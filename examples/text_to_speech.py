"""Qwen3-TTS 推理 demo —— v2 ``LLMSession`` API 风格。

整套 TTS 流水线（HF tokenizer → text_projection prefill → talker 自
回归 → code_predictor → speech_decoder → 24 kHz mono float32 WAV
bytes）全部封装在 SDK 的 ``TTSBackend`` 里（参见
``.venv/src/rkai/src/rkai/backends/tts.py``），调用方只负责：

1. 构造 ``model = SystemLanguageModel.qwen3_tts()``
2. ``session = LLMSession(model=model)``
3. ``result = await session.respond(attachment=[Attachment(text, kind="text")], generating=TTSResult)``
4. 从 ``result.attachment[0].data`` 拿到 WAV bytes，自己做后处理（落盘 + 播放）

参考 ``detect_image.py`` 的模式（list of Attachment + ``@Generable`` 继承
``LMOutput`` 的 ``generating`` 类），同时对齐 ``detect_image.py`` /
``detect_video.py`` 的 CLI 风格（argparse + 顶层 ``main``）。

模型布局：

    /userdata/models/Qwen3-TTS/
        text_projection.1828.rknn + .weight
        talker.1828.rknn + .weight
        code_predictor.1828.rknn + .weight
        speech_decoder.1828.rknn + .weight
        talker_text_embed.fp16.bin
        talker_input_embed.fp16.bin
        codec_embed.fp16.bin
        tokenizer.json

可用环境变量 ``RKAI_MODELS_DIR`` 覆盖根目录。

执行：

    .venv/bin/python .venv/src/rkai/examples/text_to_speech.py \\
        --text "你好世界" --ref-speaker girl_base
"""
from __future__ import annotations

import argparse
import io
import logging
import struct
from pathlib import Path
from typing import List

import numpy as np

from rkai import (
    Attachment,
    Field,
    Generable,
    LMOutput,
    LLMSession,
    SystemLanguageModel,
)

LOG = logging.getLogger("text_to_speech")


# --------------------------------------------------------------------------- #
# Structured output schema
# --------------------------------------------------------------------------- #
@Generable
class TTSResult(LMOutput):
    """TTS 输出的结构化描述。

    * ``text`` 是输入 prompt 的回显（v2 SDK 总是 echo 一次）。
    * ``sample_rate`` / ``channels`` 描述了音频格式。
    * ``attachment[0].data`` 携带 24 kHz mono float32 的 WAV bytes（已
      包含完整 RIFF 头；不需要再编码）。
    """
    text:        str = Field(description="Input text that was synthesised.")
    sample_rate: int = Field(default=24000)
    channels:    int = Field(default=1)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--text", required=True,
                   help="要合成的文本")
    p.add_argument("--language", default="auto",
                   help="语言 (auto / chinese / english / japanese / ...)")
    p.add_argument("--speaker", default="",
                   help="内置 SPK_ID（serena/vivian/ryan/...），与 --ref-speaker 互斥")
    p.add_argument("--ref-speaker", default="girl_base",
                   help="qwen3tts_speaker_embed.h 里的 voice-clone speaker (girl_base / ahu)")
    p.add_argument("--instruct", default="",
                   help="可选 CustomVoice instruct")
    p.add_argument("--volume", type=float, default=0.1,
                   help="播放音量 0.0 ~ 1.0（线性缩放 PCM 幅度，默认 0.1）")
    p.add_argument("--no-play", action="store_true",
                   help="只落盘 wav，不出声")
    p.add_argument("--output", "-o", type=Path, default=Path("output.wav"),
                   help="输出 WAV 文件路径")
    p.add_argument("--model-dir", type=Path, default=None,
                   help="覆盖模型根目录；默认 $RKAI_MODELS_DIR/Qwen3-TTS")
    p.add_argument("--non-streaming-mode", type=int, default=0, choices=[0, 1],
                   help="1 = 使用 non-streaming prefill（对应 C++ non_streaming_mode=1）")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Post-processing: WAV duration + volume scaling
# --------------------------------------------------------------------------- #
def _wav_duration_seconds(wav_bytes: bytes) -> float:
    """Decode audio duration from a RIFF/WAVE header.

    Tries the ``data`` chunk's sample count first; falls back to the
    ``fmt `` byte rate if the chunk layout is unexpected.
    """
    if not wav_bytes or len(wav_bytes) < 44:
        return 0.0
    bio = io.BytesIO(wav_bytes)
    # RIFF/WAVE layout: "RIFF" (4) + size (4) + "WAVE" (4) + chunks.
    # Read 12 bytes at once so the second 4-byte compare lands on
    # "WAVE" instead of the (also-4-byte) file size field.
    head = bio.read(12)
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        return 0.0
    sample_rate = 24000
    bits_per_sample = 32
    num_channels = 1
    data_size = 0
    while True:
        h = bio.read(8)
        if len(h) < 8:
            break
        cid, csize = h[:4], struct.unpack("<I", h[4:8])[0]
        cdata = bio.read(csize)
        if cid == b"fmt ":
            # fmt chunk: format_tag(2) channels(2) sample_rate(4) byte_rate(4)
            #          block_align(2) bits_per_sample(2)
            (_fmt, num_channels, sample_rate,
             _br, _ba, bits_per_sample) = struct.unpack("<HHIIHH", cdata[:16])
        elif cid == b"data":
            data_size = csize
            break
        if csize % 2:
            bio.read(1)
    if sample_rate > 0 and num_channels > 0 and bits_per_sample > 0 and data_size > 0:
        bytes_per_frame = num_channels * bits_per_sample // 8
        return data_size / (sample_rate * bytes_per_frame)
    return 0.0


def _apply_volume_to_wav(wav_bytes, volume):
    """Return a copy of *wav_bytes* with the float32 PCM chunk rescaled by *volume*.

    The upstream TTS model emits a raw waveform whose peak amplitude is
    sometimes >0.5 on sibilant / plosive consonants (e.g. the 世 in
    "你好世界" hits 0.557 on 世's sh-sound). The SDK applies
    ``chunk *= self.volume`` only inside the live-playback path, so the
    on-disk file keeps that loud peak. This helper rewrites the ``data``
    chunk in place so the saved file matches the chosen volume.

    Falls back to the original *wav_bytes* if the header is malformed
    or the volume is already 1.0 (no-op).
    """
    if volume == 1.0 or len(wav_bytes) < 44:
        return wav_bytes
    bio = io.BytesIO(wav_bytes)
    # RIFF/WAVE layout: "RIFF" (4) + size (4) + "WAVE" (4) + chunks.
    head = bio.read(12)
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        return wav_bytes
    bio.seek(0)
    out = bytearray()
    while True:
        chunk = bio.read(8)
        if len(chunk) < 8:
            break
        cid, csize = chunk[:4], struct.unpack("<I", chunk[4:8])[0]
        cdata = bio.read(csize)
        out += chunk
        out += cdata
        if cid == b"data":
            floats = np.frombuffer(cdata, dtype=np.float32).copy()
            floats = np.clip(floats * volume, -1.0, 1.0).astype(np.float32, copy=False)
            out[-csize:] = floats.tobytes()
        if csize % 2:
            pad = bio.read(1)
            out += pad
    # RIFF size = (out_size - 8); keep it in sync with the header bytes.
    struct.pack_into("<I", out, 4, len(out) - 8)
    return bytes(out)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
async def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    model = SystemLanguageModel.qwen3_tts()
    if args.model_dir:
        model.tts_model_dir = str(args.model_dir)
    session = LLMSession(model=model)
    try:
        # Forward --no-play directly to the SDK so its internal
        # ``AudioPlayer`` (the chunk-by-chunk streamer) stays silent
        # when the user wants silence. When the flag is False the SDK
        # streams live; we don't replay afterwards.
        result = await session.respond(
            attachment=[Attachment(args.text, kind="text")],
            generating=TTSResult,
            ref_speaker=args.ref_speaker,
            language=args.language,
            speaker=args.speaker,
            instruct=args.instruct,
            volume=args.volume,
            no_play=args.no_play,
            non_streaming_mode=bool(args.non_streaming_mode),
        )

        atts = list(getattr(result, "attachment", []) or [])
        if not atts:
            print("[tts] backend produced no audio attachment")
            return
        wav_bytes = atts[0].data
        duration = _wav_duration_seconds(wav_bytes)
        sample_rate = getattr(result, "sample_rate", 24000)

        # Apply the user's volume to the saved file too. The SDK only
        # scales the live PCM stream; the file on disk stays at the
        # raw model output unless we rewrite the data chunk here.
        file_bytes = _apply_volume_to_wav(wav_bytes, args.volume)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "wb") as f:
            f.write(file_bytes)
        LOG.info(
            "saved %d bytes (%.2fs @ %d Hz) to %s",
            len(wav_bytes), duration, sample_rate, args.output,
        )

        print(
            f"[tts] done. text={result.text!r} sample_rate={sample_rate} "
            f"channels={result.channels} duration={duration:.2f}s"
        )
    finally:
        session.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
