"""Qwen2.5-Omni-3B ALM 推理示例（wav-in, text-out）。

模型文件位于 ``/userdata/models/Qwen2.5-Omni-3B/``（覆盖用
``WDNAI_MODELS_DIR`` / ``WDNAI_QWEN_2_5_OMNI_3B_DIR``）；后端会自动在用户
``prompt`` 开头补 ``<audio>`` 占位符。每次 ``respond()`` 前都
``session.reset()`` 清 KV cache，方便 ``--iter K`` 跑多轮测一致性。

Usage::

    .venv/bin/python examples/audio_understanding.py --wav demo.wav
    .venv/bin/python examples/audio_understanding.py --wav demo.wav --iter 3
    .venv/bin/python examples/audio_understanding.py --wav demo.wav \\
        --prompt '请用一句话概括这段音频的主题' --perf
"""
from __future__ import annotations

import argparse
import io
import logging
import struct
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from wdnai import (
    AudioAttachment,
    LLMSession,
    SystemLanguageModel,
)

LOG = logging.getLogger("audio_understanding")


# CLI
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--wav", type=Path, default=Path("demo.wav"))
    p.add_argument("--model-dir", type=Path, default=None,
                   help="覆盖 Omni 模型根目录；默认 $WDNAI_MODELS_DIR/Qwen2.5-Omni-3B")
    p.add_argument("--prompt", default="请描述这段音频的内容",
                   help="用户 prompt；后端会自动在前面补 <audio> 占位符")
    p.add_argument("--iter", type=int, default=1,
                   help="连续推理轮数（每轮 reset KV cache）")
    p.add_argument("--max-new-tokens", type=int, default=1024,
                   help="LLM 单轮最大生成 tokens")
    p.add_argument("--perf", action="store_true",
                   help="打印 Prefill / Generate / Audio 延迟表")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


# WAV I/O — supports PCM s16le/s24le/s32le (fmt 1) and IEEE float32 (fmt 3,
# TTS output shape, downmixed to mono + rescaled to s16le). Mirrors
# ``examples/audio_to_text.py``.
def read_wav(path: Path) -> Tuple[bytes, int, int, int]:
    """Return ``(pcm_bytes, sample_rate, n_channels, sample_width)``."""
    if not path.is_file():
        raise FileNotFoundError(f"wav not found: {path}")
    fmt, n_channels, sample_rate, sample_width, data = _parse_wav(path.read_bytes())
    if fmt == 3:
        audio = np.frombuffer(data, dtype="<f4").copy()
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)
        np.clip(audio, -1.0, 1.0, out=audio)
        pcm = (audio * 32767.0).astype("<i2").tobytes()
        sample_width = 2
    elif fmt == 1 and sample_width in (1, 2, 3, 4):
        pcm = data
    else:
        raise ValueError(
            f"unsupported WAV format tag={fmt} sample_width={sample_width}"
        )
    LOG.info(
        "wav: %s fmt=%d sr=%d ch=%d width=%d bytes=%d",
        path, fmt, sample_rate, n_channels, sample_width, len(pcm),
    )
    return pcm, sample_rate, n_channels, sample_width


def _parse_wav(raw: bytes):
    """Minimal RIFF/WAVE parser. ``(format_tag, n_channels,
    sample_rate, sample_width, data_bytes)``."""
    if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    bio = io.BytesIO(raw)
    bio.read(12)
    fmt_tag = n_channels = sample_rate = sample_width = 0
    data = b""
    while True:
        head = bio.read(8)
        if len(head) < 8:
            break
        cid, csize = head[:4], struct.unpack("<I", head[4:8])[0]
        cdata = bio.read(csize)
        if cid == b"fmt " and len(cdata) >= 16:
            fmt_tag, n_channels, sample_rate, _, _, bps = struct.unpack(
                "<HHIIHH", cdata[:16]
            )
            sample_width = bps // 8
        elif cid == b"data":
            data = cdata
            break
        if csize % 2:
            bio.read(1)
    if fmt_tag == 0 or not data:
        raise ValueError("missing/incomplete fmt / data chunk in WAV")
    return fmt_tag, n_channels, sample_rate, sample_width, data


# Main
async def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    pcm, sample_rate, n_channels, sample_width = read_wav(args.wav)

    model = SystemLanguageModel.qwen_2_5_omni_3b()
    if args.model_dir:
        model = model.with_paths(omni_model_dir=str(args.model_dir))

    session = LLMSession(model=model)
    try:
        for i in range(max(1, args.iter)):
            print(f"\n========== run {i + 1}/{max(1, args.iter)} ==========")
            t0 = time.perf_counter()
            session.reset()

            result = await session.respond(
                prompt=args.prompt,
                attachment=[AudioAttachment(pcm, mime_type="audio/pcm")],
                sample_rate=sample_rate,
                n_channels=n_channels,
                sample_width=sample_width,
                max_new_tokens=args.max_new_tokens,
                perf=args.perf,
            )

            print(f"  result  = {result!r}")
            print(f"  wall    = {time.perf_counter() - t0:.2f} s")
    finally:
        session.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())