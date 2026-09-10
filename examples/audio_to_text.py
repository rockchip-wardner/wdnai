"""Qwen3-ASR demo — wav-in, transcript-out.

Pipeline is encapsulated in :class:`wdnai.backends.asr.ASRBackend`;
this script handles WAV I/O and parses ``"language <lang>\\n<text>"``
back into two fields.

Usage::

    # Offline (encoder.1828.rknn, one-shot)
    .venv/bin/python examples/audio_to_text.py --wav /path/to.wav --iter 3

    # Online / streaming (encoder_online.1828.rknn, 1-second chunks,
    # partial text committed each round after a rollback warm-up)
    .venv/bin/python examples/audio_to_text.py --wav /path/to.wav --online

``--online`` prints each round's confirmed prefix / unfixed tail to
stdout; the final transcript is the ``"language <lang>\\n<text>"``
return value of ``ASRBackend.run_audio`` (same shape as offline).

``WDNAI_MODELS_DIR`` overrides the model root.
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

LOG = logging.getLogger("audio_to_text")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--wav", type=Path, default=Path("asr_en.wav"))
    p.add_argument("--model-dir", type=Path, default=None,
                   help="覆盖模型根目录；默认 $WDNAI_MODELS_DIR/Qwen3-ASR")
    p.add_argument("--iter", type=int, default=10,
                   help="连续识别轮数（每轮 reset KV cache）")
    p.add_argument("--max-new-tokens", type=int, default=1024 * 6,
                   help="LLM 单轮最大生成 tokens（offline 生效）")
    p.add_argument("--language", default=None,
                   help="可选 ASR 语言提示；当前 SDK 透传但不强制")
    p.add_argument("--perf", action="store_true",
                   help="打印 Prefill / Generate 延迟表")
    p.add_argument("--online", action="store_true",
                   help="Use encoder_online.1828.rknn (streaming, 1s/chunk)")
    p.add_argument("--online-chunk-seconds", type=float, default=1.0,
                   help="Online chunk size (default 1.0s)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# WAV I/O — stdlib ``wave`` rejects WAVE_FORMAT_IEEE_FLOAT (format tag 3),
# which is exactly what ``Qwen3TTSDriver._write_wav`` produces. Decode the
# header by hand and convert IEEE float to s16le so the backend's integer
# PCM path sees the same shape for any upstream wav.
# --------------------------------------------------------------------------- #
def read_wav(path: Path) -> Tuple[bytes, int, int, int]:
    """Return ``(pcm_bytes, sample_rate, n_channels, sample_width)``.

    Accepts PCM s16le/s24le/s32le (format tag 1) and IEEE float32
    (format tag 3, downmixed and rescaled to s16le).
    """
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
    elif fmt == 1 and sample_width in (2, 3, 4):
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


def split_language_and_text(raw: str) -> Tuple[str, str]:
    """``ASRBackend`` emits ``"language <lang>\\n<text>\\n"`` (the
    151704 separator is replaced with ``\\n`` on the way out)."""
    if "\n" not in raw:
        return "", raw.strip() if raw else ""
    head, body = raw.split("\n", 1)
    if not head.lower().startswith("language "):
        return "", raw.strip()
    return head[len("language "):].strip(), body.strip()


# --------------------------------------------------------------------------- #
# Online-commit handler — ``\r\033[K`` reprints the same console line so
# the watcher sees the running committed / unfixed prefix (matches the
# single-line trick in cpp/online_main.cc::print_single_line_result).
# --------------------------------------------------------------------------- #
class _StreamingPrinter:
    RED = "\033[31m"
    BLUE = "\033[34m"
    RESET = "\033[0m"
    CLEAR = "\033[K"

    def __init__(self):
        self.printed_committed_len = 0

    def __call__(self, committed: str, unfixed: str, language: str):
        tail = committed[self.printed_committed_len:]
        self.printed_committed_len = len(committed)
        print(
            f"\rcommitted: {tail}{self.BLUE}{unfixed}{self.RESET}{self.CLEAR}",
            end="", flush=True,
        )

    def newline(self):
        print("")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
async def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    pcm, sample_rate, n_channels, sample_width = read_wav(args.wav)
    model = SystemLanguageModel.qwen3_asr()
    if args.model_dir:
        model = model.with_paths(asr_model_dir=str(args.model_dir))
    if args.online:
        model = model.with_options(
            use_online_encoder=True,
            online_chunk_seconds=args.online_chunk_seconds,
        )

    session = LLMSession(model=model)
    try:
        printer = _StreamingPrinter() if args.online else None
        for i in range(max(1, args.iter)):
            print(f"\n========== run {i} ==========")
            t0 = time.perf_counter()
            session.reset()
            if printer is not None:
                printer.printed_committed_len = 0
            on_commit = printer if args.online else None

            raw = await session.respond(
                attachment=[AudioAttachment(pcm, mime_type="audio/pcm")],
                sample_rate=sample_rate,
                n_channels=n_channels,
                sample_width=sample_width,
                max_new_tokens=args.max_new_tokens,
                language=args.language,
                perf=args.perf,
                on_asr_commit=on_commit,
            )

            if printer is not None:
                printer.newline()
            lang, text = split_language_and_text(raw)
            print(f"  language : {lang!r}")
            print(f"  text     : {text!r}")
            print(f"  wall     : {time.perf_counter() - t0:.2f} s")
    finally:
        session.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())