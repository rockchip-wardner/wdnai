"""Backend factory: build the right Backend from a ModelSpec."""
from __future__ import annotations

from typing import Dict, Type

from ..model import ModelSpec
from .text import TextBackend
from .tts import TTSBackend
from .vlm import VLMBackend

_REGISTRY: Dict[str, Type] = {
    "llm": TextBackend,
    "vlm": VLMBackend,
    "tts": TTSBackend,
}


def build_backend(spec: ModelSpec):
    """Pick the backend implementation based on ``spec.kind``.

    ``"asr"`` is loaded on demand — :mod:`rkai.backends.asr` imports
    rknn3lite at module load time, which segfaults on non-RK hosts.
    """
    if spec.kind == "asr":
        from .asr import ASRBackend  # type: ignore  # noqa: WPS433
        return ASRBackend(spec)
    cls = _REGISTRY.get(spec.kind)
    if cls is None:
        raise ValueError(
            f"Unknown kind: {spec.kind!r} (expected 'llm' / 'vlm' / 'tts' / 'asr')"
        )
    return cls(spec)


__all__ = ["TextBackend", "VLMBackend", "TTSBackend", "build_backend"]
