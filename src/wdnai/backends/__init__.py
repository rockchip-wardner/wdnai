"""Backend factory: build the right Backend from a ModelSpec."""
from __future__ import annotations

from typing import Dict, Optional, Type, Union

from ..model import ModelSpec
from ._common import resolve_device_id
from .embedding import EmbeddingBackend
from .text import TextBackend
from .tts import TTSBackend
from .vlm import VLMBackend

_REGISTRY: Dict[str, Type] = {
    "llm": TextBackend,
    "vlm": VLMBackend,
    "tts": TTSBackend,
    "embedding": EmbeddingBackend,
}


def build_backend(
    spec: ModelSpec,
    *,
    npu: Optional[int] = None,
    device_id: Optional[Union[bytes, str]] = None,
):
    """Pick the backend implementation based on ``spec.kind``.

    ``"asr"`` is loaded on demand — :mod:`wdnai.backends.asr` imports
    rknn3lite at module load time, which segfaults on non-RK hosts.

    ``npu`` and ``device_id`` are NPU-selection hints forwarded to the
    backend constructor. They are mutually exclusive at the
    :class:`LLMSession` level; this factory just passes them through.
    Actual resolution against ``RKNN3Lite().get_devices_id()`` happens
    inside each backend's ``__init__`` — keeping the RKNN3 native
    driver init lazy and out of the way of any co-resident ``wdnav``
    process.
    """
    if spec.kind == "asr":
        from .asr import ASRBackend  # type: ignore  # noqa: WPS433
        return ASRBackend(spec, npu=npu, device_id=device_id)
    if spec.kind == "omni":
        from .omni import OmniBackend  # type: ignore  # noqa: WPS433
        return OmniBackend(spec, npu=npu, device_id=device_id)
    cls = _REGISTRY.get(spec.kind)
    if cls is None:
        raise ValueError(
            f"Unknown kind: {spec.kind!r} (expected 'llm' / 'vlm' / 'tts' / 'omni' / 'asr' / 'embedding')"
        )
    return cls(spec, npu=npu, device_id=device_id)


__all__ = [
    "TextBackend", "VLMBackend", "TTSBackend", "EmbeddingBackend",
    "OmniBackend", "ASRBackend",
    "build_backend", "resolve_device_id",
]
