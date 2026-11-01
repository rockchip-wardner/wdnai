"""Backend factory: build the right Backend from a ModelSpec."""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Type, Union

from ..model import ModelSpec
from ._common import resolve_device_id
from .depth_anything import DepthAnythingBackend
from .embedding import EmbeddingBackend
from .locate_anything import LocateAnythingBackend
from .text import TextBackend
from .tts import TTSBackend
from .vlm import VLMBackend

_REGISTRY: Dict[str, Type] = {
    "llm": TextBackend,
    "vlm": VLMBackend,
    "tts": TTSBackend,
    "embedding": EmbeddingBackend,
    "grounding": LocateAnythingBackend,
    "depth": DepthAnythingBackend,
}


def build_backend(
    spec: ModelSpec,
    *,
    npu: Optional[int] = None,
    device_id: Optional[Union[bytes, str]] = None,
    npu_indices: Optional[Sequence[int]] = None,
):
    """Pick the backend implementation based on ``spec.kind``.

    ``"asr"`` is loaded on demand — :mod:`wdnai.backends.asr` imports
    :mod:`wdnai._librknn` (which loads librknn3) at module load time,
    which segfaults on non-RK hosts.

    ``npu`` and ``device_id`` are NPU-selection hints forwarded to the
    backend constructor. They are mutually exclusive at the
    :class:`LLMSession` level; this factory just passes them through.
    Actual resolution against ``RKNN3Lite().get_devices_id()`` happens
    inside each backend's ``__init__`` — keeping the librknn3 native
    driver init lazy and out of the way of any co-resident ``wdnav``
    process.
    """
    if spec.kind == "asr":
        from .asr import ASRBackend  # type: ignore  # noqa: WPS433
        return ASRBackend(spec, npu=npu, device_id=device_id)
    if spec.kind == "omni":
        from .omni import OmniBackend  # type: ignore  # noqa: WPS433
        return OmniBackend(spec, npu=npu, device_id=device_id)
    if spec.kind == "vlm_multicard":
        from .vlm_multicard_rknn3lite import (  # type: ignore  # noqa: WPS433
            Rknn3LiteMultiCardVLMBackend,
        )
        return Rknn3LiteMultiCardVLMBackend(
            spec, npu_indices=npu_indices,
            npu=npu, device_id=device_id,
        )
    cls = _REGISTRY.get(spec.kind)
    if cls is None:
        raise ValueError(
            f"Unknown kind: {spec.kind!r} (expected 'llm' / 'vlm' / 'tts' / 'omni' / 'asr' / 'embedding' / 'grounding' / 'depth')"
        )
    return cls(spec, npu=npu, device_id=device_id)


__all__ = [
    "TextBackend", "VLMBackend", "Rknn3LiteMultiCardVLMBackend",
    "TTSBackend", "EmbeddingBackend", "LocateAnythingBackend",
    "OmniBackend", "ASRBackend", "DepthAnythingBackend",
    "build_backend", "resolve_device_id",
]
