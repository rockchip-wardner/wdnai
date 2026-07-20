"""Attachment: image or list of images (BGR ndarrays).

The VLM backend takes care of BGR -> RGB -> resize -> normalization. rkai does
not handle video decoding; callers must hand in already-decoded frames via
hardware paths (e.g. ``rkv.VideoCapture``) and pass them as a list of ndarrays.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:  # noqa: F401
    import numpy as np


class Attachment:
    """Attachment sent to the LLM / VLM.

    Construction supports:

    - Single image: ``Attachment(img)``, where ``img`` is a BGR ndarray
      (e.g. from ``cv2.imread``).
    - Multiple images: ``Attachment([img1, img2, ...])`` — frames sampled by
      the caller (e.g. via ``rkv.VideoCapture``).
    """

    def __init__(self, content: Union["np.ndarray", List["np.ndarray"]]):
        try:
            import numpy as _np  # noqa: WPS433
        except ImportError:
            _np = None  # type: ignore[assignment]
        self._np = _np

        if _np is not None and isinstance(content, _np.ndarray):
            self.kind = "image"
            self.images: List["np.ndarray"] = [content]
        elif _np is not None and isinstance(content, list) and all(
            isinstance(x, _np.ndarray) for x in content
        ):
            self.kind = "image"
            self.images = list(content)
        else:
            raise TypeError(
                f"Unsupported attachment type: {type(content).__name__!r}. "
                "Expected np.ndarray | List[np.ndarray]. "
                "Video decoding is not supported in rkai; use rkv to decode frames first."
            )

    # ------------------------------------------------------------------ #
    def to_payload(self) -> dict:  # pragma: no cover
        return {"kind": self.kind, "n": len(self.images)}

    def __repr__(self) -> str:  # pragma: no cover
        return f"Attachment(images={len(self.images)})"