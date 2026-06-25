"""Attachment: image / video / byte stream.

Exposes ``Attachment.images`` / ``Attachment.frames`` as lists of BGR ndarrays;
the VLM backend takes care of the subsequent BGR -> RGB -> resize -> normalization.
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
    - Multiple images: ``Attachment([img1, img2, ...])``.
    - Video file path: ``Attachment(video_path)`` — frames are sampled at
      construction time.
    """

    def __init__(self, content: Union["np.ndarray", List["np.ndarray"], str, bytes, bytearray]):
        try:
            import numpy as _np  # noqa: WPS433
        except ImportError:
            _np = None  # type: ignore[assignment]
        self._np = _np

        if _np is not None and isinstance(content, _np.ndarray):
            self.kind = "image"
            self.images: List["np.ndarray"] = [content]
            self.frames: List["np.ndarray"] = []
        elif _np is not None and isinstance(content, list) and all(isinstance(x, _np.ndarray) for x in content):
            self.kind = "image"
            self.images = list(content)
            self.frames = []
        elif isinstance(content, str):
            # Treated as a video path; frames are sampled immediately.
            self.kind = "video"
            self.images = []
            self.frames = self._sample_video(content)
        elif isinstance(content, (bytes, bytearray)):
            self.kind = "video"
            self.images = []
            # Raw video bytes need a temp file to be read by cv2; not supported here for simplicity.
            raise NotImplementedError("Raw video bytes are not supported; please write to disk and pass the path.")
        else:
            raise TypeError(
                f"Unsupported attachment type: {type(content).__name__!r}. "
                "Expected np.ndarray | List[np.ndarray] | str(video path) | bytes."
            )

    # ------------------------------------------------------------------ #
    def _sample_video(self, path: str, n_frames: int = 5, frame_size: int = 384):
        """Evenly sample ``n_frames`` from the video. Returns a list of BGR ndarrays."""
        try:
            import cv2  # noqa: WPS433
        except ImportError as e:
            raise ImportError("Video handling requires opencv-python (cv2)") from e
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if total <= 0:
            # Fallback: read one frame every 2 seconds
            frames = []
            for i in range(n_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(2 * i * fps)))
                ok, f = cap.read()
                if not ok or f is None:
                    break
                frames.append(f)
        else:
            idxs = [int(round(total * (i + 0.5) / n_frames)) for i in range(n_frames)]
            frames = []
            for idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, f = cap.read()
                if not ok or f is None:
                    continue
                frames.append(f)
        cap.release()
        if not frames:
            raise RuntimeError(f"Failed to sample frames from video: {path}")
        return frames

    # ------------------------------------------------------------------ #
    def to_payload(self) -> dict:  # pragma: no cover
        return {"kind": self.kind, "n": len(self.images) + len(self.frames)}

    def __repr__(self) -> str:  # pragma: no cover
        if self.kind == "image":
            return f"Attachment(images={len(self.images)})"
        return f"Attachment(frames={len(self.frames)})"
