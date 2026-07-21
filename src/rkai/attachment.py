"""Attachment: text / image / video / audio / raw bytes.

Unified carrier for inputs and outputs of an ``LLMSession``. The
``kind`` field selects which channel populates (``images`` /
``frames`` / ``text`` / ``data``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:  # noqa: F401
    import numpy as np


class Attachment:
    """Attachment sent to / received from an LLM / VLM / TTS session.

    Construction supports:

    - Single image: ``Attachment(img)`` where ``img`` is a BGR ndarray.
    - Multiple images: ``Attachment([img1, img2, ...])``.
    - Text: ``Attachment("the prompt text")`` or ``Attachment("…", kind="text")``.
      When the string happens to also be an existing file path, pass
      ``kind="text"`` explicitly to opt out of the video heuristic.
    - Audio bytes (TTS output): ``Attachment(wav_bytes, kind="audio")``.
    - Video file path: ``Attachment("/path/to.mp4", kind="video")``.
    - Raw bytes (video or generic): ``Attachment(b"...", kind="video")``.

    Attributes (the channel that's populated depends on ``kind``):

    - ``kind``     — ``"text" | "image" | "video" | "audio"``.
    - ``text``     — str content (when ``kind="text"``).
    - ``images``   — list[BGR ndarray] (when ``kind="image"``).
    - ``frames``   — list[BGR ndarray] (when ``kind="video"``; sampled at
                    construction).
    - ``data``     — raw bytes (when ``kind="audio"`` or raw video).
    - ``mime_type`` — optional, e.g. ``"audio/wav"``.
    """

    def __init__(
        self,
        content: Union["np.ndarray", List["np.ndarray"], str, bytes, bytearray],
        kind: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> None:
        try:
            import numpy as _np  # noqa: WPS433
        except ImportError:
            _np = None  # type: ignore[assignment]
        self._np = _np

        self.text: Optional[str] = None
        self.images: List["np.ndarray"] = []
        self.frames: List["np.ndarray"] = []
        self.data: bytes = b""
        self.mime_type = mime_type

        # ---------- image (ndarray / list[ndarray]) ----------
        if _np is not None and isinstance(content, _np.ndarray):
            self.kind = "image"
            self.images = [content]
            return
        if (
            _np is not None
            and isinstance(content, list)
            and all(isinstance(x, _np.ndarray) for x in content)
        ):
            self.kind = "image"
            self.images = list(content)
            return

        # ---------- raw bytes ----------
        if isinstance(content, (bytes, bytearray)):
            data = bytes(content)
            # Default bytes → audio (the more common TTS output shape);
            # callers can override with kind="video" for raw video streams.
            self.kind = kind or "audio"
            self.data = data
            if self.mime_type is None:
                self.mime_type = "audio/wav"
            return

        # ---------- string ----------
        if isinstance(content, str):
            # Honor explicit kind first.
            if kind == "text":
                self.kind = "text"
                self.text = content
                return
            if kind == "video":
                self.kind = "video"
                self.frames = self._sample_video(content)
                return
            # Heuristic: existing file → video; otherwise treat as text.
            if self._looks_like_path(content):
                try:
                    frames = self._sample_video(content)
                except Exception:
                    self.kind = "text"
                    self.text = content
                else:
                    self.kind = "video"
                    self.frames = frames
            else:
                self.kind = "text"
                self.text = content
            return

        raise TypeError(
            f"Unsupported attachment type: {type(content).__name__!r}. "
            "Expected np.ndarray | List[np.ndarray] | str | bytes | bytearray."
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _looks_like_path(s: str) -> bool:
        """Heuristic: treat ``str`` as a video path only if it points to
        an existing file with a video-y extension. Otherwise assume text."""
        import os
        if "\n" in s or "\r" in s:
            return False
        if not os.path.exists(s) or not os.path.isfile(s):
            return False
        _, ext = os.path.splitext(s)
        return ext.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv"}

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
        out = {"kind": self.kind}
        if self.kind == "image":
            out["n_images"] = len(self.images)
        elif self.kind == "video":
            out["n_frames"] = len(self.frames)
        elif self.kind == "text":
            out["n_chars"] = len(self.text or "")
        elif self.kind == "audio":
            out["n_bytes"] = len(self.data)
            out["mime_type"] = self.mime_type
        return out

    def __repr__(self) -> str:  # pragma: no cover
        if self.kind == "image":
            return f"Attachment(images={len(self.images)})"
        if self.kind == "video":
            return f"Attachment(frames={len(self.frames)})"
        if self.kind == "text":
            n = len(self.text or "")
            return f"Attachment(text={n}chars)"
        if self.kind == "audio":
            return f"Attachment(audio={len(self.data)}B, mime={self.mime_type})"
        return f"Attachment(kind={self.kind!r})"

    # Convenience factories -------------------------------------------- #
    @classmethod
    def text(cls, content: str) -> "Attachment":
        """Explicit text attachment (skips the path heuristic)."""
        return cls(content, kind="text")

    @classmethod
    def video_path(cls, path: str) -> "Attachment":
        """Explicit video-from-file attachment (skips the path heuristic)."""
        return cls(path, kind="video")

    @classmethod
    def audio_bytes(cls, data: bytes, mime_type: str = "audio/wav") -> "Attachment":
        """Explicit audio attachment (TTS output carrier)."""
        return cls(data, kind="audio", mime_type=mime_type)
