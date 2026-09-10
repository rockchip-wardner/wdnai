"""Attachment hierarchy.

    Attachment  (base class + legacy factory constructor)
    ├── TextAttachment    — ``kind="text"``  (plain string)
    ├── ImageAttachment   — ``kind="image"`` (one or more BGR ndarray frames)
    └── AudioAttachment   — ``kind="audio"`` (raw bytes + MIME type)

New code should construct the explicit subclass
(``TextAttachment(s)`` / ``ImageAttachment(arr)`` /
``AudioAttachment(b)``) so the channel is visible at the call site and
``isinstance`` dispatch at the consumer side is type-safe.

The legacy ``Attachment(s)`` / ``Attachment(arr)`` /
``Attachment(b"...", kind="audio")`` syntax still works — the base
class' ``__new__`` routes the call to the matching concrete subclass
and returns it. There is intentionally no ``VideoAttachment`` subclass:
the legacy ``kind="video"`` path that sampled 5 frames via cv2 is
removed; callers wanting video frames now sample themselves (typically
via ``wdnav.VideoCapture.read0()``) and pass the list as
``ImageAttachment(frames)``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:  # noqa: F401
    import numpy as np

# ``wdnav`` is an optional peer dependency. ``ImageAttachment`` only
# needs the ``DecodedFrame`` class to detect ``wdnav.VideoCapture.read0()``
# outputs; keep the import lazy so a system without ``wdnav`` installed
# can still import ``wdnai`` and use the ndarray / text / bytes
# constructors.
try:
    from wdnav.mpp import DecodedFrame as _WdnavDecodedFrame  # noqa: F401
except ImportError:
    _WdnavDecodedFrame = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Base class + legacy factory routing
# --------------------------------------------------------------------------- #
class Attachment:
    """Base class for inputs to / outputs from an :class:`LLMSession`.

    Subclasses (:class:`TextAttachment`, :class:`ImageAttachment`,
    :class:`AudioAttachment`) carry exactly the payload that matches
    their channel — no union ``.text / .images / .frames / .data`` slot
    bag. The base class itself carries no payload; it exists so the
    caller can ``isinstance(item, Attachment)``-check and so the
    legacy ``Attachment(...)`` factory syntax still resolves.

    **Legacy factory syntax**: ``Attachment(s)`` / ``Attachment(arr)`` /
    ``Attachment(b"...", kind="audio")`` are still accepted — the
    ``__new__`` method below routes to the matching concrete subclass
    and returns it. New code should call the explicit subclass
    constructor so the channel is obvious at the call site.
    """

    def __new__(cls, *args, **kwargs):
        # Subclass constructors (``ImageAttachment(...)``, etc.) bypass
        # the factory branch (cls != Attachment). Only the bare
        # ``Attachment(...)`` call routes through here.
        if cls is Attachment:
            if not args and not kwargs:
                raise TypeError(
                    "Attachment is abstract — use TextAttachment, "
                    "ImageAttachment, or AudioAttachment."
                )
            return _attachment_factory(*args, **kwargs)
        return super().__new__(cls)

    # Subclasses MUST override ``kind`` with a class-level string
    # (``kind = "text"`` / ``"image"`` / ``"audio"``). The base class
    # declares it as an empty string sentinel so attribute access never
    # raises — concrete instances always have a non-empty ``kind``.
    kind: str = ""

    def to_payload(self) -> dict:  # pragma: no cover
        """Diagnostic summary; overridden by subclasses."""
        return {"kind": self.kind}

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}(kind={self.kind!r})"


# --------------------------------------------------------------------------- #
# Concrete subclasses
# --------------------------------------------------------------------------- #
class TextAttachment(Attachment):
    """Plain text carrier (chat prompts, TTS text-to-speak)."""

    kind = "text"

    def __init__(self, text: str, **_) -> None:
        # The ``**_`` sink absorbs legacy factory kwargs — when
        # ``Attachment.__new__`` routes ``Attachment("text", kind="text")``
        # to ``TextAttachment("text")``, Python then re-invokes this
        # ``__init__`` with the original kwargs (including ``kind="text"``);
        # the sink swallows them so the public signature stays strict
        # for direct construction.
        if not isinstance(text, str):
            raise TypeError(
                f"TextAttachment expects str, got {type(text).__name__}"
            )
        self.text: str = text

    def to_payload(self) -> dict:
        return {"kind": self.kind, "n_chars": len(self.text)}

    def __repr__(self) -> str:  # pragma: no cover
        return f"TextAttachment(text={len(self.text)}chars)"


class ImageAttachment(Attachment):
    """One or more BGR ndarray frames, fed to the VLM backend.

    Accepts:

    - ``ImageAttachment(img)`` where ``img`` is a BGR ndarray.
    - ``ImageAttachment([img1, img2, ...])`` for multi-frame inputs.
      This replaces the legacy ``kind="video"`` constructor path:
      callers sample frames themselves (e.g. via
      ``wdnav.VideoCapture.read0()``) and pass the list.
    - 0-copy frame from ``wdnav.VideoCapture.read0()``:
      ``ImageAttachment(frame0)`` where ``frame0`` is an
      ``wdnav.mpp.DecodedFrame``. The constructor calls
      ``frame0.numpy()`` (RGA DMA→CPU materialisation) and proceeds
      as a regular ndarray.
    """

    kind = "image"

    def __init__(
        self,
        content: Union["np.ndarray", List["np.ndarray"]],
        **_,
    ) -> None:
        # The ``**_`` sink absorbs legacy factory kwargs (``kind`` /
        # ``mime_type``) — see ``TextAttachment.__init__`` for the
        # rationale. Public signature stays strict for direct
        # construction.
        try:
            import numpy as _np  # noqa: WPS433
        except ImportError:
            _np = None  # type: ignore[assignment]

        # 0-copy frame: wdnav.mpp.DecodedFrame
        if _WdnavDecodedFrame is not None and isinstance(content, _WdnavDecodedFrame):
            arr = content.numpy()
            if _np is None or not isinstance(arr, _np.ndarray):
                raise TypeError(
                    f"{type(content).__name__}.numpy() must return np.ndarray, "
                    f"got {type(arr).__name__}"
                )
            self.images: List["np.ndarray"] = [arr]
            return

        if _np is not None and isinstance(content, _np.ndarray):
            self.images = [content]
            return

        if (
            _np is not None
            and isinstance(content, list)
            and all(isinstance(x, _np.ndarray) for x in content)
        ):
            self.images = list(content)
            return

        raise TypeError(
            f"ImageAttachment expects np.ndarray or List[np.ndarray]; "
            f"got {type(content).__name__!r}"
        )

    def to_payload(self) -> dict:
        return {"kind": self.kind, "n_images": len(self.images)}

    def __repr__(self) -> str:  # pragma: no cover
        return f"ImageAttachment(images={len(self.images)})"


class AudioAttachment(Attachment):
    """Raw audio bytes + MIME type, fed to TTS / ASR / Omni backends.

    The SDK doesn't decode the bytes — backend-specific ``run_audio``
    implementations consume them as-is (TTS uses ``audio/wav`` 24 kHz
    mono 32-bit IEEE float; Omni / ASR use 16 kHz mono s16le PCM).
    """

    kind = "audio"

    def __init__(
        self,
        data: Union[bytes, bytearray],
        mime_type: str = "audio/wav",
        **_,
    ) -> None:
        # The ``**_`` sink absorbs legacy factory kwargs (``kind``) —
        # see ``TextAttachment.__init__`` for the rationale. Public
        # signature stays strict for direct construction.
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(
                f"AudioAttachment expects bytes, got {type(data).__name__}"
            )
        self.data: bytes = bytes(data)
        self.mime_type: Optional[str] = mime_type

    def to_payload(self) -> dict:
        return {
            "kind": self.kind,
            "n_bytes": len(self.data),
            "mime_type": self.mime_type,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"AudioAttachment(audio={len(self.data)}B, mime={self.mime_type})"


# --------------------------------------------------------------------------- #
# Legacy factory routing — used by ``Attachment.__new__`` when the bare
# base class is invoked. Preserves the old ``Attachment(content, kind=...)``
# syntax so call sites written before the hierarchy refactor keep working.
# --------------------------------------------------------------------------- #
def _attachment_factory(
    content,
    kind: Optional[str] = None,
    mime_type: Optional[str] = None,
) -> Attachment:
    """Route the legacy ``Attachment(content, kind=...)`` syntax to the
    matching concrete subclass.

    Mirrors the historical heuristics so existing call sites keep
    working:

    - ``Attachment(ndarray)`` → :class:`ImageAttachment`
    - ``Attachment([ndarray, ...])`` → :class:`ImageAttachment` (multi-frame)
    - ``Attachment(str)`` → :class:`TextAttachment` (no more path→video sampling)
    - ``Attachment(bytes)`` → :class:`AudioAttachment` (default; TTS output shape)
    - ``Attachment(str, kind="text")`` → :class:`TextAttachment` (explicit)
    - ``Attachment(bytes, kind="audio")`` → :class:`AudioAttachment` (explicit)

    The legacy ``kind="video"`` constructor path is removed — there is
    no ``VideoAttachment`` subclass. Callers wanting video frames sample
    themselves (typically via ``wdnav.VideoCapture.read0()``) and pass
    the list as ``ImageAttachment(frames)``.
    """
    # Reject the deprecated video path with an informative error so the
    # caller can migrate cleanly instead of silently dropping frames.
    if kind == "video":
        raise TypeError(
            "Attachment(kind='video') is no longer supported — there is no "
            "VideoAttachment subclass. Sample the video yourself "
            "(e.g. via wdnav.VideoCapture.read0()) and pass the frames "
            "as ImageAttachment(frames)."
        )

    try:
        import numpy as _np  # noqa: WPS433
    except ImportError:
        _np = None  # type: ignore[assignment]

    # ndarray / list[ndarray] / wdnav.DecodedFrame → ImageAttachment
    if _WdnavDecodedFrame is not None and isinstance(content, _WdnavDecodedFrame):
        return ImageAttachment(content)
    if _np is not None and isinstance(content, _np.ndarray):
        return ImageAttachment(content)
    if (
        _np is not None
        and isinstance(content, list)
        and all(isinstance(x, _np.ndarray) for x in content)
    ):
        return ImageAttachment(content)

    # bytes → AudioAttachment (default; TTS output shape)
    if isinstance(content, (bytes, bytearray)):
        if kind not in (None, "audio"):
            raise TypeError(
                f"Attachment(bytes, kind={kind!r}): only kind='audio' is "
                f"supported; bytes with kind='video' is no longer supported."
            )
        return AudioAttachment(bytes(content), mime_type=mime_type or "audio/wav")

    # str → TextAttachment (no more path→video heuristic)
    if isinstance(content, str):
        if kind not in (None, "text"):
            raise TypeError(
                f"Attachment(str, kind={kind!r}): only kind='text' is "
                f"supported (no VideoAttachment subclass); pass an "
                f"ImageAttachment(frames) for video inputs."
            )
        return TextAttachment(content)

    raise TypeError(
        f"Unsupported attachment type: {type(content).__name__!r}. "
        "Use TextAttachment(text), ImageAttachment(img | [img, ...]), "
        "or AudioAttachment(bytes, mime_type=...)."
    )


__all__ = [
    "Attachment",
    "TextAttachment",
    "ImageAttachment",
    "AudioAttachment",
]