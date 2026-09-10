"""wdnai — LLM / VLM / TTS access SDK for Rockchip NPUs (librknn3 ctypes).

The SDK talks to librknn3 (`/usr/lib/librknn3_api.so`) directly via the
vendored ctypes shim in :mod:`wdnai._librknn`; it does not depend on
the ``rknn3lite`` / ``rknn3-toolkit-lite`` Python wheel. See AGENTS.md
for the design notes.

Quick start (Qwen3-VL image):

    from typing import Literal
    from wdnai import (
        Generable, Field, ImageAttachment, LMOutput, LLMSession, SystemLanguageModel,
    )

    @Generable
    class Event(LMOutput):
        act: str = Field(description="action")
        level: Literal["Normal", "Warning"] = Field(default="Normal")

    model = SystemLanguageModel.qwen3_vl_4b()
    session = LLMSession(model=model)
    out = await session.respond(
        msg="Analyze this image",
        attachment=[ImageAttachment(img)],
        generating=Event,
    )
"""
from __future__ import annotations

from pydantic import *  # noqa: F401  re-exported
from typing import *

from ._callable import Callable
from ._generable import Generable
from .attachment import Attachment, AudioAttachment, ImageAttachment, TextAttachment
from .lm_output import LMOutput
from .model import ModelSpec, SystemLanguageModel
from .session import LLMSession
from .backends.locate_anything import LocateAnythingBackend

__all__ = [
    "BaseModel",
    "Field",
    "Generable",
    "Callable",
    "Attachment",
    "TextAttachment",
    "ImageAttachment",
    "AudioAttachment",
    "LMOutput",
    "ModelSpec",
    "SystemLanguageModel",
    "LLMSession",
    "LocateAnythingBackend",
]

__version__ = "0.4.0"
