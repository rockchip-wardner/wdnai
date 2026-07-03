"""rksdk — LLM / VLM access SDK for Rockchip NPUs (rknn3-toolkit-lite).

Public API:

    from rksdk import BaseModel, Field, Generable, Callable, Attachment, LLMSession, SystemLanguageModel

Quick start (Qwen3-VL image):

    from typing import Literal
    from rksdk import Generable, Field

    @Generable
    class Event:
        act: str = Field(description="action")
        level: Literal["Normal", "Warning"] = Field(default="Normal")

    model = SystemLanguageModel.qwen3_vl_4b
    session = LLMSession(model=model)
    out = await session.respond(
        msg="Analyze this image",
        attachment=Attachment(img),
        generating=Event,
    )
"""
from __future__ import annotations

from pydantic import BaseModel, Field  # noqa: F401  re-exported

from ._callable import Callable
from ._generable import Generable
from .attachment import Attachment
from .model import ModelSpec, SystemLanguageModel
from .session import LLMSession
from .video import VideoCapture

__all__ = [
    "BaseModel",
    "Field",
    "Generable",
    "Callable",
    "Attachment",
    "ModelSpec",
    "SystemLanguageModel",
    "LLMSession",
    "VideoCapture",
]

__version__ = "0.2.0"
