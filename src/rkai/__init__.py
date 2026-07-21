"""rkai — LLM / VLM / TTS access SDK for Rockchip NPUs (rknn3-toolkit-lite).

Quick start (Qwen3-VL image):

    from typing import Literal
    from rkai import Generable, Field, Attachment, LMOutput, LLMSession, SystemLanguageModel

    @Generable
    class Event(LMOutput):
        act: str = Field(description="action")
        level: Literal["Normal", "Warning"] = Field(default="Normal")

    model = SystemLanguageModel.qwen3_vl_4b()
    session = LLMSession(model=model)
    out = await session.respond(
        msg="Analyze this image",
        attachment=[Attachment(img)],
        generating=Event,
    )
"""
from __future__ import annotations

from pydantic import BaseModel, Field  # noqa: F401  re-exported

from ._callable import Callable
from ._generable import Generable
from .attachment import Attachment
from .lm_output import LMOutput
from .model import ModelSpec, SystemLanguageModel
from .session import LLMSession

__all__ = [
    "BaseModel",
    "Field",
    "Generable",
    "Callable",
    "Attachment",
    "LMOutput",
    "ModelSpec",
    "SystemLanguageModel",
    "LLMSession",
]

__version__ = "0.1.1"
