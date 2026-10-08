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

NPU status / telemetry (memory, temperature, device list, version,
reset) is exposed via :class:`wdnai.RknnSmi`, a ctypes shim over
``/usr/lib/librknnsmi.so`` — the same C ABI the ``rknn-smi`` CLI uses,
without shelling out to a subprocess:

    from wdnai import RknnSmi
    smi = RknnSmi()
    for dev in smi.get_device_list():
        print(dev["index"], dev["mode"], smi.get_device_temperature(dev["index"]))
"""
from __future__ import annotations

from pydantic import *  # noqa: F401  re-exported
from typing import *

from ._callable import Callable
from ._generable import Generable
from ._librknnsmi import (
    ChipHealthStatus,
    CommMode,
    PrefillMode,
    ResetReason,
    RknnSmi,
    RknnSmiCallError,
    RknnSmiError,
    ThermalMode,
    VersionType,
    WorkMode,
)
from .attachment import Attachment, AudioAttachment, ImageAttachment, TextAttachment
from .lm_output import LMOutput
from .model import ModelSpec, SystemLanguageModel
from .prompt import LocatePrompt, Prompt
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
    "Prompt",
    "LocatePrompt",
    "LLMSession",
    "LocateAnythingBackend",
    "RknnSmi",
    "RknnSmiCallError",
    "RknnSmiError",
    "CommMode",
    "VersionType",
    "ChipHealthStatus",
    "ResetReason",
    "ThermalMode",
    "WorkMode",
    "PrefillMode",
]

__version__ = "0.6.0"
