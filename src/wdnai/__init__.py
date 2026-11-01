"""wdnai — LLM / VLM / TTS access SDK for Rockchip NPUs (librknn3 ctypes).

The SDK talks to librknn3 (`/usr/lib/librknn3_api.so`) directly via the
vendored ctypes shim in :mod:`wdnai._librknn`; it does not depend on
the ``rknn3lite`` / ``rknn3-toolkit-lite`` Python wheel. See AGENTS.md
for the design notes.

Quick start (Qwen3-VL image):

    from typing import Literal
    from wdnai import (
        Generable, Field, ImageAttachment, LMOutput, LLMLocalSession,
        SystemLanguageModel,
    )

    @Generable
    class Event(LMOutput):
        act: str = Field(description="action")
        level: Literal["Normal", "Warning"] = Field(default="Normal")

    model = SystemLanguageModel.qwen3_vl_4b()
    session = LLMLocalSession(model=model)
    out = await session.respond(
        prompt="Analyze this image",
        attachment=[ImageAttachment(img)],
        generating=Event,
    )

``LLMSession`` is the transport-agnostic interface; ``LLMLocalSession``
runs the model in-process on a local NPU. Remote execution over DDS is
provided by ``wdndds.LLMRPCSession``, which subclasses the same interface —
so ``generating=`` / ``tools=`` / ``on_delta=`` are spelled identically
whichever one you hold.

NPU status / telemetry (memory, temperature, device list, version,
reset) is exposed via :class:`wdnai.RknnSmi`, a ctypes shim over
``/usr/lib/librknnsmi.so`` — the same C ABI the ``rknn-smi`` CLI uses,
without shelling out to a subprocess:

    from wdnai import RknnSmi
    smi = RknnSmi()
    for dev in smi.get_device_list():
        print(dev["index"], dev["mode"], smi.get_device_temperature(dev["index"]))

``get_device_temperature()`` 返回 °C；设备级命令在部分 EP 固件上没人应答
（RK1828），此时自动回退到 chip 级读取，调用方无须关心两级差异。
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
from .capability import (
    LEGACY_ALIASES,
    RETIRED_CAPABILITIES,
    RetiredCapabilityError,
    SoftwareCapability,
    known_values,
    migrate_legacy,
)
from .lm_output import LMOutput
from .model import ModelSpec, SystemLanguageModel
from .prompt import LocatePrompt, Prompt, build_chatml_prompt
from .session import LLMLocalSession, LLMSession
from .backends.locate_anything import LocateAnythingBackend
from .backends.depth_anything import DepthAnythingBackend, DepthResult

__all__ = [
    "BaseModel",
    "Field",
    "Generable",
    "Callable",
    "Attachment",
    "TextAttachment",
    "ImageAttachment",
    "AudioAttachment",
    "SoftwareCapability",
    "RetiredCapabilityError",
    "migrate_legacy",
    "known_values",
    "LEGACY_ALIASES",
    "RETIRED_CAPABILITIES",
    "LMOutput",
    "ModelSpec",
    "SystemLanguageModel",
    "Prompt",
    "LocatePrompt",
    "build_chatml_prompt",
    "LLMSession",
    "LLMLocalSession",
    "LocateAnythingBackend",
    "DepthAnythingBackend",
    "DepthResult",
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

__version__ = "0.7.0"
