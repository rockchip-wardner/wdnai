"""``LMOutput`` — base class for structured LLM / VLM / TTS outputs.

Adds a binary ``attachment`` slot to a pydantic ``BaseModel`` so a
session can return free-form payloads (audio bytes from TTS, decoded
images from a multimodal head, raw JSON blobs, …) alongside the
parsed structured fields. The SDK fills ``attachment`` automatically
when the backend produces any.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field

from .attachment import Attachment


class LMOutput(BaseModel):
    """Structured output with optional binary attachments."""

    # ``Attachment`` isn't a pydantic model; tell pydantic to skip
    # schema introspection (we never serialise attachments to JSON).
    model_config = ConfigDict(arbitrary_types_allowed=True)

    attachment: List[Attachment] = Field(
        default_factory=list,
        description="Output attachments. Populated by the session after respond().",
    )
