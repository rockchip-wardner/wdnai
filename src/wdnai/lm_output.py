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

    # ``Attachment`` and its subclasses aren't pydantic models; tell
    # pydantic to skip schema introspection (we never serialise
    # attachments to JSON). Any subclass of :class:`Attachment`
    # (TextAttachment / ImageAttachment / AudioAttachment) is accepted
    # in the list at runtime — the type hint is just a hint for static
    # checkers.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    attachment: List[Attachment] = Field(
        default_factory=list,
        description=(
            "Output attachments. Populated by the session after respond(). "
            "In practice this list contains AudioAttachment instances for "
            "TTS / Omni outputs, ImageAttachment for VLM-decoded images, "
            "TextAttachment for free-form text payloads, or a mix thereof."
        ),
    )
