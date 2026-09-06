"""``@Generable`` schema for ``<ref>...</ref><box>...</box>`` grounding output.

Qwen-VL (and similar VLM families) emit a text response that combines a
natural-language explanation with one or more ``<box>...</box>`` regions,
each preceded by a ``<ref>label</ref>`` tag pointing back at the original
phrase. Parsing this back into structured fields is error-prone because
the model can produce slight surface variations (``<box>`` / ``<box> ... />``
/ extra whitespace) and interleave multiple boxes inside one paragraph.

This module exposes two ``@Generable`` schemas that capture the structured
result of that parse:

* ``BoundingBox`` — one ``<box>`` region with its ``label`` and four
  0–1000 normalised coordinates (``x1`` / ``y1`` / ``x2`` / ``y2``). The
  coordinate range mirrors Qwen-VL's own normalised output (the model is
  trained to emit values in ``[0, 1000]`` regardless of source
  resolution), so callers can multiply by the actual image dimensions
  later.
* ``GroundingResult`` — the top-level payload. Carries the original
  grounding text the model echoed, the parsed boxes, and a ``level``
  discriminator. Extends :class:`wdnai.LMOutput` so the SDK can attach
  binary payloads (e.g. an annotated JPEG with the boxes drawn on top)
  alongside the structured fields.

The schemas follow the existing ``@Generable`` convention: they look like
plain annotated classes and the decorator promotes them to pydantic
``BaseModel`` (or, for ``GroundingResult``, to a subclass of
``LMOutput``). All field descriptions are written in Chinese to match the
existing project style (see ``_build_target_doc_class`` in
``app/services/detector.py`` and the ``qwen3_vl_image`` examples).
"""
from __future__ import annotations

from typing import List, Literal

from pydantic import Field

from ._generable import Generable
from .lm_output import LMOutput


@Generable
class BoundingBox:
    """单个 ``<box>`` 区域：类别 + 0–1000 归一化的左上/右下坐标。"""

    label: str = Field(description="目标类别名（对应 ``<ref>...</ref>`` 标签文本）。")
    x1: float = Field(description="ROI 左上角 x，0–1000 归一化。", ge=0, le=1000)
    y1: float = Field(description="ROI 左上角 y，0–1000 归一化。", ge=0, le=1000)
    x2: float = Field(description="ROI 右下角 x，0–1000 归一化。", ge=0, le=1000)
    y2: float = Field(description="ROI 右下角 y，0–1000 归一化。", ge=0, le=1000)


@Generable
class GroundingResult(LMOutput):
    """grounding 模型的结构化输出：所有解析出的 ``<box>`` 列表 + 原始文本 + 等级标签。"""

    boxes: List[BoundingBox] = Field(description="从模型输出中解析出的所有 ``<box>`` 区域。")
    text: str = Field(description="模型回显的原始 grounding 文本（含 ``<ref>``/``<box>`` 标记）。")
    level: Literal["Normal", "Warning"] = Field(
        default="Normal", description="事件等级标签；常规观察为 ``Normal``，异常/风险提示为 ``Warning``。",
    )
