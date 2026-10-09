"""Prompt 构造器。

基类 ``Prompt`` 提供 ``str()`` 输出最终提示词（给 ``session.respond(prompt=...)``
或 ``LocateAnythingBackend.run_images(prompt=...)`` 使用）。子类在构造时
接收用户友好的字段，``__str__`` 套出 LocateAnything 模型训练时固定下来
的 prompt 模板。

设计动机：LocateAnything / Qwen-VL / Qwen3-VL 等 grounding / VLM 模型的
输出格式高度依赖训练时固定下来的 prompt 模板 + token-level 分隔符
（``</c>`` 是 LocateAnything tokenizer 的 special token，见
``src/wdnai/gguf_tokenizer.py:87``，模型训练时多类就是用 ``</c>`` 分隔）。
自由字符串容易跑偏 —— 例如逗号分隔多类时，模型会把整段当成单个
``<ref>...</ref>`` 标签（实测样本里 ``bottle, earth`` 输出
``<ref>bottle, earth</ref><box>...``，下游 SDK 解析器忠实地把
``bottle, earth`` 当成单个 label）。集中放在构造器里，wdnwebui
后续上 grounding 页时直接复用。

用法::

    from wdnai import LocatePrompt

    p = LocatePrompt(["car", "person", "dog"])
    print(str(p))
    # Locate all the instances that matches the following description: car</c>person</c>dog.

    p = LocatePrompt(["the front-most car"], mode="single")
    print(str(p))
    # Locate a single instance that matches the following description: the front-most car.

    p = LocatePrompt(["car"], mode="point")
    print(str(p))
    # Point to: car.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Prompt(ABC):
    """提示词基类。

    子类负责把构造参数渲染成完整字符串；调用方用 ``str(prompt)`` 拿到
    最终给 LLM 的文本。``__str__`` 是 SDK 实际消费的入口（``session.respond``
    的 ``prompt=`` 参数需要 ``str``）。
    """

    @abstractmethod
    def __str__(self) -> str:
        """返回构造好的完整提示词。"""
        raise NotImplementedError


# LocateAnything tokenizer 的多类分隔 special token（见
# wdnai/gguf_tokenizer.py:87，注册成 ``additional_special_tokens``，
# ByteLevel pre-tokenizer 不会切碎）。模型训练时多类标签就是用 ``</c>``
# 分隔；paper Table 9 写 "comma-separated" 与训练行为不一致，实际
# prompt 必须用 ``</c>``。
_CATEGORY_SEP = "</c>"


class LocatePrompt(Prompt):
    """LocateAnything 视觉定位提示词构造器。

    模板对齐 C++ demo ``examples/LocateAnything/cpp/main.cc`` + paper
    arXiv:2605.27365v2 Appendix B.3 (Table 9) 的 task prompt。
    ``items`` 是字符串数组，含义由 ``mode`` 决定：

    * ``"all"``（默认）— Object Detection。多类用 ``</c>`` 分隔::

          Locate all the instances that matches the following description: <a></c><b></c><c>.

      （"matches" 是 paper 的原文 typo；模型训练时两种都见过，
      这里沿用 paper 写法以保证一致。）

    * ``"single"`` — Phrase Grounding Single Box。``items`` 必须单元素::

          Locate a single instance that matches the following description: <item>.

    * ``"multi"`` — Phrase Grounding Multiple Boxes。整段拼成一个 phrase，
      用 ``</c>`` 分隔::

          Locate all the instances that match the following description: <a></c><b>.

    * ``"point"`` — Pointing。``items`` 单元素::

          Point to: <item>.

    * ``"region"`` — GUI Grounding (Box)。``items`` 单元素::

          Locate the region that matches the following description: <item>.

    * ``"text_ref"`` — Text Grounding。给定文本找位置（``</c>`` 分隔）::

          Please locate the text referred as <a></c><b>.

    * ``"scene_text"`` — Scene Text Detection。``items`` 忽略::

          Detect all the text in box format.

    * ``"layout"`` — Document Layout Analysis。多类用 ``</c>`` 分隔::

          Detect all the objects in the image that belong to the category set: <a></c><b>.

    多类一律用 ``</c>`` 分隔（tokenizer 已注册 special token，模型
    训练数据也是 ``</c>``；paper 文字写 "comma-separated" 与训练不
    一致 —— 走逗号会被模型吃成单个 label）。
    """

    _MODES = ("all", "single", "multi", "point", "region", "text_ref", "scene_text", "layout")

    def __init__(self, items: list[str], mode: str = "all"):
        if not isinstance(items, (list, tuple)):
            raise TypeError(f"items 必须是 list[str]，收到 {type(items).__name__}")
        if mode not in self._MODES:
            raise ValueError(f"mode 必须是 {self._MODES} 之一，收到 {mode!r}")
        cleaned = [str(x).strip() for x in items]
        if mode in ("single", "point", "region") and len(cleaned) != 1:
            raise ValueError(
                f"mode={mode!r} 要求 items 恰好 1 个元素，收到 {len(cleaned)} 个"
            )
        if mode in ("scene_text",) and not cleaned:
            raise ValueError("mode='scene_text' 至少要 1 个 item（即便模板不展开）")
        if mode in ("all", "multi", "layout", "text_ref") and not cleaned:
            raise ValueError(f"mode={mode!r} 至少要 1 个 item")
        self.items: list[str] = cleaned
        self.mode: str = mode

    def _joined(self) -> str:
        return _CATEGORY_SEP.join(self.items)

    def __str__(self) -> str:
        if self.mode == "all":
            return f"Locate all the instances that matches the following description: {self._joined()}."
        if self.mode == "single":
            return f"Locate a single instance that matches the following description: {self._joined()}."
        if self.mode == "multi":
            return f"Locate all the instances that match the following description: {self._joined()}."
        if self.mode == "point":
            return f"Point to: {self._joined()}."
        if self.mode == "region":
            return f"Locate the region that matches the following description: {self._joined()}."
        if self.mode == "text_ref":
            return f"Please locate the text referred as {self._joined()}."
        if self.mode == "scene_text":
            return "Detect all the text in box format."
        if self.mode == "layout":
            return f"Detect all the objects in the image that belong to the category set: {self._joined()}."
        raise AssertionError(f"unreachable mode: {self.mode!r}")


# --------------------------------------------------------------------------- #
# Multi-turn ChatML prompt assembly
# --------------------------------------------------------------------------- #
# Reinforcement appended after the tool block so the model emits exactly
# ONE tool_call JSON object in the answer segment, avoiding the
# "example first, real one second" multi-JSON failure mode. The parser
# (wdnai._callable.try_parse_tool_call) takes the first match as a backstop.
_TOOL_CALL_INSTRUCTION = (
    "When you decide to call a tool, output **exactly one** JSON object "
    "matching the format above. Do not output multiple JSON objects, "
    "placeholder/example JSONs, or any text before/after the JSON."
)


def build_chatml_prompt(history: List[dict], user_text: str,
                        tool_specs: Optional[List[dict]] = None) -> str:
    """拼 ChatML 文本 prompt（多轮对话）。

    ``tool_specs``（可选，``get_tool_specs(tools)`` 返回值）非空时，
    渲染 wdnai 的 ``render_tools_block`` 并以独立段形式追加到 system
    段之后、对话历史之前（与 ``LLMSession._compose_prompt`` 单轮路径把
    工具块追加到末尾不同——多轮对话走独立段，结构清晰）。模型只需看到
    工具描述，无需被历史 / 用户问题注意力分散。

    同时追加 ``_TOOL_CALL_INSTRUCTION`` 强化指令，让 LLM 在 answer 段
    只输出**单个** tool_call JSON。

    输出是**裸 ChatML**（无 ``<|im_*|>`` marker）—— marker 由 C 侧的
    chat template 负责插入，见 ``backends._common.apply_chat_template``。
    """
    # Lazy import: _common pulls in the RKNN shim, and prompt.py is
    # imported by the session layer before backends are built.
    from ._callable import render_tools_block
    from .backends._common import (
        CHATML_TURN_ASSISTANT, CHATML_TURN_SYSTEM, CHATML_TURN_USER,
    )

    parts = [f"{CHATML_TURN_SYSTEM}You are a helpful assistant.\n"]
    if tool_specs:
        parts.append("\n")
        parts.append(render_tools_block(tool_specs))
        parts.append("\n\n")
        parts.append(_TOOL_CALL_INSTRUCTION)
        parts.append("\n\n---\n\n")
    for msg in history:
        role = msg.get("role")
        if role == "user":
            parts.append(f"{CHATML_TURN_USER}{msg.get('content') or ''}\n")
        elif role == "assistant":
            parts.append(f"{CHATML_TURN_ASSISTANT}{msg.get('content') or ''}\n")
    parts.append(f"{CHATML_TURN_USER}{user_text}\n")
    parts.append(CHATML_TURN_ASSISTANT)
    return "".join(parts)


__all__ = ["Prompt", "LocatePrompt", "build_chatml_prompt"]