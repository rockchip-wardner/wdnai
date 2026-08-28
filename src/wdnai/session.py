"""LLM / VLM / TTS session: routes a list of ``Attachment`` objects and
supports structured output (``LMOutput`` subclasses + plain ``@Generable``
classes alike)."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np

from pydantic import BaseModel

from ._callable import (
    dispatch_tool_call,
    get_tool_specs,
    render_tools_block,
    try_parse_tool_call,
)
from .attachment import Attachment
from .backends import build_backend
from .lm_output import LMOutput
from .model import ModelSpec
from .utils import extract_json, strip_thinking_blocks

T = TypeVar("T", bound=BaseModel)


class LLMSession:
    """Session with a single RKNN model.

    Usage::

        model = SystemLanguageModel.qwen3_vl_4b().with_paths(
            rknn_llm_path=..., rknn_vision_path=...,
            tokenizer_path=..., embed_path=...,
        )
        async with LLMSession(model=model) as sess:
            out = await sess.respond(prompt="...", attachment=[Attachment(img)])

    ``attachment`` is now a **list** of :class:`Attachment` (or a single
    ``Attachment`` for backward compat). This lets one session take in
    heterogeneous payloads — e.g. a text prompt + an image for VLM,
    or an audio config + a text instruction for TTS.

    ``generating`` accepts either a ``@Generable``-decorated class or a
    subclass of :class:`LMOutput`. ``LMOutput`` carries an
    ``attachment`` field; when a TTS-style backend produces audio
    bytes, the SDK fills it in automatically.

    NPU selection: pass exactly one of ``npu=`` (zero-based index into
    ``RKNN3Lite().get_devices_id()``) or ``device_id=`` (raw device
    identifier — bytes / str — e.g. a board serial for local
    coprocessors, or an SoC name like ``"rk3588"`` for ADB targets).
    Both ``None`` selects the default device. Resolution happens
    inside the backend's ``__init__`` so the RKNN3 native driver
    init stays lazy and never races with a co-resident ``wdnav``
    process.
    """

    def __init__(
        self,
        model: ModelSpec,
        *,
        npu: Optional[int] = None,
        device_id: Optional[Union[bytes, str]] = None,
        **options: Any,
    ) -> None:
        if npu is not None and device_id is not None:
            raise ValueError("npu= and device_id= are mutually exclusive")
        if isinstance(device_id, str):
            device_id = device_id.encode("utf-8")
        self.spec = model
        self.npu = npu
        self.device_id = device_id
        self._backend = build_backend(model, npu=npu, device_id=device_id)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _coerce_attachments(
        attachment: Union[None, Attachment, List[Attachment]],
    ) -> List[Attachment]:
        """Accept ``None`` / a single ``Attachment`` / a list of them.

        Older call sites pass a single ``Attachment``; new ones pass a
        list. Both are accepted; we normalise to ``List[Attachment]``.
        """
        if attachment is None:
            return []
        if isinstance(attachment, Attachment):
            return [attachment]
        if isinstance(attachment, list):
            return attachment
        raise TypeError(
            "attachment must be None, an Attachment, or a list of Attachments; "
            f"got {type(attachment).__name__}"
        )

    # ------------------------------------------------------------------ #
    async def respond(
        self,
        prompt: Optional[str] = None,
        attachment: Union[None, Attachment, List[Attachment]] = None,
        generating: Optional[Type[T]] = None,
        debug: bool = False,
        tools: Optional[List[Any]] = None,
        perf: bool = False,
        **overrides: Any,
    ) -> Any:
        """Fire one request and return the result.

        - ``attachment`` — list (or single) of :class:`Attachment`. Any
          text attachments are prepended to the prompt; images are
          forwarded to the VLM backend; audio/video attachments are
          handled by their respective backends.
        - ``generating`` — ``@Generable`` class or :class:`LMOutput`
          subclass. When given, the model is asked to emit JSON (auto-
          appended schema hint) and the result is parsed into an
          instance of the class. ``LMOutput`` subclasses additionally
          have ``attachment`` populated from any output attachments
          the backend produced.
          - ``debug=True`` — additionally prints the raw user input
          (``prompt=``), the actual ``full_prompt`` sent to the LLM,
          the raw model output, the parsed result, and the backend
          perf table.
          - ``perf=True`` — print the backend perf table (Prefill /
          Generate / vision / total latency + FPS). Default ``False``;
          the data is still recorded on ``session._backend.sink.perf``
          even when the table is suppressed.
        - ``tools`` — a list of ``@Callable``-decorated classes. Each
          class's field schema is serialised into a JSON tool
          definition and appended to the prompt. When the model emits
          a tool-call JSON (``{"name": ..., "arguments": ...}``), a
          *populated instance* of the matching ``@Callable`` class is
          returned to the caller. The SDK does NOT auto-invoke —
          ``isinstance(result, AlarmCls)`` identifies which tool the
          model picked, and ``result()`` triggers the side effect on
          the developer-defined ``__call__``. When ``generating=`` is
          also supplied, the model may instead respond with the
          structured payload — in that case the parsed ``generating=``
          instance is returned. The caller picks the branch via
          ``isinstance``.
        """
        attachments = self._coerce_attachments(attachment)

        # ``tools=`` is a list of ``@Callable`` classes. Validate up
        # front so a non-decorated entry fails before we burn an
        # inference cycle.
        tool_specs: List[Dict[str, Any]] = get_tool_specs(tools) if tools else []
        # Forward the perf flag to the backend so it can suppress the
        # perf-table printout when the caller doesn't want it.
        overrides.setdefault("perf", perf)

        full_prompt = self._compose_prompt(prompt, generating, tool_specs)
        if debug:
            print(f"[wdnai.debug] prompt      = {prompt!r}")
            print(f"[wdnai.debug] full_prompt = {full_prompt!r}")
            print(f"[wdnai.debug] attachments={[a.to_payload() for a in attachments]}")
            print(f"[wdnai.debug] tools  = {tool_specs}")
        text, attachments_out = self._dispatch(full_prompt, attachments, **overrides)

        if debug:
            print(f"[wdnai.debug] raw    = {text!r}")
            sink = getattr(self._backend, "sink", None)
            if sink is not None and sink.perf:
                print(f"[wdnai.debug] perf   = {sink.perf}")

        # Output classification — passive, the caller decides what to do:
        #
        # 1. If ``tools=[…]`` was passed AND the model emitted a
        #    tool-call JSON (``{"name": ..., "arguments": ...}``),
        #    return a populated instance of the matching ``@Callable``
        #    class (so ``isinstance(result, Alarm)`` discriminates
        #    between multiple tools). The SDK does NOT auto-invoke;
        #    the user calls ``result()`` (or any other method) to
        #    trigger side effects.
        # 2. Otherwise run the regular parse path, which honours
        #    ``generating=…`` (returns the parsed instance) or falls
        #    back to the raw text when no schema was supplied.
        #
        # The two branches coexist: the model picks one or the other
        # depending on its prompt, and the caller inspects ``result``
        # to decide what to do.
        if tools and tool_specs:
            tool_call = try_parse_tool_call(text)
            if tool_call is not None:
                name, arguments = tool_call
                instance = dispatch_tool_call(tools, name, arguments)
                if instance is not None:
                    if debug:
                        print(f"[wdnai.debug] tool_call -> {instance!r}")
                    return instance
                if debug:
                    print(f"[wdnai.debug] tool_call: no @Callable named {name!r} "
                          f"(or args failed validation); will try parsing "
                          f"arguments as the generating= schema")

                # NEW (thinking-mode hallucinated-tool-name recovery):
                # Qwen3 / Qwen3.5 and other reasoning models sometimes
                # wrap the schema payload in tool-call JSON with a
                # hallucinated ``name`` that doesn't match any registered
                # tool. Without this fallback the dispatch silently
                # returns None and we fall through to ``_parse_output``
                # with the *outer* dict (``{"name": "...", "arguments":
                # {...}}``), which the schema validator then rejects —
                # and the schema-fallback path would stuff the entire
                # raw text (think-block included, before the strip fix)
                # into the schema's ``text`` field.
                #
                # Try validating ``arguments`` against the generating=
                # schema first. If it validates, return that. If not,
                # continue to the regular JSON-parse path (the dict is
                # still informative for the schema validator even when
                # this branch fails).
                if generating is not None and isinstance(arguments, dict):
                    try:
                        instance = _model_validate(generating, arguments)
                        if debug:
                            print(
                                f"[wdnai.debug] tool_call.arguments "
                                f"validated as {generating.__name__}: "
                                f"{instance!r}"
                            )
                        return instance
                    except Exception as e:  # noqa: BLE001
                        if debug:
                            print(
                                f"[wdnai.debug] tool_call.arguments "
                                f"failed {generating.__name__} validation: "
                                f"{type(e).__name__}: {e}"
                            )

        result = self._parse_output(
            text, generating=generating, attachments_out=attachments_out,
        )
        if debug:
            print(f"[wdnai.debug] parsed = {result!r}")
        return result

    # ------------------------------------------------------------------ #
    def _dispatch(
        self,
        prompt: str,
        attachments: List[Attachment],
        **overrides: Any,
    ) -> Tuple[str, List[Attachment]]:
        """Route ``prompt`` + a list of attachments to the right backend
        method. Returns ``(text, attachments_out)`` — ``attachments_out``
        is populated by backends that produce binary outputs (e.g. TTS
        audio bytes); today it is always empty for the VLM backend.
        """
        if not attachments:
            run = getattr(self._backend, "run_text", None) or self._backend.run
            return run(prompt, **overrides), []

                # Split attachments by channel. Audio is routed to
        # ``backend.run_audio`` when the backend exposes one (ASR);
        # image / video / text-prefixes work as before; unknown kinds
        # fall through to ``run_text``.
        images: List = []
        video_frames: List = []
        text_prefixes: List[str] = []
        audio_payloads: List[Attachment] = []
        other: List[Attachment] = []

        for att in attachments:
            if att.kind == "image":
                images.extend(att.images)
            elif att.kind == "video":
                video_frames.extend(att.frames)
            elif att.kind == "text":
                text_prefixes.append(att.text or "")
            elif att.kind == "audio":
                audio_payloads.append(att)
            else:
                other.append(att)

        full_prompt = prompt or ""
        if text_prefixes:
            joined = "\n".join(p for p in text_prefixes if p)
            full_prompt = (joined + ("\n" + full_prompt if full_prompt else "")).strip()

        if audio_payloads and hasattr(self._backend, "run_audio"):
            pcm = b"".join(att.data for att in audio_payloads)
            sample_rate = int(overrides.pop("sample_rate", 16000))
            n_channels = int(overrides.pop("n_channels", 1))
            sample_width = int(overrides.pop("sample_width", 2))
            text = self._backend.run_audio(
                pcm,
                sample_rate=sample_rate,
                n_channels=n_channels,
                sample_width=sample_width,
                prompt=full_prompt,
                **overrides,
            )
            return text, []




        if images:
            text = self._backend.run_images(full_prompt, images, **overrides)
            return text, []
        if video_frames:
            text = self._backend.run_video(full_prompt, video_frames, **overrides)
            return text, []
        if other:
            # Pass-through to run_text for now (TTS-style backends will
            # pick the audio attachments up via a future hook).
            result = self._backend.run_text(full_prompt, **overrides)
        else:
            result = self._backend.run_text(full_prompt, **overrides)
        # TTS-style backends return ``(text, [attachment, ...])``; legacy
        # text-only backends return just ``text``. Unpack accordingly.
        if isinstance(result, tuple) and len(result) == 2:
            text, atts = result
            return text, list(atts or [])
        return result, []

    # ------------------------------------------------------------------ #
    # JSON prompt injection
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compose_prompt(
        prompt: Optional[str],
        generating: Optional[Type[T]],
        tool_specs: List[Dict[str, Any]],
    ) -> str:
        """Build the final prompt from the user message, the optional
        ``generating=`` schema hint, and the optional tool block.

        When ``tools=[…]`` and ``generating=…`` are both supplied, the
        model is told to pick exactly one: either call a tool OR emit
        the structured payload. The output-classification step in
        ``respond()`` decides which one the model picked.
        """
        base = (prompt or "").rstrip()
        has_user_prompt = bool(base)
        sections: List[str] = [base] if base else []


        if tool_specs and generating is not None:
            # Both ways of responding are available — put a clear
            # "pick one" instruction up top so the model doesn't
            # silently default to whichever format has the strongest
            # tail instruction.
            #
            # Uses the canonical Qwen3 Hermes-style ``<tool_call>``
            # wrapper (see ``render_tools_block`` for the rationale).
            # The schema hint uses the legacy ``repr(dict)`` shape
            # because Qwen models are sensitive to schema formatting
            # (json.dumps(indent=2) was tried in 141ee17 and produced
            # visibly different output — see ``_safe_schema``).
            schema = _safe_schema(generating)
            choice_block = (
                "# Response format\n"
                "Pick **exactly one** of the following two ways. "
                "Do NOT output both. Do NOT include any explanation.\n\n"
                "**Option A — Call a tool.** Wrap a JSON object with "
                "the function name and arguments in <tool_call></tool_call> "
                "XML tags:\n"
                "<tool_call>\n"
                '{"name": <function-name>, "arguments": <args-json-object>}\n'
                "</tool_call>\n\n"
                "**Option B — Structured response.** Output ONLY a JSON "
                "object matching this schema:\n"
                "```json\n"
                f"{schema}\n"
                "```\n"
            )
            sections.append(choice_block)
            sections.append(render_tools_block(tool_specs))
        else:
            if tool_specs:
                sections.append(render_tools_block(tool_specs))
            if generating is not None:
                # The schema hint is appended as its own section. The
                # choice of default_task must be made here, NOT inside
                # ``_maybe_inject_json_hint`` (which only knows about
                # the prompt it was passed).  When the caller has
                # already supplied a user message in ``sections[0]``,
                # we use the "strictly follow" branch; otherwise we
                # fall back to the placeholder prompt that nudges the
                # model toward a "people / safety" structured output
                # (the original behaviour of the no-user-prompt path).
                hint = LLMSession._schema_hint_section(generating, has_user_prompt)
                if hint:
                    sections.append(hint)

        return "\n\n".join(s for s in sections if s)

    @staticmethod
    def _maybe_inject_json_hint(prompt: Optional[str], generating: Optional[Type[T]]) -> str:
        if generating is None:
            return prompt or ""
        user_msg = (prompt or "").rstrip()
        schema_text = _safe_schema(generating)
        if not schema_text:
            return user_msg
        sep = "\n\n" if user_msg else ""
        hint = LLMSession._schema_hint_section(generating, has_user_prompt=bool(user_msg))
        if not hint:
            return user_msg
        return f"{user_msg}{sep}{hint}"

    @staticmethod
    def _schema_hint_section(generating: Optional[Type[T]], has_user_prompt: bool) -> str:
        """Build the JSON-Schema hint block (default task instruction +
        fenced JSON schema). Returns "" when ``generating`` carries no
        schema.  When the caller has already supplied a user message the
        hint uses the concise "strictly follow" instruction; otherwise it
        uses the placeholder prompt that nudges the model toward a
        "people / safety" structured output.

        Note: the schema is rendered as the Python ``repr`` of the
        pydantic v1 / v2 schema dict (single line, single-quoted keys,
        Python-native types like ``str`` / ``dict``), matching the
        legacy ``_maybe_inject_json_hint`` behaviour from 141ee17.
        ``json.dumps(..., indent=2)`` was tried here and produced
        different model output for the same image — the LLM is
        sensitive to schema formatting; we keep the legacy shape.
        """
        if generating is None:
            return ""
        # Use the same rendering path as the legacy 141ee17 helper:
        # ``generating.schema()`` on pydantic v1 returns a dict, and
        # f"…{schema}…" interpolates it via ``str(dict)`` which is
        # the same as ``repr(dict)`` for plain-dict content.
        # ``_safe_schema`` already normalises both pydantic v1 and v2
        # to the same dict, so this single call works across versions.
        schema_text = _safe_schema(generating)
        if not schema_text:
            return ""
        default_task = (
            "Please strictly follow the JSON Schema below to return the result. "
            "**Output JSON only, no other text**:"
            if has_user_prompt
            else
                "Please carefully observe the behavior of people in the image, "
                "describe it according to the JSON Schema below, and assign a safety level. "
                "**Output JSON only, no other text**:"
        )
        return f"{default_task}\n```json\n{schema_text}\n```"

    @staticmethod
    def _parse_output(
        raw_text: Union[str, Dict[str, Any], BaseModel, tuple],
        generating: Optional[Type[T]] = None,
        attachments_out: Optional[List[Attachment]] = None,
    ) -> Union[str, T, Dict[str, Any]]:
        if generating is None:
            return raw_text  # type: ignore[return-value]

        # TTS-style backends: ``raw_text`` is the echoed prompt string
        # AND ``attachments_out`` is a non-empty list carrying the
        # binary payload (audio bytes for TTS, decoded images for
        # multimodal heads, …). We short-circuit the JSON parse path
        # because TTS output isn't JSON at all.
        #
        # The ``bool(attachments_out)`` check is what makes this branch
        # *opt-in*: VLM / text-only backends return ``[]`` for
        # ``attachments_out``, so they fall through to the regular JSON
        # path below. Without that guard, every LMOutput subclass would
        # be misrouted to the TTS shortcut (because
        # ``issubclass(Event, LMOutput)`` is True even for VLM).
        is_tts_path = (
            issubclass(generating, LMOutput)
            and bool(attachments_out)
        )
        if is_tts_path:
            text_value = raw_text if isinstance(raw_text, str) else (
                raw_text[0] if isinstance(raw_text, tuple) and raw_text else ""
            )
            result = _model_validate(
                generating,
                {"text": text_value, "sample_rate": 24000, "channels": 1},
            )
            if isinstance(result, LMOutput):
                result.attachment = list(attachments_out)
            return result

        if isinstance(raw_text, tuple) and attachments_out:
            text, _attachments = raw_text
            if issubclass(generating, LMOutput):
                result = _model_validate(
                    generating,
                    {"text": text, "sample_rate": 24000, "channels": 1},
                )
                if isinstance(result, LMOutput):
                    result.attachment = list(attachments_out)
            else:
                result = text
            return result

        if isinstance(raw_text, generating):
            result = raw_text
        elif isinstance(raw_text, dict):
            result = _safe_validate(generating, raw_text, raw_text)
        elif isinstance(raw_text, str):
            data = extract_json(raw_text)
            if isinstance(data, dict):
                result = _safe_validate(generating, data, raw_text)
            elif data is None:
                # The model didn't emit valid JSON. Hand back a
                # model instance that carries the raw text in a
                # ``text`` field. If the class doesn't declare ``text``,
                # we fall back to ``model_construct(**{"text": ...})`` -
                # pydantic-v2 lets us inject an undeclared field via
                # ``model_config = ConfigDict(extra="allow")`` on the
                # class; otherwise we construct a bare instance and
                # set ``text`` as a normal Python attribute.
                result = _construct_text_instance(generating, raw_text)
                if result is None:
                    return raw_text  # type: ignore[return-value]
            else:
                result = _model_validate_json(generating, data)
        else:
            raise TypeError(
                f"Cannot parse {type(raw_text).__name__!r} as {generating.__name__}"
            )

        # LMOutput subclasses have an ``attachment`` slot that the SDK
        # fills from any binary payload the backend produced (audio
        # bytes from TTS, decoded images from a diffusion head, …).
        if isinstance(result, LMOutput) and attachments_out:
            result.attachment = list(attachments_out)
        return result

    # ------------------------------------------------------------------ #
    async def __aenter__(self) -> "LLMSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
            self._backend = None

    # ------------------------------------------------------------------ #
    # Embedding passthroughs (kind="embedding" backend only)
    # ------------------------------------------------------------------ #
    def embed_text(self, text: str) -> np.ndarray:
        """Return a normalized embedding vector for a text query."""
        return self._backend.embed_text(text)

    def embed_video(self, frames: List[np.ndarray], prompt: str = "") -> np.ndarray:
        """Return a normalized embedding vector for a list of BGR video frames."""
        return self._backend.embed_video_chunk(frames, text=prompt)

    # ------------------------------------------------------------------ #
    # Stability helpers
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Clear the C library's KV cache / internal state.

        Call this between loop iterations to force the model to start from a
        clean state — trades a bit of speed for determinism.
        """
        rknn_llm = getattr(self._backend, "rknn_llm", None)
        if rknn_llm is not None and hasattr(rknn_llm, "clear_kvcache"):
            try:
                rknn_llm.clear_kvcache()
            except Exception:
                pass

    def majority_vote(self, k: int, **kwargs: Any) -> Any:
        """Run the current ``prompt`` + attachment ``k`` times synchronously and return the most common result.

        Implementation: ``k`` sequential ``session_run`` calls, then de-duplicate
        by ``repr`` and pick the most frequent entry. ~``k``x slower than a single
        inference, but much more robust against occasional NPU mis-decodes.

        ⚠️ This function is synchronous (it uses ``asyncio.run`` to drive
        ``respond``), so it's meant for scripts that are not already inside an
        event loop. If you already are, do ``asyncio.gather`` of ``k`` ``respond``
        calls yourself and tally the results.
        """
        import asyncio
        from collections import Counter

        async def _run_once():
            return await self.respond(**kwargs)

        results = [asyncio.run(_run_once()) for _ in range(k)]
        keys = [repr(r) for r in results]
        top = Counter(keys).most_common(1)[0][0]
        for r in results:
            if repr(r) == top:
                return r
        return results[0]


# --------------------------------------------------------------------------- #
# Compatibility shim: pydantic v1 uses parse_obj / parse_raw, v2 uses
# model_validate / model_validate_json
# --------------------------------------------------------------------------- #
def _safe_schema(generating: Optional[Type[T]]) -> str:
    """Render ``generating``'s pydantic schema as a string for use in
    the JSON-Schema hint block. Falls back to an empty string when
    the class can't produce a schema (e.g. it carries a non-pydantic
    ``attachment`` field). Never raises.

    The rendering is the Python ``str(dict)`` of the schema — i.e.
    single line, single-quoted keys, Python-native type names like
    ``str`` / ``dict``. This matches the legacy ``_maybe_inject_json_hint``
    helper from 141ee17 exactly (``f"{user_msg}\n\n{default_task}\\n
    ```json\\n{schema}\\n```"`` with ``schema`` interpolated as a
    dict). The Qwen3-VL model is sensitive to schema formatting —
    ``json.dumps(..., indent=2)`` was tried here and produced visibly
    different output for the same image, so we keep the legacy shape.
    """
    if generating is None:
        return ""
    try:
        schema = (
            generating.model_json_schema()
            if hasattr(generating, "model_json_schema")
            else (
                generating.schema()
                if hasattr(generating, "schema")
                else None
            )
        )
    except Exception:
        return ""
    if not schema:
        return ""
    try:
        return str(schema)
    except Exception:
        return ""


def _model_validate(model: Type[T], data: Dict[str, Any]) -> T:
    validator = getattr(model, "model_validate", None) or getattr(model, "parse_obj", None)
    if validator is None:  # pragma: no cover
        raise RuntimeError("No suitable pydantic validator found")
    return validator(data)  # type: ignore[return-value]


def _model_validate_json(model: Type[T], data: str) -> T:
    validator = getattr(model, "model_validate_json", None) or getattr(model, "parse_raw", None)
    if validator is None:  # pragma: no cover
        raise RuntimeError("No suitable pydantic validator found")
    return validator(data)  # type: ignore[return-value]


def _construct_text_instance(
    model: Type[T], raw_text: str,
) -> Optional[T]:
    """Build a model instance that carries ``raw_text`` in a ``text`` field.

    Used as a fallback when the model output doesn't validate against the
    schema: either the model returned a dict with the wrong field names
    (e.g. ``{"description": …}`` instead of ``{"text": …}``) or it didn't
    emit valid JSON at all. We preserve the model's content so the caller
    doesn't lose the description / 推理内容, and we keep the schema
    instance shape so downstream code (``isinstance(result, VLMDoc)``,
    ``getattr(result, "level", None)``, etc.) keeps working.

    Returns ``None`` when the model doesn't declare a ``text`` field *and*
    doesn't allow extras — in that case the caller should fall back to
    returning the raw string.

    **v0.x 修复**：原来对没有 ``text`` 字段的 schema（如 ``TargetDoc`` 的
    ``x1/y1/x2/y2/desc``）会 fallback ``model.model_construct(text=raw_text)``。
    pydantic 默认 ``extra="ignore"`` 会把这个 ``text`` 当 extra 字段收下，
    返回的实例 ``type().__name__ == "TargetDoc"`` 但所有 schema 字段保持
    默认值（``model_dump() = {}``）—— 调用方（如 compute_worker 的
    ``infer_handler``）看到 "TargetDoc instance" 就 ``reply["result_data"] =
    _dump``，主进程 proxy 端 ``TargetDoc.model_validate({})`` 抛 5 个字段
    missing 的 ValidationError，fallback 到 ``_ProxyResult(text=None)``，
    detector 端既拿到空 text 又不是 TargetDoc 实例，最终 silently drop
    整帧。对没有 ``text`` 字段的 schema，**必须返回 None**，让 caller 走
    raw string 路径 —— 调用方可以自己读 raw text / 决定要不要报错，至少
    不再假装 parse 成功了。
    """
    try:
        if "text" in getattr(model, "model_fields", {}):
            # Strip ``<think>...</think>`` / ``<tool_call>...</tool_call>``
            # wrappers / ```json``` fences from the raw text before
            # stuffing it into the schema's ``text`` field. Without
            # this, the second-stage OCR call (which has no tools=
            # and so gets no schema hint re-injection) sometimes
            # receives the model's literal think block + JSON fence
            # as the "plate number", e.g.
            # ``Event(text='<think>...</think>\n```json\n{"text":...}\n```')``.
            clean_text = _strip_fence_for_fallback(raw_text)
            return model.model_construct(
                _fields_set={"text"}, text=clean_text,
            )
        # No declared ``text`` field: cannot meaningfully carry the raw
        # text on this schema without misleading the caller. Return None
        # so ``_parse_output`` falls back to returning the raw string
        # unchanged.
        return None
    except Exception:
        return None


def _extract_text_from_mismatched_dict(
    data: Dict[str, Any], _depth: int = 0,
) -> Optional[str]:
    """Best-effort: find the model's actual text content in a dict that
    failed schema validation. Recurses into nested dicts up to depth 2.

    The original implementation only scanned the top level. That
    silently missed the common case where the model wraps the
    schema fields inside ``{"arguments": {...}}`` (i.e. tool-call
    JSON whose tool name didn't match anything registered, so the
    SDK fell through to the schema parse path) — the actual content
    is in ``arguments["text"]``, not at the top level. With
    recursion, we recover the model's intent instead of falling
    back to the raw-text dump (which would otherwise include the
    ``<think>`` reasoning block if stripping was incomplete).

    Container keys we drill into: ``arguments``, ``args``,
    ``parameters``, ``response``, ``result``, ``data``, ``output``.
    Depth is capped at 2 so a pathological / cyclic payload can't
    hang the caller; in practice a depth-1 recursion already covers
    every tool-call-style wrap we see in practice.
    """
    if _depth > 2 or not isinstance(data, dict):
        return None
    aliases = ("text", "description", "content", "message", "caption", "summary")
    for key in aliases:
        v = data.get(key)
        if isinstance(v, str) and v.strip():
            return strip_thinking_blocks(v)
    # Recurse into known container keys (tool-call-style wraps). We do
    # this BEFORE the "first non-trivial string" fallback because Qwen
    # models sometimes echo the schema envelope ``{"Event": {...}}``
    # when they lose the schema hint — recursing first finds the inner
    # ``text`` field instead of returning the bare key name "Event".
    container_keys = (
        "arguments", "args", "parameters",
        "response", "result", "data", "output",
    )
    for k in container_keys:
        sub = data.get(k)
        if isinstance(sub, dict):
            found = _extract_text_from_mismatched_dict(sub, _depth=_depth + 1)
            if found is not None:
                return found
    # First non-trivial string field that isn't a severity / enum-ish key.
    # Also recurse into nested dicts here (catches ``{"Event": {"text":
    # ...}}`` and similar envelope echoes that aren't in ``container_keys``).
    skip = {
        "level", "severity", "type", "category", "tag",
        "name", "tool", "function",  # tool-call identifier keys
    }
    for k, v in data.items():
        if k in skip:
            continue
        if isinstance(v, dict):
            found = _extract_text_from_mismatched_dict(v, _depth=_depth + 1)
            if found is not None:
                return found
        if isinstance(v, str) and len(v.strip()) >= 4:
            return strip_thinking_blocks(v)
    return None


def _strip_fence_for_fallback(text: str) -> str:
    """Strip markers that would pollute the schema's ``text`` field
    when the schema validation fails and we fall back to the raw
    model output as the schema's text content.

    Three classes of markers are stripped:

    1. ``<think>...</think>`` blocks — model reasoning the user
       shouldn't see (already done by
       :func:`wdnai.utils.strip_thinking_blocks`; we call it here
       defensively so the fallback path is self-contained).
    2. `````json`` / `````` Markdown fences wrapping the JSON
       payload — without this strip, the schema's text would carry
       the literal ```json`` prefix.
    3. ``<tool_call>`` / ``</tool_call>`` XML wrappers — the model
       uses these to delimit tool calls (Hermes / Qwen-Agent style);
       the JSON between them is meaningful but the tags themselves
       are noise in the user-facing fallback.

    The remaining text is whitespace-collapsed at the edges but the
    inner JSON object is *kept* — we don't try to extract and reformat
    because the fallback is a last-resort dump, not a structured parse.
    """
    if not text:
        return text
    out = strip_thinking_blocks(text)
    # Markdown ```json...``` / ```...``` fences (greedy across newlines
    # so multi-line JSON fences are handled in one pass).
    out = re.sub(r"```(?:json)?\s*\n?", "", out)
    out = re.sub(r"\s*```", "", out)
    # Lone ``<tool_call>`` / ``</tool_call>`` XML tags. The JSON
    # between them was kept by ``strip_thinking_blocks`` (which only
    # strips the tags themselves); this regex is a no-op safety net
    # for any stragglers if that behaviour changes.
    out = re.sub(r"</?tool_call>", "", out)
    return out.strip()


def _safe_validate(
    model: Type[T], data: Dict[str, Any], raw_text_for_fallback: str,
) -> T:
    """Run ``_model_validate`` and fall back on ``ValidationError``.

    Some VLM backends occasionally emit a JSON object whose keys don't
    match the schema (e.g. legacy ``{"description": …}`` field name, or
    extra fields the model added on its own). Without this guard the
    whole ``sess.respond`` call raises ``pydantic_core.ValidationError``
    and the worker's ``DetectorEvent`` collapses to a "(推理失败)"
    placeholder, wasting the 1s frame-inference cycle. By catching the
    validation error and falling back to ``_construct_text_instance``
    we keep the model's text content and stay on the same schema
    branch the caller already wrote.

    Any other exception (TypeError, RuntimeError, …) is re-raised so
    real bugs aren't masked.

    **v0.x 修复**：原来对没有 ``text`` 字段的 schema（如 ``TargetDoc`` 的
    ``x1/y1/x2/y2/desc``）走 ``raise`` 分支，让 ``ValidationError`` 一直
    抛到 ``respond()`` 边界，worker 把整条 inference 当成 error 报告。
    Qwen3.5-VL-4B 这类 4B 小模型对纯坐标 schema 跟得勉强，模型实际
    输出的可能是：

    - JSON schema 本身（``{"description": "...", "title": "TargetDoc",
      "type": "object"}``，即它把 schema 复读回来了）
    - 字段名错的 dict（``{"description": "..."}``，没 ``x1/y1/x2/y2/desc``）
    - 完全不是 JSON 的纯文本

    对这些情况，把 ``ValidationError`` 抛回去会让用户看到 "(推理失败)"
    占位、waste 整条 inference cycle。更友好的做法是返回
    ``raw_text_for_fallback``（让 caller 拿到模型的真实输出，可以
    decide 是显示 "(解析失败)" 还是自己再尝试 regex 解析 bbox），
    而不是 raise。caller 拿到的是 str 而不是 schema instance，自然
    不会走 ``isinstance(result, TargetDoc)`` 分支，整条路径退化到
    把 raw text 当 VLMDoc.text 渲染给用户。

    **v0.x 修复 2 (Qwen3.x Hermes tool-call wrapper)**：当模型走
    ``<tool_call>...</tool_call>`` 格式且工具名/字段对不上 schema 时，
    fallback 路径之前会把 `````json`` fence + ``<tool_call>`` 标签都
    留在 raw text 里，导致 ``Event.text='```json\n{"name":"ocr",...}</tool_call>'``。
    现在 ``_strip_fence_for_fallback`` 把这些 wrapper 全剥掉。
    """
    try:
        return _model_validate(model, data)
    except Exception as e:  # noqa: BLE001
        cls_name = type(e).__name__
        if cls_name == "ValidationError" or "pydantic" in cls_name.lower():
            if "text" not in getattr(model, "model_fields", {}):
                # No ``text`` field — no meaningful in-schema fallback
                # for the text content. Return the raw text (with any
                # ``<think>`` / ``<tool_call>`` / ```json``` markers
                # stripped) so the caller can surface it (or signal
                # parse failure) instead of getting an opaque "5 fields
                # missing" ValidationError.
                return _strip_fence_for_fallback(raw_text_for_fallback)  # type: ignore[return-value]
            # Prefer the model's actual content (often sitting in a
            # ``description``/``content`` key, possibly nested in
            # ``arguments``) over the raw string — the raw string
            # usually wraps the JSON object as-is and would show up
            # as a literal dict on the UI. Even the recovered text
            # gets its ``<think>`` / fence / ``<tool_call>`` markers
            # stripped defensively.
            text_value = _extract_text_from_mismatched_dict(data)
            if text_value is None:
                text_value = _strip_fence_for_fallback(raw_text_for_fallback)
            else:
                text_value = _strip_fence_for_fallback(text_value)
            fallback = _construct_text_instance(model, text_value)
            if fallback is not None:
                return fallback
        raise
