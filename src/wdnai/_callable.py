"""``@Callable`` class decorator — define a model-callable tool.

A ``@Callable``-decorated class is turned into a pydantic ``BaseModel``
*and* registered as a tool the model can call. Each annotated class
attribute becomes a tool parameter; the first ``str`` form of
``Annotated[T, "…"]`` (or ``Field(description=…)``) becomes the
parameter's description; the class's ``__doc__`` becomes the tool's
description; the resulting ``BaseModel`` class is the thing that
``respond(tools=[…])`` returns when the model picks the tool.

The session is **passive**: it never invokes the tool on the user's
behalf. The developer decides whether (and when) to call
``instance()`` to fire the side effect, enabling multi-turn agent
loops and custom dispatch logic.

Usage::

    from typing import Annotated
    from wdnai import Generable, Callable, Field

    @Callable
    class Alarm:
        '''Site alarm dispatcher.'''
        level: Annotated[str, "事件严重等级"]
        msg:   Annotated[str, "事件描述文本"]

        def __call__(self):
            print(f"ALARM {self.level}/{self.msg}")

    @Generable
    class Event:
        text:  str = Field(default="无")
        level: str = Field(default="低")

    result = await session.respond(
        msg="...",
        attachment=[img],
        generating=Event,
        tools=[Alarm],
    )

    if isinstance(result, Alarm):
        # Model picked the tool; ``result`` is a populated Alarm instance.
        print(f"alarm fired: {result.level}, {result.msg}")
        result()                  # invoke the side effect
    elif isinstance(result, Event):
        # Model emitted the structured payload instead.
        ...

Sync and async ``__call__`` are both supported — define
``async def __call__(self)`` and ``await result()``.

Why class-form, not function-form
---------------------------------
The decorator returns the same class object the user wrote — a
real Python class. ``respond()`` returns an *instance* of that class
when the model picks the tool, which means ``isinstance(result,
Alarm)`` (or any other ``@Callable`` class) tells you *exactly*
which tool the model called, with no extra ``ToolCall`` carrier
to switch on.
"""
from __future__ import annotations

import inspect
import json
import sys
import typing
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    get_args,
    get_origin,
)

import pydantic

from .utils import extract_json


# --------------------------------------------------------------------------- #
# Annotation helpers — lifted from ``@Generable`` so the two decorators
# handle forward refs / Annotated metadata uniformly.
# --------------------------------------------------------------------------- #
def _unwrap_optional(ann: Any) -> Any:
    origin = get_origin(ann)
    if origin is None:
        return ann
    args = [a for a in get_args(ann) if a is not type(None)]
    if len(args) == 1:
        return args[0]
    return ann


def _split_annotated(ann: Any) -> Tuple[Any, List[Any]]:
    metadata = getattr(ann, "__metadata__", None)
    if metadata is not None:
        return getattr(ann, "__origin__", ann), list(metadata)
    return ann, []


def _resolve_user_module(cls: type) -> Tuple[str, Dict[str, Any]]:
    """Return ``(module_name, globals_dict)`` for forward-ref resolution.

    Mirrors ``@Generable``: prefer the class's ``__module__`` for the
    module name; fall back to the caller's frame globals for classes
    defined inside functions.
    """
    import sys as _sys
    caller_globals: Dict[str, Any] = {}
    try:
        caller_frame = inspect.stack()[1]
        caller_globals = dict(caller_frame.frame.f_globals)
    except Exception:
        pass
    cls_module = getattr(cls, "__module__", None) or caller_globals.get("__name__")
    mod = _sys.modules.get(cls_module) if cls_module else None
    if mod is not None:
        for k, v in vars(mod).items():
            caller_globals.setdefault(k, v)
    return cls_module or "wdnai._callable", caller_globals


# --------------------------------------------------------------------------- #
# Field harvest + pydantic-class construction (the body is shared with
# ``@Generable`` — kept duplicated here so the two decorators stay
# independent files)
# --------------------------------------------------------------------------- #
def _collect_fields(cls: type) -> Dict[str, Tuple[Any, Any]]:
    """Return ``{name: (annotation, default)}`` for every annotated
    attribute on ``cls``.

    Three annotation styles are supported and reconciled into one
    ``(annotation, default)`` tuple per field:

    * Plain attribute with ``Field(...)`` (or any default) — the
      ``cls.__dict__`` entry is taken as the default; the annotation
      is stripped of ``Annotated``.
    * ``Annotated[T, FieldInfo(...)]`` — the ``Annotated`` wrapper is
      preserved (pydantic extracts the ``FieldInfo`` from it natively).
    * ``Annotated[T, "description"]`` (bare string style) — converted
      to a synthetic ``FieldInfo(description=…)`` default so the
      schema carries it through.
    """
    from pydantic.fields import FieldInfo

    annotations: Dict[str, Any] = getattr(cls, "__annotations__", {}) or {}
    fields: Dict[str, Tuple[Any, Any]] = {}
    for name, raw_ann in annotations.items():
        underlying, metadata = _split_annotated(raw_ann)

        # Find FieldInfo / bare-string descriptions in metadata.
        field_info_in_meta: Optional[FieldInfo] = None
        string_desc_in_meta: Optional[str] = None
        if metadata:
            for m in metadata:
                if field_info_in_meta is None and isinstance(m, FieldInfo):
                    field_info_in_meta = m
                if string_desc_in_meta is None and isinstance(m, str):
                    string_desc_in_meta = m

        if name in cls.__dict__:
            # User wrote ``name = Field(...)`` or ``name = "default"``.
            fields[name] = (underlying, cls.__dict__[name])
        elif field_info_in_meta is not None:
            # ``Annotated[T, FieldInfo(...)]`` — keep Annotated so
            # pydantic sees the FieldInfo inside the wrapper.
            fields[name] = (raw_ann, field_info_in_meta)
        elif string_desc_in_meta is not None:
            # ``Annotated[T, "str"]`` — synthesize FieldInfo(description=…)
            # and pass plain ``T`` (pydantic only understands Annotated
            # wrapping FieldInfo, not bare strings).
            fields[name] = (underlying, FieldInfo(description=string_desc_in_meta))
        else:
            # Plain annotation, no description, no default.
            pf = getattr(cls, "model_fields", {}).get(name)
            default = pf if pf is not None else ...
            fields[name] = (underlying, default)
    return fields


def _build_callable_model(cls: type) -> type:
    """Turn ``cls`` into a ``pydantic.BaseModel`` subclass while
    preserving any user-defined methods (``__call__``, ``run``, …),
    the docstring, and the qualname.
    """
    fields = _collect_fields(cls)
    cls_module, caller_globals = _resolve_user_module(cls)

    new_cls = pydantic.create_model(cls.__name__, **fields)  # type: ignore[call-overload]
    try:
        new_cls.__module__ = cls_module
    except (AttributeError, TypeError):
        pass
    try:
        new_cls.model_rebuild(_types_namespace=caller_globals)
    except Exception:
        pass

    # Re-attach every non-field attribute from the user's class.
    # pydantic.create_model builds a fresh class and only sees the
    # passed-in field kwargs, so user-defined methods, properties and
    # class-level constants have to be re-pinned. We skip dunder
    # internals and the field names we already passed in.
    annotated_names = set(fields)
    for name, value in cls.__dict__.items():
        if name in annotated_names:
            continue
        if name in ("__dict__", "__weakref__"):
            continue
        # Don't copy descriptors that pydantic already manages.
        if name.startswith("model_") or name.startswith("model"):
            continue
        setattr(new_cls, name, value)

    new_cls.__doc__ = cls.__doc__
    new_cls.__qualname__ = getattr(cls, "__qualname__", cls.__name__)
    return new_cls


# --------------------------------------------------------------------------- #
# Tool spec
# --------------------------------------------------------------------------- #
def _build_tool_spec(new_cls: type) -> Dict[str, Any]:
    """Compose the tool metadata (name / description / parameters) and
    stash it on the class so the session can find it.
    """
    try:
        parameters_schema = new_cls.model_json_schema()
    except Exception as e:
        raise TypeError(
            f"@Callable class {new_cls.__name__!r} has no JSON schema "
            f"({e!r})"
        ) from e
    spec = {
        "name": new_cls.__name__,
        "description": (new_cls.__doc__ or "").strip(),
        "parameters": parameters_schema,
        "_cls": new_cls,
    }
    new_cls.__wdnai_callable__ = True        # legacy marker
    new_cls.__wdnai_tool__ = spec            # new: full tool spec
    return spec


# --------------------------------------------------------------------------- #
# Decorator
# --------------------------------------------------------------------------- #
def Callable(cls=None, *, name: Optional[str] = None, description: Optional[str] = None):
    """Class decorator: register a class as a model-callable tool.

    Bare form::

        @Callable
        class Alarm: ...

    Parametrised form (override ``name`` / ``description``)::

        @Callable(name="alarm", description="Fire alarm dispatcher")
        class _AlarmImpl: ...
    """

    def _wrap(target: type) -> type:
        new_cls = _build_callable_model(target)
        spec = _build_tool_spec(new_cls)
        if name is not None:
            spec["name"] = name
        if description is not None:
            spec["description"] = description
        # Re-stamp in case name/description overrides changed.
        new_cls.__wdnai_tool__ = spec
        return new_cls

    if cls is not None and isinstance(cls, type):
        return _wrap(cls)
    return _wrap


# --------------------------------------------------------------------------- #
# Tool-call helpers (consumed by ``LLMSession.respond``)
# --------------------------------------------------------------------------- #
def get_tool_specs(tools: Optional[List[type]]) -> List[Dict[str, Any]]:
    """Return the model-facing tool specs for every entry in ``tools``.

    Internal fields (``_cls``) are stripped — this is what gets
    serialised into the prompt. Raises ``TypeError`` if any entry
    isn't a ``@Callable`` class.
    """
    out: List[Dict[str, Any]] = []
    for t in tools or []:
        spec = getattr(t, "__wdnai_tool__", None)
        if spec is None:
            raise TypeError(
                f"Tool {t!r} is not decorated with @Callable; only "
                f"@Callable classes may be passed via tools=[…]"
            )
        out.append({
            "name": spec["name"],
            "description": spec["description"],
            "parameters": spec["parameters"],
        })
    return out


def render_tools_block(tool_specs: List[Dict[str, Any]]) -> str:
    """Render the tool definitions + the call instruction for injection
    into the model prompt.

    The format follows Qwen-Agent's ``NousFnCallPrompt`` (Hermes-style
    tool use — the canonical format for Qwen3 / Qwen3.5 / Qwen2.5; see
    https://qwen.readthedocs.io/en/latest/framework/function_call.html
    and ``qwen_agent/llm/fncall_prompts/nous_fncall_prompt.py``).

    The legacy wdnai format used a ``<tool_name>`` placeholder which
    Qwen3.x models can't follow — they have been SFT-ed on the literal
    ``<tool_call>``/``</tool_call>`` XML wrapper + a ``<tools>`` block
    of ``{"type": "function", "function": {...}}`` JSON, and they
    hallucinate tool names (e.g. ``"ocr"``) when the prompt only shows
    a generic placeholder. Switching to the canonical format:

      * puts each tool's *literal* name (``Crop``, ``Alarm``, …) in the
        definitions so the model never has to guess;
      * tells the model to wrap its call in ``<tool_call>...</tool_call>``,
        which the parser already strips in
        :func:`wdnai.utils.strip_thinking_blocks` — no new parsing path
        needed;
      * uses the OpenAI-compatible ``{"type": "function", "function":
        {...}}`` schema that matches both the HF transformers chat
        template and vLLM's ``--tool-call-parser hermes``.

    See also ``Qwen-Agent/qwen_agent/llm/fncall_prompts/nous_fncall_prompt.py:FN_CALL_TEMPLATE``.
    """
    if not tool_specs:
        return ""
    # Tool definitions in OpenAI-compatible ``{"type": "function",
    # "function": {...}}`` shape, one JSON object per line inside the
    # ``<tools>...</tools>`` block. The model reads this block verbatim
    # to discover the available functions and their schemas.
    tool_entries = [
        json.dumps({"type": "function", "function": {
            "name": spec["name"],
            "description": spec["description"] or "(no description)",
            "parameters": spec["parameters"],
        }}, ensure_ascii=False)
        for spec in tool_specs
    ]
    tool_descs = "\n".join(tool_entries)
    return (
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n"
        f"{tool_descs}\n"
        "</tools>\n\n"
        "For each function call, return a json object with function name and "
        "arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call>"
    )


def try_parse_tool_call(text: str) -> Optional[Tuple[str, Any]]:
    """If ``text`` contains a tool-call JSON, return ``(name, arguments)``.

    Otherwise return ``None``.

    Accepts two layouts:

    * **Hermes / Qwen-Agent style** (canonical for Qwen3 / Qwen3.5 /
      Qwen2.5 — see ``NousFnCallPrompt``): the JSON is wrapped in
      ``<tool_call>...</tool_call>`` tags (stripped by
      :func:`wdnai.utils.strip_thinking_blocks` before this point).
    * **Legacy bare JSON**: ``{"name": ..., "arguments": {...}}`` or
      ``{"function": ..., "args": {...}}`` etc. (kept for
      backwards-compat with older exports / different prompt styles).

    Key aliases accepted: ``name`` / ``tool`` / ``function`` for the
    identifier; ``arguments`` / ``args`` / ``parameters`` for the
    payload. ``arguments`` may itself be a *JSON string* (Hermes spec
    embeds the JSON-encoded string) — we ``json.loads`` it
    transparently in that case so the caller's :func:`dispatch_tool_call`
    always sees a dict.

    If no argument key is present, every top-level key other than the
    name is treated as the argument payload.
    """
    data = extract_json(text)
    if not isinstance(data, dict):
        return None
    name = data.get("name") or data.get("tool") or data.get("function")
    if not isinstance(name, str):
        return None
    args = (
        data.get("arguments")
        or data.get("args")
        or data.get("parameters")
    )
    # Hermes spec allows ``arguments`` to be a JSON string rather than
    # an inline dict. Decode transparently so dispatch_tool_call
    # always validates against a dict.
    if isinstance(args, str):
        try:
            decoded = json.loads(args)
            if isinstance(decoded, dict):
                args = decoded
        except json.JSONDecodeError:
            # Keep the raw string; dispatch_tool_call will try to
            # parse it again and surface a ValidationError on failure.
            pass
    if args is None:
        args = {k: v for k, v in data.items()
                if k not in {"name", "tool", "function"}}
    return name, args


def dispatch_tool_call(
    tools: Optional[List[type]],
    name: str,
    arguments: Any,
) -> Optional[Any]:
    """Find the ``@Callable`` class whose spec name matches ``name``
    and return an instance validated against the model's ``arguments``.

    Returns ``None`` if no registered tool matches the name, or if
    ``arguments`` can't validate against the tool's schema — the
    caller can decide to fall through to the regular parse path in
    that case.
    """
    for t in tools or []:
        spec = getattr(t, "__wdnai_tool__", None)
        if not spec or spec["name"] != name:
            continue
        cls = spec["_cls"]
        if isinstance(arguments, dict):
            return cls.model_validate(arguments)
        if isinstance(arguments, str):
            try:
                payload = json.loads(arguments)
            except json.JSONDecodeError:
                return None
            if not isinstance(payload, dict):
                return None
            return cls.model_validate(payload)
        return None
    return None