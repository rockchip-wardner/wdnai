"""``@Generable`` decorator — promote a plain annotated class to a pydantic BaseModel.

Inspired by Apple Foundation Models' ``@Generable`` protocol: users don't have to
inherit from BaseModel — they just write type annotations + ``Field(description=...)``
and get schema validation, JSON parsing, ``isinstance``, ``repr`` for free.

Implementation: at decoration time we read ``cls.__annotations__`` plus
``cls.__dict__`` to collect each field's type and ``FieldInfo``. If the user
subclassed from another pydantic-friendly class (e.g. ``LMOutput``), we
pass that as the ``__base__`` of the generated pydantic model so its fields
are inherited. Otherwise we fall back to the implicit pydantic BaseModel.
"""
from __future__ import annotations

from typing import Any, Dict

import pydantic


def Generable(cls: type) -> type:
    """Class decorator: convert an annotated class into a pydantic BaseModel.

    Usage::

        from rkai import Generable, Field
        from typing import Literal, Optional

        @Generable
        class Event:
            '''custom event template'''
            act: str = Field(description="action")
            level: Literal["Normal", "Warning"] = Field(
                default="Normal", description="level",
            )

        # Inherit fields from another pydantic-friendly class:
        from rkai import LMOutput
        @Generable
        class TTSResult(LMOutput):
            text: str = Field(description="synthesised transcript")

    After this, the class is a full pydantic BaseModel::

        Event.schema()                       # JSON Schema
        Event.model_validate({"act": "x"})  # dict -> instance
        Event.model_validate_json('{"..."}') # json str -> instance
        isinstance(ins, Event)              # True
    """
    import inspect
    import sys as _sys

    annotations: Dict[str, Any] = getattr(cls, "__annotations__", {}) or {}
    fields: Dict[str, Any] = {}
    for name, ann in annotations.items():
        # Field default: a plain class carries FieldInfo in cls.__dict__;
        # a pydantic BaseModel subclass stores it in cls.model_fields via
        # __init_subclass__ and leaves cls.__dict__ empty for the field —
        # fall back to that, otherwise Field(default=...) on a subclass
        # silently turns into a required field.
        if name in cls.__dict__:
            default = cls.__dict__[name]
        else:
            pf = getattr(cls, "model_fields", {}).get(name)
            default = pf if pf is not None else ...
        fields[name] = (ann, default)

    # If the class inherits from something other than ``object`` (e.g.
    # ``LMOutput``), treat that as the pydantic base so its fields and
    # config are preserved.
    bases = getattr(cls, "__bases__", ()) or ()
    extra_base = next(
        (b for b in bases if b is not object), None
    )

    # Pick the right ``__module__`` for the generated pydantic model.
    # ``cls.__module__`` is whatever Python inferred when the user
    # wrote ``class Event(...)`` (typically the enclosing module name —
    # but it can be ``None`` if the class was defined inside a function
    # with no explicit ``__module__``). Forward-ref resolution in
    # pydantic looks up unqualified type names (e.g. ``Optional``)
    # using the generated class's ``__module__``, so getting this wrong
    # leads to ``PydanticUndefinedAnnotation``.
    #
    # We try in order: (1) the user's class' ``__module__``; (2) the
    # caller frame's module (where @Generable was applied); (3) fall
    # back to ``rkai._generable``.
    caller_module = None
    try:
        caller_frame = inspect.stack()[1]
        caller_module = caller_frame.frame.f_globals.get("__name__")
    except Exception:
        pass
    cls_module = getattr(cls, "__module__", None) or caller_module or __name__

    # Globals for forward-ref resolution: start from the user's module
    # so type names imported there (e.g. ``Optional``) are visible.
    caller_globals: Dict[str, Any] = {}
    try:
        caller_frame = inspect.stack()[1]
        caller_globals = dict(caller_frame.frame.f_globals)
    except Exception:
        pass
    # Merge the user's module's globals as a fallback so type names
    # that aren't in the caller's frame (e.g. ``Optional`` imported at
    # the top of the user's module) are still discoverable.
    mod = _sys.modules.get(cls_module)
    if mod is not None:
        for k, v in vars(mod).items():
            caller_globals.setdefault(k, v)

    if extra_base is not None:
        new_cls = pydantic.create_model(  # type: ignore[call-overload]
            cls.__name__, __base__=extra_base, **fields
        )
    else:
        new_cls = pydantic.create_model(cls.__name__, **fields)  # type: ignore[call-overload]

    # Override pydantic's default module so forward refs can find
    # ``Optional`` / ``Literal`` / ``List`` etc. from the user's module.
    try:
        new_cls.__module__ = cls_module
    except (AttributeError, TypeError):
        pass
    # Re-bind forward refs against the user's globals so instantiation
    # works even when the class is defined inside a function (where
    # ``cls.__module__`` may be missing or wrong).
    try:
        new_cls.model_rebuild(_types_namespace=caller_globals)
    except Exception:
        pass

    # Preserve docstring + qualname (debug-friendly)
    new_cls.__doc__ = cls.__doc__
    new_cls.__qualname__ = getattr(cls, "__qualname__", cls.__name__)
    return new_cls
