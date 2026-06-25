"""``@Generable`` decorator — promote a plain annotated class to a pydantic BaseModel.

Inspired by Apple Foundation Models' ``@Generable`` protocol: users don't have to
inherit from BaseModel — they just write type annotations + ``Field(description=...)``
and get schema validation, JSON parsing, ``isinstance``, ``repr`` for free.

Implementation: at decoration time we read ``cls.__annotations__`` plus
``cls.__dict__`` to collect each field's type and ``FieldInfo``, then call
``pydantic.create_model`` to dynamically generate a BaseModel subclass.
Downstream code (``Event.schema()``, ``Event.model_validate(...)``,
``Event.model_validate_json(...)``) works out of the box, and is
pydantic v1 / v2 compatible.
"""
from __future__ import annotations

from typing import Any, Dict

import pydantic


def Generable(cls: type) -> type:
    """Class decorator: convert an annotated class into a pydantic BaseModel.

    Usage::

        from rksdk import Generable, Field
        from typing import Literal, Optional

        @Generable
        class Event:
            '''custom event template'''
            act: str = Field(description="action")
            level: Literal["Normal", "Warning"] = Field(
                default="Normal", description="level",
            )

    After this, ``Event`` is a full pydantic BaseModel:
        Event.schema()                       # JSON Schema
        Event.model_validate({"act": "x"})  # dict -> instance
        Event.model_validate_json('{"..."}') # json str -> instance
        isinstance(ins, Event)              # True
    """
    annotations: Dict[str, Any] = getattr(cls, "__annotations__", {}) or {}
    fields: Dict[str, Any] = {}
    for name, ann in annotations.items():
        # Field default: FieldInfo / literal / Ellipsis (required) all from __dict__
        default = cls.__dict__.get(name, ...)
        fields[name] = (ann, default)

    new_cls = pydantic.create_model(cls.__name__, **fields)  # type: ignore[arg-type]
    # Preserve docstring + qualname (debug-friendly)
    new_cls.__doc__ = cls.__doc__
    new_cls.__qualname__ = getattr(cls, "__qualname__", cls.__name__)
    return new_cls
