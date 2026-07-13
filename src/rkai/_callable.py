"""``@Callable`` decorator — mark a function as a completion callback for ``session.respond``.

In the same spirit as Apple Foundation Models' ``@Callable``: the decorator itself
does no real work, it just stamps the function with a ``__rkai_callable__`` flag
so that:
- IDEs / readers can see at a glance that "this function will be invoked by the framework"
- there's a hook available for future signature validation (e.g. must accept exactly
  one positional parameter)

Actual invocation is done by ``LLMSession.respond(on_complete=...)`` (or ``tools=[...]``)
synchronously after the model output is successfully parsed. Async callbacks are
supported (auto-awaited).
"""
from __future__ import annotations

from typing import Callable as _Callable


def Callable(func: _Callable) -> _Callable:
    """Decorator: mark a function as a session.respond completion callback.

    Usage::

        @Callable
        def alarm(parm: Event):
            print(f"Alarm: {parm}")

        result = await session.respond(
            msg="...",
            generating=Event,
            on_complete=alarm,   # auto-invoked after a successful parse
        )
    """
    func.__rkai_callable__ = True  # type: ignore[attr-defined]
    return func
