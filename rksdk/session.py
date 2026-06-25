"""LLM / VLM session: routes ``Attachment`` to the right backend and supports structured output."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel

from .attachment import Attachment
from .backends import build_backend
from .model import ModelSpec
from .utils import extract_json

T = TypeVar("T", bound=BaseModel)


class LLMSession:
    """Session with a single RKNN model.

    Usage::

        model = SystemLanguageModel.qwen3_vl_2b().with_paths(
            rknn_llm_path=..., rknn_vision_path=...,
            tokenizer_path=..., embed_path=...,
        )
        async with LLMSession(model=model) as sess:
            out = await sess.respond(msg="...", attachment=Attachment(img))
    """

    def __init__(self, model: ModelSpec, **options: Any) -> None:
        self.spec = model
        self._backend = build_backend(model)
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    async def respond(
        self,
        msg: Optional[str] = None,
        attachment: Optional[Attachment] = None,
        generating: Optional[Type[T]] = None,
        debug: bool = False,
        tools: Optional[List[Any]] = None,
        **overrides: Any,
    ) -> Union[str, T, Dict[str, Any]]:
        """Fire one request and return the result.

        - ``attachment`` — if it's an ``Attachment``, the VLM backend is used
          (image / video path).
        - ``generating`` — if given, requires the model to emit JSON
          (auto-appends a schema hint and parses the result).
        - ``debug=True`` — additionally prints ``msg``, the actual ``prompt``
          sent to the LLM, the raw model output, the parsed result, and the
          backend perf table.
        - ``tools`` — a list of completion callbacks (a ``@Callable``-marked
          function or any 1-arg callable): invoked in order after a
          successful parse, with the parse result as the argument. Both sync
          and async callbacks are supported.
        """
        import inspect

        prompt = self._maybe_inject_json_hint(msg, generating)
        if debug:
            print(f"[rksdk.debug] msg    = {msg!r}")
            print(f"[rksdk.debug] prompt = {prompt!r}")
        text = self._dispatch(prompt, attachment, **overrides)

        if debug:
            print(f"[rksdk.debug] raw    = {text!r}")
            sink = getattr(self._backend, "sink", None)
            if sink is not None and sink.perf:
                print(f"[rksdk.debug] perf   = {sink.perf}")

        # Maintain history
        self.history.append({"role": "user", "content": msg or ""})
        self.history.append({"role": "assistant", "content": text})

        result = self._parse_output(text, generating=generating)
        if debug:
            print(f"[rksdk.debug] parsed = {result!r}")

        # Completion callbacks (invoked in order; sync / async both supported)
        if tools:
            for tool in tools:
                ret = tool(result)
                if inspect.iscoroutine(ret):
                    await ret
        return result

    # ------------------------------------------------------------------ #
    def _dispatch(
        self, prompt: str, attachment: Optional[Attachment], **overrides: Any,
    ) -> str:
        """Route ``prompt`` + ``attachment`` to the right backend method."""
        if attachment is None:
            # Pure text: text backend uses run; VLM backend uses run_text (inputs=None)
            run = getattr(self._backend, "run_text", None) or self._backend.run
            return run(prompt, **overrides)
        kind = attachment.kind
        if kind == "image":
            return self._backend.run_images(prompt, attachment.images, **overrides)
        if kind == "video":
            return self._backend.run_video(prompt, attachment.frames, **overrides)
        raise TypeError(f"Unsupported attachment kind: {kind}")

    # ------------------------------------------------------------------ #
    # JSON prompt injection
    # ------------------------------------------------------------------ #
    @staticmethod
    def _maybe_inject_json_hint(prompt: Optional[str], generating: Optional[Type[T]]) -> str:
        if generating is None:
            return prompt or ""
        user_msg = (prompt or "").rstrip()
        schema = generating.schema() if hasattr(generating, "schema") else generating.__doc__
        sep = "\n\n" if user_msg else ""
        # When msg is empty, give the model a default task so it doesn't "do nothing"
        default_task = (
            "Please carefully observe the behavior of people in the image / video, "
            "describe it according to the JSON Schema below, and assign a safety level. "
            "**Output JSON only, no other text**:"
            if not user_msg
            else "Please strictly follow the JSON Schema below to return the result. "
                 "**Output JSON only, no other text**:"
        )
        return (
            f"{user_msg}{sep}"
            f"{default_task}\n"
            f"```json\n{schema}\n```"
        )

    @staticmethod
    def _parse_output(
        raw: Union[str, Dict[str, Any], BaseModel],
        generating: Optional[Type[T]],
    ) -> Union[str, T, Dict[str, Any]]:
        if generating is None:
            return raw  # type: ignore[return-value]
        if isinstance(raw, generating):
            return raw
        if isinstance(raw, dict):
            return _model_validate(generating, raw)
        if isinstance(raw, str):
            data = extract_json(raw) or raw  # Fallback to raw text; pydantic will reject it
            if isinstance(data, str):
                return _model_validate_json(generating, data)
            if isinstance(data, dict):
                return _model_validate(generating, data)
        raise TypeError(f"Cannot parse {type(raw).__name__} as {generating.__name__}")

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
    # Stability helpers
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Clear history + the C library's KV cache / internal state.

        Call this between loop iterations to force the model to start from a
        clean state — trades a bit of speed for determinism.
        """
        self.history.clear()
        rknn_llm = getattr(self._backend, "rknn_llm", None)
        if rknn_llm is not None and hasattr(rknn_llm, "clear_kvcache"):
            try:
                rknn_llm.clear_kvcache()
            except Exception:
                pass

    def majority_vote(self, k: int, **kwargs) -> Any:
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
        # Use repr as the hash key (works for str / pydantic models / dict)
        keys = [repr(r) for r in results]
        top = Counter(keys).most_common(1)[0][0]
        # Return the first instance matching the winning key
        for r in results:
            if repr(r) == top:
                return r
        return results[0]


# --------------------------------------------------------------------------- #
# Compatibility shim: pydantic v1 uses parse_obj / parse_raw, v2 uses
# model_validate / model_validate_json
# --------------------------------------------------------------------------- #
def _model_validate(model: Type[T], data: Dict[str, Any]) -> T:
    validator = getattr(model, "model_validate", None) or getattr(model, "parse_obj", None)
    if validator is None:  # pragma: no cover
        raise RuntimeError("No suitable pydantic validator found")
    return validator(data)  # type: ignore[arg-type]


def _model_validate_json(model: Type[T], data: str) -> T:
    validator = getattr(model, "model_validate_json", None) or getattr(model, "parse_raw", None)
    if validator is None:  # pragma: no cover
        raise RuntimeError("No suitable pydantic JSON validator found")
    return validator(data)  # type: ignore[arg-type]
