"""LLM / VLM / TTS session: routes a list of ``Attachment`` objects and
supports structured output (``LMOutput`` subclasses + plain ``@Generable``
classes alike)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from pydantic import BaseModel

from .attachment import Attachment
from .backends import build_backend
from .lm_output import LMOutput
from .model import ModelSpec
from .utils import extract_json

T = TypeVar("T", bound=BaseModel)


class LLMSession:
    """Session with a single RKNN model.

    Usage::

        model = SystemLanguageModel.qwen3_vl_4b().with_paths(
            rknn_llm_path=..., rknn_vision_path=...,
            tokenizer_path=..., embed_path=...,
        )
        async with LLMSession(model=model) as sess:
            out = await sess.respond(msg="...", attachment=[Attachment(img)])

    ``attachment`` is now a **list** of :class:`Attachment` (or a single
    ``Attachment`` for backward compat). This lets one session take in
    heterogeneous payloads — e.g. a text prompt + an image for VLM,
    or an audio config + a text instruction for TTS.

    ``generating`` accepts either a ``@Generable``-decorated class or a
    subclass of :class:`LMOutput`. ``LMOutput`` carries an
    ``attachment`` field; when a TTS-style backend produces audio
    bytes, the SDK fills it in automatically.
    """

    def __init__(self, model: ModelSpec, **options: Any) -> None:
        self.spec = model
        self._backend = build_backend(model)
        self.history: List[Dict[str, Any]] = []

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
        msg: Optional[str] = None,
        attachment: Union[None, Attachment, List[Attachment]] = None,
        generating: Optional[Type[T]] = None,
        debug: bool = False,
        tools: Optional[List[Any]] = None,
        to: Optional[str] = None,
        **overrides: Any,
    ) -> Union[str, T, Dict[str, Any]]:
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
        - ``debug=True`` — additionally prints ``msg``, the actual
          ``prompt`` sent to the LLM, the raw model output, the parsed
          result, and the backend perf table.
        - ``tools`` — a list of completion callbacks (a ``@Callable``-
          marked function or any 1-arg callable): invoked in order
          after a successful parse, with the parse result as the
          argument. Both sync and async callbacks are supported.
        """
        import inspect

        # Apple Foundation Models' style: ``to="..."`` is an alias for
        # ``msg="..."``. Accept both for convenience.
        if to is not None and msg is None:
            msg = to

        attachments = self._coerce_attachments(attachment)

        prompt = self._maybe_inject_json_hint(msg, generating)
        if debug:
            print(f"[rkai.debug] msg    = {msg!r}")
            print(f"[rkai.debug] prompt = {prompt!r}")
            print(f"[rkai.debug] attachments={[a.to_payload() for a in attachments]}")
        text, attachments_out = self._dispatch(prompt, attachments, **overrides)

        if debug:
            print(f"[rkai.debug] raw    = {text!r}")
            sink = getattr(self._backend, "sink", None)
            if sink is not None and sink.perf:
                print(f"[rkai.debug] perf   = {sink.perf}")

        # Maintain history
        self.history.append({"role": "user", "content": msg or ""})
        self.history.append({"role": "assistant", "content": text})

        result = self._parse_output(
            text, generating=generating, attachments_out=attachments_out,
        )
        if debug:
            print(f"[rkai.debug] parsed = {result!r}")

        # Completion callbacks (invoked in order; sync / async both supported)
        if tools:
            for tool in tools:
                ret = tool(result)
                if inspect.iscoroutine(ret):
                    await ret
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

        # Sort attachments by channel. We split into images / video frames
        # / text prefixes. Unknown kinds (audio, …) are forwarded as-is
        # for future backends; for now they fall through to run_text.
        images: List = []
        video_frames: List = []
        text_prefixes: List[str] = []
        other: List[Attachment] = []

        for att in attachments:
            if att.kind == "image":
                images.extend(att.images)
            elif att.kind == "video":
                video_frames.extend(att.frames)
            elif att.kind == "text":
                text_prefixes.append(att.text or "")
            else:
                # audio / bytes / unknown — pass through so future TTS /
                # multimodal backends can pick them up.
                other.append(att)

        # Build a single prompt with text attachments prepended.
        full_prompt = prompt or ""
        if text_prefixes:
            joined = "\n".join(p for p in text_prefixes if p)
            full_prompt = (joined + ("\n" + full_prompt if full_prompt else "")).strip()

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
    def _maybe_inject_json_hint(prompt: Optional[str], generating: Optional[Type[T]]) -> str:
        if generating is None:
            return prompt or ""
        user_msg = (prompt or "").rstrip()
        # Generate the JSON schema for the prompt hint. If this fails
        # (e.g. ``generating`` carries an ``Attachment`` field — a
        # non-pydantic class — pydantic refuses to produce a JSON
        # schema for it), fall back to no hint. That's fine for
        # audio / multimodal output where the model doesn't emit JSON.
        try:
            schema = (
                generating.schema()
                if hasattr(generating, "schema")
                else generating.__doc__
            )
        except Exception:
            return user_msg
        if not schema:
            return user_msg
        sep = "\n\n" if user_msg else ""
        # When msg is empty, give the model a default task so it doesn't "do nothing"
        default_task = (
                "Please carefully observe the behavior of people in the image, "
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
            result = _model_validate(generating, raw_text)
        elif isinstance(raw_text, str):
            data = extract_json(raw_text)
            if isinstance(data, dict):
                result = _model_validate(generating, data)
            elif data is None:
                # The model didn't emit valid JSON. Hand back a
                # model instance that carries the raw text in a
                # ``text`` field. If the class doesn't declare ``text``,
                # we fall back to ``model_construct(**{"text": ...})`` -
                # pydantic-v2 lets us inject an undeclared field via
                # ``model_config = ConfigDict(extra="allow")`` on the
                # class; otherwise we construct a bare instance and
                # set ``text`` as a normal Python attribute.
                try:
                    if "text" in getattr(generating, "model_fields", {}):
                        result = generating.model_construct(
                            _fields_set={"text"}, text=raw_text,
                        )
                    else:
                        result = generating.model_construct()
                        result.text = raw_text  # type: ignore[attr-defined]
                except Exception:
                    # Last-resort: return the raw string and let the
                    # caller notice the missing fields.
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
