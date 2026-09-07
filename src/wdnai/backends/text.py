"""Pure-text LLM backend — for Qwen2.5 / Qwen3 (structures are identical)."""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Union

from ._common import (
    StreamSink,
    apply_chat_template,
    build_llm_args,
    build_llm_callback,
    load_embed_table,
    load_tokenizer,
    printf_perf,
    resolve_device_id,
)


class TextBackend:
    """Pure-text RKNN LLM.

    On construction, runs ``load_rknn`` + ``init_runtime`` + ``set_chat_template``;
    call :py:meth:`run` to perform a single inference.
    """

    def __init__(
        self,
        spec,
        *,
        npu: Optional[int] = None,
        device_id: Optional[Union[bytes, str]] = None,
    ):
        from .._librknn import RKNN3Lite

        if not spec.rknn_path or not spec.weight_path:
            raise ValueError(
                f"{spec.name!r} is missing rknn_path / weight_path; "
                "fill them in via .with_paths(...)."
            )
        self.spec = spec
        self.rknn = RKNN3Lite(llm_mode=True, verbose=spec.verbose)
        if self.rknn.load_rknn(spec.rknn_path, spec.weight_path) != 0:
            raise RuntimeError(f"load_rknn failed: {spec.rknn_path}")

        # tokenizer + embed
        self.tokenizer = load_tokenizer(spec.tokenizer_path)
        self.embeds = load_embed_table(spec.embed_path, spec.vocab_size)

        # Resolve NPU selection **after** RKNN3Lite is imported so the
        # native driver init order stays predictable (and doesn't race
        # with a co-resident wdnav process on dev boxes).
        resolved_device_id = resolve_device_id(RKNN3Lite, npu, device_id)

        # callback / init
        self.sink = StreamSink(tokenizer=self.tokenizer)
        cb = build_llm_callback(self.tokenizer, self.embeds, self.sink)
        if self.rknn.init_runtime(
            target=spec.target,
            core_mask=spec.core_mask,
            llm_args=build_llm_args(spec),
            llm_callback=cb,
            device_id=resolved_device_id,
        ) != 0:
            raise RuntimeError("init_runtime failed")
        apply_chat_template(self.rknn)

    # ------------------------------------------------------------------ #
    def run(self, prompt: str, **overrides: Any) -> str:
        """Run a single text inference and return the full text."""
        self.sink.reset()
        t0 = time.perf_counter()
        kwargs: Dict[str, Any] = {"prompt": prompt}
        for k in ("max_new_tokens", "enable_thinking", "keep_history"):
            if k in overrides and overrides[k] is not None:
                kwargs[k] = overrides[k]
        ret, stats = self.rknn.session_run(**kwargs)
        if ret != 0:
            raise RuntimeError(f"session_run failed: ret={ret}")
        n_decode, n_prefill, llm_start, llm_end = stats
        if self.sink.first_token_time is not None:
            self.sink.perf.update({
                "n_decode_tokens": n_decode,
                "n_prefill_tokens": n_prefill,
                "prefill_ms": (self.sink.first_token_time - llm_start) * 1000.0,
                "decode_ms": (llm_end - self.sink.first_token_time) * 1000.0,
            })
            if overrides.get("perf", False):
                printf_perf(self.sink)
        else:  # pragma: no cover
            if overrides.get("perf", False):
                print(f"[TextBackend] total {(time.perf_counter() - t0) * 1000:.1f} ms, "
                      f"prefill={n_prefill} decode={n_decode}")
        return self.sink.text

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        try:
            self.rknn.release()
        except Exception:  # noqa: BLE001
            pass
