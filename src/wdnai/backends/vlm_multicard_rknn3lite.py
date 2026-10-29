"""Qwen3.5-VL-9B backend built on the vendor ``rknn3lite`` pipeline.

The single-card 4B path stays on wdnai's ctypes shim. The segmented 9B
export has two LLM sessions on different cards, which the shim cannot run
reliably (the first ``rknn3_session_run`` hangs once the second session
exists). The vendor pipeline uses ``rknn3lite.api`` and its
``RKLLMCallback`` entry points, and runs both cards reliably.

This module wraps that vendor implementation behind the same backend
surface as ``VLMBackend`` so ``LLMLocalSession`` / the worker do not need
a second code path for 9B. Text and vision share the same pipeline;
vision runs on stage 0's card.
"""
from __future__ import annotations

import os

# Qwen3.5-VL's MRoPE fallback logs are very noisy during text decode
# (one pair per stage per token). Match the vendor demo and silence the
# RKNN logger before rknn3lite / librknn3 is loaded.
os.environ["RKNN_LOG_LEVEL"] = "0"

import json
import re
import threading
from typing import Any, List, Optional, Sequence

import numpy as np
from jinja2 import Environment

from . import _qwen35_vl_9b_vendor as V
from ._common import StreamSink


class _StageHandle:
    def __init__(self, owner: "Rknn3LiteMultiCardVLMBackend") -> None:
        self._owner = owner

    def clear_kvcache(self, *args: Any, **kwargs: Any) -> int:
        self._owner.reset()
        return 0

    def session_stop(self) -> int:
        self._owner.stop()
        return 0

    def set_chat_template(self, *args: Any, **kwargs: Any) -> int:
        rc = 0
        for stage in getattr(self._owner, "stages", []) or []:
            try:
                rc = stage.rknn.set_chat_template("", "", "")
            except Exception:  # noqa: BLE001
                rc = -1
        return rc


class Rknn3LiteMultiCardVLMBackend:
    """Two-stage Qwen3.5-VL-9B using the vendor ``rknn3lite`` pipeline."""

    def __init__(
        self,
        spec,
        *,
        npu_indices: Optional[Sequence[int]] = None,
        npu: Optional[int] = None,
        device_id: Any = None,
    ) -> None:
        if spec.stage_count < 2:
            raise ValueError("rknn3lite 9B backend requires stage_count >= 2")
        os.environ["RKNN_LOG_LEVEL"] = "0"
        self.spec = spec
        self._stop = threading.Event()
        self.sink = StreamSink()

        self.tokenizer = V.GGUFBPETokenizer(spec.tokenizer_path)
        V.tokenizer = self.tokenizer
        self.sink.tokenizer = self.tokenizer
        self._chat_env = Environment(keep_trailing_newline=True)
        self._chat_env.filters["tojson"] = lambda value: json.dumps(
            value, ensure_ascii=False,
        )
        self._chat_env.globals["raise_exception"] = (
            lambda message: (_ for _ in ()).throw(ValueError(message))
        )
        self._chat_template = (
            self._chat_env.from_string(self.tokenizer.chat_template)
            if getattr(self.tokenizer, "chat_template", None) else None
        )
        self._embeds = np.memmap(
            spec.embed_path, dtype=np.float16, mode="r",
        ).reshape(spec.vocab_size, -1)
        V.embeds_data = self._embeds

        ids = V.RKNN3Lite().get_devices_id()
        if npu_indices is not None:
            device_ids = [ids[int(i)] for i in npu_indices]
        elif npu is not None:
            device_ids = [ids[int(npu)]]
        elif device_id is not None:
            raw = device_id.encode("utf-8") if isinstance(device_id, str) else device_id
            device_ids = [raw]
        else:
            device_ids = ids[: spec.stage_count]
        if len(device_ids) < spec.stage_count:
            raise RuntimeError(
                f"{spec.name!r}: need {spec.stage_count} cards, "
                f"found {len(device_ids)}"
            )

        seg_paths = list(spec.llm_seg_paths or [])
        seg_weights = list(spec.weight_seg_paths or [])
        if len(seg_paths) != spec.stage_count:
            raise ValueError(
                f"{spec.name!r}: llm_seg_paths must have {spec.stage_count} entries"
            )
        if len(seg_weights) != spec.stage_count:
            raise ValueError(
                f"{spec.name!r}: weight_seg_paths must have {spec.stage_count} entries"
            )

        self.stages: List[V.StageRuntime] = [
            V.StageRuntime(f"stage{i}", seg_paths[i], seg_weights[i])
            for i in range(spec.stage_count)
        ]
        self.pipeline = V.PipelineState(spec.stage_count, 128, False)
        self.vision = V.VisionRuntime(
            spec.rknn_vision_path, spec.weight_vision_path,
        )
        self.shared_mems = []
        V.LAST_RESULT.reset_generation()
        share_internal = True

        for i, stage in enumerate(self.stages):
            if not V.init_stage_context(
                stage=stage,
                stage_idx=i,
                device_id=device_ids[i],
                target=spec.target,
                core_mask=spec.core_mask,
                max_context_len=spec.max_context_len,
                decrypt_key_path=None,
                verbose=False,
                share_internal=(share_internal and i == 0),
            ):
                raise RuntimeError(f"stage{i} init context failed")

        if not V.init_vision(
            vision=self.vision,
            target=spec.target,
            core_mask=spec.core_mask,
            device_id=device_ids[0],
            decrypt_key_path=None,
            verbose=False,
            share_internal=share_internal,
        ):
            raise RuntimeError("vision init failed")
        if share_internal:
            self.shared_mems = V.setup_shared_internal_mem(
                self.vision, self.stages[0])

        for i, stage in enumerate(self.stages):
            if not V.init_stage_session(
                stage=stage,
                pipeline=self.pipeline,
                stage_idx=i,
                max_context_len=spec.max_context_len,
                ignore_eos=False,
                special_eos_ids=[int(spec.special_eos_id)],
                special_bos_ids=[],
                presence_penalty=float(
                    spec.options.get("presence_penalty", 1.5)
                ),
                kvcache_policy="default",
                verbose=False,
            ):
                raise RuntimeError(f"stage{i} init session failed")

        self.rknn_llm = _StageHandle(self)

    # ------------------------------------------------------------------ #
    def _messages_from_prompt(self, prompt: str) -> list[dict]:
        """Convert LLMSession's bare ChatML transcript into messages.

        ``LLMSession._compose_prompt`` uses bare ``system\\n`` / ``user\\n``
        / ``assistant\\n`` line prefixes for multi-turn prompts. The 9B
        GGUF tokenizer has the authoritative Qwen3.5 chat template, so use
        it instead of wrapping the whole transcript inside one user message.
        """
        text = prompt or ""
        matches = list(re.finditer(r"(?m)^(system|user|assistant)\n", text))
        if not matches or matches[0].start() != 0:
            return [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text},
            ]
        messages = []
        for i, match in enumerate(matches):
            role = match.group(1)
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].rstrip("\n")
            if role == "assistant" and not content.strip():
                continue
            messages.append({"role": role, "content": content})
        if not messages or messages[0]["role"] != "system":
            messages.insert(0, {
                "role": "system", "content": "You are a helpful assistant.",
            })
        return messages

    def _build_prompt(self, prompt: str, enable_thinking: bool) -> str:
        if self._chat_template is not None:
            return self._chat_template.render(
                messages=self._messages_from_prompt(prompt),
                add_generation_prompt=True,
                enable_thinking=bool(enable_thinking),
            )
        body = V.PROMPT_PREFIX + prompt + V.PROMPT_POSTFIX
        return body + (
            "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"
        )

    def _build_image_prompt(self, prompt: str, n_images: int) -> str:
        placeholders = prompt.count(V.IMAGE_TOKEN_TEXT)
        if n_images <= 0:
            if placeholders:
                raise ValueError(
                    "prompt contains <image> placeholders but no image was given"
                )
            return prompt
        if placeholders == 0:
            return V.IMAGE_TOKEN_TEXT * n_images + prompt
        if placeholders == n_images:
            return prompt
        raise ValueError(
            f"prompt contains {placeholders} <image> placeholders "
            f"but {n_images} images were given"
        )

    def _flush(self) -> None:
        with V.LAST_RESULT.lock:
            tokens = list(V.LAST_RESULT.generated_tokens)
        if not tokens:
            return
        try:
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        except Exception:  # noqa: BLE001
            return
        if "\ufffd" in text:
            text = text.split("\ufffd", 1)[0]
        prev = V.LAST_RESULT.last_output_text
        new = text[len(prev):] if text.startswith(prev) else text
        if new:
            V.LAST_RESULT.last_output_text = text
            self.sink.emit(new)

    def _generate(
        self,
        prompt: str,
        *,
        images: Optional[List[np.ndarray]] = None,
        enable_thinking: Optional[bool] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        if enable_thinking is None:
            enable_thinking = bool(self.spec.options.get("enable_thinking", False))
        if max_new_tokens is None:
            max_new_tokens = int(self.spec.options.get("max_new_tokens", 512))
        max_new_tokens = max(1, int(max_new_tokens))

        self._stop.clear()
        self.sink.reset()
        V.LAST_RESULT.reset_generation()
        image_input = None
        if images:
            if self.vision is None:
                raise RuntimeError("9B vision runtime is not initialised")
            image_input, _packed, _vlat, _ntok = V.run_vision(
                self.vision, list(images), False,
            )
            full_prompt = self._build_prompt(
                self._build_image_prompt(prompt, len(images)),
                bool(enable_thinking),
            )
        else:
            full_prompt = self._build_prompt(prompt, bool(enable_thinking))

        ok, prefill_tokens = V.run_pipeline_once(
            self.stages,
            self.pipeline,
            prompt=full_prompt,
            inputs=[image_input] if image_input is not None else None,
            enable_thinking=bool(enable_thinking),
        )
        if not ok:
            raise RuntimeError("9B prefill pipeline failed")
        next_token = V.LAST_RESULT.wait_next_token(timeout=3.0)
        if next_token is None:
            raise RuntimeError("9B prefill produced no token")
        self._flush()

        decoded = 0
        while decoded < max_new_tokens:
            if self._stop.is_set() or self.sink.stop_event.is_set():
                self.sink.stopped = True
                break
            ok, _ = V.run_pipeline_once(
                self.stages,
                self.pipeline,
                input_tokens=[int(next_token)],
                enable_thinking=bool(enable_thinking),
            )
            if not ok:
                raise RuntimeError("9B decode pipeline failed")
            next_token = V.LAST_RESULT.wait_next_token(timeout=3.0)
            if next_token is None:
                break
            self._flush()
            decoded += 1
            if int(next_token) == int(self.spec.special_eos_id):
                break
        self.sink.finished = True
        return self.sink.text

    # ------------------------------------------------------------------ #
    def run_text(self, prompt: str, **overrides: Any) -> str:
        return self._generate(
            prompt,
            enable_thinking=overrides.get("enable_thinking"),
            max_new_tokens=overrides.get("max_new_tokens"),
        )

    def run(self, prompt: str, **overrides: Any) -> str:
        return self.run_text(prompt, **overrides)

    def run_images(self, prompt: str, images: List[np.ndarray], **overrides: Any) -> str:
        if not images:
            return self.run_text(prompt, **overrides)
        return self._generate(
            prompt,
            images=list(images),
            enable_thinking=overrides.get("enable_thinking"),
            max_new_tokens=overrides.get("max_new_tokens"),
        )

    def available_image_sizes(self) -> List[int]:
        vision = getattr(self, "vision", None)
        return [int(vision.width)] if vision is not None else []

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        for stage in getattr(self, "stages", []) or []:
            try:
                stage.rknn.clear_kvcache()
                stage.rknn.set_chat_template("", "", "")
            except Exception:  # noqa: BLE001
                pass
        V.LAST_RESULT.reset_generation()
        try:
            self.pipeline.reset()
        except Exception:  # noqa: BLE001
            pass

    def stop(self) -> None:
        self._stop.set()
        self.sink.stop()
        try:
            self.pipeline.fail()
        except Exception:  # noqa: BLE001
            pass
        for stage in getattr(self, "stages", []) or []:
            fn = getattr(stage.rknn, "session_stop", None)
            if callable(fn):
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    pass

    def close(self) -> None:
        try:
            self.stop()
        except Exception:  # noqa: BLE001
            pass
        if getattr(self, "shared_mems", None):
            try:
                V.destroy_shared_internal_mem(self.vision, self.shared_mems)
            except Exception:  # noqa: BLE001
                pass
            self.shared_mems = []
        if getattr(self, "vision", None) is not None:
            try:
                self.vision.release()
            except Exception:  # noqa: BLE001
                pass
        for stage in getattr(self, "stages", []) or []:
            try:
                stage.release()
            except Exception:  # noqa: BLE001
                pass
        self.stages = []
        self.vision = None


__all__ = ["Rknn3LiteMultiCardVLMBackend"]
