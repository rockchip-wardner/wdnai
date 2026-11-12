"""Qwen3.5-VL-9B backend for the segmented two-card export.

The 9B export is split into ``seg0`` / ``seg1``: each stage lives on its
own RK182x card and the two are chained by a token-by-token pipeline.
This module wraps the vendored pipeline in
:mod:`._qwen35_vl_9b_vendor` behind the same backend surface as
``VLMBackend``, so ``LLMLocalSession`` / the worker need no second code
path for 9B.

Like every other backend, it talks to librknn3 through
:mod:`wdnai._librknn` (plain ctypes) — no ``rknn3lite`` wheel. Text and
vision share the same pipeline; vision runs on stage 0's card.
"""
from __future__ import annotations

import os

# Qwen3.5-VL's MRoPE fallback logs are very noisy during text decode
# (one pair per stage per token). Match the vendor demo and silence the
# RKNN logger before librknn3 is loaded.
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
    def __init__(self, owner: "MultiCardVLMBackend") -> None:
        self._owner = owner

    def clear_kvcache(self, *args: Any, **kwargs: Any) -> int:
        self._owner.reset()
        return 0

    def session_stop(self) -> int:
        self._owner.stop()
        return 0

    # ⚠️ **这里故意没有 ``set_chat_template``。**
    #
    # 9B 的 C 会话**必须**保持「无模板」状态：vendor 在 init 时
    # ``set_chat_template("", "", "")`` 清空模板，后端 ``reset()`` 又清一次
    # —— 因为这个后端**自己**拼完整的 ChatML prompt（见 ``_build_prompt``：
    # ``<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n``），C 运行时
    # 只负责原样送进模型。
    #
    # 如果给 9B 套上 ChatML 模板，C 运行时会把**已经完整**的 prompt 再套
    # 一遍（``<|im_start|>user\n`` + 我们那段 + ``<|im_end|>…assistant\n``），
    # 模型看到的是一段错位的对话。实测后果：普通问答还凑合，但**结构化
    # 输出直接崩** —— 电脑控制的 agent 要的是 JSON，对话里套着对话时模型
    # 只吐出一个裸词（``click``），JSON 外壳整个丢了。
    #
    # 所以这里**故意**留成 AttributeError：任何还走
    # ``apply_chat_template(rknn)`` 的老路径都会**大声失败**，而不是悄悄
    # 给 9B 套上模板。9B 自己的模式切换走
    # ``MultiCardVLMBackend.set_thinking_mode()``（每次调用带模式，不锁存）。



class MultiCardVLMBackend:
    """Two-stage Qwen3.5-VL-9B pipeline on two RK182x cards."""

    def __init__(
        self,
        spec,
        *,
        npu_indices: Optional[Sequence[int]] = None,
        npu: Optional[int] = None,
        device_id: Any = None,
    ) -> None:
        if spec.stage_count < 2:
            raise ValueError("9B backend requires stage_count >= 2")
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

        The C-side chat template cannot replace this: on the segmented 9B
        stage0 a multi-turn bare transcript renders incorrectly (the model
        answers with a thinking dump), while the Jinja2 template handles it
        correctly. Keep this parser even though the backend is fully on the
        ctypes shim.
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
            if not self.sink.text:
                new = new.lstrip("\n")
            V.LAST_RESULT.last_output_text = text
            if new:
                self.sink.emit(new)

    def _generate(
        self,
        prompt: str,
        *,
        images: Optional[List[np.ndarray]] = None,
        enable_thinking: Optional[bool] = None,
        max_new_tokens: Optional[int] = None,
        keep_history: Optional[bool] = None,
    ) -> str:
        if enable_thinking is None:
            enable_thinking = bool(self.spec.options.get("enable_thinking", False))
        if max_new_tokens is None:
            max_new_tokens = int(self.spec.options.get("max_new_tokens", 512))
        max_new_tokens = max(1, int(max_new_tokens))

        # ``keep_history=False`` → 丢掉 C 会话里**上一轮**残留的 KV cache。
        #
        # 这个后端的会话级 ``llm_args["keep_history"] = 1`` 是 vendor 写死的
        # （分段 pipeline 内部需要它：decode 步要接着 prefill 往下走），
        # 代价是**跨请求**也会累积上下文。而应用层的每个提示都是自包含的
        # （电脑控制是「任务 + 已执行动作 + 当前截图」，文本对话是完整
        # ChatML 记录，全都在文本里），累积下来的历史纯属有害。
        #
        # 不清的后果实测过（2026-11-12）：让 9B「点击任务栏最左边的 Windows
        # 开始按钮」，它 thought 里写的却是**上一轮**「打开记事本」的目标，
        # 坐标也照着上一轮给（误差 60 px 起，落在隔壁图标上）。清一次 KV
        # 之后同一个提示立刻答对任务、输出也从「复读 schema 外壳」变回
        # 正常 JSON。
        if keep_history is False:
            self.reset()

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
            keep_history=overrides.get("keep_history"),
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
            keep_history=overrides.get("keep_history"),
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

    def set_thinking_mode(self, enable_thinking: bool) -> int:
        """Switch the thinking mode of this backend (worker control op).

        Unlike the single-card backend, the 9B's mode is **not** latched
        in the C session — it travels with **every call**:

        * the prompt carries the think block (``<think>\\n`` to open it,
          ``<think>\\n\\n</think>\\n\\n`` to close it immediately), and
        * ``session_run(enable_thinking=…)`` carries the flag.

        So there is nothing to *switch* here.  What this hook does is
        drop the KV cache, so a previous turn's context (and any thinking
        state it carried) cannot leak into the next one.

        ⚠️ It deliberately does **not** apply the ChatML template.  This
        backend builds a complete ChatML prompt itself and keeps the
        C-side template empty; applying the template would wrap that
        prompt a second time and break structured output.  See
        :class:`_StageHandle` for the full story.
        """
        self.reset()
        return 0

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


__all__ = ["MultiCardVLMBackend"]
