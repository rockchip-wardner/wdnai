"""Vision-language model backend — for Qwen2.5-VL / Qwen3-VL (image).

Flow (aligned with examples/Qwen2.5VL/test.py, Qwen3VL/test.py):

1. Load the vision RKNN + LLM RKNN.
2. For each image: BGR -> RGB -> resize -> [optional] pruned preprocessing
   -> vision.inference -> float16.
3. Pack the vision outputs into an RKNN3Image for the LLM.
4. Qwen3-VL additionally concatenates 3 deepstack aux tensors (index 2, 3, 4).
5. Call ``llm.session_run(inputs=..., prompt=...)``.

Video decoding is not handled here; callers feed pre-decoded frames via
``rkv.VideoCapture`` and pass them as a list of ndarrays to ``run_images``.
"""
from __future__ import annotations

import ctypes
import time
from typing import Any, List, Optional, Tuple

import numpy as np

from ..utils import (
    bgr_to_rgb,
    qwen25vl_prune,
    qwen3vl_prune_single,
    resize,
)
from ._common import (
    StreamSink,
    apply_chat_template,
    build_llm_args,
    build_llm_callback,
    load_embed_table,
    load_tokenizer,
    printf_perf,
)


# --------------------------------------------------------------------------- #
class VLMBackend:
    """Vision-language model backend. ``spec.variant`` decides between 2.5VL and 3VL."""

    VARIANT_25VL = "qwen2.5-vl"
    VARIANT_3VL = "qwen3-vl"

    def __init__(self, spec):
        from rknn3lite.api import RKNN3Lite  # type: ignore

        if not spec.rknn_llm_path or not spec.rknn_vision_path:
            raise ValueError(
                f"{spec.name!r} is missing rknn_llm_path / rknn_vision_path; "
                "fill them in via .with_paths(...)."
            )
        self.spec = spec
        self.variant = spec.variant

        # ---- vision -----------------------------------------------------
        self.rknn_vision = RKNN3Lite()
        vp = spec.rknn_vision_path
        if self.rknn_vision.load_rknn(vp, vp.replace(".1828.rknn", ".weight")) != 0:
            raise RuntimeError(f"vision load_rknn failed: {vp}")

        # ---- llm --------------------------------------------------------
        self.rknn_llm = RKNN3Lite(llm_mode=True, verbose=spec.verbose)
        lp = spec.rknn_llm_path
        if self.rknn_llm.load_rknn(lp, lp.replace(".1828.rknn", ".weight")) != 0:
            raise RuntimeError(f"llm load_rknn failed: {lp}")

        self.tokenizer = load_tokenizer(spec.tokenizer_path)
        self.embeds = load_embed_table(spec.embed_path, spec.vocab_size)

        # ---- init runtimes ---------------------------------------------
        if self.rknn_vision.init_runtime(
            target=spec.target, core_mask=spec.core_mask
        ) != 0:
            raise RuntimeError("vision init_runtime failed")

        self.sink = StreamSink(tokenizer=self.tokenizer)
        cb = build_llm_callback(self.tokenizer, self.embeds, self.sink)
        if self.rknn_llm.init_runtime(
            target=spec.target,
            core_mask=spec.core_mask,
            llm_args=build_llm_args(spec),
            llm_callback=cb,
        ) != 0:
            raise RuntimeError("llm init_runtime failed")
        apply_chat_template(self.rknn_llm)

    # ------------------------------------------------------------------ #
    # Image preprocessing — dispatched to the variant-specific implementation
    # ------------------------------------------------------------------ #
    def _preprocess_image(self, img_bgr: np.ndarray) -> np.ndarray:
        """Feed a BGR ndarray into the vision model.

        - ``use_prune=True``  -> pruned-model preprocessing (matches the examples);
                                 returns a 2D feature.
        - ``use_prune=False`` -> full model: NHWC ``(1, H, W, 3)``. If the Qwen3-VL
                                 full vision model expects ``shape[0] == 2``, the
                                 single image is duplicated to a 2-image stack.
        """
        size = self.spec.image_size
        rgb = resize(bgr_to_rgb(img_bgr), size)
        nchw = rgb.transpose(2, 0, 1).reshape(1, 3, size, size)

        if self.spec.use_prune:
            if self.variant == self.VARIANT_25VL:
                feat = qwen25vl_prune(nchw)
            else:  # 3VL
                feat = qwen3vl_prune_single(nchw)
            return np.float16(feat)

        # Full model: first check the vision model's actual input shape
        in_attr = self.rknn_vision.get_inputs_tensor_attr()[0]
        nhwc = rgb.reshape(1, size, size, 3)
        # Qwen3-VL full vision model expects 2 stacked images (to support the
        # "two-frame input" video mode)
        if (
            self.variant == self.VARIANT_3VL
            and in_attr.n_dims == 4
            and in_attr.shape[0] == 2
        ):
            nhwc = np.concatenate([nhwc, nhwc], axis=0)
        return np.float16(nhwc)

    def _vision_infer(self, img_bgr: np.ndarray):
        t0 = time.perf_counter()
        feat = self._preprocess_image(img_bgr)
        outs = self.rknn_vision.inference(inputs=[feat])
        return outs, time.perf_counter() - t0

    # ------------------------------------------------------------------ #
    # LLM input assembly
    # ------------------------------------------------------------------ #
    def _build_image_inputs(
        self, image_bgr_list: List[np.ndarray]
    ) -> Tuple[str, list, float]:
        """Return ``(prompt_with_prefix, rknn_inputs, vision_latency_sec)``."""
        from rknn3lite.api import (  # type: ignore
            Float16, RKNN3Image, RKNN3AuxTensorWrapper,
        )

        feats: List[np.ndarray] = []
        deepstack: List[List[np.ndarray]] = [[], [], []]
        vision_latency_total = 0.0

        for img in image_bgr_list:
            outs, lat = self._vision_infer(img)
            vision_latency_total += lat
            out0 = outs[0]
            # Most models emit a 2D (n_tokens, dim) tensor; add a batch dim if so
            if out0.ndim == 2:
                out0 = np.expand_dims(out0, 0)
            out0 = np.float16(out0)
            feats.append(out0)
            if self.variant == self.VARIANT_3VL and len(outs) >= 4:
                # deepstack lives in outputs[1..3]
                deepstack[0].append(np.float16(outs[1]))
                deepstack[1].append(np.float16(outs[2]))
                deepstack[2].append(np.float16(outs[3]))

        features = np.concatenate(feats, axis=0)
        inputs = []

        llm_img = RKNN3Image()
        llm_img.image_embed = features.ctypes.data_as(ctypes.POINTER(Float16))
        llm_img.n_image_tokens = features.shape[1]
        llm_img.n_image = features.shape[0]
        llm_img.image_width = self.spec.image_size
        llm_img.image_height = self.spec.image_size
        llm_img.image_start = "<|vision_start|>".encode("utf-8")
        llm_img.image_end = "<|vision_end|>".encode("utf-8")
        llm_img.image_content = "<|image_pad|>".encode("utf-8")
        inputs.append(llm_img)

        if self.variant == self.VARIANT_3VL and deepstack[0]:
            # Match examples/Qwen3VL/test.py single-image behavior: align_size is
            # NOT multiplied by features.shape[0]
            aligned = self.rknn_vision.get_outputs_tensor_attr()[0].aligned_size
            for i in range(3):
                aux = RKNN3AuxTensorWrapper()
                aux.index = 2 + i
                aux.aux_data = np.concatenate(deepstack[i], axis=0)
                aux.align_size = aligned
                inputs.append(aux)

        prefix = "<image>" * len(image_bgr_list)
        return prefix, inputs, vision_latency_total

    # ------------------------------------------------------------------ #
    def run_text(self, prompt: str, **overrides: Any) -> str:
        """Pure-text inference (no image). The prompt is sent directly to the LLM."""
        return self._run_llm(prompt, inputs=None, vision_latency=None, **overrides)

    def run_images(
        self, prompt: str, images: List[np.ndarray], **overrides: Any
    ) -> str:
        if not images:
            raise ValueError("images must not be empty")
        prefix, inputs, vlat = self._build_image_inputs(images)
        full_prompt = f"{prefix}{prompt}"
        return self._run_llm(full_prompt, inputs, vision_latency=vlat, **overrides)

    def _run_llm(
        self, prompt: str, inputs, vision_latency: Optional[float], **overrides: Any,
    ):
        self.sink.reset()
        kwargs = {"prompt": prompt}
        if inputs is not None:
            kwargs["inputs"] = inputs
        # Per-call overrides (max_new_tokens / enable_thinking / keep_history)
        for k in ("max_new_tokens", "enable_thinking", "keep_history"):
            if k in overrides and overrides[k] is not None:
                kwargs[k] = overrides[k]
        ret, stats = self.rknn_llm.session_run(**kwargs)
        if ret != 0:
            raise RuntimeError(f"llm session_run failed: ret={ret}")
        n_decode, n_prefill, llm_start, llm_end = stats
        if self.sink.first_token_time is not None:
            self.sink.perf.update({
                "n_decode_tokens": n_decode,
                "n_prefill_tokens": n_prefill,
                "prefill_ms": (self.sink.first_token_time - llm_start) * 1000.0,
                "decode_ms": (llm_end - self.sink.first_token_time) * 1000.0,
            })
        if overrides.get("perf", False):
            printf_perf(self.sink, vision_latency=vision_latency)
        return self.sink.text

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        for r in (self.rknn_vision, self.rknn_llm):
            try:
                r.release()
            except Exception:  # noqa: BLE001
                pass
