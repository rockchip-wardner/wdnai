"""LocateAnything grounding backend — image-in, text-out with bbox / point coords.

Mirrors ``examples/LocateAnything/cpp/main.cc``:

* 448×448 vision RKNN encodes the image into patch tokens.
* Text LLM RKNN runs in ``llm_mode`` (with the standard
  ``StreamSink`` / ``RKLLMCallback`` stack) and consumes the vision
  embeddings via a single ``RKNN3Image`` (one image per call).
* Output is a sequence of ``<ref>label</ref><box>(x1, y1, x2, y2)</box>``
  / ``<ref>label</ref><box>(x, y)</box>`` spans; parsed by
  :func:`wdnai.utils._parse_locate_output` into structured boxes /
  points lists.
"""
from __future__ import annotations

import ctypes
from typing import Any, Dict, Optional, Union

import numpy as np

from ._common import (
    build_llm_args,
    build_llm_callback,
    load_embed_table,
    load_tokenizer,
    resolve_device_id,
    StreamSink,
)


class LocateAnythingBackend:
    """LocateAnything grounding model (vision RKNN + text LLM RKNN).

    On construction, runs ``load_rknn`` + ``init_runtime`` for both the
    LLM and the vision encoder, sets the ChatML template, and exposes
    :py:meth:`run_image` for a single grounding inference.
    """

    def __init__(
        self,
        spec,
        *,
        npu: Optional[int] = None,
        device_id: Optional[Union[bytes, str]] = None,
    ):
        from rknn3lite.api import RKNN3Lite  # type: ignore

        required = (
            ("rknn_path", spec.rknn_path),
            ("weight_path", spec.weight_path),
            ("rknn_vision_path", spec.rknn_vision_path),
            ("weight_vision_path", spec.weight_vision_path),
            ("tokenizer_path", spec.tokenizer_path),
            ("embed_path", spec.embed_path),
        )
        missing = [name for name, value in required if not value]
        if missing:
            raise ValueError(
                f"{spec.name!r} is missing required path(s): {missing}; "
                "fill them in via .with_paths(...)."
            )
        if spec.image_size % 14 != 0:
            raise ValueError(
                f"LocateAnything image_size must be a multiple of 14, got {spec.image_size}"
            )
        self.spec = spec

        # Text LLM (llm_mode). Mirrors TextBackend.__init__.
        self.rknn_llm = RKNN3Lite(llm_mode=True, verbose=spec.verbose)
        if self.rknn_llm.load_rknn(spec.rknn_path, spec.weight_path) != 0:
            raise RuntimeError(f"load_rknn (llm) failed: {spec.rknn_path}")

        # Tokenizer + embed.
        self.tokenizer = load_tokenizer(spec.tokenizer_path)
        self.embeds = load_embed_table(spec.embed_path, spec.vocab_size)

        # Resolve NPU selection lazily, after RKNN3Lite is imported.
        resolved_device_id = resolve_device_id(RKNN3Lite, npu, device_id)

        # Sink + LLM callback + init_runtime.
        self.sink = StreamSink(tokenizer=self.tokenizer)
        cb = build_llm_callback(self.tokenizer, self.embeds, self.sink)
        if self.rknn_llm.init_runtime(
            target=spec.target,
            core_mask=spec.core_mask,
            llm_args=build_llm_args(spec),
            llm_callback=cb,
            device_id=resolved_device_id,
        ) != 0:
            raise RuntimeError("init_runtime (llm) failed")

        # ChatML template — same system/user/assistant skeleton as
        # the rest of the SDK. Inline call so the literal token name
        # stays visible to test/introspection helpers that grep the
        # method source for ``set_chat_template``.
        from ._common import CHATML_ASSISTANT, CHATML_SYSTEM, CHATML_USER
        ret = self.rknn_llm.set_chat_template(CHATML_SYSTEM, CHATML_USER, CHATML_ASSISTANT)
        if ret != 0:
            raise RuntimeError(f"set_chat_template failed: ret={ret}")

        # Vision encoder (NHWC uint8 input, fp16 feature output).
        self.rknn_vision = RKNN3Lite(verbose=spec.verbose)
        if self.rknn_vision.load_rknn(spec.rknn_vision_path, spec.weight_vision_path) != 0:
            raise RuntimeError(f"load_rknn (vision) failed: {spec.rknn_vision_path}")

        # Buffer reused across calls; populated in run_image.
        self._embeds_buf: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    def run_image(self, prompt: str, image: np.ndarray, *, max_new_tokens: int = 1024) -> Dict[str, Any]:
        """Run a single grounding inference and return ``{text, boxes, points}``."""
        from rknn3lite.api import Float16, RKNN3Image  # type: ignore  # noqa: WPS433
        from ..utils import _parse_locate_output  # noqa: WPS433

        # 1. Preprocess: BGR -> RGB -> letterbox -> NHWC uint8.
        from ..utils import bgr_to_rgb, letterbox_resize  # noqa: WPS433
        rgb = bgr_to_rgb(image)
        arr = letterbox_resize(rgb, self.spec.image_size)
        nhwc = np.ascontiguousarray(arr[None, ...], dtype=np.uint8)

        # 2. Vision RKNN inference.
        outs = self.rknn_vision.inference(inputs=[nhwc])

        # 3. Pack vision output as fp16 RKNN3Image token stream.
        self._embeds_buf = np.ascontiguousarray(outs[0], dtype=np.float16)
        if self._embeds_buf.ndim == 2:
            self._embeds_buf = self._embeds_buf[None, ...]

        llm_img = RKNN3Image()
        llm_img.image_embed = self._embeds_buf.ctypes.data_as(ctypes.POINTER(Float16))
        llm_img.n_image_tokens = self._embeds_buf.shape[1]
        llm_img.n_image = self._embeds_buf.shape[0]
        llm_img.image_width = self.spec.image_size
        llm_img.image_height = self.spec.image_size
        llm_img.image_start = b""
        llm_img.image_end = b"</img>"
        llm_img.image_content = b"<IMG_CONTEXT>"

        # 4. Reset sink, build full prompt, run LLM session.
        self.sink.reset()
        full_prompt = "<image> " + prompt
        ret, _stats = self.rknn_llm.session_run(
            inputs=[llm_img],
            prompt=full_prompt,
            enable_thinking=False,
            max_new_tokens=max_new_tokens,
        )
        if ret != 0:
            raise RuntimeError(f"session_run failed: ret={ret}")

        text = self.sink.text
        boxes, points = _parse_locate_output(text)
        return {"text": text, "boxes": boxes, "points": points}

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        try:
            self.rknn_llm.release()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.rknn_vision.release()
        except Exception:  # noqa: BLE001
            pass