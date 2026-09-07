"""Vision-language model backend — for Qwen2.5-VL / Qwen3-VL (image).

Flow (aligned with examples/Qwen2.5VL/test.py, Qwen3VL/test.py):

1. Load the vision RKNN(s) + LLM RKNN.
   * **Multi-resolution vision** (Qwen3-VL-4B and similar): the model
     directory is scanned for ``vision_<name>.<WxH>.1828.rknn`` files
     (e.g. ``vision_Qwen3-VL-4B.384x384.1828.rknn`` +
     ``vision_Qwen3-VL-4B.768x768.1828.rknn``). One ``RKNN3Lite``
     context is loaded per variant; they all share the LLM RKNN. The
     ``.weight`` sibling of each ``.1828.rknn`` is paired in the same
     scan.
   * **Legacy single-resolution** (no ``.<WxH>.`` suffix): falls back to
     a single vision context loaded at ``spec.image_size``. This keeps
     older installations working unchanged.
2. For each image: BGR -> RGB -> resize to the chosen variant's input
   size -> [optional] pruned preprocessing -> ``vision.inference`` ->
   float16.
3. Pack the vision outputs into an ``RKNN3Image`` for the LLM.
4. Qwen3-VL additionally concatenates 3 deepstack aux tensors
   (output indices 1..3).
5. Call ``llm.session_run(inputs=..., prompt=...)``.

Video decoding is not handled here; callers feed pre-decoded frames via
``wdnav.VideoCapture`` and pass them as a list of ndarrays to ``run_images``.

Variant selection
-----------------

Each ``run_images`` call picks ONE vision variant for the whole batch,
because the LLM consumes a single ``RKNN3Image`` with a shared token
shape and a single ``image_width`` / ``image_height``. The selection
rule is: the variant whose ``image_size`` is closest to the first
image's longest side; ties broken by smaller size (cheaper
inference). Images in the same call that would prefer a different
variant are force-fit to the first image's variant (with a warning
logged); typical VLM workloads — video frames at fixed source
resolution — only ever touch one variant per call.

Memory note: loading N vision variants costs N x vision weights. The
Qwen3-VL-4B ``memory_estimate_mb`` budget in
``wdnwebui/app/services/compute_registry.py`` still assumes a single
variant — bump it if you ship both 384x384 and 768x768 on the same
NPU card (384 ≈ 234 MB, 768 ≈ ~700 MB on the 4B export).
"""
from __future__ import annotations

import ctypes
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..utils import (
    bgr_to_rgb,
    letterbox_resize,
    qwen25vl_prune,
    qwen3vl_prune_single,
)
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

LOG = logging.getLogger("wdnai.backends.vlm")


# --------------------------------------------------------------------------- #
# Vision variant discovery
# --------------------------------------------------------------------------- #
# Match ``<prefix>.<WxH>.1828.rknn`` (new naming) or ``<prefix>.1828.rknn``
# (legacy). Captures the prefix as group 1 and the optional W/H as
# groups 2/3. Non-greedy on the prefix so the trailing ``.1828.rknn``
# stays anchored.
_VISION_FILENAME_RE = re.compile(
    r"^(?P<prefix>.+?)(?:\.(?P<w>\d+)x(?P<h>\d+))?\.1828\.rknn$"
)


def _discover_vision_paths(rknn_vision_path: str) -> List[Tuple[int, str, str]]:
    """Scan the directory of ``rknn_vision_path`` for vision variants.

    Returns a list of ``(image_size, rknn_path, weight_path)`` tuples
    sorted by ``image_size`` ascending. ``image_size`` is ``-1`` for the
    legacy single-file case (the caller resolves it from
    ``spec.image_size``).

    Discovery order:

    1. Extract ``<prefix>`` from the filename (everything before the
       optional ``.WxH.1828.rknn``).
    2. Look for ``<prefix>.*x*.1828.rknn`` siblings in the same
       directory. For each match with both width and height equal
       (``WxW``) **and** the corresponding ``.weight`` sibling present,
       include the pair.
    3. If at least one new-style variant was found, return those.
    4. Otherwise fall back to the single file at ``rknn_vision_path``
       (legacy) with ``image_size=-1``.
    """
    p = Path(rknn_vision_path)
    if not p.name:
        raise ValueError(f"invalid rknn_vision_path: {rknn_vision_path!r}")
    d = p.parent
    name = p.name
    m = _VISION_FILENAME_RE.match(name)
    if not m:
        raise ValueError(
            f"unrecognized rknn_vision filename: {name!r} "
            "(expected '<prefix>[.<WxH>].1828.rknn')"
        )
    base = m.group("prefix")

    # 1) New-style variants in the same directory
    variants: List[Tuple[int, str, str]] = []
    for f in sorted(d.glob(f"{base}.*x*.1828.rknn")):
        fm = _VISION_FILENAME_RE.match(f.name)
        if not fm or fm.group("prefix") != base:
            continue
        ws, hs = fm.group("w"), fm.group("h")
        if not ws or not hs:
            continue
        w, h = int(ws), int(hs)
        if w != h:
            LOG.warning("vlm: skip non-square vision variant: %s", f.name)
            continue
        weight = str(f).replace(".1828.rknn", ".weight")
        if not Path(weight).is_file():
            LOG.warning(
                "vlm: skip %s (missing sibling %s)", f.name, weight,
            )
            continue
        variants.append((w, str(f), weight))

    if variants:
        return sorted(variants, key=lambda x: x[0])

    # 2) Legacy fallback — single file, no .WxH. suffix
    weight = rknn_vision_path.replace(".1828.rknn", ".weight")
    return [(-1, rknn_vision_path, weight)]


# --------------------------------------------------------------------------- #
class _VisionVariant:
    """One resolution variant of the Qwen3-VL vision model.

    Attributes
    ----------
    image_size : int
        Square training resolution (e.g. 384 or 768). For legacy
        single-file variants this is filled in from ``spec.image_size``
        during ``VLMBackend.__init__``.
    rknn : RKNN3Lite
        Initialised vision RKNN context (one per variant).
    rknn_path : str
        Absolute path to the ``.1828.rknn`` file (for logging).
    weight_path : str
        Absolute path to the ``.weight`` file (for logging).
    """

    __slots__ = ("image_size", "rknn", "rknn_path", "weight_path",
                 "_outputs_attr_cache")

    def __init__(self, image_size: int, rknn, rknn_path: str,
                 weight_path: str) -> None:
        self.image_size = image_size
        self.rknn = rknn
        self.rknn_path = rknn_path
        self.weight_path = weight_path
        self._outputs_attr_cache: Optional[list] = None

    def get_outputs_tensor_attr(self):
        """Lazy: rknn.get_outputs_tensor_attr() the first time it's needed."""
        if self._outputs_attr_cache is None:
            self._outputs_attr_cache = self.rknn.get_outputs_tensor_attr()
        assert self._outputs_attr_cache is not None
        return self._outputs_attr_cache


# --------------------------------------------------------------------------- #
class VLMBackend:
    """Vision-language model backend. ``spec.variant`` decides between 2.5VL and 3VL."""

    VARIANT_25VL = "qwen2.5-vl"
    VARIANT_3VL = "qwen3-vl"

    def __init__(
        self,
        spec,
        *,
        npu: Optional[int] = None,
        device_id: Optional[Union[bytes, str]] = None,
    ):
        from .._librknn import RKNN3Lite

        if not spec.rknn_llm_path or not spec.rknn_vision_path:
            raise ValueError(
                f"{spec.name!r} is missing rknn_llm_path / rknn_vision_path; "
                "fill them in via .with_paths(...)."
            )
        self.spec = spec
        self.variant = spec.variant

        # ---- llm (single, unchanged) ------------------------------------
        self.rknn_llm = RKNN3Lite(llm_mode=True, verbose=spec.verbose)
        lp = spec.rknn_llm_path
        if self.rknn_llm.load_rknn(lp, lp.replace(".1828.rknn", ".weight")) != 0:
            raise RuntimeError(f"llm load_rknn failed: {lp}")

        self.tokenizer = load_tokenizer(spec.tokenizer_path)
        self.embeds = load_embed_table(spec.embed_path, spec.vocab_size)

        # Resolve NPU selection **once** after RKNN3Lite is imported;
        # vision + llm init_runtime share the same resolved bytes so the
        # whole VLM stays on a single card.
        resolved_device_id = resolve_device_id(RKNN3Lite, npu, device_id)

        # ---- vision (multi-resolution) ----------------------------------
        # Scan ``spec.rknn_vision_path``'s directory for every
        # ``<prefix>.<WxH>.1828.rknn`` sibling. Each variant gets its own
        # RKNN3Lite handle but shares ``resolved_device_id`` (single NPU).
        # All variant init_runtime calls happen before llm init_runtime
        # so any per-variant load failure aborts cleanly before the LLM
        # is even brought up.
        self._vision_variants: List[_VisionVariant] = self._load_vision_variants(
            spec.rknn_vision_path,
            target=spec.target,
            core_mask=spec.core_mask,
            device_id=resolved_device_id,
            fallback_image_size=spec.image_size,
        )

        # Backward-compat alias: code (and external callers) that still
        # reference ``self.rknn_vision`` get the smallest variant, which
        # matches the previous single-model behaviour (the smallest
        # variant is always the cheapest fallback choice).
        self.rknn_vision = self._vision_variants[0].rknn

        # ---- llm init_runtime -------------------------------------------
        self.sink = StreamSink(tokenizer=self.tokenizer)
        cb = build_llm_callback(self.tokenizer, self.embeds, self.sink)
        if self.rknn_llm.init_runtime(
            target=spec.target,
            core_mask=spec.core_mask,
            llm_args=build_llm_args(spec),
            llm_callback=cb,
            device_id=resolved_device_id,
        ) != 0:
            raise RuntimeError("llm init_runtime failed")
        apply_chat_template(self.rknn_llm)

        # Pin the vision feature tensor that backs ``RKNN3Image.image_embed``
        # across ``_build_image_inputs`` calls. The ctypes pointer inside
        # the struct is just an integer — without an extra reference the
        # numpy buffer is freed the moment the helper returns and the
        # driver reads garbage.
        self._vision_embed_pin: List[np.ndarray] = []

    @staticmethod
    def _load_vision_variants(
        rknn_vision_path: str,
        *,
        target: str,
        core_mask: int,
        device_id,
        fallback_image_size: int,
    ) -> List[_VisionVariant]:
        from .._librknn import RKNN3Lite

        discovered = _discover_vision_paths(rknn_vision_path)
        if not discovered:
            raise RuntimeError(
                f"no vision variants discovered under {rknn_vision_path!r}"
            )
        variants: List[_VisionVariant] = []
        for size, rknn_path, weight_path in discovered:
            rknn = RKNN3Lite()
            if rknn.load_rknn(rknn_path, weight_path) != 0:
                raise RuntimeError(
                    f"vision load_rknn failed: {rknn_path} "
                    "(if a .1828.rknn exists but the file is corrupt, "
                    "remove or replace it; the backend will then "
                    "skip it during the next start)"
                )
            if rknn.init_runtime(
                target=target, core_mask=core_mask, device_id=device_id,
            ) != 0:
                raise RuntimeError(
                    f"vision init_runtime failed: {rknn_path}"
                )
            actual_size = size if size > 0 else fallback_image_size
            variant = _VisionVariant(
                image_size=actual_size,
                rknn=rknn,
                rknn_path=rknn_path,
                weight_path=weight_path,
            )
            variants.append(variant)
            LOG.info(
                "vlm: loaded vision variant image_size=%d rknn=%s",
                actual_size, rknn_path,
            )
        if not variants:
            raise RuntimeError(
                f"no vision variants loaded for {rknn_vision_path!r}"
            )
        return variants

    # ------------------------------------------------------------------ #
    # Variant selection
    # ------------------------------------------------------------------ #
    def available_image_sizes(self) -> List[int]:
        """Return the sorted list of vision-model native resolutions
        loaded into this backend.

        Use this for UI dropdowns (the 音视频描述 page in wdnwebui
        queries this list at page-open time to populate its
        ``#vd-detect-size`` selector). The list contains the per-variant
        ``image_size`` values — i.e. the square dimensions each vision
        RKNN was exported at — in ascending order.

        With only one variant (legacy single-resolution install) the
        list has exactly one entry; the values match ``spec.image_size``.
        When both 384x384 and 768x768 variants are loaded the list is
        ``[384, 768]``.
        """
        return [v.image_size for v in self._vision_variants]

    def _pick_variant(
        self,
        img_bgr: Optional[np.ndarray] = None,
        *,
        image_size: Optional[int] = None,
    ) -> _VisionVariant:
        """Pick a vision variant for the current call.

        Selection rule:

        * ``image_size`` given → pick the variant whose ``image_size``
          matches exactly; if no exact match, fall back to the
          **nearest** size with a warning logged (this is the path the
          UI takes when the user picks a dropdown value the model
          doesn't actually expose, e.g. legacy installs with only 384
          and the user asks for 768).
        * ``image_size`` not given → auto-pick by input image's longest
          side (legacy behaviour preserved).
        * With only one variant loaded → always return it, regardless
          of ``image_size`` / image shape.

        Ties on the auto path are broken by smaller size (cheaper
        inference wins).
        """
        variants = self._vision_variants
        if not variants:
            raise RuntimeError("VLMBackend has no vision variants loaded")
        if len(variants) == 1:
            return variants[0]
        if image_size is not None:
            target = int(image_size)
            exact = [v for v in variants if v.image_size == target]
            if exact:
                return exact[0]
            # No exact match — nearest wins, log a warning so callers
            # know the requested size isn't actually exposed by this
            # model's loaded variants.
            best = min(variants, key=lambda v: abs(v.image_size - target))
            LOG.warning(
                "vlm: requested image_size=%d not exposed by any "
                "loaded variant (have %s); falling back to nearest %d",
                target, [v.image_size for v in variants], best.image_size,
            )
            return best
        # Auto-pick path: minimise |image_size - max(h, w)|.
        if img_bgr is None:
            return variants[0]
        try:
            h, w = img_bgr.shape[:2]
            target = max(int(h), int(w))
        except Exception:
            LOG.warning("vlm: cannot read image shape; defaulting to smallest variant")
            return variants[0]
        best = variants[0]
        best_dist = abs(best.image_size - target)
        for v in variants[1:]:
            d = abs(v.image_size - target)
            if d < best_dist or (d == best_dist and v.image_size < best.image_size):
                best = v
                best_dist = d
        return best

    # ------------------------------------------------------------------ #
    # Image preprocessing — variant-dispatched
    # ------------------------------------------------------------------ #
    def _preprocess_to_variant(
        self, img_bgr: np.ndarray, variant: _VisionVariant,
    ) -> np.ndarray:
        """Resize ``img_bgr`` to ``variant.image_size`` and run variant-
        specific preprocessing.

        Vision export shape is decided by inspecting the model's
        actual ``in_attr.n_dims`` at runtime, **not** by ``spec.use_prune``:

        * ``n_dims == 4``  → full-model NHWC input ``(1, H, W, 3)``,
          optionally stacked to ``(2, H, W, 3)`` for the Qwen3-VL
          "two-frame input" video mode.
        * ``n_dims != 4``  → pruned/patchified input ``(num_patches,
          embed_dim)`` produced by ``qwen3vl_prune_single`` (3VL) or
          ``qwen25vl_prune`` (2.5VL).

        The previous version of this method used ``self.spec.use_prune``
        as the dispatch key, which broke for the 4B 768x768 export
        in this environment: the file ``vision_Qwen3-VL-4B.768x768.
        1828.rknn`` is actually a pruned model (``n_dims=2``) even
        though the preset's ``use_prune=False`` flagged it as full.
        """
        size = variant.image_size
        rgb = letterbox_resize(bgr_to_rgb(img_bgr), size)
        in_attr = variant.rknn.get_inputs_tensor_attr()[0]

        if in_attr.n_dims == 4:
            # Full-model path: NHWC ``(1, H, W, 3)``; some Qwen3-VL
            # exports expect a 2-image stack at index 0 (the
            # "two-frame input" video mode for VLM).
            nhwc = rgb.reshape(1, size, size, 3)
            if in_attr.shape[0] == 2:
                nhwc = np.concatenate([nhwc, nhwc], axis=0)
            # The model may expect uint8 (raw pixels, patch embedding
            # happens inside the graph) or float16 (preprocessed
            # embeddings). Match the queried input dtype so the model
            # sees the bytes it was exported against.
            from .._librknn import RKNN3TensorType
            model_dtype = RKNN3TensorType(int(in_attr.dtype))
            if model_dtype == RKNN3TensorType.RKNN3_TENSOR_UINT8:
                nhwc = nhwc.astype(np.uint8)
            else:
                nhwc = nhwc.astype(np.float16)
            return nhwc

        # Pruned / patchified path: 2-D input (num_patches, embed_dim).
        # ``qwen3vl_prune_single`` / ``qwen25vl_prune`` produce the
        # right flattened-patch layout that this export expects.
        nchw = rgb.transpose(2, 0, 1).reshape(1, 3, size, size)
        if self.variant == self.VARIANT_25VL:
            feat = qwen25vl_prune(nchw)
        else:  # 3VL
            feat = qwen3vl_prune_single(nchw)
        return np.float16(feat)

    def _preprocess_image(self, img_bgr: np.ndarray) -> Tuple[_VisionVariant, np.ndarray]:
        variant = self._pick_variant(img_bgr)
        return variant, self._preprocess_to_variant(img_bgr, variant)

    def _vision_infer(
        self,
        img_bgr: np.ndarray,
        variant: Optional[_VisionVariant] = None,
        image_size: Optional[int] = None,
    ):
        """Run vision inference.

        Resolution priority: explicit ``image_size`` → explicit
        ``variant`` → auto-pick from image resolution. When
        ``image_size`` is given, the matching variant is selected
        via ``_pick_variant(image_size=...)``; this is the path the
        wdnwebui UI takes when the user picks a target size in the
        音视频描述 page's ``#vd-detect-size`` dropdown.
        """
        if variant is None:
            variant = self._pick_variant(img_bgr, image_size=image_size)
        feat = self._preprocess_to_variant(img_bgr, variant)
        t0 = time.perf_counter()
        outs = variant.rknn.inference(inputs=[feat])
        return variant, outs, time.perf_counter() - t0

    # ------------------------------------------------------------------ #
    # LLM input assembly
    # ------------------------------------------------------------------ #
    def _build_image_inputs(
        self,
        image_bgr_list: List[np.ndarray],
        *,
        image_size: Optional[int] = None,
    ) -> Tuple[str, list, float]:
        """Return ``(prompt_with_prefix, rknn_inputs, vision_latency_sec)``.

        All images in one call share ONE vision variant — the LLM
        consumes a single ``RKNN3Image`` (concatenated tokens + a
        single ``image_width`` / ``image_height``), so different
        vision models with different token shapes cannot be batched.
        The variant is picked once: an explicit ``image_size`` wins
        over the auto-pick from image resolution; subsequent images
        that would prefer a different variant are force-fit to the
        chosen one (with a warning logged). Typical VLM workloads —
        video frames at fixed source resolution — only touch one
        variant per call.
        """
        from .._librknn import (
            Float16,
            RKNN3Image,
            RKNN3AuxTensorWrapper,
        )

        if not image_bgr_list:
            raise ValueError("image_bgr_list must not be empty")

        chosen_variant = self._pick_variant(image_bgr_list[0], image_size=image_size)
        feats: List[np.ndarray] = []
        deepstack: List[List[np.ndarray]] = [[], [], []]
        vision_latency_total = 0.0

        for idx, img in enumerate(image_bgr_list):
            preferred = self._pick_variant(img)
            if preferred is not chosen_variant:
                LOG.warning(
                    "vlm: image[%d] (size %dx%d) prefers %dx%d variant; "
                    "force-fitting to call's chosen %dx%d variant "
                    "(LLM consumes a single RKNN3Image per call).",
                    idx, img.shape[1], img.shape[0],
                    preferred.image_size, preferred.image_size,
                    chosen_variant.image_size, chosen_variant.image_size,
                )
            variant, outs, lat = self._vision_infer(img, variant=chosen_variant)
            vision_latency_total += lat

            out0 = outs[0]
            LOG.info("vlm: vision output[%d] shape=%s dtype=%s", idx, out0.shape, out0.dtype)
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
        # ``features`` is always packed as ``(batch, n_tokens, embed_dim)``
        # (single image ⇒ ``(1, n_patches, dim)`` for Qwen3.5-VL's
        # 2-D vision output; the original 1-D token-only path also goes
        # through the ``expand_dims`` branch). The C-side
        # ``n_image_tokens`` is the per-image token count; ``n_image`` is
        # the batch count. Reading ``features.shape[0] / features.shape[1]``
        # gives the right values for both layouts.
        llm_img.n_image_tokens = features.shape[1] if features.ndim >= 2 else features.shape[0]
        llm_img.n_image = features.shape[0] if features.ndim >= 3 else 1
        llm_img.image_width = chosen_variant.image_size
        llm_img.image_height = chosen_variant.image_size
        llm_img.image_start = "<|vision_start|>".encode("utf-8")
        llm_img.image_end = "<|vision_end|>".encode("utf-8")
        llm_img.image_content = "<|image_pad|>".encode("utf-8")
        inputs.append(llm_img)

        if self.variant == self.VARIANT_3VL and deepstack[0]:
            # Match examples/Qwen3VL/test.py single-image behavior:
            # align_size is NOT multiplied by features.shape[0].
            aligned = chosen_variant.get_outputs_tensor_attr()[0].aligned_size
            for i in range(3):
                aux = RKNN3AuxTensorWrapper()
                aux.index = 2 + i
                aux.aux_data = np.concatenate(deepstack[i], axis=0)
                aux.align_size = aligned
                inputs.append(aux)

        # ``llm_img.image_embed`` is a raw pointer into ``features``;
        # without an extra reference the numpy array is garbage-collected
        # the moment this method returns, and the driver reads freed
        # memory. Pin until the next ``_build_image_inputs`` call (or
        # session reset) clears the list.
        self._vision_embed_pin = [features]

        prefix = "<image>" * len(image_bgr_list)
        return prefix, inputs, vision_latency_total

    # ------------------------------------------------------------------ #
    def run_text(self, prompt: str, **overrides: Any) -> str:
        """Pure-text inference (no image). The prompt is sent directly to the LLM."""
        return self._run_llm(prompt, inputs=None, vision_latency=None, **overrides)

    def run_images(
        self,
        prompt: str,
        images: List[np.ndarray],
        *,
        image_size: Optional[int] = None,
        **overrides: Any,
    ) -> str:
        """Run VLM inference over a batch of BGR frames.

        ``image_size`` (optional) — force the call onto the vision
        variant trained at this resolution. Must be one of the values
        returned by ``available_image_sizes()``. When ``None`` (the
        default) the backend picks the closest variant to the input
        image's longest side, preserving the legacy auto-pick
        behaviour.
        """
        if not images:
            raise ValueError("images must not be empty")
        if image_size is not None:
            available = self.available_image_sizes()
            if image_size not in available:
                # Caller-side validation should have caught this; warn
                # loudly so misconfiguration is debuggable.
                LOG.warning(
                    "run_images: requested image_size=%d not in "
                    "available_image_sizes=%s; nearest will be used",
                    image_size, available,
                )
        prefix, inputs, vlat = self._build_image_inputs(images, image_size=image_size)
        full_prompt = f"{prefix}{prompt}"
        return self._run_llm(full_prompt, inputs, vision_latency=vlat, **overrides)

    def _run_llm(
        self, prompt: str, inputs, vision_latency: Optional[float], **overrides: Any,
    ):
        self.sink.reset()
        kwargs: Dict[str, Any] = {"prompt": prompt}
        if inputs is not None:
            kwargs["inputs"] = inputs
        # Per-call overrides (max_new_tokens / enable_thinking / keep_history)
        for k in ("max_new_tokens", "enable_thinking", "keep_history"):
            if k in overrides and overrides[k] is not None:
                kwargs[k] = overrides[k]
        ret, stats = self.rknn_llm.session_run(**kwargs)
        # Drop the vision feature pin once the LLM has consumed the
        # embeddings; the pin exists so the numpy array survives until
        # ``session_run`` reads the pointer, not for the lifetime of the
        # process. Resetting here also frees the previous call's buffer
        # in long-running video loops.
        if inputs is not None:
            self._vision_embed_pin.clear()
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
        for v in getattr(self, "_vision_variants", []) or []:
            try:
                v.rknn.release()
            except Exception:  # noqa: BLE001
                pass
        try:
            self.rknn_llm.release()
        except Exception:  # noqa: BLE001
            pass