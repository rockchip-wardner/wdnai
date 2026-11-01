"""Depth Anything V3 backend — monocular depth estimation (image-in, depth-out).

The RKNN export of DA3-BASE is split into **three** stages that have to be
driven as one pipeline (mirrors ``rknn3-model-zoo/examples/depth_anything_v3/
cpp/main.cpp``)::

    local   images [1,H,W,3] uint8 NHWC   ->  local_tokens [1,L,D]   fp16
    global  tokens [V,L,1,D]             ->  feat_5/7/9/11 [V,F,1,G] fp16
    head    feat_* [V,F,G]                ->  depth/conf     [1,V,H,W] fp16

``local`` runs once per image, ``global`` fuses the concatenated tokens of
all views, ``head`` turns the fused features into one depth map plus one
confidence map per view. The number of views ``V`` is **baked into the
export** (``global``'s input shape is ``[V,L,1,D]``), so it is read off the
model at load time instead of being a runtime parameter.

Everything in this module that touches pixels is a deliberate port of the
C++ demo — the preprocessing is a *stretch* bilinear resize (not a
letterbox), the visualisation uses the same Turbo polynomial, and the
inverse-depth percentile window is shared across all views. Keeping the two
implementations bit-compatible is what makes the Python example comparable
to the C++ demo output.

The backend is **not** a text generator, so it deliberately does not
implement ``run_text``: a depth map cannot be expressed as a prompt. It
exposes ``run_images`` instead, and the result carries the raw arrays on
the object (``DepthResult.depth`` / ``.confidence``) plus a JSON summary in
``.text`` so the value still travels the text-shaped RPC wire unchanged.
"""
from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from ._common import resolve_device_id

__all__ = [
    "DepthResult",
    "DepthAnythingBackend",
    "preprocess_image_rgb",
    "build_depth_valid_mask",
    "turbo_color",
    "save_depth_heatmaps",
    "depth_weight_path",
]

#: ``RKNN3_QUERY_CORE_MEM_SIZE`` / ``RKNN3_FLAG_MEMORY_CACHEABLE`` — the
#: query commands and flags are part of the stable rknn3_api.h ABI (same
#: values the ASR backend uses).
_CMD_CORE_MEM_SIZE = 6
_FLAG_CACHEABLE = 1


def _fp16_to_fp32(fp16: np.ndarray) -> np.ndarray:
    """Convert fp16 → fp32 **exactly** as the C++ demo's ``fp16_to_fp32``.

    The demo (and every RKNN3 C example) uses the bit-twiddling converter
    in ``rknpu3/include/float16.h`` rather than a hardware conversion.
    It handles normals, infinities and NaN correctly, but it is **lossy
    for fp16 denormals**: when the exponent field is 0 it leaves the
    fp32 exponent at 0 and only shifts the mantissa, so a denormal fp16
    lands at ~1e-39 instead of its true value (fp16 denormals are
    below 6.1e-5 and the DA3 tokens do contain a few of them).

    Those values are ~0 either way, but they are *not* the same 0: the
    global stage sees them, and the difference survives into the depth
    map (the depth mean moves by ~5e-4 and ~80 % of the global features
    shift by one fp16 ULP). Since this backend's job is to reproduce the
    C++ pipeline's numbers, the same conversion is applied here — using
    numpy's ``astype(np.float32)`` instead would silently produce a
    slightly different depth map than the reference demo.
    """
    bits = np.ascontiguousarray(fp16, dtype=np.float16).view(np.uint16).astype(np.uint32)
    sign = (bits & np.uint32(0x8000)) << np.uint32(16)
    exp = (bits >> np.uint32(10)) & np.uint32(0x1F)
    frac = bits & np.uint32(0x03FF)

    # exp == 31: Inf (frac == 0) / NaN (frac != 0, collapsed to frac 1).
    is_special = exp == np.uint32(31)
    out_exp = np.where(
        is_special,
        np.uint32(255),
        np.where(exp == np.uint32(0), np.uint32(0),
                 (exp.astype(np.int64) - 15 + 127).astype(np.uint32)),
    )
    out_frac = np.where(
        is_special,
        np.where(frac != np.uint32(0), np.uint32(1), np.uint32(0)),
        frac << np.uint32(13),
    )
    out_bits = sign | (out_exp << np.uint32(23)) | out_frac
    return np.ascontiguousarray(out_bits).view(np.float32)


# --------------------------------------------------------------------------- #
# Turbo colormap — polynomial fit, ported from ``turbo_color`` in main.cpp  #
# --------------------------------------------------------------------------- #
def turbo_color(values: np.ndarray) -> np.ndarray:
    """Map ``values`` in ``[0, 1]`` to Turbo-colour RGB (``uint8``).

    Vectorised port of ``turbo_color`` in the C++ demo: the same 5th-order
    polynomial per channel, the same ``[0, 1]`` clamp, and the same
    ``(uint8_t)(c * 255.0f + 0.5f)`` rounding.
    """
    x = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    red = 0.13572138 + 4.61539260 * x - 42.66032258 * x2 + 132.13108234 * x3 \
        - 152.94239396 * x4 + 59.28637943 * x5
    green = 0.09140261 + 2.19418839 * x + 4.84296658 * x2 - 14.18503333 * x3 \
        + 4.27729857 * x4 + 2.82956604 * x5
    blue = 0.10667330 + 12.64194608 * x - 60.58204836 * x2 + 110.36276771 * x3 \
        - 89.90310912 * x4 + 27.34824973 * x5
    out = np.stack([
        np.clip(red, 0.0, 1.0),
        np.clip(green, 0.0, 1.0),
        np.clip(blue, 0.0, 1.0),
    ], axis=-1)
    # (uint8_t)(c * 255.0f + 0.5f) — round half up, then truncate.
    return (out * 255.0 + 0.5).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Preprocessing — port of ``preprocess_image_rgb`` / ``crop_and_scale_image_c``
# --------------------------------------------------------------------------- #
def _fma32(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """float32 fused multiply-add — ``round_f32(exact(a*b) + c)``.

    numpy has no FMA, but for float32 operands the product ``a*b`` is
    **exact** in float64 (24 + 24 = 48 mantissa bits ≤ 53), so doing the
    multiply and the add in float64 and rounding **once** to float32 at
    the end reproduces the hardware FMA bit for bit.
    """
    return (a.astype(np.float64) * b.astype(np.float64)
            + c.astype(np.float64)).astype(np.float32)


def preprocess_image_rgb(bgr: np.ndarray, dst_height: int, dst_width: int) -> np.ndarray:
    """Resize a BGR image to ``(dst_height, dst_width)`` RGB ``uint8``.

    The C++ demo calls ``convert_image`` with **no** source/destination
    rectangle, i.e. a plain *stretch* resize (aspect ratio is **not**
    preserved) and **not** a letterbox. The interpolation is the bilinear
    variant of ``crop_and_scale_image_c``:

    * ``src_x = int(dst_x * src_w / dst_w)``, ``x_diff = dst_x * ratio - src_x``
    * at the right/bottom edge the "neighbour" pixel is the one on the
      *inside* of the image (the C code steps back by one instead of
      clamping).

    The C++ also gets **fused** into FMAs by the compiler (GCC's default
    ``-ffp-contract=fast``), which changes the result by one LSB on ~0.4 %
    of the pixels compared to a literal transcription of the expression.
    The evaluation order below mirrors what the shipped demo binary
    actually executes — see ``crop_and_scale_image_c.constprop.0`` in
    ``rknn_da3_base_demo``::

        acc = (B * x_diff) * (1 - y_diff)          // two rounded multiplies
        acc = fma(A * (1 - x_diff), (1 - y_diff), acc)
        acc = fma(C * y_diff, (1 - x_diff), acc)
        acc = fma(y_diff, (D * x_diff), acc)

    Reproducing it makes the preprocessed bytes handed to the ``local``
    stage **identical** to the C++ demo's, so the depth maps come out
    bit-for-bit equal instead of one fp16 ULP apart.
    """
    if bgr is None or bgr.size == 0:
        raise ValueError("empty image")
    src_h, src_w = bgr.shape[:2]
    if src_h == dst_height and src_w == dst_width:
        return np.ascontiguousarray(bgr[:, :, ::-1])

    x_ratio = np.float32(src_w) / np.float32(dst_width)
    y_ratio = np.float32(src_h) / np.float32(dst_height)

    dx = np.arange(dst_width, dtype=np.float32)
    dy = np.arange(dst_height, dtype=np.float32)
    fx = dx * x_ratio
    fy = dy * y_ratio
    src_x = fx.astype(np.int64)
    src_y = fy.astype(np.int64)
    x_diff = (fx - src_x.astype(np.float32)).astype(np.float32)
    y_diff = (fy - src_y.astype(np.float32)).astype(np.float32)

    # Right / bottom neighbour: one step *inside* the image at the edges
    # (the C code subtracts one pixel instead of clamping to itself).
    x1 = np.where(src_x >= src_w - 1, src_x - 1, src_x + 1)
    y1 = np.where(src_y >= src_h - 1, src_y - 1, src_y + 1)

    src = bgr.astype(np.float32)
    a = src[src_y[:, None], src_x[None, :], :]     # top-left
    b = src[src_y[:, None], x1[None, :], :]         # top-right
    c = src[y1[:, None], src_x[None, :], :]         # bottom-left
    d = src[y1[:, None], x1[None, :], :]            # bottom-right

    one = np.float32(1.0)
    wx = x_diff[None, :, None]
    wy = y_diff[:, None, None]
    omx = one - wx
    omy = one - wy

    acc = (b * wx) * omy
    acc = _fma32(a * omx, omy, acc)
    acc = _fma32(c * wy, omx, acc)
    acc = _fma32(wy, d * wx, acc)
    # (unsigned char) cast — truncation toward zero.
    return np.ascontiguousarray(acc.astype(np.uint8)[:, :, ::-1])


def build_depth_valid_mask(rgb: np.ndarray) -> np.ndarray:
    """Mask of the pixels that belong to the real picture.

    Port of ``build_depth_valid_mask``: the letterboxed / padded border of
    a resized image is black, and a black pixel has no meaningful depth, so
    the C++ demo scans the borders of the **preprocessed** image for the
    first column/row that is bright enough and treats the rectangle
    between those as valid. If the detected rectangle is degenerate
    (smaller than half the frame in either axis) the whole frame is used.

    Returns a ``uint8`` array of the same shape, ``1`` = valid.
    """
    height, width = rgb.shape[:2]
    dark_threshold = 16
    min_column_pixels = height // 20 if height // 20 > 0 else 1
    min_row_pixels = width // 20 if width // 20 > 0 else 1

    bright = (rgb.max(axis=2) > dark_threshold)

    left = 0
    while left < width:
        if int(bright[:, left].sum()) >= min_column_pixels:
            break
        left += 1
    right = width
    while right > left:
        if int(bright[:, right - 1].sum()) >= min_column_pixels:
            break
        right -= 1
    top = 0
    while top < height:
        if int(bright[top, :].sum()) >= min_row_pixels:
            break
        top += 1
    bottom = height
    while bottom > top:
        if int(bright[bottom - 1, :].sum()) >= min_row_pixels:
            break
        bottom -= 1

    if (right - left) < width // 2 or (bottom - top) < height // 2:
        left, right, top, bottom = 0, width, 0, height

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[top:bottom, left:right] = 1
    return mask


def save_depth_heatmaps(
    output_dir: str,
    depth: np.ndarray,
    valid_masks: np.ndarray,
    views: int,
    height: int,
    width: int,
) -> Dict[str, Any]:
    """Render the depth maps as Turbo heatmaps and write them as JPEGs.

    Port of ``save_depth_heatmaps``: positive depth is turned into
    *inverse* depth, the 2% / 98% percentiles are computed over **all**
    views at once (so the same colour means the same relative depth in
    every view), and each view is written as ``depth_<NN>.jpg`` plus a
    ``depth_montage.jpg`` with 5 columns.

    Returns a dict with the computed ``lower`` / ``upper`` window and the
    number of valid pixels, so callers can assert parity with the C++ run.
    """
    import cv2

    pixels = height * width
    depth = np.asarray(depth, dtype=np.float32).reshape(views, pixels)
    valid_masks = np.asarray(valid_masks, dtype=np.uint8).reshape(views, pixels)

    positive = valid_masks.astype(bool) & (depth > 0.0) & np.isfinite(depth)
    inverse = np.where(positive, 1.0 / np.where(positive, depth, 1.0), 0.0)
    valid_values = inverse[positive]
    valid_count = int(valid_values.size)
    if valid_count <= 10:
        raise RuntimeError("all depth views have too few valid pixels")

    sorted_values = np.sort(valid_values)
    lower = float(sorted_values[int(0.02 * (valid_count - 1))])
    upper = float(sorted_values[int(0.98 * (valid_count - 1))])
    if upper <= lower:
        upper = lower + 1e-6

    os.makedirs(output_dir, exist_ok=True)

    # The C++ writes the heatmaps with libjpeg-turbo at quality 95 and
    # **4:2:2** chroma subsampling (``TJSAMP_422``); OpenCV's default is
    # 4:2:0, which visibly bleeds colour across the sharp Turbo ramps.
    # The remaining byte difference against the C++ output is the JPEG
    # codec itself (libjpeg-turbo vs OpenCV's libjpeg) — the pixels
    # handed to the encoder are identical.
    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    sampling = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_422", None)
    if sampling is not None:
        jpeg_params += [int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR), int(sampling)]

    montage_columns = views if views < 5 else 5
    montage_rows = (views + montage_columns - 1) // montage_columns
    montage_width = montage_columns * width
    montage_height = montage_rows * height
    montage = np.zeros((montage_height, montage_width, 3), dtype=np.uint8)

    for view in range(views):
        view_depth = depth[view]
        view_valid = positive[view]
        rgb = np.zeros((pixels, 3), dtype=np.uint8)
        if view_valid.any():
            normalized = (inverse[view][view_valid] - lower) / (upper - lower)
            rgb[view_valid] = turbo_color(normalized)
        image = rgb.reshape(height, width, 3)
        path = os.path.join(output_dir, "depth_%02d.jpg" % view)
        cv2.imwrite(path, image[:, :, ::-1], jpeg_params)
        montage_y = (view // montage_columns) * height
        montage_x = (view % montage_columns) * width
        montage[montage_y:montage_y + height, montage_x:montage_x + width] = image

    montage_path = os.path.join(output_dir, "depth_montage.jpg")
    cv2.imwrite(montage_path, montage[:, :, ::-1], jpeg_params)

    return {
        "lower": lower,
        "upper": upper,
        "valid_count": valid_count,
        "montage_columns": montage_columns,
        "montage_rows": montage_rows,
        "montage_width": montage_width,
        "montage_height": montage_height,
        "montage_path": montage_path,
    }


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class DepthResult:
    """What the depth backend produced for one batch of views.

    ``text`` is the JSON summary that travels the (text-shaped) RPC wire;
    the raw arrays stay on the object for local callers. ``depth`` has
    shape ``(views, H, W)`` and ``confidence`` the same.

    ``include_maps=True`` additionally base64-encodes both maps into the
    JSON. That is meant for **local** use: one fp16 view of 280×280 is
    ~157 KB raw (~209 KB base64), so all ten views together exceed the
    1 MiB per-message ceiling of the DDS transport
    (``wdndds/ai_session.py``). Over RPC, use the summary only — a
    dedicated op would be needed to ship the maps themselves.
    """

    depth: np.ndarray
    confidence: np.ndarray
    views: int
    height: int
    width: int
    #: Valid-pixel mask per view (``uint8``, 1 = valid) — the same rectangle
    #: the C++ demo derives from the preprocessed image.
    masks: Optional[np.ndarray] = None
    include_maps: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """JSON summary of the run (the wire representation)."""
        payload: Dict[str, Any] = {
            "model": "DepthAnythingV3",
            "views": int(self.views),
            "height": int(self.height),
            "width": int(self.width),
            "depth": _stats(self.depth),
            "confidence": _stats(self.confidence),
            "per_view": [
                {
                    "view": i,
                    "depth": _stats(self.depth[i]),
                    "confidence": _stats(self.confidence[i]),
                }
                for i in range(int(self.views))
            ],
        }
        payload.update(self.extra)
        if self.include_maps:
            payload["depth_b64"] = base64.b64encode(
                np.ascontiguousarray(self.depth, dtype=np.float16).tobytes()
            ).decode("ascii")
            payload["confidence_b64"] = base64.b64encode(
                np.ascontiguousarray(self.confidence, dtype=np.float16).tobytes()
            ).decode("ascii")
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

    def decode_maps(self) -> None:
        """Decode the base64 maps back into ``depth`` / ``confidence``."""
        if not self.include_maps:
            return
        d = self.extra.get("_depth_b64")
        c = self.extra.get("_confidence_b64")
        if d:
            self.depth = np.frombuffer(base64.b64decode(d), dtype=np.float16).astype(
                np.float32).reshape(self.views, self.height, self.width)
        if c:
            self.confidence = np.frombuffer(base64.b64decode(c), dtype=np.float16).astype(
                np.float32).reshape(self.views, self.height, self.width)


def _stats(arr: np.ndarray) -> Dict[str, float]:
    """min / max / mean of a float array (the C++ ``print_tensor_summary``).

    The mean accumulates in **float64** — the C++ sums into a ``double``
    and a float32 accumulation can land on the neighbouring last digit.
    """
    a = np.asarray(arr, dtype=np.float32).reshape(-1)
    return {
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.astype(np.float64).sum()) / a.size,
    }


# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #
def depth_weight_path(rknn_path: str) -> str:
    """Derive the ``.weight`` path that belongs to an ``.rknn`` file.

    Mirrors ``make_weight_path`` in the C++ demo: replace a trailing
    ``.rknn`` with ``.weight``, otherwise append ``.weight``.
    """
    if rknn_path.endswith(".rknn"):
        return rknn_path[: -len(".rknn")] + ".weight"
    return rknn_path + ".weight"


class DepthAnythingBackend:
    """RKNN Depth Anything V3 backend (kind="depth").

    Owns the three RKNN stages and drives them as the C++ demo does:
    ``local`` once per view, ``global`` once over all views, ``head`` once.
    """

    def __init__(
        self,
        spec,
        *,
        npu: Optional[int] = None,
        device_id: Optional[Union[bytes, str]] = None,
    ):
        from .._librknn import RKNN3Lite

        stages = {
            "local": getattr(spec, "depth_local_path", ""),
            "global": getattr(spec, "depth_global_path", ""),
            "head": getattr(spec, "depth_head_path", ""),
        }
        missing = [f"{k}={v!r}" for k, v in stages.items() if not v]
        if missing:
            raise ValueError(
                f"{spec.name!r} is missing depth_local_path / depth_global_path / "
                f"depth_head_path; fill them in via .with_paths(...). Got: {missing}"
            )

        # Resolve the NPU selection **after** RKNN3Lite is imported so the
        # native driver init order stays predictable (same rule as the
        # other backends).
        resolved_device_id = resolve_device_id(RKNN3Lite, npu, device_id)
        self.spec = spec
        self._core_mask = spec.core_mask
        self._target = spec.target

        self.rknn_local = RKNN3Lite()
        self.rknn_global = RKNN3Lite()
        self.rknn_head = RKNN3Lite()
        #: Shared internal-memory arena (core_id → ``rknn3_tensor_mem*``).
        self._arena: Dict[int, int] = {}

        for attr, path in (
            ("rknn_local", stages["local"]),
            ("rknn_global", stages["global"]),
            ("rknn_head", stages["head"]),
        ):
            weight = depth_weight_path(path)
            if getattr(self, attr).load_rknn(path, weight) != 0:
                raise RuntimeError(f"{attr} load_rknn failed: {path} (+{weight})")

        # The three stages must **share** one internal-memory arena per NPU
        # core: DA3's stages together need ~1.6 GB of internal memory, more
        # than the runtime would reserve on its own. The C++ demo does the
        # same (``da3_init_internal_mem_share``) — it allocates the arena
        # by hand and hands it to all three contexts. ``user_mem_internal=1``
        # tells the runtime "the caller supplies the internal memory", which
        # is what makes ``model_init`` succeed in the first place.
        for attr in ("rknn_local", "rknn_global", "rknn_head"):
            getattr(self, attr).rknn_config.user_mem_internal = 1

        for attr in ("rknn_local", "rknn_global", "rknn_head"):
            if getattr(self, attr).init_runtime(
                target=self._target,
                core_mask=self._core_mask,
                device_id=resolved_device_id,
            ) != 0:
                raise RuntimeError(f"{attr} init_runtime failed")

        self._share_internal_memory()
        self._read_model_geometry()

    # ------------------------------------------------------------------ #
    # Shared internal memory
    # ------------------------------------------------------------------ #
    def _share_internal_memory(self) -> None:
        """Share one internal-memory arena per NPU core across the stages.

        Port of ``da3_init_internal_mem_share``: every stage reports how
        much internal memory it wants per core, the **maximum** over the
        stages wins, and one arena per core is then handed to all three
        contexts (each receiving the blocks for the cores it uses, in its
        own core order).

        Same pattern the ASR backend uses for its two stages (see
        ``wdnai/backends/asr.py::_setup_shared_internal_mem``).
        """
        stages = (self.rknn_local, self.rknn_global, self.rknn_head)
        max_size_by_core: Dict[int, int] = {}
        per_stage_cores: List[List[int]] = []
        for stage in stages:
            infos = stage.rknn3_query(_CMD_CORE_MEM_SIZE) or []
            cores: List[int] = []
            for info in infos:
                core_id = int(info.core_id)
                internal = int(info.internal_size)
                cores.append(core_id)
                if internal > max_size_by_core.get(core_id, 0):
                    max_size_by_core[core_id] = internal
            per_stage_cores.append(cores)

        if not max_size_by_core:
            return

        # The first stage owns the arena (the C++ demo uses ``models[0]``).
        owner = self.rknn_local
        arena: Dict[int, int] = {}
        self._arena = arena
        try:
            for core_id in sorted(max_size_by_core):
                size = max_size_by_core[core_id]
                if size <= 0:
                    continue
                arena[core_id] = owner.create_mem(size, core_id, _FLAG_CACHEABLE)
            if not arena:
                return
            for stage, cores in zip(stages, per_stage_cores):
                blocks = [arena[c] for c in cores if c in arena]
                if blocks:
                    stage.set_internal_mem(blocks)
        except Exception:
            self._release_internal_memory()
            raise

    def _release_internal_memory(self) -> None:
        """Free the shared arena (called from :meth:`close`)."""
        arena = getattr(self, "_arena", None)
        if not arena:
            return
        owner = getattr(self, "rknn_local", None)
        for mem_ptr in arena.values():
            if owner is not None:
                owner.destroy_user_mem(mem_ptr)
        self._arena = {}

    # ------------------------------------------------------------------ #
    # Geometry — read off the models, never assumed
    # ------------------------------------------------------------------ #
    def _read_model_geometry(self) -> None:
        """Read the shapes the export baked in.

        The input resolution (``H``/``W``), the number of views (``V``) and
        the token counts all come from the tensor attributes, exactly like
        the C++ demo reads them before running anything.
        """
        local_in = self.rknn_local.get_inputs_tensor_attr()[0]
        self.image_height = int(local_in.shape[1])
        self.image_width = int(local_in.shape[2])
        self.image_channels = int(local_in.shape[3])

        global_in = self.rknn_global.get_inputs_tensor_attr()[0]
        self.views = int(global_in.shape[0])
        self.local_tokens_per_view = int(global_in.shape[1])
        self.local_hidden = int(global_in.shape[3])

        self.local_outputs = 1
        self.global_features = len(self.rknn_global.get_outputs_tensor_attr())
        self.head_outputs = len(self.rknn_head.get_outputs_tensor_attr())

    # ------------------------------------------------------------------ #
    # Inference — one method per stage, mirroring the C++ demo's pipeline
    # ------------------------------------------------------------------ #
    def run_local_stage(self, rgb: np.ndarray) -> np.ndarray:
        """``local``: one preprocessed view → ``[L, D]`` tokens (fp32)."""
        batch = np.ascontiguousarray(rgb[None, ...], dtype=np.uint8)
        outs = self.rknn_local.inference(inputs=[batch])
        if not outs:
            raise RuntimeError("local stage returned no outputs")
        tokens = np.asarray(outs[0])
        if tokens.ndim == 3:
            tokens = tokens[0]
        # fp16 → fp32 through the demo's converter (see ``_fp16_to_fp32``).
        return _fp16_to_fp32(tokens)

    def run_global_stage(self, tokens: np.ndarray) -> List[np.ndarray]:
        """``global``: ``[V, L, D]`` tokens → four ``[V, F, G]`` features."""
        v = self.views
        l = self.local_tokens_per_view
        d = self.local_hidden
        buf = np.ascontiguousarray(tokens, dtype=np.float32).reshape(v, l, 1, d)
        outs = self.rknn_global.inference(inputs=[buf])
        if not outs:
            raise RuntimeError("global stage returned no outputs")
        feats: List[np.ndarray] = []
        for out in outs:
            arr = np.asarray(out)
            # The C++ passes the raw bytes of the ``[V,F,1,G]`` tensor to
            # the head stage, whose own input is ``[V,F,G]``. Squeezing the
            # singleton axis keeps the byte order identical.
            if arr.ndim == 4:
                arr = arr[:, :, 0, :]
            feats.append(np.ascontiguousarray(arr))
        return feats

    def run_head_stage(self, feats: Sequence[np.ndarray]) -> tuple:
        """``head``: features → ``(depth, confidence)`` both ``[V,H,W]``."""
        outs = self.rknn_head.inference(inputs=list(feats))
        if not outs or len(outs) < 2:
            raise RuntimeError("head stage returned no depth/confidence outputs")
        depth = _fp16_to_fp32(np.asarray(outs[0]))
        confidence = _fp16_to_fp32(np.asarray(outs[1]))
        if depth.ndim == 4:
            depth = depth[0]
        if confidence.ndim == 4:
            confidence = confidence[0]
        return depth, confidence

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def estimate(
        self,
        images: Sequence[np.ndarray],
        include_maps: bool = False,
    ) -> DepthResult:
        """Estimate depth for ``images`` (BGR ndarrays).

        The number of images must match the view count baked into the
        export — a 10-view model cannot take 3 images.
        """
        frames = list(images or [])
        if not frames:
            raise ValueError("depth estimation needs at least one image")
        if len(frames) != self.views:
            raise ValueError(
                f"this DA3 export was built for {self.views} views, got {len(frames)}"
            )

        tokens: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        for frame in frames:
            rgb = preprocess_image_rgb(frame, self.image_height, self.image_width)
            masks.append(build_depth_valid_mask(rgb))
            tokens.append(self.run_local_stage(rgb))

        feats = self.run_global_stage(np.stack(tokens, axis=0))
        depth, confidence = self.run_head_stage(feats)

        return DepthResult(
            depth=depth,
            confidence=confidence,
            views=self.views,
            height=self.image_height,
            width=self.image_width,
            masks=np.stack(masks, axis=0),
            include_maps=bool(include_maps),
        )

    def run_images(
        self,
        prompt: str,
        images: Sequence[np.ndarray],
        include_maps: bool = False,
        **_: Any,
    ) -> DepthResult:
        """Session-facing entry point — see :meth:`estimate`."""
        return self.estimate(images, include_maps=include_maps)

    def run_text(self, prompt: str, **_: Any):
        """Depth estimation has no text-only mode."""
        raise ValueError(
            "Depth Anything V3 needs images — call run_images() / respond() "
            "with an ImageAttachment."
        )

    def close(self) -> None:
        # Release order mirrors the C++ demo: the stages that *borrow* the
        # shared arena go first, then the arena itself, then its owner.
        for attr in ("rknn_head", "rknn_global"):
            rknn = getattr(self, attr, None)
            if rknn is None:
                continue
            try:
                rknn.release()
            except Exception:  # noqa: BLE001
                pass
            setattr(self, attr, None)
        self._release_internal_memory()
        rknn = getattr(self, "rknn_local", None)
        if rknn is not None:
            try:
                rknn.release()
            except Exception:  # noqa: BLE001
                pass
            self.rknn_local = None
