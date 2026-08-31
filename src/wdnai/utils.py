"""Common utilities: image preprocessing, JSON extraction, etc."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, NamedTuple, Optional, Union

import numpy as np


# --------------------------------------------------------------------------- #
# Image: BGR -> RGB -> resize -> normalization
# --------------------------------------------------------------------------- #
def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 3 and img_bgr.shape[-1] == 3:
        # Lazy import: avoid hard dependency on cv2 at SDK import time
        try:
            import cv2  # noqa: WPS433
        except ImportError as e:
            raise ImportError(
                "Image handling requires opencv-python (cv2); please `pip install opencv-python`"
            ) from e
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_bgr


def letterbox_resize(img: np.ndarray, size: int) -> np.ndarray:
    """Resize ``img`` to a ``(size, size)`` square **without** stretching.

    Behaviour:

    * The longer side of ``img`` is scaled so it equals ``size``.
    * The shorter side is scaled by the same factor and the result is
      centred inside a ``(size, size)`` canvas; the remaining border is
      filled with grey ``(114, 114, 114)`` (BGR; the ImageNet mean that
      Qwen-VL's official ``smart_resize`` and most letterbox pipelines
      use as the "empty patch" colour).
    * The original aspect ratio is preserved exactly — never stretched.

    This is the only image-resize entry point the SDK exposes. Calling
    code MUST NOT bypass it via plain ``cv2.resize`` because:

    1. VLM crop ROIs (e.g. license-plate bbox → ~250×60 strip) would
       otherwise get squashed into a square and the characters would
       distort badly enough to break OCR.
    2. The vision RKNNs (Qwen3-VL / Qwen3.5-VL / Embedding-2B) are all
       exported with square inputs but were trained against letterboxed
       data, so grey-padded squares are the canonical input shape.

    Interpolation follows the same direction-aware rule the legacy
    ``resize`` used:

    * downsample (``size < max(h, w)``) → ``cv2.INTER_AREA`` (averages
      every source pixel covered by each destination cell, preserves
      anti-aliased mean intensity for small high-frequency content
      like license-plate strokes).
    * upsample (``size > max(h, w)``) → ``cv2.INTER_CUBIC`` (sharper
      than bilinear for 4×-class upscales).
    * isometric (equal) → ``cv2.INTER_LINEAR``.

    Edge cases:

    * Square input at exact target size: returns a copy of the input
      padded to the canvas (the paste into a fresh canvas already
      forces a copy, so callers may mutate the returned array freely).
    * 1-channel or 4-channel input: the canvas is filled with the
      matching fill value broadcast along the channel axis
      (``(114, 114, 114)`` becomes ``114`` for grayscale, ``(114,114,114,114)``
      for BGRA).

    If you need to map the model's bbox output (which is in the
    letterboxed canvas's coordinate space) back to the **original**
    image's coordinate space, use :func:`letterbox_resize_with_meta` and
    :func:`unletterbox_bbox` instead.
    """
    canvas, _meta = letterbox_resize_with_meta(img, size)
    return canvas


class LetterboxMeta(NamedTuple):
    """Geometry produced by :func:`letterbox_resize_with_meta`.

    Feed this to :func:`unletterbox_bbox` / :func:`unletterbox_point`
    to map coordinates that the model returned in the **letterboxed
    canvas's** space back to the **original image's** space.

    Attributes
    ----------
    size : int
        Canvas edge length (e.g. 384).
    orig_h, orig_w : int
        Original image height / width.
    scale : float
        ``size / max(orig_h, orig_w)`` — uniform scale applied to both
        axes. Inverse of the scaling done inside ``letterbox_resize``.
    content_h, content_w : int
        Pixel size of the centred content area inside the canvas
        (``round(orig_h * scale)`` / ``round(orig_w * scale)``).
    pad_top, pad_left : int
        Grey-padding offset of the content area's top-left corner
        inside the canvas. Always ``(size - content_*) // 2``; the
        opposite-side padding is ``size - content_* - pad_*`` and may
        differ by 1 pixel for odd ``size - content_*`` differences.
    """

    size: int
    orig_h: int
    orig_w: int
    scale: float
    content_h: int
    content_w: int
    pad_top: int
    pad_left: int


def letterbox_resize_with_meta(
    img: np.ndarray, size: int,
) -> tuple:
    """Same as :func:`letterbox_resize` but also returns the geometry.

    The returned :class:`LetterboxMeta` lets callers invert the
    letterbox — i.e. map coordinates that the VLM emitted in the
    letterboxed canvas's space back to the original image's space.

    Typical usage (the canonical "send to VLM → crop original frame"
    pattern)::

        canvas, meta = letterbox_resize_with_meta(img, 384)
        # ask VLM what to crop
        result = await session.respond(attachment=Attachment(canvas), ...)
        # bbox is in 0-1000 coords of `canvas`
        x1, y1, x2, y2 = unletterbox_bbox(result.bbox, meta)
        roi = img[int(y1):int(y2), int(x1):int(x2)]
    """
    try:
        import cv2  # noqa: WPS433
    except ImportError as e:
        raise ImportError(
            "Image letterbox requires opencv-python; please `pip install opencv-python`"
        ) from e

    h, w = img.shape[:2]
    s = max(h, w)
    if s == size and h == w:
        canvas = np.full((size, size) + img.shape[2:], 114, dtype=img.dtype)
        canvas[...] = img
        meta = LetterboxMeta(
            size=size, orig_h=h, orig_w=w, scale=1.0,
            content_h=size, content_w=size, pad_top=0, pad_left=0,
        )
        return canvas, meta

    scale = size / s
    nh, nw = int(round(h * scale)), int(round(w * scale))
    nh = max(1, nh)
    nw = max(1, nw)
    if size < s:
        interp = cv2.INTER_AREA
    elif size > s:
        interp = cv2.INTER_CUBIC
    else:
        interp = cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)

    fill = 114
    if img.ndim == 3:
        if img.shape[2] == 1:
            canvas = np.full((size, size, 1), fill, dtype=img.dtype)
        else:
            canvas = np.full((size, size, img.shape[2]), fill, dtype=img.dtype)
    else:
        canvas = np.full((size, size), fill, dtype=img.dtype)
    pad_top = (size - nh) // 2
    pad_left = (size - nw) // 2
    canvas[pad_top:pad_top + nh, pad_left:pad_left + nw] = resized
    meta = LetterboxMeta(
        size=size, orig_h=h, orig_w=w, scale=scale,
        content_h=nh, content_w=nw, pad_top=pad_top, pad_left=pad_left,
    )
    return canvas, meta


def unletterbox_bbox(
    bbox_1000: Union[List[float], tuple],
    meta: LetterboxMeta,
    *,
    clamp: bool = True,
) -> List[float]:
    """Map a 0-1000 bbox in the letterboxed canvas back to the original image.

    The VLM sees the letterboxed canvas and emits bbox coords in 0-1000
    of **that canvas** (which contains grey padding around the original
    content). Calling code that wants to crop the **original** image
    (no grey padding, original aspect ratio) must run the bbox through
    this function.

    ``bbox_1000`` is ``[x1, y1, x2, y2]`` (any order is fine — the
    function inverts top-left and bottom-right independently and
    returns them in input order). Values outside ``[0, 1000]`` are
    accepted; if ``clamp=True`` (default), the result is clipped to
    ``[0, orig_w]`` / ``[0, orig_h]`` so the cropped ROI never reads
    out-of-bounds pixels.

    Returns ``[x1_orig, y1_orig, x2_orig, y2_orig]`` in **pixels** of
    the original image (floats; cast to ``int`` when slicing).
    """
    if len(bbox_1000) != 4:
        raise ValueError(
            f"bbox must have 4 elements, got {len(bbox_1000)}: {bbox_1000!r}"
        )
    raw = [float(v) for v in bbox_1000]
    # 0-1000 -> canvas pixels
    canvas_xs = [v / 1000.0 * meta.size for v in (raw[0], raw[2])]
    canvas_ys = [v / 1000.0 * meta.size for v in (raw[1], raw[3])]
    # subtract padding -> content pixels (may be negative if bbox
    # strayed into the grey border; ``clamp`` handles that below)
    content_xs = [cx - meta.pad_left for cx in canvas_xs]
    content_ys = [cy - meta.pad_top for cy in canvas_ys]
    # content pixels -> original pixels
    orig_xs = [cx / meta.scale for cx in content_xs]
    orig_ys = [cy / meta.scale for cy in content_ys]
    if clamp:
        orig_xs = [max(0.0, min(float(meta.orig_w), x)) for x in orig_xs]
        orig_ys = [max(0.0, min(float(meta.orig_h), y)) for y in orig_ys]
    return [orig_xs[0], orig_ys[0], orig_xs[1], orig_ys[1]]


def unletterbox_point(
    xy_1000: Union[List[float], tuple],
    meta: LetterboxMeta,
    *,
    clamp: bool = True,
) -> List[float]:
    """Map a single 0-1000 point ``(x, y)`` from letterboxed canvas
    back to original image coordinates (pixels).

    Same arithmetic as :func:`unletterbox_bbox` but for a single point.
    Returns ``[x_orig, y_orig]``.
    """
    if len(xy_1000) != 2:
        raise ValueError(
            f"point must have 2 elements, got {len(xy_1000)}: {xy_1000!r}"
        )
    bbox = unletterbox_bbox(
        [xy_1000[0], xy_1000[1], xy_1000[0], xy_1000[1]], meta, clamp=clamp,
    )
    return [bbox[0], bbox[1]]


# --------------------------------------------------------------------------- #
# Qwen2.5-VL pruned-model preprocessing (matches examples/Qwen2.5VL/test.py L175-186)
# --------------------------------------------------------------------------- #
def qwen25vl_prune(img_nchw: np.ndarray) -> np.ndarray:
    """Qwen2.5-VL 3B pruned-model visual input preprocessing. ``img_nchw`` shape (1, 3, H, W)."""
    img = np.float32(img_nchw)
    # BGR-channel statistics (note: H and W must be multiples of 14)
    img[0, 2, ...] = (img[0, 2, ...] - 104.09) / 70.3
    img[0, 1, ...] = (img[0, 1, ...] - 116.74) / 66.6
    img[0, 0, ...] = (img[0, 0, ...] - 122.70) / 68.5
    patches = np.concatenate([img, img], axis=1)
    h = img.shape[2]
    w = img.shape[3]
    patches = patches.reshape(1, 2, 3, h // 2 // 14, 2, 14, w // 2 // 14, 2, 14)
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return patches.reshape(1 * h // 14 * w // 14, 3 * 2 * 14 * 14)


# --------------------------------------------------------------------------- #
# Qwen3-VL pruned-model preprocessing (single-image path; matches examples/Qwen3VL/test.py)
# --------------------------------------------------------------------------- #
def _qwen3vl_normalize(img_nchw: np.ndarray) -> np.ndarray:
    img = np.float32(img_nchw)
    img[0, 2, ...] = (img[0, 2, ...] - 127.5) / 127.5
    img[0, 1, ...] = (img[0, 1, ...] - 127.5) / 127.5
    img[0, 0, ...] = (img[0, 0, ...] - 127.5) / 127.5
    return img


def qwen3vl_prune_single(img_nchw: np.ndarray) -> np.ndarray:
    """Qwen3-VL pruned-model single-image visual input. ``img_nchw`` shape (1, 3, H, W)."""
    img = _qwen3vl_normalize(img_nchw)
    patches = np.concatenate([img, img], axis=1)
    h = img.shape[2]
    w = img.shape[3]
    patches = patches.reshape(1, 2, 3, h // 2 // 16, 2, 16, w // 2 // 16, 2, 16)
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return patches.reshape(1 * h // 16 * w // 16, 3 * 2 * 16 * 16)


# --------------------------------------------------------------------------- #
# JSON extraction from model output (tolerates ```json``` fences and extra text)
# --------------------------------------------------------------------------- #
_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)
_JSON_BARE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


# Matches ``<think>...`` blocks emitted by Qwen3 / Qwen3.5 / DeepSeek-R1
# style reasoning models. We strip these BEFORE any JSON extraction so
# that:
#
# 1. The regex below doesn't accidentally pick up a ``{}`` inside the
#    reasoning as the model's "real" answer.
# 2. ``_safe_validate``'s ``raw_text_for_fallback`` path doesn't stuff
#    the entire reasoning block into the schema's ``text`` field (this
#    was the user-visible regression in v0.x — the SDK returned
#    ``Event(text='<think>\\n\\n</think>\\n\\n```json\\n...')``
#    instead of the actual content).
# 3. ``try_parse_tool_call`` doesn't try to dispatch JSON the model
#    meant as part of its internal monologue.
#
# Non-greedy match on the body + ``re.DOTALL`` so we span newlines, but
# we REQUIRE the closing ``</think>`` (or end-of-string) so the engine
# doesn't match an empty ``<think>`` and leave the rest of the block
# untouched (which is what ``<think>.*?`` alone does — non-greedy ``.*?``
# matches 0 chars and succeeds immediately). We loop ``re.sub`` in
# ``strip_thinking_blocks`` so two consecutive think blocks (some
# Qwen variants emit two rounds) are both stripped.
_THINK_BLOCK = re.compile(
    r"<think>" r"[\s\S]*?" r"(?:</think>|$)",
    re.DOTALL,
)

# Some Qwen tool-calling variants wrap tool calls in XML tags rather
# than emitting bare JSON. We strip just the tags (opening + closing)
# so the JSON payload inside is preserved and falls through to the
# regular JSON extraction path. Stripping the *whole* block (tags
# + content) was wrong because the JSON we want to extract lives
# inside the tags — we'd then get an empty string and ``extract_json``
# would return None. The fix is to keep the content and remove only
# the literal ``<tool_call>`` / ``</tool_call>`` markers.
# Pattern is plain ASCII; built via concatenation so zero-width /
# curly-quote characters from markdown rendering never sneak in.
_TOOL_CALL_OPEN = re.compile(
    r"<tool_call" r"\b" r"[^>]*" r">",
)
_TOOL_CALL_CLOSE = re.compile(
    r"</" r"tool_call" r"\s*" r">",
)


def strip_thinking_blocks(text: str) -> str:
    """Remove ``<think>...`` (and ``<tool_call>...</tool_call>`` XML)
    blocks from a model's raw output.

    Multiple model families emit visible reasoning before the final
    answer; without stripping, the SDK's JSON parsers pick up
    fragments of the reasoning as the answer, and the schema-fallback
    path stuffs the whole reasoning into the schema's ``text``
    field — which silently corrupts the caller's data without any
    parse failure.

    The stripped text is whitespace-collapsed at the edges but
    otherwise preserved verbatim. Returns the original string
    unchanged when no markers are found.
    """
    if not text:
        return text
    prev = None
    out = text
    # Loop so we strip every think block (some Qwen variants emit
    # two of them when the model "thinks twice").
    while prev != out:
        prev = out
        out = _THINK_BLOCK.sub("", out)
    # Strip XML tool_call wrappers — but ONLY the tags, not the JSON
    # content inside (which we want ``extract_json`` to see). The
    # initial version stripped the whole ``<tool_call>...</tool_call>``
    # block including the JSON inside, which left an empty string
    # and broke the downstream parser.
    out = _TOOL_CALL_OPEN.sub("", out)
    out = _TOOL_CALL_CLOSE.sub("", out)
    return out.strip()


def extract_json(text: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
    # Strip <think> blocks before any regex matching. See the comment
    # above ``_THINK_BLOCK`` for why this happens unconditionally
    # (cheap, idempotent, and defends against every current and
    # future thinking-mode model — Qwen3, Qwen3.5, DeepSeek-R1, ...).
    text = strip_thinking_blocks(text or "")
    if not text:
        return None
    m = _JSON_FENCE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Fallback: slice from the first { to the last }
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    m = _JSON_BARE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Last-ditch fallback: try to parse the text as a Python literal.
    # Qwen3.5-4B (and other small Qwen variants) occasionally emit
    # Python-style dict repr (``{'text': '...'}`` with single quotes)
    # inside a ```json``` fence instead of strict JSON. The strict-JSON
    # paths above already failed by this point, so we strip the
    # ````json``` fences (keep content) and try ``ast.literal_eval``,
    # which handles dicts / lists / strings / numbers / booleans / None
    # safely — it does NOT evaluate arbitrary expressions, so there's
    # no security risk. We only try this on text that *looks* like a
    # Python literal (starts with ``{`` / ``[`` / ``'`` / ``"``) so
    # plain prose doesn't trigger a needless parse attempt.
    import ast
    inner_text = _JSON_FENCE.sub(r"\1", text) if _JSON_FENCE.search(text) else text
    stripped = inner_text.strip()
    if stripped and stripped[0] in "{['\"":
        try:
            obj = ast.literal_eval(stripped)
            if isinstance(obj, (dict, list)):
                return obj
        except (ValueError, SyntaxError):
            pass
    return None
