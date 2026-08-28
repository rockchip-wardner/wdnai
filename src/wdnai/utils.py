"""Common utilities: image preprocessing, JSON extraction, etc."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Union

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


def resize(img: np.ndarray, size: int) -> np.ndarray:
    """Resize an image to ``(size, size)``.

    Picks the interpolation by direction:
      * **downsample** (``size < max(h, w)``) → ``cv2.INTER_AREA``. This
        averages every source pixel covered by the destination cell,
        which is what OpenCV documents as the recommended mode for
        decimation and what preserves anti-aliased mean intensity for
        small high-frequency content (e.g. license-plate strokes when
        scaling 960x540 → 384x384). Plain bilinear
        (``cv2.INTER_LINEAR``) only sees the 2x2 corner pixels of each
        destination cell and silently drops the rest, which mangles
        text-like patterns.
      * **upsample** (``size > max(h, w)``) → ``cv2.INTER_CUBIC``. Cubic
        spline over a 4x4 neighbourhood is markedly sharper than the
        bilinear default at the cost of a few extra multiplies — well
        worth it for small-target detection on a 384 → 768 upscale.
      * **isometric** (equal) → ``cv2.INTER_LINEAR``. No resampling
        needed, but INTER_LINEAR keeps the function symmetric.

    Equivalence: ``cv2.resize(img, (s, s), interpolation=...)`` is the
    same code path OpenCV itself uses internally; we just pick the
    flag ahead of the call.
    """
    try:
        import cv2  # noqa: WPS433
    except ImportError as e:
        raise ImportError("Image resize requires opencv-python; please `pip install opencv-python`") from e
    h, w = img.shape[:2]
    if size < max(h, w):
        interp = cv2.INTER_AREA
    elif size > max(h, w):
        interp = cv2.INTER_CUBIC
    else:
        interp = cv2.INTER_LINEAR
    return cv2.resize(img, (size, size), interpolation=interp)


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
