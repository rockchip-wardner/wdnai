"""Common utilities: image / video preprocessing, JSON extraction, etc."""
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
    try:
        import cv2  # noqa: WPS433
    except ImportError as e:
        raise ImportError("Image resize requires opencv-python; please `pip install opencv-python`") from e
    return cv2.resize(img, (size, size))


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
# Qwen3-VL pruned-model preprocessing (test_video.py 2-frame version and
# test.py 1-frame version share the same algorithm)
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


def qwen3vl_prune_pair(img1_nchw: np.ndarray, img2_nchw: np.ndarray) -> np.ndarray:
    """Qwen3-VL pruned-model two-image (consecutive video frames) input."""
    img1 = _qwen3vl_normalize(img1_nchw)
    img2 = _qwen3vl_normalize(img2_nchw)
    patches = np.concatenate([img1, img2], axis=1)
    h = img1.shape[2]
    w = img1.shape[3]
    patches = patches.reshape(1, 2, 3, h // 2 // 16, 2, 16, w // 2 // 16, 2, 16)
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return patches.reshape(1 * h // 16 * w // 16, 3 * 2 * 16 * 16)


# --------------------------------------------------------------------------- #
# JSON extraction from model output (tolerates ```json``` fences and extra text)
# --------------------------------------------------------------------------- #
_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)
_JSON_BARE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def extract_json(text: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
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
            return None
    return None
