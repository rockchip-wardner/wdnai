"""可选的 RGA 硬件缩放 —— 对齐 rknn3_model_zoo 的 ``utils/image_utils.c``。

为什么需要它
============

Intern-Decision 的官方参考实现读图后用 ``convert_image()`` 缩放，而它
在 Rockchip 平台上走的是 **RGA（librga）硬件路径**：``improcess()`` 的
双线性插值和 OpenCV 的 ``cv2.INTER_LINEAR`` **不是同一个算法**，同一张
图缩出来的像素最大能差 130+（图像边缘），进而让视觉 embedding 与最终
决策概率都跟着偏。

实测（RK1828，640×640 → 384×384）::

    官方 RGA 输出  vs  cv2.resize(INTER_LINEAR)   maxdiff=136  mean=5.10
    官方 RGA 输出  vs  cv2.resize(INTER_AREA)     maxdiff=104  mean=3.94
    官方 RGA 输出  vs  本模块的 RGA 输出           maxdiff=0    mean=0.00

所以这里的 RGA 调用是**照抄官方**的：同样的 ``importbuffer_virtualaddr``
+ ``wrapbuffer_handle`` + ``improcess``，同样的整幅 ``srect`` / ``drect``，
同样的空 ``pat``（走 RGA 默认插值）。

可用性
======

* 需要板子上有 ``librga.so.2``（RK3588 / RK3576 / RK1828 都有）；
* 官方在源图或目标图宽度**不是 16 对齐**时会退到 CPU 路径 —— 本模块
  同样在这种���况下返回 ``None``，由调用方回退到 OpenCV；
* 任何一步失败都返回 ``None``（**不抛异常**），调用方必须准备好退路。
"""
from __future__ import annotations

import ctypes
import logging
import threading
from typing import Optional, Tuple

import numpy as np

LOG = logging.getLogger("wdnai._rga")

__all__ = ["rga_available", "rga_resize_rgb888", "rga_resize"]

#: librga 的格式枚举（``/usr/include/rga/rga.h``）——
#: ``RK_FORMAT_RGB_888 = 0x2 << 8``，即 [0:23] R:G:B 8:8:8 小端。
RK_FORMAT_RGB_888 = 0x2 << 8

#: 官方 ``convert_image()`` 只有在源/目标宽度都 16 对齐时才走 RGA。
_ALIGN = 16

#: 依次尝试的动态库路径。
_LIB_CANDIDATES = (
    "libaarch64-linux-gnu/librga.so.2",
    "librga.so.2",
    "librga.so",
)

_lock = threading.Lock()
_state: Optional[Tuple[object, bool]] = None


class _ImRect(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
    ]


class _ColorKeyRange(ctypes.Structure):
    _fields_ = [("max", ctypes.c_int), ("min", ctypes.c_int)]


class _ImNn(ctypes.Structure):
    _fields_ = [
        ("scale_r", ctypes.c_int), ("scale_g", ctypes.c_int),
        ("scale_b", ctypes.c_int), ("offset_r", ctypes.c_int),
        ("offset_g", ctypes.c_int), ("offset_b", ctypes.c_int),
    ]


class _RgaBuffer(ctypes.Structure):
    """``/usr/include/rga/im2d_type.h`` 的 ``rga_buffer_t``。"""

    _fields_ = [
        ("vir_addr", ctypes.c_void_p),
        ("phy_addr", ctypes.c_void_p),
        ("fd", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("wstride", ctypes.c_int),
        ("hstride", ctypes.c_int),
        ("format", ctypes.c_int),
        ("color_space_mode", ctypes.c_int),
        ("global_alpha", ctypes.c_int),
        ("rd_mode", ctypes.c_int),
        ("color", ctypes.c_int),
        ("colorkey_range", _ColorKeyRange),
        ("nn", _ImNn),
        ("rop_code", ctypes.c_int),
        ("handle", ctypes.c_int),
    ]


class _HandleParam(ctypes.Structure):
    """``im_handle_param_t``。"""

    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("format", ctypes.c_int),
    ]


def _load() -> Optional[object]:
    """Load and configure librga (cached). Returns ``None`` when absent."""
    global _state
    with _lock:
        if _state is not None:
            return _state[0]
        lib = None
        for candidate in _LIB_CANDIDATES:
            try:
                lib = ctypes.CDLL(candidate)
                break
            except OSError:
                continue
        if lib is None:
            LOG.info("wdnai._rga: librga 不可用，图像缩放走 OpenCV")
            _state = (None, False)
            return None
        lib.importbuffer_virtualaddr.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(_HandleParam),
        ]
        lib.importbuffer_virtualaddr.restype = ctypes.c_int
        lib.wrapbuffer_handle_t.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        lib.wrapbuffer_handle_t.restype = _RgaBuffer
        lib.improcess.argtypes = [
            _RgaBuffer, _RgaBuffer, _RgaBuffer,
            _ImRect, _ImRect, _ImRect, ctypes.c_int,
        ]
        lib.improcess.restype = ctypes.c_int
        lib.releasebuffer_handle.argtypes = [ctypes.c_int]
        lib.releasebuffer_handle.restype = ctypes.c_int
        _state = (lib, True)
        return lib


def rga_available() -> bool:
    """Whether the RGA hardware path can be used on this machine."""
    return _load() is not None


def rga_resize_rgb888(
    src: np.ndarray,
    dst_width: int,
    dst_height: int,
) -> Optional[np.ndarray]:
    """Resize an RGB888 image with RGA — byte-identical to the official demo.

    Parameters
    ----------
    src
        ``(H, W, 3)`` uint8, **RGB888** (R first), C-contiguous.
    dst_width, dst_height
        Target size.

    Returns
    -------
    np.ndarray or None
        ``(dst_height, dst_width, 3)`` uint8, or ``None`` when RGA is
        unavailable / the geometry is not 16-aligned / the call failed.
        ``None`` means "use your own fallback" — never an exception.
    """
    lib = _load()
    if lib is None:
        return None
    if src.dtype != np.uint8 or src.ndim != 3 or src.shape[2] != 3:
        return None
    src_h, src_w = int(src.shape[0]), int(src.shape[1])
    if src_w % _ALIGN or dst_width % _ALIGN:
        # 官方在这种几何下退到 CPU 路径 —— 这里交回调用方处理。
        return None
    if src_w == dst_width and src_h == dst_height:
        return np.ascontiguousarray(src)

    source = np.ascontiguousarray(src)
    target = np.zeros((int(dst_height), int(dst_width), 3), dtype=np.uint8)

    src_param = _HandleParam(
        width=src_w, height=src_h, format=RK_FORMAT_RGB_888,
    )
    dst_param = _HandleParam(
        width=int(dst_width), height=int(dst_height), format=RK_FORMAT_RGB_888,
    )
    src_handle = lib.importbuffer_virtualaddr(
        source.ctypes.data, ctypes.byref(src_param),
    )
    if src_handle <= 0:
        LOG.warning("wdnai._rga: importbuffer_virtualaddr(src) 失败: %s", src_handle)
        return None
    dst_handle = -1
    try:
        dst_handle = lib.importbuffer_virtualaddr(
            target.ctypes.data, ctypes.byref(dst_param),
        )
        if dst_handle <= 0:
            LOG.warning("wdnai._rga: importbuffer_virtualaddr(dst) 失败: %s", dst_handle)
            return None
        src_buf = lib.wrapbuffer_handle_t(
            src_handle, src_w, src_h, src_w, src_h, RK_FORMAT_RGB_888,
        )
        dst_buf = lib.wrapbuffer_handle_t(
            dst_handle, int(dst_width), int(dst_height),
            int(dst_width), int(dst_height), RK_FORMAT_RGB_888,
        )
        pattern = _RgaBuffer()
        src_rect = _ImRect(0, 0, src_w, src_h)
        dst_rect = _ImRect(0, 0, int(dst_width), int(dst_height))
        pattern_rect = _ImRect(0, 0, 0, 0)
        status = lib.improcess(
            src_buf, dst_buf, pattern, src_rect, dst_rect, pattern_rect, 0,
        )
        if status <= 0:
            LOG.warning("wdnai._rga: improcess 失败: %s", status)
            return None
    finally:
        lib.releasebuffer_handle(src_handle)
        if dst_handle > 0:
            lib.releasebuffer_handle(dst_handle)
    return target


def rga_resize(
    src: np.ndarray,
    dst_width: int,
    dst_height: int,
) -> Optional[np.ndarray]:
    """Alias of :func:`rga_resize_rgb888` (kept for readability at call sites)."""
    return rga_resize_rgb888(src, dst_width, dst_height)
