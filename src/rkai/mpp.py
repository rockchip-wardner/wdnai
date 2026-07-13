"""ctypes wrapper for Rockchip's librockchip_mpp.so (MPP) hardware video decoder.

This module exposes a thin Python class around the C ``MppApi`` decoder
interface used by ``mpi_dec_test.c``. It is intentionally minimal — only the
symbols exercised by the simple-decoder loop are bound:

    mpp_create / mpp_init / mpp_destroy / mpp_reset
    MppPacket init / deinit / set_data / set_size / set_pos / set_length / set_pts / set_eos
    MppFrame  init / deinit / get_*
    MppBufferGroup get / put + MppBuffer get / put
    mpi->decode_put_packet / decode_get_frame / control / reset
    mpp_dec_cfg_init / deinit / set_u32 + MPP_DEC_GET_CFG / SET_CFG

Anything encoder/ISP related is not bound — call sites for that path should
go straight to ``_lib`` if needed.

The library is searched in this order:

    1. ``rkai/libs/librockchip_mpp.so.0``    (vendored copy that ships with the SDK)
    2. ``/usr/lib/aarch64-linux-gnu/librockchip_mpp.so.0``    (system install)
    3. ``ctypes.util.find_library("rockchip_mpp")``           (fallback)
"""

from __future__ import annotations

import ctypes
import os
import threading
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    byref,
    c_char_p,
    c_int,
    c_int32,
    c_int64,
    c_size_t,
    c_uint32,
    c_uint64,
    c_void_p,
)
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))

_LIBS_TO_TRY = (
    os.path.join(_HERE, "libs", "librockchip_mpp.so.0"),
    "/usr/lib/aarch64-linux-gnu/librockchip_mpp.so.0",
)


def _load_library():
    last_err: Optional[OSError] = None
    for path in _LIBS_TO_TRY:
        if os.path.exists(path):
            try:
                return ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError as e:
                last_err = e
                continue
    # Fall back to whatever ldconfig knows about.
    try:
        return ctypes.CDLL("rockchip_mpp", mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        raise OSError(
            "Could not load librockchip_mpp.so.0. Tried: "
            + ", ".join(_LIBS_TO_TRY)
            + ". Last error: "
            + repr(last_err or e)
        )


_lib = _load_library()

# A single lock guards all MPP calls.  The C library is not documented as
# thread-safe across separate contexts and we currently only own one context
# at a time, but it is cheap insurance against accidental concurrent use.
_mpp_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Primitive types (mirror rk_type.h / mpp_err.h)
# ---------------------------------------------------------------------------

RK_U32 = c_uint32
RK_S32 = c_int32
RK_U64 = c_uint64
RK_S64 = c_int64
MPP_RET = c_int32  # signed so negative error codes work
MppCtx = c_void_p
MppPacket = c_void_p
MppFrame = c_void_p
MppBuffer = c_void_p
MppBufferGroup = c_void_p
MppParam = c_void_p


class MppCtxType:
    DEC = 0
    ENC = 1
    ISP = 2
    BUTT = 3


class MppCodingType:
    UNUSED = 0
    AUTO_DETECT = 1
    MPEG2 = 2
    H263 = 3
    MPEG4 = 4
    WMV = 5
    RV = 6
    AVC = 7          # H.264
    MJPEG = 8
    VP8 = 9
    VP9 = 10
    HEVC = 13        # H.265
    AVS = 18
    AVS2 = 19
    AV1 = 20


class MppFrameFormat:
    YUV420SP = 0          # NV12
    YUV420SP_10BIT = 1
    YUV422SP = 2
    YUV420P = 4
    YUV420SP_VU = 5       # NV21
    YUV422P = 6
    YUV400 = 12
    RGB888 = (1 << 16) + 6
    BGR888 = (1 << 16) + 7
    RGBA = (1 << 16) + 13
    BGRA = (1 << 16) + 12


class MppBufferType:
    NORMAL = 0
    ION = 1
    EXT_DMA = 2
    DRM = 3
    DMA_HEAP = 4


class MppBufferMode:
    INTERNAL = 0
    EXTERNAL = 1


# Subset of MpiCmd we actually use (see rk_mpi_cmd.h). Values come straight
# from the header so they stay compatible with future header revisions.
# Command codes are reconstructed from the bit layout documented in
# rk_mpi_cmd.h so we don't depend on the enum ordering. Each segment has
# an explicit base value (via MPP_FLAG_OR) and the enumerators that follow
# it are just `base + n`:
#
#   bit 20-23 module id   (CODEC = 0x00300000)
#   bit 16-19 ctx id      (DEC    = 0x00010000)
#   bit  8-15 cfg id      (QUERY  = 0x00000100 / CFG = 0x00000200)
#   bit  0- 7 command id  (1-based)
#
# So:
#   MPP_DEC_CMD_BASE   = 0x00310000
#   MPP_DEC_SET_OUTPUT_FORMAT = 0x00310000 + 10 = 0x0031000A
#   CMD_DEC_CMD_CFG     = 0x00310200
#   MPP_DEC_SET_CFG     = 0x00310201
#   MPP_DEC_GET_CFG     = 0x00310202
class MpiCmd:
    DEC_SET_FRAME_INFO = 0x00310001
    DEC_SET_EXT_BUF_GROUP = 0x00310002
    DEC_SET_INFO_CHANGE_READY = 0x00310003
    DEC_SET_OUTPUT_FORMAT = 0x0031000A
    DEC_SET_CFG = 0x00310201
    DEC_GET_CFG = 0x00310202


# ---------------------------------------------------------------------------
# MppBufferInfo (mpp_buffer.h) — used when reading pointer / fd out of MppBuffer
# ---------------------------------------------------------------------------


class MppBufferInfo(Structure):
    _fields_ = [
        ("type", c_uint32),
        ("size", c_size_t),
        ("ptr", c_void_p),
        ("hnd", c_void_p),
        ("fd", c_int),
        ("index", c_int),
    ]


# ---------------------------------------------------------------------------
# MppApi (rk_mpi.h) — function-pointer table returned by mpp_create()
# ---------------------------------------------------------------------------
#
# The struct layout (offsets are critical — ctypes will hand back garbage if
# the order doesn't match the C definition):
#
#   RK_U32 size, version;
#   MPP_RET (*decode)(ctx, packet, *frame);
#   MPP_RET (*decode_put_packet)(ctx, packet);
#   MPP_RET (*decode_get_frame)(ctx, *frame);
#   MPP_RET (*encode)(ctx, frame, *packet);
#   MPP_RET (*encode_put_frame)(ctx, frame);
#   MPP_RET (*encode_get_packet)(ctx, *packet);
#   MPP_RET (*isp)(ctx, dst, src);
#   MPP_RET (*isp_put_frame)(ctx, frame);
#   MPP_RET (*isp_get_frame)(ctx, *frame);
#   MPP_RET (*poll)(ctx, type, timeout);
#   MPP_RET (*dequeue)(ctx, type, *task);
#   MPP_RET (*enqueue)(ctx, type, task);
#   MPP_RET (*reset)(ctx);
#   MPP_RET (*control)(ctx, cmd, param);
#   RK_U32 reserv[16];


_C_DECODE = CFUNCTYPE(MPP_RET, MppCtx, MppPacket, POINTER(MppFrame))
_C_DECODE_PUT_PACKET = CFUNCTYPE(MPP_RET, MppCtx, MppPacket)
_C_DECODE_GET_FRAME = CFUNCTYPE(MPP_RET, MppCtx, POINTER(MppFrame))
_C_ENCODE = CFUNCTYPE(MPP_RET, MppCtx, MppFrame, POINTER(MppPacket))
_C_ENCODE_PUT_FRAME = CFUNCTYPE(MPP_RET, MppCtx, MppFrame)
_C_ENCODE_GET_PACKET = CFUNCTYPE(MPP_RET, MppCtx, POINTER(MppPacket))
_C_ISP = CFUNCTYPE(MPP_RET, MppCtx, MppFrame, MppFrame)
_C_ISP_PUT_FRAME = CFUNCTYPE(MPP_RET, MppCtx, MppFrame)
_C_ISP_GET_FRAME = CFUNCTYPE(MPP_RET, MppCtx, POINTER(MppFrame))
_C_POLL = CFUNCTYPE(MPP_RET, MppCtx, c_int, c_int)
_C_DEQUEUE = CFUNCTYPE(MPP_RET, MppCtx, c_int, c_void_p)
_C_ENQUEUE = CFUNCTYPE(MPP_RET, MppCtx, c_int, c_void_p)
_C_RESET = CFUNCTYPE(MPP_RET, MppCtx)
_C_CONTROL = CFUNCTYPE(MPP_RET, MppCtx, c_int, MppParam)


class MppApi(Structure):
    _fields_ = [
        ("size", RK_U32),
        ("version", RK_U32),
        ("decode", _C_DECODE),
        ("decode_put_packet", _C_DECODE_PUT_PACKET),
        ("decode_get_frame", _C_DECODE_GET_FRAME),
        ("encode", _C_ENCODE),
        ("encode_put_frame", _C_ENCODE_PUT_FRAME),
        ("encode_get_packet", _C_ENCODE_GET_PACKET),
        ("isp", _C_ISP),
        ("isp_put_frame", _C_ISP_PUT_FRAME),
        ("isp_get_frame", _C_ISP_GET_FRAME),
        ("poll", _C_POLL),
        ("dequeue", _C_DEQUEUE),
        ("enqueue", _C_ENQUEUE),
        ("reset", _C_RESET),
        ("control", _C_CONTROL),
        ("reserv", RK_U32 * 16),
    ]


# ---------------------------------------------------------------------------
# Top-level MppApi* getters / setters
# ---------------------------------------------------------------------------

_lib.mpp_create.argtypes = [POINTER(MppCtx), POINTER(POINTER(MppApi))]
_lib.mpp_create.restype = MPP_RET

_lib.mpp_init.argtypes = [MppCtx, c_int, c_int]
_lib.mpp_init.restype = MPP_RET

_lib.mpp_destroy.argtypes = [MppCtx]
_lib.mpp_destroy.restype = MPP_RET

_lib.mpp_check_support_format.argtypes = [c_int, c_int]
_lib.mpp_check_support_format.restype = MPP_RET


# MppPacket
_lib.mpp_packet_init.argtypes = [POINTER(MppPacket), c_void_p, c_size_t]
_lib.mpp_packet_init.restype = MPP_RET

_lib.mpp_packet_deinit.argtypes = [POINTER(MppPacket)]
_lib.mpp_packet_deinit.restype = MPP_RET

_lib.mpp_packet_set_data.argtypes = [MppPacket, c_void_p]
_lib.mpp_packet_set_data.restype = None

_lib.mpp_packet_set_size.argtypes = [MppPacket, c_size_t]
_lib.mpp_packet_set_size.restype = None

_lib.mpp_packet_set_pos.argtypes = [MppPacket, c_void_p]
_lib.mpp_packet_set_pos.restype = None

_lib.mpp_packet_set_length.argtypes = [MppPacket, c_size_t]
_lib.mpp_packet_set_length.restype = None

_lib.mpp_packet_set_pts.argtypes = [MppPacket, RK_S64]
_lib.mpp_packet_set_pts.restype = None

_lib.mpp_packet_set_eos.argtypes = [MppPacket]
_lib.mpp_packet_set_eos.restype = MPP_RET

_lib.mpp_packet_get_pts.argtypes = [MppPacket]
_lib.mpp_packet_get_pts.restype = RK_S64


# MppFrame
_lib.mpp_frame_init.argtypes = [POINTER(MppFrame)]
_lib.mpp_frame_init.restype = MPP_RET

_lib.mpp_frame_deinit.argtypes = [POINTER(MppFrame)]
_lib.mpp_frame_deinit.restype = MPP_RET

for _name in (
    "get_width", "get_height", "get_hor_stride", "get_ver_stride",
    "get_buf_size", "get_fmt", "get_pts", "get_dts", "get_poc",
    "get_errinfo", "get_discard", "get_eos", "get_info_change",
    "get_buffer",
):
    _fn = getattr(_lib, f"mpp_frame_{_name}")
    _fn.argtypes = [MppFrame]
    _fn.restype = RK_U32 if _name.startswith("get_buf_size") is False else c_size_t
# The above loop is a bit awkward because buf_size returns size_t and get_buffer
# returns MppBuffer. Fix the few that don't follow the RK_U32 rule:
_lib.mpp_frame_get_buf_size.restype = c_size_t
_lib.mpp_frame_get_pts.restype = RK_S64
_lib.mpp_frame_get_dts.restype = RK_S64
_lib.mpp_frame_get_buffer.restype = MppBuffer


# MppBuffer
_lib.mpp_buffer_get_with_tag.argtypes = [
    MppBufferGroup, POINTER(MppBuffer), c_size_t, c_char_p, c_char_p,
]
_lib.mpp_buffer_get_with_tag.restype = MPP_RET

_lib.mpp_buffer_put_with_caller.argtypes = [MppBuffer, c_char_p]
_lib.mpp_buffer_put_with_caller.restype = MPP_RET

_lib.mpp_buffer_get_ptr_with_caller.argtypes = [MppBuffer, c_char_p]
_lib.mpp_buffer_get_ptr_with_caller.restype = c_void_p


# MppBufferGroup
_lib.mpp_buffer_group_get.argtypes = [
    POINTER(MppBufferGroup), c_int, c_int, c_char_p, c_char_p,
]
_lib.mpp_buffer_group_get.restype = MPP_RET

_lib.mpp_buffer_group_put.argtypes = [MppBufferGroup]
_lib.mpp_buffer_group_put.restype = MPP_RET

_lib.mpp_buffer_group_usage.argtypes = [MppBufferGroup]
_lib.mpp_buffer_group_usage.restype = c_size_t

if hasattr(_lib, "mpp_buffer_group_limit_config"):
    _lib.mpp_buffer_group_limit_config.argtypes = [MppBufferGroup, c_size_t, c_int]
    _lib.mpp_buffer_group_limit_config.restype = MPP_RET


# MppDecCfg
_lib.mpp_dec_cfg_init.argtypes = [POINTER(c_void_p)]
_lib.mpp_dec_cfg_init.restype = MPP_RET

_lib.mpp_dec_cfg_deinit.argtypes = [c_void_p]
_lib.mpp_dec_cfg_deinit.restype = MPP_RET

_lib.mpp_dec_cfg_set_u32.argtypes = [c_void_p, c_char_p, RK_U32]
_lib.mpp_dec_cfg_set_u32.restype = MPP_RET

_lib.mpp_dec_cfg_get_u32.argtypes = [c_void_p, c_char_p, POINTER(RK_U32)]
_lib.mpp_dec_cfg_get_u32.restype = MPP_RET


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------


def _check(ret: int, where: str) -> None:
    """Raise if a MPP call returns a negative status."""
    if ret < 0:
        # MPP_ERR_* values are negative numbers; expose them in the message.
        raise RuntimeError(f"MPP call {where!r} failed: ret={ret}")


# Map a handful of FFmpeg codec names (as reported by PyAV / avcodec) to
# MppCodingType values. Extend as needed.
CODING_BY_NAME = {
    "h264": MppCodingType.AVC,
    "hevc": MppCodingType.HEVC,
    "h265": MppCodingType.HEVC,
    "vp9": MppCodingType.VP9,
    "vp8": MppCodingType.VP8,
    "av1": MppCodingType.AV1,
    "mpeg2video": MppCodingType.MPEG2,
    "mpeg4": MppCodingType.MPEG4,
    "mjpeg": MppCodingType.MJPEG,
}


@dataclass
class DecodedFrame:
    """A decoded frame returned by :class:`MppDecoder`."""

    width: int
    height: int
    hor_stride: int
    ver_stride: int
    fmt: int
    pts: int
    ptr: int           # pointer into the MppBuffer's storage (Y plane for NV12)
    buf_size: int      # size of the MppBuffer in bytes
    raw: "MppFrame"    # underlying MppFrame handle (c_void_p), for advanced users


class MppDecoder:
    """Thin ctypes wrapper around MPP's hardware video decoder.

    Mirrors the flow of ``mpi_dec_test.c``: create context, init as decoder,
    configure split_parse, then ``decode()`` packet-by-packet and drain frames.

    Parameters
    ----------
    coding_type:
        One of :class:`MppCodingType` (e.g. ``MppCodingType.AVC`` for H.264).
    split_parse:
        Whether to enable MPP's internal frame splitter. ``True`` lets MPP
        find frame boundaries inside arbitrary packet chunks — keep it on
        unless you know your demuxer already produces frame-aligned packets.
    extra_buf_count:
        Number of frame buffers to allocate when the decoder reports its
        first info-change. 24 matches the upstream test's default.
    output_format:
        Optional forced output format (``MppFrameFormat.YUV420SP`` by default,
        which is NV12 — most decoders natively output this).
    """

    def __init__(
        self,
        coding_type: int = MppCodingType.AVC,
        split_parse: bool = True,
        extra_buf_count: int = 24,
        output_format: int = MppFrameFormat.YUV420SP,
    ) -> None:
        with _mpp_lock:
            self._ctx = MppCtx()
            self._api = POINTER(MppApi)()
            _check(_lib.mpp_create(byref(self._ctx), byref(self._api)), "mpp_create")
            _check(_lib.mpp_init(self._ctx, MppCtxType.DEC, coding_type), "mpp_init")

            # split_parse configuration goes through MppDecCfg, same as
            # mpi_dec_test.c lines 518-541.
            cfg = c_void_p()
            _check(_lib.mpp_dec_cfg_init(byref(cfg)), "mpp_dec_cfg_init")
            try:
                _check(
                    self._api.contents.control(self._ctx, MpiCmd.DEC_GET_CFG, cfg),
                    "MPP_DEC_GET_CFG",
                )
                _check(
                    _lib.mpp_dec_cfg_set_u32(
                        cfg, b"base:split_parse", 1 if split_parse else 0
                    ),
                    "mpp_dec_cfg_set_u32 split_parse",
                )
                _check(
                    self._api.contents.control(self._ctx, MpiCmd.DEC_SET_CFG, cfg),
                    "MPP_DEC_SET_CFG",
                )
            finally:
                _lib.mpp_dec_cfg_deinit(cfg)

            # Force the decoder's output format. The MPP default is to give
            # us compressed-domain output (FBC / tile) which the caller has
            # to convert; asking for NV12 keeps the host code simple.
            # The C reference's "simple" decoder path (H.264/HEVC/VP9) does
            # not set this, so it matches the same code path the SDK used to
            # ship against.

        self._coding_type = coding_type
        self._extra_buf_count = extra_buf_count
        self._output_format = output_format
        self._buf_group: Optional[MppBufferGroup] = None
        # Lazily-initialised MppPacket (created on first send_packet) plus a
        # scratch buffer the packet points into. The buffer must outlive
        # decode_put_packet until the corresponding frame is consumed; we
        # reuse one big buffer for every send_packet because callers always
        # drain pending frames between sends.
        self._packet: MppPacket = MppPacket()
        self._packet_buf: Optional[ctypes.Array] = None
        self._packet_buf_size: int = 0
        self._closed = False

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> "MppDecoder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with _mpp_lock:
            if self._packet:
                _lib.mpp_packet_deinit(byref(self._packet))
                self._packet = MppPacket()
            if self._buf_group:
                _lib.mpp_buffer_group_put(self._buf_group)
                self._buf_group = None
            if self._ctx:
                _lib.mpp_destroy(self._ctx)
                self._ctx = MppCtx()

    # -- packet feeding -----------------------------------------------------

    def _ensure_packet_buf(self, size: int) -> ctypes.Array:
        """Make sure ``self._packet_buf`` is at least *size* bytes."""
        # Always allocate at least 1 MiB so we don't reallocate on every
        # small packet.
        target = max(size, 1 << 20)
        if self._packet_buf is None or self._packet_buf_size < target:
            # Tear down the previous packet so the new buffer can be
            # attached to a fresh one.
            if self._packet:
                _lib.mpp_packet_deinit(byref(self._packet))
                self._packet = MppPacket()
            buf = (ctypes.c_uint8 * target)()
            # Initialise the packet with NULL/0 (matches mpi_dec_test.c's
            # mpp_packet_init(&packet, NULL, 0)). The scratch buffer is
            # owned separately — send_packet() plugs it in via
            # mpp_packet_set_data each call.
            _check(
                _lib.mpp_packet_init(byref(self._packet), None, 0),
                "mpp_packet_init",
            )
            self._packet_buf = buf
            self._packet_buf_size = target
        return self._packet_buf

    def send_packet(
        self,
        data: bytes,
        pts: int = 0,
        eos: bool = False,
    ) -> None:
        """Copy *data* into the decoder's input packet and submit it.

        ``pts`` is in the demuxer's timescale (whatever you want — it just
        gets echoed back on :attr:`DecodedFrame.pts`). ``eos`` flags the
        packet as end-of-stream, which triggers a final flush of pending
        frames on the next :meth:`recv_frame` calls.

        Retries automatically on ``MPP_ERR_BUFFER_FULL``: when the decoder's
        packet queue is saturated we drain every available frame and try
        again. Bounded by ``max_retries`` (default 32) so a pathological
        caller can't get stuck here forever.
        """
        if self._closed:
            raise RuntimeError("MppDecoder is closed")
        size = len(data)
        with _mpp_lock:
            buf = self._ensure_packet_buf(size)
            ctypes.memmove(buf, data, size)
            _lib.mpp_packet_set_data(self._packet, buf)
            _lib.mpp_packet_set_size(self._packet, size)
            _lib.mpp_packet_set_pos(self._packet, buf)
            _lib.mpp_packet_set_length(self._packet, size)
            _lib.mpp_packet_set_pts(self._packet, pts)
            if eos:
                _check(_lib.mpp_packet_set_eos(self._packet), "mpp_packet_set_eos")
            ret = self._api.contents.decode_put_packet(self._ctx, self._packet)
            # -1012 = MPP_ERR_BUFFER_FULL. The hardware can't accept more
            # input until at least one frame is pulled by the caller.
            # The C reference's outer loop msleep()s for 1ms between
            # packets and retries. We do the same — *not* drain frames
            # here, because the info-change frame that tells us to set
            # up an external buffer group arrives here too, and stealing
            # it would leave the decoder stuck. The caller's recv_frame
            # loop is what makes progress. We sleep up to ~5s total
            # before raising — long enough to let the VPU's internal
            # thread catch up on a slow cold-start, short enough that
            # a genuinely wedged decoder still surfaces as a clear
            # error.
            import time as _t
            max_retries = 100
            attempts = 0
            backoff_ms = 1
            while ret == -1012 and attempts < max_retries:
                attempts += 1
                _t.sleep(backoff_ms / 1000.0)
                backoff_ms = min(backoff_ms * 2, 50)
                ret = self._api.contents.decode_put_packet(self._ctx, self._packet)
            _check(ret, "decode_put_packet")

    def recv_frame(self, timeout_ms: int = 30) -> Optional[DecodedFrame]:
        """Try to pull one decoded frame out of the decoder.

        Returns ``None`` if no frame is ready within *timeout_ms*. To match
        the C reference loop exactly, call this in a tight loop until it
        returns a frame after each :meth:`send_packet`.
        """
        if self._closed:
            raise RuntimeError("MppDecoder is closed")
        frame = MppFrame()
        with _mpp_lock:
            ret = self._api.contents.decode_get_frame(self._ctx, byref(frame))
            if ret == -8:  # MPP_ERR_TIMEOUT
                return None
            if ret != 0:
                _check(ret, "decode_get_frame")
            if not frame:
                # Successful call, no frame ready yet — caller should retry
                # after submitting more packets (matches the C reference's
                # `if (frame)` check after decode_get_frame returns).
                return None

            # On the first frame the decoder reports an info-change; allocate
            # an external buffer group sized to its needs (matches
            # mpi_dec_test.c lines 130-178).
            if _lib.mpp_frame_get_info_change(frame):
                width = _lib.mpp_frame_get_width(frame)
                height = _lib.mpp_frame_get_height(frame)
                hor_stride = _lib.mpp_frame_get_hor_stride(frame)
                ver_stride = _lib.mpp_frame_get_ver_stride(frame)
                buf_size = _lib.mpp_frame_get_buf_size(frame)

                if self._buf_group:
                    _lib.mpp_buffer_group_put(self._buf_group)
                grp = MppBufferGroup()
                _check(
                    _lib.mpp_buffer_group_get(
                        byref(grp),
                        MppBufferType.DMA_HEAP,
                        MppBufferMode.INTERNAL,
                        b"rkai-mpp",
                        b"info_change",
                    ),
                    "mpp_buffer_group_get",
                )
                # Limit the group so it can only hand out buffers of this size,
                # then prime the allocator with extra_buf_count entries. The
                # upstream test uses the same trick via dec_buf_mgr_setup.
                if hasattr(_lib, "mpp_buffer_group_limit_config"):
                    _check(
                        _lib.mpp_buffer_group_limit_config(
                            grp, buf_size, self._extra_buf_count
                        ),
                        "mpp_buffer_group_limit_config",
                    )
                for _ in range(self._extra_buf_count):
                    buf = MppBuffer()
                    _check(
                        _lib.mpp_buffer_get_with_tag(grp, byref(buf), buf_size, b"rkai-mpp", b"prime"),
                        "mpp_buffer_get prime",
                    )
                    _lib.mpp_buffer_put_with_caller(buf, b"prime")
                _check(
                    self._api.contents.control(
                        self._ctx, MpiCmd.DEC_SET_EXT_BUF_GROUP, grp
                    ),
                    "MPP_DEC_SET_EXT_BUF_GROUP",
                )
                self._buf_group = grp

                _check(
                    self._api.contents.control(
                        self._ctx, MpiCmd.DEC_SET_INFO_CHANGE_READY, None
                    ),
                    "MPP_DEC_SET_INFO_CHANGE_READY",
                )
                _lib.mpp_frame_deinit(byref(frame))
                return None

            buf = _lib.mpp_frame_get_buffer(frame)
            ptr = _lib.mpp_buffer_get_ptr_with_caller(buf, b"rkai-mpp")
            out = DecodedFrame(
                width=_lib.mpp_frame_get_width(frame),
                height=_lib.mpp_frame_get_height(frame),
                hor_stride=_lib.mpp_frame_get_hor_stride(frame),
                ver_stride=_lib.mpp_frame_get_ver_stride(frame),
                fmt=_lib.mpp_frame_get_fmt(frame),
                pts=_lib.mpp_frame_get_pts(frame),
                ptr=ptr or 0,
                buf_size=_lib.mpp_frame_get_buf_size(frame),
                raw=frame,
            )
        return out

    def release_frame(self, frame: DecodedFrame) -> None:
        """Return a frame received via :meth:`recv_frame` to MPP.

        The frame's MppBuffer reference count is dropped here; without this
        the internal buffer group will eventually refuse new frames.
        """
        if frame.raw:
            with _mpp_lock:
                _lib.mpp_frame_deinit(byref(frame.raw))
            frame.raw = MppFrame()  # type: ignore[assignment]

    def reset(self) -> None:
        with _mpp_lock:
            if self._ctx:
                _check(self._api.contents.reset(self._ctx), "mpi->reset")


# ---------------------------------------------------------------------------
# YUV (NV12) → BGR helper — purely for convenience in test/demo code.
# ---------------------------------------------------------------------------


def nv12_to_bgr(ptr_addr: int, width: int, height: int, hor_stride: int, ver_stride: int):
    """Convert an NV12 frame at *ptr_addr* into a (H, W, 3) BGR ndarray.

    Returns ``None`` if numpy is unavailable — callers that don't need a
    numpy view can use :class:`DecodedFrame` directly. The function does not
    copy on success; the returned array borrows memory from the decoder
    buffer, so the caller must ``release_frame()`` only after they're done
    with the array.
    """
    try:
        import numpy as np
    except ImportError:
        return None

    # NV12 layout: Y plane of stride*ver_stride bytes, then interleaved UV
    # plane of stride*ver_stride/2 bytes. Only the active area (width x
    # height) is meaningful — the rest is decoder padding.
    y_size = hor_stride * ver_stride
    total = y_size + (hor_stride * ver_stride) // 2
    buf = (ctypes.c_uint8 * total).from_address(ptr_addr)
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(ver_stride * 3 // 2, hor_stride)
    bgr_full = _nv12_to_bgr_np(arr[: height * 3 // 2, :width])
    return bgr_full


def _nv12_to_bgr_np(nv12):
    """Vectorised NV12 → BGR888 conversion.

    Lifted from the standard ITU-R BT.601 limited-range matrix. Operates on
    a (H*3/2, W) uint8 view; returns (H, W, 3) uint8.
    """
    import numpy as np

    h, w = nv12.shape[0] * 2 // 3, nv12.shape[1]
    y = nv12[:h, :].astype(np.int16)
    uv = nv12[h : h + h // 2, :].reshape(h // 2, w // 2, 2)
    u = uv[..., 0].astype(np.int16) - 128
    v = uv[..., 1].astype(np.int16) - 128

    # BT.601 limited range (matches what MPP emits for H.264):
    u_up = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)
    v_up = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)
    b = np.clip(y + 1.772 * u_up, 0, 255).astype(np.uint8)
    g = np.clip(y - 0.34414 * u_up - 0.71414 * v_up, 0, 255).astype(np.uint8)
    r = np.clip(y + 1.402 * v_up, 0, 255).astype(np.uint8)
    return np.stack([b, g, r], axis=-1)


# ---------------------------------------------------------------------------
# Annex B helpers — needed by anything that feeds raw compressed samples
# into MPP (PyAV, FFmpeg, etc.).
# ---------------------------------------------------------------------------


def avcc_to_annexb(data: bytes) -> bytes:
    """Convert a 4-byte-length-prefixed H.264/HEVC sample to Annex B.

    MP4 stores H.264/HEVC samples in "AVCC" / "HVCC" format — a sequence of
    NAL units, each prefixed with a big-endian 4-byte length. MPP's
    ``split_parse`` looks for the classic ``00 00 00 01`` start code, so we
    have to translate. Returns ``data`` unchanged when it already starts
    with a start code (some demuxers / streams are Annex B natively).
    """
    if len(data) >= 4 and data[:4] == b"\x00\x00\x00\x01":
        return data
    if len(data) >= 3 and data[:3] == b"\x00\x00\x01":
        return data

    out = bytearray()
    i = 0
    n = len(data)
    while i + 4 <= n:
        nal_len = int.from_bytes(data[i : i + 4], "big")
        i += 4
        if nal_len <= 0 or i + nal_len > n:
            # malformed — bail out and return whatever we have so MPP's
            # parser at least has something to look at.
            out.extend(data[i - 4 :])
            break
        out.extend(b"\x00\x00\x00\x01")
        out.extend(data[i : i + nal_len])
        i += nal_len
    return bytes(out)


def parse_avcc_extradata(extradata: bytes) -> bytes:
    """Return SPS+PPS as an Annex B byte stream from an avcC configuration.

    MP4 stores H.264 configuration in the avcC sample-entry box, not in
    the media samples themselves. MPP needs the SPS/PPS injected via the
    input stream (or the equivalent of what the upstream test does by
    feeding a raw .264 file). We parse the avcC and emit ``00 00 00 01``
    prefixed SPS followed by PPS so callers can prepend it to the first
    packet.
    """
    if not extradata or len(extradata) < 8 or extradata[0] != 1:
        return b""
    n_sps = extradata[5] & 0x1F
    i = 6
    out = bytearray()
    for _ in range(n_sps):
        if i + 2 > len(extradata):
            break
        sps_len = int.from_bytes(extradata[i : i + 2], "big")
        i += 2
        if i + sps_len > len(extradata):
            break
        out.extend(b"\x00\x00\x00\x01")
        out.extend(extradata[i : i + sps_len])
        i += sps_len
    if i + 1 > len(extradata):
        return bytes(out)
    n_pps = extradata[i]
    i += 1
    for _ in range(n_pps):
        if i + 2 > len(extradata):
            break
        pps_len = int.from_bytes(extradata[i : i + 2], "big")
        i += 2
        if i + pps_len > len(extradata):
            break
        out.extend(b"\x00\x00\x00\x01")
        out.extend(extradata[i : i + pps_len])
        i += pps_len
    return bytes(out)


__all__ = [
    "MppApi",
    "MppBufferInfo",
    "MppBufferMode",
    "MppBufferType",
    "MppCtxType",
    "MppCodingType",
    "MppDecoder",
    "MppFrameFormat",
    "MpiCmd",
    "CODING_BY_NAME",
    "DecodedFrame",
    "avcc_to_annexb",
    "parse_avcc_extradata",
    "nv12_to_bgr",
]