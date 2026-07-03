"""VideoCapture — a thin OpenCV-style wrapper around the MPP hardware decoder.

Designed to mirror ``cv2.VideoCapture`` so existing OpenCV pipelines can be
retargeted at the Rockchip VPU with a one-line import swap. The heavy
lifting (PyAV demux, AVCC→Annex B, SPS/PPS injection, NV12→BGR) lives in
this module; the underlying ctypes binding to ``librockchip_mpp.so`` is
``rksdk.mpp``.

Typical usage:

    import cv2
    import rksdk

    cap = rksdk.VideoCapture("bus.mp4")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("video", frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    cap.release()

The class is also a context manager, so a ``with`` block guarantees the
decoder is torn down on exit.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Optional, Tuple

import av
import cv2
import numpy as np

from .mpp import (
    CODING_BY_NAME,
    DecodedFrame,
    MppCodingType,
    MppDecoder,
    avcc_to_annexb,
    parse_avcc_extradata,
)


# Buffer-full retry inside send_packet is short (~5 s) so the wrapper
# surfaces a stuck VPU quickly. The reader in :meth:`VideoCapture.read`
# picks up the slack by draining hard on every BUFFER_FULL before
# retrying — the same pattern ``mpi_dec_test.c`` uses with its
# msleep(1)-bounded outer loop.
_BUFFER_FULL_TOKEN = "ret=-1012"
_DRAIN_RETRIES = 400
_DRAIN_TIMEOUT_MS = 5
_FRAME_BGR_FORMATS = {0, 4, 5}  # NV12, I420, NV21 — the ones cv2 handles directly


class VideoCapture:
    """OpenCV-style interface to the MPP hardware decoder.

    Iterates an MP4 (or any container PyAV can read) and returns BGR
    frames as ``numpy.ndarray`` shaped ``(H, W, 3)``.

    Parameters
    ----------
    path:
        Path to the input media file or a network URL (``rtsp://``,
        ``http://``, ``tcp://`` …). PyAV's container layer handles the
        protocol; the codec must be one of the formats MPP knows how to
        decode (H.264, HEVC, VP9, AV1, …).
    options:
        Optional dict of demuxer options forwarded to ffmpeg. For RTSP
        you typically want ``{"rtsp_transport": "tcp"}`` so the video
        stream isn't lost to UDP jitter. Ignored if the protocol doesn't
        accept it.
    timeout:
        Seconds to wait for the container to open *and* for each
        subsequent read. Accepts a float (single value) or a
        ``(open_timeout, read_timeout)`` tuple. Useful for network streams
        that might be unreachable; ``None`` lets PyAV block indefinitely.
    """

    def __init__(
        self,
        path: str | Path,
        options: Optional[dict] = None,
        timeout: Optional[float] = 10.0,
    ) -> None:
        # str(path) lets URLs through unmodified; Path.is_file() is only
        # used to give a friendlier error for the local-file case.
        path_str = str(path)
        local = Path(path)
        if "://" not in path_str and not local.is_file():
            raise FileNotFoundError(f"VideoCapture: no such file {local}")

        kwargs = {}
        if options:
            kwargs["options"] = options
        if timeout is not None:
            kwargs["timeout"] = timeout

        self._container = av.open(path_str, **kwargs)
        if not self._container.streams.video:
            self._container.close()
            raise RuntimeError(f"{path_str}: no video stream found")
        self._stream = self._container.streams.video[0]
        codec_name = self._stream.codec_context.name
        coding = CODING_BY_NAME.get(codec_name)
        if coding is None:
            self._container.close()
            raise RuntimeError(
                f"{path_str}: codec {codec_name!r} is not supported by MPP. "
                f"Known codecs: {sorted(CODING_BY_NAME)}"
            )

        # Properties surfaced to the caller (OpenCV-compatible names).
        self.width: int = self._stream.codec_context.width or 0
        self.height: int = self._stream.codec_context.height or 0
        self.fps: float = float(self._stream.average_rate or 0)
        self.frame_pos: int = 0   # frames handed back to the caller so far

        # Live streams (RTSP, HTTP, etc.) never EOF. Detect by looking at
        # the protocol: anything with "://" is treated as live so we keep
        # the demuxer open until the caller calls release().
        self._is_live: bool = "://" in path_str
        self._path_str: str = path_str

        self._decoder = MppDecoder(coding_type=coding, split_parse=False)
        # H.264/HEVC: prepend SPS/PPS from the avcC/hvcC configuration
        # box to the first sample so the decoder knows how to parse it.
        self._sps_pps = (
            parse_avcc_extradata(self._stream.codec_context.extradata or b"")
            if codec_name in ("h264", "hevc", "h265")
            else b""
        )
        self._sent_extradata = not self._sps_pps
        self._codec_name = codec_name
        self._opened = True
        self._eof = False

    # -- OpenCV-style API ---------------------------------------------------

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Pull the next decoded frame from the source.

        For local files this returns ``(True, bgr)`` for every frame and
        ``(False, None)`` once the file is exhausted. For live streams
        (RTSP, HTTP) it loops forever; the caller has to call
        :meth:`release` to stop.
        """
        if not self._opened or self._eof:
            return False, None

        # Phase 1: feed more packets from PyAV and surface any frame
        # produced along the way. We yield the *first* decoded frame we
        # see — the caller drives pacing by calling read() again.
        for raw_packet in self._container.demux(self._stream):
            data = bytes(raw_packet)
            if not data:
                continue
            if not self._send(data, pts=int(raw_packet.pts or raw_packet.dts or 0)):
                return False, None

            frame = self._drain_one()
            if frame is not None:
                return True, frame

        # Phase 2: file EOF. Signal end-of-stream so MPP flushes any
        # frames it queued, then drain whatever comes back. For live
        # streams we skip this — there is no EOF.
        if self._is_live:
            # No frame in the queue? Wait briefly for the next packet
            # to arrive over the network, then loop back to demux().
            import time as _t
            _t.sleep(0.005)
            return self.read()

        self._decoder.send_packet(b"", pts=0, eos=True)
        idle = 0
        while idle < 10:
            frame = self._recv_bgr(timeout_ms=50)
            if frame is not None:
                return True, frame
            idle += 1

        self._eof = True
        return False, None

    def release(self) -> None:
        if not self._opened:
            return
        self._opened = False
        try:
            if self._decoder is not None:
                self._decoder.close()
        finally:
            self._decoder = None  # type: ignore[assignment]
            if self._container is not None:
                self._container.close()
            self._container = None  # type: ignore[assignment]

    # -- context manager / destructor --------------------------------------

    def __enter__(self) -> "VideoCapture":
        return self

    def __exit__(self, *_exc) -> None:
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass

    # -- internals ----------------------------------------------------------

    def _send(self, data: bytes, pts: int) -> bool:
        """Submit one packet to MPP, draining on BUFFER_FULL.

        Returns ``False`` if the stream is exhausted (read() should stop).
        """
        if self._codec_name in ("h264", "hevc", "h265"):
            annexb = avcc_to_annexb(data)
            if self._sps_pps and not self._sent_extradata:
                annexb = self._sps_pps + annexb
                self._sent_extradata = True
        else:
            annexb = data

        while True:
            try:
                self._decoder.send_packet(annexb, pts=pts, eos=False)
                return True
            except RuntimeError as e:
                if _BUFFER_FULL_TOKEN not in str(e):
                    raise
                # VPU's packet queue is saturated; pull a frame and try
                # again. If no frame is ready yet (cold start), give the
                # hardware a brief moment to catch up.
                frame = self._drain_one(timeout_ms=_DRAIN_TIMEOUT_MS)
                if frame is not None:
                    # We have to come back to the caller with this frame
                    # before retrying the send — but we're inside _send.
                    # Stash it via _prefetch so read() picks it up next.
                    self._prefetched = frame
                    return True
                import time as _t
                _t.sleep(0.005)

    def _drain_one(
        self, timeout_ms: int = _DRAIN_TIMEOUT_MS
    ) -> Optional[np.ndarray]:
        """Pull at most one frame out of the decoder, returning BGR or None."""
        if hasattr(self, "_prefetched") and self._prefetched is not None:
            out, self._prefetched = self._prefetched, None
            return out
        return self._recv_bgr(timeout_ms=timeout_ms)

    def _recv_bgr(self, timeout_ms: int) -> Optional[np.ndarray]:
        """Run recv_frame once, convert NV12/NV21/I420 → BGR, release."""
        frame = self._decoder.recv_frame(timeout_ms=timeout_ms)
        if frame is None:
            return None
        try:
            bgr = _nv12_to_bgr(frame)
        finally:
            self._decoder.release_frame(frame)
        if bgr is not None:
            self.frame_pos += 1
        return bgr


def _nv12_to_bgr(frame: DecodedFrame) -> Optional[np.ndarray]:
    """View an MPP frame's YUV buffer as a BGR ndarray.

    Supports NV12 (most common), NV21 and I420 — anything else is
    dropped (returning ``None``). The output array is a fresh copy so
    the caller is free to hold it past the next :meth:`release_frame`.
    """
    if frame.ptr == 0 or frame.buf_size == 0:
        return None
    if frame.fmt not in _FRAME_BGR_FORMATS:
        return None

    w, h = frame.width, frame.height
    hor_stride, ver_stride = frame.hor_stride, frame.ver_stride
    total = hor_stride * ver_stride * 3 // 2
    if total > frame.buf_size:
        total = frame.buf_size

    raw = np.ctypeslib.as_array(
        (ctypes.c_uint8 * total).from_address(frame.ptr)
    ).reshape(ver_stride * 3 // 2, hor_stride)

    # cv2's NV12/NV21/I420 decoders expect a (h*1.5, w) buffer; the
    # rest of the MppBuffer is decoder padding.
    yuv = raw[: h * 3 // 2, :w].copy()
    if frame.fmt == 0:
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    if frame.fmt == 5:
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
    if frame.fmt == 4:
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
    return None


__all__ = ["VideoCapture"]