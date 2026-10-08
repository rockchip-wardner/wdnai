"""ctypes shim for librknnsmi.so — wdnai's RKNN-SMI status access.

The librknnsmi.so shared object (``/usr/lib/librknnsmi.so``) ships on
RK targets as the C ABI behind the ``rknn-smi`` CLI. wdnai (and therefore
wdnwebui) needs a subset of that ABI to expose NPU telemetry (device list,
memory, temperature, power, health, version, reset …) without shelling
out to ``rknn-smi info`` — which is a 3-row console table that's awkward
to parse and which the upstream comment "rknn-smi 不扛高频轮询" in
``app/system/static/system.js`` warns against for the >1Hz polling the
system-monitor page does.

This module is the ctypes shim and a thin high-level wrapper. Mirrors
:mod:`wdnai._librknn` style: all ctypes Structures / prototypes live
private, only :class:`RknnSmi` is exported, library is loaded lazily on
the first instantiation so ``import wdnai`` doesn't open the SO and
race with the compute workers.

The structure layouts mirror the C header in
``/home/rockchip/rknn-smi/include/rknnsmi_common.h`` (Rockchip 2025).
"""
from __future__ import annotations

import ctypes
import os
import threading
from ctypes import (
    POINTER,
    Structure,
    c_bool,
    c_char,
    c_char_p,
    c_int,
    c_int32,
    c_size_t,
    c_uint8,
    c_uint16,
    c_uint32,
    c_uint64,
    c_void_p,
)
from enum import IntEnum
from typing import List, Optional

# =============================================================================
# 1. Library load
# =============================================================================
_LIB: Optional[ctypes.CDLL] = None
_LIB_LOCK = threading.Lock()
_PROTOTYPES_BOUND = False


def _candidate_paths() -> List[str]:
    env = os.environ.get("WDNAI_LIBRKNNSMI")
    if env:
        return [env]
    return [
        "/usr/lib/librknnsmi.so",
    ]


def load_library() -> ctypes.CDLL:
    """Load ``librknnsmi.so`` once. Idempotent.

    The actual ``CDLL`` open runs on the first :class:`RknnSmi` instance,
    not on ``import wdnai`` — opening the SO can talk to the NPU driver
    (PCIe rescan, USB ttyACM probe) and would race with
    :mod:`wdnai._librknn`'s ``RKNN3Lite`` on shared state.
    """
    global _LIB, _PROTOTYPES_BOUND
    if _LIB is not None:
        return _LIB
    with _LIB_LOCK:
        if _LIB is not None:
            return _LIB
        last_err: Optional[BaseException] = None
        for p in _candidate_paths():
            try:
                _LIB = ctypes.CDLL(p)
                break
            except OSError as e:
                last_err = e
        if _LIB is None:
            raise RuntimeError(
                "librknnsmi.so not found in standard paths; set "
                "WDNAI_LIBRKNNSMI to override. Last error: "
                f"{last_err}"
            )
        if not _PROTOTYPES_BOUND:
            _bind_prototypes(_LIB)
            _PROTOTYPES_BOUND = True
        return _LIB


# =============================================================================
# 2. ABI constants
# =============================================================================
MAX_FREQ_CNT = 10
MAX_DEVICE_CNT = 16
MAX_CHIP_CNT = 16
MAX_CHIP_INFO_LEN = 32
MAX_PRODUCT_INFO_LEN = 32


# =============================================================================
# 3. Return codes (mirrors RKNNSMI_* in rknnsmi_common.h)
# =============================================================================
class RknnSmiError(IntEnum):
    SUCCESS = 0
    GENERIC = -1
    PARAMETER = -2
    ACCESS_DENIED = -3
    STATE = -4
    NOT_SUPPORTED = -5
    INVALID_DEVICE = -6
    TIMEOUT = -7
    MEM = -8
    SINGLE_INSTANCE = -9


class CommMode(IntEnum):
    """Mirrors ``NSMI_COMM_MODE``."""
    PCIE = 0
    USB = 1
    ETH = 2


class VersionType(IntEnum):
    """Mirrors ``NSMI_VERSION_TYPE``."""
    NSMI = 0
    PCIE_DRV = 1
    RC_CHIPS_CONNECT = 2
    EP_CHIPS_CONNECT = 3
    FIRMWARE = 4
    LIBRKNN3_API = 5


class ChipHealthStatus(IntEnum):
    """Mirrors ``NSMI_CHIP_HEALTH_STATUS``."""
    OK = 0
    WARNING = 1
    ALARM = 2
    CRITICAL = 3
    UNKNOWN = 4


class ResetReason(IntEnum):
    """Mirrors ``NSMI_RESET_REASON``."""
    NPOR = 0
    SOFT = 1
    TSADC = 2
    SUBSOC_WDT = 3
    VRV0_WDT = 4
    VRV1_WDT = 5
    CRC_CHECK = 6


class ThermalMode(IntEnum):
    """Mirrors ``NSMI_THERMAL_MODE``."""
    SPEED_RATIO = 0
    DEVICE_ENABLE = 1
    CUSTOM_THERMAL_ENABLE = 2
    DEFAULT_THERMAL_TRIPS = 3
    CUSTOM_THERMAL_TRIPS = 4


class WorkMode(IntEnum):
    """Mirrors ``NSMI_WORK_MODE``."""
    EFFICIENT = 0
    NORMAL = 1
    PERFORMANCE = 2
    USER = 3
    UNDEFINED = 4


class PrefillMode(IntEnum):
    """Mirrors ``NSMI_PREFILL_MODE``."""
    EFFICIENT = 0
    PERFORMANCE = 1
    UNDEFINED = 2


# =============================================================================
# 4. Internal ctypes structures
# =============================================================================
class _NsmiDeviceList(Structure):
    """``struct nsmi_device_list`` — describes connected devices.

    The struct is a packed (device, mode) pair per slot — matches the
    C source exactly. The outer ``count`` is uint16_t but the inner
    anonymous-struct member is padded by 2 bytes (4-byte alignment
    for the next NSMI_COMM_MODE int), so each entry occupies 8 bytes
    on aarch64.
    """

    class _DeviceEntry(Structure):
        _fields_ = [
            ("mode", c_int),        # NSMI_COMM_MODE
            ("global_id", c_uint16),
            ("internal_id", c_uint16),
        ]

    _fields_ = [
        ("count", c_uint16),
        ("_pad0", c_uint16 * 3),    # struct padding to 8-byte align
        ("dev", _DeviceEntry * MAX_DEVICE_CNT),
    ]


class _NsmiChipInfo(Structure):
    """``struct nsmi_chip_info``."""

    _fields_ = [
        ("chip_id", c_uint16),
        ("_pad0", c_uint16 * 3),
        ("chip_name", c_char * MAX_CHIP_INFO_LEN),
        ("chip_ver", c_char * MAX_CHIP_INFO_LEN),
        ("cpuid", c_uint8 * MAX_CHIP_INFO_LEN),
        ("cpuid_len", c_uint16),
        ("cpu_cnt", c_uint16),
        ("npu_cnt", c_uint16),
    ]


class _NsmiDeviceInfo(Structure):
    """``struct nsmi_device_info``."""

    _fields_ = [
        ("device_id", c_uint16),
        ("_pad0", c_uint16 * 3),
        ("product_name", c_char * MAX_PRODUCT_INFO_LEN),
        ("manufacturer", c_char * MAX_PRODUCT_INFO_LEN),
        ("serial_num", c_char * MAX_PRODUCT_INFO_LEN),
        ("reserve", c_char * 64),
    ]


class _NsmiPcieInfo(Structure):
    """``struct nsmi_pcie_info``."""

    _fields_ = [
        ("dev_is_alive", c_bool),
        ("_pad0", c_uint8 * 3),
        ("vendor_id", c_uint32),
        ("device_id", c_uint32),
        ("sub_vendor_id", c_uint32),
        ("sub_device_id", c_uint32),
        ("domain", c_uint32),
        ("bdf_bus_id", c_uint32),
        ("bdf_device_id", c_uint32),
        ("bdf_func_id", c_uint32),
        ("reserve", c_char * 64),
    ]


class _NsmiUsbInfo(Structure):
    """``struct nsmi_usb_info``."""

    _fields_ = [
        ("dev_is_alive", c_bool),
        ("_pad0", c_uint8 * 7),
    ]


class _NsmiCpuInfo(Structure):
    """``struct nsmi_cpu_info``."""

    _fields_ = [
        ("cur_freq", c_uint16),
        ("support_freq_list", c_uint16 * MAX_FREQ_CNT),
        ("support_freq_cnt", c_uint16),
        ("core_cnt", c_uint16),
        ("util_rate", c_uint16),
        ("_pad0", c_uint16),
    ]


class _NsmiNpuInfo(Structure):
    """``struct nsmi_npu_info``."""

    _fields_ = [
        ("cur_freq", c_uint16),
        ("support_freq_list", c_uint16 * MAX_FREQ_CNT),
        ("support_freq_cnt", c_uint16),
        ("core_cnt", c_uint16),
        ("util_rate", c_uint16),
        ("_pad0", c_uint16),
    ]


class _NsmiMemoryInfo(Structure):
    """``struct nsmi_memory_info``."""

    _fields_ = [
        ("total_size", c_uint64),
        ("free_size", c_uint64),
        ("hugepage_size", c_uint32),
        ("hugepage_total_cnt", c_uint32),
        ("hugepage_free_cnt", c_uint32),
        ("util_rate", c_uint16),
        ("cur_freq", c_uint16),
        ("support_freq_list", c_uint16 * MAX_FREQ_CNT),
        ("support_freq_cnt", c_uint16),
        ("_pad0", c_uint16),
    ]


class _NsmiMemheapStatus(Structure):
    """``struct nsmi_memheap_status``."""

    _fields_ = [
        ("chip_id", c_uint16),
        ("ext_mem_heap_inited", c_uint16),
        ("_pad0", c_uint32),
        ("srv_total_size", c_uint64),
        ("srv_free_size", c_uint64),
        ("vrv_total_size", c_uint64),
        ("vrv_free_size", c_uint64),
    ]


class _PcieErrEventCount(Structure):
    """``struct pcie_err_event_count`` from cc_pcie_type.h.

    The header isn't shipped with the SO we link against, but the
    struct is small and only used by ``rknnsmi_get_pcie_err_event_count``
    which wdnai doesn't call. We declare a minimal 4-uint64 placeholder
    sized to cover the most common layouts.
    """

    _fields_ = [
        ("replay_number_rollover", c_uint32),
        ("replay_number", c_uint32),
        ("bad_tlp", c_uint32),
        ("bad_dllp", c_uint32),
        ("recovery", c_uint32),
        ("replay_rollover", c_uint32),
        ("_pad0", c_uint32 * 2),
    ]


# =============================================================================
# 5. C function prototypes (bound against the loaded SO)
# =============================================================================
def _bind_prototypes(L: ctypes.CDLL) -> None:
    # lifecycle
    L.rknnsmi_init.argtypes = []
    L.rknnsmi_init.restype = c_int

    L.rknnsmi_deinit.argtypes = []
    L.rknnsmi_deinit.restype = c_int

    # device list / info
    L.rknnsmi_get_device_list.argtypes = [POINTER(_NsmiDeviceList)]
    L.rknnsmi_get_device_list.restype = c_int

    L.rknnsmi_get_device_info.argtypes = [c_uint16, POINTER(_NsmiDeviceInfo)]
    L.rknnsmi_get_device_info.restype = c_int

    # chip
    L.rknnsmi_get_chip_list.argtypes = [
        c_uint16,
        POINTER(c_uint16),
        POINTER(c_uint16),
        c_size_t,
    ]
    L.rknnsmi_get_chip_list.restype = c_int

    L.rknnsmi_get_chip_info.argtypes = [c_uint16, c_uint16, POINTER(_NsmiChipInfo)]
    L.rknnsmi_get_chip_info.restype = c_int

    # pcie / usb
    L.rknnsmi_get_pcie_info.argtypes = [c_uint16, c_uint16, POINTER(_NsmiPcieInfo)]
    L.rknnsmi_get_pcie_info.restype = c_int

    L.rknnsmi_get_usb_info.argtypes = [c_uint16, c_uint16, POINTER(_NsmiUsbInfo)]
    L.rknnsmi_get_usb_info.restype = c_int

    # cpu / npu
    L.rknnsmi_get_cpu_info.argtypes = [c_uint16, c_uint16, POINTER(_NsmiCpuInfo)]
    L.rknnsmi_get_cpu_info.restype = c_int

    L.rknnsmi_get_npu_info.argtypes = [c_uint16, c_uint16, POINTER(_NsmiNpuInfo)]
    L.rknnsmi_get_npu_info.restype = c_int

    # device-level power / voltage / current / temperature
    L.rknnsmi_get_device_power.argtypes = [c_uint16, POINTER(c_uint32)]
    L.rknnsmi_get_device_power.restype = c_int

    L.rknnsmi_get_device_voltage.argtypes = [c_uint16, POINTER(c_uint32)]
    L.rknnsmi_get_device_voltage.restype = c_int

    L.rknnsmi_get_device_current.argtypes = [c_uint16, POINTER(c_uint32)]
    L.rknnsmi_get_device_current.restype = c_int

    L.rknnsmi_get_device_temperature.argtypes = [c_uint16, POINTER(c_int32)]
    L.rknnsmi_get_device_temperature.restype = c_int

    # chip-level
    L.rknnsmi_get_chip_temperature.argtypes = [c_uint16, c_uint16, POINTER(c_int32)]
    L.rknnsmi_get_chip_temperature.restype = c_int

    L.rknnsmi_get_chip_health.argtypes = [c_uint16, c_uint16, POINTER(c_int)]
    L.rknnsmi_get_chip_health.restype = c_int

    L.rknnsmi_get_system_time.argtypes = [c_uint16, c_uint16, POINTER(c_uint32)]
    L.rknnsmi_get_system_time.restype = c_int

    L.rknnsmi_get_reset_reason.argtypes = [c_uint16, c_uint16, POINTER(c_int)]
    L.rknnsmi_get_reset_reason.restype = c_int

    L.rknnsmi_get_work_mode.argtypes = [c_uint16, c_uint16, POINTER(c_int)]
    L.rknnsmi_get_work_mode.restype = c_int

    L.rknnsmi_get_prefill_mode.argtypes = [c_uint16, c_uint16, POINTER(c_int)]
    L.rknnsmi_get_prefill_mode.restype = c_int

    # memory
    L.rknnsmi_get_memory_info.argtypes = [c_uint16, c_uint16, POINTER(_NsmiMemoryInfo)]
    L.rknnsmi_get_memory_info.restype = c_int

    L.rknnsmi_get_memheap_status.argtypes = [c_uint16, c_uint16, POINTER(_NsmiMemheapStatus)]
    L.rknnsmi_get_memheap_status.restype = c_int

    # version
    L.rknnsmi_get_version.argtypes = [c_char_p, c_size_t, c_int, c_uint16]
    L.rknnsmi_get_version.restype = c_int

    # reset
    L.rknnsmi_reset_device.argtypes = [c_uint16]
    L.rknnsmi_reset_device.restype = c_int

    L.rknnsmi_hw_reset_device.argtypes = [c_uint16]
    L.rknnsmi_hw_reset_device.restype = c_int


# =============================================================================
# 6. Public high-level wrapper
# =============================================================================
def _raise_if_error(ret: int, op: str) -> None:
    if ret != RknnSmiError.SUCCESS:
        try:
            code = RknnSmiError(ret)
        except ValueError:
            code = RknnSmiError.GENERIC
        raise RknnSmiCallError(op, ret, code)


def _u16(x) -> int:
    return int(x) & 0xFFFF


class RknnSmiCallError(RuntimeError):
    """Raised when an rknnsmi_* call returns a non-success code."""

    def __init__(self, op: str, code: int, enum: "RknnSmiError") -> None:
        self.op = op
        self.code = int(code)
        self.enum = enum
        super().__init__(f"rknnsmi {op} failed: {enum.name} ({code})")


class RknnSmi:
    """High-level wrapper around ``librknnsmi.so``.

    The class is the entry point wdnwebui uses. It calls
    :func:`load_library` on construction (which ``CDLL`` opens
    librknnsmi.so the first time), then runs the singleton
    ``rknnsmi_init()`` once per process; :meth:`close` runs the
    matching ``rknnsmi_deinit()`` (also process-singleton).

    Most methods accept a ``device_id`` (0-based) and (where the
    underlying C function takes one) a ``chip_id`` (0-based; usually
    ``0`` on RK1828 / RK3588). All return values are plain Python
    primitives / dicts — no ctypes leakage.
    """

    _init_lock = threading.Lock()
    _init_done = False
    _init_failed: Optional[BaseException] = None

    def __init__(self) -> None:
        self._lib = load_library()
        with RknnSmi._init_lock:
            if not RknnSmi._init_done and RknnSmi._init_failed is None:
                ret = self._lib.rknnsmi_init()
                if ret != RknnSmiError.SUCCESS:
                    RknnSmi._init_failed = RknnSmiCallError(
                        "init", ret,
                        RknnSmiError(ret) if ret in {e.value for e in RknnSmiError}
                        else RknnSmiError.GENERIC,
                    )
                else:
                    RknnSmi._init_done = True
        if RknnSmi._init_failed is not None:
            raise RknnSmi._init_failed

    # -------- lifecycle --------
    def close(self) -> None:
        """Run ``rknnsmi_deinit()`` if this process owns the init.

        Safe to call from atexit; ignores double-close. Most callers
        don't need to bother — librknnsmi.so is process-singleton, and
        the next process will re-init cleanly.
        """
        with RknnSmi._init_lock:
            if RknnSmi._init_done:
                try:
                    self._lib.rknnsmi_deinit()
                except Exception:
                    pass
                RknnSmi._init_done = False

    # -------- device list --------
    def get_device_list(self) -> List[dict]:
        """Return one dict per connected device.

        Each dict has the form::

            {"index": <int>, "mode": "pcie"|"usb"|"eth",
             "global_id": <int>, "internal_id": <int>}

        ``index`` is the 0-based slot the SO used to expose this device
        to other ``rknnsmi_*`` calls — i.e. the value to pass as
        ``device_id`` to :meth:`get_device_info`, :meth:`get_device_power`,
        etc.
        """
        buf = _NsmiDeviceList()
        ret = self._lib.rknnsmi_get_device_list(ctypes.byref(buf))
        _raise_if_error(ret, "get_device_list")
        out: List[dict] = []
        n = int(buf.count)
        if n > MAX_DEVICE_CNT:
            n = MAX_DEVICE_CNT
        for i in range(n):
            entry = buf.dev[i]
            mode_val = int(entry.mode)
            try:
                mode_name = CommMode(mode_val).name.lower()
            except ValueError:
                mode_name = f"unknown({mode_val})"
            out.append({
                "index": i,
                "mode": mode_name,
                "global_id": int(entry.global_id),
                "internal_id": int(entry.internal_id),
            })
        return out

    def count(self) -> int:
        """Convenience: number of connected devices."""
        return len(self.get_device_list())

    # -------- version --------
    def get_version(
        self,
        type: "VersionType | int" = VersionType.NSMI,
        device_id: int = 0,
        buffer_size: int = 256,
    ) -> str:
        """Return a version string for the requested component.

        ``type`` selects between the host-side NSMI library
        (:attr:`VersionType.NSMI`), PCIe driver, chips_connect, NPU
        firmware, etc. ``device_id`` is only meaningful for
        per-device version types like ``FIRMWARE`` / ``LIBRKNN3_API``.
        """
        buf = ctypes.create_string_buffer(buffer_size)
        ret = self._lib.rknnsmi_get_version(
            buf, ctypes.c_size_t(buffer_size), int(type), _u16(device_id)
        )
        _raise_if_error(ret, "get_version")
        return buf.value.decode("utf-8", errors="replace").rstrip("\x00").strip()

    # -------- per-device info --------
    def get_device_info(self, device_id: int = 0) -> dict:
        buf = _NsmiDeviceInfo()
        ret = self._lib.rknnsmi_get_device_info(
            _u16(device_id), ctypes.byref(buf)
        )
        _raise_if_error(ret, "get_device_info")
        return {
            "device_id": int(buf.device_id),
            "product_name": buf.product_name.decode("utf-8", errors="replace").rstrip("\x00").strip(),
            "manufacturer": buf.manufacturer.decode("utf-8", errors="replace").rstrip("\x00").strip(),
            "serial_num": buf.serial_num.decode("utf-8", errors="replace").rstrip("\x00").strip(),
        }

    def get_pcie_info(self, device_id: int = 0, chip_id: int = 0) -> dict:
        buf = _NsmiPcieInfo()
        ret = self._lib.rknnsmi_get_pcie_info(
            _u16(device_id), _u16(chip_id), ctypes.byref(buf)
        )
        _raise_if_error(ret, "get_pcie_info")
        bdf = (
            f"{buf.domain:04x}:{buf.bdf_bus_id:02x}:"
            f"{buf.bdf_device_id:02x}.{buf.bdf_func_id}"
        )
        return {
            "dev_is_alive": bool(buf.dev_is_alive),
            "vendor_id": f"0x{buf.vendor_id:04x}",
            "device_id": f"0x{buf.device_id:04x}",
            "sub_vendor_id": f"0x{buf.sub_vendor_id:04x}",
            "sub_device_id": f"0x{buf.sub_device_id:04x}",
            "domain": int(buf.domain),
            "bdf_bus_id": int(buf.bdf_bus_id),
            "bdf_device_id": int(buf.bdf_device_id),
            "bdf_func_id": int(buf.bdf_func_id),
            "bus_id": bdf,
        }

    # -------- chip / cpu / npu info --------
    def get_chip_list(self, device_id: int = 0) -> List[int]:
        chip_num = c_uint16(0)
        chip_list = (c_uint16 * MAX_CHIP_CNT)()
        ret = self._lib.rknnsmi_get_chip_list(
            _u16(device_id),
            ctypes.byref(chip_num),
            chip_list,
            ctypes.c_size_t(MAX_CHIP_CNT),
        )
        _raise_if_error(ret, "get_chip_list")
        n = int(chip_num.value)
        if n > MAX_CHIP_CNT:
            n = MAX_CHIP_CNT
        return [int(chip_list[i]) for i in range(n)]

    def get_chip_info(self, device_id: int = 0, chip_id: int = 0) -> dict:
        buf = _NsmiChipInfo()
        ret = self._lib.rknnsmi_get_chip_info(
            _u16(device_id), _u16(chip_id), ctypes.byref(buf)
        )
        _raise_if_error(ret, "get_chip_info")
        return {
            "chip_id": int(buf.chip_id),
            "chip_name": buf.chip_name.decode("utf-8", errors="replace").rstrip("\x00").strip(),
            "chip_ver": buf.chip_ver.decode("utf-8", errors="replace").rstrip("\x00").strip(),
            "cpuid_len": int(buf.cpuid_len),
            "cpuid": bytes(buf.cpuid[: int(buf.cpuid_len)]).hex()
            if buf.cpuid_len else "",
            "cpu_cnt": int(buf.cpu_cnt),
            "npu_cnt": int(buf.npu_cnt),
        }

    def get_cpu_info(self, device_id: int = 0, chip_id: int = 0) -> dict:
        buf = _NsmiCpuInfo()
        ret = self._lib.rknnsmi_get_cpu_info(
            _u16(device_id), _u16(chip_id), ctypes.byref(buf)
        )
        _raise_if_error(ret, "get_cpu_info")
        n = int(buf.support_freq_cnt)
        if n > MAX_FREQ_CNT:
            n = MAX_FREQ_CNT
        return {
            "cur_freq_mhz": int(buf.cur_freq),
            "support_freq_list_mhz": [int(buf.support_freq_list[i]) for i in range(n)],
            "support_freq_cnt": n,
            "core_cnt": int(buf.core_cnt),
            "util_rate_pct": int(buf.util_rate),
        }

    def get_npu_info(self, device_id: int = 0, chip_id: int = 0) -> dict:
        buf = _NsmiNpuInfo()
        ret = self._lib.rknnsmi_get_npu_info(
            _u16(device_id), _u16(chip_id), ctypes.byref(buf)
        )
        _raise_if_error(ret, "get_npu_info")
        n = int(buf.support_freq_cnt)
        if n > MAX_FREQ_CNT:
            n = MAX_FREQ_CNT
        return {
            "cur_freq_mhz": int(buf.cur_freq),
            "support_freq_list_mhz": [int(buf.support_freq_list[i]) for i in range(n)],
            "support_freq_cnt": n,
            "core_cnt": int(buf.core_cnt),
            "util_rate_pct": int(buf.util_rate),
        }

    # -------- power / voltage / current / temperature --------
    def get_device_power(self, device_id: int = 0) -> int:
        """Return device power in mW (0 if not supported)."""
        out = c_uint32(0)
        ret = self._lib.rknnsmi_get_device_power(_u16(device_id), ctypes.byref(out))
        _raise_if_error(ret, "get_device_power")
        return int(out.value)

    def get_device_voltage(self, device_id: int = 0) -> int:
        """Return device voltage in mV (0 if not supported)."""
        out = c_uint32(0)
        ret = self._lib.rknnsmi_get_device_voltage(_u16(device_id), ctypes.byref(out))
        _raise_if_error(ret, "get_device_voltage")
        return int(out.value)

    def get_device_current(self, device_id: int = 0) -> int:
        """Return device current in mA (0 if not supported)."""
        out = c_uint32(0)
        ret = self._lib.rknnsmi_get_device_current(_u16(device_id), ctypes.byref(out))
        _raise_if_error(ret, "get_device_current")
        return int(out.value)

    def get_device_temperature(self, device_id: int = 0) -> float:
        """Return device temperature in °C (milli-°C resolution).

        The SO returns the value as a millidegree int (×1000); we
        return it as float °C so it matches what ``rknn-smi info``
        prints in the table.
        """
        out = c_int32(0)
        ret = self._lib.rknnsmi_get_device_temperature(
            _u16(device_id), ctypes.byref(out)
        )
        _raise_if_error(ret, "get_device_temperature")
        return int(out.value) / 1000.0

    def get_chip_temperature(self, device_id: int = 0, chip_id: int = 0) -> float:
        out = c_int32(0)
        ret = self._lib.rknnsmi_get_chip_temperature(
            _u16(device_id), _u16(chip_id), ctypes.byref(out)
        )
        _raise_if_error(ret, "get_chip_temperature")
        return int(out.value) / 1000.0

    def get_chip_health(
        self, device_id: int = 0, chip_id: int = 0
    ) -> "ChipHealthStatus":
        out = c_int(0)
        ret = self._lib.rknnsmi_get_chip_health(
            _u16(device_id), _u16(chip_id), ctypes.byref(out)
        )
        _raise_if_error(ret, "get_chip_health")
        try:
            return ChipHealthStatus(int(out.value))
        except ValueError:
            return ChipHealthStatus.UNKNOWN

    def get_reset_reason(
        self, device_id: int = 0, chip_id: int = 0
    ) -> "ResetReason":
        out = c_int(0)
        ret = self._lib.rknnsmi_get_reset_reason(
            _u16(device_id), _u16(chip_id), ctypes.byref(out)
        )
        _raise_if_error(ret, "get_reset_reason")
        try:
            return ResetReason(int(out.value))
        except ValueError:
            return ResetReason.SOFT

    # -------- memory --------
    def get_memory_info(self, device_id: int = 0, chip_id: int = 0) -> dict:
        """Return device / chip memory telemetry.

        All byte counts are raw bytes; convert to MB / GB at the
        presentation layer (wdnwebui does the conversion).
        """
        buf = _NsmiMemoryInfo()
        ret = self._lib.rknnsmi_get_memory_info(
            _u16(device_id), _u16(chip_id), ctypes.byref(buf)
        )
        _raise_if_error(ret, "get_memory_info")
        n = int(buf.support_freq_cnt)
        if n > MAX_FREQ_CNT:
            n = MAX_FREQ_CNT
        return {
            "total_size": int(buf.total_size),
            "free_size": int(buf.free_size),
            "hugepage_size": int(buf.hugepage_size),
            "hugepage_total_cnt": int(buf.hugepage_total_cnt),
            "hugepage_free_cnt": int(buf.hugepage_free_cnt),
            "util_rate_pct": int(buf.util_rate),
            "cur_freq_mhz": int(buf.cur_freq),
            "support_freq_list_mhz": [int(buf.support_freq_list[i]) for i in range(n)],
            "support_freq_cnt": n,
        }

    # -------- reset --------
    def reset_device(self, device_id: int = 0) -> None:
        """Soft reset the device (mirrors ``rknn-smi reset -d <id>``)."""
        ret = self._lib.rknnsmi_reset_device(_u16(device_id))
        _raise_if_error(ret, "reset_device")

    def hw_reset_device(self, device_id: int = 0) -> None:
        """Hardware reset (mirrors ``rknn-smi reset -t hw -d <id>``).

        This is the operation the legacy ``/api/coprocessor/ws`` and
        ``restart_coprocessors.sh`` did four times in a row; the
        upstream rationale was "某些板子第一次复位不生效,需要多试
        几次", so callers may want to loop on transient errors.
        """
        ret = self._lib.rknnsmi_hw_reset_device(_u16(device_id))
        _raise_if_error(ret, "hw_reset_device")


__all__ = [
    "RknnSmi",
    "RknnSmiCallError",
    "RknnSmiError",
    "CommMode",
    "VersionType",
    "ChipHealthStatus",
    "ResetReason",
    "ThermalMode",
    "WorkMode",
    "PrefillMode",
    "load_library",
]
