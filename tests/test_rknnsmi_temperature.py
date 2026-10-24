"""``RknnSmi`` 温度读取的单位与回退测试。

两处都曾经**静默出错**，症状不是崩溃而是「数字不对 / 抛异常」：

1. **chip 级单位换算错了** —— chip 级 API 返的是**整度**（raw=43 → 43 °C，
   与 ``rknn-smi info`` 打印的 Temp(C) 一致），旧实现按毫度 ÷1000 →
   **0.043 °C**。任何阈值判断（"> 85 °C"）都会被这个 0.043 骗过去。

2. **设备级命令没有回退** —— 设备级温度要过 PCIe 问 EP 固件，RK1828 的
   固件**不应答**（cmd 12 → ret=-1）→ 旧实现直接抛 ``RknnSmiCallError``。
   同一块固件的 chip 级却是好的。

修法：设备级失败时回退 chip 级，并记住「该卡设备级不通」，后续直接走
chip 级 —— 不再反复触发注定失败的 PCIe 命令（每次触发 C 层都会打一行
``[nsmi_pcie.c:651] Receive error, command: 12``）。

用假 lib 驱动，不碰真硬件。
"""
import ctypes

import pytest

from wdnai._librknnsmi import RknnSmi, RknnSmiCallError


class _FakeLib:
    """最小假 lib：只实现两个温度命令。"""

    def __init__(self, *, device_raw=None, device_ret=0, chip_raw=43,
                 chip_ret=0):
        self.device_raw = device_raw
        self.device_ret = device_ret
        self.chip_raw = chip_raw
        self.chip_ret = chip_ret
        self.device_calls = 0
        self.chip_calls = 0

    def rknnsmi_get_device_temperature(self, device_id, out_ptr):
        self.device_calls += 1
        if self.device_ret != 0:
            return self.device_ret
        ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_int32))[0] = self.device_raw
        return 0

    def rknnsmi_get_chip_temperature(self, device_id, chip_id, out_ptr):
        self.chip_calls += 1
        if self.chip_ret != 0:
            return self.chip_ret
        ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_int32))[0] = self.chip_raw
        return 0


def _smi(fake: _FakeLib) -> RknnSmi:
    """造一个不加载真 .so 的 RknnSmi（绕过 __init__ 的 load_library）。"""
    obj = RknnSmi.__new__(RknnSmi)
    obj._lib = fake
    return obj


def _unsupported_cache() -> set:
    """设备级不通的卡集合（老版本没这个字段 → 返回空集）。"""
    cache = getattr(RknnSmi, "_temp_device_unsupported", None)
    return cache if isinstance(cache, set) else set()


@pytest.fixture(autouse=True)
def _clear_cache():
    """类级缓存跨用例共享 —— 每个用例都要从干净状态开始。"""
    cache = getattr(RknnSmi, "_temp_device_unsupported", None)
    if isinstance(cache, set):
        cache.clear()
    yield
    if isinstance(cache, set):
        cache.clear()


# --------------------------------------------------------------------------- #
# 单位
# --------------------------------------------------------------------------- #
def test_chip_temperature_is_whole_degrees():
    """chip 级 raw=43 → 43.0 °C（**不是** 0.043）。"""
    smi = _smi(_FakeLib(chip_raw=43))
    assert smi.get_chip_temperature(0, 0) == 43.0


def test_chip_temperature_does_not_divide_by_1000():
    """回归点：旧实现对 chip 级也 ÷1000 → 0.043。"""
    smi = _smi(_FakeLib(chip_raw=43))
    value = smi.get_chip_temperature(0, 0)
    assert value != 0.043
    assert value > 1.0, "43 °C 被当成毫度除就变成了 0.043"


def test_device_temperature_milli_degrees_when_device_level_works():
    """设备级可用时仍是毫度 ×1000（该路径本机无法实测，保持原语义）。"""
    smi = _smi(_FakeLib(device_raw=43000))
    assert smi.get_device_temperature(0) == 43.0


# --------------------------------------------------------------------------- #
# 回退
# --------------------------------------------------------------------------- #
def test_device_temperature_falls_back_to_chip_level():
    """设备级不应答（ret=-1）→ 回退 chip 级拿到温度，而不是抛异常。"""
    smi = _smi(_FakeLib(device_ret=-1, chip_raw=45))
    assert smi.get_device_temperature(0) == 45.0
    assert smi._lib.chip_calls == 1


def test_device_level_failure_is_remembered():
    """记住该卡设备级不通 → 后续不再触发注定失败的 PCIe 命令。"""
    fake = _FakeLib(device_ret=-1, chip_raw=43)
    smi = _smi(fake)
    smi.get_device_temperature(0)
    smi.get_device_temperature(0)
    smi.get_device_temperature(0)
    assert fake.device_calls == 1, "设备级命令被重复触发 → C 层又打 Receive error"
    assert fake.chip_calls == 3


def test_remembering_is_per_device():
    """卡 0 不通不影响卡 1 走设备级。"""
    fake = _FakeLib(device_ret=-1, chip_raw=43)
    smi = _smi(fake)
    smi.get_device_temperature(0)
    assert 0 in _unsupported_cache()
    assert 1 not in _unsupported_cache()


def test_both_levels_failing_raises():
    """两级都不通 → 仍然抛 RknnSmiCallError（op 要是设备级）。"""
    smi = _smi(_FakeLib(device_ret=-1, chip_ret=-1))
    with pytest.raises(RknnSmiCallError) as exc:
        smi.get_device_temperature(0)
    assert exc.value.op == "get_device_temperature"
    # 两级都失败时**不能**把卡记成「设备级不通」—— 下次还得重试。
    assert 0 not in _unsupported_cache()
