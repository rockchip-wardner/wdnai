# coding=utf-8
"""ECAPA-TDNN speaker encoder (RKNN) — 把一段参考音频变成 2048 维音色向量。

这是 :mod:`wdnai.backends.tts` 里内嵌的那几个参考音色的**现场提取**版本：
内嵌的向量是离线算好签进 SDK 的，这里的向量是从用户上传的音频现场算出来的。

实现要点
--------
* **纯 numpy 前处理**：不依赖 torch / librosa / scipy。mel 滤波器组是
  一个常量矩阵（``speaker_encoder_mel.npy``，128x513），随模型一起发布，
  板端只做 STFT + 矩阵乘。
* 前处理必须与参考实现（PyTorch 侧 ``mel_spectrogram``）逐位对齐，
  三处容易踩的细节：

  1. hann 窗是 **periodic**（``torch.hann_window`` 默认），不是
     ``numpy.hanning`` 的 symmetric；
  2. reflect pad 是「不重复边缘值」的镜像（torch 与 numpy 一致）；
  3. mel 压缩是 ``log(clamp(x, 1e-5))``，不是 ``log(x + eps)``。

  实测 numpy 与 torch 的 mel 相对 L2 差 0.0001%，余弦 1.00000012。
* **RKNN 推理**：走 SDK 自己的 ``wdnai._librknn`` shim（不是
  ``rknn3lite`` wheel —— 后者是第二套 NPU 驱动栈，和 shim 抢同一个 NPU
  不安全）。模型由 ``rknn3_model_zoo`` 的
  ``examples/Qwen3_TTS/python/embeds/export_speaker_encoder_rknn.py``
  导出，``target_platform='rk1820'``，不量化（跑 fp16）。
  fp16 相对 fp32 参考的 embedding 偏差约 4~8%，实测对合成结果听不出
  差别（组间 log-mel 谱距离 / 组内自然波动 ≈ 1.09x）。
* **固定输入长度**：468 帧 = 5 秒。RKNN 动态 shape 容易踩坑，这里
  用固定 shape 换稳健性。长音频截前 5 秒，短音频**尾部补零**到 5 秒。
* **懒加载**：``RKNN3Lite`` 只在 :meth:`SpeakerEncoder.__init__` 内部
  import —— librknn3 的 native driver 绝不能在模块 import 阶段打开。
"""

from __future__ import annotations

import os
import struct
import threading
from typing import Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# 常量 —— 与参考实现严格对应
# --------------------------------------------------------------------------- #
SR = 24000                 # speaker encoder 输入采样率（硬要求）
N_FFT = 1024
HOP = 256
WIN = 1024
N_MELS = 128
FMIN = 0
FMAX = 12000
PAD = (N_FFT - HOP) // 2          # 384 —— 参考实现 stft 前的手工 reflect pad

EMBED_DIM = 2048         # ECAPA-TDNN 输出维度

# 分析窗口：5.0 秒。这个长度不是随便取的 —— 导出脚本就是拿 120000 个
# 样点（5.0 s @ 24 kHz）算的 mel，得到的帧数正好是 468。**必须用同一个
# 长度**：截到 119808（刚好凑够 468 帧的最短长度）会有 1.1e-2 的相对误差
# —— 最后一帧的尾部从真实音频变成了右端 reflect pad，log-mel 把这点差异
# 放大了。
WINDOW_SECONDS = 5.0
MAX_SAMPLES = int(SR * WINDOW_SECONDS)      # 120000
FRAMES = 468                                 # 固定输入帧数

# 短音频补零策略：尾部补零（不是帧轴前面补零）。实测（3 秒真实人声，
# 金标准 = torch 模型在自然长度上的 embedding）：尾部补零 cos = 0.99832，
# 尾部 reflect cos = 0.99735，帧轴前补零 cos = 0.97353 —— 明显最差
# （静音在前，像没说话）。

MODEL_BASENAME = "speaker_encoder"


def frames_for_samples(n: int) -> int:
    """给定采样数，算 mel 帧数（center=False + 前后各 reflect pad PAD）。"""
    return (n + 2 * PAD - N_FFT) // HOP + 1


# 分析窗口长度和帧数必须对得上 —— 不对就说明常量被改坏了。
assert frames_for_samples(MAX_SAMPLES) == FRAMES, (
    "MAX_SAMPLES={} 算不出 {} 帧".format(MAX_SAMPLES, FRAMES)
)


# --------------------------------------------------------------------------- #
# wav 读取
# --------------------------------------------------------------------------- #
class WavFormatError(ValueError):
    """WAV 文件结构不合法，或者格式不支持。

    继承 ``ValueError`` —— 老调用方 ``except ValueError`` 的地方不用改。
    """


def _supported_format(tag: int, bits: int) -> bool:
    """这个 (format tag, bit depth) 组合我们解得了吗？

    和 :func:`read_wav` 的解码分支一一对应 —— 两处各写各的就会出现
    "app 层放过去了、SDK 解不了" 的洞（最后变成 500 而不是 400）。
    """
    if tag == 3 and bits in (32, 64):
        return True                                   # IEEE float
    return bits in (8, 16, 24, 32)                   # 整数 PCM


def _parse_riff(blob: bytes) -> dict:
    """解析 RIFF/WAVE 的 chunk 结构（**不解码** PCM）。

    返回 ``{"format_tag", "channels", "sample_rate", "bits",
    "block_align", "data_offset", "data_bytes", "frames"}``。
    结构不对就抛 :class:`WavFormatError`。
    """
    if len(blob) < 12 or blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise WavFormatError(
            "不是 RIFF/WAVE 文件（目前只支持 WAV/PCM）")
    pos = 12
    fmt = None
    data_offset = 0
    data_bytes = 0
    seen_data = False
    while pos + 8 <= len(blob):
        cid = blob[pos:pos + 4]
        csz = struct.unpack("<I", blob[pos + 4:pos + 8])[0]
        body_start = pos + 8
        # 只算长度，不切片 —— data chunk 可能有 20 MB，探测路径下
        # 切一刀就是白复制 20 MB。
        body_len = min(csz, len(blob) - body_start)
        if cid == b"fmt ":
            body = blob[body_start:body_start + body_len]
            if len(body) < 16:
                raise WavFormatError(
                    "fmt chunk 只有 {} 字节（至少要 16）".format(len(body)))
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            # 有些录音软件写的 chunk size 不准 —— 取实际剩下的字节，
            # 别把后面的 chunk 当成音频。
            data_offset = body_start
            data_bytes = body_len
            seen_data = True
        pos += 8 + csz + (csz & 1)

    if fmt is None:
        raise WavFormatError("缺 fmt chunk")
    if not seen_data:
        raise WavFormatError("缺 data chunk")

    tag, nch, sr, _, _, bits = fmt
    if nch < 1:
        raise WavFormatError("声道数非法: {}".format(nch))
    if sr <= 0:
        raise WavFormatError("采样率非法: {}".format(sr))
    if not _supported_format(tag, bits):
        raise WavFormatError(
            "不支持的音频格式（format tag={} bits={}）：只支持 PCM "
            "8/16/24/32 位和 32/64 位 float".format(tag, bits))
    block_align = nch * (bits // 8)
    if data_bytes % block_align:
        raise WavFormatError(
            "data 长度 {} 不是声道对齐 {} 的整数倍（文件被截断或损坏）".format(
                data_bytes, block_align))
    return {
        "format_tag": int(tag),
        "channels": int(nch),
        "sample_rate": int(sr),
        "bits": int(bits),
        "block_align": int(block_align),
        "data_offset": int(data_offset),
        "data_bytes": int(data_bytes),
        "frames": int(data_bytes // block_align),
    }


def probe_wav_bytes(blob: bytes) -> dict:
    """只读 WAV 头 → 头部信息 + 时长（**不解码** PCM）。

    给上层做输入校验用 —— 没必要为了看一眼时长就把整段音频解码成
    float32（20 MB 文件解完就是 80 MB 内存）。

    返回 ``{format_tag, channels, sample_rate, bits, block_align,
    data_bytes, frames, duration}``，``duration`` 单位秒。
    """
    info = _parse_riff(bytes(blob))
    sr = info["sample_rate"]
    info["duration"] = info["frames"] / float(sr) if sr else 0.0
    return info


def read_wav(path: str) -> Tuple[np.ndarray, int]:
    """
    读 wav → ``(pcm float32 mono, sample_rate)``。

    自己解析 RIFF 而不是用 stdlib ``wave``：后者遇到 IEEE float
    （format tag 3）直接抛 ``Error: unknown format: 3``，而板端 TTS
    输出的 WAV 恰好就是 32-bit IEEE float。

    支持 PCM8 / PCM16 / PCM24 / PCM32 与 float32 / float64。
    立体声取**能量最大的那个声道** —— 分轨/卡拉OK 式录音里人声常常
    只在一侧，固定取左声道会提到静音。
    """
    with open(path, "rb") as f:
        blob = f.read()
    info = _parse_riff(blob)
    tag, bits, nch = info["format_tag"], info["bits"], info["channels"]
    data = blob[info["data_offset"]:info["data_offset"] + info["data_bytes"]]

    if tag == 3 and bits == 32:
        pcm = np.frombuffer(data, dtype="<f4").astype(np.float32)
    elif tag == 3 and bits == 64:
        pcm = np.frombuffer(data, dtype="<f8").astype(np.float32)
    elif bits == 16:
        pcm = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    elif bits == 32:
        pcm = np.frombuffer(data, dtype="<i4").astype(np.float32) / 2147483648.0
    elif bits == 24:
        raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        val = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        val = np.where(val & 0x800000, val - 0x1000000, val)
        pcm = val.astype(np.float32) / 8388608.0
    elif bits == 8:
        pcm = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        # _parse_riff 已经筛过一遍了，到这儿说明两处逻辑对不上
        raise WavFormatError(
            "不支持的格式 tag={} bits={}".format(tag, bits))

    if nch > 1:
        frames = pcm.reshape(-1, nch)
        energies = [float((frames[:, c] * frames[:, c]).mean())
                    for c in range(nch)]
        pcm = np.ascontiguousarray(frames[:, int(np.argmax(energies))])
    return pcm, info["sample_rate"]


def resample_to_24k(pcm: np.ndarray, sr: int) -> np.ndarray:
    """重采样到 24 kHz（线性插值）。已经是 24 kHz 时原样返回。"""
    if sr == SR:
        return pcm.astype(np.float32)
    n_out = int(round(len(pcm) * SR / sr))
    x_old = np.linspace(0.0, 1.0, num=len(pcm), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, pcm).astype(np.float32)


# --------------------------------------------------------------------------- #
# mel 前处理
# --------------------------------------------------------------------------- #
def hann_periodic(n: int) -> np.ndarray:
    """``torch.hann_window(n)`` 的 periodic 窗。

    注意不是 ``numpy.hanning``（那是 symmetric，periodic=False）。
    """
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float32)


def mel_spectrogram(pcm: np.ndarray, mel_basis: np.ndarray) -> np.ndarray:
    """``(T,) @24 kHz`` → ``(frames, 128)`` log-mel。

    对应参考实现::

        y    = F.pad(y, (384, 384), mode="reflect")
        spec = torch.stft(y, 1024, hop_length=256, win_length=1024,
                           window=hann_window(1024), center=False,
                           normalized=False, onesided=True)
        mag  = sqrt(re**2 + im**2 + 1e-9)
        mel  = log(clamp(mel_basis @ mag, min=1e-5))
    """
    y = np.pad(pcm.astype(np.float32), (PAD, PAD), mode="reflect")
    n_frames = (len(y) - N_FFT) // HOP + 1
    win = hann_periodic(WIN)
    idx = np.arange(N_FFT)[None, :] + (np.arange(n_frames) * HOP)[:, None]
    frames = y[idx] * win[None, :]
    spec = np.fft.rfft(frames, n=N_FFT, axis=1)
    mag = np.sqrt(spec.real ** 2 + spec.imag ** 2 + 1e-9).astype(np.float32)
    mel = mel_basis.astype(np.float32) @ mag.T              # (128, frames)
    mel = np.log(np.maximum(mel, 1e-5))
    return mel.T.astype(np.float32)                         # (frames, 128)


# --------------------------------------------------------------------------- #
# 提取器
# --------------------------------------------------------------------------- #
class SpeakerEncoder:
    """
    RKNN 版 ECAPA-TDNN speaker encoder。

    典型用法::

        enc = SpeakerEncoder(model_dir)            # model_dir = .../Qwen3-TTS
        emb = enc.encode_file("/path/to/ref.wav")   # -> ndarray(2048,) float32

    ``model_dir`` 里需要三个文件::

        speaker_encoder.1828.rknn    # RKNN 模型（fp16，不量化）
        speaker_encoder.weight       # RKNN 权重
        speaker_encoder_mel.npy      # mel 滤波器组 (128, 513) float32
    """

    def __init__(
        self,
        model_dir,
        target: str = "rk1820",
        core_mask: int = 0x02,
        device_id: Optional[bytes] = None,
        verbose: bool = False,
        resident: bool = False,
    ):
        self.model_dir = str(model_dir)
        self.target = target
        self.core_mask = core_mask
        self.device_id = device_id
        self.verbose = verbose
        self.rknn = None
        self.frames = FRAMES
        self.n_mels = N_MELS

        rknn_path, weight_path = self._resolve_model_paths(self.model_dir)
        mel_path = os.path.join(self.model_dir, MODEL_BASENAME + "_mel.npy")
        for p in (rknn_path, weight_path, mel_path):
            if not os.path.isfile(p):
                raise FileNotFoundError("missing speaker encoder file: {}".format(p))
        mel_basis = np.load(mel_path)
        if mel_basis.shape != (N_MELS, N_FFT // 2 + 1):
            raise ValueError(
                "mel filterbank shape mismatch: got {}, expect ({}, {})".format(
                    mel_basis.shape, N_MELS, N_FFT // 2 + 1)
            )
        self.mel_basis = mel_basis.astype(np.float32)

        # ``resident=True``：构造完就占着 NPU 上下文。
        # ``False``（默认）：第一次 encode 时才开，用完立刻释放 —— 这样
        # TTS 主链路平时完全看不到这个上下文。
        self._resident = bool(resident)
        # 非常驻模式下 open/encode/close 是一整套，必须串行化 —— 两个
        # 请求同时进来会各开一个 NPU 上下文，白白多占一份显存。
        self._lock = threading.Lock()
        if self._resident:
            self.open_context()

    # ------------------------------------------------------------------ #
    # NPU 上下文生命周期
    # ------------------------------------------------------------------ #
    def open_context(self) -> bool:
        """打开 NPU 上下文（幂等）。返回是否成功。

        RKNN3 的 native lib 只能懒加载 —— 顶层 import 会让非 RK 机器
        import wdnai 时 segfault（见 AGENTS.md）。
        """
        if self.rknn is not None:
            return True
        from .._librknn import RKNN3Lite

        rknn = RKNN3Lite(verbose=self.verbose)
        ret = rknn.load_rknn(*self._resolve_model_paths(self.model_dir))
        if ret != 0:
            raise RuntimeError("load_rknn failed ({}): {}".format(self.model_dir, ret))
        ret = rknn.init_runtime(target=self.target, core_mask=self.core_mask,
                                device_id=self.device_id)
        if ret != 0 and self.core_mask != 0xFF:
            # 某些板子核数少，专属核可能不存在 —— 退回「所有核」再试一次。
            if self.verbose:
                print("[speaker_encoder] core_mask={} 不可用, 回退 0xff".format(self.core_mask))
            ret = rknn.init_runtime(target=self.target, core_mask=0xFF,
                                    device_id=self.device_id)
        if ret != 0:
            raise RuntimeError("init_runtime failed: {}".format(ret))

        self.rknn = rknn
        self.frames, self.n_mels = self._read_input_shape()
        if self.n_mels != N_MELS:
            raise RuntimeError(
                "speaker_encoder: 模型 mel 维度 {} != {}".format(self.n_mels, N_MELS)
            )
        if self.verbose:
            print("[speaker_encoder] ready: frames={} device={!r}".format(
                self.frames, self.device_id))
        return True

    def close_context(self) -> None:
        """释放 NPU 上下文（幂等）。"""
        if self.rknn is not None:
            try:
                self.rknn.release()
            except Exception as exc:      # noqa: BLE001
                print("[speaker_encoder] release: {}".format(exc))
            self.rknn = None


    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_model_paths(model_dir):
        """``<role>.1828.rknn`` 优先，回退 ``<role>.rknn``（同 TTS 其余模型）。"""
        model_dir = str(model_dir)
        weight_path = os.path.join(model_dir, MODEL_BASENAME + ".weight")
        for rknn_name in (MODEL_BASENAME + ".1828.rknn", MODEL_BASENAME + ".rknn"):
            cand = os.path.join(model_dir, rknn_name)
            if os.path.isfile(cand):
                return cand, weight_path
        return os.path.join(model_dir, MODEL_BASENAME + ".1828.rknn"), weight_path

    def _read_input_shape(self):
        """读模型输入 shape → ``(frames, n_mels)``。

        ONNX 的输入是 2 维 ``(468, 128)``，RKNN 侧会补一个 batch 维变成
        ``(1, 468, 128)``。两种都兼容：取最后两维。
        """
        attrs = self.rknn.get_inputs_tensor_attr()
        if not attrs:
            raise RuntimeError("speaker_encoder: 没有输入 tensor 属性")
        attr = attrs[0]
        shape = list(getattr(attr, "shape", []))
        n_dims = int(getattr(attr, "n_dims", 0) or 0)
        if n_dims:
            shape = shape[:n_dims]
        if len(shape) < 2:
            raise RuntimeError(
                "speaker_encoder: 输入 shape 异常: {}".format(shape)
            )
        return int(shape[-2]), int(shape[-1])

    # ------------------------------------------------------------------ #
    def encode_pcm(self, pcm: np.ndarray) -> np.ndarray:
        """``(T,) @24 kHz float32`` → ``(2048,) float32`` 音色向量。

        长度处理：长于 5 秒截前 5 秒，短于 5 秒**尾部补零**到 5 秒 ——
        两种情况都得到恰好 468 帧，且和参考实现逐位对齐。
        """
        pcm = np.asarray(pcm, dtype=np.float32)
        if len(pcm) > MAX_SAMPLES:
            pcm = pcm[:MAX_SAMPLES]
        elif len(pcm) < MAX_SAMPLES:
            pcm = np.concatenate(
                [pcm, np.zeros(MAX_SAMPLES - len(pcm), dtype=np.float32)])
        mel = mel_spectrogram(pcm, self.mel_basis)
        if mel.shape[0] < self.frames:
            # 兜底：帧数不足时在**尾部**补零（实测尾部补零远好于头部补零）。
            mel = np.concatenate(
                [mel, np.zeros((self.frames - mel.shape[0], self.n_mels), dtype=np.float32)],
                axis=0)
        else:
            mel = mel[:self.frames]
        try:
            with self._lock:
                if self.rknn is None:
                    self.open_context()
                out = self.rknn.inference(
                    inputs=[mel.reshape(1, self.frames, self.n_mels)])
                if not out:
                    raise RuntimeError("speaker_encoder: inference 失败")
                return np.asarray(out[0]).reshape(-1).astype(np.float32)
        finally:
            # 非常驻模式：提取完立刻放掉 NPU 上下文，别占着 TTS 主链路的核。
            if not self._resident:
                self.close_context()

    def encode_file(self, path: str) -> np.ndarray:
        """音频文件 → 音色向量。采样率不是 24 kHz 会先重采样。"""
        pcm, sr = read_wav(path)
        return self.encode_pcm(resample_to_24k(pcm, sr))

    def release(self) -> None:
        self.close_context()
