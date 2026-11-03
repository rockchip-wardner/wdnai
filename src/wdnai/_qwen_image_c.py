"""wdnai cshim 的 ctypes 绑定 —— Qwen-Image-2.1 文生图（rk1820）。

⭐ **Python 只负责「调接口」**：分词、拼 prompt 模板、写 PNG；
   TE 编码 / flow-match / VAE 解码 / 归一化**全在 C 侧**
   （``src/wdnai/_cshim/``，与参考实现同源 ⇒ 数值逐位一致）。

⇒ 运行时**不需要任何 Rockchip 的闭源 bin / so**：
   cshim 只链**系统自带**的 ``librknn3_api.so``（门面会 dlopen rkcp 后端）。

构建::

    cd src/wdnai/_cshim && make      # 产出 libqi_qwen.so
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

__all__ = [
    "QwenImageShim",
    "build_t2i_template",
    "cshim_path",
    "T2I_SYS_PROMPT",
]

#: diffusers 的 t2i system 提示词（逐字，不能改）
T2I_SYS_PROMPT = "Comprehend and analyze the provided prompt."

#: Qwen 的对话特殊 token。
#: ⚠️ 这里**故意拆开写**：`` 这类字面量在部分工具链（Markdown/HTML 渲染、
#:    补丁工具）里会被当成标签吞掉 ⇒ 直接写进源码会静默丢失（踩过一次）。
#: ⇒ 运行时拼接，行为与 diffusers 的 chat template 完全一致。
_IM_START = "<" + "|im_start|>"
_IM_END = "<" + "|im_end|>"

#: 噪声模式
NOISE_FIXED = 0
NOISE_FRESH = 1

_LIB: Optional[ctypes.CDLL] = None


def cshim_path() -> Path:
    """cshim 动态库路径（``wdnai/_cshim/libqi_qwen.so``）。"""
    return Path(__file__).parent / "_cshim" / "libqi_qwen.so"


def build_t2i_template(user_prompt: str) -> str:
    """diffusers 的 ``prompt_template_t2i``（**逐字**，不能改）。

    ⚠️ 必须带 Qwen 的对话特殊 token（``<|im_start|>`` / ``<|im_end|>``）——
    漏掉这 6 个特殊 token（4×151644 + 2×151645）会让
    ``encoder_hidden_states`` 变成「错位切片」，**出图内容完全不对但 corr 依然很高**
    （2026-11-01 修过一个致命 bug）。

    ⚠️ Rockchip 的闭源 tokenizer 会**自动**加这些标记；
       Python 侧的 gguf 分词器不会 ⇒ 必须在这里显式写上（否则 token 数 25 ≠ 30）。
    """
    return (
        _IM_START + "system\n" + T2I_SYS_PROMPT + _IM_END + "\n"
        + _IM_START + "user\n" + user_prompt + _IM_END + "\n"
        + _IM_START + "assistant\n"
    )


def _load() -> ctypes.CDLL:
    """加载 cshim（并声明好参数类型）。"""
    global _LIB
    if _LIB is not None:
        return _LIB
    so = cshim_path()
    if not so.exists():
        raise RuntimeError(
            f"wdnai cshim 未编译：{so}\n"
            "请先构建： cd src/wdnai/_cshim && make"
        )
    lib = ctypes.CDLL(str(so))
    lib.qi_qwen_open.restype = ctypes.c_void_p
    lib.qi_qwen_open.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32,
        ctypes.c_char_p, ctypes.c_int,
    ]
    lib.qi_qwen_close.restype = None
    lib.qi_qwen_close.argtypes = [ctypes.c_void_p]
    lib.qi_qwen_te_encode.restype = ctypes.c_int
    lib.qi_qwen_te_encode.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint16)), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)), ctypes.POINTER(ctypes.c_int),
        ctypes.c_char_p, ctypes.c_int,
    ]
    lib.qi_qwen_generate_batch.restype = ctypes.c_int
    lib.qi_qwen_generate_batch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int, ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.c_uint32,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_char_p, ctypes.c_int,
    ]
    lib.qi_qwen_free.restype = None
    lib.qi_qwen_free.argtypes = [ctypes.c_void_p]
    lib.qi_qwen_version.restype = ctypes.c_char_p
    lib.qi_qwen_version.argtypes = []
    _LIB = lib
    return lib


class QwenImageShim:
    """Qwen-Image-2.1 出图会话（TE + DiT 常驻，VAE 流式）。

    参数
    ----
    model_dir:
        模型包目录（必须含 ``llm_qwenimage21-te.*`` / ``dit_qwenimage21-512hc.seg*.*``
        / ``dit_qwenimage21-512.head.*`` / ``vae_qwenimage21-512.seg*.*``
        / ``qwenimage21-te.embed.bin`` / ``qwenimage21-te.tokenizer.gguf``
        / ``dit_qwenimage21-512hc.int_ledger`` / ``reference_512/``）
    card_te:
        **TE + DiT 头**所在卡（如 ``b"0000:01:00.0"``）
    card_dit:
        **DiT 4 段 + VAE**所在卡（如 ``b"0003:31:00.0"``）
    """

    def __init__(
        self,
        model_dir: os.PathLike | str,
        *,
        card_te: bytes = b"0000:01:00.0",
        card_dit: bytes = b"0003:31:00.0",
        core_mask: int = 0xFF,
        verbose: bool = True,
    ) -> None:
        self._lib = _load()
        self.model_dir = Path(model_dir)
        self._err = ctypes.create_string_buffer(512)
        self._h = self._lib.qi_qwen_open(
            str(self.model_dir).encode(),
            card_te,
            card_dit,
            ctypes.c_uint32(core_mask),
            self._err,
            ctypes.c_int(len(self._err)),
        )
        if not self._h:
            raise RuntimeError(
                "wdnai cshim 打开会话失败："
                + self._err.value.decode("utf-8", "replace")
            )
        # 分词器（纯 Python，**不依赖 Rockchip 的闭源 tokenizer 库**）
        from .gguf_tokenizer import load_tokenizer_from_gguf

        self._tokenizer = load_tokenizer_from_gguf(
            str(self.model_dir / "qwenimage21-te.tokenizer.gguf")
        )
        if verbose:
            print("[wdnai] cshim:", self._lib.qi_qwen_version().decode())

    # -- 生命周期 ---------------------------------------------------------- #
    def close(self) -> None:
        if getattr(self, "_h", None):
            self._lib.qi_qwen_close(self._h)
            self._h = None

    def __del__(self) -> None:  # noqa: D105
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self) -> "QwenImageShim":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- 分词 -------------------------------------------------------------- #
    def tokenize(self, prompt: str) -> List[int]:
        """t2i 模板 + 分词（**不带** special tokens）。"""
        text = build_t2i_template(prompt)
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    # -- TE 编码（调试用；出图走 generate）--------------------------------- #
    def encode(self, prompt: str):
        """返回 ``(ehs[128,4096] f16, mask[128] bool)``。"""
        tokens = self.tokenize(prompt)
        n = len(tokens)
        arr = (ctypes.c_int32 * n)(*tokens)
        p_ehs = ctypes.POINTER(ctypes.c_uint16)()
        p_mask = ctypes.POINTER(ctypes.c_uint8)()
        n_ehs = ctypes.c_int(0)
        n_mask = ctypes.c_int(0)
        rc = self._lib.qi_qwen_te_encode(
            self._h, arr, ctypes.c_int(n),
            ctypes.byref(p_ehs), ctypes.byref(n_ehs),
            ctypes.byref(p_mask), ctypes.byref(n_mask),
            self._err, ctypes.c_int(len(self._err)),
        )
        if rc != 0:
            raise RuntimeError("TE 编码失败：" + self._err.value.decode("utf-8", "replace"))
        ehs = np.ctypeslib.as_array(p_ehs, shape=(n_ehs.value,)).copy()
        mask = np.ctypeslib.as_array(p_mask, shape=(n_mask.value,)).copy()
        self._lib.qi_qwen_free(p_ehs)
        self._lib.qi_qwen_free(p_mask)
        return ehs, mask

    # -- 出图 -------------------------------------------------------------- #
    def generate(
        self,
        prompts: Sequence[str],
        *,
        steps: int = 8,
        mu: float = 0.5387096774,
        shift_terminal: float = 0.02,
        noise: str = "fixed",
        seed: int = 0,
    ) -> List[np.ndarray]:
        """批量出图，返回 N 张 ``uint8 RGB [H, W, 3]``。

        ⭐ **一个进程跑完 N 张**：TE / DiT 只 init 一次；
           N 张潜变量全跑完 → 释放 DiT → 逐张 VAE 解码。
        """
        if not prompts:
            return []
        mode = NOISE_FRESH if noise == "fresh" else NOISE_FIXED
        if mode == NOISE_FIXED and noise != "fixed":
            raise ValueError(f"未知噪声模式 {noise!r}（fixed / fresh）")

        token_lists = [self.tokenize(p) for p in prompts]
        lens = [len(t) for t in token_lists]
        flat: List[int] = []
        for t in token_lists:
            flat.extend(t)
        n = len(flat)

        p_tok = (ctypes.c_int32 * n)(*flat)
        p_len = (ctypes.c_int32 * len(lens))(*lens)
        p_rgb = ctypes.POINTER(ctypes.c_uint8)()
        w = ctypes.c_int(0)
        h = ctypes.c_int(0)
        rc = self._lib.qi_qwen_generate_batch(
            self._h,
            ctypes.c_int(len(prompts)),
            p_tok, p_len,
            ctypes.c_int(steps), ctypes.c_double(mu), ctypes.c_double(shift_terminal),
            ctypes.c_int(mode), ctypes.c_uint32(seed),
            ctypes.byref(p_rgb), ctypes.byref(w), ctypes.byref(h),
            self._err, ctypes.c_int(len(self._err)),
        )
        if rc != 0:
            raise RuntimeError("出图失败：" + self._err.value.decode("utf-8", "replace"))
        px = w.value * h.value * 3
        buf = np.ctypeslib.as_array(p_rgb, shape=(px * len(prompts),)).copy()
        self._lib.qi_qwen_free(p_rgb)
        return [
            buf[i * px:(i + 1) * px].reshape(h.value, w.value, 3) for i in range(len(prompts))
        ]

    # -- PNG --------------------------------------------------------------- #
    @staticmethod
    def write_png(path: os.PathLike | str, rgb: np.ndarray) -> None:
        """写 8bit RGB PNG（纯标准库 zlib，与基线 PNG **逐字节一致**）。"""
        from .backends.image import _write_png

        _write_png(Path(path), rgb)
