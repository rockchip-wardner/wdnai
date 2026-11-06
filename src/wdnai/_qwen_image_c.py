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
from typing import Callable, List, Optional, Sequence

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

#: C 侧进度回调的 ctypes 签名：
#:     void (*)(void* user, const char* stage, int step, int total, double elapsed_ms)
#:
#: ⚠️ **GIL 安全**：本模块用 ``ctypes.CDLL``（**不是** PyDLL）⇒ 调 C 函数
#:    期间 GIL 是**释放**的 ⇒ C 里回调 Python 时会重新拿 GIL
#:    （ctypes 内部用 PyGILState_Ensure）⇒ 不会自锁 ✅
#:    用 PyDLL 就不行（PyDLL 不释放 GIL ⇒ 同线程回调 Python 直接死锁）。
_PROGRESS_CB = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
    ctypes.c_double,
)

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
    lib.qi_qwen_open_ex.restype = ctypes.c_void_p
    lib.qi_qwen_open_ex.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32,
        _PROGRESS_CB, ctypes.c_void_p,
        ctypes.c_char_p, ctypes.c_int,
    ]
    # ⭐ 带 VAE 核掩码的版本：VAE 跑在**本机 RK3588 的 rknn2 NPU**上时用
    lib.qi_qwen_open_ex2.restype = ctypes.c_void_p
    lib.qi_qwen_open_ex2.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32,
        ctypes.c_uint32,
        _PROGRESS_CB, ctypes.c_void_p,
        ctypes.c_char_p, ctypes.c_int,
    ]
    # ⭐ 带「VAE 独占卡」的版本：拓扑 A（2×1828+3588）传空 card_vae，
    #    拓扑 B（3×1828）传第三张卡的 BDF。
    #    ⚠️⚠️ **restype/argtypes 必须显式声明**：ctypes 默认 restype=c_int，
    #    会把 64 位会话指针**截断成 32 位** ⇒ open 阶段 C 侧照常跑完（看不出
    #    异常），但后续 generate/te_encode 拿截断指针解引用 ⇒ **SIGSEGV**。
    lib.qi_qwen_open_ex3.restype = ctypes.c_void_p
    lib.qi_qwen_open_ex3.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_uint32, ctypes.c_uint32,
        _PROGRESS_CB, ctypes.c_void_p,
        ctypes.c_char_p, ctypes.c_int,
    ]
    lib.qi_qwen_set_progress.restype = None
    lib.qi_qwen_set_progress.argtypes = [
        ctypes.c_void_p, _PROGRESS_CB, ctypes.c_void_p,
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
    # ⭐ 图生图（img2img）：输入图 + 提示词 → 新图
    lib.qi_qwen_img2img_batch.restype = ctypes.c_int
    lib.qi_qwen_img2img_batch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int, ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_char_p, ctypes.c_int,
    ]
    lib.qi_qwen_fixed_noise.restype = ctypes.c_int
    lib.qi_qwen_fixed_noise.argtypes = [
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint16)), ctypes.POINTER(ctypes.c_int),
        ctypes.c_char_p, ctypes.c_int,
    ]
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
        / ``dit_qwenimage21-512hc.int_ledger``）

        ⭐ 第三十六轮起**包里只有模型**：初始噪声由 cshim 现场生成
        （numpy 兼容的 PCG64 + ziggurat），不再需要 ``reference_512/``。
    card_te:
        **TE + DiT 头**所在卡（如 ``b"0000:01:00.0"``）
    card_dit:
        **DiT 4 段**所在卡（如 ``b"0003:31:00.0"``）
    card_vae:
        ⭐ **VAE 独占卡**（PCI BDF，如 ``b"0006:xx:00.0"``）—— 只在
        **3×1828** 拓扑下需要：模型包里没有 ``vae_qwenimage21-512.3588.rknn``
        （本机 3588 跑 rknn2）时，VAE(rknn3 4 段) 改用它**独占**第三张卡。

        **两种支持的拓扑**（全部模型在 open 阶段一次装好、全程常驻，
        推理过程中**不卸载/不重装载**）：

        A) ``2×1828 + 3588`` —— ``card_vae`` 留空 ⇒ VAE 跑本机 3588 的 NPU。
        B) ``3×1828`` —— ``card_vae`` = 第三张卡 ⇒ VAE 独占它。

        两者都没有 ⇒ **open 直接失败**（不再退回「与 DiT 共卡 + 每批重建 39 s」）。
    core_mask:
        NPU 核掩码（默认全开）—— 只作用于 **rknn3** 侧（1828）
    vae_core_mask:
        ⭐ **VAE** 的 NPU 核掩码（只作用于**本机 RK3588 的 rknn2 NPU**）。

        VAE 跑在 rknn2 上时，DiT（1828）与 VAE（3588）**物理分开**
        ⇒ DiT 全程常驻 ⇒ 生成过程**零 model_init**。

        - ``0``  → 默认（``RKNN_NPU_CORE_0_1_2``，RK3588 的 **3 个核全用**）
        - ``7``  → 显式 3 核
        - ``1``  → 只用 1 个核（更慢，但给别的任务留资源）

        VAE 走 rknn3（同卡）时这个参数被忽略。
    verbose:
        打印版本号
    progress:
        **进度回调**，签名 ``cb(stage: str, step: int, total: int, elapsed_ms: float)``。
        ``stage`` ∈ {"init", "te", "flow", "vae", "done"}（``vae_init`` / ``release``
        是旧「与 DiT 共卡」拓扑的遗留，现已**不再出现**）；
        ``elapsed_ms`` = **距上一次回调**的毫秒数（= 这一步花了多久）。

        ⭐ 必须在**构造时**传 —— 那 ~90 s 的模型加载（TE 37 s + DiT 39 s）
        发生在 ``qi_qwen_open`` **内部**，事后再装就看不到那段了。
    """

    def __init__(
        self,
        model_dir: os.PathLike | str,
        *,
        card_te: bytes = b"0000:01:00.0",
        card_dit: bytes = b"0003:31:00.0",
        card_vae: bytes = b"",
        core_mask: int = 0xFF,
        vae_core_mask: int = 0,
        verbose: bool = True,
        progress: Optional[Callable[[str, int, int, float], None]] = None,
    ) -> None:
        self._lib = _load()
        self.model_dir = Path(model_dir)
        self._err = ctypes.create_string_buffer(512)
        # 进度回调的 CFUNCTYPE 实例必须**持有引用** —— 否则会被 GC，
        # C 侧再回调就是野指针。
        self._progress_cb = None
        if progress is not None:
            def _trampoline(_user, stage, step, total, elapsed_ms):
                try:
                    progress(
                        stage.decode("utf-8", "replace"),
                        int(step), int(total), float(elapsed_ms),
                    )
                except Exception:  # noqa: BLE001 — 回调出错不能拖垮生成
                    pass

            self._progress_cb = _PROGRESS_CB(_trampoline)
        self._h = self._lib.qi_qwen_open_ex3(
            str(self.model_dir).encode(),
            card_te,
            card_dit,
            card_vae,
            ctypes.c_uint32(core_mask),
            ctypes.c_uint32(vae_core_mask),
            self._progress_cb,
            None,
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

    # -- 图生图 ------------------------------------------------------------ #
    def generate_img2img(
        self,
        prompts: Sequence[str],
        source_rgb: np.ndarray,
        *,
        steps: int = 8,
        mu: float = 0.5387096774,
        shift_terminal: float = 0.02,
        noise: str = "fixed",
        seed: int = 0,
        strength: float = 0.75,
    ) -> List[np.ndarray]:
        """⭐ **图生图**：输入图 + 提示词 → N 张新图。

        ``source_rgb`` 必须是 **512×512 的 uint8 RGB**（VAE 的输入尺寸固定，
        缩放由调用方用 RGA 做）。``strength`` = 保留原图的程度（0~1）：

        * 1.0 ⇒ 从纯噪声起步 ⇒ 与文生图**逐字节一致**；
        * 0.75 ⇒ 从 25% 的进度起步 ⇒ 原图结构保留、但被文本描述改写。

        返回 N 张 ``uint8 RGB [512, 512, 3]``。
        """
        if not prompts:
            return []
        mode = NOISE_FRESH if noise == "fresh" else NOISE_FIXED
        if mode == NOISE_FIXED and noise != "fixed":
            raise ValueError(f"未知噪声模式 {noise!r}（fixed / fresh）")

        src = np.ascontiguousarray(source_rgb, dtype=np.uint8)
        if src.ndim != 3 or src.shape[2] != 3:
            raise ValueError(f"输入图必须是 H×W×3 的 RGB，实际 {src.shape}")
        if src.shape[0] != 512 or src.shape[1] != 512:
            raise ValueError(
                f"输入图必须是 512×512，实际 {src.shape[1]}×{src.shape[0]}")

        token_lists = [self.tokenize(p) for p in prompts]
        lens = [len(t) for t in token_lists]
        flat: List[int] = []
        for t in token_lists:
            flat.extend(t)
        n = len(flat)

        p_tok = (ctypes.c_int32 * n)(*flat)
        p_len = (ctypes.c_int32 * len(lens))(*lens)
        p_src = (ctypes.c_uint8 * src.size).from_buffer(
            np.ascontiguousarray(src).reshape(-1).view(np.uint8))
        p_rgb = ctypes.POINTER(ctypes.c_uint8)()
        w = ctypes.c_int(0)
        h = ctypes.c_int(0)
        rc = self._lib.qi_qwen_img2img_batch(
            self._h,
            ctypes.c_int(len(prompts)),
            p_tok, p_len,
            ctypes.c_int(steps), ctypes.c_double(mu), ctypes.c_double(shift_terminal),
            ctypes.c_int(mode), ctypes.c_uint32(seed),
            p_src, ctypes.c_int(src.shape[1]), ctypes.c_int(src.shape[0]),
            ctypes.c_double(float(strength)),
            ctypes.byref(p_rgb), ctypes.byref(w), ctypes.byref(h),
            self._err, ctypes.c_int(len(self._err)),
        )
        if rc != 0:
            raise RuntimeError("图生图失败：" + self._err.value.decode("utf-8", "replace"))
        px = w.value * h.value * 3
        buf = np.ctypeslib.as_array(p_rgb, shape=(px * len(prompts),)).copy()
        self._lib.qi_qwen_free(p_rgb)
        return [
            buf[i * px:(i + 1) * px].reshape(h.value, w.value, 3) for i in range(len(prompts))
        ]


# --------------------------------------------------------------------------- #
# 「固定初始噪声」（不需要会话）
# --------------------------------------------------------------------------- #
#: 历史基线 reference_512/hidden_states.bin 的 md5 —— cshim 现场生成的必须与它一致
FIXED_NOISE_MD5 = "b808cf7ecd34338323b783704c516128"


def fixed_noise() -> np.ndarray:
    """生成「固定初始噪声」（1024 token × 64 通道，fp16）。

    cshim 内部是 numpy 兼容的 **PCG64 + ziggurat**（不是 libstdc++ 的极坐标法），
    与历史基线 ``reference_512/hidden_states.bin`` **逐字节一致**
    （md5 ``b808cf7ecd34338323b783704c516128``）
    ⇒ 模型包里不需要再带那个文件。

    返回
    ----
    np.ndarray
        形状 (65536,) 的 float16 数组。
    """
    lib = _load()
    p = ctypes.POINTER(ctypes.c_uint16)()
    n = ctypes.c_int(0)
    err = ctypes.create_string_buffer(256)
    rc = lib.qi_qwen_fixed_noise(ctypes.byref(p), ctypes.byref(n), err, ctypes.c_int(len(err)))
    if rc != 0:
        raise RuntimeError("生成固定噪声失败：" + err.value.decode("utf-8", "replace"))
    arr = np.ctypeslib.as_array(p, shape=(n.value,)).copy()
    lib.qi_qwen_free(p)
    return arr.view(np.float16)   # ctypes 给的是 uint16 位模式，重新解释成 fp16
