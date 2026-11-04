"""Text-to-image backend — Qwen-Image-2.1 on RKNN3 (rk1820).

Qwen-Image-2.1 is a three-part diffusion model:

===========  =================================  ==================================
part         role                              RKNN artefacts
===========  =================================  ==================================
TE           prompt → ``encoder_hidden_states``  ``llm_qwenimage21-te.*``
DiT          flow-match *N* steps → latents      ``dit_qwenimage21-512hc.seg{0..3}.*``
VAE          latents → 512×512 image            ``vae_qwenimage21-512.seg{0..3}.*``
===========  =================================  ==================================

Why a C shim instead of pure Python?
------------------------------------
The three-part orchestration on rk1820 is **not** a straight port of the
upstream pipeline. Every one of the rules below was measured on real
hardware and is encoded in the C shim ``wdnai/_cshim/libqi_qwen.so``
(built from source in this repo — **no closed-source binary anywhere**):

* the DiT is cut into 4 × 8-block segments; each segment used to carry
  its own copy of the transformer *head* (``timestep_embedder`` +
  ``modulation``), so the head was **cut out** into a standalone model
  (zero numerical loss) and the 4 segments then fit on **one** card
  with shared ``internal`` memory;
* the head model **must** live on a different card than the segments;
* the VAE is a 4-segment chain that must run **streaming** — all four
  segments resident at once dies with ``MODEL_SETUP fail``;
* the DiT and the VAE **cannot share a card** ⇒ the batch order is
  "all latents first, then release the DiT, then decode";
* TE **can** co-reside with the head (486.0 + 6.4 = 492.4 MiB/core)
  ⇒ it stays resident in the same process and encodes every prompt in
  ~0.16 s instead of paying a 36.8 s ``model_init`` per prompt.

Those rules (plus the fp16 layout conventions, the FMA rounding and the
``std::mt19937`` noise chain) are **exact numerical** requirements, so the
algorithms live in C — the same source that produced the golden baseline —
and Python only calls them through ctypes.  A pure-Python re-implementation
was measured to be bit-exact too, but it duplicated a lot of subtle,
hard-won knowledge (three separate one-ULP traps) for ~0.2 s per image.

The shim links **only the system-provided** ``librknn3_api.so`` (the facade
dlopens the rkcp backend) ⇒ the model package ships **models only** — no
Rockchip ``.so`` and no opaque binary.

Card layout (2026-11-02 第三十五轮)::

    card 0 : TE 常驻 (486.0 MiB/core) + DiT 头 (6.4 MiB/core) = 492.4
    card 1 : DiT 4 段 (525.0 MiB/core, 共享 internal)
             → N 张图跑完 → 释放 → VAE 4 段流式 (峰值 516.5)

Weights are read **straight from the model package** (NVMe) — no tmpfs
staging. Measured: the ``load`` (weight registration) time is identical
either way; the only cost is ~3.6 s per segment inside ``model_init``
(a one-off ~14 s for the 4 DiT segments).

Layout of the model package (see ``/home/rockchip/models/qwenimage21``)::

    llm_qwenimage21-te.1828.rknn / .weight
    qwenimage21-te.embed.bin / .tokenizer.gguf
    dit_qwenimage21-512hc.seg{0..3}.1828.rknn / .weight   DiT 4 段（切头）
    dit_qwenimage21-512.head.1828.rknn / .weight          DiT 的「头」（另一张卡）
    dit_qwenimage21-512hc.int_ledger                     每核 internal 账本（跳探针）
    vae_qwenimage21-512.seg{0..3}.1828.rknn / .weight

⭐ 模型包里**只有模型**：「固定初始噪声」由 cshim 现场生成
   （numpy 兼容的 PCG64 + ziggurat，与历史基线逐字节一致）。
"""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..attachment import Attachment, ImageAttachment

__all__ = ["ImageBackend"]


# --------------------------------------------------------------------------- #
# Model-package layout (mirrors /home/rockchip/models/qwenimage21)
# --------------------------------------------------------------------------- #
_TE_RKNN = "llm_qwenimage21-te.1828.rknn"
_TE_WEIGHT = "llm_qwenimage21-te.weight"
_TE_TOKENIZER = "qwenimage21-te.tokenizer.gguf"
_TE_EMBED = "qwenimage21-te.embed.bin"
_N_SEG = 4

#: DiT 段模型 / 权重名模板（切头单卡：4 段在同一张卡 + 头在另一张卡）
_DIT_MODEL_TPL = "dit_qwenimage21-512hc.seg{i}.1828.rknn"
_DIT_WEIGHT_TPL = "dit_qwenimage21-512hc.seg{i}.weight"
_HEAD_MODEL = "dit_qwenimage21-512.head.1828.rknn"
_HEAD_WEIGHT_NAME = "dit_qwenimage21-512.head.weight"

#: 切头变体的每核 internal 账本（跳探针用）。跟这批 .rknn 绑定，换模型必须重测。
_INT_LEDGER_NAME = "dit_qwenimage21-512hc.int_ledger"

#: 512×512 uses ``mu = 0.5387096774`` (latent sequence length 1024).
#: The 1024×1024 value (0.6935483871) is **wrong** here.
_FM_MU_512 = "0.5387096774"

#: Default board topology (2 × RK1828 behind the RK3588 host).
_CARD0 = "0000:01:00.0"
_CARD1 = "0003:31:00.0"

#: Candidate model-package directories, in priority order.
_MODEL_DIR_CANDIDATES = (
    "/home/rockchip/models/qwenimage21",
    "/userdata/models/qwenimage21",
)

_DTYPES = {
    "float16": "float16", "fp16": "float16",
    "float32": "float32", "fp32": "float32",
    "float64": "float64", "fp64": "float64",
    "int32": "int32", "int64": "int64",
    "uint8": "uint8", "bool": "bool",
}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _log(msg: str) -> None:
    print(f"[wdnai.image] {msg}", flush=True)


def _read_meta(path: Path) -> Tuple[Optional[str], Optional[List[int]]]:
    """Parse a ``.bin.meta`` sidecar written by the C++ runtime.

    Format (see ``write_bin_meta`` in ``src/main.cpp``)::

        # dtype shape
        float16 1,4,512,512
    """
    meta = Path(str(path) + ".meta")
    if not meta.is_file():
        return None, None
    try:
        for line in meta.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            dtype = parts[0].lower() if parts else None
            shape: Optional[List[int]] = None
            if len(parts) >= 2:
                try:
                    shape = [int(x) for x in parts[-1].split(",")]
                except ValueError:
                    shape = None
            return dtype, shape
    except OSError:
        return None, None
    return None, None


def _write_png(path: Path, rgb: Any) -> None:
    """Write an 8-bit RGB PNG — pure stdlib (zlib), no PIL needed.

    Byte-compatible with the deploy-side ``scripts/fp16_to_png.py``
    (same filter-0 scanlines, same zlib level), so the PNG the SDK
    writes and the PNG the runtime writes are interchangeable.
    """
    import numpy as np

    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6))
    png += chunk(b"IEND", b"")
    Path(path).write_bytes(png)


# --------------------------------------------------------------------------- #
# Progress sink
# --------------------------------------------------------------------------- #
class _ProgressSink:
    """A ``sink``-alike for the image backend.

    :class:`wdnai.session.LLMLocalSession` looks for ``backend.sink`` to
    (a) deliver ``on_delta`` callbacks and (b) implement ``stop()``.
    The image backend has no token stream, so "deltas" are **progress
    messages** (``编码提示词…`` / ``生成中…``) rather than model tokens —
    the concatenation rule that applies to text models does not hold
    here.

    ``stop()`` sets a flag that the subprocess watcher polls, so a
    long-running generation can be aborted.
    """

    def __init__(self) -> None:
        self._observers: List[Any] = []
        self._stopped = False
        self.perf: Dict[str, Any] = {}
        self.text = ""

    # -- observer protocol (LLMSession._dispatch) ------------------------- #
    def reset(self) -> None:
        self._stopped = False
        self.text = ""
        self.perf = {}

    def add_observer(self, cb: Any) -> None:
        if cb not in self._observers:
            self._observers.append(cb)

    def remove_observer(self, cb: Any) -> None:
        try:
            self._observers.remove(cb)
        except ValueError:
            pass

    def emit(self, message: str) -> None:
        for cb in list(self._observers):
            try:
                cb(message)
            except Exception:  # noqa: BLE001 — a bad observer must not kill the run
                pass

    # -- abort path (LLMLocalSession.stop) -------------------------------- #
    def stop(self) -> None:
        self._stopped = True

    @property
    def stopped(self) -> bool:
        return self._stopped


# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #
class ImageBackend:
    """Qwen-Image-2.1 text-to-image (RKNN3 / rk1820).

    Construction only *resolves and validates* the on-disk layout — no
    NPU work happens until :meth:`warmup` / :meth:`run_text` is called.

    ⭐ 2026-11-03 第三十七轮：``LLMLocalSession`` 构造完 backend 后会调
    可选的 ``warmup()`` 钩子（见 :meth:`warmup`）⇒ **构造 session 不再是
    零 NPU 成本**（会付 ~90 s 的 cshim init）。这是刻意的：把这笔开销挪到
    worker 的 load 阶段，推理阶段就只剩 34 s，不会顶穿 webui 的 120 s
    RPC 推理超时。

    :meth:`run_text` returns ``(prompt, [ImageAttachment])``: the text is
    the prompt echo (the same convention the TTS backend uses) and the
    attachment carries the generated **BGR** ndarray, ready for
    ``cv2.imshow`` / ``ImageAttachment``.
    """

    def __init__(
        self,
        spec,
        *,
        npu: Optional[int] = None,
        device_id: Optional[Any] = None,
    ):
        self.spec = spec
        self.model_dir = _resolve_model_dir(spec)

        missing = self._missing_files()
        if missing:
            raise FileNotFoundError(
                "Qwen-Image-2.1 模型包不完整（{}）：缺 {}".format(
                    self.model_dir, ", ".join(missing),
                )
            )

        self.sink = _ProgressSink()
        self.last_log: str = ""
        self.last_artifacts: Dict[str, str] = {}
        self._workdir: Optional[Path] = None
        self._owns_workdir = False
        self._keep_artifacts = False
        self._shim = None          # 懒加载（init 要 90 s，不能在构造时做）

    # ------------------------------------------------------------------ #
    # cshim 会话
    # ------------------------------------------------------------------ #
    def _ensure_shim(self, opts: Dict[str, Any]):
        """打开（或复用）cshim 会话。

        ⭐ 会话里 TE + DiT 常驻 ⇒ 批量出图时只 init 一次。
           ⚠️ 但 **DiT 与 VAE 不能共卡** ⇒ 每批图结束后 DiT 被释放，
             下一批要重新 init（39 s）—— 这是物理限制，不是实现问题。
        """
        if self._shim is not None:
            return self._shim
        from .._qwen_image_c import QwenImageShim

        self._ensure_proxy()
        self.sink.emit("[Qwen-Image-2.1] 加载模型（TE + DiT 常驻，约 90 s）…")
        t0 = time.perf_counter()
        self._shim = QwenImageShim(
            self.model_dir,
            card_te=str(opts["card0"]).encode(),
            card_dit=str(opts["card1"]).encode(),
            verbose=bool(opts.get("verbose", True)),
            progress=self._on_cshim_progress,
        )
        _log("  cshim 就绪（{:.1f} s）".format(time.perf_counter() - t0))
        return self._shim

    # ------------------------------------------------------------------ #
    # 进度（cshim → sink → on_delta → 上层）
    # ------------------------------------------------------------------ #
    def _on_cshim_progress(
        self, stage: str, step: int, total: int, elapsed_ms: float,
    ) -> None:
        """把 cshim 的进度回调转成**结构化消息**发给上层。

        消息格式（``qi|`` 前缀 ⇒ 与人类可读消息区分得开）::

            qi|stage=flow|step=3|total=8|elapsed_ms=2601.5

        字段：
        * ``stage``   —— init / te / flow / vae / vae_init / release / done
        * ``step``    —— 该阶段内的 1-based 步序
        * ``total``   —— 该阶段的总步数
        * ``elapsed_ms`` —— **这一步**花了多久（距上一次回调）

        上层（Web UI 卡片）据此画进度条、显示「每一步在做什么 / 这一步多久 /
        还剩多久」。
        """
        self.sink.emit(
            "qi|stage={}|step={}|total={}|elapsed_ms={:.1f}".format(
                stage, int(step), int(total), float(elapsed_ms)
            )
        )

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    def _missing_files(self) -> List[str]:
        d = self.model_dir
        wanted = [
            _TE_RKNN, _TE_WEIGHT, _TE_TOKENIZER, _TE_EMBED,
            _HEAD_MODEL, _HEAD_WEIGHT_NAME,
            _INT_LEDGER_NAME,
        ]
        for i in range(_N_SEG):
            wanted.append(_DIT_MODEL_TPL.format(i=i))
            wanted.append(_DIT_WEIGHT_TPL.format(i=i))
            wanted.append(f"vae_qwenimage21-512.seg{i}.1828.rknn")
            wanted.append(f"vae_qwenimage21-512.seg{i}.weight")
        return [w for w in wanted if not (d / w).is_file()]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def warmup(self) -> None:
        """把 cshim 会话拉起来（TE + DiT 常驻，约 90 s）。

        ⭐ 为什么要单独一个钩子：cshim 是**懒加载**的（构造 backend 不碰
        NPU），于是第一次推理要付「90 s init + 34 s 生成」= 124 s，
        **超过 webui 的 120 s RPC 推理超时** ⇒ 第一次出图必然超时，
        而超时之后 worker 还在跑，用户看到的是「一直转圈然后报错」。

        把 init 挪到 session 构造期（worker 的 ``load`` 阶段，UI 显示
        「加载中」）⇒ 推理阶段只剩 34 s。

        幂等：``_ensure_shim`` 内部判空，重复调用不会重复 init。
        """
        self._ensure_shim(self._options({}))

    def run_text(self, prompt: str, **overrides: Any) -> Tuple[str, List[Attachment]]:
        """Generate one image from ``prompt``.

        Returns ``(prompt, [ImageAttachment(bgr_ndarray)])``.
        """
        return self.run_texts([prompt], **overrides)[0]

    def run_texts(
        self, prompts: List[str], **overrides: Any,
    ) -> List[Tuple[str, List[Attachment]]]:
        """Generate **N** images from ``prompts`` — **一个进程跑完**。

        ⭐ 2026-11-02 第三十五轮：批量不是「循环调 N 次 run_text」——
        那样每张图都要重新 init 一次 DiT（39 s）与 TE（37 s），完全白费常驻。
        真正的批量是：**一次调用跑完 N 张** ⇒ TE / DiT / VAE 都只 init 一次。

        ⭐ 2026-11-03 第三十六轮：不再起 C++ 运行时子进程 ⇒ 改成**进程内调
        wdnai 自己的 cshim**（``wdnai/_cshim/libqi_qwen.so``）⇒
        发布包里**不需要任何闭源 bin / so**。

        Returns a list of ``(prompt, [ImageAttachment(bgr_ndarray)])``,
        one entry per prompt, in order.
        """
        cleaned = [(p or "").strip() for p in (prompts or [])]
        if not cleaned or any(not p for p in cleaned):
            raise ValueError("Qwen-Image-2.1 需要一个非空的提示词")

        opts = self._options(overrides)
        work = self._make_workdir(opts)
        self._keep_artifacts = bool(opts.get("keep_artifacts"))
        out_dir = work / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        self.sink.reset()
        t_start = time.perf_counter()
        try:
            # ① 会话（TE + DiT 常驻）② 批量出图（TE → flow-match → VAE）
            shim = self._ensure_shim(opts)
            self.sink.emit("[Qwen-Image-2.1] 生成中 …")
            t0 = time.perf_counter()
            images = shim.generate(
                cleaned,
                steps=int(opts["steps"]),
                mu=float(opts["fm_mu"]),
                shift_terminal=float(opts.get("shift_terminal", 0.02)),
                noise="fresh" if opts.get("noise_fresh", True) else "fixed",
                seed=int(opts.get("noise_seed", 0) or 0),
            )
            gen_ms = (time.perf_counter() - t0) * 1000.0
            te_ms = 0.0    # TE 已含在生成里（常驻，0.16 s/张）

            results: List[Tuple[str, List[Attachment]]] = []
            copy_dir = opts.get("output_dir")
            first_png: Optional[Path] = None
            for i, (prompt, rgb) in enumerate(zip(cleaned, images)):
                png_path = out_dir / "img{:02d}.png".format(i + 1)
                _write_png(png_path, rgb)
                if first_png is None:
                    first_png = png_path
                bgr = np.ascontiguousarray(rgb[:, :, ::-1])
                results.append((prompt, [ImageAttachment(bgr)]))
                if copy_dir:
                    # ⭐ 把**后端自己写的** PNG 复制出去（不是重新编码）
                    #   ⇒ 与基线 PNG **逐字节相同**
                    dst_dir = Path(str(copy_dir))
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(str(png_path), str(dst_dir / png_path.name))
                if opts.get("verbose", True):
                    _log("  {} → {}".format(png_path.name, png_path))
            total_ms = (time.perf_counter() - t_start) * 1000.0
            # 收尾阶段（写 PNG / 复制产物）—— 让进度条有个明确的末态
            self.sink.emit("qi|stage=save|step=1|total=1|elapsed_ms=0.0")
            self.sink.perf.update({
                "te_ms": te_ms,
                "generate_ms": gen_ms,
                "total_ms": total_ms,
                "steps": int(opts["steps"]),
                "count": len(cleaned),
            })
            self.last_artifacts = {
                "png": str(first_png or ""),
                "out_dir": str(out_dir),
            }
            self.sink.emit(
                "[Qwen-Image-2.1] 完成：{} 张（{:.1f} s）".format(
                    len(cleaned), total_ms / 1000.0,
                )
            )
            if opts.get("verbose", True):
                _log(
                    "完成：{} 张  生成 {:.1f}s  总计 {:.1f}s".format(
                        len(cleaned), gen_ms / 1000.0, total_ms / 1000.0,
                    )
                )
            return results
        finally:
            if self._owns_workdir and not opts.get("keep_artifacts"):
                shutil.rmtree(work, ignore_errors=True)

    def run_images(self, prompt: str, images: list, **overrides: Any) -> str:
        """Not supported — Qwen-Image-2.1 is a text-to-image model."""
        raise NotImplementedError(
            "Qwen-Image-2.1 是文生图模型，不接受图像输入（img2img 未实现）"
        )

    def close(self) -> None:
        """Release the cshim session and drop the scratch directory.

        The scratch directory survives when the run asked for
        ``keep_artifacts=True`` — the PNGs are useful for numerical
        comparison against a PyTorch golden run.
        """
        shim = getattr(self, "_shim", None)
        if shim is not None:
            try:
                shim.close()
            except Exception:  # noqa: BLE001
                pass
            self._shim = None
        if self._workdir is not None and self._owns_workdir and not self._keep_artifacts:
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _options(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        spec_opts = getattr(self.spec, "options", {}) or {}
        merged: Dict[str, Any] = {
            "steps": 8,
            "fm_mu": _FM_MU_512,
            "shift_terminal": 0.02,
            "noise_fresh": True,
            "noise_seed": 0,
            "card0": _CARD0,
            "card1": _CARD1,
            "output_dir": None,
            "work_dir": None,
            "keep_artifacts": False,
            "verbose": bool(getattr(self.spec, "verbose", True)),
            "timeout_te": 600,
            "timeout_gen": 1800,
        }
        for key, value in spec_opts.items():
            merged[key] = value
        for key, value in overrides.items():
            if value is not None and key in merged:
                merged[key] = value
        merged["steps"] = max(1, int(merged["steps"]))
        return merged

    def _make_workdir(self, opts: Dict[str, Any]) -> Path:
        if opts.get("work_dir"):
            work = Path(str(opts["work_dir"]))
            work.mkdir(parents=True, exist_ok=True)
            self._workdir = work
            self._owns_workdir = False
            return work
        self._workdir = Path(tempfile.mkdtemp(prefix="wdnai_qi21_"))
        self._owns_workdir = True
        return self._workdir

    def _ensure_proxy(self) -> None:
        """Make sure a transfer proxy is running (never restart one).

        The proxy is a long-lived daemon, so it must be started with
        :class:`subprocess.Popen` + ``start_new_session`` — a plain
        ``subprocess.run`` would block until the daemon exits.
        """
        if self._proxy_alive():
            return
        _log("transfer proxy 未运行 ⇒ 启动一个")
        for cmd in (
            ["sudo", "-n", "-E", "/bin/rknn3_transfer_proxy"],
            ["sudo", "-E", "/bin/rknn3_transfer_proxy"],
            ["/bin/rknn3_transfer_proxy"],
        ):
            try:
                subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL, start_new_session=True,
                )
            except (OSError, subprocess.SubprocessError):
                continue
            for _ in range(15):          # 最多等 30 s
                if self._proxy_alive():
                    _log("proxy 已就绪")
                    return
                time.sleep(2)
        _log("⚠️ 没能确认 transfer proxy 已启动（继续尝试，可能仍能跑通）")

    @staticmethod
    def _proxy_alive() -> bool:
        try:
            out = subprocess.run(
                ["ps", "-eo", "comm"], capture_output=True, text=True, timeout=10,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return False
        return any(c.startswith("rknn3_transfer_") for c in out.split())


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #
def _resolve_model_dir(spec) -> Path:
    """Locate the Qwen-Image-2.1 model package.

    Order: an explicit ``spec.image_model_dir`` (strict — a wrong path is
    an error, not a reason to silently use another one) → env override →
    ``$WDNAI_MODELS_DIR/qwenimage21`` → the well-known board locations.
    """
    explicit = str(getattr(spec, "image_model_dir", "") or "")
    if explicit:
        p = Path(explicit)
        if not p.is_dir():
            raise FileNotFoundError(f"Qwen-Image-2.1 模型目录不存在：{p}")
        return p

    candidates: List[Path] = []
    for env_key in ("WDNAI_QWEN_IMAGE_21_DIR", "QI_MODEL_DIR"):
        value = os.environ.get(env_key)
        if value:
            candidates.append(Path(value))
    models_root = os.environ.get("WDNAI_MODELS_DIR")
    if models_root:
        candidates.append(Path(models_root) / "qwenimage21")
    candidates.extend(Path(p) for p in _MODEL_DIR_CANDIDATES)

    for cand in candidates:
        if cand.is_dir():
            return cand
    return candidates[-1]






