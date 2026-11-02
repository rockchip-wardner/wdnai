"""Text-to-image backend — Qwen-Image-2.1 on RKNN3 (rk1820).

Qwen-Image-2.1 is a three-part diffusion model:

===========  =================================  ==================================
part         role                              RKNN artefacts
===========  =================================  ==================================
TE           prompt → ``encoder_hidden_states``  ``llm_qwenimage21-te.*``
DiT          flow-match *N* steps → latents      ``dit_qwenimage21-512hc.seg{0..3}.*``
VAE          latents → 512×512 image            ``vae_qwenimage21-512.seg{0..3}.*``
===========  =================================  ==================================

Why a subprocess instead of ``librknn3`` ctypes?
------------------------------------------------
The three-part orchestration on rk1820 is **not** a straight port of the
upstream pipeline. Every one of the rules below was measured on real
hardware and is encoded in the C++ runtime ``qwen_image_rknn``:

* the DiT is cut into 4 × 8-block segments; each segment used to carry
  its own copy of the transformer *head* (``timestep_embedder`` +
  ``modulation``), so the head was **cut out** into a standalone model
  (zero numerical loss) and the 4 segments then fit on **one** card
  with ``--share-internal``;
* the head model **must** live on a different card than the segments;
* the VAE is a 4-segment chain that must run **streaming** — all four
  segments resident at once dies with ``MODEL_SETUP fail``;
* the VAE's per-segment IO has to satisfy
  ``core0 internal peak + that segment's IO ≤ 639.5 MiB``, because the
  runtime pins every tensor of a segment to ``slot.attr.core_id``;
* TE **can** now co-reside with the head (486.0 + 6.4 = 492.4 MiB/core)
  ⇒ it stays resident in the same process and encodes every prompt in
  ~0.16 s instead of paying a 36.8 s ``model_init`` per prompt.

Re-deriving all of that in Python would duplicate a lot of subtle,
hard-won knowledge for no benefit, so this backend **drives that binary**
as a subprocess and reads the resulting image back — the same shape as
:mod:`wdnai.backends.tts` shelling out to ``aplay`` for playback.

The consequence: this module contains **no** ``librknn3`` import at all,
so importing it on a non-RK host is harmless.

Card layout (2026-11-02 第三十五轮)::

    card 0 : TE 常驻 (486.0 MiB/core) + DiT 头 (6.4 MiB/core) = 492.4
    card 1 : DiT 4 段 (525.0 MiB/core, --share-internal)
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
    reference_512/hidden_states.bin      ← 初始噪声
    runtime/qwen_image_rknn               ← C++ 运行时（+ runtime/lib/aarch64）
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

from ..attachment import Attachment, ImageAttachment

__all__ = ["ImageBackend"]


# --------------------------------------------------------------------------- #
# Model-package layout (mirrors /home/rockchip/models/qwenimage21)
# --------------------------------------------------------------------------- #
_TE_RKNN = "llm_qwenimage21-te.1828.rknn"
_TE_WEIGHT = "llm_qwenimage21-te.weight"
_TE_TOKENIZER = "qwenimage21-te.tokenizer.gguf"
_TE_EMBED = "qwenimage21-te.embed.bin"
_REF_DIR = "reference_512"
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

#: ⚠️ 2026-11-02 第三十五轮：**不再有 deploy 目录兜底**。
#:
#: 之前 ``_DEPLOY_ROOTS`` 会在模型包没带 ``runtime/`` 时去
#: ``/home/rockchip/projects/deploy/bin/`` 找 —— 而那个目录里**曾经**存着
#: **另一份** ``qwen_image_rknn`` ⇒ 一份更新一份没更新 ⇒ 行为分裂
#: ⇒ 就是「模型包里的二进制静默过期」那个坑的根源。
#:
#: ⇒ 现在**运行时只有一份**（模型包的 ``runtime/``），由 ``deploy/build.sh``
#: 编译完自动同步 ⇒ 不可能再分叉。
_DEPLOY_ROOTS: Tuple[str, ...] = ()

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
    NPU work happens until :meth:`run_text` is called, so building a
    session is cheap and does not touch the cards.

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
        self.binary = _resolve_binary(self.model_dir)
        self.lib_dir = _resolve_lib_dir(self.binary)

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
        self._proc: Optional[subprocess.Popen] = None
        self._workdir: Optional[Path] = None
        self._owns_workdir = False
        self._keep_artifacts = False

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    def _missing_files(self) -> List[str]:
        d = self.model_dir
        wanted = [
            _TE_RKNN, _TE_WEIGHT, _TE_TOKENIZER, _TE_EMBED,
            str(Path(_REF_DIR) / "hidden_states.bin"),
            _HEAD_MODEL, _HEAD_WEIGHT_NAME,
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
        真正的批量是：给每个 prompt 一个 te 目录（放 ``prompt.txt``），
        一次性交给 ``serve --te-dirs d1,d2,...`` ⇒ **TE / DiT / VAE 都只 init 一次**。

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
            # ① TE — 每个 prompt 一个目录（放 prompt.txt），由 serve 里的
            #    **常驻 TE** 编码（TE 与 DiT 的头共卡 0 ⇒ 不再单独起进程、
            #    不再每张图付 36.8 s 的 model_init）
            self.sink.emit("[Qwen-Image-2.1] 编码提示词 …")
            t0 = time.perf_counter()
            te_dirs = self._run_te(cleaned, work, opts)
            te_ms = (time.perf_counter() - t0) * 1000.0

            # ② DiT (resident, zero load/init during generation) + ③ VAE
            self.sink.emit("[Qwen-Image-2.1] 生成中 …")
            t0 = time.perf_counter()
            self._run_serve(te_dirs, out_dir, opts)
            gen_ms = (time.perf_counter() - t0) * 1000.0

            results: List[Tuple[str, List[Attachment]]] = []
            copy_dir = opts.get("output_dir")
            first_png: Optional[Path] = None
            for i, prompt in enumerate(cleaned):
                bgr, png_path = self._load_image(out_dir, i + 1)
                if first_png is None:
                    first_png = png_path
                results.append((prompt, [ImageAttachment(bgr)]))
                if copy_dir:
                    # ⭐ 把**后端自己写的** PNG 复制出去（不是重新编码）
                    #   ⇒ 与运行时写的那份**逐字节相同**（同一套 zlib/filter 规则）
                    dst_dir = Path(str(copy_dir))
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(str(png_path), str(dst_dir / png_path.name))
                if opts.get("verbose", True):
                    _log("  {} → {}".format(png_path.name, png_path))
            total_ms = (time.perf_counter() - t_start) * 1000.0
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
                "te_dir": str(te_dirs[0]),
            }
            self.sink.emit(
                "[Qwen-Image-2.1] 完成：{} 张（{:.1f} s）".format(
                    len(cleaned), total_ms / 1000.0,
                )
            )
            if opts.get("verbose", True):
                _log(
                    "完成：{} 张  TE {:.1f}s  生成 {:.1f}s  总计 {:.1f}s".format(
                        len(cleaned), te_ms / 1000.0, gen_ms / 1000.0,
                        total_ms / 1000.0,
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
        """Abort any running generation and drop the scratch directory.

        The scratch directory survives when the run asked for
        ``keep_artifacts=True`` — the latents / raw VAE tensor are useful
        for numerical comparison against a PyTorch golden run.
        """
        self._abort()
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

    # -- environment ---------------------------------------------------- #
    def _subprocess_env(self) -> Dict[str, str]:
        env = dict(os.environ)
        lib = str(self.lib_dir)
        env["LD_LIBRARY_PATH"] = (
            f"{lib}:{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else lib
        )
        # ⛔ QI_SPREAD_IO must stay 0: the runtime only writes results into
        #    the pool it allocated them in — spreading the *output* IO makes
        #    every output read back as zeros.
        # ✅ QI_SPREAD_IO_IN=1 spreads the *input* IO only, which is what
        #    makes the VAE's third segment fit in the core0 pool.
        env["QI_SPREAD_IO"] = "0"
        env["QI_SPREAD_IO_IN"] = "1"
        env.setdefault("RKNN3_NETWORK_SOCKET_FILE", "/tmp/rk-mdns.ini")
        # ⭐⭐⭐ 切头变体：跳过「探针」。
        #   --share-internal 必须先探出每段的 internal 需求才能算共享块，
        #   而**探针 = 又一次完整的 load + model_init**（4 段 ≈ 25.4 s）
        #   ⇒ 模型被 init 了**两遍** ⇒ 常驻 init 51.0 s。
        #   喂一份外置账本（QI_INT_PER_CORE）⇒ init 降到 24.9 s，**数值逐字节不变**。
        #   ⚠️ 账本跟这批 .rknn 绑定（换模型必须重测），偏小会让 runtime 写越界
        #   ⇒ 只认**随模型包发布**的账本文件，缺失就退回探针（慢但正确）。
        if not env.get("QI_INT_PER_CORE"):
            ledger = self.model_dir / _INT_LEDGER_NAME
            if ledger.is_file():
                try:
                    env["QI_INT_PER_CORE"] = ledger.read_text().strip().replace(" ", "")
                except OSError:
                    pass
        return env

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

    # -- weight paths ------------------------------------------------------ #
    def _weight_paths(self) -> Tuple[List[str], List[str]]:
        """Return the DiT / VAE weight paths to hand to the runtime.

        ⚠️ 2026-11-02 第三十五轮：**直接用模型包里的 .weight**（不再暂存到 tmpfs）。

        实测（板端）：NVMe 直读与 tmpfs 的 ``load``（权重登记）时间一样
        （17~40 ms），唯一代价是 ``model_init`` 慢 ~3.6 s/段
        （DiT 4 段 ≈ +14 s，**一次性**）。

        ⇒ 因此彻底去掉 tmpfs 那一整套（暂存 / 互斥清理 / 大小校验），
          也顺带消灭了「两套变体同名权重错配」那个雷（变体现在只有一套）。
        """
        dit: List[str] = []
        vae: List[str] = []
        for i in range(_N_SEG):
            dit.append(str(self.model_dir / _DIT_WEIGHT_TPL.format(i=i)))
            vae.append(str(self.model_dir / f"vae_qwenimage21-512.seg{i}.weight"))
        return dit, vae

    # -- the two runtime invocations ------------------------------------- #
    def _run_te(
        self, prompts: List[str], work: Path, opts: Dict[str, Any],
    ) -> List[Path]:
        # ⭐⭐⭐ 2026-11-02 第三十五轮：**TE 常驻**。
        #   TE 的 model_init 要 36.8 s、真推理只要 0.16 s ⇒ 不再单独起 te-plain 进程，
        #   改成「把 prompt 写进目录，serve 里的常驻 TE 负责编码」。
        #   ⇒ 每张图省掉 37 s（批量 N 张时省 N×37 s）。
        # ⭐ 批量：每个 prompt 一个目录 ⇒ serve 一次跑完 ⇒ TE/DiT/VAE 都只 init 一次。
        te_dirs: List[Path] = []
        for i, prompt in enumerate(prompts):
            te_dir = work / "te{}".format(i + 1)
            te_dir.mkdir(parents=True, exist_ok=True)
            (te_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
            te_dirs.append(te_dir)
        return te_dirs

    def _run_serve(
        self, te_dirs: List[Path], out_dir: Path, opts: Dict[str, Any],
    ) -> None:
        d = self.model_dir
        dit_w, vae_w = self._weight_paths()

        dit_models = [str(d / _DIT_MODEL_TPL.format(i=i)) for i in range(_N_SEG)]
        vae_models = [str(d / f"vae_qwenimage21-512.seg{i}.1828.rknn") for i in range(_N_SEG)]
        # ⭐⭐⭐ 2026-11-02 第三十五轮的新布局：
        #   卡 0 : TE 常驻（486.0 MiB/核）+ 头模型（6.4 MiB/核）= 492.4 ✅
        #   卡 1 : DiT 4 段（切头单卡，--share-internal）           = 525.0 ✅
        #          → N 张图跑完 → 释放 → VAE 4 段流式（峰值 516.5）✅
        # ⚠️ 头**必须与段不同卡** ⇒ 头在卡 0、段在卡 1。
        device_list = ",".join([opts["card1"]] * _N_SEG)

        cmd = [
            str(self.binary), "serve",
            "--models", ",".join(dit_models),
            "--weights", ",".join(dit_w),
            "--device-list", device_list,
        ]
        # ⚠️ --share-internal 是单卡的**必要条件**（不加 ⇒ 第 3 段 MODEL_SETUP fail）
        # ⚠️ 头模型**必须**放在与段不同的卡上 ⇒ 头在卡 0、段在卡 1
        cmd += [
            "--share-internal",
            "--head-model", str(d / _HEAD_MODEL),
            "--head-weight", str(d / _HEAD_WEIGHT_NAME),
            "--head-device", str(opts["card0"]),
            # ⭐ TE 常驻（与头共卡 0）：每张图只做一次 0.16 s 的推理
            "--te-resident",
            "--te-model", str(d / _TE_RKNN),
            "--te-weight", str(d / _TE_WEIGHT),
            "--te-tokenizer", str(d / _TE_TOKENIZER),
            "--te-embed", str(d / _TE_EMBED),
            "--te-device", str(opts["card0"]),
            "--te-dirs", ",".join(str(td) for td in te_dirs),
            "--input-dir", str(d / _REF_DIR),
            "--output", str(out_dir),
            "--repeat", "1",
            "--fm-steps", str(opts["steps"]),
            "--fm-mu", str(opts["fm_mu"]),
            "--vae-models", ",".join(vae_models),
            "--vae-weights", ",".join(vae_w),
            "--vae-device", str(opts["card1"]),
            # ⚠️ --vae-stream is mandatory: without it all four VAE segments
            #    stay resident and segment #2 dies with MODEL_SETUP fail
            #    (after the whole DiT run has already been paid for).
            "--vae-stream",
            # ⚠️ TE is gone by now, so the VAE may take the DiT's pool.
            "--vae-release-dit",
            # The SDK writes the PNG itself from the raw tensor, so the
            # runtime does not need python3 + numpy for the conversion.
            "--no-png",
        ]
        if opts.get("noise_fresh", True):
            cmd.append("--noise-fresh")
            seed = int(opts.get("noise_seed", 0) or 0)
            cmd += ["--noise-seed", str(seed)]

        self._run_cmd(cmd, opts, float(opts["timeout_gen"]), "DiT+VAE")

    # -- subprocess driver ------------------------------------------------ #
    def _run_cmd(self, cmd: List[str], opts: Dict[str, Any],
                 timeout: float, label: str) -> None:
        env = self._subprocess_env()
        _log(f"{label}: {' '.join(cmd[1:4])} …")
        try:
            proc = subprocess.Popen(
                cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
        except OSError as exc:
            raise RuntimeError(f"无法启动 {self.binary}: {exc}") from exc

        self._proc = proc
        lines: List[str] = []

        def _reader() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                lines.append(line)

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()

        deadline = time.monotonic() + timeout
        shown = 0
        try:
            while proc.poll() is None:
                if self.sink.stopped:
                    self._kill(proc)
                    raise RuntimeError("生成被取消（session.stop()）")
                if time.monotonic() > deadline:
                    self._kill(proc)
                    raise TimeoutError(f"{label} 超时（{timeout:.0f} s）")
                while shown < len(lines):
                    text = lines[shown].rstrip()
                    shown += 1
                    if text:
                        self.last_log += text + "\n"
                        if opts.get("verbose", True):
                            print(f"[wdnai.image] {text}", flush=True)
                time.sleep(0.2)
        finally:
            self._proc = None
            thread.join(timeout=5)

        while shown < len(lines):
            text = lines[shown].rstrip()
            shown += 1
            if text:
                self.last_log += text + "\n"
                if opts.get("verbose", True):
                    print(f"[wdnai.image] {text}", flush=True)

        if proc.returncode != 0:
            tail = "\n".join(lines[-25:])
            raise RuntimeError(
                f"{label} 失败（退出码 {proc.returncode}）：\n{tail}"
            )

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        except (OSError, subprocess.SubprocessError):
            pass

    def _abort(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            self._kill(proc)

    # -- image decoding ---------------------------------------------------- #
    def _load_image(self, out_dir: Path, index: int = 1) -> Tuple[Any, Path]:
        """Decode the VAE output into a BGR ndarray and write the PNG.

        The runtime writes ``img01_vae_image.bin`` (+ ``.meta``) as
        NCHW ``[1, 4, 512, 512]`` fp16. Channels 0/1/2 are RGB; the 4th
        channel is not part of the visible image.

        ``index`` is 1-based and selects ``img<NN>`` — the runtime numbers
        its per-image outputs that way, so a batch of N lands as
        ``img01`` … ``imgNN``.
        """
        import numpy as np

        stem = "img{:02d}".format(int(index))
        raw = out_dir / (stem + "_vae_image.bin")
        png_path = out_dir / (stem + ".png")
        if raw.is_file():
            dtype_name, shape = _read_meta(raw)
            dtype = _DTYPES.get(dtype_name or "", "float16")
            data = np.fromfile(str(raw), dtype=np.dtype(dtype)).astype(np.float32)
            if shape and len(shape) >= 3:
                total_c = int(shape[1])
                h, w = int(shape[2]), int(shape[3])
            else:
                total_c = 4
                side = int(round((data.size / total_c) ** 0.5))
                h = w = side
            if data.size != total_c * h * w:
                total_c = 4
                side = int(round((data.size / total_c) ** 0.5))
                h = w = side
            img = data.reshape(total_c, h, w)
            rgb = np.stack([img[0], img[1], img[2]], axis=-1)
            if float(rgb.min()) < -0.01:
                rgb = (rgb + 1.0) * 127.5
            else:
                rgb = rgb * 255.0
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            _write_png(png_path, rgb)
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
            return bgr, png_path

        # Fallback: the runtime already wrote a PNG (e.g. someone ran the
        # binary without --no-png).
        if png_path.is_file():
            try:
                import cv2

                bgr = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
                if bgr is not None:
                    return bgr, png_path
            except Exception:  # noqa: BLE001
                pass
        raise RuntimeError(
            f"没有找到出图文件：{raw} / {png_path}（{out_dir}）"
        )


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


def _resolve_binary(model_dir: Path) -> Path:
    """Locate the ``qwen_image_rknn`` runtime binary."""
    candidates: List[Path] = []
    env_bin = os.environ.get("WDNAI_QWEN_IMAGE_BIN")
    if env_bin:
        candidates.append(Path(env_bin))
    candidates.append(model_dir / "runtime" / "qwen_image_rknn")
    env_root = os.environ.get("WDNAI_QWEN_IMAGE_ROOT")
    if env_root:
        candidates.append(Path(env_root) / "bin" / "qwen_image_rknn")
    for root in _DEPLOY_ROOTS:
        candidates.append(Path(root) / "bin" / "qwen_image_rknn")

    for cand in candidates:
        if cand.is_file() and os.access(str(cand), os.X_OK):
            _check_runtime_flags(cand)
            return cand
    raise FileNotFoundError(
        "找不到 qwen_image_rknn 运行时（试过：{}）".format(
            ", ".join(str(c) for c in candidates)
        )
    )


#: 运行时必须认识的参数。少一个就说明二进制**过期**了。
#:
#: ⛔ 2026-11-02 第三十五轮踩到：模型包 ``runtime/`` 里那份二进制是旧的
#: （没有 ``--te-resident``）⇒ 它会**默默忽略**新参数 ⇒ 走回「读 te 目录里
#: 现成的 encoder_hidden_states.bin」的老路 ⇒ 报「TE 目录 xxx 里缺
#: encoder_hidden_states.bin」，**白等 79 s**。
_REQUIRED_RUNTIME_FLAGS = ("te-resident", "share-internal", "head-model", "vae-stream")


def _check_runtime_flags(binary: Path) -> None:
    """Raise :class:`RuntimeError` if ``binary`` predates a required flag."""
    try:
        blob = binary.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"读不了运行时 {binary}: {exc}") from exc
    missing = [f for f in _REQUIRED_RUNTIME_FLAGS if f.encode() not in blob]
    if missing:
        raise RuntimeError(
            "运行时太旧：{} 不支持 --{}。"
            "（模型包 runtime/ 里的副本可能是旧的）"
            "⇒ 重新 make，并把 bin/qwen_image_rknn 同步到模型包的 runtime/".format(
                binary, ", ".join(missing),
            )
        )


def _resolve_lib_dir(binary: Path) -> Path:
    """Locate the directory holding the ``librknn3_api*`` shared objects."""
    candidates = [
        binary.parent / "lib" / "aarch64",
        binary.parent.parent / "lib" / "aarch64",
    ]
    for root in _DEPLOY_ROOTS:
        candidates.append(Path(root) / "lib" / "aarch64")
    for cand in candidates:
        if cand.is_dir():
            return cand
    return candidates[-1]
