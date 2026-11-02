"""Text-to-image backend — Qwen-Image-2.1 on RKNN3 (rk1820).

Qwen-Image-2.1 is a three-part diffusion model:

===========  =================================  ==================================
part         role                              RKNN artefacts
===========  =================================  ==================================
TE           prompt → ``encoder_hidden_states``  ``llm_qwenimage21-te.*``
DiT          flow-match *N* steps → latents      ``dit_qwenimage21-512.seg{0..3}.*``
VAE          latents → 512×512 image            ``vae_qwenimage21-512.seg{0..3}.*``
===========  =================================  ==================================

Why a subprocess instead of ``librknn3`` ctypes?
------------------------------------------------
The three-part orchestration on rk1820 is **not** a straight port of the
upstream pipeline. Every one of the rules below was measured on real
hardware and is encoded in the C++ runtime ``qwen_image_rknn``:

* the DiT is cut into 4 × 8-block segments that must sit **two per card**
  (four on one card does not fit into the 639.5 MiB/core pool);
* the VAE is a 4-segment chain that must run **streaming** — all four
  segments resident at once dies with ``MODEL_SETUP fail``;
* the VAE's per-segment IO has to satisfy
  ``core0 internal peak + that segment's IO ≤ 639.5 MiB``, because the
  runtime pins every tensor of a segment to ``slot.attr.core_id``;
* TE cannot co-reside with DiT: its weights alone are 486 MiB/core, so
  TE has to run in its own process and release the pool afterwards.

Re-deriving all of that in Python would duplicate a lot of subtle,
hard-won knowledge for no benefit, so this backend **drives that binary**
as a subprocess and reads the resulting image back — the same shape as
:mod:`wdnai.backends.tts` shelling out to ``aplay`` for playback.

The consequence: this module contains **no** ``librknn3`` import at all,
so importing it on a non-RK host is harmless.

Layout of the model package (see ``/home/rockchip/models/qwenimage21``)::

    llm_qwenimage21-te.1828.rknn / .weight
    qwenimage21-te.embed.bin / .tokenizer.gguf
    dit_qwenimage21-512.seg{0..3}.1828.rknn / .weight
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

#: tmpfs cache names for the staged weights. These deliberately reuse the
#: names the deploy-side ``test.sh`` already staged — the tmpfs only has
#: ~0.5 GiB free, so re-staging under different names would fail.
_DIT_CACHE = ("m512a.weight", "m512b.weight", "m512c.weight", "m512d.weight")
_VAE_CACHE = (
    "vae512_f0m.weight", "vae512_f1m.weight",
    "vae512_f2m.weight", "vae512_f3m.weight",
)

#: 512×512 uses ``mu = 0.5387096774`` (latent sequence length 1024).
#: The 1024×1024 value (0.6935483871) is **wrong** here.
_FM_MU_512 = "0.5387096774"

#: Default board topology (2 × RK1828 behind the RK3588 host).
_CARD0 = "0000:01:00.0"
_CARD1 = "0003:31:00.0"

#: Where the weights are staged for fast (RAM-speed) loading.
_TMPFS_DIR = "/mnt/qi_w"

#: Fallback deployment roots, searched when the model package does not
#: carry its own ``runtime/`` copy.
_DEPLOY_ROOTS = ("/home/rockchip/projects/deploy",)

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


def _free_bytes(path: Path) -> int:
    """Free bytes on the filesystem holding ``path`` (0 if unknown)."""
    try:
        st = os.statvfs(str(path))
        return int(st.f_bavail) * int(st.f_frsize)
    except (OSError, AttributeError):
        return 0


def _is_tmpfs(path: Path) -> bool:
    """True when ``path`` lives on a tmpfs (RAM) mount."""
    try:
        with open("/proc/mounts", "r") as fh:
            for line in fh:
                fields = line.split()
                if len(fields) < 3:
                    continue
                mount, fstype = fields[1], fields[2]
                if fstype != "tmpfs":
                    continue
                if str(path) == mount or str(path).startswith(mount.rstrip("/") + "/"):
                    return True
    except OSError:
        pass
    return False


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
        ]
        for i in range(_N_SEG):
            wanted.append(f"dit_qwenimage21-512.seg{i}.1828.rknn")
            wanted.append(f"dit_qwenimage21-512.seg{i}.weight")
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
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("Qwen-Image-2.1 需要一个非空的提示词")

        opts = self._options(overrides)
        work = self._make_workdir(opts)
        self._keep_artifacts = bool(opts.get("keep_artifacts"))
        te_dir = work / "te"
        out_dir = work / "out"
        te_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        self.sink.reset()
        t_start = time.perf_counter()
        try:
            # ① TE — must own the card pool alone; the process exits and
            #    releases it before the DiT is initialised.
            self.sink.emit("[Qwen-Image-2.1] 编码提示词 …")
            t0 = time.perf_counter()
            self._run_te(prompt, te_dir, opts)
            te_ms = (time.perf_counter() - t0) * 1000.0

            # ② DiT (resident, zero load/init during generation) + ③ VAE
            self.sink.emit("[Qwen-Image-2.1] 生成中 …")
            t0 = time.perf_counter()
            self._run_serve(te_dir, out_dir, opts)
            gen_ms = (time.perf_counter() - t0) * 1000.0

            bgr, png_path = self._load_image(out_dir)
            total_ms = (time.perf_counter() - t_start) * 1000.0
            self.sink.perf.update({
                "te_ms": te_ms,
                "generate_ms": gen_ms,
                "total_ms": total_ms,
                "steps": int(opts["steps"]),
            })
            self.last_artifacts = {
                "png": str(png_path),
                "out_dir": str(out_dir),
                "te_dir": str(te_dir),
            }
            self.sink.emit(
                "[Qwen-Image-2.1] 完成：{}（{:.1f} s）".format(
                    png_path.name, total_ms / 1000.0,
                )
            )
            if opts.get("verbose", True):
                _log(
                    "完成：{}  TE {:.1f}s  生成 {:.1f}s  总计 {:.1f}s".format(
                        png_path, te_ms / 1000.0, gen_ms / 1000.0, total_ms / 1000.0,
                    )
                )
            return prompt, [ImageAttachment(bgr)]
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
            "tmpfs_dir": _TMPFS_DIR,
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

    def _stage_weights(self, opts: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Return the weight paths to hand to the runtime.

        Prefers a tmpfs copy (RAM-speed load) and falls back to the
        on-disk model package when the tmpfs is missing or full.
        """
        tmpfs = Path(str(opts["tmpfs_dir"]))
        use_tmpfs = tmpfs.is_dir() and _is_tmpfs(tmpfs)
        dit: List[str] = []
        vae: List[str] = []
        if not use_tmpfs:
            _log("tmpfs 不可用 ⇒ 直接从模型包读权重（会慢一些）")
            for i in range(_N_SEG):
                dit.append(str(self.model_dir / f"dit_qwenimage21-512.seg{i}.weight"))
                vae.append(str(self.model_dir / f"vae_qwenimage21-512.seg{i}.weight"))
            return dit, vae

        free = _free_bytes(tmpfs)
        for i in range(_N_SEG):
            dit.append(self._stage_one(
                tmpfs, _DIT_CACHE[i], self.model_dir / f"dit_qwenimage21-512.seg{i}.weight", free,
            ))
            vae.append(self._stage_one(
                tmpfs, _VAE_CACHE[i], self.model_dir / f"vae_qwenimage21-512.seg{i}.weight", free,
            ))
        return dit, vae

    def _stage_one(self, tmpfs: Path, name: str, src: Path, free: int) -> str:
        dst = tmpfs / name
        if dst.is_file():
            return str(dst)
        try:
            size = src.stat().st_size
        except OSError:
            return str(src)
        if free < size + (64 << 20):   # 留 64 MiB 余量
            _log(f"tmpfs 空间不足（剩 {free >> 20} MiB）⇒ {name} 直接从模型包读")
            return str(src)
        _log(f"暂存权重到 tmpfs：{name}")
        try:
            shutil.copyfile(str(src), str(dst))
        except OSError as exc:
            _log(f"暂存失败（{exc}）⇒ 直接从模型包读")
            return str(src)
        return str(dst)

    # -- the two runtime invocations ------------------------------------- #
    def _run_te(self, prompt: str, te_dir: Path, opts: Dict[str, Any]) -> None:
        d = self.model_dir
        cmd = [
            str(self.binary), "te-plain",
            "--model", str(d / _TE_RKNN),
            "--weight", str(d / _TE_WEIGHT),
            "--tokenizer", str(d / _TE_TOKENIZER),
            "--embed", str(d / _TE_EMBED),
            "--prompt", prompt,
            "--output", str(te_dir),
        ]
        self._ensure_proxy()
        self._run_cmd(cmd, opts, float(opts["timeout_te"]), "TE")
        if not (te_dir / "encoder_hidden_states.bin").is_file():
            raise RuntimeError("TE 没有产出 encoder_hidden_states.bin")

    def _run_serve(self, te_dir: Path, out_dir: Path, opts: Dict[str, Any]) -> None:
        d = self.model_dir
        dit_w, vae_w = self._stage_weights(opts)

        dit_models = [str(d / f"dit_qwenimage21-512.seg{i}.1828.rknn") for i in range(_N_SEG)]
        vae_models = [str(d / f"vae_qwenimage21-512.seg{i}.1828.rknn") for i in range(_N_SEG)]
        device_list = ",".join([
            opts["card0"], opts["card0"], opts["card1"], opts["card1"],
        ])

        cmd = [
            str(self.binary), "serve",
            "--models", ",".join(dit_models),
            "--weights", ",".join(dit_w),
            "--device-list", device_list,
            "--te-dirs", str(te_dir),
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
    def _load_image(self, out_dir: Path) -> Tuple[Any, Path]:
        """Decode the VAE output into a BGR ndarray and write the PNG.

        The runtime writes ``img01_vae_image.bin`` (+ ``.meta``) as
        NCHW ``[1, 4, 512, 512]`` fp16. Channels 0/1/2 are RGB; the 4th
        channel is not part of the visible image.
        """
        import numpy as np

        raw = out_dir / "img01_vae_image.bin"
        png_path = out_dir / "img01.png"
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
            return cand
    raise FileNotFoundError(
        "找不到 qwen_image_rknn 运行时（试过：{}）".format(
            ", ".join(str(c) for c in candidates)
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
