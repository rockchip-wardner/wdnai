"""Qwen2.5-Omni-3B backend — wav-in, text-out.

Pipeline (mirrors ``rknn3_model_zoo/examples/Qwen2_5_Omni/python_example/
rknn_qwen2_5_omni_audio_demo.py``)::

    raw PCM  →  16 kHz mono fp32  →  log-mel-spectrogram (128 × n_frame)
           →  audio encoder (dynamic shape 0/1) one shot
           →  fp16 audio embeddings  →  RKNN3Audio + ``<audio>`` prompt placeholder
           →  llm.session_run(inputs=[audio_in], prompt=…)
           →  raw text.

The audio encoder is a dynamic-shape RKNN with two pre-exported shapes;
``select_audio_shape`` picks the smallest one whose ``n_frame`` fits, and
``apply_audio_shape`` patches ``rknn3lite`` v1.0.4's stale attr cache so
the next ``run()`` actually sees the new shape.

Unlike ASR, the LLM input is a free-form prompt string (with ``<audio>``
auto-prepended) plus an ``RKNN3Audio`` struct injected via
``session_run(inputs=[…])``. There is no shared internal-memory setup —
the encoder runs once, the LLM runs once.
"""
from __future__ import annotations

import ctypes
import logging
import time as _time
from typing import Any, Dict, Optional, Union

import numpy as np

from ._common import (
    StreamSink,
    apply_chat_template,
    build_llm_args,
    build_llm_callback,
    load_embed_table,
    load_tokenizer,
    printf_perf,
    resolve_device_id,
)
from .asr import (  # noqa: F401  — reused (identical mel formula)
    _hann_window,
    _load_mel_filters,
    _log_mel_spectrogram,
    _pcm_to_mono_float32,
    _reflect_pad,
    _resample_linear,
)

LOG = logging.getLogger("wdnai.backends.omni")

SAMPLE_RATE = 16000
HOP_LENGTH = 160
N_FFT = 400
N_MELS = 128
MELS_FILTERS_COLS = N_FFT // 2 + 1  # 201

AUDIO_SHAPES: Dict[int, Dict[str, int]] = {
    0: {"n_frame": 600, "embeds_dim0": 150},
    1: {"n_frame": 300, "embeds_dim0":  75},
}


def select_audio_shape(audio_length: int) -> int:
    """Pick the smallest shape whose ``n_frame`` fits ``ceil(audio_length / HOP_LENGTH)``."""
    n_frame_ = (audio_length + HOP_LENGTH - 1) // HOP_LENGTH
    for sid in sorted(AUDIO_SHAPES.keys(), reverse=True):
        if n_frame_ <= AUDIO_SHAPES[sid]["n_frame"]:
            return sid
    return 0


def apply_audio_shape(rknn_audio, shape_id: int) -> None:
    """Switch the audio RKNN runtime to ``shape_id`` and patch attr caches.

    The ``rknn3lite`` Python wrapper (v1.0.4) has a bug where ``set_shape()``
    updates ``shape_infos[]`` but NOT the runtime's cached
    ``inputs[i].attr`` / ``outputs[i].attr`` ctypes structs. Without a
    manual patch the next ``rknn3_run()`` fails with::

        inputs[0].attr->dims[2] = 600, but shape_infos[1].input_attrs[0].dims[2] = 300

    This function patches those attrs in-place from the freshly-returned
    ``get_inputs_tensor_attr()`` / ``get_outputs_tensor_attr()`` queries,
    which DO reflect the new shape.
    """
    runtime = rknn_audio.rknn_runtime
    if runtime is None or getattr(runtime, "set_shape", None) is None:
        return
    runtime.set_shape(shape_id)
    in_attrs = runtime.get_inputs_tensor_attr()
    out_attrs = runtime.get_outputs_tensor_attr()
    for i, src in enumerate(in_attrs):
        dst = runtime.inputs[i].attr.contents
        dst.n_dims = src.n_dims
        for d in range(16):
            dst.shape[d] = src.shape[d]
        dst.n_elems = src.n_elems
        dst.n_stride = src.n_stride
        for d in range(17):
            dst.stride[d] = src.stride[d]
        dst.aligned_size = src.aligned_size
        dst.dtype = src.dtype
        dst.layout = src.layout
    for i, src in enumerate(out_attrs):
        dst = runtime.outputs[i].attr.contents
        dst.n_dims = src.n_dims
        for d in range(16):
            dst.shape[d] = src.shape[d]
        dst.n_elems = src.n_elems
        dst.n_stride = src.n_stride
        for d in range(17):
            dst.stride[d] = src.stride[d]
        dst.aligned_size = src.aligned_size
        dst.dtype = src.dtype
        dst.layout = src.layout


def _build_padded_mask(actual_len: int, n_frame: int) -> np.ndarray:
    """``(1, 1, n_frame)`` float16 mask — 1.0 for valid frames else 0.0."""
    mask = np.zeros((1, 1, n_frame), dtype=np.float16)
    valid = min(actual_len, n_frame)
    mask[0, 0, :valid] = 1.0
    return mask


def _build_attention_mask(n_frame: int) -> np.ndarray:
    """``(n_frame / 2, n_frame / 2)`` all-zero float16 mask (matches the C++
    encoder's loop, which overwrites every cell with 0.0 regardless of the
    initial -1e8f)."""
    return np.zeros((n_frame // 2, n_frame // 2), dtype=np.float16)


class OmniBackend:
    """WAV bytes in → free-form text out (Qwen2.5-Omni-3B audio path)."""

    def __init__(
        self,
        spec,
        *,
        npu: Optional[int] = None,
        device_id: Optional[Union[bytes, str]] = None,
    ):
        from rknn3lite.api import RKNN3Lite  # type: ignore

        if not spec.omni_model_dir:
            raise ValueError(
                f"{spec.name!r} requires spec.omni_model_dir; "
                "fill it via .with_paths(omni_model_dir=…)."
            )
        self.spec = spec
        from pathlib import Path
        model_dir = Path(spec.omni_model_dir)

        audio_rknn = model_dir / "Qwen2.5-Omni-3B-audio.1828.rknn"
        audio_wt = model_dir / "Qwen2.5-Omni-3B-audio.weight"
        llm_rknn = model_dir / "Qwen2.5-Omni-3B-llm.1828.rknn"
        llm_wt = model_dir / "Qwen2.5-Omni-3B-llm.weight"
        for p in (
            audio_rknn, audio_wt,
            llm_rknn, llm_wt,
            Path(spec.tokenizer_path), Path(spec.embed_path),
            model_dir / "mel_128_filters.txt",
        ):
            if not p.is_file():
                raise FileNotFoundError(f"Qwen2.5-Omni-3B: missing required file {p}")

        self._mel_filters = _load_mel_filters(
            str(model_dir / "mel_128_filters.txt"), N_MELS, MELS_FILTERS_COLS,
        )

        resolved_device_id = resolve_device_id(RKNN3Lite, npu, device_id)

        self.rknn_audio = RKNN3Lite(verbose=spec.verbose)
        if self.rknn_audio.load_rknn(str(audio_rknn), str(audio_wt)) != 0:
            raise RuntimeError(f"audio load_rknn failed: {audio_rknn}")
        if self.rknn_audio.init_runtime(
            target=spec.target, core_mask=spec.core_mask,
            device_id=resolved_device_id,
        ) != 0:
            raise RuntimeError("audio init_runtime failed")

        in_attr = self.rknn_audio.get_inputs_tensor_attr()[0]
        in_shape = list(in_attr.shape[: in_attr.n_dims])
        LOG.info(
            "audio encoder: in_shape=%s layout=%d dtype=%d n_dims=%d",
            in_shape, in_attr.layout, in_attr.dtype, in_attr.n_dims,
        )
        # Omni's audio encoder is a dynamic-shape export — layout is
        # reported as UNDEFINED (0); we only validate n_dims / n_mels.
        if in_attr.n_dims != 3:
            raise RuntimeError(
                f"audio encoder: expected 3-D input, got n_dims={in_attr.n_dims} "
                f"shape={in_shape}"
            )
        self._n_mels = int(in_shape[1])
        if self._n_mels != N_MELS:
            raise RuntimeError(
                f"audio encoder: expected n_mels={N_MELS}, got {self._n_mels}"
            )

        self.tokenizer = load_tokenizer(spec.tokenizer_path)
        self._ensure_audio_special_tokens()
        self.embeds = load_embed_table(spec.embed_path, spec.vocab_size)

        self.rknn_llm = RKNN3Lite(llm_mode=True, verbose=spec.verbose)
        if self.rknn_llm.load_rknn(str(llm_rknn), str(llm_wt)) != 0:
            raise RuntimeError(f"llm load_rknn failed: {llm_rknn}")

        self.sink = StreamSink(tokenizer=self.tokenizer)
        cb = build_llm_callback(self.tokenizer, self.embeds, self.sink)
        if self.rknn_llm.init_runtime(
            target=spec.target, core_mask=spec.core_mask,
            llm_args=build_llm_args(spec), llm_callback=cb,
            device_id=resolved_device_id,
        ) != 0:
            raise RuntimeError("llm init_runtime failed")
        apply_chat_template(self.rknn_llm)

        self._last_audio_latency: float = 0.0

    def _ensure_audio_special_tokens(self) -> None:
        """Register ``<|audio_bos|>`` / ``<|audio_eos|>`` / ``<|AUDIO|>`` as
        single tokens so RKNN3's ``session_run`` can locate them in the
        tokenised prompt. Without this, the GGUF-fed ByteLevel
        pre-tokenizer splits each marker into ~7 regular tokens and
        ``session_run`` fails with ``Failed to find audio_start_token
        in tokens!``.
        """
        markers = ["<|audio_bos|>", "<|audio_eos|>", "<|AUDIO|>"]
        existing = [m for m in markers if m in self.tokenizer.get_vocab()]
        if not existing:
            return
        added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": existing}
        )
        if added > 0:
            LOG.info(
                "registered %d audio multimodal markers as special tokens: %r",
                added, existing,
            )

    def run_audio(
        self,
        pcm_bytes: bytes,
        sample_rate: int = SAMPLE_RATE,
        n_channels: int = 1,
        sample_width: int = 2,
        prompt: str = "",
        **overrides: Any,
    ) -> str:
        """Run a single audio-understanding pass over a raw PCM payload.

        ``prompt`` is the user instruction; the backend auto-prepends
        ``<audio>`` (no separator — matches the demo's default). Returns
        the model's free-form text response. ``overrides`` are forwarded
        to ``rknn_llm.session_run`` (``max_new_tokens`` etc.).
        """
        from rknn3lite.api.rknn3_types import (  # type: ignore
            Float16, RKNN3Audio,
        )

        self.sink.reset()
        audio = _pcm_to_mono_float32(pcm_bytes, n_channels, sample_width)
        if sample_rate != SAMPLE_RATE:
            audio = _resample_linear(audio, sample_rate, SAMPLE_RATE)
        if len(audio) < N_FFT:
            raise RuntimeError(f"audio too short: {len(audio)} samples (< {N_FFT})")

        shape_id = select_audio_shape(len(audio))
        n_frame = AUDIO_SHAPES[shape_id]["n_frame"]
        embeds_dim0 = AUDIO_SHAPES[shape_id]["embeds_dim0"]
        actual_len = len(audio) // HOP_LENGTH
        LOG.info(
            "selected shape %d: n_frame=%d embeds_dim0=%d actual_len=%d",
            shape_id, n_frame, embeds_dim0, actual_len,
        )

        mel = _log_mel_spectrogram(audio, self._mel_filters)
        if mel.shape[0] != self._n_mels:
            raise RuntimeError(
                f"mel n_mels mismatch: encoder expects {self._n_mels}, got {mel.shape[0]}"
            )
        # Pad/truncate mel to ``n_frame`` columns exactly. The short-audio
        # branch MUST zero-pad — feeding a shorter tensor shifts the
        # encoder's attention window and produces wrong audio embeddings.
        if mel.shape[1] >= n_frame:
            mel_padded = mel[:, :n_frame].astype(np.float16)
        else:
            mel_padded = np.zeros((self._n_mels, n_frame), dtype=np.float16)
            mel_padded[:, : mel.shape[1]] = mel.astype(np.float16)
        mel_batched = mel_padded[np.newaxis, :, :]
        padded_mask = _build_padded_mask(actual_len, n_frame)
        attention_mask = _build_attention_mask(n_frame)[np.newaxis, :, :]

        apply_audio_shape(self.rknn_audio, shape_id)

        t_audio_start = _time.perf_counter()
        outputs = self.rknn_audio.inference(
            inputs=[mel_batched, padded_mask, attention_mask]
        )
        audio_latency = _time.perf_counter() - t_audio_start
        if outputs is None or len(outputs) == 0:
            raise RuntimeError("audio encoder inference failed")
        audio_embeds = np.asarray(outputs[0], dtype=np.float16)
        if audio_embeds.ndim == 2:
            audio_embeds = audio_embeds[np.newaxis, ...]
        LOG.info(
            "audio_embeds %s latency=%.2f ms",
            audio_embeds.shape, audio_latency * 1000.0,
        )
        self._last_audio_latency = audio_latency

        audio_in = RKNN3Audio()
        audio_in.audio_embed = audio_embeds.ctypes.data_as(ctypes.POINTER(Float16))
        audio_in.n_audio_tokens = audio_embeds.shape[1]
        audio_in.n_audio = 1
        audio_in.audio_start = b"<|audio_bos|>"
        audio_in.audio_end = b"<|audio_eos|>"
        audio_in.audio_content = b"<|AUDIO|>"

        full_prompt = "<audio>" + (prompt or "")

        kwargs: Dict[str, Any] = {
            "inputs": [audio_in], "prompt": full_prompt, "keep_history": 0,
        }
        if overrides.get("max_new_tokens") is not None:
            kwargs["max_new_tokens"] = overrides["max_new_tokens"]

        ret, stats = self.rknn_llm.session_run(**kwargs)
        if ret != 0:
            raise RuntimeError(f"llm session_run failed: ret={ret}")
        n_decode, n_prefill, llm_start, llm_end = stats

        if self.sink.first_token_time is not None:
            self.sink.perf.update({
                "n_decode_tokens": n_decode,
                "n_prefill_tokens": n_prefill,
                "prefill_ms": (self.sink.first_token_time - llm_start) * 1000.0,
                "decode_ms": (llm_end - self.sink.first_token_time) * 1000.0,
            })
        if overrides.get("perf", False):
            printf_perf(self.sink, vision_latency=audio_latency)
        return self.sink.text

    def run_text(self, prompt: str, **kw: Any) -> str:
        raise RuntimeError(
            "OmniBackend.run_text(): pass Attachment(wav_bytes, kind='audio') "
            "to LLMSession.respond() and the SDK will route to run_audio()."
        )

    def run_images(self, prompt: str, images: list, **kw: Any) -> str:
        raise RuntimeError("Omni backend does not support images.")

    def run_video(self, prompt: str, frames: list, **kw: Any) -> str:
        raise RuntimeError("Omni backend does not support video.")

    def close(self) -> None:
        for r in (getattr(self, "rknn_audio", None),
                  getattr(self, "rknn_llm", None)):
            if r is None:
                continue
            try:
                r.release()
            except Exception:  # noqa: BLE001
                pass
