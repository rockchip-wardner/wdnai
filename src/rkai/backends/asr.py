"""Qwen3-ASR backend — wav-in, text-out.

Pipeline (mirrors ``examples/Qwen3_ASR/cpp``):

    raw PCM (any rate / s16le / s24le / s32le / IEEE float)
        → 16 kHz mono fp32 → log-mel-spectrogram (pure numpy)
        → audio encoder (NCHW, ``encoder.1828.rknn``) chunk-by-chunk
        → fp16 audio embeddings
        → concat with hardcoded ``prefix`` (9 tokens) + ``postfix`` (6 tokens)
        → ``llm.session_run(embeds=…)`` with ``keep_history=0``
        → raw text, with the language-separator ``[PAD151704]`` replaced by ``\n``.

``run_audio`` returns the full transcript string
(``"language <lang>\\n<text>"``). The example in ``examples/audio_to_text.py``
splits on the first ``\n``.

Supported model layout (matches :class:`rkai.model.SystemLanguageModel.qwen3_asr`):

    <model_dir>/
        encoder.1828.rknn            + encoder.weight
        encoder_online.1828.rknn     + encoder_online.weight
        llm.1828.rknn                + llm.weight
        llm.embed.bin                fp16 (vocab_size × dim)
        llm.tokenizer.gguf           GPT-2 BPE vocab
        mel_128_filters.txt          (n_mels × N_FFT/2+1) fp32
"""
from __future__ import annotations

import ctypes
import logging
import time as _time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from ._common import (
    StreamSink,
    load_embed_table,
    load_tokenizer,
    make_embed_callback,
    make_tokenizer_callback,
    printf_perf,
)

LOG = logging.getLogger("rkai.backends.asr")

# --------------------------------------------------------------------------- #
# Constants — match cpp/qwen3_asr.h + cpp/audio/rknn_qwen3_asr_audio.cc.
# Hardcoded because RKNN3 exported the LLM against a fixed prompt layout;
# changing these breaks the RoPE alignment.
# --------------------------------------------------------------------------- #
HOP_LENGTH = 160
N_FFT = 400
DOWN_SAMPLE_RATE = 8
TARGET_SAMPLE_RATE = 16000
N_MELS_DEFAULT = 128
MELS_FILTERS_COLS = N_FFT // 2 + 1  # 201

PREFIX_N_TOKENS = 9                    # fixed by the LLM export
POSTFIX_N_TOKENS = 6                   # fixed by the LLM export
AUDIO_START_TOKEN = 151669
AUDIO_END_TOKEN = 151670

MAX_NEW_TOKENS_DEFAULT = 1024 * 6


# --------------------------------------------------------------------------- #
# WAV decoding — drop-in replacement for utils/audio_utils.cc (no miniaudio).
# --------------------------------------------------------------------------- #
def _pcm_to_mono_float32(pcm: bytes, n_channels: int, sample_width: int) -> np.ndarray:
    """Decode raw PCM payload to mono fp32 in [-1, 1].

    Supports s16le, s24le, s32le. ``asr_en.wav`` ships as s24le."""
    if sample_width == 2:
        raw = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        raw = np.frombuffer(pcm, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sample_width == 3:
        # numpy has no native int24; decode 3 little-endian bytes manually.
        b = np.frombuffer(pcm, dtype=np.uint8).reshape(-1, 3)
        v = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )
        sign = (b[:, 2] & 0x80) != 0
        v[sign] -= 0x1000000
        raw = v.astype(np.float32) / 8388608.0
    else:
        raise ValueError(f"unsupported sample_width={sample_width} (want 2/3/4)")
    if n_channels > 1:
        raw = raw.reshape(-1, n_channels).mean(axis=1).astype(np.float32)
    return raw


def _resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear interpolation (matches the C demo's hand-rolled resampler)."""
    if src_rate == dst_rate or len(audio) == 0:
        return audio.astype(np.float32, copy=False)
    out_len = int(round(len(audio) * dst_rate / src_rate))
    if out_len <= 1:
        return np.zeros(1, dtype=np.float32)
    idx = np.arange(out_len) * src_rate / dst_rate
    left = np.floor(idx).astype(np.int64)
    right = np.minimum(left + 1, len(audio) - 1)
    frac = idx - left
    return (audio[left] * (1.0 - frac) + audio[right] * frac).astype(np.float32)


# --------------------------------------------------------------------------- #
# Log-mel-spectrogram — pure numpy. Mirrors utils/audio_utils.cc.
# --------------------------------------------------------------------------- #
def _hann_window(n_fft: int) -> np.ndarray:
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_fft) / (n_fft - 1)))


def _reflect_pad(audio: np.ndarray, pad_width: int) -> np.ndarray:
    out = np.empty(len(audio) + 2 * pad_width, dtype=audio.dtype)
    out[pad_width:pad_width + len(audio)] = audio
    out[:pad_width] = audio[:pad_width][::-1]
    out[pad_width + len(audio):] = audio[-pad_width:][::-1]
    return out


def _log_mel_spectrogram(
    audio: np.ndarray,
    mel_filters: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    n = max(len(audio), n_fft)
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))
    padded = _reflect_pad(audio, n_fft // 2)
    num_frames = (len(padded) - n_fft) // hop_length + 1
    win = _hann_window(n_fft)
    frames = np.stack(
        [padded[i * hop_length:i * hop_length + n_fft] * win for i in range(num_frames)],
        axis=0,
    )
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    mag = (spec.real ** 2 + spec.imag ** 2).T  # (n_fft/2 + 1, num_frames)
    mag = mag[:, :num_frames - 1]            # drop the last frame (matches C)
    mel_spec = mel_filters @ mag             # (n_mels, num_frames - 1)
    np.maximum(mel_spec, 1e-10, out=mel_spec)
    np.log10(mel_spec, out=mel_spec)
    threshold = mel_spec.max() - 8.0
    np.maximum(mel_spec, threshold, out=mel_spec)
    return (mel_spec + 4.0) / 4.0


def _load_mel_filters(path: str, n_mels: int, n_cols: int) -> np.ndarray:
    """Load ``mel_128_filters.txt``. Some exports ship an over-long buffer;
    truncate the surplus before reshaping to ``(n_mels, n_cols)``."""
    with open(path, "r") as f:
        arr = np.fromstring(" ".join(f.read().split()), dtype=np.float32, sep=" ")
    expected = n_mels * n_cols
    if arr.size < expected:
        raise ValueError(
            f"{path}: expected at least {expected} floats, got {arr.size}"
        )
    return arr[:expected].reshape(n_mels, n_cols)


# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #
class ASRBackend:
    """WAV bytes in → raw transcript string out."""

    def __init__(self, spec):
        from rknn3lite.api import RKNN3Lite, RKLLMCallback  # type: ignore
        from rknn3lite.api.rknn3_types import (
            RKNN3QueryCmd, RKNN3MemAllocFlags,
        )

        if not spec.asr_model_dir:
            raise ValueError(
                f"{spec.name!r} requires spec.asr_model_dir; "
                "fill it via .with_paths(asr_model_dir=…)."
            )
        self.spec = spec
        model_dir = Path(spec.asr_model_dir)

        encoder_rknn = model_dir / "encoder.1828.rknn"
        encoder_wt = model_dir / "encoder.weight"
        llm_rknn = model_dir / "llm.1828.rknn"
        llm_wt = model_dir / "llm.weight"
        for p in (
            encoder_rknn, encoder_wt, llm_rknn, llm_wt,
            Path(spec.tokenizer_path), Path(spec.embed_path),
            model_dir / "mel_128_filters.txt",
        ):
            if not p.is_file():
                raise FileNotFoundError(f"Qwen3-ASR: missing required file {p}")

        self._mel_filters = _load_mel_filters(
            str(model_dir / "mel_128_filters.txt"), N_MELS_DEFAULT, MELS_FILTERS_COLS,
        )

        # ---- Audio encoder: user_mem_internal=1 so shared internal memory
        # can be set after the LLM is also initialised.
        self.rknn_audio = RKNN3Lite(verbose=spec.verbose)
        if self.rknn_audio.load_rknn(str(encoder_rknn), str(encoder_wt)) != 0:
            raise RuntimeError(f"audio load_rknn failed: {encoder_rknn}")
        self.rknn_audio.rknn_config.user_mem_internal = 1
        if self.rknn_audio.init_runtime(
            target=spec.target, core_mask=spec.core_mask,
        ) != 0:
            raise RuntimeError("audio init_runtime failed")

        audio_inputs = self.rknn_audio.get_inputs_tensor_attr()
        audio_outputs = self.rknn_audio.get_outputs_tensor_attr()
        in_attr = audio_inputs[0]
        out_attr = audio_outputs[0]
        # RKNN3TensorAttr.shape is a fixed-length array; slice by n_dims.
        in_shape = list(in_attr.shape[: in_attr.n_dims])
        out_shape = list(out_attr.shape[: out_attr.n_dims])
        # Layout enum: 1 = RKNN3_TENSOR_NCHW.
        if in_attr.layout != 1:
            raise RuntimeError(
                f"audio encoder: expected NCHW input layout, got layout={in_attr.layout}"
            )
        self._batch_size = int(in_shape[0])
        self._n_mels = int(in_shape[2])
        self._window_size = int(in_shape[3])
        self._embeds_dim = int(out_shape[-1])
        LOG.info(
            "audio encoder: batch=%d n_mels=%d window_size=%d embeds_dim=%d",
            self._batch_size, self._n_mels, self._window_size, self._embeds_dim,
        )

        # ---- Tokenizer + embed table; embed dim must match encoder output.
        self.tokenizer = load_tokenizer(spec.tokenizer_path)
        self.embeds = load_embed_table(spec.embed_path, spec.vocab_size)
        if self.embeds.shape[1] != self._embeds_dim:
            raise RuntimeError(
                f"LLM embed dim ({self.embeds.shape[1]}) must match audio "
                f"encoder embeds_dim ({self._embeds_dim})"
            )

        # ---- LLM RKNN.
        self.rknn_llm = RKNN3Lite(llm_mode=True, verbose=spec.verbose)
        if self.rknn_llm.load_rknn(str(llm_rknn), str(llm_wt)) != 0:
            raise RuntimeError(f"llm load_rknn failed: {llm_rknn}")
        self.rknn_llm.rknn_config.user_mem_internal = 1

        self.sink = StreamSink(tokenizer=self.tokenizer)
        cb = RKLLMCallback()
        cb.result_callback = _make_result_callback(self)
        cb.tokenizer_callback, cb.tokenizer_userdata = make_tokenizer_callback(
            self.tokenizer
        )
        cb.embed_callback, cb.embed_userdata = make_embed_callback(self.embeds)
        if self.rknn_llm.init_runtime(
            target=spec.target, core_mask=spec.core_mask,
            llm_args=self._build_llm_args(spec), llm_callback=cb,
        ) != 0:
            raise RuntimeError("llm init_runtime failed")

        self._setup_shared_internal_mem()
        self._prefix_embeds, self._postfix_embeds = self._build_chat_embeds()

        # NCHW input buffer (batch, channels=1, n_mels, window_size).
        self._audio_in_buf = np.zeros(
            (self._batch_size, 1, self._n_mels, self._window_size),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ #
    def _build_llm_args(self, spec) -> List[Dict[str, Any]]:
        o = spec.options
        return [{
            "max_new_tokens": o.get("max_new_tokens", MAX_NEW_TOKENS_DEFAULT),
            "top_k": o.get("top_k", 1),
            "top_p": o.get("top_p", 0.9),
            "temperature": o.get("temperature", 1.0),
            "repeat_penalty": o.get("repeat_penalty", 1.2),
            "vocab_size": spec.vocab_size,
            "special_eos_id": spec.special_eos_id,
            "max_context_len": spec.max_context_len,
            "keep_history": o.get("keep_history", 0),
        }]

    # ------------------------------------------------------------------ #
    def _setup_shared_internal_mem(self) -> None:
        """Allocate one ``RKNN3TensorMemory`` per NPU core and hand it to
        both contexts via ``set_internal_mem``.

        Mirrors ``init_internal_share`` in cpp/qwen3_asr.cc. Each
        core's ``internal_size`` is ``max(audio_internal, llm_internal)``
        — without sharing, the LLM's pre-allocation would clip the
        audio encoder's working set.

        ``RKNN3Lite.create_mem`` only accepts ``RKNN3TensorAttr``, so
        we call the underlying ``lib.rknn3_create_mem`` directly with
        ``(size, core_id, flags)`` to get a bare POINTER back.
        """
        from rknn3lite.api.rknn3_types import (
            RKNN3QueryCmd, RKNN3MemAllocFlags, RKNN3CoreMemSize,
        )

        def _query_core_mems(handle):
            core_num = int(handle.rknn3_query(
                RKNN3QueryCmd.RKNN3_QUERY_CORE_NUMBER,
            ))
            if core_num <= 0:
                return []
            sizes = (RKNN3CoreMemSize * core_num)()
            ret = handle.rknn_runtime.lib.rknn3_query(
                handle.rknn_runtime.context,
                int(RKNN3QueryCmd.RKNN3_QUERY_CORE_MEM_SIZE),
                ctypes.cast(sizes, ctypes.c_void_p),
                ctypes.c_size_t(ctypes.sizeof(sizes)),
            )
            if ret < 0:
                raise RuntimeError(f"rknn3_query CORE_MEM_SIZE failed: ret={ret}")
            return [(s.internal_size, s.core_id) for s in sizes]

        audio_cores = _query_core_mems(self.rknn_audio)
        llm_cores = _query_core_mems(self.rknn_llm)
        n = max(len(audio_cores), len(llm_cores))
        shared = []
        for i in range(n):
            a = audio_cores[i] if i < len(audio_cores) else None
            l = llm_cores[i] if i < len(llm_cores) else None
            internal = max(
                a[0] if a is not None else 0,
                l[0] if l is not None else 0,
            )
            if internal <= 0:
                shared.append(None)
                continue
            mem = self.rknn_llm.rknn_runtime.create_mem(
                internal, i,
                RKNN3MemAllocFlags.RKNN3_FLAG_MEMORY_CACHEABLE,
            )
            if not mem:
                raise RuntimeError(
                    f"create_mem failed for core {i} size={internal}"
                )
            shared.append(mem)

        valid = [m for m in shared if m is not None]
        if valid:
            self.rknn_audio.set_internal_mem(valid)
            self.rknn_llm.set_internal_mem(valid)
        LOG.info(
            "shared internal memory: cores=%d blocks=%d",
            n, sum(1 for m in shared if m is not None),
        )

    # ------------------------------------------------------------------ #
    def _build_chat_embeds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Tokenise the hardcoded prefix / postfix prompts and look up
        each token's fp16 embedding from the LLM embed table.

        Mirrors ``init_qwen3_asr_model`` in cpp/qwen3_asr.cc:184-204.
        The 8-token prefix / 5-token postfix are produced by both the
        upstream llama.cpp BPE session *and* our GGUF-fed
        HuggingFace tokenizer (gpt2 BPE + ByteLevel); the 9th / 6th
        token (the audio boundary marker) is hardcoded to match.
        """
        prefix_ids = self._tokenize_to_list(
            "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n"
        )[: PREFIX_N_TOKENS - 1]
        if len(prefix_ids) != PREFIX_N_TOKENS - 1:
            raise RuntimeError(
                f"prefix tokenize: expected {PREFIX_N_TOKENS - 1} tokens, "
                f"got {len(prefix_ids)}"
            )
        prefix_ids.append(AUDIO_START_TOKEN)

        postfix_ids = self._tokenize_to_list(
            "<|im_end|>\n<|im_start|>assistant\n"
        )[: POSTFIX_N_TOKENS - 1]
        if len(postfix_ids) != POSTFIX_N_TOKENS - 1:
            raise RuntimeError(
                f"postfix tokenize: expected {POSTFIX_N_TOKENS - 1} tokens, "
                f"got {len(postfix_ids)}"
            )
        postfix_ids = [AUDIO_END_TOKEN] + postfix_ids

        return (
            self.embeds[prefix_ids].astype(np.float16),
            self.embeds[postfix_ids].astype(np.float16),
        )

    def _tokenize_to_list(self, text: str) -> List[int]:
        """Normalise HuggingFace + transformers-only outputs to ``List[int]``.

        ``PreTrainedTokenizerFast.__call__`` returns
        ``{"input_ids": ndarray([1, n])}`` when PyTorch is installed and
        ``{"input_ids": [int, ...]}`` when transformers is run without it.
        """
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if isinstance(ids, list) and ids and not isinstance(ids[0], list):
            return list(ids)
        if hasattr(ids, "shape"):
            return np.asarray(ids).flatten().tolist()
        flat: List[int] = []
        for v in ids:
            if isinstance(v, list):
                flat.extend(v)
            else:
                flat.append(int(v))
        return flat

    # ------------------------------------------------------------------ #
    def _audio_token_count(self, mel_frames: int) -> int:
        """Mirror ``get_n_audio`` in cpp/audio/rknn_qwen3_asr_audio.cc."""
        if mel_frames <= 0:
            return 0
        left = mel_frames % self._window_size
        n_left = -(-left // DOWN_SAMPLE_RATE)
        n_per_window = -(-self._window_size // DOWN_SAMPLE_RATE)
        return (mel_frames // self._window_size) * n_per_window + n_left

    def _run_audio_encoder(self, mel_spec: np.ndarray) -> np.ndarray:
        """Run the audio encoder over ``mel_spec`` chunk-by-chunk.

        Returns ``(n_audio_tokens, embeds_dim)`` fp16."""
        n_mels, num_frames = mel_spec.shape
        if n_mels != self._n_mels:
            raise RuntimeError(
                f"audio encoder: expected n_mels={self._n_mels}, got {n_mels}"
            )

        n_token_per_window = -(-self._window_size // DOWN_SAMPLE_RATE)
        total_tokens = self._audio_token_count(num_frames)
        if total_tokens <= 0:
            raise RuntimeError("audio too short after STFT")

        out = np.empty((total_tokens, self._embeds_dim), dtype=np.float16)
        n_chunks = (num_frames + self._window_size - 1) // self._window_size
        for chunk_idx in range(n_chunks):
            start = chunk_idx * self._window_size
            real_len = min(self._window_size, num_frames - start)
            self._audio_in_buf.fill(0.0)
            if real_len > 0:
                self._audio_in_buf[0, 0, :, :real_len] = mel_spec[:, start:start + real_len]

            outs = self.rknn_audio.inference(inputs=[self._audio_in_buf])
            if outs is None or len(outs) == 0:
                raise RuntimeError("audio encoder inference failed")
            enc_out = np.asarray(outs[0])
            if enc_out.ndim == 3:  # (batch, n_token, dim) → (n_token, dim)
                enc_out = enc_out[0]
            base = chunk_idx * n_token_per_window
            n_valid = min(enc_out.shape[0], total_tokens - base)
            if n_valid > 0:
                out[base:base + n_valid] = enc_out[:n_valid].astype(np.float16)
        return out

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run_audio(
        self,
        pcm_bytes: bytes,
        sample_rate: int = TARGET_SAMPLE_RATE,
        n_channels: int = 1,
        sample_width: int = 2,
        **overrides: Any,
    ) -> str:
        """Run a single ASR pass over a raw PCM payload.

        ``**overrides`` (``max_new_tokens`` / ``keep_history``) goes
        straight to ``rknn_llm.session_run``. Returns the LLM text
        output (``"language <lang>\\n<text>"``).
        """
        self.sink.reset()
        audio = _pcm_to_mono_float32(pcm_bytes, n_channels, sample_width)
        if sample_rate != TARGET_SAMPLE_RATE:
            audio = _resample_linear(audio, sample_rate, TARGET_SAMPLE_RATE)
        if len(audio) < N_FFT:
            raise RuntimeError(f"audio too short: {len(audio)} samples (< {N_FFT})")
        mel_spec = _log_mel_spectrogram(audio, self._mel_filters)
        audio_embeds = self._run_audio_encoder(mel_spec)

        full = np.concatenate(
            [
                self._prefix_embeds[None, ...],
                audio_embeds[None, ...],
                self._postfix_embeds[None, ...],
            ],
            axis=1,
        ).astype(np.float16)

        kwargs: Dict[str, Any] = {"embeds": full, "keep_history": 0}
        if overrides.get("max_new_tokens") is not None:
            kwargs["max_new_tokens"] = overrides["max_new_tokens"]
        ret, stats = self.rknn_llm.session_run(**kwargs)
        if ret != 0:
            raise RuntimeError(f"llm session_run failed: ret={ret}")
        n_decode, n_prefill, llm_start, llm_end = stats
        # The language separator (token 151704) is in the GGUF tokenizer's
        # "control token" whitelist but not in HF's "special tokens"
        # whitelist, so it leaks out as the literal ``[PAD151704]``
        # glued to the language header. Replace it with ``\n`` so
        # callers see a clean split. (The C demo splits on the token
        # id directly inside its streaming callback.)
        text = self.sink.text.replace("[PAD151704]", "\n", 1)
        if self.sink.first_token_time is not None:
            self.sink.perf.update({
                "n_decode_tokens": n_decode,
                "n_prefill_tokens": n_prefill,
                "prefill_ms": (self.sink.first_token_time - llm_start) * 1000.0,
                "decode_ms": (llm_end - self.sink.first_token_time) * 1000.0,
            })
            if overrides.get("perf", False):
                printf_perf(self.sink)
        return text

    # Stub fall-throughs — ``LLMSession._dispatch`` probes for these
    # names when it can't route to a more specific method (audio /
    # image / video). Keep them as informative errors rather than
    # relying on AttributeError.
    def run_text(self, prompt: str, **kw: Any) -> str:
        raise RuntimeError(
            "ASRBackend.run_text(): pass Attachment(wav_bytes, kind='audio') "
            "to LLMSession.respond() and the SDK will route to run_audio()."
        )

    def run_images(self, prompt: str, images: list, **kw: Any) -> str:
        raise RuntimeError("ASR backend does not support images.")

    def run_video(self, prompt: str, frames: list, **kw: Any) -> str:
        raise RuntimeError("ASR backend does not support video.")

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        for r in (getattr(self, "rknn_audio", None),
                  getattr(self, "rknn_llm", None)):
            if r is None:
                continue
            try:
                r.release()
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# Result callback — builds tokens into StreamSink as the LLM streams.
# --------------------------------------------------------------------------- #
def _make_result_callback(b: "ASRBackend"):
    from rknn3lite.api import LLMCallState, LLMResultCallback  # type: ignore

    def _cb(userdata, result_ptr, state):
        s = int(state)
        if s == int(LLMCallState.RKLLM_RUN_ERROR):
            b.sink.error = "inference error"
            return 0
        if s == int(LLMCallState.RKLLM_RUN_WAITING):
            return 0
        if s in (
            int(LLMCallState.RKLLM_RUN_FINISH),
            int(LLMCallState.RKLLM_RUN_STOP),
            int(LLMCallState.RKLLM_RUN_MAX_NEW_TOKEN_REACHED),
        ):
            try:
                full = _decode_safe(b.sink.tokenizer, b.sink.tokens)
                delta = full[len(b.sink.text):]
                if delta:
                    b.sink.text += delta
            except Exception as exc:  # noqa: BLE001
                b.sink.error = f"decode error: {exc}"
            b.sink.finished = True
            return 0
        if s == int(LLMCallState.RKLLM_RUN_NORMAL):
            n = result_ptr.contents.num_tokens
            new_tokens = [
                int(result_ptr.contents.token_ids[i]) for i in range(n)
            ]
            b.sink.tokens.extend(new_tokens)
            if b.sink.first_token_time is None:
                b.sink.first_token_time = _time.perf_counter()
            try:
                full = _decode_safe(b.sink.tokenizer, b.sink.tokens)
                delta = full[len(b.sink.text):]
                if delta:
                    b.sink.text += delta
            except Exception:
                # Trailing UTF-8 chunk — wait for the next call.
                pass
        return 0

    return LLMResultCallback(_cb)


def _decode_safe(tokenizer, tokens: List[int]) -> str:
    text = tokenizer.decode(
        tokens, skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return text.split("\ufffd", 1)[0] if "\ufffd" in text else text