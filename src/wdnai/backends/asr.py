"""Qwen3-ASR backend — wav-in, text-out.

Offline pipeline (mirrors ``examples/Qwen3_ASR/cpp/main.cc``)::

    raw PCM  →  16 kHz mono fp32  →  log-mel-spectrogram
           →  audio encoder (NCHW, ``encoder.1828.rknn``) one shot
           →  fp16 audio embeddings  →  prefix (9) + audio + postfix (6)
           →  llm.session_run(embeds=…, keep_history=0)
           →  raw text, with separator ``[PAD151704]`` replaced by ``\n``.

Online pipeline (mirrors ``examples/Qwen3_ASR/cpp/online_main.cc``)
runs when ``spec.options["use_online_encoder"]`` is True. The audio
is split into 1-second chunks; each chunk runs the streaming
encoder (``encoder_online.1828.rknn``) and the LLM once, with all
previous audio embeddings and confirmed-text tokens carried in the
LLM input. Rollback (4 warm-up rounds + 5 trailing tokens) keeps
the committed prefix stable.

``run_audio`` returns the final transcript (``"language <lang>\\n<text>"``).
Online mode additionally calls ``on_asr_commit(committed, unfixed,
language)`` after each round if provided. The example in
``examples/audio_to_text.py`` consumes both.

Supported model layout (matches :class:`wdnai.model.SystemLanguageModel.qwen3_asr`):

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
import json
import logging
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ._common import (
    StreamSink,
    load_embed_table,
    load_tokenizer,
    printf_perf,
    resolve_device_id,
)

LOG = logging.getLogger("wdnai.backends.asr")

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
LANGUAGE_SEP_TOKEN = 151704            # token after which the ASR body starts

MAX_NEW_TOKENS_DEFAULT = 1024 * 6

# Online-encoder knobs. Defaults mirror cpp/online_main.cc
# (ONLINE_CHUNK_MS=1000, ONLINE_MAX_NEW_TOKENS=50,
#  ONLINE_ROLLBACK_TOKENS=5, ONLINE_ROLLBACK_ROUNDS=4).
ONLINE_CHUNK_SECONDS_DEFAULT = 1.0
ONLINE_MAX_NEW_TOKENS_DEFAULT = 50
ONLINE_ROLLBACK_TOKENS_DEFAULT = 5
ONLINE_ROLLBACK_ROUNDS_DEFAULT = 4

# --------------------------------------------------------------------------- #
# Incremental streaming sessions.
#
# The streaming path is *incremental*: the caller keeps feeding audio and
# reads back one round per chunk. The session state (audio embedding
# history + committed tokens + the text accumulator) has to survive
# **across** calls, which the file-driven path does not need — there the
# whole buffer arrives in one ``run_audio`` call.
#
# The three keys below travel through the RPC ``options`` dict, which the
# transport treats as free-form (``wdndds`` puts ``dict(overrides)`` on the
# wire verbatim). So a caller reaches the incremental path with
# ``session.respond(..., asr_stream_id="s1")`` and **no transport change
# is required** — the worker stays a thin relay.
# --------------------------------------------------------------------------- #
#: Session id — groups the calls that share one audio history.
ASR_STREAM_ID_KEY = "asr_stream_id"
#: True when the block carried by this call is the session's last one.
#: The final block is not rolled back (mirrors ``is_last`` in the C demo).
ASR_STREAM_FINAL_KEY = "asr_stream_final"
#: Timeline origin of a streaming session, in seconds: the position of
#: the session's **first** sample on the caller's timeline. The caller
#: owns the timeline (it knows which file / stream the audio came from),
#: so it supplies the origin and the backend derives every block's start
#: from it. Letting the backend accumulate the position would
#: desynchronise after a seek — the backend cannot know how much was
#: skipped. Seeking = a new session (new id) with a new origin.
ASR_STREAM_ORIGIN_SEC_KEY = "asr_stream_origin_sec"

#: Idle TTL for a streaming session. The worker is a long-lived process, so
#: a leaked session would grow ``audio_embed_history`` forever.
ASR_STREAM_TTL_SEC_DEFAULT = 120.0


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
    """Linear-interpolation resampler, ``src_rate`` → ``dst_rate``.

    Why linear and not a band-limited (windowed-sinc) resampler?
    ----------------------------------------------------------
    The C demo hands the WAV to **miniaudio** (``ma_decoder_init_file``),
    which resamples 48 kHz → 16 kHz internally with its own
    band-limited interpolator. That filter is part of the *demo's file
    I/O*, not of the ASR algorithm.

    Measured on ``asr_en.wav`` (48 kHz s24le → 16 kHz):

    * **linear interpolation** (this function) → the final committed
      transcript is **byte-identical** to the C demo's, and 14 of the 16
      streaming rounds match exactly (the 2 that differ do so only in
      the *speculative* "unfixed" tail, which the rollback policy
      discards anyway — the committed text never diverged).
    * **windowed-sinc** → a different (equally plausible) transcript:
      ``"Oh yeah, he wasn't even that big…"`` instead of
      ``"Uh huh. Oh yeah, yeah! He wasn't even that big…"``.

    The ASR model is sensitive at the margins, so the resampler choice
    is *observable*. Since the C demo is the reference this SDK is
    validated against, we keep the resampler that reproduces it.

    Residual (documented) divergence, in short: the C's miniaudio
    resampler and this linear one differ in the last bits of the mel
    spectrogram, which occasionally flips a single token in the
    speculative tail of a round. The rollback strategy absorbs it — the
    committed transcript and the final result are unaffected.
    """
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


class _StreamResampler:
    """Streaming counterpart of :func:`_resample_linear`.

    Why not just resample each block on its own?

    The offline path resamples the **whole** buffer in one go, so every
    output sample is interpolated from the **global** input index
    ``j * src_rate / dst_rate``.  A block-local resampler restarts that
    index at 0 on every block, which shifts the phase by one block per
    block — the two paths would not agree sample-for-sample, and the ASR
    model is sensitive enough at the margins for that to flip a token.
    (Concretely: 48 kHz → 16 kHz, 48 000 samples per block. The offline
    path emits input indices ``0, 3, 6, …``; a block-local resampler
    would emit ``0, 3, 6, …`` again in block 2 instead of
    ``48 000, 48 003, …``.)

    This class keeps a *carry* of the not-yet-consumed input tail and
    indexes the **global** output counter, so the concatenated output of
    N pushes equals the offline resample of the concatenated input —
    byte for byte.

    Two subtleties make that exact rather than approximate:

    1. **The output count.** The offline path emits
       ``round(total_in / ratio)`` samples, and a streaming resampler
       cannot know the total length in advance. So the streaming path
       emits only the outputs whose *second* interpolation neighbour is
       a real sample, and defers the offline path's last-sample
       extrapolation (:meth:`flush`) to the end — where the total is
       finally known. The two then agree exactly, including the count.
    2. **The index.** Output ``j`` is interpolated from the global input
       index ``(j * src_rate) / dst_rate``, not from a block-local
       index. A block-local resample would restart that index at 0 on
       every block and shift the phase by one block per block — and the
       ASR model is sensitive enough at the margins for that to flip a
       token.

    Only **downsampling** is supported (``src_rate >= dst_rate``), which is
    the only direction the ASR backend ever needs (48 kHz / 32 kHz / 24 kHz
    → 16 kHz).
    """

    def __init__(self, src_rate: int, dst_rate: int) -> None:
        if src_rate < dst_rate:
            raise ValueError(
                "_StreamResampler supports downsampling only "
                "(src_rate >= dst_rate), got {} -> {}".format(src_rate, dst_rate)
            )
        self.src_rate = int(src_rate)
        self.dst_rate = int(dst_rate)
        self.ratio = float(src_rate) / float(dst_rate)
        self.carry = np.zeros((0,), dtype=np.float32)
        self.out_index = 0
        self.total_in = 0

    # -- internals ------------------------------------------------------- #
    def _render(self, x: np.ndarray, base_global: int, j0: int, n_new: int) -> np.ndarray:
        """Render output samples ``j0 … j0+n_new-1`` from ``x``.

        ``base_global`` is the global sample index that ``x[0]`` carries.
        The index arithmetic mirrors :func:`_resample_linear` exactly —
        including the ``(j * src) / dst`` float form, so the two produce
        bit-identical samples.
        """
        idx = np.arange(j0, j0 + n_new) * self.src_rate / self.dst_rate
        left_global = np.floor(idx).astype(np.int64)
        left = left_global - base_global
        right = np.minimum(left + 1, x.size - 1)
        frac = idx - left_global
        return (x[left] * (1.0 - frac) + x[right] * frac).astype(np.float32)

    def _offline_count(self) -> int:
        """``round(total_in / ratio)`` — the offline path's output count."""
        return int(round(self.total_in * self.dst_rate / self.src_rate))

    # -- public ---------------------------------------------------------- #
    def push(self, audio: np.ndarray) -> np.ndarray:
        """Consume ``audio``; return the output samples it completes."""
        if audio.size:
            x = (np.concatenate([self.carry, audio]) if self.carry.size
                 else audio.astype(np.float32, copy=False))
            self.total_in += int(audio.size)
        else:
            x = self.carry

        if x.size == 0:
            return np.zeros((0,), dtype=np.float32)

        # Output j needs input floor(j*ratio) **and** floor(j*ratio)+1.
        # Emit only while that second neighbour is a real sample; the
        # offline path's extrapolation of the very last sample is left
        # to flush(), which is the only place that knows the total.
        j_limit = int((self.total_in - 2) // self.ratio) + 1
        n_target = min(j_limit, self._offline_count())
        n_new = n_target - self.out_index
        if n_new <= 0:
            self.carry = x
            return np.zeros((0,), dtype=np.float32)

        base_global = self.total_in - x.size
        out = self._render(x, base_global, self.out_index, n_new)
        self.out_index += n_new

        # Everything from the last consumed input sample onwards is carry.
        consumed_global = int(np.floor((self.out_index - 1) * self.ratio)) + 1
        self.carry = x[consumed_global - base_global:]
        return out

    def flush(self) -> np.ndarray:
        """Emit the tail the offline path would have produced.

        The offline path emits ``round(total_in / ratio)`` samples and
        extrapolates the last one with ``right = min(left + 1, len - 1)``;
        the streaming path stopped short because it cannot extrapolate
        before it knows the input has ended.
        """
        if self.total_in == 0:
            return np.zeros((0,), dtype=np.float32)
        n_new = self._offline_count() - self.out_index
        if n_new <= 0 or self.carry.size == 0:
            self.carry = np.zeros((0,), dtype=np.float32)
            return np.zeros((0,), dtype=np.float32)
        base_global = self.total_in - self.carry.size
        out = self._render(self.carry, base_global, self.out_index, n_new)
        self.out_index += n_new
        self.carry = np.zeros((0,), dtype=np.float32)
        return out


# --------------------------------------------------------------------------- #
# Log-mel-spectrogram — pure numpy. Mirrors utils/audio_utils.cc.
# --------------------------------------------------------------------------- #
def _hann_window(n_fft: int) -> np.ndarray:
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_fft) / (n_fft - 1)))


def _reflect_pad(audio: np.ndarray, pad_width: int) -> np.ndarray:
    """Reflect-pad ``audio`` by ``pad_width`` on both sides.

    Mirrors ``reflect_pad`` in the C reference (``utils/audio_utils.cc``) —
    **except** for the case the C gets wrong: when the buffer is shorter
    than ``pad_width``, the C writes ``audio.begin() + pad_width`` (past
    the end) and reads ``audio.end() - pad_width`` (before the begin),
    i.e. undefined behaviour.  The numpy port of that expression raises

        ValueError: could not broadcast input array from shape (160,)
        into shape (200,)

    — a hard failure on a legal input (a short tail block).  So the
    reflection here uses **only the samples that exist** and leaves the
    uncovered edge zero-filled: deterministic, and no worse than the C's
    garbage.
    """
    n = len(audio)
    out = np.zeros(n + 2 * pad_width, dtype=audio.dtype)
    if n == 0:
        return out
    out[pad_width:pad_width + n] = audio
    w = min(pad_width, n)
    out[pad_width - w:pad_width] = audio[:w][::-1]
    out[pad_width + n:pad_width + n + w] = audio[-w:][::-1]
    return out


def _log_mel_spectrogram(
    audio: np.ndarray,
    mel_filters: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Log-mel spectrogram, **byte-for-byte layout-compatible with the C demo**.

    Mirrors ``audio_preprocess`` + ``log_mel_spectrogram`` in the model-zoo's
    ``utils/audio_utils.cc``:

    1. ``n_frame = floor((len - n_fft) / hop) + 1`` — the C's frame count;
    2. the C then calls ``audio_preprocess(..., max_audio_length = n_frame * hop, ...)``
       and **truncates** the audio to that length when it is longer. For a
       1-second chunk (16000 samples) that keeps 15680 samples → 98 mel
       columns, *not* 100. Without this trim the encoder sees two extra
       (non-zero) frames the C never fed it, and the token/feature layout
       drifts from the reference implementation.
    3. reflect-pad by ``n_fft // 2``, STFT with a Hann window, drop the last
       frame, project through the mel filters, ``log10``, clamp to
       ``max - 8 dB``, scale to ``(x + 4) / 4``.

    Returns ``(n_mels, n_frame)`` — the same column count the C feeds the
    encoder.
    """
    audio_length = len(audio)
    if audio_length < n_fft:
        audio = np.pad(audio, (0, n_fft - audio_length))
        audio_length = n_fft

    # (1) C frame count, (2) C truncation.
    n_frame = (audio_length - n_fft) // hop_length + 1
    max_audio_length = n_frame * hop_length
    if audio_length >= max_audio_length:
        audio = audio[:max_audio_length]
        audio_length = max_audio_length

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
# Streaming session state
# --------------------------------------------------------------------------- #
@dataclass
class _OnlineStreamState:
    """Everything one incremental streaming session accumulates.

    The file-driven path keeps the same values in local variables for the
    duration of a single ``run_audio`` call.  The incremental path has to
    keep them **across** calls, because a session spans many requests:
    the audio embedding history and the confirmed-token list are the
    model's context, and losing them would restart the transcript on
    every block.
    """

    stream_id: str
    samples_per_chunk: int
    max_new_tokens: int
    rollback_tokens: int
    rollback_rounds: int
    chunk_seconds: float

    audio_embed_history: List[np.ndarray] = field(default_factory=list)
    committed_token_ids: List[int] = field(default_factory=list)
    decoder: Any = None

    #: Index of the next block to run — **global** across calls, so the
    #: warm-up window (``rollback_rounds``) counts from the session start.
    chunk_idx: int = 0
    #: Position of the session's first sample on the caller's timeline.
    origin_sec: float = 0.0
    #: Samples already consumed, in 16 kHz units.
    samples_consumed: int = 0
    #: Tail of the caller's audio that does not fill a whole block yet.
    pending: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    #: Streaming resampler (only used when the source rate != 16 kHz).
    resampler: Optional[_StreamResampler] = None

    created_at: float = 0.0
    last_seen: float = 0.0

    @property
    def chunk_start_sec(self) -> float:
        """Start of the *next* block on the caller's timeline."""
        return (self.origin_sec
                + self.chunk_idx * self.samples_per_chunk
                / float(TARGET_SAMPLE_RATE))


# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #
class ASRBackend:
    """WAV bytes in → raw transcript string out."""

    def __init__(
        self,
        spec,
        *,
        npu: Optional[int] = None,
        device_id: Optional[Union[bytes, str]] = None,
    ):
        from .._librknn import RKNN3Lite, build_llm_callback

        if not spec.asr_model_dir:
            raise ValueError(
                f"{spec.name!r} requires spec.asr_model_dir; "
                "fill it via .with_paths(asr_model_dir=…)."
            )
        self.spec = spec
        model_dir = Path(spec.asr_model_dir)

        # Switch encoder based on the streaming flag. The two encoder
        # rknns share the same input shape / dtype, so the runtime
        # query below is identical for both.
        self._use_online = bool(spec.options.get("use_online_encoder", False))
        if self._use_online:
            encoder_rknn = model_dir / "encoder_online.1828.rknn"
            encoder_wt = model_dir / "encoder_online.weight"
        else:
            encoder_rknn = model_dir / "encoder.1828.rknn"
            encoder_wt = model_dir / "encoder.weight"

        llm_rknn = model_dir / "llm.1828.rknn"
        llm_wt = model_dir / "llm.weight"
        # Only the encoder actually selected is required. A streaming-only
        # deployment (encoder_online) has no offline encoder on disk, and
        # vice versa — demanding both made the backend refuse to load a
        # perfectly usable model folder.
        for p in (
            encoder_rknn, encoder_wt,
            llm_rknn, llm_wt,
            Path(spec.tokenizer_path), Path(spec.embed_path),
            model_dir / "mel_128_filters.txt",
        ):
            if not p.is_file():
                raise FileNotFoundError(f"Qwen3-ASR: missing required file {p}")

        self._mel_filters = _load_mel_filters(
            str(model_dir / "mel_128_filters.txt"), N_MELS_DEFAULT, MELS_FILTERS_COLS,
        )

        # Resolve NPU selection **once** after RKNN3Lite is imported;
        # audio encoder + llm init_runtime share the same resolved bytes.
        resolved_device_id = resolve_device_id(RKNN3Lite, npu, device_id)

        # ---- Audio encoder: user_mem_internal=1 so shared internal memory
        # can be set after the LLM is also initialised.
        self.rknn_audio = RKNN3Lite(verbose=spec.verbose)
        if self.rknn_audio.load_rknn(str(encoder_rknn), str(encoder_wt)) != 0:
            raise RuntimeError(f"audio load_rknn failed: {encoder_rknn}")
        self.rknn_audio.rknn_config.user_mem_internal = 1
        if self.rknn_audio.init_runtime(
            target=spec.target, core_mask=spec.core_mask,
            device_id=resolved_device_id,
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
        cb = build_llm_callback(
            result_cb=_make_result_callback(self),
            tokenizer=self.tokenizer,
            embeds_data=self.embeds,
        )
        if self.rknn_llm.init_runtime(
            target=spec.target, core_mask=spec.core_mask,
            llm_args=self._build_llm_args(spec), llm_callback=cb,
            device_id=resolved_device_id,
        ) != 0:
            raise RuntimeError("llm init_runtime failed")

        self._setup_shared_internal_mem()
        self._prefix_embeds, self._postfix_embeds = self._build_chat_embeds()

        # ---- Incremental streaming sessions.
        #: Live sessions keyed by ``asr_stream_id``. The worker is a
        #: long-lived process, so sessions are reaped by an idle TTL
        #: (:meth:`_gc_streams`) — otherwise a caller that stops feeding
        #: would leak the audio embedding history forever.
        self._online_streams: Dict[str, _OnlineStreamState] = {}
        self._stream_ttl_sec = float(
            spec.options.get("asr_stream_ttl_sec", ASR_STREAM_TTL_SEC_DEFAULT)
        )

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
        """Allocate one ``rknn3_tensor_mem*`` per NPU core and hand it
        to both contexts via ``set_internal_mem``.

        Mirrors ``init_internal_share`` in cpp/qwen3_asr.cc. Each
        core's ``internal_size`` is ``max(audio_internal, llm_internal)``
        — without sharing, the LLM's pre-allocation would clip the
        audio encoder's working set.
        """
        # RKNN3 query command int values match rknn3_api.h:
        #   13 = RKNN3_QUERY_CORE_NUMBER
        #   6  = RKNN3_QUERY_CORE_MEM_SIZE
        #   1  = RKNN3_FLAG_MEMORY_CACHEABLE
        CMD_CORE_NUMBER = 13
        CMD_CORE_MEM_SIZE = 6
        FLAG_CACHEABLE = 1

        def _query_core_mems(handle):
            core_num = int(handle.rknn3_query(CMD_CORE_NUMBER) or 0)
            if core_num <= 0:
                return []
            return handle.rknn3_query(CMD_CORE_MEM_SIZE) or []

        audio_cores = _query_core_mems(self.rknn_audio)
        llm_cores = _query_core_mems(self.rknn_llm)
        n = max(len(audio_cores), len(llm_cores))
        shared = []
        for i in range(n):
            a = audio_cores[i] if i < len(audio_cores) else None
            l = llm_cores[i] if i < len(llm_cores) else None
            internal = max(
                int(a.internal_size) if a is not None else 0,
                int(l.internal_size) if l is not None else 0,
            )
            if internal <= 0:
                shared.append(None)
                continue
            mem = self.rknn_llm.create_mem(internal, i, FLAG_CACHEABLE)
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
        on_asr_commit: Optional[Any] = None,
        **overrides: Any,
    ) -> str:
        """Run a single ASR pass over a raw PCM payload.

        The encoder choice is locked at backend construction time via
        ``spec.options["use_online_encoder"]``: ``False`` (default)
        uses the offline ``encoder.1828.rknn`` and returns the full
        transcript in one shot; ``True`` uses the streaming
        ``encoder_online.1828.rknn`` and runs the LLM once per
        ``online_chunk_seconds`` window.

        ``on_asr_commit`` — *(online only, optional)* ``Callable``
        invoked after each round as
        ``on_asr_commit(committed_text, unfixed_text, language)``.
        ``committed_text`` is the text confirmed by the rollback
        policy so far; ``unfixed_text`` may still be revised by the
        next round.

        **Incremental mode** — pass ``asr_stream_id=<id>`` in
        ``overrides`` to turn the call into one step of a long-lived
        streaming session (see :meth:`_run_online_stream`). The session
        state then survives across calls, and the call returns the
        transcript confirmed so far rather than a final transcript.

        Returns ``"language <lang>\\n<text>"`` for both modes.
        """
        self.sink.reset()
        audio = _pcm_to_mono_float32(pcm_bytes, n_channels, sample_width)

        stream_id = overrides.pop(ASR_STREAM_ID_KEY, None)
        if stream_id is not None:
            # The incremental path owns the resampling: a block-local
            # resample would land on a different phase than the offline
            # one and the two would not agree token-for-token.
            final = bool(overrides.pop(ASR_STREAM_FINAL_KEY, False))
            return self._run_online_stream(
                audio, sample_rate, str(stream_id), final,
                on_asr_commit, overrides,
            )

        if sample_rate != TARGET_SAMPLE_RATE:
            audio = _resample_linear(audio, sample_rate, TARGET_SAMPLE_RATE)
        if len(audio) < N_FFT:
            raise RuntimeError(f"audio too short: {len(audio)} samples (< {N_FFT})")

        if self._use_online:
            return self._run_online(audio, on_asr_commit, overrides)
        return self._run_offline(audio, overrides)

    # ------------------------------------------------------------------ #
    def _run_offline(self, audio: np.ndarray, overrides: Dict[str, Any]) -> str:
        """Single-shot path; mirrors ``cpp/main.cc::inference_qwen3_asr_model``."""
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
        # Token 151704 is in the GGUF control-token list but not in HF's
        # "special tokens" whitelist, so it leaks out as ``[PAD151704]``
        # glued to the language header. Swap for ``\n`` so callers see
        # a clean split.
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

    # ------------------------------------------------------------------ #
    # Streaming session plumbing
    # ------------------------------------------------------------------ #
    def _new_stream_state(self, stream_id: str, overrides: Dict[str, Any]) -> _OnlineStreamState:
        """Create a session; the streaming knobs are frozen here.

        The knobs are read **once**, when the session starts.  Changing
        ``online_chunk_seconds`` mid-session would move the block
        boundaries and desynchronise the accumulated context, so a later
        call that disagrees is ignored (with a warning) rather than
        silently corrupting the transcript.
        """
        chunk_seconds = float(overrides.get(
            "online_chunk_seconds", ONLINE_CHUNK_SECONDS_DEFAULT,
        ))
        max_new_tokens = int(overrides.get(
            "online_max_new_tokens", ONLINE_MAX_NEW_TOKENS_DEFAULT,
        ))
        rollback_tokens = int(overrides.get(
            "online_rollback_tokens", ONLINE_ROLLBACK_TOKENS_DEFAULT,
        ))
        rollback_rounds = int(overrides.get(
            "online_rollback_rounds", ONLINE_ROLLBACK_ROUNDS_DEFAULT,
        ))
        samples_per_chunk = max(N_FFT, int(TARGET_SAMPLE_RATE * chunk_seconds))
        now = _time.time()
        return _OnlineStreamState(
            stream_id=stream_id,
            samples_per_chunk=samples_per_chunk,
            max_new_tokens=max_new_tokens,
            rollback_tokens=rollback_tokens,
            rollback_rounds=rollback_rounds,
            chunk_seconds=chunk_seconds,
            decoder=_OnlineDecoder(self.tokenizer),
            origin_sec=float(overrides.get(ASR_STREAM_ORIGIN_SEC_KEY, 0.0) or 0.0),
            created_at=now,
            last_seen=now,
        )

    def _gc_streams(self) -> None:
        """Drop sessions that have been idle for longer than the TTL."""
        if not self._online_streams:
            return
        now = _time.time()
        dead = [
            sid for sid, st in self._online_streams.items()
            if now - st.last_seen > self._stream_ttl_sec
        ]
        for sid in dead:
            st = self._online_streams.pop(sid, None)
            if st is not None:
                LOG.info(
                    "asr stream %s reaped after %.1fs idle (ttl=%.1fs)",
                    sid, now - st.last_seen, self._stream_ttl_sec,
                )

    def reset_stream(self, stream_id: str) -> None:
        """Forget a streaming session (explicit teardown)."""
        self._online_streams.pop(str(stream_id), None)

    def _stream_text(self, state: _OnlineStreamState) -> str:
        """Current transcript of a session, in the ``run_audio`` shape."""
        lang = state.decoder.language or ""
        body = state.decoder.committed or ""
        if not lang:
            return body
        return f"language {lang}\n{body}"

    # ------------------------------------------------------------------ #
    # Online path
    # ------------------------------------------------------------------ #
    def _run_online(
        self,
        audio: np.ndarray,
        on_asr_commit: Optional[Any],
        overrides: Dict[str, Any],
    ) -> str:
        """Streaming path; mirrors ``cpp/online_main.cc::main``.

        The whole buffer arrives in one call, so the session lives for
        the duration of this method — but it is the *same* session object
        and the *same* per-block routine the incremental path uses, so
        the two produce identical transcripts for identical audio.
        """
        chunk_seconds = float(overrides.pop(
            "online_chunk_seconds", ONLINE_CHUNK_SECONDS_DEFAULT,
        ))
        max_new_tokens = int(overrides.pop(
            "online_max_new_tokens", ONLINE_MAX_NEW_TOKENS_DEFAULT,
        ))
        rollback_tokens = int(overrides.pop(
            "online_rollback_tokens", ONLINE_ROLLBACK_TOKENS_DEFAULT,
        ))
        rollback_rounds = int(overrides.pop(
            "online_rollback_rounds", ONLINE_ROLLBACK_ROUNDS_DEFAULT,
        ))

        samples_per_chunk = max(N_FFT, int(TARGET_SAMPLE_RATE * chunk_seconds))
        n_chunks = (len(audio) + samples_per_chunk - 1) // samples_per_chunk
        if n_chunks <= 0:
            raise RuntimeError("audio too short for online streaming")

        state = self._new_stream_state("<file>", {
            "online_chunk_seconds": chunk_seconds,
            "online_max_new_tokens": max_new_tokens,
            "online_rollback_tokens": rollback_tokens,
            "online_rollback_rounds": rollback_rounds,
        })

        for chunk_idx in range(n_chunks):
            start = chunk_idx * samples_per_chunk
            end = min(start + samples_per_chunk, len(audio))
            self._run_online_block(
                state,
                audio[start:end],
                is_last=(chunk_idx == n_chunks - 1),
                chunk_start_sec=state.chunk_start_sec,
                on_asr_commit=on_asr_commit,
            )

        final_text = state.decoder.final_text(
            state.committed_token_ids, self.tokenizer)
        self.sink.publish(_final_payload(
            language=state.decoder.language,
            text=final_text,
        ))
        return final_text

    def _run_online_block(
        self,
        state: _OnlineStreamState,
        chunk_audio: np.ndarray,
        *,
        is_last: bool,
        chunk_start_sec: float,
        on_asr_commit: Optional[Any],
    ) -> None:
        """Run **one** streaming block against the session's context.

        This is the shared core of the file-driven and incremental paths:
        mel → encoder → context assembly → LLM → rollback → publish.
        Keeping them on one routine is what makes "incremental N blocks"
        and "file N blocks" produce the same transcript.
        """
        # 一块音频至少要能凑出 **2 帧** mel（``N_FFT + HOP_LENGTH``）。
        #
        # 只补到 ``N_FFT`` 是不够的：``_log_mel_spectrogram`` 按 C 的语义
        # 把音频裁到 ``n_frame * hop_length``（1 帧 → 160 采样），而反射
        # 填充需要 ``n_fft // 2``（200）采样 —— 短于它就会踩到
        # ``_reflect_pad`` 的越界（见那里的注释）。
        #
        # 什么时候会拿到这么短的块？**会话的最后一块**：末块的长度 = 音频
        # 总长 mod 块长，只要总长没落在块网格上，末块就可能只有几十个
        # 采样（97 分钟真人访谈视频的末块：196 采样 = 12 ms）。
        min_block = N_FFT + HOP_LENGTH
        if len(chunk_audio) < min_block:
            chunk_audio = np.pad(
                chunk_audio, (0, min_block - len(chunk_audio))
            )
        chunk_mel = _log_mel_spectrogram(chunk_audio, self._mel_filters)
        chunk_embeds = self._run_audio_encoder(chunk_mel)
        state.audio_embed_history.append(chunk_embeds)

        committed_segment = (
            self.embeds[state.committed_token_ids].astype(np.float16)[None, ...]
            if state.committed_token_ids
            else np.zeros((1, 0, self._embeds_dim), dtype=np.float16)
        )
        full = np.concatenate(
            [
                self._prefix_embeds[None, ...],
                np.concatenate(state.audio_embed_history, axis=0)[None, ...],
                self._postfix_embeds[None, ...],
                committed_segment,
            ],
            axis=1,
        ).astype(np.float16)

        self.sink.reset()
        ret, _ = self.rknn_llm.session_run(
            embeds=full, keep_history=0, max_new_tokens=state.max_new_tokens,
        )
        if ret != 0:
            raise RuntimeError(
                f"llm session_run failed at chunk {state.chunk_idx}: ret={ret}"
            )

        round_tokens = list(self.sink.tokens)

        # rollback policy — mirrors cpp/online_main.cc:482-496
        if is_last:
            stable_count = len(round_tokens)
        elif state.chunk_idx < state.rollback_rounds:
            stable_count = 0
        elif len(round_tokens) > state.rollback_tokens:
            stable_count = len(round_tokens) - state.rollback_tokens
        else:
            stable_count = 0

        new_stable = round_tokens[:stable_count]
        state.committed_token_ids = state.committed_token_ids + new_stable
        new_unfixed = round_tokens[stable_count:]

        _, committed, unfixed = state.decoder.add(new_stable, new_unfixed)
        if on_asr_commit is not None:
            on_asr_commit(committed, unfixed, state.decoder.language)

        # Publish the round through the **sink**, which is the SDK's
        # one streaming channel. That is what makes the incremental
        # output survive the process boundary: ``LLMLocalSession``
        # registers the caller's ``on_delta`` as a sink observer, and
        # ``wdndds`` forwards observer calls over its RELIABLE delta
        # side topic. Publishing here (rather than only through the
        # ``on_asr_commit`` callback) means the *same* code path
        # serves an in-process caller and a remote worker.
        self.sink.publish(_round_payload(
            round=state.chunk_idx + 1,
            committed=committed,
            unfixed=unfixed,
            language=state.decoder.language,
            chunk_start_sec=chunk_start_sec,
            chunk_samples=int(len(chunk_audio)),
        ))
        state.chunk_idx += 1
        state.samples_consumed += int(len(chunk_audio))

    def _run_online_stream(
        self,
        audio: np.ndarray,
        sample_rate: int,
        stream_id: str,
        final: bool,
        on_asr_commit: Optional[Any],
        overrides: Dict[str, Any],
    ) -> str:
        """One step of an incremental streaming session.

        Contract
        --------
        * **Block boundaries.** The backend cuts the audio on the 16 kHz
          grid; the caller may hand over any amount of audio it likes
          and the backend buffers the remainder.  The boundaries are
          therefore identical to the file-driven path's — which is what
          makes the two transcripts identical.
        * **The last block.** The call carrying ``asr_stream_final=True``
          must carry the session's **last** block.  The last block is
          not rolled back (mirrors ``is_last`` in the C demo), so it has
          to be the one that knows it is the last.  A final call that
          carries no new block therefore leaves the last
          ``rollback_tokens`` tokens unconfirmed — the caller is warned.
        * **Timeline.** ``asr_stream_origin_sec`` is the position of the
          session's first sample on the caller's timeline; it is frozen
          when the session starts.  Every block's start is derived from
          it, so the caller never has to track the position itself.
          Seeking = a new session (new id) with a new origin.
        """
        self._gc_streams()
        state = self._online_streams.get(stream_id)
        if state is None:
            state = self._new_stream_state(stream_id, overrides)
            self._online_streams[stream_id] = state
        else:
            # The knobs are frozen at session start; ignore later drift.
            for key, frozen in (
                ("online_chunk_seconds", state.chunk_seconds),
                ("online_max_new_tokens", state.max_new_tokens),
                ("online_rollback_tokens", state.rollback_tokens),
                ("online_rollback_rounds", state.rollback_rounds),
                (ASR_STREAM_ORIGIN_SEC_KEY, state.origin_sec),
            ):
                if key in overrides and overrides[key] != frozen:
                    LOG.warning(
                        "asr stream %s: ignoring %s=%s (frozen at %s)",
                        stream_id, key, overrides[key], frozen,
                    )
        state.last_seen = _time.time()

        # --- resample. The streaming resampler keeps the phase aligned
        #     with the offline one; a block-local resample would not.
        if sample_rate == TARGET_SAMPLE_RATE:
            block_audio = audio.astype(np.float32, copy=False)
        else:
            if state.resampler is None:
                state.resampler = _StreamResampler(sample_rate, TARGET_SAMPLE_RATE)
            block_audio = state.resampler.push(audio)

        if final and state.resampler is not None:
            extra = state.resampler.flush()
            if extra.size:
                block_audio = (
                    np.concatenate([block_audio, extra]) if block_audio.size
                    else extra
                )

        # --- cut into whole blocks ---
        spc = state.samples_per_chunk
        if state.pending.size:
            block_audio = np.concatenate([state.pending, block_audio])
            state.pending = np.zeros((0,), dtype=np.float32)
        n_blocks = int(block_audio.size // spc)
        remainder = block_audio[n_blocks * spc:]
        #: A trailing partial block means the session's last block is the
        #: **remainder**, not the last whole block.  Marking both as "last"
        #: would skip the rollback on a block that still has a successor —
        #: and that is exactly how the file-driven path splits the same
        #: audio (the last block is the short one).
        has_tail = remainder.size > 0

        for i in range(n_blocks):
            blk = block_audio[i * spc:(i + 1) * spc]
            # In the final call the *last executed* block is the session's
            # last one — which is the remainder when there is one.
            is_last = final and (i == n_blocks - 1) and not has_tail
            self._run_online_block(
                state, blk, is_last=is_last,
                chunk_start_sec=state.chunk_start_sec,
                on_asr_commit=on_asr_commit,
            )
        state.pending = remainder

        if not final:
            return self._stream_text(state)

        # --- final: the remainder (if any) is the session's last block ---
        if has_tail:
            tail = remainder
            state.pending = np.zeros((0,), dtype=np.float32)
            self._run_online_block(
                state, tail, is_last=True,
                chunk_start_sec=state.chunk_start_sec,
                on_asr_commit=on_asr_commit,
            )
        elif n_blocks == 0:
            LOG.warning(
                "asr stream %s: final call carried no audio; the last "
                "%d token(s) stay unconfirmed (the final call must carry "
                "the session's last block)",
                stream_id, state.rollback_tokens,
            )

        final_text = state.decoder.final_text(
            state.committed_token_ids, self.tokenizer)
        self.sink.publish(_final_payload(
            language=state.decoder.language,
            text=final_text,
        ))
        self._online_streams.pop(stream_id, None)
        return final_text

    # Stub fall-throughs — ``LLMSession._dispatch`` probes for these
    # names when it can't route to a more specific method (audio /
    # image / video). Keep them as informative errors rather than
    # relying on AttributeError.
    def run_text(self, prompt: str, **kw: Any) -> str:
        raise RuntimeError(
            "ASRBackend.run_text(): pass AudioAttachment(wav_bytes) "
            "to LLMSession.respond() and the SDK will route to run_audio()."
        )

    def run_images(self, prompt: str, images: list, **kw: Any) -> str:
        raise RuntimeError("ASR backend does not support images.")

    def run_video(self, prompt: str, frames: list, **kw: Any) -> str:
        raise RuntimeError("ASR backend does not support video.")

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        # Streaming sessions hold the audio embedding history; drop them
        # with the context so a released backend leaves nothing behind.
        self._online_streams.clear()
        for r in (getattr(self, "rknn_audio", None),
                  getattr(self, "rknn_llm", None)):
            if r is None:
                continue
            try:
                r.release()
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# Streaming payload — one JSON object per streaming round.
#
# The SDK's one streaming channel is the ``StreamSink`` observer list, and
# ``wdndds`` forwards observer calls over its RELIABLE delta side topic.
# The payload therefore has to be **text** — it is the same channel the
# chat models stream plain tokens over. For ASR the observable unit is a
# *round*, not a token, and a round carries two pieces of text (the
# confirmed prefix and the still-speculative tail), so the payload is a
# JSON object rather than a bare string.
#
# Consumers that only want the plain transcript can ignore the JSON and
# use the final ``text`` field / the RPC reply.
# --------------------------------------------------------------------------- #
def _language_of(head: str) -> str:
    """``"language English"`` → ``"English"``; anything else → ``""``."""
    head = (head or "").strip()
    if head.lower().startswith("language "):
        return head[len("language "):].strip()
    return head


def _round_payload(
    *,
    round: int,
    committed: str,
    unfixed: str,
    language: str,
    chunk_start_sec: float = 0.0,
    chunk_samples: int = 0,
) -> str:
    """Serialise one streaming round for the delta channel.

    ``chunk_start_sec`` / ``chunk_samples`` are **additive** fields: they
    tell the consumer where on the timeline this round's audio sits, so
    a caller can align the transcript with a media player.  A consumer
    that predates them simply ignores the two keys.
    """
    return json.dumps({
        "type": "asr_round",
        "round": int(round),
        "chunk_start_sec": float(chunk_start_sec),
        "chunk_samples": int(chunk_samples),
        "committed": committed,
        "unfixed": unfixed,
        "language": language,
    }, ensure_ascii=False)


def _final_payload(*, language: str, text: str) -> str:
    """Serialise the closing payload (the complete transcript)."""
    return json.dumps({
        "type": "asr_final",
        "language": language,
        "text": text,
    }, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Online-mode text accumulator — mirrors ``decode_asr_text`` /
# ``round_idx`` bookkeeping in cpp/online_main.cc. Splits each round
# into a "stable" prefix (confirmed by the rollback policy) and a
# "unfixed" tail (still subject to revision next round), and stitches
# the language header once the 151704 separator lands.
#
# Important: because the LLM session_run input keeps the same
# ChatML prefix every round, the LLM re-emits ``language
# <lang>[PAD151704]`` from every round. ``add`` strips that prefix
# from whichever side carries it. The final ``language`` field is
# recovered in ``final_text`` from the full committed-token stream
# (more accurate than tracking per-round — early rounds' guess is
# often wrong because the audio context is too short).
# --------------------------------------------------------------------------- #
class _OnlineDecoder:
    def __init__(self, tokenizer):
        self.tk = tokenizer
        self.committed = ""
        self.unfixed = ""
        self.text_started = False  # has the 151704 separator been seen yet?
        #: Detected language ("English", "Chinese", …). Filled in when the
        #: ``151704`` separator first lands — the model emits
        #: ``language <lang>[PAD151704]`` before the transcript body.
        self.language = ""

    def add(
        self,
        new_stable_tokens: List[int],
        new_unfixed_tokens: List[int],
    ):
        """Fold a round's tokens in.

        Returns ``(delta, committed, unfixed)`` for streaming loggers:
        ``delta`` is the text newly committed in this round,
        ``committed`` is the total committed text so far, ``unfixed``
        is the tail still subject to revision.
        """
        stable_text = self._decode(new_stable_tokens)
        unfixed_text = self._decode(new_unfixed_tokens)

        if not self.text_started:
            # First time the 151704 separator lands: extract the
            # language header from whichever side carries it, drop the
            # other half, and remember to strip the prefix from later
            # rounds.
            sep = "[PAD151704]"
            for src, is_stable in ((stable_text, True), (unfixed_text, False)):
                idx = src.find(sep)
                if idx < 0:
                    continue
                tail = src[idx + len(sep):]
                if is_stable:
                    stable_text = tail
                    unfixed_text = ""
                else:
                    stable_text = ""
                    unfixed_text = tail
                self.language = _language_of(src[:idx])
                self.text_started = True
                break
        else:
            # The LLM re-emits the ``language <lang>[PAD151704]`` prefix
            # every round. Strip it from whichever side has it so the
            # committed accumulator only sees the actual content.
            sep = "[PAD151704]"
            for attr in ("stable", "unfixed"):
                t = stable_text if attr == "stable" else unfixed_text
                idx = t.find(sep)
                if idx < 0:
                    continue
                body = t[idx + len(sep):]
                if attr == "stable":
                    stable_text = body
                else:
                    unfixed_text = body
                # Later rounds see more audio, so their language guess is
                # better than the first one — keep the newest.
                self.language = _language_of(t[:idx])

        delta = stable_text
        self.committed += delta
        self.unfixed = unfixed_text
        return delta, self.committed, self.unfixed

    def final_text(self, committed_token_ids: List[int], tokenizer) -> str:
        """Decode the full committed-token stream and split on 151704.

        Returned as ``"language <lang>\\n<text>"`` (matches the
        offline ``ASRBackend.run_audio`` shape so ``examples/audio_to_text.py``
        doesn't need separate handling).
        """
        if not committed_token_ids:
            return ""
        # ``skip_special_tokens=True`` — control tokens render as zero bytes
        # in the C demo (llama.cpp ``TokenToPiece``), so they must not leak
        # into the transcript here either.
        full = tokenizer.decode(committed_token_ids, skip_special_tokens=True)
        sep = "[PAD151704]"
        idx = full.find(sep)
        if idx < 0:
            return full
        head = full[:idx].strip()
        body = full[idx + len(sep):]
        if head.lower().startswith("language "):
            head = head[len("language "):].strip()
        return f"language {head}\n{body}".rstrip("\n")

    def _decode(self, tokens: List[int]) -> str:
        if not tokens:
            return ""
        # ``skip_special_tokens=True`` mirrors llama.cpp's ``TokenToPiece``,
        # which renders control tokens (EOS ``<|im_end|>``, …) as *zero
        # bytes*. Leaving them in leaked a trailing ``<|im_end|>`` into the
        # transcript and into every streamed "unfixed" fragment. The
        # ``[PAD151704]`` separator is *not* a special token, so it still
        # survives and the split below keeps working.
        return self.tk.decode(tokens, skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# Result callback — builds tokens into StreamSink as the LLM streams.
# --------------------------------------------------------------------------- #
# LLMCallState values (mirroring rknn3lite's enum). Kept here as plain
# ints so this module doesn't import the (now internal) shim IntEnum.
_LLM_RUN_NORMAL = 0
_LLM_RUN_WAITING = 1
_LLM_RUN_FINISH = 2
_LLM_RUN_STOP = 3
_LLM_RUN_MAX_NEW_TOKEN_REACHED = 4
_LLM_RUN_ERROR = 5


def _make_result_callback(b: "ASRBackend"):
    """Build the user-side ``result_cb`` callable for the ASR LLM.

    Signature matches :func:`wdnai._librknn.build_llm_callback`:
    ``(userdata, token_ids: List[int], state: int) -> int``. The shim
    auto-pins ``b`` as userdata so we receive the backend instance
    here.
    """
    def _cb(_userdata, new_tokens, state):
        s = int(state)
        if s == _LLM_RUN_ERROR:
            b.sink.error = "inference error"
            return 0
        if s == _LLM_RUN_WAITING:
            return 0
        if s in (
            _LLM_RUN_FINISH,
            _LLM_RUN_STOP,
            _LLM_RUN_MAX_NEW_TOKEN_REACHED,
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
        if s == _LLM_RUN_NORMAL:
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

    return _cb


def _decode_safe(tokenizer, tokens: List[int]) -> str:
    text = tokenizer.decode(
        tokens, skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return text.split("\ufffd", 1)[0] if "\ufffd" in text else text