"""Qwen3-TTS backend — text-in, audio-out.

Wraps the upstream ``qwen3_tts_demo.Qwen3TTSDemo`` shipped with
``rknn3-toolkit-lite`` and exposes it through the v2 ``LLMSession``
surface. ``run_text(prompt)`` returns the prompt unchanged along
with a single :class:`Attachment` carrying the synthesised WAV bytes —
the SDK then assigns that attachment to ``LMOutput.attachment`` so the
caller sees a unified output object.

Model layout (on disk, matches ``qwen3_tts_demo``):

    <model_dir>/
        text_projection.1828.rknn + .weight
        talker.1828.rknn          + .weight
        code_predictor.1828.rknn  + .weight
        speech_decoder.1828.rknn  + .weight
        talker_text_embed.fp16.bin
        talker_input_embed.fp16.bin
        codec_embed.fp16.bin
        tokenizer.json
        qwen3tts_speaker_embed.h          # optional, for VoiceClone

The directory is resolved from the ``ModelSpec`` in this order:
``spec.tts_model_dir`` → ``spec.options["model_dir"]`` →
``$TTS_MODEL_DIR`` → ``$RKAI_MODELS_DIR/Qwen3-TTS`` → ``/userdata/models/Qwen3-TTS``.

Optional overrides (forwarded from ``LLMSession.respond(**overrides)``):

* ``ref_speaker``   — voice-clone speaker name (``"girl_base"`` / ``"ahu"``).
* ``language``      — ``"auto"`` / ``"chinese"`` / ``"english"`` / …
* ``speaker``       — built-in SPK_ID entry (``serena`` / ``vivian`` / …).
* ``volume``        — 0.0–1.0 (applied to the live PCM by the upstream
                      ``AudioPlayer``).
* ``no_play``       — skip live playback; only produce WAV bytes.
* ``instruct``      — optional CustomVoice instruct text.
* ``non_streaming_mode`` — 1 to use non-streaming prefill.
* ``on_step``       — ``callable(n_frames, audio_sec)`` invoked per
                      decoder window for real-time progress reporting.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, List, Tuple

from ..attachment import Attachment

LOG = logging.getLogger("rkai.backends.tts")


def _resolve_tts_model_dir(spec) -> Path:
    """Find the Qwen3-TTS model directory from a ``ModelSpec``."""
    explicit = getattr(spec, "tts_model_dir", None)
    if explicit:
        return Path(explicit)
    opts = spec.options or {}
    if "model_dir" in opts:
        return Path(opts["model_dir"])
    env = os.environ.get("TTS_MODEL_DIR")
    if env:
        return Path(env)
    root = os.environ.get("RKAI_MODELS_DIR", "/userdata/models")
    return Path(root) / "Qwen3-TTS"


class TTSBackend:
    """TTS backend — text-in, audio-out.

    Wraps the upstream ``Qwen3TTSDemo`` shipped with
    ``rknn3-toolkit-lite``. ``run_text(prompt)`` returns the prompt
    unchanged plus a single :class:`Attachment` carrying the
    synthesised WAV bytes — the SDK then assigns it to
    ``LMOutput.attachment`` so the v2 caller sees a unified output.
    """

    def __init__(self, spec):
        # Lazy: the upstream module pulls in rknn3lite + numpy + the C
        # rknn3 runtime, which we don't want to pay for in a text/VLM run.
        #
        # The upstream demo lives outside this wheel (under
        # ``~/Desktop/rknn3/rknn3-toolkit/rknn3-toolkit-lite/examples/Qwen3-TTS``
        # or ``~/Desktop/rknn3_Qwen3_TTS_demo`` on a stock board image);
        # we search for it in order:
        #   1. ``$RKAI_TTS_DEMO_DIR``
        #   2. walking up from the model dir
        #   3. walking up from this file
        #   4. ``$HOME/Desktop/rknn3/.../Qwen3-TTS`` for each well-known user
        import sys as _sys
        model_dir = _resolve_tts_model_dir(spec)
        candidates: list[str] = []
        env_override = os.environ.get("RKAI_TTS_DEMO_DIR")
        if env_override:
            candidates.append(env_override)

        def _is_qwen3_tts_demo(d: Path) -> bool:
            try:
                return d.is_dir() and (d / "qwen3_tts_demo.py").is_file()
            except (PermissionError, OSError):
                return False

        # (2) walk up from model_dir
        d = Path(model_dir).resolve()
        for _ in range(6):
            d = d.parent
            for cand in (
                d / "rknn3-toolkit" / "rknn3-toolkit-lite" / "examples" / "Qwen3-TTS",
                d / "Qwen3-TTS",
                d / "rknn3_Qwen3_TTS_demo",
            ):
                if _is_qwen3_tts_demo(cand):
                    candidates.append(str(cand))
                    break
            if candidates:
                break

        # (3) walk up from this file (rkaicore sits deep inside .venv)
        d = Path(__file__).resolve()
        for _ in range(8):
            d = d.parent
            for cand in (
                d / "rknn3-toolkit" / "rknn3-toolkit-lite" / "examples" / "Qwen3-TTS",
                d / "Qwen3-TTS",
                d / "rknn3_Qwen3_TTS_demo",
            ):
                if _is_qwen3_tts_demo(cand):
                    candidates.append(str(cand))
                    break
            if candidates:
                break

        # (4) $HOME + (5) hardcoded fallbacks
        home = Path.home()
        for h in (home, Path("/root"), Path("/home/linaro"), Path("/home/pi")):
            for cand in (
                h / "Desktop" / "rknn3" / "rknn3-toolkit" / "rknn3-toolkit-lite" / "examples" / "Qwen3-TTS",
                h / "Desktop" / "rknn3_Qwen3_TTS_demo",
            ):
                if _is_qwen3_tts_demo(cand):
                    candidates.append(str(cand))

        print(f"[TTSBackend] qwen3_tts_demo search candidates:", flush=True)
        for d in candidates:
            print(f"  {d}: {'YES' if _is_qwen3_tts_demo(Path(d)) else 'NO'}", flush=True)

        for d in candidates:
            if d and d not in _sys.path and _is_qwen3_tts_demo(Path(d)):
                _sys.path.insert(0, d)
        print(f"[TTSBackend] final sys.path[0:5]={_sys.path[:5]}", flush=True)

        try:
            import qwen3_tts_demo as _qwen3_tts_demo  # noqa: E402
        except ImportError as e:  # pragma: no cover - host-specific
            raise RuntimeError(
                f"TTSBackend requires the upstream qwen3_tts_demo module; "
                f"not importable: {e}. Set RKAI_TTS_DEMO_DIR to the directory "
                f"containing qwen3_tts_demo.py. Tried: {candidates}"
            )

        self.spec = spec
        self._qwen3_tts_demo = _qwen3_tts_demo
        self.model_dir = _resolve_tts_model_dir(spec)
        if not self.model_dir.is_dir():
            raise FileNotFoundError(
                f"TTS model directory not found: {self.model_dir}"
            )

        # Inject the .1828.rknn path adapter + progress hooks so the
        # upstream's hard-coded ``<role>.rknn`` filenames resolve to the
        # on-disk ``<role>.1828.rknn`` naming used elsewhere on this
        # board, and so ``run_text`` produces an (prompt, attachment) pair
        # that fits the v2 ``LLMSession`` contract.
        self._install_path_adapter(_qwen3_tts_demo.Qwen3TTSDemo)
        self._install_run_text(_qwen3_tts_demo.Qwen3TTSDemo)

        self.demo = _qwen3_tts_demo.Qwen3TTSDemo(model_dir=str(self.model_dir))
        LOG.info("TTSBackend initialised; model_dir=%s", self.model_dir)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _install_path_adapter(cls) -> None:
        """Rewire the upstream ``init_*`` methods so ``<role>.rknn``
        resolves to the on-disk ``<role>.1828.rknn`` naming used here.
        """
        if getattr(cls, "_rkai_paths_patched", False):
            return
        from qwen3_tts_demo import (
            TEXT_PROJECTION_CORE_MASK,
            QWEN3_TTS_MAX_CONTEXT_LEN,
            TALKER_VOCAB_SIZE,
            CODEC_MAX_CONTEXT_LEN,
            CODEC_VOCAB_SIZE,
            CODEC_EOS_TOKEN_ID,
        )
        import qwen3_tts_demo as _qtd
        _wrap = _qtd._wrap_userdata

        def _resolve_rknn(model_dir, role):
            rknn_path = model_dir / f"{role}.1828.rknn"
            weight_path = model_dir / f"{role}.weight"
            return rknn_path, weight_path

        def _init_text_projection(self):
            from rknn3lite.api import RKNN3Lite
            print('[TTSBackend] _init_text_projection called (rkai patched)', flush=True)
            rknn_path, weight_path = _resolve_rknn(self.model_dir, "text_projection")
            self.text_projection = RKNN3Lite(verbose=self.verbose)
            ret = self.text_projection.load_rknn(str(rknn_path), str(weight_path))
            assert ret == 0, f"load text_projection failed ({rknn_path}): {ret}"
            ret = self.text_projection.init_runtime(
                target=self.target,
                core_mask=TEXT_PROJECTION_CORE_MASK,
                device_id=self.device_id,
            )
            assert ret == 0, "init text_projection runtime failed"
            # The upstream code reads these attributes for the prefill
            # chunked path; mirror the original logic so the rest of
            # ``build_prefill_embedding`` keeps working.
            attrs = self.text_projection.get_inputs_tensor_attr()
            if not attrs:
                raise RuntimeError("text_projection: no input tensor attributes")
            shape = [attrs[0].shape[i] for i in range(attrs[0].n_dims)]
            self.text_proj_seq_len = int(shape[1])
            self.text_proj_in_dim = int(shape[2])
            print(f"[TTSBackend] _init_text_projection DONE: text_proj_seq_len={self.text_proj_seq_len} text_proj_in_dim={self.text_proj_in_dim}", flush=True)

        def _init_speech_decoder(self):
            from rknn3lite.api import RKNN3Lite
            rknn_path, weight_path = _resolve_rknn(self.model_dir, "speech_decoder")
            self.speech_decoder = RKNN3Lite(verbose=self.verbose)
            ret = self.speech_decoder.load_rknn(str(rknn_path), str(weight_path))
            assert ret == 0, f"load speech_decoder failed ({rknn_path}): {ret}"
            ret = self.speech_decoder.init_runtime(
                target=self.target, core_mask=self.core_mask, device_id=self.device_id,
            )
            assert ret == 0, "init speech_decoder runtime failed"
            self.speech_decoder_window = 50

        def _init_code_predictor(self):
            from rknn3lite.api import (
                RKNN3Lite, RKLLMCallback, LLMResultCallback, LLMGetEmbedCallback,
            )
            rknn_path, weight_path = _resolve_rknn(self.model_dir, "code_predictor")
            self.code_predictor = RKNN3Lite(llm_mode=True, verbose=self.verbose)
            ret = self.code_predictor.load_rknn(str(rknn_path), str(weight_path))
            assert ret == 0, f"load code_predictor failed ({rknn_path}): {ret}"
            cb = RKLLMCallback()
            cb.result_callback = LLMResultCallback(self._cp_result_callback)
            cb.result_userdata = _wrap(self, "_cp_userdata")
            cb.embed_callback = LLMGetEmbedCallback(self._cp_embed_callback)
            cb.embed_userdata = _wrap(self, "_cp_embed_userdata")
            args = [{
                "max_context_len": CODEC_MAX_CONTEXT_LEN,
                "vocab_size":      CODEC_VOCAB_SIZE,
                "special_eos_id":  CODEC_EOS_TOKEN_ID,
                "max_new_tokens":  CODEC_MAX_CONTEXT_LEN,
                "keep_history":    0,
                **self.code_predictor_sampling,
            }]
            ret = self.code_predictor.init_runtime(
                target=self.target, core_mask=self.core_mask, device_id=self.device_id,
                llm_args=args, llm_callback=cb,
            )
            assert ret == 0, "init code_predictor runtime failed"

        def _init_talker(self):
            from rknn3lite.api import (
                RKNN3Lite, RKLLMCallback, LLMResultCallback,
                LLMGetEmbedCallback, LLMOutputCallback,
            )
            rknn_path, weight_path = _resolve_rknn(self.model_dir, "talker")
            self.talker = RKNN3Lite(llm_mode=True, verbose=self.verbose)
            ret = self.talker.load_rknn(str(rknn_path), str(weight_path))
            assert ret == 0, f"load talker failed ({rknn_path}): {ret}"
            ret = self.talker.init_runtime(
                target=self.target, core_mask=self.core_mask, device_id=self.device_id,
            )
            assert ret == 0, "init talker runtime failed"
            output_tensors, n_outputs = self.talker.create_output_tensors([0, 1])
            self._talker_output_tensors = output_tensors
            cb = RKLLMCallback()
            cb.result_callback = LLMResultCallback(self._talker_result_callback)
            cb.result_userdata = _wrap(self, "_talker_result_userdata")
            cb.embed_callback = LLMGetEmbedCallback(self._talker_embed_callback)
            cb.embed_userdata = _wrap(self, "_talker_embed_userdata")
            cb.output_callback = LLMOutputCallback(self._talker_output_callback)
            cb.output_userdata = _wrap(self, "_talker_output_userdata")
            cb.output_tensors = output_tensors
            cb.n_output_tensors = n_outputs
            args = [{
                "max_context_len": QWEN3_TTS_MAX_CONTEXT_LEN,
                "vocab_size":      TALKER_VOCAB_SIZE,
                "special_eos_id":  CODEC_EOS_TOKEN_ID,
                "max_new_tokens":  QWEN3_TTS_MAX_CONTEXT_LEN,
                "keep_history":    0,
                **self.talker_sampling,
            }]
            ret = self.talker.init_llm_session(llm_args=args, llm_callback=cb)
            assert ret == 0, f"init talker llm session failed: {ret}"

        cls.init_text_projection = _init_text_projection
        cls.init_speech_decoder  = _init_speech_decoder
        cls.init_code_predictor  = _init_code_predictor
        cls.init_talker           = _init_talker
        cls._rkai_paths_patched = True

    # ------------------------------------------------------------------ #
    @staticmethod
    def _install_run_text(cls) -> None:
        """Install ``run_text`` on the upstream demo that returns ``(prompt, attachments)``.

        The upstream demo's ``run()`` returns just the prompt string. We
        add a parallel ``run_text`` that runs the synthesiser, writes to a
        temp WAV, reads it back as bytes, and returns it as an
        :class:`Attachment` so ``LLMSession.respond`` can surface it via
        ``LMOutput.attachment``.

        For ``run_images`` / ``run_video`` we provide stub methods that fall
        back to plain text — TTS doesn't consume images.
        """
        if getattr(cls, "_rkai_run_text_patched", False):
            return
        import qwen3_tts_demo as _qtd

        # ``progress hooks``: install ``run_with_progress`` on the demo
        # if not already present (the upstream demo ships without it).
        cls.run_with_progress = getattr(cls, "run_with_progress", None)
        if cls.run_with_progress is None:
            # Same shape as ``app/services/tts_session.py``'s
            # ``_install_run_with_progress`` — see that file for a
            # detailed explanation.
            import queue as _queue
            import numpy as _np
            from qwen3_tts_demo import (
                kSampleRate, kFeatSize, kWindowLen, kLeftContextSize,
                kChunkSize, kTotalUpsample, _write_wav,
            )

            def run_with_progress(
                self, text, output_path,
                language="auto", speaker="", ref_speaker="",
                instruct="", non_streaming_mode=False,
                play=True, volume=0.5, audio_device=None,
                stop_event=None, on_step=None,
            ):
                self.load_resources()
                self.init_text_projection()
                self.init_speech_decoder()
                self.init_code_predictor()
                self.init_talker()
                self.audio_player = (
                    _qtd.AudioPlayer(kSampleRate, device=audio_device, volume=volume)
                    if play else None
                )
                try:
                    prefill = self.build_prefill_embedding(
                        text=text, instruct=instruct,
                        language=language, speaker=speaker,
                        ref_speaker=ref_speaker,
                        non_streaming_mode=non_streaming_mode,
                    )
                    audio = self._run_talker_streaming_with_progress(
                        prefill, stop_event=stop_event, on_step=on_step,
                    )
                    if audio.size == 0:
                        pass
                    else:
                        _write_wav(output_path, audio, kSampleRate)
                    return audio
                finally:
                    if self.audio_player is not None:
                        self.audio_player.close()
                        self.audio_player = None
                    self.release()

            def _run_talker_streaming_with_progress(self, prefill_embeds, stop_event=None, on_step=None):
                import queue as _q
                import numpy as _np
                import threading as _threading
                with self.lock:
                    self.talker_failed = False
                    self.talker_last_state = None
                    self.talker_steps = 0
                    self.collected_codes = []
                    self.past_hidden = None
                    self._first_window = True
                self.frame_queue = _q.Queue()
                self._streamed_audio = []
                self._talker_done_event = _threading.Event()

                def _decoder_worker_with_stop():
                    import numpy as _np
                    from qwen3_tts_demo import kSampleRate as _sr
                    def _get(timeout):
                        try: return self.frame_queue.get(timeout=timeout)
                        except _q.Empty: return None
                    buffer = []
                    prev_context = []
                    while len(buffer) < 50 and not (stop_event is not None and stop_event.is_set()):
                        frame = _get(0.1)
                        if frame is None:
                            if self._talker_done_event.is_set(): break
                            continue
                        buffer.append(frame)
                    if buffer:
                        codes = self.transcode_for_speech_decoder(buffer)
                        audio = self._decode_window_full(codes, first=True)
                        if audio.size:
                            self._streamed_audio.append(audio)
                            if self.audio_player is not None:
                                self.audio_player.play_chunk(audio)
                        if on_step is not None:
                            on_step(len(self.collected_codes),
                                    sum(a.size for a in self._streamed_audio) / _sr)
                        prev_context = buffer[-25:] if len(buffer) > 25 else buffer[:]
                        buffer = []
                    while not (stop_event is not None and stop_event.is_set()):
                        while len(buffer) < 25:
                            frame = _get(0.1)
                            if frame is None:
                                if self._talker_done_event.is_set() and self.frame_queue.empty(): break
                                continue
                            buffer.append(frame)
                        if not buffer: break
                        new_codes = self.transcode_for_speech_decoder(buffer)
                        ctx_codes = (
                            self.transcode_for_speech_decoder(prev_context)
                            if prev_context else None
                        )
                        audio = self._decode_window_with_context(ctx_codes, new_codes)
                        if audio.size:
                            self._streamed_audio.append(audio)
                            if self.audio_player is not None:
                                self.audio_player.play_chunk(audio)
                        if on_step is not None:
                            on_step(len(self.collected_codes),
                                    sum(a.size for a in self._streamed_audio) / _sr)
                        combined = (prev_context + buffer) if prev_context else buffer
                        prev_context = combined[-25:] if len(combined) > 25 else combined[:]
                        buffer = buffer[25:] if len(buffer) > 25 else []

                decoder_t = _threading.Thread(
                    target=_decoder_worker_with_stop,
                    name="qwen3tts-rkai", daemon=True,
                )
                decoder_t.start()
                ret, _ = self.talker.session_run_async(embeds=prefill_embeds)
                if ret != 0:
                    self._talker_done_event.set()
                    self.frame_queue.put(None)
                    raise RuntimeError(f"talker.session_run_async failed: ret={ret}")
                while not self._talker_done_event.is_set():
                    if stop_event is not None and stop_event.is_set():
                        try: self.talker.session_stop()
                        except Exception: pass
                        break
                    import time as _t; _t.sleep(0.02)
                if self.talker_failed:
                    self.frame_queue.put(None)
                    decoder_t.join(timeout=5)
                    raise RuntimeError(
                        f"talker generation failed, last_state={self.talker_last_state}"
                    )
                self.frame_queue.put(None)
                decoder_t.join(timeout=30)
                if not self._streamed_audio: return _np.zeros(0, dtype=_np.float32)
                return _np.concatenate(self._streamed_audio, axis=0)

            cls.run_with_progress = run_with_progress
            cls._run_talker_streaming_with_progress = _run_talker_streaming_with_progress

        # Now install run_text on top.
        def run_text(self, prompt, **overrides):
            """v2 SDK entry point: returns ``(prompt_str, [audio_attachment])``.

            Synthesises ``prompt`` using the upstream demo, writes the
            WAV to a temp file, reads it back as bytes, and packages it
            as an :class:`Attachment` (``kind="audio"``).
            """
            ref_speaker = overrides.get("ref_speaker", "girl_base")
            language = overrides.get("language", "auto")
            speaker = overrides.get("speaker", "")
            instruct = overrides.get("instruct", "")
            volume = overrides.get("volume", 0.5)
            no_play = bool(overrides.get("no_play", False))
            non_streaming = bool(overrides.get("non_streaming_mode", False))
            stop_event = overrides.get("stop_event", None)
            on_step = overrides.get("on_step", None)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
                tmp_path = fp.name
            try:
                audio = self.run_with_progress(
                    text=prompt,
                    output_path=tmp_path,
                    language=language, speaker=speaker, ref_speaker=ref_speaker,
                    instruct=instruct, non_streaming_mode=non_streaming,
                    play=not no_play, volume=volume, audio_device=None,
                    stop_event=stop_event, on_step=on_step,
                )
                if audio is None or audio.size == 0:
                    raise RuntimeError("TTS produced no audio")
                with open(tmp_path, "rb") as fr:
                    wav_bytes = fr.read()
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

            return prompt, [
                Attachment(wav_bytes, kind="audio", mime_type="audio/wav"),
            ]

        def run_images(self, prompt, images, **overrides):
            """TTS ignores images — fall back to plain text."""
            return self.run_text(prompt, **overrides)

        def run_video(self, prompt, frames, **overrides):
            """TTS ignores video frames — fall back to plain text."""
            return self.run_text(prompt, **overrides)

        cls.run_text = run_text
        cls.run_images = run_images
        cls.run_video = run_video
        cls._rkai_run_text_patched = True

    # ------------------------------------------------------------------ #
    def run_text(self, prompt: str, **overrides: Any) -> Tuple[str, List[Attachment]]:
        """Public entry point invoked by ``LLMSession._dispatch``.

        Drives the TTS pipeline via the upstream demo's
        ``run_text`` (patched in ``_install_run_text``) and returns
        ``(prompt_str, attachments_out)`` where ``attachments_out[0]``
        is an :class:`Attachment` carrying the WAV bytes.

        ``LLMSession._dispatch`` unpacks this tuple; downstream
        ``_parse_output`` sees the tuple and uses the
        ``attachments_out`` argument to populate ``LMOutput.attachment``.
        """
        # Forward to the upstream demo's ``run_text`` (patched in
        # ``_install_run_text``) which already returns
        # ``(prompt, [Attachment(audio_bytes)])``.
        result = self.demo.run_text(prompt, **overrides)
        # Normalise: upstream ``run_text`` returns ``(text, [att])``;
        # any other shape (a bare string, a model instance) is treated
        # as text-only with no attachments.
        if isinstance(result, tuple) and len(result) == 2:
            text, atts = result
            return text, list(atts or [])
        return result, []

    # ------------------------------------------------------------------ #
    def run_images(self, prompt: str, images: list, **overrides: Any) -> str:
        return self.run_text(prompt, **overrides)[0]

    def run_video(self, prompt: str, frames: list, **overrides: Any) -> str:
        return self.run_text(prompt, **overrides)[0]

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        try:
            self.demo.release()
        except Exception:  # noqa: BLE001
            pass
