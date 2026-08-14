"""Qwen3-TTS backend — text-in, audio-out.

The full RKNN3 driver (the four ``.1828.rknn`` models, embed tables,
prefill construction, talker/code_predictor/speech_decoder pipeline,
audio playback) is vendored in this module — originally imported
from ``rknn3-toolkit-lite/examples/Qwen3-TTS/qwen3_tts_demo.py``
via ``sys.path`` injection, which made the SDK un-importable on
hosts without the upstream demo directory. ``import wdnai.backends.tts``
is now self-contained.

``run_text(prompt)`` returns the prompt unchanged along with a
single :class:`Attachment` carrying the synthesised WAV bytes; the
SDK assigns that attachment to ``LMOutput.attachment``.

Model layout on disk (matches :mod:`wdnai.models`):

    <model_dir>/
        text_projection.1828.rknn + .weight
        talker.1828.rknn          + .weight
        code_predictor.1828.rknn  + .weight
        speech_decoder.1828.rknn  + .weight
        talker_text_embed.fp16.bin
        talker_input_embed.fp16.bin
        codec_embed.fp16.bin
        tokenizer.json

VoiceClone speakers (``girl_base`` / ``ahu``) are bundled in this
module — no external ``qwen3tts_speaker_embed.h`` is needed.

The directory is resolved from the ``ModelSpec`` in this order:
``spec.tts_model_dir`` → ``spec.options["model_dir"]`` →
``$TTS_MODEL_DIR`` → ``$WDNAI_MODELS_DIR/Qwen3-TTS`` →
``/userdata/models/Qwen3-TTS``.

Optional overrides (forwarded from ``LLMSession.respond(**overrides)``):

* ``ref_speaker``   — voice-clone speaker name (``"girl_base"`` / ``"ahu"``).
* ``language``      — ``"auto"`` / ``"chinese"`` / ``"english"`` / …
* ``speaker``       — built-in SPK_ID entry (``serena`` / ``vivian`` / …).
* ``volume``        — 0.0–1.0 (applied to the live PCM by the
                      ``AudioPlayer``).
* ``no_play``       — skip live playback; only produce WAV bytes.
* ``instruct``      — optional CustomVoice instruct text.
* ``non_streaming_mode`` — 1 to use non-streaming prefill.
* ``on_step``       — ``callable(n_frames, audio_sec)`` invoked per
                      decoder window for real-time progress reporting.
"""
from __future__ import annotations

import base64
import ctypes
import logging
import os
import queue
import struct
import subprocess
import tempfile
import threading
import time
import zlib
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np

try:
    from rknn3lite.api import RKNN3Lite, RKLLMCallback  # noqa: WPS433
    from rknn3lite.api.rknn3_types import (
        LLMResultCallback,
        LLMGetEmbedCallback,
        LLMOutputCallback,
        LLMCallState,
    )
except Exception as exc:  # pragma: no cover - import error path
    raise SystemExit(
        "rknn3lite package not installed. Please install the wheel under "
        "rknn3-toolkit-lite/packages and rerun."
    ) from exc

from ..attachment import Attachment
from ._common import resolve_device_id

LOG = logging.getLogger("wdnai.backends.tts")


# --------------------------------------------------------------------------- #
# Constants copied from rknn3_model_zoo/examples/Qwen3_TTS/cpp/talker.h
# --------------------------------------------------------------------------- #
CODEC_MAX_CONTEXT_LEN = 15
QWEN3_TTS_MAX_CONTEXT_LEN = 8192
QWEN3_TTS_EMBED_DIM = 2048
TEXT_PROJECTION_CORE_MASK = 0x01

TEXT_VOCAB_SIZE = 151936
TALKER_VOCAB_SIZE = 3072
CODEC_VOCAB_SIZE = 2048

TTS_BOS_TOKEN_ID = 151672
TTS_EOS_TOKEN_ID = 151673
TTS_PAD_TOKEN_ID = 151671
CODEC_BOS_ID = 2149
CODEC_EOS_TOKEN_ID = 2150
CODEC_THINK_ID = 2154
CODEC_NOTHINK_ID = 2155
CODEC_PAD_ID = 2148
CODEC_THINK_BOS_ID = 2156
CODEC_THINK_EOS_ID = 2157

# Audio decoding windowing
kFeatSize = 16
kChunkSize = 25
kLeftContextSize = 25
kWindowLen = kChunkSize + kLeftContextSize  # 50 frames
kTotalUpsample = 1920
kSampleRate = 24000

# Speaker / language tables (taken from cpp/talker.h)
SPK_ID = {
    "serena": 3066, "vivian": 3065, "uncle_fu": 3010, "ryan": 3061,
    "aiden": 2861, "ono_anna": 2873, "sohee": 2864, "eric": 2875,
    "dylan": 2878,
}

CODEC_LANGUAGE_ID = {
    "chinese": 2055, "english": 2050, "german": 2053, "italian": 2070,
    "portuguese": 2071, "spanish": 2054, "japanese": 2058, "korean": 2064,
    "french": 2061, "russian": 2069,
    "beijing_dialect": 2074, "sichuan_dialect": 2062,
}

SPK_IS_DIALECT = {
    "serena": "false", "vivian": "false", "uncle_fu": "false",
    "ryan": "false", "aiden": "false", "ono_anna": "false",
    "sohee": "false", "eric": "sichuan_dialect", "dylan": "beijing_dialect",
}

# Sampling parameters mirror the C++ demo's constants
# ``TALKER_SAMPLE_PARAMS`` / ``CODE_PREDICTOR_SAMPLE_PARAMS``.
TALKER_SAMPLING = {
    "top_k": 50,
    "top_p": 1.0,
    "temperature": 0.9,
    "repeat_penalty": 1.05,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
}
CODE_PREDICTOR_SAMPLING = {
    "top_k": 50,
    "top_p": 1.0,
    "temperature": 0.9,
    "repeat_penalty": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _userdata_to_obj(ptr):
    """Recover a Python object from a void* userdata pointer."""
    return ctypes.cast(ptr, ctypes.POINTER(ctypes.py_object)).contents.value


def _obj_to_userdata(obj):
    """Return ``(void_ptr, anchor)``.

    The returned ``py_object`` instance must stay alive as long as the
    callback may be invoked; callers usually pin it as a class attribute.
    """
    holder = ctypes.py_object(obj)
    return ctypes.cast(ctypes.pointer(holder), ctypes.c_void_p), holder


def _fp16_view(ptr, n_elems):
    """Create a fp16 ``numpy`` view of a raw memory pointer."""
    return np.ctypeslib.as_array(
        ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint16)),
        shape=(n_elems,),
    ).view(np.float16)


def _wrap_userdata(self, attr_name):
    """Build a void* userdata that recovers ``self`` from the callback."""
    ptr, holder = _obj_to_userdata(self)
    if attr_name is not None:
        setattr(self, attr_name, holder)
    return ptr


# --------------------------------------------------------------------------- #
# Bundled speaker embeddings
# --------------------------------------------------------------------------- #
# VoiceClone reference vectors from the upstream C++ demo's
# ``qwen3tts_speaker_embed.h``. That header only ships with the rknn3
# dev tree, so without this bundle a clean board would silently fall
# through to the built-in voice and ``girl_base`` / ``ahu`` would
# produce identical audio. Two 2048-dim fp16 vectors are stored inline
# as zlib-compressed base64 (~8.5 KB on disk, ~4 KB decoded).
_BUNDLED_SPEAKER_EMBEDDINGS_B64: dict[str, tuple[str, ...]] = {
    "girl_base": (
        "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",
    ),
    "ahu": (
        "eNoN1edDFNfCgHF7e6MQRQFZ2DY79ZyZ2QUERA1gu7YYCxbUmCiWWGLBRjRiTGxRDGJBkLJtdqecKYuI3Yi9d+w10UQssVw16NX45k94vvyej7An3bFtJ3AZ66GqyH2waQKm3MeL3CuqTFqe8mnAFUqvLJc7CwPtjdmtqIfcjT1PbsZN8dlqFNoDfivz4+8LjsunuDvi"
        "YhjOdMgPkbKxwZJjjwy0r5imzKf98lCUSZnABOdFsS187RtK1+Am+hg5wdipuomDNrt9aTJQhpAjqL0Jd+kubKp6RYj2DYbNQgXof/5b2gb2H98p8oiynlon3QVDzLhvEahRc5z9sHX+RuYYbmbokjSnMC/uNDWIWuRda8uGs5R2wRmC0/mLOlgYoDywPwCL0ZCyT4xb"
        "NGkmlQ7ml/Fv7MMDgnFK3si99D8gvjFWEf01vCxdH8mM9Pciv2ZPC82Lz7Gzrc1gOxCl3mV72Z+yTYp1aSUqMJ4YUcm2yh+l5VVv2j9y5pMztafgLXYPUeXz3E8OVkpN/WfVy9Ipyqydk6cgGc63tNcg2hmbF/nY+YJ5Gh1OJorfgv3CXjkRhFAKW4tfB2ej18q2uLHw"
        "T/TYjmGjSjDtfnAQfMA208fCLcTGHY0c/LaJxG9qFlvMf1S/lHn6EryvHKm85xZBR7kXeckxwrW6bCxzQXhtP033EB5FRKvZGuv9ScxATWG7rd/BbOaWPM8yv3wpaCJcwEdDJnTA35tcSK/wQ1e8f7qjDjtOJuJH6INUDX5WWoa14V86immHcBv/sbMJnKB6ohfKYvQ/"
        "Rx/33ooo9UinVDtNLqi8bxTtmsAUmLf4Ff5zDNHj9cHcfEtLx1v7eNUKjvoOScn8U6shqXp6iUNa5phNlMJMaT1zQL4NHmtdQ3vt/RMP2Wvpu/hkZFV12rulM7bHPVUbsyFzWx6ZY98Sovlr0ihxnhHLqMgkNvDY6U8dJjCHjYM/4a+DlVxbx2pnNT7C1b/tJX5w1TrL"
        "E623lIpOw1XghblAmgss8jO+NjCUC2IXkItHioOapzaB4VSeHL2pyhQgmxe32fqx5D0wC2FoUMw7vhsaWDKWZ7hYcqpyjZkeHi72iPOEjgjPWUm8jq7BMMv5oMOuwZfCWDSNmtOxudrTEVJvBibRIcxB3SJNYn/QLLEF6Eb3D+2AbYLLEY6WyLUKbmQ4nxD58C/UC3XS"
        "6n3/Ibsrs5md0R/FpfAr28XmnUKh4AO0m8wlnpPzNqcw76VqYQn/laqx7yhChkIdBUSJPsT2FV/5DWu8OM++nkvZNtG1RUt1RPk9iOUdaxzee9J26gu9Dz8/YQxMJR6iKeCsdQrYpxrr6qtnCaw8GhU6d8m79N+xcNSUmhtIF1X3RHcOO0yOo/+kroJYaRJaVmG3pbAM"
        "ygJPIzOiw6Ux8n7bVGvIa0KD5TRhFJWp6kKU7WDpIMaGCCFizW9wkzuAh8HZbg9xQuggFtInmYsV18XvYLI2TH0HKfEWn0ue9s+RkmzP0B1yemkl/T0noGPgrFYltJOWmtqA771J6lTUxVUVbBuUyL+VdN5rz9QOWzujuSWnhPKqpfxM/3A+Qx8IP9/Y1DuVyqbeKE/Q"
        "H0Q3saE+2sFjWcpu+2F5k6ZKo4xCLJ+qw4YgnUurfKUE3Zn6J/J3/gu2+1hf4hEaDp57IrSBtAt9z41Al9VkI47IU2ZbT2L5lXuEJFIm1pReFTDzusA+aeDq/6JIb2Jwb2VgtyQtBcOM5qGuUTrbYsMY/ASoEf6DL4MNlL8st43e5D2yRsn3F8HRxCryR3Gv8Kt2wvep"
        "nK6+p34gd0knjXLLEHGr42NgJL0o2FdrZeTC9SQuEqAv2i10V7sFOSiCyV6eTLanqfVeY+sU0AD+Dc7E26unWk4Fg8x+pg0Xk1+N6p2PQAQeBKec59Xi4Of8WGkJvV9dye6FMPaw5rBuJgexIioSdGU5iPPN4SgiUmshLjAXw1lUa22hnFSa7VhvsHRfbrWUTvWCzazt"
        "ZadjiNEYnQaLufOwCJwVnke/V9KVLM2vJisW4azcj1gGU4XtXhO5FAvKh8ju2p/UYn0bPO2b4B6uhNneqSvVDFVANWSNVArHOPoU7y41GZ/AI2CFMsdIp+nqj+RCdarNJq+w/K6lgQDKUzdxGfIIjQcqLGKmua7AGGMmdQfOCjR1XghBdjN7lfmao+RX7qXkQEe9b7wz"
        "LiVZC+GY2oi9q66ABQrElvNfGIXUMWoF1oJfCIsSMkJ3tUtUsDKcMZX8Zv4oDxLa4zN4vwqJyPg/qBzzJ8Kf6k73fnIjylbaqJPsij5A7kjssdSQPxsN6CC8LsewJIqTfiAnaW1Zc6crkR9gqX7HYXa+xz10YVSSvhBgXc4HSJWzd6Ru4O/AdeaZXfO1kn7UBntbswXs"
        "XSkvuidoEfuccmvJcKhipgbYSPfNksL4NzvTtYYGZ1yw36EsscNL4hJiEPJNCiWB0U6B0NFTssZeAkfBG3R+1FUpT2viFeyLt1yCfaQzVrhVcm90XQ5lgE0JT7BClaiMIGaFhcSWWl9bBqCMLHjdd1MA6JVitv3NjQ4ukN/Qx4QXqBGoA4hKF39NmNxgAMOp57Brwmh8"
        "obwHX+g5wyYJR8jjpgjqZ7UlN8LTBOT5nzvGyq28uUo+OY27SnpKPeofxJCotKqwqMb8wYR962cFeoMpBku1UCfJ75hZxi8qg2KNerpATYDDUZhrruwomarP597KKMaKE0KQm0NLu6LdB/xmIZFaju4lfJWwRK8TKetbeVG84k5R/9q2gpxLDjEvElPdX2M2/R/qTskq"
        "Nh3fhdJ8v5s6hdZpbqmhP41X42vjN/ubhKr4f7T88iVQEKqMh446aSqIUBugM9QjbLVjuD3kWYzucHQ55/80JiJ+uvgIHWMuSxNcTdRmoQXRt61hdARsSaQa3Sw7NE17qV/EgV4OF7G3YEumTeke8ah1W+COGInMYBj5WvxR1V2dvQeRyzGAdKFqPAt1BWGsGsg05OqF"
        "YJH3oViv7icuMGtgqdtHyR1/cM8JfkM3cD/zB0Aat9WY52lBfEa11ksc2ZRfwLkJSg11nv5f4LV0Mng21iv+bTmLrfUcMDcErY1BQpLx2EiRpkmd8Xqvg9kXU7N9qhIt+xwR2odqHhss+OlidoA9DGVJlx0LSnu7LOua8D3Rbs+uuOne5lVnvKLSRY6Gy/Qi7AR4KOHG"
        "h7IlJRns91w4nQ9Pxr41ddLvW2v/VaPE2GMu5d+KnYwlzFBHEzGf6akcEluHfrdVcHMCX4BI1I3crmfxs0Fz7F8P1KWojE/oqJfloHDHU3ZKbBb8SHRlbws59l/I6dte87Pxa4qFH8YQYj9Qg52qnllxtyUnNZBuhjqimyA3ujN1VG0TnIkw6ob8GRjlqhMy1GWhbMkE"
        "s6mB8JI9k80Ez+RTnlmSLfQo7Fx1LtiI6phN8nhFVjLJtyJpOUMx2PGIE6W7mS6FrwtN/Gz2Z8SAo8Q89omtE2wbHJbAw/yITHDDk+ur2CYSorsJ2NExd92awLOqyXjg34LGYi71m3ARdad04nump2ZBE9Om6eMqvpEO6XRxf5qhv6Q+L7eI48SmcLuSglJJP10ZXUYC"
        "Yw03xMVgl8BPcgd0VRtPdcd68k+FkLpK6gSm/+vlA3Z6QQYYu2MglasU8Q/1FXQYpNjDxGfkTWct8JS0la7ZF7h2kcOUNJrSBT4OX1L+hg5YW6OpTB/qAaoJ/lKpOyqA1/ZAqqcbFjfDOdnrrZUGkbcTPoAusDu/E7siLaK/dVxTlgCrWIh2K3GBdrIpaY0SWzVLHCR3"
        "sC7E2m4NCH65sXNURWdtlYrIekrX45PXi5vRNOxzPgZ8IR3GBm6tRev9Zfht7BCYhypcKaEq9R+qi5BCFVWS28ahDZxHnMds4I0EXWnp3Fi1X3xBNWMuAo/Iipiw0vYX6ud/CWbCbHKuPVKO0m3KJk8zsi85uP0ko2PlDM9j8R/tBrQCH3T7D9IdsPNEHzjc2xf9obtt"
        "47g0lCG1svUGCaHB7lT/WPtI+CJwwb9Zm8Gr/gQ2CWZ6Z8j7nKfRKX+DQJp6NLBXTScuaqo4Q/sKd4H/AxKeU50njWQqkG5dRp0Eq/594zlpHBoiNuS7BnMYE/fUIGMaS1uZG2qtNkVMQz62uT7RVmtvz1/edCUWQwtd+0oHVM7ylcDF5oN8PldO5ao2tquxunIgdwPv"
        "IPcjbaV+so9aa6phulPtoUcf6Rtr+UNZJtaR29qvIr6hePwYTGWSg+ll2R6Mcjvw8h7GC2WaetHYodz0HJDvozZR15jt1J3AL55SvjPAxD0yYN1MhRYBPlN2VGgdjMTmamLCvkB/Z3/0DtyX1niXbllmrEM+6mZF6xgP7vObq8ZqW4XJRo06Uj1KtJS/knOIHmWt3LMS"
        "cqm+O6qdJxBOrNRnMp03rQg2oI4pebE+uveWcdJhtSkYAF+RroTd4gE1aNoo1csLyn9lmllGiTFKE8+36DPwXBqpX6egmkR9K7rxgs05/pNxsdYUdXsF7V8JrlbdMryBWnjXPBX28leDf+Qj8TVUUkgV3wXaUPcC54IvpZswQsTQFenejnklURQD3+BOkgnRcG+789uO"
        "yzb00PFW+ia0TswK1MUcpwrI38WJZrNpFVbPzme7K5LjgFwgZznPqFXI5hwaVWxJovobZx1P5U1Sqbg8IU5vqR8KBENe+kliP22RcKIslf0Y/ga2JhqKOxLClEfYUXBPXKFm0Gvlk/BLZS1wRs1VLzG5wQawJ9YVGVg2V0+PVwX6rPJdvwvi12S5Xw19smEEPOXziTVK"
        "d/kxtd5RxZV5z0uH6OfSJOI6tZa74X7neWDcpFs4ynFrZDIZq0wni5gSep09S0kW/2Sf8gkJY7jFxcM6lTNF3stgM3uEWkmG43l0O7ot/ACfwSrpGv1ebAEPOlb4Z9i2KJsZVjwv9vXT+Hbjv0xjUIf31eyWI6Ub2m2QGlUWiu1ga2WlcRJOAi9ptzqecsk1MRGUG52z"
        "pPAfXBxHG5usM0uJ7SaUSXalv+UaJf9AkGwr11/SKPo+KFB+knXCQsfjudZEbhT2IjFE1MR2I+PoLO69EBk6LuuOHLoZE4ZukF2oyc5niETDUZ3zIdFWOgI76zdpl8o4X8nFm6H1PnnQYZUXKf8Ppq4rVA==",
    ),
}


def _get_bundled_speaker_embeddings() -> dict[str, np.ndarray]:
    """Lazy-decoded copy of :data:`_BUNDLED_SPEAKER_EMBEDDINGS_B64`.

    Returns a ``{name: np.ndarray fp16 (2048,)}`` dict. Decoding is
    deferred to the first call so module import stays cheap (and the
    ~8.5 KB of zlib+base64 in the source only materialises into
    ~4 KB of numpy arrays when VoiceClone is actually used).
    """
    if not hasattr(_get_bundled_speaker_embeddings, "_cache"):
        _get_bundled_speaker_embeddings._cache = {
            name: np.frombuffer(
                zlib.decompress(base64.b64decode("".join(b64_chunks))), dtype=np.float16
            ).copy()
            for name, b64_chunks in _BUNDLED_SPEAKER_EMBEDDINGS_B64.items()
        }
    return _get_bundled_speaker_embeddings._cache


# --------------------------------------------------------------------------- #
# Audio playback sink
# --------------------------------------------------------------------------- #
class AudioPlayer:
    """Real-time playback sink.

    Tries ``sounddevice`` (PortAudio-based, low latency) first and falls
    back to an ``aplay`` subprocess if PortAudio or a usable output device
    is not available. When neither backend works, the class degrades to a
    no-op and the demo still writes the WAV file.
    """

    def __init__(self, sample_rate, device=None, volume=0.5):
        self.sample_rate = sample_rate
        if device is None:
            self.device = _resolve_alsa_device(None)
        else:
            self.device = device
        self.volume = float(volume)
        self._backend = None
        self._sd = None
        self._stream = None
        self._proc = None
        self._warned = False
        self._closed = False

        try:
            import sounddevice as sd
            self._sd = sd
            self._stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                device=device,
            )
            self._stream.start()
            self._backend = "sounddevice"
            print(
                f"[audio] streaming via sounddevice @ {sample_rate} Hz mono, "
                f"volume={self.volume:.0%}",
                flush=True,
            )
            return
        except ImportError:
            pass
        except Exception as exc:
            self._sd = None
            self._stream = None
            if not self._warned:
                print(
                    f"[audio] sounddevice init failed ({exc}); falling back to aplay",
                    flush=True,
                )
                self._warned = True

        try:
            cmd = ["aplay", "-v", "-t", "raw", "-f", "FLOAT_LE",
                   "-r", str(sample_rate), "-c", "1"]
            if self.device is not None:
                cmd += ["-D", self.device]
            cmd.append("-")
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            self._aplay_stderr = self._proc.stderr
            self._backend = "aplay"
            print(
                f"[audio] streaming via aplay ({' '.join(cmd)}) @ {sample_rate} Hz mono, "
                f"volume={self.volume:.0%}; device: {self.device!r}",
                flush=True,
            )
        except FileNotFoundError:
            if not self._warned:
                print(
                    "[audio] neither sounddevice nor aplay is available; "
                    "falling back to WAV-only output.",
                    flush=True,
                )
                self._warned = True

    @property
    def available(self):
        return self._backend is not None and not self._closed

    def play_chunk(self, audio):
        if not self.available:
            return
        chunk = np.asarray(audio, dtype=np.float32) * self.volume
        np.clip(chunk, -1.0, 1.0, out=chunk)
        if self._backend == "sounddevice":
            try:
                self._stream.write(chunk)
            except Exception as exc:
                if not self._warned:
                    print(f"[audio] sounddevice write failed: {exc}", flush=True)
                    self._warned = True
        elif self._backend == "aplay":
            # aplay 启动后可能立刻退（headless 板子 / HDMI 默认 / 格式不
            # 支持 / 权限问题）。原先 Popen 完只 catch BrokenPipeError，
            # 不 poll()，stderr 也被收着没读 —— 既不诊断也每帧打一行错。
            # 现在写之前先 poll() 一次，死了就 drain stderr + 关掉 backend，
            # 后续帧走 available=False 早退，不再反复触发 Broken pipe。
            proc = self._proc
            if proc is not None and proc.poll() is not None:
                rc = proc.returncode
                err = ""
                try:
                    if proc.stderr is not None:
                        err = proc.stderr.read().decode(
                            "utf-8", errors="replace")
                except Exception:
                    pass
                if not self._warned:
                    print(
                        f"[audio] aplay exited rc={rc} before any write; "
                        f"disabling playback (stderr: {err.strip()[:500]})",
                        flush=True,
                    )
                    self._warned = True
                self._backend = None
                self._proc = None
                self._closed = True
                return
            try:
                self._proc.stdin.write(chunk.tobytes())
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                if not self._warned:
                    err = ""
                    try:
                        if proc is not None and proc.stderr is not None:
                            err = proc.stderr.read().decode(
                                "utf-8", errors="replace")
                    except Exception:
                        pass
                    print(
                        f"[audio] aplay write failed: {exc}; disabling "
                        f"playback (stderr: {err.strip()[:500]})",
                        flush=True,
                    )
                    self._warned = True
                self._backend = None
                self._proc = None
                self._closed = True

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._backend == "sounddevice" and self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        elif self._backend == "aplay" and self._proc is not None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=10)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass


def _resolve_alsa_device(device):
    """Pick a usable ALSA output for ``aplay`` when no device is set.

    On Rockchip boards, the system ``default`` often points to HDMI which
    has no audio sink, so the onboard ES8323 (or similar) codec would stay
    silent. ``aplay -l`` lists every card; we pick the first entry whose
    description does not contain ``hdmi``.
    """
    if device is not None:
        return device
    try:
        out = subprocess.run(
            ["aplay", "-l"], capture_output=True, text=True, timeout=3
        )
    except Exception as exc:
        print(f"[audio] aplay -l failed: {exc}", flush=True)
        return None
    if out.returncode != 0:
        return None
    for line in (out.stdout or "").splitlines():
        line = line.strip()
        if not line.lower().startswith("card "):
            continue
        if "hdmi" in line.lower():
            continue
        try:
            n = int(line.split(":", 1)[0].split()[1])
        except Exception:
            continue
        return f"plughw:{n},0"
    return None


def _write_wav(path, audio, sample_rate):
    """Write a mono 32-bit IEEE-float WAV file (matches the C++ demo)."""
    path = str(path)
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    data = audio.astype("<f4", copy=False).tobytes()
    with open(path, "wb") as f:
        f.write(b"RIFF")
        byte_rate = sample_rate * 1 * 4
        block_align = 1 * 4
        data_size = len(data)
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        # fmt chunk: IEEE float (format tag 3) so the file matches the C++ output
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 3))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", 32))
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(data)


def _to_lower(s):
    return s.lower() if isinstance(s, str) else s


def _load_tokenizer(model_dir: Path):
    """Return a callable ``text -> List[int]``.

    Tries ``transformers.AutoTokenizer`` first and falls back to the bare
    ``tokenizers`` package. Raises ``RuntimeError`` if neither is available.
    """
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
        return lambda text: list(tk.encode(text, add_special_tokens=False))
    except Exception:
        pass
    try:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        return lambda text: list(tk.encode(text).ids)
    except Exception:
        pass
    raise RuntimeError(
        "Unable to tokenize text. Please install the 'transformers' or the "
        "'tokenizers' Python package (e.g. `pip install tokenizers`)."
    )


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
    root = os.environ.get("WDNAI_MODELS_DIR", "/userdata/models")
    return Path(root) / "Qwen3-TTS"


# --------------------------------------------------------------------------- #
# RKNN3 driver — vendored from qwen3_tts_demo.py
# --------------------------------------------------------------------------- #
class Qwen3TTSDriver:
    """RKNN3 driver for the Qwen3-TTS pipeline.

    Four RKNN3 models run back-to-back:

    text → text_projection → talker (LLM) ─┐
                                            ├─► code_predictor (LLM)
                                            └─► speech_decoder ──► wav

    ``run_with_progress`` is the SDK-friendly entry point — it accepts
    ``stop_event`` / ``on_step`` / ``on_audio_chunk`` hooks used by
    :class:`TTSBackend`.
    """

    def __init__(self, model_dir, target="rk1820", core_mask=0xff,
                 device_id=None,
                 talker_sampling=None, code_predictor_sampling=None,
                 verbose=False):
        self.model_dir = Path(model_dir)
        self.target = target
        self.core_mask = core_mask
        self.device_id = device_id.encode("utf-8") if isinstance(device_id, str) else device_id
        self.verbose = verbose
        self.talker_sampling = dict(talker_sampling) if talker_sampling is not None else dict(TALKER_SAMPLING)
        self.code_predictor_sampling = (
            dict(code_predictor_sampling) if code_predictor_sampling is not None
            else dict(CODE_PREDICTOR_SAMPLING)
        )

        # Embedding tables (fp16 numpy arrays, contiguous)
        self.text_embed = None
        self.input_embed = None
        self.codec_embed = None
        self.speaker_embeddings = None

        # Tokenizer callable ``text -> List[int]``
        self._encode_text = None

        # Shared state for the callbacks
        self.lock = threading.Lock()
        self.talker_steps = 0
        self.talker_failed = False
        self.talker_last_state = None
        self.past_hidden = None            # fp16 (EMBED_DIM,)
        self.code_predictor_out_ids = []   # list of int
        self.code_predictor_step = 0
        self.collected_codes = []           # list of 16-token frames
        self.tts_pad_embed = None          # fp16 (EMBED_DIM,)
        self.trailing_text_hiddens = None  # fp16 (T, EMBED_DIM)

        # RKNN3Lite instances (one per model)
        self.text_projection = None
        self.talker = None
        self.code_predictor = None
        self.speech_decoder = None
        # Optional real-time playback sink.
        self.audio_player = None
        # Streaming pipeline state.
        self.frame_queue = None
        self._streamed_audio = []
        self._talker_done_event = None
        self._decoder_thread = None

    # ------------------------------------------------------------------ #
    # Resource paths
    # ------------------------------------------------------------------ #
    def _resolve_model_paths(self, role: str):
        """Return ``(rknn_path, weight_path)`` for ``role``.

        Looks up ``<role>.1828.rknn`` first (the RK1828 NPU export naming
        used on this board) and falls back to the bare ``<role>.rknn``
        filename for legacy installs that haven't been renamed.
        """
        weight_path = self.model_dir / f"{role}.weight"
        for rknn_name in (f"{role}.1828.rknn", f"{role}.rknn"):
            cand = self.model_dir / rknn_name
            if cand.is_file():
                return cand, weight_path
        # Default to the .1828.rknn name even if missing — ``load_rknn``
        # will raise the usual FileNotFoundError with the right path.
        return self.model_dir / f"{role}.1828.rknn", weight_path

    # ------------------------------------------------------------------ #
    # Resources
    # ------------------------------------------------------------------ #
    def _load_speaker_embeddings(self):
        """Return the bundled voice-clone speaker embeddings."""
        bundled = _get_bundled_speaker_embeddings()
        print(
            f"[speaker] using {len(bundled)} bundled speaker(s): "
            f"{', '.join(sorted(bundled))}"
        )
        return bundled

    def ensure_loaded(self):
        """Eagerly load embed tables + tokeniser + all four RKNN3 models.

        Idempotent — each underlying ``init_*`` method short-circuits on a
        non-None handle, and ``load_resources`` short-circuits on a non-None
        embed / tokenizer. Safe to call multiple times.

        Called by :class:`TTSBackend.__init__` so that ``LLMSession(...)``
        instantiation triggers the full TTS load (matching the eager
        semantics of ``TextBackend`` / ``VLMBackend`` / ``ASRBackend`` —
        those run ``load_rknn`` + ``init_runtime`` directly in their
        constructors). Without this, the user-facing "运行中" status on
        the 算力配置 page is misleading: NPU runtime is not built, weights
        are not even mmap'd until the first ``run_text`` triggers
        ``run_with_progress``.
        """
        if getattr(self, "_ensure_loaded_done", False):
            return
        self.load_resources()
        self.init_text_projection()
        self.init_speech_decoder()
        self.init_code_predictor()
        self.init_talker()
        self._ensure_loaded_done = True

    def load_resources(self):
        """Load embedding tables and configure the tokenizer.

        Idempotent: after the first call the embed tables (~750 MB) and
        tokenizer are reused on subsequent calls. pyrkaiweb relies on
        this — ``TtsSession._infer_one`` issues one ``respond`` per
        ``tts_alert`` trigger; re-loading 750 MB each time made every
        call ~10 s before this guard landed.
        """
        if self.text_embed is not None and self._encode_text is not None:
            return
        text_path = self.model_dir / "talker_text_embed.fp16.bin"
        input_path = self.model_dir / "talker_input_embed.fp16.bin"
        codec_path = self.model_dir / "codec_embed.fp16.bin"
        tokenizer_path = self.model_dir / "tokenizer.json"
        for p in (text_path, input_path, codec_path, tokenizer_path):
            if not p.exists():
                raise FileNotFoundError(f"missing required file: {p}")

        self.speaker_embeddings = self._load_speaker_embeddings()

        self.text_embed = np.fromfile(text_path, dtype=np.float16).reshape(
            TEXT_VOCAB_SIZE, QWEN3_TTS_EMBED_DIM
        )
        self.input_embed = np.fromfile(input_path, dtype=np.float16).reshape(
            TALKER_VOCAB_SIZE, QWEN3_TTS_EMBED_DIM
        )
        self.codec_embed = np.fromfile(codec_path, dtype=np.float16).reshape(
            15, CODEC_VOCAB_SIZE, QWEN3_TTS_EMBED_DIM
        )
        print(
            f"[embeds] text={self.text_embed.shape} input={self.input_embed.shape} "
            f"codec={self.codec_embed.shape}"
        )

        self._encode_text = _load_tokenizer(self.model_dir)

    # ------------------------------------------------------------------ #
    # text_projection (separate inference, not LLM)
    # ------------------------------------------------------------------ #
    def init_text_projection(self):
        if self.text_projection is not None:
            return
        rknn_path, weight_path = self._resolve_model_paths("text_projection")
        self.text_projection = RKNN3Lite(verbose=self.verbose)
        ret = self.text_projection.load_rknn(str(rknn_path), str(weight_path))
        assert ret == 0, f"load text_projection failed ({rknn_path}): {ret}"
        ret = self.text_projection.init_runtime(
            target=self.target,
            core_mask=TEXT_PROJECTION_CORE_MASK,
            device_id=self.device_id,
        )
        assert ret == 0, "init text_projection runtime failed"

        attrs = self.text_projection.get_inputs_tensor_attr()
        if not attrs:
            raise RuntimeError("text_projection: no input tensor attributes")
        shape = [attrs[0].shape[i] for i in range(attrs[0].n_dims)]
        self.text_proj_seq_len = int(shape[1])
        self.text_proj_in_dim = int(shape[2])
        print(f"[text_projection] input shape={shape} (1, seq_len, dim)")

    def text_projection_chunked(self, input_ids):
        """Run text_projection per ``text_proj_seq_len`` chunk."""
        if self.text_projection is None:
            raise RuntimeError("text_projection not initialized")
        seq_len = len(input_ids)
        chunk = self.text_proj_seq_len
        out_dim = QWEN3_TTS_EMBED_DIM
        out = np.zeros((seq_len, out_dim), dtype=np.float16)

        buf = np.zeros((chunk, self.text_proj_in_dim), dtype=np.float16)
        for start in range(0, seq_len, chunk):
            valid_len = min(chunk, seq_len - start)
            buf.fill(0)
            for i in range(valid_len):
                tid = int(input_ids[start + i])
                if 0 <= tid < TEXT_VOCAB_SIZE:
                    buf[i] = self.text_embed[tid]

            inp = buf.reshape(1, chunk, self.text_proj_in_dim)
            outputs = self.text_projection.inference(inputs=[inp])
            if outputs is None:
                raise RuntimeError("text_projection inference failed")
            chunk_out = np.asarray(outputs[0]).reshape(chunk, out_dim)
            out[start:start + valid_len] = chunk_out[:valid_len]
        return out

    # ------------------------------------------------------------------ #
    # speech_decoder (separate inference, not LLM)
    # ------------------------------------------------------------------ #
    def init_speech_decoder(self):
        if self.speech_decoder is not None:
            return
        rknn_path, weight_path = self._resolve_model_paths("speech_decoder")
        self.speech_decoder = RKNN3Lite(verbose=self.verbose)
        ret = self.speech_decoder.load_rknn(str(rknn_path), str(weight_path))
        assert ret == 0, f"load speech_decoder failed ({rknn_path}): {ret}"
        ret = self.speech_decoder.init_runtime(
            target=self.target, core_mask=self.core_mask, device_id=self.device_id
        )
        assert ret == 0, "init speech_decoder runtime failed"

        attrs = self.speech_decoder.get_inputs_tensor_attr()
        if not attrs:
            raise RuntimeError("speech_decoder: no input tensor attributes")
        shape = [attrs[0].shape[i] for i in range(attrs[0].n_dims)]
        if len(shape) != 3 or shape[1] != kFeatSize or shape[2] != kWindowLen:
            print(
                f"[speech_decoder] WARNING unexpected input shape={shape}, "
                f"expected (1, {kFeatSize}, {kWindowLen})"
            )
        self.speech_decoder_window = kWindowLen
        print(
            f"[speech_decoder] input shape={shape} dtype=int32, "
            f"output samples per window = {kWindowLen * kTotalUpsample}"
        )

    # ------------------------------------------------------------------ #
    # code_predictor (LLM)
    # ------------------------------------------------------------------ #
    def init_code_predictor(self):
        if self.code_predictor is not None:
            return
        rknn_path, weight_path = self._resolve_model_paths("code_predictor")
        self.code_predictor = RKNN3Lite(llm_mode=True, verbose=self.verbose)
        ret = self.code_predictor.load_rknn(str(rknn_path), str(weight_path))
        assert ret == 0, f"load code_predictor failed ({rknn_path}): {ret}"

        cb = RKLLMCallback()
        cb.result_callback = LLMResultCallback(self._cp_result_callback)
        cb.result_userdata = _wrap_userdata(self, "_cp_userdata")
        cb.embed_callback = LLMGetEmbedCallback(self._cp_embed_callback)
        cb.embed_userdata = _wrap_userdata(self, "_cp_embed_userdata")

        args = [{
            "max_context_len": CODEC_MAX_CONTEXT_LEN,
            "vocab_size": CODEC_VOCAB_SIZE,
            "special_eos_id": CODEC_EOS_TOKEN_ID,
            "max_new_tokens": CODEC_MAX_CONTEXT_LEN,
            "keep_history": 0,
            **self.code_predictor_sampling,
        }]

        ret = self.code_predictor.init_runtime(
            target=self.target,
            core_mask=self.core_mask,
            device_id=self.device_id,
            llm_args=args,
            llm_callback=cb,
        )
        assert ret == 0, "init code_predictor runtime failed"

    def run_code_predictor(self, code_input_embeds):
        """Run code_predictor synchronously and return the 15 codec token ids."""
        with self.lock:
            self.code_predictor_out_ids = []
            self.code_predictor_step = 0

        ret, _ = self.code_predictor.session_run(embeds=code_input_embeds)
        if ret != 0:
            raise RuntimeError(f"code_predictor.session_run failed: {ret}")

        with self.lock:
            ids = list(self.code_predictor_out_ids)
        if len(ids) != CODEC_MAX_CONTEXT_LEN:
            print(
                f"[code_predictor] WARNING produced {len(ids)} tokens, "
                f"expected {CODEC_MAX_CONTEXT_LEN}"
            )
            if len(ids) < CODEC_MAX_CONTEXT_LEN:
                ids += [0] * (CODEC_MAX_CONTEXT_LEN - len(ids))
        return ids

    # ------------------------------------------------------------------ #
    # talker (LLM)
    # ------------------------------------------------------------------ #
    def init_talker(self):
        if self.talker is not None:
            return
        rknn_path, weight_path = self._resolve_model_paths("talker")
        self.talker = RKNN3Lite(llm_mode=True, verbose=self.verbose)
        ret = self.talker.load_rknn(str(rknn_path), str(weight_path))
        assert ret == 0, f"load talker failed ({rknn_path}): {ret}"

        ret = self.talker.init_runtime(
            target=self.target, core_mask=self.core_mask, device_id=self.device_id
        )
        assert ret == 0, "init talker runtime failed"

        output_tensors, n_outputs = self.talker.create_output_tensors([0, 1])
        assert n_outputs == 2, f"talker: expected 2 outputs, got {n_outputs}"
        self._talker_output_tensors = output_tensors

        cb = RKLLMCallback()
        cb.result_callback = LLMResultCallback(self._talker_result_callback)
        cb.result_userdata = _wrap_userdata(self, "_talker_result_userdata")
        cb.embed_callback = LLMGetEmbedCallback(self._talker_embed_callback)
        cb.embed_userdata = _wrap_userdata(self, "_talker_embed_userdata")
        cb.output_callback = LLMOutputCallback(self._talker_output_callback)
        cb.output_userdata = _wrap_userdata(self, "_talker_output_userdata")
        cb.output_tensors = output_tensors
        cb.n_output_tensors = n_outputs

        args = [{
            "max_context_len": QWEN3_TTS_MAX_CONTEXT_LEN,
            "vocab_size": TALKER_VOCAB_SIZE,
            "special_eos_id": CODEC_EOS_TOKEN_ID,
            "max_new_tokens": QWEN3_TTS_MAX_CONTEXT_LEN,
            "keep_history": 0,
            **self.talker_sampling,
        }]

        ret = self.talker.init_llm_session(llm_args=args, llm_callback=cb)
        assert ret == 0, f"init talker llm session failed: {ret}"

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
    def _talker_result_callback(self, userdata, result_ptr, state):
        self = _userdata_to_obj(userdata)
        state_val = int(state)
        with self.lock:
            self.talker_last_state = state_val
            if state_val == int(LLMCallState.RKLLM_RUN_NORMAL):
                self.talker_steps += 1
            step = self.talker_steps
            if state_val in (
                int(LLMCallState.RKLLM_RUN_FINISH),
                int(LLMCallState.RKLLM_RUN_STOP),
                int(LLMCallState.RKLLM_RUN_MAX_NEW_TOKEN_REACHED),
                int(LLMCallState.RKLLM_RUN_ERROR),
            ):
                if state_val == int(LLMCallState.RKLLM_RUN_ERROR):
                    self.talker_failed = True
                if self._talker_done_event is not None:
                    self._talker_done_event.set()
        if state_val == int(LLMCallState.RKLLM_RUN_NORMAL):
            n = result_ptr.contents.num_tokens
            tokens = [int(result_ptr.contents.token_ids[i]) for i in range(n)]
        elif state_val == int(LLMCallState.RKLLM_RUN_FINISH):
            print(f"[talker] FINISH, total steps={step}")
        elif state_val == int(LLMCallState.RKLLM_RUN_STOP):
            print(f"[talker] STOP")
        elif state_val == int(LLMCallState.RKLLM_RUN_MAX_NEW_TOKEN_REACHED):
            print(f"[talker] MAX_NEW_TOKEN_REACHED, total steps={step}")
        elif state_val == int(LLMCallState.RKLLM_RUN_ERROR):
            print(f"[talker] ERROR")
        return 0

    def _talker_output_callback(self, userdata, tensors, n_tensors, state):
        """Copy the talker's ``past_hidden`` (output index 1) back to host."""
        self = _userdata_to_obj(userdata)
        for i in range(n_tensors):
            tensor_ptr = tensors[i]
            if tensor_ptr.attr is None or tensor_ptr.mem is None:
                continue
            attr = tensor_ptr.attr.contents
            if attr.index != 1:
                continue
            view = _fp16_view(tensor_ptr.mem.contents.virt_addr, attr.n_elems)
            with self.lock:
                self.past_hidden = view.reshape(-1).copy()
        return 0

    def _talker_embed_callback(self, userdata, tokens_ptr, num_tokens, embed_ptr, length):
        """Build the next talker input embedding for ``num_tokens`` positions."""
        self = _userdata_to_obj(userdata)
        dim = QWEN3_TTS_EMBED_DIM

        expected_len = num_tokens * dim * np.dtype(np.float16).itemsize
        if length != expected_len:
            print(f"[talker embed] invalid buffer: got {length}, want {expected_len}")
            return -1

        token_id = int(tokens_ptr[0])
        if token_id < 0 or token_id >= TALKER_VOCAB_SIZE:
            token_id = 0
        last_id_hidden = self.input_embed[token_id].astype(np.float32).copy()

        with self.lock:
            past_hidden = self.past_hidden
        if past_hidden is None:
            print("[talker embed] past_hidden not yet available")
            return -1

        code_input = np.concatenate(
            [past_hidden.astype(np.float32), last_id_hidden]
        ).astype(np.float16)
        code_input = code_input.reshape(1, 2, dim)
        code_ids = self.run_code_predictor(code_input)

        frame = [token_id] + [int(t) for t in code_ids]
        with self.lock:
            self.collected_codes.append(frame)
            step = self.talker_steps
        if self.frame_queue is not None:
            self.frame_queue.put(frame)

        next_embed = last_id_hidden.copy()
        for i, tid in enumerate(code_ids):
            tid = int(tid)
            if tid < 0 or tid >= CODEC_VOCAB_SIZE:
                tid = 0
            next_embed += self.codec_embed[i, tid].astype(np.float32)

        with self.lock:
            trailing = self.trailing_text_hiddens
            pad = self.tts_pad_embed
        idx = step - 1
        if trailing is not None and 0 <= idx < trailing.shape[0]:
            next_embed += trailing[idx].astype(np.float32)
        else:
            next_embed += pad.astype(np.float32)

        dst = _fp16_view(embed_ptr, num_tokens * dim)
        dst[:] = next_embed.astype(np.float16)
        return 0

    def _cp_result_callback(self, userdata, result_ptr, state):
        self = _userdata_to_obj(userdata)
        if int(state) != int(LLMCallState.RKLLM_RUN_NORMAL):
            return 0
        n = result_ptr.contents.num_tokens
        tokens = [int(result_ptr.contents.token_ids[i]) for i in range(n)]
        with self.lock:
            self.code_predictor_out_ids.extend(tokens)
            self.code_predictor_step += 1
        return 0

    def _cp_embed_callback(self, userdata, tokens_ptr, num_tokens, embed_ptr, length):
        """Fill the code_predictor embed buffer from the codec embedding table."""
        self = _userdata_to_obj(userdata)
        dim = QWEN3_TTS_EMBED_DIM

        expected_len = num_tokens * dim * np.dtype(np.float16).itemsize
        if length != expected_len:
            return -1

        with self.lock:
            step = self.code_predictor_step

        row = step - 1 if step > 0 else 0
        if row >= self.codec_embed.shape[0]:
            row = self.codec_embed.shape[0] - 1

        dst = _fp16_view(embed_ptr, num_tokens * dim)
        for n in range(int(num_tokens)):
            tid = int(tokens_ptr[n])
            if tid < 0 or tid >= CODEC_VOCAB_SIZE:
                tid = 0
            dst[n * dim:(n + 1) * dim] = self.codec_embed[row, tid]
        return 0

    # ------------------------------------------------------------------ #
    # Prefill construction
    # ------------------------------------------------------------------ #
    def build_prefill_embedding(
        self,
        text,
        instruct="",
        language="auto",
        speaker="",
        ref_speaker="",
        non_streaming_mode=False,
    ):
        """Construct the prefill embeddings, mirroring ``process_prefill_embedding``.

        Returns a ``(1, n_tokens, EMBED_DIM)`` fp16 ndarray suitable for
        ``session_run(embeds=...)``.
        """
        if not text:
            raise ValueError("input text is empty")

        # Wrap the text in Qwen3 chat markers. The token layout is:
        #   input_ids[0]     = <|im_start|>         (151644)
        #   input_ids[1]     = 'assistant'         (77091)
        #   input_ids[2]     = '\n'                 (198)
        #   input_ids[3..-3] = content tokens
        #   input_ids[-3]    = <|im_end|>           (151645)
        # So ``first_id = input_ids[3]`` is the first content token
        # and ``trailing_ids = input_ids[4:-5]`` covers the rest.
        assist_text = "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n"
        input_ids = self._encode_text(assist_text)
        dim = QWEN3_TTS_EMBED_DIM

        # 1) instruct embedding (chunked text_projection)
        instruct_projected = None
        if instruct:
            instruct_text = "<|im_start|>user\n" + instruct + "<|im_end|>\n"
            instruct_ids = self._encode_text(instruct_text)
            instruct_projected = self.text_projection_chunked(instruct_ids)

        # 2) Speaker embedding selection
        speaker_embed = None
        ref_speaker_lower = _to_lower(ref_speaker) if ref_speaker else ""
        if ref_speaker_lower and self.speaker_embeddings and ref_speaker_lower in self.speaker_embeddings:
            speaker_embed = self.speaker_embeddings[ref_speaker_lower].astype(np.float32).copy()
            print(f"[speaker] using voice-clone ref_speaker='{ref_speaker_lower}'")
        elif ref_speaker_lower:
            print(
                f"[speaker] ref_speaker='{ref_speaker_lower}' not found in "
                f"{'available: '+ ', '.join(sorted(self.speaker_embeddings)) if self.speaker_embeddings else 'no speaker_embeddings loaded'}"
            )

        input_speaker_lower = _to_lower(speaker) if speaker else ""
        if speaker_embed is None and input_speaker_lower:
            sid = SPK_ID.get(input_speaker_lower)
            if sid is not None:
                speaker_embed = self.input_embed[int(sid)].astype(np.float32).copy()
            else:
                print(f"[speaker] '{input_speaker_lower}' not in SPK_ID, skipping")

        # 3) Language id
        language_id = -1
        language_lower = _to_lower(language)
        if language_lower != "auto":
            language_id = CODEC_LANGUAGE_ID.get(language_lower, -1)

        is_config_speaker = bool(speaker_embed is not None)
        dialect = SPK_IS_DIALECT.get(input_speaker_lower, "false")
        if (
            (language_lower == "auto" or language_lower == "chinese")
            and is_config_speaker
            and dialect != "false"
        ):
            language_id = CODEC_LANGUAGE_ID.get(dialect, language_id)

        # 4) base tts control tokens, projected through text_projection
        base_ids = [TTS_BOS_TOKEN_ID, TTS_EOS_TOKEN_ID, TTS_PAD_TOKEN_ID]
        base_proj = self.text_projection_chunked(base_ids)
        tts_bos = base_proj[0]
        tts_eos = base_proj[1]
        tts_pad = base_proj[2]
        with self.lock:
            self.tts_pad_embed = tts_pad.astype(np.float16)

        # 5) codec-side control tokens, looked up via input_embed
        if language_id == -1:
            codec_ctrl = [CODEC_NOTHINK_ID, CODEC_THINK_BOS_ID, CODEC_THINK_EOS_ID]
        else:
            codec_ctrl = [CODEC_THINK_ID, CODEC_THINK_BOS_ID, language_id, CODEC_THINK_EOS_ID]
        codec_input = np.zeros((len(codec_ctrl), dim), dtype=np.float32)
        for i, tid in enumerate(codec_ctrl):
            tid = int(tid)
            if tid < 0 or tid >= TALKER_VOCAB_SIZE:
                tid = 0
            codec_input[i] = self.input_embed[tid].astype(np.float32)

        if speaker_embed is not None:
            codec_input = np.concatenate(
                [codec_input, speaker_embed[None, :].astype(np.float32)],
                axis=0,
            )

        for tid in (CODEC_PAD_ID, CODEC_BOS_ID):
            tid = int(tid)
            if tid < 0 or tid >= TALKER_VOCAB_SIZE:
                tid = 0
            codec_input = np.concatenate(
                [codec_input, self.input_embed[tid].astype(np.float32)[None, :]],
                axis=0,
            )
        codec_total = codec_input.shape[0]

        # 6) role header (text projection)
        role_ids = [151644, 77091, 198]
        role_proj = self.text_projection_chunked(role_ids)

        # 7) fused header: text-side (tpad except last which is tbos) + codec-side
        fused = np.zeros((codec_total - 1, dim), dtype=np.float32)
        for i in range(codec_total - 1):
            text_node = tts_pad if i < codec_total - 2 else tts_bos
            fused[i] = text_node.astype(np.float32) + codec_input[i]

        # 8) streaming vs non-streaming content
        if non_streaming_mode:
            content_ids = [input_ids[i] for i in range(3, max(3, len(input_ids) - 5))]
            if len(content_ids) > 0:
                content_proj = self.text_projection_chunked(content_ids)
                content_proj = np.concatenate(
                    [content_proj, tts_eos[None, :].astype(np.float32)], axis=0
                )
            else:
                content_proj = tts_eos[None, :].astype(np.float32)

            content_len_plus_eos = content_proj.shape[0]
            codec_pad_id = int(CODEC_PAD_ID)
            if codec_pad_id < 0 or codec_pad_id >= TALKER_VOCAB_SIZE:
                codec_pad_id = 0
            codec_pad = self.input_embed[codec_pad_id].astype(np.float32)
            for i in range(content_len_plus_eos):
                content_proj[i] = content_proj[i] + codec_pad

            tail_fused = (tts_pad.astype(np.float32)
                          + self.input_embed[int(CODEC_BOS_ID)].astype(np.float32))
            trailing = np.concatenate(
                [content_proj.astype(np.float32), tail_fused[None, :]], axis=0
            )
            with self.lock:
                self.trailing_text_hiddens = trailing.astype(np.float16)

            final = np.concatenate([fused, content_proj, tail_fused[None, :]], axis=0)
            final = final.astype(np.float16)
        else:
            # streaming mode: prefill with first content token, store trailing.
            first_id = input_ids[3] if len(input_ids) > 3 else 0
            if first_id < 0 or first_id >= TEXT_VOCAB_SIZE:
                first_id = 0
            first_proj = self.text_projection_chunked([first_id])[0]
            first_fused = first_proj.astype(np.float32) + codec_input[codec_total - 1]
            trailing_ids = input_ids[4:max(4, len(input_ids) - 5)]
            if trailing_ids:
                trailing_proj = self.text_projection_chunked(trailing_ids)
            else:
                trailing_proj = np.zeros((0, dim), dtype=np.float16)
            trailing = np.concatenate(
                [trailing_proj.astype(np.float32), tts_eos[None, :].astype(np.float32)],
                axis=0,
            )
            with self.lock:
                self.trailing_text_hiddens = trailing.astype(np.float16)

            final = np.concatenate([fused, first_fused[None, :]], axis=0)
            final = final.astype(np.float16)

        final = np.concatenate([role_proj.astype(np.float16), final], axis=0)
        if instruct_projected is not None:
            final = np.concatenate(
                [instruct_projected.astype(np.float16), final], axis=0
            )

        n_tokens = final.shape[0]
        embeds = final.reshape(1, n_tokens, dim)
        print(
            f"[prefill] tokens={n_tokens}, dim={dim}, non_streaming={non_streaming_mode}"
        )
        return embeds

    # ------------------------------------------------------------------ #
    # Audio decoding
    # ------------------------------------------------------------------ #
    def transcode_for_speech_decoder(self, frames):
        """Convert collected frames into a single ``(kFeatSize, L)`` int32 array.

        Each frame from the talker contains ``kFeatSize=16`` tokens laid out as
        ``[main_token, code_predictor[0], ..., code_predictor[14]]``.
        The speech decoder expects one channel per token position, so channel
        ``c`` at frame ``t`` is simply ``frames[t][c]``.
        """
        n_frames = len(frames)
        if n_frames == 0:
            raise RuntimeError("no frames generated by talker")

        codes = np.zeros((kFeatSize, n_frames), dtype=np.int32)
        for t, frame in enumerate(frames):
            for c in range(kFeatSize):
                codes[c, t] = int(frame[c]) if c < len(frame) else 0
        return codes

    def _decode_window_with_context(self, ctx_codes, new_codes):
        """Decode a (25 left + N new) window and return the new audio.

        ``ctx_codes`` is the ``(kFeatSize, kLeftContextSize)`` left-context
        block from the previous window (or ``None`` to use zeros). ``new_codes``
        is the freshly-produced frame block (may be shorter than 25 at
        the end of generation). Output: 1D float32 of length
        ``new_frames * kTotalUpsample``.
        """
        n_new = 0 if new_codes is None else new_codes.shape[1]
        if n_new == 0:
            return np.zeros(0, dtype=np.float32)

        block = np.zeros((kFeatSize, kWindowLen), dtype=np.int32)
        n_ctx = 0
        if ctx_codes is not None:
            n_ctx = min(ctx_codes.shape[1], kLeftContextSize)
            block[:, :n_ctx] = ctx_codes[:, :n_ctx]
        block[:, n_ctx:n_ctx + n_new] = new_codes

        outputs = self.speech_decoder.inference(inputs=[block.reshape(1, kFeatSize, kWindowLen)])
        if outputs is None:
            raise RuntimeError("speech_decoder inference failed")
        audio_full = np.asarray(outputs[0]).reshape(-1).astype(np.float32)
        take_start = min(n_ctx * kTotalUpsample, audio_full.size)
        take_end = min(take_start + n_new * kTotalUpsample, audio_full.size)
        if take_end <= take_start:
            return np.zeros(0, dtype=np.float32)
        return audio_full[take_start:take_end].copy()

    def _decode_window_full(self, codes_16xl, first=False):
        """Decode a complete (up to 50-frame) window.

        Helper for the streaming worker. ``codes_16xl`` may be shorter
        than 50 frames for the final drain; we just feed whatever we
        have into the model with zero padding.
        """
        n = codes_16xl.shape[1]
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        block = np.zeros((kFeatSize, kWindowLen), dtype=np.int32)
        block[:, :n] = codes_16xl
        ctx = 0 if first else kLeftContextSize
        new = n if first else min(kChunkSize, n)
        if first:
            new = min(n, kWindowLen)
        outputs = self.speech_decoder.inference(inputs=[block.reshape(1, kFeatSize, kWindowLen)])
        if outputs is None:
            raise RuntimeError("speech_decoder inference failed")
        audio_full = np.asarray(outputs[0]).reshape(-1).astype(np.float32)
        take_start = min(ctx * kTotalUpsample, audio_full.size)
        take_end = min(take_start + new * kTotalUpsample, audio_full.size)
        if take_end <= take_start:
            return np.zeros(0, dtype=np.float32)
        return audio_full[take_start:take_end].copy()

    # ------------------------------------------------------------------ #
    # End-to-end (SDK entry point)
    # ------------------------------------------------------------------ #
    def run_with_progress(
        self,
        text,
        output_path,
        language="auto",
        speaker="",
        ref_speaker="",
        instruct="",
        non_streaming_mode=False,
        play=True,
        volume=0.5,
        audio_device=None,
        stop_event: Optional[threading.Event] = None,
        on_step: Optional[Callable[[int, float], None]] = None,
        on_audio_chunk: Optional[Callable[[np.ndarray, int], None]] = None,
    ):
        """Synthesise ``text`` and write a WAV.

        Mirrors the upstream ``qwen3_tts_demo.run`` plus three SDK
        hooks the v2 session layer needs: ``stop_event`` aborts the
        talker mid-flight; ``on_step(n_frames, audio_sec)`` reports
        progress per decoder window; ``on_audio_chunk(audio, sr)``
        streams PCM live to the browser without re-reading the file.
        """
        self.load_resources()
        self.init_text_projection()
        self.init_speech_decoder()
        self.init_code_predictor()
        self.init_talker()

        self.audio_player = (
            AudioPlayer(kSampleRate, device=audio_device, volume=volume)
            if play else None
        )

        try:
            prefill = self.build_prefill_embedding(
                text=text,
                instruct=instruct,
                language=language,
                speaker=speaker,
                ref_speaker=ref_speaker,
                non_streaming_mode=non_streaming_mode,
            )

            t0 = time.time()
            audio = self._run_talker_streaming_with_progress(
                prefill,
                stop_event=stop_event,
                on_step=on_step,
                on_audio_chunk=on_audio_chunk,
            )
            print(
                f"[talker+decoder] {len(self.collected_codes)} frames, "
                f"{audio.size / kSampleRate:.2f}s of audio in {time.time() - t0:.2f}s"
            )

            if audio.size == 0:
                print("[warn] no audio produced", flush=True)
            else:
                _write_wav(output_path, audio, kSampleRate)
                print(f"[done] saved wav to {output_path}")
            return audio
        finally:
            if self.audio_player is not None:
                self.audio_player.close()
                self.audio_player = None

    def _run_talker_streaming_with_progress(
        self,
        prefill_embeds,
        stop_event: Optional[threading.Event] = None,
        on_step: Optional[Callable[[int, float], None]] = None,
        on_audio_chunk: Optional[Callable[[np.ndarray, int], None]] = None,
    ):
        """Run talker + decoder worker with stop + progress hooks."""
        with self.lock:
            self.talker_failed = False
            self.talker_last_state = None
            self.talker_steps = 0
            self.collected_codes = []
            self.past_hidden = None
            self._first_window = True

        self.frame_queue = queue.Queue()
        self._streamed_audio = []
        self._talker_done_event = threading.Event()

        decoder_t = threading.Thread(
            target=self._decoder_worker_with_stop,
            name="qwen3tts-wdnai",
            args=(stop_event, on_step, on_audio_chunk),
            daemon=True,
        )
        decoder_t.start()

        ret, _ = self.talker.session_run_async(embeds=prefill_embeds)
        if ret != 0:
            self._talker_done_event.set()
            self.frame_queue.put(None)
            raise RuntimeError(f"talker.session_run_async failed: ret={ret}")

        while not self._talker_done_event.is_set():
            if stop_event is not None and stop_event.is_set():
                try:
                    self.talker.session_stop()
                except Exception:
                    pass
                break
            time.sleep(0.02)

        if self.talker_failed:
            self.frame_queue.put(None)
            decoder_t.join(timeout=5)
            raise RuntimeError(
                f"talker generation failed, last_state={self.talker_last_state}"
            )

        self.frame_queue.put(None)
        decoder_t.join(timeout=30)
        if decoder_t.is_alive():
            print("[decoder] worker did not finish in time", flush=True)

        if not self._streamed_audio:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._streamed_audio, axis=0)

    def _decoder_worker_with_stop(
        self,
        stop_event: Optional[threading.Event],
        on_step: Optional[Callable[[int, float], None]],
        on_audio_chunk: Optional[Callable[[np.ndarray, int], None]],
    ):
        """Consume frames from ``self.frame_queue`` and stream decoded audio.

        Mirrors the upstream ``_decoder_worker`` but checks ``stop_event``
        on every loop iteration and emits ``on_step`` / ``on_audio_chunk``
        callbacks after each decoded window.
        """

        def _get(timeout):
            try:
                return self.frame_queue.get(timeout=timeout)
            except queue.Empty:
                return None

        def _stopped():
            return stop_event is not None and stop_event.is_set()

        buffer = []
        prev_context = []

        # First window: collect up to 50 frames, decode.
        while len(buffer) < kWindowLen and not _stopped():
            frame = _get(0.1)
            if frame is None:
                if self._talker_done_event.is_set():
                    break
                continue
            buffer.append(frame)
        if buffer and not _stopped():
            codes = self.transcode_for_speech_decoder(buffer)
            audio = self._decode_window_full(codes, first=True)
            if audio.size:
                self._streamed_audio.append(audio)
                if self.audio_player is not None:
                    self.audio_player.play_chunk(audio)
                if on_audio_chunk is not None:
                    try:
                        on_audio_chunk(audio, kSampleRate)
                    except Exception:
                        pass
            if on_step is not None:
                try:
                    on_step(
                        len(self.collected_codes),
                        sum(a.size for a in self._streamed_audio) / kSampleRate,
                    )
                except Exception:
                    pass
            if len(buffer) > kLeftContextSize:
                prev_context = buffer[-kLeftContextSize:]
            else:
                prev_context = buffer[:]
            buffer = []

        # Subsequent windows.
        while not _stopped():
            while len(buffer) < kChunkSize:
                frame = _get(0.1)
                if frame is None:
                    if self._talker_done_event.is_set() and self.frame_queue.empty():
                        break
                    continue
                buffer.append(frame)
            if not buffer:
                break
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
                if on_audio_chunk is not None:
                    try:
                        on_audio_chunk(audio, kSampleRate)
                    except Exception:
                        pass
            if on_step is not None:
                try:
                    on_step(
                        len(self.collected_codes),
                        sum(a.size for a in self._streamed_audio) / kSampleRate,
                    )
                except Exception:
                    pass
            if prev_context:
                combined = prev_context + buffer
            else:
                combined = buffer
            if len(combined) > kLeftContextSize:
                prev_context = combined[-kLeftContextSize:]
            else:
                prev_context = combined[:]
            if len(buffer) <= kChunkSize:
                buffer = []
            else:
                buffer = buffer[kChunkSize:]

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #
    def release(self):
        for name in ("talker", "code_predictor", "text_projection", "speech_decoder"):
            rk = getattr(self, name)
            if rk is not None:
                try:
                    rk.release()
                except Exception as exc:
                    print(f"[release] {name}: {exc}")
                setattr(self, name, None)


# --------------------------------------------------------------------------- #
# v2 SDK backend
# --------------------------------------------------------------------------- #
class TTSBackend:
    """TTS backend — text-in, audio-out.

    ``run_text(prompt)`` returns ``(prompt_str, [Attachment])`` where
    the attachment carries the synthesised WAV bytes. ``LLMSession``
    unpacks the tuple and assigns the attachment to ``LMOutput.attachment``
    so the v2 caller sees a unified output.
    """

    def __init__(
        self,
        spec,
        *,
        npu: Optional[int] = None,
        device_id: Optional[Union[bytes, str]] = None,
    ):
        self.spec = spec
        self.model_dir = _resolve_tts_model_dir(spec)
        if not self.model_dir.is_dir():
            raise FileNotFoundError(
                f"TTS model directory not found: {self.model_dir}"
            )

        # Resolve NPU selection **once** (the four RKNN3Lite handles in
        # Qwen3TTSDriver — text_projection / talker / code_predictor /
        # speech_decoder — all share this resolved bytes).
        resolved_device_id = resolve_device_id(RKNN3Lite, npu, device_id)

        # Instantiating ``Qwen3TTSDriver`` loads the four RKNN3 models
        # + embed tables up front; that cost is paid for in
        # ``run_text`` and friends. ``TTSBackend`` is intentionally
        # not importable for non-TTS tasks — keep the import
        # localised here.
        self.driver = Qwen3TTSDriver(
            model_dir=str(self.model_dir),
            device_id=resolved_device_id,
        )
        # Eagerly trigger the full TTS load (embed tables + 4× load_rknn
        # + 4× init_runtime). Without this, the heavy init work would be
        # deferred to the first ``run_text`` call, making the "运行中"
        # status on the 算力配置 page misleading (NPU runtime not built,
        # .weight files not even mmap'd). Mirrors the eager semantics
        # already used by TextBackend / VLMBackend / ASRBackend.
        self.driver.ensure_loaded()
        LOG.info("TTSBackend initialised; model_dir=%s", self.model_dir)

    # ------------------------------------------------------------------ #
    def run_text(self, prompt: str, **overrides: Any) -> Tuple[str, List[Attachment]]:
        """Public entry point invoked by ``LLMSession._dispatch``.

        Drives the TTS pipeline via the demo's ``run_with_progress``
        and packages the resulting WAV bytes as an
        :class:`Attachment` (``kind="audio"``).
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
        on_audio_chunk = overrides.get("on_audio_chunk", None)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            tmp_path = fp.name
        try:
            audio = self.driver.run_with_progress(
                text=prompt,
                output_path=tmp_path,
                language=language, speaker=speaker, ref_speaker=ref_speaker,
                instruct=instruct, non_streaming_mode=non_streaming,
                play=not no_play, volume=volume, audio_device=None,
                stop_event=stop_event, on_step=on_step,
                on_audio_chunk=on_audio_chunk,
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

    # ------------------------------------------------------------------ #
    def run_images(self, prompt: str, images: list, **overrides: Any) -> str:
        """TTS ignores images — fall back to plain text."""
        return self.run_text(prompt, **overrides)[0]

    def run_video(self, prompt: str, frames: list, **overrides: Any) -> str:
        """TTS ignores video frames — fall back to plain text."""
        return self.run_text(prompt, **overrides)[0]

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        try:
            self.driver.release()
        except Exception:  # noqa: BLE001
            pass