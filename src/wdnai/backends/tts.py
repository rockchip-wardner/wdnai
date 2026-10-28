"""Qwen3-TTS backend — text-in, audio-out.

The full RKNN3 driver (the four ``.1828.rknn`` models, embed tables,
prefill construction, talker/code_predictor/speech_decoder pipeline,
audio playback) is vendored in this module — originally imported
from ``rknn3-toolkit-lite/examples/Qwen3-TTS/qwen3_tts_demo.py``
via ``sys.path`` injection, which made the SDK un-importable on
hosts without the upstream demo directory. ``import wdnai.backends.tts``
is now self-contained.

``run_text(prompt)`` returns the prompt unchanged along with a
single :class:`AudioAttachment` carrying the synthesised WAV bytes; the
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

VoiceClone speakers (``news_female`` / ``baby_girl`` / ``lovely_girl``
/ ``cool_lady`` / ``male_announcer`` / ``uncle_bass``) are bundled in
this module — no external ``qwen3tts_speaker_embed.h`` is needed.

The directory is resolved from the ``ModelSpec`` in this order:
``spec.tts_model_dir`` → ``spec.options["model_dir"]`` →
``$TTS_MODEL_DIR`` → ``$WDNAI_MODELS_DIR/Qwen3-TTS`` →
``/userdata/models/Qwen3-TTS``.

Optional overrides (forwarded from ``LLMSession.respond(**overrides)``):

* ``ref_speaker``   — voice-clone speaker name (``"news_female"`` /
                      ``"baby_girl"`` / ``"lovely_girl"`` / …).
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
import re
import struct
import subprocess
import tempfile
import threading
import time
import unicodedata
import zlib
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np

from .._librknn import RKNN3Lite, build_llm_callback
from ..attachment import AudioAttachment
from ._common import resolve_device_id

# LLMCallState integer values used by the TTS callbacks below. Kept
# inline so this module doesn't reach into the shim's internal
# IntEnum; the values mirror rknn3_api.h.
_LLM_RUN_NORMAL = 0
_LLM_RUN_FINISH = 2
_LLM_RUN_STOP = 3
_LLM_RUN_MAX_NEW_TOKEN_REACHED = 4
_LLM_RUN_ERROR = 5

LOG = logging.getLogger("wdnai.backends.tts")


# --------------------------------------------------------------------------- #
# Constants copied from rknn3_model_zoo/examples/Qwen3_TTS/cpp/talker.h
# --------------------------------------------------------------------------- #
CODEC_MAX_CONTEXT_LEN = 15
QWEN3_TTS_MAX_CONTEXT_LEN = 8192
QWEN3_TTS_EMBED_DIM = 2048
TEXT_PROJECTION_CORE_MASK = 0x01
# 音色提取器（ECAPA-TDNN speaker encoder）的 NPU 上下文模式：
#
#   resident  —— 常驻。worker 启动时就开好 NPU 上下文，第一次提取不用等
#                init_runtime（约 0.45 s）。
#   on_demand —— 默认。每次提取时开上下文、用完立刻释放 —— TTS 主链路
#                平时完全看不到这个上下文。代价是每次提取多付一次
#                init_runtime（约 0.45 s），上传音色是低频操作，可以接受。
#   off       —— 不启用音色提取（模型文件都不碰）。
#
# 三种模式对 TTS 合成速度的影响实测在噪声内（同一块板、同一外部负载下
# 逐条对比 talker 每帧耗时）。用环境变量 ``WDNAI_SPEAKER_ENCODER_MODE`` 覆盖。
SPEAKER_ENCODER_MODE_ENV = "WDNAI_SPEAKER_ENCODER_MODE"
SPEAKER_ENCODER_MODE_DEFAULT = "on_demand"
# 音色提取器独占的核。它是低频操作，给 1 个核就够。初始化失败时
# :meth:`Qwen3TTSDriver.init_speaker_encoder` 会回退到 0xff 再试一次。
# 用环境变量 ``WDNAI_SPEAKER_ENCODER_CORE_MASK`` 覆盖（十进制整数）。
SPEAKER_ENCODER_CORE_MASK = 0x02
SPEAKER_ENCODER_CORE_MASK_ENV = "WDNAI_SPEAKER_ENCODER_CORE_MASK"

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

# --------------------------------------------------------------------------- #
# 用户音色（现场提取的 voice-clone 向量）
# --------------------------------------------------------------------------- #
# 用户音色目录：app 进程把路径写进环境变量 ``WDNAI_TTS_USER_VOICES_DIR``，
# worker 启动时 ``os.environ.copy()`` 继承 app 的环境，所以 worker 拿得到。
#
# 目录里每个音色一对文件：
#     <name>.npy    2048 维 float32 的 x-vector
#     <name>.json   元信息（原始文件名 / 时长 / 采样率 / 创建时间）
#
# **app 是用户数据的归属方，负责写；worker 负责在启动时读回来** —— 这样
# worker 重启后用户音色自动恢复，不依赖 app 去推送。
TTS_USER_VOICES_DIR_ENV = "WDNAI_TTS_USER_VOICES_DIR"
USER_SPEAKER_EMBED_DIM = 2048
_USER_SPEAKER_NAME_MAX = 32
# 音色名规则：**Unicode 字母 / 数字 / 下划线 / 连字符**，且以字母数字开头。
#
# 允许中文：名字是给人看的（「我的声音」比 ``my_voice`` 好认），Linux
# 文件名对 UTF-8 没有限制。真正要挡住的是**文件系统上危险**的字符 ——
# 路径分隔符、``.``、隐藏文件、控制字符 —— 这条正则正好只放行字母数字
# 和 ``_`` ``-``。``[^\W_]`` = 字母或数字（``\w`` 在 str 模式下是 Unicode
# 感知的，会匹配汉字；排除 ``_`` 是因为下划线不能开头）。
_USER_SPEAKER_NAME_RE = re.compile(r"^[^\W_][\w-]*$", re.UNICODE)


def _normalize_user_speaker_key(name) -> str:
    """音色名 → key（NFC 归一化 + 去空白 + 转小写）。

    注册和注销必须走同一套归一化，否则名字对不上 —— 会出现「文件删了、
    worker 内存里还留着」的不一致。

    统一做 **NFC 归一化**：macOS 之类的系统会把带音标的字母拆成
    「基字母 + 组合符号」（NFD），不归一化的话同一个名字会有好几种
    字节形态 —— 存进去的是一种、查表用的是另一种，就会「明明注册过
    却查不到」。
    """
    return unicodedata.normalize("NFC", str(name or "")).strip().lower()

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
# Helpers
# --------------------------------------------------------------------------- #
def _fp16_view(ptr, n_elems):
    """Create a fp16 ``numpy`` view of a raw memory pointer.

    Used by the embed callbacks below: the shim passes the C-side
    ``embed_ptr`` as a Python int (the raw address) and ``length`` in
    bytes; we wrap the address as a writable fp16 ndarray.
    """
    return np.ctypeslib.as_array(
        ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint16)),
        shape=(n_elems,),
    ).view(np.float16)


# --------------------------------------------------------------------------- #
# Bundled speaker embeddings
# --------------------------------------------------------------------------- #
# VoiceClone reference vectors from the upstream C++ demo's
# ``qwen3tts_speaker_embed.h``. That header only ships with the rknn3
# dev tree, so without this bundle a clean board would silently fall
# through to the built-in voice and every reference speaker would
# produce identical audio. Six 2048-dim fp16 vectors are stored inline
# as zlib-compressed base64 (~31 KB on disk, ~24 KB decoded).
#
# 这张表和 ``examples/Qwen3_TTS/cpp/qwen3tts_speaker_embed.h`` 是**同一份
# 向量**的两种编码(C++ 侧存字面量、Python 侧存 zlib+base64)，必须同步
# 更新 —— 用 ``scripts/build_ref_voices.py`` 一次生成两份，别手改。
_BUNDLED_SPEAKER_EMBEDDINGS_B64: dict[str, tuple[str, ...]] = {
    "news_female": (
        "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",
    ),
    "baby_girl": (
        "eNoNlodDE2fDwHEPLCoOUBESuCTP3T03koBsRbGOukdd4B6oVV5FsO5dqYpaBzIUJCF3ufHcSgI4X3FP+tZR7edsadVqtUWs1tX69Y/4jdZqkg6p9ng5aFd2iTVOl4pT9Q34fr4Z+gmt1Pe496hBsWLnTIFDV+ggJQTuZ95r+1CvWEKtlkPxSZ61Whl9kKERJbrhI1s340lknLs33qqKptYonckNntnoH1cPdiq1KTBNbQMJObnkOfNGTBEjAiq5BB/vehwxm81xhcZmE6lxw9mlwgWlWBsj3tGDlQ+V3/Bebbqpv+e6a75aBLfhubGteZdrNWiuNzpydSdbTRSLc3UGfs0uMC5q3oiRXB/AsndRoQYDUvTrMl485s3zn8M+1arhsorbltvmLlwYa40dJS9VnXGCpQnTQ/5HOWFOcIepjVoRbOMBmJtO1H8kQ9AC1w26E3m6IlnH464yd2Ax/I/4dfGg4mFxwmEITkQlOfey7QIvD7kMjx5u26YnwNye805dAUWeNXwWlEGhZ7fSV+iPTMwZ9Ir4P+UFbsSHyHmyKGXCy+ATuFvfj8z+eHWPZaU4g0rhFoMhMNf6pa/OG8QCUvfOBd2MdawMs+3XfIvY+fTP2G8onIivtLQ9DGzuc1hTqpK8jJ2Q91oZ9gdhC1tn0mG77i9sPYyWRJzFIDO9xfpIT6RPt/jgPNCXfgaWKSfd4eVb0UFUD45EhhqPqTVwmqUJ/YAQDk5HZn2llKrEeUYLsNNzJXDQyi84uClqByqifqdOR0Qr6yENOllR4Y+CRThbM9N4qJ2PyDS2s/nRcezZGMI4SX8Bs8R7+p+eRk+0PtVo4uhhe6S2i4gTWgXCuH6YwFZzfuXTqAjLFZVHBcxYo6n6UTwfF13QXRpBzIrvC1Tr1fJEUw3asO/DrlDfJjVN3R25WFrHRTHI9YYFyg/KV8wr01LURvtNycUmkb192ViwmGO+pgyMtXBtPLUoicV6YXAuwmKOgYXKVe92crd7DPlcaEqs1z5XWyOcXQxaCBnsadcl1uqIwC9KLu2y3lFIr7lExdDb8Q6UT++NujnH0+u10YFuYKhyE6VVFrLFioJaRZdqnDKCTpT/J2Tjl+F7XzToreDWHG9F7FpFp3K02dZ6o1q3E3Ls2pB6523PXoNzVyuzLTnG4kAbriV76Whn1JI4zRpCKTkFhhhDNEq5Cwoh0Wk6lWfcUU9VfSI48MnKM2mcOxMUioOMDlIwO9zzA4DyROsyvjnb0tmOeAgleNru7LaMb7A3jZ4dM9O7nlpoHmXfqN7fUme5L/2HOOm5Zr8gxRuV0jCiD9uc2mrhsLbVKtZM3WBs9IYRYdReZqKmYvnyHQITXZHvRA02PdyfQuo25kVlZ2m3NZIicUMMrTqGu5m7lQcqNrvnillRGGGlV3vHRCQZm8jIwAn5HLgRk6d8BsZSf5U3sgN8BcJllMpMO/RasKK5Yhj065MPLHHkoyhMpBX8A3tc6GycAw1KGGju+JZ6ysVKXZnPK5vDr4SRdG1ZeHHnuM1e77586XdrTdkWGAI86AbqQm9zRRGZxnOyPVsktFAmU7/Qt418kyyto1Z7WpZ1081Mc1a1DGKSJZRYJk8jR0CNLIta4amzL6BugL+0TtYQcyaxw7KczyXmKKPRAvaS7HeMwlqjgQEAcP2SOpheDJooeWyKXG/+yfYZLBLPmn0kT2RE3EIb8e+lDIWuzgj3Cx7fZmmVn4czuBV+jVlm6Sld85uMo/bhPEZOAyGepfxc13CcRIu0M8JnB0IlBdthTPfOiMkHcWq28Fa4SXzUirVfqDUFF5lgbytsFLSgEthf/xmZgJn8Rt3tCqEMMoloALMCzdQ4x5/oHkP7tjjOEZv5XnpvmhI5jIoPq2xnKZIJUoxcIWcwqUQtOUaN4cwH7uLIyKDbwgDMhvmCRiwWbPSyiI+WpJg68An/t9ZL3+yqjHqOMdoUYR1rK2aKgyQMNuVukg9FQGTZtoRfi+RhYWUtiqWqEQkT4ZvivURezyN4q+5FSgQON34FdxBhB+5yQzoSwgbpCFNAhWnZrglKMftl+GQsyDil/AiGwY9uIASjHGoxOt8zQSkQJ5E8e09ZUJYG98hD2Inudd5x8Bpajn3vWGntj6WZblo72EuFPAoHedoZZQUN9OYKxC9YuyXU2t7xo/XXMNtTojQk7BLnHlwjNyiTUSv6jiL6uYTuzGX1G627ujM6mW2vVcldaLNH859XPobOgEc8L7tsMLorN80pvr/ss/hQYCVtnKvkJr9N9lvHSh7yF2GTaFduxnxuWqyM5K9W5Emj//XsfylTVCHbRcIPZx18TG/XX7IABrjp5BBjPZfBjjryRHBpnfEcarh81NGJqsd9tlBipf6gPGZPE09h2RXLM+oRbgGKMho+UDpQE4X38jzvVGco2RLNadsbEMo7tY9e58b0Vj3y1afxt8DwpACxGndrBnTAMyhUTME22IIdfWCp6FBSelJSUYVTiqMGsRuFWHqN3IJW8UT/EmXs4UHgKWxjawIn82+0UJohX613602Z0VQJWmreqjmxQjqEOqvsOTiNmBlX2zVEShL/JMPswdazrmNqS8lsfuAtt++krOWplN7jzf41NEZv4L5Hz7EstS/fA9Tg812y+SgbrvfsNRRcZFJqVsEvfJTwN/oj5qjake2rNii15mHWReqyqA/4ydg1nnPUM2u4OswSYV8AJpru2QdXzip3A2SfSfwu9ynb7P4Cz/HNj7fIs/U69qRwljoJ+jNPAsXyFJBNkLYR8DpR4QijnfoNNEB/rWepfrDSsxflGHusD8kj+gp7g/ZSTN4fIa73dfW8I9P1c+wU7Db5R0ks+kNtD8+iU+K1fwvewt3o2ykUcx28r9QF1J/q3xWFbKMM1LN4tfk1nVXu9+Uzt40r5PgA4I5gX9E6sZ29g1WzVfo93x9aqL3m4Lye5oo6KoTxwUlssL9AGycmqp29JfhUPZsshu/FZHVZVUf5tfQWV9VtcB6e5vbo/5AjmAZxHjVALyldJl4VBrockSfIu2xblgNe8bhCUCa8JfBW3KZZ/SRdaUuxzmYGa4M4U5f+kNKmKa3wNGGrOIa6wzfAqugybQuRk1BELe/5DObI6dXphJfS1JeQg8fIWuyMNE+Y6XjaJFhsLTfVJlEtrMeMGiHdu1UfDiV0GDmo6/pjqtr2KzgTjlv6UJVGTGB/hUCWwgf0e9u3VBBshFV4lDaUeQAmxExhTfINcJ9YazsuPmT3i5+imxU75a+s1+Wr+HPyPDndPpnqQeZx1yQtpghM0c4SQ8ii6GFwp9qbGaZf9Qfj08q625cymdEbnG3Vr33RcqgCSpf4CCNAVVjfqKvIYPDAFWSPJWda86hS12hsl4GLMbEF6CMa6CxGd+kyaSX5lNoqHJfjtSDfCG+ytLxjNlqE6o1DPSP5J9ZS4Z49jhyAbmsj4KfMnIDD1lRM8vNcoj6J2AyGUc/UUh43DY19hEbYT1Vt9txi6+kcIVW8cWR8VS6/jmLZ+drv1Hj5JZiNLScSmWfhqyMz0+qIWZKdDyPW4GzlZAbKtejTyl+9zbW27njW7TtKOzlFXUqrxK7wWmlhfH+Fp3IMS8UAsTcaz+bCXcJKvVY/FJMb3oZ7CSYkvYWr4PLKarl8eyE6WbyWaUuVuk+hFfQw+p08EAbxx/HJDh3Vc41cPeplpcoG2O/wpeiIuNR8SUmnZ2mD2N+Id8R6uiPWRGtklsPvtTFxK/U1jhryKAUSusIrxGT7YMs8Ywl+R5CUBs+2+BOWBO+Z8iswnn5nCdXtZT2p+wxrLDo03hWGu2ETPUV3Gzo+A19ne4z34xuZdv4gZpW0mJ0gfY07YBphMS56b4lj7aqzFdhWNdX+ABzA2+D51N9x9aVp4kx/F5KH7cOz6EEWnnpIdLTdNw6RCcpL6Tbcq9bpVUyGPBQW8IfIbxTMPw58Hu1CaWIHe29yNqxj7yog4Tv+hrqc6GzKwAOWJ2CtnlzGspOsx8hlMd8wF0EBsUs5ViFhtwKlvFMbjnz0UDYJXhQ+p37w/88UJC3Vf7ZvFbxwEODofvgGdrE3LyWf6Qu3e48aJYLZkylo0hPsJ7ZOCiOaWIbq5y31zMry4fLPdjk02XvQFwXfiq2QXbuGz0Fj3dmSn0hInO5JZVtAqcwq1LEjwd9dJ8KZ0G5sUFxkd/9mECF3hm1hPptrSvLdhQP1w1o7OjqQEBmFfPp2T1dtgdGMP+Wdj39ArcrvG+e4tXwJ+U57BDnluLFQfUymSa2FTf8aPEb4xXNB0fCB5vPCSVjg7WgOpgjHYGaTiPEvagbom3plKF/CdKUf3E48wRaJq/FO8m1bUypZj+BSuYWxaz07iBtKgl7gb2Mg9SLVzScgDMhHlsirU5dQLZVA2XXQ0rbP/VzaoWx27SEGxKYK3xlF2BY1i49BIYeiOr1QU6jfURdbuvE2vNx8Gi8h38gTtRWmdHyZZas6SXRZcHGutZcIbRmcTXPoU/XvxG5kvmpz/EizaIaSVtKZe3Zgtzqo6gK4kpRl7KVqxJ3KzfJEcZa3nqsB/8CKqlEgFs2hByidlJdKML5aS635Bb6CkxTPv8cmUKf1RbJb+tIzobIhUGtdxDx1bVSHwEyqGT0S+YVjIKjSwloDSWxjxTnrLMdl9gDcZzTDjrK7MRIlq2ZqtvMO2gld6B1/fZ/bN1wvY5Y6nYFXQrS9n7aO3Ch8i9+C8e5V+l/QVLoR/Vq5wNGI0on++khBR3O8hryPTR0yj1molyi7D2Xr7VmZ+AJkqvstTY1S7UelqthvzZJGoXw0ks0l3vqKHN9b2jIT6Ea1k5oo/Nfc2xPhXeDY5SxDs9Bu8gl7Jc6NrRLCEk7rHuM5+NZs5nMjI5m7wmzUUYwkMZjl0Ln7aArMpubxkeYe4Yb1hhIqh6E8fRzr4XfZNqmh9oP0Aa7edoUs6PFGvO31yDY2gbqv7Gbv0yMQRqz39sNrUWb0LcqRlC4WmBBxwZyrd4Dv9RfYHmZ58lyVo9fHpksy88hZzH5AFsLkuIza0b86n1Q+rhrFMPQr/K7rtDxHGmk8V0YD57+Er0cN3AXvVXoMnOltiO4FJ7qvqYes+5UsPIOYpw+2jWOBFkr8AP17l4v/D26RKXY=",
    ),
    "lovely_girl": (
        "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",
    ),
    "cool_lady": (
        "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",
    ),
    "male_announcer": (
        "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",
    ),
    "uncle_bass": (
        "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",
    ),
}

#: 默认参考音色 —— 内嵌表的**第一条**。
#:
#: 不能写死某个名字:参考音色表会被整体替换(增删参考音色),写死的话
#: 删掉那个音色后,全链路都拿着一个不存在的 ref_speaker 去合成 ——
#: 表现是「音色没生效」,而不是「音色列表里少了一项」。
DEFAULT_REF_SPEAKER: str = next(iter(_BUNDLED_SPEAKER_EMBEDDINGS_B64), "")


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
# 内置音色(config speaker)可用性探测
# --------------------------------------------------------------------------- #
_CONFIG_SPEAKER_CACHE: dict = {}


def available_config_speakers(model_dir) -> dict:
    """本模型**真正可用**的内置音色 → ``{名字: input_embed 行号}``。

    ``SPK_ID`` 的 9 个内置音色只在 **CustomVoice** 权重里训练过;部署
    **Base**(声音克隆)权重时那些行是没训练的占位噪声(实测 norm
    0.022~0.023,而训练过的行是 0.056~1.64,参考音色是 17)—— 塞进
    talker 等于没给音色条件,模型每次生成都自由发挥。

    所以按行内容**实测有效性**:norm 要高出「表内噪声地板」一个数量级。
    不是按名字白名单 —— 换成 CustomVoice 转换的模型,这里自动放行。
    任何异常退化成 ``{}``:宁可少给选项,不能给「选了却不生效」的假选项。
    """
    path = Path(model_dir) / "talker_input_embed.fp16.bin"
    try:
        st = path.stat()
    except OSError:
        return {}
    key = (str(path), st.st_mtime_ns, st.st_size)
    hit = _CONFIG_SPEAKER_CACHE.get(key)
    if hit is not None:
        return dict(hit)

    try:
        emb = np.fromfile(path, dtype=np.float16).reshape(
            TALKER_VOCAB_SIZE, QWEN3_TTS_EMBED_DIM)
        norms = np.linalg.norm(emb.astype(np.float32), axis=1)
        # 噪声地板 = 表内最低 5% 的典型量级(比 min 稳,万一有几行
        # 恰好是 0)。实测 Base 模型:地板 0.022,门槛 0.176。
        floor = float(np.percentile(norms, 5))
        threshold = 8.0 * floor
        out = {
            name: int(row)
            for name, row in SPK_ID.items()
            if 0 <= int(row) < TALKER_VOCAB_SIZE
            and float(norms[int(row)]) >= threshold
        }
    except Exception as e:  # noqa: BLE001
        LOG.warning("available_config_speakers failed for %s: %s", path, e)
        return {}

    _CONFIG_SPEAKER_CACHE[key] = dict(out)
    return dict(out)


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
        # 音色提取器（ECAPA-TDNN speaker encoder）—— 独立于上面四个
        # TTS 模型，是「用户上传音频 → 现场提取音色」用的。
        self.speaker_encoder = None
        # 用户音色（现场提取的 x-vector）的名字集合。音色本体存在
        # ``speaker_embeddings`` 里（和内置音色同一张表）。
        self._user_speaker_names: set = set()
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

    # ------------------------------------------------------------------ #
    # Speaker encoder (音色提取)
    # ------------------------------------------------------------------ #
    def init_speaker_encoder(self):
        """初始化音色提取器（ECAPA-TDNN speaker encoder）。

        用途：把用户上传的参考音频现场算成 2048 维 x-vector，塞进
        :attr:`speaker_embeddings` 后即可用该音色做 VoiceClone。

        **容错**：模型文件缺失 / NPU 初始化失败时只打印一条警告并返回
        ``None`` —— 音色克隆是可选功能，不能拖垮 TTS 主链路。幂等。
        """
        if self.speaker_encoder is not None:
            return self.speaker_encoder
        mode = (os.environ.get(SPEAKER_ENCODER_MODE_ENV)
                or SPEAKER_ENCODER_MODE_DEFAULT).strip().lower()
        if mode == "off":
            print("[speaker_encoder] 已禁用 ({})".format(SPEAKER_ENCODER_MODE_ENV))
            return None
        try:
            from .speaker_encode import SpeakerEncoder

            try:
                core_mask = int(os.environ.get(
                    SPEAKER_ENCODER_CORE_MASK_ENV) or SPEAKER_ENCODER_CORE_MASK)
            except (TypeError, ValueError):
                core_mask = SPEAKER_ENCODER_CORE_MASK
            enc = SpeakerEncoder(
                model_dir=str(self.model_dir),
                target=self.target,
                core_mask=core_mask,
                device_id=self.device_id,
                verbose=self.verbose,
                resident=(mode == "resident"),
            )
        except Exception as exc:      # noqa: BLE001
            print(
                f"[speaker_encoder] 初始化失败，音色克隆功能不可用: {exc}"
            )
            return None
        self.speaker_encoder = enc
        print(
            f"[speaker_encoder] ready: model_dir={self.model_dir} "
            f"mode={mode} frames={enc.frames} dim={enc.mel_basis.shape[0]}"
        )
        return enc

    # ------------------------------------------------------------------ #
    # User speakers (用户音色)
    # ------------------------------------------------------------------ #
    @staticmethod
    def normalize_user_speaker_name(name) -> str:
        """把用户给的音色名归一化成合法的 key（小写 + 字符集校验）。

        允许中文（名字是给人看的），但只放行**字母 / 数字 / ``_`` / ``-``**
        —— 名字同时会当文件名用，这样就杜绝了 ``..``、路径分隔符、隐藏
        文件、控制字符这类输入。
        """
        key = _normalize_user_speaker_key(name)
        if not key:
            raise ValueError("音色名不能为空")
        if len(key) > _USER_SPEAKER_NAME_MAX:
            raise ValueError(
                f"音色名过长（最多 {_USER_SPEAKER_NAME_MAX} 个字符）"
            )
        if not _USER_SPEAKER_NAME_RE.match(key):
            raise ValueError(
                "音色名只能包含字母、数字、下划线、连字符（支持中文），"
                "且必须以字母或数字开头"
            )
        return key

    def set_user_speaker(self, name, embedding) -> str:
        """注册一个用户音色（现场提取的 x-vector）。

        写入 :attr:`speaker_embeddings` —— 那是 talker 查音色时用的同一张
        表，所以注册完立刻就能用这个音色合成。返回归一化后的音色名。
        """
        key = self.normalize_user_speaker_name(name)
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if emb.shape[0] != USER_SPEAKER_EMBED_DIM:
            raise ValueError(
                f"音色向量维度应为 {USER_SPEAKER_EMBED_DIM}，"
                f"实际 {emb.shape[0]}"
            )
        if not bool(np.all(np.isfinite(emb))):
            raise ValueError("音色向量含 NaN/Inf")
        if key in _get_bundled_speaker_embeddings():
            raise ValueError(
                f"音色名 {key!r} 与内置音色冲突，换个名字"
            )
        if self.speaker_embeddings is None:
            raise RuntimeError("speaker_embeddings 尚未加载")
        self.speaker_embeddings[key] = emb
        self._user_speaker_names.add(key)
        print(f"[speaker] registered user speaker '{key}' (dim={emb.shape[0]})")
        return key

    def remove_user_speaker(self, name) -> bool:
        """注销一个用户音色。返回是否确实移除了。"""
        key = _normalize_user_speaker_key(name)
        if key not in self._user_speaker_names:
            return False
        if self.speaker_embeddings is not None:
            self.speaker_embeddings.pop(key, None)
        self._user_speaker_names.discard(key)
        print(f"[speaker] removed user speaker '{key}'")
        return True

    def list_user_speakers(self) -> List[str]:
        """当前进程里已注册的用户音色名（已排序）。"""
        return sorted(self._user_speaker_names)

    def load_user_speakers(self, directory=None) -> List[str]:
        """从目录把持久化的用户音色读回来。

        目录来自 ``directory`` 参数，没给就读环境变量
        :data:`TTS_USER_VOICES_DIR_ENV`。目录不存在 / 为空 → 什么都不做。
        损坏的文件跳过并打印警告，**不抛异常** —— 一个坏文件不该让整个
        TTS 起不来。
        """
        if directory is None:
            directory = os.environ.get(TTS_USER_VOICES_DIR_ENV) or ""
        directory = str(directory or "").strip()
        if not directory:
            return []
        root = Path(directory)
        if not root.is_dir():
            return []
        loaded: List[str] = []
        for npy in sorted(root.glob("*.npy")):
            try:
                emb = np.load(npy).astype(np.float32).reshape(-1)
                key = self.set_user_speaker(npy.stem, emb)
                loaded.append(key)
            except Exception as exc:      # noqa: BLE001
                print(f"[speaker] 跳过损坏的用户音色 {npy}: {exc}")
        if loaded:
            print(
                f"[speaker] loaded {len(loaded)} user speaker(s) "
                f"from {root}: {', '.join(loaded)}"
            )
        return loaded

    def encode_speaker_file(self, path):
        """参考音频文件 → 2048 维音色向量（float32）。

        音色提取器没起来（模型文件缺失）时抛 ``RuntimeError``。

        Note: this runs an NPU inference — do NOT call it from the
        worker's DDS pump thread; the SDK's own ``run_with_progress``
        has the same constraint.
        """
        enc = self.speaker_encoder or self.init_speaker_encoder()
        if enc is None:
            raise RuntimeError("speaker encoder 未初始化，无法提取音色")
        return enc.encode_file(path)

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
        # 音色提取器一起初始化 —— 默认 on_demand 模式这里只加载 mel
        # 滤波器组（几百 KB，不占 NPU），真正提取时才开 NPU 上下文；
        # resident 模式则在这里就把上下文开好（init_runtime 约 0.45 s）。
        # 失败不阻断（见上面的容错）。
        self.init_speaker_encoder()
        # 把持久化的用户音色读回来（app 写的 <name>.npy）—— worker 重启
        # 后用户音色自动恢复，不用 app 重新推一遍。
        self.load_user_speakers()
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

        cb = build_llm_callback(
            result_cb=self._cp_result_callback,
            embed_callback=self._cp_embed_callback,
        )

        args = [{
            "max_context_len": CODEC_MAX_CONTEXT_LEN,
            "vocab_size": CODEC_VOCAB_SIZE,
            "special_eos_id": CODEC_EOS_TOKEN_ID,
            "max_new_tokens": CODEC_MAX_CONTEXT_LEN,
            "keep_history": 0,
            # logits tensor is named ``"logits"`` not ``"output"``;
            # see talker.cc:1192.
            "logits_name": "logits",
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

        cb = build_llm_callback(
            result_cb=self._talker_result_callback,
            embed_callback=self._talker_embed_callback,
            output_cb=self._talker_output_callback,
            output_tensors=output_tensors,
        )

        args = [{
            "max_context_len": QWEN3_TTS_MAX_CONTEXT_LEN,
            "vocab_size": TALKER_VOCAB_SIZE,
            "special_eos_id": CODEC_EOS_TOKEN_ID,
            "max_new_tokens": QWEN3_TTS_MAX_CONTEXT_LEN,
            "keep_history": 0,
            # logits tensor is named ``"logits"`` not ``"output"``;
            # see talker.cc:1095.
            "logits_name": "logits",
            **self.talker_sampling,
        }]

        ret = self.talker.init_llm_session(llm_args=args, llm_callback=cb)
        assert ret == 0, f"init talker llm session failed: {ret}"

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
    # The shim's trampolines hand these callables Python objects:
    # * result_cb(self, token_ids: List[int], state: int) -> int
    # * embed_callback(self, token_ids: List[int], embed_ptr: int,
    #                   length: int) -> int
    # * output_cb(self, outputs: List[_TensorView], state: int) -> int
    # ``self`` here is the bound backend instance (the shim pins it as
    # the C-side userdata).
    def _talker_result_callback(self, _self, tokens, state):
        state_val = int(state)
        with _self.lock:
            _self.talker_last_state = state_val
            if state_val == _LLM_RUN_NORMAL:
                _self.talker_steps += 1
            step = _self.talker_steps
            if state_val in (
                _LLM_RUN_FINISH,
                _LLM_RUN_STOP,
                _LLM_RUN_MAX_NEW_TOKEN_REACHED,
                _LLM_RUN_ERROR,
            ):
                if state_val == _LLM_RUN_ERROR:
                    _self.talker_failed = True
                if _self._talker_done_event is not None:
                    _self._talker_done_event.set()
        if state_val == _LLM_RUN_NORMAL:
            pass  # tokens ignored by the talker; we only track step count
        elif state_val == _LLM_RUN_FINISH:
            print(f"[talker] FINISH, total steps={step}")
        elif state_val == _LLM_RUN_STOP:
            print(f"[talker] STOP")
        elif state_val == _LLM_RUN_MAX_NEW_TOKEN_REACHED:
            print(f"[talker] MAX_NEW_TOKEN_REACHED, total steps={step}")
        elif state_val == _LLM_RUN_ERROR:
            print(f"[talker] ERROR")
        return 0

    def _talker_output_callback(self, _self, outputs, state):
        """Copy the talker's ``past_hidden`` (output index 1) back to host."""
        for tv in outputs:
            if tv.index != 1 or tv.data is None:
                continue
            with _self.lock:
                _self.past_hidden = tv.data.reshape(-1).copy()
        return 0

    def _talker_embed_callback(self, _self, tokens, embed_ptr, length):
        """Build the next talker input embedding for ``num_tokens`` positions."""
        dim = QWEN3_TTS_EMBED_DIM

        num_tokens = len(tokens)
        expected_len = num_tokens * dim * np.dtype(np.float16).itemsize
        if length != expected_len:
            print(f"[talker embed] invalid buffer: got {length}, want {expected_len}")
            return -1

        token_id = int(tokens[0])
        if token_id < 0 or token_id >= TALKER_VOCAB_SIZE:
            token_id = 0
        last_id_hidden = _self.input_embed[token_id].astype(np.float32).copy()

        with _self.lock:
            past_hidden = _self.past_hidden
        if past_hidden is None:
            print("[talker embed] past_hidden not yet available")
            return -1

        code_input = np.concatenate(
            [past_hidden.astype(np.float32), last_id_hidden]
        ).astype(np.float16)
        code_input = code_input.reshape(1, 2, dim)
        code_ids = _self.run_code_predictor(code_input)

        frame = [token_id] + [int(t) for t in code_ids]
        with _self.lock:
            _self.collected_codes.append(frame)
            step = _self.talker_steps
        if _self.frame_queue is not None:
            _self.frame_queue.put(frame)

        next_embed = last_id_hidden.copy()
        for i, tid in enumerate(code_ids):
            tid = int(tid)
            if tid < 0 or tid >= CODEC_VOCAB_SIZE:
                tid = 0
            next_embed += _self.codec_embed[i, tid].astype(np.float32)

        with _self.lock:
            trailing = _self.trailing_text_hiddens
            pad = _self.tts_pad_embed
        idx = step - 1
        if trailing is not None and 0 <= idx < trailing.shape[0]:
            next_embed += trailing[idx].astype(np.float32)
        else:
            next_embed += pad.astype(np.float32)

        dst = _fp16_view(embed_ptr, num_tokens * dim)
        dst[:] = next_embed.astype(np.float16)
        return 0

    def _cp_result_callback(self, _self, tokens, state):
        if int(state) != _LLM_RUN_NORMAL:
            return 0
        with _self.lock:
            _self.code_predictor_out_ids.extend(tokens)
            _self.code_predictor_step += 1
        return 0

    def _cp_embed_callback(self, _self, tokens, embed_ptr, length):
        """Fill the code_predictor embed buffer from the codec embedding table."""
        dim = QWEN3_TTS_EMBED_DIM

        num_tokens = len(tokens)
        expected_len = num_tokens * dim * np.dtype(np.float16).itemsize
        if length != expected_len:
            return -1

        with _self.lock:
            step = _self.code_predictor_step

        row = step - 1 if step > 0 else 0
        if row >= _self.codec_embed.shape[0]:
            row = _self.codec_embed.shape[0] - 1

        dst = _fp16_view(embed_ptr, num_tokens * dim)
        for i in range(num_tokens):
            tid = int(tokens[i])
            if tid < 0 or tid >= CODEC_VOCAB_SIZE:
                tid = 0
            dst[i * dim:(i + 1) * dim] = _self.codec_embed[row, tid]
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

        # 音色提取器不是 RKNN3Lite，是包装对象 —— 单独释放。
        enc = getattr(self, "speaker_encoder", None)
        if enc is not None:
            try:
                enc.release()
            except Exception as exc:
                print(f"[release] speaker_encoder: {exc}")
            self.speaker_encoder = None


# --------------------------------------------------------------------------- #
# v2 SDK backend
# --------------------------------------------------------------------------- #
class TTSBackend:
    """TTS backend — text-in, audio-out.

    ``run_text(prompt)`` returns ``(prompt_str, [AudioAttachment])`` where
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
    def run_text(self, prompt: str, **overrides: Any) -> Tuple[str, List["Attachment"]]:
        """Public entry point invoked by ``LLMSession._dispatch``.

        Drives the TTS pipeline via the demo's ``run_with_progress``
        and packages the resulting WAV bytes as an
        :class:`AudioAttachment` (24 kHz mono 32-bit IEEE float WAV).
        """
        # Forward-declared in the signature for type tools — imported
        # locally below where needed.
        from ..attachment import Attachment  # noqa: F401
        ref_speaker = overrides.get("ref_speaker", DEFAULT_REF_SPEAKER)
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
            AudioAttachment(wav_bytes, mime_type="audio/wav"),
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