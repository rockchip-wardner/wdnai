"""ctypes shim for librknn3_api.so — the wdnai SDK's only RKNN3 entry point.

This module replaces the ``rknn3lite`` Python wheel that wdnai used to
depend on. It exposes a minimal Python surface that mirrors the parts
of ``rknn3lite.api`` actually exercised by the wdnai backends:

* :class:`RKNN3Lite` — the main entry point.
* :class:`RKNN3Image` / :class:`RKNN3Audio` — multimodal input structs
  constructed by backends and passed to :meth:`RKNN3Lite.session_run`.
* :class:`RKNN3AuxTensorWrapper` — Qwen3-VL "deepstack" aux tensors.
* :class:`Float16` — ctype alias for ``uint16`` storage, used in
  ``ctypes.POINTER(Float16)`` typing for image/audio embeds.
* :class:`RKNN3TensorType` — IntEnum used by the embedding backend to
  distinguish UINT8 vs FP16 vision inputs.
* :func:`build_llm_callback` — builder for LLM callback bundles that
  takes plain Python callables; the resulting bundle is opaque and
  consumed by :meth:`RKNN3Lite.init_runtime` / :meth:`init_llm_session`.

Everything else — the C function prototypes, the
``rknn3_llm_input``/``rknn3_llm_multimodal_tensor`` packing, the
``RKLLMCallback`` struct, the 5 ``CFUNCTYPE`` callback types, all
internal ctypes Structures (``RKNN3TensorAttr``, ``RKNN3CoreMemSize``,
``RKNN3QueryCmd`` …) — is implemented inside this module and never
leaves it. The ``from rknn3lite.api import RKLLMCallback`` / ctypes
plumbing that the old backends did is gone: every backend now only
constructs a Python object, hands it to :class:`RKNN3Lite`, and lets
this shim do the C marshalling.

The implementation follows the C++ reference pattern from
``rknn3_model_zoo/examples/Qwen3_5_VL/cpp/main.cc``,
``qwen3_5_vl.cc`` and ``qwen3_vl.cc``: one librknn3 context per
RKNN model, ``rknn3_session_init`` for LLM mode, ``rknn3_create_mem``
+ ``rknn3_run`` for plain inference. Memory ownership mirrors the
C++ flow — ``rknn3_create_mem`` returns ``rknn3_tensor_mem*`` that
must be released with ``rknn3_destroy_mem``; the shim tracks these
and frees them in :meth:`RKNN3Lite.release`.
"""
from __future__ import annotations

import ctypes
import os
import platform
import threading
import time
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    Union,
    c_bool,
    c_char,
    c_char_p,
    c_double,
    c_float,
    c_int,
    c_int32,
    c_int64,
    c_uint8,
    c_uint16,
    c_uint32,
    c_uint64,
    c_void_p,
    cast as ctypes_cast,
    pointer,
)
from enum import IntEnum
from typing import Any, Callable, List, Optional, Sequence, Tuple, TYPE_CHECKING, cast as typing_cast

if TYPE_CHECKING:
    pass

import numpy as np


# =============================================================================
# 1. Library load
# =============================================================================
_LIB: Optional[ctypes.CDLL] = None
_LIB_LOCK = threading.Lock()


def _candidate_paths() -> List[str]:
    env = os.environ.get("WDNAI_LIBRKNN3_API")
    if env:
        return [env]
    paths = [
        "/usr/lib/librknn3_api.so",
        "/usr/lib/librknn3_api_rkcp.so",
        "/usr/lib/librknn3_api_native.so",
    ]
    if platform.architecture()[0] == "32bit":
        paths += ["/usr/lib32/librknn3_api.so", "/usr/lib32/librknn3_api_rkcp.so"]
    return paths


def load_library() -> ctypes.CDLL:
    """Load librknn3_api.so once. Subsequent calls return the cached handle.

    Importing this module does *not* load the SO (which would init the
    NPU driver at import time and race with ``wdnav`` in the same
    process). The first :class:`RKNN3Lite` construction triggers the
    actual ``CDLL`` open. :func:`load_library` is idempotent.
    """
    global _LIB
    if _LIB is not None:
        return _LIB
    with _LIB_LOCK:
        if _LIB is not None:
            return _LIB
        last_err: Optional[Exception] = None
        for p in _candidate_paths():
            try:
                _LIB = ctypes.CDLL(p)
                break
            except OSError as e:
                last_err = e
        if _LIB is None:
            raise RuntimeError(
                "librknn3_api.so not found in standard paths; set "
                "WDNAI_LIBRKNN3_API to override. Last error: "
                f"{last_err}"
            )
        _bind_prototypes(_LIB)
        return _LIB


# =============================================================================
# 2. ABI constants
# =============================================================================
_MAX_DIMS = 16
_MAX_STRIDE_DIMS = _MAX_DIMS + 1
_MAX_NAME_LEN = 256
_MAX_DEVS = 64
_MAX_DEV_LEN = 64
_MAX_NPU_NODE_NUM = 128
_MAX_SPECIAL_BOS_ID_NUM = 64
_MAX_SPECIAL_EOS_ID_NUM = 64


# =============================================================================
# 3. Public surface
# =============================================================================
class RKNN3TensorType(IntEnum):
    RKNN3_TENSOR_FLOAT32 = 0
    RKNN3_TENSOR_FLOAT16 = 1
    RKNN3_TENSOR_INT8 = 2
    RKNN3_TENSOR_UINT8 = 3
    RKNN3_TENSOR_INT16 = 4
    RKNN3_TENSOR_UINT16 = 5
    RKNN3_TENSOR_INT32 = 6
    RKNN3_TENSOR_UINT32 = 7
    RKNN3_TENSOR_INT64 = 8
    RKNN3_TENSOR_UINT64 = 9
    RKNN3_TENSOR_BOOL = 10
    RKNN3_TENSOR_INT4 = 11
    RKNN3_TENSOR_FLOAT8E4M3FN = 12
    RKNN3_TENSOR_BFLOAT16 = 13
    RKNN3_TENSOR_FLOAT8E8M0 = 14
    RKNN3_TENSOR_FLOAT4E2M1 = 15


class Float16(Structure):
    """Half-precision IEEE 754 storage type (16-bit)."""

    _fields_ = [("v", c_uint16)]


class RKNN3Image(Structure):
    """Multimodal image descriptor. wdnai sets the start/end/content
    markers (UTF-8 bytes) to the model's ``<|vision_*|>`` /
    ``<|image_pad|>`` tokens, and ``image_embed`` to a contiguous
    ``(n_image, n_image_tokens, embed_dim)`` fp16 array via
    ``np.ctypes.data_as(POINTER(Float16))``."""

    _fields_ = [
        ("image_embed", POINTER(Float16)),
        ("n_image_tokens", c_uint64),
        ("n_image", c_uint64),
        ("image_start", c_char_p),
        ("image_end", c_char_p),
        ("image_content", c_char_p),
        ("image_width", c_uint64),
        ("image_height", c_uint64),
    ]


class RKNN3Audio(Structure):
    """Multimodal audio descriptor. wdnai sets the audio markers and
    ``audio_embed`` to a contiguous ``(n_audio, n_audio_tokens,
    embed_dim)`` fp16 array."""

    _fields_ = [
        ("audio_embed", POINTER(Float16)),
        ("n_audio_tokens", c_uint64),
        ("n_audio", c_uint64),
        ("audio_start", c_char_p),
        ("audio_end", c_char_p),
        ("audio_content", c_char_p),
    ]


class RKNN3AuxTensorWrapper:
    """Qwen3-VL "deepstack" aux tensor. wdnai constructs one per
    aux output (index 1..3) and appends to the ``session_run``
    ``inputs=`` list."""

    __slots__ = ("aux_data", "index", "align_size")

    def __init__(self, aux_data: Any = None, index: int = 0, align_size: int = 0) -> None:
        self.aux_data = aux_data
        self.index = index
        self.align_size = align_size


# =============================================================================
# 4. Internal ABI types (mirroring rknn3_api.h, never re-exported)
# =============================================================================
class _QuantInfo(Structure):
    _fields_ = [("scale", c_float), ("zero_point", c_int32)]


class _InputOutputNum(Structure):
    _fields_ = [("n_input", c_uint32), ("n_output", c_uint32)]


class _TensorAttr(Structure):
    _fields_ = [
        ("index", c_uint32),
        ("name", c_char * _MAX_NAME_LEN),
        ("n_dims", c_uint32),
        ("shape", c_uint32 * _MAX_DIMS),
        ("aligned_size", c_uint64),
        ("n_stride", c_uint32),
        ("stride", c_uint64 * _MAX_STRIDE_DIMS),
        ("n_elems", c_uint32),
        ("dtype", c_int),
        ("layout", c_int),
        ("qnt_type", c_int),
        ("qnt_info", _QuantInfo),
        ("n_orig_dims", c_uint32),
        ("orig_shape", c_uint32 * _MAX_DIMS),
        ("orig_layout", c_int),
        ("core_id", c_int32),
    ]


class _SDKVersion(Structure):
    _fields_ = [
        ("api_version", c_char * 256),
        ("drv_version", c_char * 256),
    ]


class _CoreMemSize(Structure):
    _fields_ = [
        ("core_id", c_int32),
        ("weight_size", c_uint64),
        ("internal_size", c_uint64),
        ("reserved", c_uint8 * 32),
    ]


class _CustomString(Structure):
    _fields_ = [("string", c_char * 1024)]


class _TensorMemory(Structure):
    _fields_ = [
        ("virt_addr", c_void_p),
        ("phys_addr", c_uint64),
        ("fd", c_int32),
        ("buffer_id", c_uint64),
        ("offset", c_uint64),
        ("size", c_uint64),
        ("flags", c_uint64),
        ("core_id", c_int32),
        ("priv_data", c_void_p),
        ("mem_type", c_int),
    ]


class _RknnTensor(Structure):
    _fields_ = [
        ("mem", POINTER(_TensorMemory)),
        ("attr", POINTER(_TensorAttr)),
    ]


class _RknnConfig(Structure):
    _fields_ = [
        ("priority", c_int32),
        ("run_timeout", c_uint32),
        ("run_core_mask", c_uint32),
        ("user_mem_weight", c_uint8),
        ("user_mem_internal", c_uint8),
        ("user_sram", c_uint8),
        ("reserved", c_uint8 * 128),
    ]


class _InitExtend(Structure):
    _fields_ = [("device_id", c_char_p), ("reserved", c_uint8 * 128)]


class _NodeMemInfo(Structure):
    _fields_ = [("total", c_uint64), ("free", c_uint64)]


class _DevMemInfo(Structure):
    _fields_ = [
        ("node_num", c_uint32),
        ("sys_total", c_uint64),
        ("sys_free", c_uint64),
        ("node_mem_info", _NodeMemInfo * _MAX_NPU_NODE_NUM),
    ]


class _RknnDevice(Structure):
    _fields_ = [
        ("id", c_char * _MAX_DEV_LEN),
        ("type", c_char * _MAX_DEV_LEN),
        ("mem_info", _DevMemInfo),
    ]


class _RknnDevices(Structure):
    _fields_ = [
        ("n_devices", c_uint32),
        ("devices", _RknnDevice * _MAX_DEVS),
    ]


class _SamplingParams(Structure):
    _fields_ = [
        ("top_k", c_int32),
        ("top_p", c_float),
        ("temperature", c_float),
        ("repeat_penalty", c_float),
        ("frequency_penalty", c_float),
        ("presence_penalty", c_float),
    ]


class _VocabInfo(Structure):
    _fields_ = [
        ("vocab_size", c_int),
        ("special_bos_id", c_int * _MAX_SPECIAL_BOS_ID_NUM),
        ("special_eos_id", c_int * _MAX_SPECIAL_EOS_ID_NUM),
        ("n_special_bos_id", c_int),
        ("n_special_eos_id", c_int),
        ("linefeed_id", c_int),
        ("skip_special_token", c_bool),
        ("ignore_eos_token", c_bool),
        ("reserved", c_uint8 * 64),
    ]


class _LLMExtendParam(Structure):
    _fields_ = [("reserved", c_uint8 * 128)]


class _AttentionKvcacheLens(Structure):
    """Mirrors rknn3_api.h's ``rknn3_attention_kvcache_lens``.

    Used inside :class:`_LLMConfig` for the per-attention-type kvcache
    buffer length list (one entry per attention type, e.g. full /
    sliding / linear).
    """
    _fields_ = [
        ("attention_type", c_int),
        ("n_kvcache_buffer_lens", c_uint32),
        ("kvcache_buffer_lens", c_int32 * 16),  # RKNN3_MAX_KVCACHE_LEN_GROUPS = 16
    ]


class _LLMConfig(Structure):
    """Mirrors rknn3_api.h's ``rknn3_llm_config`` (queried via
    ``RKNN3_QUERY_LLM_CONFIG`` to read the model's intrinsic
    ``max_ctx_len``, ``vocab_size``, etc.)."""
    _fields_ = [
        ("chat_template", c_char_p),
        ("vocab_size", c_uint32),
        ("embedding_dim", c_uint32),
        ("max_ctx_len", c_uint32),
        ("max_position_embeddings", c_uint32),
        ("kvcache_store_method", c_int),
        ("kvcache_dtype", c_int),
        ("kvcache_group_size", c_uint32),
        ("kvcache_residual_depth", c_uint32),
        ("model_type", c_char_p),
        ("task_type", c_int),
        ("rope_cache_host_storage", c_uint8),
        ("n_attention_kvcache_lens", c_uint32),
        ("attention_kvcache_lens", _AttentionKvcacheLens * 3),
        ("reserved", c_uint8 * 127),
    ]


class _LLMParam(Structure):
    _fields_ = [
        ("logits_name", c_char_p),
        ("max_context_len", c_int32),
        ("sampling_param", _SamplingParams),
        ("vocab_info", _VocabInfo),
        ("extend_param", _LLMExtendParam),
    ]


class _LLMTensor(Structure):
    _fields_ = [
        ("name", c_char_p),
        ("prompt", c_char_p),
        ("embed", c_void_p),
        ("tokens", POINTER(c_int32)),
        ("n_tokens", c_uint64),
        ("enable_thinking", c_bool),
    ]


class _Video(Structure):
    _fields_ = [
        ("video_embed", c_void_p),
        ("n_frame_tokens", c_uint64),
        ("n_frame_per_video", c_uint64),
        ("n_video", c_uint64),
        ("video_start", c_char_p),
        ("video_end", c_char_p),
        ("video_content", c_char_p),
        ("frame_width", c_uint64),
        ("frame_height", c_uint64),
    ]


class _LLMMultiModelTensor(Structure):
    _fields_ = [
        ("name", c_char_p),
        ("prompt", c_char_p),
        ("tokens", POINTER(c_int32)),
        ("n_tokens", c_uint64),
        ("enable_thinking", c_bool),
        ("image", RKNN3Image),
        ("audio", RKNN3Audio),
        ("video", _Video),
    ]


class _LLMInputUnion(Union):
    _fields_ = [
        ("llm_input", _LLMTensor),
        ("multimodal_input", _LLMMultiModelTensor),
        ("aux_input", _RknnTensor),
    ]


class _LLMInput(Structure):
    _fields_ = [
        ("role", c_char_p),
        ("input_type", c_int),
        ("_u", _LLMInputUnion),
    ]


class _LLMInferParam(Structure):
    _fields_ = [
        ("keep_history", c_int),
        ("max_new_tokens", c_int32),
        ("prefill_only", c_bool),
        ("disable_sampling", c_bool),
        ("reserved", c_uint8 * 128),
    ]


class _LLMResult(Structure):
    _fields_ = [
        ("token_ids", POINTER(c_int32)),
        ("num_tokens", c_int),
    ]


class _LLMRunState(Structure):
    _fields_ = [
        ("n_total_tokens", c_uint64),
        ("n_reuse_tokens", c_uint64),
        ("n_max_tokens", c_uint64),
        ("n_prefill_tokens", c_uint64),
        ("n_decode_tokens", c_uint64),
        ("n_input_tokens", c_uint64),
        ("n_output_tokens", c_uint64),
        ("input_tokens", POINTER(c_int32)),
        ("output_tokens", POINTER(c_int32)),
        ("kvcache_policy", c_int),
        ("n_loras_enabled", c_int32),
        ("loras_enabled", c_void_p),
    ]


class _QueryCmd(IntEnum):
    IN_OUT_NUM = 0
    INPUT_ATTR = 1
    OUTPUT_ATTR = 2
    SDK_VERSION = 5
    CORE_MEM_SIZE = 6
    ORIGIN_INPUT_ATTR = 8
    ORIGIN_OUTPUT_ATTR = 9
    DEVICE_MEM_INFO = 12
    CORE_NUMBER = 13
    DYNAMIC_SHAPE_CONFIG = 15
    DYNAMIC_SHAPE_INFO = 16
    LLM_CONFIG = 17


# Public alias — example scripts and downstream callers that came from
# ``rknn3lite.api`` import this name (``from wdnai._librknn import
# RKNN3QueryCmd``). The shim keeps the leading underscore on the
# canonical definition so internal usage reads as "this is a shim
# implementation detail"; the alias is the user-facing spelling.
RKNN3QueryCmd = _QueryCmd


class _LLMInputType(IntEnum):
    PROMPT = 0
    TOKEN = 1
    EMBED = 2
    MULTIMODAL = 3
    AUX = 4


class _LLMCallState(IntEnum):
    NORMAL = 0
    WAITING = 1
    FINISH = 2
    STOP = 3
    MAX_NEW_TOKEN_REACHED = 4
    ERROR = 5
    PAUSE = 6
    RESUME = 7


class _LLMOutputCallbackState(IntEnum):
    PREFILL_PROCESSING = 0
    PREFILL_FINISHED = 1
    DECODE_PROCESSING = 2
    DECODE_FINISHED = 3


class _KVCacheClearPolicy(IntEnum):
    CLEAR_ALL = 0
    KEEP_SYSTEM_PROMPT = 1


class _MemAllocFlags(IntEnum):
    CACHEABLE = 1 << 0
    NON_CACHEABLE = 1 << 1


class _MemSyncMode(IntEnum):
    TO_DEVICE = 1
    FROM_DEVICE = 2


# =============================================================================
# 5. C function prototypes (bound against the loaded SO)
# =============================================================================
def _bind_prototypes(L: ctypes.CDLL) -> None:
    L.rknn3_init.argtypes = [POINTER(c_uint64), c_void_p]
    L.rknn3_init.restype = c_int

    L.rknn3_destroy.argtypes = [c_uint64]
    L.rknn3_destroy.restype = c_int

    L.rknn3_load_model_from_path.argtypes = [c_uint64, c_char_p, c_char_p]
    L.rknn3_load_model_from_path.restype = c_int

    L.rknn3_set_decrypt_key_from_path.argtypes = [c_uint64, c_char_p]
    L.rknn3_set_decrypt_key_from_path.restype = c_int

    L.rknn3_query.argtypes = [c_uint64, c_int, c_void_p, c_uint64]
    L.rknn3_query.restype = c_int

    L.rknn3_model_init.argtypes = [c_uint64, c_void_p]
    L.rknn3_model_init.restype = c_int

    L.rknn3_set_shape.argtypes = [c_uint64, c_int32]
    L.rknn3_set_shape.restype = c_int

    L.rknn3_create_mem.argtypes = [c_uint64, c_uint64, c_int32, c_int32]
    L.rknn3_create_mem.restype = c_void_p

    L.rknn3_destroy_mem.argtypes = [c_uint64, c_void_p]
    L.rknn3_destroy_mem.restype = c_int

    L.rknn3_set_internal_mem.argtypes = [c_uint64, c_void_p, c_uint32]
    L.rknn3_set_internal_mem.restype = c_int

    L.rknn3_mem_sync.argtypes = [c_uint64, c_void_p, c_int32]
    L.rknn3_mem_sync.restype = c_int

    L.rknn3_run.argtypes = [c_uint64, c_void_p, c_uint32, c_void_p, c_uint32]
    L.rknn3_run.restype = c_int

    L.rknn3_find_devices.argtypes = [c_void_p]
    L.rknn3_find_devices.restype = c_int

    L.rknn3_session_init.argtypes = [c_uint64, c_void_p, c_int]
    L.rknn3_session_init.restype = c_void_p

    L.rknn3_session_destroy.argtypes = [c_void_p]
    L.rknn3_session_destroy.restype = c_int

    L.rknn3_session_set_chat_template.argtypes = [
        c_void_p, c_char_p, c_char_p, c_char_p,
    ]
    L.rknn3_session_set_chat_template.restype = c_int

    L.rknn3_session_set_callback.argtypes = [c_void_p, c_void_p]
    L.rknn3_session_set_callback.restype = c_int

    L.rknn3_session_run.argtypes = [c_void_p, c_void_p, c_uint32, c_void_p]
    L.rknn3_session_run.restype = c_int

    L.rknn3_session_run_async.argtypes = [c_void_p, c_void_p, c_uint32, c_void_p]
    L.rknn3_session_run_async.restype = c_int

    L.rknn3_session_clear_kvcache.argtypes = [c_void_p, c_int32]
    L.rknn3_session_clear_kvcache.restype = c_int

    L.rknn3_session_stop.argtypes = [c_void_p]
    L.rknn3_session_stop.restype = c_int

    L.rknn3_session_pause.argtypes = [c_void_p]
    L.rknn3_session_pause.restype = c_int

    L.rknn3_session_resume.argtypes = [c_void_p]
    L.rknn3_session_resume.restype = c_int

    L.rknn3_session_set_llm_param.argtypes = [c_void_p, c_void_p, c_int]
    L.rknn3_session_set_llm_param.restype = c_int

    L.rknn3_session_query_state.argtypes = [c_void_p, c_void_p]
    L.rknn3_session_query_state.restype = c_int


# =============================================================================
# 6. LLM callback bundle
# =============================================================================
# C function-pointer types for the 5 callbacks the C side can invoke.
# They all match rknn3_api.h exactly; the shim's trampolines marshal
# their ``void* userdata`` to/from a Python object via ctypes.py_object
# and translate the C payload (e.g. RKLLMResult* with token_ids[]) to
# a Python list of ints before calling the user-supplied callable.
_RESULT_CB = CFUNCTYPE(
    c_int, c_void_p, POINTER(_LLMResult), c_int,
)
_TOKENIZER_CB = CFUNCTYPE(
    c_int, c_void_p, c_char_p, c_int32, POINTER(c_int32), c_int32,
)
_EMBED_CB = CFUNCTYPE(
    c_int, c_void_p, POINTER(c_int32), c_uint64, c_void_p, c_uint64,
)
_SAMPLING_CB = CFUNCTYPE(c_int, c_void_p, c_void_p, c_char_p)
_OUTPUT_CB = CFUNCTYPE(c_int, c_void_p, POINTER(_RknnTensor), c_uint32, c_int)
_INPUT_CB = CFUNCTYPE(c_int, c_void_p, POINTER(_RknnTensor), c_uint32, c_void_p)


class _LLMCallback(Structure):
    """C-side callback bundle; field order matches rknn3_api.h:1026."""

    _fields_ = [
        ("result_callback", _RESULT_CB),
        ("result_userdata", c_void_p),
        ("sampling_callback", _SAMPLING_CB),
        ("sampling_userdata", c_void_p),
        ("tokenizer_callback", _TOKENIZER_CB),
        ("tokenizer_userdata", c_void_p),
        ("embed_callback", _EMBED_CB),
        ("embed_userdata", c_void_p),
        ("output_callback", _OUTPUT_CB),
        ("output_userdata", c_void_p),
        ("output_tensors", POINTER(_RknnTensor)),
        ("n_output_tensors", c_uint32),
        ("input_callback", _INPUT_CB),
        ("input_userdata", c_void_p),
        ("input_tensors_index", POINTER(c_int32)),
        ("n_input_tensors", c_uint32),
    ]


class _TensorView:
    """Per-output snapshot exposed to the user in ``output_cb``.

    ``.data`` is a numpy fp16 view of the NPU's output buffer
    (CPU-readable, no copy). ``.index`` / ``.n_elems`` / ``.name``
    mirror :class:`_TensorAttr`. ``.data is None`` for tensors that
    the C side did not populate.
    """

    __slots__ = ("index", "name", "n_elems", "dtype", "data")

    def __init__(self, index: int, name: str, n_elems: int, dtype: int, data):
        self.index = index
        self.name = name
        self.n_elems = n_elems
        self.dtype = dtype
        self.data = data


class _LLMCallbackBundle:
    """Opaque object that :meth:`RKNN3Lite.init_runtime` /
    :meth:`init_llm_session` consume as ``llm_callback=``.

    Holds:
    * the C-side :class:`_LLMCallback` struct, kept alive for the
      lifetime of the parent :class:`RKNN3Lite`.
    * the ``py_object`` anchors that pin the user-supplied callables'
      bound ``self`` instances (so the GC doesn't free them while C
      still has a ``void*`` into them).
    * the pre-allocated ``output_tensors`` array (for the talker
      ``create_output_tensors`` path) plus its backing buffer.
    """

    __slots__ = (
        "struct", "pins", "output_tensors", "n_output_tensors",
        "_trampolines", "_output_buffer_refs", "_output_attr_refs",
    )

    def __init__(self) -> None:
        self.struct = _LLMCallback()
        self.pins: List[Any] = []            # keep py_object anchors alive
        self._trampolines: List[Any] = []    # keep CFUNCTYPE refs alive
        self.output_tensors: Any = None      # ctypes array[ _RknnTensor ]
        self.n_output_tensors: c_uint32 = c_uint32(0)
        self._output_buffer_refs: List[Any] = []
        self._output_attr_refs: List[Any] = []


def _pin_userdata(bundle: _LLMCallbackBundle, obj: Any) -> c_void_p:
    """Pin ``obj`` as a C ``void*`` for the lifetime of ``bundle``."""
    holder = ctypes.py_object(obj)
    bundle.pins.append(holder)
    return ctypes_cast(pointer(holder), c_void_p)


def _result_trampoline(user_cb: Callable[..., int]):
    """Build the C trampoline for the result callback.

    ``user_cb`` is invoked as ``user_cb(userdata, token_ids, state)``
    where ``token_ids`` is a Python ``List[int]`` and ``state`` is the
    integer LLMCallState. The user's bound ``self`` (if any) is
    automatically restored from the void* userdata.
    """
    @_RESULT_CB
    def _cb(userdata, result_ptr, state):
        py_obj = ctypes.cast(userdata, POINTER(ctypes.py_object)).contents.value
        # ``result_ptr`` is NULL in the FINISH / WAITING / ERROR states;
        # only NORMAL / decode carries actual token IDs. Match rknn3lite
        # by passing an empty list when the pointer is None.
        if result_ptr:
            n = result_ptr.contents.num_tokens
            tokens = [int(result_ptr.contents.token_ids[i]) for i in range(n)]
        else:
            tokens = []
        try:
            rc = int(user_cb(py_obj, tokens, int(state)))
        except Exception:  # noqa: BLE001 — never let a Python error into C
            rc = -1
        return rc

    return _cb


def _tokenizer_trampoline_for_hf_tokenizer(tokenizer) -> Tuple[Any, c_void_p]:
    """Build a C tokenizer callback that wraps a HuggingFace tokenizer.

    Mirrors rknn3lite's default behaviour: text → ``tokenizer(text)``
    → ``input_ids[0][:n_tokens_max]`` → int32 array. ``userdata`` is
    unused; the closure pins the tokenizer itself.
    """
    @_TOKENIZER_CB
    def _cb(userdata, text_ptr, text_len, tokens_ptr, n_tokens_max):
        text = ctypes.string_at(text_ptr, text_len).decode("utf-8")
        out = tokenizer(text, return_tensors="np", truncation=True)
        ids = out["input_ids"][0][:n_tokens_max]
        n = len(ids)
        if n <= 0:
            return 0
        for i in range(n):
            tokens_ptr[i] = int(ids[i])
        return n

    holder = ctypes.py_object(tokenizer)
    ud = ctypes_cast(pointer(holder), c_void_p)
    return _cb, ud


def _embed_trampoline_for_table(embeds: np.ndarray) -> Tuple[Any, c_void_p]:
    """Build a C embed callback that looks up fp16 token embeddings."""

    @_EMBED_CB
    def _cb(userdata, tokens_ptr, num_tokens, embed, length):
        dim = embeds.shape[1]
        expected = int(num_tokens) * dim * np.dtype(np.float16).itemsize
        if int(length) != expected:
            return -1
        ids = [int(tokens_ptr[i]) for i in range(int(num_tokens))]
        dst = np.ctypeslib.as_array(
            ctypes_cast(embed, POINTER(c_uint16)),
            shape=(int(num_tokens) * dim,),
        ).view(np.float16)
        dst[:] = embeds[ids].ravel()
        return 0

    holder = ctypes.py_object(embeds)
    ud = ctypes_cast(pointer(holder), c_void_p)
    return _cb, ud


def build_llm_callback(
    *,
    result_cb: Optional[Callable[..., int]] = None,
    tokenizer: Any = None,
    embeds_data: Optional[np.ndarray] = None,
    embed_callback: Optional[Callable[..., int]] = None,
    output_cb: Optional[Callable[..., int]] = None,
    output_tensors: Optional[Any] = None,
    sampling_cb: Optional[Callable[..., int]] = None,
    input_cb: Optional[Callable[..., int]] = None,
) -> _LLMCallbackBundle:
    """Build the opaque LLM callback bundle consumed by
    :meth:`RKNN3Lite.init_runtime` / :meth:`init_llm_session`.

    Parameters mirror the slots in rknn3lite's ``RKLLMCallback`` struct,
    but every callback is a plain Python callable (no ctypes
    wrapping). The shim auto-pins bound-method ``self`` objects via
    ``ctypes.py_object`` and dispatches them to the user callable.

    * ``result_cb(userdata, token_ids: List[int], state: int) -> int`` —
      ``state`` is one of the LLMCallState int values (see
      :data:`_LLMCallState`). ``userdata`` is the bound ``self`` of
      ``result_cb`` if it has one, else the ``result_cb`` itself.
      Return 0 to continue, non-zero to abort.
    * ``tokenizer`` — if given, a default tokenizer callback is built
      that calls ``tokenizer(text, return_tensors="np", truncation=True)``
      (matches rknn3lite's behaviour for HuggingFace tokenizers). Pass
      ``None`` to leave the tokenizer slot empty (the LLM must already
      know how to tokenize — not currently exercised by wdnai).
    * ``embeds_data`` — if given, a default embed callback is built
      that looks up ``embeds_data[token_ids]`` (shape
      ``(vocab_size, dim)``) and writes fp16 to the C buffer. Mutually
      exclusive with ``embed_callback`` (custom per-token embed).
    * ``embed_callback(userdata, tokens: List[int], embed_ptr: int,
      length: int) -> int`` — custom per-token embed callback. The
      user is responsible for writing fp16 data into the buffer
      pointed to by ``embed_ptr`` (use
      ``np.ctypeslib.as_array(ctypes.cast(embed_ptr, POINTER(c_uint16)),
      shape=(length // 2,)).view(np.float16)`` for an editable view).
      ``length`` is the buffer size in bytes. Used by the TTS talker
      to dynamically build the next-step embedding.
    * ``output_cb(userdata, outputs: List[_TensorView], state: int) -> int``
      — fired for each prefill/decode step that touches an output
      tensor set up via :meth:`RKNN3Lite.create_output_tensors`.
    * ``sampling_cb`` / ``input_cb`` — left empty by default; the LLM
      uses its built-in sampler. If supplied, must be a plain Python
      callable (the shim does no marshalling for these two).
    """
    bundle = _LLMCallbackBundle()

    # ---- result ----------------------------------------------------- #
    if result_cb is not None:
        tramp = _result_trampoline(result_cb)
        bundle.struct.result_callback = tramp
        bundle._trampolines.append(tramp)
        # Pin the bound method's ``__self__`` if any; else pin the cb
        # itself (e.g. closure).
        target = getattr(result_cb, "__self__", result_cb)
        bundle.struct.result_userdata = _pin_userdata(bundle, target)

    # ---- tokenizer ------------------------------------------------- #
    if tokenizer is not None:
        tramp, ud = _tokenizer_trampoline_for_hf_tokenizer(tokenizer)
        bundle.struct.tokenizer_callback = tramp
        bundle.struct.tokenizer_userdata = ud
        bundle._trampolines.append(tramp)
        bundle.pins.append(ud)            # keep the py_object anchor alive

    # ---- embed ------------------------------------------------------ #
    if embeds_data is not None and embed_callback is not None:
        raise ValueError("embeds_data and embed_callback are mutually exclusive")
    if embeds_data is not None:
        if not isinstance(embeds_data, np.ndarray) or embeds_data.dtype != np.float16:
            raise TypeError("embeds_data must be a float16 ndarray")
        tramp, ud = _embed_trampoline_for_table(embeds_data)
        bundle.struct.embed_callback = tramp
        bundle.struct.embed_userdata = ud
        bundle._trampolines.append(tramp)
        bundle.pins.append(ud)
    elif embed_callback is not None:
        target = getattr(embed_callback, "__self__", embed_callback)

        @_EMBED_CB
        def _emb_cb(userdata, tokens_ptr, num_tokens, embed_ptr, length):
            py_obj = ctypes.cast(
                userdata, POINTER(ctypes.py_object)
            ).contents.value
            n = int(num_tokens)
            tok_list = [int(tokens_ptr[i]) for i in range(n)]
            try:
                rc = int(embed_callback(py_obj, tok_list, int(embed_ptr), int(length)))
            except Exception:  # noqa: BLE001
                rc = -1
            return rc

        bundle.struct.embed_callback = _emb_cb
        bundle.struct.embed_userdata = _pin_userdata(bundle, target)
        bundle._trampolines.append(_emb_cb)

    # ---- output ----------------------------------------------------- #
    if output_cb is not None:
        target = getattr(output_cb, "__self__", output_cb)

        @_OUTPUT_CB
        def _cb(userdata, tensors_ptr, n_tensors, state):
            py_obj = ctypes.cast(userdata, POINTER(ctypes.py_object)).contents.value
            n = int(n_tensors)
            views: List[_TensorView] = []
            for i in range(n):
                t = tensors_ptr[i]
                mem = t.mem
                attr = t.attr
                if attr:
                    a = attr.contents
                    idx = int(a.index)
                    n_elems = int(a.n_elems)
                    name = a.name.decode("utf-8", "replace") if a.name else ""
                    dtype = int(a.dtype)
                else:
                    idx, n_elems, name, dtype = 0, 0, "", 0
                if mem and mem.contents.virt_addr and n_elems > 0:
                    data = np.ctypeslib.as_array(
                        ctypes_cast(mem.contents.virt_addr, POINTER(c_uint8)),
                        shape=(mem.contents.size,),
                    )
                    # Treat as fp16 by default — all the output tensors
                    # wdnai touches (e.g. talker past_hidden) are fp16.
                    data = data.view(np.float16)[:n_elems]
                else:
                    data = None
                views.append(_TensorView(idx, name, n_elems, dtype, data))
            try:
                rc = int(output_cb(py_obj, views, int(state)))
            except Exception:  # noqa: BLE001
                rc = -1
            return rc

        bundle.struct.output_callback = _cb
        bundle.struct.output_userdata = _pin_userdata(bundle, target)
        bundle._trampolines.append(_cb)

    # ---- output tensors (pre-allocated, for the talker path) -------- #
    # Must live on the same callback struct as ``output_callback`` or
    # the C library invokes the callback against an empty tensor array
    # and ``past_hidden`` (output index 1) never arrives.
    if output_tensors is not None:
        n = len(output_tensors)
        bundle.output_tensors = ctypes_cast(
            pointer(output_tensors), POINTER(_RknnTensor),
        )
        bundle.n_output_tensors = c_uint32(n)
        bundle.struct.output_tensors = bundle.output_tensors
        bundle.struct.n_output_tensors = c_uint32(n)
        bundle.pins.append(output_tensors)   # keep ctypes array alive

    # ---- sampling / input (passthrough) ----------------------------- #
    if sampling_cb is not None:
        bundle.struct.sampling_callback = _SAMPLING_CB(sampling_cb)
        bundle.struct.sampling_userdata = _pin_userdata(
            bundle, getattr(sampling_cb, "__self__", sampling_cb),
        )

    if input_cb is not None:
        bundle.struct.input_callback = _INPUT_CB(input_cb)
        bundle.struct.input_userdata = _pin_userdata(
            bundle, getattr(input_cb, "__self__", input_cb),
        )

    return bundle


# =============================================================================
# 7. Internal Runtime
# =============================================================================
class _Runtime:
    """Owns the librknn3 ``rknn3_context`` and the pre-allocated
    input/output tensor memory for a single RKNN model.

    Mirrors the slice of rknn3lite's ``RKNN3Runtime`` that the wdnai
    backends actually touch (``lib``, ``context``, ``inputs``,
    ``outputs``, ``set_shape``, ``create_mem``, ``set_internal_mem``).
    :meth:`set_shape` re-queries fresh ``rknn3_query(INPUT_ATTR/OUTPUT_ATTR)``
    and re-allocates the input/output buffers, so callers don't need
    to re-query attrs after a shape switch (unlike the old rknn3lite
    v1.0.4 wheel, which left ``inputs[i].attr`` stale).
    """

    def __init__(self, lib: ctypes.CDLL, target: str, device_id: Optional[bytes]):
        self.lib = lib
        self.target = target
        self.device_id = device_id

        # rknn3_init
        ctx = c_uint64(0)
        ext = _InitExtend()
        ext_p = ctypes_cast(pointer(ext), c_void_p) if device_id is not None else None
        if device_id is not None:
            ext.device_id = device_id        # bytes → c_char_p
            ret = lib.rknn3_init(pointer(ctx), pointer(ext))
        else:
            ret = lib.rknn3_init(pointer(ctx), None)
        if ret != 0:
            raise RuntimeError(f"rknn3_init failed: ret={ret}")
        self.context = ctx.value

        self.config = _RknnConfig()
        self.user_mem_internal = 0
        self.user_mem_weight = 0
        # Pinned attribute / tensor structs — must outlive the model.
        self._attr_refs: List[_TensorAttr] = []
        self._input_tensor_refs: List[_RknnTensor] = []
        self._output_tensor_refs: List[_RknnTensor] = []
        # Numpy arrays that back the tensor memory (kept alive so the
        # NPU never sees a freed virt_addr).
        self._input_buffer_refs: List[np.ndarray] = []
        self._output_buffer_refs: List[np.ndarray] = []
        # ``rknn3_tensor_mem*`` that we created via rknn3_create_mem and
        # need to release in :meth:`release`.
        self._owned_mem: List[int] = []
        self._user_mem_blocks: List[int] = []
        self.inputs: List[_RknnTensor] = []
        self.outputs: List[_RknnTensor] = []
        self.n_inputs = 0
        self.n_outputs = 0
        self._tensor_mem_pointers: List[int] = []
        # Per-runtime user data (used for context-lifetime-bound data).
        self._created = True

    # ------------------------------------------------------------------ #
    def _check_alive(self) -> None:
        if not getattr(self, "_created", False):
            raise RuntimeError("runtime has been released")

    # ------------------------------------------------------------------ #
    def load_model(self, model_path: str, weight_path: str) -> None:
        self._check_alive()
        ret = self.lib.rknn3_load_model_from_path(
            self.context, model_path.encode("utf-8"), weight_path.encode("utf-8"),
        )
        if ret != 0:
            raise RuntimeError(f"rknn3_load_model_from_path failed: ret={ret}")

    # ------------------------------------------------------------------ #
    def model_init(self) -> None:
        self._check_alive()
        ret = self.lib.rknn3_model_init(self.context, pointer(self.config))
        if ret != 0:
            raise RuntimeError(f"rknn3_model_init failed: ret={ret}")

    # ------------------------------------------------------------------ #
    def create_tensor(self) -> None:
        """Pre-allocate input/output tensor memory and build the
        ``_RknnTensor`` arrays the user (or :meth:`inference`) reads.

        Idempotent: subsequent calls re-allocate (the previous
        buffers are released first). Called by :meth:`set_shape` on
        every shape switch so the runtime always sees fresh
        ``rknn3_query(INPUT_ATTR/OUTPUT_ATTR)`` values.
        """
        self._check_alive()
        self._release_user_mem()

        io = _InputOutputNum()
        if self.lib.rknn3_query(
            self.context, int(_QueryCmd.IN_OUT_NUM),
            pointer(io), c_uint64(ctypes.sizeof(io)),
        ) != 0:
            raise RuntimeError("rknn3_query(IN_OUT_NUM) failed")
        self.n_inputs = int(io.n_input)
        self.n_outputs = int(io.n_output)

        # Build the input / output arrays.
        self._input_tensor_refs = [self._build_tensor(i, is_input=True) for i in range(self.n_inputs)]
        self._output_tensor_refs = [self._build_tensor(i, is_input=False) for i in range(self.n_outputs)]
        self.inputs = list(self._input_tensor_refs)
        self.outputs = list(self._output_tensor_refs)

    def _build_tensor(self, index: int, is_input: bool) -> _RknnTensor:
        attr = _TensorAttr()
        attr.index = c_uint32(index)
        cmd = _QueryCmd.INPUT_ATTR if is_input else _QueryCmd.OUTPUT_ATTR
        if self.lib.rknn3_query(
            self.context, int(cmd),
            pointer(attr), c_uint64(ctypes.sizeof(attr)),
        ) != 0:
            raise RuntimeError(f"rknn3_query({'INPUT' if is_input else 'OUTPUT'}_ATTR {index}) failed")
        # Keep the attr alive for the lifetime of this Runtime.
        self._attr_refs.append(attr)

        # Allocate the buffer. rknn3_create_mem returns a heap pointer
        # to a rknn3_tensor_mem that the runtime owns.
        size = int(attr.aligned_size) or int(attr.n_elems) * 2
        mem_ptr = self.lib.rknn3_create_mem(
            self.context,
            c_uint64(size),
            c_int32(int(attr.core_id)),
            c_int32(int(_MemAllocFlags.CACHEABLE)),
        )
        if not mem_ptr:
            raise RuntimeError(f"rknn3_create_mem failed for tensor {index}")
        self._owned_mem.append(int(mem_ptr))

        # Build a numpy array backed by the C virt_addr so the
        # underlying memory is reachable as a writable buffer (and the
        # GC can't free it under us).
        mem_struct = ctypes_cast(mem_ptr, POINTER(_TensorMemory))
        virt = mem_struct.contents.virt_addr
        size_b = int(mem_struct.contents.size) or size
        np_buf = np.ctypeslib.as_array(
            ctypes_cast(virt, POINTER(c_uint8)),
            shape=(size_b,),
        )
        if is_input:
            self._input_buffer_refs.append(np_buf)
        else:
            self._output_buffer_refs.append(np_buf)

        t = _RknnTensor()
        t.mem = ctypes_cast(mem_ptr, POINTER(_TensorMemory))
        t.attr = pointer(attr)
        return t

    def _release_user_mem(self) -> None:
        """Release all tensor memory and clear the cached buffers."""
        for ptr in self._owned_mem:
            try:
                self.lib.rknn3_destroy_mem(self.context, c_void_p(ptr))
            except Exception:  # noqa: BLE001
                pass
        self._owned_mem.clear()
        self._user_mem_blocks.clear()
        self._input_tensor_refs.clear()
        self._output_tensor_refs.clear()
        self._input_buffer_refs.clear()
        self._output_buffer_refs.clear()
        self._attr_refs.clear()
        self.inputs = []
        self.outputs = []
        self.n_inputs = 0
        self.n_outputs = 0

    # ------------------------------------------------------------------ #
    def set_internal_mem(self, mem_blocks: Sequence[int]) -> None:
        """Hand pre-allocated ``rknn3_tensor_mem*`` blocks to the runtime
        (the shared-internal-memory trick from
        ``rknn3_model_zoo/examples/Qwen3_ASR``)."""
        self._check_alive()
        if not mem_blocks:
            return
        n = len(mem_blocks)
        arr = (c_void_p * n)(*mem_blocks)
        ret = self.lib.rknn3_set_internal_mem(self.context, arr, c_uint32(n))
        if ret != 0:
            raise RuntimeError(f"rknn3_set_internal_mem failed: ret={ret}")
        # Keep the (borrowed) pointers alive until release; the C side
        # is the actual owner.
        self._user_mem_blocks.extend(int(p) for p in mem_blocks)

    def set_shape(self, shape_id: int) -> None:
        self._check_alive()
        ret = self.lib.rknn3_set_shape(self.context, c_int32(int(shape_id)))
        if ret != 0:
            raise RuntimeError(f"rknn3_set_shape({shape_id}) failed: ret={ret}")
        # Re-allocate input/output buffers for the new shape.
        self.create_tensor()

    # ------------------------------------------------------------------ #
    def set_input_data(self, idx: int, data: np.ndarray, data_format: Optional[str]) -> None:
        """Copy a numpy array into the ``idx``-th input buffer.

        ``data_format`` is accepted for parity with rknn3lite but
        ignored: wdnai's backends always pass NCHW/NHWC layouts that
        match the model's expected input; if the user truly needs a
        layout transpose they should do it before calling ``inference``.

        The model's input dtype is read from the queried
        ``RKNN3TensorAttr.dtype``. The caller may pass float32 while
        the model expects float16 (typical for rknn3 exports) — we
        cast before writing so the GPU sees the bytes it expects.
        """
        self._check_alive()
        if idx >= self.n_inputs:
            raise IndexError(f"input index {idx} >= n_inputs {self.n_inputs}")
        t = self._input_tensor_refs[idx]
        if not t.mem or not t.mem.contents.virt_addr:
            raise RuntimeError(f"input tensor {idx} has no memory")
        dst = self._input_buffer_refs[idx]
        size_bytes = int(t.mem.contents.size)
        if data is not None:
            # Match the model's expected dtype so the buffer bytes are
            # interpreted correctly by rknn3_run (e.g. caller hands in
            # float32 while the export is float16).
            model_dtype = _np_dtype_for(int(t.attr.contents.dtype))
            if data.dtype != model_dtype:
                # Cast, preserving shape — for layout conversion the
                # caller should pass an NCHW/NHWC array of the right
                # dtype already.
                data = data.astype(model_dtype, copy=False)
            flat = np.ascontiguousarray(data).reshape(-1)
            n = min(flat.size, size_bytes // flat.itemsize)
            dst[: n * flat.itemsize].view(flat.dtype)[:n] = flat[:n]

    def get_output(self, idx: int) -> np.ndarray:
        self._check_alive()
        if idx >= self.n_outputs:
            raise IndexError(f"output index {idx} >= n_outputs {self.n_outputs}")
        t = self._output_tensor_refs[idx]
        attr = t.attr.contents
        dtype = _np_dtype_for(int(attr.dtype))
        n_elems = int(attr.n_elems) or int(attr.aligned_size) // np.dtype(dtype).itemsize
        flat = self._output_buffer_refs[idx].view(dtype)[:n_elems].copy()
        # Reshape to the model's declared output shape (e.g. (144, 2560)
        # for Qwen3.5-VL vision). Falls back to 1D when n_dims is 0
        # (rare) or when the buffer size doesn't match the shape product
        # — in that case the caller still gets the raw 1D data.
        n_dims = int(attr.n_dims)
        if n_dims > 1 and n_elems > 0:
            shape = tuple(int(attr.shape[i]) for i in range(n_dims))
            if all(d > 0 for d in shape) and int(np.prod(shape)) == n_elems:
                return flat.reshape(shape)
        return flat

    def get_input_attr(self, idx: int) -> _TensorAttr:
        self._check_alive()
        if idx >= self.n_inputs:
            raise IndexError(f"input index {idx} >= n_inputs {self.n_inputs}")
        return self._input_tensor_refs[idx].attr.contents

    def get_output_attr(self, idx: int) -> _TensorAttr:
        self._check_alive()
        if idx >= self.n_outputs:
            raise IndexError(f"output index {idx} >= n_outputs {self.n_outputs}")
        return self._output_tensor_refs[idx].attr.contents

    # ------------------------------------------------------------------ #
    def rknn3_query(self, cmd: int, *, index: int = 0) -> Any:
        self._check_alive()
        ec = int(cmd)
        if ec == int(_QueryCmd.IN_OUT_NUM):
            out = _InputOutputNum()
            self._query_into(ec, out, ctypes.sizeof(out))
            return out
        if ec in (int(_QueryCmd.INPUT_ATTR), int(_QueryCmd.OUTPUT_ATTR),
                  int(_QueryCmd.ORIGIN_INPUT_ATTR), int(_QueryCmd.ORIGIN_OUTPUT_ATTR)):
            out = _TensorAttr()
            out.index = c_uint32(index)
            self._query_into(ec, out, ctypes.sizeof(out))
            return out
        if ec == int(_QueryCmd.SDK_VERSION):
            out = _SDKVersion()
            self._query_into(ec, out, ctypes.sizeof(out))
            return out
        if ec == int(_QueryCmd.CORE_NUMBER):
            n = c_int32(0)
            self._query_into(ec, n, ctypes.sizeof(n))
            return int(n.value)
        if ec == int(_QueryCmd.CORE_MEM_SIZE):
            n = c_int32(0)
            self._query_into(int(_QueryCmd.CORE_NUMBER), n, ctypes.sizeof(n))
            if n.value <= 0:
                return []
            arr = (_CoreMemSize * n.value)()
            self._query_into(ec, arr, ctypes.sizeof(arr))
            return list(arr)
        if ec == int(_QueryCmd.DEVICE_MEM_INFO):
            out = _DevMemInfo()
            self._query_into(ec, out, ctypes.sizeof(out))
            return out
        if ec == int(_QueryCmd.LLM_CONFIG):
            out = _LLMConfig()
            self._query_into(ec, out, ctypes.sizeof(out))
            return out
        raise NotImplementedError(f"rknn3_query cmd={ec} not implemented")

    def _query_into(self, cmd: int, buf: Any, size: int) -> None:
        ret = self.lib.rknn3_query(
            self.context, c_int(cmd), ctypes_cast(pointer(buf), c_void_p), c_uint64(size),
        )
        if ret != 0:
            raise RuntimeError(f"rknn3_query({cmd}) failed: ret={ret}")

    # ------------------------------------------------------------------ #
    def run(self) -> int:
        self._check_alive()
        if not self._input_tensor_refs or not self._output_tensor_refs:
            raise RuntimeError("run() before create_tensor()")
        n_in = len(self._input_tensor_refs)
        n_out = len(self._output_tensor_refs)
        in_arr = (_RknnTensor * n_in)(*self._input_tensor_refs)
        out_arr = (_RknnTensor * n_out)(*self._output_tensor_refs)
        # The C examples (e.g. Qwen3_5_VL/cpp/vision/rknn_qwen3_5_vl_vision.cc)
        # call ``rknn3_mem_sync(.., RKNN3_MEMORY_SYNC_TO_DEVICE)`` for each
        # input tensor right before ``rknn3_run`` and ``.._FROM_DEVICE`` for
        # each output right after. Skipping those steps leaves the NPU
        # reading stale bytes from the cacheable input buffers, which on this
        # driver manifests as the model producing all-zero outputs (the
        # buffer was memset to 0 on create_mem and never flushed).
        for i in range(n_in):
            if in_arr[i].mem:
                ret = self.lib.rknn3_mem_sync(
                    self.context, in_arr[i].mem, c_int32(int(_MemSyncMode.TO_DEVICE)),
                )
                if ret != 0:
                    raise RuntimeError(f"rknn3_mem_sync input[{i}] failed: ret={ret}")
        ret = self.lib.rknn3_run(
            self.context,
            ctypes_cast(pointer(in_arr), c_void_p), c_uint32(n_in),
            ctypes_cast(pointer(out_arr), c_void_p), c_uint32(n_out),
        )
        if ret != 0:
            raise RuntimeError(f"rknn3_run failed: ret={ret}")
        for i in range(n_out):
            if out_arr[i].mem:
                ret2 = self.lib.rknn3_mem_sync(
                    self.context, out_arr[i].mem, c_int32(int(_MemSyncMode.FROM_DEVICE)),
                )
                if ret2 != 0:
                    raise RuntimeError(f"rknn3_mem_sync output[{i}] failed: ret={ret2}")
        return int(ret)

    # ------------------------------------------------------------------ #
    def release(self) -> None:
        if not getattr(self, "_created", False):
            return
        try:
            self._release_user_mem()
            self.lib.rknn3_destroy(self.context)
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._created = False
            self.context = 0
            self.inputs = []
            self.outputs = []


def _np_dtype_for(rknn_dtype: int) -> Any:
    return {
        0: np.float32, 1: np.float16, 2: np.int8, 3: np.uint8,
        4: np.int16, 5: np.uint16, 6: np.int32, 7: np.uint32,
        8: np.int64, 9: np.uint64, 10: np.bool_,
    }.get(int(rknn_dtype), np.float16)


# =============================================================================
# 8. LLM session — wraps rknn3_session_t* + input packing
# =============================================================================
class _LLMSession:
    """One rknn3 session handle + cached inputs / params."""

    def __init__(self, runtime: "_LLMRuntime", session: int, params: List[dict], callback: Optional[_LLMCallbackBundle]):
        self.runtime = runtime
        self.session_index = session
        self.lib = runtime.lib
        self._params = params
        self._callback = callback
        self._llm_params: List[_LLMParam] = []
        self._c_params: List[int] = []   # addresses of _LLMParams (ints)
        self._packed_inputs: List[_LLMInput] = []
        self._input_pin_refs: List[Any] = []
        self._input_anchor_refs: List[Any] = []   # byte string anchors
        self._infer_param = _LLMInferParam()
        self._infer_p = pointer(self._infer_param)
        self._session: Optional[int] = None
        self._init()

    def _init(self) -> None:
        # Query the model's own LLM config and override max_context_len
        # with whatever the model says — the C++ reference examples do
        # this and the warning at rknn3_session_init time confirms the
        # runtime wants the model's intrinsic value, not whatever the
        # user passes. If the query fails we fall back to the user value.
        try:
            llm_cfg = self.runtime.rknn3_query(_QueryCmd.LLM_CONFIG)
            model_max_ctx = int(getattr(llm_cfg, "max_ctx_len", 0))
        except Exception:  # noqa: BLE001
            model_max_ctx = 0
        if model_max_ctx > 0:
            for d in self._params:
                d["max_context_len"] = model_max_ctx
        self._llm_params, self._c_params = _pack_llm_params(self._params)
        # C signature: rknn3_session* rknn3_session_init(rknn3_context,
        #     rknn3_llm_param* params, int n_params);
        # ``params`` is a pointer to a (contiguous) array of n_params
        # structs — not a pointer to an array of pointers. With
        # n_params=1 (the wdnai-supported case) this collapses to a
        # pointer to a single _LLMParam struct. ``pointer(self._llm_params[0])``
        # gives that address.
        n = len(self._llm_params)
        sess = self.lib.rknn3_session_init(
            self.runtime.context, pointer(self._llm_params[0]), c_int(n),
        )
        if not sess:
            raise RuntimeError("rknn3_session_init returned NULL")
        self._session = int(sess)
        if self._callback is not None:
            cb = ctypes_cast(pointer(self._callback.struct), c_void_p)
            if self.lib.rknn3_session_set_callback(self._session, cb) != 0:
                raise RuntimeError("rknn3_session_set_callback failed")

    def set_chat_template(self, system: str, prefix: str, postfix: str) -> int:
        return self.lib.rknn3_session_set_chat_template(
            self._session,
            system.encode("utf-8"), prefix.encode("utf-8"), postfix.encode("utf-8"),
        )

    def clear_kvcache(self, clear_policy: int = 0) -> int:
        return self.lib.rknn3_session_clear_kvcache(
            self._session, c_int32(int(clear_policy)),
        )

    def _set_infer_param(
        self,
        *,
        keep_history: Optional[int],
        max_new_tokens: Optional[int],
        prefill_only: Optional[bool] = None,
        disable_sampling: Optional[bool] = None,
    ) -> None:
        if keep_history is not None:
            self._infer_param.keep_history = c_int(int(keep_history))
        if max_new_tokens is not None:
            self._infer_param.max_new_tokens = c_int32(int(max_new_tokens))
        elif self._infer_param.max_new_tokens == 0:
            # Default for the underlying ``rknn3_llm_infer_param``
            # struct is 0 (the C memset default), but 0 in the C API
            # means "unbounded" — Qwen3-VL / Qwen3.5-VL with that
            # setting allocate a huge prefill buffer and crash with
            # ``std::bad_alloc`` as soon as the prompt is non-trivial.
            # Prefer the session-configured ``max_new_tokens`` from the
            # init args (rknn3lite semantics: code_predictor=15,
            # talker=8192 — see examples/Qwen3_TTS/cpp/talker.cc:266).
            sess_max = 0
            if self._params and isinstance(self._params[0], dict):
                sess_max = int(self._params[0].get("max_new_tokens", 0) or 0)
            if sess_max > 0:
                self._infer_param.max_new_tokens = c_int32(sess_max)
            else:
                self._infer_param.max_new_tokens = c_int32(1024)
        if prefill_only is not None:
            self._infer_param.prefill_only = c_bool(bool(prefill_only))
        if disable_sampling is not None:
            self._infer_param.disable_sampling = c_bool(bool(disable_sampling))

    def _pack_inputs(
        self,
        *,
        prompt: Optional[str],
        embeds: Optional[np.ndarray],
        tokens: Optional[Any],
        inputs: Optional[Sequence[Any]],
        enable_thinking: bool,
    ) -> List[_LLMInput]:
        packed: List[_LLMInput] = []
        self._input_pin_refs.clear()

        has_multimodal = any(
            isinstance(x, (RKNN3Image, RKNN3Audio)) for x in (inputs or [])
        )

        # ``rknn3_session_run`` 严格要求 ``inputs[0]`` 必须是
        # multimodal（``RKNN3_LLM_INPUT_MULTIMODAL``），aux 必须
        # 排在它后面 —— 看 C 库错误
        # ``RKLLMSession: Please put the LLM input at the first
        # position!``，以及 C++ 参考
        # ``examples/Qwen3_VL/cpp/llm/rknn_qwen3_vl_llm.cc:174-180``
        # 先 ``inputs[0].input_type = RKNN3_LLM_INPUT_MULTIMODAL``
        # 再 ``for i in range(3) inputs[i+1].input_type = ...AUX``。
        # 早期版本把 aux 排在前面，结果 C 库把首个 aux 当成 LLM
        # input 校验失败 → ``ret=-1``。注释里说"double-prompt caused
        # ret=-1"是把表面症状记成原因；真正原因是 inputs 顺序颠倒。
        if has_multimodal:
            for x in inputs:
                if isinstance(x, (RKNN3Image, RKNN3Audio)):
                    packed.append(self._pack_multimodal(x, prompt, enable_thinking))
        # Auxiliary inputs (deepstack, etc.) — kept in order
        # AFTER the multimodal. 严格按 vlm.py 构造顺序保留 aux 索引
        # （vlm.py 给每个 aux 设 ``aux.index = 2 + i``，与模型
        # ``rknn3_query`` 查到的 deepstack input attr.index 一致）。
        if inputs:
            for x in inputs:
                if isinstance(x, RKNN3AuxTensorWrapper):
                    aux = self._pack_aux(x)
                    packed.append(aux)
        if has_multimodal:
            return packed
        # 3) Single-payload (prompt / embeds / tokens) — text-only path
        if prompt is not None:
            packed.append(self._pack_prompt(prompt, enable_thinking))
        elif embeds is not None:
            packed.append(self._pack_embed(embeds, enable_thinking))
        elif tokens is not None:
            packed.append(self._pack_tokens(tokens, enable_thinking))
        else:
            # Match rknn3lite: empty call still sends an empty prompt input
            packed.append(self._pack_prompt("", enable_thinking))
        return packed

    def _pack_prompt(self, prompt: str, enable_thinking: bool) -> _LLMInput:
        li = _LLMInput()
        li.role = None
        li.input_type = c_int(int(_LLMInputType.PROMPT))
        t = li._u.llm_input
        # Pin the c_char_p wrapper itself — once the wrapper is GC'd
        # its internal buffer would be freed and the pointer inside
        # ``t.prompt`` would dangle. The wrapper owns its own copy of
        # the bytes.
        anchor = prompt.encode("utf-8")
        cprompt = c_char_p(anchor)
        self._input_anchor_refs.append(cprompt)
        t.name = None
        t.prompt = cprompt
        t.embed = c_void_p(0)
        t.tokens = None
        t.n_tokens = c_uint64(0)
        t.enable_thinking = c_bool(bool(enable_thinking))
        return li

    def _pack_embed(self, embeds: np.ndarray, enable_thinking: bool) -> _LLMInput:
        if embeds.dtype != np.float16:
            embeds = embeds.astype(np.float16)
        embeds = np.ascontiguousarray(embeds)
        li = _LLMInput()
        li.role = None
        li.input_type = c_int(int(_LLMInputType.EMBED))
        t = li._u.llm_input
        self._input_pin_refs.append(embeds)   # keep buffer alive
        # LLM graphs export the embed input as ``input_embeds`` — the
        # C++ reference sets it (talker.cc:250) and so must we, else
        # the C library can't bind the input and the run returns
        # RKLLM_RUN_ERROR. Mirrors the multimodal path's
        # ``_pack_multimodal`` requirement.
        name_cp = c_char_p(b"input_embeds")
        self._input_anchor_refs.append(name_cp)   # pins the c_char_p + its bytes
        t.name = name_cp
        t.prompt = None
        t.embed = c_void_p(embeds.ctypes.data)
        t.tokens = None
        t.n_tokens = c_uint64(int(np.prod(embeds.shape)) // embeds.shape[-1] if embeds.ndim > 1 else embeds.shape[0])
        t.enable_thinking = c_bool(bool(enable_thinking))
        return li

    def _pack_tokens(self, tokens: Any, enable_thinking: bool) -> _LLMInput:
        if isinstance(tokens, np.ndarray):
            arr = np.ascontiguousarray(tokens.astype(np.int32))
        else:
            arr = np.asarray(list(tokens), dtype=np.int32)
        li = _LLMInput()
        li.role = None
        li.input_type = c_int(int(_LLMInputType.TOKEN))
        t = li._u.llm_input
        self._input_pin_refs.append(arr)
        t.name = None
        t.prompt = None
        t.embed = c_void_p(0)
        t.tokens = ctypes_cast(arr.ctypes.data, POINTER(c_int32))
        t.n_tokens = c_uint64(arr.size)
        t.enable_thinking = c_bool(bool(enable_thinking))
        return li

    def _pack_multimodal(self, mm: Any, prompt: Optional[str], enable_thinking: bool) -> _LLMInput:
        li = _LLMInput()
        li.role = None
        li.input_type = c_int(int(_LLMInputType.MULTIMODAL))
        mm_t = li._u.multimodal_input
        # The LLM graph was exported with an input tensor named
        # ``input_embeds`` (Qwen3-VL / Qwen3.5-VL family); without this
        # name the session can't bind the image data and returns -1.
        # Pin both the bytes source and the c_char_p wrapper so neither
        # is GC'd before ``rknn3_session_run`` reads the pointer.
        if not hasattr(self, "_input_name_anchor"):
            self._input_name_anchor = b"input_embeds"
        name_cp = c_char_p(self._input_name_anchor)
        self._input_anchor_refs.append((self._input_name_anchor, name_cp))
        mm_t.name = name_cp
        if prompt is not None:
            anchor = prompt.encode("utf-8")
            prompt_cp = c_char_p(anchor)
            self._input_anchor_refs.append((anchor, prompt_cp))
            mm_t.prompt = prompt_cp
        else:
            mm_t.prompt = None
        mm_t.tokens = None
        mm_t.n_tokens = c_uint64(0)
        mm_t.enable_thinking = c_bool(bool(enable_thinking))
        if isinstance(mm, RKNN3Image):
            mm_t.image = mm
        elif isinstance(mm, RKNN3Audio):
            mm_t.audio = mm
        else:
            raise TypeError(f"unsupported multimodal input: {type(mm).__name__}")
        return li

    def _pack_aux(self, aux: RKNN3AuxTensorWrapper) -> _LLMInput:
        """Pack one aux input — with lazy **per-slot pre-allocation**.

        Critical lesson learned on RK3588 (2026-09-08):
        ``rknn3_create_mem`` for aux tensors is a **finite-pool**
        resource on the NPU driver. Calling it once per inference
        exhausts the pool within ~8 frames and the driver returns
        ``MSG_MEM_CREATE_ACK fail`` + ``rknn3_create_mem: device
        failed to create memory``. With the wrong pre-allocation
        strategy, ``_pack_aux`` would then raise a Python exception
        that crosses through the ctypes boundary into the reply
        path, and the worker segfaults (``rc=-11``) — the user's
        symptom is "video detection dies after a few frames".

        The fix follows the C++ reference
        ``examples/Qwen3_VL/cpp/llm/rknn_qwen3_vl_llm.cc:61-120``:
        allocate the 3 deepstack mem blocks **once** in
        ``_LLMRuntime.__init__`` (lazily, on first ``_pack_aux``
        call) and reuse them for every subsequent frame — only the
        **bytes inside** the buffer are overwritten each call. The
        attr (shape / dtype / index / aligned_size) is also cached
        on first call so we never re-allocate.

        Sized to ``aux.align_size`` (which vlm.py sets to the vision
        output's ``aligned_size`` at session init), so the buffer
        exactly fits the largest deepstack output the model will
        emit at this ``image_size`` — no over-allocation, no
        re-allocation, no driver pool leak.
        """
        li = _LLMInput()
        li.role = None
        li.input_type = c_int(int(_LLMInputType.AUX))
        aux_t = li._u.aux_input
        if aux.aux_data is not None:
            data = np.ascontiguousarray(aux.aux_data.astype(np.float16, copy=False))
        else:
            data = np.zeros(0, dtype=np.float16)

        # Lazy alloc: first ``_pack_aux`` call creates the mem pool,
        # every subsequent call reuses the cached blocks. We key the
        # pool by ``(align_size, aux_index)`` — a model with two
        # vision variants (e.g. 384 / 768) sees different align_size
        # per image_size and gets a separate pool; within one size the
        # **same** aux input across frames reuses the same buffer.
        #
        # aux_index is part of the key because Qwen3-VL-4B emits 3
        # deepstack aux tensors (at model input indices 2/3/4) and
        # vlm.py currently sets all three to the same ``align_size``;
        # a pure ``(align_size,)`` key would have all three share one
        # mem block + one attr, which leaks ~1.5–1.8 GB of NPU memory
        # on RK3588 because the driver can't reconcile the duplicate
        # (mem, attr) entries on session teardown. See AGENTS.md for
        # the full root-cause writeup.
        pool_key = (int(aux.align_size), int(aux.index))
        pool = self.runtime._aux_pool.get(pool_key)
        if pool is None:
            size = max(int(aux.align_size), 1)
            mem_ptr = self.runtime.lib.rknn3_create_mem(
                self.runtime.context,
                c_uint64(size),
                c_int32(0),
                c_int32(int(_MemAllocFlags.CACHEABLE)),
            )
            if not mem_ptr:
                # Driver refused. The right move is to surface a
                # clean error to the worker reply path (so the main
                # process can log + respawn) rather than letting a
                # Python exception escape through ctypes and segfault.
                # We stash the message in a shared dict and let the
                # caller of ``_pack_inputs`` raise — see
                # ``_pack_inputs``.
                self.runtime._aux_last_err = (
                    f"rknn3_create_mem (aux, size={size}) failed"
                )
                raise RuntimeError(self.runtime._aux_last_err)
            attr = _TensorAttr()
            attr.index = c_uint32(int(aux.index))
            attr.n_elems = c_uint32(int(data.size))
            attr.n_dims = c_uint32(int(min(data.ndim, _MAX_DIMS)))
            for i in range(min(data.ndim, _MAX_DIMS)):
                attr.shape[i] = c_uint32(int(data.shape[i]))
            attr.dtype = c_int(int(RKNN3TensorType.RKNN3_TENSOR_FLOAT16))
            attr.aligned_size = c_uint64(size)
            self.runtime._aux_attr_refs.append(attr)
            pool = {
                "mem_ptr": int(mem_ptr),
                "size": size,
                "attr": attr,
            }
            self.runtime._aux_pool[pool_key] = pool
        # ``mem_ptr`` in cache is a raw ctypes pointer address (int).
        # Wrap in c_void_p so the downstream cast treats it as an
        # opaque pointer (matches what rknn3_create_mem returned).
        mem_ptr = c_void_p(pool["mem_ptr"])
        self._input_pin_refs.append(data)

        # Copy fresh bytes into the pre-allocated buffer.
        mem_struct = ctypes_cast(mem_ptr, POINTER(_TensorMemory))
        if data.size > 0 and mem_struct.contents.virt_addr:
            # ``data`` may be a 2-D ``(n_patches, embed_dim)`` array
            # (vision RKNN deepstack outputs, e.g. Qwen3-VL-4B @ 384
            # emits ``(144, 2560)``). The destination is a 1-D fp16
            # view of the same total length, so numpy cannot auto-
            # broadcast a 2-D source into it — ``dst[:] = data``
            # raises ``ValueError: could not broadcast input array
            # from shape (144, 2560) into shape (368640,)``.
            # ``ravel()`` returns a flat view when possible (no copy);
            # the assignment writes bytes in C order, matching the
            # C-side ``rknn3_session_run`` layout.
            flat = data.ravel() if data.ndim != 1 else data
            buf_bytes = mem_struct.contents.size
            dst = np.ctypeslib.as_array(
                ctypes_cast(mem_struct.contents.virt_addr, POINTER(c_uint8)),
                shape=(buf_bytes,),
            ).view(np.float16)[:flat.size]
            dst[:] = flat

        attr = pool["attr"]
        aux_t.mem = ctypes_cast(mem_ptr, POINTER(_TensorMemory))
        aux_t.attr = pointer(attr)
        return li

    def run(
        self,
        *,
        keep_history: Optional[int],
        max_new_tokens: Optional[int],
        prefill_only: Optional[bool] = None,
        disable_sampling: Optional[bool] = None,
        inputs: List[_LLMInput],
    ) -> int:
        self._set_infer_param(
            keep_history=keep_history, max_new_tokens=max_new_tokens,
            prefill_only=prefill_only, disable_sampling=disable_sampling,
        )
        n = len(inputs)
        arr = (_LLMInput * n)(*inputs)
        return self.lib.rknn3_session_run(
            self._session, ctypes_cast(pointer(arr), c_void_p), c_uint32(n),
            self._infer_p,
        )

    def run_async(
        self,
        *,
        keep_history: Optional[int],
        max_new_tokens: Optional[int],
        prefill_only: Optional[bool] = None,
        disable_sampling: Optional[bool] = None,
        inputs: List[_LLMInput],
    ) -> int:
        self._set_infer_param(
            keep_history=keep_history, max_new_tokens=max_new_tokens,
            prefill_only=prefill_only, disable_sampling=disable_sampling,
        )
        n = len(inputs)
        arr = (_LLMInput * n)(*inputs)
        return self.lib.rknn3_session_run_async(
            self._session, ctypes_cast(pointer(arr), c_void_p), c_uint32(n),
            self._infer_p,
        )

    def destroy(self) -> None:
        if self._session is None:
            return
        try:
            self.lib.rknn3_session_destroy(self._session)
        except Exception:  # noqa: BLE001
            pass
        self._session = None


def _pack_llm_params(
    args: Sequence[dict],
    *,
    logits_name: str = "output",
) -> Tuple[List[_LLMParam], List[int]]:
    """Build a list of C-side ``_LLMParam`` structs from Python dicts.

    The dict shape mirrors the legacy rknn3lite API; we only need a
    handful of keys for the models wdnai touches:

    * ``max_new_tokens`` (int, default 512)
    * ``top_k`` (int, default 1)
    * ``top_p`` (float, default 0.9)
    * ``temperature`` (float, default 0.7)
    * ``repeat_penalty`` (float, default 1.0)
    * ``frequency_penalty`` (float, default 0.0)
    * ``presence_penalty`` (float, default 0.0)
    * ``vocab_size`` (int, required)
    * ``special_eos_id`` (int, required) — also ``special_bos_id`` accepted
    * ``max_context_len`` (int, required)
    * ``keep_history`` (int, default 0)
    * ``linefeed_id`` (int, optional)

    ``logits_name`` is the ``name`` field of the output tensor that the
    LLM session should treat as the logits. The C++ reference examples
    hard-code ``"output"`` (the canonical export name from
    rknn3-model-zoo); without this, the session runs in a mode that
    doesn't drive the embedding-style hidden-state output, and the
    result is all zeros.
    """
    out: List[_LLMParam] = []
    for d in args:
        # Per-arg override of the logits tensor name — most rknn3-model-zoo
        # graphs name it ``"output"``, but the Qwen3-TTS talker /
        # code_predictor export it as ``"logits"`` (talker.cc:1095/1192).
        # Wrong name → RKLLM_RUN_ERROR.
        arg_logits = d.pop("logits_name", logits_name) if isinstance(d, dict) else logits_name
        logits_anchor = arg_logits.encode("utf-8")
        p = _LLMParam()
        p.logits_name = c_char_p(logits_anchor)
        p.max_context_len = c_int32(int(d.get("max_context_len", 4096)))
        sp = _SamplingParams()
        sp.top_k = c_int32(int(d.get("top_k", 1)))
        sp.top_p = c_float(float(d.get("top_p", 0.9)))
        sp.temperature = c_float(float(d.get("temperature", 0.7)))
        sp.repeat_penalty = c_float(float(d.get("repeat_penalty", 1.0)))
        sp.frequency_penalty = c_float(float(d.get("frequency_penalty", 0.0)))
        sp.presence_penalty = c_float(float(d.get("presence_penalty", 0.0)))
        p.sampling_param = sp
        v = _VocabInfo()
        v.vocab_size = c_int(int(d.get("vocab_size", 0)))
        eos = d.get("special_eos_id", d.get("special_eos", []))
        bos = d.get("special_bos_id", d.get("special_bos", []))
        if isinstance(eos, int):
            eos = [eos]
        if isinstance(bos, int):
            bos = [bos]
        v.n_special_eos_id = c_int(len(eos))
        v.n_special_bos_id = c_int(len(bos))
        for i, x in enumerate(eos[:_MAX_SPECIAL_EOS_ID_NUM]):
            v.special_eos_id[i] = c_int(int(x))
        for i, x in enumerate(bos[:_MAX_SPECIAL_BOS_ID_NUM]):
            v.special_bos_id[i] = c_int(int(x))
        v.linefeed_id = c_int(int(d.get("linefeed_id", 0)))
        p.vocab_info = v
        ep = _LLMExtendParam()
        p.extend_param = ep
        out.append(p)
    addrs = [ctypes.addressof(p) for p in out]
    return out, addrs


# =============================================================================
# 9. Public RKNN3Lite
# =============================================================================
class _LLMRuntime(_Runtime):
    """Runtime variant for ``llm_mode=True``. Adds the LLM session
    plumbing. Kept separate so the regular :class:`_Runtime` can stay
    single-purpose (just create_tensor + run)."""

    def __init__(self, lib, target, device_id):
        super().__init__(lib, target, device_id)
        self._aux_mem_refs: List[int] = []
        self._aux_attr_refs: List[_TensorAttr] = []
        # Per-align-size aux-mem pool (see ``_pack_aux`` docstring).
        # Key = ``aux.align_size``; value = ``{"mem_ptr", "size", "attr"}``.
        # Lazy alloc on first ``_pack_aux``; reused for every subsequent
        # frame so we never re-call ``rknn3_create_mem`` mid-session.
        self._aux_pool: Dict[int, dict] = {}
        self.session: Optional[_LLMSession] = None

    def release(self) -> None:
        if self.session is not None:
            try:
                self.session.destroy()
            except Exception:  # noqa: BLE001
                pass
            self.session = None
        # Destroy every cached aux-mem block from ``_aux_pool``. Note
        # ``_aux_pool`` keys by ``align_size``, so a session that ran
        # two vision variants (384 + 768) drops two blocks; a
        # single-variant session drops one. Old transient-tracking
        # lists (``_aux_mem_refs`` / ``_aux_attr_refs``) are kept for
        # ABI compat but are unused by the lazy-pool path.
        for entry in self._aux_pool.values():
            try:
                self.lib.rknn3_destroy_mem(
                    self.context, c_void_p(int(entry["mem_ptr"])))
            except Exception:  # noqa: BLE001
                pass
        self._aux_pool.clear()
        for ptr in self._aux_mem_refs:
            try:
                self.lib.rknn3_destroy_mem(self.context, c_void_p(ptr))
            except Exception:  # noqa: BLE001
                pass
        self._aux_mem_refs.clear()
        self._aux_attr_refs.clear()
        super().release()


class RKNN3Lite:
    """Public RKNN3 entry point used by every wdnai backend.

    Surface is intentionally narrower than rknn3lite's
    :class:`rknn3lite.api.RKNN3Lite` — only the methods wdnai
    actually calls are exposed; everything else is encapsulated
    inside this shim.

    Construction is lazy with respect to librknn3 — :func:`load_library`
    only runs the first time a :class:`RKNN3Lite` is instantiated.
    The NPU driver is initialised inside :meth:`init_runtime` (or, in
    the tts talker path, inside :meth:`init_llm_session`).
    """

    def __init__(self, llm_mode: bool = False, verbose: bool = False) -> None:
        self.llm_mode = bool(llm_mode)
        self.verbose = bool(verbose)
        self.lib: ctypes.CDLL = load_library()
        self.model_path: Optional[str] = None
        self.weight_path: Optional[str] = None
        self.rknn_config = _RknnConfig()        # user-mutable before init
        self.rknn_runtime: Optional[_Runtime] = None
        self._llm_args: Optional[List[dict]] = None
        self._llm_callback: Optional[_LLMCallbackBundle] = None
        self._init_device_id: Optional[bytes] = None
        self._released = False

    # ------------------------------------------------------------------ #
    def load_rknn(self, model_path: str, weight_path: str) -> int:
        if not os.path.isfile(model_path):
            return -1
        if weight_path and not os.path.isfile(weight_path):
            return -1
        self.model_path = model_path
        self.weight_path = weight_path
        return 0

    def init_runtime(
        self,
        target: str,
        core_mask: int,
        *,
        llm_args: Optional[Sequence[dict]] = None,
        llm_callback: Optional[_LLMCallbackBundle] = None,
        device_id: Optional[bytes] = None,
        **_: Any,
    ) -> int:
        if self.model_path is None or self.weight_path is None:
            return -1
        try:
            self.rknn_config.run_core_mask = c_uint32(int(core_mask))
            self._init_device_id = device_id
            if self.llm_mode:
                runtime = _LLMRuntime(self.lib, target, device_id)
            else:
                runtime = _Runtime(self.lib, target, device_id)
            runtime.config = self.rknn_config
            runtime.user_mem_internal = int(self.rknn_config.user_mem_internal)
            runtime.user_mem_weight = int(self.rknn_config.user_mem_weight)
            runtime.load_model(self.model_path, self.weight_path)
            runtime.model_init()
            if not self.llm_mode:
                runtime.create_tensor()
            self.rknn_runtime = runtime
            if self.llm_mode and llm_args is not None:
                self._llm_args = list(llm_args)
                self._llm_callback = llm_callback
                runtime.session = _LLMSession(
                    typing_cast(_LLMRuntime, runtime), session=0, params=self._llm_args,
                    callback=llm_callback,
                )
            return 0
        except Exception:  # noqa: BLE001
            if self.rknn_runtime is not None:
                self.rknn_runtime.release()
                self.rknn_runtime = None
            return -1

    def init_llm_session(
        self,
        llm_args: Sequence[dict],
        llm_callback: _LLMCallbackBundle,
        session_index: int = 0,
    ) -> int:
        """Attach an LLM session to an already-initialised runtime
        (used by the TTS talker after ``init_runtime`` for non-LLM
        mode)."""
        if not isinstance(self.rknn_runtime, _LLMRuntime):
            return -1
        try:
            self._llm_args = list(llm_args)
            self._llm_callback = llm_callback
            rt = typing_cast(_LLMRuntime, self.rknn_runtime)
            rt.session = _LLMSession(
                rt, session=session_index, params=self._llm_args,
                callback=llm_callback,
            )
            return 0
        except Exception:  # noqa: BLE001
            return -1

    # ------------------------------------------------------------------ #
    def inference(
        self,
        inputs: Optional[Sequence[np.ndarray]] = None,
        data_format: Optional[Sequence[str]] = None,
    ) -> Optional[List[np.ndarray]]:
        if not isinstance(self.rknn_runtime, _Runtime) or isinstance(self.rknn_runtime, _LLMRuntime):
            return None
        if inputs is None:
            return None
        n = self.rknn_runtime.n_inputs
        if len(inputs) != n:
            return None
        for i, arr in enumerate(inputs):
            fmt = data_format[i] if data_format is not None and i < len(data_format) else None
            self.rknn_runtime.set_input_data(i, arr, fmt)
        rc = self.rknn_runtime.run()
        if rc != 0:
            print(f"[shim debug] run() returned {rc}", flush=True)
            return None
        return [self.rknn_runtime.get_output(i) for i in range(self.rknn_runtime.n_outputs)]

    # ------------------------------------------------------------------ #
    def _do_session_run(
        self,
        run_fn: Callable[..., int],
        *,
        inputs: Optional[Sequence[Any]],
        prompt: Optional[str],
        embeds: Optional[np.ndarray],
        tokens: Optional[Any],
        keep_history: Optional[int],
        max_new_tokens: Optional[int],
        prefill_only: Optional[bool] = None,
        disable_sampling: Optional[bool] = None,
        enable_thinking: bool = False,
    ) -> Tuple[int, List[Any]]:
        rt = self.rknn_runtime
        if not isinstance(rt, _LLMRuntime) or rt.session is None:
            return -1, []
        packed = rt.session._pack_inputs(
            prompt=prompt, embeds=embeds, tokens=tokens,
            inputs=inputs, enable_thinking=enable_thinking,
        )
        # Wall-clock anchors for ``printf_perf``: callers compute
        # prefill_ms = (sink.first_token_time - llm_start) * 1000
        # decode_ms  = (llm_end             - sink.first_token_time) * 1000
        # so both ends need real ``time.perf_counter()`` values — NOT 0.0
        # (which previously made ``prefill_ms`` print
        # ``first_token_time * 1000``, i.e. seconds since process start).
        llm_start = time.perf_counter()
        ret = run_fn(
            keep_history=keep_history, max_new_tokens=max_new_tokens,
            prefill_only=prefill_only, disable_sampling=disable_sampling,
            inputs=packed,
        )
        llm_end = time.perf_counter()

        # Pull the authoritative prefill / decode token counts from the
        # C side via ``rknn3_session_query_state`` — same API the C++
        # LocateAnything / Qwen3_5_VL demos use to read state into
        # ``rknn_perf_metrics_t`` (see e.g.
        # ``rknn3-model-zoo/examples/LocateAnything/cpp/llm/
        # rknn_locate_anything_llm.cc:156-164``). The shim's
        # ``_LLMRunState`` mirrors librknn3's RKLLMRunState byte-for-byte
        # (verified against ``rknn3_api.h:1058-1072``: same field order,
        # 12 fields, identical widths including the 4-byte kvcache_policy
        # enum at offset 72). Layout drift on a future librknn3 bump
        # wouldn't raise — ctypes would just read garbage at the wrong
        # offset — so the sanity bounds below fall back to
        # ``len(sink.tokens)`` (decode) / 0 (prefill) when values look
        # out of range. The upper bound 1_000_000 is intentionally
        # generous (no model exceeds max_context_len that high) so it
        # never trips on legitimate runs.
        n_prefill_tokens = 0
        n_decode_tokens = 0
        try:
            state = _LLMRunState()
            qret = self.lib.rknn3_session_query_state(
                rt.session._session,
                ctypes_cast(pointer(state), c_void_p),
            )
            if qret == 0:
                raw_prefill = int(state.n_prefill_tokens)
                raw_decode = int(state.n_decode_tokens)
                # Sanity bounds — sanity gate catches struct-layout drift
                # before the garbage values propagate to ``printf_perf``.
                if 0 < raw_prefill < 1_000_000 and 0 <= raw_decode < 1_000_000:
                    n_prefill_tokens = raw_prefill
                    n_decode_tokens = raw_decode
                else:
                    import logging as _logging
                    _logging.getLogger("wdnai._librknn").warning(
                        "rknn3_session_query_state returned out-of-range values "
                        "(prefill=%d decode=%d qret=%d) — likely struct layout "
                        "drift between shim's _LLMRunState and librknn3's "
                        "RKLLMRunState. Falling back to sink.tokens for decode.",
                        raw_prefill, raw_decode, qret,
                    )
                    n_decode_tokens = len(self.sink.tokens) if self.sink else raw_decode
        except Exception as e:  # noqa: BLE001
            # Symbol missing on this librknn3 build. Don't break the
            # inference — degrade to 0/0 like the original shim did.
            import logging as _logging
            _logging.getLogger("wdnai._librknn").debug(
                "rknn3_session_query_state unavailable, falling back to 0/0: %s", e
            )

        return ret, [n_decode_tokens, n_prefill_tokens, llm_start, llm_end]

    def session_run(
        self,
        *,
        inputs: Optional[Sequence[Any]] = None,
        prompt: Optional[str] = None,
        embeds: Optional[np.ndarray] = None,
        tokens: Optional[Any] = None,
        keep_history: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        prefill_only: Optional[bool] = None,
        disable_sampling: Optional[bool] = None,
        enable_thinking: bool = False,
        session_index: int = 0,
    ) -> Tuple[int, List[Any]]:
        rt = self.rknn_runtime
        if not isinstance(rt, _LLMRuntime) or rt.session is None:
            return -1, []
        return self._do_session_run(
            rt.session.run,
            inputs=inputs, prompt=prompt, embeds=embeds, tokens=tokens,
            keep_history=keep_history, max_new_tokens=max_new_tokens,
            prefill_only=prefill_only, disable_sampling=disable_sampling,
            enable_thinking=enable_thinking,
        )

    def session_run_async(
        self,
        *,
        inputs: Optional[Sequence[Any]] = None,
        prompt: Optional[str] = None,
        embeds: Optional[np.ndarray] = None,
        tokens: Optional[Any] = None,
        keep_history: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        prefill_only: Optional[bool] = None,
        disable_sampling: Optional[bool] = None,
        enable_thinking: bool = False,
        session_index: int = 0,
    ) -> Tuple[int, List[Any]]:
        rt = self.rknn_runtime
        if not isinstance(rt, _LLMRuntime) or rt.session is None:
            return -1, []
        return self._do_session_run(
            rt.session.run_async,
            inputs=inputs, prompt=prompt, embeds=embeds, tokens=tokens,
            keep_history=keep_history, max_new_tokens=max_new_tokens,
            prefill_only=prefill_only, disable_sampling=disable_sampling,
            enable_thinking=enable_thinking,
        )

    # ------------------------------------------------------------------ #
    def set_chat_template(self, system: str, prefix: str, postfix: str) -> int:
        if not isinstance(self.rknn_runtime, _LLMRuntime) or self.rknn_runtime.session is None:
            return -1
        return self.rknn_runtime.session.set_chat_template(system, prefix, postfix)

    def clear_kvcache(self, clear_policy: Optional[int] = None) -> int:
        if not isinstance(self.rknn_runtime, _LLMRuntime) or self.rknn_runtime.session is None:
            return -1
        cp = int(clear_policy) if clear_policy is not None else int(_KVCacheClearPolicy.CLEAR_ALL)
        return self.rknn_runtime.session.clear_kvcache(cp)

    def session_stop(self) -> int:
        if not isinstance(self.rknn_runtime, _LLMRuntime) or self.rknn_runtime.session is None:
            return -1
        return self.lib.rknn3_session_stop(self.rknn_runtime.session._session)

    # ------------------------------------------------------------------ #
    def get_devices_id(self) -> List[bytes]:
        devs = _RknnDevices()
        if self.lib.rknn3_find_devices(pointer(devs)) != 0:
            # Fall back: rknn3lite's get_devices_id also returns []
            return []
        n = int(devs.n_devices)
        return [bytes(devs.devices[i].id).rstrip(b"\x00") for i in range(n)]

    def get_inputs_tensor_attr(self) -> List[_TensorAttr]:
        if self.rknn_runtime is None or isinstance(self.rknn_runtime, _LLMRuntime):
            return []
        if not self.rknn_runtime.inputs:
            self.rknn_runtime.create_tensor()
        return [t.attr.contents for t in self.rknn_runtime.inputs]

    def get_outputs_tensor_attr(self) -> List[_TensorAttr]:
        if self.rknn_runtime is None or isinstance(self.rknn_runtime, _LLMRuntime):
            return []
        if not self.rknn_runtime.outputs:
            self.rknn_runtime.create_tensor()
        return [t.attr.contents for t in self.rknn_runtime.outputs]

    def rknn3_query(self, cmd: int, index: int = 0) -> Any:
        if self.rknn_runtime is None:
            return None
        return self.rknn_runtime.rknn3_query(int(cmd), index=index)

    # ------------------------------------------------------------------ #
    def create_output_tensors(self, indices: Optional[Sequence[int]] = None) -> Tuple[Any, int]:
        """Pre-allocate output tensors and attach them to the LLM
        callback bundle for the next :meth:`init_llm_session` /
        :meth:`init_runtime` call.

        The TTS talker uses this to read ``past_hidden`` (output 1)
        per decode step. Returns a tuple ``(output_tensors_array, n)``
        where ``output_tensors_array`` is a ctypes array of
        ``_RknnTensor`` matching what rknn3lite returns.
        """
        if not isinstance(self.rknn_runtime, _LLMRuntime):
            return None, 0
        if self.rknn_runtime._created is False:
            return None, 0
        # ensure inputs/outputs arrays are built
        if not self.rknn_runtime.outputs:
            self.rknn_runtime.create_tensor()
        if indices is None:
            indices = list(range(self.rknn_runtime.n_outputs))
        n = len(indices)
        out_tensors = []
        for idx in indices:
            t = _RknnTensor()
            src = self.rknn_runtime._output_tensor_refs[idx]
            t.mem = src.mem
            t.attr = src.attr
            out_tensors.append(t)
        arr = (_RknnTensor * n)(*out_tensors)
        # Stash on the callback bundle so the next init_llm_session
        # wires it up.
        if self._llm_callback is None:
            self._llm_callback = _LLMCallbackBundle()
        self._llm_callback.output_tensors = ctypes_cast(pointer(arr), POINTER(_RknnTensor))
        self._llm_callback.n_output_tensors = c_uint32(n)
        # Patch into the (existing or pending) callback struct.
        self._llm_callback.struct.output_tensors = self._llm_callback.output_tensors
        self._llm_callback.struct.n_output_tensors = c_uint32(n)
        return arr, n

    # ------------------------------------------------------------------ #
    def create_mem(self, size: int, core_id: int, flags: int = 1) -> int:
        """Allocate a ``rknn3_tensor_mem*`` for shared internal memory.

        Used by :class:`wdnai.backends.asr.ASRBackend._setup_shared_internal_mem`
        which mirrors ``init_internal_share`` in
        ``rknn3_model_zoo/examples/Qwen3_ASR/cpp/qwen3_asr.cc``: the
        same memory blocks are passed to both the audio encoder and the
        LLM via :meth:`set_internal_mem`. The returned ``int`` is the
        raw ``rknn3_tensor_mem*`` cast to a Python int.
        """
        if self.rknn_runtime is None:
            raise RuntimeError("runtime not initialised")
        return _runtime_create_mem(self.rknn_runtime, size, core_id, flags)

    def set_internal_mem(self, mem_blocks: Sequence[int]) -> None:
        """Hand pre-allocated ``rknn3_tensor_mem*`` blocks to the runtime
        (the shared-internal-memory trick used by the ASR backend)."""
        if self.rknn_runtime is None:
            raise RuntimeError("runtime not initialised")
        self.rknn_runtime.set_internal_mem(mem_blocks)

    def set_shape(self, shape_id: int) -> None:
        """Switch to a different dynamic shape (re-allocates input/output buffers)."""
        if self.rknn_runtime is None:
            raise RuntimeError("runtime not initialised")
        self.rknn_runtime.set_shape(shape_id)

    def release(self, session_index: Optional[int] = None) -> None:
        if self._released:
            return
        if self.rknn_runtime is not None:
            self.rknn_runtime.release()
            self.rknn_runtime = None
        self._released = True

    # ------------------------------------------------------------------ #
    def __enter__(self) -> "RKNN3Lite":
        return self

    def __exit__(self, *_: Any) -> None:
        self.release()


# Backwards-compat aliases (some legacy wdnai code paths reach into
# ``rknn_runtime.create_mem`` directly — see asr.py).
def _runtime_create_mem(runtime: _Runtime, size: int, core_id: int, flags: int) -> int:
    """Allocate a ``rknn3_tensor_mem*`` via the runtime's lib.

    Used by :class:`wdnai.backends.asr.ASRBackend._setup_shared_internal_mem`
    which mirrors ``init_internal_share`` in
    ``rknn3_model_zoo/examples/Qwen3_ASR/cpp/qwen3_asr.cc``: the
    same ``rknn3_tensor_mem*`` blocks are handed to both the audio
    encoder and the LLM via :meth:`_Runtime.set_internal_mem`.
    """
    ptr = runtime.lib.rknn3_create_mem(
        runtime.context, c_uint64(int(size)), c_int32(int(core_id)), c_int32(int(flags)),
    )
    if not ptr:
        raise RuntimeError(f"rknn3_create_mem failed: size={size} core={core_id}")
    runtime._user_mem_blocks.append(int(ptr))
    return int(ptr)


# Patch ``_Runtime.create_mem`` to also expose the (size, core, flags)
# flavour rknn3lite provides for direct shared-mem use.
_Runtime.create_mem = _runtime_create_mem          # type: ignore[attr-defined]


# =============================================================================
# 10. Public exports
# =============================================================================
__all__ = [
    "RKNN3Lite",
    "RKNN3Image",
    "RKNN3Audio",
    "RKNN3AuxTensorWrapper",
    "Float16",
    "RKNN3TensorType",
    "RKNN3QueryCmd",
    "build_llm_callback",
]
