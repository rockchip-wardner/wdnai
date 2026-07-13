"""Shared logic for all backends: tokenizer / embed / callbacks / streaming result sink."""
from __future__ import annotations

import ctypes
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Set the HF mirror at import time, matching the example scripts
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com/")


# --------------------------------------------------------------------------- #
# Streaming result aggregation
# --------------------------------------------------------------------------- #
@dataclass
class StreamSink:
    """Collects tokens / text / perf data produced by the RKNN streaming callback."""

    tokenizer: Any = None
    tokens: List[int] = field(default_factory=list)
    text: str = ""
    first_token_time: Optional[float] = None
    finished: bool = False
    error: Optional[str] = None
    perf: Dict[str, float] = field(default_factory=dict)

    def reset(self) -> None:
        self.tokens.clear()
        self.text = ""
        self.first_token_time = None
        self.finished = False
        self.error = None
        self.perf.clear()


def _decode_safe(tokenizer, tokens: List[int]) -> str:
    text = tokenizer.decode(
        tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return text.split("�", 1)[0] if "�" in text else text


# State constants (matching the example scripts)
_STATE_ERROR = 5
_STATE_WAITING = 1
_STATE_NORMAL = 0
_STATE_FINISH = (2, 3, 4)
_FINISH_MSG = {2: "Finished", 3: "Stop", 4: "Max new token reached"}


def make_result_callback(sink: StreamSink):
    """Build an LLMResultCallback the C library can invoke."""
    from rknn3lite.api import LLMResultCallback  # type: ignore  # noqa: WPS433

    def _cb(userdata, result_ptr, state):
        if state == _STATE_ERROR:
            sink.error = "inference error"
            return 0
        if state == _STATE_WAITING:
            return 0
        if state in _STATE_FINISH:
            if sink.tokens:
                try:
                    full = _decode_safe(sink.tokenizer, sink.tokens)
                    delta = full[len(sink.text):]
                    if delta:
                        sink.text += delta
                except Exception as e:  # noqa: BLE001
                    sink.error = f"decode error: {e}"
            sink.finished = True
            sink.perf["finish_state"] = state
            return 0
        if state == _STATE_NORMAL:
            n = result_ptr.contents.num_tokens
            new_tokens = [result_ptr.contents.token_ids[i] for i in range(n)]
            sink.tokens.extend(new_tokens)
            if sink.first_token_time is None:
                sink.first_token_time = time.perf_counter()
            try:
                full = _decode_safe(sink.tokenizer, sink.tokens)
                delta = full[len(sink.text):]
                if delta:
                    sink.text += delta
            except Exception:  # noqa: BLE001
                # Trailing UTF-8 truncation; wait for the next chunk
                return 0
        return 0

    return LLMResultCallback(_cb)


def make_tokenizer_callback(tokenizer) -> Tuple[Any, Any]:
    from rknn3lite.api import LLMTokenizerCallback  # type: ignore

    def _cb(userdata, text_ptr, text_len, tokens_ptr, n_tokens_max):
        text = text_ptr.decode("utf-8")
        inputs = tokenizer(text, return_tensors="np", truncation=True)
        tokens = inputs["input_ids"][0][:n_tokens_max]
        n = len(tokens)
        if n <= 0:
            return 0
        for i in range(n):
            tokens_ptr[i] = int(tokens[i])
        return n

    userdata = ctypes.py_object(tokenizer)
    return LLMTokenizerCallback(_cb), ctypes.cast(ctypes.pointer(userdata), ctypes.c_void_p)


def make_embed_callback(embeds_data: np.ndarray) -> Tuple[Any, Any]:
    from rknn3lite.api import LLMGetEmbedCallback  # type: ignore

    def _cb(userdata, tokens_ptr, num_tokens, embed, length):
        embedding_dim = embeds_data.shape[1]
        expected_len = num_tokens * embedding_dim * np.dtype(np.float16).itemsize
        if length != expected_len:
            return -1
        dst = np.ctypeslib.as_array(
            ctypes.cast(embed, ctypes.POINTER(ctypes.c_uint16)),
            shape=(num_tokens * embedding_dim,),
        ).view(np.float16)
        tokens = [tokens_ptr[i] for i in range(num_tokens)]
        dst[:] = embeds_data[tokens].ravel()
        return 0

    userdata = ctypes.py_object(embeds_data)
    return LLMGetEmbedCallback(_cb), ctypes.cast(ctypes.pointer(userdata), ctypes.c_void_p)


# --------------------------------------------------------------------------- #
# Tokenizer / embed loading
# --------------------------------------------------------------------------- #
def load_tokenizer(path: str):
    """Load a tokenizer.

    - ``.gguf`` — parse the GPT-2 BPE vocabulary + merges from RKNN's
      ``*.tokenizer.gguf`` file.
    - Otherwise — load as a HuggingFace model name or directory.
    """
    if path.lower().endswith(".gguf"):
        from ..gguf_tokenizer import load_tokenizer_from_gguf  # noqa: WPS433
        return load_tokenizer_from_gguf(path)
    from transformers import AutoTokenizer  # noqa: WPS433
    return AutoTokenizer.from_pretrained(path)


def load_embed_table(path: str, vocab_size: int) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float16)
    return arr.reshape(vocab_size, -1)


# --------------------------------------------------------------------------- #
# Assemble RKLLMCallback (result + tokenizer + embed)
# --------------------------------------------------------------------------- #
def build_llm_callback(tokenizer, embeds_data: np.ndarray, sink: StreamSink):
    from rknn3lite.api import RKLLMCallback  # type: ignore

    cb = RKLLMCallback()
    cb.result_callback = make_result_callback(sink)
    cb.result_userdata = None
    cb.tokenizer_callback, cb.tokenizer_userdata = make_tokenizer_callback(tokenizer)
    cb.embed_callback, cb.embed_userdata = make_embed_callback(embeds_data)
    return cb


# --------------------------------------------------------------------------- #
# ChatML template — identical across all four example scripts
# --------------------------------------------------------------------------- #
CHATML_SYSTEM = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
CHATML_USER = "<|im_start|>user\n"
CHATML_ASSISTANT = "<|im_end|>\n<|im_start|>assistant\n"


def apply_chat_template(rknn_llm) -> None:
    ret = rknn_llm.set_chat_template(CHATML_SYSTEM, CHATML_USER, CHATML_ASSISTANT)
    if ret != 0:
        raise RuntimeError(f"set_chat_template failed: ret={ret}")


# --------------------------------------------------------------------------- #
# Merge spec.options into the llm_args dict expected by rknn.init_runtime
# --------------------------------------------------------------------------- #
def build_llm_args(spec) -> List[Dict[str, Any]]:
    o = spec.options
    return [{
        "max_new_tokens": o.get("max_new_tokens", 512),
        "top_k": o.get("top_k", 1),
        "top_p": o.get("top_p", 0.9),
        "temperature": o.get("temperature", 0.7),
        "repeat_penalty": o.get("repeat_penalty", 1.0),
        "vocab_size": spec.vocab_size,
        "special_eos_id": spec.special_eos_id,
        "max_context_len": spec.max_context_len,
        "keep_history": o.get("keep_history", 0),
    }]


# --------------------------------------------------------------------------- #
# Perf-table printout (style matches the example scripts)
# --------------------------------------------------------------------------- #
def printf_perf(sink: StreamSink, vision_latency: Optional[float] = None) -> None:
    p = sink.perf
    if not p:
        return
    decode_n = int(p.get("n_decode_tokens", 0))
    prefill_n = int(p.get("n_prefill_tokens", 0))
    prefill_ms = float(p.get("prefill_ms", 0.0))
    decode_ms = float(p.get("decode_ms", 0.0))
    print("\n--------------------------------------------------------------------------------------")
    print(" %-12s  %-15s  %-8s  %-23s  %-23s" %
          ("Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second"))
    print("--------------------------------------------------------------------------------------")
    prefill_tpt = prefill_ms / prefill_n if prefill_n else 0.0
    prefill_tps = (prefill_n * 1000.0) / prefill_ms if prefill_ms else 0.0
    decode_tpt = decode_ms / decode_n if decode_n else 0.0
    decode_tps = (decode_n * 1000.0) / decode_ms if decode_ms else 0.0
    print(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f" %
          ("Prefill", prefill_ms, prefill_n, prefill_tpt, prefill_tps))
    print(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f" %
          ("Generate", decode_ms, decode_n, decode_tpt, decode_tps))
    print("--------------------------------------------------------------------------------------")
    if vision_latency is not None:
        ms = vision_latency * 1000
        fps = 1.0 / vision_latency if vision_latency > 0 else 0.0
        print(f" Vision latency = {ms:.2f} ms, FPS = {fps:.2f}")
