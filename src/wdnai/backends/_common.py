"""Shared logic for all backends: tokenizer / embed / callbacks / streaming result sink."""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .._librknn import build_llm_callback as _build_llm_callback_impl
from ..model import ChatTemplate

# Set the HF mirror at import time, matching the example scripts
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com/")


# --------------------------------------------------------------------------- #
# NPU selection
# --------------------------------------------------------------------------- #
def resolve_device_id(
    rknn_cls,
    npu: Optional[int],
    device_id: Optional[Union[bytes, str]],
):
    """Resolve the ``device_id`` bytes to pass into ``RKNN3Lite.init_runtime``.

    - ``device_id`` already set (bytes or str) → returned normalised to
      bytes (``str.encode("utf-8")`` if needed; bytes passed through).
    - ``npu`` given → call ``rknn_cls().get_devices_id()[npu]`` and
      return the corresponding bytes. Raises ``IndexError`` if out of
      range (the message includes the available list).
    - both ``None`` → returns ``None`` so the underlying call goes to
      ``device_id=NULL`` (rknn3lite picks the default device).

    The enumeration is performed lazily here — callers should invoke
    this **inside** a backend's ``__init__`` (after ``RKNN3Lite`` is
    already imported) so the RKNN3 native driver init does not race
    with ``wdnav`` in the same process.
    """
    if device_id is not None:
        if isinstance(device_id, str):
            return device_id.encode("utf-8")
        return device_id
    if npu is None:
        return None
    ids = rknn_cls().get_devices_id()
    if not (0 <= npu < len(ids)):
        raise IndexError(
            f"npu={npu} out of range; {len(ids)} device(s) found: {ids!r}"
        )
    return ids[npu]


# --------------------------------------------------------------------------- #
# Streaming result aggregation
# --------------------------------------------------------------------------- #
@dataclass
class StreamSink:
    """Collects tokens / text / perf data produced by the RKNN streaming callback.

    Also the fan-out point for incremental streaming. Callers register a
    plain Python callable with :meth:`add_observer` and receive every text
    delta as soon as it is decoded, without patching the SDK.

    **Observers run on the inference thread** (inside the RKNN result
    callback), not on the event-loop thread — so they must stay short and
    never block; hand heavy work off to your own queue. An observer that
    raises is isolated: the exception is swallowed and the remaining
    observers still run, because a broken callback must not abort
    inference.
    """

    tokenizer: Any = None
    tokens: List[int] = field(default_factory=list)
    text: str = ""
    first_token_time: Optional[float] = None
    finished: bool = False
    error: Optional[str] = None
    perf: Dict[str, float] = field(default_factory=dict)
    # Cooperative cancellation, polled by the result callback once per
    # decode step. This stops the *stream*: the pending token is dropped and
    # no further deltas are emitted. It does **not** release the NPU — on
    # this librknn3 a non-zero callback return leaves ``session_run``
    # decoding to ``max_new_tokens`` (measured 21.9 s vs 24.7 s). The NPU
    # side is ``rknn3_session_stop()``, called by ``LLMLocalSession.stop``.
    # Excluded from repr/eq for the same reason as ``_observers``.
    stop_event: threading.Event = field(
        default_factory=threading.Event, repr=False, compare=False
    )
    # Set on the token where the stop was observed, so callers can tell
    # "user stopped" apart from "session_run returned != 0".
    stopped: bool = field(default=False, repr=False, compare=False)
    # Observer registry. Excluded from repr/eq so the dataclass' printed
    # form and equality are unchanged by the streaming support.
    _observers: List[Callable[[str], None]] = field(
        default_factory=list, repr=False, compare=False
    )

    def emit(self, delta: str) -> None:
        """Append one decoded text delta and fan it out to observers.

        Used by backends that decode tokens themselves (the multi-stage
        9B pipeline) instead of going through ``_ResultCallbackAdapter``
        (which owns the same job for single-stage backends).
        """
        if not delta:
            return
        self.text += delta
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
        self._notify(delta)

    def publish(self, delta: str) -> None:
        """Fan ``delta`` out to observers **without** touching :attr:`text`.

        For backends whose observable output is not the model's own token
        stream — the ASR backend publishes one payload per *streaming
        round* (committed text + speculative tail) rather than per token.
        Those payloads must not pollute ``self.text``, which those
        backends use for their own bookkeeping (the ASR backend resets
        the sink per round and reads ``tokens`` back out of it).
        """
        if not delta:
            return
        self._notify(delta)

    def add_observer(self, cb: Callable[[str], None]) -> None:
        """Register ``cb(delta)``. No-op for ``None``; ignores duplicates."""
        if cb is not None and cb not in self._observers:
            self._observers.append(cb)

    def remove_observer(self, cb: Callable[[str], None]) -> None:
        """Unregister ``cb``. Unregistering an unknown callback is a no-op."""
        try:
            self._observers.remove(cb)
        except ValueError:
            pass

    def _notify(self, delta: str) -> None:
        if not delta or not self._observers:
            return
        # Snapshot: an observer is allowed to add/remove observers.
        for cb in list(self._observers):
            try:
                cb(delta)
            except Exception:  # noqa: BLE001 — see class docstring
                pass

    def stop(self) -> None:
        """Request cancellation of the run currently in flight.

        Stops the stream within one token (see :meth:`_ResultCallbackAdapter.__call__`).
        Releasing the NPU is ``LLMLocalSession.stop()``'s job. Safe from any
        thread; a no-op when nothing is running.
        """
        self.stop_event.set()

    def reset(self) -> None:
        self.tokens.clear()
        self.text = ""
        self.first_token_time = None
        self.finished = False
        self.error = None
        self.perf.clear()
        self.stop_event.clear()
        self.stopped = False
        # ``_observers`` deliberately survives reset(): observers are bound
        # to the session lifetime (or to one respond() call registered by
        # LLMLocalSession), not to a single inference run.


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


class _ResultCallbackAdapter:
    """Bridges a :class:`StreamSink` to the shim's result callback contract.

    The shim's :func:`wdnai._librknn.build_llm_callback` trampoline
    auto-pins the bound method's ``__self__`` and invokes the callable
    as ``cb(self, tokens, state)``. This adapter wraps the sink in a
    small object so that the shim can pick up ``self = sink`` correctly
    via ``__self__`` introspection, instead of treating the bare
    closure as the userdata.
    """

    __slots__ = ("sink",)

    def __init__(self, sink: StreamSink) -> None:
        self.sink = sink

    def __call__(self, _userdata, new_tokens, state):
        sink = self.sink
        s = int(state)
        # Stop the stream: drop the pending token and report a non-zero return so
        # the callback stops emitting. Checked on NORMAL only — WAITING /
        # FINISH are bookkeeping callbacks, and aborting from one of those
        # would turn an already-finished run into an error. Note this does
        # not by itself free the NPU; see ``StreamSink.stop_event``.
        if s == _STATE_NORMAL and sink.stop_event.is_set():
            sink.stopped = True
            sink.finished = True
            return 1
        if s == _STATE_ERROR:
            sink.error = "inference error"
            return 0
        if s == _STATE_WAITING:
            return 0
        if s in _STATE_FINISH:
            if sink.tokens:
                try:
                    full = _decode_safe(sink.tokenizer, sink.tokens)
                    delta = full[len(sink.text):]
                    if delta:
                        sink.text += delta
                        sink._notify(delta)
                except Exception as e:  # noqa: BLE001
                    sink.error = f"decode error: {e}"
            sink.finished = True
            sink.perf["finish_state"] = s
            return 0
        if s == _STATE_NORMAL:
            sink.tokens.extend(new_tokens)
            if sink.first_token_time is None:
                sink.first_token_time = time.perf_counter()
            try:
                full = _decode_safe(sink.tokenizer, sink.tokens)
                delta = full[len(sink.text):]
                if delta:
                    sink.text += delta
                    sink._notify(delta)
            except Exception:  # noqa: BLE001 — Trailing UTF-8 chunk
                return 0
        return 0


def make_result_callback(sink: StreamSink):
    """Build the user-side ``result_cb`` callable for
    :func:`build_llm_callback`.

    The returned callable has the shim's signature
    ``(userdata, new_tokens: List[int], state: int) -> int``; the
    stream sink is mutated in place (``.tokens``, ``.text``,
    ``.error``, ``.finished``, ``.perf``) and every text delta is
    fanned out to the observers registered via ``sink.add_observer``.

    Returns a bound method (via :class:`_ResultCallbackAdapter`) so the
    sink can be auto-pinned as C-side userdata; a free closure would be
    passed as the userdata itself and the C trampoline would have no way
    to reach the sink.
    """
    return _ResultCallbackAdapter(sink).__call__


# --------------------------------------------------------------------------- #
# Tokenizer / embed loading
# --------------------------------------------------------------------------- #
def load_tokenizer(path: str, extra_special_tokens=None):
    """Load a tokenizer.

    - ``.gguf`` — parse the GPT-2 BPE vocabulary + merges from RKNN's
      ``*.tokenizer.gguf`` file.
    - Otherwise — load as a HuggingFace model name or directory.

    ``extra_special_tokens`` — optional per-instance extra control
    tokens to register as non-splittable specials (see
    :func:`wdnai.gguf_tokenizer.load_tokenizer_from_gguf`). Only
    meaningful for the GGUF path.
    """
    if path.lower().endswith(".gguf"):
        from ..gguf_tokenizer import load_tokenizer_from_gguf  # noqa: WPS433
        return load_tokenizer_from_gguf(
            path, extra_special_tokens=extra_special_tokens,
        )
    from transformers import AutoTokenizer  # noqa: WPS433
    return AutoTokenizer.from_pretrained(path)


def load_embed_table(path: str, vocab_size: int) -> np.ndarray:
    # Use mmap instead of np.fromfile so the embed table's pages are
    # released to the OS as soon as the last reference is dropped —
    # np.fromfile mallocs the buffer into the glibc arena which only
    # ever grows and is not returned to the OS until process exit,
    # causing worker RSS to accumulate ~600 MB per load/unload cycle.
    # All 6 callers (Text/VLM/Omni/Embedding/LocateAnything/ASR) are
    # read-only on the embed table (asr.py's fancy indexing returns
    # copies), so ``mode='r'`` is safe.
    arr = np.memmap(path, dtype=np.float16, mode='r')
    return arr.reshape(vocab_size, -1)


def raise_for_session_run(ret: int, sink: StreamSink, what: str) -> None:
    """Turn a non-zero ``session_run`` into an error — unless we stopped it.

    A stopped run returns early with a non-clean status; keying off
    ``sink.stopped`` (set by the callback, not the caller) keeps "user
    pressed stop" from being reported as an inference failure.
    """
    if ret == 0 or sink.stopped:
        return
    raise RuntimeError(f"{what} session_run failed: ret={ret}")


# --------------------------------------------------------------------------- #
# Assemble the LLM callback bundle consumed by ``RKNN3Lite.init_runtime``.
#
# Returns an opaque ``_LLMCallbackBundle`` from :mod:`wdnai._librknn`;
# the bundle's C-side trampoline reads each token from the C
# ``RKLLMResult`` and writes into ``sink``.
# --------------------------------------------------------------------------- #
def build_llm_callback(
    tokenizer,
    embeds_data: np.ndarray,
    sink: Optional[StreamSink] = None,
    *,
    result_cb: Optional[Callable[..., int]] = None,
    output_cb: Optional[Callable[..., int]] = None,
    output_tensors: Optional[Any] = None,
):
    """Build the LLM callback bundle for ``init_runtime(llm_callback=...)``.

    Kept as a function for source compatibility with the previous
    rknn3lite version; the new shim handles ctypes userdata
    marshalling internally.

    ``result_cb`` — a custom result callback. Defaults to the
    :class:`StreamSink` adapter built from ``sink``; the C library
    **refuses to run without one** (``no result_callback provided!``),
    so a backend that does not consume a token stream (the
    Intern-Decision decision model) must pass a no-op of its own.

    ``output_cb`` / ``output_tensors`` — optional **output** callback
    wiring. A backend that reads model *tensors* instead of a token
    stream (the Intern-Decision decision model reads its 62 candidate
    logits) passes a bound ``output_cb(userdata, outputs, state)`` plus
    the pre-allocated tensor array from
    :meth:`wdnai._librknn.RKNN3Lite.create_output_tensors`.

    ``sink`` may be ``None`` when the caller does not consume a token
    stream (the decision backend does not).
    """
    if result_cb is None and sink is not None:
        result_cb = make_result_callback(sink)
    return _build_llm_callback_impl(
        result_cb=result_cb,
        tokenizer=tokenizer,
        embeds_data=embeds_data,
        output_cb=output_cb,
        output_tensors=output_tensors,
    )


# --------------------------------------------------------------------------- #
# ChatML template — identical across all four example scripts
# --------------------------------------------------------------------------- #
# Bare (marker-free) turn prefixes. This is the form actually handed to
# ``session_run``; RKNN's C-side chat template — installed by
# ``apply_chat_template`` below — wraps each turn in the ``<|im_*|>``
# markers. ``LLMSession._compose_prompt`` builds multi-turn prompts from
# these, so they are the single source of truth for both the bare and the
# marked form.
CHATML_TURN_SYSTEM = "system\n"
CHATML_TURN_USER = "user\n"
CHATML_TURN_ASSISTANT = "assistant\n"

_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"

# Marked form, expressed in terms of the bare prefixes above so the two
# cannot drift apart. Byte-identical to the literals these replaced.
CHATML_SYSTEM = f"{_IM_START}{CHATML_TURN_SYSTEM}You are a helpful assistant.{_IM_END}\n"
CHATML_USER = f"{_IM_START}{CHATML_TURN_USER}"
CHATML_ASSISTANT = f"{_IM_END}\n{_IM_START}{CHATML_TURN_ASSISTANT}"

#: The ChatML triple as a :class:`~wdnai.model.ChatTemplate`. This is the
#: **default** for every backend that does not override it.
CHATML_TEMPLATE = ChatTemplate(
    system=CHATML_SYSTEM, prefix=CHATML_USER, postfix=CHATML_ASSISTANT,
)


def apply_chat_template(rknn_llm, template: Optional[ChatTemplate] = None) -> None:
    """Install the C-side chat template on ``rknn_llm``.

    ``template`` is a :class:`~wdnai.model.ChatTemplate`; ``None`` selects
    the ChatML default (every Qwen-family preset relies on it).

    The C side composes ``system + prefix + <prompt> + postfix``, so a
    model family with its own control tokens (HY-MT's Hunyuan markers)
    only has to supply a different triple here — the **content** of the
    prompt is never touched by this function.

    A backend that assembles the whole transcript itself (the 9B
    multi-card pipeline) must **not** call this: it clears the C-side
    template instead (``set_chat_template("", "", "")``) so the prompt is
    not wrapped twice.
    """
    tpl = template if template is not None else CHATML_TEMPLATE
    ret = rknn_llm.set_chat_template(tpl.system, tpl.prefix, tpl.postfix)
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
        # Optional frequency / presence penalty. Default 0.0 (= disabled),
        # matches the C++ ``rknn3_sampling_params`` zero-init. Qwen3.5-VL
        # tunes ``presence_penalty=1.5`` (see examples/Qwen3_5_VL/cpp/main.cc)
        # to discourage repetition on long multimodal generations; the
        # rest of the presets keep the default of 0.0.
        "frequency_penalty": o.get("frequency_penalty", 0.0),
        "presence_penalty":  o.get("presence_penalty", 0.0),
    }]


# --------------------------------------------------------------------------- #
# Perf table
# --------------------------------------------------------------------------- #
#
# ⚠️ **为什么要把它从 printf_perf 里抠出来**（2026-11-17）：
#
# 以前这些派生量（tokens/s、每 token 耗时、LLM 延迟）**只在 printf_perf
# 里算完就打印掉了** —— 调用方拿不到，只能去解析 stdout（而 stdout 在
# **worker 进程**里，Web UI 那边根本看不见）。
#
# ⇒ 板端 Web UI 要显示「这次翻译花了多久、多少 tokens/s」时，只能自己
# **重新算一遍** —— 于是同一套公式散落在两个地方（SDK 打印一份、宿主
# 算一份），**迟早会漂**（改了 SDK 的算法，界面上的数就悄悄对不上了）。
#
# ⇒ 这里把它提成**纯函数** ``perf_table(perf)``：
#   · :func:`printf_perf` 复用它（打印的和返回的**同一份算法**）；
#   · 调用方（宿主翻译 provider）也复用它（拿累加后的原始计数算汇总值）。
#
# 公式**一个字都没改** —— 还是 SDK 原来那套（见 printf_perf 的历史）。
def perf_table(perf: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """把 perf 表里的原始计数算成一份**派生数字字典**。

    输入是 ``sink.perf`` 那种原始计数：
    ``n_decode_tokens`` / ``n_prefill_tokens`` / ``prefill_ms`` / ``decode_ms``。

    输出（**全是数字**，可以直接进 JSON / 过 wire）：

    ==========================  ==========================================
    ``prefill_tokens``          预填充 token 数
    ``decode_tokens``           解码（生成）token 数
    ``prefill_ms`` / ``decode_ms``  两段的耗时（毫秒）
    ``prefill_ms_per_token``    预填充的每 token 耗时
    ``decode_ms_per_token``     解码的每 token 耗时
    ``prefill_tokens_per_second``  预填充的吞吐（tokens/s）
    ``decode_tokens_per_second``   **解码的吞吐（tokens/s）** ← 界面显示的
    ``llm_latency_ms``          LLM 总延迟 = 预填充 + 解码
    ==========================  ==========================================

    ⚠️ **计数为 0 时对应的速率也是 0**（不是 inf/nan）—— 这些数要进 JSON
    响应，``inf`` / ``NaN`` 编不过去 ⇒ 整个响应序列化失败 ⇒ **推理成功了
    但前端收到 500**。
    """
    p = perf if isinstance(perf, dict) else {}
    decode_n = int(p.get("n_decode_tokens", 0) or 0)
    prefill_n = int(p.get("n_prefill_tokens", 0) or 0)
    prefill_ms = float(p.get("prefill_ms", 0.0) or 0.0)
    decode_ms = float(p.get("decode_ms", 0.0) or 0.0)
    return {
        "prefill_tokens": prefill_n,
        "decode_tokens": decode_n,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "prefill_ms_per_token": prefill_ms / prefill_n if prefill_n else 0.0,
        "decode_ms_per_token": decode_ms / decode_n if decode_n else 0.0,
        "prefill_tokens_per_second": (
            (prefill_n * 1000.0) / prefill_ms if prefill_ms else 0.0),
        "decode_tokens_per_second": (
            (decode_n * 1000.0) / decode_ms if decode_ms else 0.0),
        "llm_latency_ms": prefill_ms + decode_ms,
    }


def perf_json(perf: Optional[Dict[str, Any]]) -> str:
    """把 perf 表算成**一段 JSON 文本**（可以直接贴进日志 / 响应体）。

    ⚠️ **为什么要有「文本」这一版**：perf 表最终要**过 wire**（DDS reply
    → HTTP 响应 → 前端）。中间任何一环只要带了编不过去的值（``inf`` /
    ``NaN`` / 非数字对象），**整条响应就序列化失败** ⇒ 推理明明成功了，
    调用方收到的却是传输错误 —— 最难查的一种失败。

    ⇒ 所以这里**先序列化一次**：编不过去就地降级（``NaN``/``inf`` → 0），
    保证产出的**一定是一段合法 JSON**。

    用法::

        print(perf_json(sink.perf))     # 日志里贴一行 JSON
        reply["perf_json"] = perf_json(sink.perf)   # 或者直接带过 wire
    """
    table = perf_table(perf)
    # ⚠️ ``allow_nan=False``：``NaN``/``inf`` **不是合法 JSON**（虽然
    # Python 的 json 默认会把它们写成 NaN/Infinity，但别的解析器不认）。
    # 这里先自己清一遍，再让 json 严格校验。
    clean = {}
    for key, val in table.items():
        try:
            num = float(val)
        except (TypeError, ValueError):
            continue
        if num != num or num in (float("inf"), float("-inf")):
            num = 0.0
        clean[str(key)] = num
    try:
        return json.dumps(clean, ensure_ascii=False, sort_keys=True,
                          allow_nan=False)
    except (TypeError, ValueError):
        # 兜底：连 ``clean`` 都编不过去（理论上不会）⇒ 给一份空的
        return "{}"


def printf_perf(sink: StreamSink, vision_latency: Optional[float] = None) -> None:
    p = sink.perf
    if not p:
        return
    t = perf_table(p)
    decode_n = t["decode_tokens"]
    prefill_n = t["prefill_tokens"]
    prefill_ms = t["prefill_ms"]
    decode_ms = t["decode_ms"]
    print("\n--------------------------------------------------------------------------------------")
    print(" %-12s  %-15s  %-8s  %-23s  %-23s" %
          ("Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second"))
    print("--------------------------------------------------------------------------------------")
    prefill_tpt = t["prefill_ms_per_token"]
    prefill_tps = t["prefill_tokens_per_second"]
    decode_tpt = t["decode_ms_per_token"]
    decode_tps = t["decode_tokens_per_second"]
    print(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f" %
          ("Prefill", prefill_ms, prefill_n, prefill_tpt, prefill_tps))
    print(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f" %
          ("Generate", decode_ms, decode_n, decode_tpt, decode_tps))
    print("--------------------------------------------------------------------------------------")
    # Combined LLM-only latency (prefill + decode). Excludes vision
    # preprocessing so it stays comparable across text-only and VLM
    # backends.
    llm_ms = t["llm_latency_ms"]
    print(f" LLM latency = {llm_ms:.2f} ms")
    if vision_latency is not None:
        # ``vision_latency`` measures only vision preprocessing +
        # vision inference — the LLM portion is reported above. The
        # end-to-end FPS combines both and reflects the real
        # per-frame throughput.
        vision_ms = vision_latency * 1000.0
        print(f" Vision latency = {vision_ms:.2f} ms")
        total_sec = vision_latency + llm_ms / 1000.0
        if total_sec > 0:
            print(f" Total latency = {total_sec * 1000.0:.2f} ms, FPS = {1.0 / total_sec:.2f}")
    else:
        total_sec = llm_ms / 1000.0
        if total_sec > 0:
            print(f" Total latency = {total_sec * 1000.0:.2f} ms, FPS = {1.0 / total_sec:.2f}")
