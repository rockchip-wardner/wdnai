"""增量流式会话（``asr_stream_id``）的契约测试。

分两档：

* **契约档**（默认，无需 NPU）—— 用「裸实例 + 替换单块执行器」把
  :meth:`ASRBackend._run_online_stream` 的状态机、块切分、时间轴、
  TTL 回收钉死；再单独验 :class:`_StreamResampler` 与整段
  :func:`_resample_linear` **逐样本等价**（这是「增量 ≡ 文件」的前提）。
* **实机档**（``WDNAI_ASR_NPU=1``）—— 同一段音频走增量路径与走文件
  路径，比较最终转写文本。

为什么要有「裸实例」这种写法：``ASRBackend.__init__`` 会加载 RKNN 模型、
占 NPU、申请显存。契约档要测的只是**编排逻辑**（切块 / 状态 / 时间轴），
不该为此碰硬件。于是用 ``object.__new__`` 建一个只填了必要属性的实例，
再把 ``_run_online_block`` 换成记录器 —— 被测的那段逻辑原样保留，
需要硬件的那段被摘掉。

⚠️ **为什么是惰性导入**：``wdnai.backends._common`` 在 import 时就加载
RKNN 原生库。模块级 ``import wdnai.backends.asr`` 会让它在 pytest 的
**收集期** 加载 —— 改变了原生库的加载顺序，会让整个套件在 teardown
时段错误（一个只做 import 的六行探针即可复现，与 ASR 后端本身无关的
既有脆弱点）。把 import 放进测试体里就恢复了原来的顺序。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# Lazy access to the ASR backend module — see the note in the docstring.
# --------------------------------------------------------------------------- #
_ASR = None


def _asr():
    global _ASR
    if _ASR is None:
        import wdnai.backends.asr as mod
        _ASR = mod
    return _ASR


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _bare_backend(ttl: float = 120.0):
    """An ``ASRBackend`` with just enough state for the session logic."""
    from wdnai.backends._common import StreamSink

    class _Tok:
        def decode(self, tokens, skip_special_tokens=True):
            return "".join(chr(int(t)) for t in tokens)

    backend = object.__new__(_asr().ASRBackend)
    backend.tokenizer = _Tok()
    backend.sink = StreamSink(tokenizer=backend.tokenizer)
    backend._online_streams = {}
    backend._stream_ttl_sec = ttl
    return backend


class _BlockRecorder:
    """Stand-in for ``_run_online_block`` — records what it was asked to do."""

    def __init__(self):
        self.calls = []   # (chunk_idx, n_samples, is_last, chunk_start_sec)

    def __call__(self, state, chunk_audio, *, is_last, chunk_start_sec,
                 on_asr_commit):
        self.calls.append((
            state.chunk_idx, int(len(chunk_audio)), bool(is_last),
            float(chunk_start_sec),
        ))
        # Emulate what the real routine does to the session state.
        state.chunk_idx += 1
        state.samples_consumed += int(len(chunk_audio))


def _feed(backend, stream_id, n_samples, *, final=False, origin=None):
    """One ``_run_online_stream`` call with silence."""
    overrides = {}
    if origin is not None:
        overrides[_asr().ASR_STREAM_ORIGIN_SEC_KEY] = origin
    return backend._run_online_stream(
        np.zeros(n_samples, dtype=np.float32),
        _asr().TARGET_SAMPLE_RATE,
        stream_id, final, None, overrides,
    )


def _pcm(samples: int) -> bytes:
    """16 kHz mono s16le silence."""
    return np.zeros(samples, dtype=np.int16).tobytes()


# --------------------------------------------------------------------------- #
# 1) Streaming resampler — must match the offline resample sample for sample
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("src_rate", [48000, 32000, 24000])
def test_stream_resampler_matches_offline(src_rate):
    """Pushing the same audio in blocks must yield exactly the offline result.

    The offline path resamples the whole buffer, so output sample ``j`` is
    interpolated from the **global** input index ``j * src/dst``. A
    block-local resampler would restart that index at 0 and shift the phase
    by one block per block — and the ASR model is sensitive enough at the
    margins for that to flip a token.
    """
    m = _asr()
    n_src = src_rate * 3          # 3 seconds
    rng = np.random.default_rng(7)
    audio = (rng.standard_normal(n_src) * 0.1).astype(np.float32)

    offline = m._resample_linear(audio, src_rate, m.TARGET_SAMPLE_RATE)

    resampler = m._StreamResampler(src_rate, m.TARGET_SAMPLE_RATE)
    out = []
    for i in range(0, n_src, src_rate):     # one second per block
        out.append(resampler.push(audio[i:i + src_rate]))
    out.append(resampler.flush())
    streamed = np.concatenate(out) if out else np.zeros((0,), np.float32)

    assert streamed.shape == offline.shape, (
        "length mismatch: streamed {} vs offline {}".format(
            streamed.shape, offline.shape))
    assert np.array_equal(streamed, offline), (
        "streaming resample diverged from the offline one")


@pytest.mark.parametrize("block", [16000, 24000, 48000, 7000])
def test_stream_resampler_is_independent_of_the_block_size(block):
    """The global index makes the result independent of how it was chunked.

    The ASR backend buffers the output on the 16 kHz grid anyway, so the
    caller may hand over the source in whatever pieces are convenient.
    """
    m = _asr()
    src_rate = 48000
    rng = np.random.default_rng(13)
    audio = (rng.standard_normal(src_rate * 2 + 1234) * 0.1).astype(np.float32)
    offline = m._resample_linear(audio, src_rate, m.TARGET_SAMPLE_RATE)

    resampler = m._StreamResampler(src_rate, m.TARGET_SAMPLE_RATE)
    out = []
    for i in range(0, audio.size, block):
        out.append(resampler.push(audio[i:i + block]))
    out.append(resampler.flush())
    streamed = np.concatenate(out)

    assert streamed.shape == offline.shape
    assert np.array_equal(streamed, offline)


def test_stream_resampler_flush_reproduces_last_sample():
    """The offline path extrapolates its last sample; flush must too."""
    m = _asr()
    src_rate = 48000
    for n in (48000, 48001, 48002, 48003):
        rng = np.random.default_rng(3)
        audio = (rng.standard_normal(n) * 0.1).astype(np.float32)
        offline = m._resample_linear(audio, src_rate, m.TARGET_SAMPLE_RATE)

        resampler = m._StreamResampler(src_rate, m.TARGET_SAMPLE_RATE)
        out = np.concatenate([resampler.push(audio), resampler.flush()])
        assert out.shape == offline.shape, "n={}".format(n)
        assert np.array_equal(out, offline), "n={}".format(n)


def test_stream_resampler_rejects_upsampling():
    with pytest.raises(ValueError):
        _asr()._StreamResampler(16000, 48000)


# --------------------------------------------------------------------------- #
# 2) Round payload — the new fields are purely additive
# --------------------------------------------------------------------------- #
def test_round_payload_is_additive():
    """Old keys keep their meaning; the new ones are extra.

    A consumer that predates the timestamp fields must keep working, so
    the payload may only **grow**.
    """
    payload = json.loads(_asr()._round_payload(
        round=7, committed="hello", unfixed=" world", language="English",
        chunk_start_sec=6.0, chunk_samples=16000,
    ))
    assert payload["type"] == "asr_round"
    assert payload["round"] == 7
    assert payload["committed"] == "hello"
    assert payload["unfixed"] == " world"
    assert payload["language"] == "English"
    assert payload["chunk_start_sec"] == pytest.approx(6.0)
    assert payload["chunk_samples"] == 16000

    # Defaults keep the old shape meaningful.
    legacy = json.loads(_asr()._round_payload(
        round=1, committed="", unfixed="", language=""))
    assert legacy["chunk_start_sec"] == 0.0
    assert legacy["chunk_samples"] == 0


# --------------------------------------------------------------------------- #
# 3) Session bookkeeping
# --------------------------------------------------------------------------- #
def test_stream_state_defaults():
    st = _asr()._OnlineStreamState(
        stream_id="s", samples_per_chunk=16000, max_new_tokens=50,
        rollback_tokens=5, rollback_rounds=4, chunk_seconds=1.0,
    )
    assert st.chunk_idx == 0
    assert st.samples_consumed == 0
    assert st.origin_sec == 0.0
    assert st.audio_embed_history == []
    assert st.committed_token_ids == []
    assert st.pending.size == 0
    assert st.resampler is None
    assert st.chunk_start_sec == 0.0


def test_incremental_blocks_and_global_chunk_index():
    """Blocks are cut on the 16 kHz grid and the index is global."""
    backend = _bare_backend()
    recorder = _BlockRecorder()
    backend._run_online_block = recorder

    for _ in range(3):
        _feed(backend, "s1", 16000)

    assert len(recorder.calls) == 3
    assert [c[0] for c in recorder.calls] == [0, 1, 2]      # global index
    assert all(c[1] == 16000 for c in recorder.calls)
    assert all(c[2] is False for c in recorder.calls)      # not the last one


def test_partial_audio_is_buffered_until_a_whole_block():
    """The caller may hand over any amount; the backend buffers the rest."""
    backend = _bare_backend()
    recorder = _BlockRecorder()
    backend._run_online_block = recorder

    _feed(backend, "s1", 5000)
    assert recorder.calls == []          # not enough for a block yet
    assert backend._online_streams["s1"].pending.size == 5000

    _feed(backend, "s1", 11000)         # 5000 + 11000 = 16000
    assert len(recorder.calls) == 1
    assert recorder.calls[0][1] == 16000


def test_final_call_marks_the_last_block():
    """Only the session's last block is exempt from the rollback."""
    backend = _bare_backend()
    recorder = _BlockRecorder()
    backend._run_online_block = recorder

    _feed(backend, "s1", 16000)
    _feed(backend, "s1", 16000, final=True)

    assert len(recorder.calls) == 2
    assert recorder.calls[0][2] is False
    assert recorder.calls[1][2] is True     # ← the last block


def test_final_call_with_a_partial_block():
    """A short final block is the session's last block too."""
    backend = _bare_backend()
    recorder = _BlockRecorder()
    backend._run_online_block = recorder

    _feed(backend, "s1", 5000, final=True)
    assert len(recorder.calls) == 1
    assert recorder.calls[0][1] == 5000
    assert recorder.calls[0][2] is True


def test_final_with_a_whole_block_and_a_tail_marks_only_the_tail():
    """Regression: the final call may carry a whole block *and* a tail.

    Only the **last executed** block is the session's last one — the
    tail.  Marking the whole block as "last" as well would skip its
    rollback, and the transcript would then differ from the file-driven
    path (which splits the same audio with the short block last).

    The real-NPU test caught this: the incremental path confirmed five
    extra tokens and the transcript grew a spurious trailing sentence.
    """
    backend = _bare_backend()
    recorder = _BlockRecorder()
    backend._run_online_block = recorder

    _feed(backend, "s1", 16000)
    _feed(backend, "s1", 16000 + 500, final=True)

    assert len(recorder.calls) == 3
    assert recorder.calls[1][2] is False      # the whole block is not last
    assert recorder.calls[2][2] is True       # the tail is
    assert recorder.calls[2][1] == 500


def test_final_removes_the_session():
    backend = _bare_backend()
    backend._run_online_block = _BlockRecorder()

    _feed(backend, "s1", 16000)
    assert "s1" in backend._online_streams

    _feed(backend, "s1", 16000, final=True)
    assert "s1" not in backend._online_streams


def test_same_id_after_final_starts_a_new_session():
    """Re-using a finished id must not resurrect the old state."""
    backend = _bare_backend()
    recorder = _BlockRecorder()
    backend._run_online_block = recorder

    _feed(backend, "s1", 16000, final=True)
    assert len(recorder.calls) == 1
    assert recorder.calls[0][0] == 0        # first block of a *new* session

    _feed(backend, "s1", 16000)
    assert len(recorder.calls) == 2
    assert recorder.calls[1][0] == 0        # index restarts at 0


def test_idle_sessions_are_reaped():
    """A caller that stops feeding must not leak the session forever."""
    backend = _bare_backend(ttl=0.0)
    backend._run_online_block = _BlockRecorder()

    _feed(backend, "s1", 16000)
    assert "s1" in backend._online_streams

    # The next call reaps it (TTL 0 ⇒ always idle) and starts fresh.
    _feed(backend, "s1", 16000)
    assert backend._online_streams["s1"].chunk_idx == 1


def test_reset_stream_drops_the_session():
    backend = _bare_backend()
    backend._run_online_block = _BlockRecorder()
    _feed(backend, "s1", 16000)
    assert "s1" in backend._online_streams
    backend.reset_stream("s1")
    assert "s1" not in backend._online_streams


def test_16k_input_does_not_create_a_resampler():
    """At 16 kHz there is nothing to resample — and no phase to lose."""
    backend = _bare_backend()
    backend._run_online_block = _BlockRecorder()
    _feed(backend, "s1", 16000)
    assert backend._online_streams["s1"].resampler is None


def test_non_16k_input_uses_the_streaming_resampler():
    m = _asr()
    backend = _bare_backend()
    backend._run_online_block = _BlockRecorder()
    backend._run_online_stream(
        np.zeros(48000, dtype=np.float32), 48000, "s1", False, None, {},
    )
    rs = backend._online_streams["s1"].resampler
    assert rs is not None
    assert rs.src_rate == 48000 and rs.dst_rate == m.TARGET_SAMPLE_RATE


def test_final_without_new_audio_warns(caplog):
    """The final call must carry the last block — say so if it doesn't."""
    backend = _bare_backend()
    backend._run_online_block = _BlockRecorder()

    with caplog.at_level("WARNING", logger="wdnai.backends.asr"):
        _feed(backend, "s1", 16000)
        _feed(backend, "s1", 0, final=True)
    assert any("final call carried no audio" in r.message
               for r in caplog.records)


# --------------------------------------------------------------------------- #
# 4) Timeline
# --------------------------------------------------------------------------- #
def test_chunk_start_seconds_follow_the_session_origin():
    """The origin is set once; every block is derived from it."""
    backend = _bare_backend()
    recorder = _BlockRecorder()
    backend._run_online_block = recorder

    _feed(backend, "s1", 16000, origin=12.5)
    _feed(backend, "s1", 16000)
    assert [c[3] for c in recorder.calls] == [12.5, 13.5]


def test_multiple_blocks_in_one_call_step_the_timeline():
    backend = _bare_backend()
    recorder = _BlockRecorder()
    backend._run_online_block = recorder

    _feed(backend, "s1", 32000)
    assert [c[3] for c in recorder.calls] == [0.0, 1.0]


def test_timestamps_are_strictly_increasing():
    """D1 — the timeline must not stall or go backwards."""
    backend = _bare_backend()
    recorder = _BlockRecorder()
    backend._run_online_block = recorder

    for _ in range(20):
        _feed(backend, "s1", 16000)
    starts = [c[3] for c in recorder.calls]
    assert starts == sorted(starts)
    assert all(b - a == pytest.approx(1.0) for a, b in zip(starts, starts[1:]))


def test_origin_is_frozen_for_the_session():
    """A later origin change is ignored (seeking = a new session)."""
    backend = _bare_backend()
    recorder = _BlockRecorder()
    backend._run_online_block = recorder

    _feed(backend, "s1", 16000, origin=0.0)
    _feed(backend, "s1", 16000, origin=99.0)
    assert [c[3] for c in recorder.calls] == [0.0, 1.0]


def test_last_block_reports_its_real_length():
    """D2 — a short final block reports the samples it actually carried."""
    backend = _bare_backend()
    recorder = _BlockRecorder()
    backend._run_online_block = recorder

    _feed(backend, "s1", 16000)
    _feed(backend, "s1", 4321, final=True)
    assert recorder.calls[-1][1] == 4321


# --------------------------------------------------------------------------- #
# 5) Option keys
# --------------------------------------------------------------------------- #
def test_option_keys_are_the_documented_ones():
    m = _asr()
    assert m.ASR_STREAM_ID_KEY == "asr_stream_id"
    assert m.ASR_STREAM_FINAL_KEY == "asr_stream_final"
    assert m.ASR_STREAM_ORIGIN_SEC_KEY == "asr_stream_origin_sec"
    assert m.ASR_STREAM_TTL_SEC_DEFAULT == 120.0


def test_run_audio_dispatches_to_the_incremental_path():
    """``asr_stream_id`` in the overrides must select the streaming path."""
    m = _asr()
    backend = _bare_backend()
    seen = {}

    def _fake_stream(audio, sample_rate, stream_id, final,
                     on_asr_commit, overrides):
        seen.update(dict(stream_id=stream_id, final=final,
                         sample_rate=sample_rate,
                         origin=overrides.get(m.ASR_STREAM_ORIGIN_SEC_KEY)))
        return "language English\nok"

    backend._run_online_stream = _fake_stream
    out = backend.run_audio(
        _pcm(16000),
        sample_rate=m.TARGET_SAMPLE_RATE,
        asr_stream_id="live-1",
        asr_stream_final=True,
        asr_stream_origin_sec=3.0,
    )
    assert out == "language English\nok"
    assert seen["stream_id"] == "live-1"
    assert seen["final"] is True
    assert seen["origin"] == pytest.approx(3.0)


def test_run_audio_without_stream_id_keeps_the_offline_path():
    """A2 — no ``asr_stream_id`` ⇒ the legacy path, untouched."""
    backend = _bare_backend()
    backend._use_online = False
    called = {}

    def _fake_offline(audio, overrides):
        called["n"] = int(len(audio))
        return "language English\nfile"

    backend._run_offline = _fake_offline
    out = backend.run_audio(_pcm(16000), sample_rate=_asr().TARGET_SAMPLE_RATE)
    assert out == "language English\nfile"
    assert called["n"] == 16000


# --------------------------------------------------------------------------- #
# 6) Real hardware — incremental N blocks == file N blocks
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not os.environ.get("WDNAI_ASR_NPU"),
    reason="needs a real NPU + the Qwen3-ASR model folder",
)
def test_incremental_matches_file_on_real_npu():
    """A1/A8 — the same audio, fed incrementally, must yield the same text.

    The incremental path is only worth having if it does **not** change
    the transcript. This is the test that pins that.

    The source PCM is fed at its **native** rate (48 kHz), so the
    streaming resampler is exercised too — that is the part most likely
    to drift from the offline path.
    """
    import wdnai
    from wdnai import SystemLanguageModel

    wav_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "examples", "Qwen3_ASR", "asr_en.wav",
    )
    if not os.path.exists(wav_path):
        pytest.skip("asr_en.wav not present")

    import wave
    with wave.open(wav_path, "rb") as w:
        sr = w.getframerate()
        width = w.getsampwidth()
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())

    spec = SystemLanguageModel.qwen3_asr()
    npu_idx = os.environ.get("WDNAI_ASR_NPU_INDEX")
    sess = wdnai.LLMSession(
        model=spec,
        **({"npu": int(npu_idx)} if npu_idx else {}),
    )
    try:
        backend = sess._backend

        # ---- file path ----
        file_text = backend.run_audio(
            raw, sample_rate=sr, n_channels=ch, sample_width=width)

        # ---- incremental path: native-rate blocks, last one is final ----
        # Feed the **raw PCM bytes** (not a converted float array): the
        # backend decodes the payload according to ``sample_width``.
        frame_bytes = max(1, ch * width)
        total_frames = len(raw) // frame_bytes
        src_block = sr          # one second of source per call
        n_full = int(total_frames // src_block)
        if n_full < 2:
            pytest.skip("audio too short for a multi-block comparison")

        for i in range(n_full - 1):
            blk = raw[i * src_block * frame_bytes:(i + 1) * src_block * frame_bytes]
            backend.run_audio(
                blk, sample_rate=sr, n_channels=ch, sample_width=width,
                asr_stream_id="live-test",
            )
        tail = raw[(n_full - 1) * src_block * frame_bytes:]
        inc_text = backend.run_audio(
            tail, sample_rate=sr, n_channels=ch, sample_width=width,
            asr_stream_id="live-test",
            asr_stream_final=True,
        )

        assert inc_text == file_text, (
            "incremental transcript differs:\n  file: {!r}\n  incr: {!r}"
            .format(file_text, inc_text))
    finally:
        sess.close()
