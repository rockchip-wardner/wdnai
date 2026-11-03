import ctypes

import numpy as np

import wdnai.backends._qwen35_vl_9b_vendor as V


def _imports():
    return V


class _FakeBundle:
    def __init__(self, **kwargs):
        self.result_cb = kwargs.get("result_cb")
        self.tokenizer = kwargs.get("tokenizer")
        self.embed_callback = kwargs.get("embed_callback")
        self.output_cb = kwargs.get("output_cb")
        self.output_tensors = kwargs.get("output_tensors")


class _View:
    def __init__(self, name, dtype, n_elems, data):
        self.name = name
        self.dtype = dtype
        self.n_elems = n_elems
        self.data = data


class _FakeRknn:
    def __init__(self, stage_index, dim, tokens):
        self.stage_index = stage_index
        self.dim = dim
        self.tokens = tokens
        self.bundle = None
        self.calls = []
        self.shared = []
        self.destroyed = []
        self.session_runs = 0

    def load_rknn(self, model_path, weight_path):
        self.calls.append(("load", model_path))
        return 0

    def init_runtime(self, target, core_mask, device_id=None, **kwargs):
        self.calls.append(("init", device_id))
        return 0

    def rknn3_query(self, cmd, index=0):
        V = _imports()
        if int(cmd) == int(V.RKNN3QueryCmd.CORE_NUMBER):
            return 3
        if int(cmd) == int(V.RKNN3QueryCmd.CORE_MEM_SIZE):
            class _M:
                def __init__(self, v):
                    self.internal_size = v
            return [_M(100), _M(300), _M(200)]
        if int(cmd) == int(V.RKNN3QueryCmd.IN_OUT_NUM):
            class _IO:
                n_input = 2
                n_output = 1
            return _IO()
        raise AssertionError(f"unexpected query {cmd}")

    def create_output_tensors(self):
        return ["tensor"], 1

    def set_internal_mem(self, mems):
        self.shared = list(mems)

    def create_mem(self, size, core_id, flags=1):
        assert size == 300
        return 10000 + core_id

    def destroy_user_mem(self, mem):
        self.destroyed.append(mem)

    def init_llm_session(self, llm_args, llm_callback, session_index=0):
        self.bundle = llm_callback
        self.llm_args = llm_args
        return 0

    def set_chat_template(self, *args):
        return 0

    def clear_kvcache(self, *args):
        return 0

    def session_run(self, **kwargs):
        self.session_runs += 1
        if self.stage_index == 0:
            return self._run_stage0(**kwargs)
        return self._run_stage1(**kwargs)

    def _run_stage0(self, prompt=None, tokens=None, inputs=None, **kwargs):
        assert self.bundle is not None
        if prompt is not None:
            ids = self.bundle.tokenizer(prompt)["input_ids"][0].tolist()
        else:
            ids = list(tokens)
        n = len(ids)
        buf = (ctypes.c_uint16 * (n * self.dim))()
        rc = self.bundle.embed_callback(
            None, ids, ctypes.addressof(buf), n * self.dim * 2,
        )
        assert rc == 0
        view = _View("output", 1, n * self.dim, np.arange(n * self.dim, dtype=np.float16))
        rc = self.bundle.output_cb(None, [view], 1)
        assert rc == 0
        return 0, [0, n, 0.0, 0.0]

    def _run_stage1(self, embeds=None, **kwargs):
        assert embeds is not None
        assert embeds.shape[1] == self.dim
        rc = self.bundle.result_cb(None, [self.tokens[0]], 0)
        assert rc == 0
        rc = self.bundle.result_cb(None, [], 4)
        assert rc == 0
        return 0, [1, 0, 0.0, 0.0]

    def release(self):
        pass


class _FakeTokenizer:
    def __init__(self, ids):
        self.ids = ids

    def encode(self, text, max_tokens=None):
        return list(self.ids)


def _make_stages(V, dim=4, tokens=(11, 12, 13), ids=(7, 8)):
    stage0 = V.StageRuntime("stage0", "m0", "w0")
    stage1 = V.StageRuntime("stage1", "m1", "w1")
    stage0.embedding_dim = dim
    stage1.embedding_dim = dim
    stage0.vocab_size = 32
    stage1.vocab_size = 32
    stage0.max_ctx_len = 128
    stage1.max_ctx_len = 128
    stage0.rknn = _FakeRknn(0, dim, tokens)
    stage1.rknn = _FakeRknn(1, dim, tokens)
    pipeline = V.PipelineState(2, 128, False)
    stage0.rknn.bundle = _FakeBundle(
        tokenizer=V._StageTokenizeAdapter(_FakeTokenizer(ids)),
        embed_callback=V._StageEmbedCallback(pipeline),
        output_cb=V._StageOutputCallback(pipeline, 0, dim),
        result_cb=V._StageResultCallback(0, False),
    )
    stage1.rknn.bundle = _FakeBundle(
        embed_callback=V._StageEmbedCallback(pipeline),
        result_cb=V._StageResultCallback(1, True),
    )
    return stage0, stage1, pipeline


def test_tokenize_adapter_contract():
    V = _imports()
    adapter = V._StageTokenizeAdapter(_FakeTokenizer([5, 6, 7]))
    out = adapter("hi", return_tensors="np", truncation=True)
    assert out["input_ids"].shape == (1, 3)
    assert out["input_ids"][0].tolist() == [5, 6, 7]


def test_embed_callback_writes_and_counts(monkeypatch):
    V = _imports()
    embeds = np.arange(8, dtype=np.float16).reshape(4, 2)
    monkeypatch.setattr(V, "embeds_data", embeds)
    _, _, pipeline = _make_stages(V, dim=2)
    cb = V._StageEmbedCallback(pipeline)
    raw = (ctypes.c_uint16 * 4)()
    rc = cb(None, [1, 3], ctypes.addressof(raw), 8)
    assert rc == 0
    got = np.frombuffer(bytes(raw), dtype=np.float16)
    assert got.tolist() == embeds[[1, 3]].reshape(-1).tolist()
    assert pipeline.slots[0].expected_tokens == 2


def test_embed_callback_rejects_bad_token(monkeypatch):
    V = _imports()
    monkeypatch.setattr(V, "embeds_data", np.zeros((4, 2), dtype=np.float16))
    _, _, pipeline = _make_stages(V, dim=2)
    cb = V._StageEmbedCallback(pipeline)
    raw = (ctypes.c_uint16 * 2)()
    assert cb(None, [9], ctypes.addressof(raw), 4) == -1


def test_output_callback_uses_active_input_tokens():
    V = _imports()
    _, _, pipeline = _make_stages(V)
    cb = V._StageOutputCallback(pipeline, 0, 4)
    pipeline.slots[0].active_input_tokens = 3
    view = _View("output", 1, 12, np.zeros(12, dtype=np.float16))
    assert cb(None, [view], 0) == 0
    assert pipeline.slots[0].batches[0].n_tokens == 3
    assert pipeline.slots[0].batches[0].tensors[0].n_elems == 12


def test_output_callback_prefill_finished_uses_remaining():
    V = _imports()
    _, _, pipeline = _make_stages(V)
    cb = V._StageOutputCallback(pipeline, 0, 4)
    pipeline.slots[0].expected_tokens = 2
    view = _View("output", 1, 8, np.zeros(8, dtype=np.float16))
    assert cb(None, [view], 1) == 0
    assert pipeline.slots[0].batches[0].n_tokens == 2


def test_result_callback_only_last_stage():
    V = _imports()
    V.LAST_RESULT.reset_generation()
    cb0 = V._StageResultCallback(0, False)
    cb1 = V._StageResultCallback(1, True)
    assert cb0(None, [5], 0) == 0
    assert cb1(None, [], 0) == 0
    assert cb1(None, [99], 0) == 0
    assert list(V.LAST_RESULT.generated_tokens) == [99]
    V.LAST_RESULT.reset_generation()


def test_run_pipeline_once_chains_token_to_embed_to_token(monkeypatch):
    V = _imports()
    V.LAST_RESULT.reset_generation()
    monkeypatch.setattr(V, "embeds_data", np.zeros((32, 4), dtype=np.float16))
    monkeypatch.setattr(
        V, "run_vision",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("vision not expected")),
    )
    stage0, stage1, pipeline = _make_stages(V)
    ok, n_tokens = V.run_pipeline_once(
        [stage0, stage1], pipeline, prompt="hi", enable_thinking=False,
    )
    assert ok is True
    assert n_tokens == 2
    assert list(V.LAST_RESULT.generated_tokens) == [stage1.rknn.tokens[0]]
    assert pipeline.slots[0].expected_tokens == 2
    assert pipeline.slots[0].emitted_tokens == 2
    V.LAST_RESULT.reset_generation()


def test_query_core_internal_size_uses_max_entry():
    V = _imports()
    rknn = _FakeRknn(0, 4, ())
    core_num, size = V.query_core_internal_size(rknn)
    assert core_num == 3
    assert size == 300


def test_shared_internal_mem_roundtrip():
    V = _imports()
    stage = V.StageRuntime("stage0", "m", "w")
    vision = V.VisionRuntime("vm", "vw")
    stage.rknn = _FakeRknn(0, 4, ())
    vision.rknn = _FakeRknn(1, 4, ())
    mems = V.setup_shared_internal_mem(vision, stage)
    assert mems == [10000, 10001, 10002]
    assert stage.rknn.shared == mems
    assert vision.rknn.shared == mems
    V.destroy_shared_internal_mem(vision, mems)
    assert mems == []
    assert vision.rknn.destroyed == [10000, 10001, 10002]
