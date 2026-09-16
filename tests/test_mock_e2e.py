"""端到端 mock 测试：用伪 RKNN3Lite + 伪 tokenizer 把整条链路跑一遍。"""
import os
import tempfile
from typing import Annotated

import numpy as np


# --------------------------------------------------------------------------- #
# Attachment hierarchy — direct constructor tests
# --------------------------------------------------------------------------- #
def test_attachment_subclass_constructors():
    """``TextAttachment`` / ``ImageAttachment`` / ``AudioAttachment`` are
    the public carriers; ``Attachment`` is the abstract base + legacy
    factory (``Attachment(content, kind=...)`` syntax)."""
    from wdnai import (
        Attachment, TextAttachment, ImageAttachment, AudioAttachment,
    )

    # Explicit subclass constructors
    t = TextAttachment("hello")
    assert t.kind == "text" and t.text == "hello"
    assert isinstance(t, Attachment)

    img = np.zeros((4, 5, 3), dtype=np.uint8)
    i = ImageAttachment(img)
    assert i.kind == "image" and len(i.images) == 1 and i.images[0] is img
    assert isinstance(i, Attachment)

    frames = [img, img.copy()]
    im = ImageAttachment(frames)
    assert im.kind == "image" and len(im.images) == 2
    assert im.images[0] is frames[0] and im.images[1] is frames[1]
    assert isinstance(im, Attachment)

    au = AudioAttachment(b"\x00\x01\x02")
    assert au.kind == "audio" and au.data == b"\x00\x01\x02"
    assert au.mime_type == "audio/wav"
    assert isinstance(au, Attachment)

    # Cross-class isinstance is negative
    assert not isinstance(t, ImageAttachment)
    assert not isinstance(i, TextAttachment)
    assert not isinstance(au, ImageAttachment)

    # Subclass strict types
    try:
        TextAttachment(123)  # type: ignore[arg-type]
        raise AssertionError("TextAttachment(123) should raise")
    except TypeError:
        pass

    try:
        ImageAttachment("not an array")  # type: ignore[arg-type]
        raise AssertionError("ImageAttachment('str') should raise")
    except TypeError:
        pass

    try:
        AudioAttachment("not bytes")  # type: ignore[arg-type]
        raise AssertionError("AudioAttachment('str') should raise")
    except TypeError:
        pass


def test_attachment_legacy_factory_routing():
    """Legacy ``Attachment(content, kind=...)`` syntax still routes to
    the matching concrete subclass (back-compat for pre-refactor call
    sites)."""
    from wdnai import (
        Attachment, TextAttachment, ImageAttachment, AudioAttachment,
    )
    import numpy as np

    img = np.zeros((4, 5, 3), dtype=np.uint8)

    # ndarray → ImageAttachment
    a = Attachment(img)
    assert isinstance(a, ImageAttachment)

    # list of ndarrays → ImageAttachment (multi-frame)
    a = Attachment([img, img])
    assert isinstance(a, ImageAttachment) and len(a.images) == 2

    # str → TextAttachment (default and explicit)
    a = Attachment("plain")
    assert isinstance(a, TextAttachment) and a.text == "plain"
    a = Attachment("plain", kind="text")
    assert isinstance(a, TextAttachment) and a.text == "plain"

    # bytes → AudioAttachment (default and explicit)
    a = Attachment(b"\x00")
    assert isinstance(a, AudioAttachment) and a.mime_type == "audio/wav"
    a = Attachment(b"\x00", kind="audio", mime_type="audio/pcm")
    assert isinstance(a, AudioAttachment) and a.mime_type == "audio/pcm"

    # The kind="video" path is removed (use ImageAttachment(frames))
    try:
        Attachment("/tmp/foo.mp4", kind="video")
        raise AssertionError("Attachment(kind='video') should raise")
    except TypeError:
        pass

    # Bare Attachment() raises (no channel)
    try:
        Attachment()
        raise AssertionError("Attachment() should raise")
    except TypeError:
        pass


def test_session_coerce_attachments_accepts_hierarchy():
    """``LLMSession._coerce_attachments`` accepts a single subclass
    instance, a list of mixed subclasses, or None — and normalises to
    ``List[Attachment]``."""
    from wdnai import (
        LLMSession, TextAttachment, ImageAttachment, AudioAttachment,
    )
    import numpy as np

    img = np.zeros((4, 5, 3), dtype=np.uint8)

    assert LLMSession._coerce_attachments(None) == []

    single = LLMSession._coerce_attachments(TextAttachment("hi"))
    assert isinstance(single, list) and len(single) == 1
    assert isinstance(single[0], TextAttachment)

    mixed = LLMSession._coerce_attachments(
        [TextAttachment("hi"), ImageAttachment(img), AudioAttachment(b"\x00")],
    )
    assert len(mixed) == 3
    assert [type(x).__name__ for x in mixed] == [
        "TextAttachment", "ImageAttachment", "AudioAttachment",
    ]

    # Legacy factory-produced instance is also accepted
    from wdnai import Attachment
    legacy = LLMSession._coerce_attachments(Attachment("hi"))
    assert isinstance(legacy[0], TextAttachment)

    # Reject non-Attachment, non-list, non-None at the top level
    # (list elements are trusted — caller is responsible for the
    # contents; downstream ``_dispatch`` would crash on a bad element
    # at the ``isinstance`` branch, which is the right place for that
    # error to surface).
    try:
        LLMSession._coerce_attachments("just a string")
        raise AssertionError("non-Attachment single should raise")
    except TypeError:
        pass


# --------------------------------------------------------------------------- #
def _make_workspace():
    tmp = tempfile.mkdtemp(prefix="wdnai_mock_")
    emb = np.arange(64 * 8, dtype=np.uint16).view(np.float16).reshape(64, 8).copy()
    embed_path = os.path.join(tmp, "embed.bin")
    emb.tofile(embed_path)
    tok_dir = os.path.join(tmp, "tokenizer")
    os.makedirs(tok_dir)
    return embed_path, tok_dir


# --------------------------------------------------------------------------- #
class _MockTokenizer:
    def __init__(self, vocab_size: int = 64):
        self._vocab = vocab_size

    def __call__(self, text: str, return_tensors=None, truncation: bool = True):
        ids = [ord(c) % self._vocab for c in text if ord(c) < 128]
        return {"input_ids": np.array([ids], dtype=np.int64)}

    def decode(self, tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        out = []
        for t in tokens:
            try:
                out.append(chr(int(t)))
            except (ValueError, OverflowError):
                pass
        return "".join(out)


class _FakeTensorAttr:
    def __init__(self, n_dims=4, shape=(1, 384, 384, 3), aligned_size=4096, dtype=0):
        self.n_dims, self.shape, self.aligned_size, self.dtype = n_dims, shape, aligned_size, dtype


class _FakeRknn:
    """Mock RKNN3Lite — replaces the vendored ``wdnai._librknn.RKNN3Lite``
    for tests that don't have a real NPU.

    The session_run contract is the same as :class:`wdnai._librknn.RKNN3Lite`:
    ``(ret, [n_decode, n_prefill, llm_start, llm_end])``. The mock
    writes the decoded text into the sink attached via the callback
    bundle so ``_run_llm`` / ``run`` see it.
    """

    # Class-level list of mock device IDs returned by ``get_devices_id()``.
    mock_devices = [b"0001:11:00.0", b"0003:31:00.0", b"deadbeef"]

    def __init__(self, llm_mode=False, deepstack_n=0, vocab_size=64):
        self.llm_mode = llm_mode
        self.deepstack_n = deepstack_n
        self._in_attr = _FakeTensorAttr()
        self._out_attr = _FakeTensorAttr(aligned_size=4096)
        self.last_inputs = None
        self.last_prompt = None
        self.last_init_runtime_kwargs = None
        self.session_calls = 0
        self._sinks: list = []
        self._tokenizer = _MockTokenizer(vocab_size)
        # The shim's ``rknn_config`` is the user-mutable RKNN3Config struct.
        from types import SimpleNamespace
        self.rknn_config = SimpleNamespace()
        self.rknn_config.user_mem_internal = 0
        self.rknn_config.user_mem_weight = 0

    def load_rknn(self, *a, **kw): return 0
    def init_runtime(self, *a, **kw):
        self.last_init_runtime_kwargs = kw
        return 0
    def set_chat_template(self, *a, **kw): return 0
    def clear_kvcache(self, *a, **kw): return 0
    def release(self): return 0

    def get_inputs_tensor_attr(self): return [self._in_attr]
    def get_outputs_tensor_attr(self): return [self._out_attr]
    def get_devices_id(self):
        return list(self.mock_devices)

    def inference(self, inputs=None, data_format=None):
        # Vision RKNN main output: (n_patches, dim) — VLM backend
        # expand_dims -> (1, n_patches, dim).
        n_patches, dim = 32, 8
        feat = np.random.randn(n_patches, dim).astype(np.float32)
        outs = [feat]
        # Deepstack aux outputs share the same (n_patches, dim) layout
        # — Qwen3-VL-4B @ 384 emits (144, 2560); Qwen3.5-VL does
        # the same. Tests rely on this 2-D shape so the aux-packing
        # path is exercised (1-D mocks previously hid a numpy 2D→1D
        # broadcast bug).
        for _ in range(self.deepstack_n):
            outs.append(np.random.randn(n_patches, dim).astype(np.float32))
        return outs

    def session_run(self, inputs=None, prompt=None, embeds=None, **kw):
        self.session_calls += 1
        self.last_prompt = prompt
        self.last_inputs = inputs
        tok = self._tokenizer
        if self.llm_mode:
            ids = [ord(c) % tok._vocab for c in (prompt or "") if ord(c) < 128]
            if "JSON Schema" in (prompt or ""):
                text = '{"act": "run", "level": 2}'
            else:
                text = tok.decode(ids) if ids else (prompt or "")
        else:
            text = f"[vision-mock] {prompt}"
        for sink in self._sinks:
            sink.text = text
        return 0, [10, 5, 0.0, 0.001]

    def rknn3_query(self, cmd, index=0): return 0


# --------------------------------------------------------------------------- #
def _install_mocks(deepstack_n: int = 0, vocab_size: int = 64):
    """Mock wdnai._librknn.RKNN3Lite + transformers + key helpers.

    Also rebinds ``build_llm_callback`` so the produced callback
    bundle stashes the most-recently-created :class:`_FakeRknn` on
    ``last_rknn['obj']`` (used by ``session_run`` to know which sink
    to update).
    """
    import wdnai.backends._common as common
    import wdnai.backends.text as text_mod
    import wdnai.backends.vlm as vlm_mod
    import wdnai._librknn as librknn

    # ---- 1) RKNN3Lite 构造器：返回 _FakeRknn ----
    def make(llm_mode=False, verbose=False):
        return _FakeRknn(llm_mode=llm_mode, deepstack_n=deepstack_n, vocab_size=vocab_size)
    librknn.RKNN3Lite = make

    # ---- 2) Tokenizer / Embed 加载 ----
    common.load_tokenizer = lambda path: _MockTokenizer(vocab_size)
    common.load_embed_table = lambda path, vs: np.zeros((vs, 8), dtype=np.float16)

    # ---- 3) 各后端模块里也都重新绑一遍（它们是 from import） ----
    text_mod.load_tokenizer = common.load_tokenizer
    text_mod.load_embed_table = common.load_embed_table
    vlm_mod.load_tokenizer = common.load_tokenizer
    vlm_mod.load_embed_table = common.load_embed_table

    # ---- 4) build_llm_callback：返回的 bundle 把 sink 串到 rknn._sinks ----
    # Strategy: build the real bundle (so result_cb gets called with
    # ``(sink_, token_ids, state)``), then on the produced bundle
    # append the sink to the most-recently-created ``_FakeRknn``'s
    # ``_sinks`` list. ``session_run`` will then update sink.text
    # exactly like the production shim's trampoline does.
    last_rknn: dict = {"obj": None}
    orig_build = common.build_llm_callback

    def hooked_build(tokenizer, embeds, sink):
        bundle = orig_build(tokenizer, embeds, sink)
        if last_rknn["obj"] is not None:
            last_rknn["obj"]._sinks.append(sink)
        return bundle
    common.build_llm_callback = hooked_build
    text_mod.build_llm_callback = hooked_build
    vlm_mod.build_llm_callback = hooked_build

    # ---- 5) backend __init__ monkey-patches: track "last created rknn" ----
    orig_text_init = text_mod.TextBackend.__init__

    def h_text(self, spec, *, npu=None, device_id=None):
        from wdnai._librknn import RKNN3Lite  # 已 monkey-patch 成 make()
        if not spec.rknn_path or not spec.weight_path:
            raise ValueError("missing path")
        self.rknn = RKNN3Lite(llm_mode=True, verbose=spec.verbose)
        if self.rknn.load_rknn(spec.rknn_path, spec.weight_path) != 0:
            raise RuntimeError("load failed")
        last_rknn["obj"] = self.rknn
        from wdnai.backends._common import (
            StreamSink, build_llm_args, apply_chat_template,
        )
        self.tokenizer = text_mod.load_tokenizer(spec.tokenizer_path)
        self.embeds = text_mod.load_embed_table(spec.embed_path, spec.vocab_size)
        self.sink = StreamSink(tokenizer=self.tokenizer)
        cb = text_mod.build_llm_callback(self.tokenizer, self.embeds, self.sink)
        from wdnai.backends import resolve_device_id
        resolved = resolve_device_id(RKNN3Lite, npu, device_id)
        if self.rknn.init_runtime(
            target=spec.target, core_mask=spec.core_mask,
            llm_args=build_llm_args(spec), llm_callback=cb,
            device_id=resolved,
        ) != 0:
            raise RuntimeError("init_runtime failed")
        apply_chat_template(self.rknn)
    text_mod.TextBackend.__init__ = h_text

    orig_vlm_init = vlm_mod.VLMBackend.__init__

    def h_vlm(self, spec, *, npu=None, device_id=None):
        from wdnai._librknn import RKNN3Lite
        if not spec.rknn_llm_path or not spec.rknn_vision_path:
            raise ValueError("missing path")
        self.spec = spec
        # vision
        self.rknn_vision = RKNN3Lite()
        vp = spec.rknn_vision_path
        if self.rknn_vision.load_rknn(vp, vp.replace(".rknn3", ".weight")) != 0:
            raise RuntimeError("vision load failed")
        from wdnai.backends.vlm import _VisionVariant
        self._vision_variants = [_VisionVariant(
            image_size=spec.image_size,
            rknn=self.rknn_vision,
            rknn_path=vp,
            weight_path=vp.replace(".rknn3", ".weight"),
        )]
        # llm
        self.rknn_llm = RKNN3Lite(llm_mode=True, verbose=spec.verbose)
        lp = spec.rknn_llm_path
        if self.rknn_llm.load_rknn(lp, lp.replace(".rknn3", ".weight")) != 0:
            raise RuntimeError("llm load failed")
        self.variant = spec.variant
        last_rknn["obj"] = self.rknn_llm

        from wdnai.backends._common import (
            StreamSink, build_llm_args, apply_chat_template,
        )
        self.tokenizer = vlm_mod.load_tokenizer(spec.tokenizer_path)
        self.embeds = vlm_mod.load_embed_table(spec.embed_path, spec.vocab_size)
        from wdnai.backends import resolve_device_id
        resolved = resolve_device_id(RKNN3Lite, npu, device_id)
        if self.rknn_vision.init_runtime(
            target=spec.target, core_mask=spec.core_mask,
            device_id=resolved,
        ) != 0:
            raise RuntimeError("vision init failed")
        self.sink = StreamSink(tokenizer=self.tokenizer)
        cb = vlm_mod.build_llm_callback(self.tokenizer, self.embeds, self.sink)
        if self.rknn_llm.init_runtime(
            target=spec.target, core_mask=spec.core_mask,
            llm_args=build_llm_args(spec), llm_callback=cb,
            device_id=resolved,
        ) != 0:
            raise RuntimeError("llm init failed")
        apply_chat_template(self.rknn_llm)
    vlm_mod.VLMBackend.__init__ = h_vlm


# --------------------------------------------------------------------------- #
def test_text_qwen3():
    from wdnai import LLMSession, SystemLanguageModel
    embed, tok_dir = _make_workspace()
    _install_mocks(deepstack_n=0, vocab_size=64)

    spec = SystemLanguageModel.qwen3_1_7b().with_paths(
        rknn_path="/fake.rknn3", weight_path="/fake.weight",
        tokenizer_path=tok_dir, embed_path=embed,
    )
    sess = LLMSession(model=spec)
    text = sess._backend.run("hello world")
    print("[Qwen3] text ->", repr(text))
    assert text
    sess.close()


def test_text_qwen25():
    from wdnai import LLMSession, SystemLanguageModel
    embed, tok_dir = _make_workspace()
    _install_mocks(deepstack_n=0, vocab_size=64)

    spec = SystemLanguageModel.qwen2_5_0_5b().with_paths(
        rknn_path="/fake.rknn3", weight_path="/fake.weight",
        tokenizer_path=tok_dir, embed_path=embed,
    )
    sess = LLMSession(model=spec)
    text = sess._backend.run("你好")
    print("[Qwen2.5] text ->", repr(text))
    assert text
    sess.close()


def test_image_qwen3vl():
    from wdnai import LLMSession, SystemLanguageModel
    from pydantic import BaseModel, Field
    from typing import Optional

    embed, tok_dir = _make_workspace()
    _install_mocks(deepstack_n=3, vocab_size=64)

    spec = SystemLanguageModel.qwen3_vl_2b().with_paths(
        rknn_llm_path="/fake_llm.rknn3", rknn_vision_path="/fake_v.rknn3",
        tokenizer_path=tok_dir, embed_path=embed,
    )
    sess = LLMSession(model=spec)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    text = sess._backend.run_images("请描述", [img])
    print("[Qwen3-VL] image ->", repr(text))
    assert text

    # 结构化输出
    class Event(BaseModel):
        act: str = Field(description="行为")
        level: Optional[int] = None

    parsed = LLMSession._parse_output('{"act": "run", "level": 3}', Event)
    assert parsed.act == "run" and parsed.level == 3
    print("[Qwen3-VL] structured parse ->", parsed)

    sess.close()


def test_pack_aux_2d_deepstack_does_not_broadcast_fail():
    """Regression: ``_LLMSession._pack_aux`` must accept 2-D ``(n_patches,
    embed_dim)`` aux tensors — Qwen3-VL-4B @ 384 emits ``(144, 2560)``
    per deepstack output, and the C-side destination buffer is a flat
    1-D fp16 view. Before the fix, the assignment
    ``dst[:] = data`` raised
    ``ValueError: could not broadcast input array from shape
    (144, 2560) into shape (368640,)`` because numpy cannot auto-
    broadcast a 2-D source into a 1-D destination even when the total
    element count matches. The fix ravel()-s the data first.

    The test stubs ``rknn3_create_mem`` to hand back a python-side
    byte buffer (no real NPU needed) so we can call ``_pack_aux``
    directly with the real (144, 2560) shape and assert both that
    the broadcast error is gone AND that the bytes written to the
    destination match the source after ravel()-ing.
    """
    import ctypes as _ct
    import wdnai._librknn as librknn

    # Allocate a destination byte buffer the stub C call will hand
    # back. Sized for the 144*2560 fp16 payload (368640 elements *
    # 2 bytes = 737280) plus a little slack so the assignment still
    # has room to land if the runtime ever grows the alloc.
    sentinel_n, sentinel_d = 144, 2560
    nbytes = sentinel_n * sentinel_d * np.dtype(np.float16).itemsize
    dst_buf = (_ct.c_uint8 * nbytes)()

    class _FakeMemStruct(_ct.Structure):
        # Mirror the C-side ``rknn3_tensor_mem`` just enough for the
        # virt_addr / size fields that ``_pack_aux`` reads.
        _fields_ = [
            ("virt_addr", _ct.c_void_p),
            ("phys_addr", _ct.c_uint64),
            ("fd", _ct.c_int32),
            ("buffer_id", _ct.c_uint64),
            ("offset", _ct.c_uint64),
            ("size", _ct.c_uint64),
            ("flags", _ct.c_uint64),
            ("core_id", _ct.c_int32),
            ("priv_data", _ct.c_void_p),
            ("mem_type", _ct.c_int),
        ]

    fake_mem = _FakeMemStruct()
    fake_mem.virt_addr = _ct.cast(dst_buf, _ct.c_void_p).value
    fake_mem.size = nbytes

    class _StubLib:
        """Minimal stand-in for the librknn3 CDLL — only the
        symbols ``_LLMSession.__init__`` / ``_pack_aux`` reach.
        ``rknn3_session_init`` returns a non-NULL handle so the
        post-init bookkeeping (``self._session = sess``) succeeds;
        ``_pack_aux`` calls ``rknn3_create_mem`` which we redirect
        to our python-side buffer."""

        def __init__(self):
            self.rknn3_create_mem = self._create_mem
            self.rknn3_session_init = self._session_init
            self.rknn3_query = self._query

        @staticmethod
        def _create_mem(ctx, size, core_id, flags):
            return _ct.cast(_ct.pointer(fake_mem), _ct.c_void_p).value

        @staticmethod
        def _session_init(ctx, params, n_params):
            # Return a non-zero pointer so ``_LLMSession._init``
            # treats the call as success. The actual session handle
            # is never dereferenced by ``_pack_aux``.
            return 0x1

        @staticmethod
        def _query(ctx, cmd, buf, size):
            # ``_LLMSession._init`` calls ``rknn3_query`` to read
            # ``LLM_CONFIG.max_ctx_len``. We never use the result, so
            # returning 0 (success) is fine — the except branch in
            # ``_init`` handles failures gracefully.
            return 0

    # Build a real ``_LLMRuntime`` + ``_LLMSession`` against the
    # stub lib so ``_pack_aux`` runs the real code path (including
    # the bytes-write step). The runtime only needs ``lib`` and
    # ``context`` populated enough to satisfy ``_pack_aux``'s
    # preconditions; we never call ``rknn3_session_run``.
    runtime = librknn._LLMRuntime.__new__(librknn._LLMRuntime)
    runtime.lib = _StubLib()
    runtime.context = 0
    runtime._aux_mem_refs = []
    runtime._aux_attr_refs = []
    runtime._aux_pool = {}

    llm_session = librknn._LLMSession(
        runtime=runtime,
        session=0,  # placeholder; not used by _pack_aux
        params=[{"vocab_size": 64, "max_context_len": 4096,
                 "special_eos_id": 0}],
        callback=None,
    )

    # Build the deterministic 2-D deepstack payload that mirrors
    # the real Qwen3-VL-4B @ 384 output. ``(144, 2560)`` — chosen
    # specifically so its total element count (368640) is *not* a
    # power of two, locking the regression contract against future
    # "simplifications" that try to assume 1-D sources.
    rng = np.random.default_rng(seed=0x5753_4E_54)
    sentinel_2d = rng.standard_normal(
        (sentinel_n, sentinel_d)
    ).astype(np.float16)
    aux = librknn.RKNN3AuxTensorWrapper(
        aux_data=sentinel_2d,
        index=2,
        align_size=nbytes,
    )
    packed = llm_session._pack_aux(aux)

    # The packed C struct must carry the 144*2560 fp16 count and
    # 2 dims so the LLM session knows the layout.
    assert packed.role is None
    assert int(packed.input_type) == int(librknn._LLMInputType.AUX)
    attr = packed._u.aux_input.attr.contents
    assert attr.index == 2
    assert attr.n_elems == (sentinel_n * sentinel_d)
    assert attr.n_dims == 2
    # ``shape[]`` and ``dtype`` are the fields the C library checks
    # against the model's aux tensor contract. Both used to be
    # left zero — ``shape[]`` stayed ``[0, 0, ...]`` and ``dtype``
    # defaulted to ``0`` (``RKNN3_TENSOR_FLOAT32``) — which made the
    # C library compute ``n_elems * dtype_size(4) = 1474560`` against
    # the actual ``aligned_size(737280)`` and reject the input via
    # ``rknn3_session_run -> ret=-1``. Without these assertions a
    # revert that drops the manual ``shape[]`` / ``dtype`` assignment
    # silently slips back to the bug.
    assert attr.shape[0] == sentinel_n
    assert attr.shape[1] == sentinel_d
    assert int(attr.dtype) == int(
        librknn.RKNN3TensorType.RKNN3_TENSOR_FLOAT16
    )
    assert attr.aligned_size == nbytes

    # The destination buffer must hold the ravel()-d source bytes in
    # C order (the assignment the bug used to blow up on).
    written = np.ctypeslib.as_array(
        dst_buf, shape=(nbytes,)
    ).view(np.float16)[:sentinel_n * sentinel_d]
    np.testing.assert_array_equal(written, sentinel_2d.ravel())

    print("[Qwen3-VL] _pack_aux 2-D deepstack regression OK")


def test_pack_inputs_multimodal_must_be_first():
    """Regression: ``_LLMSession._pack_inputs`` must put the multimodal
    input at ``inputs[0]`` and deepstack aux inputs AFTER it.

    ``rknn3_session_run`` rejects the call with
    ``RKLLMSession: Please put the LLM input at the first position!``
    when any aux tensor appears before the multimodal entry — see
    user incident on RK3588 board (2026-09-08) where the C library
    stderr showed the message and ``session_run`` returned ``-1`` for
    every frame despite the previous shape/dtype fixes having landed.
    Root cause was that the Python shim packed aux first, then pushed
    multimodal at the end; the C library iterates ``inputs[]`` in
    order and treats the first one as the LLM input. C++ reference
    ``examples/Qwen3_VL/cpp/llm/rknn_qwen3_vl_llm.cc:174-180`` puts
    multimodal at ``inputs[0]`` and aux at ``inputs[1..3]``.

    The test stubs ``_do_session_run`` (the function that calls
    ``rknn3_session_run``) to capture the packed ``inputs`` list,
    then asserts the ordering invariant.
    """
    import ctypes as _ct
    import wdnai._librknn as librknn

    captured_inputs: list = []

    # Stub librknn3 entry points: ``_pack_aux`` calls
    # ``rknn3_create_mem`` and then dereferences the result as a
    # ``_TensorMemory*``; a bare integer (e.g. ``return 1``) would
    # segfault on first read. Return a pointer to a real, zeroed
    # ``_TensorMemory`` instead — ``virt_addr`` and ``size`` are the
    # only fields touched and zeros satisfy both.
    runtime = librknn._LLMRuntime.__new__(librknn._LLMRuntime)

    class _StubLib:
        def __init__(self):
            self.rknn3_create_mem = self._create_mem
            self.rknn3_session_init = lambda *a, **k: 1
            self.rknn3_query = lambda *a, **k: 0

        @staticmethod
        def _create_mem(ctx, size, core_id, flags):
            tm = librknn._TensorMemory()
            tm.virt_addr = 0
            # ``size`` arrives as a ctypes integer (c_uint64 on the
            # signature); convert to Python int for assignment.
            tm.size = int(size.value) if hasattr(size, "value") else int(size)
            return _ct.cast(_ct.pointer(tm), _ct.c_void_p).value

    runtime.lib = _StubLib()
    runtime.context = 0
    runtime._aux_mem_refs = []
    runtime._aux_attr_refs = []
    runtime._aux_pool = {}
    llm_session = librknn._LLMSession(
        runtime=runtime, session=0,
        params=[{"vocab_size": 64, "max_context_len": 4096,
                 "special_eos_id": 0}],
        callback=None,
    )

    # 2-D aux data matching Qwen3-VL-4B @ 384 deepstack output shape.
    sentinel_n, sentinel_d = 12, 8
    sentinel_2d = np.zeros(
        (sentinel_n, sentinel_d), dtype=np.float16
    )
    # All three aux tensors share the SAME ``align_size`` — this is
    # exactly what ``wdnai.backends.vlm._build_image_inputs`` does
    # for Qwen3-VL-4B's 3 deepstack outputs (sets every aux's
    # ``align_size`` to the first vision output's ``aligned_size``).
    # Before the fix, this caused all 3 aux inputs to share one mem
    # block + one attr (with ``index=2``), which leaked ~1.5–1.8 GB
    # of NPU memory on RK3588. The fix keys the lazy-pool by
    # ``(align_size, aux_index)`` so each aux gets its own mem block
    # and its own correctly-indexed attr.
    shared_align = sentinel_n * sentinel_d * 2
    aux_wrappers = [
        librknn.RKNN3AuxTensorWrapper(
            aux_data=sentinel_2d.copy(), index=2 + i,
            align_size=shared_align,
        )
        for i in range(3)
    ]

    # Build a minimal RKNN3Image stand-in (no C calls; we only need
    # ``_pack_inputs`` to recognise the type via ``isinstance`` and
    # call ``_pack_multimodal``). We stub the image-attr by giving
    # it a dummy bytes buffer (the multimodal packer never reads
    # the image bytes for this test — the test cares only about
    # ordering of the returned ``packed`` list, not the bytes
    # inside the multimodal struct).
    import ctypes as _ct
    from wdnai._librknn import RKNN3Image, Float16
    dummy_emit = (Float16 * 4)()
    mm_image = RKNN3Image()
    mm_image.image_embed = _ct.cast(dummy_emit, _ct.POINTER(Float16))
    mm_image.n_image_tokens = 1
    mm_image.n_image = 1
    mm_image.image_width = 384
    mm_image.image_height = 384
    mm_image.image_start = b"<|vision_start|>"
    mm_image.image_end = b"<|vision_end|>"
    mm_image.image_content = b"<|image_pad|>"

    packed = llm_session._pack_inputs(
        prompt="白色车辆", embeds=None, tokens=None,
        inputs=[*aux_wrappers, mm_image],
        enable_thinking=False,
    )

    # The multimodal input MUST be at index 0; aux inputs follow.
    # (Early-bug version had aux at indices 0..2 and multimodal at
    # index 3, which made the C library reject every call.)
    assert len(packed) == 4, f"want 4 packed inputs, got {len(packed)}"
    assert int(packed[0].input_type) == int(
        librknn._LLMInputType.MULTIMODAL
    ), (
        f"inputs[0] must be MULTIMODAL, got input_type="
        f"{int(packed[0].input_type)}"
    )
    for i in range(1, 4):
        assert int(packed[i].input_type) == int(
            librknn._LLMInputType.AUX
        ), (
            f"inputs[{i}] must be AUX, got input_type="
            f"{int(packed[i].input_type)}"
        )
    # ``attr.index`` on each aux must round-trip (2 + i in vlm.py).
    for i in range(3):
        assert int(packed[i + 1]._u.aux_input.attr.contents.index) == 2 + i

    # Regression guard: even when all 3 aux share one ``align_size``
    # (Qwen3-VL-4B production layout — vlm.py copies the first vision
    # output's aligned_size to every aux), each must get its own
    # mem block. Sharing leaks ~1.5–1.8 GB of NPU memory on RK3588.
    pool_mem_ids = {id(entry["mem_ptr"]) for entry in runtime._aux_pool.values()}
    assert len(pool_mem_ids) == 3, (
        f"3 aux inputs share only {len(pool_mem_ids)} unique mem_ptr(s); "
        f"duplicate mem blocks leak NPU memory on session teardown. Pool "
        f"key must include aux_index, not just align_size."
    )
    assert len(runtime._aux_pool) == 3, (
        f"_aux_pool has {len(runtime._aux_pool)} entries after 3 aux "
        f"with shared align_size; expected 3 (one per aux_index)."
    )

    # Pool reuse: a second ``_pack_inputs`` call (simulating the next
    # video frame) must NOT re-call ``rknn3_create_mem`` — the lazy
    # pool hands back the same buffer slots. The stub counts every
    # create call; after the first round we expect zero growth.
    first_pool_size = len(runtime._aux_pool)
    second = llm_session._pack_inputs(
        prompt="白色车辆", embeds=None, tokens=None,
        inputs=[*aux_wrappers, mm_image],
        enable_thinking=False,
    )
    assert len(second) == 4, f"want 4 packed inputs, got {len(second)}"
    assert len(runtime._aux_pool) == first_pool_size, (
        f"_aux_pool grew from {first_pool_size} to "
        f"{len(runtime._aux_pool)} — driver-side memory would leak "
        f"across frames. Real RK3588 boards crash after ~8 frames "
        f"when this happens (rc=-11)."
    )

    print("[Qwen3-VL] _pack_inputs ordering + per-aux mem regression OK")


def test_image_qwen25vl():
    from wdnai import LLMSession, SystemLanguageModel
    embed, tok_dir = _make_workspace()
    _install_mocks(deepstack_n=0, vocab_size=64)  # Qwen2.5-VL 没有 deepstack

    spec = SystemLanguageModel.qwen2_5_vl_3b().with_paths(
        rknn_llm_path="/fake_llm.rknn3", rknn_vision_path="/fake_v.rknn3",
        tokenizer_path=tok_dir, embed_path=embed,
    )
    sess = LLMSession(model=spec)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    text = sess._backend.run_images("请描述", [img])
    print("[Qwen2.5-VL] image ->", repr(text))
    assert text
    sess.close()


def test_overrides_passed_to_session_run():
    """``max_new_tokens`` / ``enable_thinking`` / ``keep_history`` 能通过 overrides 透传。"""
    from wdnai import LLMSession, SystemLanguageModel
    embed, tok_dir = _make_workspace()
    _install_mocks(deepstack_n=0, vocab_size=64)

    spec = SystemLanguageModel.qwen3_1_7b().with_paths(
        rknn_path="/fake.rknn3", weight_path="/fake.weight",
        tokenizer_path=tok_dir, embed_path=embed,
    )
    sess = LLMSession(model=spec)
    # 让 mock session_run 记录调用时的 kwargs
    captured = {}
    orig_run = sess._backend.rknn.session_run
    def spy(**kw):
        captured.update(kw); return orig_run(**kw)
    sess._backend.rknn.session_run = spy

    sess._backend.run("hi", max_new_tokens=8, enable_thinking=True, keep_history=1)
    assert captured.get("max_new_tokens") == 8
    assert captured.get("enable_thinking") is True
    assert captured.get("keep_history") == 1
    print("[override] session_run kwargs =", {k: captured[k] for k in ("max_new_tokens", "enable_thinking", "keep_history")})

    # VLM 也同样
    _install_mocks(deepstack_n=3, vocab_size=64)
    vlm_spec = SystemLanguageModel.qwen3_vl_2b().with_paths(
        rknn_llm_path="/fake_llm.rknn3", rknn_vision_path="/fake_v.rknn3",
        tokenizer_path=tok_dir, embed_path=embed,
    )
    sess2 = LLMSession(model=vlm_spec)
    captured2 = {}
    orig_run2 = sess2._backend.rknn_llm.session_run
    def spy2(**kw):
        captured2.update(kw); return orig_run2(**kw)
    sess2._backend.rknn_llm.session_run = spy2
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    sess2._backend.run_images("desc", [img], max_new_tokens=64, enable_thinking=True)
    assert captured2.get("max_new_tokens") == 64
    assert captured2.get("enable_thinking") is True
    print("[override] VLM session_run kwargs =", {k: captured2[k] for k in ("max_new_tokens", "enable_thinking")})
    sess.close()
    sess2.close()


def test_session_respond_full_flow():
    """通过 LLMSession.respond() 走完整链路：prompt 注入 + JSON 解析 + pydantic。"""
    import asyncio
    from wdnai import LLMSession, SystemLanguageModel, ImageAttachment
    from pydantic import BaseModel, Field
    from typing import Optional

    class Event(BaseModel):
        act: str = Field(description="行为")
        level: Optional[int] = None

    embed, tok_dir = _make_workspace()
    _install_mocks(deepstack_n=3, vocab_size=64)

    spec = SystemLanguageModel.qwen3_vl_2b().with_paths(
        rknn_llm_path="/fake_llm.rknn3", rknn_vision_path="/fake_v.rknn3",
        tokenizer_path=tok_dir, embed_path=embed,
    )

    async def run():
        sess = LLMSession(model=spec)
        # 1) 纯文本
        out1 = await sess.respond(prompt="请介绍 RK 芯片")
        assert isinstance(out1, str) and out1
        print("[session] text respond ->", repr(out1))

        # 2) 图像
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        out2 = await sess.respond(
            prompt="请分析这张图的行为", attachment=ImageAttachment(img),
        )
        assert isinstance(out2, str) and out2
        print("[session] image respond ->", repr(out2))

        # 3) 图像 + 结构化输出（注入 JSON hint + 解析）
        out3 = await sess.respond(
            prompt="请分析这张图的行为",
            attachment=ImageAttachment(img),
            generating=Event,
        )
        # mock 返回的 prompt 是 "[vision-mock] <image>请分析这张图的行为 + JSON hint"，
        # 解析不到 JSON，会回退到 str（pydantic 抛错时我们的代码会让 pydantic 兜底）
        # 实际上 extract_json 找不到 JSON，_parse_output 会把 str 直接给 pydantic 兜底；
        # pydantic 对纯字符串解析会失败——所以这里我们手工验证 _maybe_inject_json_hint
        hint = LLMSession._maybe_inject_json_hint("请分析", Event)
        assert "JSON Schema" in hint and "act" in hint
        print("[session] json hint injected ok")

        # 4) 直接给 dict 路径：模拟模型返回 JSON
        parsed = LLMSession._parse_output(
            '{"act": "jump", "level": 4}', Event,
        )
        assert isinstance(parsed, Event) and parsed.act == "jump" and parsed.level == 4
        print("[session] pydantic parse ->", parsed)
        sess.close()

    asyncio.run(run())


# --------------------------------------------------------------------------- #
def test_device_id_passed_through():
    """``LLMSession(model=..., device_id=b"...")`` 下传到 RKNN3Lite.init_runtime。"""
    from wdnai import LLMSession, SystemLanguageModel

    embed, tok_dir = _make_workspace()
    _install_mocks(deepstack_n=0, vocab_size=64)

    spec = SystemLanguageModel.qwen3_1_7b().with_paths(
        rknn_path="/fake.rknn3", weight_path="/fake.weight",
        tokenizer_path=tok_dir, embed_path=embed,
    )

    raw = b"rk-custom-serial"
    sess = LLMSession(model=spec, device_id=raw)
    assert sess.device_id == raw
    assert sess.npu is None
    captured = sess._backend.rknn.last_init_runtime_kwargs
    assert captured is not None
    assert captured.get("device_id") == raw
    print("[device_id] TextBackend init_runtime kw device_id =", captured["device_id"])
    sess.close()

    # str 形态也会被自动 encode
    sess2 = LLMSession(model=spec, device_id="rk3588")
    assert sess2.device_id == b"rk3588"
    captured2 = sess2._backend.rknn.last_init_runtime_kwargs
    assert captured2.get("device_id") == b"rk3588"
    print("[device_id] str -> bytes, kw =", captured2["device_id"])
    sess2.close()


def test_npu_index_resolves_via_get_devices_id():
    """``LLMSession(model=..., npu=K)`` 走 ``RKNN3Lite().get_devices_id()[K]``。"""
    from wdnai import LLMSession, SystemLanguageModel

    embed, tok_dir = _make_workspace()
    _install_mocks(deepstack_n=3, vocab_size=64)

    spec = SystemLanguageModel.qwen3_vl_2b().with_paths(
        rknn_llm_path="/fake_llm.rknn3", rknn_vision_path="/fake_v.rknn3",
        tokenizer_path=tok_dir, embed_path=embed,
    )

    # _FakeRknn.mock_devices = [b'0001:11:00.0', b'0003:31:00.0', b'deadbeef']
    sess = LLMSession(model=spec, npu=1)
    assert sess.npu == 1
    expected = b"0003:31:00.0"

    # vision + llm 都应该拿到同一个解析后的 device_id
    assert sess._backend.rknn_vision.last_init_runtime_kwargs.get("device_id") == expected
    assert sess._backend.rknn_llm.last_init_runtime_kwargs.get("device_id") == expected
    print(
        "[npu] vision=%r llm=%r"
        % (
            sess._backend.rknn_vision.last_init_runtime_kwargs["device_id"],
            sess._backend.rknn_llm.last_init_runtime_kwargs["device_id"],
        )
    )
    sess.close()


def test_npu_out_of_range_raises():
    """``npu`` 越界时抛 IndexError，错误消息列出可用列表。"""
    from wdnai import LLMSession, SystemLanguageModel

    embed, tok_dir = _make_workspace()
    _install_mocks(deepstack_n=0, vocab_size=64)

    spec = SystemLanguageModel.qwen3_1_7b().with_paths(
        rknn_path="/fake.rknn3", weight_path="/fake.weight",
        tokenizer_path=tok_dir, embed_path=embed,
    )

    try:
        LLMSession(model=spec, npu=10)
    except IndexError as e:
        msg = str(e)
        assert "npu=10" in msg
        assert "device(s) found" in msg
        # 可用列表里有这 3 个 mock id 之一
        assert b"0001:11:00.0" in msg.encode() or b"deadbeef" in msg.encode()
        print("[npu] out-of-range IndexError:", msg)
    else:
        raise AssertionError("expected IndexError")


def test_npu_and_device_id_mutually_exclusive():
    """同时传 npu + device_id 直接在 LLMSession.__init__ 抛 ValueError（不依赖 backend）。"""
    from wdnai import LLMSession, SystemLanguageModel

    embed, tok_dir = _make_workspace()
    _install_mocks(deepstack_n=0, vocab_size=64)

    spec = SystemLanguageModel.qwen3_1_7b().with_paths(
        rknn_path="/fake.rknn3", weight_path="/fake.weight",
        tokenizer_path=tok_dir, embed_path=embed,
    )

    try:
        LLMSession(model=spec, npu=0, device_id=b"abc")
    except ValueError as e:
        assert "mutually exclusive" in str(e)
        print("[npu] mutual-exclusion ValueError:", e)
    else:
        raise AssertionError("expected ValueError")


# --------------------------------------------------------------------------- #
def test_discover_vision_paths_multi_resolution():
    """``_discover_vision_paths`` finds every ``<prefix>.<WxH>.1828.rknn`` sibling.

    The naming convention is: ``vision_<name>.<W>x<H>.1828.rknn`` + a
    matching ``vision_<name>.<W>x<H>.weight``. Discovery is per
    ``<prefix>``, so a directory containing both
    ``vision_Qwen3-VL-4B.384x384.*`` and
    ``vision_Qwen3-VL-4B.768x768.*`` returns two entries sorted by
    image_size ascending.
    """
    import os
    import tempfile
    from wdnai.backends.vlm import _discover_vision_paths

    with tempfile.TemporaryDirectory(prefix="wdnai_vision_") as d:
        d_path = d
        # Multi-resolution layout
        for size in (384, 768):
            base = f"vision_Qwen3-VL-4B.{size}x{size}"
            with open(os.path.join(d_path, f"{base}.1828.rknn"), "wb") as f:
                f.write(b"\x00" * 16)
            with open(os.path.join(d_path, f"{base}.weight"), "wb") as f:
                f.write(b"\x00" * 16)

        # Anchor: point at the 384x384 file so the backend scans the
        # right directory.
        anchor = os.path.join(d_path, "vision_Qwen3-VL-4B.384x384.1828.rknn")
        variants = _discover_vision_paths(anchor)
        sizes = sorted([s for s, _r, _w in variants])
        assert sizes == [384, 768], f"expected [384, 768], got {sizes}"
        for s, rknn_p, weight_p in variants:
            assert rknn_p.endswith(f".{s}x{s}.1828.rknn")
            assert weight_p.endswith(f".{s}x{s}.weight")
        print("[discover] multi-resolution ->", variants)


def test_discover_vision_paths_legacy_single_file():
    """Legacy ``<prefix>.1828.rknn`` (no ``.<WxH>.`` suffix) returns one entry."""
    import os
    import tempfile
    from wdnai.backends.vlm import _discover_vision_paths

    with tempfile.TemporaryDirectory(prefix="wdnai_vision_legacy_") as d:
        d_path = d
        with open(os.path.join(d_path, "vision_Qwen3-VL-4B.1828.rknn"), "wb") as f:
            f.write(b"\x00" * 16)
        with open(os.path.join(d_path, "vision_Qwen3-VL-4B.weight"), "wb") as f:
            f.write(b"\x00" * 16)

        anchor = os.path.join(d_path, "vision_Qwen3-VL-4B.1828.rknn")
        variants = _discover_vision_paths(anchor)
        assert len(variants) == 1
        size, rknn_p, weight_p = variants[0]
        assert size == -1  # legacy sentinel; caller resolves from spec.image_size
        assert rknn_p == anchor
        assert weight_p.endswith(".weight")
        print("[discover] legacy single file ->", variants)


def test_discover_vision_paths_skips_half_present_variants():
    """A ``.1828.rknn`` without its ``.weight`` sibling is skipped."""
    import os
    import tempfile
    from wdnai.backends.vlm import _discover_vision_paths

    with tempfile.TemporaryDirectory(prefix="wdnai_vision_half_") as d:
        d_path = d
        # 384x384: complete pair
        with open(os.path.join(d_path, "vision_X.384x384.1828.rknn"), "wb") as f:
            f.write(b"\x00")
        with open(os.path.join(d_path, "vision_X.384x384.weight"), "wb") as f:
            f.write(b"\x00")
        # 768x768: missing .weight — must be skipped
        with open(os.path.join(d_path, "vision_X.768x768.1828.rknn"), "wb") as f:
            f.write(b"\x00")
        # no .weight sibling for 768x768

        anchor = os.path.join(d_path, "vision_X.384x384.1828.rknn")
        variants = _discover_vision_paths(anchor)
        sizes = sorted([s for s, _r, _w in variants])
        assert sizes == [384], f"expected only 384, got {sizes}"
        print("[discover] skip half-present variant ->", sizes)


def test_discover_vision_paths_ignores_other_prefixes():
    """Discovery is scoped to the anchor's prefix; other models are ignored."""
    import os
    import tempfile
    from wdnai.backends.vlm import _discover_vision_paths

    with tempfile.TemporaryDirectory(prefix="wdnai_vision_other_") as d:
        d_path = d
        # Our model: 384x384 only
        with open(os.path.join(d_path, "vision_Qwen3-VL-4B.384x384.1828.rknn"), "wb") as f:
            f.write(b"\x00")
        with open(os.path.join(d_path, "vision_Qwen3-VL-4B.384x384.weight"), "wb") as f:
            f.write(b"\x00")
        # Different model (embedding) also in the same dir — must NOT be returned
        with open(os.path.join(d_path, "vision_Qwen3-VL-Embedding-2B.1828.rknn"), "wb") as f:
            f.write(b"\x00")
        with open(os.path.join(d_path, "vision_Qwen3-VL-Embedding-2B.weight"), "wb") as f:
            f.write(b"\x00")

        anchor = os.path.join(d_path, "vision_Qwen3-VL-4B.384x384.1828.rknn")
        variants = _discover_vision_paths(anchor)
        sizes = sorted([s for s, _r, _w in variants])
        assert sizes == [384], f"other-model files leaked into discovery: {variants}"
        # All paths must point at Qwen3-VL-4B (not embedding)
        for _s, rknn_p, _w in variants:
            assert "Qwen3-VL-4B" in rknn_p
            assert "Embedding" not in rknn_p
        print("[discover] prefix-scoped ->", variants)


def test_pick_variant_closest_by_longest_side():
    """``VLMBackend._pick_variant`` picks the variant whose image_size
    is closest to the input image's longest side; ties broken by smaller size.
    """
    from wdnai.backends.vlm import _VisionVariant

    class _StubRknn:
        # ``_VisionVariant`` doesn't call any RKNN3Lite methods during
        # ``_pick_variant``; this stub is only here to satisfy the
        # constructor.
        pass

    fake_backend = type(
        "_StubBackend",
        (),
        {"_vision_variants": [
            _VisionVariant(image_size=384, rknn=_StubRknn(),
                           rknn_path="/fake/384.rknn", weight_path="/fake/384.weight"),
            _VisionVariant(image_size=768, rknn=_StubRknn(),
                           rknn_path="/fake/768.rknn", weight_path="/fake/768.weight"),
        ]},
    )()

    # ``_pick_variant`` is a method — bind it to the stub
    from wdnai.backends.vlm import VLMBackend
    pick = VLMBackend._pick_variant.__get__(fake_backend, type(fake_backend))

    # Tiny image → 384 (closer than 768)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    v = pick(img)
    assert v.image_size == 384, f"tiny image should pick 384, got {v.image_size}"

    # High-res image (1500 wide) → 768 (closer than 384)
    img = np.zeros((900, 1500, 3), dtype=np.uint8)
    v = pick(img)
    assert v.image_size == 768, f"1500-wide image should pick 768, got {v.image_size}"

    # Mid image (576 square) → 384: |384-576|=192, |768-576|=192 → tie → smaller wins
    img = np.zeros((576, 576, 3), dtype=np.uint8)
    v = pick(img)
    assert v.image_size == 384, f"576-square tie should pick 384, got {v.image_size}"

    # Boundary: 600-wide → 768 wins (|384-600|=216, |768-600|=168)
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    v = pick(img)
    assert v.image_size == 768, f"600-wide should pick 768, got {v.image_size}"

    # Portrait image: max(h, w) drives the decision
    img = np.zeros((2000, 500, 3), dtype=np.uint8)  # 2000 tall
    v = pick(img)
    assert v.image_size == 768, f"2000-tall portrait should pick 768, got {v.image_size}"

    print("[pick] longest-side selection ok")


def test_pick_variant_single_variant_short_circuits():
    """A single-variant backend always returns its only variant, regardless of size."""
    from wdnai.backends.vlm import _VisionVariant, VLMBackend

    class _StubRknn:
        pass

    fake_backend = type(
        "_StubBackend",
        (),
        {"_vision_variants": [
            _VisionVariant(image_size=384, rknn=_StubRknn(),
                           rknn_path="/fake/384.rknn", weight_path="/fake/384.weight"),
        ]},
    )()

    pick = VLMBackend._pick_variant.__get__(fake_backend, type(fake_backend))
    for hw in [(100, 100), (640, 480), (1920, 1080), (50, 50)]:
        img = np.zeros((hw[1], hw[0], 3), dtype=np.uint8)
        v = pick(img)
        assert v.image_size == 384, f"single-variant: {hw} -> {v.image_size}"
    print("[pick] single-variant short-circuit ok")


# --------------------------------------------------------------------------- #
def test_available_image_sizes_returns_variant_sizes():
    """``VLMBackend.available_image_sizes()`` returns the sorted
    list of variant image sizes — used by wdnwebui's 音视频描述
    page ``#vd-detect-size`` dropdown to populate dynamically."""
    from wdnai.backends.vlm import _VisionVariant, VLMBackend

    class _StubRknn:
        pass

    # Two-variant backend
    backend = object.__new__(VLMBackend)
    backend._vision_variants = [
        _VisionVariant(image_size=384, rknn=_StubRknn(),
                       rknn_path="/f/384.rknn", weight_path="/f/384.weight"),
        _VisionVariant(image_size=768, rknn=_StubRknn(),
                       rknn_path="/f/768.rknn", weight_path="/f/768.weight"),
    ]
    sizes = backend.available_image_sizes()
    assert sizes == [384, 768], f"expected [384, 768], got {sizes}"

    # Single-variant backend (legacy single-resolution install)
    backend._vision_variants = [
        _VisionVariant(image_size=384, rknn=_StubRknn(),
                       rknn_path="/f/384.rknn", weight_path="/f/384.weight"),
    ]
    sizes = backend.available_image_sizes()
    assert sizes == [384], f"expected [384], got {sizes}"

    print("[available_image_sizes] returns variant sizes sorted")


def test_pick_variant_image_size_override_exact_match():
    """``_pick_variant(image_size=N)`` returns the variant whose
    ``image_size`` matches ``N`` exactly — no auto-pick heuristic
    is involved."""
    from wdnai.backends.vlm import _VisionVariant, VLMBackend

    class _StubRknn:
        pass

    backend = object.__new__(VLMBackend)
    backend._vision_variants = [
        _VisionVariant(image_size=384, rknn=_StubRknn(),
                       rknn_path="/f/384.rknn", weight_path="/f/384.weight"),
        _VisionVariant(image_size=768, rknn=_StubRknn(),
                       rknn_path="/f/768.rknn", weight_path="/f/768.weight"),
    ]
    pick = VLMBackend._pick_variant.__get__(backend, type(backend))

    # Exact match → returns that variant even when an auto-pick would
    # have picked the other (here, an 800x600 frame would auto-pick
    # 768, but image_size=384 must override).
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    v = pick(img, image_size=384)
    assert v.image_size == 384, f"override should win, got {v.image_size}"

    v = pick(img, image_size=768)
    assert v.image_size == 768, f"explicit 768 should win, got {v.image_size}"

    # Single-variant backend: image_size override is irrelevant; the
    # only variant is returned regardless of requested size.
    backend._vision_variants = [
        _VisionVariant(image_size=384, rknn=_StubRknn(),
                       rknn_path="/f/384.rknn", weight_path="/f/384.weight"),
    ]
    v = pick(image_size=768)
    assert v.image_size == 384, f"single-variant: any size → 384, got {v.image_size}"

    print("[pick_variant] image_size override exact match")


def test_preprocess_to_variant_dispatches_by_n_dims():
    """``_preprocess_to_variant`` must inspect ``in_attr.n_dims`` at
    runtime, not ``spec.use_prune``, because the 4B exports in this
    env have different export formats per resolution:

      * 384 file (full NHWC):  ``in_attr.n_dims == 4``  → NHWC
      * 768 file (pruned):    ``in_attr.n_dims != 4``  → patchified

    Mismatching the two is what caused the regression with the
    previous ``use_prune=False`` dispatch.
    """
    from types import SimpleNamespace
    from wdnai.backends.vlm import _VisionVariant, VLMBackend

    class _StubRknn:
        def get_inputs_tensor_attr(self):
            # The 384/768 stubs report different ``n_dims`` so we
            # can verify that the dispatch key in
            # ``_preprocess_to_variant`` follows the runtime query,
            # not a preset config flag.
            if self._image_size == 384:
                return [SimpleNamespace(n_dims=4, shape=(1, 384, 384, 3), dtype=0)]
            return [SimpleNamespace(n_dims=2, shape=(2304, 1536), dtype=0)]

    v_384 = _VisionVariant(
        image_size=384, rknn=_StubRknn(), rknn_path="/f/384.rknn", weight_path="/f/384.weight",
    )
    v_384.rknn._image_size = 384
    v_768 = _VisionVariant(
        image_size=768, rknn=_StubRknn(), rknn_path="/f/768.rknn", weight_path="/f/768.weight",
    )
    v_768.rknn._image_size = 768

    backend = object.__new__(VLMBackend)
    backend.spec = SimpleNamespace(use_prune=False)   # preset says full
    backend.variant = VLMBackend.VARIANT_3VL
    backend._vision_variants = [v_384, v_768]

    # Patch the prune helper so we can assert it was (or was not)
    # called without importing the real rknn. Leave ``resize`` real
    # so the input is actually resized to the variant's image_size
    # before ``_preprocess_to_variant`` reshapes it.
    import wdnai.backends.vlm as vlm_mod
    calls = {"prune": 0}
    def _fake_prune(_nchw):
        calls["prune"] += 1
        return np.zeros((1, 1), dtype=np.float32)
    orig_prune = vlm_mod.qwen3vl_prune_single
    vlm_mod.qwen3vl_prune_single = _fake_prune
    try:
        img = np.zeros((100, 100, 3), dtype=np.uint8)  # arbitrary source size
        # 384 variant (n_dims=4) → must take the NHWC raw path,
        # NOT the prune path.  Shape assertion below confirms.
        out = backend._preprocess_to_variant(img, v_384)
        assert out.shape == (1, 384, 384, 3), (
            f"384 n_dims=4 should produce (1,384,384,3) NHWC, got {out.shape}"
        )
        assert calls["prune"] == 0, "384 path must NOT call qwen3vl_prune_single"

        # 768 variant (n_dims=2) → must take the prune / patchified
        # path.  Shape assertion confirms the prune helper was called
        # and produced a 2-D (num_patches, embed_dim) tensor.
        out = backend._preprocess_to_variant(img, v_768)
        assert out.ndim == 2, f"768 n_dims=2 should produce 2-D tensor, got ndim={out.ndim}"
        assert calls["prune"] == 1, "768 path must call qwen3vl_prune_single exactly once"
    finally:
        vlm_mod.qwen3vl_prune_single = orig_prune
    print("[preprocess] dispatches by in_dims (384=NHWC, 768=pruned)")


def test_letterbox_resize_preserves_aspect_ratio():
    """``letterbox_resize`` must NEVER stretch — non-square inputs get
    centred in a square canvas with grey padding, and the scaled
    content keeps its original aspect ratio exactly.
    """
    from wdnai.utils import letterbox_resize

    # ---- 1. Wide plate-strip: 100 (h) × 400 (w), target 384 ----
    img_wide = np.zeros((100, 400, 3), dtype=np.uint8)
    img_wide[..., 0] = 200  # blue strip, easy to spot
    out = letterbox_resize(img_wide, 384)
    assert out.shape == (384, 384, 3), f"wide: expected (384,384,3), got {out.shape}"

    # Scaled content width: 400/4 * 384 = 384 (already fills the long side)
    # Scaled content height: 100/4 * 384 = 96
    # Letterbox padding: (384-96)//2 = 144 rows on top & bottom; 0 cols L/R
    expected_nh = round(100 * 384 / 400)
    expected_pad_top = (384 - expected_nh) // 2
    pad_left = (384 - expected_nh) // 2  # the loop sets x0 from nw, not nh; recompute correctly
    # The strip is wider than tall (nw == 384, nh < 384), so pad is
    # vertical only (top/bottom) and L/R are zero.
    pad_h_each = expected_pad_top
    # Top pad rows should be fill grey (114,114,114).
    assert np.all(out[:pad_h_each, :, :] == 114), \
        "wide: top pad rows must be grey fill (114)"
    assert np.all(out[pad_h_each + expected_nh:, :, :] == 114), \
        "wide: bottom pad rows must be grey fill (114)"
    # Left & right columns within the content band are NOT pad (full width).
    # Content band's aspect ratio: 384/expected_nh ≈ 400/100.
    content = out[pad_h_each:pad_h_each + expected_nh]
    blue_rows = np.where(content[..., 0] == 200)[0]
    assert blue_rows.size > 0, "wide: original blue content must survive inside the letterbox"
    # Aspect ratio preserved: width / height of the surviving content ≈ 4:1.
    bw = float(content.shape[1])
    bh = float(content.shape[0])
    assert abs(bw / bh - 4.0) < 0.05, \
        f"wide: aspect ratio must be preserved (~4:1), got {bw/bh:.2f}"

    # ---- 2. Tall portrait: 400 (h) × 100 (w), target 384 ----
    img_tall = np.zeros((400, 100, 3), dtype=np.uint8)
    img_tall[..., 1] = 150  # green
    out = letterbox_resize(img_tall, 384)
    assert out.shape == (384, 384, 3)
    expected_nw = round(100 * 384 / 400)
    pad_w_each = (384 - expected_nw) // 2
    assert np.all(out[:, :pad_w_each, :] == 114), "tall: left pad must be grey fill"
    assert np.all(out[:, pad_w_each + expected_nw:, :] == 114), "tall: right pad must be grey fill"
    content = out[:, pad_w_each:pad_w_each + expected_nw]
    bw, bh = float(content.shape[1]), float(content.shape[0])
    assert abs(bw / bh - 0.25) < 0.05, \
        f"tall: aspect ratio must be preserved (~1:4), got {bw/bh:.2f}"

    # ---- 3. Already square at target size: idempotent + writable copy ----
    img_sq = np.full((384, 384, 3), 200, dtype=np.uint8)
    out = letterbox_resize(img_sq, 384)
    assert out.shape == (384, 384, 3)
    assert np.all(out == 200), "square: must pass-through (no padding needed)"
    out[0, 0] = 0  # mutating the result must NOT mutate the input
    assert int(img_sq[0, 0, 0]) == 200, "square: result must be a copy, not a view"

    # ---- 4. Upscale: 50x50 → 768 (longest side = 50 < 768) ----
    small = np.full((50, 50, 3), 50, dtype=np.uint8)
    out = letterbox_resize(small, 768)
    assert out.shape == (768, 768, 3)
    pad_each = (768 - 50) // 2  # scale = 768/50, content should be 768×768? No — 50x50 is square → no pad.
    # A square input never gets padding, regardless of size.
    assert np.all(out == 50), "square small → 768: must fill the whole canvas"

    print("[letterbox] preserves aspect ratio, fills with grey (114)")


def test_letterbox_resize_embeds_into_preprocess():
    """Verify ``VLMBackend._preprocess_to_variant`` now letterboxes
    instead of stretching — i.e. feeding a wide plate-strip yields a
    square that still contains the original colours in a centred band.
    """
    from types import SimpleNamespace
    from wdnai.backends.vlm import _VisionVariant, VLMBackend

    class _StubRknn:
        def get_inputs_tensor_attr(self):
            return [SimpleNamespace(n_dims=4, shape=(1, 384, 384, 3), dtype=0)]
        def inference(self, inputs=None):
            return [np.zeros((1, 384, 384, 16), dtype=np.float16)]

    v = _VisionVariant(
        image_size=384, rknn=_StubRknn(),
        rknn_path="/f/384.rknn", weight_path="/f/384.weight",
    )
    backend = object.__new__(VLMBackend)
    backend.spec = SimpleNamespace(use_prune=False)
    backend.variant = VLMBackend.VARIANT_3VL
    backend._vision_variants = [v]

    # Wide plate strip: 60 (h) × 240 (w), all red.
    # Pass as BGR (name ``img_bgr`` in the SDK signature); the SDK
    # converts internally to RGB before letterboxing.
    strip_bgr = np.zeros((60, 240, 3), dtype=np.uint8)
    strip_bgr[..., 2] = 200  # red channel in BGR (index 2)
    out = backend._preprocess_to_variant(strip_bgr, v)
    assert out.shape == (1, 384, 384, 3)
    out_sq = out[0]
    # After BGR->RGB the red channel is now at index 0.
    # The strip's aspect ratio is 4:1, so the letterbox content band
    # is the full 384 wide and 96 tall, centred vertically.
    expected_nh = round(60 * 384 / 240)  # 96
    pad_each = (384 - expected_nh) // 2
    # Above the content: grey fill (114,114,114).
    assert np.all(out_sq[:pad_each, :, :] == 114), \
        "preprocess: top letterbox pad must be grey (114)"
    assert np.all(out_sq[pad_each + expected_nh:, :, :] == 114), \
        "preprocess: bottom letterbox pad must be grey (114)"
    # Inside the content band the original colour (red) must be present.
    content = out_sq[pad_each:pad_each + expected_nh]
    red_rows = np.where(content[..., 0] == 200)[0]
    assert red_rows.size > 0, \
        "preprocess: original red content must survive inside the letterbox"
    print("[letterbox] _preprocess_to_variant letterboxes wide strips")


def test_unletterbox_bbox_round_trip():
    """``unletterbox_bbox`` must map a 0-1000 bbox in the letterboxed
    canvas back to original-image pixels, accounting for the scale and
    the centred padding.

    The canonical pattern is: caller pre-letterboxes with
    ``letterbox_resize_with_meta``, sends the canvas to the VLM, gets
    back a bbox in 0-1000 of **the canvas** (with grey padding), then
    maps that bbox to original-frame pixels so the ROI crop has the
    original aspect ratio and no grey border.
    """
    from wdnai.utils import letterbox_resize_with_meta, unletterbox_bbox

    # ---- 1. 16:9 source (1920x1080) → 384x384 canvas, wide case ----
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    canvas, meta = letterbox_resize_with_meta(img, 384)
    assert canvas.shape == (384, 384, 3)
    assert meta.size == 384
    assert meta.orig_h == 1080 and meta.orig_w == 1920
    # 1920 > 1080 → scale = 384/1920 = 0.2, content_w = 384, content_h = 216
    assert abs(meta.scale - 0.2) < 1e-6, f"16:9 scale: expected 0.2, got {meta.scale}"
    assert meta.content_w == 384
    assert meta.content_h == 216
    assert meta.pad_top == (384 - 216) // 2  # 84
    assert meta.pad_left == 0

    # Bbox covers the entire content area in 0-1000 of the canvas:
    #   - canvas pixels: (0, 84) → (384, 300)  (full width, content height)
    #   - 0-1000 of canvas: x=[0, 1000], y=[84*1000/384, 300*1000/384]
    canvas_y_top = 84 / 384 * 1000  # ~218.75
    canvas_y_bot = 300 / 384 * 1000  # ~781.25
    bbox_canvas = [0, canvas_y_top, 1000, canvas_y_bot]
    bbox_orig = unletterbox_bbox(bbox_canvas, meta)
    # Expected: full width, full height
    assert abs(bbox_orig[0] - 0) < 1.0, f"x1 expected ~0, got {bbox_orig[0]}"
    assert abs(bbox_orig[1] - 0) < 1.0, f"y1 expected ~0, got {bbox_orig[1]}"
    assert abs(bbox_orig[2] - 1920) < 1.0, f"x2 expected ~1920, got {bbox_orig[2]}"
    assert abs(bbox_orig[3] - 1080) < 1.0, f"y2 expected ~1080, got {bbox_orig[3]}"

    # Bbox straddles the top grey padding (y=50..100 of 1000 lands at
    # canvas rows 19..38, which is inside the 84-row top padding). The
    # ``y`` coords must clamp to ``0``; the ``x`` coords stay valid
    # (they're in the content band because pad_left == 0).
    bbox_in_pad = [400, 50, 600, 100]
    clamped = unletterbox_bbox(bbox_in_pad, meta)
    assert clamped[1] == 0.0 and clamped[3] == 0.0, \
        f"bbox y must clamp to 0 (in pad), got {clamped}"
    # x1=400/1000*384/0.2 = 768, x2=600/1000*384/0.2 = 1152
    assert abs(clamped[0] - 768.0) < 1.0, f"x1 expected ~768, got {clamped[0]}"
    assert abs(clamped[2] - 1152.0) < 1.0, f"x2 expected ~1152, got {clamped[2]}"

    # ---- 2. 9:16 source (1080x1920) → 384x384 canvas, tall case ----
    img_tall = np.zeros((1920, 1080, 3), dtype=np.uint8)
    canvas, meta = letterbox_resize_with_meta(img_tall, 384)
    assert abs(meta.scale - 0.2) < 1e-6
    assert meta.content_w == 216 and meta.content_h == 384
    assert meta.pad_top == 0
    assert meta.pad_left == (384 - 216) // 2  # 84

    # Bbox covering the entire content area in 0-1000 of canvas
    bbox_canvas = [84/384*1000, 0, (84+216)/384*1000, 1000]
    bbox_orig = unletterbox_bbox(bbox_canvas, meta)
    assert abs(bbox_orig[0]) < 1.0
    assert abs(bbox_orig[1]) < 1.0
    assert abs(bbox_orig[2] - 1080) < 1.0
    assert abs(bbox_orig[3] - 1920) < 1.0

    # ---- 3. Square source 800x800 → 384x384 canvas (downsampled) ----
    # 800 > 384 → downsample, scale = 384/800 = 0.48, no padding.
    img_sq = np.zeros((800, 800, 3), dtype=np.uint8)
    canvas, meta = letterbox_resize_with_meta(img_sq, 384)
    assert meta.content_w == 384 and meta.content_h == 384
    assert meta.pad_top == 0 and meta.pad_left == 0
    assert abs(meta.scale - 0.48) < 1e-6, f"800→384 scale: expected 0.48, got {meta.scale}"
    # 0-1000 bbox (200, 300, 700, 900) in the canvas maps back to the
    # 800x800 frame at (200/1000*384 / 0.48, ...) = (160, ...).
    bbox_canvas = [200, 300, 700, 900]
    bbox_orig = unletterbox_bbox(bbox_canvas, meta)
    inv = 1.0 / meta.scale
    assert abs(bbox_orig[0] - 200/1000*384*inv) < 1.0
    assert abs(bbox_orig[1] - 300/1000*384*inv) < 1.0
    assert abs(bbox_orig[2] - 700/1000*384*inv) < 1.0
    assert abs(bbox_orig[3] - 900/1000*384*inv) < 1.0

    # ---- 3b. Square source at exact canvas size: identity scale ----
    img_eq = np.zeros((384, 384, 3), dtype=np.uint8)
    canvas, meta = letterbox_resize_with_meta(img_eq, 384)
    assert meta.scale == 1.0
    bbox_canvas = [200, 300, 700, 900]
    bbox_orig = unletterbox_bbox(bbox_canvas, meta)
    assert abs(bbox_orig[0] - 200/1000*384) < 1.0
    assert abs(bbox_orig[2] - 700/1000*384) < 1.0

    # ---- 4. clamp=False keeps negative / oversize coords ----
    bbox_neg = [10, 10, 20, 20]   # inside top-left padding of a 16:9 letterbox
    unclamped = unletterbox_bbox(bbox_neg, meta, clamp=False)
    # meta here is the 800x800→384 case (no padding), so the result is
    # just the canvas-pixel scaled. Just confirm it didn't clamp.
    assert unclamped == [10/1000*384, 10/1000*384, 20/1000*384, 20/1000*384]

    print("[unletterbox_bbox] round-trip preserves canvas↔original mapping")


def test_pick_variant_image_size_override_nearest_fallback():
    """When ``image_size`` doesn't match any variant, fall back to
    nearest (smaller on tie) and log a warning — preserves the
    safety net for legacy single-resolution installs that don't have
    the requested resolution."""
    import logging as _logging
    from wdnai.backends.vlm import _VisionVariant, VLMBackend

    class _StubRknn:
        pass

    backend = object.__new__(VLMBackend)
    backend._vision_variants = [
        _VisionVariant(image_size=384, rknn=_StubRknn(),
                       rknn_path="/f/384.rknn", weight_path="/f/384.weight"),
        _VisionVariant(image_size=768, rknn=_StubRknn(),
                       rknn_path="/f/768.rknn", weight_path="/f/768.weight"),
    ]
    pick = VLMBackend._pick_variant.__get__(backend, type(backend))

    # 512 not exposed → nearest wins. |384-512|=128 < |768-512|=256 → 384.
    v = pick(image_size=512)
    assert v.image_size == 384, f"512 not exposed → nearest 384, got {v.image_size}"

    # 1000 closer to 768 than 384 → 768.
    v = pick(image_size=1000)
    assert v.image_size == 768, f"1000 → 768, got {v.image_size}"

    # 576 (tie) → smaller wins → 384.
    v = pick(image_size=576)
    assert v.image_size == 384, f"576 tie → 384, got {v.image_size}"

    print("[pick_variant] nearest fallback works for unknown sizes")


def test_run_images_forwards_image_size():
    """``VLMBackend.run_images(image_size=N)`` is the entry point the
    wdnwebui detector hits. The kwarg flows into ``_pick_variant``
    and overrides auto-pick."""
    from wdnai.backends.vlm import _VisionVariant, VLMBackend

    class _StubRknn:
        def get_inputs_tensor_attr(self):
            from types import SimpleNamespace
            return [SimpleNamespace(n_dims=4, shape=(1, 384, 384, 3), dtype=0)]
        def get_outputs_tensor_attr(self):
            return [SimpleNamespace(aligned_size=4096, dtype=0)]
        def inference(self, inputs):
            return [np.zeros((1, 32), dtype=np.float32)]

    backend = object.__new__(VLMBackend)
    backend.spec = type("S", (), {
        "name": "fake", "variant": "qwen3-vl", "image_size": 384,
        "use_prune": False,
    })()
    backend.variant = "qwen3-vl"
    backend._vision_variants = [
        _VisionVariant(image_size=384, rknn=_StubRknn(),
                       rknn_path="/f/384.rknn", weight_path="/f/384.weight"),
        _VisionVariant(image_size=768, rknn=_StubRknn(),
                       rknn_path="/f/768.rknn", weight_path="/f/768.weight"),
    ]
    backend.rknn_vision = backend._vision_variants[0].rknn
    backend.tokenizer = None
    backend.embeds = None
    backend.sink = None

    # Spy on ``_pick_variant`` to capture the ``image_size=`` kwarg
    # forwarded by ``_build_image_inputs``. The kwarg is consumed
    # inside ``_pick_variant`` to choose the variant, then
    # ``_vision_infer`` is called with the pinned variant (no
    # further image_size arg).
    pick_calls = []
    orig_pick = backend._pick_variant
    def spy_pick(img_bgr=None, *, image_size=None):
        pick_calls.append({"img_shape": img_bgr.shape if img_bgr is not None else None,
                           "image_size": image_size})
        return orig_pick(img_bgr, image_size=image_size)
    backend._pick_variant = spy_pick

    # 800x600 frame + image_size=384 → the chosen variant must be 384
    # (override wins even though auto-pick would have chosen 768).
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    backend._build_image_inputs([img], image_size=384)

    # The first call (auto-pick based on first image) carries
    # image_size=384; subsequent calls from the loop may carry
    # image_size=None (they fall back to auto-pick per image but
    # force-fit to the chosen variant).
    assert pick_calls, "_pick_variant was not called"
    assert pick_calls[0]["image_size"] == 384, (
        f"_build_image_inputs must forward image_size=384 to "
        f"_pick_variant, got {pick_calls[0]['image_size']}"
    )

    backend._pick_variant = orig_pick
    print("[run_images] image_size=384 forwarded to _pick_variant")


# --------------------------------------------------------------------------- #
def test_strip_thinking_blocks_basic():
    """``strip_thinking_blocks`` removes ``<think>...`` and ``<tool_call>...</tool_call>`` blocks."""
    from wdnai.utils import strip_thinking_blocks

    # Plain think block — single-pass
    assert strip_thinking_blocks(
        '<think>reasoning</think>{"text":"ok"}'
    ) == '{"text":"ok"}'

    # Empty think block (the user's actual bug) + JSON fence
    assert strip_thinking_blocks(
        '<think>\n\n</think>\n\n```json\n{"text":"plate"}\n```'
    ) == '```json\n{"text":"plate"}\n```'

    # Multi-block (some Qwen variants emit two think rounds)
    assert strip_thinking_blocks(
        '<think>a</think>foo<think>b</think>bar'
    ) == "foobar"

    # XML tool_call wrapper — only the tags are stripped; the JSON
    # payload inside is preserved so ``extract_json`` can find it.
    assert strip_thinking_blocks(
        '<tool_call>{"name":"x","arguments":{}}</tool_call>keep'
    ) == '{"name":"x","arguments":{}}keep'

    # Idempotent on plain text
    assert strip_thinking_blocks("plain text") == "plain text"

    # Empty / None-ish
    assert strip_thinking_blocks("") == ""
    assert strip_thinking_blocks(None) is None  # type: ignore[arg-type]

    # Unclosed think tag — non-greedy ``.*?`` reaches end-of-string and
    # the closing ``(?:</think>|$)`` alt matches ``$`` so the whole
    # trailing block is consumed.
    assert strip_thinking_blocks("<think>only open") == ""

    print("[strip] basic cases OK")


def test_extract_json_strips_think_blocks():
    """``extract_json`` finds the JSON fence AFTER stripping any leading think block."""
    from wdnai.utils import extract_json

    # Think block + JSON fence: must extract the JSON, not a {} inside
    # the think block (which doesn't exist here, but the regex would
    # otherwise be confused if it did).
    out = extract_json(
        '<think>reasoning with {} in it</think>\n```json\n{"text":"plate"}\n```'
    )
    assert out == {"text": "plate"}, f"got {out!r}"

    # Think block + bare JSON
    out = extract_json('<think>r</think>{"text":"plate"}')
    assert out == {"text": "plate"}, f"got {out!r}"

    # Think block + plain text (no JSON anywhere) → None
    assert extract_json("<think>reasoning</think>no json here") is None

    print("[extract_json] think-block stripping OK")


def test_extract_text_from_mismatched_dict_recurses():
    """``_extract_text_from_mismatched_dict`` drills into ``arguments`` / ``response`` / ``data``."""
    from wdnai.session import _extract_text_from_mismatched_dict

    # Top-level hit (legacy case) — unchanged behaviour
    assert _extract_text_from_mismatched_dict({"text": "hi"}) == "hi"
    assert _extract_text_from_mismatched_dict({"description": "hi"}) == "hi"

    # Nested inside ``arguments`` (the tool-call-wrap case)
    assert _extract_text_from_mismatched_dict(
        {"name": "ocr", "arguments": {"text": "京A88***"}}
    ) == "京A88***"

    # Nested inside ``response`` (some models wrap in response/data)
    assert _extract_text_from_mismatched_dict(
        {"response": {"content": "answer"}}
    ) == "answer"
    assert _extract_text_from_mismatched_dict(
        {"data": {"summary": "answer"}}
    ) == "answer"

    # Depth cap: 2 levels deep is OK, 3+ returns None
    assert _extract_text_from_mismatched_dict(
        {"a": {"b": {"c": {"text": "deep"}}}}
    ) is None

    # Non-dict input → None
    assert _extract_text_from_mismatched_dict("not a dict") is None  # type: ignore[arg-type]
    assert _extract_text_from_mismatched_dict(None) is None  # type: ignore[arg-type]

    # Skip-list includes tool-call identifier keys so they don't
    # shadow the real content via the "first non-trivial string" path.
    # ``name="x"`` is short (1 char), so the existing len>=4 check
    # already filters it; this test pins that behaviour explicitly.
    out = _extract_text_from_mismatched_dict(
        {"name": "tool", "arguments": {"content": "real answer"}}
    )
    assert out == "real answer", f"tool-identifier 'name' leaked through: {out!r}"

    print("[extract_text] recursion + skip-list OK")


def test_safe_validate_raw_text_strips_think():
    """``_safe_validate``'s raw-text fallback strips ``<think>`` blocks defensively."""
    from pydantic import BaseModel, Field
    from wdnai.session import _safe_validate

    class Event(BaseModel):
        text: str = Field(description="event description")

    # Valid case — should pass straight through.
    out = _safe_validate(Event, {"text": "ok"}, "<think>reasoning</think>")
    assert out.text == "ok"

    # Schema-mismatch + raw_text has a think block: fallback to raw_text
    # but the think block MUST be stripped. Before the fix the model
    # would see ``Event(text='<think>reasoning\n\n...')`` — full
    # reasoning dumped into the schema's ``text`` field.
    bad = {"name": "ocr", "arguments": {"text": "answer"}}  # no top-level text
    raw = '<think>\nreasoning\n</think>\n```json\n{"name":"ocr","arguments":{"text":"answer"}}\n```'
    out = _safe_validate(Event, bad, raw)
    # ``arguments.text`` is recovered via the recursive helper, then
    # ``strip_thinking_blocks`` strips any think content for safety.
    assert out.text == "answer", f"got {out.text!r}"

    # When even recursion can't find anything (the JSON really is
    # garbage), we fall back to the raw text — but still strip the
    # think block.
    bad2 = {"random_key": [1, 2, 3]}
    raw2 = "<think>long internal monologue</think>real text after"
    out2 = _safe_validate(Event, bad2, raw2)
    assert out2.text == "real text after", f"got {out2.text!r}"

    print("[_safe_validate] raw-text fallback strips think OK")


def _build_session_with_fake_output(text: str):
    """Build a LLMSession whose ``_run_llm`` returns ``text`` verbatim."""
    import asyncio
    from wdnai import LLMSession, SystemLanguageModel

    embed, tok_dir = _make_workspace()
    _install_mocks(deepstack_n=3, vocab_size=64)
    spec = SystemLanguageModel.qwen3_vl_2b().with_paths(
        rknn_llm_path="/fake_llm.rknn3",
        rknn_vision_path="/fake_v.rknn3",
        tokenizer_path=tok_dir, embed_path=embed,
    )
    sess = LLMSession(model=spec)

    # Monkey-patch the actual call path so the model returns ``text``.
    # ``VLMBackend.run_images`` is the async entry; we patch
    # ``_run_llm`` which is the synchronous inner call.
    sess._backend._run_llm = lambda *a, **kw: text  # type: ignore[assignment]

    return sess


def test_respond_thinking_block_stripped_in_schema():
    """respond() with ``generating=Event`` strips a leading think
    block AND extracts the JSON cleanly — the schema's ``text``
    field carries the answer, not the think content.
    """
    import asyncio
    from pydantic import BaseModel, Field

    class Event(BaseModel):
        text: str = Field(description="event description")

    fake_text = '<think>reasoning</think>\n```json\n{"text":"answer"}\n```'
    sess = _build_session_with_fake_output(fake_text)
    try:
        out = asyncio.run(sess.respond(
            prompt="describe",
            attachment=__import__("wdnai").ImageAttachment(
                np.zeros((480, 640, 3), dtype=np.uint8)
            ),
            generating=Event,
        ))
        assert isinstance(out, Event), f"want Event, got {type(out).__name__}"
        assert out.text == "answer", f"want 'answer', got {out.text!r}"
    finally:
        sess.close()
    print("[respond] think block stripped before schema parse OK")


def test_respond_hallucinated_tool_name_recovers_schema():
    """respond() with ``tools=[Crop]`` + ``generating=Event``: model
    emits a tool-call JSON with a hallucinated name (``"ocr"``).
    Before the fix, the dispatch silently returned None and the
    schema-fallback path stuffed the raw text (think block included)
    into ``Event.text``. After the fix, we try ``arguments`` against
    the schema and recover the real answer.
    """
    import asyncio
    from typing import Annotated, Literal
    from pydantic import BaseModel, Field
    from wdnai import Callable

    class Event(BaseModel):
        text: str = Field(description="plate number or '无'")

    @Callable(description="plate crop tool")
    class Crop:
        bbox: Annotated[list[float], "0-1000 bbox [x1,y1,x2,y2]"]

    # Exact reproduction of the user's bug report.
    fake_text = (
        '<think>\n\n</think>\n\n'
        '```json\n'
        '{"name": "ocr", "arguments": {"text": "京A88***"}}\n'
        '```'
    )
    sess = _build_session_with_fake_output(fake_text)
    try:
        out = asyncio.run(sess.respond(
            prompt="detect plate",
            attachment=__import__("wdnai").ImageAttachment(
                np.zeros((480, 640, 3), dtype=np.uint8)
            ),
            generating=Event,
            tools=[Crop],
        ))
        # ``name="ocr"`` is not registered, so Crop isn't picked;
        # the new fallback validates ``arguments={"text":"京A88***"}``
        # against Event → Event instance.
        assert isinstance(out, Event), f"want Event, got {type(out).__name__}: {out!r}"
        assert out.text == "京A88***", f"want '京A88***', got {out.text!r}"
        # Crucially, NOT the raw think-block+JSON blob.
        assert "<think>" not in out.text
        assert "```" not in out.text
    finally:
        sess.close()
    print("[respond] hallucinated tool name -> schema recovery OK")


def test_respond_hallucinated_tool_name_with_invalid_arguments_falls_through():
    """When the hallucinated tool name's ``arguments`` also doesn't
    match the schema, fall through to the generic JSON parse path —
    same as before the fix. We must not regress on existing
    behaviour."""
    import asyncio
    from typing import Annotated, Literal
    from pydantic import BaseModel, Field
    from wdnai import Callable

    class Event(BaseModel):
        text: str = Field(description="event description")

    @Callable(description="plate crop tool")
    class Crop:
        bbox: Annotated[list[float], "0-1000 bbox"]

    # Arguments don't validate as Event (no ``text`` field, has bogus key).
    fake_text = (
        '<think>r</think>```json\n'
        '{"name": "ocr", "arguments": {"foo": 1}}\n'
        '```'
    )
    sess = _build_session_with_fake_output(fake_text)
    try:
        out = asyncio.run(sess.respond(
            prompt="detect",
            attachment=__import__("wdnai").ImageAttachment(
                np.zeros((480, 640, 3), dtype=np.uint8)
            ),
            generating=Event,
            tools=[Crop],
        ))
        # The arguments-as-schema path fails; we then try the outer
        # dict against Event (also fails because no ``text``); finally
        # the raw text is wrapped (with think block stripped) into
        # Event.text — which is the legacy safe behaviour.
        assert isinstance(out, Event), f"want Event, got {type(out).__name__}: {out!r}"
        # Crucially, no think block leaked.
        assert "<think>" not in out.text
        assert "</think>" not in out.text
    finally:
        sess.close()
    print("[respond] invalid hallucinated args -> safe fallback OK")


def test_respond_tool_call_xml_wrapper_stripped():
    """When the model wraps the tool-call JSON in ``<tool_call>...</tool_call>``
    XML tags (some Qwen variants do this), the strip-then-extract
    pipeline recovers the JSON inside."""
    import asyncio
    from typing import Annotated, Literal
    from pydantic import BaseModel, Field
    from wdnai import Callable

    class Event(BaseModel):
        text: str = Field(description="event description")

    @Callable(description="plate crop tool")
    class Crop:
        bbox: Annotated[list[float], "0-1000 bbox"]

    fake_text = (
        '<tool_call>\n'
        '{"name": "Crop", "arguments": {"bbox": [100, 200, 300, 400]}}\n'
        '</tool_call>'
    )
    sess = _build_session_with_fake_output(fake_text)
    try:
        out = asyncio.run(sess.respond(
            prompt="detect",
            attachment=__import__("wdnai").ImageAttachment(
                np.zeros((480, 640, 3), dtype=np.uint8)
            ),
            tools=[Crop],
        ))
        # The XML wrapper is stripped; the inner JSON parses as a
        # valid Crop tool call → Crop instance returned.
        assert isinstance(out, Crop), f"want Crop, got {type(out).__name__}: {out!r}"
        assert out.bbox == [100, 200, 300, 400]
    finally:
        sess.close()
    print("[respond] XML tool_call wrapper stripped OK")


def test_extract_json_python_literal_fallback():
    """Qwen3.5-4B sometimes emits Python-style dict repr (``{'text':
    '…'}``) instead of strict JSON when the schema hint is short.
    ``extract_json`` should accept it via ``ast.literal_eval`` so the
    downstream schema-validation doesn't fall through to the raw-text
    fallback path.
    """
    from wdnai.utils import extract_json

    out = extract_json("{'text': '京A88***'}")
    assert out == {"text": "京A88***"}, f"got {out!r}"

    # Works inside a think block too (strip runs first).
    out = extract_json('<think>r</think>{"text": "京A88***"}')
    assert out == {"text": "京A88***"}

    # Lists too.
    out = extract_json("[1, 2, 3]")
    assert out == [1, 2, 3], f"got {out!r}"

    # Plain text must NOT trigger literal_eval — keeps the function
    # safe and avoids needless parse attempts.
    assert extract_json("hello world") is None
    # Only triggers on text that LOOKS like a literal — leading ``{``
    # / ``[`` / ``'`` / ``"``.
    assert extract_json("plain {not really json}") is None
    print("[extract_json] Python literal fallback OK")


def test_extract_text_handles_schema_envelope():
    """When the schema is echoed back as ``{"Event": {...}}`` (Qwen
    sometimes does this on the second-stage OCR call), the recursive
    extraction should drill into the nested dict instead of returning
    the bare key name "Event".
    """
    from wdnai.session import _extract_text_from_mismatched_dict

    # Top-level text wins over envelope nesting
    assert _extract_text_from_mismatched_dict(
        {"Event": {"text": "plate"}, "text": "override"}
    ) == "override"

    # Envelope case: drill into "Event" → "text"
    out = _extract_text_from_mismatched_dict({"Event": {"text": "plate"}})
    assert out == "plate", f"envelope case failed, got {out!r}"

    # Envelope + think in text field
    out = _extract_text_from_mismatched_dict(
        {"Event": {"text": "<think>r</think>real"}}
    )
    assert out == "real", f"got {out!r}"
    print("[extract_text] schema envelope handled OK")


def test_render_tools_block_uses_nous_fncall_format():
    """``render_tools_block`` emits the canonical NousFnCallPrompt
    (Hermes-style) tool prompt — the format Qwen3 / Qwen3.5 were SFT-ed
    on. This is what stops the model from hallucinating tool names
    like ``"ocr"``.

    Reference: qwen_agent/llm/fncall_prompts/nous_fncall_prompt.py:FN_CALL_TEMPLATE
    """
    from wdnai._callable import Callable as _Callable, render_tools_block

    @_Callable(description="plate crop tool")
    class Crop:
        bbox: Annotated[list[float], "0-1000 bbox"]

    specs = [{
        "name": "Crop",
        "description": "plate crop tool",
        "parameters": {
            "type": "object",
            "properties": {"bbox": {"type": "array", "items": {"type": "number"}}},
            "required": ["bbox"],
        },
    }]
    out = render_tools_block(specs)

    # Canonical elements per NousFnCallPrompt.FN_CALL_TEMPLATE
    assert "<tools>" in out and "</tools>" in out, "missing <tools> block"
    # Tool defs are OpenAI-compatible: {"type": "function", "function": {...}}
    assert '"type": "function"' in out, "missing type=function wrapper"
    assert '"name": "Crop"' in out, "tool name not literal (was placeholder)"
    assert "<tool_call>" in out and "</tool_call>" in out, (
        "missing <tool_call>...</tool_call> wrapper in call instruction"
    )
    # Legacy placeholder must be gone (it confuses Qwen3)
    assert "<tool_name>" not in out, "legacy <tool_name> placeholder still present"
    # Defensive: the call template should mention literal tool name (or
    # at least the literal-word "function-name") so the model sees a
    # real example of what to emit.
    assert "function-name" in out or "Crop" in out
    print("[render_tools_block] NousFnCallPrompt format OK")


def test_compose_prompt_choice_block_uses_tool_call_wrapper():
    """When both ``tools=`` and ``generating=`` are given, the
    "Pick one of two ways" block must reference the canonical
    ``<tool_call>...</tool_call>`` wrapper, not the legacy ``<tool_name>``
    placeholder.
    """
    from pydantic import BaseModel, Field
    from wdnai._callable import Callable as _Callable
    from wdnai.session import LLMSession

    class Event(BaseModel):
        text: str = Field(description="event")

    @_Callable(description="crop")
    class Crop:
        bbox: Annotated[list[float], "0-1000 bbox"]

    tool_specs = [{"name": "Crop", "description": "crop",
                   "parameters": {"type": "object", "properties": {}}}]
    full = LLMSession._compose_prompt("请检测车牌", Event, tool_specs)
    # Canonical wrapper present
    assert "<tool_call>" in full and "</tool_call>" in full
    # Legacy placeholder gone
    assert "<tool_name>" not in full
    # Tool block also has the canonical structure
    assert "<tools>" in full and "</tools>" in full
    # Schema hint still uses repr(dict) (legacy shape — see _safe_schema)
    assert "Event" in full, "schema hint must still mention Event"
    print("[compose_prompt] choice_block uses NousFnCallPrompt format OK")


def test_try_parse_tool_call_handles_hermes_string_arguments():
    """Hermes spec allows ``arguments`` to be a JSON string instead
    of an inline dict. ``try_parse_tool_call`` should transparently
    decode it so ``dispatch_tool_call`` always sees a dict.
    """
    from wdnai._callable import try_parse_tool_call

    # Hermes-style: arguments is a JSON-encoded string
    text = (
        '<tool_call>\n'
        '{"name": "Crop", "arguments": "{\\"bbox\\": [1, 2, 3, 4]}"}\n'
        '</tool_call>'
    )
    parsed = try_parse_tool_call(text)
    assert parsed == ("Crop", {"bbox": [1, 2, 3, 4]}), f"got {parsed!r}"

    # Legacy: arguments is inline dict — still works
    text2 = '{"name": "Crop", "arguments": {"bbox": [1, 2, 3, 4]}}'
    parsed2 = try_parse_tool_call(text2)
    assert parsed2 == ("Crop", {"bbox": [1, 2, 3, 4]}), f"got {parsed2!r}"
    print("[try_parse_tool_call] Hermes string-arguments decoded OK")


def test_safe_validate_fallback_strips_json_fence_and_tool_call_wrapper():
    """The schema-validation raw-text fallback must remove ````json```` fence
    AND ``<tool_call>``/``</tool_call>`` tags so the schema's ``text``
    field doesn't carry literal wrapper markup.
    """
    from pydantic import BaseModel, Field
    from wdnai.session import _safe_validate

    class Event(BaseModel):
        text: str = Field(description="event text")

    # Schema-mismatched JSON; raw_text has both ```json``` fence AND
    # ``<tool_call>`` wrapper — the fallback must clean both.
    bad = {"name": "ocr", "arguments": {"region": [10, 20, 30, 40]}}
    raw = (
        '<think>reasoning</think>'
        '```json\n'
        '<tool_call>\n'
        '{"name": "ocr", "arguments": {"region": [10, 20, 30, 40]}}\n'
        '</tool_call>\n'
        '```'
    )
    out = _safe_validate(Event, bad, raw)
    # No wrapper markup should leak
    assert "<think>" not in out.text
    assert "</think>" not in out.text
    assert "```" not in out.text, f"json fence leaked: {out.text!r}"
    assert "<tool_call>" not in out.text
    assert "</tool_call>" not in out.text
    # Inner JSON preserved (it's our best-guess for the model's
    # intent when schema validation genuinely fails)
    assert "ocr" in out.text or "region" in out.text
    print("[_safe_validate] fence + tool_call wrapper both stripped OK")


def test_render_tools_block_empty():
    """No tools → empty string (no spurious markup in prompt)."""
    from wdnai._callable import render_tools_block
    assert render_tools_block([]) == ""
    print("[render_tools_block] empty input → empty output OK")


def test_qwen3_5_vl_with_new_prompt_calls_crop_correctly():
    """End-to-end smoke test: with the new NousFnCallPrompt-format
    prompt, the mock backend that returns
    ``<tool_call>{"name": "Crop", "arguments": {"bbox": [...]}}</tool_call>``
    must be dispatched to a Crop instance.

    This pins the full contract: prompt format + strip pipeline +
    dispatch + schema validation. Without the prompt rewrite the
    model hallucinated ``"ocr"`` instead of the registered ``Crop``.
    """
    import asyncio
    from typing import Annotated
    from pydantic import BaseModel, Field
    from wdnai import ImageAttachment, Callable as _Callable

    class Event(BaseModel):
        text: str = Field(description="event text")

    @_Callable(description="plate crop tool")
    class Crop:
        bbox: Annotated[list[float], "0-1000 bbox"]

    # Model output in the new canonical Hermes format — this is
    # exactly what Qwen3.5-VL-4B emits when the prompt is correctly
    # formatted per NousFnCallPrompt.
    fake_text = (
        '<think>\nreasoning\n</think>'
        '<tool_call>\n'
        '{"name": "Crop", "arguments": {"bbox": [100, 200, 300, 400]}}\n'
        '</tool_call>'
    )
    sess = _build_session_with_fake_output(fake_text)
    try:
        out = asyncio.run(sess.respond(
            prompt="detect plate",
            attachment=ImageAttachment(np.zeros((480, 640, 3), dtype=np.uint8)),
            generating=Event,
            tools=[Crop],
        ))
        assert isinstance(out, Crop), (
            f"want Crop, got {type(out).__name__}: {out!r}"
        )
        assert out.bbox == [100, 200, 300, 400]
    finally:
        sess.close()
    print("[end-to-end] new prompt format dispatches Crop correctly")


# --------------------------------------------------------------------------- #
# LocateAnything grounding backend — mock-based end-to-end tests
# --------------------------------------------------------------------------- #
def _build_locate_anything_backend_for_test(num_tokens: int = 4, embed_dim: int = 64):
    """Build a LocateAnythingBackend backed by mocked RKNN3Lite handles.

    ``_install_mocks()`` only patches ``TextBackend.__init__`` /
    ``VLMBackend.__init__`` — it never tracks the
    ``LocateAnythingBackend`` LLM handle, so we register the sink to
    ``rknn_llm._sinks`` by hand. ``rknn_vision.inference`` is also
    overridden to return a synthetic ``(1, num_tokens, embed_dim)``
    fp16 feature so the backend's NHWC→fp16 cast path runs without
    complaining.
    """
    embed, tok_dir = _make_workspace()
    _install_mocks(deepstack_n=0, vocab_size=64)

    import wdnai.backends._common as common
    import wdnai.backends.locate_anything as la_mod
    la_mod.load_tokenizer = common.load_tokenizer
    la_mod.load_embed_table = common.load_embed_table
    la_mod.build_llm_callback = common.build_llm_callback

    from wdnai.model import _locate_anything_spec
    spec = _locate_anything_spec().with_paths(
        rknn_path="/fake_llm.rknn3",
        weight_path="/fake_llm.weight",
        rknn_vision_path="/fake_v.rknn3",
        weight_vision_path="/fake_v.weight",
        tokenizer_path=tok_dir,
        embed_path=embed,
    )

    from wdnai.backends.locate_anything import LocateAnythingBackend
    backend = LocateAnythingBackend(spec)
    backend.rknn_llm._sinks.append(backend.sink)

    def fake_vision_inference(inputs=None, **kw):
        return [np.zeros((1, num_tokens, embed_dim), dtype=np.float16)]
    backend.rknn_vision.inference = fake_vision_inference

    return backend


def _set_locate_anything_output(backend, text: str):
    """Replace ``rknn_llm.session_run`` so it injects ``text`` into the sink.

    Mirrors what the C-side ``make_result_callback`` would do at the end
    of a real run: extend the sink's token list with ASCII ord-of-char
    ids (filtering out non-ASCII), then write the desired text verbatim
    into ``sink.text``. Returns the ``(0, [stats])`` tuple the caller
    expects from ``session_run``.
    """
    def fake_session_run(inputs=None, prompt=None, embeds=None, **kw):
        tok = backend.tokenizer
        ids = [ord(c) % tok._vocab for c in text if ord(c) < 128]
        backend.sink.tokens.extend(ids)
        backend.sink.text = text
        return 0, [10, 5, 0.0, 0.001]
    backend.rknn_llm.session_run = fake_session_run


def _build_grounding_result(run_images_out):
    """Identity helper — :py:meth:`LocateAnythingBackend.run_images` already returns a ``GroundingResult``."""
    return run_images_out


def test_locate_anything_grounding_result_round_trip():
    """Full pipeline: vision RKNN → LLM RKNN → text → ``_parse_locate_output``.

    The synthetic model output ``<ref>person</ref><box><43><475><137><865></box>``
    must parse to one ``BoundingBox`` with ``label="person"`` and the
    four coords exactly as written.
    """
    backend = _build_locate_anything_backend_for_test()
    synthetic = "<ref>person</ref><box><43><475><137><865></box>"
    _set_locate_anything_output(backend, synthetic)

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    raw = backend.run_images("detect person", [img])
    result = _build_grounding_result(raw)

    assert result.text == synthetic, f"text round-trip mismatch: {result.text!r}"

    assert len(result.boxes) == 1, f"want 1 box, got {len(result.boxes)}"
    box = result.boxes[0]
    assert box.label == "person", f"label: {box.label!r}"
    assert box.x1 == 43, f"x1: {box.x1}"
    assert box.y1 == 475, f"y1: {box.y1}"
    assert box.x2 == 137, f"x2: {box.x2}"
    assert box.y2 == 865, f"y2: {box.y2}"

    assert result.level == "Normal", f"level: {result.level!r}"

    print("[locate_anything] grounding result round-trip OK")


def test_locate_anything_no_boxes_returns_empty_list():
    """A grounding call whose model response contains no ``<box>`` tags
    must produce ``boxes=[]`` and pass the original text through verbatim.
    """
    backend = _build_locate_anything_backend_for_test()
    plain = "no person detected in this image"
    _set_locate_anything_output(backend, plain)

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    raw = backend.run_images("describe", [img])
    result = _build_grounding_result(raw)

    assert result.text == plain, f"text passthrough: {result.text!r}"
    assert result.boxes == [], f"want [] boxes, got {result.boxes!r}"
    assert result.level == "Normal"
    print("[locate_anything] no-boxes → empty list OK")


def test_locate_anything_session_reset_clears_state():
    """Two consecutive ``run_images()`` calls. The second call must NOT
    carry any leftover text or tokens from the first call (or from manual
    sink pollution) — ``run_images()``'s internal ``self.sink.reset()``
    is what guarantees the fresh state.
    """
    backend = _build_locate_anything_backend_for_test()
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    first_text = "<ref>a</ref><box><10><20><30><40></box>"
    _set_locate_anything_output(backend, first_text)
    r1 = backend.run_images("first", [img])
    assert r1.text == first_text, f"first call: {r1.text!r}"
    assert backend.sink.text == first_text

    backend.sink.tokens.extend([100, 101, 102, 103])
    backend.sink.text = "STALE_LEFTOVER"
    assert "STALE" in backend.sink.text
    assert len(backend.sink.tokens) >= 4

    second_text = "<ref>b</ref><box><50><60><70><80></box>"
    _set_locate_anything_output(backend, second_text)
    r2 = backend.run_images("second", [img])

    assert r2.text == second_text, (
        f"reset failed — got {r2.text!r}"
    )
    assert "STALE" not in r2.text, (
        f"stale state leaked into second result: {r2.text!r}"
    )
    assert "<ref>a</ref>" not in r2.text, (
        f"first-call text leaked into second result: {r2.text!r}"
    )
    assert 100 not in backend.sink.tokens, (
        "manual pollution tokens survived the reset"
    )
    result = _build_grounding_result(r2)
    assert result.boxes[0].label == "b"
    assert result.boxes[0].x1 == 50
    print("[locate_anything] session reset clears state OK")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    test_text_qwen3()
    test_text_qwen25()
    test_image_qwen3vl()
    test_image_qwen25vl()
    test_overrides_passed_to_session_run()
    test_session_respond_full_flow()
    test_device_id_passed_through()
    test_npu_index_resolves_via_get_devices_id()
    test_npu_out_of_range_raises()
    test_npu_and_device_id_mutually_exclusive()
    test_discover_vision_paths_multi_resolution()
    test_discover_vision_paths_legacy_single_file()
    test_discover_vision_paths_skips_half_present_variants()
    test_discover_vision_paths_ignores_other_prefixes()
    test_pick_variant_closest_by_longest_side()
    test_pick_variant_single_variant_short_circuits()
    test_available_image_sizes_returns_variant_sizes()
    test_pick_variant_image_size_override_exact_match()
    test_pick_variant_image_size_override_nearest_fallback()
    test_run_images_forwards_image_size()
    # --- thinking-mode bug fix coverage ---
    test_strip_thinking_blocks_basic()
    test_extract_json_strips_think_blocks()
    test_extract_text_from_mismatched_dict_recurses()
    test_safe_validate_raw_text_strips_think()
    test_respond_thinking_block_stripped_in_schema()
    test_respond_hallucinated_tool_name_recovers_schema()
    test_respond_hallucinated_tool_name_with_invalid_arguments_falls_through()
    test_respond_tool_call_xml_wrapper_stripped()
    # --- NousFnCallPrompt (Hermes) tool-call format ---
    test_render_tools_block_uses_nous_fncall_format()
    test_render_tools_block_empty()
    test_compose_prompt_choice_block_uses_tool_call_wrapper()
    test_try_parse_tool_call_handles_hermes_string_arguments()
    test_safe_validate_fallback_strips_json_fence_and_tool_call_wrapper()
    test_qwen3_5_vl_with_new_prompt_calls_crop_correctly()
    # --- deepstack 2-D aux broadcast regression (wdnwebui 视频目标检测) ---
    test_pack_aux_2d_deepstack_does_not_broadcast_fail()
    # --- inputs[0]=multimodal ordering regression (RKLLMSession error) ---
    test_pack_inputs_multimodal_must_be_first()
    print("\nALL MOCK TESTS PASSED")
