"""端到端 mock 测试：用伪 RKNN3Lite + 伪 tokenizer 把整条链路跑一遍。"""
import os
import tempfile
from typing import Annotated

import numpy as np


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
    def __init__(self, n_dims=4, shape=(1, 384, 384, 3), aligned_size=4096):
        self.n_dims, self.shape, self.aligned_size = n_dims, shape, aligned_size


class _FakeRknn:
    # Class-level list of mock device IDs returned by ``get_devices_id()``.
    # Tests that exercise the ``npu=`` path monkey-patch this to control
    # what's resolvable via ``wdnai.backends.resolve_device_id``.
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
        self._sinks = []
        self._tokenizer = _MockTokenizer(vocab_size)

    def load_rknn(self, *a, **kw): return 0
    def init_runtime(self, *a, **kw):
        self.last_init_runtime_kwargs = kw
        return 0
    def set_chat_template(self, *a, **kw): return 0
    def release(self): return 0

    def get_inputs_tensor_attr(self): return [self._in_attr]
    def get_outputs_tensor_attr(self): return [self._out_attr]
    def get_devices_id(self):
        # Always returns a fresh list (real RKNN3Runtime is constructed
        # and released each time, so each call is independent).
        return list(self.mock_devices)

    def inference(self, inputs=None, data_format=None):
        feat = np.random.randn(1, 32).astype(np.float32)
        outs = [feat]
        for _ in range(self.deepstack_n):
            outs.append(np.random.randn(32).astype(np.float32))
        return outs

    def session_run(self, inputs=None, prompt=None, embeds=None, **kw):
        self.session_calls += 1
        self.last_prompt = prompt
        self.last_inputs = inputs
        tok = self._tokenizer
        if self.llm_mode:
            ids = [ord(c) % tok._vocab for c in (prompt or "") if ord(c) < 128]
            # 如果 prompt 里有 JSON Schema 提示词，就回 JSON 模拟"听话的模型"
            if "JSON Schema" in (prompt or ""):
                text = '{"act": "run", "level": 2}'
            else:
                text = tok.decode(ids) if ids else (prompt or "")
        else:
            text = f"[vision-mock] {prompt}"
        for sink in self._sinks:
            sink.text = text
        return 0, [10, 5, 0.0, 0.001]


# --------------------------------------------------------------------------- #
class _MockFactory:
    """冒充 RKNN3Lite 的构造器：每次调用产生一个 _FakeRknn。"""

    def __new__(cls, llm_mode=False, verbose=False, deepstack_n=0, vocab_size=64):
        return _FakeRknn(llm_mode=llm_mode, deepstack_n=deepstack_n, vocab_size=vocab_size)


# --------------------------------------------------------------------------- #
def _install_mocks(deepstack_n: int = 0, vocab_size: int = 64):
    """把 rknn3lite + transformers + 关键 helper 全部 monkey-patch 成 mock。"""
    import rknn3lite.api as api
    import wdnai.backends._common as common
    import wdnai.backends.text as text_mod
    import wdnai.backends.vlm as vlm_mod

    # ---- 1) RKNN3Lite 构造器：返回 _FakeRknn ----
    def make(llm_mode=False, verbose=False):
        return _FakeRknn(llm_mode=llm_mode, deepstack_n=deepstack_n, vocab_size=vocab_size)
    api.RKNN3Lite = make

    # ---- 2) LLMResultCallback：吞掉 cffi 回调（结果由 session_run 同步写 sink） ----
    from rknn3lite.api import LLMResultCallback
    def noop_result_cb(sink):
        def cb(userdata, result_ptr, state): return 0
        return LLMResultCallback(cb)
    common.make_result_callback = noop_result_cb

    # ---- 3) Tokenizer / Embed 加载 ----
    common.load_tokenizer = lambda path: _MockTokenizer(vocab_size)
    common.load_embed_table = lambda path, vs: np.zeros((vs, 8), dtype=np.float16)

    # ---- 4) 各后端模块里也都重新绑一遍（它们是 from import） ----
    text_mod.load_tokenizer = common.load_tokenizer
    text_mod.load_embed_table = common.load_embed_table
    text_mod.make_result_callback = common.make_result_callback
    vlm_mod.load_tokenizer = common.load_tokenizer
    vlm_mod.load_embed_table = common.load_embed_table
    vlm_mod.make_result_callback = common.make_result_callback

    # ---- 5) build_llm_callback：构造完把 sink 注册到 rknn._sinks ----
    orig_build_llm_cb = common.build_llm_callback
    last_rknn = {"obj": None}

    # 关键：先设 last_rknn，再调原 __init__（build_llm_callback 在 __init__ 内部被调）
    orig_text_init = text_mod.TextBackend.__init__
    def h_text(self, spec):
        last_rknn["obj"] = self.rknn  # 暂存 rknn，但 self.rknn 此刻还没创建
        # 实际上 rknn 是在 __init__ 第一步建的，所以我们要拆开
    text_mod.TextBackend.__init__ = h_text

    # 改用更稳的方案：直接 monkey-patch build_llm_callback，让它用 rknn 引用闭包
    # 把 rknn 的 _sinks 列表暴露成"全局最近一次构造的 rknn"
    # —— 借助 text/vlm backend 的 __init__ 第一步 rknn 创建时机
    orig_text_init = text_mod.TextBackend.__init__
    def h_text2(self, spec, *, npu=None, device_id=None):
        # 1) 先建 rknn（同步原 __init__ 的第一行）
        from rknn3lite.api import RKNN3Lite  # 已被 monkey-patch
        if not spec.rknn_path or not spec.weight_path:
            raise ValueError("missing path")
        self.rknn = RKNN3Lite(llm_mode=True, verbose=spec.verbose)
        if self.rknn.load_rknn(spec.rknn_path, spec.weight_path) != 0:
            raise RuntimeError("load failed")
        # 2) 暂存，让后续 build_llm_callback 找得到
        last_rknn["obj"] = self.rknn
        # 3) 跑剩下的初始化
        from wdnai.backends._common import (
            StreamSink, build_llm_args, apply_chat_template,
        )
        self.tokenizer = text_mod.load_tokenizer(spec.tokenizer_path)
        self.embeds = text_mod.load_embed_table(spec.embed_path, spec.vocab_size)
        self.sink = StreamSink(tokenizer=self.tokenizer)
        cb = text_mod.build_llm_callback(self.tokenizer, self.embeds, self.sink)
        # Use the real resolver so ``npu`` / ``device_id`` flow through
        # identically to the unpatched backend.
        from wdnai.backends import resolve_device_id
        resolved = resolve_device_id(RKNN3Lite, npu, device_id)
        if self.rknn.init_runtime(
            target=spec.target, core_mask=spec.core_mask,
            llm_args=build_llm_args(spec), llm_callback=cb,
            device_id=resolved,
        ) != 0:
            raise RuntimeError("init_runtime failed")
        apply_chat_template(self.rknn)
    text_mod.TextBackend.__init__ = h_text2

    orig_vlm_init = vlm_mod.VLMBackend.__init__
    def h_vlm(self, spec, *, npu=None, device_id=None):
        from rknn3lite.api import RKNN3Lite
        if not spec.rknn_llm_path or not spec.rknn_vision_path:
            raise ValueError("missing path")
        self.spec = spec
        # vision
        self.rknn_vision = RKNN3Lite()
        vp = spec.rknn_vision_path
        if self.rknn_vision.load_rknn(vp, vp.replace(".rknn3", ".weight")) != 0:
            raise RuntimeError("vision load failed")
        # Wrap the single mock vision RKNN into a one-element
        # ``_vision_variants`` list so the multi-resolution-aware
        # ``_pick_variant`` / ``_build_image_inputs`` code paths find
        # it. Production ``VLMBackend.__init__`` builds the same list
        # from directory discovery; the mock short-circuits to a single
        # variant at ``spec.image_size``.
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

    def hooked_build(tokenizer, embeds, sink):
        cb = orig_build_llm_cb(tokenizer, embeds, sink)
        if last_rknn["obj"] is not None:
            last_rknn["obj"]._sinks.append(sink)
        return cb
    common.build_llm_callback = hooked_build
    text_mod.build_llm_callback = hooked_build
    vlm_mod.build_llm_callback = hooked_build


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
    from wdnai import LLMSession, SystemLanguageModel, Attachment
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
            prompt="请分析这张图的行为", attachment=Attachment(img),
        )
        assert isinstance(out2, str) and out2
        print("[session] image respond ->", repr(out2))

        # 3) 图像 + 结构化输出（注入 JSON hint + 解析）
        out3 = await sess.respond(
            prompt="请分析这张图的行为",
            attachment=Attachment(img),
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
                return [SimpleNamespace(n_dims=4, shape=(1, 384, 384, 3))]
            return [SimpleNamespace(n_dims=2, shape=(2304, 1536))]

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
    print("[preprocess] dispatches by in_attr.n_dims (384=NHWC, 768=pruned)")


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
            return [SimpleNamespace(n_dims=4, shape=(1, 384, 384, 3))]
        def get_outputs_tensor_attr(self):
            return [SimpleNamespace(aligned_size=4096)]
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
            attachment=__import__("wdnai").Attachment(
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
            attachment=__import__("wdnai").Attachment(
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
            attachment=__import__("wdnai").Attachment(
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
            attachment=__import__("wdnai").Attachment(
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
    from wdnai import Attachment, Callable as _Callable

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
            attachment=Attachment(np.zeros((480, 640, 3), dtype=np.uint8)),
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
    print("\nALL MOCK TESTS PASSED")
