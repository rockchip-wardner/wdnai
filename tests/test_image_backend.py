"""Qwen-Image-2.1 文生图后端（``wdnai.backends.image``）的单元测试。

覆盖的是**不需要 NPU** 的部分：路径解析、.bin → 图像的解码数学、PNG 编码
与运行时的一致性、以及「文生图模型绝不能看到 JSON schema 提示」这条硬约束。

真正跑 NPU 的端到端验证在 ``examples/qwen_image_21.py``（板端实测 ≈ 94 s）。
"""
from __future__ import annotations

import asyncio
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

from wdnai import (
    ImageAttachment,
    LMOutput,
    LLMLocalSession,
    ModelSpec,
    SoftwareCapability,
    SystemLanguageModel,
)
from wdnai.backends import image as image_mod
from wdnai.backends.image import (
    ImageBackend,
    _DTYPES,
    _read_meta,
    _resolve_model_dir,
    _write_png,
)


# --------------------------------------------------------------------------- #
# 能力词表
# --------------------------------------------------------------------------- #
def test_text_to_image_capability_exists():
    assert SoftwareCapability.TEXT_TO_IMAGE == "swcap_text_to_image"
    assert "swcap_text_to_image" in SoftwareCapability.values()


# --------------------------------------------------------------------------- #
# 路径解析
# --------------------------------------------------------------------------- #
def test_explicit_model_dir_is_strict(tmp_path, monkeypatch):
    """显式指定的目录不存在 ⇒ 直接报错，不静默换到别的目录。"""
    monkeypatch.setenv("QI_MODEL_DIR", str(tmp_path))
    spec = ModelSpec(name="x", kind="image", variant="qwen-image-2.1",
                      image_model_dir=str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError):
        _resolve_model_dir(spec)


def test_model_dir_falls_back_to_env(tmp_path, monkeypatch):
    (tmp_path / "llm_qwenimage21-te.1828.rknn").write_bytes(b"x")
    monkeypatch.setenv("QI_MODEL_DIR", str(tmp_path))
    spec = ModelSpec(name="x", kind="image", variant="qwen-image-2.1")
    assert _resolve_model_dir(spec) == tmp_path


# --------------------------------------------------------------------------- #
# cshim（第三十六轮：不再有闭源二进制）
# --------------------------------------------------------------------------- #
def test_cshim_lives_inside_the_python_package():
    """cshim 必须是 wdnai 包的一部分（源码 + 编译产物都在仓库里）。"""
    from wdnai._qwen_image_c import cshim_path

    so = cshim_path()
    assert so.parent.name == "_cshim"
    assert so.name == "libqi_qwen.so"
    assert so.parent.parent.name == "wdnai"


def test_cshim_exports_the_expected_api():
    """⛔ 没有闭源二进制 ⇒ 出图能力必须来自**我们自己的** .so。"""
    from wdnai._qwen_image_c import cshim_path

    so = cshim_path()
    if not so.exists():
        pytest.skip("cshim 未编译（cd src/wdnai/_cshim && make）")
    import ctypes

    lib = ctypes.CDLL(str(so))
    for sym in ("qi_qwen_open", "qi_qwen_close", "qi_qwen_te_encode",
                "qi_qwen_generate_batch", "qi_qwen_free"):
        assert hasattr(lib, sym), sym


def test_t2i_template_keeps_the_special_tokens():
    """⭐ 回归护栏：模板必须带 Qwen 的对话特殊 token。

    漏掉它们 ⇒ TE 输出变成「错位切片」⇒ **出图内容完全不对但 corr 依然很高**
    （2026-11-01 修过一个致命 bug；2026-11-03 在 cshim 侧又踩了一次）。
    """
    from wdnai._qwen_image_c import build_t2i_template

    t = build_t2i_template("a cat")
    assert t.count("<" + "|im_start|>") == 3
    assert t.count("<" + "|im_end|>") == 2
    assert t.startswith("<" + "|im_start|>system")


def test_model_package_ships_no_closed_source_binary():
    """⭐ 用户硬要求：发布包里**不得**有不开源的 bin / so。"""
    from wdnai._qwen_image_c import cshim_path

    pkg = Path("/home/rockchip/models/Qwen-Image-2.1")
    if not pkg.is_dir():
        pytest.skip("板端模型包不在本机")
    assert not (pkg / "runtime" / "qwen_image_rknn").exists()
    assert not (pkg / "runtime").exists()
    # 系统自带的门面 + rkcp 就够用（native 从未被加载）
    assert Path("/usr/lib/librknn3_api.so").exists()
    assert Path("/usr/lib/librknn3_api_rkcp.so").exists()


def test_fixed_noise_is_generated_not_shipped():
    """⭐ 固定初始噪声必须**现场生成**，且与历史基线逐字节一致。

    历史基线是模型包里的 ``reference_512/hidden_states.bin``，实测由
    ``np.random.default_rng(0).standard_normal(64**3).reshape(64,64,64)[::2,::2,:]``
    生成。numpy 只保证「同一 build 内」的流一致，跨版本可能换算法
    （官方文档举的例子就是 Box-Muller → Ziggurat）⇒ 不能依赖 numpy 现算，
    cshim 里搬了一份 PCG64 + ziggurat ⇒ 永久逐字节一致。
    """
    import hashlib

    from wdnai._qwen_image_c import FIXED_NOISE_MD5, cshim_path, fixed_noise

    if not cshim_path().exists():
        pytest.skip("cshim 未编译（cd src/wdnai/_cshim && make）")
    noise = fixed_noise()
    assert noise.dtype == np.float16
    assert noise.size == 1024 * 64
    assert hashlib.md5(noise.tobytes()).hexdigest() == FIXED_NOISE_MD5


def test_model_package_needs_no_data_file():
    """⭐ 模型包里**只剩模型**：初始噪声不再需要随包发布。"""
    from wdnai.model import _QWEN_IMAGE_21_REQUIRED

    assert "reference_512/hidden_states.bin" not in _QWEN_IMAGE_21_REQUIRED
    assert not any("reference_512" in f for f in _QWEN_IMAGE_21_REQUIRED)


# --------------------------------------------------------------------------- #
# .meta / .bin 解码
# --------------------------------------------------------------------------- #
def test_read_meta_parses_runtime_sidecar(tmp_path):
    meta = tmp_path / "x.bin.meta"
    meta.write_text("# dtype shape\nfloat16 1,4,512,512\n")
    dtype, shape = _read_meta(tmp_path / "x.bin")
    assert dtype == "float16"
    assert shape == [1, 4, 512, 512]


def test_read_meta_missing_returns_none(tmp_path):
    assert _read_meta(tmp_path / "nope.bin") == (None, None)


def test_png_writer_matches_runtime_reference(tmp_path):
    """SDK 写的 PNG 必须和运行时 ``scripts/fp16_to_png.py`` **逐位相同**。

    运行时那份是纯标准库实现（zlib level 6 + filter-0 scanlines），这里
    照同样的规则写一遍并比对字节 —— 否则「板端出的图」和「SDK 出的图」
    会因为编码差异而对不上。
    """
    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    out = tmp_path / "a.png"
    _write_png(out, rgb)

    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    expect = b"\x89PNG\r\n\x1a\n"
    expect += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    expect += chunk(b"IDAT", zlib.compress(raw, 6))
    expect += chunk(b"IEND", b"")
    assert out.read_bytes() == expect


def test_run_texts_converts_rgb_to_bgr_and_writes_png(tmp_path, monkeypatch):
    """cshim 返回 RGB uint8 ⇒ 后端写 PNG 并把附件转成 **BGR**。

    （第三十六轮：解码不再走「读 .bin + .meta」，而是 cshim 直接给 RGB。）
    """
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 0] = 10   # R
    rgb[..., 1] = 20   # G
    rgb[..., 2] = 30   # B

    class _FakeShim:
        def generate(self, prompts, **kw):
            assert list(prompts) == ["a cat"]
            return [rgb]

        def close(self):
            pass

    backend = ImageBackend.__new__(ImageBackend)   # 不碰路径解析 / NPU
    backend.spec = ModelSpec(name="x", kind="image", variant="qwen-image-2.1")
    backend.model_dir = tmp_path
    backend.sink = image_mod._ProgressSink()
    backend.last_log = ""
    backend.last_artifacts = {}
    backend._workdir = None
    backend._owns_workdir = False
    backend._keep_artifacts = False
    backend._shim = _FakeShim()

    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)
    backend._make_workdir = lambda opts: work

    results = backend.run_texts(["a cat"], output_dir=str(tmp_path / "copy"))
    assert len(results) == 1
    prompt, attachments = results[0]
    assert prompt == "a cat"
    bgr = attachments[0].images[0]
    assert bgr.shape == (4, 4, 3)
    assert bgr.dtype == np.uint8
    assert int(bgr[0, 0, 2]) == 10   # BGR 的第 2 通道 = R
    assert int(bgr[0, 0, 0]) == 30   # BGR 的第 0 通道 = B
    # PNG 与后端自己写的那份**逐字节相同**（复制到 output_dir）
    png = work / "out" / "img01.png"
    assert png.is_file()
    assert (tmp_path / "copy" / "img01.png").read_bytes() == png.read_bytes()


def test_dtype_table_covers_runtime_dtypes():
    for name in ("float16", "float32"):
        assert name in _DTYPES


# --------------------------------------------------------------------------- #
# session 接线（用假后端，不碰 NPU）
# --------------------------------------------------------------------------- #
class _FakeImageBackend:
    def __init__(self, spec, **_kwargs):
        self.spec = spec
        self.seen = []
        self.sink = None

    def run_text(self, prompt, **overrides):
        self.seen.append((prompt, dict(overrides)))
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        return prompt, [ImageAttachment(img)]

    def close(self):  # pragma: no cover
        pass


@pytest.fixture()
def fake_image_session(monkeypatch):
    import wdnai.session as session_mod

    monkeypatch.setattr(session_mod, "build_backend",
                        lambda spec, **kw: _FakeImageBackend(spec, **kw))
    spec = ModelSpec(name="fake", kind="image", variant="qwen-image-2.1")
    return LLMLocalSession(model=spec)


def test_generate_image_returns_bgr(fake_image_session):
    """``generate_image`` 是最短路径：直接拿到 BGR 像素。"""
    img = asyncio.run(fake_image_session.generate_image("a cat", steps=20))
    assert img.shape == (512, 512, 3)
    assert fake_image_session._backend.seen[-1][1]["steps"] == 20


def test_run_prompt_returns_attachment(fake_image_session):
    text, atts = asyncio.run(fake_image_session.run_prompt("a dog"))
    assert text == "a dog"
    assert len(atts) == 1
    assert isinstance(atts[0], ImageAttachment)


def test_respond_with_lmoutput_carries_image(fake_image_session):
    out = asyncio.run(fake_image_session.respond("a bird", generating=LMOutput))
    assert isinstance(out, LMOutput)
    assert len(out.attachment) == 1
    assert isinstance(out.attachment[0], ImageAttachment)


def test_schema_hint_never_reaches_an_image_prompt(fake_image_session):
    """文生图模型把提示词**画进图里** ⇒ JSON schema 提示绝不能混进去。"""
    from wdnai import Field, Generable

    @Generable
    class Doc(LMOutput):
        act: str = Field(description="action")

    with pytest.raises(Exception):
        asyncio.run(fake_image_session.respond("a fish", generating=Doc))
    assert fake_image_session._backend.seen[-1][0] == "a fish"


# --------------------------------------------------------------------------- #
# preset
# --------------------------------------------------------------------------- #
def test_preset_shape():
    """preset 指向的目录不存在时也应当给出可读的错误。"""
    try:
        spec = SystemLanguageModel.qwen_image_21_512()
    except FileNotFoundError as exc:
        # 目录名 = HF 仓库名（Qwen-Image-2.1）；旧名 qwenimage21 仍可能被兜底命中
        assert ("Qwen-Image-2.1" in str(exc)) or ("qwenimage21" in str(exc))
        return
    assert spec.kind == "image"
    assert spec.variant == "qwen-image-2.1"
    assert spec.options["steps"] == 8
    assert spec.options["fm_mu"] == "0.5387096774"


# --------------------------------------------------------------------------- #
# 进度（第三十七轮：cshim → sink → on_delta）
# --------------------------------------------------------------------------- #
def test_cshim_progress_becomes_a_structured_sink_message():
    """cshim 的进度回调必须变成**可解析**的结构化消息。

    上层（Web UI 卡片）靠它画进度条：「现在到哪一步、这一步多久、还剩多久」。
    格式：``qi|stage=flow|step=3|total=8|elapsed_ms=2601.5``
    """
    from wdnai.backends.image import ImageBackend

    backend = ImageBackend.__new__(ImageBackend)   # 不走 __init__（不碰 NPU）
    backend.sink = image_mod._ProgressSink()
    seen = []
    backend.sink.add_observer(seen.append)

    backend._on_cshim_progress("flow", 3, 8, 2601.5)

    assert len(seen) == 1
    msg = seen[0]
    assert msg.startswith("qi|"), msg
    fields = dict(
        part.split("=", 1) for part in msg[len("qi|"):].split("|") if "=" in part
    )
    assert fields["stage"] == "flow"
    assert fields["step"] == "3"
    assert fields["total"] == "8"
    assert float(fields["elapsed_ms"]) == pytest.approx(2601.5, abs=0.1)


def test_progress_message_does_not_touch_the_npu():
    """进度消息只是字符串 ⇒ 不该有任何副作用（不写文件、不碰设备）。"""
    from wdnai.backends.image import ImageBackend

    backend = ImageBackend.__new__(ImageBackend)
    backend.sink = image_mod._ProgressSink()
    seen = []
    backend.sink.add_observer(seen.append)
    backend._on_cshim_progress("vae", 2, 4, 2228.0)
    assert "stage=vae" in seen[0]


# --------------------------------------------------------------------------- #
# 选卡：按**运行时枚举**分配（不占 NPU）
# --------------------------------------------------------------------------- #
def _bare_backend(tmp_path):
    """造一个不碰 NPU / 不起 proxy 的 ImageBackend。"""
    backend = ImageBackend.__new__(ImageBackend)
    backend.spec = ModelSpec(name="x", kind="image", variant="qwen-image-2.1")
    backend.model_dir = tmp_path
    backend.sink = image_mod._ProgressSink()
    backend._shim = None
    backend._ensure_proxy = lambda: None      # 别真去起 transfer proxy
    return backend


def _fake_shim(monkeypatch):
    """把 QwenImageShim 换成记录参数的假货，返回记录 dict。"""
    seen = {}

    class _FakeShim:
        def __init__(self, model_dir, **kw):
            seen.update(kw)

        def close(self):
            pass

    monkeypatch.setattr("wdnai._qwen_image_c.QwenImageShim", _FakeShim)
    return seen


def test_default_options_leave_cards_to_runtime_enumeration():
    """⭐ 卡号默认留空 ⇒ 交给**运行时枚举**，不写死任何 PCI 地址。"""
    backend = _bare_backend(Path("/tmp"))
    opts = backend._options({})
    assert opts["card0"] is None
    assert opts["card1"] is None
    assert opts["card2"] is None


def test_cards_come_from_runtime_enumeration(monkeypatch, tmp_path):
    """⭐ 卡号必须来自 `_ep_card_ids()` 的**现枚举**结果。"""
    eps = ["aaaa:11:00.0", "bbbb:31:00.0"]
    monkeypatch.setattr(image_mod, "_ep_card_ids", lambda: list(eps))
    seen = _fake_shim(monkeypatch)

    backend = _bare_backend(tmp_path)
    backend._ensure_shim(backend._options({}))

    assert seen["card_te"] == b"aaaa:11:00.0"
    assert seen["card_dit"] == b"bbbb:31:00.0"
    assert seen["card_vae"] == b""          # 两张卡 ⇒ VAE 走本机 3588


def test_third_card_takes_the_vae(monkeypatch, tmp_path):
    """三张卡 ⇒ 第三张独占给 VAE(rknn3)。"""
    monkeypatch.setattr(
        image_mod, "_ep_card_ids",
        lambda: ["aaaa:11:00.0", "bbbb:31:00.0", "cccc:31:00.0"],
    )
    seen = _fake_shim(monkeypatch)

    backend = _bare_backend(tmp_path)
    backend._ensure_shim(backend._options({}))

    assert seen["card_te"] == b"aaaa:11:00.0"
    assert seen["card_dit"] == b"bbbb:31:00.0"
    assert seen["card_vae"] == b"cccc:31:00.0"


def test_explicit_card_override_wins(monkeypatch, tmp_path):
    """显式指定的卡号仍然优先于枚举结果。"""
    monkeypatch.setattr(
        image_mod, "_ep_card_ids", lambda: ["aaaa:11:00.0", "bbbb:31:00.0"],
    )
    seen = _fake_shim(monkeypatch)

    backend = _bare_backend(tmp_path)
    opts = backend._options({"card0": "cccc:00:00.0", "card1": "dddd:00:00.0"})
    backend._ensure_shim(opts)

    assert seen["card_te"] == b"cccc:00:00.0"
    assert seen["card_dit"] == b"dddd:00:00.0"


def test_enumeration_failure_raises_instead_of_guessing(monkeypatch, tmp_path):
    """⭐ 枚举不到 ⇒ **报错**，不许拿一个猜的地址去初始化。"""
    monkeypatch.setattr(image_mod, "_ep_card_ids", lambda: [])
    _fake_shim(monkeypatch)

    backend = _bare_backend(tmp_path)
    with pytest.raises(RuntimeError) as exc:
        backend._ensure_shim(backend._options({}))
    assert "枚举不到" in str(exc.value)


def test_qwen_image_shim_requires_explicit_cards():
    """⭐ 卡号**必填** —— 漏传要在构造时立刻炸。"""
    import inspect

    from wdnai._qwen_image_c import QwenImageShim

    params = inspect.signature(QwenImageShim.__init__).parameters
    assert params["card_te"].default is inspect.Parameter.empty
    assert params["card_dit"].default is inspect.Parameter.empty


def test_progress_none_passes_a_null_callback(monkeypatch, tmp_path):
    """⭐ ``progress=None`` ⇒ 传**空回调实例**（NULL 指针），不能传 None。

    直接传 None 会被 ctypes 拒掉：
    ``TypeError: expected CFunctionType instance instead of NoneType``。
    """
    import wdnai._qwen_image_c as qic

    class _FakeLib:
        def __init__(self):
            self.open_args = None

        def qi_qwen_open_ex3(self, *args):
            self.open_args = args
            return 1                    # 非 NULL 句柄

        def qi_qwen_version(self):
            return b"fake"

    lib = _FakeLib()
    monkeypatch.setattr(qic, "_load", lambda: lib)
    monkeypatch.setattr(
        "wdnai.gguf_tokenizer.load_tokenizer_from_gguf", lambda p: None,
    )

    qic.QwenImageShim(tmp_path, card_te=b"a", card_dit=b"b", progress=None)

    cb = lib.open_args[6]              # arg7 = 进度回调
    assert cb is not None
    assert not bool(cb)                # NULL 函数指针

