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
    _check_runtime_flags,
    _read_meta,
    _resolve_binary,
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


def _fake_runtime(path: Path, *, with_flags: bool = True) -> Path:
    """写一个「像运行时」的假二进制。

    ⚠️ 2026-11-02 第三十五轮：``_resolve_binary`` 现在会做**版本护栏**
    （老 runtime 会**默默忽略** ``--te-resident`` ⇒ 走回老路 ⇒ 白等 79 s），
    所以测试用的假二进制必须**带上关键参数的特征串**。
    """
    blob = b"\x7fELF" + (b"\x00" * 64)
    if with_flags:
        blob += b"te-resident share-internal head-model vae-stream"
    path.write_bytes(blob)
    path.chmod(0o755)
    return path


def test_binary_resolution_prefers_packaged_runtime(tmp_path, monkeypatch):
    """模型包自带 runtime/ 时优先用它（模型包自包含）。"""
    pkg = tmp_path / "pkg"
    (pkg / "runtime").mkdir(parents=True)
    binary = _fake_runtime(pkg / "runtime" / "qwen_image_rknn")
    monkeypatch.delenv("WDNAI_QWEN_IMAGE_BIN", raising=False)
    assert _resolve_binary(pkg) == binary


def test_binary_resolution_env_override(tmp_path, monkeypatch):
    custom = _fake_runtime(tmp_path / "my_bin")
    monkeypatch.setenv("WDNAI_QWEN_IMAGE_BIN", str(custom))
    assert _resolve_binary(tmp_path) == custom


def test_stale_runtime_is_rejected(tmp_path, monkeypatch):
    """⭐ 模型包里的运行时副本**过期** ⇒ 必须**立刻报错**，不能默默跑老路。

    2026-11-02 实测：包里的旧二进制没有 ``--te-resident`` ⇒ 默默忽略 ⇒
    走回「读 te 目录里现成的 encoder_hidden_states.bin」的老路 ⇒
    报「TE 目录里缺 encoder_hidden_states.bin」，**白等 79 s**。
    """
    pkg = tmp_path / "pkg"
    (pkg / "runtime").mkdir(parents=True)
    _fake_runtime(pkg / "runtime" / "qwen_image_rknn", with_flags=False)
    monkeypatch.delenv("WDNAI_QWEN_IMAGE_BIN", raising=False)
    with pytest.raises(RuntimeError) as exc:
        _resolve_binary(pkg)
    assert "太旧" in str(exc.value)
    assert "te-resident" in str(exc.value)


def test_runtime_flag_check_names_every_missing_flag(tmp_path):
    """缺的每个参数都要在报错里列出来（否则没法判断该同步哪个版本）。"""
    bad = tmp_path / "old_runtime"
    bad.write_bytes(b"\x7fELF te-resident share-internal")
    bad.chmod(0o755)
    with pytest.raises(RuntimeError) as exc:
        _check_runtime_flags(bad)
    msg = str(exc.value)
    assert "head-model" in msg
    assert "vae-stream" in msg
    assert "te-resident" not in msg


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


def test_vae_tensor_decodes_to_bgr(tmp_path):
    """NCHW [1,4,512,512] fp16 → BGR uint8，且和运行时的归一化一致。"""
    side = 8
    plane = side * side
    # NCHW：通道 0 在 [0:64]，通道 3 是不可见通道（必须被忽略）。
    data = np.zeros(4 * plane, dtype=np.float16)
    data[:plane] = np.linspace(-1.0, 1.0, plane, dtype=np.float32)
    data[3 * plane:] = 0.5
    raw = tmp_path / "img01_vae_image.bin"
    raw.write_bytes(data.tobytes())
    (tmp_path / "img01_vae_image.bin.meta").write_text(
        "# dtype shape\nfloat16 1,4,8,8\n"
    )

    backend = ImageBackend.__new__(ImageBackend)   # 只测解码，不碰路径
    bgr, png = backend._load_image(tmp_path)
    assert bgr.shape == (8, 8, 3)
    assert bgr.dtype == np.uint8
    assert png.is_file()
    # 通道 0/1/2 是 RGB ⇒ BGR 的第 2 通道 = R = 通道 0
    assert int(bgr[0, 0, 2]) == 0            # -1 → 0
    assert int(bgr[-1, -1, 2]) == 255        # +1 → 255
    # 通道 3 不参与成像
    assert int(bgr[0, 0, 0]) == int(bgr[0, 0, 1]) == 127


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
        assert "qwenimage21" in str(exc)
        return
    assert spec.kind == "image"
    assert spec.variant == "qwen-image-2.1"
    assert spec.options["steps"] == 8
    assert spec.options["fm_mu"] == "0.5387096774"
