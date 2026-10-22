"""``SoftwareCapability`` 词表测试。

重点不是「枚举有几个值」，而是把**四条运行时性质**钉死 —— 它们全仓有几百处
调用点依赖，坏掉时症状是静默的（路由不到实例 / 用户文案变成 ``Cap.TEXT_CHAT``）：

1. 成员与裸字符串**相等**，且 ``hash`` 一致 → 按能力串做键的既有代码无需改造；
2. ``json.dumps`` 输出**裸字符串** → HTTP / config 的 wire 格式不变；
3. ``__str__`` 覆写后 f-string / ``%s`` / ``+`` 全部还原为裸值；
4. 严格校验（``coerce``）与老配置迁移（``migrate_legacy``）**语义分离** ——
   别名不能混进校验，否则拼错的 manifest 能蒙混过关。
"""
import json

import pytest

from wdnai import (
    LEGACY_ALIASES,
    RETIRED_CAPABILITIES,
    RetiredCapabilityError,
    SoftwareCapability,
    known_values,
    migrate_legacy,
)


#: 声明顺序即 ``SoftwareCapability`` 里的顺序（报错信息与 manifest 校验
#: 输出靠它可读，所以单独按序断言，不只比对集合）。
EXPECTED = (
    "swcap_text_chat",
    "swcap_vision_chat",
    "swcap_vision_grounding",
    "swcap_audio_chat",
    "swcap_vision_embedding",
    "swcap_text_to_speech",
    "swcap_tool_call",
)


# --------------------------------------------------------------------------- #
# 词表范围
# --------------------------------------------------------------------------- #
def test_exact_member_set():
    assert set(SoftwareCapability.values()) == set(EXPECTED)
    assert known_values() == set(EXPECTED)


def test_values_order_stable():
    """顺序即声明顺序 —— 报错信息与 manifest 校验输出靠它可读。"""
    assert SoftwareCapability.values() == EXPECTED


# --------------------------------------------------------------------------- #
# 性质 1：与裸字符串互相等价、哈希一致
# --------------------------------------------------------------------------- #
def test_equality_and_hash_match_plain_string():
    cap = SoftwareCapability.VISION_CHAT
    assert cap == "swcap_vision_chat"
    assert "swcap_vision_chat" in [cap]
    assert cap in SoftwareCapability
    assert hash(cap) == hash("swcap_vision_chat")


def test_usable_as_dict_key_both_directions():
    """全仓到处是 ``cfg.swcap_to_model[cap]`` 这类按能力串取值的写法。"""
    cap = SoftwareCapability.TOOL_CALL
    by_str = {"swcap_tool_call": {"npu_index": 0}}
    by_enum = {cap: {"npu_index": 0}}
    assert by_str[cap] == {"npu_index": 0}
    assert by_enum["swcap_tool_call"] == {"npu_index": 0}
    assert by_str == by_enum


# --------------------------------------------------------------------------- #
# 性质 2：JSON / wire 格式不变
# --------------------------------------------------------------------------- #
def test_json_roundtrip_emits_plain_string():
    payload = {"capability": SoftwareCapability.AUDIO_CHAT}
    assert json.dumps(payload) == '{"capability": "swcap_audio_chat"}'
    assert json.loads(json.dumps(payload))["capability"] == SoftwareCapability.AUDIO_CHAT


# --------------------------------------------------------------------------- #
# 性质 3：用户可见文案不被枚举名污染
# --------------------------------------------------------------------------- #
def test_str_overridden_so_messages_stay_readable():
    cap = SoftwareCapability.TEXT_CHAT
    assert str(cap) == "swcap_text_chat"
    assert f"功能未启用: {cap}" == "功能未启用: swcap_text_chat"
    assert "功能未启用: %s" % cap == "功能未启用: swcap_text_chat"
    assert "功能未启用: " + cap == "功能未启用: swcap_text_chat"


# --------------------------------------------------------------------------- #
# 性质 4：校验严格、迁移宽松
# --------------------------------------------------------------------------- #
def test_coerce_accepts_current_names_only():
    assert SoftwareCapability.coerce("swcap_tool_call") is SoftwareCapability.TOOL_CALL
    assert SoftwareCapability.coerce(SoftwareCapability.TOOL_CALL) is SoftwareCapability.TOOL_CALL


@pytest.mark.parametrize("bad", [
    "swcap_video_describe",     # 上一版名字：校验路径必须拒（迁移才认）
    "video_describe",           # 裸旧名
    "swcap_computer_control",   # 上一版名字
    "swcap_video_query",        # 从未上线的幽灵能力
    "swcap_video_analyse",      # 同上
    "swcap_",                   # 空前缀
    "TOOL_CALL",                # 成员名而非值
    "",
])
def test_coerce_rejects_everything_else(bad):
    with pytest.raises(ValueError) as exc:
        SoftwareCapability.coerce(bad)
    # 报错必须列出合法值，否则调用方无从修正
    assert "swcap_tool_call" in str(exc.value)


def test_coerce_rejects_non_string():
    with pytest.raises(ValueError):
        SoftwareCapability.coerce(42)


def test_contains_is_boolean_coerce():
    assert SoftwareCapability.contains("swcap_audio_chat")
    assert SoftwareCapability.contains(SoftwareCapability.AUDIO_CHAT)
    assert not SoftwareCapability.contains("swcap_audio_understand")
    assert not SoftwareCapability.contains(None)


def test_migrate_legacy_maps_current_without_rename():
    cap, renamed = migrate_legacy("swcap_text_chat")
    assert cap is SoftwareCapability.TEXT_CHAT
    assert renamed is False


@pytest.mark.parametrize("old,expected", [
    ("video_describe", SoftwareCapability.VISION_CHAT),
    ("audio_tts", SoftwareCapability.TEXT_TO_SPEECH),
    ("audio_understand", SoftwareCapability.AUDIO_CHAT),
    ("swcap_video_describe", SoftwareCapability.VISION_CHAT),
    ("swcap_grounding", SoftwareCapability.VISION_GROUNDING),
    ("swcap_audio_understand", SoftwareCapability.AUDIO_CHAT),
    ("swcap_video_retrieval", SoftwareCapability.VISION_EMBEDDING),
    ("swcap_audio_tts", SoftwareCapability.TEXT_TO_SPEECH),
    ("swcap_computer_control", SoftwareCapability.TOOL_CALL),
])
def test_migrate_legacy_every_alias(old, expected):
    cap, renamed = migrate_legacy(old)
    assert cap is expected
    assert renamed is True


def test_migrate_legacy_passes_enum_through():
    cap, renamed = migrate_legacy(SoftwareCapability.TOOL_CALL)
    assert cap is SoftwareCapability.TOOL_CALL
    assert renamed is False


def test_alias_table_targets_are_real_members():
    """每个别名都必须指向真实成员，否则老配置迁到一个不存在的值。"""
    for old, target in LEGACY_ALIASES.items():
        assert isinstance(target, SoftwareCapability), old
        assert target in SoftwareCapability, old


@pytest.mark.parametrize("gone", sorted(RETIRED_CAPABILITIES))
def test_retired_capabilities_raise_instead_of_silently_mapping(gone):
    """下线的能力必须报错 —— 静默映射到无人声明的名字会让老配置变成
    永久的「功能未启用」，且没有任何线索。"""
    with pytest.raises(RetiredCapabilityError) as exc:
        migrate_legacy(gone)
    assert gone in str(exc.value)


def test_retired_error_is_valueerror_subclass():
    """调用方想统一按 ValueError 兜底时仍能捕获。"""
    assert issubclass(RetiredCapabilityError, ValueError)


def test_migrate_legacy_rejects_unknown():
    with pytest.raises(ValueError):
        migrate_legacy("swcap_not_a_thing")


def test_no_alias_collides_with_current_names():
    """别名不能和当前成员同名，否则 migrate 会走不到改名分支。"""
    assert not (set(LEGACY_ALIASES) & set(SoftwareCapability.values()))


def test_retired_and_alias_tables_disjoint():
    assert not (set(RETIRED_CAPABILITIES) & set(LEGACY_ALIASES))