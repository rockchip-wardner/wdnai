"""软件能力词表（``swcap_*``）—— 全仓唯一权威定义。

能力回答的是**「模型能做什么」**，不是「哪个业务用它」：

* 命名轴统一为 ``<输入模态>_<输出形态或功能>``；
* 「什么业务场景调用它」属于**绑定**（哪个模型 / 哪张卡），归调用方与配置管，
  不许混进能力名 —— 历史上 ``swcap_computer_control`` 就是把绑定问题伪装成
  能力造的（同一个 RKNN 同时服务多个能力时，「拿到哪个模型」取决于卡号排列）。

``swcap_*`` 之前只在 docstring 里被引用（见 :mod:`wdnai.model`），定义散落在
各应用仓库的 manifest 里，于是每个人都能自造一个。本模块把它收回 SDK，并且
用枚举把范围钉死。

Python 3.9 兼容
===============

``pyproject.toml`` 声明 ``requires-python = ">=3.9"``，因此只能用
``class X(str, Enum)``，**不可用** :class:`enum.StrEnum`（3.11+）。

``__str__`` 必须覆写
====================

``str`` 混入的 Enum 在 3.11 之前 ``str(member)`` 返回 ``"SoftwareCapability.X"``，
于是所有 f-string 的用户可见文案会变成「功能未启用: SoftwareCapability.TEXT_CHAT」。
实测覆写 ``__str__`` 后 f-string / ``%s`` / ``+`` 拼接 / ``str()`` 全部还原为裸值。

而**不**覆写 ``__hash__`` / ``__eq__``：实测 ``hash(member) == hash(value)``，
所以按能力串做键的既有代码（``cfg.swcap_to_model[cap]`` 之类）双向命中，
``json.dumps`` 仍输出裸字符串 —— wire 格式不变。
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, FrozenSet, Iterable, List, Tuple

__all__ = [
    "SoftwareCapability",
    "RetiredCapabilityError",
    "LEGACY_ALIASES",
    "RETIRED_CAPABILITIES",
    "migrate_legacy",
]


class SoftwareCapability(str, Enum):
    """一个 RKNN 模型实例在本机**实际**能做的事。

    只描述本机 RKNN 端口真正实现了的通路，不是上游权重理论支持的范围。例：
    Qwen2.5-Omni 上游是全模态（图像/视频/音频进、语音出），但 wdnai 的 omni
    backend 只导出了 wav-in→text-out 一条（``backends/omni.py`` 的
    ``run_images`` 直接 raise），所以这里**不**给它 vision 相关能力 ——
    声明一个只有运行时才会炸的能力，比不声明更糟。
    """

    #: 文本 → 文本。多轮上下文 + 流式逐字输出。
    TEXT_CHAT = "swcap_text_chat"
    #: 图像 → 文本。通用 VLM 描述 / 目标检测描述（schema 属调用方）。
    VISION_CHAT = "swcap_vision_chat"
    #: 图像 → 0–1000 归一化 bbox + 文本。需专用定位 encoder
    #: （LocateAnything），不是「VLM 加个 prompt」就能替代的。
    VISION_GROUNDING = "swcap_vision_grounding"
    #: 音频 → 文本。
    AUDIO_CHAT = "swcap_audio_chat"
    #: 图像 → 向量。供向量检索 / 相似度反查。
    VISION_EMBEDDING = "swcap_vision_embedding"
    #: 文本 → 语音。
    TEXT_TO_SPEECH = "swcap_text_to_speech"
    #: → **可执行动作 / 工具调用**，超越自由文本输出。两种实现都算：
    #: ``@Generable`` 动作 schema（agent 的 click/type/key…）与 ``@Callable``
    #: 工具调度（副作用在主进程执行）。
    TOOL_CALL = "swcap_tool_call"

    def __str__(self) -> str:
        # 见模块 docstring：不覆写会让全站 f-string 文案变成 "Cap.TEXT_CHAT"。
        return self.value

    @classmethod
    def values(cls) -> Tuple[str, ...]:
        """全部合法能力串（按声明顺序）。"""
        return tuple(m.value for m in cls)

    @classmethod
    def coerce(cls, value: Any) -> "SoftwareCapability":
        """严格解析**当前**能力名，不接受任何历史别名。

        用于 manifest 校验 —— 别名属于「读老配置」的迁移语义（见
        :func:`migrate_legacy`），放进校验会让拼错的 manifest 蒙混过关。

        Raises
        ------
        ValueError
            ``value`` 不是合法成员时；消息里列出全部合法值。
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value)
            except ValueError:
                pass
        raise ValueError(
            "未知的软件能力 {!r}；合法值为: {}".format(
                value, ", ".join(cls.values())
            )
        )

    @classmethod
    def contains(cls, value: Any) -> bool:
        """:meth:`coerce` 的布尔版（校验路径用，不抛异常）。"""
        return isinstance(value, cls) or (
            isinstance(value, str) and value in cls._value2member_map_
        )


class RetiredCapabilityError(ValueError):
    """存储的旧能力名对应的功能已下线，无法映射到任何现存能力。"""


#: 历史能力名 → 当前成员。含两种形态：
#:
#: 1. **裸旧名**（最早的 config 里没带前缀）—— ``video_describe`` 等；
#: 2. **带前缀的上一版名字** —— ``swcap_video_describe`` 等，本次改名前的
#:    当前名。老 config 可能存着任意一种。
LEGACY_ALIASES: Dict[str, SoftwareCapability] = {
    # 裸旧名
    "video_describe": SoftwareCapability.VISION_CHAT,
    "audio_tts": SoftwareCapability.TEXT_TO_SPEECH,
    "audio_understand": SoftwareCapability.AUDIO_CHAT,
    # 上一版 swcap_* 名
    "swcap_video_describe": SoftwareCapability.VISION_CHAT,
    "swcap_grounding": SoftwareCapability.VISION_GROUNDING,
    "swcap_audio_understand": SoftwareCapability.AUDIO_CHAT,
    "swcap_video_retrieval": SoftwareCapability.VISION_EMBEDDING,
    "swcap_audio_tts": SoftwareCapability.TEXT_TO_SPEECH,
    "swcap_computer_control": SoftwareCapability.TOOL_CALL,
}

#: 已下线、**没有**替代能力的旧名。迁移时必须明确报错，不能静默映射 ——
#: 静默迁到一个无人声明的名字，等于把老配置变成永久的「功能未启用」，
#: 而这正是这些条目一直躺在迁移表里没人发现的原因。
RETIRED_CAPABILITIES: Dict[str, str] = {
    "video_query": "视频查询能力从未在任何 worker 上线，无替代能力",
    "swcap_video_query": "视频查询能力从未在任何 worker 上线，无替代能力",
    "video_analyse": "已并入视觉描述（swcap_vision_chat），请改用该能力",
    "swcap_video_analyse": "已并入视觉描述（swcap_vision_chat），请改用该能力",
}


def migrate_legacy(value: Any) -> Tuple[SoftwareCapability, bool]:
    """把**存储态**的能力名（当前名或历史别名）解析成当前成员。

    与 :meth:`SoftwareCapability.coerce` 的区别：本函数接受历史别名（读老
    config 用），遇到已下线的能力则抛 :class:`RetiredCapabilityError`。

    Returns
    -------
    (能力, 是否发生了改名)
        ``renamed`` 为 ``True`` 表示输入是历史别名，调用方据此决定要不要
        落盘回写。

    Raises
    ------
    RetiredCapabilityError
        能力已下线且无替代。
    ValueError
        既不是当前名也不是已知别名。
    """
    if isinstance(value, SoftwareCapability):
        return value, False
    if not isinstance(value, str):
        raise ValueError("软件能力必须是字符串，得到 {!r}".format(type(value).__name__))
    if value in SoftwareCapability:
        return SoftwareCapability(value), False
    reason = RETIRED_CAPABILITIES.get(value)
    if reason is not None:
        raise RetiredCapabilityError("软件能力 {!r} 已下线：{}".format(value, reason))
    alias = LEGACY_ALIASES.get(value)
    if alias is not None:
        return alias, True
    raise ValueError(
        "未知的软件能力 {!r}；合法值为: {}".format(
            value, ", ".join(SoftwareCapability.values())
        )
    )


def known_values() -> FrozenSet[str]:
    """全部合法能力串的集合（卡片 manifest 校验用）。"""
    return frozenset(SoftwareCapability.values())