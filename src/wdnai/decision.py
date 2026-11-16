"""Intern-Decision 的结构化决策协议：任务解析、prompt 构造、概率校准。

这个模块是**纯 CPU** 的：它不 import ``librknn3``，因此可以在任何机器上
import / 单测。真正跑 NPU 的部分在
:mod:`wdnai.backends.intern_decision`。

模型与参考实现
==============

Intern-Decision（internlm/Intern-Decision-{0.8B,2B,4B}）不是普通的对话
模型：它不生成文本，而是针对**若干决策字段**各输出一次「候选符号的
logits」，由 CPU 侧换算成选项概率。参考实现是 rknn3_model_zoo 的
``examples/Intern_Decision/cpp/main.cpp``，本模块逐条对齐它的行为。

三种字段类型
------------

``choice``
    从若干选项中选一个（客服分流：account / payment / other）。
``score``
    选项键是**数字字符串**（"0" / "1" / "2"），输出是各选项概率按数值
    加权得到的评分。
``noul``
    是否判断（yes / no），输出「肯定概率」。

候选符号
========

62 个：``A-Z`` + ``a-z`` + ``0-9``。某个字段的第 *i* 个选项拿到第 *i*
个符号，符号本身没有语义 —— 语义来自任务配置里的选项键。

温度校准
========

模型输出的 logits 先做 softmax，再按 checkpoint 对应的温度做一次
「log 域重标定」：``p_i ∝ exp((log p_i - max log p) / T)``。温度由
embedding 宽度查表（:data:`CHECKPOINT_TEMPERATURES`）—— 4B 是
``1.99241824``。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "DECISION_SYMBOLS",
    "DECISION_SYSTEM_PROMPT",
    "DECISION_MARKER",
    "CHECKPOINT_TEMPERATURES",
    "DEFAULT_IMAGE_STATE",
    "DecisionOption",
    "DecisionField",
    "DecisionTask",
    "DecisionFieldResult",
    "DecisionResult",
    "parse_decision_task",
    "build_decision_prompt",
    "calibrate_probabilities",
    "weighted_score",
    "pick_best_option",
]


#: 62 个候选符号，顺序固定：``A-Z`` → ``a-z`` → ``0-9``。
#: 参考实现里是 ``static const char symbols[] = "ABC...9";``。
DECISION_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

#: 参考实现里逐字照抄的 system prompt。**不要改写** —— 模型是按这一段
#: 训练 / 校准的，改一个字都会让概率偏移。
DECISION_SYSTEM_PROMPT = (
    "You are a careful decision assistant. Use the state and decision schema in "
    "the user message to make the requested decisions. For every field, choose "
    "exactly one answer symbol (e.g. A, B, C, ...) from its listed options and "
    "return one valid JSON object mapping each field name to its chosen symbol. "
    "Use the field names and symbols exactly as given. Do not include "
    "explanations, Markdown, or extra text."
)

#: 字段占位符。参考实现用模型自带的 tokenizer 把它切成一个 token，
#: 再在 token 序列里定位「第 i 个字段的 ``<decision>``」的位置 ——
#: 该字段的推理输入就是**这个位置之前**的 token 前缀。
DECISION_MARKER = "<decision>"

#: embedding 宽度 → 概率校准温度。参考实现
#: ``checkpoint_temperatures[]``。
CHECKPOINT_TEMPERATURES: Dict[int, float] = {
    1024: 2.747760550703,   # Intern-Decision-0.8B
    2048: 2.100509348278,   # Intern-Decision-2B
    2560: 1.99241824,       # Intern-Decision-4B
}

#: 只给图不给文本时使用的 state（参考实现 ``intern_request_build``）。
DEFAULT_IMAGE_STATE = "请根据图片回答问题。"

#: 一个任务里最多多少个决策字段（参考实现硬上限 16）。
MAX_FIELDS = 16


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DecisionOption:
    """一个候选项：``key`` 是语义键（"account" / "0" / "yes"），``description``
    是给模型看的说明。

    符号不在这里 —— 它由选项在 :class:`DecisionField` 里的**序号**决定
    （见 :meth:`DecisionField.symbol_of`）。
    """

    key: str
    description: str


@dataclass(frozen=True)
class DecisionField:
    """一个决策字段（= 参考实现里 ``questions`` 的一项）。

    ``type`` 取 ``"choice"`` / ``"score"`` / ``"noul"``；``options`` 的
    **顺序**就是符号分配顺序（A、B、C…），必须与任务 JSON 里的书写顺序
    一致。
    """

    name: str
    type: str
    instructions: str
    options: Tuple[DecisionOption, ...]

    def symbol_of(self, option: DecisionOption) -> str:
        """Return the candidate symbol bound to ``option``."""
        return DECISION_SYMBOLS[self.options.index(option)]

    @property
    def symbols(self) -> Tuple[str, ...]:
        """每个选项对应的符号，顺序与 :attr:`options` 一致。"""
        return tuple(DECISION_SYMBOLS[i] for i in range(len(self.options)))


@dataclass
class DecisionTask:
    """一次决策请求：状态 + 若干字段。

    ``state`` 是「当前世界状态」，通常是用户文本（一段话），也可以是
    任意 JSON 可序列化的结构（参考实现允许对象）。字段顺序即
    :attr:`fields` 的顺序。
    """

    state: Any = None
    fields: List[DecisionField] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize back to the reference task-JSON shape."""
        questions: Dict[str, Any] = {}
        for f in self.fields:
            if f.type == "noul":
                criteria: Any = {o.key: o.description for o in f.options}
            elif f.type == "score" and all(
                o.key.isdigit() for o in f.options
            ):
                criteria = [o.description for o in f.options]
            else:
                criteria = {o.key: o.description for o in f.options}
            questions[f.name] = {
                "type": f.type,
                "instructions": f.instructions,
                "criteria": criteria,
            }
        return {"state": self.state, "questions": questions}


@dataclass
class DecisionFieldResult:
    """单个字段的决策结果。

    * ``symbol`` —— 选中的候选符号（"A" / "b" / "7"）。
    * ``label`` —— 选中的**选项键**（"account" / "2" / "yes"）。
    * ``probabilities`` —— 选项键 → 校准后的概率（合计为 1）。
    * ``logits`` —— 该字段有效选项的原始 logits（按选项顺序）。
    * ``score`` —— ``type="score"`` 时的概率加权评分。
    * ``noul`` —— ``type="noul"`` 时的肯定概率。
    """

    name: str
    type: str
    symbol: str
    label: str
    probabilities: Dict[str, float]
    logits: List[float]
    score: Optional[float] = None
    noul: Optional[float] = None

    @property
    def value(self) -> Any:
        """The field's value: the option key for choice/noul, the weighted
        score for score."""
        if self.type == "score":
            return self.score
        return self.label

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "type": self.type,
            "symbol": self.symbol,
            "label": self.label,
            "probabilities": dict(self.probabilities),
            "logits": list(self.logits),
        }
        if self.score is not None:
            out["score"] = self.score
        if self.noul is not None:
            out["noul"] = self.noul
        return out


@dataclass
class DecisionResult:
    """一次决策请求的完整结果。

    :attr:`fields` 按请求里的字段顺序排列。
    """

    fields: List[DecisionFieldResult] = field(default_factory=list)
    temperature: float = 0.0

    # -- 便捷视图 ------------------------------------------------------- #
    @property
    def by_name(self) -> Dict[str, DecisionFieldResult]:
        return {f.name: f for f in self.fields}

    def values(self) -> Dict[str, Any]:
        """``{字段名: 值}`` —— choice/noul 是选项键，score 是加权评分。"""
        return {f.name: f.value for f in self.fields}

    @property
    def text(self) -> str:
        """JSON 文本：``{"字段名": 值}``。

        这是给 session / RPC 通道用的「文本答案」——它是一个**合法
        JSON 对象**，所以 ``respond(..., generating=SomePydanticModel)``
        可以直接校验它（字段名对上、类型对上）。
        """
        return json.dumps(self.values(), ensure_ascii=False)

    def to_dict(self) -> Dict[str, Any]:
        """完整结果（含每个选项的概率与 logits），与参考实现落盘的 JSON
        结构一致（外层多一个 ``temperature``）。"""
        return {
            "decisions": {f.name: f.to_dict() for f in self.fields},
            "temperature": self.temperature,
        }

    def summary(self) -> str:
        """人类可读的一行式摘要（放进 ``TextAttachment`` 随 wire 传出）。"""
        parts = []
        for f in self.fields:
            if f.type == "score":
                parts.append(
                    "{}[{}]={:.4f}".format(f.name, f.symbol, float(f.score or 0.0))
                )
            elif f.type == "noul":
                parts.append(
                    "{}[{}]={} ({:.2%})".format(
                        f.name, f.symbol, f.label, float(f.noul or 0.0)
                    )
                )
            else:
                prob = f.probabilities.get(f.label, 0.0)
                parts.append(
                    "{}[{}]={} ({:.2%})".format(f.name, f.symbol, f.label, prob)
                )
        return "; ".join(parts)


# --------------------------------------------------------------------------- #
# 任务解析（对齐参考实现的 intern_options_parse / intern_request_build）
# --------------------------------------------------------------------------- #
class OrderedChoices:
    """Insertion-ordered string→string map (``dict`` 在 3.7+ 已经是
    ordered，但单独一个类让「键是字符串、值是字符串」这个约束显式化，
    并提供 ``set`` 语义 —— 重复键保留首次出现的位置与取值）。"""

    __slots__ = ("_data",)

    def __init__(self) -> None:
        self._data: Dict[str, str] = {}

    def set(self, key: str, value: str) -> None:
        self._data.setdefault(key, value)

    def assign(self, key: str, value: str) -> None:
        """Overwrite ``key``'s value, **keeping its original position**.

        参考实现里是 ``(*choices)["no"] = it.value();`` —— 赋值**覆盖**，
        而不是「首次出现优先」。noul 字段先用默认说明占位、再用任务
        JSON 里的 criteria 覆盖它，位置保持不变（``no`` 仍在 ``yes``
        前面，符号分配才不会错位）。
        """
        self._data[key] = value

    def items(self):
        return self._data.items()

    def __len__(self) -> int:
        return len(self._data)

    def __bool__(self) -> bool:
        return bool(self._data)


def _parse_options(
    name: str, question: Dict[str, Any],
) -> Tuple[str, Tuple[DecisionOption, ...]]:
    """Parse one ``questions`` entry into its type + option list.

    Mirrors the reference implementation's ``intern_options_parse``:

    * ``noul`` —— 固定 ``no`` / ``yes`` 两项；``criteria`` 存在时必须
      恰好是两项、且键分别命中 no/false/0 与 yes/true/1（大小写不敏感）。
    * ``choice`` / ``score`` —— ``criteria`` 是对象时按对象顺序取。
    * ``score`` —— ``criteria`` 是数组时，键取数组下标的字符串形式
      （"0" / "1" / "2"…）。
    """
    qtype = question.get("type")
    if not isinstance(qtype, str):
        raise ValueError(
            "决策字段 {!r} 缺少 type（choice / score / noul）".format(name)
        )
    if "instructions" in question and not isinstance(question["instructions"], str):
        raise ValueError("决策字段 {!r} 的 instructions 必须是字符串".format(name))

    has_criteria = "criteria" in question
    criteria = question.get("criteria", {})
    choices = OrderedChoices()

    if qtype == "noul":
        choices.set("no", "The answer is no (negative, or disagree with the claim).")
        choices.set("yes", "The answer is yes (affirmative, or align with the claim).")
        if has_criteria:
            if not isinstance(criteria, dict) or len(criteria) != 2:
                raise ValueError(
                    "noul 字段 {!r} 的 criteria 必须是恰好两项的对象".format(name)
                )
            no_set = False
            yes_set = False
            for key, value in criteria.items():
                if not isinstance(value, str):
                    raise ValueError(
                        "noul 字段 {!r} 的 criteria 值必须是字符串".format(name)
                    )
                low = key.lower()
                if not no_set and (low in ("no", "false") or key == "0"):
                    choices.assign("no", value)
                    no_set = True
                elif not yes_set and (low in ("yes", "true") or key == "1"):
                    choices.assign("yes", value)
                    yes_set = True
                else:
                    raise ValueError(
                        "noul 字段 {!r} 的 criteria 键 {!r} 无法映射到 yes/no".format(
                            name, key
                        )
                    )
            if not (no_set and yes_set):
                raise ValueError(
                    "noul 字段 {!r} 的 criteria 必须同时给出 no 与 yes".format(name)
                )
    elif qtype in ("choice", "score") and isinstance(criteria, dict):
        for key, value in criteria.items():
            choices.set(key, value)
    elif qtype == "score" and isinstance(criteria, list):
        for index, value in enumerate(criteria):
            choices.set(str(index), value)
    else:
        raise ValueError(
            "决策字段 {!r} 的 type={!r} 与 criteria 形状不匹配".format(name, qtype)
        )

    for key, value in choices.items():
        if not isinstance(value, str):
            raise ValueError(
                "决策字段 {!r} 的选项 {!r} 的说明必须是字符串".format(name, key)
            )
        if qtype == "score":
            try:
                score = float(key)
            except ValueError:
                raise ValueError(
                    "score 字段 {!r} 的选项键 {!r} 必须是数字".format(name, key)
                ) from None
            if not math.isfinite(score):
                raise ValueError(
                    "score 字段 {!r} 的选项键 {!r} 必须是有限数".format(name, key)
                )
    if not choices:
        raise ValueError("决策字段 {!r} 没有任何选项".format(name))
    if len(choices) > len(DECISION_SYMBOLS):
        raise ValueError(
            "决策字段 {!r} 的选项数超过 {}".format(name, len(DECISION_SYMBOLS))
        )
    return qtype, tuple(
        DecisionOption(key=key, description=value) for key, value in choices.items()
    )


def parse_decision_task(payload: Any) -> DecisionTask:
    """Parse the reference task JSON into a :class:`DecisionTask`.

    ``payload`` is either the parsed JSON (a ``dict``) or a JSON string.

    Shape::

        {
          "state": "...",                     # optional; the user text
          "questions": {
            "category": {
              "type": "choice",              # choice | score | noul
              "instructions": "...",
              "criteria": {"account": "...", "payment": "..."}
            }
          }
        }

    Raises
    ------
    ValueError
        The task is malformed (missing ``questions``, too many fields,
        bad option shape, …).
    """
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError("决策任务必须是 JSON 对象")
    questions = payload.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("决策任务缺少非空的 questions")
    if len(questions) > MAX_FIELDS:
        raise ValueError(
            "决策字段数 {} 超过上限 {}".format(len(questions), MAX_FIELDS)
        )
    fields: List[DecisionField] = []
    for name, question in questions.items():
        if not isinstance(question, dict):
            raise ValueError("决策字段 {!r} 必须是对象".format(name))
        qtype, options = _parse_options(name, question)
        fields.append(
            DecisionField(
                name=name,
                type=qtype,
                instructions=question.get("instructions", ""),
                options=options,
            )
        )
    return DecisionTask(state=payload.get("state"), fields=fields)


# --------------------------------------------------------------------------- #
# Prompt 构造（对齐参考实现的 intern_prompt_build）
# --------------------------------------------------------------------------- #
def build_decision_prompt(
    task: DecisionTask,
    *,
    image_tokens: int = 0,
    image_count: int = 0,
) -> str:
    """Build the model's prompt for ``task``.

    Layout (byte-identical to the reference implementation)::

        <|im_start|>system\\n<system prompt><|im_end|>\\n<|im_start|>user\\n
        [Picture 1: <|vision_start|><|image_pad|>×N<|vision_end|>] ...
        <user block><|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n
        {\\n    "field": "<decision>",\\n    ...\\n}<|im_end|>\\n

    ⚠️ The ChatML control markers are **part of the prompt text**, not a
    chat template: the exported model has an empty ``chat_template``, so
    the C side must keep ``set_chat_template("", "", "")`` and the markers
    have to be spelled out here. Dropping them shifts every token by 66
    bytes / 3 tokens and measurably changes the calibrated probabilities.

    ``image_count`` / ``image_tokens`` control the picture placeholders;
    pass ``0`` for the text-only case.
    """
    if image_count and image_tokens <= 0:
        raise ValueError("有图时必须给出 image_tokens（视觉模型输出的 token 数）")
    user = (
        "Return one answer for every field using the supplied answer symbols."
        "\n\n## State\n"
    )
    user += json.dumps(task.state, indent=2, ensure_ascii=False)
    user += "\n## Decision schema\n"
    skeleton: Dict[str, str] = {}
    first = True
    for f in task.fields:
        if not first:
            user += "\n"
        first = False
        user += "{name}: {instructions}".format(
            name=f.name, instructions=f.instructions
        )
        for index, option in enumerate(f.options):
            user += "\n    {sym} = {key}: {desc}".format(
                sym=DECISION_SYMBOLS[index], key=option.key, desc=option.description
            )
        skeleton[f.name] = DECISION_MARKER
    if DECISION_MARKER in user:
        raise ValueError("state / instructions 里不能出现 {}".format(DECISION_MARKER))

    prompt = "<|im_start|>system\n" + DECISION_SYSTEM_PROMPT + "<|im_end|>\n<|im_start|>user\n"
    for i in range(image_count):
        prompt += "Picture {n}: <|vision_start|>".format(n=i + 1)
        prompt += "<|image_pad|>" * image_tokens
        prompt += "<|vision_end|>"
    prompt += user + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prompt += json.dumps(skeleton, indent=4, ensure_ascii=False) + "<|im_end|>\n"
    return prompt


# --------------------------------------------------------------------------- #
# 概率校准（对齐参考实现的 intern_probabilities）
# --------------------------------------------------------------------------- #
def calibrate_probabilities(
    logits: Sequence[float],
    temperature: float,
) -> List[float]:
    """Turn raw logits into calibrated probabilities.

    Two stages, both in **FP32** (the reference implementation does it in
    C ``float``, not double — the difference is visible in the last ULP
    of the printed percentages):

    1. ``softmax(logits)`` over the **field's own options only**;
    2. re-calibrate in log space: ``p_i ∝ exp((log p_i - max log p) / T)``.

    ``temperature == 1`` skips stage 2 (that is what the reference does
    too — ``if (temperature == 1.0f) return 0;``).

    The result is **bit-identical** to the reference implementation: same
    ``float32`` rounding order, same sequential (not pairwise)
    accumulation, and ``numpy``'s ``float32`` ``exp``/``log`` agree with
    glibc's ``expf``/``logf`` bit for bit.
    """
    count = len(logits)
    if count <= 0 or count > len(DECISION_SYMBOLS):
        raise ValueError("logits 数量必须在 1..{} 之间".format(len(DECISION_SYMBOLS)))
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature 必须是有限正数")
    values = np.asarray([float(x) for x in logits], dtype=np.float32)
    for x in values:
        if not np.isfinite(x):
            raise ValueError("logits 含有非有限值")

    # ---- 阶段 1：softmax（FP32，逐项累加）------------------------------
    maximum = values[0]
    for value in values:
        if value > maximum:
            maximum = value
    raw = np.empty(count, dtype=np.float32)
    total = np.float32(0.0)
    for index in range(count):
        exp = np.float32(np.exp(np.float32(values[index] - maximum)))
        raw[index] = exp
        total = np.float32(total + exp)

    probs = np.empty(count, dtype=np.float32)
    logs = np.empty(count, dtype=np.float32)
    max_log = np.float32(-np.inf)
    for index in range(count):
        prob = np.float32(raw[index] / total)
        probs[index] = prob
        # 保留 FP32 下溢产生的 0 概率（参考实现也是如此）。
        log = np.float32(np.log(prob)) if prob > 0 else np.float32(-np.inf)
        logs[index] = log
        if log > max_log:
            max_log = log

    temperature32 = np.float32(temperature)
    if temperature32 == np.float32(1.0):
        return [float(p) for p in probs]

    # ---- 阶段 2：log 域温度重标定（FP32）--------------------------------
    scaled = np.empty(count, dtype=np.float32)
    norm = np.float32(0.0)
    for index in range(count):
        exp = np.float32(
            np.exp(np.float32(np.float32(logs[index] - max_log) / temperature32))
        )
        scaled[index] = exp
        norm = np.float32(norm + exp)
    return [float(np.float32(value / norm)) for value in scaled]


def weighted_score(
    keys: Sequence[str],
    probabilities: Sequence[float],
) -> float:
    """Probability-weighted score for a ``type="score"`` field.

    Mirrors the reference implementation's ``float expected`` accumulator::

        expected += strtof(key.c_str(), NULL) * probability;

    Everything is **float32** — the C code accumulates in ``float``, so a
    double accumulator drifts by a few ULP away from the official number.
    """
    total = np.float32(0.0)
    for key, probability in zip(keys, probabilities):
        weight = np.float32(float(key))
        total = np.float32(total + np.float32(weight * np.float32(probability)))
    return float(total)


def pick_best_option(
    probabilities: Sequence[float],
    keys: Sequence[str],
) -> int:
    """Index of the winning option.

    Ties are broken by the **lexicographically smaller option key** —
    the reference implementation does the same ("Upstream resolves equal
    probabilities by lexical option label").
    """
    best = 0
    for index in range(len(probabilities)):
        if index == 0:
            best = 0
            continue
        if probabilities[index] > probabilities[best]:
            best = index
        elif probabilities[index] == probabilities[best] and keys[index] < keys[best]:
            best = index
    return best
