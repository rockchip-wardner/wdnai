"""Intern-Decision backend — structured decision (分类 / 是非 / 评分) on RK182x.

对齐 rknn3_model_zoo 的 ``examples/Intern_Decision/cpp/main.cpp``。

流程
====

1. **加载**：LLM RKNN（``llm_mode=True``）+ tokenizer + embed 表；视觉
   RKNN **按需加载**（第一次带图的请求时才加载，纯文本任务不占这份显存）。
2. **建 prompt**：CPU 侧拼出完整 prompt（含 ``<decision>`` 骨架），用
   模型自带的 tokenizer 分词，定位每个字段的 ``<decision>`` token 位置。
3. **逐字段推理**：每个字段用**全新的 session** 跑一次 prefill
   （``prefill_only=True`` / ``max_new_tokens=1`` / ``disable_sampling=True``），
   输入是该字段 ``<decision>`` 标记**之前**的 token 前缀；输出 62 维候选
   符号 logits。
4. **算概率**：CPU 侧对「本字段有效选项」的那几维做 softmax + 温度校准，
   再按字段类型给出分类结果 / 加权评分 / 肯定概率。

与普通 LLM 后端的三个关键差异
=============================

* **不生成文本**。模型输出的是 62 个候选符号的 logits，不是 token 流。
  因此本后端走 ``output_callback``（``PREFILL_FINISHED``）拿张量，而不是
  靠 result callback 解码文本。
* **每个字段一次 prefill**，且必须**换 session**。该模型有 linear-attention
  层，递归状态会跨调用残留；``keep_history=0`` 只清 KV cache，不清递归
  状态 —— 参考实现因此每个字段都 ``rknn3_session_init`` / ``destroy``。
  这里用 :meth:`wdnai._librknn.RKNN3Lite.recreate_llm_session` 复刻。
* **prompt 由 CPU 全权构造**。模型导出时 ``chat_template`` 是空的
  （``Intern-Decision-4B-llm.config.pkl`` 里 system/prompt_prefix/
  prompt_postfix/chat_template 全空），所以 C 侧的 chat template 必须
  显式清空，否则 prompt 会被 ChatML 再包一层。
"""
from __future__ import annotations

import ctypes
import json
import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..decision import (
    DEFAULT_IMAGE_STATE,
    DECISION_MARKER,
    DECISION_SYMBOLS,
    CHECKPOINT_TEMPERATURES,
    DecisionField,
    DecisionFieldResult,
    DecisionResult,
    DecisionTask,
    build_decision_prompt,
    calibrate_probabilities,
    parse_decision_task,
    pick_best_option,
    weighted_score,
)
from .._rga import rga_resize_rgb888
from ..utils import bgr_to_rgb
from ._common import (
    build_llm_args,
    build_llm_callback,
    load_embed_table,
    load_tokenizer,
    resolve_device_id,
)

LOG = logging.getLogger("wdnai.backends.intern_decision")


#: 控制 token —— 必须是**单 token**，否则参考实现那种「在 token 序列里
#: 定位 ``<decision>``」的做法就无从谈起（分词器会把 ``<decision>`` 拆成
#: ``<`` / ``decision`` / ``>`` 三个 id）。
#:
#: ``<think>`` / ``</think>`` 同理：模型训练时它们各是一个 id（248068 /
#: 248069）。它们**不**注册进 Qwen 的全局 tokenizer —— Qwen 路径靠解码
#: 文本里的字面量切 thinking 通道，注册成 special 会被
#: ``skip_special_tokens=True`` 抹掉。这里按实例注册，只影响本后端。
DECISION_CONTROL_TOKENS = ("<decision>", "<think>", "</think>")


class _NullResultSink:
    """一个什么都不做的 result callback。

    C 库**要求**必须有 result callback（缺了会打印
    ``no result_callback provided!`` 并让 ``session_run`` 返回 -1），
    即使这次跑的是 ``prefill_only``、根本不产出 token 流。决策模型
    只读输出张量，所以这里给一个空实现。
    """

    __slots__ = ()

    def __call__(self, _userdata: Any, _tokens: Sequence[int], _state: int) -> int:
        return 0


class _DecisionOutputSink:
    """接收 ``PREFILL_FINISHED`` 时刻的 62 维候选符号 logits。

    必须是**绑定方法**传给 ``output_cb`` —— shim 靠 ``__self__``
    反射拿到 userdata（见 ``_librknn.build_llm_callback``）。
    """

    __slots__ = ("n_elems", "logits", "received", "error")

    def __init__(self, n_elems: int) -> None:
        self.n_elems = int(n_elems)
        self.logits: List[float] = []
        self.received = False
        self.error: Optional[str] = None

    def reset(self) -> None:
        self.logits = []
        self.received = False
        self.error = None

    def __call__(self, _userdata: Any, outputs: Sequence[Any], state: int) -> int:
        from .._librknn import _LLMOutputCallbackState

        if int(state) != int(_LLMOutputCallbackState.PREFILL_FINISHED):
            return 0
        if len(outputs) != 1:
            self.error = "output callback 期望 1 个张量，收到 {}".format(len(outputs))
            return -1
        view = outputs[0]
        if int(view.n_elems) != self.n_elems:
            self.error = (
                "output 张量元素数 {} != 期望的候选符号数 {}".format(
                    int(view.n_elems), self.n_elems,
                )
            )
            return -1
        data = view.data
        if data is None:
            self.error = "output 张量没有数据"
            return -1
        values = [float(x) for x in data[: self.n_elems]]
        for value in values:
            if not np.isfinite(value):
                self.error = "logits 含有非有限值"
                return -1
        self.logits = values
        self.received = True
        return 0


class _VisionModel:
    """视觉 RKNN 的运行时封装（懒加载）。"""

    __slots__ = ("rknn", "width", "height", "tokens", "embed_dim", "path")

    def __init__(self, rknn, width: int, height: int, tokens: int, embed_dim: int, path: str):
        self.rknn = rknn
        self.width = width
        self.height = height
        self.tokens = tokens
        self.embed_dim = embed_dim
        self.path = path


def _result_to_payload(result: DecisionResult) -> Tuple[str, List[Any]]:
    """把 :class:`DecisionResult` 转成 session 层要的 ``(text, attachments)``。

    * ``text`` —— 答案 JSON（``{"字段名": 值}``）。它是**合法 JSON 对象**，
      所以 ``respond(..., generating=SomeModel)`` 可以直接校验。
    * 附件 —— 一条 :class:`~wdnai.attachment.TextAttachment`，内容是完整
      明细（每个字段的符号 / 概率 / 原始 logits / 评分），随 wire 一起
      传给调用方。
    """
    from ..attachment import TextAttachment

    detail = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    summary = result.summary()
    attachment_text = summary + "\n" + detail
    return result.text, [TextAttachment(attachment_text)]


class InternDecisionBackend:
    """Intern-Decision backend (text + image → structured decisions)."""

    def __init__(
        self,
        spec,
        *,
        npu: Optional[int] = None,
        device_id: Optional[Union[bytes, str]] = None,
    ):
        from .._librknn import RKNN3Lite

        if not spec.rknn_llm_path or not spec.weight_llm_path:
            raise ValueError(
                f"{spec.name!r} is missing rknn_llm_path / weight_llm_path; "
                "fill them in via .with_paths(...)."
            )
        self.spec = spec
        self._device_id = resolve_device_id(RKNN3Lite, npu, device_id)

        # ---- LLM ----------------------------------------------------------
        self.rknn_llm = RKNN3Lite(llm_mode=True, verbose=spec.verbose)
        if self.rknn_llm.load_rknn(spec.rknn_llm_path, spec.weight_llm_path) != 0:
            raise RuntimeError(f"llm load_rknn failed: {spec.rknn_llm_path}")

        self.tokenizer = load_tokenizer(
            spec.tokenizer_path,
            extra_special_tokens=list(DECISION_CONTROL_TOKENS),
        )
        self.embeds = load_embed_table(spec.embed_path, spec.vocab_size)

        # 第一次 init_runtime **不带** llm_args：这样只会把模型 load 起来，
        # 不会立刻建 session —— 我们需要先分配 output 张量（62 维 logits），
        # 再把 callback bundle 交给 session。
        if self.rknn_llm.init_runtime(
            target=spec.target,
            core_mask=spec.core_mask,
            llm_args=None,
            llm_callback=None,
            device_id=self._device_id,
        ) != 0:
            raise RuntimeError("llm init_runtime failed")

        self._out_attr = self._query_output_attr(0)
        n_symbols = len(DECISION_SYMBOLS)
        if int(self._out_attr.n_elems) != n_symbols:
            raise RuntimeError(
                "LLM 输出张量元素数 {} != 候选符号数 {} —— 这不是 Intern-Decision "
                "的导出？".format(int(self._out_attr.n_elems), n_symbols)
            )
        self._sink = _DecisionOutputSink(n_symbols)
        self._null_sink = _NullResultSink()
        out_tensors, n_out = self.rknn_llm.create_output_tensors([0])
        if not out_tensors or int(n_out) != 1:
            raise RuntimeError("create_output_tensors 失败")
        self._out_tensors = out_tensors

        cb = build_llm_callback(
            tokenizer=self.tokenizer,
            embeds_data=self.embeds,
            result_cb=self._null_sink,
            output_cb=self._sink,
            output_tensors=out_tensors,
        )
        if self.rknn_llm.init_llm_session(
            llm_args=build_llm_args(spec), llm_callback=cb,
        ) != 0:
            raise RuntimeError("init_llm_session failed")
        self._clear_chat_template()

        # 校准温度：优先用模型自己报的 embedding 宽度查表（与参考实现
        # 一致），查不到就用 spec 里显式给的 calibration_temperature。
        self.temperature = self._resolve_temperature()

        # ---- 推理性能表（2026-11-17）-------------------------------------
        #
        # ⚠️ 这个 backend **没有** ``StreamSink``（它用的是
        # ``_DecisionOutputSink``，只收 logits），所以标准的
        # ``backend.sink.perf`` 这条路走不通 ⇒ 自己记一份。
        #
        # ⚠️ **每个字段跑一次独立的 prefill-only 推理**
        # （``max_new_tokens=1``）⇒ 解码 token 恒为 0 ⇒ 真正的吞吐是
        # **预填充速率**（提示词处理速度），不是解码速率。
        #
        # DDS worker 侧会读 ``backend.perf`` 并带过 wire（见
        # ``wdndds/ai_session.py`` 的 ``_read_perf``）。
        self._perf: Dict[str, float] = {}

        # ---- 视觉（懒加载）-------------------------------------------------
        self._vision: Optional[_VisionModel] = None
        self._vision_geom: Optional[Tuple[int, int, int]] = None
        self._keep_vision_resident = bool(
            spec.options.get("keep_vision_resident", False)
        )
        # 视觉特征的内存引用 —— ``RKNN3Image.image_embed`` 只是一个裸指针，
        # 不 pin 住 numpy 数组的话 session_run 会读到已释放的内存。
        self._feature_pin: Optional[np.ndarray] = None

        # marker 的 token id —— 用来在 token 序列里定位每个字段。
        marker_id = self.tokenizer.convert_tokens_to_ids(DECISION_MARKER)
        if marker_id is None or int(marker_id) < 0:
            raise RuntimeError(
                "tokenizer 里找不到 {} —— 该控制 token 必须注册为特殊 token".format(
                    DECISION_MARKER,
                )
            )
        self._marker_id = int(marker_id)

        # ---- 视觉：常驻的话**现在就加载** ---------------------------------
        #
        # ⚠️ 常驻 ≠ 只是「不释放」—— 默认的懒加载路径是**第一次带图的请求**
        # 才 ``_ensure_vision()``。要真的「常驻」（启动即占好显存、带图请求
        # 没有首次 7.5 秒加载），必须在这里就把它拉起来。
        #
        # 失败不致命：降级回按需加载（纯文本决策照常能用），只记一条 warning。
        if self._keep_vision_resident:
            try:
                self._ensure_vision()
            except Exception as exc:  # noqa: BLE001
                LOG.warning(
                    "intern-decision: 视觉模型常驻加载失败（降级为按需加载）: %s",
                    exc,
                )

    # ------------------------------------------------------------------ #
    # 初始化辅助
    # ------------------------------------------------------------------ #
    def _query_output_attr(self, index: int):
        """Query one output tensor's attribute.

        ``RKNN3Lite.get_outputs_tensor_attr()`` returns ``[]`` for an LLM
        runtime (that helper is for the plain, non-LLM path) — the LLM
        side has to ask the driver directly.
        """
        from .._librknn import RKNN3QueryCmd

        return self.rknn_llm.rknn3_query(int(RKNN3QueryCmd.OUTPUT_ATTR), index=index)

    def _clear_chat_template(self) -> None:
        """清空 C 侧的 chat template。

        Intern-Decision 的导出里 chat_template 是空的，prompt 完全由
        CPU 侧构造（``system\\n...\\nuser\\n...``）。如果沿用 SDK 默认的
        ChatML，prompt 会被再包一层 ``<|im_start|>`` —— 模型没见过这个
        形状，概率会明显偏移。
        """
        ret = self.rknn_llm.set_chat_template("", "", "")
        if ret != 0:
            raise RuntimeError(f"set_chat_template failed: ret={ret}")

    def _resolve_temperature(self) -> float:
        """校准温度（参考实现的 ``checkpoint_temperatures[]``）。

        返回值是 **float32** —— 参考实现里温度表是 ``float`` 数组，
        4B 对应 ``1.99241824f``，实际值是 ``1.9924182891845703``。
        直接返回 double 会让上报的温度与官方差最后一位。
        """
        explicit = self.spec.options.get("calibration_temperature")
        if explicit:
            return float(np.float32(explicit))
        try:
            from .._librknn import RKNN3QueryCmd

            cfg = self.rknn_llm.rknn3_query(int(RKNN3QueryCmd.LLM_CONFIG))
            dim = int(getattr(cfg, "embedding_dim", 0))
        except Exception:  # noqa: BLE001 — 查询失败就退回硬编码
            dim = 0
        if dim in CHECKPOINT_TEMPERATURES:
            temperature = float(np.float32(CHECKPOINT_TEMPERATURES[dim]))
            LOG.info(
                "intern-decision: embedding_dim=%d → 校准温度 %s",
                dim, temperature,
            )
            return temperature
        raise RuntimeError(
            "未知的 embedding 宽度 {!r}，无法确定概率校准温度；"
            "请在 spec.options 里显式指定 calibration_temperature".format(dim)
        )

    # ------------------------------------------------------------------ #
    # 视觉
    # ------------------------------------------------------------------ #
    def _ensure_vision(self) -> _VisionModel:
        """加载视觉 RKNN（幂等）。

        第一次带图的请求才加载 —— 纯文本决策任务不该占这份显存。
        """
        if self._vision is not None:
            return self._vision
        from .._librknn import RKNN3Lite

        path = self.spec.rknn_vision_path
        weight = self.spec.weight_vision_path
        if not path:
            raise RuntimeError(
                "该请求需要图片，但 spec 里没有 rknn_vision_path —— "
                "Intern-Decision 的视觉模型没装？"
            )
        rknn = RKNN3Lite(verbose=self.spec.verbose)
        if rknn.load_rknn(path, weight) != 0:
            raise RuntimeError(f"vision load_rknn failed: {path}")
        if rknn.init_runtime(
            target=self.spec.target,
            core_mask=self.spec.core_mask,
            device_id=self._device_id,
        ) != 0:
            raise RuntimeError(f"vision init_runtime failed: {path}")

        in_attr = rknn.get_inputs_tensor_attr()[0]
        out_attr = rknn.get_outputs_tensor_attr()[0]
        # 视觉输入：uint8 NHWC [1, H, W, 3]；输出：fp16 [tokens, dim]
        if int(in_attr.n_dims) != 4:
            raise RuntimeError(
                "视觉输入维度 {} != 4 —— 期望 NHWC [1,H,W,3]".format(int(in_attr.n_dims))
            )
        height = int(in_attr.shape[1])
        width = int(in_attr.shape[2])
        if out_attr.n_dims == 2:
            tokens = int(out_attr.shape[0])
        elif int(out_attr.n_dims) == 3 and int(out_attr.shape[0]) == 1:
            tokens = int(out_attr.shape[1])
        else:
            raise RuntimeError(
                "视觉输出形状 {} 无法解析".format(list(out_attr.shape))
            )
        embed_dim = int(out_attr.shape[out_attr.n_dims - 1])
        self._vision_geom = (width, height, tokens)
        self._vision = _VisionModel(rknn, width, height, tokens, embed_dim, path)
        LOG.info(
            "intern-decision: vision loaded %s (%dx%d, %d tokens, dim %d)",
            path, width, height, tokens, embed_dim,
        )
        return self._vision

    def _release_vision(self) -> None:
        """释放视觉模型。

        参考实现就是**这么做的**：图编码完就把 vision 释放掉，再去做
        各字段的 LLM prefill —— 视觉权重（~682 MB）没必要陪着 LLM 的
        prefill 一起占着 NPU 内存。

        ``spec.options["keep_vision_resident"] = True`` 可以改成常驻。
        """
        if self._vision is None or self._keep_vision_resident:
            return
        try:
            self._vision.rknn.release()
        except Exception:  # noqa: BLE001
            pass
        self._vision = None

    def _encode_images(self, images: Sequence[np.ndarray]) -> Tuple[np.ndarray, int]:
        """编码一组图片，返回 ``(features, tokens_per_image)``。

        ``features`` 形状 ``(n_image * tokens, dim)`` —— 与参考实现的
        ``calloc(n_image * embedding_elements)`` 布局一致（按 token 轴
        拼接）。
        """
        import cv2

        vision = self._ensure_vision()
        feats: List[np.ndarray] = []
        for index, img in enumerate(images):
            if img is None or getattr(img, "size", 0) == 0:
                raise ValueError("第 {} 张图片为空".format(index))
            rgb = bgr_to_rgb(np.ascontiguousarray(img))
            # 参考实现的 convert_image 是**整幅拉伸**（不是 letterbox），
            # 归一化已经烙进模型里，这里只送原始 RGB 字节。
            #
            # 缩放优先走 **RGA 硬件**（与官方 image_utils.c 的
            # convert_image 完全一致）：RGA 的双线性与 OpenCV 的
            # INTER_LINEAR 不是同一个算法，同一张图缩出来像素最大能差
            # 130+，会带偏视觉 embedding，进而带偏决策概率。RGA 不可用
            # 或几何不 16 对齐时才回退 OpenCV。
            resized = rga_resize_rgb888(rgb, vision.width, vision.height)
            if resized is None:
                resized = cv2.resize(
                    rgb, (vision.width, vision.height),
                    interpolation=cv2.INTER_LINEAR,
                )
            nhwc = np.ascontiguousarray(
                resized.reshape(1, vision.height, vision.width, 3).astype(np.uint8)
            )
            outs = vision.rknn.inference(inputs=[nhwc])
            if not outs:
                raise RuntimeError("视觉推理没有返回输出")
            out0 = np.asarray(outs[0])
            if out0.ndim == 3 and out0.shape[0] == 1:
                out0 = out0[0]
            if out0.ndim != 2:
                raise RuntimeError(
                    "视觉输出形状 {} 无法解析".format(list(out0.shape))
                )
            feats.append(np.float16(out0))
        features = np.concatenate(feats, axis=0) if len(feats) > 1 else feats[0]
        return features, vision.tokens

    # ------------------------------------------------------------------ #
    # 推理
    # ------------------------------------------------------------------ #
    def _tokenize(self, prompt: str) -> List[int]:
        return list(self.tokenizer(prompt)["input_ids"])

    def _field_prefixes(
        self, tokens: Sequence[int], n_fields: int,
    ) -> List[List[int]]:
        """按 ``<decision>`` 的位置切出每个字段的推理前缀。

        参考实现：``tokens[0:position[i]]`` —— 即**截止该字段
        ``<decision>`` 标记之前**的前缀。
        """
        positions = [i for i, t in enumerate(tokens) if int(t) == self._marker_id]
        if len(positions) != n_fields:
            raise RuntimeError(
                "decision token 数量 {} != 字段数 {}".format(len(positions), n_fields)
            )
        return [list(tokens[: pos]) for pos in positions]

    def _record_perf(self, stats: Any, wall_ms: float) -> None:
        """把一次 prefill-only 推理的计时数据**累加**进性能表。

        ⚠️ ``stats`` 是 ``session_run`` 返回的
        ``(n_decode, n_prefill, llm_start, llm_end)``。后两个是 **C 侧**
        的墙钟锚点（``time.perf_counter()``，见 ``wdnai/_librknn.py``）——
        比 Python 侧的墙钟更贴近真正的推理时间（不含打包输入的开销）。

        ⚠️ **prefill-only ⇒ 解码 token 恒为 0**：这次推理的时间全算在
        ``prefill_ms`` 上。所以这个模型的吞吐 = **预填充速率**。

        ⚠️ 墙钟退路：C 侧锚点缺失（老版本 / 异常）时用 Python 侧的
        墙钟 —— 宁可粗一点，也不能没有。
        """
        n_decode = 0
        n_prefill = 0
        dur_ms = wall_ms
        try:
            n_decode, n_prefill, llm_start, llm_end = stats
        except (TypeError, ValueError):
            pass
        else:
            try:
                if llm_end and llm_start:
                    dur_ms = (float(llm_end) - float(llm_start)) * 1000.0
            except (TypeError, ValueError):
                pass
        self._perf["n_decode_tokens"] = self._perf.get("n_decode_tokens", 0.0) + float(n_decode or 0)
        self._perf["n_prefill_tokens"] = self._perf.get("n_prefill_tokens", 0.0) + float(n_prefill or 0)
        self._perf["prefill_ms"] = self._perf.get("prefill_ms", 0.0) + float(dur_ms or 0.0)

    @property
    def perf(self) -> Dict[str, float]:
        """最近一次决策推理的性能表（原始计数）。

        ⚠️ **每次决策前会被清空**（见 :meth:`decide`）⇒ 它只反映**最近
        一次**请求，跟 ``StreamSink.perf`` 的语义一样。

        DDS worker 侧读它、把数字带过 wire（见 ``wdndds/ai_session.py``）。
        """
        return dict(self._perf)

    def _run_field(
        self,
        prefix: Sequence[int],
        image_input: Optional[Any],
    ) -> List[float]:
        """跑一个字段：全新 session + 一次 prefill，返回候选符号 logits。"""
        # 每个字段一个**新 session** —— 隔离 KV / linear-attention / conv 状态。
        if self.rknn_llm.recreate_llm_session() != 0:
            raise RuntimeError("recreate_llm_session failed")
        self._clear_chat_template()

        self._sink.reset()
        kwargs: Dict[str, Any] = {
            "tokens": list(prefix),
            "keep_history": 0,
            "max_new_tokens": 1,
            "prefill_only": True,
            "disable_sampling": True,
        }
        if image_input is not None:
            kwargs["inputs"] = [image_input]
        t_run = time.perf_counter()
        ret, stats = self.rknn_llm.session_run(**kwargs)
        # ⚠️ **stats 以前是丢掉的**（``_stats``）—— 现在留下计时数据。
        # 它是 ``(n_decode, n_prefill, llm_start, llm_end)``：前两个是
        # token 数，后两个是 C 侧的墙钟锚点。
        self._record_perf(stats, (time.perf_counter() - t_run) * 1000.0)
        if ret != 0:
            raise RuntimeError(f"session_run failed: ret={ret}")
        if self._sink.error:
            raise RuntimeError(f"output callback 报错: {self._sink.error}")
        if not self._sink.received:
            raise RuntimeError(
                "没有收到 prefill 结束时的候选符号 logits —— "
                "模型输出张量与 Intern-Decision 不匹配？"
            )
        return self._sink.logits

    def _score_field(
        self,
        field: DecisionField,
        logits: Sequence[float],
        temperature: float,
    ) -> DecisionFieldResult:
        n = len(field.options)
        if n > len(logits):
            raise RuntimeError(
                "字段 {!r} 有 {} 个选项，但只拿到 {} 维 logits".format(
                    field.name, n, len(logits),
                )
            )
        probabilities = calibrate_probabilities(logits[:n], temperature)
        keys = [o.key for o in field.options]
        best = pick_best_option(probabilities, keys)
        symbol = DECISION_SYMBOLS[best]
        label = keys[best]
        score: Optional[float] = None
        noul: Optional[float] = None
        if field.type == "score":
            score = weighted_score(keys, probabilities)
        if field.type == "noul":
            noul = float(probabilities[keys.index("yes")]) if "yes" in keys else None
        return DecisionFieldResult(
            name=field.name,
            type=field.type,
            symbol=symbol,
            label=label,
            probabilities={k: float(p) for k, p in zip(keys, probabilities)},
            logits=[float(x) for x in logits[:n]],
            score=score,
            noul=noul,
        )

    # ------------------------------------------------------------------ #
    # 对外接口
    # ------------------------------------------------------------------ #
    def _resolve_task(
        self, overrides: Dict[str, Any],
    ) -> Tuple[DecisionTask, Dict[str, Any]]:
        """从 override 里取出任务配置，并返回**去掉 task 之后**的剩余 override。

        ``task`` 可以是：

        * :class:`~wdnai.decision.DecisionTask`；
        * 任务 JSON（``dict``）—— 形如 ``{"questions": {...}}``；
        * JSON 字符串。

        ``task`` 必须从 override 里**摘掉**再往下传 —— 否则
        ``decide(state, task, ...)`` 会拿到两份 ``task``（位置参数 +
        关键字参数）→ ``TypeError: got multiple values for argument 'task'``。
        """
        rest = {k: v for k, v in overrides.items() if k != "task"}
        task = overrides.get("task")
        if task is None:
            raise ValueError(
                "Intern-Decision 需要任务配置：请传 task=（DecisionTask / 任务 JSON）"
            )
        if isinstance(task, DecisionTask):
            return task, rest
        if isinstance(task, (str, dict)):
            return parse_decision_task(task), rest
        raise TypeError(
            "task 必须是 DecisionTask / dict / str，得到 {}".format(type(task).__name__)
        )

    def decide(
        self,
        state: Any,
        task: DecisionTask,
        images: Optional[Sequence[np.ndarray]] = None,
        **overrides: Any,
    ) -> DecisionResult:
        """跑一次决策。

        ``state`` 覆盖任务里的 state（通常是用户文本）；``images`` 是
        BGR ndarray 列表（可空）。
        """
        # 不改调用方的 task 对象 —— 复制一份再覆盖 state。
        task = DecisionTask(state=task.state, fields=list(task.fields))
        if state:
            task.state = state
        elif task.state is None:
            # 参考实现的请求构造：
            # ``(*row)["state"] = text[0] ? text : "请根据图片回答问题。";``
            # —— 没给文本（典型是只给图）就回落到默认 state。
            # 不回落的话 prompt 里会写成 ``null``，模型会按「没有 state」
            # 解读，概率与官方对不上。
            task.state = DEFAULT_IMAGE_STATE
        if not task.fields:
            raise ValueError("任务里没有任何决策字段")

        image_list = list(images or [])
        features = None
        tokens_per_image = 0
        # ⚠️ **每次决策前清空性能表** —— 它只反映**最近一次**请求
        # （跟 ``StreamSink.perf`` 的语义一样）。
        self._perf = {}
        if image_list:
            features, tokens_per_image = self._encode_images(image_list)
            # 视觉编码完成 → 释放视觉模型（与参考实现一致），给 LLM 的
            # prefill 腾出 NPU 内存。
            self._release_vision()

        prompt = build_decision_prompt(
            task,
            image_tokens=tokens_per_image,
            image_count=len(image_list),
        )
        tokens = self._tokenize(prompt)
        prefixes = self._field_prefixes(tokens, len(task.fields))

        image_input = None
        if features is not None:
            from .._librknn import Float16, RKNN3Image

            width, height, _ = self._vision_geom or (
                self.spec.image_size, self.spec.image_size, tokens_per_image,
            )
            llm_img = RKNN3Image()
            llm_img.image_embed = features.ctypes.data_as(ctypes.POINTER(Float16))
            llm_img.n_image_tokens = tokens_per_image
            llm_img.n_image = len(image_list)
            llm_img.image_width = width
            llm_img.image_height = height
            llm_img.image_start = "<|vision_start|>".encode("utf-8")
            llm_img.image_end = "<|vision_end|>".encode("utf-8")
            llm_img.image_content = "<|image_pad|>".encode("utf-8")
            image_input = llm_img
            # features 的内存必须活到 session_run 结束
            self._feature_pin = features

        temperature = float(
            overrides.get("temperature") or self.temperature
        )
        results: List[DecisionFieldResult] = []
        try:
            for field, prefix in zip(task.fields, prefixes):
                logits = self._run_field(prefix, image_input)
                results.append(self._score_field(field, logits, temperature))
        finally:
            self._feature_pin = None
        return DecisionResult(fields=results, temperature=temperature)

    def run_text(self, prompt: str, **overrides: Any) -> Tuple[str, List[Any]]:
        """纯文本决策。

        ``prompt`` 是「状态」（用户消息）；任务配置走 ``task=`` override。

        Returns ``(text, attachments)`` —— ``text`` 是答案 JSON
        （``{"字段名": 值}``），附件里是带完整概率 / logits 的明细。
        """
        task, rest = self._resolve_task(overrides)
        result = self.decide(prompt, task, None, **rest)
        return _result_to_payload(result)

    def run(self, prompt: str, **overrides: Any) -> Tuple[str, List[Any]]:
        """Alias of :meth:`run_text` (the session layer calls ``run``)."""
        return self.run_text(prompt, **overrides)

    def run_images(
        self,
        prompt: str,
        images: List[np.ndarray],
        **overrides: Any,
    ) -> Tuple[str, List[Any]]:
        """图文决策。"""
        task, rest = self._resolve_task(overrides)
        result = self.decide(prompt, task, images, **rest)
        return _result_to_payload(result)

    def close(self) -> None:
        try:
            if self._vision is not None:
                self._vision.rknn.release()
        except Exception:  # noqa: BLE001
            pass
        self._vision = None
        try:
            self.rknn_llm.release()
        except Exception:  # noqa: BLE001
            pass
