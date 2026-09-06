"""LocateAnything grounding demo —— v2 ``LLMSession`` API 风格。

LocateAnything 是 Qwen-VL 系列里专门做「在图里找东西」的模型：
给定一张 BGR 图 + 一段自然语言 query，模型回一段
``<ref>label</ref><box>x1 y1 x2 y2</box>`` 形式的 grounding 文本，
SDK 在 backend 里解析成结构化的 :class:`~wdnai.grounding.BoundingBox`
列表。

模型布局（与 ``examples/LocateAnything/cpp/main.cc`` 对齐）：

    /userdata/models/LocateAnything-3B/
        llm_LocateAnything.1828.rknn + .weight
        vision_LocateAnything.448x448.1828.rknn + .weight
        LocateAnything.embed.bin
        LocateAnything.tokenizer.gguf

可用环境变量 ``WDNAI_MODELS_DIR`` 覆盖根目录，或
``WDNAI_LOCATE_ANYTHING_DIR`` 直接指向具体目录。

执行（默认读取本目录的 bus.jpg —— 与 detect_image.py 的 demo.jpg 同款套路；
    按需 ``--image other.jpg`` 覆盖即可）：

    .venv/bin/python .venv/src/wdnai/examples/locate_image.py
    .venv/bin/python .venv/src/wdnai/examples/locate_image.py \\
        --image bus.jpg \\
        --prompt "定位图里的公共汽车"
"""
from __future__ import annotations

import argparse
import sys
from typing import List

from wdnai import Attachment, LLMSession, SystemLanguageModel

from wdnai.grounding import GroundingResult


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--image", default="bus.jpg",
                   help="输入图片路径（BGR 顺序，cv2.imread 解码；默认本目录下的 bus.jpg —— "
                        "rknn3-model-zoo 的 LocateAnything 演示样张，作为本脚本的 canonical test image）")
    p.add_argument("--prompt", default="",
                   help="grounding 查询，例如「定位图里的公共汽车」（默认空字符串；SDK 会按空 prompt 处理）")
    p.add_argument("--max-new-tokens", type=int, default=1024,
                   help="LLM 最大生成 token 数（默认 1024）")
    p.add_argument("--npu", type=int, default=0,
                   help="NPU 卡索引，对应 RKNN3Lite.get_devices_id() 的下标（默认 0）")
    p.add_argument("--debug", action="store_true",
                   help="透传 SDK 的 debug 标志，输出 perf / prefill / generate 等耗时")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
async def main() -> None:
    args = parse_args()

    try:
        model = SystemLanguageModel.locate_anything()
    except FileNotFoundError as e:
        sys.stderr.write(
            "[locate_image] LocateAnything 模型文件未就绪。"
            "请把模型放到 $WDNAI_LOCATE_ANYTHING_DIR 或 $WDNAI_MODELS_DIR/LocateAnything-3B 下，"
            "并确认 .1828.rknn / .weight / .embed.bin / .tokenizer.gguf 齐全。"
            f"\n  原始错误：{e}\n"
        )
        sys.exit(1)

    import cv2

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"无法读取图片：{args.image}")

    session = LLMSession(model=model, npu=args.npu)
    try:
        result = await session.respond(
            prompt=args.prompt,
            attachment=[Attachment(img)],
            generating=GroundingResult,
            max_new_tokens=args.max_new_tokens,
            debug=args.debug,
        )

        if isinstance(result, GroundingResult):
            print(f"  boxes = {result.boxes!r}")
            print(f"  text  = {result.text!r}")
            print(f"  level = {result.level!r}")
        else:
            print(f"  raw   = {result!r}")
    finally:
        session.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
