"""Qwen3.5-VL-4B 视觉推理示例（图像描述）。

模型文件位于固定的本地路径 ``/userdata/models/Qwen3.5-4B/``：

    llm_Qwen3.5-4B-VL.1828.rknn / .weight           (VLM-mode LLM，MRoPE 烙印)
    vision_Qwen3.5-4B-VL.384x384.1828.rknn / .weight (必需 384x384 vision)
    vision_Qwen3.5-4B-VL.768x768.1828.rknn / .weight (可选 768x768 vision)
    llm_Qwen3.5-4B-VL.embed.bin
    llm_Qwen3.5-4B-VL.tokenizer.gguf

可通过环境变量更改根目录：

    WDNAI_QWEN3_5_4B_VL_DIR=/path/to/Qwen3.5-4B

与 ``qwen3_5_text.py`` 的区别：本示例加载的是 VLM-mode LLM
（``llm_Qwen3.5-4B-VL.1828.rknn``），可以消费 vision tokens。视觉
预处理路径与 Qwen3-VL 修剪版一致（``qwen3vl_prune_single``）。

When the ``Qwen3.5-4B/`` directory carries both the 384x384 and the
768x768 vision model pairs, ``LLMSession`` auto-loads both at init and
picks the closest variant per call based on the input image's longest
side. See [wdnai/models/README.md](../src/wdnai/models/README.md) for
the multi-resolution naming convention.

Usage::

    .venv/bin/python examples/qwen3_5_vl_image.py
    .venv/bin/python examples/qwen3_5_vl_image.py --image demo.jpg --iter 3 --perf
    .venv/bin/python examples/qwen3_5_vl_image.py --model-dir /opt/Qwen3.5-4B
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2

from wdnai import (
    Attachment,
    Generable,
    LMOutput,
    LLMSession,
    SystemLanguageModel,
)
from pydantic import Field
from typing import Literal

LOG = logging.getLogger("qwen3_5_vl_image")


@Generable
class ImageCaption(LMOutput):
    """图片描述结构化输出。"""
    text: str = Field(description="图片内容的自然语言描述")
    level: Literal["Normal", "Warning"] = Field(
        default="Normal", description="描述的安全等级标签",
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--image", default="demo.jpg",
                   help="输入图片路径（默认 demo.jpg）")
    p.add_argument("--prompt", default="请用一句话描述这张图片的内容。",
                   help="图片描述 prompt")
    p.add_argument("--iter", type=int, default=1,
                   help="连续推理轮数（每轮 reset KV cache）")
    p.add_argument("--image-size", type=int, default=None,
                   help="强制使用某个 vision 变体（384 或 768）；不传则按图像尺寸自动选")
    p.add_argument("--model-dir", type=Path, default=None,
                   help="覆盖模型根目录；等价于 WDNAI_QWEN3_5_4B_VL_DIR")
    p.add_argument("--npu", type=int, default=None,
                   help="多卡场景下选第 K 张卡（0-based index）")
    p.add_argument("--perf", action="store_true",
                   help="打印 Prefill / Generate 延迟表")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


async def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"图片读取失败: {args.image}")

    if args.model_dir:
        os.environ["WDNAI_QWEN3_5_4B_VL_DIR"] = str(args.model_dir)
    model = SystemLanguageModel.qwen3_5_vl_4b()
    LOG.info("loaded model preset (image_size=%d, use_prune=%s, max_context_len=%d)",
             model.image_size, model.use_prune, model.max_context_len)

    session = LLMSession(model=model, npu=args.npu)
    try:
        for i in range(max(1, args.iter)):
            print(f"\n========== run {i} ==========")
            t0 = time.perf_counter()
            session.reset()
            result = await session.respond(
                prompt=args.prompt,
                attachment=[Attachment(img)],
                generating=ImageCaption,
                image_size=args.image_size,
                perf=args.perf,
            )
            wall = time.perf_counter() - t0
            if isinstance(result, ImageCaption):
                print(f"  text  = {result.text!r}")
                print(f"  level = {result.level!r}")
            else:
                # Model didn't emit JSON matching the schema — show the
                # raw result so the user can see what came back.
                print(f"  raw   = {result!r}")
            print(f"  wall  = {wall:.2f} s")
    finally:
        session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
