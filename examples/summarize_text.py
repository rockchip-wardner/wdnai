"""Qwen3-8B 推理示例 —— 纯文本总结。

模型文件位于固定的本地路径 ``/userdata/models/Qwen3-8B/``：

    Qwen3-8B.1828.rknn / .weight
    Qwen3-8B.embed.bin
    Qwen3-8B.tokenizer.gguf

可通过环境变量更改根目录：

    RKAI_QWEN3_8B_DIR=/path/to/Qwen3-8B

Usage::

    .venv/bin/python examples/summarize_text.py
    .venv/bin/python examples/summarize_text.py --iter 3 --perf
    .venv/bin/python examples/summarize_text.py --model-dir /opt/Qwen3-8B
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

from rkai import LLMSession, SystemLanguageModel

LOG = logging.getLogger("summarize_text")


_NEWS_PARAGRAPH = (
    "请总结一下以下这段话，要求不超过50个字："
    "“8月3日，财闻海外资讯消息，美国国土安全部牵头的相关工作组更新了对中国企业的"
    "进口限制清单，将在美国当地时间8月3日生效。包括洽洽食品、思念食品、七匹狼等"
    "在内的多家中国企业纳入限制范围，对其生产的相关产品进入美国市场实施进口限制。"
    "这意味着，曾经摆上美国消费者餐桌的瓜子、饺子、汤圆等中国食品，以及进入全球"
    "市场多年的中国服装品牌，都面临更高的贸易准入门槛。\n"
    "根据美国相关规定，被列入企业生产的商品进入美国市场将受到进口限制，除非进口商"
    "能够提供证明材料证明相关产品符合要求。\n"
    "此次调整涉及食品、电子、金属等多个行业，是美国近年来扩大相关贸易限制措施的"
    "一部分。\n"
    "据南华早报8月1日报道，白宫周五再次拿莫须有的新疆“强迫劳动”说事，将43家中国"
    "企业列入一份黑名单。\n"
    "为执行所谓“维吾尔强迫劳动预防法”，美国国土安全部将相关实体清单扩大了30%，"
    "新增企业涵盖食品、棉花、制药、金属以及锂生产等多个行业。\n"
    "该法案于2021年由美国国会通过，旨在针对来自中国新疆的进口商品，并阻止被指在"
    "所谓“强迫劳动”条件下生产的产品进入美国市场。\n"
    "报道称，这是特朗普政府上任以来首次扩大该名单，使名单上的实体数量从144家"
    "增至187家，也是自该法案于2021年签署成为法律以来，单次新增实体数量最多的"
    "一次。美国海关与边境保护局将于8月3日起开始禁止这些企业的产品进口美国。\n"
    "美国国会“中国问题特别委员会”主席约翰·穆勒纳尔（John Moolenaar）叫嚣，"
    "特朗普政府此次行动是在向中国传递信号，不会对其行为视而不见。\n"
    "中国驻美国大使馆发言人谴责美方扩大黑名单，中方坚决反对美国利用所谓‘强迫"
    "劳动’问题抹黑中国、干涉中国内政、打压中国企业。\n"
    "中国外交部也曾指出，所谓新疆存在“强迫劳动”原本就是反华势力为抹黑中国炮制"
    "的世纪谎言，同新疆各族人民劳动权益得到切实保障的事实完全相反。实质是破坏"
    "新疆繁荣稳定，遏制中国发展。\n"
    "就在7月24日，美国就以301调查为幌子，指责强迫劳动产品进入供应链，对全球60个"
    "国家和地区实施新一轮关税。”"
)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--prompt", default=_NEWS_PARAGRAPH,
                   help="要总结的文本（默认是内置的中文新闻段落）")
    p.add_argument("--iter", type=int, default=1,
                   help="连续推理轮数（每轮 reset KV cache）")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="覆盖 spec.options['max_new_tokens']（不传则用 preset 默认）")
    p.add_argument("--model-dir", type=Path, default=None,
                   help="覆盖模型根目录；等价于 RKAI_QWEN3_8B_DIR")
    p.add_argument("--npu", type=int, default=None,
                   help="多卡场景下选第 K 张卡（0-based index）")
    p.add_argument("--perf", action="store_true",
                   help="打印 Prefill / Generate 延迟表")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
async def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.model_dir:
        os.environ["RKAI_QWEN3_8B_DIR"] = str(args.model_dir)
    model = SystemLanguageModel.qwen3_8b()
    if args.max_new_tokens is not None:
        model = model.with_options(max_new_tokens=args.max_new_tokens)

    session = LLMSession(model=model, npu=args.npu)
    try:
        for i in range(max(1, args.iter)):
            print(f"\n========== run {i} ==========")
            t0 = time.perf_counter()
            session.reset()
            result = await session.respond(
                prompt=args.prompt,
                perf=args.perf,
            )
            wall = time.perf_counter() - t0
            if isinstance(result, str):
                print(f"  result = {result!r}")
            else:
                print(f"  result = {result!r} (unexpected type)")
            print(f"  wall   = {wall:.2f} s")
    finally:
        session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)