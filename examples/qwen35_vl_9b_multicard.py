"""Qwen3.5-VL-9B two-card smoke test.

Requires RK182x #0 and #1 to be free (the model occupies both).
"""
from __future__ import annotations

import argparse
import asyncio

import cv2

from wdnai import Attachment, LLMLocalSession, SystemLanguageModel


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npu", default="0,1")
    ap.add_argument("--image", default="/userdata/test/bus.jpg")
    ap.add_argument("--prompt", default="请用一句话描述这张图片。")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--thinking", action="store_true")
    args = ap.parse_args()

    npu_indices = [int(x) for x in args.npu.split(",") if x.strip()]
    model = SystemLanguageModel.qwen3_5_vl_9b()
    sess = LLMLocalSession(model=model, npu_indices=npu_indices)
    try:
        print("== text ==")
        print(await sess.respond(
            prompt="你好，请用一句话介绍你自己。",
            max_new_tokens=args.max_new_tokens,
        ))

        img = cv2.imread(args.image)
        if img is None:
            raise FileNotFoundError(args.image)
        print("== image ==")
        print(await sess.respond(
            prompt=args.prompt,
            attachment=[Attachment(img)],
            max_new_tokens=args.max_new_tokens,
        ))

        sess.reset()
        print("== thinking ==")
        print(await sess.respond(
            prompt="用一句话解释什么是视觉语言模型。",
            enable_thinking=args.thinking,
            max_new_tokens=args.max_new_tokens,
        ))
    finally:
        sess.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
