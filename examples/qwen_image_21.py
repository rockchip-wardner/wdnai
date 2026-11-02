"""Qwen-Image-2.1 文生图 demo —— 512×512，RKNN3 / rk1820。

整套流水线（text encoder → DiT flow-match → VAE 解码）封装在 SDK 的
``ImageBackend`` 里（参见 ``.venv/src/wdnai/src/wdnai/backends/image.py``），
调用方只负责三件事：

1. 构造 ``model = SystemLanguageModel.qwen_image_21_512()``
2. ``session = LLMLocalSession(model=model)``
3. ``img = await session.generate_image("a cute cat")`` → 拿到 **BGR ndarray**

三条取图路径（等价，按场景选）::

    # ① 最短路径：直接拿像素
    img = await session.generate_image(prompt)

    # ② 走 run_prompt：返回 (text, attachments)，附件里是 ImageAttachment
    text, atts = await session.run_prompt(prompt)
    img = atts[0].images[0]

    # ③ 走 respond + LMOutput：结构化外壳（attachment 自动填充）
    out = await session.respond(prompt, generating=LMOutput)
    img = out.attachment[0].images[0]

模型布局（默认 ``$WDNAI_MODELS_DIR/qwenimage21``，板上实际在
``/home/rockchip/models/qwenimage21``；可用 ``WDNAI_QWEN_IMAGE_21_DIR``
或 ``QI_MODEL_DIR`` 覆盖）::

    llm_qwenimage21-te.1828.rknn + .weight       text encoder
    qwenimage21-te.embed.bin + .tokenizer.gguf   tokenizer / embedding
    dit_qwenimage21-512hc.seg{0..3}.1828.rknn + .weight   DiT（4 × 8 blocks，切头）
    dit_qwenimage21-512.head.1828.rknn + .weight            DiT 的「头」（独立卡）
    dit_qwenimage21-512hc.int_ledger                         每核 internal 账本
    vae_qwenimage21-512.seg{0..3}.1828.rknn + .weight   VAE（4 段链）
    reference_512/hidden_states.bin              初始噪声
    runtime/qwen_image_rknn                       C++ 运行时

⏱ 耗时（板端实测，RK3588 + 2×RK1828，8 步）：**TE 与 DiT 常驻**
（卡 0 = TE + 头，卡 1 = DiT 4 段 → VAE），一次性 init ≈ 78 s，
之后**每张图 ≈ 34 s**（flow-match 21 s + VAE 11 s），TE 编码只要 0.16 s。
步数是纯运行时参数，模型不用重建。

执行::

    .venv/bin/python .venv/src/wdnai/examples/qwen_image_21.py \\
        --prompt "a cute cat sitting on a wooden chair" --steps 8 \\
        --output /tmp/qi21_cat.png
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path
from typing import List

from wdnai import (
    LLMLocalSession,
    SystemLanguageModel,
)

LOG = logging.getLogger("qwen_image_21")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--prompt", required=True, help="要画的提示词")
    p.add_argument("--steps", type=int, default=8,
                   help="flow-match 步数（8 步稳态 ≈ 34 s/张；20 步更清晰 ≈ 55 s/张；40 步 = 上游默认）")
    p.add_argument("--output", "-o", type=Path, default=Path("output.png"),
                   help="输出 PNG 路径")
    p.add_argument("--model-dir", type=Path, default=None,
                   help="覆盖模型包目录；默认自动解析 $WDNAI_MODELS_DIR/qwenimage21")
    p.add_argument("--seed", type=int, default=0,
                   help="初始噪声种子（配合 --fixed-noise 可复现）")
    p.add_argument("--fixed-noise", action="store_true",
                   help="复用模型包里的固定噪声（可与 PyTorch 金标准对拍）")
    p.add_argument("--keep-artifacts", action="store_true",
                   help="保留中间产物（潜变量 / VAE 原始张量）")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def resolve_output_path(raw: Path) -> Path:
    """把 ``--output`` 归一成「带受支持后缀的文件路径」。

    OpenCV 靠扩展名挑编码器：给目录、或者干脆不给后缀，都会报
    ``could not find a writer for the specified extension``（而且
    ``imwrite`` 只返回 False、不抛异常 ⇒ 很容易被当成「出图失败」）。
    这里统一补成 ``<目录>/img01.png`` 或 ``<路径>.png``。
    """
    if raw.is_dir():
        return raw / "img01.png"
    if not raw.suffix:
        return raw.with_suffix(".png")
    return raw


async def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    model = SystemLanguageModel.qwen_image_21_512()
    if args.model_dir:
        model.image_model_dir = str(args.model_dir)

    session = LLMLocalSession(model=model)
    try:
        t0 = time.perf_counter()
        img = await session.generate_image(
            args.prompt,
            steps=args.steps,
            noise_fresh=not args.fixed_noise,
            noise_seed=args.seed,
            keep_artifacts=args.keep_artifacts,
        )
        elapsed = time.perf_counter() - t0

        out_path = resolve_output_path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import cv2

            if not cv2.imwrite(str(out_path), img):
                LOG.warning("写 PNG 失败（%s）", out_path)
        except Exception as exc:  # noqa: BLE001 — 落盘失败不该掩盖出图结果
            LOG.warning("写 PNG 失败（%s）", exc)

        LOG.info("图像 shape=%s dtype=%s", img.shape, img.dtype)
        print(
            f"[qwen-image-2.1] done. steps={args.steps} "
            f"shape={img.shape} elapsed={elapsed:.1f}s -> {out_path}"
        )
    finally:
        session.close()


if __name__ == "__main__":
    asyncio.run(main())
