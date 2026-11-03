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

⭐ **批量**（2026-11-02 第三十五轮）：提示词用 ``|`` 分隔 ⇒ **一个进程跑完 N 张**
（TE / DiT / VAE 都只 ``model_init`` 一次）⇒ 每张图的边际成本只有
flow-match + VAE ≈ 34 s::

    imgs = await session.generate_images(["a cat", "a dog"])

模型布局（默认 ``$WDNAI_MODELS_DIR/qwenimage21``，板上实际在
``/home/rockchip/models/qwenimage21``；可用 ``WDNAI_QWEN_IMAGE_21_DIR``
或 ``QI_MODEL_DIR`` 覆盖）::

    llm_qwenimage21-te.1828.rknn + .weight       text encoder
    qwenimage21-te.embed.bin + .tokenizer.gguf   tokenizer / embedding
    dit_qwenimage21-512hc.seg{0..3}.1828.rknn + .weight   DiT（4 × 8 blocks，切头）
    dit_qwenimage21-512.head.1828.rknn + .weight            DiT 的「头」（独立卡）
    dit_qwenimage21-512hc.int_ledger                         每核 internal 账本
    vae_qwenimage21-512.seg{0..3}.1828.rknn + .weight   VAE（4 段链）

⭐ **第三十六轮（2026-11-03）：发布包里只剩「模型」** —— 运行时是 wdnai 自己的
C shim（``wdnai/_cshim/libqi_qwen.so``，**源码随包发布**），
它只链**系统自带**的 ``librknn3_api.so`` ⇒
**不再需要 ``runtime/qwen_image_rknn``，也不需要任何 Rockchip 的 .so**。

⭐ 连「固定初始噪声」也是 cshim **现场生成**的（numpy 兼容的 PCG64 + ziggurat，
与历史基线**逐字节一致**）⇒ 模型包里**连数据文件都不用带**。

⏱ 耗时（板端实测，RK3588 + 2×RK1828，8 步）：**TE 与 DiT 常驻**
（卡 0 = TE + 头，卡 1 = DiT 4 段 → VAE），一次性 init ≈ 78 s，
之后**每张图 ≈ 34 s**（flow-match 21 s + VAE 11 s），TE 编码只要 0.16 s。
步数是纯运行时参数，模型不用重建。

执行::

    # 单张
    .venv/bin/python .venv/src/wdnai/examples/qwen_image_21.py \\
        "a cute cat sitting on a wooden chair" 8 --out /tmp/qi21

    # 批量（一个进程跑完 N 张）
    .venv/bin/python .venv/src/wdnai/examples/qwen_image_21.py \\
        "a cute cat sitting on a wooden chair|一只戴眼镜的猫" 8 --out /tmp/qi21
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import time
from pathlib import Path
from typing import List, Optional

from wdnai import (
    LLMLocalSession,
    SystemLanguageModel,
)

LOG = logging.getLogger("qwen_image_21")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """解析命令行。

    刻意与旧的 ``samples/qwenimage21.sh`` **对齐**（同一个脚本干同一件事）：

    * 提示词是**位置参数**（也接受 ``--prompt``）
    * 步数是**位置参数**（也接受 ``--steps``）
    * ``--out`` 给的是**目录** ⇒ 产物是 ``img01.png`` / ``img02.png`` …
    * ``|`` 分隔多个提示词 ⇒ **批量**（一个进程跑完 N 张）
    * ``--fresh`` / ``--fixed`` 切噪声模式
    """
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("prompt", nargs="?", default=None,
                   help='要画的提示词；用 "|" 分隔多个 ⇒ 批量（一个进程跑完 N 张）')
    p.add_argument("steps", nargs="?", type=int, default=None,
                   help="flow-match 步数（8 步稳态 ≈ 34 s/张；20 步更清晰 ≈ 55 s/张）")
    p.add_argument("--prompt", dest="prompt_opt", default=None,
                   help="提示词（位置参数的别名）")
    p.add_argument("--steps", dest="steps_opt", type=int, default=None,
                   help="flow-match 步数（位置参数的别名）")
    p.add_argument("--out", "--output", "-o", dest="out", type=Path,
                   default=Path("/tmp/qi21"),
                   help="输出**目录** ⇒ 产物为 img01.png / img02.png …")
    p.add_argument("--model-dir", type=Path, default=None,
                   help="覆盖模型包目录；默认自动解析 $WDNAI_MODELS_DIR/qwenimage21")
    p.add_argument("--seed", type=int, default=0,
                   help="初始噪声种子（配合 --fixed 可复现）")
    p.add_argument("--fixed", dest="fixed", action="store_true",
                   help="用固定噪声（cshim 现场生成，与历史基线逐字节一致 ⇒ 可对拍 md5）")
    p.add_argument("--fresh", dest="fixed", action="store_false",
                   help="每张图用随机噪声（默认）")
    p.add_argument("--keep-artifacts", action="store_true",
                   help="保留中间产物（潜变量 / VAE 原始张量）")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fixed=False)
    args = p.parse_args(argv)

    # 位置参数与 --xxx 别名合并（位置参数优先）
    if args.prompt_opt is not None:
        args.prompt = args.prompt_opt
    if args.steps_opt is not None:
        args.steps = args.steps_opt
    if args.prompt is None:
        p.error("需要提示词（位置参数，或 --prompt）")
    if args.steps is None:
        args.steps = 8
    return args


def split_prompts(raw: str) -> List[str]:
    """按 ``|`` 拆成多个提示词（去空白、丢空项）。"""
    return [p.strip() for p in raw.split("|") if p.strip()]


def resolve_output_dir(raw: Path) -> tuple[Path, Optional[Path]]:
    """把 ``--out`` 拆成「(输出目录, 可选的重命名目标)」。

    * 目录 / 无后缀 ⇒ ``(raw, None)``，产物为 ``img01.png`` / ``img02.png`` …
    * ``.png`` 文件路径 ⇒ ``(父目录, 该路径)``，单张时改名过去

    ⚠️ PNG 由**后端**写（纯 stdlib zlib，与运行时那份**逐字节相同**），
    这里只做复制/改名 ⇒ **不重新编码** ⇒ 输出与运行时完全一致。
    """
    if raw.suffix.lower() == ".png" and not raw.is_dir():
        parent = raw.parent if str(raw.parent) else Path(".")
        return parent, raw
    return raw, None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
async def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    prompts = split_prompts(args.prompt)
    if not prompts:
        raise SystemExit("提示词为空")

    model = SystemLanguageModel.qwen_image_21_512()
    if args.model_dir:
        model.image_model_dir = str(args.model_dir)

    out_dir, rename_to = resolve_output_dir(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = LLMLocalSession(model=model)
    try:
        t0 = time.perf_counter()
        # ⭐ 批量：**一个进程**跑完 N 张（TE / DiT / VAE 都只 init 一次）
        images = await session.generate_images(
            prompts,
            steps=args.steps,
            noise_fresh=not args.fixed,
            noise_seed=args.seed,
            keep_artifacts=args.keep_artifacts,
            output_dir=str(out_dir),
        )
        elapsed = time.perf_counter() - t0

        for i, img in enumerate(images, start=1):
            png = out_dir / "img{:02d}.png".format(i)
            if rename_to is not None:
                # 用户给了具体文件名 ⇒ 单张直接改名，多张加序号
                dst = rename_to if len(images) == 1 else rename_to.with_name(
                    "{}_{:02d}{}".format(rename_to.stem, i, rename_to.suffix)
                )
                if png.is_file() and png != dst:
                    shutil.move(str(png), str(dst))
                    png = dst
            LOG.info(
                "第 %d/%d 张 shape=%s dtype=%s -> %s",
                i, len(images), img.shape, img.dtype, png,
            )

        print(
            "[qwen-image-2.1] done. {} 张, steps={} 总耗时 {:.1f}s "
            "（{:.1f}s/张）-> {}".format(
                len(images), args.steps, elapsed,
                elapsed / max(len(images), 1), out_dir,
            )
        )
    finally:
        session.close()


if __name__ == "__main__":
    asyncio.run(main())
