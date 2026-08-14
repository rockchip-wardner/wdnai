"""Qwen3-VL-4B 视频流 VLM 推理示例。

按一定帧间隔把视频帧送给 ``LLMSession`` 推理，并把分析结果打在终端。

为了避免 VLM 推理阻塞视频播放，检测调用被放到后台 ``asyncio.Task`` 里：
主循环只负责读帧 / 显示帧 / 按 ``--every`` 间隔调度检测，上一帧检测
返回时才在终端输出结果并落盘；主循环每帧之间 ``await asyncio.wait(...)``
显式给后台任务最多一个显示帧（≈25ms）的预算，保证结果回收时机确定，
所以即使 VLM 推理耗时较长，视频也能持续显示（无检测时显示原始帧，
有最近一次标注时短暂显示标注帧）。

模型文件位于固定的本地路径 ``/userdata/models/Qwen3-VL-4B/``，可通过
环境变量直接指定模型目录（调试时用）：

    WDNAI_QWEN3_VL_4B_DIR=/path/to/Qwen3-VL-4B  # 直接指定 4B 模型目录，默认 /userdata/models/Qwen3-VL-4B

命令行参数::

    video                       输入视频路径（位置参数，可选）
    --every N                   每隔 N 帧跑一次 VLM（默认 30 ≈ 1s @ 30fps）
    --max-frames N              最多处理 N 帧，0 表示跑到 EOF（默认 0）
    --no-display                不调用 cv2.imshow（无显示环境 / 批量跑时用）
    --no-rga                    禁用 RGA 硬件色彩转换
    --mode {single,two-stage}   推理模式（默认 single）
    --majority-vote K           每次 VLM 跑 K 次取多数派（默认 1 = 不投票）
    --debug-vlm                 把每次 VLM 调用的 prompt/raw output/parsed result 打出来

两种 ``--mode`` 详解：

``single``（单阶段，默认）
    一次 VLM 调用同时输出 ``carplate``（车牌包围盒）和 ``plate_text``
    （车牌号码）。速度最快、调用次数最少，但 VLM 要同时做定位 + 字符
    识别，对小 / 模糊车牌的号码召回略差。

``two-stage``（两阶段）
    第一次 VLM 调用只输出 ``carplate`` 包围盒；如果有车牌，按 1.5×
    扩边裁剪出 ROI，再对 ROI 跑第二次 VLM（OCR）专门识别号码。
    定位和识别解耦后号码召回更好，但每次命中车牌会跑两次 VLM，速度
    更慢。``two-stage`` 模式下 OCR 也会被丢到后台，主循环同样不阻塞。

坐标系统：``carplate`` 一律使用 **0-1000 归一化坐标**（图像左上角=(0,0)、
右下角=(1000,1000)），prompt 里已经写死，解析端也只走这一条路径，不再
走"启发式判断坐标空间"的老逻辑，避免同一段视频多次跑出来的 bbox
位置在像素空间和归一化空间之间漂移。
"""
import argparse
import asyncio
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import wdnav
from wdnai import Attachment, Generable, Field, LLMSession, SystemLanguageModel
from typing import Literal
from dataclasses import replace


@Generable
class PlateBox:
    """两阶段模式第一步：仅输出车牌位置"""
    carplate: str = Field(
        default="未检测到",
        description="车牌包围盒，**字符串必须带方括号**，格式 [x1,y1,x2,y2]，"
                  "0-1000 归一化坐标（左上=(0,0)、右下=(1000,1000)，x1,y1 左上、x2,y2 右下）；"
                  "无车牌时填 \"未检测到\"。",
    )


@Generable
class PlateOCR:
    """两阶段模式第二步：仅识别车牌号"""
    plate_text: str = Field(
        default="未识别",
        description="车牌号码 = 汉字(省份) + 字母 + 数字，可能含 \"·\"；"
                  "完全无法辨认时填 \"未识别\"。",
    )


@Generable
class Event:
    """单阶段模式：同时输出车牌位置和号码"""
    carplate: str = Field(
        default="未检测到",
        description="车牌包围盒，**字符串必须带方括号**，格式 [x1,y1,x2,y2]，"
                  "0-1000 归一化坐标（左上=(0,0)、右下=(1000,1000)，x1,y1 左上、x2,y2 右下）；"
                  "无车牌时填 \"未检测到\"。",
    )
    plate_text: str = Field(
        default="未识别",
        description="车牌号码 = 汉字(省份) + 字母 + 数字，可能含 \"·\"；"
                  "完全无法辨认时填 \"未识别\"。",
    )


def build_box_prompt(width: int, height: int) -> str:
    return (
        f"这是一段 {width}x{height} 的视频帧。请判断图中是否有中国车牌。\n"
        f"中国车牌特征：细长矩形（长宽比约 4.4:1），常见为蓝/黄/绿/白底。"
        f"如有，输出车牌的完整包围盒 [x1,y1,x2,y2]，使用 **0-1000 归一化坐标**"
        f"（左上=(0,0)、右下=(1000,1000)，x1,y1 左上、x2,y2 右下）。\n"
        f"如无车牌，carplate 填字符串 \"未检测到\"。\n"
        f"输出格式（**必须用下面这种 JSON wrapper**）：\n"
        f'{{"carplate":"[x1,y1,x2,y2]"}}\n'
        f'或 {{"carplate":"未检测到"}}'
    )


def build_single_prompt(width: int, height: int) -> str:
    return (
        f"这是一段 {width}x{height} 的视频帧。请判断图中是否有中国车牌。\n"
        f"中国车牌特征：细长矩形（长宽比约 4.4:1），常见为蓝底白字 / 黄底黑字 / 绿底黑字 / 白底黑字。"
        f"如有，同时输出 (1) 包围盒 [x1,y1,x2,y2]（**0-1000 归一化坐标**："
        f"左上=(0,0)、右下=(1000,1000)，x1,y1 左上、x2,y2 右下）和 (2) 车牌号码。\n"
        f"车牌号码 = 汉字(省份) + 字母 + 数字，可能含居中圆点 \"·\"。"
        f"按图中实际字符输出；看不清的字符用 ? 占位。\n"
        f"无车牌：carplate=\"未检测到\"，plate_text=\"未识别\"。\n"
        f"输出格式（**必须用下面这种 JSON wrapper，字段名 carplate / plate_text 一字不差**）：\n"
        f'{{"carplate":"[x1,y1,x2,y2]","plate_text":"<图中实际字符>"}}\n'
        f'或 {{"carplate":"未检测到","plate_text":"未识别"}}'
    )


def build_ocr_prompt() -> str:
    return (
        "请按图中实际字符识别车牌号码。\n"
        "车牌号码 = 汉字(省份) + 字母 + 数字，可能含居中圆点 \"·\"。\n"
        "看清的字符按图中原样输出；看不清的字符用 ? 占位；"
        "整张图无法辨认返回 \"未识别\"。\n"
        "输出格式（**必须用下面这种 JSON wrapper，字段名 plate_text**）：\n"
        f'{{"plate_text":"<图中实际字符>"}}\n'
        f'或 {{"plate_text":"未识别"}}'
    )


_PLATE_RE = re.compile(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]")
_PLATE_FALLBACK_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)[,\s;]+(\d+(?:\.\d+)?)[,\s;]+(\d+(?:\.\d+)?)[,\s;]+(\d+(?:\.\d+)?)(?![\d.])")


def annotate_frame(frame: np.ndarray, x1: float, y1: float, x2: float, y2: float,
                  text: str) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont
    out_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).copy()
    draw = ImageDraw.Draw(out_pil)
    font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 22)
    ix1, iy1, ix2, iy2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    draw.rectangle([ix1, iy1, ix2, iy2], outline=(0, 255, 0), width=2)
    text_y = iy2 + 26
    if text_y > frame.shape[0] - 4:
        text_y = iy1 - 4
    text_pos = (ix1, text_y - 22)
    bbox = draw.textbbox(text_pos, text, font=font)
    draw.rectangle((bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2), fill=(0, 0, 0))
    draw.text(text_pos, text, font=font, fill=(0, 255, 0))
    return cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)


async def _majority_vote_async(session, k: int, **kwargs) -> Any:
    """Run ``session.respond(**kwargs)`` k times and return the most common parsed result.

    Each call is preceded by ``session.reset()`` so KV cache from
    the previous attempt doesn't leak into the next one — important for
    stability because the LLM is autoregressive: a tiny first-token
    difference can snowball into a totally different final string.

    De-dup key is ``repr(result)`` so two outputs whose parsed schema are
    identical (including bbox coordinates and plate_text) collapse to one
    bucket. ``~k×`` slower than a single inference but the official SDK
    ``majority_vote`` is documented as the recommended mitigation for
    "NPU mis-decodes" (``src/wdnai/session.py:485``).

    ``Counter.most_common(1)[0][0]`` always exists (Counter is non-empty
    by construction: we just appended k items), but mypy doesn't see that,
    so we guard ``top`` explicitly.
    """
    from collections import Counter

    results = []
    for _ in range(k):
        session.reset()
        results.append(await session.respond(**kwargs))
    keys = [repr(r) for r in results]
    top = Counter(keys).most_common(1)[0][0]
    for r in results:
        if repr(r) == top:
            return r
    return results[0]


async def _run_detect_once(session, prompt, frame, schema, *, vote_k: int = 1, debug: bool = False):
    """Run a single detect-mode VLM call on a copy of frame."""
    kwargs = dict(
        prompt=prompt,
        attachment=Attachment(frame),
        generating=schema,
        debug=debug,
    )
    if vote_k <= 1:
        session.reset()
        return await session.respond(**kwargs)
    return await _majority_vote_async(session, vote_k, **kwargs)


async def _run_ocr_once(session, roi, *, vote_k: int = 1, debug: bool = False):
    """Run a single OCR-mode VLM call on a copy of roi."""
    kwargs = dict(
        prompt=build_ocr_prompt(),
        attachment=Attachment(roi),
        generating=PlateOCR,
        debug=debug,
    )
    if vote_k <= 1:
        session.reset()
        return await session.respond(**kwargs)
    return await _majority_vote_async(session, vote_k, **kwargs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "video",
        nargs="?",
        default="/userdata/3339824657b2c8e9a861cf0e2ac53f68_raw.mp4",
        help="输入视频路径（默认 /userdata/.../raw.mp4）",
    )
    p.add_argument(
        "--every", type=int, default=30,
        help="每隔多少帧跑一次 VLM（默认 30 ≈ 1s @ 30fps；VLM 处理慢时可调大）",
    )
    p.add_argument(
        "--max-frames", type=int, default=0,
        help="最多处理多少帧（0 表示一直跑到 EOF）",
    )
    p.add_argument(
        "--no-display", action="store_true",
        help="无显示环境时使用：不调用 cv2.imshow",
    )
    p.add_argument(
        "--no-rga", action="store_true",
        help="禁用 RGA 硬件色彩转换（遇 RK3588 等机型上 RGA 段错误时使用）",
    )
    p.add_argument(
        "--mode", choices=["single", "two-stage"], default="single",
        help="推理模式: 'single' = 单次同时输出位置+号码; 'two-stage' = 先位置后号码",
    )
    p.add_argument(
        "--majority-vote", type=int, default=1, metavar="K",
        help="每次 VLM 调用跑 K 次取多数派（默认 1 = 不投票）。NPU mis-decode "
             "时设 3 通常能把 bbox/号码的抖动压平，代价是 K× 延迟。",
    )
    p.add_argument(
        "--debug-vlm", action="store_true",
        help="把每次 VLM 调用的 prompt / attachments / raw output / parsed result "
             "都打到终端。两次跑一对比就能看到 NPU 在哪个 token 上开始分叉。",
    )
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    # 模型文件需要在 /userdata/models/Qwen3-VL-4B/ 下（可用 WDNAI_QWEN3_VL_4B_DIR 覆盖）
    # Pruned vision RKNN: Python 用 qwen3vl_prune_single 算 patch features 喂进去,
    # 可在 32 倍数 image_size 之间切换(384/768/512)不用重新导出 RKNN。
    model = SystemLanguageModel.qwen3_vl_4b()
    model = replace(model, use_prune=True, image_size=768)
    # model = replace(model, use_prune=False, image_size=768)
    # 强制 greedy：默认 options 里 temperature=1e-6, top_k=1 已经接近 argmax，
    # 但 top_p=0.9 在 C 库里可能先于 temperature 起作用、保留 2~3 个候选 token
    # 再采样。把 top_p 也置 1.0 把这条潜在路径关掉，配合 top_k=1 走纯 argmax。
    model = model.with_options(top_p=1.0)
    print(
        f"[model] {model.name}  use_prune={model.use_prune}  "
        f"image_size={model.image_size}  "
        f"temperature={model.options['temperature']}  "
        f"top_k={model.options['top_k']}  top_p={model.options['top_p']}"
    )
    session = LLMSession(model=model)

    cap = wdnav.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    print(
        f"[video] {video_path.name}  {cap.width}x{cap.height} @ {cap.fps:.2f}fps  "
        f"every {args.every} frame(s)"
    )

    if args.mode == "single":
        prompt = build_single_prompt(cap.width, cap.height)
        schema = Event
    else:
        prompt = build_box_prompt(cap.width, cap.height)
        schema = PlateBox
    out_dir = Path.cwd() / video_path.stem
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    print(f"[out] {out_dir}")
    processed = 0
    plate_idx = 0

    # ---- 后台 VLM 状态 ----
    # detect_task: 当前正在跑的后台 VLM 任务（detect 或 ocr）；None 表示空闲
    # detect_kind: "detect" | "ocr" | None
    # detect_frame: 触发本次后台任务的原始帧（用于在 detect/ocr 完成后画标注）
    # detect_pos:   触发本次后台任务时的 frame_pos（用于日志和过期的窗口）
    # ocr_box / ocr_out_path: 两阶段模式下，OCR 完成时需要回填的坐标 + 落盘路径
    detect_task: Optional[asyncio.Task] = None
    detect_kind: Optional[str] = None
    detect_frame: Optional[np.ndarray] = None
    detect_pos: int = -1
    ocr_box: Optional[Tuple[float, float, float, float]] = None
    ocr_out_path: Optional[Path] = None

    # 最近一次成功检测的标注帧；显示到 display_until_pos 之后回到原始帧
    display_frame: Optional[np.ndarray] = None
    display_until_pos: int = -1

    try:
        while True:
            # ---- 1. --no-display 快路径：不到检测间隔时只前进不解码 ----
            if (
                args.no_display
                and detect_task is None
                and display_frame is None
                and cap.frame_pos % args.every != 0
            ):
                if not cap.grab():
                    print("[video] EOF")
                    break
                await asyncio.sleep(0.001)
                continue

            # ---- 2. 读取下一帧 ----
            ret, frame = cap.read()
            if not ret:
                print("[video] EOF")
                break

            # ---- 3. 若上次后台任务完成，收取结果并落盘 ----
            if detect_task is not None and detect_task.done():
                kind = detect_kind
                try:
                    result = detect_task.result()
                except Exception as e:
                    print(f"  vlm {kind} error: {e}")
                    result = None
                detect_task = None
                detect_kind = None

                if kind == "detect" and result is not None:
                    # detect 路径接受两种 schema：two-stage 第一步返回 PlateBox，
                    # single 模式返回 Event（同时含 carplate 和 plate_text）。
                    if not isinstance(result, (PlateBox, Event)):
                        # NPU mis-decode / schema 不匹配时，SDK 会 fallback 到
                        # 直接返回 raw string（src/wdnai/session.py:439）。这种
                        # 帧直接 skip，不计数、不调度 OCR、不更新 display。
                        snippet = result if isinstance(result, str) else type(result).__name__
                        print(
                            f"\n========== frame {detect_pos} (parse failed) ==========\n"
                            f"  ⚠ vlm 未输出合法 PlateBox/Event (got {type(result).__name__})，跳过本帧\n"
                            f"  raw = {repr(snippet)[:200]}"
                        )
                    else:
                        processed += 1
                        coord = result.carplate.strip()
                        print(f"\n========== frame {detect_pos} ==========")
                        print(f"  carplate  = {coord!r}")

                        m = _PLATE_RE.search(coord) or _PLATE_FALLBACK_RE.search(coord)
                        if m is None:
                            print("  saved    = (skipped, no plate)")
                        else:
                            nx1, ny1, nx2, ny2 = (float(g) for g in m.groups())
                            inverted = (nx2 < nx1) or (ny2 < ny1)
                            nx1, nx2 = min(nx1, nx2), max(nx1, nx2)
                            ny1, ny2 = min(ny1, ny2), max(ny2, ny1)
                            # 始终按 0-1000 归一化坐标解析（与 prompt 中的硬约束一致）；
                            # 这里再做一次范围裁剪，防止 VLM 偶尔给出 [-5, 1050] 之类的越界值
                            # 把 ROI 拉扯出图像外。
                            nx1, nx2 = max(0.0, min(1000.0, nx1)), max(0.0, min(1000.0, nx2))
                            ny1, ny2 = max(0.0, min(1000.0, ny1)), max(0.0, min(1000.0, ny2))
                            h, w = detect_frame.shape[:2]
                            sx, sy = w / 1000.0, h / 1000.0
                            scheme = "norm1000"
                            x1, x2 = nx1 * sx, nx2 * sx
                            y1, y2 = ny1 * sy, ny2 * sy
                            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                            # Adaptive margin: when the predicted box is
                            # comfortably inside the frame, expand 1.5×
                            # (good OCR coverage for the use_prune=False path
                            # whose bbox is typically a few pixels off). When
                            # the box is close to a frame edge, a 1.5×
                            # expansion would overflow and force us to clip
                            # half of the plate — fall back to a tighter 1.2×
                            # so OCR still sees a complete character set.
                            border_dist = min(x1, y1, w - x2, h - y2)
                            margin = 1.5 if border_dist > 10 else 1.2
                            bw, bh = (x2 - x1) * margin, (y2 - y1) * margin
                            x1, x2 = cx - bw / 2.0, cx + bw / 2.0
                            y1, y2 = cy - bh / 2.0, cy + bh / 2.0
                            if x1 >= w or y1 >= h or x2 <= 0 or y2 <= 0:
                                print(f"  saved    = (skipped, box out of frame: {nx1},{ny1},{nx2},{ny2})")
                            else:
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(w, x2), min(h, y2)
                                if x2 <= x1 or y2 <= y1:
                                    print(f"  saved    = (skipped, degenerate box after clip: {nx1},{ny1},{nx2},{ny2})")
                                else:
                                    plate_idx += 1
                                    out_path = out_dir / f"plate_{plate_idx:06d}.png"
                                    roi = detect_frame[int(round(y1)):int(round(y2)),
                                                        int(round(x1)):int(round(x2))]
                                    cv2.imwrite(str(out_path), roi)
                                    tag = f" ({scheme})" + (" (inverted)" if inverted else "") + f" (x{margin})"
                                    print(f"  saved    = {out_path.name}{tag}")

                                    if args.mode == "single":
                                        plate_text = result.plate_text.strip()
                                        print(f"  plate    = {plate_text!r}")
                                        txt_path = out_path.with_suffix(".txt")
                                        txt_path.write_text(plate_text, encoding="utf-8")
                                        print(f"  txt      = {txt_path.name}")
                                        display_frame = annotate_frame(
                                            detect_frame, x1, y1, x2, y2, plate_text,
                                        )
                                        display_until_pos = cap.frame_pos + args.every
                                    else:
                                        # 把 OCR 也丢到后台跑，不阻塞主循环
                                        ocr_box = (x1, y1, x2, y2)
                                        ocr_out_path = out_path
                                        detect_kind = "ocr"
                                        detect_task = asyncio.create_task(
                                            _run_ocr_once(
                                                session, roi.copy(),
                                                vote_k=args.majority_vote,
                                                debug=args.debug_vlm,
                                            )
                                        )

                elif kind == "ocr" and result is not None:
                    if not isinstance(result, PlateOCR):
                        snippet = result if isinstance(result, str) else type(result).__name__
                        print(
                            f"  ⚠ ocr 未输出合法 PlateOCR (got {type(result).__name__})，跳过\n"
                            f"  raw = {repr(snippet)[:200]}"
                        )
                        ocr_box = None
                        ocr_out_path = None
                    else:
                        plate_text = result.plate_text.strip()
                        print(f"  ocr      = {plate_text!r}")
                        txt_path = ocr_out_path.with_suffix(".txt")
                        txt_path.write_text(plate_text, encoding="utf-8")
                        print(f"  txt      = {txt_path.name}")
                        x1, y1, x2, y2 = ocr_box
                        display_frame = annotate_frame(
                            detect_frame, x1, y1, x2, y2, plate_text,
                        )
                        display_until_pos = cap.frame_pos + args.every
                        ocr_box = None
                        ocr_out_path = None

            # ---- 4. 调度新一轮检测（仅当后台空闲且到了 --every 间隔） ----
            if detect_task is None and cap.frame_pos % args.every == 0:
                detect_kind = "detect"
                detect_pos = cap.frame_pos
                detect_frame = frame.copy()
                detect_task = asyncio.create_task(
                    _run_detect_once(
                        session, prompt, detect_frame, schema,
                        vote_k=args.majority_vote,
                        debug=args.debug_vlm,
                    )
                )

            # ---- 5. 显示当前帧（无最近标注时是原始帧） ----
            if display_frame is not None and cap.frame_pos > display_until_pos:
                display_frame = None  # 标注窗口结束，回到原帧
            view = display_frame if display_frame is not None else frame
            if not args.no_display:
                cv2.imshow("Video Player", view)
                # 25ms 延迟 + 检测 'q' 退出
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    print("[video] user quit")
                    break

            # ---- 6. 显式等待后台 VLM 任务（最多 25ms = 一个显示帧） ----
            # 原本 await asyncio.sleep(0) 只让出一次事件循环；如果后台 VLM
            # 还没跑完，主循环又会进入下一帧，结果回收时机完全不确定，
            # 多次跑同一段视频得到的框/号码会忽有忽无。
            # 这里改用 asyncio.wait 给后台任务最多一个显示帧的预算：
            # - 任务已完成 → 立即返回（下一轮开头 detect_task.done() 为 True 就回收）
            # - 任务还在跑 → 让出直到超时或完成，确保后台至少有固定时长推进
            if detect_task is not None:
                _, _ = await asyncio.wait({detect_task}, timeout=0.025)

            if args.max_frames and cap.frame_pos >= args.max_frames:
                print(f"[video] reached max-frames={args.max_frames}")
                break
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        # 退出前取消还在跑的后台任务，避免 session.close() 与在飞请求冲突
        if detect_task is not None and not detect_task.done():
            detect_task.cancel()
            try:
                await detect_task
            except (asyncio.CancelledError, Exception):
                pass
        session.close()

    print(f"\n[done] {processed} frame(s) processed, {plate_idx} plate(s) saved to {out_dir}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)