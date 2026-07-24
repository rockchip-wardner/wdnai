"""Qwen3-VL-4B 视频流 VLM 推理示例。

按一定帧间隔把视频帧送给 ``LLMSession`` 推理，并把分析结果打在终端。

为了避免 VLM 推理阻塞视频播放，检测调用被放到后台 ``asyncio.Task`` 里：
主循环只负责读帧 / 显示帧 / 按 ``--every`` 间隔调度检测，上一帧检测
返回时才在终端输出结果并落盘；主循环每帧之间 ``await asyncio.sleep(0)``
让事件循环有机会推进后台任务，所以即使 VLM 推理耗时较长，视频也能
持续显示（无检测时显示原始帧，有最近一次标注时短暂显示标注帧）。

模型文件位于固定的本地路径 ``/userdata/models/Qwen3-VL-4B/``，可通过
环境变量直接指定模型目录（调试时用）：

    RKAI_QWEN3_VL_4B_DIR=/path/to/Qwen3-VL-4B  # 直接指定 4B 模型目录，默认 /userdata/models/Qwen3-VL-4B

命令行参数::

    video                       输入视频路径（位置参数，可选）
    --every N                   每隔 N 帧跑一次 VLM（默认 30 ≈ 1s @ 30fps）
    --max-frames N              最多处理 N 帧，0 表示跑到 EOF（默认 0）
    --no-display                不调用 cv2.imshow（无显示环境 / 批量跑时用）
    --no-rga                    禁用 RGA 硬件色彩转换
    --mode {single,two-stage}   推理模式（默认 single）

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
"""
import argparse
import asyncio
import re
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import rkv
from rkai import Attachment, Generable, Field, LLMSession, SystemLanguageModel
from typing import Literal
from dataclasses import replace


@Generable
class PlateBox:
    """两阶段模式第一步：仅输出车牌位置"""
    carplate: str = Field(
        default="车牌坐标",
        description="车牌完整包围盒，格式 [x1,y1,x2,y2]，四个角必须紧贴车牌外框边缘；未检测到时返回'未检测到'",
    )


@Generable
class PlateOCR:
    """两阶段模式第二步：仅识别车牌号"""
    plate_text: str = Field(
        default="未识别",
        description="车牌号码（汉字+字母+数字，例如'京A12345'）；无法清晰识别时返回'未识别'",
    )


@Generable
class Event:
    """单阶段模式：同时输出车牌位置和号码"""
    carplate: str = Field(
        default="车牌坐标",
        description="车牌完整包围盒，格式 [x1,y1,x2,y2]，四个角必须紧贴车牌外框边缘；未检测到时返回'未检测到'",
    )
    plate_text: str = Field(
        default="未识别",
        description="车牌号码（汉字+字母+数字，例如'京A12345'）；无法清晰识别时返回'未识别'",
    )


def build_box_prompt(width: int, height: int) -> str:
    return (
        f"这是一段 {width}x{height} 的视频帧。请判断图中是否有车牌。\n"
        f"如有车牌，请给出一行完整包围盒坐标 [x1,y1,x2,y2]，"
        f"要求四个角紧贴车牌外框四个边缘，确保所有汉字、字母和数字完整可见、不被裁切；"
        f"如果无车牌，写'未检测到'。"
    )


def build_single_prompt(width: int, height: int) -> str:
    return (
        f"这是一段 {width}x{height} 的视频帧。请判断图中是否有车牌。\n"
        f"车牌特征：细长矩形（长宽比约 4.4:1），中国常见为蓝底白字 / 黄底黑字 / 绿底黑字，"
        f"内容包含汉字、字母和数字。\n"
        f"如有车牌，请同时给出：\n"
        f"  carplate   = 车牌完整包围盒 [x1,y1,x2,y2]，四个角紧贴车牌外框，确保字符完整可见\n"
        f"  plate_text = 车牌号码（如'京A12345'），如果字符不够清晰也要尽量识别\n"
        f"如果无车牌，carplate 写'未检测到'，plate_text 写'未识别'。"
    )


def build_ocr_prompt() -> str:
    return (
        "请识别图中的车牌号码。这是裁剪并已扩边过的车牌图像，应只包含 1 个车牌。"
        "如果字符清晰可读，直接给出完整车牌内容（例如'京A12345'或'粤B6D89H'）；"
        "如果图像模糊或无法识别，返回'未识别'。"
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


async def _run_detect_once(session, prompt, frame, schema):
    """Run a single detect-mode VLM call on a copy of frame."""
    session.reset()
    return await session.respond(
        prompt=prompt,
        attachment=Attachment(frame),
        generating=schema,
    )


async def _run_ocr_once(session, roi):
    """Run a single OCR-mode VLM call on a copy of roi."""
    return await session.respond(
        prompt=build_ocr_prompt(),
        attachment=Attachment(roi),
        generating=PlateOCR,
    )


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
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    # 模型文件需要在 /userdata/models/Qwen3-VL-4B/ 下（可用 RKAI_QWEN3_VL_4B_DIR 覆盖）
    model = SystemLanguageModel.qwen3_vl_4b()
    # model = replace(model, use_prune=True, image_size=768)
    model = replace(model, use_prune=False, image_size=768)
    print(f"[model] {model.name}  use_prune={model.use_prune}  image_size={model.image_size}")
    session = LLMSession(model=model)

    cap = rkv.VideoCapture(str(video_path))
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
                await asyncio.sleep(0)
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
                        ny1, ny2 = min(ny1, ny2), max(ny1, ny2)
                        h, w = detect_frame.shape[:2]
                        if nx2 > w or ny2 > h:
                            sx, sy = w / 1000.0, h / 1000.0
                            scheme = "norm1000"
                        else:
                            sx, sy = 1.0, 1.0
                            scheme = "pixel"
                        x1, x2 = nx1 * sx, nx2 * sx
                        y1, y2 = ny1 * sy, ny2 * sy
                        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        bw, bh = (x2 - x1) * 1.5, (y2 - y1) * 1.5
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
                                tag = f" ({scheme})" + (" (inverted)" if inverted else "") + " (x1.5)"
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
                                    display_until_pos = detect_pos + args.every
                                else:
                                    # 把 OCR 也丢到后台跑，不阻塞主循环
                                    ocr_box = (x1, y1, x2, y2)
                                    ocr_out_path = out_path
                                    detect_kind = "ocr"
                                    detect_task = asyncio.create_task(
                                        _run_ocr_once(session, roi.copy())
                                    )

                elif kind == "ocr" and result is not None:
                    plate_text = result.plate_text.strip()
                    print(f"  ocr      = {plate_text!r}")
                    txt_path = ocr_out_path.with_suffix(".txt")
                    txt_path.write_text(plate_text, encoding="utf-8")
                    print(f"  txt      = {txt_path.name}")
                    x1, y1, x2, y2 = ocr_box
                    display_frame = annotate_frame(
                        detect_frame, x1, y1, x2, y2, plate_text,
                    )
                    display_until_pos = detect_pos + args.every
                    ocr_box = None
                    ocr_out_path = None

            # ---- 4. 调度新一轮检测（仅当后台空闲且到了 --every 间隔） ----
            if detect_task is None and cap.frame_pos % args.every == 0:
                detect_kind = "detect"
                detect_pos = cap.frame_pos
                detect_frame = frame.copy()
                detect_task = asyncio.create_task(
                    _run_detect_once(session, prompt, detect_frame, schema)
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

            # ---- 6. 让出事件循环，让后台 VLM 任务有机会推进 ----
            await asyncio.sleep(0)

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