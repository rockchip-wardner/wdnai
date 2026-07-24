"""Qwen3-VL-4B 视频流 VLM 推理示例。

按一定帧间隔把视频帧送给 ``LLMSession`` 推理，并把分析结果打在终端。

模型文件位于固定的本地路径 ``/userdata/models/Qwen3-VL-4B/``，可通过
环境变量直接指定模型目录（调试时用）：

    RKAI_QWEN3_VL_4B_DIR=/path/to/Qwen3-VL-4B  # 直接指定 4B 模型目录，默认 /userdata/models/Qwen3-VL-4B
"""
import argparse
import asyncio
import sys
from pathlib import Path

import cv2
import rkv
from rkai import Attachment, Generable, Field, LLMSession, SystemLanguageModel
from typing import Literal


@Generable
class Event:
    """从视频帧中抽取的结构化事件"""
    act: Literal["吸污", "阀门开关", "关闭盖板", "巡检中"] = Field(
        default="巡检中", description="工作人员相关动作"
    )


PROMPT = (
    "这是铁路巡检的视频，请分析视频中的工作人员是否做了相应的动作，比如吸污、阀门开关、关闭盖板等"
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
        help="每隔多少帧跑一次 VLM（默认 30 ≈ 1s @ 30fps）",
    )
    p.add_argument(
        "--max-frames", type=int, default=0,
        help="最多处理多少帧（0 表示一直跑到 EOF）",
    )
    p.add_argument(
        "--no-display", action="store_true",
        help="无显示环境时使用：不调用 cv2.imshow",
    )
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    if rkv.VideoCapture is None:
        raise RuntimeError(
            "rkv.VideoCapture 不可用：rkv.video 模块加载失败（多半是 "
            "librockchip_mpp.so.0 缺符号或 librockchip-mpp / librga 没装）。"
            "在装了原生库的环境上再跑这个脚本。"
        )

    # 模型文件需要在 /userdata/models/Qwen3-VL-4B/ 下（可用 RKAI_QWEN3_VL_4B_DIR 覆盖）
    model = SystemLanguageModel.qwen3_vl_4b()
    session = LLMSession(model=model)

    cap = rkv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    print(
        f"[video] {video_path.name}  {cap.width}x{cap.height} @ {cap.fps:.2f}fps  "
        f"every {args.every} frame(s)"
    )

    processed = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[video] EOF")
                break

            if not args.no_display:
                cv2.imshow("Video Player", frame)
                # 25ms 延迟 + 检测 'q' 退出
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    print("[video] user quit")
                    break

            if cap.frame_pos % args.every == 0:
                print(f"\n========== frame {cap.frame_pos} ==========")
                # reset() 清空 history + C 库 KV cache，避免上一轮状态污染下一轮
                session.reset()
                result = await session.respond(
                    prompt=PROMPT,
                    attachment=Attachment(frame),
                    generating=Event,
                )
                print(f"  act  = {result.act!r}")
                processed += 1

            if args.max_frames and cap.frame_pos >= args.max_frames:
                print(f"[video] reached max-frames={args.max_frames}")
                break
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        session.close()

    print(f"\n[done] {processed} frame(s) processed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)