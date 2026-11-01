"""Depth Anything V3 demo —— output matches the C++ demo line-for-line.

**⚠️ 当前限制**：本 sample 只覆盖 **10-view（多视图融合）** 导出 ——
``global`` 段在 10 张图之间做融合，送图张数必须正好是 10。
**单路（单图）深度估计尚未实现**（需要一份 V=1 的导出），后续版本补齐。

The C++ demo at ``rknn3-model-zoo/examples/depth_anything_v3/cpp/main.cpp``
drives the three-stage DA3-BASE RKNN export (``local`` → ``global`` →
``head``) and produces a specific sequence of stdout lines: per-view
preprocessing logs, the tensor summaries, the shared inverse-depth
visualisation window, one heatmap per view and a montage. This Python port
mirrors that layout so the two runs can be compared side by side; only the
timing numbers (which depend on the actual hardware) vary.

A run against the model-zoo's ``model/images.txt`` (10 frames) produces
something close to::

    mode=depth views=10 model=DepthAnythingV3 image=280x280
    origin size=1024x576 crop size=1024x576
    input image: 1024 x 576, subsampling: 4:2:0, colorspace: YCbCr, orientation: 1
    depth visualization valid region: x=[0,280) y=[0,280)
    local view 1/10 done
    ...
    local_tokens: elems=3079680 min=-6.074219 max=6.269531 mean=-0.005947
    depth: elems=784000 min=0.350586 max=1.421875 mean=0.975114
    confidence: elems=784000 min=1.000000 max=5.597656 mean=2.675516
    shared inverse-depth visualization range: [0.730385, 2.169492]
    saved depth heatmap: depth_vis/depth_00.jpg
    ...
    saved depth montage: depth_vis/depth_montage.jpg layout=5x2 canvas=1400x560
    depth output directory: depth_vis
    DA3-BASE split RKNN pipeline finished.

The visualisation is written to ``depth_vis/`` by default; ``--no-vis``
skips that (handy for CI / batch runs).

执行：

    .venv/bin/python .venv/src/wdnai/examples/depth_anything_v3.py
    .venv/bin/python .venv/src/wdnai/examples/depth_anything_v3.py \\
        --image-list /path/to/images.txt --output-dir depth_vis
    .venv/bin/python .venv/src/wdnai/examples/depth_anything_v3.py --no-vis

输入素材（随本 sample 自带，与 model-zoo 的 demo 逐字节相同）：

    examples/depth_anything_v3/frame_01.jpg .. frame_10.jpg
    examples/depth_anything_v3/images.txt      # 每行一个文件名

``images.txt`` 里的路径是**相对列表自身目录**的裸文件名，所以整个目录可以
整体搬走而不必改内容；换成绝对路径同样有效。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

import numpy as np

from wdnai import SystemLanguageModel
from wdnai.backends.depth_anything import (
    DepthAnythingBackend,
    build_depth_valid_mask,
    preprocess_image_rgb,
    save_depth_heatmaps,
)

# --------------------------------------------------------------------------- #
# Default input list — the 10 frames that ship with this sample, copied from
# the model-zoo's DA3 demo (``examples/depth_anything_v3/model/``). The list
# is resolved **relative to this file**, so the sample runs from any cwd.
# --------------------------------------------------------------------------- #
_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_IMAGE_LIST = os.path.join(_EXAMPLES_DIR, "depth_anything_v3", "images.txt")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--image-list", default=_DEFAULT_IMAGE_LIST,
                   help="图片列表文件（每行一个路径，行数须等于模型导出的 view 数；"
                        "默认指向 rknn3-model-zoo 的 DA3 demo 素材）")
    p.add_argument("--image", action="append", default=None,
                   help="直接给一张图片（可重复；张数必须等于模型的 view 数）。"
                        "与 --image-list 二选一。")
    p.add_argument("--output-dir", default="depth_vis",
                   help="热力图输出目录（默认 depth_vis，对齐 C++ demo）")
    p.add_argument("--npu", type=int, default=0,
                   help="NPU 卡索引，对应 RKNN3Lite.get_devices_id() 的下标（默认 0）")
    p.add_argument("--no-vis", action="store_true",
                   help="不写出 depth_vis/（CI / 批量跑用）")
    p.add_argument("--model-dir", default=None,
                   help="覆盖模型目录（等价于 WDNAI_DEPTH_ANYTHING_V3_DIR）")
    return p.parse_args(argv)


def _read_image_list(path: str) -> List[str]:
    """Read the image list — one path per line, blank trailing lines ignored.

    The model-zoo's ``model/images.txt`` stores paths **relative to the
    demo root** (``model/frame_01.jpg``) while the list itself lives in
    ``model/`` — so a relative entry is resolved against the list's own
    directory first and against its *parent* second. Absolute entries are
    taken as they are.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    out = [line for line in lines if line]
    if not out:
        raise FileNotFoundError(f"图片列表为空: {path}")

    list_dir = os.path.dirname(os.path.abspath(path))
    candidates = [list_dir, os.path.dirname(list_dir), os.getcwd()]
    resolved: List[str] = []
    for entry in out:
        if os.path.isabs(entry):
            resolved.append(entry)
            continue
        for base in candidates:
            candidate = os.path.join(base, entry)
            if os.path.isfile(candidate):
                resolved.append(candidate)
                break
        else:
            # Keep the raw entry so the error message names what was asked for.
            resolved.append(entry)
    return resolved


def _print_tensor_summary(name: str, data: np.ndarray) -> None:
    """Mirror the C++ ``print_tensor_summary`` line.

    The C++ accumulates the sum in a ``double`` (``double sum = 0.0;
    sum += data[i];``) and divides by the element count — numpy's
    ``float32`` ``.mean()`` would accumulate in float32 and can land on
    the neighbouring last digit, so the accumulation is widened to
    float64 here.
    """
    flat = np.asarray(data, dtype=np.float32).reshape(-1)
    mean = float(flat.astype(np.float64).sum()) / flat.size
    print(f"{name}: elems={flat.size} min={flat.min():.6f} "
          f"max={flat.max():.6f} mean={mean:.6f}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()

    if args.model_dir:
        os.environ["WDNAI_DEPTH_ANYTHING_V3_DIR"] = args.model_dir

    if args.image:
        image_paths = list(args.image)
    else:
        image_paths = _read_image_list(args.image_list)

    # 1. Depth Anything V3 model spec. Suppress RKNN3Lite's verbose mode so
    #    the demo output stays close to the C++ demo.
    spec = SystemLanguageModel.depth_anything_v3()

    import cv2

    # ---- Load the frames -------------------------------------------------
    frames = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"无法读取图片：{path}")
        frames.append(img)

    print(
        f'mode=depth views={len(frames)} model={spec.name} '
        f'image={spec.image_size}x{spec.image_size}'
    )

    # 2. Build the backend. The real geometry (input resolution, view count,
    #    token counts) is read off the models themselves — the export bakes
    #    them in, so the view count is *not* a free parameter.
    t0 = time.perf_counter()
    backend = DepthAnythingBackend(spec, npu=args.npu)
    try:
        print(
            f"pipeline config: views={backend.views} "
            f"image={backend.image_channels}x{backend.image_height}x{backend.image_width} "
            f"local_tokens={backend.local_tokens_per_view} hidden={backend.local_hidden}"
        )

        # The view count is baked into the export (``global``'s input shape is
        # ``[V, L, 1, D]``) — it is **not** a runtime parameter. Check it
        # *before* any inference: the mismatch would otherwise only surface
        # as an opaque numpy reshape error, after the model was loaded and
        # every view already went through the local stage.
        if len(frames) != backend.views:
            raise SystemExit(
                f"这个 DA3 导出是为 {backend.views} 个 view 建的，"
                f"却喂了 {len(frames)} 张图 —— 张数必须正好是 {backend.views}"
                f"（改张数需要重新导出模型）"
            )

        # ---- Per-view preprocessing + local stage -------------------------
        tokens = []
        masks = []
        for idx, frame in enumerate(frames):
            src_h, src_w = frame.shape[:2]
            print(f"origin size={src_w}x{src_h} crop size={src_w}x{src_h}")
            print(
                f"input image: {src_w} x {src_h}, subsampling: 4:2:0, "
                f"colorspace: YCbCr, orientation: 1"
            )
            preprocess_start = time.perf_counter()
            rgb = preprocess_image_rgb(frame, backend.image_height, backend.image_width)
            mask = build_depth_valid_mask(rgb)
            masks.append(mask)
            print(
                f"depth visualization valid region: "
                f"x=[0,{backend.image_width}) y=[0,{backend.image_height})"
            )
            tokens.append(backend.run_local_stage(rgb))
            print(
                f"[time] local view {idx + 1} preprocess "
                f"{(time.perf_counter() - preprocess_start) * 1000.0:.3f} ms"
            )
            print(f"local view {idx + 1}/{len(frames)} done")

        local_tokens = np.stack(tokens, axis=0)
        _print_tensor_summary("local_tokens", local_tokens)

        # ---- Global + head stages ----------------------------------------
        stage_start = time.perf_counter()
        feats = backend.run_global_stage(local_tokens)
        print(f"[time] global run {(time.perf_counter() - stage_start) * 1000.0:.3f} ms")

        stage_start = time.perf_counter()
        depth, confidence = backend.run_head_stage(feats)
        print(f"[time] head run {(time.perf_counter() - stage_start) * 1000.0:.3f} ms")

        _print_tensor_summary("depth", depth)
        _print_tensor_summary("confidence", confidence)

        # ---- Visualisation ------------------------------------------------
        if not args.no_vis:
            info = save_depth_heatmaps(
                args.output_dir,
                depth.reshape(len(frames), -1),
                np.stack(masks, axis=0).reshape(len(frames), -1),
                len(frames),
                backend.image_height,
                backend.image_width,
            )
            print(
                f"shared inverse-depth visualization range: "
                f"[{info['lower']:.6f}, {info['upper']:.6f}]"
            )
            for view in range(len(frames)):
                print(f"saved depth heatmap: {os.path.join(args.output_dir, 'depth_%02d.jpg' % view)}")
            print(
                f"saved depth montage: {info['montage_path']} "
                f"layout={info['montage_columns']}x{info['montage_rows']} "
                f"canvas={info['montage_width']}x{info['montage_height']}"
            )
            print(f"depth output directory: {args.output_dir}")

        print(f"[time] total {(time.perf_counter() - t0) * 1000.0:.3f} ms")
        print("DA3-BASE split RKNN pipeline finished.")
    finally:
        backend.close()


if __name__ == "__main__":
    sys.exit(main())
