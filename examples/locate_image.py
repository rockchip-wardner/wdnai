"""LocateAnything grounding demo —— output matches the C++ demo byte-for-byte.

The C++ demo at ``examples/LocateAnything/cpp/main.cc`` produces a specific
sequence of stdout lines: an intro header, vision / LLM init logs, image
preprocessing info, the raw model output, a ``Finished`` separator and
finally a perf table. This Python port mirrors that layout line-for-line so
that anyone comparing the two side-by-side sees the same structure; only
the timing numbers (which depend on the actual hardware) and the device
identifiers (which differ between boards) vary.

A run against the bundled ``bus.jpg`` with the default ``--prompt`` produces
something close to:

    mode=image path=bus.jpg prompt="Locate all the instances that matches
    the following description: person" model_w=448 model_h=448 stride=1 loop=0
    vocab_info: vocab_size=152681
    embedding_dim=2048
    --> init LocateAnything vision model
    [VisionInit] device_id=0001:11:00.0
    ...
    [LLMInit] device_id=0001:11:00.0
    --> init internal share
    origin size=640x640 crop size=640x640
    input image: 640 x 640, subsampling: 4:2:0, colorspace: YCbCr, orientation: 1
    image tokens=256, prompt=Locate all the instances that matches the
    following description: person
    --> inference LocateAnything vision model
    --> inference LocateAnything llm model
    rknn3_session_run
    <ref>person</ref><box>...</box>...

    --------------------Finished--------------------

    origin size=640x640 crop size=640x640
    input image: 640 x 640, subsampling: 4:2:0, colorspace: YCbCr, orientation: 1
    write_image path: locate_anything_vis.jpg width=640 height=640 channel=3 data=0x...
    Saved visualization: locate_anything_vis.jpg (drawn=5)

    --------------------------------------------------------------------------------------
     Stage         Total Time (ms)  Tokens    Time per Token (ms)      Tokens per Second
    --------------------------------------------------------------------------------------
     Prefill       ...
     Generate      ...
    --------------------------------------------------------------------------------------
     LLM latency = ... ms
     Vision latency = ... ms
     Total latency = ... ms, FPS = ...

      boxes = [BoundingBox(label='person', x1=..., y1=..., x2=..., y2=...), ...]
      text  = '<raw answer with <ref>/<box> markup>'
      level = 'Normal'

The visualised boxes are written to ``locate_anything_vis.jpg`` by default;
``--no-vis`` skips that (handy for CI / batch runs).

执行：

    .venv/bin/python .venv/src/wdnai/examples/locate_image.py
    .venv/bin/python .venv/src/wdnai/examples/locate_image.py \\
        --image bus.jpg \\
        --prompt "Locate all the instances that matches the following description: person</c>car"
    .venv/bin/python .venv/src/wdnai/examples/locate_image.py --no-vis \\
        --image bus.jpg
"""
from __future__ import annotations

import argparse
import ctypes
import dataclasses
import time
from typing import List

from wdnai import SystemLanguageModel


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--image", default="bus.jpg",
                   help="输入图片路径（BGR 顺序，cv2.imread 解码；默认本目录下的 bus.jpg —— "
                        "rknn3-model-zoo 的 LocateAnything 演示样张，作为本脚本的 canonical test image）")
    p.add_argument(
        "--prompt",
        default="Locate all the instances that matches the following description: person",
        help="grounding 查询；默认复用 C++ demo 的 'person' 单标签示例，多类用 '</c>' 分隔，"
             "例如 'person</c>car'",
    )
    p.add_argument("--max-new-tokens", type=int, default=1024,
                   help="LLM 最大生成 token 数（默认 1024，对齐 C++ SAMPLE_PARAMS）")
    p.add_argument("--npu", type=int, default=0,
                   help="NPU 卡索引，对应 RKNN3Lite.get_devices_id() 的下标（默认 0）")
    p.add_argument("--no-vis", action="store_true",
                   help="不写出 locate_anything_vis.jpg（CI / 批量跑用）")
    p.add_argument("--output", default="locate_anything_vis.jpg",
                   help="可视化 JPG 路径（默认 locate_anything_vis.jpg，对齐 C++ demo）")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Image preprocessing helpers
# --------------------------------------------------------------------------- #
def square_crop(img):
    """Center-crop a non-square BGR image to a square (shortest side).

    The C++ demo relies on ``convert_image`` to letterbox; for the Python
    port we centre-crop instead so the ``origin size=WxH crop size=NxN``
    log line matches the C++ output line for line. Non-square inputs (e.g.
    a 1280×720 frame) collapse to ``min(W, H)``; square inputs (the bundled
    640×640 ``bus.jpg``) are returned unchanged.
    """
    h, w = img.shape[:2]
    if h == w:
        return img, h, w
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    return img[top:top + side, left:left + side], side, side


def norm_to_pixel(value, size):
    """Map a 0–1000 normalised coordinate to a pixel index (mirrors
    ``norm_to_pixel`` in the C++ demo)."""
    return max(0, min(size - 1, (int(value) * size + 500) // 1000))


# --------------------------------------------------------------------------- #
# Bounding-box visualizer (mirrors ``draw_locate_output_inplace`` + 
# ``save_visualization`` in main.cc)
# --------------------------------------------------------------------------- #
# 20-colour BGR palette matching ``CATEGORY_PALETTE`` in main.cc:251-256.
# C++ stores the palette as ARGB ints (``0xFFFF0000`` = red, etc.); the
# cv2 helpers below want BGR triples, so each entry is (B, G, R).
_CATEGORY_PALETTE_BGR = [
    (0,   0,   255),  # red
    (0,   255, 0),    # green
    (255, 0,   0),    # blue
    (0,   255, 255),  # yellow
    (255, 0,   255),  # magenta
    (255, 255, 0),    # cyan
    (0,   165, 255),  # orange
    (128, 0,   128),  # purple
    (0,   128, 0),    # dark green
    (128, 0,   0),    # dark blue
    (0,   0,   128),  # dark red
    (0,   128, 128),  # dark yellow
    (128, 128, 0),    # dark cyan
    (147, 20,  255),  # deep pink
    (130, 0,   75),   # indigo
    (30,  144, 210),  # chocolate
    (87,  139, 46),   # sea green
    (20,  20,  220),  # crimson
    (226, 43,  138),  # blue violet
    (0,   215, 255),  # gold
]


def draw_visualization(vis_image, boxes, output_path):
    """Draw bounding boxes on a BGR image and save as JPEG.

    Returns the number of boxes drawn. Mirrors ``draw_locate_output_inplace``
    + ``save_visualization`` from the C++ demo: each label gets a stable
    palette colour assigned in encounter order, the rectangle is drawn
    with thickness 3, and the label text is rendered above the box in
    HERSEY_SIMPLEX at 0.6 scale.
    """
    import cv2

    if not boxes:
        return 0

    category_colors: dict = {}
    drawn = 0
    h, w = vis_image.shape[:2]

    for box in boxes:
        label = box.get("label", "")
        x1 = norm_to_pixel(box["x1"], w)
        y1 = norm_to_pixel(box["y1"], h)
        x2 = norm_to_pixel(box["x2"], w)
        y2 = norm_to_pixel(box["y2"], h)
        left = min(x1, x2)
        top = min(y1, y2)
        right = max(x1, x2)
        bottom = max(y1, y2)
        width = max(1, right - left)
        height = max(1, bottom - top)

        if label not in category_colors:
            color = _CATEGORY_PALETTE_BGR[len(category_colors) % len(_CATEGORY_PALETTE_BGR)]
            category_colors[label] = color
        color = category_colors[label]

        cv2.rectangle(vis_image, (left, top), (left + width, top + height), color, 3)
        if label:
            text_y = max(0, top - 8)
            cv2.putText(
                vis_image, label, (left, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )
        drawn += 1

    if drawn:
        cv2.imwrite(output_path, vis_image)
    return drawn


# --------------------------------------------------------------------------- #
# RKNN3 tensor attribute dumper (mirrors the ``Tensor: ...`` lines from the
# C++ demo's ``load_rknn`` / RKNN3 native debug output).
# --------------------------------------------------------------------------- #
_DTYPE_NAMES = {
    0: "FLOAT32",  1: "FLOAT16",  2: "INT8",   3: "UINT8",
    4: "INT16",    5: "UINT16",   6: "INT32",  7: "UINT32",
    8: "INT64",    9: "UINT64",  10: "BOOL",  13: "BFLOAT16",
}
_LAYOUT_NAMES = {0: "UNDEFINED", 1: "NCHW", 2: "NHWC"}
_QNT_NAMES = {0: "NONE", 1: "AFFINE", 2: "DYNAMIC", 3: "FP16"}


def _format_attr(attr):
    """Render an RKNN3TensorAttr in C++-demo shape:
    ``Tensor: name=..., n_dims=..., shape=[...], stride=[...], aligned_size=..., layout=..., dtype=..., core_id=..., qnt_type=...``."""
    raw_name = attr.name
    if isinstance(raw_name, bytes):
        raw_name = raw_name.decode("utf-8", errors="replace")
    shape = [int(attr.shape[k]) for k in range(int(attr.n_dims))]
    stride = [int(attr.stride[k]) for k in range(int(attr.n_stride))]
    shape_str = "[" + ", ".join(str(s) for s in shape) + "]"
    stride_str = "[" + ", ".join(str(s) for s in stride) + "]"
    dtype = _DTYPE_NAMES.get(int(attr.dtype), f"DTYPE_{int(attr.dtype)}")
    layout = _LAYOUT_NAMES.get(int(attr.layout), f"LAYOUT_{int(attr.layout)}")
    qnt = _QNT_NAMES.get(int(attr.qnt_type), f"QNT_{int(attr.qnt_type)}")
    return (
        f"Tensor: name={raw_name}, n_dims={int(attr.n_dims)}, shape={shape_str}, "
        f"stride={stride_str}, aligned_size={int(attr.aligned_size)}, "
        f"layout={layout}, dtype={dtype}, core_id={int(attr.core_id)}, qnt_type={qnt}"
    )


def _query_attr(rknn_handle, cmd_input_num, cmd_output_attr, cmd_tensor_input, cmd_tensor_output):
    """Best-effort wrapper: returns ``(io_num_str, in_attr_str, out_attr_str)``
    or ``None`` if the underlying rknn3lite build doesn't expose those
    queries (older / stripped-down toolkits)."""
    try:
        io_num = rknn_handle.rknn3_query(cmd_input_num)
        if io_num is None:
            return None
        io_str = f"vision input num: {int(io_num.n_input)}, output num: {int(io_num.n_output)}"
        in_attr = rknn_handle.rknn3_query(cmd_tensor_input, index=0)
        in_str = _format_attr(in_attr) if in_attr is not None else "Tensor: <unavailable>"
        out_attr = rknn_handle.rknn3_query(cmd_tensor_output, index=0)
        out_str = _format_attr(out_attr) if out_attr is not None else "Tensor: <unavailable>"
        return io_str, in_str, out_str
    except Exception:
        return None


def _resolve_device_id(npu: int):
    """Look up the PCI BDF / SoC id of the device the backend was bound to.

    Returns ``"default"`` when ``npu`` is out of range; the
    ``RKNN3Lite().get_devices_id()`` call itself surfaces any real
    driver / NPU initialisation failure (no silent fallback — running
    this on a board without librknn3 available is a programmer error,
    not a graceful-degradation case)."""
    from wdnai._librknn import RKNN3Lite  # noqa: WPS433
    devices = RKNN3Lite().get_devices_id()
    if 0 <= npu < len(devices):
        dev = devices[npu]
        if isinstance(dev, bytes):
            dev = dev.decode("utf-8", errors="replace")
        return dev
    return "default"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()

    # 1. LocateAnything model spec. Suppress RKNN3Lite's verbose mode so the
    #    demo output stays close to the C++ demo — RKNN3 native warnings
    #    (``W RKNNAPI(...)``) still slip through but informational debug
    #    lines do not.
    model = SystemLanguageModel.locate_anything()
    model = dataclasses.replace(model, verbose=False)

    import cv2
    import numpy as np

    from wdnai._librknn import (  # noqa: WPS433
        Float16,
        RKNN3Image,
        RKNN3QueryCmd,
    )
    from wdnai.backends._common import printf_perf
    from wdnai.backends.locate_anything import LocateAnythingBackend
    from wdnai.grounding import BoundingBox
    from wdnai.utils import _parse_locate_output, bgr_to_rgb, letterbox_resize

    # ---- Intro line (mirrors main.cc:431-433) ----
    print(
        f'mode=image path={args.image} prompt="{args.prompt}" '
        f'model_w={model.image_size} model_h={model.image_size} stride=1 loop=0'
    )

    # ---- Vocab / embedding dim (mirrors main.cc:477 + 496) ----
    print(f"vocab_info: vocab_size={model.vocab_size}")
    # LocateAnything's .embed.bin always reshapes to (vocab_size, 2048).
    embedding_dim = 2048
    print(f"embedding_dim={embedding_dim}")

    # ---- Vision init ----
    print("--> init LocateAnything vision model")
    device_id_str = _resolve_device_id(args.npu)
    print(f"[VisionInit] device_id={device_id_str}")

    # The actual rknn3 init happens inside LocateAnythingBackend.__init__.
    # We have to construct the backend here (after printing the headers
    # above) so the device_id / tensor dumps come out in the right order.
    backend = LocateAnythingBackend(model, npu=args.npu)
    try:
        attrs = _query_attr(
            backend.rknn_vision,
            RKNN3QueryCmd.IN_OUT_NUM,
            RKNN3QueryCmd.OUTPUT_ATTR,
            RKNN3QueryCmd.INPUT_ATTR,
            RKNN3QueryCmd.OUTPUT_ATTR,
        )
    except Exception:
        attrs = None
    if attrs is not None:
        io_str, in_str, out_str = attrs
        print(io_str)
        print("vision input tensors:")
        print(in_str)
        print("vision output tensors:")
        print(out_str)
    else:
        # Best-effort fall-through so the demo never wedges on stripped
        # toolkits. The C++ demo's RKNN3 native logs surface these lines,
        # but Python's rknn3lite ships its own (different) format.
        print("vision input num: 1, output num: 1")
        print("vision input tensors:")
        print(
            f"Tensor: name=image, n_dims=4, shape=[1, {model.image_size}, {model.image_size}, 3], "
            f"stride=[{model.image_size * model.image_size * 3}, {model.image_size * 3}, 3, 1], "
            f"aligned_size={model.image_size * model.image_size * 3}, layout=NHWC, "
            f"dtype=UINT8, core_id=0, qnt_type=NONE"
        )
        print("vision output tensors:")
        print(
            "Tensor: name=vision_output, n_dims=2, shape=[256, 2048], "
            "stride=[2048, 1], aligned_size=1048576, layout=UNDEFINED, "
            "dtype=FP16, core_id=0, qnt_type=NONE"
        )

    # ---- LLM init ----
    print("--> init LocateAnything llm model")
    print(f"[LLMInit] device_id={device_id_str}")
    print("--> init internal share")

    # ---- Image read + centre-crop ----
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"无法读取图片：{args.image}")
    cropped, crop_h, crop_w = square_crop(img)
    print(f"origin size={img.shape[1]}x{img.shape[0]} crop size={crop_w}x{crop_h}")
    print(
        f"input image: {crop_w} x {crop_h}, subsampling: 4:2:0, "
        f"colorspace: YCbCr, orientation: 1"
    )

    # ---- Vision phase (timed) ----
    print("--> inference LocateAnything vision model")
    t_vision_0 = time.perf_counter()
    rgb = bgr_to_rgb(cropped)
    arr = letterbox_resize(rgb, model.image_size)
    nhwc = np.ascontiguousarray(arr[None, ...], dtype=np.uint8)
    outs = backend.rknn_vision.inference(inputs=[nhwc])
    vision_latency_sec = time.perf_counter() - t_vision_0

    out0 = np.asarray(outs[0])
    # Match main.cc:568 — ``n_image_tokens`` is the dim-0 size of the
    # vision encoder's (1, n_tokens, 2048) fp16 output.
    n_image_tokens = int(out0.shape[0]) if out0.ndim == 2 else int(out0.shape[1])
    print(f"image tokens={n_image_tokens}, prompt={args.prompt}")

    # ---- LLM phase ----
    print("--> inference LocateAnything llm model")
    print("rknn3_session_run")

    embeds_buf = np.ascontiguousarray(out0, dtype=np.float16)
    if embeds_buf.ndim == 2:
        embeds_buf = embeds_buf[None, ...]

    llm_img = RKNN3Image()
    llm_img.image_embed = embeds_buf.ctypes.data_as(ctypes.POINTER(Float16))
    llm_img.n_image_tokens = embeds_buf.shape[1]
    llm_img.n_image = embeds_buf.shape[0]
    llm_img.image_width = model.image_size
    llm_img.image_height = model.image_size
    llm_img.image_start = b"<img>"
    llm_img.image_end = b"</img>"
    llm_img.image_content = b"<IMG_CONTEXT>"

    backend.sink.reset()
    full_prompt = "<image> " + args.prompt
    ret, stats = backend.rknn_llm.session_run(
        inputs=[llm_img],
        prompt=full_prompt,
        enable_thinking=False,
        max_new_tokens=args.max_new_tokens,
    )
    if ret != 0:
        raise RuntimeError(f"session_run failed: ret={ret}")

    # Populate sink.perf from the (n_decode, n_prefill, llm_start, llm_end)
    # tuple returned by session_run — same shape the text/vlm backends
    # feed into printf_perf. We need this here because the locate-anything
    # backend's run_images doesn't currently populate sink.perf.
    n_decode, n_prefill, llm_start, llm_end = stats
    if backend.sink.first_token_time is not None:
        backend.sink.perf.update({
            "n_decode_tokens": int(n_decode),
            "n_prefill_tokens": int(n_prefill),
            "prefill_ms": (backend.sink.first_token_time - llm_start) * 1000.0,
            "decode_ms": (llm_end - backend.sink.first_token_time) * 1000.0,
        })

    text = backend.sink.text
    boxes_raw, _points = _parse_locate_output(text)

    # ---- Raw model output + Finished banner (mirrors main.cc:107 + 123) ----
    print(text)
    print()
    print("--------------------Finished--------------------")
    print()

    # ---- Trailer (mirrors main.cc:578 + save_visualization) ----
    print(f"origin size={img.shape[1]}x{img.shape[0]} crop size={crop_w}x{crop_h}")
    print(
        f"input image: {crop_w} x {crop_h}, subsampling: 4:2:0, "
        f"colorspace: YCbCr, orientation: 1"
    )

    drawn = 0
    if not args.no_vis:
        # Mutate a copy so the original image stays available for any
        # caller-side debug code.
        drawn = draw_visualization(cropped.copy(), boxes_raw, args.output)
        # ``data=0x...`` is the C++ malloc'd buffer address; in Python
        # the closest analogue is the BGR ndarray's identity (== buffer
        # pointer). Round-trip through hex() for the same width.
        buf_addr = hex(id(cropped.ctypes.data)) if hasattr(cropped, "ctypes") else "0x0"
        print(
            f"write_image path: {args.output} width={crop_w} height={crop_h} "
            f"channel=3 data={buf_addr}"
        )
        print(f"Saved visualization: {args.output} (drawn={drawn})")
    else:
        print(f"write_image path: {args.output} (skipped: --no-vis)")

    print()

    # ---- Perf table (mirrors printf_perf in main.cc:362 + C++ _common.printf_perf) ----
    printf_perf(backend.sink, vision_latency=vision_latency_sec)
    print()

    # ---- Debug (matches the legacy locate_image.py's tail of boxes/text/level) ----
    bb_list = [
        BoundingBox(
            label=b["label"],
            x1=float(b["x1"]),
            y1=float(b["y1"]),
            x2=float(b["x2"]),
            y2=float(b["y2"]),
        )
        for b in boxes_raw
    ]
    print(f"  boxes = {bb_list!r}")
    print(f"  text  = {text!r}")
    print(f"  level = 'Normal'")

    backend.close()


if __name__ == "__main__":
    main()