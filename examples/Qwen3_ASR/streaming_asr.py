"""Qwen3-ASR 流式语音识别 —— sample。

对齐的参考实现是 model-zoo 的 C++ demo::

    rknn3-model-zoo/examples/Qwen3_ASR/cpp/online_main.cc

C demo 有两种输出方式，本 sample 一一对应：

============================  ===========================================
C demo                        本 sample
============================  ===========================================
默认（每轮打印 commit/unfix）  ``--mode rounds``（默认）
``-s``（单行原地刷新）         ``--mode single``
============================  ===========================================

**两种模式吃的是同一份数据**（每轮的 committed / unfixed），只是渲染
方式不同 —— 和 C demo 一样。

用法::

    # 默认：逐轮明细
    .venv/bin/python .venv/src/wdnai/examples/Qwen3_ASR/streaming_asr.py

    # 单行实时刷新（对应 C demo 的 -s）
    .venv/bin/python .venv/src/wdnai/examples/Qwen3_ASR/streaming_asr.py --mode single

    # 指定音频 / 模型目录
    .venv/bin/python .venv/src/wdnai/examples/Qwen3_ASR/streaming_asr.py \\
        --wav /path/to/other.wav --model-dir /userdata/models/Qwen3-ASR

    # 和 C demo 自动比对（见 compare_with_c_demo.py）
    .venv/bin/python .venv/src/wdnai/examples/Qwen3_ASR/compare_with_c_demo.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# 让 sample 脱离 wdnai 的安装位置也能跑（仓库内直接执行）。
_HERE = Path(__file__).resolve().parent
_REPO_SRC = _HERE.parents[2] / "src"
if _REPO_SRC.is_dir() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from wdnai import (  # noqa: E402
    AudioAttachment,
    LLMLocalSession,
    SystemLanguageModel,
)

#: 素材目录（与 sample 同名）。
ASSET_DIR = _HERE


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--wav", type=Path, default=ASSET_DIR / "asr_en.wav",
                   help="输入 wav（默认：sample 自带的 asr_en.wav）")
    p.add_argument("--model-dir", type=Path, default=None,
                   help="覆盖模型目录；默认 $WDNAI_QWEN3_ASR_DIR 或 "
                        "$WDNAI_MODELS_DIR/Qwen3-ASR")
    p.add_argument("--npu", type=int, default=None,
                   help="用哪张本机 EP 协卡（默认第一张）。同一个模型占满一张卡的 "
                        "全部 182x 核，所以和 webui 里已启用的 worker 不能抢同一张")
    p.add_argument("--mode", choices=("rounds", "single"), default="rounds",
                   help="输出方式：rounds=逐轮明细（C 默认）；single=单行刷新（C 的 -s）")
    p.add_argument("--chunk-seconds", type=float, default=1.0,
                   help="流式分块长度（秒），默认 1.0 —— 与 C 的 ONLINE_CHUNK_MS 一致")
    p.add_argument("--max-new-tokens", type=int, default=50,
                   help="每块 LLM 最多生成多少 token（默认 50 —— C 的 "
                        "ONLINE_MAX_NEW_TOKENS）")
    p.add_argument("--rollback-tokens", type=int, default=5,
                   help="每轮回退末尾几个 token（默认 5 —— C 的 "
                        "ONLINE_ROLLBACK_TOKENS）")
    p.add_argument("--rollback-rounds", type=int, default=4,
                   help="前几轮不提交（默认 4 —— C 的 ONLINE_ROLLBACK_ROUNDS）")
    p.add_argument("--save-json", type=Path, default=None,
                   help="把逐轮结果写成 JSON（给比对工具用）")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# WAV 读取
# --------------------------------------------------------------------------- #
def read_wav(path: Path):
    """Return ``(pcm_bytes, sample_rate, n_channels, sample_width)``.

    支持 PCM s16/s24/s32（format tag 1）与 IEEE float32（tag 3）。
    标准库 ``wave`` 不认 float32，所以自己解析。
    """
    import io
    import struct

    raw = path.read_bytes()
    bio = io.BytesIO(raw)
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("不是 RIFF/WAVE 文件: {}".format(path))
    bio.read(12)
    fmt_tag = n_ch = sr = sw = 0
    data = b""
    while True:
        head = bio.read(8)
        if len(head) < 8:
            break
        cid, csize = head[:4], struct.unpack("<I", head[4:8])[0]
        payload = bio.read(csize)
        if cid == b"fmt " and len(payload) >= 16:
            fmt_tag, n_ch, sr, _, _, bits = struct.unpack("<HHIIHH", payload[:16])
            sw = bits // 8
        elif cid == b"data":
            data = payload
            break
        if csize % 2:
            bio.read(1)
    if fmt_tag == 3 and sw == 4:
        import numpy as np
        audio = np.frombuffer(data, dtype="<f4").copy()
        if n_ch > 1:
            audio = audio.reshape(-1, n_ch).mean(axis=1)
        np.clip(audio, -1.0, 1.0, out=audio)
        data = (audio * 32767.0).astype("<i2").tobytes()
        sw = 2
    elif fmt_tag != 1 or sw not in (2, 3, 4):
        raise ValueError(
            "不支持的 WAV 格式: tag={} width={}".format(fmt_tag, sw))
    return data, sr, n_ch, sw


# --------------------------------------------------------------------------- #
# 输出 —— 两种模式
# --------------------------------------------------------------------------- #
class _RoundsPrinter:
    """逐轮明细（对应 C demo 的默认输出）。"""

    def __init__(self) -> None:
        self.round_idx = 0

    def __call__(self, payload: Dict[str, Any]) -> None:
        self.round_idx += 1
        print("\n================ Online Round {} ================".format(self.round_idx))
        print("commit_add_text:")
        print(_delta(payload))
        print("unfix_text:")
        print(payload.get("unfixed") or "")
        sys.stdout.flush()


class _SingleLinePrinter:
    """单行原地刷新（对应 C demo 的 ``-s``）。

    逐字节对齐 C 的 ``print_single_line_result()``
    （``cpp/online_main.cc:201``）::

        printf("\\r%s", committed_text);
        if (!unfix_text.empty()) printf(BLUE "%s" RESET, unfix_text);
        printf("\\033[K");

    即：**已确认文本正常色 + 未确认尾巴蓝色**，未确认尾巴为空时**不输出
    颜色转义**（这一点与 C 一致 —— 早几轮还没有未确认内容）。
    """

    RED = "\033[31m"
    BLUE = "\033[34m"
    RESET = "\033[0m"
    CLEAR = "\033[K"

    #: 与 C 的 ``ANSI_COLOR_RED "single-line mode enabled: ..."`` 文案一致。
    HEADER = ("single-line mode enabled: only suitable for single-line ASR "
              "output, unfix text is shown in blue.")

    def __init__(self) -> None:
        self.printed_len = 0

    def print_header(self) -> None:
        print(self.RED + self.HEADER + "\n" + self.RESET, end="", flush=True)

    def __call__(self, payload: Dict[str, Any]) -> None:
        committed = payload.get("committed") or ""
        unfixed = payload.get("unfixed") or ""
        out = "\r" + committed
        if unfixed:
            out += self.BLUE + unfixed + self.RESET
        out += self.CLEAR
        print(out, end="", flush=True)
        self.printed_len = len(committed)


def _delta(payload: Dict[str, Any]) -> str:
    """本轮**新增**的已确认文本（C demo 的 commit_add_text 是增量）。"""
    # payload 里 committed 是累计值；增量由调用方用上一轮的累计值相减得到。
    return payload.get("_delta", "") or ""


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    import logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    pcm, sample_rate, n_channels, sample_width = read_wav(args.wav)
    print("wav: {} sr={} ch={} width={} bytes={} ({:.2f}s)".format(
        args.wav, sample_rate, n_channels, sample_width, len(pcm),
        len(pcm) / float(sample_width * n_channels * sample_rate),
    ))

    model = SystemLanguageModel.qwen3_asr()
    if args.model_dir:
        model = model.with_paths(asr_model_dir=str(args.model_dir))
    model = model.with_options(
        use_online_encoder=True,
        online_chunk_seconds=args.chunk_seconds,
        online_max_new_tokens=args.max_new_tokens,
        online_rollback_tokens=args.rollback_tokens,
        online_rollback_rounds=args.rollback_rounds,
    )

    printer = _SingleLinePrinter() if args.mode == "single" else _RoundsPrinter()
    rounds: List[Dict[str, Any]] = []
    prev_committed = ""

    def on_delta(raw: str) -> None:
        nonlocal prev_committed
        try:
            payload = json.loads(raw)
        except Exception:  # noqa: BLE001
            return
        kind = payload.get("type")
        if kind == "asr_final":
            # 收尾载荷（完整结果）—— 不算一轮，结果由下面的
            # "Final Commit Result" 块统一打印。
            return
        if kind != "asr_round":
            return
        committed = payload.get("committed") or ""
        payload["_delta"] = (committed[len(prev_committed):]
                             if committed.startswith(prev_committed) else committed)
        prev_committed = committed
        rounds.append(payload)
        printer(payload)

    if args.mode == "single":
        printer.print_header()
    else:
        print("rounds mode: 每轮打印 commit_add_text / unfix_text（对应 C 默认输出）")

    t_load = time.perf_counter()
    session = LLMLocalSession(model=model, **({"npu": args.npu} if args.npu is not None else {}))
    load_sec = time.perf_counter() - t_load
    print("model loaded in {:.2f}s (npu={})".format(load_sec, args.npu))
    t0 = time.perf_counter()
    try:
        import asyncio
        result = asyncio.run(session.respond(
            attachment=[AudioAttachment(pcm, mime_type="audio/pcm")],
            sample_rate=sample_rate,
            n_channels=n_channels,
            sample_width=sample_width,
            on_delta=on_delta,
        ))
        elapsed = time.perf_counter() - t0
    finally:
        session.close()

    if args.mode == "single":
        print("")

    # ---- 与 C demo 同样的收尾块 ----
    print("\n================ Final Commit Result ================")
    print(_final_body(result))
    print("\nrounds={} wall={:.2f}s".format(len(rounds), elapsed))

    if args.save_json:
        args.save_json.write_text(json.dumps({
            "rounds": [{k: v for k, v in r.items() if not k.startswith("_")}
                        for r in rounds],
            "final": result,
        }, ensure_ascii=False, indent=1))
        print("saved:", args.save_json)


def _final_body(result: Any) -> str:
    """``"language English\\n<body>"`` → ``body``（C demo 的 final 只打正文）。"""
    text = result if isinstance(result, str) else str(result)
    if "\n" in text:
        head, body = text.split("\n", 1)
        if head.lower().startswith("language "):
            return body
    return text


if __name__ == "__main__":
    main()
