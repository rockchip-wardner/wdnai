"""Qwen3-ASR **增量流式会话** —— sample。

``streaming_asr.py`` 演示的是「文件驱动的流式」：整段音频一次性交给
``respond()``，SDK 内部按 1 秒切块、逐轮解码。

本 sample 演示的是**真正的增量会话**：音频一块一块喂进去，每喂一块
就出一轮结果。麦克风实时 ASR（网页端 ``getUserMedia``、板载 ALSA）
走的就是这条路。

============================  ==========================================
概念                          说明
============================  ==========================================
``asr_stream_id``             会话 id。同一个 id 的多次调用共享
                              「音频 embedding 历史 + 已确认 token」
                              —— 这就是增量会话与一次性推理的区别。
``asr_stream_final``          本次调用带的是不是**最后一块**。最后
                              一块不砍尾（对应 C demo 的 ``is_last``）。
``asr_stream_origin_sec``     会话在调用方时间轴上的起点（秒）。
                              每块的时间戳由它推导；seek = 换新会话
                              + 新起点。
============================  ==========================================

**契约**（重要）：带 ``asr_stream_final=True`` 的那次调用**必须携带
本会话的最后一块**。最后一块不砍尾，所以它必须是知道自己「是最后一块」
的那一块；若最后一块已在之前的调用里跑过，那 5 个回退 token 就留在
未确认区（SDK 会打一条 warning）。

用法::

    # 增量会话 + 与一次性推理逐字节比对
    .venv/bin/python .venv/src/wdnai/examples/Qwen3_ASR/streaming_session.py

    # 指定音频 / 模型目录 / 协卡
    .venv/bin/python .venv/src/wdnai/examples/Qwen3_ASR/streaming_session.py \\
        --wav /path/to/other.wav --model-dir /userdata/models/Qwen3-ASR --npu 1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
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
from streaming_asr import read_wav  # noqa: E402  （同目录的 wav 解析器）


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
                   help="用哪张本机 EP 协卡（默认第一张）。同一个模型占满一张卡的"
                        "全部 182x 核，所以和 webui 里已启用的 worker 不能抢同一张")
    p.add_argument("--feed-seconds", type=float, default=1.0,
                   help="每次喂进去多少秒的**源**音频（默认 1.0）。"
                        "SDK 内部仍按 1 秒（16000 样本）切块")
    p.add_argument("--chunk-seconds", type=float, default=1.0,
                   help="SDK 的推理块长（秒），默认 1.0 —— C 的 ONLINE_CHUNK_MS")
    p.add_argument("--max-new-tokens", type=int, default=50,
                   help="每块 LLM 最多生成多少 token（默认 50 —— C 的 "
                        "ONLINE_MAX_NEW_TOKENS）")
    p.add_argument("--rollback-tokens", type=int, default=5,
                   help="每轮回退末尾几个 token（默认 5 —— C 的 "
                        "ONLINE_ROLLBACK_TOKENS）")
    p.add_argument("--rollback-rounds", type=int, default=4,
                   help="前几轮不挂前缀（默认 4 —— C 的 "
                        "ONLINE_ROLLBACK_ROUNDS）")
    p.add_argument("--skip-oneshot", action="store_true",
                   help="不跑一次性推理（省一次时间），只演示增量会话")
    p.add_argument("--save-json", type=Path, default=None,
                   help="把逐轮结果存成 JSON")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# 逐轮输出
# --------------------------------------------------------------------------- #
class _RoundPrinter:
    """打印每轮的已确认增量 / 未确认尾巴 / 时间戳。"""

    def __init__(self) -> None:
        self.prev_committed = ""
        self.rounds: List[Dict[str, Any]] = []

    def __call__(self, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except Exception:  # noqa: BLE001
            return
        kind = payload.get("type")
        if kind == "asr_final":
            return          # 收尾载荷不算一轮
        if kind != "asr_round":
            return

        committed = payload.get("committed") or ""
        unfixed = payload.get("unfixed") or ""
        delta = (committed[len(self.prev_committed):]
                 if committed.startswith(self.prev_committed) else committed)
        self.prev_committed = committed
        payload["_delta"] = delta
        self.rounds.append(payload)

        start = payload.get("chunk_start_sec")
        end = start + payload.get("chunk_samples", 0) / 16000.0 \
            if start is not None else None
        span = "  [{:.2f}s → {:.2f}s]".format(start, end) if end else ""
        print("\n--- round {:>3}  ({}){}".format(
            payload.get("round"), payload.get("language") or "?", span))
        print("  + {}".format(delta))
        if unfixed:
            print("  ~ {}".format(unfixed))
        sys.stdout.flush()


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
async def _run_incremental(session, pcm: bytes, sample_rate: int,
                           n_channels: int, sample_width: int,
                           feed_seconds: float, stream_id: str,
                           printer: _RoundPrinter) -> str:
    """把音频一块一块喂进去，返回最终文本。"""
    frame_bytes = max(1, n_channels * sample_width)
    total_frames = len(pcm) // frame_bytes
    feed_frames = max(1, int(round(feed_seconds * sample_rate)))

    n_calls = 0
    pos = 0
    result = ""
    while pos < total_frames:
        end = min(pos + feed_frames, total_frames)
        block = pcm[pos * frame_bytes:end * frame_bytes]
        pos = end
        n_calls += 1
        is_final = pos >= total_frames
        # The **final** call carries the session's last block and returns
        # the complete transcript; the intermediate calls return the
        # transcript confirmed so far.
        result = await session.respond(
            attachment=[AudioAttachment(block, mime_type="audio/pcm")],
            sample_rate=sample_rate,
            n_channels=n_channels,
            sample_width=sample_width,
            on_delta=printer,
            asr_stream_id=stream_id,
            asr_stream_origin_sec=0.0,
            asr_stream_final=is_final,
        )
    if n_calls == 0:
        raise RuntimeError("音频为空")
    return result if isinstance(result, str) else str(result)


async def _run_oneshot(session, pcm: bytes, sample_rate: int,
                       n_channels: int, sample_width: int) -> str:
    """一次性把整段音频交给 SDK（文件驱动的流式）。"""
    return await session.respond(
        attachment=[AudioAttachment(pcm, mime_type="audio/pcm")],
        sample_rate=sample_rate,
        n_channels=n_channels,
        sample_width=sample_width,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    pcm, sample_rate, n_channels, sample_width = read_wav(args.wav)
    duration = len(pcm) / float(sample_width * n_channels * sample_rate)
    print("wav: {} sr={} ch={} width={} bytes={} ({:.2f}s)".format(
        args.wav, sample_rate, n_channels, sample_width, len(pcm), duration,
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

    printer = _RoundPrinter()
    t_load = time.perf_counter()
    session = LLMLocalSession(
        model=model,
        **({"npu": args.npu} if args.npu is not None else {}),
    )
    print("model loaded in {:.2f}s (npu={})".format(
        time.perf_counter() - t_load, args.npu))

    try:
        # ---- 1) 增量会话 ----
        print("\n================ incremental session ================")
        t0 = time.perf_counter()
        inc_result = asyncio.run(_run_incremental(
            session, pcm, sample_rate, n_channels, sample_width,
            args.feed_seconds, "demo-session", printer,
        ))
        inc_elapsed = time.perf_counter() - t0
        inc_text = _final_body(inc_result)
        print("\n--- incremental final ---")
        print(inc_text)
        print("rounds={} wall={:.2f}s".format(len(printer.rounds), inc_elapsed))

        # ---- 2) 一次性推理（对照） ----
        one_text = ""
        if not args.skip_oneshot:
            print("\n================ one-shot (file-driven) ================")
            t0 = time.perf_counter()
            one_result = asyncio.run(_run_oneshot(
                session, pcm, sample_rate, n_channels, sample_width))
            print("one-shot wall={:.2f}s".format(time.perf_counter() - t0))
            one_text = _final_body(one_result)
            print("\n--- one-shot final ---")
            print(one_text)

            print("\n================ comparison ================")
            if inc_text == one_text:
                print("IDENTICAL — 增量会话与一次性推理逐字节相同 ✓")
            else:
                print("DIFFERENT ✗")
                print("  incremental: {!r}".format(inc_text))
                print("  one-shot   : {!r}".format(one_text))
    finally:
        session.close()

    if args.save_json:
        args.save_json.write_text(json.dumps({
            "rounds": [{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in printer.rounds],
            "incremental_final": inc_text,
            "one_shot_final": one_text,
        }, ensure_ascii=False, indent=1))
        print("saved:", args.save_json)


def _final_body(result: Any) -> str:
    """``"language English\\n<body>"`` → ``body``。"""
    text = result if isinstance(result, str) else str(result)
    if "\n" in text:
        head, body = text.split("\n", 1)
        if head.lower().startswith("language "):
            return body
    return text


if __name__ == "__main__":
    main()
