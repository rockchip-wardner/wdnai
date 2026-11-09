"""Qwen3-ASR —— 与 C++ demo 自动比对。

跑一遍 C demo（``rknn_qwen3_asr_demo_online``）和本 SDK 的流式 ASR，
把两份输出逐轮对齐比较：

* **最终结果**必须**逐字节相同** —— 不相同就 FAIL；
* **每轮的已确认文本（committed）**必须相同；
* **每轮的未确认尾巴（unfix）允许不同** —— 它按设计就是「还没确认、
  下一轮可能被改写」的部分；C demo 用 miniaudio 做 48 kHz -> 16 kHz 重采样，
  SDK 用线性插值，两者在 mel 频谱的末位比特上有差异，偶尔会让某一个
  token 选到不同的分支。回退策略会把这种差异吸收掉 —— **已确认文本从
  不曾分歧**（实测 16 轮全部一致）。

⚠️ **两张卡要错开跑**（见下方「设备选择」一节）：
C demo 的 ``rknn3_init(&ctx, NULL)`` 走驱动枚举的**第一张卡**
（板上是 ``0001:11:00.0``），且用 ``0xff`` 占满该卡全部 8 个 182x 核。
在**同一张卡上背靠背**跑 C demo 和 SDK（C demo 退出 -> 立刻建新 context）
会触发 RKNN3 PCIe transfer proxy 的驱动级 hang（proxy 卡在 ``ioctl()``，
需要 ``kill -9`` 手动救，且不会自动重启）。

所以本脚本默认让 **C demo 用第一张卡、SDK 用第二张卡**。

用法::

    .venv/bin/python .venv/src/wdnai/examples/Qwen3_ASR/compare_with_c_demo.py

    # 指定 C demo 目录 / 模型目录 / 音频
    .venv/bin/python .venv/src/wdnai/examples/Qwen3_ASR/compare_with_c_demo.py \\
        --c-demo-dir /path/to/rknn_Qwen3_ASR_demo \\
        --model-dir /userdata/models/Qwen3-ASR \\
        --wav /path/to/asr_en.wav

    # 显式指定 SDK 用哪张卡（不要和 C demo 同卡！）
    .venv/bin/python .venv/src/wdnai/examples/Qwen3_ASR/compare_with_c_demo.py --sdk-npu 1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_REPO_SRC = _HERE.parents[2] / "src"
if _REPO_SRC.is_dir() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

DEFAULT_C_DEMO_DIR = Path(
    "/userdata/rknn3/rknn3-model-zoo/install/rk3588_linux_aarch64/"
    "rknn_Qwen3_ASR_demo"
)

#: C demo 的每轮块长（秒）。C demo 固定 1 秒一块（ONLINE_CHUNK_MS）。
C_CHUNK_SECONDS = 1.0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--c-demo-dir", type=Path, default=DEFAULT_C_DEMO_DIR,
                   help="C demo 的安装目录（含 rknn_qwen3_asr_demo_online 与 model/）")
    p.add_argument("--model-dir", type=Path, default=None,
                   help="SDK 侧的模型目录；默认 $WDNAI_QWEN3_ASR_DIR 或 "
                        "$WDNAI_MODELS_DIR/Qwen3-ASR")
    p.add_argument("--wav", type=Path, default=_HERE / "asr_en.wav",
                   help="输入音频（两边必须用同一个文件）")
    p.add_argument("--sdk-npu", type=int, default=None,
                   help="SDK 侧用哪张本机 EP 协卡。默认**自动挑一张和 C demo "
                        "不同的卡**（C demo 固定用驱动枚举的第一张）—— 同卡背靠背"
                        "会触发 RKNN3 驱动 hang，别手动指成同一张")
    p.add_argument("--c-baseline", type=Path, default=None,
                   help="直接拿**已保存的 C demo 输出**当参考（不再跑 C demo）。"
                        "C demo 是确定性的（同一二进制/模型/wav 重复跑逐字节相同），"
                        "所以固化一份基线比每次重跑更快；板子某张卡挂掉时也能用")
    p.add_argument("--keep-output", type=Path, default=None,
                   help="把两边的原始输出写到这个目录")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# C demo
# --------------------------------------------------------------------------- #
_ROUND_RE = re.compile(
    r"=+ Online Round (\d+) =+\ncommit_add_text:\n(.*?)\nunfix_text:\n(.*?)\nperf:",
    re.S,
)

#: RKNN3 在「没指定 device_id」时打印的告警行。C demo 的
#: ``rknn3_init(&ctx, NULL)`` 正好命中这条 —— 它用的就是**第一张卡**。
_C_DEVICE_RE = re.compile(
    r"Using device\[(\d+)\]: id=([0-9A-Fa-f:.]+), type=(\w+)", re.S)


def parse_c_output(raw: str) -> Tuple[List[Dict[str, Any]], str]:
    """从 C demo 的 stdout 里解析 ``(rounds, final_text)``。"""
    rounds = [
        {"round": int(m.group(1)), "add": m.group(2), "unfix": m.group(3)}
        for m in _ROUND_RE.finditer(raw)
    ]
    final = ""
    m = re.search(r"=+ Final Commit Result =+\n(.*?)(?:\n\n|\Z)", raw, re.S)
    if m:
        final = m.group(1).strip()
    return rounds, final


def run_c_demo(demo_dir: Path, wav: Path) -> Tuple[List[Dict[str, Any]], str, str]:
    """Run the C demo; return ``(rounds, final_text, raw_output)``.

    ``rounds`` is a list of ``{"round", "add", "unfix"}``.
    """
    binary = demo_dir / "rknn_qwen3_asr_demo_online"
    if not binary.is_file():
        raise FileNotFoundError("C demo 不存在: {}".format(binary))
    model = demo_dir / "model"
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(demo_dir / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")

    cmd = [
        str(binary),
        str(model / "encoder_online.rknn"),
        str(model / "encoder_online.weight"),
        str(model / "llm.rknn"),
        str(model / "llm.weight"),
        str(model / "llm.tokenizer.gguf"),
        str(model / "llm.embed.bin"),
        "0xff",
        "0xff",
        str(wav),
    ]
    proc = subprocess.run(cmd, cwd=str(demo_dir), env=env,
                          capture_output=True, text=True, timeout=900)
    # ⚠️ 只从 **stdout** 解析结果。C demo 自己 printf 的是 stdout，
    # 而 llama.cpp 的模型元数据 dump 走 **stderr** —— 把两者拼起来
    # ("stdout" + "stderr") 会把那段 dump 追加到「Final Commit Result」后面，
    # 让最终结果比对假失败。
    raw = proc.stdout
    err = proc.stderr

    if "MODEL_SETUP fail" in (raw + err) and "Online Round" not in raw:
        raise RuntimeError(
            "C demo 的 model_init 失败（MODEL_SETUP fail）—— NPU 被占了。\n"
            "C demo 固定用驱动枚举的第一张卡（device[0]）且占满 8 个 182x 核，\n"
            "所以比对前必须先把 webui 里启用的 ASR worker 停掉：\n"
            "    curl -X DELETE http://127.0.0.1:8000/api/compute/instances "
            "-H 'Content-Type: application/json' -d '{\"model\":\"qwen3-asr\"}'\n"
            "（跑完再启用回来。）"
        )
    rounds, final = parse_c_output(raw)
    # stderr 只留作诊断（不进比对）。
    raw = raw + "\n" + err
    return rounds, final, raw


# --------------------------------------------------------------------------- #
# 设备选择 —— 两边必须用不同的卡
# --------------------------------------------------------------------------- #
def c_demo_device(raw: str) -> Tuple[Optional[int], Optional[str]]:
    """从 C demo 输出里解析它实际用的设备。

    RKNN3 在没指定 device_id 时会打印::

        W RKNN3: Multiple RKNN devices found (6) but no device_id specified.
        Using device[0]: id=0001:11:00.0, type=PCIE by default.

    返回 ``(device_index, device_id)``；解析不到时返回 ``(None, None)``。
    """
    m = _C_DEVICE_RE.search(raw)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2)


def device_index_of(device_id: str) -> Optional[int]:
    """device_id 在 ``RKNN3Lite.get_devices_id()`` 里的下标。"""
    try:
        from wdnai._librknn import RKNN3Lite
        ids = RKNN3Lite().get_devices_id()
    except Exception:  # noqa: BLE001
        return None
    for i, raw_id in enumerate(ids):
        s = raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id)
        if s == device_id:
            return i
    return None


def pick_sdk_npu(c_device_id: Optional[str], explicit: Optional[int]) -> Optional[int]:
    """决定 SDK 用哪张卡。

    - 显式指定 → 原样返回（但会和 C demo 同卡时给出警告）；
    - 没指定 → 自动挑一张**和 C demo 不同的**卡。
    """
    if explicit is not None:
        return explicit
    if c_device_id is None:
        return None
    c_idx = device_index_of(c_device_id)
    if c_idx is None:
        return None
    try:
        from wdnai._librknn import RKNN3Lite
        ids = RKNN3Lite().get_devices_id()
    except Exception:  # noqa: BLE001
        return None
    for i in range(len(ids)):
        if i != c_idx:
            return i
    return None


# --------------------------------------------------------------------------- #
# SDK 侧
# --------------------------------------------------------------------------- #
def run_sdk(wav: Path, model_dir: Optional[Path], npu: Optional[int]):
    """Run the SDK's streaming ASR; return ``(rounds, final_text)``."""
    from wdnai import AudioAttachment, LLMLocalSession, SystemLanguageModel

    sys.path.insert(0, str(_HERE))
    from streaming_asr import read_wav  # noqa: E402

    pcm, sample_rate, n_channels, sample_width = read_wav(wav)

    model = SystemLanguageModel.qwen3_asr()
    if model_dir:
        model = model.with_paths(asr_model_dir=str(model_dir))
    model = model.with_options(use_online_encoder=True,
                               online_chunk_seconds=C_CHUNK_SECONDS)

    rounds: List[Dict[str, Any]] = []
    final = ""
    prev = ""

    def on_delta(raw: str) -> None:
        nonlocal prev, final
        try:
            payload = json.loads(raw)
        except Exception:  # noqa: BLE001
            return
        if payload.get("type") == "asr_final":
            final = payload.get("text") or ""
            return
        if payload.get("type") != "asr_round":
            return
        committed = payload.get("committed") or ""
        rounds.append({
            "round": int(payload.get("round") or 0),
            "add": (committed[len(prev):] if committed.startswith(prev) else committed),
            "unfix": payload.get("unfixed") or "",
        })
        prev = committed

    kwargs = {"npu": npu} if npu is not None else {}
    session = LLMLocalSession(model=model, **kwargs)
    try:
        import asyncio
        result = asyncio.run(session.respond(
            attachment=[AudioAttachment(pcm, mime_type="audio/pcm")],
            sample_rate=sample_rate,
            n_channels=n_channels,
            sample_width=sample_width,
            on_delta=on_delta,
        ))
    finally:
        session.close()
    if not final:
        final = result if isinstance(result, str) else str(result)
    return rounds, final


# --------------------------------------------------------------------------- #
# 比对
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    if args.c_baseline:
        print("=== 1. 用已保存的 C demo 基线 ({}) ===".format(args.c_baseline))
        c_raw = args.c_baseline.read_text(errors="replace")
        c_rounds, c_final = parse_c_output(c_raw)
        print("    rounds={} final_len={}".format(len(c_rounds), len(c_final)))
        c_dev_idx, c_dev_id = c_demo_device(c_raw)
        print("    基线记录的设备: device[{}] = {}".format(c_dev_idx, c_dev_id))
    else:
        print("=== 1. 跑 C demo ({}) ===".format(args.c_demo_dir))
        c_rounds, c_final, c_raw = run_c_demo(args.c_demo_dir, args.wav)
        print("    rounds={} final_len={}".format(len(c_rounds), len(c_final)))
        c_dev_idx, c_dev_id = c_demo_device(c_raw)
        print("    C demo 用的设备: device[{}] = {}".format(c_dev_idx, c_dev_id))

    sdk_npu = pick_sdk_npu(c_dev_id, args.sdk_npu)
    print("=== 2. 跑 SDK（npu={}）===".format(sdk_npu))
    if sdk_npu is not None and c_dev_idx is not None and sdk_npu == c_dev_idx:
        print("    [WARN] SDK 和 C demo 用的是同一张卡！同卡背靠背会触发 "
              "RKNN3 驱动 hang（proxy 卡在 ioctl）。")
    py_rounds, py_final = run_sdk(args.wav, args.model_dir, sdk_npu)
    print("    rounds={} final_len={}".format(len(py_rounds), len(py_final)))

    if args.keep_output:
        args.keep_output.mkdir(parents=True, exist_ok=True)
        (args.keep_output / "c_demo_raw.txt").write_text(c_raw)
        (args.keep_output / "sdk_final.txt").write_text(py_final)

    ok = True

    print("\n=== 3. 最终结果比对 ===")
    c_body = c_final.strip()
    py_body = py_final.strip()
    if "\n" in py_body:
        head, py_body = py_body.split("\n", 1)
        if head.lower().startswith("language "):
            py_body = py_body.strip()
    if c_body == py_body:
        print("    [OK] 最终结果逐字节相同")
        print("         {}".format(c_body))
    else:
        ok = False
        print("    [FAIL] 最终结果不同")
        print("         C  : {!r}".format(c_body))
        print("         SDK: {!r}".format(py_body))

    print("\n=== 4. 逐轮比对 ===")
    n = min(len(c_rounds), len(py_rounds))
    if len(c_rounds) != len(py_rounds):
        ok = False
        print("    [FAIL] 轮数不同: C={} SDK={}".format(len(c_rounds), len(py_rounds)))
    add_mismatch = []
    unfix_mismatch = []
    for i in range(n):
        c, py = c_rounds[i], py_rounds[i]
        if c["add"] != py["add"]:
            add_mismatch.append((i + 1, c["add"], py["add"]))
        if c["unfix"] != py["unfix"]:
            unfix_mismatch.append((i + 1, c["unfix"], py["unfix"]))

    if add_mismatch:
        ok = False
        print("    [FAIL] {} 轮的**已确认文本**不同:".format(len(add_mismatch)))
        for r, a, b in add_mismatch:
            print("      round {}: C={!r} SDK={!r}".format(r, a, b))
    else:
        print("    [OK] {} 轮的已确认文本全部相同".format(n))

    if unfix_mismatch:
        print("    [INFO] {} 轮的**未确认尾巴**不同（允许 —— 下一轮会被回退修正）:"
              .format(len(unfix_mismatch)))
        for r, a, b in unfix_mismatch:
            print("      round {}: C={!r}".format(r, a))
            print("               SDK={!r}".format(b))
    else:
        print("    [OK] {} 轮的未确认尾巴全部相同".format(n))

    print("\n=== 结论: {} ===".format("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
