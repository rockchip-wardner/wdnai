#!/usr/bin/env bash
# ============================================================================
# qwenimage21.sh —— Qwen-Image-2.1 → RKNN(rk1820) 512×512 出图一键脚本
#
#   用法：
#     bash qwenimage21.sh "a cute cat sitting on a wooden chair"      # 默认 8 步
#     bash qwenimage21.sh "一只猫" 20                                  # 20 步（更准）
#     bash qwenimage21.sh "一只猫" 8 --out /tmp/myimg                  # 指定输出目录
#     bash qwenimage21.sh "一只猫" 8 --fresh                           # 每次用随机噪声
#
#   产物：<out>/img01.png            最终图（512×512 RGB）
#         <out>/img01_latents.bin    DiT 潜变量（token 主序 fp16）
#         <out>/img01_vae_image.bin  VAE 输出（NCHW [1,4,512,512] fp16）
#         <out>/te/                  TE 中间产物
#         <out>/test.log             完整日志
#
#   流程（板端实测，2026-11-01）：
#     ① TE（text encoder）单独跑一次 —— 必须占满卡池 → 跑完进程退出 → 池释放
#     ② DiT（4 个合并 8-block 模型，双卡常驻）+ VAE 4 段链
#        ⇒ 生成过程 **零 load / 零 model_init**
#
#   ⚠️ 三条硬性要求（改脚本前先看 debug.md §73）：
#     * QI_SPREAD_IO=0    —— 输出的 IO 绝不能分散（runtime 只往它指定的池写结果）
#     * QI_SPREAD_IO_IN=1 —— 只分散「输入」IO（否则 VAE 第 3 段 create_mem 失败）
#     * --vae-stream      —— VAE 链必须逐段释放（否则 4 段同时常驻 ⇒ MODEL_SETUP fail）
#
#   模型包（默认 /home/rockchip/models/qwenimage21，可用 QI_MODEL_DIR 覆盖）：
#     llm_qwenimage21-te.1828.rknn + .weight
#     qwenimage21-te.embed.bin + .tokenizer.gguf
#     dit_qwenimage21-512.seg{0..3}.1828.rknn + .weight
#     vae_qwenimage21-512.seg{0..3}.1828.rknn + .weight
#     reference_512/hidden_states.bin          ← 初始噪声
#     runtime/qwen_image_rknn + runtime/lib/aarch64/   ← C++ 运行时
#
#   ⏱ 耗时（RK3588 + 2×RK1828）：8 步 ≈ 94 s / 20 步 ≈ 125 s / 40 步 ≈ 177 s
#      步数是纯运行时参数，模型不用重建。TE 的权重占 486 MiB/核，与 DiT
#      物理无法共卡 ⇒ 每次换提示词都要重跑 TE（这是架构天花板，不是 bug）。
# ============================================================================
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)

# ── 参数 ────────────────────────────────────────────────────────────────────
PROMPT="${1:-}"
STEPS="${2:-8}"
OUT=""
NOISE_MODE=${QI_NOISE:-fresh}     # fresh = 每张换随机噪声；ref = 复用固定噪声（可与金标准对拍）
NOISE_FLAG=""
[ "$NOISE_MODE" = "ref" ] && NOISE_FLAG="--fixed"
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2;;
    --fresh) NOISE_MODE=fresh; shift;;
    --fixed) NOISE_MODE=ref; shift;;
    *) shift;;
  esac
done
[ -z "$PROMPT" ] && { echo "用法: bash qwenimage21.sh \"提示词\" [步数] [--out 目录]"; exit 1; }
[ -z "$OUT" ] && OUT=/tmp/qi21_$(date +%H%M%S)

# ── 运行时 / 模型包 ────────────────────────────────────────────────────────
MD=${QI_MODEL_DIR:-${WDNAI_QWEN_IMAGE_21_DIR:-/home/rockchip/models/qwenimage21}}
if [ ! -d "$MD" ]; then
  MD="$HERE/../models/qwenimage21"
fi
if [ ! -f "$MD/llm_qwenimage21-te.1828.rknn" ] || [ ! -f "$MD/dit_qwenimage21-512.seg0.1828.rknn" ]; then
  echo "❌ 模型包不完整：$MD"
  echo "   需要 llm_qwenimage21-te.1828.rknn + dit_qwenimage21-512.seg0.1828.rknn 等"
  exit 2
fi

# 运行时：优先用模型包自带的 runtime/，否则退回部署目录
if [ -x "$MD/runtime/qwen_image_rknn" ]; then
  BIN="$MD/runtime/qwen_image_rknn"
  LIB="$MD/runtime/lib/aarch64"
else
  DEP=${WDNAI_QWEN_IMAGE_ROOT:-/home/rockchip/projects/deploy}
  BIN="$DEP/bin/qwen_image_rknn"
  LIB="$DEP/lib/aarch64"
fi
[ -x "$BIN" ] || { echo "❌ 找不到运行时 $BIN"; exit 2; }

export LD_LIBRARY_PATH=$LIB:${LD_LIBRARY_PATH:-}
export QI_SPREAD_IO=${QI_SPREAD_IO:-0}
export QI_SPREAD_IO_IN=${QI_SPREAD_IO_IN:-1}
export RKNN3_NETWORK_SOCKET_FILE=/tmp/rk-mdns.ini

TE_MODEL="$MD/llm_qwenimage21-te.1828.rknn"
TE_WEIGHT="$MD/llm_qwenimage21-te.weight"
TE_TOKENIZER="$MD/qwenimage21-te.tokenizer.gguf"
TE_EMBED="$MD/qwenimage21-te.embed.bin"
REF_DIR="$MD/reference_512"

DIT_RKNN=(); DIT_WTMP=(); DIT_WSRC=()
for i in 0 1 2 3; do
  s=$(echo abcd | cut -c$((i+1)))
  DIT_RKNN+=("$MD/dit_qwenimage21-512.seg${i}.1828.rknn")
  DIT_WTMP+=("m512${s}.weight")                 # tmpfs 缓存沿用旧名，避免重复占空间
  DIT_WSRC+=("$MD/dit_qwenimage21-512.seg${i}.weight")
done
VAE_RKNN=(); VAE_WTMP=(); VAE_WSRC=()
for i in 0 1 2 3; do
  VAE_RKNN+=("$MD/vae_qwenimage21-512.seg${i}.1828.rknn")
  VAE_WTMP+=("vae512_f${i}m.weight")
  VAE_WSRC+=("$MD/vae_qwenimage21-512.seg${i}.weight")
done

WD=${QI_WORK_DIR:-/home/rockchip/qi_work}
TMPFS_W=${QI_TMPFS:-/mnt/qi_w}
CARD0=${CARD0:-0000:01:00.0}
CARD1=${CARD1:-0003:31:00.0}
FM_MU=${FM_MU:-0.5387096774}          # 512 专用（latent 序列长 1024；1024 的 0.6935 不能用）

ms()   { echo $(( ($(date +%s%N) - $1) / 1000000 )); }
say()  { echo "$*"; }

mkdir -p "$OUT"
LOG=$OUT/test.log
: > "$LOG"

# ── banner ─────────────────────────────────────────────────────────────────
say "╔══════════════════════════════════════════════════════════════╗"
say "║  Qwen-Image-2.1 → RKNN(rk1820)  512×512 出图                  ║"
say "╚══════════════════════════════════════════════════════════════╝"
say "  提示词 : $PROMPT"
say "  步数   : $STEPS   (mu=$FM_MU)"
say "  输出   : $OUT"
say "  模型包 : $MD"
say "  运行时 : $BIN"
say "  噪声   : $NOISE_MODE"
say "  时间   : $(date '+%Y-%m-%d %H:%M:%S')"
say ""

T_ALL=$(date +%s%N)

# ── ① 环境准备 ─────────────────────────────────────────────────────────────
say "── ① 环境准备 ──────────────────────────────────────────"
T0=$(date +%s%N)

if ! mount | grep -q "on $TMPFS_W "; then
  sudo mkdir -p "$TMPFS_W"
  sudo mount -t tmpfs -o size=5242880k tmpfs "$TMPFS_W"
  say "  挂载 tmpfs $TMPFS_W ✅"
fi

# proxy：已在跑就**不重启**（别碰卡的状态）
if ps -eo comm | grep -q "^rknn3_transfer_"; then
  say "  proxy 已在跑 ⇒ 不重启"
else
  say "  起 proxy …"
  sudo -E setsid nohup /bin/rknn3_transfer_proxy > "$WD/proxy.log" 2>&1 < /dev/null &
  sleep 30
fi

# 权重暂存到 tmpfs（已在则跳过）
for i in 0 1 2 3; do
  for pair in "${DIT_WTMP[$i]}:${DIT_WSRC[$i]}" "${VAE_WTMP[$i]}:${VAE_WSRC[$i]}"; do
    name=${pair%%:*}; src=${pair#*:}
    [ -f "$TMPFS_W/$name" ] && continue
    [ -f "$src" ] && cp "$src" "$TMPFS_W/$name" && say "    暂存 $name"
  done
done
df -h "$TMPFS_W" | tail -1 | sed 's/^/  /'
PREP_MS=$(ms $T0)
say "  准备耗时 ${PREP_MS} ms"
say ""

# ── ② TE（单独跑）───────────────────────────────────────────────────────────
say "── ② TE（text encoder，单独占卡池）──────────────────────"
T0=$(date +%s%N)
TE_OUT=$OUT/te
"$BIN" te-plain \
  --model "$TE_MODEL" \
  --weight "$TE_WEIGHT" \
  --tokenizer "$TE_TOKENIZER" \
  --embed "$TE_EMBED" \
  --prompt "$PROMPT" --output "$TE_OUT" 2>&1 | tee -a "$LOG" \
  | grep -E "model_init|推理完成|已写|错误"
RC=${PIPESTATUS[0]}
TE_MS=$(ms $T0)
if [ ! -f "$TE_OUT/encoder_hidden_states.bin" ]; then
  say "  ❌ TE 没产出 encoder_hidden_states.bin（退出码 $RC）"; exit 2
fi
say "  TE 小计 ${TE_MS} ms"
say ""

# ── ③ DiT + VAE ────────────────────────────────────────────────────────────
say "── ③ DiT（双卡常驻，零 load/init）+ VAE 4 段链 ─────────────"
T0=$(date +%s%N)
ML=""; WL=""; DL=""
for i in 0 1 2 3; do
  ML="$ML,${DIT_RKNN[$i]}"
  if [ -f "$TMPFS_W/${DIT_WTMP[$i]}" ]; then
    WL="$WL,$TMPFS_W/${DIT_WTMP[$i]}"
  else
    WL="$WL,${DIT_WSRC[$i]}"
  fi
  if [ "$i" -lt 2 ]; then DL="$DL,$CARD0"; else DL="$DL,$CARD1"; fi
done
ML=${ML#,}; WL=${WL#,}; DL=${DL#,}

VML=""; VWL=""
for i in 0 1 2 3; do
  VML="$VML,${VAE_RKNN[$i]}"
  if [ -f "$TMPFS_W/${VAE_WTMP[$i]}" ]; then
    VWL="$VWL,$TMPFS_W/${VAE_WTMP[$i]}"
  else
    VWL="$VWL,${VAE_WSRC[$i]}"
  fi
done
VML=${VML#,}; VWL=${VWL#,}

if [ "$NOISE_MODE" = "fresh" ]; then
  NOISE_FLAG="--noise-fresh"
else
  NOISE_FLAG=""
fi

"$BIN" serve \
  --models "$ML" --weights "$WL" --device-list "$DL" \
  --te-dirs "$TE_OUT" \
  --input-dir "$REF_DIR" \
  --output "$OUT" --repeat 1 \
  --fm-steps "$STEPS" --fm-mu "$FM_MU" $NOISE_FLAG \
  --vae-models "$VML" --vae-weights "$VWL" \
  --vae-device "$CARD1" --vae-stream --vae-release-dit 2>&1 | tee -a "$LOG" \
  | grep -E "常驻 init|flow-match|vae-chain|阶段合计|全部 1 张|错误|E RKNNAPI|MODEL_SETUP"
RC=${PIPESTATUS[0]}
GEN_MS=$(ms $T0)
say "  生成小计 ${GEN_MS} ms"
say ""

# ── ④ 汇总 ─────────────────────────────────────────────────────────────────
TOTAL_MS=$(ms $T_ALL)
say "═══════════════════════════════════════════════════════════════"
say " 耗时汇总"
say "═══════════════════════════════════════════════════════════════"
say "  ① 环境准备        $(awk -v v=$PREP_MS 'BEGIN{printf "%8.2f s", v/1000}')"
say "  ② TE（编码）      $(awk -v v=$TE_MS 'BEGIN{printf "%8.2f s", v/1000}')"
say "  ③ DiT + VAE 生成  $(awk -v v=$GEN_MS 'BEGIN{printf "%8.2f s", v/1000}')"
say "  ────────────────────────────────"
say "  ⭐ 总计            $(awk -v v=$TOTAL_MS 'BEGIN{printf "%8.2f s", v/1000}')"
say ""
if [ -f "$OUT/img01.png" ]; then
  say "  ✅ 出图成功：$OUT/img01.png"
  ls -la "$OUT"/img01* | sed 's/^/     /'
else
  say "  ❌ 没出图（退出码 $RC），看日志：$LOG"
fi
say "  完整日志：$LOG"
exit $RC
