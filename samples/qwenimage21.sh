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
#   流程（板端实测，2026-11-02 第三十五轮）：
#     ① 一个 serve 进程里全做完 —— **TE 常驻** + DiT 4 段常驻 + VAE 4 段流式
#        卡 0 : TE（486.0 MiB/核）+ DiT 的头（6.4 MiB/核）= 492.4 ✅
#        卡 1 : DiT 4 段（切头，--share-internal，525.0 MiB/核）
#               → N 张图跑完 → 释放 → VAE 4 段流式（峰值 516.5）
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
#     dit_qwenimage21-512hc.seg{0..3}.1828.rknn + .weight   DiT 4 段（切头）
#     dit_qwenimage21-512.head.1828.rknn + .weight         DiT 的「头」
#     dit_qwenimage21-512hc.int_ledger                     每核 internal 账本
#     vae_qwenimage21-512.seg{0..3}.1828.rknn + .weight
#     reference_512/hidden_states.bin          ← 初始噪声
#     runtime/qwen_image_rknn + runtime/lib/aarch64/   ← C++ 运行时
#
#   ⏱ 耗时（RK3588 + 2×RK1828，8 步）：一次性 init ≈ 78 s，
#      之后**每张图 ≈ 34 s**（TE 常驻，编码只要 0.16 s）。
#      步数是纯运行时参数，模型不用重建。
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
if [ ! -f "$MD/llm_qwenimage21-te.1828.rknn" ] || [ ! -f "$MD/dit_qwenimage21-512hc.seg0.1828.rknn" ] || \
   [ ! -f "$MD/dit_qwenimage21-512.head.1828.rknn" ]; then
  echo "❌ 模型包不完整：$MD"
  echo "   需要 llm_qwenimage21-te.1828.rknn + dit_qwenimage21-512hc.seg0.1828.rknn + dit_qwenimage21-512.head.1828.rknn"
  echo "   （只支持「切头单卡」那一套；老的双卡版已废弃）"
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

# ⛔ 运行时版本护栏（2026-11-02 第三十五轮踩到）
#   模型包 runtime/ 里那份二进制可能**过期**（例：没有 --te-resident）⇒
#   老 runtime 会**默默忽略**新参数 ⇒ 走回「读 te 目录里现成的 ehs」的老路
#   ⇒ 报「TE 目录 xxx 里缺 encoder_hidden_states.bin」，白等 79 s。
NEEDED_FLAGS="te-resident share-internal head-model vae-stream"
BIN_STR=$(strings "$BIN" 2>/dev/null)
for fl in $NEEDED_FLAGS; do
  case "$BIN_STR" in
    *"$fl"*) : ;;
    *) echo "❌ 运行时太旧：$BIN 不支持 --$fl"
       echo "   （模型包 runtime/ 里的副本可能是旧的）"
       echo "   ⇒ 重新 make，并把 bin/qwen_image_rknn 同步到 $MD/runtime/"
       exit 4;;
  esac
done

export LD_LIBRARY_PATH=$LIB:${LD_LIBRARY_PATH:-}
export QI_SPREAD_IO=${QI_SPREAD_IO:-0}
export QI_SPREAD_IO_IN=${QI_SPREAD_IO_IN:-1}
export RKNN3_NETWORK_SOCKET_FILE=/tmp/rk-mdns.ini

TE_MODEL="$MD/llm_qwenimage21-te.1828.rknn"
TE_WEIGHT="$MD/llm_qwenimage21-te.weight"
TE_TOKENIZER="$MD/qwenimage21-te.tokenizer.gguf"
TE_EMBED="$MD/qwenimage21-te.embed.bin"
REF_DIR="$MD/reference_512"

# ── 模型（**只认「切头单卡」那一套**：4 段 + 独立的头模型）────────────────
# ⚠️ 老的双卡版（dit_qwenimage21-512.seg*）已废弃，不再支持。
# ⚠️ 权重**直接读模型包**（不再走 tmpfs）：实测 NVMe 直读的 load 时间与 tmpfs 一样。
DIT_RKNN=(); DIT_WSRC=()
for i in 0 1 2 3; do
  DIT_RKNN+=("$MD/dit_qwenimage21-512hc.seg${i}.1828.rknn")
  DIT_WSRC+=("$MD/dit_qwenimage21-512hc.seg${i}.weight")
done
HEAD_RKNN="$MD/dit_qwenimage21-512.head.1828.rknn"
HEAD_WSRC="$MD/dit_qwenimage21-512.head.weight"

VAE_RKNN=(); VAE_WSRC=()
for i in 0 1 2 3; do
  VAE_RKNN+=("$MD/vae_qwenimage21-512.seg${i}.1828.rknn")
  VAE_WSRC+=("$MD/vae_qwenimage21-512.seg${i}.weight")
done

WD=${QI_WORK_DIR:-/home/rockchip/qi_work}
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
say "  布局   : 卡0 = TE 常驻 + 头  |  卡1 = DiT 4 段 → VAE"
say "  运行时 : $BIN"
say "  噪声   : $NOISE_MODE"
say "  时间   : $(date '+%Y-%m-%d %H:%M:%S')"
say ""

T_ALL=$(date +%s%N)

# ── ① 环境准备 ─────────────────────────────────────────────────────────────
say "── ① 环境准备 ──────────────────────────────────────────"
T0=$(date +%s%N)

# proxy：已在跑就**不重启**（别碰卡的状态）
if ps -eo comm | grep -q "^rknn3_transfer_"; then
  say "  proxy 已在跑 ⇒ 不重启"
else
  say "  起 proxy …"
  sudo -E setsid nohup /bin/rknn3_transfer_proxy > "$WD/proxy.log" 2>&1 < /dev/null &
  sleep 30
fi

PREP_MS=$(ms $T0)
say "  准备耗时 ${PREP_MS} ms"
say ""

# ── ② TE（**常驻**，进程内）─────────────────────────────────────────────────
# ⭐⭐⭐ TE 的 model_init 要 36.8 s，真推理只要 0.16 s ⇒ 常驻后每张图省 37 s。
#    serve 读每个目录里的 prompt.txt → 常驻 TE 编码 → 直接写进工作目录。
# ⭐⭐⭐ 批量：PROMPT 用 "|" 分隔多个 ⇒ serve **一次进程**跑完所有图
#    ⇒ TE / DiT / VAE 都只 init 一次。
say "── ② TE（常驻，进程内；每个 prompt 一份）────────────────"
T0=$(date +%s%N)
TE_DIRS=""
IFS='|' read -r -a PROMPTS <<< "$PROMPT"
for pi in "${!PROMPTS[@]}"; do
  TE_OUT=$OUT/te$((pi + 1))
  mkdir -p "$TE_OUT"
  printf '%s' "${PROMPTS[$pi]}" > "$TE_OUT/prompt.txt"
  TE_DIRS="$TE_DIRS,$TE_OUT"
done
TE_DIRS=${TE_DIRS#,}
say "  ${#PROMPTS[@]} 个 prompt：${TE_DIRS}"
TE_MS=0
say ""

# ── ③ DiT + VAE ────────────────────────────────────────────────────────────
say "── ③ TE 常驻 + DiT（单卡常驻，零 load/init）+ VAE 4 段链 ────"
T0=$(date +%s%N)
ML=""; WL=""; DL=""
for i in 0 1 2 3; do
  ML="$ML,${DIT_RKNN[$i]}"
  WL="$WL,${DIT_WSRC[$i]}"
  DL="$DL,$CARD1"          # ⭐ DiT 4 段全在卡 1（--share-internal 是**必要条件**）
done
ML=${ML#,}; WL=${WL#,}; DL=${DL#,}

VML=""; VWL=""
for i in 0 1 2 3; do
  VML="$VML,${VAE_RKNN[$i]}"
  VWL="$VWL,${VAE_WSRC[$i]}"
done
VML=${VML#,}; VWL=${VWL#,}

if [ "$NOISE_MODE" = "fresh" ]; then
  NOISE_FLAG="--noise-fresh"
else
  NOISE_FLAG=""
fi

"$BIN" serve \
  --models "$ML" --weights "$WL" --device-list "$DL" \
  --share-internal \
  --head-model "$HEAD_RKNN" --head-weight "$HEAD_WSRC" --head-device "$CARD0" \
  --te-resident \
  --te-model "$TE_MODEL" --te-weight "$TE_WEIGHT" \
  --te-tokenizer "$TE_TOKENIZER" --te-embed "$TE_EMBED" --te-device "$CARD0" \
  --te-dirs "$TE_DIRS" \
  --input-dir "$REF_DIR" \
  --output "$OUT" --repeat 1 \
  --fm-steps "$STEPS" --fm-mu "$FM_MU" $NOISE_FLAG \
  --vae-models "$VML" --vae-weights "$VWL" \
  --vae-device "$CARD1" --vae-stream --vae-release-dit 2>&1 | tee -a "$LOG" \
  | grep -E "TE 常驻|常驻 init|常驻 TE|flow-match|vae-chain|阶段合计|全部|错误|E RKNNAPI|MODEL_SETUP"
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
if ls "$OUT"/img*.png >/dev/null 2>&1; then
  NPNG=$(ls "$OUT"/img*.png 2>/dev/null | wc -l)
  say "  ✅ 出图成功（${NPNG} 张）："
  ls -la "$OUT"/img*.png | sed 's/^/     /'
else
  say "  ❌ 没出图（退出码 $RC），看日志：$LOG"
fi
say "  完整日志：$LOG"
exit $RC
