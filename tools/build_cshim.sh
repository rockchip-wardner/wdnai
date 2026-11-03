#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 构建 wdnai 的 cshim（Qwen-Image-2.1 出图运行时，C 实现）
#
#   bash tools/build_cshim.sh            # 板端原生编译（aarch64）
#   bash tools/build_cshim.sh --clean
#
# 产物：src/wdnai/_cshim/libqi_qwen.so
#
# ⚠️ 依赖：**只有系统自带的 librknn3_api.so**
#    （门面 librknn3_api.so 会 dlopen 系统自带的 librknn3_api_rkcp.so）
#    ⇒ 编译产物**不携带任何 Rockchip 的 .so / .bin** ⇒ 发布包只有模型。
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CSHIM="$ROOT/src/wdnai/_cshim"

if [[ "${1:-}" == "--clean" ]]; then
  make -C "$CSHIM" clean
  echo "已清理。"
  exit 0
fi

echo "== 构建 cshim =="
make -C "$CSHIM" "${@:-all}"

echo
echo "产物: $CSHIM/libqi_qwen.so"
echo "依赖: $(ldd "$CSHIM/libqi_qwen.so" | grep -i rknn || echo '（无 rknn 依赖?!）')"
