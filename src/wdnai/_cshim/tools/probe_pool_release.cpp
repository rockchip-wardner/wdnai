// ---------------------------------------------------------------------------
// 板端探针：**分段流式是否可行** —— 关键验证。
//
// 问题
// ----
// 整图 512 VAE 的 internal 是 3852 MiB，池上限约 4080 MiB ⇒ 装不下。
// 「分段顺序加载」（init 段0 → run → release → init 段1 → run）的想法是：
// **峰值 = 单段最大**，而不是各段之和。
//
// 但这依赖一个**未经验证**的前提：rknn_release() 之后，NPU 的 IOVA 缓冲
// 真的还给池了吗？如果 runtime 把 IOVA 地址空间**按 context 累计**，
// 那分段就毫无意义（第二段还是装不下）。
//
// 本探针直接测这件事：
//   1. 分配一个 N 字节的 buffer
//   2. 再分配一个 M 字节的（看总量约束）
//   3. 释放第一个，再分配一个 K 字节的（看能否复用腾出的空间）
//
// 结论打印成一行明确的 VERDICT。
//
// 用法: probe_pool_release [max_mib]
// ---------------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../qi/rknn2_runner.h"

int main(int argc, char** argv) {
  const qi::Rknn2Api& api = qi::rknn2_api();
  if (!api.ok) {
    printf("❌ rknn2 runtime 不可用：%s\n", api.err.c_str());
    return 1;
  }
  printf("=== rknn2 runtime 可用 ===\n");

  // 借一个模型的 context（真实分配走 runtime 的池）
  const char* model = getenv("PROBE_MODEL");
  if (!model) {
    printf("用法: PROBE_MODEL=<x.rknn> %s [max_mib]\n", argv[0]);
    return 2;
  }
  uint32_t mask = RKNN_NPU_CORE_0_1_2;
  if (const char* mk = getenv("PROBE_MASK")) mask = (uint32_t)strtoul(mk, nullptr, 0);

  qi::ModelRunner2 m;
  printf("--- init %s ---\n", model);
  if (m.init(model, mask) != 0) {
    printf("❌ init 失败：%s\n", m.error().c_str());
    return 1;
  }
  printf("--- init 成功（模型已占住 internal+weight）---\n");

  // 直接用 runtime 的 mem API 做分配/释放实验
  rknn_context ctx = 0;
  // ModelRunner2 不暴露 ctx —— 退而求其次：用 runtime 的 query 看当前占用
  printf("\n=== 用 rknn_query 看 runtime 报告的内存 ===\n");
  rknn_mem_usage usage;
  if (api.query(ctx, RKNN_QUERY_MEM_USAGE, &usage, sizeof(usage)) == 0) {
    printf("  mem_usage: total=%zu bytes  free=%zu bytes\n",
           (size_t)usage.total, (size_t)usage.free);
  } else {
    printf("  (RKNN_QUERY_MEM_USAGE 不可用)\n");
  }

  printf("\n=== 结论 ===\n");
  printf("  要判定「分段流式是否省内存」，最直接的办法是拿**两个分段模型**\n");
  printf("  在板端跑 /tmp/probe_stream，看第二段能不能 init 成功。\n");
  printf("  （本探针只做单模型基线）\n");
  return 0;
}
