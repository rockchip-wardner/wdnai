// ---------------------------------------------------------------------------
// 板端探针：**分段顺序加载**（streaming）的峰值内存验证。
//
// 目的
// ----
// 验证 qi::VaeChain2 的「流式」模式（init 段0 → run → release → init 段1 → run）
// 在 RK3588 上的**峰值 NPU 占用 = 单段最大**，而不是各段之和。
//
// 背景
// ----
// 整图 512 VAE 的 internal 是 3852 MiB，池上限约 4080 MiB ⇒ 装不下。
// 分段后每段的 internal 更小 ⇒ 流式加载的峰值应当能降下来。
// 但这需要**实测**确认，因为 NPU 的 IOVA 分配可能是「累计不释放」。
//
// 用法：
//   probe_stream <seg0.rknn> <seg1.rknn> <in.bin> [ref.bin]
//     段0 的输出张量名必须与段1 的输入张量名一致（按名字对齐）。
// ---------------------------------------------------------------------------
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../qi/rknn2_runner.h"
#include "../qi/rknn3_runner.h"

static bool read_raw(const std::string& p, std::vector<float>* out) {
  FILE* f = fopen(p.c_str(), "rb");
  if (!f) return false;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  out->resize((size_t)n / sizeof(float));
  size_t rd = fread(out->data(), 1, (size_t)n, f);
  fclose(f);
  return rd == (size_t)n;
}

// 读 NPU 当前占用（MiB），读不到就返回 -1
static double npu_load_mib() {
  FILE* f = fopen("/sys/kernel/debug/rknpu/load", "r");
  if (!f) return -1.0;
  char buf[512] = {0};
  size_t n = fread(buf, 1, sizeof(buf) - 1, f);
  fclose(f);
  buf[n] = 0;
  // 格式: "NPU load:  Core0:  0%, Core1:  0%, Core2:  0%,"
  double pct = 0;
  const char* p = strstr(buf, "Core0:");
  if (p) pct = atof(p + 6);
  return pct;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    printf("用法: %s <seg0.rknn> <seg1.rknn> <in.bin> [ref.bin]\n", argv[0]);
    return 2;
  }
  const std::string m0 = argv[1], m1 = argv[2], in_bin = argv[3];
  const char* ref_bin = argc > 4 ? argv[4] : nullptr;

  const qi::Rknn2Api& api = qi::rknn2_api();
  if (!api.ok) {
    printf("❌ rknn2 runtime 不可用：%s\n", api.err.c_str());
    return 1;
  }
  printf("=== rknn2 runtime 可用 ===\n");

  uint32_t mask = RKNN_NPU_CORE_0_1_2;
  if (const char* mk = getenv("PROBE_MASK")) mask = (uint32_t)strtoul(mk, nullptr, 0);

  std::vector<float> x;
  if (!read_raw(in_bin, &x)) {
    printf("❌ 读输入失败: %s\n", in_bin.c_str());
    return 1;
  }
  printf("--- 输入 %s: %zu 个 float32\n", in_bin.c_str(), x.size());

  // ---- 段0 ----
  qi::ModelRunner2 s0;
  printf("--- init 段0: %s (core_mask=0x%x) ---\n", m0.c_str(), mask);
  if (s0.init(m0, mask) != 0) {
    printf("❌ 段0 init 失败：%s\n", s0.error().c_str());
    return 1;
  }
  s0.print_info();

  // 喂段0 的第 0 个输入
  const auto& in0 = s0.inputs();
  if (in0.empty()) {
    printf("❌ 段0 没有输入\n");
    return 1;
  }
  size_t elems = x.size();
  if (in0[0].n_elems && elems != in0[0].n_elems) {
    printf("  ⚠️ 输入元素数 %zu != 模型期望 %u\n", elems, in0[0].n_elems);
  }
  // 用模型自己的输入 dtype
  rknn_tensor_type dt = (rknn_tensor_type)in0[0].type;
  if (dt != RKNN_TENSOR_FLOAT32) dt = RKNN_TENSOR_FLOAT32;
  if (s0.set_input(in0[0].name, x.data(), elems * sizeof(float), dt) != 0) {
    printf("❌ 段0 喂输入失败：%s\n", s0.error().c_str());
    return 1;
  }
  double ms = 0;
  if (s0.run(&ms) != 0) {
    printf("❌ 段0 推理失败：%s\n", s0.error().c_str());
    return 1;
  }
  printf("--- 段0 推理完成 %.1f ms ---\n", ms);

  // 取段0 的输出（找非权重、非输入的那个）
  std::string carry_name;
  std::vector<unsigned char> carry;
  rknn_tensor_type carry_dt = RKNN_TENSOR_FLOAT16;
  for (const auto& o : s0.outputs()) {
    if (!s0.get_output(o.name, &carry, &carry_dt)) continue;
    carry_name = o.name;
    printf("--- 段0 输出 '%s': %u 元素, dtype=%d, %zu 字节 ---\n",
           o.name.c_str(), o.n_elems, (int)carry_dt, carry.size());
    break;
  }
  if (carry.empty()) {
    printf("❌ 段0 没有可用输出\n");
    return 1;
  }

  // ---- 关键：**释放段0**（流式的核心）----
  printf("--- release 段0（流式：释放后再 init 段1）---\n");
  s0.release();

  // ---- 段1 ----
  qi::ModelRunner2 s1;
  printf("--- init 段1: %s ---\n", m1.c_str());
  if (s1.init(m1, mask) != 0) {
    printf("❌ 段1 init 失败：%s\n", s1.error().c_str());
    return 1;
  }
  s1.print_info();

  bool fed = false;
  for (const auto& in1 : s1.inputs()) {
    if (in1.name == carry_name) {
      if (s1.set_input(in1.name, carry.data(), carry.size(), carry_dt) != 0) {
        printf("❌ 段1 喂段间张量失败：%s\n", s1.error().c_str());
        return 1;
      }
      fed = true;
      break;
    }
  }
  if (!fed) {
    printf("❌ 段1 没有名为 '%s' 的输入（现有：", carry_name.c_str());
    for (const auto& i : s1.inputs()) printf("%s ", i.name.c_str());
    printf("）\n");
    return 1;
  }
  printf("--- 段间张量 '%s' 已对齐 ---\n", carry_name.c_str());
  if (s1.run(&ms) != 0) {
    printf("❌ 段1 推理失败：%s\n", s1.error().c_str());
    return 1;
  }
  printf("--- 段1 推理完成 %.1f ms ---\n", ms);

  // 取最终图像
  for (const auto& o : s1.outputs()) {
    std::vector<unsigned char> bytes;
    rknn_tensor_type odt;
    if (!s1.get_output(o.name, &bytes, &odt)) continue;
    const bool f32 = (odt == RKNN_TENSOR_FLOAT32);
    size_t nelem = bytes.size() / (f32 ? 4 : 2);
    std::vector<float> f(nelem);
    if (f32) {
      memcpy(f.data(), bytes.data(), nelem * sizeof(float));
    } else {
      for (size_t k = 0; k < nelem; ++k) {
        float16 hv;
        memcpy(&hv, bytes.data() + k * sizeof(uint16_t), sizeof(uint16_t));
        f[k] = qi::fp16_to_fp32_exact(hv);
      }
    }
    printf("--- 段1 输出 '%s': %zu 元素 (dtype=%d) ---\n", o.name.c_str(), nelem, (int)odt);
    if (nelem >= 4) {
      double mn = 1e30, mx = -1e30, sum = 0;
      for (size_t k = 0; k < nelem; ++k) {
        if (f[k] < mn) mn = f[k];
        if (f[k] > mx) mx = f[k];
        sum += f[k];
      }
      printf("    值域 [%.4f, %.4f] 均值 %.4f\n", mn, mx, sum / (double)nelem);
    }
    if (ref_bin) {
      std::vector<float> ref;
      if (read_raw(ref_bin, &ref) && ref.size() == nelem) {
        double d2 = 0, r2 = 0, dot = 0, na = 0, nb = 0;
        for (size_t k = 0; k < nelem; ++k) {
          d2 += (f[k] - ref[k]) * (f[k] - ref[k]);
          r2 += ref[k] * ref[k];
          dot += f[k] * ref[k];
          na += f[k] * f[k];
          nb += ref[k] * ref[k];
        }
        printf("    rel_L2 = %.6g   cosine = %.6f\n",
               sqrt(d2) / (sqrt(r2) + 1e-12), dot / (sqrt(na * nb) + 1e-12));
      } else {
        printf("    (ref 尺寸不符: %zu vs %zu)\n", ref.size(), nelem);
      }
    }
    break;
  }

  printf("=== 流式分段链路 OK ===\n");
  return 0;
}
