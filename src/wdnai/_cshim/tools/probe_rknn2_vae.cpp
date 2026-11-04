// ---------------------------------------------------------------------------
// 板端探针：用**cshim 自己的 rknn2_runner** 跑一个 .rknn，
// 并和 PyTorch 金标准对拍（max|d| / rel_L2 / cosine）。
//
// 为什么单独写这个：
//   - 验证 qi::ModelRunner2 这条**真实代码路径**（不只是 runtime 能不能跑）
//   - 验证布局（喂进去的是逻辑 NCHW，取出来也要是逻辑 NCHW）
//   - 直接给出板端**真实精度**（比模拟器可信得多）
//
// 用法：
//   probe_rknn2_vae <model.rknn> <in.bin> [ref.bin]
//     in.bin  : float32 [1,64,1,32,32]（**已按 latents_std/mean 归一化**的潜变量）
//     ref.bin : float32 [512,512,4]（PyTorch fp32 金标准，可选）
// ---------------------------------------------------------------------------
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../qi/rknn2_runner.h"
#include "../qi/rknn3_runner.h"   // fp16->fp32 精确转换（与 cshim 同一份）

static bool read_raw(const std::string& p, std::vector<float>* out, size_t expect) {
  FILE* f = fopen(p.c_str(), "rb");
  if (!f) return false;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  size_t cnt = (size_t)n / sizeof(float);
  out->resize(cnt);
  size_t rd = fread(out->data(), 1, (size_t)n, f);
  fclose(f);
  if (expect && cnt != expect) {
    printf("  ⚠️ %s 元素数 %zu != 期望 %zu\n", p.c_str(), cnt, expect);
  }
  return rd == (size_t)n;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("用法: %s <model.rknn> <in.bin> [ref.bin]\n", argv[0]);
    return 2;
  }
  const char* model = argv[1];
  const char* in_bin = argv[2];
  const char* ref_bin = argc > 3 ? argv[3] : nullptr;

  const qi::Rknn2Api& api = qi::rknn2_api();
  if (!api.ok) {
    printf("❌ rknn2 runtime 不可用：%s\n", api.err.c_str());
    return 1;
  }
  printf("=== rknn2 runtime 可用 ===\n");

  // ---- 1. init（这一步决定内存够不够）----
  qi::ModelRunner2 m;
  // 核掩码：默认 3 核全用；PROBE_MASK=1/2/4/7 可覆盖。
  //   ⚠️ 用它来验证「NPU arena 是否随**启用核数**缩放」——
  //     fp16 整图 3 核要 4335 MiB > 池上限 4080 MiB，若 1~2 核能装下，
  //     说明 arena 是按核数分摊的 ⇒ 可以用 fp16（精度正确）跑。
  uint32_t mask = RKNN_NPU_CORE_0_1_2;
  if (const char* mk = getenv("PROBE_MASK")) mask = (uint32_t)strtoul(mk, nullptr, 0);
  printf("--- init %s (core_mask=0x%x) ---\n", model, mask);
  int ret = m.init(model, mask);   // 默认 3 个核全用
  if (ret != 0) {
    printf("❌ init 失败：%s\n", m.error().c_str());
    printf("   ⇒ 这个 .rknn 的 NPU 占用超过 RK3588 的缓冲池\n");
    return 1;
  }
  m.print_info();

  // ---- 2. 喂潜变量 ----
  std::vector<float> x;
  // 期望的元素数从**模型自己的输入属性**拿（512 分辨率是 64*32*32，
  // 256 分辨率是 64*16*16）—— 写死 32*32 会让 256 的模型没法测。
  size_t want_elems = 64u * 32u * 32u;
  if (!m.inputs().empty()) {
    uint32_t ne = m.inputs()[0].n_elems;
    if (ne) want_elems = ne;
  }
  if (!read_raw(in_bin, &x, want_elems)) {
    printf("❌ 读输入失败: %s\n", in_bin);
    return 1;
  }
  printf("--- 输入 %zu 元素 ---\n", x.size());
  const std::string& in_name = m.inputs().empty() ? std::string() : m.inputs()[0].name;
  ret = m.set_input(in_name, x.data(), x.size() * sizeof(float), RKNN_TENSOR_FLOAT32);
  if (ret != 0) {
    printf("❌ set_input 失败：%s\n", m.error().c_str());
    return 1;
  }

  // ---- 3. 跑 ----
  int runs = 3;
  if (const char* e = getenv("PROBE_RUNS")) runs = atoi(e);
  double ms = 0.0;
  for (int r = 0; r < runs; ++r) {
    ret = m.run(&ms);
    if (ret != 0) {
      printf("❌ run 失败：%s\n", m.error().c_str());
      return 1;
    }
    printf("--- run #%d 耗时 %.1f ms ---\n", r + 1, ms);
  }

  // ---- 4. 取输出 + 对拍 ----
  const char* out_name = "image";
  bool have = false;
  for (const auto& o : m.outputs()) {
    if (o.name == out_name) have = true;
  }
  if (!have && !m.outputs().empty()) out_name = m.outputs()[0].name.c_str();

  std::vector<unsigned char> bytes;
  rknn_tensor_type dt = RKNN_TENSOR_FLOAT16;
  if (!m.get_output(out_name, &bytes, &dt)) {
    printf("❌ 取输出 '%s' 失败\n", out_name);
    return 1;
  }
  const size_t nelem = bytes.size() / (dt == RKNN_TENSOR_FLOAT32 ? 4 : 2);
  std::vector<float> y(nelem);
  for (size_t i = 0; i < nelem; ++i) {
    if (dt == RKNN_TENSOR_FLOAT32) {
      memcpy(&y[i], bytes.data() + i * 4, 4);
    } else {
      uint16_t hv;
      memcpy(&hv, bytes.data() + i * 2, 2);
      float16 h16;
      memcpy(&h16, &hv, sizeof(uint16_t));
      y[i] = qi::fp16_to_fp32_exact(h16);
    }
  }
  printf("--- 输出 '%s' %zu 元素 dtype=%d ---\n", out_name, nelem, (int)dt);

  // ---- 可选：把输出 dump 出来（离线做排列/位移/尺度分析用）----
  if (const char* dp = getenv("PROBE_DUMP")) {
    FILE* f = fopen(dp, "wb");
    if (f) {
      fwrite(y.data(), sizeof(float), nelem, f);
      fclose(f);
      printf("    已 dump 到 %s（fp32, NCHW 逻辑布局）\n", dp);
    }
  }

  // ---- 输出统计（确认不是全 0 / 不是垃圾）----
  {
    double mn = 1e30, mx = -1e30, sum = 0;
    for (size_t i = 0; i < nelem; ++i) {
      double v = y[i];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
      sum += v;
    }
    printf("    值域 [%.4f, %.4f]  均值 %.4f\n", mn, mx, sum / (double)nelem);
  }

  if (ref_bin) {
    std::vector<float> ref;
    if (!read_raw(ref_bin, &ref, 512 * 512 * 4)) {
      printf("⚠️ 读金标准失败\n");
      return 1;
    }
    // 输出是 [1,4,512,512] 的逻辑 NCHW ⇒ 转成 [512,512,4] 再比
    const size_t hw = 512 * 512;
    if (nelem < hw * 4) {
      printf("⚠️ 输出元素数不够\n");
      return 1;
    }
    double maxd = 0, sumd = 0;
    double dot = 0, ny = 0, nr = 0, num = 0;
    for (size_t p = 0; p < hw; ++p) {
      for (int ch = 0; ch < 4; ++ch) {
        float o = y[(size_t)ch * hw + p];        // NCHW
        float r = ref[p * 4 + ch];              // HWC
        double d = fabs((double)o - (double)r);
        if (d > maxd) maxd = d;
        sumd += d;
        dot += (double)o * (double)r;
        ny += (double)o * (double)o;
        nr += (double)r * (double)r;
        double e = (double)o - (double)r;
        num += e * e;
      }
    }
    double rel = sqrt(num) / (sqrt(nr) + 1e-12);
    double cos = dot / (sqrt(ny) * sqrt(nr) + 1e-12);
    printf("\n=== 板端精度 vs PyTorch 金标准 ===\n");
    printf("  max|d|   = %.6g\n", maxd);
    printf("  mean|d|  = %.6g\n", sumd / (double)(hw * 4));
    printf("  rel_L2   = %.6g\n", rel);
    printf("  cosine   = %.6f\n", cos);
  }
  m.release();
  return 0;
}
