// ---------------------------------------------------------------------------
// RKNN2（RK3588 本地 NPU）runner 实现 —— 见 rknn2_runner.h 的说明
// ---------------------------------------------------------------------------
#include "rknn2_runner.h"

#include <dlfcn.h>

#include <cstdio>
#include <cstring>
#include <ctime>
#include <mutex>

#include "rknn3_runner.h"   // 复用 fp16<->fp32 的精确转换

namespace qi {

namespace {

double now_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

template <typename T>
bool sym(void* h, const char* name, T* fn) {
  void* p = dlsym(h, name);
  if (!p) return false;
  *fn = reinterpret_cast<T>(p);
  return true;
}

// 这个 dtype 是不是**量化**的（int8/int16/...）⇒ 输出要用 want_float 取
bool is_quantized_type(rknn_tensor_type t) {
  switch (t) {
    case RKNN_TENSOR_INT8:
    case RKNN_TENSOR_UINT8:
    case RKNN_TENSOR_INT16:
    case RKNN_TENSOR_UINT16:
    case RKNN_TENSOR_INT32:
    case RKNN_TENSOR_UINT32:
    case RKNN_TENSOR_INT64:
    case RKNN_TENSOR_INT4:
      return true;
    default:
      return false;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// dlopen librknnrt
// ---------------------------------------------------------------------------
const Rknn2Api& rknn2_api() {
  static Rknn2Api api;
  static std::once_flag once;
  std::call_once(once, [&]() {
    // 依次尝试几个常见的 soname（apt 装的是 librknnrt.so.2.5.0 + .so 软链）
    const char* candidates[] = {
        "librknnrt.so.2.5.0", "librknnrt.so.2", "librknnrt.so",
        "librknnrt.so.1",    "librknnrt.so",
    };
    void* h = nullptr;
    for (const char* n : candidates) {
      h = dlopen(n, RTLD_NOW | RTLD_LOCAL);
      if (h) break;
    }
    if (!h) {
      api.err = "找不到 librknnrt（rknn2 runtime）—— 本机没有装 rknn2-runtime";
      return;
    }
    bool ok = true;
    ok &= sym(h, "rknn_init", &api.init);
    ok &= sym(h, "rknn_destroy", &api.destroy);
    ok &= sym(h, "rknn_query", &api.query);
    ok &= sym(h, "rknn_set_core_mask", &api.set_core_mask);
    ok &= sym(h, "rknn_inputs_set", &api.inputs_set);
    ok &= sym(h, "rknn_run", &api.run);
    ok &= sym(h, "rknn_outputs_get", &api.outputs_get);
    ok &= sym(h, "rknn_outputs_release", &api.outputs_release);
    if (!ok) {
      api.err = "librknnrt 里缺少需要的符号（版本不匹配？）";
      dlclose(h);
      return;
    }
    api.ok = true;
  });
  return api;
}

// ---------------------------------------------------------------------------
// ModelRunner2
// ---------------------------------------------------------------------------
ModelRunner2::~ModelRunner2() { release(); }

int ModelRunner2::init(const std::string& model_path, uint32_t core_mask) {
  release();
  const Rknn2Api& api = rknn2_api();
  if (!api.ok) {
    err_ = api.err;
    return -1;
  }
  model_path_ = model_path;

  rknn_init_extend ext;
  memset(&ext, 0, sizeof(ext));
  int ret = api.init(&ctx_, (void*)model_path.c_str(), 0, 0, &ext);
  if (ret != 0) {
    char buf[256];
    snprintf(buf, sizeof(buf), "rknn_init 失败 ret=%d（%s）", ret, model_path.c_str());
    err_ = buf;
    ctx_ = 0;
    return ret;
  }

  // 核掩码：RK3588 有 3 个核，默认用满（0_1_2）
  if (core_mask == 0) core_mask = RKNN_NPU_CORE_0_1_2;
  ret = api.set_core_mask(ctx_, (rknn_core_mask)core_mask);
  if (ret != 0) {
    printf("[rknn2] ⚠️ set_core_mask(%u) 失败 ret=%d（继续用默认核）\n", core_mask, ret);
  }

  // ---- 查 IO 属性 ----
  rknn_input_output_num io;
  memset(&io, 0, sizeof(io));
  ret = api.query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
  if (ret != 0) {
    char buf[128];
    snprintf(buf, sizeof(buf), "rknn_query(IN_OUT_NUM) 失败 ret=%d", ret);
    err_ = buf;
    release();
    return ret;
  }
  for (uint32_t i = 0; i < io.n_input; ++i) {
    rknn_tensor_attr a;
    memset(&a, 0, sizeof(a));
    a.index = i;
    if (api.query(ctx_, RKNN_QUERY_INPUT_ATTR, &a, sizeof(a)) != 0) continue;
    TensorAttr2 t;
    t.name = a.name;
    t.n_dims = a.n_dims > 8 ? 8 : a.n_dims;
    for (uint32_t d = 0; d < t.n_dims; ++d) t.dims[d] = a.dims[d];
    t.n_elems = a.n_elems;
    t.size = a.size;
    t.fmt = a.fmt;
    t.type = a.type;
    in_.push_back(t);
  }
  for (uint32_t i = 0; i < io.n_output; ++i) {
    rknn_tensor_attr a;
    memset(&a, 0, sizeof(a));
    a.index = i;
    if (api.query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &a, sizeof(a)) != 0) continue;
    TensorAttr2 t;
    t.name = a.name;
    t.n_dims = a.n_dims > 8 ? 8 : a.n_dims;
    for (uint32_t d = 0; d < t.n_dims; ++d) t.dims[d] = a.dims[d];
    t.n_elems = a.n_elems;
    t.size = a.size;
    t.fmt = a.fmt;
    t.type = a.type;
    out_.push_back(t);
  }
  return 0;
}

void ModelRunner2::release() {
  const Rknn2Api& api = rknn2_api();
  if (ctx_ && api.ok) {
    if (!last_outputs_.empty()) {
      api.outputs_release(ctx_, (uint32_t)last_outputs_.size(), last_outputs_.data());
      last_outputs_.clear();
    }
    api.destroy(ctx_);
  }
  ctx_ = 0;
  in_.clear();
  out_.clear();
  last_bytes_.clear();
}

int ModelRunner2::set_input(const std::string& name, const void* data, size_t bytes,
                            rknn_tensor_type dtype) {
  const Rknn2Api& api = rknn2_api();
  if (!api.ok || !ctx_) return -1;

  uint32_t idx = 0;
  bool found = false;
  for (uint32_t i = 0; i < in_.size(); ++i) {
    if (in_[i].name == name) { idx = i; found = true; break; }
  }
  if (!found) {
    err_ = "模型没有名为 '" + name + "' 的输入张量";
    return -1;
  }

  rknn_input rin;
  memset(&rin, 0, sizeof(rin));
  rin.index = idx;
  rin.buf = const_cast<void*>(data);
  rin.size = (uint32_t)bytes;
  rin.pass_through = 0;
  rin.type = dtype;
  // ⭐ 喂的是**逻辑 NCHW**；runtime 负责转成模型内部布局
  rin.fmt = RKNN_TENSOR_NCHW;
  int ret = api.inputs_set(ctx_, 1, &rin);
  if (ret != 0) {
    char buf[160];
    snprintf(buf, sizeof(buf), "rknn_inputs_set('%s') 失败 ret=%d", name.c_str(), ret);
    err_ = buf;
    return ret;
  }
  return 0;
}

int ModelRunner2::run(double* out_ms) {
  const Rknn2Api& api = rknn2_api();
  if (!api.ok || !ctx_) return -1;

  // 释放上一次的输出缓冲
  if (!last_outputs_.empty()) {
    api.outputs_release(ctx_, (uint32_t)last_outputs_.size(), last_outputs_.data());
    last_outputs_.clear();
    last_bytes_.clear();
  }

  double t0 = now_ms();
  int ret = api.run(ctx_, nullptr);
  if (ret != 0) {
    char buf[128];
    snprintf(buf, sizeof(buf), "rknn_run 失败 ret=%d", ret);
    err_ = buf;
    return ret;
  }

  last_outputs_.assign(out_.size(), rknn_output());
  actual_types_.assign(out_.size(), RKNN_TENSOR_FLOAT16);
  for (size_t i = 0; i < out_.size(); ++i) {
    last_outputs_[i].index = (uint32_t)i;
    // ⭐ **量化模型**（w8a8）的输出是 int8 ⇒ 必须 want_float=1 让 runtime
    //   反量化成 fp32，否则拿到的是裸 int8（当 fp16 读就是垃圾）。
    const bool quant = is_quantized_type(out_[i].type);
    last_outputs_[i].want_float = quant ? 1 : 0;
    last_outputs_[i].is_prealloc = 0;
    actual_types_[i] = quant ? RKNN_TENSOR_FLOAT32 : out_[i].type;
  }
  ret = api.outputs_get(ctx_, (uint32_t)last_outputs_.size(), last_outputs_.data(), nullptr);
  if (ret != 0) {
    char buf[128];
    snprintf(buf, sizeof(buf), "rknn_outputs_get 失败 ret=%d", ret);
    err_ = buf;
    return ret;
  }
  // 立刻把字节拷出来（outputs_get 的缓冲在 outputs_release 后就没了）
  last_bytes_.assign(out_.size(), std::vector<unsigned char>());
  for (size_t i = 0; i < out_.size(); ++i) {
    if (!last_outputs_[i].buf || last_outputs_[i].size == 0) continue;
    const unsigned char* p = (const unsigned char*)last_outputs_[i].buf;
    last_bytes_[i].assign(p, p + last_outputs_[i].size);
  }
  double t1 = now_ms();
  if (out_ms) *out_ms = t1 - t0;
  return 0;
}

bool ModelRunner2::get_output(const std::string& name, std::vector<unsigned char>* bytes,
                               rknn_tensor_type* dtype) const {
  for (size_t i = 0; i < out_.size(); ++i) {
    if (out_[i].name != name) continue;
    if (i >= last_bytes_.size() || last_bytes_[i].empty()) return false;
    rknn_tensor_type dt = (i < actual_types_.size()) ? actual_types_[i] : out_[i].type;
    if (dtype) *dtype = dt;
    // 布局按 **attr 的 fmt** 判（量化模型的 attr.fmt 仍描述逻辑布局），
    // 元素大小按**实际拿到的** dtype 算。
    TensorAttr2 a = out_[i];
    a.type = dt;
    return rknn2_to_logical_nchw(a, last_bytes_[i].data(), bytes);
  }
  return false;
}

void ModelRunner2::print_info() const {
  printf("[rknn2] 模型 %s\n", model_path_.c_str());
  for (const auto& t : in_) {
    printf("[rknn2]   IN  %-16s dims=[", t.name.c_str());
    for (uint32_t d = 0; d < t.n_dims; ++d) printf("%s%u", d ? "," : "", t.dims[d]);
    printf("] elems=%u size=%u fmt=%d type=%d\n", t.n_elems, t.size, (int)t.fmt, (int)t.type);
  }
  for (const auto& t : out_) {
    printf("[rknn2]   OUT %-16s dims=[", t.name.c_str());
    for (uint32_t d = 0; d < t.n_dims; ++d) printf("%s%u", d ? "," : "", t.dims[d]);
    printf("] elems=%u size=%u fmt=%d type=%d\n", t.n_elems, t.size, (int)t.fmt, (int)t.type);
  }
}

// ---------------------------------------------------------------------------
// 布局归一化：原生布局 -> 逻辑 NCHW
// ---------------------------------------------------------------------------
bool rknn2_to_logical_nchw(const TensorAttr2& attr, const unsigned char* src,
                           std::vector<unsigned char>* dst) {
  const size_t elem = (attr.type == RKNN_TENSOR_FLOAT32) ? 4 : 2;

  if (attr.fmt == RKNN_TENSOR_NCHW) {
    // 已经是逻辑 NCHW ⇒ 零拷贝
    dst->assign(src, src + (size_t)attr.n_elems * elem);
    return true;
  }

  if (attr.fmt == RKNN_TENSOR_NHWC && attr.n_dims == 4) {
    // [n,h,w,c] -> [n,c,h,w]
    const uint32_t n = attr.dims[0], h = attr.dims[1], w = attr.dims[2], c = attr.dims[3];
    dst->assign((size_t)attr.n_elems * elem, 0);
    for (uint32_t ni = 0; ni < n; ++ni)
      for (uint32_t ci = 0; ci < c; ++ci)
        for (uint32_t hi = 0; hi < h; ++hi)
          for (uint32_t wi = 0; wi < w; ++wi) {
            size_t si = ((size_t)ni * h * w * c + (size_t)hi * w * c + (size_t)wi * c + ci);
            size_t di = ((size_t)ni * c * h * w + (size_t)ci * h * w + (size_t)hi * w + wi);
            memcpy(dst->data() + di * elem, src + si * elem, elem);
          }
    return true;
  }

  if (attr.fmt == RKNN_TENSOR_NC1HWC2) {
    // 原生 [n][c1][h][w][c2] -> 逻辑 [n][c1*c2][h][w]（4D）
    if (attr.n_dims != 5) return false;
    const uint32_t n = attr.dims[0], c1 = attr.dims[1], h = attr.dims[2],
                   w = attr.dims[3], c2 = attr.dims[4];
    const uint32_t c = c1 * c2;
    dst->assign((size_t)attr.n_elems * elem, 0);
    for (uint32_t ni = 0; ni < n; ++ni)
      for (uint32_t ci = 0; ci < c1; ++ci)
        for (uint32_t hi = 0; hi < h; ++hi)
          for (uint32_t wi = 0; wi < w; ++wi)
            for (uint32_t cj = 0; cj < c2; ++cj) {
              size_t si = ((size_t)ni * c1 * h * w * c2 + (size_t)ci * h * w * c2 +
                           (size_t)hi * w * c2 + (size_t)wi * c2 + cj);
              size_t di = ((size_t)ni * c * h * w + (size_t)(ci * c2 + cj) * h * w +
                           (size_t)hi * w + wi);
              memcpy(dst->data() + di * elem, src + si * elem, elem);
            }
    return true;
  }

  if (attr.fmt == RKNN_TENSOR_UNDEFINED) {
    // ⚠️ **UNDEFINED = runtime 给的就是逻辑布局**（零拷贝）。
    //
    // 什么时候会看到它：VAE **编码器**（`vae_qwenimage21-512.encoder.*.rknn`）
    // 的输出张量 `latents` 的 attr.fmt 就是 UNDEFINED（3），而解码器的
    // `image` 是 NCHW（0）。
    //
    // 「UNDEFINED 是不是逻辑布局」不是猜的 —— 板端实测过：同一张图，
    // 编码器输出按 [1,64,1,32,32]（NCDHW）解释，与 119 的 PyTorch 金标准
    // rel_L2 = **0.0071**（fp16 精度）⇒ 顺序就是 NCDHW，零拷贝是对的。
    //
    // ⚠️ 走这条分支时**只支持 4D/5D**（NCHW / NCDHW）—— 更高维的模型
    //    本来也不该走 rknn2 这条路（VAE 是 4D/5D）。
    if (attr.n_dims != 4 && attr.n_dims != 5) return false;
    dst->assign(src, src + (size_t)attr.n_elems * elem);
    return true;
  }

  // 未知布局 ⇒ 原样返回（调用方负责）
  dst->assign(src, src + (size_t)attr.n_elems * elem);
  return false;
}

}  // namespace qi
