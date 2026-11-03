#include "rknn3_runner.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <time.h>
#include <ctime>
#include <algorithm>
#include <map>
#include <vector>

namespace qi {

namespace {

double now_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

std::string join_path(const std::string& dir, const std::string& name) {
  if (dir.empty()) return name;
  if (dir.back() == '/') return dir + name;
  return dir + "/" + name;
}

// ---------------------------------------------------------------------------
// 布局转换
// ---------------------------------------------------------------------------
// 本程序内部统一用 **ONNX 的 NCHW** 作为标准布局（参考文件、分段模型之间
// 传的数据都是 NCHW）。RKNN 的 NPU 原生是 channels-last，device mem 里的
// 4D 张量可能是 NHWC，所以边界上要按运行时自己报出来的 attr.layout 转一次。
//
// 注意：不能靠猜。RKNN simulator 的 inference() 默认 data_format='nhwc'，
// 但它内部会把用户数据转成模型布局（喂 NCHW 或 NHWC 结果一样），
// 所以 simulator 说明不了 device mem 到底是什么布局 —— 只有 attr.layout 能说明。
//
// 转置方向：
//   输入：NCHW(标准) -> NHWC(模型)
//   输出：NHWC(模型) -> NCHW(标准)

inline bool needs_nhwc_transpose(const rknn3_tensor_attr& a) {
  return a.n_dims == 4 && a.layout == RKNN3_TENSOR_NHWC;
}

// 5D 的 NC1HWC2：runtime 把 4D NCHW 的 C 维拆成 C1 组 x C2 通道
// （C = C1 * C2，物理顺序 [n][c1][h][w][c2]）。
// device mem 里放的是**原生**布局，我们手上的数据是 NCHW，必须自己重排。
inline bool needs_nc1hwc2_convert(const rknn3_tensor_attr& a) {
  return a.n_dims == 5 && a.layout == RKNN3_TENSOR_NC1HWC2;
}

// NC1HWC2 的通道归属是不是「交错」的（默认否；QI_NC1HWC2_SWAP=1 打开）
inline bool nc1hwc2_swap_order() {
  const char* e = getenv("QI_NC1HWC2_SWAP");
  return e && e[0] == '1';
}

// 从 attr.shape 取 NCHW 意义的 (n,c,h,w)。
//   layout==NHWC 时 attr.shape 已经是 (n,h,w,c)，要换过来。
void nchw_dims(const rknn3_tensor_attr& a, uint32_t* n, uint32_t* c, uint32_t* h,
               uint32_t* w) {
  if (a.n_dims != 4) {
    *n = *c = *h = *w = 0;
    return;
  }
  if (a.layout == RKNN3_TENSOR_NHWC) {
    *n = a.shape[0];
    *h = a.shape[1];
    *w = a.shape[2];
    *c = a.shape[3];
  } else {
    *n = a.shape[0];
    *c = a.shape[1];
    *h = a.shape[2];
    *w = a.shape[3];
  }
}

const char* layout_name(rknn3_tensor_layout l) {
  switch (l) {
    case RKNN3_TENSOR_NCHW: return "NCHW";
    case RKNN3_TENSOR_NHWC: return "NHWC";
    case RKNN3_TENSOR_NC1HWC2: return "NC1HWC2";
    case RKNN3_TENSOR_UNDEFINED: return "UNDEFINED";
    default: return "other";
  }
}

}  // namespace

// ---- ModelRunner 的静态成员（定义放在匿名 namespace 之后，方便 selftest 调用）----

bool ModelRunner::needs_nhwc_transpose(const rknn3_tensor_attr& a) {
  return qi::needs_nhwc_transpose(a);
}

bool ModelRunner::needs_nc1hwc2_convert(const rknn3_tensor_attr& a) {
  return qi::needs_nc1hwc2_convert(a);
}

// NCHW (n,c,h,w) -> NHWC (n,h,w,c)，按元素大小 esz 搬运
void ModelRunner::transpose_nchw_nhwc(const unsigned char* src, unsigned char* dst, uint32_t n,
                                     uint32_t c, uint32_t h, uint32_t w, size_t esz) {
  for (uint32_t i = 0; i < n; ++i) {
    for (uint32_t cc = 0; cc < c; ++cc) {
      for (uint32_t hh = 0; hh < h; ++hh) {
        for (uint32_t ww = 0; ww < w; ++ww) {
          size_t si = (((size_t)i * c + cc) * h + hh) * w + ww;
          size_t di = (((size_t)i * h + hh) * w + ww) * c + cc;
          memcpy(dst + di * esz, src + si * esz, esz);
        }
      }
    }
  }
}

// NCHW (n,c,h,w) -> NC1HWC2 (n,c1,h,w,c2)，其中 c = c1*c2 + c2
//
// ⚠️ 关于通道归属：NC1HWC2 是 Rockchip 经典 **NCHW16c** 布局的推广 ——
//    物理顺序 [n][c1][h][w][c2]，最内层 C2 个通道是**连续**的，
//    即通道 c 落在 (c1 = c / C2, c2 = c % C2)。
//    这是按 NCHW16c 推出来的（板端 C2 正好是 16 = NCHW16c 的块大小）。
//    如果板端实测端到端 rel_L2 不对（> 0.5，像是通道被打乱），
//    设环境变量 **QI_NC1HWC2_SWAP=1** 可以换成「交错」归属
//    (c1 = c % C1, c2 = c / C1) 再试，不用重新编译。
void ModelRunner::transpose_nchw_nc1hwc2(const unsigned char* src, unsigned char* dst,
                                         uint32_t n, uint32_t c1, uint32_t h, uint32_t w,
                                         uint32_t c2, size_t esz) {
  const bool sw = nc1hwc2_swap_order();
  const uint32_t c = c1 * c2;
  for (uint32_t i = 0; i < n; ++i) {
    for (uint32_t cc = 0; cc < c; ++cc) {
      const uint32_t c1i = sw ? (cc % c1) : (cc / c2);
      const uint32_t c2i = sw ? (cc / c1) : (cc % c2);
      for (uint32_t hh = 0; hh < h; ++hh) {
        for (uint32_t ww = 0; ww < w; ++ww) {
          size_t si = (((size_t)i * c + cc) * h + hh) * w + ww;
          size_t di = ((((size_t)i * c1 + c1i) * h + hh) * w + ww) * c2 + c2i;
          memcpy(dst + di * esz, src + si * esz, esz);
        }
      }
    }
  }
}

// NC1HWC2 (n,c1,h,w,c2) -> NCHW (n,c,h,w)（上面那个的逆）
void ModelRunner::transpose_nc1hwc2_nchw(const unsigned char* src, unsigned char* dst,
                                         uint32_t n, uint32_t c1, uint32_t h, uint32_t w,
                                         uint32_t c2, size_t esz) {
  const bool sw = nc1hwc2_swap_order();
  const uint32_t c = c1 * c2;
  for (uint32_t i = 0; i < n; ++i) {
    for (uint32_t cc = 0; cc < c; ++cc) {
      const uint32_t c1i = sw ? (cc % c1) : (cc / c2);
      const uint32_t c2i = sw ? (cc / c1) : (cc % c2);
      for (uint32_t hh = 0; hh < h; ++hh) {
        for (uint32_t ww = 0; ww < w; ++ww) {
          size_t si = ((((size_t)i * c1 + c1i) * h + hh) * w + ww) * c2 + c2i;
          size_t di = (((size_t)i * c + cc) * h + hh) * w + ww;
          memcpy(dst + di * esz, src + si * esz, esz);
        }
      }
    }
  }
}

// --- IEEE-754 binary16 <-> binary32 的位运算辅助 -------------------------
// 注意：SDK 自带的 float16.h 里 fp32_to_fp16() 是 **截断**（frac >> 13），
// 不是 IEEE 要求的 round-to-nearest-even，和 numpy 的 astype(np.float16)
// 会有最多 1 ULP 的差异；它的 fp16_to_fp32() 也没处理 **次正规数**。
// 这里自己实现正确版本，保证板端算出来的数和 numpy / PC 端完全一致。
//
// 依赖 float16 是位域 struct {frac:10, exp:5, sign:1} 且小端存放
// （bit0-9=frac, bit10-14=exp, bit15=sign）——这正是 IEEE fp16 的内存布局。
// rk1820 是 aarch64 小端，成立。

inline uint32_t bits_of(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof(u));
  return u;
}
inline float float_from_bits(uint32_t u) {
  float f;
  memcpy(&f, &u, sizeof(f));
  return f;
}
inline uint16_t bits_of_f16(const float16& h) {
  uint16_t u;
  memcpy(&u, &h, sizeof(u));
  return u;
}
inline float16 f16_from_bits(uint16_t u) {
  float16 h;
  memcpy(&h, &u, sizeof(h));
  return h;
}

// float -> fp16，round-to-nearest-even（与 numpy astype(np.float16) 一致）
float16 fp32_to_fp16_rne(float f) {
  uint32_t x = bits_of(f);
  uint16_t sign = (uint16_t)((x >> 16) & 0x8000u);
  uint32_t raw_exp = (x >> 23) & 0xFFu;
  uint32_t mant = x & 0x7FFFFFu;

  if (raw_exp == 0xFFu) {                       // Inf / NaN
    if (mant) {
      uint16_t payload = (uint16_t)(mant >> 13);
      if (payload == 0) payload = 1;            // NaN 的 payload 不能全 0
      return f16_from_bits((uint16_t)(sign | 0x7C00u | payload));
    }
    return f16_from_bits((uint16_t)(sign | 0x7C00u));
  }

  int32_t exp = (int32_t)raw_exp - 127 + 15;   // fp16 的指数
  if (exp >= 0x1F) return f16_from_bits((uint16_t)(sign | 0x7C00u));  // 溢出 -> Inf

  if (exp <= 0) {                              // 次正规数 / 0
    if (exp < -10) return f16_from_bits(sign);  // 太小 -> 0
    mant |= 0x800000u;                         // 补隐含位
    int shift = 14 - exp;                      // 需要右移的位数 (1..24)
    uint32_t r = mant >> shift;
    uint32_t rem = mant & ((1u << shift) - 1u);
    uint32_t halfway = 1u << (shift - 1);
    if (rem > halfway || (rem == halfway && (r & 1u))) r++;
    return f16_from_bits((uint16_t)(sign | r));
  }

  uint32_t r = mant >> 13;
  uint32_t rem = mant & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) {
    r++;
    if (r == 0x400u) {                         // 进位到下一档指数
      r = 0;
      exp++;
      if (exp >= 0x1F) return f16_from_bits((uint16_t)(sign | 0x7C00u));
    }
  }
  return f16_from_bits((uint16_t)(sign | ((uint32_t)exp << 10) | r));
}

// fp16 -> float，正确处理次正规数（float16.h 的版本在次正规数上是错的）
float fp16_to_fp32_exact(float16 h) {
  uint16_t bits = bits_of_f16(h);
  uint32_t sign = (uint32_t)(bits & 0x8000u) << 16;
  uint32_t exp = (bits >> 10) & 0x1Fu;
  uint32_t mant = bits & 0x3FFu;

  if (exp == 0) {                              // 0 或次正规数
    if (mant == 0) return float_from_bits(sign);
    // 次正规数: value = mant * 2^-24
    int shift = 0;
    while (!(mant & 0x400u)) {                 // 规格化
      mant <<= 1;
      shift++;
    }
    mant &= 0x3FFu;
    // 次正规数: value = mant * 2^-24
    // 左移 shift 位后 mant' = 2^10 + frac, 于是
    //   value = (1 + frac/2^10) * 2^(-14-shift)
    // 即 e - 127 = -14 - shift  =>  e = 127 - 14 - shift
    uint32_t e = (uint32_t)(127 - 14 - shift);
    return float_from_bits(sign | (e << 23) | (mant << 13));
  }
  if (exp == 0x1F) {                           // Inf / NaN
    if (mant) return float_from_bits(sign | 0x7F800000u | (mant << 13));
    return float_from_bits(sign | 0x7F800000u);
  }
  uint32_t e = exp - 15 + 127;
  return float_from_bits(sign | (e << 23) | (mant << 13));
}

// ---------------------------------------------------------------------------
// dtype 工具
// ---------------------------------------------------------------------------

size_t ModelRunner::dtype_size(rknn3_tensor_type t) {
  switch (t) {
    case RKNN3_TENSOR_FLOAT32: return 4;
    case RKNN3_TENSOR_FLOAT16: return 2;
    case RKNN3_TENSOR_INT8:
    case RKNN3_TENSOR_UINT8:
    case RKNN3_TENSOR_BOOL:
    case RKNN3_TENSOR_INT4:
    case RKNN3_TENSOR_FLOAT8E4M3FN:
    case RKNN3_TENSOR_FLOAT8E8M0:
    case RKNN3_TENSOR_FLOAT4E2M1: return 1;
    case RKNN3_TENSOR_INT16:
    case RKNN3_TENSOR_UINT16:
    case RKNN3_TENSOR_BFLOAT16: return 2;
    case RKNN3_TENSOR_INT32:
    case RKNN3_TENSOR_UINT32: return 4;
    case RKNN3_TENSOR_INT64:
    case RKNN3_TENSOR_UINT64: return 8;
    default: return 0;
  }
}

const char* ModelRunner::dtype_name(rknn3_tensor_type t) {
  switch (t) {
    case RKNN3_TENSOR_FLOAT32: return "float32";
    case RKNN3_TENSOR_FLOAT16: return "float16";
    case RKNN3_TENSOR_INT8: return "int8";
    case RKNN3_TENSOR_UINT8: return "uint8";
    case RKNN3_TENSOR_INT16: return "int16";
    case RKNN3_TENSOR_UINT16: return "uint16";
    case RKNN3_TENSOR_INT32: return "int32";
    case RKNN3_TENSOR_UINT32: return "uint32";
    case RKNN3_TENSOR_INT64: return "int64";
    case RKNN3_TENSOR_UINT64: return "uint64";
    case RKNN3_TENSOR_BOOL: return "bool";
    case RKNN3_TENSOR_INT4: return "int4";
    case RKNN3_TENSOR_FLOAT8E4M3FN: return "float8e4m3fn";
    case RKNN3_TENSOR_BFLOAT16: return "bfloat16";
    default: return "unknown";
  }
}

rknn3_tensor_type ModelRunner::dtype_from_name(const std::string& s) {
  std::string t;
  for (char c : s) t += (char)tolower((unsigned char)c);
  if (t == "float32" || t == "fp32" || t == "f32") return RKNN3_TENSOR_FLOAT32;
  if (t == "float16" || t == "fp16" || t == "f16") return RKNN3_TENSOR_FLOAT16;
  if (t == "bool" || t == "boolean" || t == "uint8" || t == "u8") return RKNN3_TENSOR_UINT8;
  if (t == "int8" || t == "i8") return RKNN3_TENSOR_INT8;
  if (t == "int16" || t == "i16") return RKNN3_TENSOR_INT16;
  if (t == "int32" || t == "i32") return RKNN3_TENSOR_INT32;
  if (t == "uint32" || t == "u32") return RKNN3_TENSOR_UINT32;
  if (t == "int64" || t == "i64") return RKNN3_TENSOR_INT64;
  if (t == "uint64" || t == "u64") return RKNN3_TENSOR_UINT64;
  return RKNN3_TENSOR_FLOAT32;
}

bool ModelRunner::is_fake_name(const std::string& name) {
  // 多 task 模型里 task 之间的内部同步张量：core0_fake_in / core3_fake_out ...
  return name.rfind("core", 0) == 0 && name.find("_fake_") != std::string::npos;
}

bool ModelRunner::convert(const void* src, rknn3_tensor_type src_dtype, void* dst,
                          rknn3_tensor_type dst_dtype, uint32_t n) {
  if (src_dtype == dst_dtype) {
    size_t sz = dtype_size(src_dtype);
    if (sz == 0) return false;
    memcpy(dst, src, sz * n);
    return true;
  }
  // float32 -> float16  (round-to-nearest-even, 与 numpy astype(np.float16) 一致)
  if (src_dtype == RKNN3_TENSOR_FLOAT32 && dst_dtype == RKNN3_TENSOR_FLOAT16) {
    const float* s = (const float*)src;
    float16* d = (float16*)dst;
    for (uint32_t i = 0; i < n; ++i) d[i] = fp32_to_fp16_rne(s[i]);
    return true;
  }
  // float16 -> float32  (正确处理次正规数)
  if (src_dtype == RKNN3_TENSOR_FLOAT16 && dst_dtype == RKNN3_TENSOR_FLOAT32) {
    const float16* s = (const float16*)src;
    float* d = (float*)dst;
    for (uint32_t i = 0; i < n; ++i) d[i] = fp16_to_fp32_exact(s[i]);
    return true;
  }
  // float32 -> bool
  if (src_dtype == RKNN3_TENSOR_FLOAT32 && dst_dtype == RKNN3_TENSOR_BOOL) {
    const float* s = (const float*)src;
    uint8_t* d = (uint8_t*)dst;
    for (uint32_t i = 0; i < n; ++i) d[i] = (s[i] != 0.0f) ? 1 : 0;
    return true;
  }
  // uint8 -> bool
  if (src_dtype == RKNN3_TENSOR_UINT8 && dst_dtype == RKNN3_TENSOR_BOOL) {
    memcpy(dst, src, (size_t)n);
    return true;
  }
  // float32 -> int32 / int64
  if (src_dtype == RKNN3_TENSOR_FLOAT32 && dst_dtype == RKNN3_TENSOR_INT32) {
    const float* s = (const float*)src;
    int32_t* d = (int32_t*)dst;
    for (uint32_t i = 0; i < n; ++i) d[i] = (int32_t)s[i];
    return true;
  }
  if (src_dtype == RKNN3_TENSOR_FLOAT32 && dst_dtype == RKNN3_TENSOR_INT64) {
    const float* s = (const float*)src;
    int64_t* d = (int64_t*)dst;
    for (uint32_t i = 0; i < n; ++i) d[i] = (int64_t)s[i];
    return true;
  }
  // int64 -> int32
  if (src_dtype == RKNN3_TENSOR_INT64 && dst_dtype == RKNN3_TENSOR_INT32) {
    const int64_t* s = (const int64_t*)src;
    int32_t* d = (int32_t*)dst;
    for (uint32_t i = 0; i < n; ++i) d[i] = (int32_t)s[i];
    return true;
  }
  // int32 -> int64
  if (src_dtype == RKNN3_TENSOR_INT32 && dst_dtype == RKNN3_TENSOR_INT64) {
    const int32_t* s = (const int32_t*)src;
    int64_t* d = (int64_t*)dst;
    for (uint32_t i = 0; i < n; ++i) d[i] = (int64_t)s[i];
    return true;
  }
  printf("[runner] 不支持的 dtype 转换: %s -> %s\n", dtype_name(src_dtype),
         dtype_name(dst_dtype));
  return false;
}

// ---------------------------------------------------------------------------
// 文件 IO
// ---------------------------------------------------------------------------

bool read_file(const std::string& path, void* buf, size_t size) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return false;
  size_t got = fread(buf, 1, size, f);
  fclose(f);
  return got == size;
}

bool write_file(const std::string& path, const void* buf, size_t size) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) return false;
  size_t put = fwrite(buf, 1, size, f);
  fclose(f);
  return put == size;
}

bool read_meta(const std::string& bin_path, MetaInfo* out) {
  if (!out) return false;
  std::string meta_path = bin_path + ".meta";
  FILE* f = fopen(meta_path.c_str(), "r");
  if (!f) return false;
  char line[512];
  bool found = false;
  while (fgets(line, sizeof(line), f)) {
    // 跳过注释和空行
    const char* p = line;
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == '#' || *p == '\n' || *p == '\r' || *p == 0) continue;
    char dtype_str[64] = {0};
    char shape_str[256] = {0};
    int n = sscanf(p, "%63s %255s", dtype_str, shape_str);
    if (n >= 1) {
      out->dtype = ModelRunner::dtype_from_name(dtype_str);
      found = true;
    }
    if (n >= 2) {
      out->shape.clear();
      const char* s = shape_str;
      while (*s) {
        char* end = nullptr;
        long v = strtol(s, &end, 10);
        if (end == s) break;
        out->shape.push_back((uint32_t)v);
        s = end;
        while (*s == ',' || *s == 'x' || *s == ' ') ++s;
      }
    }
    break;
  }
  fclose(f);
  return found;
}

// ---------------------------------------------------------------------------
// ⭐⭐⭐ 2026-10-31：IO 缓冲**进程级均衡核分配**（按节点池空闲量）
//
// 坑 1：计数器原本是 **ModelRunner 的成员** ⇒ 每个段的计数器都从 0 开始
//      ⇒ 4 段的 33 MiB 输入缓冲**全落在 core0**（4 × 33 = 132 MiB）
//      ⇒ core0 池被 IO 吃穿 ⇒ 共享 internal 根本装不下（实测缺 144 MiB）。
//
// 坑 2（2026-10-31 第二十一轮）：只按「已分字节最少」挑核**还不够** ——
//      16-block 模型的「峰值核」本身就只剩 ~6 MiB 余量，
//      而新起的 runner 计数从 0 开始 ⇒ 第一个 IO 缓冲（33 MiB）正好落在峰值核上
//      ⇒ create_mem 失败。
//      ⇒ 改成**按节点池空闲量**挑：model_init 之后查一次
//         rknn3_query(RKNN3_QUERY_DEVICE_MEM_INFO) 拿到每核空闲，
//         每次挑「空闲最多」的核 ⇒ 自动避开峰值核。
//
// ⚠️ 跨卡不共享计数（mem 绑定 device）⇒ 按 device_id 分桶。
// ---------------------------------------------------------------------------
namespace {

std::map<std::string, std::vector<uint64_t>> g_io_used;     // 每 device 每核已分字节
std::map<std::string, std::vector<uint64_t>> g_pool_free;   // 每 device 每核池空闲（model_init 后快照）
// ⭐⭐⭐ 2026-11-02：每 device 每核**预留**字节（共享 internal 块等「还没分配但一定要有」的空间）。
//   为什么需要：共享 internal 是在**所有段的 IO 缓冲之后**才分配的
//   （resident_chain.cpp 第 4 步），而 pick_io_core 选的是「池空闲最多」的核
//   ⇒ 池最大的 core0 会被优先塞满 IO，等共享块要分配时 core0 已经没空间了
//   ⇒ 单卡方案（4 段共享 internal）必然在 core0 上先炸。
//   ⇒ 解法：把共享块的每核需求**提前登记成预留**，pick_io_core 扣掉它再挑核。
std::map<std::string, std::vector<uint64_t>> g_io_reserve;

// 记录 model_init 之后的池空闲量（供 pick_io_core 用）
void set_pool_free(const std::string& dev, const std::vector<uint64_t>& free_bytes) {
  g_pool_free[dev] = free_bytes;
}

// 挑一个 IO 缓冲该放哪：优先「池空闲最多（扣掉预留）」的核，其次「已分字节最少」的核
int pick_io_core(const std::string& dev, int core_num, uint64_t sz) {
  if (core_num <= 0) core_num = 8;
  std::vector<uint64_t>& v = g_io_used[dev];
  if ((int)v.size() < core_num) v.resize(core_num, 0);
  std::vector<uint64_t>& pf = g_pool_free[dev];
  std::vector<uint64_t>& rsv = g_io_reserve[dev];
  int best = 0;
  int64_t best_score = 0;
  bool have_pool = (int)pf.size() >= core_num;
  for (int c = 0; c < core_num; ++c) {
    int64_t score;
    int64_t r = (c < (int)rsv.size()) ? (int64_t)rsv[c] : 0;
    if (have_pool) {
      score = (int64_t)pf[c] - (int64_t)v[c] - r;   // 还剩多少（扣掉已分 + 预留）
    } else {
      score = -(int64_t)v[c] - r;                    // 没有池信息 ⇒ 选已用最少的
    }
    if (c == 0 || score > best_score) { best = c; best_score = score; }
  }
  v[best] += sz;
  return best;
}

// ⛔⭐⭐⭐⭐⭐ IO 分散**已禁用**（2026-10-31 11:30 实测发现是致命 bug）
//
//   症状：`QI_SPREAD_IO=1` 时，**所有模型的输出恒为全 0**
//         （rknn3_run 返回成功、耗时正常、输出缓冲一个非零元素都没有）。
//   根因：`rknn3_create_mem(ctx, size, core_id, ...)` 的 core_id 参数
//         **不是「你想放哪核就放哪核」**，而是「runtime 为这个 IO 张量
//         分配的节点池 ID」。runtime 只往**它自己指定的那个池**写结果；
//         我们另挑一个核（pick_io_core）⇒ runtime 写的是池 A，
//         我们读的是池 B ⇒ 永远读到全 0。
//   证据（1024 w8a16 现役模型 seg8fac_0，历史上产出 corr 0.938）：
//         QI_SPREAD_IO=1 → rms=0.00000, nonzero=0
//         QI_SPREAD_IO=0 → rms=30.21015, nonzero=17300584  ✅
//   ⇒ IO 缓冲**必须**用 runtime 报告的 slot.attr.core_id。
//   ⇒ 代价：33 MiB 的 joint stream 会全压在 runtime 指定的核上
//     （通常 core0）。这会影响「每卡能常驻几段」的内存账，
//     但正确性优先 —— 宁可慢，不能错。
//
//   保留环境变量仅作**诊断开关**：QI_SPREAD_IO=1 会重新打开这个坏行为，
//   默认（不设 / 设为 0）一律走安全的 runtime 原生 core_id。
bool spread_io_enabled() {
  const char* e = getenv("QI_SPREAD_IO");
  if (!e || !e[0]) return false;      // ⭐ 默认关闭
  return !(e[0] == '0');
}

// ---------------------------------------------------------------------------
// ⭐⭐⭐ 只分散「**输入**」IO（QI_SPREAD_IO_IN=1 开启，默认关闭）
//
// 背景（debug.md §73.1）：
//   runtime 把每段的输入/输出张量都指定到 `slot.attr.core_id`（实测 = core 0），
//   而 core 0 恰好是「热核」（承载该模型最大的 internal）。
//   ⇒ 真机硬约束 = **core0 的 internal + 该段 IO 合计 ≤ 639.5 MiB**。
//   反例：vae512_f2m 静态 516.4 + IO 144 = 660.4 > 639.5 ⇒ create_mem 失败。
//
// 为什么「只分散输入」是安全的（与 §65 根因 B 不矛盾）：
//   §65 的结论是 **runtime 只往它自己指定的那个池写结果** ⇒ **输出**不能换池
//   （换池 ⇒ runtime 写池 A、我们读池 B ⇒ 输出恒为全 0）。
//   而 **输入** 是**我们自己** mem_sync 写进去的，runtime 只是把它当作
//   输入张量的数据源 ⇒ 把输入缓冲放到别的池，runtime 照样从我们给的指针读。
//   ⚠️ 本条**必须数值验证**：跑完拿 simulator 参考对拍，输出全 0 就说明不成立。
// ---------------------------------------------------------------------------
bool spread_io_input_enabled() {
  const char* e = getenv("QI_SPREAD_IO_IN");
  if (!e || !e[0]) return false;      // ⭐ 默认关闭
  return !(e[0] == '0');
}

}  // namespace

// ---------------------------------------------------------------------------
// ⭐⭐⭐ 2026-11-02：登记「每核预留字节」（公开接口）
//
// 用途：共享 internal 块是在所有段的 IO 缓冲**之后**才分配的，
//       而 pick_io_core 会优先挑「池空闲最多」的核（= core0）
//       ⇒ 共享块还没分配时，IO 就把 core0 塞满了 ⇒ 共享块分配必失败。
//       ⇒ 调用方（ResidentChain）在 init 各段**之前**先登记预留，
//          pick_io_core 挑核时就会自动避开这些核。
// ---------------------------------------------------------------------------
void set_io_reserve(const std::string& dev, const std::vector<uint64_t>& bytes) {
  g_io_reserve[dev] = bytes;
  if ((int)g_io_reserve[dev].size() > 8) g_io_reserve[dev].resize(8);
}


// ---------------------------------------------------------------------------
// ModelRunner
// ---------------------------------------------------------------------------
ModelRunner::~ModelRunner() { release(); }

int ModelRunner::init(const std::string& model_path, const std::string& weight_path,
                      uint32_t core_mask, bool user_mem_internal, const char* device_id) {
  int ret = 0;
  rknn3_config config;
  uint32_t core_num = 0;

  // 多卡时可以用 RKNN3_DEVICE_ID 环境变量指定设备（和 model zoo 官方示例一致）
  // ⚠️ 2026-10-30 实测：**librknn3_api.so 自己不读这个环境变量**，
  //    必须自己填 rknn3_init_extend.device_id 才会真的选到那张卡。
  //    （init_extend.device_id 为空时库会打印 "use the first device"。）
  rknn3_init_extend init_extend;
  memset(&init_extend, 0, sizeof(init_extend));
  const char* dev_id = device_id;
  if ((!dev_id || !dev_id[0])) dev_id = getenv("RKNN3_DEVICE_ID");
  printf("[runner] init: device_id=%s\n", (dev_id && dev_id[0]) ? dev_id : "(默认 device[0])");
  fflush(stdout);
  if (dev_id && dev_id[0]) {
    // ⭐⭐⭐ device_id 和 init_extend 都存进 runner 的**持久成员**：
    //   runtime 会长期持有 init_extend.device_id 这个指针，
    //   之前它们分别是「调用方的临时字符串」和「init() 的栈变量」⇒ 悬空 ⇒ SIGSEGV。
    device_id_ = dev_id;
    init_extend_ = init_extend;
    init_extend_.device_id = (char*)device_id_.c_str();
    printf("[runner] 使用设备 %s\n", device_id_.c_str());
    ret = rknn3_init(&ctx_, &init_extend_);
  } else {
    ret = rknn3_init(&ctx_, nullptr);
  }
  if (ret != RKNN3_SUCCESS) {
    printf("[runner] rknn3_init 失败 ret=%d\n", ret);
    return ret;
  }
  // ⭐ 计时（2026-10-30 加）：「时间花在哪」一目了然。
  //   40 步 flow-match 每步 ~112 s，而 8 段**纯推理**只有 ~25 s
  //   ⇒ 剩下 ~88 s 花在「load + init + destroy」上，
  //   而 `rknn3_load_model_from_path` 是**通过 RCPC 把权重传给卡端**（8.3 GB/步）。
  struct timespec t_load0, t_load1, t_init0, t_init1;
  clock_gettime(CLOCK_MONOTONIC, &t_load0);
  ret = rknn3_load_model_from_path(ctx_, model_path.c_str(),
                                   weight_path.empty() ? nullptr : weight_path.c_str());
  clock_gettime(CLOCK_MONOTONIC, &t_load1);
  if (ret != RKNN3_SUCCESS) {
    printf("[runner] rknn3_load_model_from_path 失败 ret=%d  model=%s weight=%s\n", ret,
           model_path.c_str(), weight_path.c_str());
    release();
    return ret;
  }
  double load_ms = (t_load1.tv_sec - t_load0.tv_sec) * 1000.0 +
                   (t_load1.tv_nsec - t_load0.tv_nsec) / 1e6;

  memset(&config, 0, sizeof(config));
  config.priority = 0;
  config.run_timeout = 0;      // 不超时
  config.user_mem_internal = user_mem_internal ? 1 : 0;

  ret = rknn3_query(ctx_, RKNN3_QUERY_CORE_NUMBER, &core_num, sizeof(core_num));
  if (ret == RKNN3_SUCCESS && core_num > 0) {
    // ★ 关键: core_mask 的 popcount 必须等于模型的 task 数。
    //   0xffffffff 对任何模型都不合法(rk1828 上是 8 核 -> 0xff)。
    //   板端实测: 掩码不对时直接报 "core_mask ffffffff is not match with npu core number N"。
    //   所以默认(0)按模型实际的核数自动推导; 用户显式指定时才用用户的值。
    uint32_t mask = core_mask;
    if (mask == 0) {
        mask = (core_num >= 32) ? 0xffffffffu : ((1u << core_num) - 1u);
    }
    printf("[runner] NPU core 数 = %u, run_core_mask = 0x%x%s\n", core_num, mask,
           core_mask ? "" : " (自动推导)");
    config.run_core_mask = mask;
  } else {
    printf("[runner] 查询 core 数失败 ret=%d（用 run_core_mask=0x%x 继续）\n", ret, core_mask);
    config.run_core_mask = core_mask;
  }

  clock_gettime(CLOCK_MONOTONIC, &t_init0);
  ret = rknn3_model_init(ctx_, &config);
  clock_gettime(CLOCK_MONOTONIC, &t_init1);
  if (ret != RKNN3_SUCCESS) {
    printf("[runner] rknn3_model_init 失败 ret=%d\n", ret);
    release();
    return ret;
  }
  double init_ms = (t_init1.tv_sec - t_init0.tv_sec) * 1000.0 +
                   (t_init1.tv_nsec - t_init0.tv_nsec) / 1e6;
  printf("[runner] ⏱ load(权重传输) %.1f ms | model_init %.1f ms | 合计 %.1f ms\n",
         load_ms, init_ms, load_ms + init_ms);

  // ⭐⭐⭐ 2026-10-31：记下 model_init 之后的**每核池空闲量**，
  //   后面分配 IO 缓冲时按「空闲最多」挑核 ⇒ 自动避开模型的峰值核。
  //   （16-block 模型峰值核只剩 ~6 MiB，IO 缓冲若落在那里必失败。）
  {
    rknn3_dev_mem_info mi;
    memset(&mi, 0, sizeof(mi));
    if (rknn3_query(ctx_, RKNN3_QUERY_DEVICE_MEM_INFO, &mi, sizeof(mi)) == RKNN3_SUCCESS &&
        mi.node_num > 0) {
      std::vector<uint64_t> pf(mi.node_num, 0);
      for (uint32_t n = 0; n < mi.node_num && n < RKNN3_MAX_NPU_NODE_NUM; n++) {
        pf[n] = mi.node_mem_info[n].free;
      }
      set_pool_free(device_id_, pf);
      printf("[runner] 池空闲/核:");
      for (uint32_t n = 0; n < mi.node_num && n < 8; n++) {
        printf(" %6.1f", pf[n] / 1048576.0);
      }
      printf(" MiB\n");
    }
  }

  rknn3_input_output_num io_num;
  memset(&io_num, 0, sizeof(io_num));
  ret = rknn3_query(ctx_, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN3_SUCCESS) {
    printf("[runner] rknn3_query(IN_OUT_NUM) 失败 ret=%d\n", ret);
    release();
    return ret;
  }
  printf("[runner] 输入 %u 个, 输出 %u 个\n", io_num.n_input, io_num.n_output);

  in_.resize(io_num.n_input);
  out_.resize(io_num.n_output);

  for (uint32_t i = 0; i < io_num.n_input; ++i) {
    TensorSlot& slot = in_[i];
    slot.attr.index = i;
    ret = rknn3_query(ctx_, RKNN3_QUERY_INPUT_ATTR, &slot.attr, sizeof(slot.attr));
    if (ret != RKNN3_SUCCESS) {
      printf("[runner] 查询输入[%u] 属性失败 ret=%d\n", i, ret);
      release();
      return ret;
    }
    slot.fake = is_fake_name(slot.attr.name);
    // ⭐⭐⭐ 2026-10-31：IO 缓冲**核分散**同样适用于**输入**。
    //   段间张量 hidden_states / joint_stream 各 33 MiB，runtime 默认全放 core_id=0。
    //   每卡常驻 4 段时 ⇒ core0 要吃 4×(33+33) = 264 MiB 的 IO ⇒ 直接撑爆池子。
    //   `rknn3_create_mem` 的 core_id 由**调用方**指定 ⇒ 可以摊到 8 核。
    int32_t in_core = slot.attr.core_id;
    if (in_core >= 0 && (spread_io_enabled() || spread_io_input_enabled())) {
      in_core = pick_io_core(device_id_, core_num, slot.attr.aligned_size);
    }
    printf("[runner] 输入[%u] %-28s %8.2f MiB  核 %d%s\n", i, slot.attr.name,
           slot.attr.aligned_size / 1048576.0, in_core,
           (in_core != slot.attr.core_id) ? "（已分散）" : "");
    fflush(stdout);
    slot.mem = rknn3_create_mem(ctx_, slot.attr.aligned_size, in_core,
                                RKNN3_FLAG_MEMORY_CACHEABLE);
    if (!slot.mem) {
      printf("[runner] rknn3_create_mem 输入[%u] (%s, %llu B) 失败\n", i, slot.attr.name,
             (unsigned long long)slot.attr.aligned_size);
      release();
      return -1;
    }
  }

  for (uint32_t i = 0; i < io_num.n_output; ++i) {
    TensorSlot& slot = out_[i];
    slot.attr.index = i;
    ret = rknn3_query(ctx_, RKNN3_QUERY_OUTPUT_ATTR, &slot.attr, sizeof(slot.attr));
    if (ret != RKNN3_SUCCESS) {
      printf("[runner] 查询输出[%u] 属性失败 ret=%d\n", i, ret);
      release();
      return ret;
    }
    slot.fake = is_fake_name(slot.attr.name);
    // ⭐⭐⭐ 2026-10-31：IO 缓冲**核分散**（可选，QI_SPREAD_IO=1 开启）
    //   段间张量 joint stream 是 (1,4224,4096) fp16 = **33 MiB**，
    //   而 runtime 把它们全放在 core_id=0 ⇒ core0 池被吃掉 33 MiB/段。
    //   `rknn3_create_mem` 的 core_id 是**调用方指定的** ⇒ 可以摊到 8 个核。
    //   ⇒ 每核只占 33/8 ≈ 4.1 MiB ⇒ 大幅放宽「每卡能常驻几段」的门槛。
    int32_t out_core = slot.attr.core_id;
    if (spread_io_enabled() && out_core >= 0) {
      out_core = pick_io_core(device_id_, core_num, slot.attr.aligned_size);
    }
    printf("[runner] 输出[%u] %-28s %8.2f MiB  核 %d%s\n", i, slot.attr.name,
           slot.attr.aligned_size / 1048576.0, out_core,
           (out_core != slot.attr.core_id) ? "（已分散）" : "");
    fflush(stdout);
    slot.mem = rknn3_create_mem(ctx_, slot.attr.aligned_size, out_core,
                                RKNN3_FLAG_MEMORY_CACHEABLE);
    if (!slot.mem) {
      printf("[runner] rknn3_create_mem 输出[%u] (%s, %llu B) 失败\n", i, slot.attr.name,
             (unsigned long long)slot.attr.aligned_size);
      release();
      return -1;
    }
  }

  return RKNN3_SUCCESS;
}

// ---------------------------------------------------------------------------
// 注入用户自管的 internal 缓冲
//
// ⚠️ 调用顺序：**必须在 init(user_mem_internal=1) 之后**。
//    先 set 再 model_init 会被卡端拒（ACK_INVALD_ARGUMENT）。
//    本轮已穷举 6 种顺序（scripts/probe_setmem_order.cpp）验证。
// ---------------------------------------------------------------------------
int ModelRunner::set_shared_internal_mem(std::vector<rknn3_tensor_mem*>* mems) {
  if (!mems || mems->empty()) {
    printf("[runner] set_shared_internal_mem: mem 列表为空\n");
    return -1;
  }
  int ret = rknn3_set_internal_mem(ctx_, mems->data(), (uint32_t)mems->size());
  printf("[runner] set_shared_internal_mem(%zu 核) ret=%d\n", mems->size(), ret);
  return ret;
}

// ---------------------------------------------------------------------------
// 查询每核真实占用（ALLOCATION_INFO 必须传数组）
// ---------------------------------------------------------------------------
bool ModelRunner::query_core_alloc(std::vector<uint64_t>* cmd, std::vector<uint64_t>* weight,
                                   std::vector<uint64_t>* internal,
                                   std::vector<uint64_t>* kvcache) {
  int32_t core_num = 0;
  if (rknn3_query(ctx_, RKNN3_QUERY_CORE_NUMBER, &core_num, sizeof(core_num)) !=
          RKNN3_SUCCESS ||
      core_num <= 0) {
    return false;
  }
  // ⚠️ 必须按**数组**查：size = sizeof(struct) × core_num。
  //    传单个 struct 会报 "size = 456, expect 3648!"。
  std::vector<rknn3_allocation_info> ais(core_num);
  memset(ais.data(), 0, ais.size() * sizeof(rknn3_allocation_info));
  for (int c = 0; c < core_num; ++c) ais[c].core_id = c;
  int ret = rknn3_query(ctx_, RKNN3_QUERY_ALLOCATION_INFO, ais.data(),
                        ais.size() * sizeof(rknn3_allocation_info));
  if (ret != RKNN3_SUCCESS) {
    printf("[runner] query_core_alloc 失败 ret=%d\n", ret);
    return false;
  }
  if (cmd) cmd->resize(core_num);
  if (weight) weight->resize(core_num);
  if (internal) internal->resize(core_num);
  if (kvcache) kvcache->resize(core_num);
  for (int c = 0; c < core_num; ++c) {
    if (cmd) (*cmd)[c] = ais[c].command_mem.size;
    if (weight) (*weight)[c] = ais[c].weight_mem.size;
    if (internal) (*internal)[c] = ais[c].internal_mem.size;
    if (kvcache) (*kvcache)[c] = ais[c].kvcache_mem.size;
  }
  return true;
}

void ModelRunner::release() {
  for (auto& s : in_) {
    if (s.mem) {
      rknn3_destroy_mem(ctx_, s.mem);
      s.mem = nullptr;
    }
  }
  for (auto& s : out_) {
    if (s.mem) {
      rknn3_destroy_mem(ctx_, s.mem);
      s.mem = nullptr;
    }
  }
  in_.clear();
  out_.clear();
  if (ctx_ != 0) {
    rknn3_destroy(ctx_);
    ctx_ = 0;
  }
}

void ModelRunner::print_info() const {
  rknn3_sdk_version ver;
  memset(&ver, 0, sizeof(ver));
  if (rknn3_query(ctx_, RKNN3_QUERY_SDK_VERSION, &ver, sizeof(ver)) == RKNN3_SUCCESS) {
    printf("SDK api 版本 : %s\n", ver.api_version);
    printf("SDK 驱动版本 : %s\n", ver.drv_version);
  }

  rknn3_devices devs;
  memset(&devs, 0, sizeof(devs));
  if (rknn3_find_devices(&devs) == RKNN3_SUCCESS) {
    printf("发现 %u 个 RKNN 设备:\n", devs.n_devices);
    for (uint32_t i = 0; i < devs.n_devices && i < RKNN3_MAX_DEVS; ++i) {
      printf("  [%u] id=%s type=%s\n", i, devs.devices[i].id, devs.devices[i].type);
    }
  } else {
    printf("[warn] rknn3_find_devices 失败\n");
  }

  printf("\n输入张量:\n");
  for (const auto& s : in_) {
    printf("  [%u] %-32s %-8s shape=[", s.attr.index, s.attr.name,
           rknn3_get_type_string(s.attr.dtype));
    for (uint32_t d = 0; d < s.attr.n_dims; ++d) {
      printf("%s%u", d ? "," : "", s.attr.shape[d]);
    }
    printf("]  layout=%s  n_elems=%u  aligned=%llu B  core=%d%s\n",
           layout_name(s.attr.layout), s.attr.n_elems,
           (unsigned long long)s.attr.aligned_size, s.attr.core_id,
           s.fake ? "  [内部 glue 张量]" : "");
  }
  printf("\n输出张量:\n");
  for (const auto& s : out_) {
    printf("  [%u] %-32s %-8s shape=[", s.attr.index, s.attr.name,
           rknn3_get_type_string(s.attr.dtype));
    for (uint32_t d = 0; d < s.attr.n_dims; ++d) {
      printf("%s%u", d ? "," : "", s.attr.shape[d]);
    }
    printf("]  layout=%s  n_elems=%u  aligned=%llu B  core=%d%s\n",
           layout_name(s.attr.layout), s.attr.n_elems,
           (unsigned long long)s.attr.aligned_size, s.attr.core_id,
           s.fake ? "  [内部 glue 张量]" : "");
  }
}

int ModelRunner::load_inputs(const std::string& dir, bool verbose,
                              const std::map<std::string, std::vector<unsigned char>>*
                                  overrides,
                              const std::set<std::string>* raw_names) {
  std::vector<unsigned char> host_buf;
  std::vector<unsigned char> f32_buf;
  std::vector<unsigned char> tmp_buf;

  for (auto& slot : in_) {
    const std::string name = slot.attr.name;

    // 调用方要求「原样写入」的张量：这里跳过，由 set_raw_input 负责
    // （例：rope_cos_cache 的 NC1HWC2 标记和通用转换规则对不上）
    if (raw_names && raw_names->count(name)) {
      if (verbose)
        printf("[runner] 输入 %-30s <- 跳过（调用方用 set_raw_input 原样写入）\n", name.c_str());
      continue;
    }

    if (slot.fake) {
      // 内部 glue 张量（coreN_fake_in）：runtime 自己会用，这里清零保险
      memset(slot.mem->virt_addr, 0, (size_t)slot.attr.aligned_size);
      if (verbose) printf("[runner] 输入 %s 是内部 glue 张量, 已清零\n", name.c_str());
      continue;
    }

    // ---- 1. 取源数据（overrides 优先，其次目录里的 .bin）----
    const unsigned char* src = nullptr;
    size_t src_bytes = 0;
    rknn3_tensor_type src_dtype = slot.attr.dtype;
    std::vector<unsigned char> owned;

    bool has_override = false;
    if (overrides) {
      auto oit = overrides->find(name);
      if (oit != overrides->end()) {
        owned = oit->second;
        src = owned.data();
        src_bytes = owned.size();
        src_dtype = slot.attr.dtype;   // overrides 已是该张量的原生 dtype
        has_override = true;
      }
    }
    if (!has_override) {
      std::string path = join_path(dir, name + ".bin");
      FILE* f = fopen(path.c_str(), "rb");
      if (f) {
        // 大小和「原生 dtype」对不上 -> 退回 .f32.bin
        // （参考文件可能是 fp16 而模型 IO 是 fp32，或者反过来）
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        size_t need0 = dtype_size(slot.attr.dtype) * slot.attr.n_elems;
        if (sz != (long)need0) {
          fclose(f);
          f = nullptr;
          path = join_path(dir, name + ".f32.bin");
          f = fopen(path.c_str(), "rb");
          src_dtype = RKNN3_TENSOR_FLOAT32;
        }
      } else {
        path = join_path(dir, name + ".f32.bin");
        f = fopen(path.c_str(), "rb");
        src_dtype = RKNN3_TENSOR_FLOAT32;
      }
      if (!f) {
        // ⭐⭐⭐ 兜底：**carry 里只有一个张量时，直接拿它当这个输入**。
        //   合并模型（merge_dit_blocks.py）给每段输出起了不同的名字
        //   （joint_stream_m8a / m8b / m16a …），跨模型衔接时
        //   「上一段的输出名」与「下一段的输入名」可能对不上
        //   ⇒ 按名找不到 + 目录里也没文件 ⇒ 用唯一的 carry 张量顶上。
        if (overrides && overrides->size() == 1) {
          const auto& only = *overrides->begin();
          printf("[runner] 输入 %s 找不到文件 ⇒ 用上一段的唯一输出 %s 顶上（%zu B）\n",
                 name.c_str(), only.first.c_str(), only.second.size());
          owned = only.second;
          src = owned.data();
          src_bytes = owned.size();
          src_dtype = slot.attr.dtype;
          has_override = true;
        } else {
          printf("[runner] 找不到输入文件: %s.bin  (或 %s.f32.bin)\n", name.c_str(),
                 name.c_str());
          printf("[runner] 当前模型需要这些输入:\n");
          for (const auto& s : in_) {
            if (!s.fake) printf("           %s (%s)\n", s.attr.name,
                                rknn3_get_type_string(s.attr.dtype));
          }
          return -1;
        }
      } else {
        size_t esz = dtype_size(src_dtype);
        size_t need = esz * slot.attr.n_elems;
        host_buf.resize(need);
        size_t got = fread(host_buf.data(), 1, need, f);
        fclose(f);
        if (got != need) {
          printf("[runner] 读 %s 失败: 期望 %zu B, 实际 %zu B\n", path.c_str(), need, got);
          return -1;
        }
        src = host_buf.data();
        src_bytes = need;
      }
    }

    size_t esz = dtype_size(src_dtype);
    if (esz == 0) {
      printf("[runner] 输入 %s 的 dtype 未知\n", name.c_str());
      return -1;
    }
    size_t need = esz * slot.attr.n_elems;
    if (src_bytes < need) {
      printf("[runner] 输入 %s 数据不足: %zu < %zu\n", name.c_str(), src_bytes, need);
      return -1;
    }

    // ---- 2. 布局转换（模型要 NHWC / NC1HWC2 而我们是 NCHW 时）----
    const unsigned char* final_src = src;
    rknn3_tensor_type final_dtype = src_dtype;
    const unsigned char* transpose_src = nullptr;

    if (needs_nhwc_transpose(slot.attr)) {
      uint32_t n, c, h, w;
      nchw_dims(slot.attr, &n, &c, &h, &w);
      if (n == 0) {
        printf("[runner] 输入 %s 维度不是 4D, 无法做布局转换\n", name.c_str());
        return -1;
      }
      // 源数据先转 float32（如果已经是 float32 就直接用）
      const unsigned char* f32 = nullptr;
      if (src_dtype == RKNN3_TENSOR_FLOAT32) {
        f32 = src;
      } else {
        f32_buf.resize((size_t)slot.attr.n_elems * sizeof(float));
        if (!convert(src, src_dtype, f32_buf.data(), RKNN3_TENSOR_FLOAT32,
                     slot.attr.n_elems)) {
          printf("[runner] 输入 %s 转 float32 失败\n", name.c_str());
          return -1;
        }
        f32 = f32_buf.data();
      }
      tmp_buf.resize((size_t)slot.attr.n_elems * sizeof(float));
      ModelRunner::transpose_nchw_nhwc(f32, tmp_buf.data(), n, c, h, w, sizeof(float));
      transpose_src = tmp_buf.data();
      final_dtype = RKNN3_TENSOR_FLOAT32;
      final_src = transpose_src;
    } else if (needs_nc1hwc2_convert(slot.attr)) {
      // 5D NC1HWC2：device mem 里是 [n][c1][h][w][c2] 的**原生**顺序，
      // 我们手上的数据是 NCHW，必须把通道维拆成 c1 组 x c2 重排。
      // （runtime 的 rknn_inputs_set 里有 normalize 步骤会做这个转换，
      //   但我们走的是「直接写 device mem + mem_sync」这条路，runtime 不管，
      //   所以必须自己做 —— 4D NHWC 的转置同理，见上面的 transpose_nchw_nhwc。）
      const uint32_t n = slot.attr.shape[0];
      const uint32_t c1 = slot.attr.shape[1];
      const uint32_t h = slot.attr.shape[2];
      const uint32_t w = slot.attr.shape[3];
      const uint32_t c2 = slot.attr.shape[4];
      size_t esz = dtype_size(src_dtype);
      if (esz == 0) {
        printf("[runner] 输入 %s 的 dtype 未知\n", name.c_str());
        return -1;
      }
      tmp_buf.resize(need);
      ModelRunner::transpose_nchw_nc1hwc2(src, tmp_buf.data(), n, c1, h, w, c2, esz);
      transpose_src = tmp_buf.data();
      final_dtype = src_dtype;
      final_src = transpose_src;
    }

    // ---- 3. 写进 device memory ----
    if (final_src == src && final_dtype == slot.attr.dtype) {
      memcpy(slot.mem->virt_addr, src, need);
    } else {
      if (!convert(final_src, final_dtype, slot.mem->virt_addr, slot.attr.dtype,
                   slot.attr.n_elems)) {
        printf("[runner] 输入 %s dtype 转换失败\n", name.c_str());
        return -1;
      }
    }

    int ret = rknn3_mem_sync(ctx_, slot.mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
    if (ret != RKNN3_SUCCESS) {
      printf("[runner] rknn3_mem_sync(TO_DEVICE) 输入[%s] 失败 ret=%d\n", name.c_str(),
             ret);
      return ret;
    }
    if (verbose) {
      printf("[runner] 输入 %-30s <- %s (%s, %u 元素, 布局 %s%s%s)\n", name.c_str(),
             has_override ? "override" : name.c_str(),
             rknn3_get_type_string(slot.attr.dtype), slot.attr.n_elems,
             layout_name(slot.attr.layout),
             needs_nhwc_transpose(slot.attr) ? " -> 已转成 NHWC" : "",
             needs_nc1hwc2_convert(slot.attr) ? " -> 已转成 NC1HWC2" : "");
    }
  }
  return RKNN3_SUCCESS;
}

int ModelRunner::run(double* out_ms) {
  std::vector<rknn3_tensor> in_tensors(in_.size());
  std::vector<rknn3_tensor> out_tensors(out_.size());
  for (size_t i = 0; i < in_.size(); ++i) {
    in_tensors[i].mem = in_[i].mem;
    in_tensors[i].attr = &in_[i].attr;
  }
  for (size_t i = 0; i < out_.size(); ++i) {
    out_tensors[i].mem = out_[i].mem;
    out_tensors[i].attr = &out_[i].attr;
  }

  double t0 = now_ms();
  int ret = rknn3_run(ctx_, in_tensors.data(), (uint32_t)in_tensors.size(),
                      out_tensors.data(), (uint32_t)out_tensors.size());
  double t1 = now_ms();
  if (ret != RKNN3_SUCCESS) {
    printf("[runner] rknn3_run 失败 ret=%d\n", ret);
    return ret;
  }
  if (out_ms) *out_ms = t1 - t0;

  for (auto& slot : out_) {
    ret = rknn3_mem_sync(ctx_, slot.mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
    if (ret != RKNN3_SUCCESS) {
      printf("[runner] rknn3_mem_sync(FROM_DEVICE) 输出[%s] 失败 ret=%d\n",
             slot.attr.name, ret);
      return ret;
    }
  }
  return RKNN3_SUCCESS;
}

// ---- 官方调试接口 ---------------------------------------------------------
// 这些接口用来定位「模型 init 成功但推理跑不完」的模型（VAE 的 preb/band 就是）。
// rknn3_profile_ops 会逐算子执行并打印耗时/cycles —— 如果在某个算子上挂住，
// 那个算子就是嫌疑对象。

int ModelRunner::profile_ops(uint32_t log_level) {
  std::vector<rknn3_tensor> in_tensors(in_.size());
  std::vector<rknn3_tensor> out_tensors(out_.size());
  for (size_t i = 0; i < in_.size(); ++i) {
    in_tensors[i].mem = in_[i].mem;
    in_tensors[i].attr = &in_[i].attr;
  }
  for (size_t i = 0; i < out_.size(); ++i) {
    out_tensors[i].mem = out_[i].mem;
    out_tensors[i].attr = &out_[i].attr;
  }
  printf("[runner] rknn3_profile_ops(log_level=%u) ...\n", log_level);
  fflush(stdout);
  int ret = rknn3_profile_ops(ctx_, in_tensors.data(), (uint32_t)in_tensors.size(),
                               out_tensors.data(), (uint32_t)out_tensors.size(),
                               log_level);
  printf("[runner] rknn3_profile_ops 返回 ret=%d\n", ret);
  fflush(stdout);
  return ret;
}

int ModelRunner::profile_mem() {
  printf("[runner] rknn3_profile_mem() ...\n");
  fflush(stdout);
  int ret = rknn3_profile_mem(ctx_);
  printf("[runner] rknn3_profile_mem 返回 ret=%d\n", ret);
  fflush(stdout);
  return ret;
}

int ModelRunner::dump_features(const std::string& dir, uint32_t core_mask) {
  std::vector<rknn3_tensor> in_tensors(in_.size());
  for (size_t i = 0; i < in_.size(); ++i) {
    in_tensors[i].mem = in_[i].mem;
    in_tensors[i].attr = &in_[i].attr;
  }
  printf("[runner] rknn3_dump_features(dir=%s, core_mask=0x%x) ...\n", dir.c_str(),
         core_mask);
  fflush(stdout);
  int ret = rknn3_dump_features(ctx_, in_tensors.data(), (uint32_t)in_tensors.size(),
                                nullptr, 0, dir.c_str(), core_mask, "*");
  printf("[runner] rknn3_dump_features 返回 ret=%d\n", ret);
  fflush(stdout);
  return ret;
}

bool ModelRunner::set_raw_input(const std::string& name, const void* data, size_t bytes) {
  for (auto& slot : in_) {
    if (slot.fake || slot.attr.name != name) continue;
    if (!slot.mem || !slot.mem->virt_addr) return false;
    size_t cap = (size_t)slot.attr.aligned_size;
    if (bytes > cap) {
      printf("[runner] set_raw_input(%s): 数据 %zu B > aligned_size %zu B\n", name.c_str(), bytes,
             cap);
      return false;
    }
    memset(slot.mem->virt_addr, 0, cap);
    memcpy(slot.mem->virt_addr, data, bytes);
    int ret = rknn3_mem_sync(ctx_, slot.mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
    if (ret != RKNN3_SUCCESS) {
      printf("[runner] set_raw_input(%s): mem_sync(TO_DEVICE) 失败 ret=%d\n", name.c_str(), ret);
      return false;
    }
    return true;
  }
  return false;
}

bool ModelRunner::get_output_bytes(const std::string& name,
                                   std::vector<unsigned char>* out) const {
  if (!out) return false;
  for (const auto& slot : out_) {
    if (slot.fake || slot.attr.name != name) continue;
    size_t esz = dtype_size(slot.attr.dtype);
    if (esz == 0) return false;
    size_t valid = esz * slot.attr.n_elems;

    // 模型内部是 NHWC 时，对外统一给 NCHW（我们的标准布局）
    if (needs_nhwc_transpose(slot.attr)) {
      uint32_t n, c, h, w;
      nchw_dims(slot.attr, &n, &c, &h, &w);
      if (n == 0) return false;
      std::vector<unsigned char> tmp(valid);
      ModelRunner::transpose_nchw_nhwc((const unsigned char*)slot.mem->virt_addr, tmp.data(), n, c, h,
                          w, esz);
      out->swap(tmp);
      return true;
    }
    // 5D NC1HWC2 输出同理：原生 [n][c1][h][w][c2] -> NCHW [n][c][h][w]
    if (needs_nc1hwc2_convert(slot.attr)) {
      const uint32_t n = slot.attr.shape[0];
      const uint32_t c1 = slot.attr.shape[1];
      const uint32_t h = slot.attr.shape[2];
      const uint32_t w = slot.attr.shape[3];
      const uint32_t c2 = slot.attr.shape[4];
      std::vector<unsigned char> tmp(valid);
      ModelRunner::transpose_nc1hwc2_nchw((const unsigned char*)slot.mem->virt_addr, tmp.data(), n, c1,
                             h, w, c2, esz);
      out->swap(tmp);
      return true;
    }
    out->assign((const unsigned char*)slot.mem->virt_addr,
                (const unsigned char*)slot.mem->virt_addr + valid);
    return true;
  }
  return false;
}

int ModelRunner::save_outputs(const std::string& prefix) const {
  for (const auto& slot : out_) {
    if (slot.fake) {
      printf("[runner] 跳过内部 glue 输出 %s\n", slot.attr.name);
      continue;
    }
    size_t esz = dtype_size(slot.attr.dtype);
    if (esz == 0) {
      printf("[runner] 输出 %s 的 dtype 未知, 跳过\n", slot.attr.name);
      continue;
    }
    // 防御性检查: 本程序按 **连续内存** 写出（和 model zoo 官方示例一致）。
    // stride 若是 -1（layout=UNDEFINED 时就是这样），说明厂商没给出，
    // 按连续处理；只有在 stride 明确已知且不连续时才告警，
    // 免得默默给错顺序的数据。
    {
      bool stride_known = true;
      uint64_t expect = 1;
      bool contiguous = true;
      for (uint32_t d = 0; d < slot.attr.n_dims; ++d) {
        uint32_t k = slot.attr.n_dims - 1 - d;   // 从最后一维往前
        uint64_t s = (k < slot.attr.n_stride) ? slot.attr.stride[k] : 0;
        if (s == 0 || s == (uint64_t)-1) {
          stride_known = false;
          break;
        }
        if (s != expect) {
          contiguous = false;
          break;
        }
        expect *= slot.attr.shape[k];
      }
      if (stride_known && !contiguous) {
        printf("[runner] [警告] 输出 %s 的 stride 不是连续的, 写出的 .bin 元素顺序可能不对\n",
               slot.attr.name);
      }
    }
    // aligned_size 可能大于 n_elems * esz（对齐填充），只写有效部分
    size_t valid = esz * slot.attr.n_elems;
    std::string path = prefix + "_" + slot.attr.name + ".bin";
    std::vector<unsigned char> bytes;
    if (!get_output_bytes(slot.attr.name, &bytes) || bytes.size() < valid) {
      // 回退：直接写 device memory
      bytes.assign((const unsigned char*)slot.mem->virt_addr,
                   (const unsigned char*)slot.mem->virt_addr + valid);
    }
    if (!write_file(path, bytes.data(), valid)) {
      printf("[runner] 写 %s 失败\n", path.c_str());
      return -1;
    }
    std::string meta_path = path + ".meta";
    FILE* f = fopen(meta_path.c_str(), "w");
    if (f) {
      fprintf(f, "# dtype shape\n");
      fprintf(f, "%s ", dtype_name(slot.attr.dtype));
      for (uint32_t d = 0; d < slot.attr.n_dims; ++d) {
        fprintf(f, "%s%u", d ? "," : "", slot.attr.shape[d]);
      }
      fprintf(f, "\n");
      fclose(f);
    }
    printf("[runner] 输出 %-30s -> %s (%zu B)\n", slot.attr.name, path.c_str(), valid);
  }
  return RKNN3_SUCCESS;
}

}  // namespace qi
