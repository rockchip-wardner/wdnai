// ---------------------------------------------------------------------------
// numpy 兼容的「固定初始噪声」生成器（Qwen-Image-2.1 / 512 路线）
//
// 为什么需要这个文件：
//   出图的「固定噪声」（原来随模型包发布的 reference_512/hidden_states.bin，
//   128 KB）实测是这么生成的：
//
//       np.random.default_rng(0).standard_normal(64**3)
//           .reshape(64, 64, 64)[::2, ::2, :].reshape(-1).astype(np.float16)
//
//   numpy 的标准正态用的是 **PCG64 + ziggurat**（不是 libstdc++ 的极坐标法，
//   两者连「消耗多少个随机数」都不一样），而 numpy 官方只保证
//   「同一 build 内」的流一致性 —— 跨版本可能换算法。
//   ⇒ 用 numpy 现算 ⇒ 升级 numpy 就可能换流 ⇒ 验收 md5 基线失效。
//   ⇒ 这里把 PCG64 + ziggurat **完整搬进 C** ⇒ 永久逐字节一致，且零外部依赖。
//
// ⚠️ 数值注意：
//   ① **必须关掉 FMA 收缩**（-ffp-contract=off）：慢路径判定里有
//      `(fi[i-1] - fi[i]) * u + fi[i]`，被收缩成 FMA 会与 numpy 差 1 ULP。
//      （numpy 自己编译时也没收缩，已用 262144 个 double 逐位对拍验证。）
//   ② **double → fp16 要一步到位**：先降到 float32 再到 fp16 会有
//      double rounding（≈1/2048 的元素会差 1 ULP）⇒ 65536 个里约 32 个不同。
//      所以这里直接实现 double→fp16 的 RNE。
// ---------------------------------------------------------------------------
#include "np_noise.h"

#include <cmath>
#include <cstring>

#include "np_zig_tables.h"

#if !defined(__SIZEOF_INT128__)
#error "需要 GCC/Clang 的 __uint128_t（aarch64 GCC 有）"
#endif

// ⛔⛔ 关键：禁止 FMA 收缩（见文件头 ①）
//   慢路径判定里有 `(fi[i-1] - fi[i]) * u + fi[i]`，被收缩成 FMA 会与 numpy
//   差 1 ULP ⇒ 在 margin ~6e-7 的临界样本上**判定翻转** ⇒ 整条流错位。
//   （numpy 自己编译时也没收缩 —— 已用 262144 个 double 逐位对拍验证。）
#ifdef QI_NP_ALLOW_FMA
#pragma GCC optimize("fp-contract=fast")
#else
#pragma GCC optimize("fp-contract=off")
#endif

namespace qi {
namespace {

typedef __uint128_t u128;

// 128 位常量（拆成高/低 64 位，⛔ 不能直接写成 ULL 后缀——那会被截断）
static const u128 kPcgMult = ((u128)0x2360ED051FC65DA4ULL << 64) | 0x4385DF649FCCF645ULL;
static const u128 kPcgInc = ((u128)0x418DDADB3AF71A82ULL << 64) | 0x588133BC447873A9ULL;

// ---------------------------------------------------------------------------
// PCG64（XSL-RR 128/64）—— numpy 的 Generator 默认 bit generator
//   state: 128 位；输出取高 64 位做 xorshift + rotate
// ---------------------------------------------------------------------------
struct Pcg64 {
  u128 state;
  u128 inc;

  void init(u128 s, u128 i) {
    state = s;
    inc = i;
  }

  uint64_t next64() {
    u128 old = state;
    state = old * kPcgMult + inc;
    uint64_t hi = (uint64_t)(old >> 64);
    uint64_t lo = (uint64_t)old;
    uint64_t xorshifted = hi ^ lo;
    unsigned rot = (unsigned)(hi >> 58);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 63));
  }

  // numpy 的 next_double：取 53 位
  double next_double() {
    return (double)(next64() >> 11) * (1.0 / 9007199254740992.0);
  }
};

// numpy random_standard_normal() 的逐行移植（numpy/random/src/distributions/distributions.c）
double np_standard_normal(Pcg64* bg) {
  uint64_t r;
  int sign;
  uint64_t rabs;
  int idx;
  double x, xx, yy;
  for (;;) {
    r = bg->next64();
    idx = (int)(r & 0xff);
    r >>= 8;
    sign = (int)(r & 0x1);
    rabs = (r >> 1) & 0x000fffffffffffffULL;
    x = (double)rabs * NP_ZIG_WI[idx];
    if (sign & 0x1) x = -x;
    if (rabs < NP_ZIG_KI[idx]) return x;   /* 99.3% 的情况走这里 */
    if (idx == 0) {
      for (;;) {
        /* 用 1.0 - U 避免 log(0.0)（numpy GH 13361） */
        xx = -NP_ZIG_NOR_INV_R * std::log1p(-bg->next_double());
        yy = -std::log1p(-bg->next_double());
        if (yy + yy > xx * xx) {
          return ((rabs >> 8) & 0x1) ? -(NP_ZIG_NOR_R + xx) : (NP_ZIG_NOR_R + xx);
        }
      }
    } else {
      if (((NP_ZIG_FI[idx - 1] - NP_ZIG_FI[idx]) * bg->next_double() + NP_ZIG_FI[idx]) <
          std::exp(-0.5 * x * x)) {
        return x;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// double → fp16（round-to-nearest-even）—— **一步到位**，不经 float32
//   （见文件头 ②）
// ---------------------------------------------------------------------------
uint16_t fp64_to_fp16_rne(double d) {
  uint64_t b;
  memcpy(&b, &d, sizeof(b));
  uint16_t sign = (uint16_t)((b >> 48) & 0x8000ULL);   // b 的第 63 位 = 符号位
  uint32_t raw_exp = (uint32_t)((b >> 52) & 0x7FFULL);
  uint64_t mant = b & 0x000FFFFFFFFFFFFFULL;

  if (raw_exp == 0x7FFULL) {                     // Inf / NaN
    if (mant) {
      uint16_t payload = (uint16_t)(mant >> 42);
      if (payload == 0) payload = 1;             // NaN 的 payload 不能全 0
      return (uint16_t)(sign | 0x7C00u | payload);
    }
    return (uint16_t)(sign | 0x7C00u);
  }

  int32_t exp = (int32_t)raw_exp - 1023 + 15;   // fp16 的指数
  if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);   // 溢出 -> Inf

  if (exp <= 0) {                               // 次正规数 / 0
    if (exp < -10) return sign;                 // 太小 -> ±0
    mant |= 0x0010000000000000ULL;              // 补隐含位（2^52）
    // 值 = (2^52 + mant) * 2^(exp-67)；fp16 次正规表示为 m * 2^-24
    //   ⇒ m = (2^52 + mant) * 2^(exp-43)  ⇒ 右移 (43 - exp) 位（1..53）
    int shift = 43 - exp;
    uint32_t r = (uint32_t)(mant >> shift);
    uint64_t rem = mant & ((1ULL << shift) - 1ULL);
    uint64_t halfway = 1ULL << (shift - 1);
    if (rem > halfway || (rem == halfway && (r & 1u))) r++;
    // r 进位到 1024 时，位模式正好等于「最小正常数」（指数域 1、尾数 0），
    // 所以这里不需要额外处理。
    return (uint16_t)(sign | r);
  }

  uint32_t r = (uint32_t)(mant >> 42);          // 52 位 -> 10 位
  uint64_t rem = mant & 0x000003FFFFFFFFFFULL;  // 低 42 位
  uint64_t halfway = 0x0000020000000000ULL;     // 2^41
  if (rem > halfway || (rem == halfway && (r & 1u))) {
    r++;
    if (r == 0x400u) {                          // 进位到下一档指数
      r = 0;
      exp++;
      if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    }
  }
  return (uint16_t)(sign | ((uint16_t)exp << 10) | r);
}

}  // namespace

// ---------------------------------------------------------------------------
// 生成 512 路线的固定初始噪声：1024 个 token × 64 通道（fp16）
//
//   等价于（Python）：
//       np.random.default_rng(0).standard_normal(64**3)
//           .reshape(64, 64, 64)[::2, ::2, :].reshape(-1).astype(np.float16)
//
//   ⚠️ 起始 state 是「default_rng(0) 初始化后再推进 1 次」的值
//      （numpy 的 bit_generator.state 快照与「下一次输出」差一步，
//        已实测确认：raw 序列从该 state 起算需要先跳过一个 uint64）。
// ---------------------------------------------------------------------------
bool np_fixed_noise_512(std::vector<uint16_t>* out) {
  if (!out) return false;
  const size_t kSrc = 64 * 64 * 64;   // 源数组 [64,64,64]
  const size_t kDst = 32 * 32 * 64;   // 取 [::2, ::2, :] 之后 = 65536
  out->assign(kDst, 0);

  Pcg64 bg;
  // state = default_rng(0) 初始化后再推进 1 次的值（见文件头说明）
  bg.init(((u128)0x3C535930F580265FULL << 64) | 0xC9B4B5D1B4AFF7D8ULL, kPcgInc);

  size_t dst = 0;
  for (size_t s = 0; s < kSrc; ++s) {
    double v = np_standard_normal(&bg);
    size_t a = s / (64 * 64);              // 第 0 维
    size_t b = (s % (64 * 64)) / 64;       // 第 1 维
    if ((a & 1) == 0 && (b & 1) == 0) {    // [::2, ::2, :]（第 2 维全取）
      (*out)[dst++] = fp64_to_fp16_rne(v);
    }
  }
  return dst == kDst;
}

}  // namespace qi
