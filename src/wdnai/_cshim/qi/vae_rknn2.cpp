// ---------------------------------------------------------------------------
// VAE 解码器的 RKNN2 链实现（跑在 RK3588 本地 NPU）—— 见 vae_rknn2.h
// ---------------------------------------------------------------------------
#include "vae_rknn2.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>

#include "rknn3_runner.h"   // fp16<->fp32 精确转换
#include "vae_stats.h"

namespace qi {

namespace {

// 名字 -> 段输出缓冲
struct Carry {
  std::string name;
  std::vector<unsigned char> bytes;
  rknn_tensor_type dtype = RKNN_TENSOR_FLOAT16;
};

}  // namespace

VaeChain2::~VaeChain2() { release(); }

// ---------------------------------------------------------------------------
// token 主序 (T, C) -> 通道主序 (C, T)，并乘 latents_std / 加 latents_mean
// 与 rknn3 的 VaeChain::latent_to_nchw **逐字同口径**（数值必须一致）。
// ---------------------------------------------------------------------------
bool VaeChain2::latent_to_nchw_f32(const std::vector<unsigned char>& lat_fp16,
                                   int channels, std::vector<float>* out) {
  if (channels <= 0 || channels > 64) return false;
  size_t n = lat_fp16.size() / 2;
  if (n == 0 || n % (size_t)channels != 0) return false;
  size_t tokens = n / (size_t)channels;
  size_t side = (size_t)std::lround(std::sqrt((double)tokens));
  if (side * side != tokens) return false;

  std::vector<float> tmp(n);
  if (!ModelRunner::convert(lat_fp16.data(), RKNN3_TENSOR_FLOAT16, (void*)tmp.data(),
                            RKNN3_TENSOR_FLOAT32, (uint32_t)n)) {
    return false;
  }
  out->assign(n, 0.0f);
  // token 主序 (T, C) -> 通道主序 (C, T)：out[c*T + t]
  for (size_t t = 0; t < tokens; ++t) {
    for (int c = 0; c < channels; ++c) {
      float v = tmp[t * (size_t)channels + (size_t)c];
      (*out)[(size_t)c * tokens + t] = v * kVaeLatentsStd[c] + kVaeLatentsMean[c];
    }
  }
  return true;
}

int VaeChain2::init_piece(size_t i) {
  int ret = runners_[i]->init(models_[i], core_mask_);
  if (ret != 0) {
    char buf[256];
    snprintf(buf, sizeof(buf), "第 %zu 段 init 失败：%s", i, runners_[i]->error().c_str());
    err_ = buf;
  }
  return ret;
}

void VaeChain2::release_piece(size_t i) {
  if (i < runners_.size() && runners_[i]) runners_[i]->release();
}

// ---------------------------------------------------------------------------
int VaeChain2::init(const std::vector<std::string>& models, uint32_t core_mask,
                     bool prefer_resident) {
  release();
  if (models.empty()) {
    err_ = "VAE(rknn2) 链为空";
    return -1;
  }
  const Rknn2Api& api = rknn2_api();
  if (!api.ok) {
    err_ = api.err;
    return -1;
  }
  models_ = models;
  core_mask_ = core_mask ? core_mask : (uint32_t)RKNN_NPU_CORE_0_1_2;
  runners_.clear();
  for (size_t i = 0; i < models_.size(); ++i) {
    runners_.push_back(std::unique_ptr<ModelRunner2>(new ModelRunner2()));
  }
  inited_ = true;

  if (prefer_resident) {
    for (size_t i = 0; i < runners_.size(); ++i) {
      if (init_piece(i) != 0) {
        // ⭐⭐ 要分清两种「装不下」：
        //
        //   a) **整图**（只有 1 个模型）：装不下就是**根本装不下**。
        //      「退化成流式」在这里毫无意义 —— 流式只是把 init 推迟到
        //      解码那一刻，而那一刻**同样装不下**。
        //      ⇒ 必须返回失败，让 api.cpp 退回 rknn3 的 VAE 链。
        //      （踩过的坑：模型包里放了个装不下的 fp16 整图，
        //      init 误报成功 ⇒ 生成时在解码阶段炸 "第 0 段 init 失败"，
        //      整条出图链路断掉。）
        //
        //   b) **分段**（多个模型）：每段单独装得下、但全体装不下 ——
        //      这时退化成流式（逐段 init/release）是对的。
        if (runners_.size() <= 1) {
          printf("[vae-rknn2] ❌ 整图装不下（%s）⇒ 交回调用方（退回 rknn3 VAE 链）\n",
                 err_.c_str());
          release();
          return -1;
        }
        printf("[vae-rknn2] 常驻失败（%s）⇒ 退化成流式（逐段 init/release）\n",
               err_.c_str());
        for (size_t k = 0; k < runners_.size(); ++k) release_piece(k);
        resident_ = false;
        err_.clear();
        return 0;   // 分段流式也算成功
      }
    }
    resident_ = true;
    printf("[vae-rknn2] %zu 段全部常驻（生成过程零 load/init）\n", runners_.size());
  } else {
    resident_ = false;
  }
  return 0;
}

void VaeChain2::release() {
  for (size_t i = 0; i < runners_.size(); ++i) release_piece(i);
  runners_.clear();
  models_.clear();
  resident_ = false;
  inited_ = false;
}

// ---------------------------------------------------------------------------
int VaeChain2::decode(const std::vector<unsigned char>& lat_fp16, int channels,
                      std::vector<unsigned char>* out_rgb, int* out_w, int* out_h,
                      double* out_ms) {
  if (!inited_ || runners_.empty()) {
    err_ = "VAE(rknn2) 链没初始化";
    return -1;
  }
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  // ---- 1. 潜变量：token 主序 -> 通道主序 + 乘 std/mean（float32）----
  std::vector<float> nchw;
  if (!latent_to_nchw_f32(lat_fp16, channels, &nchw)) {
    err_ = "潜变量布局/大小不对（token 主序 -> 通道主序 失败）";
    return -1;
  }

  // ---- 2. 逐段跑；段间按**张量名**对齐 ----
  std::vector<Carry> carry;   // 上一段的输出

  for (size_t i = 0; i < runners_.size(); ++i) {
    ModelRunner2& m = *(runners_[i]);

    // 流式模式下：必须**先 init 本段**才能拿到输入张量的名字/形状
    if (!resident_ && !m.ready()) {
      if (init_piece(i) != 0) return -1;
    }

    // ---- 3. 准备输入 ----
    for (const auto& s : m.inputs()) {
      const std::string& nm = s.name;

      // 先找同名 carry
      const Carry* hit = nullptr;
      for (const auto& c : carry) {
        if (c.name == nm) { hit = &c; break; }
      }
      // 兜底：carry 只有 1 个且本段只有一个真输入 ⇒ 直接顶上
      if (!hit && carry.size() == 1 && m.inputs().size() == 1) hit = &carry[0];

      if (!hit) {
        // 第 0 段的输入就是潜变量（名字按模型实际输入名来，首个输入）
        if (i == 0 && &s == &m.inputs()[0]) {
          int ret = m.set_input(nm, nchw.data(), nchw.size() * sizeof(float),
                                RKNN_TENSOR_FLOAT32);
          if (ret != 0) { err_ = "喂潜变量失败：" + m.error(); return -1; }
        }
        continue;
      }

      int ret = m.set_input(nm, hit->bytes.data(), hit->bytes.size(), hit->dtype);
      if (ret != 0) { err_ = "喂段间张量失败：" + m.error(); return -1; }
    }

    // ---- 4. 跑 ----
    double ms = 0.0;
    int ret = m.run(&ms);
    if (ret != 0) {
      err_ = "第 " + std::to_string(i) + " 段推理失败：" + m.error();
      return -1;
    }
    printf("[vae-rknn2] 第 %zu/%zu 段推理完成，耗时 %.1f ms\n", i + 1, runners_.size(), ms);
    qi_progress_emit(progress_ctx_, "vae", (int)i + 1, (int)runners_.size());

    // ---- 5. 取输出，作为下一段的输入 ----
    carry.clear();
    for (const auto& s : m.outputs()) {
      std::vector<unsigned char> bytes;
      rknn_tensor_type dt = RKNN_TENSOR_FLOAT16;
      if (!m.get_output(s.name, &bytes, &dt)) continue;
      Carry c;
      c.name = s.name;
      c.bytes.swap(bytes);
      c.dtype = dt;
      carry.push_back(std::move(c));
    }

    // ---- 6. 最后一段：取出图像 ----
    if (i + 1 == runners_.size()) {
      const Carry* img = nullptr;
      for (const auto& c : carry) {
        if (c.name == "image") { img = &c; break; }
      }
      if (!img) {
        // 兜底：4 维、通道 3/4、空间 >= 256 的那个
        for (const auto& s : m.outputs()) {
          if (s.n_dims != 4) continue;
          uint32_t c = s.dims[1], hh = s.dims[2], ww = s.dims[3];
          if ((c == 3 || c == 4) && hh >= 256 && ww >= 256) {
            for (const auto& cc : carry) {
              if (cc.name == s.name) { img = &cc; break; }
            }
            break;
          }
        }
      }
      if (!img) { err_ = "最后一段没有图像输出"; return -1; }

      // 输出是 [1, C, H, W]（逻辑 NCHW）⇒ 取前 3 通道归一化成 uint8 RGB
      //
      // ⭐⭐ **dtype 必须看实际拿到的值，不能写死 fp16**：
      //    * fp16 模型（`vae_qwenimage21-512.rknn2.rknn` 早期版本）→ fp16
      //    * **量化模型（w8a8）→ runtime 用 want_float=1 反量化成 fp32**
      //      （不置 want_float 拿到的是裸 int8，当 fp16 读直接是垃圾）
      //    写死 fp16 会把 fp32 的字节两两当成一个 fp16 读 ⇒ **元素数翻倍**
      //    ⇒ 后面 `c*h*w != nelem` 触发「按元素数推」的兜底分支
      //    ⇒ 整张图是错位的垃圾（而且不报错，静默出坏图）。
      const bool out_f32 = (img->dtype == RKNN_TENSOR_FLOAT32);
      const size_t nelem = img->bytes.size() / (out_f32 ? 4 : 2);
      std::vector<float> f(nelem);
      if (out_f32) {
        memcpy(f.data(), img->bytes.data(), nelem * sizeof(float));
      } else {
        for (size_t k = 0; k < nelem; ++k) {
          float16 hv;
          memcpy(&hv, img->bytes.data() + k * sizeof(uint16_t), sizeof(uint16_t));
          f[k] = fp16_to_fp32_exact(hv);
        }
      }
      // 从该段的输出属性拿形状
      uint32_t c = 4, h = 512, w = 512;
      for (const auto& s : m.outputs()) {
        if (s.name != img->name) continue;
        if (s.n_dims >= 4) {
          c = s.dims[1];
          h = s.dims[2];
          w = s.dims[3];
        }
        break;
      }
      if ((size_t)c * h * w != nelem) {
        // 形状对不上 ⇒ 退回「按元素数推」
        if (nelem % 4 != 0) { err_ = "图像输出大小异常"; return -1; }
        h = w = 512;
        c = (uint32_t)(nelem / (h * w));
      }
      out_rgb->assign((size_t)h * w * 3, 0);
      float lo = 1e30f;
      std::vector<float> rgb((size_t)h * w * 3);
      for (uint32_t ch = 0; ch < 3 && ch < c; ++ch) {
        for (size_t p = 0; p < (size_t)h * w; ++p) {
          float v = f[(size_t)ch * h * w + p];
          rgb[p * 3 + ch] = v;
          if (v < lo) lo = v;
        }
      }
      const bool norm01 = (lo < -0.01f);
      for (size_t k = 0; k < rgb.size(); ++k) {
        float v = norm01 ? (rgb[k] + 1.0f) * 127.5f : rgb[k] * 255.0f;
        if (v < 0.0f) v = 0.0f;
        if (v > 255.0f) v = 255.0f;
        (*out_rgb)[k] = (uint8_t)v;   // 与 numpy astype(uint8) 一样是截断
      }
      if (out_w) *out_w = (int)w;
      if (out_h) *out_h = (int)h;
    }

    // ---- 7. 流式模式：立刻释放这一段 ----
    if (!resident_) release_piece(i);
  }

  clock_gettime(CLOCK_MONOTONIC, &t1);
  if (out_ms) {
    *out_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
  }
  return 0;
}

}  // namespace qi
