// ---------------------------------------------------------------------------
// VAE decoder 分段链的实现（见 vae_chain.h 的说明）
// ---------------------------------------------------------------------------
#include "vae_chain.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>

#include "vae_stats.h"

namespace qi {

namespace {

// 写 <path>.meta（格式和 ModelRunner::save_outputs 一致）
bool write_meta(const std::string& path, rknn3_tensor_type dtype, const uint32_t* shape,
                uint32_t n_dims) {
  std::string mp = path + ".meta";
  FILE* f = fopen(mp.c_str(), "w");
  if (!f) return false;
  fprintf(f, "# dtype shape\n");
  fprintf(f, "%s ", ModelRunner::dtype_name(dtype));
  for (uint32_t d = 0; d < n_dims; ++d) fprintf(f, "%s%u", d ? "," : "", shape[d]);
  fprintf(f, "\n");
  fclose(f);
  return true;
}

// 一段链里的中间张量
struct Carry {
  std::string name;
  std::vector<unsigned char> bytes;
  rknn3_tensor_type dtype = RKNN3_TENSOR_FLOAT32;
  uint32_t elems = 0;
};

}  // namespace

VaeChain::~VaeChain() { release(); }

// ---------------------------------------------------------------------------
bool VaeChain::latent_to_nchw(const std::vector<unsigned char>& lat_fp16, int channels,
                               std::vector<float>* out) {
  if (channels <= 0) return false;
  size_t n = lat_fp16.size() / 2;
  if (n == 0 || n % (size_t)channels != 0) return false;
  size_t tokens = n / (size_t)channels;
  size_t side = (size_t)std::lround(std::sqrt((double)tokens));
  if (side * side != tokens) return false;
  if (channels > 64) return false;   // latents_std/mean 只有 64 通道

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

// ---------------------------------------------------------------------------
int VaeChain::init_piece(size_t i) {
  const char* dev = device_id_.empty() ? nullptr : device_id_.c_str();
  int ret = runners_[i]->init(models_[i], weights_[i], core_mask_, false, dev);
  if (ret != RKNN3_SUCCESS) {
    char buf[256];
    snprintf(buf, sizeof(buf), "第 %zu 段 init 失败 ret=%d（%s）", i, ret, models_[i].c_str());
    err_ = buf;
  }
  return ret;
}

void VaeChain::release_piece(size_t i) { if (runners_[i]) runners_[i]->release(); }

// ---------------------------------------------------------------------------
int VaeChain::init(const std::vector<std::string>& models,
                   const std::vector<std::string>& weights, uint32_t core_mask,
                   const char* device_id, bool prefer_resident) {
  release();
  if (models.empty()) {
    err_ = "VAE 链为空";
    return -1;
  }
  models_ = models;
  weights_.assign(models.size(), "");
  for (size_t i = 0; i < weights.size() && i < models.size(); ++i) weights_[i] = weights[i];
  core_mask_ = core_mask;
  device_id_ = device_id ? device_id : "";
  runners_.clear();
  for (size_t i = 0; i < models_.size(); ++i) {
    runners_.push_back(std::unique_ptr<ModelRunner>(new ModelRunner()));
  }
  inited_ = true;

  if (prefer_resident) {
    for (size_t i = 0; i < runners_.size(); ++i) {
      if (init_piece(i) != RKNN3_SUCCESS) {
        // 装不下 => 全部释放，退化成流式
        printf("[vae-chain] 常驻失败（%s）⇒ 退化成流式（逐段 init/release）\n", err_.c_str());
        for (size_t k = 0; k < runners_.size(); ++k) release_piece(k);
        resident_ = false;
        err_.clear();
        return RKNN3_SUCCESS;
      }
    }
    resident_ = true;
    printf("[vae-chain] %zu 段全部常驻（生成过程零 load/init）\n", runners_.size());
  } else {
    resident_ = false;
  }
  return RKNN3_SUCCESS;
}

void VaeChain::release() {
  for (size_t i = 0; i < runners_.size(); ++i) release_piece(i);
  runners_.clear();
  models_.clear();
  weights_.clear();
  resident_ = false;
  inited_ = false;
}

// ---------------------------------------------------------------------------
int VaeChain::decode(const std::vector<unsigned char>& lat_fp16, int channels,
                     const std::string& scratch_dir, const std::string& out_prefix,
                     double* out_ms) {
  if (!inited_ || runners_.empty()) {
    err_ = "VAE 链没初始化";
    return -1;
  }
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  // ---- 0. 流式模式下先把第 0 段 init 出来（要知道它的输入张量名/形状）----
  //   ⚠️ 常驻模式下 runners_[0] 早就 init 好了；流式模式必须先 init。
  bool piece0_ready = false;
  if (!resident_) {
    if (init_piece(0) != RKNN3_SUCCESS) return -1;
    piece0_ready = true;
  }

  // ---- 1. 潜变量：token 主序 -> 通道主序 + 乘 std/mean ----
  if (channels <= 0) {
    // 从第 0 段第一个真输入张量的形状里取通道数
    for (const auto& s : runners_[0]->inputs()) {
      if (s.fake) continue;
      channels = (s.attr.n_dims >= 4) ? (int)s.attr.shape[1] : 0;
      break;
    }
    if (channels <= 0) {
      err_ = "无法确定潜变量通道数";
      if (piece0_ready) release_piece(0);
      return -1;
    }
  }
  std::vector<float> nchw;
  if (!latent_to_nchw(lat_fp16, channels, &nchw)) {
    err_ = "潜变量布局/大小不对（token 主序 -> 通道主序 失败）";
    return -1;
  }

  std::vector<Carry> carry;   // 上一段的输出

  for (size_t i = 0; i < runners_.size(); ++i) {
    ModelRunner& m = *(runners_[i]);

    // ---- 2. ⭐⭐⭐ 流式模式下**必须先 init 本段** ----
    //   原因：输入张量的属性（名字/形状/dtype）要等 model_init 之后才有。
    //   顺序写反的话，第 2 段拿到的 m.inputs() 是**空列表** ⇒ carry 根本没机会填进去
    //   ⇒ load_inputs 只能去磁盘找 add_19.bin ⇒ 「找不到输入文件」。
    //   （2026-11-01 02:38 实测踩到：n0 推理成功，n1 却报「找不到输入文件: add_19.bin」）
    if (!resident_ && i > 0) {
      if (init_piece(i) != RKNN3_SUCCESS) return -1;
    }

    // ---- 3. 准备输入（用上一段的 carry）----
    std::map<std::string, std::vector<unsigned char>> ov;
    for (const auto& s : m.inputs()) {
      if (s.fake) continue;
      const std::string& nm = s.attr.name;

      // 先找同名 carry
      const Carry* hit = nullptr;
      for (const auto& c : carry) {
        if (c.name == nm) { hit = &c; break; }
      }
      // 兜底：carry 只有 1 个 && 本段只有一个真输入 ⇒ 直接顶上
      if (!hit && carry.size() == 1) {
        int n_real = 0;
        for (const auto& s2 : m.inputs()) if (!s2.fake) n_real++;
        if (n_real == 1) hit = &carry[0];
      }
      if (!hit) continue;   // 留给 load_inputs 从目录里找

      // dtype 不一致要转（override 假定数据已是该张量的原生 dtype）
      if (hit->dtype == s.attr.dtype) {
        ov[nm] = hit->bytes;
      } else {
        std::vector<unsigned char> tmp((size_t)s.attr.n_elems *
                                       ModelRunner::dtype_size(s.attr.dtype));
        if (!ModelRunner::convert(hit->bytes.data(), hit->dtype, tmp.data(),
                                  s.attr.dtype, s.attr.n_elems)) {
          err_ = "段间张量 dtype 转换失败：" + nm;
          if (!resident_) release_piece(i);
          return -1;
        }
        ov[nm] = tmp;
      }
    }

    // ---- 4. 第 0 段的输入 = 潜变量 ----
    if (i == 0) {
      // 找本段第一个真输入张量
      const rknn3_tensor_attr* in_attr = nullptr;
      std::string in_name;
      for (const auto& s : m.inputs()) {
        if (!s.fake) { in_attr = &s.attr; in_name = s.attr.name; break; }
      }
      if (!in_attr) {
        err_ = "第 0 段没有输入张量";
        if (!resident_) release_piece(i);
        return -1;
      }
      std::vector<unsigned char> tmp((size_t)in_attr->n_elems *
                                     ModelRunner::dtype_size(in_attr->dtype));
      if (!ModelRunner::convert(nchw.data(), RKNN3_TENSOR_FLOAT32, tmp.data(),
                                in_attr->dtype, in_attr->n_elems)) {
        err_ = "潜变量转 " + in_name + " 的 dtype 失败";
        if (!resident_) release_piece(i);
        return -1;
      }
      ov[in_name] = tmp;
    }

    int ret = m.load_inputs(scratch_dir, false, &ov);
    if (ret != RKNN3_SUCCESS) {
      err_ = "第 " + std::to_string(i) + " 段装载输入失败";
      if (!resident_) release_piece(i);
      return -1;
    }
    double ms = 0.0;
    ret = m.run(&ms);
    if (ret != RKNN3_SUCCESS) {
      err_ = "第 " + std::to_string(i) + " 段推理失败 ret=" + std::to_string(ret);
      if (!resident_) release_piece(i);
      return -1;
    }
    printf("[vae-chain] 第 %zu/%zu 段推理完成，耗时 %.1f ms\n", i + 1, runners_.size(), ms);

    // ---- 5. 取输出，作为下一段的输入 ----
    carry.clear();
    for (const auto& s : m.outputs()) {
      if (s.fake) continue;
      std::vector<unsigned char> bytes;
      if (!m.get_output_bytes(s.attr.name, &bytes)) continue;
      Carry c;
      c.name = s.attr.name;
      c.bytes.swap(bytes);
      c.dtype = s.attr.dtype;
      c.elems = s.attr.n_elems;
      carry.push_back(std::move(c));
    }

    // 最后一段：把图像输出单独存一份
    if (i + 1 == runners_.size()) {
      const Carry* img = nullptr;
      for (const auto& c : carry) {
        if (c.name == "image") { img = &c; break; }
      }
      if (!img) {
        // ⭐⭐⭐ 兜底按**形状**判定图像，而不是「元素数最多」
        //   原因：新切法（n0..n5）的最后一段可能同时输出多个张量，
        //   而中间张量（[1,1152,64,64] = 4.7M 元素）比图像（[1,3,512,512] = 0.8M）**还大**
        //   ⇒ 「元素数最多」会选错！
        //   图像的特征：4 维、通道数 3 或 4、空间 >= 256×256。
        const rknn3_tensor_attr* best = nullptr;
        for (const auto& s : m.outputs()) {
          if (s.fake) continue;
          if (s.attr.n_dims != 4) continue;
          uint32_t c = s.attr.shape[1];
          uint32_t hh = s.attr.shape[2], ww = s.attr.shape[3];
          if ((c == 3 || c == 4) && hh >= 256 && ww >= 256) { best = &s.attr; break; }
        }
        if (best) {
          for (const auto& c : carry) {
            if (c.name == best->name) { img = &c; break; }
          }
        }
      }
      if (!img) {
        // 再兜底：元素数最多的那个（老逻辑，作为最后手段）
        for (const auto& c : carry) {
          if (!img || c.elems > img->elems) img = &c;
        }
      }
      if (!img) { err_ = "最后一段没有输出"; return -1; }
      std::string path = out_prefix + "_image.bin";
      if (!write_file(path, img->bytes.data(), img->bytes.size())) {
        err_ = "写图像失败 " + path;
        return -1;
      }
      // shape：从该段输出张量的 attr 拿（重新查一次）
      for (const auto& s : m.outputs()) {
        if (s.attr.name == img->name) {
          write_meta(path, s.attr.dtype, s.attr.shape, s.attr.n_dims);
          break;
        }
      }
    }

    // ---- 6. 流式模式下立刻释放这一段 ----
    if (!resident_) release_piece(i);
  }

  clock_gettime(CLOCK_MONOTONIC, &t1);
  if (out_ms) {
    *out_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
  }
  return 0;
}

}  // namespace qi
