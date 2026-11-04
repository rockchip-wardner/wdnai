// ---------------------------------------------------------------------------
// VAE decoder 的「分段链」（512 用）
//
// 为什么
// ------
// 512 的 VAE decoder **单模型装不下**：每核 internal 2600~2900 MiB，
// 而 RK1828 的节点池只有 614~640 MiB ⇒ 必须沿上采样边界切成若干段，
// 每段一个小模型，段间用中间张量在宿主内存里传（和 DiT 分段一个套路）。
//
// 三个必须记住的坑（2026-10-31 踩过）
// ------------------------------------
//  1. **潜变量是 token 主序**：DiT 输出的潜变量是 (1, side*side, 64)（token 主序，
//     = 管道 unpack 之后的布局），而 VAE 要的是 **通道主序** (1, 64, 1, side, side)。
//     直接按通道主序读会**整张图被转置** ⇒ 解出来是紫绿噪点（corr ≈ -0.37）。
//  2. **要乘 latents_std + latents_mean**（逐通道，见 vae_stats.h）—— 漏了整体色调全错。
//  3. **段间中间张量不用自己转布局**：`get_output_bytes` 返回的已经是**逻辑 NCHW**，
//     而 `load_inputs` 自己会做 NCHW -> NHWC / NC1HWC2 的转换 ⇒ 直接互传即可。
// ---------------------------------------------------------------------------
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "rknn3_runner.h"
#include "progress.h"

namespace qi {

class VaeChain {
 public:
  VaeChain() = default;
  ~VaeChain();
  VaeChain(const VaeChain&) = delete;
  VaeChain& operator=(const VaeChain&) = delete;

  // 初始化整条链。
  // prefer_resident=true：全部 init 常驻（生成过程零 load/init）；
  //   任何一段 init 失败 ⇒ 全部释放并退化成**流式**（逐段 init/release）。
  int init(const std::vector<std::string>& models, const std::vector<std::string>& weights,
           uint32_t core_mask, const char* device_id, bool prefer_resident);

  void release();

  bool resident() const { return resident_; }
  size_t n_pieces() const { return runners_.size(); }
  const std::string& error() const { return err_; }

  // ---- 进度回调（可选）----
  // 每跑完一段报一次（stage="vae"）⇒ 上层能显示「VAE 第 2/4 段」。
  // ⚠️ 传的是**会话的共享 ctx 指针** ⇒ elapsed 口径全局一致。
  void set_progress(qi_progress_ctx* ctx) { progress_ctx_ = ctx; }

  // 解一张图。
  //   lat_fp16   : DiT 的潜变量（fp16，token 主序 (1, T, C)）
  //   channels   : 潜变量通道数（512 ⇒ 64）
  //   out_prefix : 图像写成 <out_prefix>_image.bin + <out_prefix>_image.bin.meta
  // 返回 0 成功；out_ms 返回总耗时（毫秒）。
  //   channels=0 ⇒ 自动从第 0 段输入张量的形状里取（512 ⇒ 64）
  int decode(const std::vector<unsigned char>& lat_fp16, int channels,
             const std::string& scratch_dir, const std::string& out_prefix, double* out_ms);

 private:
  int init_piece(size_t i);
  void release_piece(size_t i);
  // 把 token 主序潜变量转成通道主序 + 乘 std/mean（fp32）
  static bool latent_to_nchw(const std::vector<unsigned char>& lat_fp16, int channels,
                              std::vector<float>* out);

  std::vector<std::unique_ptr<ModelRunner>> runners_;
  std::vector<std::string> models_;
  std::vector<std::string> weights_;
  uint32_t core_mask_ = 0;
  std::string device_id_;
  bool resident_ = false;
  bool inited_ = false;
  std::string err_;

  // 进度回调（每段报一次）—— **共享**的 ctx（口径统一）
  qi_progress_ctx* progress_ctx_ = nullptr;
};

}  // namespace qi
