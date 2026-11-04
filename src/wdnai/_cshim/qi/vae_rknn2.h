// ---------------------------------------------------------------------------
// VAE 解码器的 **RKNN2 链**（跑在 RK3588 本地 NPU 上）
//
// 与 rknn3 的 VaeChain（vae_chain.h）一一对应，只是底层换成 rknn2：
//   - rknn3 链：4 段，1828 协处理器（rkcp 网络），**流式**加载
//                （DiT 与 VAE 不能共卡 ⇒ 每批图结束要释放 DiT 再重建，39 s）
//   - rknn2 链：跑在 **本机 RK3588 的 3 个 NPU 核**上
//                ⇒ 与 DiT **物理分开** ⇒ DiT 可以一直常驻，零 model_init
//
// 三段式（和 rknn3 的 VAE 分段边界一致）：
//   part0: latents            -> add_19, add_28
//   part1: add_19, add_28     -> add_43
//   part2: add_43             -> add_58
//   part3: add_58             -> image
//
// ⚠️ 段与段之间按**张量名**对齐（add_19/add_28/add_43/add_58 两边同名），
//    所以链的形状（几段、每段几个输入输出）完全由模型决定，代码不写死。
//
// ⚠️ **常驻 vs 流式**
//    RK3588 的 NPU 缓冲池实测上限 ≈ 4032 MiB（单次 4032 OK / 4096 FAIL）。
//    - 所有段加起来装得下 ⇒ 全部常驻 ⇒ 生成过程零 model_init（最优）
//    - 装不下 ⇒ 退化成流式（逐段 init / release），仍比 rknn3 链快得多
//      （rknn2 的权重在本机 NVMe 上，且段比 rknn3 的段小）
// ---------------------------------------------------------------------------
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "progress.h"
#include "rknn2_runner.h"

namespace qi {

class VaeChain2 {
 public:
  VaeChain2() = default;
  ~VaeChain2();
  VaeChain2(const VaeChain2&) = delete;
  VaeChain2& operator=(const VaeChain2&) = delete;

  // 初始化整条链。
  //   models      : 各段的 .rknn 路径（按顺序）
  //   core_mask   : NPU 核掩码（RK3588 用 0_1_2 = 7 ⇒ 3 个核全用）
  //   prefer_resident: true ⇒ 尽量全部常驻；装不下就自动退化成流式
  int init(const std::vector<std::string>& models, uint32_t core_mask,
           bool prefer_resident);

  void release();

  bool resident() const { return resident_; }
  size_t n_pieces() const { return runners_.size(); }
  const std::string& error() const { return err_; }

  void set_progress(qi_progress_ctx* ctx) { progress_ctx_ = ctx; }

  // 解一张图。
  //   lat_fp16 : DiT 的潜变量（fp16，token 主序 (1, T, C)）
  //   channels : 潜变量通道数（512 ⇒ 64；0 ⇒ 自动取）
  //   out_rgb  : RGB888 uint8（h × w × 3，行主序）
  //   out_w/out_h : 图像尺寸
  //   out_ms   : 总耗时（毫秒）
  int decode(const std::vector<unsigned char>& lat_fp16, int channels,
             std::vector<unsigned char>* out_rgb, int* out_w, int* out_h,
             double* out_ms);

 private:
  int init_piece(size_t i);
  void release_piece(size_t i);
  // token 主序潜变量 -> 通道主序 (1, C, 1, side, side) 的 float32
  static bool latent_to_nchw_f32(const std::vector<unsigned char>& lat_fp16,
                                 int channels, std::vector<float>* out);

  std::vector<std::unique_ptr<ModelRunner2>> runners_;
  std::vector<std::string> models_;
  uint32_t core_mask_ = 0;
  bool resident_ = false;
  bool inited_ = false;
  std::string err_;
  qi_progress_ctx* progress_ctx_ = nullptr;
};

}  // namespace qi
