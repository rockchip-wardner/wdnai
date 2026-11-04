// ---------------------------------------------------------------------------
// Qwen-Image-2.1 的 **RKNN2**（RK3588 本地 NPU）模型 runner
//
// 为什么需要它
// ------------
// 现有链路里 TE + DiT 走 **RKNN3 / rk1820**（两张 1828 协处理器板，经 rkcp
// 网络连过来），而 **VAE 解码器**原本也在 1828 上跑。问题是：
//   DiT 与 VAE 物理上不能共卡（VAE 4 段 internal 合计 > 节点池）
//   ⇒ 每批图跑完必须**释放 DiT**，下一批再重新 init（39 s）
//
// 把 VAE 挪到 **RK3588 本地 NPU（rknn2）** 之后：
//   DiT 常驻不动（零 model_init），VAE 也在本地常驻 ⇒ 生成过程一次 init 都不用
//
// 只依赖 **系统自带的 librknnrt.so**（apt: rknn2-runtime 2.5.0）
// ⇒ 模型包里不带任何 Rockchip 的 .so / .bin。
//
// ⭐ 用 **dlopen** 而不是直接链接
//   - 没有装 rknn2 runtime 的板子照样能编译 / 运行（rknn2 VAE 自动不可用，
//     退回 rknn3 的 VAE 链 ⇒ 行为与改造前完全一致）
//   - 也避免把 librknnrt 变成 cshim 的硬依赖
//
// 布局约定（与 rknn3_runner 一致）
// --------------------------------
//   喂进去的数据一律是**逻辑 NCHW**（[n][c][...][h][w] 连续），
//   runtime 按 rknn_input.fmt / .type 负责转成模型内部布局；
//   取出来的输出一律先归一化成**逻辑 NCHW** 再交给下一段。
//   ⇒ 段与段之间只按**张量名**对齐，不关心各自的内部布局。
// ---------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "rknn_api.h"

namespace qi {

// 一个张量的元信息
struct TensorAttr2 {
  std::string name;
  uint32_t n_dims = 0;
  uint32_t dims[8] = {0};   // 形状（NC1HWC2 时是**原生**形状）
  uint32_t n_elems = 0;
  uint32_t size = 0;        // 字节数
  rknn_tensor_format fmt = RKNN_TENSOR_NCHW;
  rknn_tensor_type type = RKNN_TENSOR_FLOAT32;
};

// rknn2 runtime 的动态符号表（dlopen 出来的）
struct Rknn2Api {
  bool ok = false;
  std::string err;

  int (*init)(rknn_context*, void*, uint32_t, uint32_t, rknn_init_extend*) = nullptr;
  int (*destroy)(rknn_context) = nullptr;
  int (*query)(rknn_context, rknn_query_cmd, void*, uint32_t) = nullptr;
  int (*set_core_mask)(rknn_context, rknn_core_mask) = nullptr;
  int (*inputs_set)(rknn_context, uint32_t, rknn_input[]) = nullptr;
  int (*run)(rknn_context, rknn_run_extend*) = nullptr;
  int (*outputs_get)(rknn_context, uint32_t, rknn_output[], rknn_output_extend*) = nullptr;
  int (*outputs_release)(rknn_context, uint32_t, rknn_output[]) = nullptr;
};

// 全局单例（首次调用时才 dlopen）；没有 rknn2 runtime 时 ok=false
const Rknn2Api& rknn2_api();

class ModelRunner2 {
 public:
  ModelRunner2() = default;
  ~ModelRunner2();
  ModelRunner2(const ModelRunner2&) = delete;
  ModelRunner2& operator=(const ModelRunner2&) = delete;

  // 初始化：rknn_init → set_core_mask → query IO 属性。
  // core_mask 用 rknn_core_mask（RK3588 有 3 个核 ⇒ RKNN_NPU_CORE_0_1_2）。
  int init(const std::string& model_path, uint32_t core_mask);
  void release();

  bool ready() const { return ctx_ != 0; }
  const std::vector<TensorAttr2>& inputs() const { return in_; }
  const std::vector<TensorAttr2>& outputs() const { return out_; }

  // 喂输入：data 是**逻辑 NCHW**，dtype 说明缓冲区里的实际精度。
  // runtime 会按 fmt/type 转成模型内部布局。
  int set_input(const std::string& name, const void* data, size_t bytes, rknn_tensor_type dtype);

  // 执行一次推理（内部计时）
  int run(double* out_ms = nullptr);

  // 取输出张量，归一化成**逻辑 NCHW**（dtype 与模型输出一致）。
  bool get_output(const std::string& name, std::vector<unsigned char>* bytes,
                  rknn_tensor_type* dtype) const;

  const std::string& error() const { return err_; }
  void print_info() const;

 private:
  rknn_context ctx_ = 0;
  std::vector<TensorAttr2> in_;
  std::vector<TensorAttr2> out_;
  std::string model_path_;
  std::string err_;

  // 最近一次 run 的输出缓冲（run() 里拿、get_output() 里取）
  std::vector<rknn_output> last_outputs_;
  std::vector<std::vector<unsigned char>> last_bytes_;
  // ⭐ 每个输出**实际**拿到的 dtype（量化模型用 want_float=1 取回 fp32）
  std::vector<rknn_tensor_type> actual_types_;
};

// 把 rknn2 的原生输出布局（NHWC / NC1HWC2）归一化成逻辑 NCHW。
// NCHW 原样返回（零拷贝）。
bool rknn2_to_logical_nchw(const TensorAttr2& attr, const unsigned char* src,
                           std::vector<unsigned char>* dst);

}  // namespace qi
