// ---------------------------------------------------------------------------
// Qwen-Image-2.1 on RKNN3 (rk1820) —— 通用模型 runner
//
// 只依赖 librknn3_api.so 的公开接口（rknn3_api.h）：
//   rknn3_init / rknn3_load_model_from_path / rknn3_model_init / rknn3_query
//   rknn3_create_mem / rknn3_mem_sync / rknn3_run / rknn3_destroy_mem / rknn3_destroy
//
// 设计要点：
//   1. 所有 IO 张量都通过 rknn3_query(RKNN3_QUERY_*_ATTR) 动态发现，
//      不硬编码张量个数和名字（DiT 是 4 task、VAE 是 2 task，task 间还有
//      coreN_fake_in / coreN_fake_out 这种内部 glue 张量）。
//   2. 输入数据按 **张量名** 从目录里读：<input-dir>/<attr.name>.bin
//      文件内容是裸数据，dtype = 该张量的原生 dtype（DiT/VAE 是 fp16，
//      mask 是 bool）。若存在 <attr.name>.f32.bin 则按 float32 读入再转换。
//   3. 输出写成 <prefix>.bin + <prefix>.meta（meta 里记 dtype 和 shape），
//      方便 compare 子命令做数值比对。
// ---------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "rknn3_api.h"

namespace qi {

// 张量元信息 + 分配的 device memory
struct TensorSlot {
  rknn3_tensor_attr attr{};
  rknn3_tensor_mem* mem = nullptr;
  bool fake = false;  // coreN_fake_in / coreN_fake_out：多 task 之间的内部同步张量
};

class ModelRunner {
 public:
  ModelRunner() = default;
  ~ModelRunner();

  ModelRunner(const ModelRunner&) = delete;
  ModelRunner& operator=(const ModelRunner&) = delete;

  // 初始化：init -> load model -> model_init -> query IO -> 分配 IO 内存
  //
  // ⭐ device_id（2026-10-30，第十九轮加）：
  //   `librknn3_api.so` **不读 `RKNN3_DEVICE_ID` 环境变量** ——
  //   传 `nullptr` 给 rknn3_init 时它会用 `device[0]`（PCIe 枚举顺序第一张卡）。
  //   要指定卡必须自己填 `rknn3_init_extend.device_id`。
  //   device_id 为 nullptr/空 ⇒ 退回「环境变量 RKNN3_DEVICE_ID，再退回 device[0]」。
  int init(const std::string& model_path, const std::string& weight_path,
           uint32_t core_mask, bool user_mem_internal, const char* device_id = nullptr);

  // ⭐ 注入用户自管的 internal 缓冲（多段共享一份 ⇒ 「8 段常驻」的前提）。
  //
  // ⚠️ 调用顺序**必须**是：init(user_mem_internal=1) 之后、第一次 run 之前。
  //    反过来（先 set 再 model_init）会被卡端拒：
  //      "SET_INTERNAL_MEM fail, ack = ACK_INVALD_ARGUMENT"
  //    （本轮已穷举 6 种顺序验证，见 scripts/probe_setmem_order.cpp）
  //
  // mems[c] 必须是**本 context** 上用 rknn3_create_mem 分配的那一块
  //（跨 context 的 mem 会被拒）。
  int set_shared_internal_mem(std::vector<rknn3_tensor_mem*>* mems);

  // 查询本模型每核的 command/weight/internal/kvcache 实际占用（字节）。
  // ⚠️ ALLOCATION_INFO 必须传**数组**（size = sizeof(struct) × core_num）。
  bool query_core_alloc(std::vector<uint64_t>* cmd, std::vector<uint64_t>* weight,
                        std::vector<uint64_t>* internal, std::vector<uint64_t>* kvcache);

  void release();

  // 打印 SDK 版本 / 设备信息 / IO 张量表
  void print_info() const;

  // 从目录读输入并写入 device memory（按张量名匹配）。
  // overrides 里如果有同名张量，优先用 overrides 的数据（不依赖 dtype 转换，
  // 数据已是该张量的原生 dtype），用于分段模型之间的中间结果传递。
  // raw_names 里的张量**跳过**（调用方要用 set_raw_input 自己写，比如 rope cache）。
  int load_inputs(const std::string& dir, bool verbose,
                  const std::map<std::string, std::vector<unsigned char>>* overrides = nullptr,
                  const std::set<std::string>* raw_names = nullptr);

  // 取某个输出张量的原始字节（原生 dtype），用于喂给下一个分段模型
  bool get_output_bytes(const std::string& name, std::vector<unsigned char>* out) const;

  // **原样**把数据写进某个输入张量的 device mem 并同步到设备。
  // 与 load_inputs 的区别：不做任何布局转换 / dtype 转换 —— 数据必须已经是
  // 该张量的**原生布局 + 原生 dtype**。
  // 用途：模型里有些张量虽然被标成 NC1HWC2，但逻辑形状和通用的
  // 「NCHW → NC1HWC2」规则对不上（例如 rope_cos_cache [1,8,1,1024,16]，
  // 它的 c1=8 是 head 数、c2=16 是 rotary 维，h=1，不是通道分片）。
  // 这种张量必须走这条路，不能用通用转换。
  bool set_raw_input(const std::string& name, const void* data, size_t bytes);

  // 执行一次推理（含计时）
  int run(double* out_ms = nullptr);

  // ---- 官方调试接口（定位「跑不起来」的模型）----
  // 逐算子 profile：打印每个 op 的类型/耗时/cycles/带宽
  int profile_ops(uint32_t log_level);
  // 打印每核内存分配（command/weight/internal/kvcache）
  int profile_mem();
  // 逐层 dump 中间特征到 .npy
  int dump_features(const std::string& dir, uint32_t core_mask);

  // 把输出写到 <prefix>.bin（原生 dtype）+ <prefix>.meta
  int save_outputs(const std::string& prefix) const;

  const std::vector<TensorSlot>& inputs() const { return in_; }
  const std::vector<TensorSlot>& outputs() const { return out_; }

  rknn3_context context() const { return ctx_; }

 private:
  rknn3_context ctx_ = 0;
  std::vector<TensorSlot> in_;
  std::vector<TensorSlot> out_;

  // ⭐⭐⭐ 2026-10-31 修一个**段错误**：
  //   `rknn3_init_extend` 里带 `char* device_id`，runtime 会**长期持有**这个指针。
  //   之前它是 **init() 栈上的临时变量**，且 device_id 指向调用方的临时字符串
  //   （ResidentChain 的 `devs[i].c_str()`）⇒ 指针悬空 ⇒ 后续 create_mem 时 SIGSEGV。
  //   （`dit` 子命令因为走 getenv() 拿到的是环境变量存储，永久有效，所以不崩。）
  // ⇒ 现在把 device_id 和 init_extend 都做成 **runner 的持久成员**。
  std::string device_id_;
  rknn3_init_extend init_extend_{};
  // ⚠️ IO 缓冲核分散计数器已提到**进程级**（rknn3_runner.cpp 的 pick_io_core），
  //    按 device_id 分桶做「累计字节最少」均衡。
  //    （原为 per-runner 成员 ⇒ 每段都从 core0 开始 ⇒ 4 段的 33 MiB 输入全砸 core0。）

 public:
  // ---- 小工具 ----
  static size_t dtype_size(rknn3_tensor_type t);
  static const char* dtype_name(rknn3_tensor_type t);
  // 从 "float32" / "fp16" / "bool" / "int32" 之类的字符串解析 dtype
  static rknn3_tensor_type dtype_from_name(const std::string& s);
  // coreN_fake_in / coreN_fake_out 判定
  static bool is_fake_name(const std::string& name);
  // 把 src（src_dtype）转成 dst（dst_dtype），n 个元素
  static bool convert(const void* src, rknn3_tensor_type src_dtype, void* dst,
                      rknn3_tensor_type dst_dtype, uint32_t n);
  // ---- 布局转换（device mem 里放的是**原生**布局，runtime 不帮忙转）----
  // NCHW (n,c,h,w) -> NHWC (n,h,w,c)
  static void transpose_nchw_nhwc(const unsigned char* src, unsigned char* dst, uint32_t n,
                                  uint32_t c, uint32_t h, uint32_t w, size_t esz);
  // NCHW (n,c,h,w) -> NC1HWC2 (n,c1,h,w,c2)，其中 c = c1*c2 + c2
  static void transpose_nchw_nc1hwc2(const unsigned char* src, unsigned char* dst, uint32_t n,
                                    uint32_t c1, uint32_t h, uint32_t w, uint32_t c2,
                                    size_t esz);
  // NC1HWC2 (n,c1,h,w,c2) -> NCHW (n,c,h,w)（上面那个的逆）
  static void transpose_nc1hwc2_nchw(const unsigned char* src, unsigned char* dst, uint32_t n,
                                    uint32_t c1, uint32_t h, uint32_t w, uint32_t c2,
                                    size_t esz);
  // 4D 张量是否需要 NCHW<->NHWC 转置
  static bool needs_nhwc_transpose(const rknn3_tensor_attr& a);
  // 5D 张量是否需要 NCHW<->NC1HWC2 重排
  static bool needs_nc1hwc2_convert(const rknn3_tensor_attr& a);
};

// 正确的 IEEE-754 fp16 <-> fp32 转换（round-to-nearest-even）。
// SDK 自带的 float16.h 里 fp32_to_fp16() 是截断而非 RNE，
// 且 fp16_to_fp32() 对次正规数处理有误 —— 这里用自己实现的版本，
// 保证与 numpy 的 astype 逐位一致。
float16 fp32_to_fp16_rne(float f);
float fp16_to_fp32_exact(float16 h);

// ⭐⭐⭐ 2026-11-02：登记「每核预留字节」—— 共享 internal 块等
//   「还没分配但稍后一定要有」的空间。pick_io_core 挑核时会自动扣掉它，
//   避免把 IO 缓冲塞到池最大、但稍后要给共享块留位置的核上。
//   （共享 internal 的分配发生在所有段 IO 之后，见 resident_chain.cpp 第 4 步）
void set_io_reserve(const std::string& dev, const std::vector<uint64_t>& bytes);

// 读裸二进制文件；失败返回 false
bool read_file(const std::string& path, void* buf, size_t size);
// 写裸二进制文件
bool write_file(const std::string& path, const void* buf, size_t size);
// 解析 <file>.meta：返回 dtype 和 shape 描述（找不到时用默认值）
struct MetaInfo {
  rknn3_tensor_type dtype = RKNN3_TENSOR_FLOAT32;
  std::vector<uint32_t> shape;
};
bool read_meta(const std::string& bin_path, MetaInfo* out);

}  // namespace qi
