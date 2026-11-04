// ---------------------------------------------------------------------------
// DiT「常驻模式」：8 段 context 一次 init，40 步 flow-match 不再重建模型。
//
// 为什么需要（2026-10-30，第十八轮，debug.md §51）
// ---------------------------------------------------------------------------
// 现状：每步 8 段串行执行，每段都「load 权重 → model_init → 推理 → 释放」。
//   实测（w8a16）model_init = 10.6 s/段 × 8 = 85 s/步，占 112 s/步的 76%。
//   ⇒ 40 步图里 78% 的时间花在**重复重建模型**上。
//
// 目标：让 8 段**同时常驻**在卡上，40 步只 init 一次。
//
// 为什么能做到（本轮实测打通的关键）
// ---------------------------------------------------------------------------
// 1) 每段的 internal（中间激活工作区）恒为 ~1878 MiB（与层数/量化无关），
//    8 段各自一份 = 15024 MiB ≫ 整卡 5122 MiB ⇒ **物理上不可能各自独占**。
// 2) 但 internal 是**工作区**：8 段严格串行执行 ⇒ 同一时刻只有一段在跑
//    ⇒ 8 段**可以共享同一块 internal 缓冲**。
// 3) RKNN3 提供 `config.user_mem_internal=1` + `rknn3_set_internal_mem()`，
//    让宿主分配并注入 internal 缓冲。
//    ⚠️ **调用顺序必须是「先 model_init(user_mem_internal=1)，再 set_internal_mem」** ——
//       反过来（先 set 再 init）会被卡端拒：
//         "SET_INTERNAL_MEM fail, ack = ACK_INVALD_ARGUMENT"
//       本轮已穷举 6 种顺序验证（scripts/probe_setmem_order.cpp 留档）。
//
// 内存账（w4a16，每核实测，来自 rknn3_query(ALLOCATION_INFO)）
// ---------------------------------------------------------------------------
//   每核 weight 70.4 MiB + cmd 3.6 MiB + internal(共享后) 357.2 MiB(峰值核)
//   8 段常驻 ⇒ 每核 8×70.4 + 8×3.6 + 357.2 = 949 MiB > 614.5 MiB 节点池 ❌
//   ⇒ **单卡装不下 8 段**。但每卡 4 段：4×74 + 357.2 = 653 MiB（仍超 39 MiB）
//   ⇒ 4 段/卡也需要 internal 更小，或者用**双卡各 4 段**。
//
// 本文件实现的能力
// ---------------------------------------------------------------------------
//   * `ResidentChain`：持有 N 个 ModelRunner + **一份**共享 internal 缓冲；
//     init 一次，之后每步只做 run。
//   * 共享 internal 的分配：先用一个「探针 context」跑普通 model_init，
//     用 rknn3_query(ALLOCATION_INFO) 拿到**每核** internal 真实字节数，
//     再按每核实际需求分配（而不是一刀切峰值），避免浪费。
//
// 跨卡说明：8 段链是**严格串行**的（seg0→seg1→…→seg7），
//   所以「双卡各 4 段」在**单张图**上没有加速意义（还多一次 PCIe 中转）。
//   常驻模式的价值在于**消灭每步 85 s 的重建**，这个收益单卡即可拿到 ——
//   只要 internal 能小到 8 段挤进一张卡。
// ---------------------------------------------------------------------------

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "rknn3_api.h"
#include "rknn3_runner.h"
#include "progress.h"

namespace qi {

// 一份可被多个 context 共享的 internal 缓冲
struct SharedInternal {
  rknn3_context owner = 0;                       // 创建这些 mem 的 context
  std::vector<rknn3_tensor_mem*> mem;           // 每核一块
  std::vector<uint64_t> per_core_bytes;          // 每核实际字节数

  ~SharedInternal() { release(); }
  void release();
  bool valid() const { return !mem.empty(); }
  uint64_t total_bytes() const;
};

// 「常驻链」：一次 init，之后反复 run
class ResidentChain {
 public:
  ~ResidentChain();

  // 初始化 N 段（每段一个 .rknn + .weight），并共享一份 internal。
  // 失败返回非 0，并把错误信息打出来。
  //
  // ⭐ device_ids（2026-10-30，第十九轮加）：
  //   每段可以指定不同的设备（双卡方案：前 N 段在卡 A，剩下的在卡 B）。
  //   空字符串 ⇒ 用环境变量 RKNN3_DEVICE_ID / device[0]。
  //   ⭐ 不共享 internal 时（每卡只放一个模型）**不要跑探针** ——
  //     探针会临时多占一份卡上内存，单卡单模型时可能直接 MODEL_SETUP fail。
  //
  // ⭐ stream_flags（2026-10-30，第十九轮加，方案 A 需要）：
  //   stream_flags[i] = true ⇒ 该段是**流式段**：**不常驻**，
  //   每次 run_once 都走「load → model_init → run → 释放」。
  //   常驻段只做 run（零 PCIe 权重传输、零 model_init）。
  //   为什么需要：卡上每核池 614.5 MiB ⇒ **每卡最多常驻 1 个 8-block 模型**
  //   （实测每核峰 569.9 MiB，见 debug.md §54）⇒ 32 块要 4 个模型，只有 2 张卡
  //   ⇒ 「2 段常驻（各一卡）+ 2 段流式」是唯一装得下的形态。
  int init(const std::vector<std::string>& models, const std::vector<std::string>& weights,
           uint32_t core_mask, bool share_internal,
           const std::vector<std::string>* device_ids = nullptr,
           const std::vector<bool>* stream_flags = nullptr);

  // 跑完整条链一次。carry 里存放段间传递的中间张量。
  // input_dir 提供每段独有的输入（第 0 段的 hidden_states 等）。
  int run_once(const std::string& input_dir, std::map<std::string, std::vector<unsigned char>>* carry,
               double* total_ms);

  size_t size() const { return runners_.size(); }
  const SharedInternal* shared() const { return shared_.get(); }

  // ⭐⭐⭐ flow-match 常驻模式（2026-10-31，第二十轮）
  //
  // 目标：**生成过程中零 load / 零 model_init**。
  //   之前 x 的回写（x = x + (σ_next − σ)·v）在 Python 层（run_flowmatch.py），
  //   每一步都要起一个 `dit-chain` 子进程 ⇒ 每步 8 次 load + 8 次 model_init。
  //   ⇒ 把整个 flow-match 循环搬进**同一个常驻进程**：
  //       每步只做 run（常驻段），x 直接在进程内更新并写回输入缓冲。
  //
  // 依赖：所有段都必须是**常驻段**（stream_ 全 false），否则仍有 load/init。
  //
  // sigmas 序列与 scripts/run_flowmatch.py 的 sigmas_for() **逐位一致**：
  //   linspace(1, 1/N, N) → time_shift_exponential(mu) → stretch_shift_to_terminal(0.02)
  //   → concat([..., 0])。
  //
  // 返回 0 = 成功；非 0 = 失败。
  int run_flowmatch(const std::string& input_dir, int steps, double mu, double shift_terminal,
                    const std::string& x_path, const std::string& out_latents_path,
                    const std::string& dump_dir);

  // ⭐⭐⭐ 主动释放全部段（模型 + 共享 internal）⇒ **卡池回收**
  //   用途：512 的 DiT 常驻（每核 ~451 MiB）与 512 的 VAE 分段链
  //   （每核还要几百 MiB）**物理上不能同时常驻** ⇒
  //   先用 DiT 把 N 张图的潜变量全跑出来，再释放 DiT、让 VAE 解码。
  //   （DiT 阶段内部仍然是零 load / 零 model_init。）
  void release();

  // ⭐⭐⭐ 方案 D（第三十三轮）：把 DiT 的「头」切出来单独算。
  //
  //   头 = timestep_embedder + modulation，**只依赖 timestep**，4 个段各带一份
  //   （≈ 53.7 MiB int4/段 ⇒ 卡上白占 4 份 ≈ 20.2 MiB/核，纯浪费）。
  //   ⇒ 切成一个独立的小模型放到**另一张卡**上；它的输出
  //   （modulation_out / timestep_emb_out）按**名字**进 carry map
  //   ⇒ 段按名取输入（load_inputs 的 overrides 机制，现成的）。
  //
  //   device_id 为空 ⇒ 自动挑一张**没被段占用**的卡（rknn3_find_devices）。
  int init_head(const std::string& model, const std::string& weight,
                const char* device_id = nullptr);
  // 释放头模型（VAE 阶段要占同一张卡时用）
  void release_head();
  bool has_head() const { return head_ != nullptr; }
  std::string head_device() const { return head_device_; }

  // 打印内存占用摘要
  void print_mem_summary() const;

  // ---- 进度回调（可选）----
  // flow-match 每跑完一步就报一次（stage="flow"），
  // 上层据此画进度条 + 估剩余时间。
  // ⚠️ 传的是**会话的共享 ctx 指针**（不是自己的副本）⇒ elapsed 口径
  //    与会话层（init/te/release/done）完全一致。
  void set_progress(qi_progress_ctx* ctx) { progress_ctx_ = ctx; }

 private:
  std::vector<std::unique_ptr<ModelRunner>> runners_;
  // ⭐ 方案 D：切出来的「头」模型（单独一张卡）
  std::unique_ptr<ModelRunner> head_;
  std::string head_device_;
  std::unique_ptr<SharedInternal> shared_;
  // ⭐ 多卡时**每张卡一份**共享块（mem 绑定在 device 上，不能跨卡共用）
  std::vector<std::pair<rknn3_context, std::vector<rknn3_tensor_mem*>>> extra_shared_;
  std::vector<uint64_t> seg_internal_bytes_;   // 每段各自的 internal 需求（探针得到）
  std::vector<uint64_t> seg_weight_bytes_;

  // ⭐ 流式段（方案 A）：不常驻，每次 run_once 现场建/拆。
  std::vector<std::string> models_;      // 全部段的 .rknn 路径
  std::vector<std::string> weights_;     // 全部段的 .weight 路径
  std::vector<std::string> device_ids_;  // 全部段的设备 id
  std::vector<bool> stream_;             // 该段是否流式
  std::vector<size_t> runner_seg_;       // runner 下标 → 段号
  uint32_t core_mask_ = 0;               // 流式段也要用

  // 进度回调（flow-match 每步报一次）—— **共享**的 ctx（口径统一）
  qi_progress_ctx* progress_ctx_ = nullptr;
};

// 探针：跑一次普通 model_init，拿到该模型的每核 internal/weight 需求。
// 返回 false 表示失败。out_internal/out_weight 按核填入字节数。
bool probe_model_mem(const std::string& model_path, const std::string& weight_path,
                     std::vector<uint64_t>* out_internal, std::vector<uint64_t>* out_weight,
                     const char* device_id = nullptr);

}  // namespace qi
