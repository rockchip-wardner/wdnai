// ---------------------------------------------------------------------------
// wdnai cshim —— Text Encoder（Qwen-Image-2.1 的 LLM/text_encoder）
//
// ⭐ 与参考实现 deploy/src/main.cpp 里的 ``TeEncoder`` **同源**
//    （只删掉了 Rockchip 的闭源 Tokenizer：分词改由 Python 侧的
//      gguf 分词器完成，token ids 由 Python 传进来）。
//
// 形态 A2（板端现役的 TE）：
//    输入  input_embeds [1,128,4096] fp16
//          attention_mask [1,128] fp16
//          position_ids   [1,128] int64
//    输出  encoder_hidden_states [1,128,4096] fp16
//
// ⚠️ drop_idx 必须在**输出侧**做（输入喂完整 token 序列，输出再取 [drop, n)）
//    —— 119 上 ORT 的决定性实验结论。
// ---------------------------------------------------------------------------
#pragma once

#include <sys/mman.h>
#include <unistd.h>

#include <cstdint>
#include <string>
#include <vector>

#include "rknn3_api.h"
#include "rknn3_runner.h"

namespace qi {

// embedding 表（mmap，只读）
struct EmbedStore {
  float16* data = nullptr;   // mmap 出来的 embedding 表
  size_t bytes = 0;
  int dim = 4096;            // hidden_size
  int rows = 0;              // 实际行数
  int fd = -1;
  ~EmbedStore() {
    if (data) munmap(data, bytes);
    if (fd >= 0) close(fd);
  }
};

class TeEncoder {
 public:
  //: DiT seg0 的 `encoder_hidden_states` 固定形状是 [1, 128, 4096]
  static constexpr uint32_t kMaxSeq = 128;
  //: diffusers: self._drop_idx = len(processor.apply_chat_template(system_msg))
  static constexpr uint32_t kDropIdx = 14;

  ~TeEncoder() = default;

  // 一次性初始化（36.8 s 的 model_init 就在这里）
  //   device_id 为 nullptr/空 ⇒ 退回 device[0]
  //   forced_seq <= 0 ⇒ 用模型自己声明的序列长度
  int open(const std::string& model_path, const std::string& weight_path,
           const std::string& embed_path, uint32_t core_mask, const char* device_id,
           const std::string& rope_dir, int forced_seq);

  // 每张图一次：填输入 → run → 取输出 → 后处理
  //   tokens 必须是**带 t2i 模板**的完整 token 序列（Python 侧分词得到）
  bool encode(const int32_t* tokens, int n_tokens);

  double last_ms() const { return last_ms_; }

  // 上一次 encode 的结果（生命周期到下一次 encode）
  const uint16_t* hs() const { return out_hs_.empty() ? nullptr : reinterpret_cast<const uint16_t*>(out_hs_.data()); }
  size_t hs_count() const { return out_hs_.size(); }
  const uint8_t* mask() const { return out_mask_.empty() ? nullptr : out_mask_.data(); }
  size_t mask_count() const { return out_mask_.size(); }
  uint32_t dim() const { return dim_; }

 private:
  const rknn3_tensor_attr* find_in(const char* nm) const;
  const rknn3_tensor_attr* find_out(const char* nm) const;

  ModelRunner m_;
  EmbedStore embed_;
  const rknn3_tensor_attr* a_hs_ = nullptr;
  uint32_t dim_ = 4096;
  uint32_t model_seq_ = 0;
  bool flavor_a_ = false;
  bool flavor_a2_ = false;
  std::string flavor_;
  std::string rope_dir_;
  int forced_seq_ = 0;
  double last_ms_ = 0.0;
  std::vector<float16> out_hs_;
  std::vector<uint8_t> out_mask_;
};

}  // namespace qi
