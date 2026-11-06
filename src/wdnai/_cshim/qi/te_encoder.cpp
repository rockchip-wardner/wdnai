#include "te_encoder.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <map>
#include <set>

#include "util.h"

namespace qi {

// ---------------------------------------------------------------------------
int TeEncoder::open(const std::string& model_path, const std::string& weight_path,
                    const std::string& embed_path, uint32_t core_mask, const char* device_id,
                    const std::string& rope_dir, int forced_seq) {
  // ---- 1. 加载模型（**这一步 36.8 s，常驻后只付一次**）----
  int ret = m_.init(model_path, weight_path, core_mask, /*user_mem_internal=*/false,
                    device_id);
  if (ret != RKNN3_SUCCESS) {
    printf("[te] 模型初始化失败 ret=%d\n", ret);
    return ret;   // ⭐ 原样回传 rknn3 错误码（调用方要靠它区分「卡地址不对」）
  }
  m_.print_info();

  // ⚠️ attr.name 是 char*，必须用 strcmp 比内容（直接 == 比的是指针！）
  const rknn3_tensor_attr* a_ids = find_in("input_ids");
  const rknn3_tensor_attr* a_emb = find_in("input_embeds");
  const rknn3_tensor_attr* a_pos = find_in("position_ids");
  if (!a_ids && !a_emb) {
    printf("[te] 这个模型既没有 input_ids 也没有 input_embeds 输入，不是可用的 TE 模型。\n");
    return 2;
  }
  flavor_a_ = (a_ids != nullptr);
  flavor_a2_ = (!flavor_a_ && a_emb && a_pos);
  flavor_ = flavor_a_ ? "A（input_ids）"
             : flavor_a2_ ? "A2（input_embeds + position_ids）"
                           : "B（input_embeds + mrope + rope 常量）";
  printf("[te] 模型形态: %s\n", flavor_.c_str());

  a_hs_ = find_out("encoder_hidden_states");
  if (!a_hs_) {
    printf("[te] 模型没有 encoder_hidden_states 输出\n");
    return 2;
  }
  dim_ = (a_hs_->n_dims >= 3) ? a_hs_->shape[2] : 4096;

  // ---- 2. 序列长度：优先用模型声明的固定长度 ----
  {
    const rknn3_tensor_attr* a = flavor_a_ ? a_ids : a_emb;
    if (a && a->n_dims >= 2) model_seq_ = a->shape[1];
  }

  rope_dir_ = rope_dir;
  forced_seq_ = forced_seq;

  // ---- 3. embedding 表（形态 A2/B 需要查表拼 input_embeds）----
  //    ⚠️ mmap 是**懒加载**的虚拟映射，不占实际 RAM
  if (!flavor_a_) {
    struct stat est;
    if (stat(embed_path.c_str(), &est) != 0) {
      printf("[te] 打不开 embedding 文件: %s\n", embed_path.c_str());
      return 2;
    }
    embed_.fd = ::open(embed_path.c_str(), O_RDONLY);   // ⚠️ 全局限定：成员 open 会遮蔽 ::open
    embed_.bytes = (size_t)est.st_size;
    embed_.data = (float16*)mmap(nullptr, embed_.bytes, PROT_READ, MAP_PRIVATE, embed_.fd, 0);
    embed_.dim = (int)dim_;
    embed_.rows = (int)(embed_.bytes / (sizeof(float16) * (size_t)embed_.dim));
    printf("[te] embedding 表: %d 行 x %d 维 (fp16)\n", embed_.rows, embed_.dim);
    if (embed_.rows <= 0 || embed_.data == MAP_FAILED) {
      printf("[te] mmap embedding 失败\n");
      return 2;
    }
  }
  return 0;
}

// ---------------------------------------------------------------------------
bool TeEncoder::encode(const int32_t* tokens, int n_tokens) {
  // ⚠️⚠️ drop 必须在**输出侧**做，不能在输入侧做！（119 上 ORT 决定性实验）
  //   ⇒ 输入要喂**完整** token 序列（含 system 段），输出再取 [drop, n_tok)。
  const uint32_t drop = kDropIdx;
  const uint32_t n_tok = (uint32_t)n_tokens;
  if (n_tok <= drop) {
    printf("[te] token 数 %u 少于 drop_idx %u，prompt 太短\n", n_tok, drop);
    return false;
  }
  const uint32_t keep = n_tok - drop;

  // ---- 序列长度 + 左补 ----
  uint32_t seq = model_seq_ ? model_seq_ : n_tok;
  if (forced_seq_ > 0) seq = (uint32_t)forced_seq_;
  if (seq < n_tok) seq = n_tok;
  const uint32_t offset = (seq > n_tok) ? (seq - n_tok) : 0;

  // position_ids：diffusers 规则 = cumsum(mask) - 1，padding 处取 0
  std::vector<int64_t> pos(seq, 0);
  {
    int64_t c = 0;
    for (uint32_t i = 0; i < seq; ++i) {
      if (i >= offset) c++;
      pos[i] = (c > 0) ? (c - 1) : 0;
    }
  }

  std::map<std::string, std::vector<unsigned char>> ov;
  std::set<std::string> raw_names;
  auto put_i64 = [&](const char* nm, const std::vector<int64_t>& v,
                     rknn3_tensor_type dt) -> bool {
    size_t esz = ModelRunner::dtype_size(dt);
    if (!esz) return false;
    std::vector<unsigned char> out(v.size() * esz);
    if (!ModelRunner::convert(v.data(), RKNN3_TENSOR_INT64, out.data(), dt, (uint32_t)v.size()))
      return false;
    ov[nm] = out;
    return true;
  };
  auto put_f32 = [&](const char* nm, const std::vector<float>& v,
                     rknn3_tensor_type dt) -> bool {
    size_t esz = ModelRunner::dtype_size(dt);
    if (!esz) return false;
    std::vector<unsigned char> out(v.size() * esz);
    if (!ModelRunner::convert(v.data(), RKNN3_TENSOR_FLOAT32, out.data(), dt, (uint32_t)v.size()))
      return false;
    ov[nm] = out;
    return true;
  };
  auto put_f16 = [&](const char* nm, const std::vector<float16>& v,
                     rknn3_tensor_type dt) -> bool {
    size_t esz = ModelRunner::dtype_size(dt);
    if (!esz) return false;
    std::vector<unsigned char> out(v.size() * esz);
    if (!ModelRunner::convert(v.data(), RKNN3_TENSOR_FLOAT16, out.data(), dt, (uint32_t)v.size()))
      return false;
    ov[nm] = out;
    return true;
  };

  if (flavor_a_) {
    // ---- 形态 A：input_ids / attention_mask / position_ids / num_logits_to_keep ----
    std::vector<int64_t> ids(seq, 0);
    std::vector<float> mask(seq, 0.0f);
    for (uint32_t i = 0; i < n_tok; ++i) {
      ids[offset + i] = tokens[i];
      mask[offset + i] = 1.0f;
    }
    const rknn3_tensor_attr* a_mask = find_in("attention_mask");
    const rknn3_tensor_attr* a_pos2 = find_in("position_ids");
    const rknn3_tensor_attr* a_nl = find_in("num_logits_to_keep");
    if (!put_i64("input_ids", ids, find_in("input_ids")->dtype)) {
      printf("[te] input_ids 转换失败\n");
      return false;
    }
    if (!put_f32("attention_mask", mask, a_mask ? a_mask->dtype : RKNN3_TENSOR_FLOAT32)) {
      printf("[te] attention_mask 转换失败\n");
      return false;
    }
    if (a_pos2 && !put_i64("position_ids", pos, a_pos2->dtype)) {
      printf("[te] position_ids 转换失败\n");
      return false;
    }
    if (a_nl) {
      std::vector<int64_t> nl(1, -1);
      if (!put_i64("num_logits_to_keep", nl, a_nl->dtype)) {
        printf("[te] num_logits_to_keep 转换失败\n");
        return false;
      }
    }
  } else {
    // ---- 形态 A2 / B：都用 embedding 表查表拼 input_embeds ----
    std::vector<float16> emb((size_t)seq * dim_);        // 0 填充
    memset(emb.data(), 0, emb.size() * sizeof(float16));
    for (uint32_t i = 0; i < n_tok; ++i) {
      int32_t t = tokens[i];
      if (t < 0 || t >= embed_.rows) {
        printf("[te] token id %d 超出 embedding 表范围 (%d 行)\n", t, embed_.rows);
        return false;
      }
      memcpy(emb.data() + (size_t)(offset + i) * dim_,
             embed_.data + (size_t)t * dim_, (size_t)dim_ * sizeof(float16));
    }
    const rknn3_tensor_attr* a_mask = find_in("attention_mask");
    const rknn3_tensor_attr* a_pos2 = find_in("position_ids");
    const rknn3_tensor_attr* a_nl = find_in("num_logits_to_keep");
    const rknn3_tensor_attr* a_mrope = find_in("mrope_id_input");
    if (!put_f16("input_embeds", emb, find_in("input_embeds")->dtype)) {
      printf("[te] input_embeds 转换失败\n");
      return false;
    }
    std::vector<float> mask(seq, 0.0f);
    for (uint32_t i = 0; i < n_tok; ++i) mask[offset + i] = 1.0f;
    if (a_mask && !put_f32("attention_mask", mask, a_mask->dtype)) {
      printf("[te] attention_mask 转换失败\n");
      return false;
    }
    // position_ids（形态 A2）：diffusers 的 cumsum(mask)-1 规则。
    if (a_pos2 && !put_i64("position_ids", pos, a_pos2->dtype)) {
      printf("[te] position_ids 转换失败\n");
      return false;
    }
    if (a_nl) {
      std::vector<int64_t> nl(1, -1);
      if (!put_i64("num_logits_to_keep", nl, a_nl->dtype)) {
        printf("[te] num_logits_to_keep 转换失败\n");
        return false;
      }
    }
    if (a_mrope) {
      std::vector<int64_t> mrope((size_t)seq * 3, 0);
      for (uint32_t i = 0; i < seq; ++i) {
        mrope[(size_t)i * 3 + 0] = pos[i];
        mrope[(size_t)i * 3 + 1] = pos[i];
        mrope[(size_t)i * 3 + 2] = pos[i];
      }
      if (!put_i64("mrope_id_input", mrope, a_mrope->dtype)) {
        printf("[te] mrope_id_input 转换失败\n");
        return false;
      }
    }

    // rope 常量：必须**原样**写入（不能走通用布局转换）
    static const char* kRopeNames[] = {"rope_cos_cache", "rope_sin_cache", "Th", "Tc", "Ts", "Tsr"};
    for (const char* nm : kRopeNames) {
      if (!find_in(nm)) continue;
      std::string p = rope_dir_.empty() ? std::string(nm) : join_path(rope_dir_, std::string(nm) + ".bin");
      size_t want = 0;
      for (const auto& s : m_.inputs())
        if (!s.fake && s.attr.name == nm)
          want = (size_t)s.attr.n_elems * ModelRunner::dtype_size(s.attr.dtype);
      FILE* f = fopen(p.c_str(), "rb");
      if (!f) {
        printf("[te] 缺少 rope 常量文件: %s\n", p.c_str());
        return false;
      }
      fseek(f, 0, SEEK_END);
      long sz = ftell(f);
      fseek(f, 0, SEEK_SET);
      std::vector<unsigned char> buf((size_t)(sz > 0 ? sz : 0));
      size_t got = fread(buf.data(), 1, buf.size(), f);
      fclose(f);
      if (got != buf.size()) {
        printf("[te] 读 %s 失败\n", p.c_str());
        return false;
      }
      if (buf.size() < want) {
        printf("[te] %s 只有 %zu B，模型需要 %zu B\n", p.c_str(), buf.size(), want);
        return false;
      }
      raw_names.insert(nm);
      ov[nm] = buf;   // 占位，真正写入走 set_raw_input
    }
  }

  // ---- 装载输入 + 跑 ----
  int ret = m_.load_inputs("", true, &ov, &raw_names);
  if (ret != RKNN3_SUCCESS) {
    printf("[te] 装载输入失败 ret=%d\n", ret);
    return false;
  }
  for (const std::string& nm : raw_names) {
    auto it = ov.find(nm);
    if (it == ov.end()) continue;
    if (!m_.set_raw_input(nm, it->second.data(), it->second.size())) {
      printf("[te] set_raw_input(%s) 失败\n", nm.c_str());
      return false;
    }
  }

  double ms = 0.0;
  ret = m_.run(&ms);
  if (ret != RKNN3_SUCCESS) {
    printf("[te] 推理失败 ret=%d\n", ret);
    return false;
  }
  last_ms_ = ms;

  // ---- 取 encoder_hidden_states，做后处理 ----
  std::vector<unsigned char> hs_raw;
  if (!m_.get_output_bytes("encoder_hidden_states", &hs_raw)) {
    printf("[te] 没有 encoder_hidden_states 输出\n");
    return false;
  }
  size_t esz = ModelRunner::dtype_size(a_hs_->dtype);
  if (esz == 0) {
    printf("[te] encoder_hidden_states 的 dtype 不支持\n");
    return false;
  }
  const uint32_t out_seq = (uint32_t)(hs_raw.size() / (esz * dim_));
  std::vector<float16> hs_f16((size_t)keep * dim_);
  if (offset + n_tok > out_seq) {
    printf("[te] 输出 seq (%u) 不足以取 [%u,%u) 的 token\n", out_seq, offset + drop,
           offset + n_tok);
    return false;
  }
  {
    const unsigned char* src = hs_raw.data() + (size_t)(offset + drop) * dim_ * esz;
    if (!ModelRunner::convert(src, a_hs_->dtype, hs_f16.data(), RKNN3_TENSOR_FLOAT16,
                               keep * dim_)) {
      printf("[te] encoder_hidden_states 的 dtype 转换失败\n");
      return false;
    }
  }

  const uint32_t max_seq = kMaxSeq;   // DiT seg0 的 ehs 固定是 [1,128,4096]
  const uint32_t out_n = (keep < max_seq) ? keep : max_seq;
  out_hs_.clear();
  out_hs_.resize((size_t)max_seq * dim_);          // 值初始化 = 全 0
  memcpy(out_hs_.data(), hs_f16.data(), (size_t)out_n * dim_ * sizeof(float16));
  out_mask_.assign((size_t)max_seq, 0);
  memset(out_mask_.data(), 1, out_n);
  return true;
}

// ---------------------------------------------------------------------------
const rknn3_tensor_attr* TeEncoder::find_in(const char* nm) const {
  for (const auto& s : m_.inputs())
    if (!s.fake && s.attr.name && strcmp(s.attr.name, nm) == 0) return &s.attr;
  return nullptr;
}

const rknn3_tensor_attr* TeEncoder::find_out(const char* nm) const {
  for (const auto& s : m_.outputs())
    if (!s.fake && s.attr.name && strcmp(s.attr.name, nm) == 0) return &s.attr;
  return nullptr;
}

}  // namespace qi
