// ---------------------------------------------------------------------------
// wdnai cshim —— Qwen-Image-2.1 出图的 C API 实现
//
// ⭐ 编排照搬参考实现（deploy/src/main.cpp 的 cmd_run_serve），
//    只是把「进程 + CLI + 文件目录」换成「会话 + C ABI」。
//    ⇒ 数值行为与基线**逐位一致**（同一份 flow-match / VAE 代码）。
// ---------------------------------------------------------------------------
#include "api.h"

#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "np_noise.h"
#include "rknn3_runner.h"
#include "resident_chain.h"
#include "te_encoder.h"
#include "util.h"
#include "vae_chain.h"

namespace {

void set_err(char* err, int err_len, const std::string& msg) {
  if (!err || err_len <= 0) return;
  snprintf(err, (size_t)err_len, "%s", msg.c_str());
}

// 512 路线的文件命名（与模型包一致）
const char* kTeModel = "llm_qwenimage21-te.1828.rknn";
const char* kTeWeight = "llm_qwenimage21-te.weight";
const char* kTeEmbed = "qwenimage21-te.embed.bin";
const char* kHeadModel = "dit_qwenimage21-512.head.1828.rknn";
const char* kHeadWeight = "dit_qwenimage21-512.head.weight";
const char* kLedger = "dit_qwenimage21-512hc.int_ledger";

std::string seg_name(const char* stem, int i, const char* ext) {
  char buf[128];
  snprintf(buf, sizeof(buf), "%s.seg%d%s", stem, i, ext);
  return std::string(buf);
}

// 现场生成初始噪声（fp16），与参考实现的 write_fresh_noise **逐位一致**
//   std::mt19937 + std::normal_distribution<float> + qi::fp32_to_fp16_rne
std::vector<uint16_t> fresh_noise(size_t n_elem, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<uint16_t> h(n_elem);
  for (size_t i = 0; i < n_elem; ++i) {
    float16 v = qi::fp32_to_fp16_rne(dist(rng));
    memcpy(&h[i], &v, sizeof(uint16_t));
  }
  return h;
}

// 读 <path>.meta 里的 shape（"# dtype shape\nfloat16 1,4,512,512"）
bool read_meta_shape(const std::string& path, std::vector<int>* shape) {
  FILE* f = fopen(path.c_str(), "r");
  if (!f) return false;
  char line[256];
  bool ok = false;
  while (fgets(line, sizeof(line), f)) {
    if (line[0] == '#' || line[0] == '\n') continue;
    // 取最后一段逗号分隔的 shape
    char* p = strrchr(line, ' ');
    if (!p) continue;
    p++;
    std::string s(p);
    while (!s.empty() && (s[s.size() - 1] == '\n' || s[s.size() - 1] == '\r')) s.erase(s.size() - 1);
    shape->clear();
    size_t pos = 0;
    while (pos <= s.size()) {
      size_t c = s.find(',', pos);
      std::string one = s.substr(pos, c == std::string::npos ? std::string::npos : c - pos);
      if (!one.empty()) shape->push_back(atoi(one.c_str()));
      if (c == std::string::npos) break;
      pos = c + 1;
    }
    ok = !shape->empty();
    break;
  }
  fclose(f);
  return ok;
}

}  // namespace

// ---------------------------------------------------------------------------
struct qi_qwen_session {
  qi::TeEncoder te;
  qi::ResidentChain dit;
  qi::VaeChain vae;
  std::string model_dir;
  std::string work_dir;
  std::string card_te;
  std::string card_dit;
  uint32_t core_mask = 0xFF;
  bool dit_ready = false;
  bool vae_ready = false;

  // ---- 进度回调（可选）----
  // ⚠️ 一个会话只有**一个** qi_progress_ctx —— 各层共享它 ⇒ elapsed 口径全局一致
  qi_progress_ctx prog;
};

// 发一条进度（走**共享**的 ctx ⇒ 与 resident_chain / vae_chain 的口径一致）
static void emit_progress(qi_qwen_session* s, const char* stage, int step, int total) {
  if (!s) return;
  qi_progress_emit(&s->prog, stage, step, total);
}

// 阶段开工：把计时基准挪到现在（见 progress.h 的说明）
static void begin_stage(qi_qwen_session* s, const char* stage) {
  if (!s) return;
  qi_progress_stage(&s->prog, stage);
}

// ---------------------------------------------------------------------------
// 把 DiT 链重新 init 起来（generate_batch 结束后 DiT 被释放，
// 因为 DiT 与 VAE 物理上不能共卡 ⇒ 下一批图要重新 init）
static int init_dit(qi_qwen_session* s, char* err, int err_len) {
  std::vector<std::string> seg_models, seg_weights, device_ids;
  for (int i = 0; i < 4; ++i) {
    seg_models.push_back(
        qi::join_path(s->model_dir, seg_name("dit_qwenimage21-512hc", i, ".1828.rknn")));
    seg_weights.push_back(
        qi::join_path(s->model_dir, seg_name("dit_qwenimage21-512hc", i, ".weight")));
    device_ids.push_back(s->card_dit);
  }
  int ret = s->dit.init(seg_models, seg_weights, s->core_mask, /*share_internal=*/true,
                        &device_ids);
  if (ret != 0) {
    set_err(err, err_len, "DiT 常驻初始化失败");
    return 1;
  }
  ret = s->dit.init_head(qi::join_path(s->model_dir, kHeadModel),
                         qi::join_path(s->model_dir, kHeadWeight),
                         s->card_te.empty() ? nullptr : s->card_te.c_str());
  if (ret != 0) {
    set_err(err, err_len, "DiT 头模型初始化失败");
    return 1;
  }
  s->dit_ready = true;
  return 0;
}

// ---------------------------------------------------------------------------
qi_qwen_session* qi_qwen_open(const char* model_dir, const char* card_te,
                              const char* card_dit, uint32_t core_mask, char* err,
                              int err_len) {
  return qi_qwen_open_ex(model_dir, card_te, card_dit, core_mask, nullptr, nullptr,
                         err, err_len);
}

qi_qwen_session* qi_qwen_open_ex(const char* model_dir, const char* card_te,
                                 const char* card_dit, uint32_t core_mask,
                                 qi_progress_cb cb, void* user, char* err,
                                 int err_len) {
  if (!model_dir) {
    set_err(err, err_len, "model_dir 为空");
    return nullptr;
  }
  qi_qwen_session* s = new qi_qwen_session();
  s->model_dir = model_dir;
  s->card_te = card_te ? card_te : "";
  s->card_dit = card_dit ? card_dit : "";
  s->core_mask = core_mask ? core_mask : 0xFF;
  // ⚠️ 进度回调必须在**加载模型之前**装上 —— 否则那 90 s 的 init
  //    （TE 37 s + DiT 39 s）完全看不到，用户面对的就是一段空白等待。
  if (cb) {
    s->prog.cb = cb;
    s->prog.user = user;
    s->dit.set_progress(&s->prog);
    s->vae.set_progress(&s->prog);
  }

  // ---- 板端铁律：输出不分散（QI_SPREAD_IO=1 会让输出恒为全 0），只分散输入 ----
  if (!getenv("QI_SPREAD_IO")) setenv("QI_SPREAD_IO", "0", 1);
  if (!getenv("QI_SPREAD_IO_IN")) setenv("QI_SPREAD_IO_IN", "1", 1);

  // ---- 共享 internal 的尺寸依据（外部账本 ⇒ 跳过探针，省 25 s）----
  if (!getenv("QI_INT_PER_CORE")) {
    std::string lp = qi::join_path(s->model_dir, kLedger);
    FILE* f = fopen(lp.c_str(), "r");
    if (f) {
      std::string txt;
      char buf[1024];
      size_t n;
      while ((n = fread(buf, 1, sizeof(buf), f)) > 0) txt.append(buf, n);
      fclose(f);
      // 只留数字、逗号、点
      std::string clean;
      for (char c : txt) {
        if ((c >= '0' && c <= '9') || c == '.' || c == ',' || c == ' ' || c == '\n' ||
            c == '\r' || c == '\t')
          clean += c;
      }
      setenv("QI_INT_PER_CORE", clean.c_str(), 1);
      printf("[qi] internal 账本: %s\n", clean.c_str());
    } else {
      printf("[qi] ⚠️ 找不到 internal 账本 %s ⇒ 会跑探针（慢 ~25 s）\n", lp.c_str());
    }
  }

  // ---- 工作目录（临时）----
  char wbuf[256];
  snprintf(wbuf, sizeof(wbuf), "/tmp/wdnai_qwen_%d", (int)getpid());
  s->work_dir = wbuf;
  std::string mk = "mkdir -p '" + s->work_dir + "'";
  if (system(mk.c_str()) != 0) {
    set_err(err, err_len, "创建工作目录失败: " + s->work_dir);
    delete s;
    return nullptr;
  }

  // ---- 1. TE（常驻；36.8 s 的 model_init 在这里）----
  begin_stage(s, "init");
  emit_progress(s, "init", 1, 3);
  int ret = s->te.open(qi::join_path(s->model_dir, kTeModel),
                       qi::join_path(s->model_dir, kTeWeight),
                       qi::join_path(s->model_dir, kTeEmbed), s->core_mask,
                       s->card_te.empty() ? nullptr : s->card_te.c_str(), "", 0);
  if (ret != 0) {
    set_err(err, err_len, "TE 初始化失败");
    delete s;
    return nullptr;
  }

  // ---- 2. DiT：4 段（切头单卡，共享 internal）+ 头模型（另一张卡）----
  emit_progress(s, "init", 2, 3);
  if (init_dit(s, err, err_len) != 0) {
    delete s;
    return nullptr;
  }
  emit_progress(s, "init", 3, 3);
  return s;
}

// ---------------------------------------------------------------------------
void qi_qwen_close(qi_qwen_session* s) {
  if (!s) return;
  s->dit.release();
  s->vae.release();
  delete s;
}

void qi_qwen_set_progress(qi_qwen_session* s, qi_progress_cb cb, void* user) {
  if (!s) return;
  s->prog.cb = cb;
  s->prog.user = user;
  qi_progress_stage(&s->prog, nullptr);  // 换了回调 ⇒ 计时重新起算
  // DiT / VAE 也要拿到**同一个** ctx ⇒ flow-match 每一步、VAE 每一段
  // 报出来的 elapsed 才和会话层的事件对得上（口径统一）。
  s->dit.set_progress(&s->prog);
  s->vae.set_progress(&s->prog);
}

// ---------------------------------------------------------------------------
int qi_qwen_te_encode(qi_qwen_session* s, const int32_t* tokens, int n_tokens,
                      uint16_t** out_ehs, int* out_ehs_n, uint8_t** out_mask,
                      int* out_mask_n, char* err, int err_len) {
  if (!s || !tokens || n_tokens <= 0) {
    set_err(err, err_len, "参数错误");
    return 1;
  }
  if (!s->te.encode(tokens, n_tokens)) {
    set_err(err, err_len, "TE 编码失败");
    return 1;
  }
  size_t hn = s->te.hs_count();
  uint16_t* h = (uint16_t*)malloc(hn * sizeof(uint16_t));
  memcpy(h, s->te.hs(), hn * sizeof(uint16_t));
  uint8_t* m = (uint8_t*)malloc(s->te.mask_count());
  memcpy(m, s->te.mask(), s->te.mask_count());
  *out_ehs = h;
  *out_ehs_n = (int)hn;
  *out_mask = m;
  *out_mask_n = (int)s->te.mask_count();
  return 0;
}

// ---------------------------------------------------------------------------
int qi_qwen_generate_batch(qi_qwen_session* s, int n_images,
                           const int32_t* all_tokens, const int32_t* all_n_tokens,
                           int steps, double mu, double shift_terminal, int noise_mode,
                           uint32_t seed, uint8_t** out_rgb, int* out_w, int* out_h,
                           char* err, int err_len) {
  if (!s || n_images <= 0 || !all_tokens || !all_n_tokens) {
    set_err(err, err_len, "参数错误");
    return 1;
  }
  if (!s->dit_ready) {
    // 上一批图结束后 DiT 被释放（不能与 VAE 共卡）⇒ 这里重新 init
    if (init_dit(s, err, err_len) != 0) {
      return 1;
    }
  }
  const bool debug_dump = getenv("QI_DEBUG_DUMP") != nullptr;

  // ---- ① 逐张 TE 编码 + ② 逐张 flow-match ----
  std::vector<std::vector<unsigned char>> latents((size_t)n_images);
  int64_t tok_off = 0;
  for (int img = 0; img < n_images; ++img) {
    int n_tok = all_n_tokens[img];
    const int32_t* toks = all_tokens + tok_off;
    tok_off += n_tok;

    uint16_t* ehs = nullptr;
    uint8_t* mask = nullptr;
    int ehs_n = 0, mask_n = 0;
    // 进度：TE 编码。**开工**重置计时，**干完**再报 ⇒ 上层看到的
    // elapsed 就是「这次编码花了多久」（而不是「worker 空等了多久」）。
    begin_stage(s, "te");
    if (qi_qwen_te_encode(s, toks, n_tok, &ehs, &ehs_n, &mask, &mask_n, err, err_len) != 0) {
      return 1;
    }
    emit_progress(s, "te", img + 1, n_images);

    // 工作目录：每张一份（runner 按文件名读输入）
    char sub[256];
    snprintf(sub, sizeof(sub), "/img%02d", img + 1);
    std::string dir = s->work_dir + sub;
    std::string mk = "mkdir -p '" + dir + "'";
    if (system(mk.c_str()) != 0) {
      set_err(err, err_len, "创建图片工作目录失败");
      qi_qwen_free(ehs); qi_qwen_free(mask);
      return 1;
    }
    qi::write_file(qi::join_path(dir, "encoder_hidden_states.bin"), ehs,
                   (size_t)ehs_n * sizeof(uint16_t));
    qi::write_file(qi::join_path(dir, "encoder_hidden_states_mask.bin"), mask,
                   (size_t)mask_n);
    if (debug_dump) {
      // 调试：把 TE 的输出单独落盘（与参考实现的 din/ 目录可比对）
      qi::write_file(qi::join_path(dir, "ehs_dump.bin"), ehs,
                     (size_t)ehs_n * sizeof(uint16_t));
      qi::write_file(qi::join_path(dir, "mask_dump.bin"), mask, (size_t)mask_n);
    }
    qi_qwen_free(ehs);
    qi_qwen_free(mask);

    // 初始噪声：固定（参考噪声）或现场生成
    const size_t n_elem = 1024 * 64;   // 512 路线：1024 token × 64 通道
    std::string x_path = qi::join_path(dir, "hidden_states.bin");
    if (noise_mode == QI_NOISE_FRESH) {
      std::vector<uint16_t> h = fresh_noise(n_elem, seed + (uint32_t)img);
      qi::write_file(x_path, h.data(), h.size() * sizeof(uint16_t));
    } else {
      // 固定噪声：**现场生成**（numpy 兼容的 PCG64 + ziggurat），
      // 与历史基线 reference_512/hidden_states.bin **逐字节一致**
      // ⇒ 模型包里不需要再带这个文件。
      // ⚠️ QI_NOISE_FILE 可强制改用外部文件（A/B 对拍 / 回退用）。
      const char* force_file = getenv("QI_NOISE_FILE");
      if (force_file && force_file[0]) {
        std::vector<unsigned char> buf;
        if (!qi::read_whole_file(force_file, &buf) || buf.size() != n_elem * sizeof(uint16_t)) {
          set_err(err, err_len, "读指定噪声文件失败: " + std::string(force_file));
          return 1;
        }
        qi::write_file(x_path, buf.data(), buf.size());
      } else {
        std::vector<uint16_t> h;
        if (!qi::np_fixed_noise_512(&h)) {
          set_err(err, err_len, "生成固定噪声失败");
          return 1;
        }
        qi::write_file(x_path, h.data(), h.size() * sizeof(uint16_t));
      }
    }

    begin_stage(s, "flow");
    std::string lat_path = qi::join_path(dir, "latents.bin");
    std::string dump_dir = debug_dump ? dir : std::string();
    int ret = s->dit.run_flowmatch(dir, steps, mu, shift_terminal, x_path, lat_path,
                                   dump_dir);
    if (ret != 0) {
      set_err(err, err_len, "flow-match 失败");
      return 1;
    }
    if (!qi::read_whole_file(lat_path, &latents[(size_t)img])) {
      set_err(err, err_len, "读潜变量失败");
      return 1;
    }
  }

  // ---- ③ 释放 DiT（与 VAE 物理上不能共卡）----
  begin_stage(s, "release");
  emit_progress(s, "release", 1, 1);
  s->dit.release();
  s->dit_ready = false;

  // ---- ④ VAE 4 段（流式）→ 归一化成 uint8 RGB ----
  std::vector<std::string> vae_models, vae_weights;
  for (int i = 0; i < 4; ++i) {
    vae_models.push_back(qi::join_path(s->model_dir, seg_name("vae_qwenimage21-512", i, ".1828.rknn")));
    vae_weights.push_back(qi::join_path(s->model_dir, seg_name("vae_qwenimage21-512", i, ".weight")));
  }
  int ret = s->vae.init(vae_models, vae_weights, s->core_mask,
                        s->card_dit.empty() ? nullptr : s->card_dit.c_str(),
                        /*prefer_resident=*/false);
  if (ret != 0) {
    set_err(err, err_len, "VAE 初始化失败: " + s->vae.error());
    return 1;
  }
  s->vae_ready = true;
  begin_stage(s, "vae_init");
  emit_progress(s, "vae_init", 1, 1);

  const size_t px = (size_t)512 * 512 * 3;
  uint8_t* all_rgb = (uint8_t*)malloc(px * (size_t)n_images);
  if (!all_rgb) {
    set_err(err, err_len, "内存不足");
    return 1;
  }
  for (int img = 0; img < n_images; ++img) {
    char pref[256];
    snprintf(pref, sizeof(pref), "%s/img%02d", s->work_dir.c_str(), img + 1);
    double ms = 0.0;
    begin_stage(s, "vae");
    ret = s->vae.decode(latents[(size_t)img], /*channels=*/0, s->work_dir, pref, &ms);
    if (ret != 0) {
      set_err(err, err_len, "VAE 解码失败: " + s->vae.error());
      qi_qwen_free(all_rgb);
      return 1;
    }
    // 读回 <prefix>_image.bin + .meta，按 NCHW 取通道 0/1/2 → 归一化 → uint8
    std::string img_bin = std::string(pref) + "_image.bin";
    std::vector<unsigned char> raw;
    if (!qi::read_whole_file(img_bin, &raw)) {
      set_err(err, err_len, "读 VAE 输出失败: " + img_bin);
      qi_qwen_free(all_rgb);
      return 1;
    }
    std::vector<int> shape;
    if (!read_meta_shape(img_bin + ".meta", &shape) || shape.size() < 4) {
      set_err(err, err_len, "VAE 输出 meta 缺失/格式不对");
      qi_qwen_free(all_rgb);
      return 1;
    }
    const int c = shape[1], h = shape[2], w = shape[3];
    const size_t n = (size_t)c * h * w;
    if (raw.size() != n * sizeof(uint16_t)) {
      set_err(err, err_len, "VAE 输出大小与 meta 不符");
      qi_qwen_free(all_rgb);
      return 1;
    }
    std::vector<float> f(n);
    for (size_t i = 0; i < n; ++i) {
      float16 hv;
      memcpy(&hv, raw.data() + i * sizeof(uint16_t), sizeof(uint16_t));
      f[i] = qi::fp16_to_fp32_exact(hv);
    }
    // 通道主序 (c, h, w) → 取前 3 通道 → RGB
    uint8_t* dst = all_rgb + px * (size_t)img;
    float lo = 1e30f;
    std::vector<float> rgb((size_t)h * w * 3);
    for (int ch = 0; ch < 3 && ch < c; ++ch) {
      for (size_t p = 0; p < (size_t)h * w; ++p) {
        float v = f[(size_t)ch * h * w + p];
        rgb[(size_t)p * 3 + ch] = v;
        if (v < lo) lo = v;
      }
    }
    const bool norm01 = (lo < -0.01f);
    for (size_t i = 0; i < rgb.size(); ++i) {
      float v = norm01 ? (rgb[i] + 1.0f) * 127.5f : rgb[i] * 255.0f;
      if (v < 0.0f) v = 0.0f;
      if (v > 255.0f) v = 255.0f;
      dst[i] = (uint8_t)v;   // 与 numpy astype(uint8) 一样是截断
    }
    if (img + 1 == n_images) {
      *out_w = w;
      *out_h = h;
    }
  }
  *out_rgb = all_rgb;
  s->vae.release();
  s->vae_ready = false;
  begin_stage(s, "done");
  emit_progress(s, "done", 1, 1);
  return 0;
}

// ---------------------------------------------------------------------------
void qi_qwen_free(void* p) { free(p); }

int qi_qwen_fixed_noise(uint16_t** out_fp16, int* n, char* err, int err_len) {
  if (!out_fp16 || !n) {
    set_err(err, err_len, "参数为空");
    return 1;
  }
  std::vector<uint16_t> h;
  if (!qi::np_fixed_noise_512(&h)) {
    set_err(err, err_len, "生成固定噪声失败");
    return 1;
  }
  uint16_t* buf = (uint16_t*)malloc(h.size() * sizeof(uint16_t));
  if (!buf) {
    set_err(err, err_len, "分配内存失败");
    return 1;
  }
  memcpy(buf, h.data(), h.size() * sizeof(uint16_t));
  *out_fp16 = buf;
  *n = (int)h.size();
  return 0;
}

const char* qi_qwen_version(void) { return "wdnai-cshim 1.0 (Qwen-Image-2.1, rk1820)"; }
