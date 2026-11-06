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
#include "vae_stats.h"
#include "vae_chain.h"
#include "vae_rknn2.h"

namespace {

// ⚠️ stdout 在**重定向到文件**时默认是**全缓冲**（4~8 KB）⇒ 进程被
//    SIGSEGV / SIGKILL 掉时，缓冲区里的日志**全部丢失** ⇒ 现场看起来
//    「worker 莫名其妙就死了，没有任何线索」。
//    改成行缓冲 ⇒ 每行 printf 立刻落盘，崩溃点前最后一行就是断点。
struct StdoutLineBuffer {
  StdoutLineBuffer() { setvbuf(stdout, nullptr, _IOLBF, 0); }
} g_stdout_line_buffer;

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

// 「VAE 没有独占设备」时的统一报错文案（open 阶段的**秒级**检查与
//  真正 init VAE 之前各用一次 ⇒ 两处文案不会漂移）。
static const char* kNoDedicatedVae =
    "VAE 没有可独占的 NPU 设备：模型包里既没有 vae_qwenimage21-512.3588.rknn"
    "（本机 3588 跑 rknn2），也没给第三张 1828 卡（跑 rknn3 的 4 段 VAE）。"
    "**不再退回「与 DiT 共卡」** —— 那会让每批图多 39 s 的 DiT 重建。";

// ⭐ rknn2（**本机 RK3588 NPU**）上的 VAE 解码器。
//   命名跟模型包里的其它文件统一成「后缀 = 跑在哪块 SoC」：
//     整图版：vae_qwenimage21-512.3588.rknn
//     分段版：vae_qwenimage21-512.seg{0..N}.3588.rknn
//   （与 DiT/TE 的 .1828.rknn 同一套命名规则）
//   旧的 `vae_qwenimage21-512.rknn2.rknn` / `.rknn2.seg{0..N}.rknn` 仍兼容，
//   老模型包不用重新打包。
//   ⚠️ 整图优先；没有整图就按 seg0/seg1/... 找分段。
constexpr int kVae2MaxSegs = 8;

// 在 model_dir 里找 rknn2 的 VAE 模型（整图优先，其次分段）
static std::vector<std::string> find_vae2_models(const std::string& model_dir) {
  std::vector<std::string> out;
  // ---- 整图（按优先级）----
  static const char* kWholeNames[] = {
      "vae_qwenimage21-512.3588.rknn",   // 新命名：.3588 = 跑在 RK3588 上
      "vae_qwenimage21-512.rknn2.rknn",   // 旧命名（兼容老模型包）
  };
  for (const char* name : kWholeNames) {
    std::string p = qi::join_path(model_dir, name);
    if (access(p.c_str(), R_OK) == 0) {
      out.push_back(p);
      return out;
    }
  }
  // ---- 分段流式 ----
  struct SegForm { const char* stem; const char* suffix; };
  static const SegForm kSegForms[] = {
      {"vae_qwenimage21-512.seg", ".3588.rknn"},   // 新命名
      {"vae_qwenimage21-512.rknn2.seg", ".rknn"},  // 旧命名（兼容老模型包）
  };
  for (const SegForm& form : kSegForms) {
    std::vector<std::string> segs;
    for (int i = 0; i < kVae2MaxSegs; ++i) {
      char buf[128];
      snprintf(buf, sizeof(buf), "%s%d%s", form.stem, i, form.suffix);
      std::string p = qi::join_path(model_dir, buf);
      if (access(p.c_str(), R_OK) != 0) break;
      segs.push_back(p);
    }
    if (!segs.empty()) return segs;
  }
  return out;
}

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
  qi::VaeChain vae;      // rknn3 的 VAE 链（1828 上，**独占第三张卡** ⇒ 与 DiT 互不干扰）
  qi::VaeChain2 vae2;    // rknn2 的 VAE 链（**本机 RK3588 NPU** ⇒ 与 DiT 物理分开）
  qi::ModelRunner2 vae_enc;  // ⭐ VAE **编码器**（rknn2，本机 NPU）—— 图生图用
  std::string model_dir;
  std::string work_dir;
  std::string card_te;
  std::string card_dit;
  std::string card_vae;   // rknn3 VAE 独占的第三张 1828 卡（空 ⇒ 只能走 rknn2）
  uint32_t core_mask = 0xFF;
  uint32_t vae_core_mask = 0;   // 0 ⇒ 默认（3 核全用）
  bool dit_ready = false;
  bool vae_ready = false;
  bool vae2_ready = false;      // ⭐ rknn2 VAE 可用 ⇒ DiT 全程常驻
  bool vae_resident = false;    // rknn3 VAE 链也常驻（独占第三张卡时）
  bool vae_enc_ready = false;   // ⭐ VAE 编码器可用 ⇒ 支持图生图

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

// rknn3 错误码 → 短说明（只列最常撞的几个；nullptr = 没收录）
static const char* rknn3_err_text(int ret) {
  switch (ret) {
    case RKNN3_ERR_DEVICE_UNAVAILABLE:
      return "设备不可用（PCI 地址不对 / 驱动没认到这张卡）";
    case RKNN3_ERR_DEVICE_UNMATCH:
      return "设备不匹配";
    case RKNN3_ERR_OUT_OF_MEMORY:
      return "显存不足";
    case RKNN3_ERR_MODEL_INVALID:
      return "模型无效";
    case RKNN3_ERR_COMMUNICATION:
      return "与卡端通信失败";
    default:
      return nullptr;
  }
}

// ---------------------------------------------------------------------------
// 把 DiT 链 init 起来（**只在 open 阶段调用一次**）
//
// ⚠️ 旧拓扑（VAE 与 DiT 共卡）里，generate_batch 结束会释放 DiT、下一批
//    再 init 回来（每批多 39 s model_init）。现在 VAE 必须独占设备，
//    DiT **全程常驻** ⇒ 这个函数**不再**有「重建」用途；
//    generate_batch 里 dit_ready=false 会**直接报错**（拒绝推理中重建模型）。
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
  return qi_qwen_open_ex2(model_dir, card_te, card_dit, core_mask, 0, cb, user, err,
                          err_len);
}

qi_qwen_session* qi_qwen_open_ex2(const char* model_dir, const char* card_te,
                                  const char* card_dit, uint32_t core_mask,
                                  uint32_t vae_core_mask, qi_progress_cb cb, void* user,
                                  char* err, int err_len) {
  // 老接口：没有第三张卡 ⇒ 只能走 rknn2（本机 3588）
  return qi_qwen_open_ex3(model_dir, card_te, card_dit, "", core_mask, vae_core_mask,
                           cb, user, err, err_len);
}

qi_qwen_session* qi_qwen_open_ex3(const char* model_dir, const char* card_te,
                                  const char* card_dit, const char* card_vae,
                                  uint32_t core_mask,
                                  uint32_t vae_core_mask, qi_progress_cb cb, void* user,
                                  char* err, int err_len) {
  if (!model_dir) {
    set_err(err, err_len, "model_dir 为空");
    return nullptr;
  }
  // ---- 0. ⭐ **秒级**检查「VAE 有没有独占设备」----
  //   为什么要放在最前面：VAE 的可用性**只取决于模型包里有没有那个文件**
  //   （加上有没有第三张卡）。如果等到 TE(37 s) + DiT(60 s) 全部加载完才发现
  //   VAE 缺失，用户要**白等 2 分钟**才知道开不了工。
  //   这里只是几个 access()，~0 ms —— 不合格就立刻失败。
  {
    std::vector<std::string> vae2_probe = find_vae2_models(model_dir);
    bool has_rknn3_vae = false;
    if (card_vae && card_vae[0]) {
      has_rknn3_vae = true;
      for (int i = 0; i < 4; ++i) {
        std::string p = qi::join_path(
            model_dir, seg_name("vae_qwenimage21-512", i, ".1828.rknn"));
        if (access(p.c_str(), R_OK) != 0) { has_rknn3_vae = false; break; }
      }
    }
    if (vae2_probe.empty() && !has_rknn3_vae) {
      set_err(err, err_len, kNoDedicatedVae);
      return nullptr;
    }
  }
  qi_qwen_session* s = new qi_qwen_session();
  s->model_dir = model_dir;
  s->card_te = card_te ? card_te : "";
  s->card_dit = card_dit ? card_dit : "";
  s->card_vae = card_vae ? card_vae : "";
  s->core_mask = core_mask ? core_mask : 0xFF;
  s->vae_core_mask = vae_core_mask;
  // ⚠️ 进度回调必须在**加载模型之前**装上 —— 否则那 90 s 的 init
  //    （TE 37 s + DiT 39 s）完全看不到，用户面对的就是一段空白等待。
  if (cb) {
    s->prog.cb = cb;
    s->prog.user = user;
    s->dit.set_progress(&s->prog);
    s->vae.set_progress(&s->prog);
    s->vae2.set_progress(&s->prog);
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
    // ⭐ 带上 **rknn3 原始 ret 码 + device_id** —— 只说「TE 初始化失败」
    //    的话，排查的人只能靠复现猜是哪张卡、哪一步炸的。
    char msg[320];
    const char* dev = s->card_te.empty() ? "(默认 device[0])" : s->card_te.c_str();
    const char* why = rknn3_err_text(ret);
    if (why) {
      snprintf(msg, sizeof(msg), "TE 初始化失败：ret=%d（%s），device_id=%s", ret, why, dev);
    } else {
      snprintf(msg, sizeof(msg), "TE 初始化失败：ret=%d，device_id=%s", ret, dev);
    }
    set_err(err, err_len, msg);
    delete s;
    return nullptr;
  }

  // ---- 2. DiT：4 段（切头单卡，共享 internal）+ 头模型（另一张卡）----
  emit_progress(s, "init", 2, 3);
  if (init_dit(s, err, err_len) != 0) {
    delete s;
    return nullptr;
  }

  // ---- 3. ⭐ VAE：**必须独占一个 NPU 设备** ----
  //   两种支持的拓扑（都由**模型包里的文件名**决定，不由运行时猜）：
  //     A) 2×1828 + 3588：`.3588.rknn`（rknn2）⇒ VAE 跑**本机 RK3588 的 NPU**
  //        ⇒ 与 DiT 物理分开，DiT 全程常驻。
  //     B) 3×1828：`.seg{0..3}.1828.rknn`（rknn3）⇒ VAE 独占**第三张卡**。
  //   ⚠️ **不再支持「VAE 与 DiT 共卡」** —— 那要求推理前释放 DiT、推理后重建，
  //      每批图多 39 s model_init。现在没有独占设备就在 open 阶段**明确失败**。
  {
    std::vector<std::string> vae2_models = find_vae2_models(s->model_dir);
    if (!vae2_models.empty()) {
      emit_progress(s, "init", 3, 3);
      int r2 = s->vae2.init(vae2_models, s->vae_core_mask, /*prefer_resident=*/true);
      if (r2 != 0) {
        set_err(err, err_len,
                std::string("rknn2 VAE 初始化失败（") + s->vae2.error() +
                    "）⇒ 本机 3588 的 NPU 跑不了这个 VAE；"
                    "请检查 rknn2-runtime（apt install rknn2-runtime）是否装好");
        s->vae2.release();
        delete s;
        return nullptr;
      }
      s->vae2_ready = true;
      printf("[qi] ⭐ VAE 走本机 rknn2（RK3588 NPU，%zu 段）⇒ DiT 全程常驻\n",
             vae2_models.size());
    } else if (!s->card_vae.empty()) {
      // ---- 拓扑 B：3×1828，VAE 独占第三张卡 ----
      emit_progress(s, "init", 3, 3);
      std::vector<std::string> vae_models, vae_weights;
      for (int i = 0; i < 4; ++i) {
        vae_models.push_back(
            qi::join_path(s->model_dir, seg_name("vae_qwenimage21-512", i, ".1828.rknn")));
        vae_weights.push_back(
            qi::join_path(s->model_dir, seg_name("vae_qwenimage21-512", i, ".weight")));
      }
      int rv = s->vae.init(vae_models, vae_weights, s->core_mask,
                            s->card_vae.c_str(), /*prefer_resident=*/true);
      if (rv != 0) {
        set_err(err, err_len,
                std::string("rknn3 VAE（独占第三张卡 ") + s->card_vae +
                    "）初始化失败：" + s->vae.error());
        s->vae.release();
        delete s;
        return nullptr;
      }
      s->vae_ready = true;
      s->vae_resident = true;
      printf("[qi] ⭐ VAE 走第三张 1828 卡（%s）⇒ 与 DiT 互不干扰，全程常驻\n",
             s->card_vae.c_str());
    } else {
      // （理论上走不到 —— open 开头的秒级检查已经拦下了；留作兜底）
      set_err(err, err_len, kNoDedicatedVae);
      delete s;
      return nullptr;
    }
  }
  emit_progress(s, "init", 3, 3);

  // ---- 4. ⭐ VAE **编码器**（图生图用；**可选**）----
  //   模型包里有 `vae_qwenimage21-512.encoder.3588.rknn` 才装 ——
  //   没有它**不影响文生图**（只是图生图不可用）。
  //   ⚠️ 它和解码器**同在本机 3588 的 NPU** 上，但实测**能同时常驻**
  //   （解码器 2311 MiB + 编码器 ≈ 0.9 GiB < 4 GiB 池）。
  {
    const std::string enc_path =
        qi::join_path(s->model_dir, "vae_qwenimage21-512.encoder.3588.rknn");
    std::vector<unsigned char> probe;
    if (qi::read_whole_file(enc_path, &probe)) {
      emit_progress(s, "init", 3, 3);
      int re = s->vae_enc.init(enc_path, s->vae_core_mask);
      if (re != 0) {
        // ⚠️ **不致命** —— 编码器装不上只是「图生图不可用」，
        //    文生图必须照常能跑（别因为一个可选件把整张卡拖死）。
        printf("[qi] ⚠️ VAE 编码器初始化失败（%s）⇒ 本机不支持图生图\n",
               s->vae_enc.error().c_str());
        s->vae_enc.release();
      } else {
        s->vae_enc_ready = true;
        printf("[qi] ⭐ VAE 编码器就绪（本机 rknn2）⇒ 支持图生图\n");
      }
    }
  }
  return s;
}

// ---------------------------------------------------------------------------
void qi_qwen_close(qi_qwen_session* s) {
  if (!s) return;
  s->dit.release();
  s->vae.release();
  s->vae2.release();
  s->vae_enc.release();
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
  s->vae2.set_progress(&s->prog);
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
    // ⚠️ **不允许在推理过程中卸载/重装载模型**。
    //    正常情况下 open 阶段就把 TE / DiT / VAE 全部装好并常驻，
    //    这里只可能是因为上一轮有人手动释放过 ⇒ 直接报错，不偷偷重建。
    set_err(err, err_len,
            "DiT 不在常驻状态（dit_ready=false）—— **拒绝推理中重建模型**。"
            "请重新打开会话（open 阶段会一次性把 TE/DiT/VAE 全部装好）。");
    return 1;
  }
  if (!s->vae2_ready && !s->vae_ready) {
    set_err(err, err_len,
            "VAE 不在常驻状态（既没有 rknn2 的 3588 VAE，也没有独占卡的 rknn3 VAE）"
            "—— **拒绝推理中重建模型**。");
    return 1;
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

  // ==================================================================
  // ③ + ④ VAE 解码
  //
  // ⭐ 两条路：
  //   A) **rknn2**（本机 RK3588 NPU）：DiT 与 VAE **物理分开**
  //      ⇒ DiT **全程常驻**，不释放、不重建 ⇒ 生成过程零 model_init
  //   B) **rknn3**（1828，与 DiT 同卡）：必须先释放 DiT，VAE 流式加载
  //      （老路径，行为与改造前完全一致）
  // ==================================================================
  const size_t px = (size_t)512 * 512 * 3;
  uint8_t* all_rgb = (uint8_t*)malloc(px * (size_t)n_images);
  if (!all_rgb) {
    set_err(err, err_len, "内存不足");
    return 1;
  }

  if (s->vae2_ready) {
    // ---- A) rknn2：本机 NPU，DiT 常驻 ----
    for (int img = 0; img < n_images; ++img) {
      std::vector<unsigned char> rgb;
      int w = 0, h = 0;
      double ms = 0.0;
      begin_stage(s, "vae");
      // ⭐ 先发一条「进入解码阶段」的事件（step=0 ⇒ 进度条不提前跑满）。
      //    不发的话，上层在整图解码的十几秒里**看不到任何阶段变化** ——
      //    卡片会一直停在「flow 8/8」，ETA 也算不出来。
      emit_progress(s, "vae", 0, 1);
      int r2 = s->vae2.decode(latents[(size_t)img], /*channels=*/64, &rgb, &w, &h, &ms);
      if (r2 != 0) {
        set_err(err, err_len, "VAE(rknn2) 解码失败: " + s->vae2.error());
        qi_qwen_free(all_rgb);
        return 1;
      }
      uint8_t* dst = all_rgb + px * (size_t)img;
      if ((size_t)w * h * 3 == rgb.size()) {
        memcpy(dst, rgb.data(), rgb.size());
      } else {
        set_err(err, err_len, "VAE(rknn2) 输出尺寸异常");
        qi_qwen_free(all_rgb);
        return 1;
      }
      if (img + 1 == n_images) {
        *out_w = w;
        *out_h = h;
      }
    }
    *out_rgb = all_rgb;
    // ⭐ 不释放 DiT！下一批图可以直接用 ⇒ 零 model_init
    begin_stage(s, "done");
    emit_progress(s, "done", 1, 1);
    return 0;
  }

  // ---- B) rknn3 路径：VAE 独占第三张 1828 卡，**open 阶段就已常驻** ----
  //   ⚠️ 这里**不再**释放 DiT、**不再**init VAE —— 全部在 open 阶段做完。
  //   推理过程中一律不卸载/重装载模型。
  if (!s->vae_ready) {
    set_err(err, err_len, "rknn3 VAE 链不可用（open 阶段应已初始化并常驻）");
    qi_qwen_free(all_rgb);
    return 1;
  }

  for (int img = 0; img < n_images; ++img) {
    char pref[256];
    snprintf(pref, sizeof(pref), "%s/img%02d", s->work_dir.c_str(), img + 1);
    double ms = 0.0;
    int ret = 0;
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
  // ⭐ **不释放 VAE** —— 它独占第三张卡，与 DiT 互不干扰 ⇒ 下一批直接用，
  //    整个会话生命周期内**零 model_init**。
  begin_stage(s, "done");
  emit_progress(s, "done", 1, 1);
  return 0;
}

// ===========================================================================
// ⭐⭐ 图生图（img2img）
//
//   输入图 --[VAE 编码器]--> z₀ --[加噪到起步点]--> x --[flow-match]--> z
//          --[VAE 解码器]--> 新图
//
//   ⚠️ 与文生图的**唯一**差别：flow-match 的**起步点**不是「纯噪声 σ=1」，
//      而是「原图潜变量加噪到 σ[k₀]」。
// ===========================================================================

// 把输入图编码成潜变量 z₀（**token 主序 fp16**，与 hidden_states 同布局）
static bool encode_source_image(qi_qwen_session* s, const uint8_t* rgb, int w, int h,
                                 std::vector<unsigned char>* z0_fp16,
                                 char* err, int err_len) {
  if (!s->vae_enc_ready) {
    set_err(err, err_len,
            "本机不支持图生图：模型包里没有 VAE **编码器**"
            "（vae_qwenimage21-512.encoder.3588.rknn）");
    return false;
  }
  if (w != 512 || h != 512) {
    set_err(err, err_len,
            "输入图必须是 512×512（VAE 的输入尺寸固定；缩放由调用方用 RGA 做）");
    return false;
  }
  // ---- RGB888 → RGBA（alpha=1）→ fp32 [-1,1] → [1,4,1,512,512] ----
  // ⚠️ 模型的输入张量是 **4 通道**（Qwen-Image-2.1 的 VAE 是 RGBA 原生的），
  //    布局 NCDHW ⇒ 内存里是「逐通道连续」⇒ 按通道分段填正好对上。
  const size_t px = (size_t)512 * 512;
  std::vector<float> pix(4 * px);
  for (size_t i = 0; i < px; ++i) {
    pix[0 * px + i] = rgb[i * 3 + 0] / 255.0f * 2.0f - 1.0f;
    pix[1 * px + i] = rgb[i * 3 + 1] / 255.0f * 2.0f - 1.0f;
    pix[2 * px + i] = rgb[i * 3 + 2] / 255.0f * 2.0f - 1.0f;
    pix[3 * px + i] = 1.0f;   // alpha = 不透明
  }
  int ret = s->vae_enc.set_input("pixel_values", pix.data(),
                                 pix.size() * sizeof(float), RKNN_TENSOR_FLOAT32);
  if (ret != 0) {
    set_err(err, err_len, "VAE 编码器 set_input 失败: " + s->vae_enc.error());
    return false;
  }
  double ms = 0.0;
  begin_stage(s, "encode");
  emit_progress(s, "encode", 1, 1);
  ret = s->vae_enc.run(&ms);
  if (ret != 0) {
    set_err(err, err_len, "VAE 编码器推理失败: " + s->vae_enc.error());
    return false;
  }
  std::vector<unsigned char> raw;
  rknn_tensor_type dt = RKNN_TENSOR_FLOAT16;
  if (!s->vae_enc.get_output("latents", &raw, &dt)) {
    set_err(err, err_len, "VAE 编码器输出读取失败: " + s->vae_enc.error());
    return false;
  }
  const size_t n_elem = 1024 * 64;   // 1024 token × 64 通道
  if (raw.size() != n_elem * 2) {
    set_err(err, err_len, "VAE 编码器输出大小不对");
    return false;
  }
  // fp16 → fp32
  std::vector<float> f32(n_elem);
  if (!qi::ModelRunner::convert(raw.data(), RKNN3_TENSOR_FLOAT16,
                                (void*)f32.data(), RKNN3_TENSOR_FLOAT32,
                                (uint32_t)n_elem)) {
    set_err(err, err_len, "VAE 编码器输出类型转换失败");
    return false;
  }
  // ---- 通道主序 (C=64, T=1024) → 归一化 → token 主序 (T=1024, C=64) ----
  // 归一化：z = (raw − mean) / std（解码器那边是它的逆：v·std + mean）
  std::vector<float> norm(n_elem);
  for (int t = 0; t < 1024; ++t) {
    for (int c = 0; c < 64; ++c) {
      float raw_v = f32[(size_t)c * 1024 + (size_t)t];
      norm[(size_t)t * 64 + (size_t)c] =
          (raw_v - qi::kVaeLatentsMean[c]) / qi::kVaeLatentsStd[c];
    }
  }
  z0_fp16->resize(n_elem * 2);
  if (!qi::ModelRunner::convert(norm.data(), RKNN3_TENSOR_FLOAT32,
                                z0_fp16->data(), RKNN3_TENSOR_FLOAT16,
                                (uint32_t)n_elem)) {
    set_err(err, err_len, "潜变量转 fp16 失败");
    return false;
  }
  printf("[qi] VAE 编码器完成（%.1f ms）⇒ z₀ 已就绪\n", ms);
  return true;
}

int qi_qwen_img2img_batch(qi_qwen_session* s, int n_images,
                           const int32_t* all_tokens, const int32_t* all_n_tokens,
                           int steps, double mu, double shift_terminal, int noise_mode,
                           uint32_t seed, const uint8_t* src_rgb, int src_w, int src_h,
                           double strength, uint8_t** out_rgb, int* out_w, int* out_h,
                           char* err, int err_len) {
  if (!s || n_images <= 0 || !all_tokens || !all_n_tokens || !src_rgb) {
    set_err(err, err_len, "参数错误");
    return 1;
  }
  if (!s->dit_ready) {
    set_err(err, err_len, "DiT 不在常驻状态（dit_ready=false）—— 拒绝推理中重建模型");
    return 1;
  }
  if (!s->vae2_ready) {
    set_err(err, err_len, "图生图只支持 rknn2 的本机 VAE（本板布局）");
    return 1;
  }
  if (steps <= 0) steps = 1;
  if (strength < 0.0) strength = 0.0;
  if (strength > 1.0) strength = 1.0;

  // ---- ① 输入图 → z₀（token 主序 fp16）----
  std::vector<unsigned char> z0;
  if (!encode_source_image(s, src_rgb, src_w, src_h, &z0, err, err_len)) {
    return 1;
  }

  // ---- 起步点：strength 越大 ⇒ 起步越早 ⇒ 越接近纯文生图 ----
  // （与 diffusers 的 img2img 同一口径：t_start = steps − int(steps·strength)）
  int start_step = steps - (int)std::lround((double)steps * strength);
  if (start_step < 0) start_step = 0;
  if (start_step > steps - 1) start_step = steps - 1;

  // ---- ② 逐张：TE 编码 + flow-match（从 start_step 起步）----
  std::vector<std::vector<unsigned char>> latents((size_t)n_images);
  int64_t tok_off = 0;
  for (int img = 0; img < n_images; ++img) {
    int n_tok = all_n_tokens[img];
    const int32_t* toks = all_tokens + tok_off;
    tok_off += n_tok;

    uint16_t* ehs = nullptr;
    uint8_t* mask = nullptr;
    int ehs_n = 0, mask_n = 0;
    begin_stage(s, "te");
    if (qi_qwen_te_encode(s, toks, n_tok, &ehs, &ehs_n, &mask, &mask_n, err, err_len) != 0) {
      return 1;
    }
    emit_progress(s, "te", img + 1, n_images);

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
    qi_qwen_free(ehs);
    qi_qwen_free(mask);

    // 原图潜变量 z₀（token 主序 fp16）
    qi::write_file(qi::join_path(dir, "z0.bin"), z0.data(), z0.size());

    // 初始噪声 ε（固定 / 现场随机 —— 与文生图完全同一套）
    const size_t n_elem = 1024 * 64;
    std::string x_path = qi::join_path(dir, "hidden_states.bin");
    if (noise_mode == QI_NOISE_FRESH) {
      std::vector<uint16_t> h = fresh_noise(n_elem, seed + (uint32_t)img);
      qi::write_file(x_path, h.data(), h.size() * sizeof(uint16_t));
    } else {
      std::vector<uint16_t> h;
      if (!qi::np_fixed_noise_512(&h)) {
        set_err(err, err_len, "生成固定噪声失败");
        return 1;
      }
      qi::write_file(x_path, h.data(), h.size() * sizeof(uint16_t));
    }

    begin_stage(s, "flow");
    std::string lat_path = qi::join_path(dir, "latents.bin");
    int ret = s->dit.run_flowmatch(dir, steps, mu, shift_terminal, x_path, lat_path,
                                   std::string(), start_step,
                                   qi::join_path(dir, "z0.bin"));
    if (ret != 0) {
      set_err(err, err_len, "flow-match 失败");
      return 1;
    }
    if (!qi::read_whole_file(lat_path, &latents[(size_t)img])) {
      set_err(err, err_len, "读潜变量失败");
      return 1;
    }
  }

  // ---- ③ VAE 解码（与文生图同一条路）----
  const size_t px = (size_t)512 * 512 * 3;
  uint8_t* all_rgb = (uint8_t*)malloc(px * (size_t)n_images);
  if (!all_rgb) {
    set_err(err, err_len, "内存不足");
    return 1;
  }
  for (int img = 0; img < n_images; ++img) {
    std::vector<unsigned char> rgb;
    int w = 0, h = 0;
    double ms = 0.0;
    begin_stage(s, "vae");
    emit_progress(s, "vae", 0, 1);
    int r2 = s->vae2.decode(latents[(size_t)img], /*channels=*/64, &rgb, &w, &h, &ms);
    if (r2 != 0) {
      set_err(err, err_len, "VAE(rknn2) 解码失败: " + s->vae2.error());
      qi_qwen_free(all_rgb);
      return 1;
    }
    uint8_t* dst = all_rgb + px * (size_t)img;
    if ((size_t)w * h * 3 == rgb.size()) {
      memcpy(dst, rgb.data(), rgb.size());
    } else {
      set_err(err, err_len, "VAE(rknn2) 输出尺寸异常");
      qi_qwen_free(all_rgb);
      return 1;
    }
    if (img + 1 == n_images) {
      *out_w = w;
      *out_h = h;
    }
  }
  *out_rgb = all_rgb;
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
