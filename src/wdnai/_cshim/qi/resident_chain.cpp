// ---------------------------------------------------------------------------
// DiT「常驻模式」实现。
//
// 背景见 resident_chain.h 的注释。核心要点：
//   1) 8 段常驻 ⇒ 每段各自独占 internal 的话需要 8×1878 = 15024 MiB，
//      远超整卡 5122 MiB ⇒ **必须共享一份 internal**。
//   2) RKNN3 的 user_mem_internal 路径实测**可用**，但顺序固定：
//        rknn3_load_model_from_path
//        → rknn3_model_init(config.user_mem_internal = 1)
//        → rknn3_set_internal_mem(ctx, mem[], n_core)
//      （反过来会报 ACK_INVALD_ARGUMENT，见 scripts/probe_setmem_order.cpp）
//   3) 每核 internal 的**真实字节数**用 rknn3_query(ALLOCATION_INFO) 查，
//      而且必须传**数组**（size = sizeof(struct) × core_num）。
// ---------------------------------------------------------------------------

#include "resident_chain.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <string>
#include <algorithm>

namespace qi {

namespace {

double now_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

}  // namespace

// ---------------------------------------------------------------------------
// SharedInternal
// ---------------------------------------------------------------------------

void SharedInternal::release() {
  if (owner) {
    for (auto m : mem) {
      if (m) rknn3_destroy_mem(owner, m);
    }
  }
  mem.clear();
  per_core_bytes.clear();
  owner = 0;
}

uint64_t SharedInternal::total_bytes() const {
  uint64_t t = 0;
  for (auto b : per_core_bytes) t += b;
  return t;
}

// ---------------------------------------------------------------------------
// 节点池容量/空闲查询（**唯一权威判据**）
//
// ⚠️ ALLOCATION_INFO 报的是「模型元数据需求量」，不是实际占用；
//    实际占用必须看 DEVICE_MEM_INFO 的 node_mem_info[n].free。
//    8 个节点池容量并不相同（639.5 / 619.9 / 614.5 MiB）⇒ 预算必须逐池算。
// ---------------------------------------------------------------------------
bool query_node_pool(rknn3_context ctx, std::vector<uint64_t>* total,
                     std::vector<uint64_t>* free_bytes) {
  rknn3_dev_mem_info mi;
  memset(&mi, 0, sizeof(mi));
  if (rknn3_query(ctx, RKNN3_QUERY_DEVICE_MEM_INFO, &mi, sizeof(mi)) != RKNN3_SUCCESS ||
      mi.node_num == 0) {
    return false;
  }
  if (total) total->assign(mi.node_num, 0);
  if (free_bytes) free_bytes->assign(mi.node_num, 0);
  for (uint32_t n = 0; n < mi.node_num && n < RKNN3_MAX_NPU_NODE_NUM; n++) {
    if (total) (*total)[n] = mi.node_mem_info[n].total;
    if (free_bytes) (*free_bytes)[n] = mi.node_mem_info[n].free;
  }
  return true;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// parse_mib_list：把 "121.9,142.7,..." 解析成字节数组
// ---------------------------------------------------------------------------

static std::vector<uint64_t> parse_mib_list(const char* s) {
  std::vector<uint64_t> out;
  std::string str(s), cur;
  for (size_t i = 0; i <= str.size(); ++i) {
    char ch = (i < str.size()) ? str[i] : ',';
    if (ch == ',') {
      if (!cur.empty()) {
        out.push_back((uint64_t)(atof(cur.c_str()) * 1048576.0));
        cur.clear();
      }
    } else if (ch != ' ') {
      cur.push_back(ch);
    }
  }
  return out;
}

// probe_model_mem：跑一次普通 model_init，拿到每核 internal / weight 需求
// ---------------------------------------------------------------------------

bool probe_model_mem(const std::string& model_path, const std::string& weight_path,
                     std::vector<uint64_t>* out_internal, std::vector<uint64_t>* out_weight,
                     const char* device_id) {
  rknn3_context ctx = 0;
  rknn3_init_extend ext;
  memset(&ext, 0, sizeof(ext));
  if (device_id && device_id[0]) {
    ext.device_id = (char*)device_id;
    rknn3_init(&ctx, &ext);
  } else {
    rknn3_init(&ctx, nullptr);
  }
  if (ctx == 0) {
    printf("[resident] probe: rknn3_init 失败\n");
    return false;
  }
  if (rknn3_load_model_from_path(ctx, model_path.c_str(), weight_path.c_str()) !=
      RKNN3_SUCCESS) {
    printf("[resident] probe: 加载模型失败 %s\n", model_path.c_str());
    rknn3_destroy(ctx);
    return false;
  }
  int32_t core_num = 0;
  rknn3_query(ctx, RKNN3_QUERY_CORE_NUMBER, &core_num, sizeof(core_num));
  rknn3_config cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.run_core_mask = (core_num >= 32) ? 0xffffffffu : ((1u << core_num) - 1u);
  int ret = rknn3_model_init(ctx, &cfg);
  if (ret != RKNN3_SUCCESS) {
    printf("[resident] probe: model_init 失败 ret=%d\n", ret);
    rknn3_destroy(ctx);
    return false;
  }

  // ⚠️ ALLOCATION_INFO 必须传**数组**（size = sizeof(struct) × core_num）
  std::vector<rknn3_allocation_info> ais(core_num);
  memset(ais.data(), 0, ais.size() * sizeof(rknn3_allocation_info));
  for (int c = 0; c < core_num; ++c) ais[c].core_id = c;
  ret = rknn3_query(ctx, RKNN3_QUERY_ALLOCATION_INFO, ais.data(),
                    ais.size() * sizeof(rknn3_allocation_info));
  if (ret != RKNN3_SUCCESS) {
    printf("[resident] probe: ALLOCATION_INFO 查询失败 ret=%d\n", ret);
    rknn3_destroy(ctx);
    return false;
  }
  if (out_internal) out_internal->resize(core_num);
  if (out_weight) out_weight->resize(core_num);
  for (int c = 0; c < core_num; ++c) {
    if (out_internal) (*out_internal)[c] = ais[c].internal_mem.size;
    if (out_weight) (*out_weight)[c] = ais[c].weight_mem.size;
  }
  rknn3_destroy(ctx);
  return true;
}

// ---------------------------------------------------------------------------
// ResidentChain
// ---------------------------------------------------------------------------

ResidentChain::~ResidentChain() {
  // ⭐ 释放其它设备组的共享块（⚠️ 不释放会留下每核数百 MiB 的泄漏！）
  for (auto& g : extra_shared_) {
    for (auto m : g.second) {
      if (m) rknn3_destroy_mem(g.first, m);
    }
  }
  extra_shared_.clear();
}

void ResidentChain::release() {
  // 顺序很重要：先销毁共享块（它要靠 owner context 销毁 mem），再销毁各段的 context
  for (auto& g : extra_shared_) {
    for (auto m : g.second) {
      if (m) rknn3_destroy_mem(g.first, m);
    }
  }
  extra_shared_.clear();
  shared_.reset();
  runners_.clear();
  release_head();
  printf("[resident] 已释放全部常驻段（卡池回收）\n");
}

// ---------------------------------------------------------------------------
// ⭐⭐⭐ 方案 D（第三十三轮）：把 DiT 的「头」切出来单独算
//
// 头 = timestep_embedder + modulation：
//   * 只依赖 timestep ⇒ 4 个段各带一份是**纯浪费**（≈ 53.7 MiB int4/段）
//   * 全是 Linear / 逐元素算子（无 Conv、无注意力、无 Transformer）
//   * 切出来后放到**另一张卡**上，卡 0 的 4 个段就都不带头了
//     ⇒ 每核省 4 × 6.7 = 26.8 MiB ⇒ core0 从 643.8 降到 617.0 MiB（池 639.5）
//
// 头的输出按**名字**进 carry map（modulation_out / timestep_emb_out）
//   ⇒ 段用现成的 load_inputs(overrides=carry) 按名取，不用改段的任何代码。
// ---------------------------------------------------------------------------
int ResidentChain::init_head(const std::string& model, const std::string& weight,
                              const char* device_id) {
  if (head_) {
    printf("[resident] 头模型已经在了（设备 %s）\n", head_device_.c_str());
    return 0;
  }
  // ---- 选卡：显式指定 > 自动挑一张没被段占用的卡 ----
  std::string dev = device_id ? device_id : "";
  if (dev.empty()) {
    rknn3_devices devs;
    memset(&devs, 0, sizeof(devs));
    if (rknn3_find_devices(&devs) == RKNN3_SUCCESS) {
      for (uint32_t i = 0; i < devs.n_devices; ++i) {
        std::string id = devs.devices[i].id;
        bool used = false;
        for (const auto& d : device_ids_) {
          if (d == id) { used = true; break; }
        }
        if (!used) { dev = id; break; }
      }
    }
    if (dev.empty()) {
      printf("[错误] 没有空闲的卡放头模型（段已占满所有设备）\n");
      printf("       ⇒ 请显式指定 --head-device，或把头放到与段不同的卡上\n");
      return 2;
    }
  }
  head_device_ = dev;

  printf("[resident] 初始化头模型（设备 %s）：%s\n", dev.c_str(), model.c_str());
  std::unique_ptr<ModelRunner> r(new ModelRunner());
  // ⛔ 头**不**共享 internal：它自己一个小模型，让 runtime 自己分配 internal 最省事
  int ret = r->init(model, weight, core_mask_, /*user_mem_internal=*/false, dev.c_str());
  if (ret != RKNN3_SUCCESS) {
    printf("[错误] 头模型初始化失败 ret=%d\n", ret);
    return 3;
  }
  printf("[resident] 头模型的输入：\n");
  for (const auto& s : r->inputs()) {
    if (s.fake) continue;
    printf("           %s\n", s.attr.name);
  }
  printf("[resident] 头模型的输出（会按名字进 carry）：\n");
  for (const auto& s : r->outputs()) {
    if (s.fake) continue;
    printf("           %s\n", s.attr.name);
  }
  head_ = std::move(r);
  return 0;
}

void ResidentChain::release_head() {
  if (!head_) return;
  printf("[resident] 释放头模型（设备 %s）\n", head_device_.c_str());
  head_.reset();
  head_device_.clear();
}

int ResidentChain::init(const std::vector<std::string>& models,
                        const std::vector<std::string>& weights, uint32_t core_mask,
                        bool share_internal, const std::vector<std::string>* device_ids,
                        const std::vector<bool>* stream_flags) {
  if (models.size() != weights.size() || models.empty()) {
    printf("[错误] models/weights 数量不一致或为空\n");
    return 1;
  }
  const size_t n = models.size();

  // 每段的设备 id
  std::vector<std::string> devs(n);
  if (device_ids) {
    for (size_t i = 0; i < n && i < device_ids->size(); ++i) devs[i] = (*device_ids)[i];
  }

  // ⭐ 流式段标记
  std::vector<bool> streams(n, false);
  if (stream_flags) {
    for (size_t i = 0; i < n && i < stream_flags->size(); ++i) streams[i] = (*stream_flags)[i];
  }

  // 存下来：流式段在 run_once 里要现场建 runner
  models_ = models;
  weights_ = weights;
  device_ids_ = devs;
  stream_ = streams;
  core_mask_ = core_mask;

  printf("[resident] 常驻模式：%zu 段，shared_internal=%d\n", n, share_internal ? 1 : 0);
  for (size_t i = 0; i < n; ++i) {
    printf("[resident] 段 %zu: %s  设备=%s  %s\n", i, models[i].c_str(),
           devs[i].empty() ? "(默认)" : devs[i].c_str(),
           streams[i] ? "【流式：每步 load→init→run→释放】" : "【常驻】");
  }

  // ---- 第 1 步：探针每段的 internal / weight 需求 ----
  // ⚠️ 只在**共享 internal** 时才需要探针（要知道每核需求才能算共享块大小）。
  //    不共享时探针会**白占一份卡上内存**（双卡单模型方案下会直接 MODEL_SETUP fail）⇒ 跳过。
  seg_internal_bytes_.assign(n, 0);
  seg_weight_bytes_.assign(n, 0);
  std::vector<std::vector<uint64_t>> seg_int(n), seg_wt(n);

  // ⭐⭐⭐ 2026-10-31：支持「外部给定 internal 需求」⇒ **完全跳过探针**。
  //
  //   为什么必须能跳过：探针对每段跑一次**完整 model_init**（读权重 + PCIe 传
  //   564 MB + 建 task + 分配 internal + destroy）。连续做 4 次时 runtime 的
  //   回收并不可靠 ⇒ **实测卡在第 1 段探针里 model_init 无返回**
  //   （2026-10-31 板端实测：卡 boot state 正常，但 model_init 永远不返回，
  //    必须 reset 卡才能恢复）。
  //
  //   而静态账本已经足够准：
  //     119 的 check_alloc.py  ⇒ 峰值核 371.0 MiB
  //     板端 probe_fit 实测     ⇒ 峰值核 371.12 MiB   （完全吻合）
  //   ⇒ 直接把每核需求喂进来，省掉 N 次 model_init，也避开上面的坑。
  //
  //   用法：QI_INT_PER_CORE="121.9,142.7,121.9,142.7,173.4,208.7,173.4,297.0"
  //   （单位 MiB，逗号分隔，按核 0..7）
  std::vector<uint64_t> ext_per_core;
  if (share_internal) {
    if (const char* env = getenv("QI_INT_PER_CORE")) {
      ext_per_core = parse_mib_list(env);
      if (ext_per_core.empty()) {
        printf("[resident] QI_INT_PER_CORE 解析失败: %s\n", env);
        return 2;
      }
      printf("[resident] 使用外部 internal 账本（**跳过探针**），每核: ");
      for (size_t c = 0; c < ext_per_core.size(); ++c)
        printf("%s%.1f", c ? "/" : "", ext_per_core[c] / 1048576.0);
      printf(" MiB\n");
    }
  }

  if (share_internal && !ext_per_core.empty()) {
    uint64_t s_int = 0;
    for (auto b : ext_per_core) s_int += b;
    for (size_t i = 0; i < n; ++i) {
      if (streams[i]) continue;
      seg_int[i] = ext_per_core;
      seg_internal_bytes_[i] = s_int;
      printf("[resident] 段 %zu: internal(外部账本)=%.1f MiB\n", i, s_int / 1048576.0);
    }
  } else if (share_internal) {
    const double t_probe_all0 = now_ms();   // ⏱ 2026-11-02 第三十四轮：探针单独计时
    for (size_t i = 0; i < n; ++i) {
      // ⚠️ 流式段**不能探针**：探针会临时建一份完整模型占住卡上内存，
      //    而常驻段已经把卡填满了（8-block 模型每核 569.9 MiB / 池 614.5 MiB）
      //    ⇒ 探针必然 MODEL_SETUP fail。
      if (streams[i]) {
        printf("[resident] 段 %zu: 流式段，跳过探针\n", i);
        continue;
      }
      if (!probe_model_mem(models[i], weights[i], &seg_int[i], &seg_wt[i],
                           devs[i].empty() ? nullptr : devs[i].c_str())) {
        printf("[resident] 探针失败: %s\n", models[i].c_str());
        return 2;
      }
      uint64_t s_int = 0, s_wt = 0;
      for (size_t c = 0; c < seg_int[i].size(); ++c) {
        s_int += seg_int[i][c];
        s_wt += seg_wt[i][c];
      }
      seg_internal_bytes_[i] = s_int;
      seg_weight_bytes_[i] = s_wt;
      printf("[resident] 段 %zu: internal=%.1f MiB weight=%.1f MiB\n", i, s_int / 1048576.0,
             s_wt / 1048576.0);
      // ⏱ 探针本身 = 又一次完整 load + model_init（**慢的真正来源**）
      printf("[resident]   段 %zu 探针耗时 %.1f ms；每核 internal = ", i, now_ms() - t_probe_all0);
      for (size_t c = 0; c < seg_int[i].size(); ++c)
        printf("%s%.3f", c ? "/" : "", seg_int[i][c] / 1048576.0);
      printf(" MiB\n");
    }
    printf("[resident] ⏱ 探针合计 %.1f ms（%zu 段，每段一次完整 model_init）\n",
           now_ms() - t_probe_all0, n);
  }

  // ---- 第 2 步：算共享 internal 的每核大小 ----
  // 8 段串行 ⇒ 每核只需要 max(各段该核需求) 的一份。
  int32_t core_num = 0;
  if (!ext_per_core.empty()) {
    // 外部账本已知每核需求 ⇒ core_num 直接取数组长度，
    // **不再 load_model + destroy**（那一步会白读一遍 564 MB 权重）。
    core_num = (int32_t)ext_per_core.size();
    printf("[resident] core_num = %d（外部账本，跳过 load 探测）\n", core_num);
  } else if (share_internal) {
    rknn3_context t = 0;
    rknn3_init_extend te;
    memset(&te, 0, sizeof(te));
    if (!devs[0].empty()) {
      te.device_id = (char*)devs[0].c_str();
      rknn3_init(&t, &te);
    } else {
      rknn3_init(&t, nullptr);
    }
    rknn3_load_model_from_path(t, models[0].c_str(), weights[0].c_str());
    rknn3_query(t, RKNN3_QUERY_CORE_NUMBER, &core_num, sizeof(core_num));
    rknn3_destroy(t);
  }
  if (core_num <= 0) core_num = 8;

  if (share_internal) {
    shared_.reset(new SharedInternal());
    shared_->per_core_bytes.assign(core_num, 0);
    for (size_t i = 0; i < n; ++i) {
      for (size_t c = 0; c < seg_int[i].size() && c < (size_t)core_num; ++c) {
        if (seg_int[i][c] > shared_->per_core_bytes[c]) {
          shared_->per_core_bytes[c] = seg_int[i][c];
        }
      }
    }
    uint64_t tot = 0;
    for (auto b : shared_->per_core_bytes) tot += b;
    printf("[resident] 共享 internal: 每核峰值 %.1f MiB, 合计 %.1f MiB (各段 max)\n",
           *std::max_element(shared_->per_core_bytes.begin(), shared_->per_core_bytes.end()) /
               1048576.0,
           tot / 1048576.0);
  }

  // ---- 第 3 步：真正 init 各段 ----
  // ⚠️ 探针（probe_model_mem）会**短暂占住节点池**，所以必须**先把所有段的
  //    探针做完并释放**（上面第 1 步已经全部返回、context 已销毁），再开始 init。
  //    否则第 2 段 init 时池子还没还给 runtime ⇒ MODEL_SETUP fail。
  //
  // ⭐⭐⭐ 2026-11-02：共享 internal 块是在**第 4 步**才分配的（那时各段的
  //    weight + cmd + IO 缓冲都已占好池子）。而 pick_io_core 选的是
  //    「池空闲最多」的核 ⇒ 池最大的 core0 会被优先塞满 IO ⇒
  //    等共享块要分配时 core0 已经没空间了 ⇒ **单卡方案必然在 core0 上先炸**。
  //    ⇒ 这里把共享块的每核需求**提前登记成预留**，pick_io_core 会自动避开。
  if (share_internal && !ext_per_core.empty()) {
    // 用外部账本：每核需求直接就是 ext_per_core
    std::map<std::string, std::vector<uint64_t>> per_dev;
    for (size_t i = 0; i < n; ++i) {
      if (streams[i]) continue;
      const std::string& dev = devs[i].empty() ? std::string("(default)") : devs[i];
      auto& v = per_dev[dev];
      if (v.size() < ext_per_core.size()) v.resize(ext_per_core.size(), 0);
      for (size_t c = 0; c < ext_per_core.size(); ++c) {
        if (ext_per_core[c] > v[c]) v[c] = ext_per_core[c];
      }
    }
    for (auto& kv : per_dev) {
      set_io_reserve(kv.first, kv.second);
      printf("[resident] 设备 %s: 预留共享 internal 每核 ", kv.first.c_str());
      for (size_t c = 0; c < kv.second.size(); ++c)
        printf("%s%.1f", c ? "/" : "", kv.second[c] / 1048576.0);
      printf(" MiB（给 IO 挑核用）\n");
    }
  }
  runners_.clear();
  runner_seg_.clear();   // runner 下标 → 段号（流式段不占 runner）
  for (size_t i = 0; i < n; ++i) {
    if (streams[i]) {
      // ⭐ 流式段：**不建常驻 runner**，run_once 时现场建/拆。
      //    （否则每卡要同时装 2 个 8-block 模型 ⇒ 每核 1138 MiB > 614.5 装不下）
      printf("[resident] 段 %zu: 流式段，初始化时跳过\n", i);
      continue;
    }
    std::unique_ptr<ModelRunner> r(new ModelRunner());
    // 先用普通 init 建好 context / 查 IO 张量
    int ret = r->init(models[i], weights[i], core_mask, share_internal ? 1 : 0,
                      devs[i].empty() ? nullptr : devs[i].c_str());
    if (ret != RKNN3_SUCCESS) {
      printf("[错误] 段 %zu 初始化失败 ret=%d\n", i, ret);
      return 3;
    }
    runners_.push_back(std::move(r));
    runner_seg_.push_back(i);
  }

  // ---- 第 4 步：把共享 internal 注入每个 context ----
  // ⚠️ 顺序：model_init 之后才能 set_internal_mem。
  //
  // ⭐⭐⭐ 2026-10-31 关键修正（实测得出，见 debug.md §61.7）：
  //   **「每段各自 create_mem」得到的是 8 份独立内存，并不共享**（旧实现就是这个，
  //   所以共享一直没真正生效）。
  //   实测可行的是：**在组内第一个 context 上 create_mem，然后把同一个 mem 数组
  //   直接传给组内其它 context 的 rknn3_set_internal_mem()** ⇒ 真正共享同一块物理内存。
  //   （probe_share3 的 Q2 实测：ctxB set_internal_mem(ctxA 的 mem) ret=0，
  //     且池空闲量**没有增加** ⇒ 没有二次分配 ⇒ 共享成立。）
  //
  //   ⚠️ 共享块**只能在同一张卡内**（mem 绑定在某个 device 上）
  //      ⇒ 必须**按设备分组**，每组一份共享块。
  if (share_internal) {
    // ---- 按设备分组 ----
    std::map<std::string, std::vector<size_t>> by_dev;
    for (size_t i = 0; i < runners_.size(); ++i) {
      by_dev[devs[runner_seg_[i]].empty() ? std::string("(default)")
                                           : devs[runner_seg_[i]]]
          .push_back(i);
    }
    printf("[resident] 共享 internal：按设备分 %zu 组\n", by_dev.size());

    for (auto& kv : by_dev) {
      const std::vector<size_t>& grp = kv.second;
      // 需求 = 组内各段的每核 max
      std::vector<uint64_t> per_core(core_num, 0);
      for (size_t gi : grp) {
        size_t seg = runner_seg_[gi];
        for (size_t c = 0; c < seg_int[seg].size() && c < (size_t)core_num; ++c) {
          if (seg_int[seg][c] > per_core[c]) per_core[c] = seg_int[seg][c];
        }
      }
      // ---- 组内第一个 context 创建共享块 ----
      ModelRunner* first = runners_[grp[0]].get();
      rknn3_context ctx0 = first->context();

      // ⭐⭐⭐ 2026-10-31：**池容量感知**的预算。
      //   此时各段的 weight + cmd + IO 缓冲**已经占掉一部分池子**，
      //   共享块只能在**剩余空间**里分配 ⇒ 必须查真实池空闲量，
      //   否则 create_mem 会以 `MSG_MEM_CREATE_ACK fail` 失败（根因就是没算 IO）。
      std::vector<uint64_t> pool_total, pool_free;
      if (!query_node_pool(ctx0, &pool_total, &pool_free)) {
        printf("[警告] 设备 %s: 查不到节点池信息，按需求原样分配\n", kv.first.c_str());
        pool_total.assign(core_num, 0);
        pool_free.assign(core_num, 0);
      }
      const uint64_t kSafety = 16ull << 20;   // 每池留 16 MiB 安全余量
      // ⛔ QI_SHARE_FIT 现在**不再**允许把共享块裁到小于需求（会写越界）。
      //    这里只把它降级成「提示」：仍然拒绝，但明确告诉用户为什么。
      const bool fit_requested = getenv("QI_SHARE_FIT") != nullptr;
      // ⭐ 共享块的余量倍率：默认 **1.0**（探针测到的就是 runtime 的申请量，
      //   再乘 1.08 会凭空多要 30 MiB/核 ⇒ 4 段/卡直接装不下）。
      //   需要更保守时用 QI_SHARE_MARGIN=1.08。
      double margin = 1.0;
      if (const char* m = getenv("QI_SHARE_MARGIN")) margin = atof(m);

      std::vector<rknn3_tensor_mem*> mems;
      uint64_t need_sum = 0, got_sum = 0;
      const double t_create0 = now_ms();      // ⏱ 2026-11-02 第三十四轮：分段计时
      for (int c = 0; c < core_num; ++c) {
        uint64_t need = per_core[c];
        if (need == 0) need = 256ull << 20;
        // +1 MiB 对齐余量
        need = (uint64_t)(need * margin) + (1 << 20);
        uint64_t sz = need;
        uint64_t avail = (c < (int)pool_free.size()) ? pool_free[c] : 0;
        uint64_t cap = (c < (int)pool_total.size()) ? pool_total[c] : 0;
        if (avail > 0 && need > avail) {
          uint64_t usable = (avail > kSafety) ? (avail - kSafety) : 0;
          printf("[错误] 设备 %s: 核 %d 池空间不足：需要 %.1f MiB，可用 %.1f MiB"
                 "（池总 %.1f MiB，缺 %.1f MiB）\n",
                 kv.first.c_str(), c, need / 1048576.0, avail / 1048576.0,
                 cap / 1048576.0, (need - avail) / 1048576.0);
          // ⛔⭐⭐⭐ 2026-11-02：**禁止**把共享块裁到小于需求。
          //   旧实现在 QI_SHARE_FIT=1 时会裁到「可用 − 16 MiB」，
          //   而 runtime 拿到的是**用户给的缓冲** ⇒ 它会照着模型的 internal
          //   需求去写 ⇒ **写越界 ⇒ 踩坏别的分配 ⇒ 内存损坏**。
          //   这比直接 FAIL 危险得多（FAIL 只是起不来，越界是结果错/崩卡）。
          //   ⇒ 现在：**只要裁剪后 < 需求，一律硬失败**，
          //     必须真的减少每核需求（换更小的段 / 减少同卡段数 / 减小 IO）。
          if (usable < need) {
            printf("[错误] 共享 internal 装不下 ⇒ 需减小每核 internal 需求"
                   "（或减少每卡常驻段数 / 用更少 IO 占用）\n");
            printf("[错误] ⛔ 拒绝裁剪到 %.1f MiB（需求 %.1f MiB）—— 裁小了 runtime "
                   "会写越界踩坏内存\n", usable / 1048576.0, need / 1048576.0);
            if (fit_requested) {
              printf("[错误] QI_SHARE_FIT 已被禁用（2026-11-02）：它只会把共享块裁小，"
                     "导致 runtime 写越界。\n");
            }
            return 4;
          }
          sz = usable;
          printf("[警告] 核 %d 裁剪到 %.1f MiB（QI_SHARE_FIT=1）\n", c, sz / 1048576.0);
        }
        rknn3_tensor_mem* m = nullptr;
        // create_mem 失败 ⇒ 逐步缩到可用空间再试（对齐/额外开销可能比预算多）
        for (int attempt = 0; attempt < 4 && !m; ++attempt) {
          m = rknn3_create_mem(ctx0, sz, c, RKNN3_FLAG_MEMORY_CACHEABLE);
          if (!m && attempt < 3) {
            sz = (uint64_t)(sz * 0.9);
            printf("[警告] 设备 %s 核 %d: create_mem 失败，重试 %.1f MiB\n",
                   kv.first.c_str(), c, sz / 1048576.0);
          }
        }
        if (!m) {
          printf("[错误] 设备 %s: create_mem(internal, core %d, %.1f MiB) 失败\n",
                 kv.first.c_str(), c, sz / 1048576.0);
          return 4;
        }
        mems.push_back(m);
        need_sum += need;
        got_sum += sz;
        printf("[resident]   核 %d: 池总 %.1f MiB / 分配前空闲 %.1f MiB / internal 需求 %.1f MiB"
               " → 实配 %.1f MiB\n",
               c, cap / 1048576.0, avail / 1048576.0, need / 1048576.0, sz / 1048576.0);
      }
      printf("[resident] 设备 %s: 共享块 %zu 段共用（每核 max，需求合计 %.1f MiB，实配 %.1f MiB）\n",
             kv.first.c_str(), grp.size(), need_sum / 1048576.0, got_sum / 1048576.0);
      printf("[resident] ⏱ 共享块 create_mem（%d 核）耗时 %.1f ms\n", core_num,
             now_ms() - t_create0);

      // ---- 组内所有 context 都 set 同一个 mem 数组 ----
      double t_set_total = 0;
      for (size_t k = 0; k < grp.size(); ++k) {
        ModelRunner* r = runners_[grp[k]].get();
        rknn3_context ctx = r->context();
        const double t_set0 = now_ms();
        int ret = rknn3_set_internal_mem(ctx, mems.data(), (uint32_t)core_num);
        const double dt_set = now_ms() - t_set0;
        t_set_total += dt_set;
        printf("[resident]   段 %zu（设备 %s）set_internal_mem ret=%d  ⏱ %.1f ms%s\n",
               runner_seg_[grp[k]], kv.first.c_str(), ret, dt_set,
               (k == 0) ? "  ← 创建者" : "  ← 复用同一块");
        if (ret != RKNN3_SUCCESS) {
          printf("[错误] 段 %zu 注入共享 internal 失败 ret=%d\n", runner_seg_[grp[k]], ret);
          return 5;
        }
      }
      printf("[resident] ⏱ set_internal_mem 合计 %.1f ms（%zu 段）\n", t_set_total, grp.size());
      // 生命周期：mem 归该组**第一个** runner 的 context（随 context 销毁回收）
      if (!shared_->owner) {
        shared_->owner = ctx0;
        shared_->mem = mems;
      } else {
        // 多组时另存（第二组及以后）
        extra_shared_.push_back({ctx0, mems});
      }
    }
  }

  printf("[resident] ✅ %zu 段全部常驻完成\n", n);
  return 0;
}

int ResidentChain::run_once(const std::string& input_dir,
                            std::map<std::string, std::vector<unsigned char>>* carry,
                            double* total_ms) {
  double t0 = now_ms();
  const size_t n = models_.size();

  // ================================================================
  // ⭐⭐⭐ 方案 D（第三十三轮）：**先跑「头」模型**
  //   头 = timestep_embedder + modulation（只依赖 timestep）
  //   它的输出（modulation_out / timestep_emb_out）按**名字**进 carry map
  //   ⇒ 段按名取输入（load_inputs 的 overrides 机制）⇒ 下游一个字不用改。
  // ================================================================
  if (head_) {
    // 头的输入只有 timestep（由 flow-match 循环每步写进 input_dir）
    int hret = head_->load_inputs(input_dir, false, nullptr);
    if (hret != RKNN3_SUCCESS) {
      printf("[错误] 头模型装载输入失败 ret=%d\n", hret);
      return 3;
    }
    double hms = 0.0;
    hret = head_->run(&hms);
    if (hret != RKNN3_SUCCESS) {
      printf("[错误] 头模型推理失败 ret=%d\n", hret);
      return 4;
    }
    int n_out = 0;
    for (const auto& slot : head_->outputs()) {
      if (slot.fake) continue;
      std::vector<unsigned char> bytes;
      if (!head_->get_output_bytes(slot.attr.name, &bytes)) continue;
      if (carry) {
        (*carry)[slot.attr.name] = bytes;
        // ⭐ 兜底：RKNN toolkit 有可能把输出名改掉（实测有时会带前缀/改写）。
        //   段的输入名是**约定的** modulation_out / timestep_emb_out，
        //   所以按「原 ONNX 名里的特征」再补一份别名 ⇒ 段永远能按名取到。
        const std::string& on = slot.attr.name;
        if (on.find("modulation.1/MatMul") != std::string::npos) {
          (*carry)["modulation_out"] = bytes;
        } else if (on.find("modulation.0/Mul") != std::string::npos) {
          (*carry)["timestep_emb_out"] = bytes;
        }
      }
      ++n_out;
    }
    printf("[resident] 头模型推理 %.1f ms（%d 个输出进 carry）\n", hms, n_out);
    fflush(stdout);
  }

  // ⭐ 按**段号**顺序跑：常驻段直接 run，流式段现场建 → run → 拆。
  //   （runners_ 里只放常驻段，下标 → 段号用 runner_seg_ 映射）
  size_t ri = 0;
  for (size_t i = 0; i < n; ++i) {
    ModelRunner* r = nullptr;
    std::unique_ptr<ModelRunner> tmp;   // 流式段的临时 runner

    if (stream_.size() > i && stream_[i]) {
      // ---- 流式段：load → model_init → run → 释放 ----
      double s0 = now_ms();
      tmp.reset(new ModelRunner());
      int ret = tmp->init(models_[i], weights_[i], core_mask_, false,
                          device_ids_[i].empty() ? nullptr : device_ids_[i].c_str());
      if (ret != RKNN3_SUCCESS) {
        // ⚠️ 已知的最常见原因：**同一张卡放不下两个 DiT 模型**。
        //   每核池 614.5 MiB，而一个 8-block 模型每核就要 569.9 MiB
        //   ⇒ 卡上只要有一个常驻段，同卡的流式段必然 MODEL_SETUP fail。
        //   ⇒ 「常驻 + 流式」必须分到**不同的卡**上。
        printf("\n[错误] 段 %zu（流式）model_init 失败 ret=%d\n", i, ret);
        printf("       ⭐ 最可能原因：这张卡上已有常驻段 ⇒ **一张卡放不下两个 DiT 模型**\n");
        printf("          （每核池 614.5 MiB；8-block 模型每核峰 569.9 MiB）\n");
        printf("          ⇒ 请用 --device-list 把常驻段和流式段分到**不同的卡**上。\n");
        // ⚠️ runtime 内部在 model_init 失败时会 std::terminate ⇒ 这里只能尽量提示，
        //    避免「静默算错」。（实测 abort 后卡上内存**没有**泄漏，池空闲能回到满额。）
        fflush(stdout);
        return 2;
      }
      r = tmp.get();
      printf("[resident] 段 %zu（流式）现场建模型 %.1f ms\n", i, now_ms() - s0);
    } else {
      if (ri >= runners_.size()) {
        printf("[错误] 段 %zu 找不到常驻 runner\n", i);
        return 2;
      }
      r = runners_[ri].get();
      ++ri;
    }

    int ret = r->load_inputs(input_dir, false, carry);
    if (ret != RKNN3_SUCCESS) {
      printf("[错误] 段 %zu 装载输入失败 ret=%d\n", i, ret);
      return 3;
    }
    double ms = 0.0;
    ret = r->run(&ms);
    if (ret != RKNN3_SUCCESS) {
      printf("[错误] 段 %zu 推理失败 ret=%d\n", i, ret);
      return 4;
    }
    printf("[resident] 段 %zu/%zu 推理 %.1f ms%s\n", i + 1, n, ms,
           (stream_.size() > i && stream_[i]) ? "（流式）" : "");
    fflush(stdout);

    // 把本段输出带给下一段
    for (const auto& slot : r->outputs()) {
      if (slot.fake) continue;
      const std::string oname(slot.attr.name);
      std::vector<unsigned char> bytes;
      if (!r->get_output_bytes(oname, &bytes)) continue;
      if (carry) (*carry)[oname] = bytes;
    }
  }
  if (total_ms) *total_ms = now_ms() - t0;
  return 0;
}

// ---------------------------------------------------------------------------
// ⭐⭐⭐ flow-match 常驻模式（2026-10-31，第二十轮）
//
// 目的：让**生成过程中零 load / 零 model_init**。
//   之前 x 的回写在 Python 层，每步起一个 `dit-chain` 子进程
//   ⇒ 每步 8 次 load（PCIe 传权重）+ 8 次 model_init（建 task + 分配 internal）。
//   实测这部分占每步 ~29 s / 56 s = 52%。
//   ⇒ 把整个 flow-match 循环搬进常驻进程：模型只 init 一次，之后每步只 run。
//
// sigma 序列与 scripts/run_flowmatch.py 的 sigmas_for() **逐位一致**。
// ---------------------------------------------------------------------------

namespace {

// diffusers FlowMatchEulerDiscreteScheduler 的 time_shift（exponential）
double time_shift_exponential(double mu, double t) {
  return std::exp(mu) / (std::exp(mu) + (1.0 / t) - 1.0);
}

// float32 → float16（round-to-nearest-even），与 Python 的 np.float16 一致
uint16_t f32_to_f16_rne(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  uint32_t sign = (u >> 16) & 0x8000u;
  int32_t exp = (int32_t)((u >> 23) & 0xFF) - 127;
  uint32_t mant = u & 0x007FFFFFu;

  if (((u >> 23) & 0xFF) == 0xFF) {           // Inf / NaN
    return (uint16_t)(sign | 0x7C00u | (mant ? 0x0200u : 0u));
  }
  if (exp > 15) return (uint16_t)(sign | 0x7C00u);   // overflow → Inf
  if (exp >= -14) {                            // normal
    uint32_t m = mant >> 13;
    uint32_t rem = mant & 0x1FFFu;
    uint32_t h = sign | (uint32_t)((exp + 15) << 10) | m;
    // RNE：tie-to-even
    if (rem > 0x1000u || (rem == 0x1000u && (m & 1u))) ++h;
    return (uint16_t)h;
  }
  if (exp >= -25) {                            // subnormal
    uint32_t m = mant | 0x00800000u;
    int shift = -exp - 14 + 13;
    uint32_t h_m = m >> shift;
    uint32_t rem = m & ((1u << shift) - 1u);
    uint32_t half = 1u << (shift - 1);
    uint16_t h = (uint16_t)(sign | h_m);
    if (rem > half || (rem == half && (h_m & 1u))) ++h;
    return h;
  }
  return (uint16_t)sign;                       // underflow → ±0
}

}  // namespace

int ResidentChain::run_flowmatch(const std::string& input_dir, int steps, double mu,
                                 double shift_terminal, const std::string& x_path,
                                 const std::string& out_latents_path,
                                 const std::string& dump_dir) {
  // ---- 前置检查：必须全部常驻，否则仍有 load/init ----
  for (size_t i = 0; i < stream_.size(); ++i) {
    if (stream_[i]) {
      printf("[错误] flow-match 常驻模式要求**所有段常驻**，但段 %zu 是流式段\n", i);
      printf("       （流式段每步都要 load+init ⇒ 违背「生成过程零 load/init」）\n");
      return 2;
    }
  }

  // ---- sigma 序列（与 Python sigmas_for 逐位一致）----
  std::vector<double> sig;
  sig.reserve(steps + 1);
  for (int k = 0; k < steps; ++k) {
    double s = 1.0 - (double)k / (double)steps;          // linspace(1, 1/N, N)
    s = time_shift_exponential(mu, s);
    sig.push_back(s);
  }
  {   // stretch_shift_to_terminal
    double last = sig.back();
    double scale = (1.0 - last) / (1.0 - shift_terminal);
    for (int k = 0; k < steps; ++k) {
      double om = 1.0 - sig[k];
      sig[k] = 1.0 - om / scale;
    }
  }
  sig.push_back(0.0);

  printf("[flowmatch] steps=%d mu=%.10f\n", steps, mu);
  printf("[flowmatch] sigmas = [");
  for (size_t k = 0; k < sig.size(); ++k) printf("%s%.6f", k ? ", " : "", sig[k]);
  printf("]\n");

  // ---- 读入初始 x（fp16）----
  std::vector<unsigned char> xbuf;
  {
    // 先按 fp16 读，再转 fp32 累加（与 Python load_f16 一致）
    std::vector<uint16_t> h;
    FILE* f = fopen(x_path.c_str(), "rb");
    if (!f) { printf("[错误] 打不开 %s\n", x_path.c_str()); return 3; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    h.resize(sz / 2);
    if (fread(h.data(), 1, h.size() * 2, f) != h.size() * 2) { fclose(f); return 3; }
    fclose(f);
    xbuf.resize(h.size() * 4);
    float* xf = (float*)xbuf.data();
    for (size_t i = 0; i < h.size(); ++i) {
      // fp16 → fp32
      uint32_t s = (h[i] >> 15) & 1u, e = (h[i] >> 10) & 0x1Fu, m = h[i] & 0x3FFu;
      uint32_t out;
      if (e == 0) {
        if (m == 0) { out = s << 31; }
        else {
          int sh = 0;
          while (!(m & 0x400u)) { m <<= 1; ++sh; }
          m &= 0x3FFu;
          uint32_t fe = (uint32_t)(127 - 15 - sh);
          out = (s << 31) | (fe << 23) | (m << 13);
        }
      } else if (e == 31) {
        out = (s << 31) | 0x7F800000u | (m << 13);
      } else {
        out = (s << 31) | ((uint32_t)(e + 112) << 23) | (m << 13);
      }
      float f;
      memcpy(&f, &out, 4);
      xf[i] = f;
    }
    printf("[flowmatch] 初始 x: %zu 元素\n", h.size());
  }

  const size_t n_elem = xbuf.size() / 4;
  std::vector<float> x(n_elem);
  memcpy(x.data(), xbuf.data(), n_elem * 4);

  std::map<std::string, std::vector<unsigned char>> carry;
  const std::string ts_path = input_dir + "/timestep.bin";

  // 写 timestep 的 meta（runner 按 meta 决定 dtype/shape）
  {
    FILE* f = fopen((ts_path + ".meta").c_str(), "w");
    if (f) { fprintf(f, "# dtype shape\nfloat16 1\n"); fclose(f); }
  }

  // 进度计时：t_prev_step = 上一步跑完的时刻（第一步 = flow 阶段开始）
  for (int k = 0; k < steps; ++k) {
    double sigma = sig[k], sigma_next = sig[k + 1];

    // ---- 写 x（fp16 RNE）到输入缓冲 ----
    {
      std::vector<uint16_t> h(n_elem);
      for (size_t i = 0; i < n_elem; ++i) h[i] = f32_to_f16_rne(x[i]);
      FILE* f = fopen(x_path.c_str(), "wb");
      if (!f) { printf("[错误] 写不了 %s\n", x_path.c_str()); return 3; }
      fwrite(h.data(), 2, n_elem, f);
      fclose(f);
    }
    // ---- 写 timestep（fp16）----
    {
      uint16_t h = f32_to_f16_rne((float)sigma);
      FILE* f = fopen(ts_path.c_str(), "wb");
      if (!f) { printf("[错误] 写不了 %s\n", ts_path.c_str()); return 3; }
      fwrite(&h, 2, 1, f);
      fclose(f);
    }

    printf("\n[flowmatch] step %d/%d  sigma=%.6f -> %.6f\n", k + 1, steps, sigma, sigma_next);
    fflush(stdout);

    double ms = 0.0;
    int ret = run_once(input_dir, &carry, &ms);
    if (ret != RKNN3_SUCCESS) {
      printf("[错误] step %d 失败 ret=%d\n", k + 1, ret);
      return 4;
    }
    printf("[flowmatch] step %d 耗时 %.1f ms（**零 load / 零 model_init**）\n", k + 1, ms);
    fflush(stdout);

    // ---- 取 v = noise_pred，做 x = x + (σ_next − σ)·v ----
    auto it = carry.find("noise_pred");
    if (it == carry.end()) {
      printf("[错误] carry 里没有 noise_pred\n");
      return 5;
    }
    const std::vector<unsigned char>& vb = it->second;
    if (vb.size() != n_elem * 2) {
      printf("[错误] noise_pred 长度 %zu != x 长度 %zu\n", vb.size(), n_elem * 2);
      return 5;
    }
    const uint16_t* vh = (const uint16_t*)vb.data();
    double dt = sigma_next - sigma;
    double xrms = 0.0, vrms = 0.0;
    for (size_t i = 0; i < n_elem; ++i) {
      // fp16 → fp32
      uint32_t s = (vh[i] >> 15) & 1u, e = (vh[i] >> 10) & 0x1Fu, m = vh[i] & 0x3FFu;
      uint32_t out;
      if (e == 0) {
        if (m == 0) { out = s << 31; }
        else {
          int sh = 0;
          while (!(m & 0x400u)) { m <<= 1; ++sh; }
          m &= 0x3FFu;
          out = (s << 31) | ((uint32_t)(127 - 15 - sh) << 23) | (m << 13);
        }
      } else if (e == 31) {
        out = (s << 31) | 0x7F800000u | (m << 13);
      } else {
        out = (s << 31) | ((uint32_t)(e + 112) << 23) | (m << 13);
      }
      float vf;
      memcpy(&vf, &out, 4);
      x[i] += (float)dt * vf;
      xrms += (double)x[i] * x[i];
      vrms += (double)vf * vf;
    }
    printf("[flowmatch] |x|rms=%.5f  |v|rms=%.5f\n", sqrt(xrms / n_elem), sqrt(vrms / n_elem));
    fflush(stdout);

    if (!dump_dir.empty()) {
      char buf[512];
      snprintf(buf, sizeof(buf), "%s/step%02d_x.bin", dump_dir.c_str(), k + 1);
      std::vector<uint16_t> h(n_elem);
      for (size_t i = 0; i < n_elem; ++i) h[i] = f32_to_f16_rne(x[i]);
      FILE* f = fopen(buf, "wb");
      if (f) { fwrite(h.data(), 2, n_elem, f); fclose(f); }
      snprintf(buf, sizeof(buf), "%s/step%02d_v.bin", dump_dir.c_str(), k + 1);
      f = fopen(buf, "wb");
      if (f) { fwrite(vb.data(), 1, vb.size(), f); fclose(f); }
    }

    // ---- 进度：这一步跑完了 ----
    // elapsed_ms 由**共享 ctx** 统一算（口径与会话层的事件一致）
    // ⇒ 上层直接拿它当「这一步花了多久」，按剩余步数外推就是 ETA。
    qi_progress_emit(progress_ctx_, "flow", k + 1, steps);
  }

  // ---- 输出最终潜变量（fp16，与 run_flowmatch.py 的 latents_nchw.bin 同格式）----
  {
    std::vector<uint16_t> h(n_elem);
    for (size_t i = 0; i < n_elem; ++i) h[i] = f32_to_f16_rne(x[i]);
    FILE* f = fopen(out_latents_path.c_str(), "wb");
    if (!f) { printf("[错误] 写不了 %s\n", out_latents_path.c_str()); return 6; }
    fwrite(h.data(), 2, n_elem, f);
    fclose(f);
    printf("[flowmatch] ✅ 潜变量已写 %s（%zu 元素 fp16）\n", out_latents_path.c_str(), n_elem);
  }
  return 0;
}

void ResidentChain::print_mem_summary() const {
  printf("\n===== 常驻内存占用 =====\n");
  uint64_t w_total = 0, i_total = 0;
  // ⭐ 优先用**运行时自己的分配表**（ALLOCATION_INFO）—— 这才是权威值。
  //   探针数据（seg_*_bytes_）只在「共享 internal」模式下才有（那时会跑探针）；
  //   不共享时探针被跳过（会白占卡上内存），所以必须回落到 query_core_alloc。
  std::vector<uint64_t> per_seg_w(runners_.size(), 0), per_seg_i(runners_.size(), 0);
  std::vector<uint64_t> core_peak(runners_.size(), 0);
  for (size_t i = 0; i < runners_.size(); ++i) {
    std::vector<uint64_t> cmd, wt, in, kv;
    if (runners_[i]->query_core_alloc(&cmd, &wt, &in, &kv)) {
      uint64_t sw = 0, si = 0, peak = 0;
      for (size_t c = 0; c < wt.size(); ++c) sw += wt[c];
      for (size_t c = 0; c < in.size(); ++c) si += in[c];
      for (size_t c = 0; c < wt.size(); ++c) {
        uint64_t p = wt[c] + (c < cmd.size() ? cmd[c] : 0) + (c < in.size() ? in[c] : 0);
        if (p > peak) peak = p;
      }
      per_seg_w[i] = sw;
      per_seg_i[i] = si;
      core_peak[i] = peak;
    } else {
      per_seg_w[i] = seg_weight_bytes_[i];
      per_seg_i[i] = seg_internal_bytes_[i];
    }
  }

  for (size_t i = 0; i < per_seg_w.size(); ++i) {
    printf("  段 %zu: weight=%8.1f MiB  internal=%8.1f MiB  每核峰=%8.1f MiB %s\n", i,
           per_seg_w[i] / 1048576.0, per_seg_i[i] / 1048576.0, core_peak[i] / 1048576.0,
           core_peak[i] > 614.5 * 1048576 ? "❌超节点池" : "✅");
    w_total += per_seg_w[i];
    i_total += per_seg_i[i];
  }
  printf("  ---------------------------------------------\n");
  printf("  权重合计        = %8.1f MiB\n", w_total / 1048576.0);
  printf("  internal 各自独占 = %8.1f MiB  (%.2f GiB)\n", i_total / 1048576.0,
         i_total / 1073741824.0);
  if (shared_ && shared_->valid()) {
    uint64_t s = shared_->total_bytes();
    printf("  共享后 internal  = %8.1f MiB  (%.2f GiB)  ⇒ 省 %.2f GiB\n", s / 1048576.0,
           s / 1073741824.0, (i_total > s ? (i_total - s) : 0) / 1073741824.0);
  }
  printf("  整卡 NPU 内存    = %8.1f MiB  (%.2f GiB, 8 核 × ~614.5 MiB)\n", 5122.0, 5.0);
  uint64_t need = w_total + (shared_ ? shared_->total_bytes() : i_total);
  printf("  权重+internal    = %8.1f MiB  ⇒ %s\n", need / 1048576.0,
         need <= 5122ull * 1048576 ? "装得下 ✅" : "装不下 ❌");
  printf("=========================\n");
}

}  // namespace qi
