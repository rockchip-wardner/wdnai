// ---------------------------------------------------------------------------
// wdnai cshim —— 进度回调
//
// ⭐ 为什么要它：出图是**重**操作（首次 load ~90 s，之后每张 8 步 ~34 s），
//    上层（Web UI 的卡片）需要知道「现在到哪一步了、这一步花了多久、
//    还剩多久」才能画进度条。没有它，上层只能看到一个黑转圈。
//
// 阶段（stage）一览：
//   "init"     —— 模型加载（TE / DiT 常驻），step 1..3
//   "te"       —— 提示词编码（TE 常驻）
//   "flow"     —— flow-match 采样，step 1..steps（**进度条的主干**）
//   "release"  —— 释放 DiT（DiT 与 VAE 不能共卡）
//   "vae_init" —— 加载 VAE
//   "vae"      —— VAE 解码（4 段流式），step 1..4
//   "save"     —— 写 PNG / 收尾（Python 侧补的）
//   "done"     —— 完成
//
// ⚠️⚠️ **只有一个时钟**：所有发射点（api.cpp / resident_chain / vae_chain）
//    都通过同一个 qi_progress_ctx 发射 ⇒ elapsed_ms 的口径全局一致。
//    （第一版让每层各自计时，结果「release」那一条报出 21 s —— 那是
//    「距离上一次**本层**发射」的间隔，跨层完全对不上。教训：进度口径
//    必须有唯一的时钟源。）
//
// elapsed_ms 的口径 = **距上一次回调**的毫秒数（不是距阶段开始）：
//   ⇒ 回调方直接拿它当「这一步花了多久」，按步数外推即可得到 ETA。
//   ⇒ 第一个回调的 elapsed 是「从会话开始到这一步完成」的时长。
//
// 线程：回调在**调用方那个线程**上同步触发（C 侧不另起线程）。
//   Python 侧用 ctypes.CDLL（GIL 在调用期间释放）⇒ 回调里重新拿 GIL
//   是安全的，不会自锁。
// ---------------------------------------------------------------------------
#ifndef QI_WDN_PROGRESS_H
#define QI_WDN_PROGRESS_H

#include <string.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* stage: 阶段名（见上表）；step/total: 该阶段内的 1-based 进度；
 * elapsed_ms: 距上一次回调的毫秒数。 */
typedef void (*qi_progress_cb)(void* user, const char* stage, int step,
                               int total, double elapsed_ms);

/* 全局唯一的进度上下文（会话持有它，各层共享指针） */
typedef struct qi_progress_ctx {
  qi_progress_cb cb;   /* 不装就是现在的行为 —— 一行日志都不多打 */
  void* user;
  double last_ms;      /* 上一次「计时基准」的单调时钟（毫秒） */
  char stage[24];      /* 当前阶段（换阶段 ⇒ 计时基准重置） */
} qi_progress_ctx;

/* 单调时钟（毫秒） */
static inline double qi_now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

/* **换阶段**：把计时基准挪到现在。
 *
 * ⭐ 为什么必须有这一步：elapsed 的口径是「距上一次计时基准」，而
 * 「上一次计时基准」若一直停在**上一次请求**的末尾，那么新请求的
 *   第一条进度会把「worker 空闲了多久」当成「这一步花了多久」
 *   （实测：TE 编码显示 44.6 秒 —— 其实是 worker 空等了 44 秒）。
 * ⇒ 每个阶段开工前调一次它 ⇒ 本阶段第一条进度的 elapsed 就是
 *   「本阶段已经花了多久」✅
 */
static inline void qi_progress_stage(qi_progress_ctx* ctx, const char* stage) {
  if (!ctx) return;
  ctx->last_ms = qi_now_ms();
  if (stage) {
    size_t n = sizeof(ctx->stage) - 1;
    strncpy(ctx->stage, stage, n);
    ctx->stage[n] = '\0';
  } else {
    ctx->stage[0] = '\0';
  }
}

/* 发一条进度（elapsed 统一由这里算 ⇒ 全局口径一致） */
static inline void qi_progress_emit(qi_progress_ctx* ctx, const char* stage,
                                     int step, int total) {
  if (!ctx || !ctx->cb) return;
  const double now = qi_now_ms();
  const double ms = now - ctx->last_ms;
  ctx->last_ms = now;
  ctx->cb(ctx->user, stage, step, total, ms);
}

#ifdef __cplusplus
}
#endif
#endif  // QI_WDN_PROGRESS_H
