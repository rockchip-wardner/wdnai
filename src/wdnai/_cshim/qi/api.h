// ---------------------------------------------------------------------------
// wdnai cshim —— Qwen-Image-2.1 出图的 C API
//
// ⭐ 设计目标：**Python 只调这几个接口**，算法全在 C 侧
//    （TE 编码 / flow-match / VAE 解码 / 归一化），与参考实现同源 ⇒ 逐位一致。
//
// 依赖：**只有系统自带的 librknn3_api.so**（门面会 dlopen rkcp 后端）
//    ⇒ 发布包里不需要带任何 Rockchip 的 .so / .bin。
//
// 典型用法：
//     s = qi_qwen_open(model_dir, card_te, card_dit, 0xff, err, sizeof err);
//     qi_qwen_te_encode(s, tokens, n_tokens, &ehs, &ehs_n, &mask, &mask_n, err, n);
//     qi_qwen_generate_batch(s, n, tokens, n_toks, steps, mu, shift, noise_mode, seed,
//                            &rgb, &w, &h, err, n);
//     qi_qwen_close(s);
// ---------------------------------------------------------------------------
#ifndef QI_WDN_QWEN_API_H
#define QI_WDN_QWEN_API_H

#include <stdint.h>

#include "progress.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qi_qwen_session qi_qwen_session;

/* 噪声模式 */
#define QI_NOISE_FIXED 0   /* 现场生成（numpy 兼容的 PCG64 + ziggurat，与历史基线逐字节一致） */
#define QI_NOISE_FRESH 1   /* 现场生成（mt19937，种子 = seed + 图片下标） */

/* 打开会话：加载 TE（常驻）+ DiT 头 + DiT 4 段（常驻，共享 internal）。
 * 失败返回 NULL，err 里是中文原因。 */
qi_qwen_session* qi_qwen_open(const char* model_dir,
                              const char* card_te,
                              const char* card_dit,
                              uint32_t core_mask,
                              char* err, int err_len);

/* 同上，但**在加载模型之前**就装上进度回调。
 * ⭐ 要看到那 ~90 s 的 init 进度（TE 37 s + DiT 39 s）就必须用这个：
 *   qi_qwen_open 里的 init 发生在返回之前，事后再 set_progress 就晚了。 */
qi_qwen_session* qi_qwen_open_ex(const char* model_dir,
                                 const char* card_te,
                                 const char* card_dit,
                                 uint32_t core_mask,
                                 qi_progress_cb cb, void* user,
                                 char* err, int err_len);

/* 同 qi_qwen_open_ex，但多一个 **VAE 核掩码**（只对**本机 RK3588 的 rknn2
 * NPU**有效；TE / DiT 仍走 card_te / card_dit 指定的 rk1820）。
 *
 *   vae_core_mask = 0    ⇒ 用默认（RKNN_NPU_CORE_0_1_2，即 RK3588 的 3 个核全用）
 *   vae_core_mask = 7    ⇒ 显式 3 核
 *   vae_core_mask = 1    ⇒ 只用 1 个核
 *
 * ⭐ 为什么要有这个：把 VAE 放到**本机**的 rknn2 NPU 上之后，它与 DiT（跑在
 *   1828 协处理器上）**物理分开** ⇒ DiT 可以一直常驻 ⇒ 生成过程零 model_init。
 *   而 VAE 用几个核是可以调的（3 核约 1.5 s/张，单核约 4.6 s/张）。 */
qi_qwen_session* qi_qwen_open_ex2(const char* model_dir,
                                  const char* card_te,
                                  const char* card_dit,
                                  uint32_t core_mask,
                                  uint32_t vae_core_mask,
                                  qi_progress_cb cb, void* user,
                                  char* err, int err_len);

/* 同 qi_qwen_open_ex2，但多一个 **VAE 独占卡**（``card_vae``，PCI BDF 文本）。
 *
 * ⭐ 两种支持的拓扑（**全部模型一次性装好并全程常驻**，推理过程中
 *   **不卸载 / 不重装载**任何模型）：
 *
 *   A) **2×1828 + 3588** —— 模型包里有 ``vae_qwenimage21-512.3588.rknn``
 *      ⇒ VAE 跑**本机 RK3588 的 NPU**（rknn2 / librknnrt），
 *      与 DiT（1828）**物理分开**。``card_vae`` 传空即可。
 *   B) **3×1828** —— 没有 3588 的 rknn2 VAE，但有第三张 1828 卡
 *      ⇒ ``card_vae`` 传那张卡的 BDF，VAE(rknn3 4 段) **独占**它，
 *      与 DiT 互不干扰、同样常驻。
 *
 * ⚠️ 两者都没有 ⇒ **open 直接失败**并给出中文原因。
 *    **不再退回「VAE 与 DiT 共卡」** —— 那要求推理前释放 DiT、推理后重建，
 *    每批图多 39 s model_init，与「全程常驻」的目标冲突。
 */
qi_qwen_session* qi_qwen_open_ex3(const char* model_dir,
                                  const char* card_te,
                                  const char* card_dit,
                                  const char* card_vae,
                                  uint32_t core_mask,
                                  uint32_t vae_core_mask,
                                  qi_progress_cb cb, void* user,
                                  char* err, int err_len);

void qi_qwen_close(qi_qwen_session* s);

/* 装进度回调（可选；不装就是现在的行为 —— 一行日志都不多打）。
 *
 * 回调在**调用方那个线程**上同步触发。阶段与口径见 progress.h。
 * 典型用法（Python）：
 *     lib.qi_qwen_set_progress(s, cb, None)   # cb = CFUNCTYPE(...) 实例
 */
void qi_qwen_set_progress(qi_qwen_session* s, qi_progress_cb cb, void* user);

/* TE 编码：tokens 是**带 t2i 模板**的完整 token 序列（Python 侧分词得到）。
 * 输出 ehs（fp16，128×4096）与 mask（bool，128），由 qi_qwen_free 释放。 */
int qi_qwen_te_encode(qi_qwen_session* s,
                      const int32_t* tokens, int n_tokens,
                      uint16_t** out_ehs, int* out_ehs_n,
                      uint8_t** out_mask, int* out_mask_n,
                      char* err, int err_len);

/* 批量出图：N 张。
 *   all_tokens / all_n_tokens：N 段 token 的拼接与各自长度
 *   输出：out_rgb = N × h × w × 3 的 uint8（RGB888，行主序），由 qi_qwen_free 释放
 *
 * 流程（与参考实现的 serve 一致）：
 *   ① 逐张 TE 编码（TE 常驻）
 *   ② 逐张 flow-match（DiT 4 段常驻，零 load / 零 model_init）
 *   ③ **释放 DiT**（DiT 与 VAE 物理上不能共卡）—— ⚠️ 只有当 VAE 走的是
 *      **rknn3 / 同卡** 时才会释放；VAE 走**本机 rknn2**（不同设备）时
 *      **DiT 全程常驻**，生成过程一次 model_init 都不会发生
 *   ④ 加载 VAE 4 段（流式），逐张解码 + 归一化成 uint8
 */
int qi_qwen_generate_batch(qi_qwen_session* s,
                           int n_images,
                           const int32_t* all_tokens, const int32_t* all_n_tokens,
                           int steps, double mu, double shift_terminal,
                           int noise_mode, uint32_t seed,
                           uint8_t** out_rgb, int* out_w, int* out_h,
                           char* err, int err_len);

/* 释放 C API 分配的内存（ehs / mask / rgb） */
void qi_qwen_free(void* p);

/* 生成「固定初始噪声」（512 路线：1024 token × 64 通道，fp16）。
 * 与历史基线 reference_512/hidden_states.bin **逐字节一致**
 * （md5 b808cf7ecd34338323b783704c516128）⇒ 模型包不需要带那个文件。
 * out_fp16 由 qi_qwen_free 释放；*n 取元素个数（65536）。 */
int qi_qwen_fixed_noise(uint16_t** out_fp16, int* n, char* err, int err_len);

const char* qi_qwen_version(void);

#ifdef __cplusplus
}
#endif
#endif  // QI_WDN_QWEN_API_H
