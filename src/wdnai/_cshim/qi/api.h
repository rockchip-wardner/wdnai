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

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qi_qwen_session qi_qwen_session;

/* 噪声模式 */
#define QI_NOISE_FIXED 0   /* 用 <model_dir>/reference_512/hidden_states.bin */
#define QI_NOISE_FRESH 1   /* 现场生成（种子 = seed + 图片下标） */

/* 打开会话：加载 TE（常驻）+ DiT 头 + DiT 4 段（常驻，共享 internal）。
 * 失败返回 NULL，err 里是中文原因。 */
qi_qwen_session* qi_qwen_open(const char* model_dir,
                              const char* card_te,
                              const char* card_dit,
                              uint32_t core_mask,
                              char* err, int err_len);

void qi_qwen_close(qi_qwen_session* s);

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
 *   ③ **释放 DiT**（DiT 与 VAE 物理上不能共卡）
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

const char* qi_qwen_version(void);

#ifdef __cplusplus
}
#endif
#endif  // QI_WDN_QWEN_API_H
