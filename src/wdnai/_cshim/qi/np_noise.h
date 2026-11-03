// ---------------------------------------------------------------------------
// numpy 兼容的固定初始噪声（Qwen-Image-2.1 / 512 路线）
//
//   与模型包里的 reference_512/hidden_states.bin **逐字节一致**，
//   但不需要随包发布那个文件 ⇒ 发布包只剩「模型」。
//
//   等价的 Python 表达式：
//       np.random.default_rng(0).standard_normal(64**3)
//           .reshape(64, 64, 64)[::2, ::2, :].reshape(-1).astype(np.float16)
// ---------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <vector>

namespace qi {

/* 生成 1024 个 token × 64 通道的 fp16 初始噪声。
 * 成功返回 true，out 里是 65536 个 fp16（IEEE 小端位模式）。 */
bool np_fixed_noise_512(std::vector<uint16_t>* out);

}  // namespace qi
