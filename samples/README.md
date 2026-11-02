# samples/

用来放 wdnai 的用法示例。每个示例建议是一个**可独立运行**的文件（`.py`
或 `.sh`），方便用户直接 `python samples/<name>.py` / `bash samples/<name>.sh`
看到效果。

## 添加新示例的约定

- 顶部加一段 docstring 说明这个示例在演示什么
- 不要依赖 pytest 之类的测试框架 —— 直接 `if __name__ == "__main__":` 跑
- 避免硬编码绝对路径，使用相对路径或环境变量
- 输出尽量用 `print()` 而不是 logging，方便用户直接看到结果

## 示例索引

| 文件 | 演示内容 |
| --- | --- |
| `qwenimage21.sh` | Qwen-Image-2.1 → RKNN(rk1820) 512×512 文生图：prompt → PNG，并打印各阶段耗时 |

## `qwenimage21.sh` —— 文生图一键脚本

```bash
bash samples/qwenimage21.sh "a cute cat sitting on a wooden chair"       # 8 步
bash samples/qwenimage21.sh "一只戴眼镜的猫" 20                            # 20 步
bash samples/qwenimage21.sh "一只猫" 8 --out /tmp/myimg                     # 指定输出目录
bash samples/qwenimage21.sh "一只猫" 8 --fixed                              # 固定噪声（可复现 / 可对拍）
```

产物（`<out>/`）：

| 文件 | 内容 |
| --- | --- |
| `img01.png` | 最终图（512×512 RGB） |
| `img01_latents.bin` | DiT 潜变量（token 主序 fp16） |
| `img01_vae_image.bin` | VAE 输出（NCHW `[1,4,512,512]` fp16）+ `.meta` |
| `te/` | text encoder 的中间产物（`encoder_hidden_states*`） |
| `test.log` | 完整运行时日志 |

脚本会打印各阶段耗时（环境准备 / TE / DiT+VAE / 总计）。板端实测（RK3588 +
2×RK1828）：8 步 ≈ **94 s**、20 步 ≈ 125 s、40 步 ≈ 177 s。

模型包默认在 `/home/rockchip/models/qwenimage21`（用 `QI_MODEL_DIR` 或
`WDNAI_QWEN_IMAGE_21_DIR` 覆盖）。

⚠️ 三条硬性要求（脚本里已经设好，改脚本前先看 `deploy/debug.md` §73）：

- `QI_SPREAD_IO=0` —— **输出** IO 绝不能分散（runtime 只往它自己指定的池写
  结果，换池 ⇒ 输出恒为全 0）
- `QI_SPREAD_IO_IN=1` —— 只分散**输入** IO（否则 VAE 第 3 段 `create_mem` 失败）
- `--vae-stream` —— VAE 分段链必须逐段释放（否则 4 段同时常驻 ⇒
  `MODEL_SETUP fail`，而且是在 DiT 跑完之后才炸）

> Python 侧的等价入口是 `examples/qwen_image_21.py`（走
> `LLMLocalSession.generate_image()`），两者跑的是同一套模型与运行时。
