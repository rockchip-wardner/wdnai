# samples/

用来放 wdnai 的**命令行入口**。每个示例建议是一个**可独立运行**的文件
（`.py` 或 `.sh`），方便用户直接 `python samples/<name>.py` /
`bash samples/<name>.sh` 看到效果。

## 添加新示例的约定

- 顶部加一段 docstring 说明这个示例在演示什么
- 不要依赖 pytest 之类的测试框架 —— 直接 `if __name__ == "__main__":` 跑
- 避免硬编码绝对路径，使用相对路径或环境变量
- 输出尽量用 `print()` 而不是 logging，方便用户直接看到结果

## 示例索引

| 文件 | 演示内容 |
| --- | --- |
| （暂无） | — |

> ⭐ **2026-11-02 第三十五轮**：`qwenimage21.sh` 已删除。
>
> 原因：**文生图只有一个入口就够**，而 shell 版本与 Python 版本**功能重复**
> （两套参数、两套噪声语义、两套 PNG 转换路径）⇒ 迟早会走偏。
>
> 现在统一用 `examples/qwen_image_21.py`（参数与旧 sh 完全对齐，见下）。
>
> ```bash
> cd /home/rockchip/wdnwebui
> .venv/bin/python .venv/src/wdnai/examples/qwen_image_21.py \
>     "a cute cat sitting on a wooden chair" 8 --out /tmp/qi21
> ```

## `examples/qwen_image_21.py` —— 文生图一键入口

```bash
# 单张
.venv/bin/python .venv/src/wdnai/examples/qwen_image_21.py \
    "a cute cat sitting on a wooden chair" 8 --out /tmp/qi21

# 20 步（更清晰，耗时更长）
.venv/bin/python .venv/src/wdnai/examples/qwen_image_21.py \
    "一只戴眼镜的猫" 20 --out /tmp/qi21

# ⭐ 批量："|" 分隔多个提示词 ⇒ **一个进程跑完 N 张**
#    （TE / DiT / VAE 都只 model_init 一次 ⇒ 每张图边际 ≈ 34 s）
.venv/bin/python .venv/src/wdnai/examples/qwen_image_21.py \
    "a cute cat sitting on a wooden chair|一只戴眼镜的猫" 8 --out /tmp/qi21

# 固定初始噪声（可复现 / 可与 PyTorch 金标准对拍）
.venv/bin/python .venv/src/wdnai/examples/qwen_image_21.py \
    "a cute cat" 8 --fixed --seed 0 --out /tmp/qi21
```

产物（`<out>/`）：

| 文件 | 内容 |
| --- | --- |
| `img01.png` | 最终图（512×512 RGB）；批量时依次 `img02.png` / `img03.png` … |
| `img01_latents.bin` | DiT 潜变量（token 主序 fp16） |
| `img01_vae_image.bin` | VAE 输出（NCHW `[1,4,512,512]` fp16）+ `.meta` |

⏱ 板端实测（RK3588 + 2×RK1828，8 步）：**TE 与 DiT 都常驻** ⇒
一次性 init ≈ 78 s，之后**每张图 ≈ 34 s**（flow-match 21 s + VAE 11 s）。
