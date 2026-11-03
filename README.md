# wdnai

RK 芯片（RK3588 + PCIe AI 协处理器 RK1828）上的 **LLM / VLM / TTS / ASR / Embedding**
访问 SDK —— 基于 `librknn3_api.so` 的 **ctypes 直连**，不依赖 `rknn3-toolkit-lite`
Python wheel。

- **纯 Python + ctypes**：只绑定 SDK 真正用到的那部分 RKNN3 ABI，`import wdnai`
  不会打开 `.so`（库是惰性加载的，避免和 compute worker 抢设备）。
- **能力词表收归 SDK**：`swcap_*` 是全仓唯一权威定义（见 `wdnai.capability`）。
- **遥测同源**：`wdnai.RknnSmi` 是 `rknn-smi` CLI 背后那套 C ABI 的进程内封装，
  高频轮询不用 fork 子进程。

> 本包**不在 PyPI**，只在公司内网 Git 上，按下面的「安装」从 SSH 装 editable。

---

## 安装

前提：设备上要有 Rockchip 运行时 —— `/usr/lib/librknn3_api.so`（推理）与
`/usr/lib/librknnsmi.so`（遥测）；两者都不在时 `import wdnai` 仍然安全，只是
用到对应能力时才报错。

```bash
# 从内网 Git 装 editable（推荐）
pip install -e "git+ssh://git@10.10.10.10/wardner/wdnai.git#egg=wdnai"

# 或者先 clone 到本地再装
git clone ssh://git@10.10.10.10/wardner/wdnai.git
pip install -e ./wdnai
```

Python 要求 **>= 3.9**。运行期依赖：`numpy`、`transformers`、`tokenizers`、
`pydantic`、`opencv-python`、`gguf`。

> **注意**：`transformers` / `tokenizers` / `gguf` 是**模型加载的必经之路**，
> 不是可选传递依赖 —— RKNN 模型自带 `*.tokenizer.gguf`，加载时要靠它们解析
> tokenizer。若用 `pip install --no-deps` 装本包，请自行把这几个包补齐，
> 否则模型加载会报 `ModuleNotFoundError: No module named 'tokenizers'`。

开发/测试额外依赖：`pip install -e ".[dev]"`（pytest + build）。

---

## 快速上手

### 文本对话

```python
import asyncio
from wdnai import LLMLocalSession, SystemLanguageModel

async def main():
    session = LLMLocalSession(model=SystemLanguageModel.qwen3_4b(), npu=0)
    try:
        text = await session.respond(prompt="用一句话介绍 RK3588。")
        print(text)
    finally:
        session.close()

asyncio.run(main())
```

> `LLMSession(model=...)` 是**旧写法**（现在会发 `DeprecationWarning`，内部
> 其实返回 `LLMLocalSession`）。进程内推理请直接用 `LLMLocalSession`；
> 跨进程 / 跨机走 DDS RPC 的话用 `wdndds.LLMRPCSession` —— 两者接口一致。

### 视觉：图片 / 视频帧

```python
import cv2
from wdnai import ImageAttachment, LLMLocalSession, SystemLanguageModel

img = cv2.imread("demo.jpg")            # **BGR** ndarray（OpenCV 约定）
session = LLMLocalSession(model=SystemLanguageModel.qwen3_vl_4b(), npu=0)
text = await session.respond(
    prompt="这张图里有什么？",
    attachment=[ImageAttachment(img)],  # 多帧就传 list
)
```

- 视频**不用**特殊类型：自己按帧采样，把帧列表交给 `ImageAttachment([f0, f1, ...])`。
- 目录里同时有 384×384 与 768×768 两套 vision 模型时，SDK 会**两个都加载**，
  并按输入图的最长边自动挑最接近的那一套（命名约定见 `src/wdnai/models/README.md`）。

### 结构化输出（`generating=`）

```python
from typing import Literal
from wdnai import Field, Generable, LMOutput

@Generable
class Event(LMOutput):
    act: str = Field(description="人物行为")
    level: Literal["Normal", "Warning", "Danger"] = Field(default="Normal")

result = await session.respond(prompt="判断画面里的行为", attachment=[img], generating=Event)
if isinstance(result, Event):
    print(result.act, result.level)
else:
    print(result)        # 模型没吐出符合 schema 的 JSON → 拿到的是原文
```

`generating=` 的 schema 由 SDK 拼进 prompt；返回值经 pydantic 校验，
校验不过时**不抛异常**，而是把模型原文返回给你（见上面 `else` 分支）。

### 工具调用（`tools=`）

```python
from wdnai import Callable

@Callable
def alarm(parm: Event):
    """发现异常行为时触发报警"""
    print("!!! 报警 !!!", parm)

await session.respond(prompt="检查危险行为", attachment=[img], generating=Event, tools=[alarm])
```

`@Callable` 装饰的函数会被注册成「模型可调用的工具」；`@Generable` 的 schema
是它的入参类型。

### 流式 / 多轮 / 其它

```python
await session.respond(prompt="...", on_delta=lambda chunk: print(chunk, end=""))
await session.respond(prompt="...", history=[{"role": "user", "content": "上一轮"}])

session.reset()          # 清 C 层 KV cache（多轮之间记得调）
session.stop()           # 取消正在跑的推理
await session.majority_vote(k=3, prompt="...")   # 多次采样投票
await session.embed_text("要检索的文本")          # embedding 模型
```

---

### 文生图（Qwen-Image-2.1）

`kind="image"` 的模型走 `wdnai.backends.image.ImageBackend`：文本 → 512×512
图像。**输出是图像不是文本**，所以要用能带附件的入口拿图。

```python
from wdnai import LLMLocalSession, SystemLanguageModel

session = LLMLocalSession(model=SystemLanguageModel.qwen_image_21_512())

# ① 最短路径：直接拿 BGR 像素
img = await session.generate_image("a cute cat sitting on a wooden chair")
# img.shape == (512, 512, 3)，dtype uint8（BGR，可直接 cv2.imshow）

# ② run_prompt：返回 (text, attachments)
text, atts = await session.run_prompt("a cute cat")
bgr = atts[0].images[0]

# ③ respond + LMOutput：结构化外壳（attachment 自动填充）
out = await session.respond("a cute cat", generating=LMOutput)
bgr = out.attachment[0].images[0]
```

⚠️ `respond()` **不带** `generating=` 时只返回文本，图片会丢 —— 那是
`respond` 的契约（它承诺给调用方一个**文本**结果），要图就用上面三条之一。

⭐⭐⭐ **运行时是 wdnai 自己的 C shim**（`src/wdnai/_cshim/libqi_qwen.so`，
**源码随包发布**）—— 不是外部二进制、也不是任何 Rockchip 的闭源产物。
它只链**系统自带**的 `librknn3_api.so` ⇒
模型包里**只有模型**，不���任何 `.so` / 可执行文件。

```bash
# cshim 在 pip install 时自动编译；也可手动：
cd src/wdnai/_cshim && make          # 或 bash tools/build_cshim.sh
```

⚠️ 编译需要 C++ 编译器（板上 `/usr/bin/g++` 已有）；缺编译器时安装不会失败，
只是文生图不可用（其它后端不受影响）。

运行时参数（可写进 `spec.options` 或逐次传）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `steps` | `8` | flow-match 步数。8 步 ≈ 94 s；20 步更清晰 ≈ 125 s；40 步 = 上游默认 |
| `fm_mu` | `0.5387096774` | 512 路线的调度系数（**不能**用 1024 路的 0.6935483871） |
| `noise_fresh` | `True` | 每次换随机初始噪声；`False` ⇒ 固定噪声（可复现 / 可与金标准对拍） |
| `keep_artifacts` | `False` | 保留中间产物（潜变量 / VAE 原始张量） |
| `work_dir` | 临时目录 | 指定工作目录（产物落在这里） |

⏱ 板端实测（RK3588 + 2×RK1828，8 步）：**TE 与 DiT 都常驻**
（卡 0 = TE + 头，卡 1 = DiT 4 段 → VAE）⇒ 一次性 init ≈ 78 s，
之后**每张图 ≈ 34 s**（flow-match 21 s + VAE 11 s，TE 编码只要 0.16 s）。

命令行等价物（不写 Python 也能出图）：

```bash
.venv/bin/python .venv/src/wdnai/examples/qwen_image_21.py \
    "a cute cat sitting on a wooden chair" 8 --out /tmp/qi21

# ⭐ 批量："|" 分隔多个提示词 ⇒ 一个进程跑完 N 张
.venv/bin/python .venv/src/wdnai/examples/qwen_image_21.py \
    "a cute cat sitting on a wooden chair|一只戴眼镜的猫" 8 --out /tmp/qi21
```

---

## 能力词表 `SoftwareCapability`

`swcap_*` 描述的是「**这个模型实例在本机实际能做什么**」，不是「哪个业务用它」。
定义收在 `wdnai.capability`，manifest 校验和运行时取 session 都用它。

| 能力串 | 含义 |
|---|---|
| `swcap_text_chat` | 文本 → 文本（多轮 + 流式） |
| `swcap_vision_chat` | 图像 → 文本（描述 / 检测描述） |
| `swcap_vision_grounding` | 图像 → 0–1000 归一化 bbox + 文本（需 LocateAnything） |
| `swcap_vision_embedding` | 图像/视频 → 检索向量 |
| `swcap_audio_chat` | 音频 → 文本 |
| `swcap_text_to_speech` | 文本 → 语音 |
| `swcap_text_to_image` | 文本 → 图像（Qwen-Image-2.1，512×512） |
| `swcap_tool_call` | 函数 / 工具调用 |

```python
from wdnai import SoftwareCapability, migrate_legacy

SoftwareCapability.VISION_CHAT.value        # 'swcap_vision_chat'
migrate_legacy("video_describe")            # → (SoftwareCapability.VISION_CHAT, True)
migrate_legacy("swcap_not_a_thing")         # → ValueError
```

`migrate_legacy()` 负责**读老配置**：裸旧名（`video_describe`）和上一版
`swcap_*` 名都能迁；已下线的能力（`video_query` / `video_analyse`）抛
`RetiredCapabilityError` 而不是静默映射成一个没人声明的名字。

---

## 遥测 `RknnSmi`

```python
from wdnai import RknnSmi

smi = RknnSmi()
for dev in smi.get_device_list():
    print(dev["index"], dev["mode"], smi.get_device_temperature(dev["index"]))
```

- 设备列表 / 型号 / PCIe BDF / 温度 / 功耗 / 电压 / 电流 / NPU 利用率 /
  显存 / 健康状态 / 版本 / 复位。
- 温度单位是 **°C**。设备级温度命令在部分 EP 固件上没人应答（RK1828），
  `get_device_temperature()` 会**自动回退到 chip 级**并记住，后续不再触发
  失败的 PCIe 命令 —— 调用方无须关心两级差异。
- 需要 root 才能拿到 `/tmp/rknnsmi.lock` 的 flock（同一时刻另起
  `rknn-smi` CLI 会因此 init 失败）。

---

## ModelSpec 速查

一个 `ModelSpec` = **一个模型实例的全部静态配置**。

| 字段 | 说明 |
|---|---|
| `name` | 模型名，如 `qwen3-vl-4b` |
| `kind` | `llm` / `vlm` / `tts` / `omni` / `asr` / `embedding` |
| `variant` | 权重族：`qwen3` / `qwen3-vl` / `qwen2.5-omni` / `qwen3-tts` / … |
| `rknn_path` / `weight_path` | LLM 的 RKNN 与权重文件 |
| `rknn_llm_path` / `rknn_vision_path` | VLM 的两套 RKNN（+ 各自的 `weight_*`） |
| `tts_model_dir` / `omni_model_dir` / `asr_model_dir` | 按目录组织的模型（TTS / Omni / ASR） |
| `embed_path` / `tokenizer_path` | embedding 向量文件 / tokenizer 文件 |
| `vocab_size` | 词表大小（Qwen3.5-VL 是 **248320**，别照抄 Qwen3 的 151936） |
| `special_eos_id` | 特殊 EOS id（Qwen3.5-VL 是 **248046**） |
| `max_context_len` | 上下文长度 |
| `image_size` | 视觉输入边长（384 / 576 / 768 …） |
| `use_prune` | 是否启用剪枝 |
| `target` / `core_mask` | `rk1820` / `rk1828`；`core_mask` 选 NPU 核心 |
| `verbose` | 运行时日志 |
| `options` | 采样参数（`temperature` / `top_p` / `top_k` / `repeat_penalty` …），会并进 `init_runtime` |

改路径 / 改采样：

```python
spec = SystemLanguageModel.qwen3_vl_4b().with_paths(rknn_llm_path="/tmp/x.rknn")
spec = spec.with_options(temperature=0.2, top_p=0.9)
```

### 现成 preset

`SystemLanguageModel.<preset>()`：

| preset | 模型目录（`$WDNAI_MODELS_DIR` 下） | 能力 |
|---|---|---|
| `qwen2_5_0_5b` / `qwen3_1_7b` / `qwen3_4b` | `Qwen3-4B/` 等 | 文本对话 |
| `qwen2_5_vl_3b` / `qwen2_5_vl_7b` | `Qwen2.5-VL-*` | 视觉对话 |
| `qwen3_vl_2b` / `qwen3_vl_4b` / `qwen3_vl_4b_template` | `Qwen3-VL-4B/` | 视觉对话（template 版不绑默认路径） |
| `qwen3_5_vl_4b` | `Qwen3.5-4B/` | 视觉对话 + 工具调用 |
| `qwen2_5_omni_3b` | `Qwen2.5-Omni-3B/` | 音频 → 文本 |
| `qwen3_tts` | `Qwen3-TTS/` | 文本 → 语音 |
| `qwen3_asr` | `Qwen3-ASR/` | 语音 → 文本 |
| `qwen3_vl_embedding` | `Qwen3-VL-Embedding-2B/` | 检索向量 |
| `locate_anything` | `LocateAnything-3B/` | 目标定位（bbox） |
| `qwen_image_21_512` | `qwenimage21/` | 文生图（512×512） |

preset 会在**构造时**检查文件是否齐全，缺文件抛 `FileNotFoundError` 并列出
缺哪些 —— 不会等到推理时才炸。

---

## 模型文件放哪

默认根目录 `/userdata/models/`（可用 `WDNAI_MODELS_DIR` 改）：

```
/userdata/models/Qwen3-VL-4B/
├── llm_Qwen3-VL-4B.1828.rknn        # LLM RKNN
├── llm_Qwen3-VL-4B.weight
├── vision_Qwen3-VL-4B.384x384.1828.rknn   # 视觉变体（多分辨率）
├── vision_Qwen3-VL-4B.384x384.weight
├── vision_Qwen3-VL-4B.768x768.1828.rknn
├── vision_Qwen3-VL-4B.768x768.weight
├── Qwen3-VL-4B.embed.bin
└── Qwen3-VL-4B.tokenizer.gguf        # RKNN 自带的 GGUF tokenizer
```

命名约定与各模型的完整文件表见 [`src/wdnai/models/README.md`](src/wdnai/models/README.md)。

### 环境变量

| 变量 | 作用 |
|---|---|
| `WDNAI_MODELS_DIR` | 模型根目录（默认 `/userdata/models`） |
| `WDNAI_QWEN3_VL_4B_DIR` / `WDNAI_QWEN3_4B_DIR` / `WDNAI_QWEN3_5_4B_VL_DIR` / `WDNAI_QWEN_2_5_OMNI_3B_DIR` / `WDNAI_LOCATE_ANYTHING_DIR` | 单个模型的目录覆盖 |
| `WDNAI_QWEN_IMAGE_21_DIR` / `QI_MODEL_DIR` | Qwen-Image-2.1 模型包目录（默认 `$WDNAI_MODELS_DIR/qwenimage21`，板上实际在 `/home/rockchip/models/qwenimage21`） |
| `QI_SPREAD_IO` / `QI_SPREAD_IO_IN` | cshim 内部默认 `0` / `1`（输出 IO 绝不能分散） |
| `WDNAI_LIBRKNN3_API` | `librknn3_api.so` 路径 |
| `WDNAI_LIBRKNNSMI` | `librknnsmi.so` 路径 |
| `WDNAI_TRACE_PROMPT` | 打印每次推理实际发出的 prompt（排查 prompt 污染用） |

---

## 已知限制

- **同一个模型句柄不可重入**：一次 `respond()` 没结束就别再发第二次；SDK 内部
  用单 worker 线程串行化，但跨进程要靠调用方自己保证。
- **显存是硬约束**：一张 RK1828 5 GB，装不下两个 4B VLM。模型常驻后别再在同一张
  卡上启第二个。
- **`Enum.__contains__` 的版本差异**：`migrate_legacy()` 已改成「按值查表」，
  在 Python 3.9 ~ 3.12 上语义一致（3.11 的 `Enum.__contains__` 对非枚举成员
  抛 TypeError，不能用 `value in SoftwareCapability` 判成员）。
- **chip 级温度与设备级温度单位不同**（整度 vs 毫度），SDK 内部各自换算，
  对外都是 °C。

---

## 目录导航

| 路径 | 内容 |
|---|---|
| `src/wdnai/session.py` | `LLMSession` 抽象接口 + `LLMLocalSession` 进程内实现 |
| `src/wdnai/model.py` | `ModelSpec` + `SystemLanguageModel` preset 目录 |
| `src/wdnai/capability.py` | `swcap_*` 词表 + 迁移表 |
| `src/wdnai/attachment.py` | `TextAttachment` / `ImageAttachment` / `AudioAttachment` |
| `src/wdnai/_generable.py` / `_callable.py` | 结构化输出与工具调用的装饰器 |
| `src/wdnai/_librknn.py` | librknn3 的 ctypes 绑定 |
| `src/wdnai/_librknnsmi.py` | librknnsmi 的 ctypes 绑定 + `RknnSmi` |
| `src/wdnai/grounding.py` | LocateAnything 的 bbox schema |
| `examples/` | 可直接跑的示例（图片检测、视频车牌、函数调用、TTS …） |
| `tests/` | pytest（不占硬件） |
| [`AGENTS.md`](AGENTS.md) | 给 AI / 维护者的详细内部约定（踩坑记录都在这） |

## 许可

Apache-2.0
