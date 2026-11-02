# wdnai examples —— sample 使用说明

本目录是 wdnai SDK 的**可运行 sample**，每个 sample 对应一个模型 / 一类能力，
是端到端消费者。`AGENTS.md` 定的开发顺序就是：

> **先在 `examples/` 写调用该模型的 demo → 再改 `model.py` / `session.py` /
> `backends/*.py` → 最后跑 demo 调通。**

因此 sample 不是玩具代码，而是**这个模型对外契约的可执行文档**：sample 怎么调、
输入长什么样、输出每一行代表什么、哪些坑会反复踩 —— 都在这里写清楚。

---

## 目录

| sample | 模型 / 能力 | 状态 |
|---|---|---|
| [`depth_anything_v3.py`](depth_anything_v3.py) | Depth Anything V3 单目深度估计 | ✅ 已记录 |
| [`audio_to_text.py`](audio_to_text.py) | Qwen3-ASR 语音转文字 | 待补 |
| [`audio_understanding.py`](audio_understanding.py) | Qwen2.5-Omni-3B 音频理解 | 待补 |
| [`detect_image.py`](detect_image.py) | Qwen3-VL-4B 图像推理 | 待补 |
| [`detect_video.py`](detect_video.py) | Qwen3-VL-4B 视频流推理 | 待补 |
| [`detect_video_carplate.py`](detect_video_carplate.py) | Qwen3-VL-4B 车牌检测 | 待补 |
| [`function_call.py`](function_call.py) | 工具调用 / function call | 待补 |
| [`locate_image.py`](locate_image.py) | LocateAnything 目标定位 | 待补 |
| [`qwen3_5_vl_image.py`](qwen3_5_vl_image.py) | Qwen3.5-VL-4B 图像描述 | 待补 |
| [`qwen35_vl_9b_multicard.py`](qwen35_vl_9b_multicard.py) | Qwen3.5-VL-9B 双卡 | 待补 |
| [`qwen_image_21.py`](qwen_image_21.py) | Qwen-Image-2.1 文生图（512×512） | ✅ 已记录 |
| [`summarize_text.py`](summarize_text.py) | Qwen3-4B 纯文本总结 | 待补 |
| [`test_three_models_one_npu.py`](test_three_models_one_npu.py) | 单卡多模型共存 | 待补 |
| [`text_to_speech.py`](text_to_speech.py) | Qwen3-TTS 语音合成 | 待补 |

---

## 规范：新增 sample 必须按此记录

**新增任何 sample，都要在本 README 里加一节，固定 4 个小节 + 素材目录约定。**
顺序与标题不要改，这样后来的人能按同一套心智模型读所有 sample。
性能敏感 / any-view 类模型还要额外写「性能实测」和「view 数与实时流」
（见下面第 6、7 条）。

### 1. 节标题

`## <sample 文件名> —— <模型 / 能力的一句话描述>`

文件名和节标题用**同一个标识符**，方便跳转。

### 2. `### 怎么跑`

- 开头给一条**可以直接复制粘贴**的默认命令（默认参数就能跑通的那条）；
- 再列常用变体：换素材、换输出目录、只算不画图、换 NPU 卡；
- 标注**大致耗时**和**是否占 NPU**（占 NPU 的要提醒别并发跑）。

### 3. `### 输入`

- 一张表：**每个命令行参数**是什么含义、默认值是什么；
- 再单独一小节写**素材目录约定**（见下）；
- 明确写清**参数之间的优先级**和**硬性数量约束**。

### 4. `### 输出`

- 一张表：**每一行 stdout**（或每个产出文件）代表什么；
- 数值型输出要注明是「什么量纲 / 什么单位」，别让读者自己猜。

### 5. `### 容易踩的坑`

- 只写**已经踩过、或读代码看不出来的**坑；
- 每条写清**症状 → 原因 → 怎么办**；
- 共享组件 / 驱动层面的坑要指明影响哪些后端。

### 6. `### 性能实测`（性能敏感的模型要写）

- 把**加载**和**推理**都拆到每个阶段，给出实测值和测量方法；
- 注明平台、模型大小、是否有 page cache 之类的变量；
- 特别区分「含加载」和「纯推理」—— 拿含加载的数字去算吞吐会错一个数量级。

### 7. `### view 数与实时流`（any-view / 视频类模型要写）

- view 数是**导出时**参数还是运行期参数？换它要改代码吗？
- 有效 view 数变化对结果的影响（要有实测表）；
- 实时流的硬约束（攒帧延迟、输入必须同源、顺序是否重要）。

### 素材目录约定

sample 需要的图片 / 音频素材，统一放**与 sample 同名的子目录**：

```
examples/
├── <sample_name>.py
└── <sample_name>/          # 素材目录，与 sample 同名
    ├── frame_01.jpg
    ├── frame_02.jpg
    └── images.txt          # 清单文件：每行一个路径
```

- `images.txt` 里用**裸文件名**（相对清单自身所在目录解析），
  这样整个目录可以整体搬走而不必改内容；写绝对路径同样有效；
- 清单里的相对路径按「**清单自身目录 → 其父目录 → 当前 cwd**」依次解析，
  空行忽略 —— 这个解析规则所有 sample 共用；
- 素材从 model-zoo 复制过来时，先核对 md5 与原始素材一致，
  否则 sample 的输出会与 C++ demo 对不上。

---

## depth_anything_v3 —— Depth Anything V3 单目深度估计

> 深度估计**不是**文本生成：图像进，**逐像素深度图 + 置信度**出。
> 这个 sample 的 stdout **逐行对齐** C++ 参考实现
> （`rknn3-model-zoo/examples/depth_anything_v3/cpp/main.cpp`），
> 所以两边可以直接并排 diff。

> **⚠️ 当前限制**：本版本**只完成基础的 10-view（多视图融合）**。
> `global` 段在 10 张图之间做融合，送图张数必须正好是 10。
> **单路（单图）深度估计尚未实现**。
>
> 后续路线（详见下方「view 数与实时流」一节）：
>
> | 路线 | 代价 | 要改什么 |
> |---|---|---|
> | **`--views 5`**（推荐先做） | 精度差 1.58%，延迟减半 | **重新导出三段 RKNN，SDK / sample / worker 零改动** |
> | `--views 1` | 精度差 4.14%，真·实时 | 同上，仍然是重新导出 |
> | 多相机几何 | — | model-zoo 要改导出脚本加 **ray** 分支，且需换更大的模型（DA3-Base 的位姿精度只有 Giant 的 1/4） |
>
> **V 是导出时参数**：backend 从模型上读 view 数，所以换 V **不需要改代码**。

### 怎么跑

前置条件：模型文件已在 `/userdata/models/Depth-Anything-V3/`（见「坑」第 6 条）。

```bash
cd /userdata/wdnwebui

# 默认：用本 sample 自带的 10 张图，热力图输出到 ./depth_vis
.venv/bin/python .venv/src/wdnai/examples/depth_anything_v3.py

# 指定输出目录
.venv/bin/python .venv/src/wdnai/examples/depth_anything_v3.py --output-dir /tmp/da3_vis

# 换自己的素材（张数必须正好 10，见「坑」第 1 条）
.venv/bin/python .venv/src/wdnai/examples/depth_anything_v3.py \
    --image-list /path/to/my_images.txt

# 直接给图（可重复；同样必须正好 10 张）
.venv/bin/python .venv/src/wdnai/examples/depth_anything_v3.py \
    --image a.jpg --image b.jpg ...

# 只算不画热力图（CI / 批量跑用）
.venv/bin/python .venv/src/wdnai/examples/depth_anything_v3.py --no-vis

# 指定 NPU 卡 / 覆盖模型目录
.venv/bin/python .venv/src/wdnai/examples/depth_anything_v3.py --npu 1
.venv/bin/python .venv/src/wdnai/examples/depth_anything_v3.py --model-dir /path/to/models
```

- **耗时**：**首次约 18.4 s**（几乎全是模型加载），**稳态推理约 1.34 s**。
  分阶段实测见下方「性能实测」。
- **占 NPU**：是。同一时刻别再跑别的 sample，否则会抢卡。
- **默认 `--output-dir` 是 `depth_vis`，相对当前工作目录**（对齐 C++ demo）。
  在 `examples/` 目录里直接跑会把热力图落在源码目录 —— 已在 `.gitignore`
  里忽略，但建议始终显式给 `--output-dir`。
- **一致性回归**：`python -m scripts.verify_da3_parity`（在 `wdnwebui` 仓）
  会并排跑 C++ demo 与本 sample，逐项比对张量统计与热力图。

### 输入

#### 命令行参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--image-list` | `examples/depth_anything_v3/images.txt` | 图片清单文件，**每行一个路径**。相对路径按「清单自身目录 → 其父目录 → 当前 cwd」依次解析；绝对路径直接用；空行忽略 |
| `--image` | 无 | 直接给图片路径，**可重复**。与 `--image-list` **二选一**；两者都给时 `--image` 优先 |
| `--output-dir` | `depth_vis` | 热力图输出目录（对齐 C++ demo 的默认目录名） |
| `--npu` | `0` | NPU 卡索引，对应 `RKNN3Lite.get_devices_id()` 的下标 |
| `--no-vis` | 关 | 不画 / 不写热力图，只跑推理与统计 |
| `--model-dir` | 无 | 覆盖模型目录，等价于设环境变量 `WDNAI_DEPTH_ANYTHING_V3_DIR` |

#### 素材

```
examples/depth_anything_v3/
├── frame_01.jpg .. frame_10.jpg   # 从 model-zoo 的 DA3 demo 复制而来（md5 已核对一致）
└── images.txt                      # 每行一个裸文件名
```

`images.txt` 内容：

```
frame_01.jpg
frame_02.jpg
...
frame_10.jpg
```

#### 数量硬约束

**图片张数必须正好等于模型的 view 数（本导出 = 10）。**
view 数是**烤进 RKNN 导出**的（`global` 段输入形状就是 `[V, L, 1, D]`），
不是运行期参数 —— 送 3 张或 12 张都不行，改张数必须重新导出模型。

本 sample 在跑任何推理**之前**就会校验并给出明确报错
（backend 侧 `estimate()` 也有同样的检查）。

### 输出

#### stdout 各行

| 行 | 含义 |
|---|---|
| `mode=depth views=10 model=DepthAnythingV3 image=384x384` | `views` = 实际喂进去的图片张数；`image=384x384` 是 `spec.image_size` 的**名义提示值，不是真实输入分辨率**（真实值看下一行） |
| `pipeline config: views=10 image=3x280x280 local_tokens=401 hidden=768` | **真实几何**，从模型张量属性上读出来：输入 3×280×280、每个 view 401 个 token、hidden 768 |
| `origin size=1024x576 crop size=1024x576` | 原图分辨率（自带素材是 1024×576） |
| `input image: 1024 x 576, subsampling: 4:2:0, colorspace: YCbCr, orientation: 1` | 复刻 C++ demo 的 libjpeg 信息行（对齐输出格式用） |
| `depth visualization valid region: x=[0,280) y=[0,280)` | 深度可视化的有效区域 |
| `local view N/10 done` | 第 N 张图过完 `local` 段（逐图出 token） |
| `[time] local view N preprocess … ms` | 第 N 张图的预处理耗时 |
| `local_tokens: elems=3079680 min=… max=… mean=…` | `local` 段输出张量统计。`elems = 10 × 401 × 768` |
| `depth: elems=784000 min=… max=… mean=…` | 深度图。`elems = 10 × 280 × 280`；**值是逆深度**（见「坑」第 2 条） |
| `confidence: elems=784000 min=… max=… mean=…` | 逐像素置信度 |
| `shared inverse-depth visualization range: [lower, upper]` | 10 张图**共享**的逆深度着色窗口（2% / 98% percentile 截断） |
| `saved depth heatmap: <dir>/depth_NN.jpg` | 每张图一张 Turbo 伪彩热力图 |
| `saved depth montage: <dir>/depth_montage.jpg layout=5x2 canvas=1400x560` | 5 列拼图，便于一眼看一组 |
| `depth output directory: <dir>` | 产出目录 |
| `[time] global run … ms` / `[time] head run … ms` / `[time] total … ms` | 各段耗时（随硬件负载浮动） |
| `DA3-BASE split RKNN pipeline finished.` | 正常结束标志 |

#### 产出文件

| 文件 | 内容 |
|---|---|
| `depth_00.jpg` … `depth_09.jpg` | 每 view 一张 Turbo 伪彩深度图 |
| `depth_montage.jpg` | 5×2 拼图（1400×560） |

#### 基准数值（自带 10 张素材，与 C++ demo **逐位相同**）

```
local_tokens: elems=3079680 min=-6.074219 max=6.269531 mean=-0.005947
depth:        elems=784000  min=0.350586  max=1.421875 mean=0.975114
confidence:   elems=784000  min=1.000000  max=5.597656 mean=2.675516
shared inverse-depth visualization range: [0.730385, 2.169492]
```

换成别的素材，数值会变 —— 上面这组只对**自带素材**成立，是「跑对了没有」的基准。

### 性能实测

> 平台：RK3588（8 核 NPU），三段模型权重共 **1276.9 MB**
> （`local` 316.6 MB + `global` 749.2 MB + `head` 197.8 MB，另有 13.3 MB `.rknn`）。
> 测法：包住 `_Runtime.__init__` / `load_model` / `model_init` / `create_tensor`
> 与 backend 的 `_share_internal_memory` / `_read_model_geometry` 计时；
> 推理跑 4 轮取后 3 轮均值。

#### 模型加载（一次性，约 18.4 s）

| 阶段 | 耗时 | 占比 | 说明 |
|---|---|---|---|
| `_Runtime.__init__` ×3 | **6.70 s** | 36% | 开 NPU 设备上下文，**2.24 s × 3**（三段各开一个 PCIe 设备句柄） |
| `model_init` ×3 | **9.83 s** | 53% | local 2.47 s / global **5.77 s** / head 1.59 s —— 与权重大小成正比 |
| 共享内部显存分配 | **0.72 s** | 4% | 8 核共约 1.6 GB（`da3_init_internal_mem_share`） |
| `load_model` ×3 | 0.04 s | — | 只是把权重路径交给 C 侧，真正的上传在 `model_init` 里 |
| `create_tensor` ×3 | 0.06 s | — | 分配输入/输出张量显存 |
| 读模型几何 | ≈ 0 s | — | 读张量属性 |
| **合计** | **≈ 18.4 s** | | 连续 5 次加载实测 18.34 / 18.35 / 18.40 / 18.45 / 18.46 s，**很稳定** |

要点：

- **加载不是 page cache 主导的** —— 第二次加载（权重已在 cache）耗时和第一次一样，
  说明这 18 s 是**真实的 PCIe 传输 + NPU 初始化**。
- 跑 sample 时看到的 `[time] total ≈ 20 s` **包含加载**，
  别把它当成推理耗时。
- 加载完的常驻占用：NPU 侧共享内部显存约 1.6 GB + 三段权重约 1.28 GB
  ≈ **2.9 GB**（与 worker manifest 里的 `memory_estimate_mb: 3000` 对得上）。

#### 推理（稳态，一次 10 view）

| 阶段 | 耗时 | 备注 |
|---|---|---|
| `local` ×10 | **0.251 s** | **0.025 s/帧**，逐图独立（见下） |
| `global` ×1 | **0.681 s** | 一次处理全部 10 个 view 的 token |
| `head` ×1 | **0.407 s** | 一次出 10 张深度图 + 10 张置信度图 |
| **合计** | **1.340 s** | 约 **0.75 次/秒** 的更新率 |

- **首轮推理没有明显的冷启动惩罚**：第 1 轮 1.367 s，之后稳定在 1.32~1.37 s。
  冷启动的代价全在**模型加载**那 18 s 上。
- **增量模式可再省 0.26 s**：`local` 段是**逐图独立**的（同一张图单独算与在序列里算，
  tokens 逐位相同），所以实时流可以缓存前 9 帧的 token，只算新帧：

  | 模式 | local | global | head | 合计 |
  |---|---|---|---|---|
  | 全量（10 帧都算） | 0.251 s | 0.681 s | 0.407 s | 1.340 s |
  | 增量（只算 1 帧） | 0.026 s | 0.662 s | 0.391 s | **1.079 s** |

- **`global` 段是真跨帧融合**：把 10 张图换成 10 张同一张，`depth[0]` 的均值
  从 0.996872 变成 0.974225（**差 4.1%**，max_diff 0.446）——
  模型确实依赖多帧上下文，**单帧推理会掉精度**。

#### 对实时流的含义

按 25 fps 相机估：攒满 10 帧要 0.36 s，加一次推理 1.1~1.4 s
→ **端到端延迟约 1.5~1.8 s**，深度更新率约 0.75 次/秒。
性能上完全够用，瓶颈在「必须攒 10 帧」这个硬约束上，不是算力。

### view 数与实时流

#### 一、DA3 的 view 数机制（依据 arXiv:2511.10647 / ICLR 2026）

DA3 = *Depth Anything 3: Recovering the Visual Space from Any Views*（ByteDance Seed）。
它不是单目深度模型，而是**任意 view 数**的几何模型：

- **架构**：单个 plain transformer（vanilla DINOv2）+ **input-adaptive cross-view
  self-attention**。骨干分两组层：前 `L_s` 层做**图内**注意力，后 `L_g` 层
  **交替**做跨 view / 图内注意力，靠 **tensor reordering** 实现，`L_s:L_g = 2:1`
  （论文消融证明「全交替」更差）。
  论文原话：*"with a single image, the model naturally reduces to monocular depth
  estimation without extra cost."*
- **训练**：504×504 下 view 数从 **[2, 18]** 均匀采样；batch size 动态调整以保持
  每步 token 数恒定（分辨率越大，可用的 view 数越少）。
- **我们的拆分**：`local` 段 = 前 `L_s` 层（逐图独立）；`global` 段 = 后 `L_g` 层
  （跨 view 建模）；`head` 段出深度 + 置信度。

#### 二、V 是**导出时**参数 —— 改 V 只需重新导出，**SDK 侧零改动**

`global` 的输入形状是 `[V, L, 1, D]`，V 在 ONNX 导出后就固定了。但 backend
**从模型上读 V**（`DepthAnythingBackend._read_model_geometry` 读
`global` 输入的 `shape[0]`），不是硬编码 —— 所以：

> **换 V 只需要重新导出三段 RKNN，SDK / sample / worker 一行都不用改。**
> `estimate()` 的张数校验、报错文案、sample 的提示都会自动跟着变。

model-zoo 的导出参数：`--views`（**已验证范围 1-10**，默认 10）、`--image_size`
（默认 280）。

#### 三、实测：有效 view 数对精度的影响

V=10 的模型跑不了 V=1，但 `local` 段逐图独立，所以可以用**重复帧填充槽位**
模拟有效 view 数递减（始终比较同一张 frame_01 的深度图）：

| 配置 | 有效不同画面 | mean\|Δ\| vs 基线 | 相对偏差 | 空间相关系数 |
|---|---|---|---|---|
| A 正序 10 帧 | 10 | 0 | 0% | 1.000 |
| B 5 帧 + 5 重复 | 5 | 0.0158 | **1.58%** | 0.995 |
| C 3 帧 + 7 重复 | 3 | 0.0259 | **2.59%** | 0.993 |
| D 全重复（≈单路） | 1 | 0.0412 | **4.14%** | 0.974 |
| F 10 帧**逆序** | 10 | 0.0397 | 3.98% | **0.998** |
| E 3 本场景 + 7 异场景 | 3 + 污染 | 0.6203 | **62.22%** | 0.845 |

三条结论：

1. **收益递减但没饱和** —— 10→5 只差 1.58%，5→1 才差 4.14%。
   攒 5 帧能吃到 10 帧 62% 的收益，**攒 5 帧比攒 10 帧划算得多**。
2. **顺序不重要** —— 逆序 3.98% ≈ 单路 4.14%，空间结构相关系数 0.998。
   `global` 的跨 view 注意力是**集合式**的，不是时序建模。
3. **绝对不能跨场景拼帧** —— 混入 7 张别的场景后偏差 **62%**，深度绝对尺度
   都塌了（0.377 vs 0.997），模型 confidence 崩到 1/3（2.92 → 1.12）。
   实时流的 V 帧必须来自**同一段连续视频**。

> **别拿 `confidence` 判断 view 数好坏**：退化输入下 confidence 反而**最高**
> （单路 3.37 > 3 帧 3.20 > 10 帧 2.92）。它衡量的是局部证据一致性，
> 不是全局正确性。

#### 四、实时流的三种形态与落地路线

| 形态 | 机制 | 延迟 | 跨帧融合 | 要改什么 |
|---|---|---|---|---|
| **A. V=1 逐帧** | 退化成单目深度估计 | ~0.1 s | 无 | **重新导出 `--views 1`**，SDK 零改动 |
| **B. V=N 滑窗**（现在） | N 帧一起过 `global` | N帧间隔 + 推理 | 有 | 重新导出 `--views N`，SDK 零改动 |
| **C. 多相机几何** | 需要位姿 | — | 有 | **model-zoo 要改导出脚本**（见下） |

**推荐路线**：

1. **`--views 5`** —— 延迟砍半，精度只差 1.58%，**SDK 零改动**，先做这个。
2. **`--views 1`** —— 真·实时（~0.1 s），代价 ~4.14%。
   论文证据：any-view 模型退化成单目后**仍然超过 DA2**（Tab.4 δ1 Rank 2.20 vs 2.60），
   所以这不是「用错模型」；只是 V=1 落在训练分布 [2,18] 之外。
   更强的选择是官方的 **DA3-Monocular**（Tab.10 ETH3D δ1 98.8 vs DA2 86.5）。
3. **多相机几何** —— 见下节的硬限制。

#### 五、多相机 / 位姿：当前走不通，原因有两条

**① DA3-Base 的位姿能力是全家族最弱的**

论文 Tab.2（位姿 Auc3）：

| | HiRoom | ETH3D | DTU | 7Scenes | ScanNet++ |
|---|---|---|---|---|---|
| DA3-Giant (1.10B) | 80.3 | 48.4 | 94.1 | 28.5 | 85.0 |
| DA3-Large (0.36B) | 58.7 | 32.2 | 70.2 | 29.2 | 60.2 |
| **DA3-Base (0.11B)** ← 我们导的就是这个 | **19.0** | **15.1** | **60.1** | **20.1** | **25.1** |
| DA3-Small (0.03B) | 9.49 | 8.59 | 30.6 | 14.0 | 10.9 |

论文自己点破了原因：*"with pose conditioning, performance gains from scaling model
size are smaller than in pose-free models, **indicating that pose estimation scales
more strongly than depth estimation**."*
→ **深度随模型缩放温和（Base 够用），位姿随模型缩放非常敏感（Base 差 4 倍）。**
多相机融合的真正瓶颈是**位姿**，不是视差。

**② 我们的导出丢掉了 ray 分支，拿不到位姿**

model-zoo 的 `da3_export/head.py` 只返回 `output.depth` 和 `output.depth_conf`
（`output_names=["depth", "depth_conf"]`），**没有 ray 导出路径**。
而多相机要融合必须有位姿 —— 论文里位姿是靠 **ray map**（每像素 6 通道的
射线原点+方向）隐式表示的。

所以当前能力边界：

- ✅ 单相机深度图（相对深度，10 view 融合）
- ❌ 相机位姿 / 外参 / 内参
- ❌ 多相机点云融合、建图、SLAM

要打通需要**两件事**（都不是配置项）：

1. **model-zoo 导出脚本加 ray 分支** —— `da3_export/head.py` 的 wrapper 要多返回
   `output.ray`（`H×W×6`，数据量比 depth 大 6 倍），并把 `"ray"` 加进
   `output_names`；
2. **换更大的模型** —— ray 分支配合 DA3-Large/Giant 才有可用的位姿精度。

> **论文里没有「视差大小 vs 精度」的定量关系**。`disparity` 一词全文只出现在
> teacher 的单目预测目标语境（DA2 预测 disparity、DA3 的 teacher 改成预测 depth），
> 与多相机几何无关；论文也没有按基线分层的评测表。能找到的只有定性结论：
> 视差大对经典 SfM/MVS 是难点（DA3 的卖点正是在这），wide-baseline 是 DA3 的
> 宣称优势场景，而「多相机甚至无重叠也能用」只出现在**项目页 demo**，
> **论文正文没有对应的定量评测**。

### 容易踩的坑

1. **张数必须正好是 10。**
   view 数烤在导出里，不是运行期参数。送 3 张曾经只在 numpy 的
   `reshape` 里炸出 `cannot reshape array of size … into shape (10,401,1,768)`
   —— 模型已经加载（**约 18.4 s**）、10 张图也已经过完 `local` 段，白等一场。
   现在 sample 在跑推理前就校验并直接报「这个 DA3 导出是为 10 个 view 建的」。

2. **`depth` 是「逆深度」，不是距离。**
   值 = `1 / depth`，**越大越近**。它不是米数、不是公尺、也不是任何真实物理量，
   只能用于相对比较（谁比谁近）。`confidence` 是模型自评的逐像素置信度，
   不是概率校准值，别拿它当「准确率」。

3. **`image=384x384` 是假的，真实输入是 280×280。**
   `mode=depth … image=384x384` 打的是 `spec.image_size`（dataclass 默认值，
   只是个提示）；`pipeline config: … image=3x280x280` 才是从模型上读出来的真值。

4. **10 张图共享一个着色窗口。**
   `shared inverse-depth visualization range` 是**跨全部 view** 一起算的
   2%/98% percentile，所以同一张图里颜色相同 = 相对深度相同。
   **不能把单张图的颜色单独解读**，也别拿单图去和另一组素材的图比颜色。

5. **热力图的 JPEG 字节和 C++ demo 不一样，但像素是一样的。**
   差异来自编码器：C++ 用 libjpeg-turbo 的 4:2:2，Python 侧用 OpenCV
   （已显式设成 `IMWRITE_JPEG_SAMPLING_FACTOR_422` 对齐）。
   解码后 `mean_diff ≈ 0.6/255`、`max_diff ≤ 14`。
   **不要用「JPEG 字节完全相同」当验收标准**，要比解码后的像素。

6. **模型文件必须就位。**
   默认路径 `/userdata/models/Depth-Anything-V3/`，需要
   `da3_base_{local,global,head}.rknn` + 同名 `.weight` 共 6 个文件
   （可用 `WDNAI_DEPTH_ANYTHING_V3_DIR` 或 `--model-dir` 覆盖）。
   缺文件会抛 `FileNotFoundError`，在 webui 侧会被 worker 转成 `errored` 状态。

7. **`--npu` 是 `get_devices_id()` 的**下标**，不是 PCI 地址。**
   板卡重启 / 驱动重载后设备枚举顺序可能变，同一个 `--npu 1` 未必还是同一块卡。
   排查时先跑一次 `RKNN3Lite().get_devices_id()` 确认当前顺序。

8. **绝不要在 NPU 调用中途 kill 进程。**
   SIGTERM 落在 `rknn3_run` 里会破坏 RK1820 的 PCIe 链路，表现为
   `rknn3_startup` 报 `ddr load addr request failed`，**C++ demo 也跑不动**，
   只能整机重启恢复。排查用 `py-spy dump --pid <pid>` 看栈，别急着 kill。

9. **三段共享显存，缺一步就会「功能未启用」。**
   DA3 的 `local` / `global` / `head` 三段必须共享同一份按 NPU 核分配的内部显存
   （C++ 的 `da3_init_internal_mem_share`，8 核合计约 1.6 GB）。
   少了 `user_mem_internal=1` + `set_internal_mem` 这步，`rknn3_model_init`
   直接 `MODEL_SETUP fail`，在 webui 里表现为
   「功能未启用: swcap_vision_depth」。这是 backend 内部保证的，
   但换模型 / 换导出时值得留意。

10. **`[time] total` 含模型加载，别当成推理耗时。**
    sample 打印的 `total ≈ 20 s` 里，**约 18.4 s 是加载**（开设备上下文 6.7 s +
    `model_init` 9.8 s + 共享显存 0.7 s），真正的推理只有 **1.34 s**。
    想知道推理有多快，看 `[time] global run` / `[time] head run`，或者直接看
    「性能实测」那张表 —— 拿 `total` 去评估实时流吞吐会算错一个数量级。

11. **实时流绝不能跨场景拼帧。**
    把 10 个槽位里的 7 个换成**别的场景**的画面，depth[0] 相对基线偏 **62%**，
    深度绝对尺度都塌了（均值 0.377 vs 0.997），模型 confidence 崩到 1.12
    （基线 2.92）。实时流的 V 帧必须来自**同一段连续视频**，
    不能拿不同摄像头/不同时段的帧凑数。

12. **view 的顺序不重要，别为此做重排序。**
    把 10 帧逆序喂进去，depth[0] 只偏 3.98%（单路是 4.14%），空间结构相关系数
    仍有 0.998 —— `global` 的跨 view 注意力是**集合式**的，不是时序建模。
    （顺带纠正一个容易犯的错：比较逆序结果时**必须比同一张图**，
    逆序后 slot0 装的是另一帧，拿它跟基线比会算出假的大偏差。）

13. **别用 `confidence` 判断 view 数给得好不好。**
    退化输入（10 张重复图）下 confidence **最高**（3.37），真实 10 帧反而更低
    （2.92）。它衡量的是局部证据一致性，不是全局正确性。
---

## qwen_image_21 —— Qwen-Image-2.1 文生图（512×512）

> 文生图**不是**文本生成：文本进，**一张 512×512 图像**出。
> 输出走 `ImageAttachment`（BGR uint8 ndarray），不是字符串。
> 这个 sample 的产物与板端 `test.sh` 出的图**逐位一致**（同一份噪声时）。

### 怎么跑

前置条件：模型包已在 `/home/rockchip/models/qwenimage21/`（见「坑」第 3 条）。

```bash
cd /home/rockchip/wdnwebui

# 默认：一只猫坐在木椅上，8 步，输出到 ./output.png
.venv/bin/python .venv/src/wdnai/examples/qwen_image_21.py \
    --prompt "a cute cat sitting on a wooden chair"

# 20 步（更清晰，耗时更长）
.venv/bin/python .venv/src/wdnai/examples/qwen_image_21.py \
    --prompt "一只戴眼镜的猫" --steps 20 --output /tmp/qi21_cn.png

# 固定初始噪声（可复现 / 可与 PyTorch 金标准对拍）
.venv/bin/python .venv/src/wdnai/examples/qwen_image_21.py \
    --prompt "a cute cat" --fixed-noise --seed 0

# 不写 Python 的等价入口
bash .venv/src/wdnai/samples/qwenimage21.sh "a cute cat" 8 --out /tmp/qi21
```

- **耗时**：**91~95 s/张**（8 步）。分段实测见下方「输出」表。
- **占 NPU**：是，占**两张卡**。同一时刻别再跑别的 sample。
- 输出目录里的中间产物（潜变量 / VAE 原始张量）默认跑完就删；
  要留就加 `--keep-artifacts`（产物目录会打印出来）。

### 输入

#### 命令行参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--prompt` | **必填** | 要画的提示词（中英文都行）。空串会抛 `ValueError` |
| `--steps` | `8` | flow-match 步数。8 步 ≈ 94 s；20 步 ≈ 125 s；40 步 ≈ 177 s。**纯运行时参数，模型不重建** |
| `--output` / `-o` | `output.png` | 输出 PNG 路径（父目录自动创建） |
| `--model-dir` | 自动解析 | 覆盖模型包目录；默认 `$WDNAI_MODELS_DIR/qwenimage21`，板上实际在 `/home/rockchip/models/qwenimage21` |
| `--seed` | `0` | 初始噪声种子（配合 `--fixed-noise`） |
| `--fixed-noise` | 关 | 复用模型包里的固定噪声（`reference_512/hidden_states.bin`），可复现 |
| `--keep-artifacts` | 关 | 保留中间产物（潜变量 / VAE 原始张量 / TE 输出） |
| `-v` / `--verbose` | 关 | 打开 SDK 的 DEBUG 日志 |

#### 素材目录约定

**不适用** —— 本 sample 的输入是纯文本提示词，没有图片 / 音频素材目录。

### 输出

| 产出 | 含义 |
|---|---|
| `<--output>` | 512×512 RGB PNG（8 bit/通道，无 alpha） |
| stdout `图像 shape=(512, 512, 3) dtype=uint8` | ndarray 形状与类型（BGR，H×W×C） |
| stdout `elapsed=94.7s` | 端到端墙钟耗时（含 TE 编码 + DiT 生成 + VAE 解码） |

分段耗时（板端实测，RK3588 + 2×RK1828，8 步）：

| 阶段 | 耗时 | 说明 |
|---|---|---|
| TE 编码 | ~37 s | **97% 是 `model_init`**（4 GB 权重灌进 NPU），推理本身只要 0.16 s |
| DiT 常驻 init | ~23 s | 一次性；之后**生成过程零 load / 零 model_init** |
| flow-match 8 步 | ~21 s | 每步 2.60 s（4 段串行，每段 ~580 ms） |
| VAE 4 段链 | ~10 s | 流式逐段释放 |

### 容易踩的坑

1. **`respond()` 不带 `generating=` 时图片会丢。**
   `respond` 的契约是给调用方一个**文本**结果，所以只返回 prompt 回显。
   要图用 `session.generate_image(prompt)`（最短）、`session.run_prompt()`
   （返回 `(text, attachments)`）或 `session.respond(prompt, generating=LMOutput)`
   （`.attachment` 自动填充）。

2. **给文生图传自定义 `generating=` schema 会抛 `ValidationError`。**
   扩散模型不产 JSON —— 这是预期行为，不是 bug。用 `LMOutput` 本身即可。

3. **模型包必须带 `runtime/qwen_image_rknn`（+ `runtime/lib/aarch64/`）。**
   缺了它 SDK 会回退去 `/home/rockchip/projects/deploy/bin/` 找；两处都没有就
   `FileNotFoundError`。可以用 `WDNAI_QWEN_IMAGE_BIN` / `WDNAI_QWEN_IMAGE_ROOT` 指。

4. **`reference_512/hidden_states.bin` 的文件名不能改。**
   运行时按这个名字读初始噪声；改名 ⇒ `找不到初始噪声` 直接退出。

5. **`fm_mu` 必须用 512 路线的 `0.5387096774`。**
   1024 路的 `0.6935483871` 用在这里会让出图完全不对（但不会报错）。

6. **TE 与 DiT 物理无法共卡 ⇒ 每次换提示词都要重跑 TE。**
   TE 权重本身占 486 MiB/核，与常驻的 DiT 抢不下同一张卡。
   所以「94 s 一张」是**架构天花板**，不是实现问题 ——
   步数只影响 flow-match 那 21 s。

7. **别在跑图的过程中 reset / rescan / 烧固件 / 重启板子。**
   这些是把 RK1820 弄坏的真凶（详见 `deploy/debug.md` §69）。
   本 sample 自己不碰卡的状态：proxy 已在跑就不重启，权重只读不写。
