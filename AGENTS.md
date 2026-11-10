# AGENTS.md

Quick reference for OpenCode sessions working on the `wdnai` SDK. The
project is a Python wrapper around librknn3 (`/usr/lib/librknn3_api.so`)
for **LLM / VLM / TTS** inference on Rockchip NPUs (RK1828); import
name `wdnai`, distribution name `wdnai`.

> Vendored in-tree as `.venv/src/wdnai/` (editable install from
> `git@github.com:rockchip-wardner/wdnai.git`). It is *not* on PyPI. See
> `wdnwebui/AGENTS.md` "One-time setup" for the install command.

## 仓库地址与同步策略（2026-11-10）

| 角色 | 地址 | 允许的操作 |
|---|---|---|
| **局域网主仓库** | http://10.10.10.10:8080/wardner | **pull + push** —— 唯一权威源，所有更新的第一站 |
| **GitHub** | https://github.com/rockchip-wardner | **只允许 push** —— 发布镜像，不从它 pull |
| **Gitee 国内镜像** | https://gitee.com/rockchip-wardner | **不允许任何 git 操作** —— 公开只读镜像，内容由上游自动同步 |

远端命名：`origin` = `git@10.10.10.10:wardner/wdnai.git`（局域网），
`github` = `git@github.com:rockchip-wardner/wdnai.git`（只 push）。

**提交顺序不可颠倒：先 `git push origin master`（局域网主仓库），再
`git push github master`（GitHub）。Gitee 一概不碰。** `master` 的跟踪分支指向
`origin`，裸 `git pull` / `git push` 即为对主仓库操作。拉取最新代码只走局域网
主仓库 —— 从 GitHub 拉的是「已发布出去的历史」，会让本地与主仓库分叉。

> **As of v0.4.x**, wdnai no longer depends on the `rknn3lite` Python
> wheel. The vendored `wdnai/_librknn.py` is a ctypes shim that
> mirrors the subset of `rknn3lite.api` the SDK actually exercises,
> driven directly against librknn3. See
> [`wdnai/_librknn.py`](#wdnaishim-library-binding) below.


## Repo map (one-line each)

- `README.md` — 面向**使用者**的说明：安装（GitHub editable）、文本/视觉/
  结构化输出/工具调用的快速上手、能力词表、`RknnSmi` 遥测、`ModelSpec`
  字段与 preset 速查、模型文件布局与环境变量、已知限制。
  `pyproject.toml` 的 `readme = "README.md"` 指向的就是它 —— **别删**，
  删了打包会退化成「声明了 markdown 长描述但正文为空」。
- `pyproject.toml` — package metadata (name=wdnai, version=0.4.0);
  deps: `numpy>=1.23`, `transformers>=4.40`, `pydantic>=1.10`,
  `opencv-python>=4.5`, `gguf>=0.6`. Optional: `[lite]`
  `opencv-python-headless`, `[dev]` `pytest`+`build`. **No runtime
  dependency on `rknn3lite` / `rknn3-toolkit-lite`**: the SDK
  talks to librknn3 directly via `wdnai._librknn`.
- `src/wdnai/__init__.py` — public API: re-exports `Generable`,
  `Callable`, `Attachment`, `LMOutput`, `ModelSpec`,
  `SystemLanguageModel`, `LLMSession`, plus pydantic's `BaseModel`
  / `Field`. Holds `__version__`. **Does not import `rknn3lite`** —
  the only RKNN3 access point is :mod:`wdnai._librknn`, which is
  lazy-imported inside each backend's `__init__` (see "Gotchas").
- `src/wdnai/attachment.py` — `Attachment(content, kind=…)`: unified
  carrier for `image` / `video` / `text` / `audio(bytes)`. BGR ndarray
  or list-of-BGR → image; string + `.mp4|.mov|.mkv|.avi|.webm|.flv`
  extension + file exists → video (samples 5 frames via cv2); bytes
  → audio (default; TTS output shape). Exposes `.to_payload()` /
  `.text` / `.images` / `.frames` / `.data` / `.mime_type`.
- `src/wdnai/_generable.py` — `@Generable` decorator: turns an
  annotated class into a pydantic `BaseModel`. Modeled after Apple
  Foundation Models' `@Generable`. Preserves subclass bases (e.g.
  `LMOutput`), copies user methods, sets `__module__` from caller
  frame so forward refs (`Optional`, `Literal`) resolve.
- `src/wdnai/_callable.py` — `@Callable` decorator: turns an annotated
  class into a pydantic `BaseModel` *and* registers it as a model-
  callable tool. The class itself is returned (not a wrapper); the
  session fills `instance()` from the model's tool-call JSON so
  `isinstance(result, AlarmCls)` discriminates between tools without
  any extra carrier type. **Passive**: the SDK never invokes the
  side effect — user calls `result()` themselves.
- `src/wdnai/lm_output.py` — `LMOutput(BaseModel)`: structured output
  base with a `attachment: List[Attachment]` slot. The SDK fills
  `attachment` automatically when the backend produces binary payloads
  (TTS audio bytes, decoded multimodal images, …).
- `src/wdnai/model.py` — `ModelSpec` dataclass (paths + sampling +
  vocab + image_size + target/core_mask) + `SystemLanguageModel`
  preset catalog: `qwen2_5_0_5b()`, `qwen3_1_7b()`, `qwen3_4b()`
  (pure-text, binds to `/userdata/models/Qwen3-4B/`, overridable via
  `WDNAI_QWEN3_4B_DIR`; raises `FileNotFoundError` listing missing
  files), `qwen2_5_vl_3b()` / `_7b()`, `qwen3_vl_2b()`,
  `qwen3_vl_4b_template()`, `qwen3_vl_4b()`,
  `qwen3_5_vl_4b()` (Qwen3.5-VL-4B; **vocab_size=248320**,
  **special_eos_id=248046** — different from Qwen3; the VLM-mode LLM
  is a regular LLM with MRoPE baked into the graph for image position
  encoding, so it also handles pure-text inference from a single
  RKNN; 384×384 + 768×768 vision variants auto-discovered in the
  model directory),   `qwen3_tts()`, `qwen3_vl_embedding()` (retrieval
  embedding, no token generation, binds to
  `/userdata/models/Qwen3-VL-Embedding-2B/` and raises
  `FileNotFoundError` listing missing files),
  `locate_anything()` (LocateAnything grounding model — image-in,
  text-out with 0–1000 normalised bbox coords; binds to
  `/userdata/models/LocateAnything/`, override via
  `WDNAI_LOCATE_ANYTHING_DIR`; raises `FileNotFoundError` listing
  missing files; capability key `swcap_vision_grounding`).
- `src/wdnai/_librknnsmi.py` — ctypes shim over `/usr/lib/librknnsmi.so`
  (the C ABI behind the `rknn-smi` CLI): device list, device/chip info,
  PCIe info, **temperature**, power/voltage/current, NPU util, memory,
  chip health, version, reset. `RknnSmi` is the only public entry.
  Two traps worth knowing before touching it:
  - **温度的单位两级不一样**：设备级 `get_device_temperature()` 返**毫度**
    （×1000，43000 → 43.0 °C），chip 级 `get_chip_temperature()` 返**整度**
    （43 → 43.0 °C）。当成同一套单位处理 → chip 级被多除 1000 → 43 °C 变成
    **0.043 °C**（静默错值，阈值判断全部失效）。
  - **设备级温度命令在 RK1828 EP 固件上没人应答**（cmd 12 走 PCIe，无应答
    → ret=-1，C 层打 `[nsmi_pcie.c:651] Receive error, command: 12`）。
    `get_device_temperature()` 因此**自动回退到 chip 级**，并记住该卡设备级
    不通（`_temp_device_unsupported`）→ 后续不再触发注定失败的命令。
    调用方（wdnwebui 的 system-monitor）只管调一次即可，不必自己实现
    「设备级 / chip 级」这层平台语义。
- `src/wdnai/grounding.py` — `@Generable` schemas for the
  LocateAnything backend: `BoundingBox` (single `<box>` region with
  `label` + 0–1000 normalised `x1`/`y1`/`x2`/`y2`) and
  `GroundingResult(LMOutput)` (the top-level payload, adds
  `boxes: List[BoundingBox]` + `text` (raw echoed grounding text)
  + `level: Literal["Normal", "Warning"]`). Parses the
  ``<ref>label</ref><box>(x1, y1, x2, y2)</box>`` spans the model
  emits into structured fields.
  The `qwen3_vl_4b()` preset binds to
  `/userdata/models/Qwen3-VL-4B/` (overridable via
  `WDNAI_QWEN3_VL_4B_DIR`) and raises `FileNotFoundError` listing
  missing files if the directory is incomplete. The
  `qwen3_5_vl_4b()` preset lives under
  `/userdata/models/Qwen3.5-4B/` (override via
  `WDNAI_QWEN3_5_4B_VL_DIR`); MRoPE baked into
  `Qwen3.5-4B-llm.1828.rknn` is what distinguishes it from a pure
  text LLM — no separate text-only Qwen3.5-4B RKNN is shipped (one
  RKNN powers both text-chat and video-describe capabilities).
- `src/wdnai/session.py` — **传输无关的会话接口 + 两个实现**：
  - `LLMSession(ABC)` — 接口。拥有**所有**必须跨进程一致的逻辑：
    prompt 组装（`_compose_prompt`）、tool-call 解析、schema 校验
    （`_parse_output`）。签名：
    `async def respond(prompt, attachment, generating, tools, debug,
    perf, on_delta, history, **overrides)`。唯一交给子类的缝是
    `abstract async _dispatch(prompt, attachments, *, on_delta) ->
    (text, attachments_out)`。
  - `LLMLocalSession` — 本机 NPU 实现，`__init__` 建 backend，
    `_dispatch` 丢线程池。**`LLMSession(model=…)` 仍是可调用的**（走
    `__new__` 垫片转成 `LLMLocalSession` + `DeprecationWarning`，
    保留一个版本周期），`isinstance(x, LLMSession)` 对两个实现都为真。
  - `wdndds.LLMRPCSession(LLMSession)` — DDS RPC 实现（wdndds 包内，
    反向依赖本包）。
  - 由此 `generating=` / `tools=` **完全不过线**：它们只在调用方进程被
    消费，worker 只见到一个 prompt 字符串 + 附件列表。
  Responsibilities:
  attachment channel-splitting (images / video frames / text prefixes
  / passthrough), ChatML prompt assembly + JSON-Schema hint injection,
  tool-call routing (`@Callable` branch vs `generating=` branch
  decided by `isinstance`), output classification (TTS shortcut via
  `bool(attachments_out)` vs raw-JSON parse), history bookkeeping,
  `reset()` (clears history + `rknn_llm.clear_kvcache()`),
  `majority_vote(k)` (k sequential calls + `Counter.most_common`).
- `src/wdnai/utils.py` — image preprocessing (`bgr_to_rgb`,
  `letterbox_resize` — **the only image-resize entry point**; always
  preserves aspect ratio by scaling longest side and centring the
  shorter side with grey `(114,114,114)` padding; NEVER stretches.
  Interpolation direction-aware: downsample→`INTER_AREA`,
  upsample→`INTER_CUBIC`, equal→`INTER_LINEAR`),
  `qwen25vl_prune` / `qwen3vl_prune_single` (patch-tokeniser for
  pruned vision RKNNs — matches the example scripts' preprocessing
  exactly), `extract_json` (tolerates ```json``` fences and bare
  braces).
  **Removed**: the old plain `resize()` (stretch-to-square) — VLM crop
  ROIs (e.g. a 250×60 license-plate strip) would get squashed into a
  square and break OCR. Use `letterbox_resize`; if you really need the
  legacy stretch, call `cv2.resize(img, (size, size), ...)` yourself.
- `src/wdnai/gguf_tokenizer.py` — `load_tokenizer_from_gguf(path)`:
  reads `tokenizer.ggml.{tokens,merges,model,pre,eos_token_id,…}` via
  `gguf.GGUFReader`, writes vocab+merges to a temp dir, builds a
  GPT-2-style BPE via `tokenizers`, wraps as a
  `PreTrainedTokenizerFast`. **Multimodal placeholders**
  (`<|vision_start|>`, `<|image_pad|>`, `<|vision_pad|>`,
  `<|video_pad|>`) are registered as `additional_special_tokens`
  — without this, the ByteLevel pre-tokenizer would split each into
  ~7 regular tokens and `rknn3` session_run would never find the
  expected IDs.
- `src/wdnai/backends/__init__.py` — `build_backend(spec)` factory:
  `{"llm": TextBackend, "vlm": VLMBackend, "tts": TTSBackend,
  "embedding": EmbeddingBackend, "grounding": LocateAnythingBackend}`
  registry (asr is loaded on demand).
- `src/wdnai/backends/embedding.py` — `EmbeddingBackend` (kind=
  "embedding"): loads llm/vision `.1828.rknn` + `.weight` + embed.bin
  + tokenizer under `WDNAI_MODELS_DIR/Qwen3-VL-Embedding-2B`, feeds
  vision at 576x576 via `letterbox_resize` (so PyAV-decoded frames
  with arbitrary aspect ratios get grey-padded squares, never
  stretched), exposes `embed_text(text)` and
  `embed_video_chunk(frames, text="")` returning L2-normalized
  2048-dim vectors (the LLM runs through the plain RKNN3Lite path,
  no token generation).
- `src/wdnai/backends/locate_anything.py` — `LocateAnythingBackend`
  (kind="grounding"): LocateAnything grounding model — pairs a
  448×448 vision RKNN encoder with a text LLM RKNN running in
  `llm_mode` (mirrors `examples/LocateAnything/cpp/main.cc`).
  Pipeline: BGR→RGB→letterbox→NHWC uint8 → vision RKNN → fp16 patch
  tokens packed into a single `RKNN3Image` → LLM `session_run` with
  `enable_thinking=False` (grounding is non-thinking) → raw text
  passed through `wdnai.utils._parse_locate_output` to split
  ``<ref>label</ref><box>(x1,y1,x2,y2)</box>`` /
  ``<ref>label</ref><box>(x,y)</box>`` spans into structured
  `boxes` / `points` lists. Exposes `run_image(prompt, image,
  *, max_new_tokens=1024)` returning `{"text", "boxes", "points"}`.
  Required paths: `rknn_path` + `weight_path` + `rknn_vision_path`
  + `weight_vision_path` + `tokenizer_path` + `embed_path`; vision
  `image_size` must be a multiple of 14.
- `src/wdnai/backends/depth_anything.py` — `DepthAnythingBackend`
  (kind="depth"): Depth Anything V3 单目深度估计 — **三段** RKNN
  （`local` 逐图 token → `global` 跨 view 融合 → `head` 出深度/置信度），
  镜像 `rknn3-model-zoo/examples/depth_anything_v3/cpp/main.cpp`。
  三段**共享**一份按 NPU 核分配的内部显存（C++ 的
  `da3_init_internal_mem_share`，约 1.6 GB）：少了 `user_mem_internal=1`
  + `set_internal_mem` 这一步，`rknn3_model_init` 直接
  `MODEL_SETUP fail`。view 数是**烤进导出**的（`global` 输入
  `[V,L,1,D]`），不是运行期参数 —— 送图张数必须与之一致。
  **当前部署的导出是 1-view（280×280）**：`global` 输入 `[1,401,1,768]`、
  `head` 输出 `[1,1,280,280]` —— 一张图即可出深度图 + 置信度。
  **换 V 不需要改 SDK** —— `self.views` 是从 `global` 的输入张量属性上读
  出来的，重新导出 `--views N`（model-zoo 已验证 1-10）即可，`estimate()`
  的张数校验与报错文案自动跟着变。
  部署侧文件命名跟 `LocateAnything-3B` 一致：
  `<role>_<Model>.<WxH>.<N>view.1828.rknn` + 同名 `.weight`（**去掉 `.1828`
  段**）—— `depth_weight_path()` 按 `.1828.rknn` → `.weight` 派生，
  旧的「`.rknn` → `.weight`」只作兼容兜底。
  实测（RK3588；用重复帧填充槽位模拟有效 view 数递减，始终比较同一张图
  的深度）：10→5 偏 **1.58%**、10→3 偏 **2.59%**、10→1 偏 **4.14%**，
  收益递减但未饱和（5 帧吃到 62% 的收益）；**顺序不重要**（逆序 3.98%，
  空间相关系数 0.998 —— `global` 是集合式跨 view 注意力，不是时序建模）；
  **跨场景拼帧是灾难**（偏 62%，深度尺度塌掉，confidence 崩到 1/3）。
  实时流的 V 帧必须来自同一段连续视频。`confidence` **不能**用来判断
  view 数好坏（退化输入下它反而最高）。
  **多相机几何当前走不通**：多相机融合需要相机位姿，而位姿在官方模型里靠
  **ray map**（每像素 6 通道）隐式表示 —— model-zoo 的
  `da3_export/head.py` 只返回 `depth` + `depth_conf`，**没有 ray 导出路径**；
  且论文 Tab.2 显示 DA3-Base 的位姿 Auc3 只有 DA3-Giant 的 1/4
  （HiRoom 19.0 vs 80.3），**位姿比深度更吃模型规模**。
  像素侧全部按 C++ 的**实际行为**对齐，三处不对齐就会让深度图差一两个
  fp16 ULP：① resize 被 GCC 融合成 FMA（见 `preprocess_image_rgb`）；
  ② fp16→fp32 用 C++ 的 `fp16_to_fp32`（**不处理 fp16 非规格化数**，
  见 `_fp16_to_fp32`）；③ 统计均值用 float64 累加（C++ 是 `double`）。
  回归防线：`python -m scripts.verify_da3_parity`（在 wdnwebui 仓）。
- `src/wdnai/backends/_common.py` — shared RKNN plumbing:
  `StreamSink` (collects tokens / text / perf from streaming
  callback), `build_llm_callback` (wraps the shim's
  `wdnai._librknn.build_llm_callback`), `load_tokenizer` (`.gguf`
  → `gguf_tokenizer.py`; otherwise `AutoTokenizer.from_pretrained`),
  `load_embed_table` (mmap fp16 → reshape `(vocab_size, dim)`),
  ChatML template (`system\nYou are a helpful assistant.\n` /
  `user\n` / `\nassistant\n`), `build_llm_args` (merges
  `spec.options` into the `llm_args[0]` dict), `printf_perf`
  (Prefill / Generate / vision / total latency table).
- `src/wdnai/_librknn.py` — **vendored ctypes shim for librknn3**.
  Public surface: `RKNN3Lite` (mirrors the slice of
  `rknn3lite.api.RKNN3Lite` wdnai uses), `RKNN3Image` /
  `RKNN3Audio` / `RKNN3AuxTensorWrapper` (multimodal input structs
  backends construct and pass to `session_run(inputs=...)`),
  `Float16` (ctype alias for `uint16` storage used in
  `ctypes.POINTER(Float16)` typing), `RKNN3TensorType` (IntEnum
  used by the embedding backend), and `build_llm_callback` (the
  builder that hides the C `RKLLMCallback` struct + CFUNCTYPE
  trampolines — backends pass plain Python callables, the shim
  pins the bound ``self`` via `ctypes.py_object` and rebuilds the
  ctypes pointer dance internally). All other ctypes Structures
  (`RKNN3TensorAttr`, `RKNN3CoreMemSize`, `RKNN3QueryCmd`,
  `RKNN3LLMParam`, `RKNN3LLMInput`, …) and the C function
  prototypes for `rknn3_init` / `rknn3_load_model_from_path` /
  `rknn3_session_init` / `rknn3_session_run` / `rknn3_run` /
  `rknn3_set_internal_mem` / … live in the same module but stay
  private (names start with `_`). Library is loaded lazily
  (`CDLL` only runs on the first `RKNN3Lite()` instantiation) so
  `import wdnai` doesn't open the SO and init the NPU driver.
  Style follows `rknn3_model_zoo/examples/Qwen3_5_VL/cpp/main.cc`
  + `qwen3_5_vl.cc` (one librknn3 context per RKNN model, plain
  `rknn3_run` for vision / embedding, `rknn3_session_init` +
  `rknn3_session_run` for LLM mode).
- `src/wdnai/backends/text.py` — `TextBackend`: loads `rknn` +
  `weight`, init_runtime with the LLMM-mode callback, `apply_chat_template`,
  exposes `run(prompt, **overrides)`. Overrides `max_new_tokens` /
  `enable_thinking` / `keep_history` are passed straight to
  `rknn.session_run` (the shim's session_run takes the same kwarg
  names as rknn3lite did).
- `src/wdnai/backends/vlm.py` — `VLMBackend`: one `RKNN3Lite` LLM
  handle + **N** vision handles (one per discovered native resolution
  variant). At init, `_load_vision_variants` calls
  `_discover_vision_paths` to scan `spec.rknn_vision_path`'s directory
  for every `<prefix>.<WxH>.1828.rknn` sibling. Qwen3-VL-4B ships
  with both 384x384 (required) and 768x768 (optional) pairs; legacy
  installations with a single `vision_<name>.1828.rknn` (no `<WxH>`
  suffix) fall through to the single-file path. Each variant gets its
  own `RKNN3Lite` context sharing the LLM's `device_id`. Variant-
  dispatched preprocessing (`VARIANT_25VL="qwen2.5-vl"` vs
  `VARIANT_3VL="qwen3-vl"`):
  - `use_prune=True` → `qwen25vl_prune(...)` /
    `qwen3vl_prune_single(...)` (2D feature, matches the example
    scripts).
  - `use_prune=False` → NHWC `(1, H, W, 3)`; Qwen3-VL full vision
    model expects a 2-image stack (the "two-frame input" video mode),
    so a single image is duplicated along axis 0.
  **Per-call variant selection** (`_pick_variant`): the variant whose
  `image_size` is closest to the first image's longest side wins; ties
  broken by smaller size. All images in one call share one variant
  (the LLM consumes a single `RKNN3Image` with concatenated tokens +
  a single `image_width` / `image_height`); images that would prefer a
  different variant are force-fit to the chosen one with a warning.
  **Image-to-variant resize** (`_preprocess_to_variant`) uses
  `letterbox_resize(img, variant.image_size)` — never plain `cv2.resize`
  — so the source aspect ratio is preserved and the shorter side is
  filled with grey `(114,114,114)`. Crop ROIs (typically very wide
  strips like a license-plate bbox) keep their character aspect ratio
  through the second-stage OCR call. Outputs the vision feature →
  `RKNN3Image` (vision tags set to `<|vision_start|>` / `<|vision_end|>`
  / `<|image_pad|>`). Qwen3-VL
  additionally concatenates 3 deepstack aux tensors (output indices
  1..3) into `RKNN3AuxTensorWrapper`s with the chosen variant's
  vision output's `aligned_size`.
- `src/wdnai/backends/vlm.py` — `VLMBackend`: one `RKNN3Lite` LLM
  handle + **N** vision handles (one per discovered native resolution
  variant). At init, `_load_vision_variants` calls
  `_discover_vision_paths` to scan `spec.rknn_vision_path`'s directory
  for every `<prefix>.<WxH>.1828.rknn` sibling. Qwen3-VL-4B ships
  with both 384x384 (required) and 768x768 (optional) pairs; legacy
  installations with a single `vision_<name>.1828.rknn` (no `<WxH>`
  suffix) fall through to the single-file path. Each variant gets its
  own RKNN3Lite context sharing the LLM's `device_id`. Variant-
  dispatched preprocessing (`VARIANT_25VL="qwen2.5-vl"` vs
  `VARIANT_3VL="qwen3-vl"`):
  - `use_prune=True` → `qwen25vl_prune(...)` /
    `qwen3vl_prune_single(...)` (2D feature, matches the example
    scripts).
  - `use_prune=False` → NHWC `(1, H, W, 3)`; Qwen3-VL full vision
    model expects a 2-image stack (the "two-frame input" video mode),
    so a single image is duplicated along axis 0.
  **Per-call variant selection** (`_pick_variant`): the variant whose
  `image_size` is closest to the first image's longest side wins; ties
  broken by smaller size. All images in one call share one variant
  (the LLM consumes a single `RKNN3Image` with concatenated tokens +
  a single `image_width` / `image_height`); images that would prefer a
  different variant are force-fit to the chosen one with a warning.
  **Image-to-variant resize** (`_preprocess_to_variant`) uses
  `letterbox_resize(img, variant.image_size)` — never plain `cv2.resize`
  — so the source aspect ratio is preserved and the shorter side is
  filled with grey `(114,114,114)`. Crop ROIs (typically very wide
  strips like a license-plate bbox) keep their character aspect ratio
  through the second-stage OCR call. Outputs the vision feature →
  `RKNN3Image` (vision tags set to `<|vision_start|>` / `<|vision_end|>`
  / `<|image_pad|>`). Qwen3-VL
  additionally concatenates 3 deepstack aux tensors (output indices
  1..3) into `RKNN3AuxTensorWrapper`s with the chosen variant's
  vision output's `aligned_size`.
- `src/wdnai/backends/vlm_multicard.py` — Qwen3.5-VL-9B 两段式双卡后端
  （`kind="vlm_multicard"`）：seg0/seg1 各一个 LLM session、各绑一张
  RK182x，vision 与 seg0 同卡并共享 internal mem；逐 token pipeline 在
  `_qwen35_vl_9b_vendor.py`（该文件同时保留可直接运行的 CLI demo）。
  **全部走 `_librknn` ctypes shim，不依赖 `rknn3lite`**。prompt 由
  Python 侧 Jinja2 渲染（GGUF `tokenizer.chat_template` +
  `_messages_from_prompt`）：真机 A/B 显示 C 侧 `set_chat_template` 在
  分段 stage0 上单轮可用，但多轮 bare transcript 会退化成 thinking
  dump（去掉 transcript 的 system 段也不行），所以 Jinja2 这层必须保留。
- `src/wdnai/backends/tts.py` — `TTSBackend` + vendored
  `Qwen3TTSDriver` (1893 lines, originally in
  `rknn3-toolkit-lite/examples/Qwen3-TTS/qwen3_tts_demo.py` and
  imported via `sys.path` injection; now in-tree so
  `import wdnai.backends.tts` is self-contained). Pipeline:
  `text → text_projection (1 core) → talker (8 cores, AR) →
  code_predictor (8 cores, AR, offline) → speech_decoder
  (sliding window 25/50) → 24 kHz mono 32-bit IEEE float WAV`.
  `AudioPlayer` (sounddevice → aplay → no-op fallback). VoiceClone
  reference embeddings (6 真人参考音色) are zlib+base64-inlined
  in `_BUNDLED_SPEAKER_EMBEDDINGS_B64` so no external
  `qwen3tts_speaker_embed.h` is required. `run_text()` returns
  `(prompt_str, [Attachment])`; the SDK unpacks the tuple and
  assigns the audio bytes to `LMOutput.attachment`. The four RKNN
  contexts each go through `rknn3_init` + `load_rknn` +
  `model_init`; only `talker` and `code_predictor` get
  `rknn3_session_init` (the LLM-mode pair). The four callbacks
  (`_talker_result_callback`, `_talker_output_callback`,
  `_talker_embed_callback`, `_cp_*` variants) are plain Python
  callables built via `wdnai._librknn.build_llm_callback`; the
  shim pins the bound `self` as userdata and dispatches the C-side
  trampolines. `_LLM_RUN_NORMAL` / `_LLM_RUN_FINISH` etc. are
  inlined as integer constants — we don't expose the shim's
  IntEnum to keep this module independent.
- `src/wdnai/backends/speaker_encode.py` — `SpeakerEncoder`: RKNN 版
  ECAPA-TDNN，把一段参考音频变成 2048 维 x-vector（用户音色的**现场提取**，
  参考音色是同一份向量的离线版本）。纯 numpy 前处理（mel 滤波器组是
  随模型发布的 `speaker_encoder_mel.npy`），固定输入 468 帧 = 5 秒
  （长音频截前 5 秒，短音频尾部补零）。走 `wdnai._librknn` shim，
  `RKNN3Lite` 只在 `open_context()` 里 import —— native driver 绝不能在
  模块 import 阶段打开。
- `src/wdnai/_cshim/` — ⭐ **文生图的运行时（C 实现，源码在本仓库）**：
  `libqi_qwen.so`（`make` 编出来；`setup.py` 在 `build_py` 阶段自动编）。
  里面是 `qi/{rknn3_runner,resident_chain,vae_chain,te_encoder,np_noise,api}` ——
  **模型分段的规则、IO 落位、layout 转换、flow-match 数学全在 C 侧**
  ⇒ 数值行为与基线**逐字节一致**（PNG md5 `4766cd27…` / `7604d7b4…`）。
  **只链系统自带的 `librknn3_api.so`**（门面会 dlopen 系统自带的 rkcp）
  ⇒ 模型包里**不需要任何 .so 或可执行文件**。
  C ABI（Python 只碰这几个）：`qi_qwen_open` / `qi_qwen_te_encode` /
  `qi_qwen_generate_batch` / `qi_qwen_fixed_noise` / `qi_qwen_free` / `qi_qwen_close`
  \+ **图生图**：`qi_qwen_img2img_batch`（见下面「图生图」那条）。
- ⭐⭐⭐ **图生图（img2img）＝ VAE 编码器 + 从中间步起步**（2026-11-06）：
  - **模型包里必须多一个文件**：`vae_qwenimage21-512.encoder.3588.rknn`
    （205 MB，跑在本机 RK3588 的 NPU 上，**与解码器同设备但可同时常驻**：
    解码器 2311 MiB + 编码器 ≈0.9 GiB < 4 GiB 池）。
    ⚠️ **解码器转好了 ≠ 图生图能用** —— 解码器 = latents→图，
    编码器 = 图→latents，是**两个不同的模型文件**。
    编码器是**可选件**：没有它 ⇒ 图生图**明确报错**，**不偷偷退化成文生图**。
  - 走的是**经典 SDEdit 式 img2img**，**不是**官方那种「DiT 加图像条件输入」
    ⇒ **DiT 一个字节都不用改**：
    `z₀ = Encoder(图)` → `x = (1−σ[k₀])·z₀ + σ[k₀]·ε` →
    DiT 从 `k₀` 跑到末步 → VAE 解码。
    `k₀ = steps − round(steps·strength)`，**与 diffusers img2img 同一口径**；
    `strength=1 ⇒ k₀=0 ⇒ 与文生图逐字节一致`。
  - C 侧：`qi/api.cpp::encode_source_image()`（RGB888→RGBA→fp32[-1,1]→
    `set_input("pixel_values")`→`run`→取 `latents`→归一化→**转 token 主序 fp16**）
    \+ `qi_qwen_img2img_batch()`；
    `qi/resident_chain.cpp::run_flowmatch()` 新增 `start_step` / `z0_path` 两个参数
    （`start_step=0` ⇒ 完全不做加噪 ⇒ 与文生图**逐字节一致**）。
  - ⚠️ **VAE 是 RGBA 4 通道原生**（不是 RGB）⇒ 编码时 alpha 填 1.0；
    输入尺寸固定 512×512（**缩放由调用方做**，板端走 wdnav 的 RGA）。
  - ⚠️ `rknn2_to_logical_nchw` 原来只认 NCHW/NHWC/NC1HWC2，而编码器输出
    的 fmt=3（`RKNN_TENSOR_UNDEFINED`）⇒ 已加 **UNDEFINED → 零拷贝**分支
    （板端实测 UNDEFINED 输出就是逻辑 NCDHW）。
  - ⚠️ 编码器**输出**是通道主序（C=64, T=1024），DiT 的 `hidden_states` 是
    **token 主序**（T=1024, C=64）⇒ 两者要互转（解码器侧的
    `latent_to_nchw_f32` 是同口径参考）。
  - 归一化常数在 `qi/vae_stats.h`（`qi::kVaeLatentsMean/Std`）：
    解码 `v·std + mean`，编码 `(raw − mean)/std`。
- `src/wdnai/_cshim/qi/np_noise.{h,cpp}` + `np_zig_tables.h` — ⭐ **固定初始噪声
  现场生成**（⇒ 模型包里**不用再带** `reference_512/hidden_states.bin`）。
  里面是 **numpy 兼容的 PCG64（XSL-RR 128/64）+ ziggurat**：
  历史基线 = `default_rng(0).standard_normal(64**3).reshape(64,64,64)[::2,::2,:]`。
  ⚠️ **不能用 numpy 现算**（官方只保证「同一 build 内」的流一致，跨版本可能换算法）⇒ 整条链搬进 C。
  ⚠️ `np_zig_tables.h` 是**自动生成**的（`deploy/scripts/gen_np_zig_tables.py`，
  数据源 numpy 2.5.3 的 `ziggurat_constants.h`）⇒ **别手改**。
  ⚠️ `np_noise.cpp` 里 `#pragma GCC optimize("fp-contract=off")` **必须保留**
  （慢路径判定的 mul+add 被收缩成 FMA 会差 1 ULP ⇒ 临界样本判定翻转）；
  ⚠️ **但全局 CXXFLAGS 不能关** —— `resident_chain.cpp` 的 FMA 收缩是基线行为的一部分。
  ⚠️ `fp64_to_fp16_rne` 的次正规位移量是 **`43 - exp`**（fp32 版才是 `14 - exp`）⇒ 别抄错。
- `src/wdnai/_qwen_image_c.py` — cshim 的 ctypes 绑定 +
  **纯 Python 的 gguf 分词器**（分词留在 Python ⇒ 不需要 Rockchip 的闭源
  `libtokenizer.a`）。⚠️ t2i 模板必须显式带 `<|im_start|>` / `<|im_end|>`
  （Rockchip 的闭源 tokenizer 会自动加，Python 的不会）⇒ 源码里用
  `"<" + "|im_start|>"` 拼，**别写字面量**（会被工具链吞掉）。
  另有 `fixed_noise()`：返回固定初始噪声（fp16），md5 恒为
  `b808cf7ecd34338323b783704c516128`（单测守着）。
  另有 `generate_img2img(prompts, source_rgb, steps, mu, shift_terminal,
  noise, seed, strength)`：图生图的 ctypes 入口（调 `qi_qwen_img2img_batch`），
  `source_rgb` 必须是 **512×512 RGB888**（缩放由调用方用 RGA 做），
  `strength` = 保留原图的程度（0~1）。
- `src/wdnai/backends/image.py` — `ImageBackend`（kind=`image`，文生图
  Qwen-Image-2.1 / RKNN3 rk1820，512×512）。**不再起子进程** ⇒ 改成
  **进程内调 cshim**（`_qwen_image_c.QwenImageShim`）。
  流水线三段：TE（text encoder，**常驻**，与 DiT 的头共卡 0）→
  DiT（4 × 8-block 段，**切头单卡**，共享 internal，常驻 ⇒ 生成过程
  零 load / 零 model_init）→ VAE（4 段链，**流式**，且必须在
  **DiT 释放之后**才加载 —— 两者物理上不能共卡）。
  两条硬约束写在 `backends/image.py` 的注释里：`QI_SPREAD_IO=0`
  （输出 IO 绝不能分散）、`QI_SPREAD_IO_IN=1`（只分散输入 IO，
  否则 VAE 第 3 段 `create_mem` 失败）。
  `run_text()` 返回 `(prompt_str, [ImageAttachment])`；图片是 **BGR
  uint8**（cshim 直接给 RGB，SDK 转 BGR），PNG 由 SDK 用纯标准库写。
  模型包默认 `/home/rockchip/models/Qwen-Image-2.1`（`WDNAI_QWEN_IMAGE_21_DIR`
  / `QI_MODEL_DIR` 可覆盖），**包里只剩模型**（无 .so、无可执行文件）。
  ⭐⭐ **2026-11-06：`run_images()` 从 `NotImplementedError` 变成真实现**
  （图生图）：BGR→RGB、**用 wdnav 的 RGA 缩放到 512×512**（不许软件 resize）、
  调 `generate_img2img`、写 PNG、复制到 `output_dir`，
  返回 `(prompt, [ImageAttachment(bgr)])`；`_options()` 里加 `"strength": 0.75`。
  ⚠️ **附件形态**：session 的 `_dispatch_sync` 会把附件**展平成 ndarray 列表**
  （`images.extend(att.images)`）⇒ `run_images` 收到的是**裸 ndarray**，
  不是 `ImageAttachment` ⇒ 取图要 `isinstance(item, np.ndarray)` 分支
  （第一版按 `.images` 取，第一版直接报「图生图需要一张输入图」）。
  ⭐ **第三十七轮新增 `warmup()` 钩子**：cshim 会话是**懒加载**的
  （构造 backend 不碰 NPU），于是「第一次推理」= 90 s init + 34 s 生成
  = 124 s，**超过 webui 的 120 s RPC 推理超时** ⇒ 第一次出图必然超时。
  `LLMLocalSession` 构造完 backend 后会调可选的 `warmup()` ⇒ init 挪到
  worker 的 load 阶段，推理阶段只剩 34 s。**给其它重后端加预热时也走
  同一个钩子**（`getattr(backend, "warmup", None)`，没实现就是 no-op）。
  ⭐⭐⭐ **第三十七轮：cshim 加了进度回调**（`qi_progress_ctx`）——
  出图是重操作（首次 load ~90 s + 每张 8 步 ~34 s），上层（Web UI 卡片）
  要画进度条就必须知道「现在到哪一步、这一步多久、还剩多久」。
  - C 侧：`qi/progress.h` 定义 `qi_progress_cb` + `qi_progress_ctx`；
    `qi_qwen_open_ex()`（比 `qi_qwen_open` 多两个参数）能在**加载模型之前**
    装上回调 ⇒ 那 90 s 的 init 进度也看得见（事后 `qi_qwen_set_progress`
    就晚了 —— init 发生在 `qi_qwen_open` 内部）。
  - ⭐ **全局只有一个时钟**：`qi_progress_ctx` 由会话持有，各层
    （api.cpp / resident_chain / vae_chain）**共享指针** ⇒ elapsed 口径
    全局一致。第一版让每层各自计时，结果「release」那条报出 21 s
    （那是「距上一次**本层**发射」的间隔，跨层对不上）。
  - ⭐ **换阶段要重置计时基准**（`qi_progress_stage()`）：否则新请求的第
    一条进度会把「worker 空闲了多久」当成「这一步花了多久」
    （实测：TE 编码显示 44.6 s，其实是 worker 空等了 44 s）。
  - Python 侧用 `ctypes.CDLL`（**不是** PyDLL）⇒ 调用期间 GIL 释放 ⇒
    C 里回调 Python 安全（PyDLL 会自锁）。
  - backend 把回调转成结构化消息
    `qi|stage=flow|step=3|total=8|elapsed_ms=2601.5`，经 `sink.emit()`
    ⇒ `on_delta` ⇒ DDS delta 通道 ⇒ 上层。
- `src/wdnai/models/README.md` — model file layout
  (`/userdata/models/<name>/`), `.1828.rknn` naming convention, the
  Qwen3-VL-4B / Qwen3-TTS file tables. The directory itself ships
  empty in-tree; weights are user-placed (or installed via the
  rknn3_model_zoo flow).
- `tests/test_mock_e2e.py` — Monkey-patches
  `wdnai._librknn.RKNN3Lite` + transformers + the result-callback +
  `build_llm_callback` so the full SDK runs without hardware. Covers
  Qwen3 / Qwen2.5 text, Qwen3-VL / Qwen2.5-VL image, override
  pass-through, and the full `respond()` path (JSON hint injection
  + pydantic parse).
- `tests/test_multicard_pipeline.py` — Qwen3.5-VL-9B 两段 pipeline 的
  离线回归（fake `RKNN3Lite`，不占硬件）：tokenizer 适配、embed
  callback 写入/计数、output/result callback 语义、`run_pipeline_once`
  的 token→embed→token 串联、shared internal mem 的建立/释放。
- `tests/test_capability.py` — `SoftwareCapability` 词表：成员与裸字符串
  相等且 hash 一致、`json.dumps` 出裸字符串、`__str__` 还原裸值、
  严格校验与老配置迁移语义分离。
- `tests/test_rknnsmi_temperature.py` — `RknnSmi` 温度读取的**单位**与
  **回退**：chip 级返整度（raw 43 → 43.0 °C，不是 0.043）；设备级失败
  （ret=-1）→ 回退 chip 级并记住该卡设备级不通（第二次不再触发注定
  失败的 PCIe 命令）；两级都失败仍抛 `RknnSmiCallError` 且**不**记缓存。
  用假 lib 驱动，不碰硬件。
- `examples/` — runnable demos, no `test_` prefix. End-to-end
  consumers; `detect_image.py` (10-round structured-output loop),
  `detect_video.py` (`wdnav.VideoCapture` + per-N-frames VLM), the
  `detect_video_carplate.py` plate detector (single / two-stage /
  `--majority-vote K` / background `asyncio.Task` for non-blocking
  display), `function_call.py` (multi-tool dispatch with
  `isinstance`), `text_to_speech.py` (TTS CLI with `--ref-speaker
  news_female` / `--volume` / `--no-play`), `qwen_image_21.py`
  (Qwen-Image-2.1 文生图 CLI —— `--prompt` / `--steps` / `--output`，
  走 `session.generate_image()`). External dataset dirs
  `光厂_*` hold plate frames + `.txt` ground truth.
  **`examples/README.md` 是 sample 的对外文档**：新增 sample 必须在
  该 README 里加一节，固定 4 个小节 —— **怎么跑 / 输入 / 输出 /
  容易踩的坑**，外加「素材目录约定」（素材放 `examples/<sample 同名>/`，
  清单用裸文件名相对自身目录解析）；性能敏感 / any-view 类模型再加
  **「性能实测」**和**「view 数与实时流」**两节。
  已按此记录的有 `depth_anything_v3` 与 `qwen_image_21` 两节，
  其余 sample 逐个补齐。
- `samples/` — 命令行入口（与 `examples/` 的 `.py` demo 平行）。
  ⭐ 2026-11-02 第三十五轮：**空目录** —— `qwenimage21.sh` 已删除
  （与 `examples/qwen_image_21.py` 功能重复，两套参数/噪声语义/PNG 转换迟早走偏）。
  ⇒ **文生图只有 `examples/qwen_image_21.py` 一个入口**，
  参数与旧 sh 对齐（`"<提示词>" [步数] [--out 目录] [--fixed|--fresh]`，
  `|` 分隔多提示词 ⇒ 批量）。


## 接口设计

### 设计原则

- **VLM 兼容 cv2 但不依赖 cv2**：`bgr_to_rgb` / `letterbox_resize` 是惰性
  import cv2，SDK 在非 RK 开发板上依然能 import；但调用方一旦走图像路径
  就必须装 `opencv-python`。`wdnav` 的 `VideoCapture.read()` 已经把 BGR
  ndarray 出来了，调用方直接 `Attachment(frame)` 即可。**图像送入 VLM
  之前不需要调用方自己 resize** —— SDK 后端会自动按 vision 变体
  `image_size` 做 letterbox（最长边等比缩放 + 短边居中填灰
  `(114,114,114)`），永远不拉伸。Crop ROI、用户上传图、视频任意比例帧
  都走同一条路径。
- **不做软解 / 不做软件兜底**：解码（视频帧）/ 视觉预处理（图像）/ TTS
  codec 解码全部走 NPU / 加速库。cv2 只用来 BGR↔RGB、letterbox、frame
  sampling（视频 attachment 抽样 5 帧）、`cv2.VideoCapture` 给 `Attachment`
  用作兜底启发式；不要拿 cv2 做 JPEG/H.264 软编解码。
- **RKNN 导入必须懒加载**：`from rknn3lite.api import RKNN3Lite` 仅出现在
  backend 模块的 `__init__` / 调用栈内部；`src/wdnai/__init__.py` 顶部永远
  不要 import `rknn3lite`，否则一台非 RK 机器上 `import wdnai` 会因为加载
  RKNN3 native libs 而 segfault（与 `wdnwebui` 里 `WdnaiDetector._load_wdnai`
  的约束同源）。
- **rknn3lite 是同步 C API，`LLMLocalSession` 已把阻塞调用丢进线程池**：
  backend 的 `run` / `run_text` / `run_images` 仍是同步阻塞调用
  （≈ 1s @ 4B），但 `LLMLocalSession._dispatch` 通过
  `loop.run_in_executor(self._executor, …)` 把它们提交到
  `ThreadPoolExecutor(max_workers=1)`，所以 `respond` 是真正的协程，
  事件循环不会被推理卡住。**单 worker 是刻意的**：一个 RKNN3 handle
  不可重入，且要跟 `wdndds.LLMRPCSession` 的串行语义对齐。要在事件
  循环里并行多次 `respond`，调用方自行 `asyncio.gather` 即可
  （每个 session 自带一把锁；跨 session 并行才吃多卡）。
  wdnwebui 的 `WdnaiDetector` 仍在自己的 `Thread` 里跑整条 `respond` 链。
- **JSON hint 的 schema 渲染用 `str(dict)`，不要换成 `json.dumps(indent=2)`**：
  legacy 实现（141ee17）写的就是 `f"{user_msg}\n\n{hint}\n```json\n{schema}\n```"`
  其中 `schema` 是 pydantic 的 `repr(dict)`（单行、单引号 key、Python
  原生类型名 `str` / `dict`）。Qwen3-VL 对 schema 格式敏感，换成
  `json.dumps(..., indent=2)` 后同样图片产出的 JSON 不一样，必须保留
  legacy shape。`_safe_schema` 统一跨 pydantic v1/v2 的 schema dict。
- **Qwen3.5 的 vocab / eos 与 Qwen3 不同**：Qwen3 走 151,936 / 151,645
  的小词表，Qwen3.5 走 248,320 / 248,046 的大词表（多语言扩展）。沿用
  Qwen3 vocab_size 加载 Qwen3.5 的 ``.embed.bin`` 会被 ``reshape`` 整除
  失败；empirical 上 ``Qwen3.5-4B.embed.bin = 1,271,398,400 bytes =
  635,699,200 fp16 values = 248,320 × 2560``。这两个值直接读 GGUF
  tokenizer header（``tok.vocab_size`` / ``tok.eos_token_id``）。
- **多模态占位符必须注册为 `additional_special_tokens`**：`gguf_tokenizer.py`
  在构建 BPE 后会把 `<|vision_start|>` 等加进 `additional_special_tokens`，
  否则 ByteLevel pre-tokenizer 会把它们切碎成 ~7 个普通 token，`rknn3`
  `session_run` 找不到期望的 ID，整个推理直接挂掉。这条对任何用 RKNN3
  的多模态模型通用。
- **TTS 的 VoiceClone 参考音色不读盘**：6 个 fp16 2048-d 参考向量
  zlib+base64 编码内联在 `tts.py:_BUNDLED_SPEAKER_EMBEDDINGS_B64`，
  启动时惰性解一次。`_load_speaker_embeddings()` 不再 walk 文件系统，
  所以 `ref_speaker="news_female"` 在没有 rknn3 dev 树的干净 board 上
  仍然能正确工作。**用户音色**（现场提取）另走一条路：app 写盘，
  worker 启动时从 `WDNAI_TTS_USER_VOICES_DIR` 读回来。
- **`.1828.rknn` 是 RK1828 NPU 量化版本后缀**：`TTSDriver._resolve_model_paths`
  先尝试 `<role>.1828.rknn`，找不到就 fallback 到 `<role>.rknn`。新增模型
  变体请沿用同一约定。
- **NPU 选卡（多卡场景）**：`LLMSession(model=..., npu=K)` 或
  `LLMSession(model=..., device_id=b"...")`。两者互斥；同时不传等价于
  单卡默认行为（向后兼容所有现存调用点）。`npu` 是
  `RKNN3Lite().get_devices_id()` 结果的下标（板上典型 4 卡 → `0..3`），
  实际传到底层 `init_runtime(device_id=...)` 的是 bytes 字符串，**不是
  整数**（RKNN3 的 `RKNN3InitExtend.device_id: c_char_p`，约定为设备唯一
  标识符，如序列号 / SoC 名 / PCI BDF / ADB IP）。`device_id` 接受 str
  或 bytes，str 会被自动 `.encode("utf-8")`。ADB 场景传
  `device_id="rk3588"` 等 SoC 名；网络 ADB 传 `"IP[:Port]"`。解析在
  `_common.resolve_device_id()` 里完成，**只在 backend `__init__` 里**
  调用（`LLMSession.__init__` 自身不触发 `get_devices_id()`，保持 RKNN3
  native driver 懒加载——与 `wdnwebui` 里 `WdnaiDetector._load_wdnai` 同源
  约束）。
- **不要在 backend 里再手写 `ctypes`/`py_object`/CFUNCTYPE 那套**——
  用 `wdnai._librknn.build_llm_callback(result_cb=…, tokenizer=…,
  embeds_data=…, output_cb=…, embed_callback=…)` 即可。shim 会自动
  把绑定方法的 `__self__` 用 `ctypes.py_object` 钉住，并 trampoline
  C 侧的 `RKLLMResult` / `RKLLMCallback` 指针 ↔ Python list /
  numpy view。如果非要直接接 ctypes 指针（例如要写 embed buffer），
  让 callback 接受 `(userdata, tokens, embed_ptr: int, length: int)`
  形态并用 `np.ctypeslib.as_array(cast(embed_ptr, POINTER(c_uint16)),
  shape=(length//2,)).view(np.float16)` 自己包 view。

### 修改守则

改动某个功能的时候，不影响其它功能；如果需要改动到公共功能，需要自动
跑真实测试每个关联的功能项是否都正常。

- 改动只触及目标功能及其直接依赖，不顺手 refactor 无关代码。
- 公共功能改动（`wdnai._librknn` shim、`gguf_tokenizer.py`、
  `backends/` 下的 `@WdnaiCallable` 工厂、`SystemLanguageModel` 预设
  工厂、`Attachment` 序列化、tokenizer 接口等被多个 backend / 多个
  调用方共享的模块）必须跑所有关联 backend / 关联功能的真实测试
  （不是仅 `tests/test_smoke.py` 那种 import-only），不允许"测试跳过"
  或"代码改对了就行"。
- 失败 → 修到全通过再交；下游 `wdnwebui` 集成页面（视频描述 / 视频
  目标检测 / 视频内容检索 / 语音生成 / 文本对话 / 微信视频）必须
  至少 smoke 一遍。

### 文本推理（LLM）

```python
from wdnai import LLMSession, SystemLanguageModel

model = SystemLanguageModel.qwen3_1_7b().with_paths(
    rknn_path="/userdata/qwen3_1.7b.rknn3",
    weight_path="/userdata/qwen3_1.7b.weight",
    tokenizer_path="/userdata/qwen3_tokenizer",
    embed_path="/userdata/qwen3_1.7b.embed.bin",
)
async with LLMSession(model=model, npu=2) as sess:    # 多卡场景选第 3 张卡
    out = await sess.respond(prompt="你好")
    # out is str when no generating= / tools= supplied
    sess.reset()  # 清历史 + rknn_llm.clear_kvcache()
```

### 视觉推理（VLM）

```python
import cv2
from wdnai import Attachment, Generable, Field, LMOutput, LLMSession, SystemLanguageModel
from typing import Literal

@Generable
class Event(LMOutput):
    act: str = Field(description="动作")
    level: Literal["Normal", "Warning"] = Field(default="Normal")

img = cv2.imread("demo.jpg")
model = SystemLanguageModel.qwen3_vl_4b()   # 默认绑 /userdata/models/Qwen3-VL-4B/
async with LLMSession(model=model, device_id=b"0001:11:00.0") as sess:  # 多卡场景指定 device_id
    result = await sess.respond(
        prompt="请描述这张图",
        attachment=[Attachment(img)],       # list 形式，可叠加 text + image
        generating=Event,                    # 解析为 pydantic 模型
    )
    # result is Event (or raw str when JSON parse failed)
    sess.reset()
```

### TTS

```python
from wdnai import Attachment, Generable, Field, LMOutput, LLMSession, SystemLanguageModel

@Generable
class TTSResult(LMOutput):
    text:        str = Field(description="Input text echoed")
    sample_rate: int = Field(default=24000)
    channels:    int = Field(default=1)

model = SystemLanguageModel.qwen3_tts()      # 默认绑 /userdata/models/Qwen3-TTS/
async with LLMSession(model=model, npu=0) as sess:    # 多卡场景选第 1 张卡
    result = await sess.respond(
        attachment=[Attachment("你好世界", kind="text")],
        generating=TTSResult,
        ref_speaker="news_female",          # 内嵌 fp16，无需外部 header
        language="auto",
        volume=0.1,                          # 0.0~1.0 线性缩放 live PCM
        no_play=False,                       # True 时只产 WAV bytes 不出声
        non_streaming_mode=False,
    )
    wav_bytes = result.attachment[0].data   # 24 kHz mono 32-bit IEEE float (format tag 3)
```

### 工具调用 / 结构化输出

```python
from wdnai import Callable, Field, LLMSession, SystemLanguageModel
from typing import Literal

@Callable
class Alarm:
    """现场报警。"""
    level: Literal["low", "high"] = Field(description="严重等级")
    message: str = Field(description="告警文本")
    def __call__(self):
        print(f"!!! alarm({self.level}): {self.message}")

async with LLMSession(model=model) as sess:
    result = await sess.respond(
        prompt="…",
        attachment=[img],
        tools=[Alarm],          # tools= 和 generating= 可以同时给，模型二选一
        generating=Event,
    )
    if isinstance(result, Alarm):
        result()                # 触发副作用（SDK 不会自动调用）
    elif isinstance(result, Event):
        ...
```

### 视觉定位 (LocateAnything)

LocateAnything 是一个 grounding 模型（视觉定位）：单张图输入 → 模型回
答带 ``<ref>label</ref><box>(x1, y1, x2, y2)</box>`` 标记的文本 + 0–1000
归一化坐标的 ROI 列表。模型同时支持点定位（``<box>(x, y)</box>``）和
框定位；SDK 解析后 `GroundingResult.boxes` 只收纳 4 坐标框，点定位落
进 `points` 字段（在 `LocateAnythingBackend.run_image` 返回值中可见；
通过 `LLMSession.respond(generating=GroundingResult)` 走标准
`@Generable` 协议时只有 `boxes` 进入 schema，因为 `GroundingResult`
刻意只声明了 `boxes` / `text` / `level` 三个字段，不污染坐标语义）。

> **grounding 链路绝不能把 JSON Schema 塞进 prompt。**
> `LLMSession._suppress_generating_in_prompt` 对 `kind == "grounding"`
> 返回 `True`（和 `kind == "tts"` 一样）——`generating=` 仍然承载输出
> 侧的解析，但 prompt 里只有用户查询。
>
> 为什么这么严（2026-10-22 真机实测，RK3588）：
>
> * LocateAnything 是 grounding 模型，它把 prompt 里的**每个名词短语**
>   都当成「要定位的目标」。JSON Schema 里的字段名 / 字段描述
>   （`description`、`ROI 左上角 x，0–1000 归一化。`、
>   `单个 ``<box>`` 区域：类别 + 0–1000 归一化的左上/右下坐标。` …）
>   全是名词短语，于是被逐个 ground 出来 —— 症状是「定位到 41 个
>   目标」，标签是 schema 的字段名，不是模型能力问题。
> * 同一批真实帧的 A/B 实测（广州地铁人流素材 + 查询「穿黄色衣服的人」）：
>   带 schema 时每帧目标数 `[1, 0, 144, 5, 2, 1]`、最慢一帧 **13.7 s**；
>   干净 prompt 时 `[1, 0, 1, 1, 1, 1]`、稳定 ~0.5 s。
>   **「污染」是逐帧随机的** —— 某些帧看着正常，某些帧直接炸成上百个
>   框，所以「12 条事件里 8 条干净」不代表那 8 条真的干净。
> * 症状看起来像「模型输出质量波动」，但**证据在输入侧**：schema 是
>   调用方拼进去的。排查这类问题先看真正发出去的 prompt ——
>   `WDNAI_TRACE_PROMPT=1` 会把每次推理实际发出的 prompt 打到 stderr
>   （带 `kind=` 和长度）。
>
> **判定必须落在 `spec.kind` 上 —— 所以调用方必须传真正的 `ModelSpec`。**
> wdnwebui 的主进程曾经把 worker manifest 的 preset dict（UI 用的
> `Dict[str, Any]`，**没有 `kind`**）当 model 传给 `LLMRPCSession`，
> 于是 `spec.kind` 解析成 `None`、抑制静默失效、schema 照旧进 prompt，
> 只改 wdnai **一点效果都没有**。现在 `LLMSession._model_kind()` 兼容
> 两种形状（`ModelSpec` 走属性、dict 走 `["kind"]`），解析不到时
> **打一条 warning** 而不是静默 —— 这类"判定依据解析不出来"的坑
> 必须吵，否则它会以「模型输出质量」的形式回来找你。

```python
import cv2
from wdnai import Attachment, LLMSession, SystemLanguageModel
from wdnai.grounding import GroundingResult

img = cv2.imread("demo.jpg")                    # BGR ndarray
model = SystemLanguageModel.locate_anything()   # 默认绑 /userdata/models/LocateAnything/
async with LLMSession(model=model, npu=0) as sess:
    result = await sess.respond(
        prompt="Locate the person in the image",
        attachment=[Attachment(img)],            # 单图输入
        generating=GroundingResult,             # 解析为 @Generable schema
    )
    print(result.boxes, result.text, result.level)
    # result.boxes[0]  -> BoundingBox(label="person", x1=120, y1=80, x2=480, y2=720)
    # result.text      -> 模型回显的原始文本（带 <ref>/<box> 标签）
    # result.level     -> "Normal" | "Warning"
    sess.reset()
```

### 输出 schema 形态

grounding 链路定义在 `src/wdnai/grounding.py`（独立模块，**不**耦合
`@Callable` 工具链路），由三个角色构成：

- **`BoundingBox`** —— 单个 ``<box>`` 区域，5 个字段：
  - `label: str` —— 目标类别名，对应模型输出的 ``<ref>label</ref>`` 标签文本；
  - `x1` / `y1` / `x2` / `y2: float` —— ROI 左上 / 右下坐标，**0–1000
    归一化**（pydantic `Field(ge=0, le=1000)` 硬约束，模型输出超界会被
    schema 校验直接拒收）。

- **`GroundingResult(LMOutput)`** —— grounding 顶层 payload。除了继承自
  `LMOutput` 的 `attachment: List[Attachment]` slot 外，自身声明 3 个
  字段：
  - `boxes: List[BoundingBox]` —— 从模型输出中解析出的所有 ``<box>`` 区域；
  - `text: str` —— 模型回显的原始 grounding 文本（含 ``<ref>`` /
    ``<box>`` 标记，方便前端调试或回放原始输出）；
  - `level: Literal["Normal", "Warning"]` —— 事件等级标签，常规观察为
    `"Normal"`，异常 / 风险提示为 `"Warning"`（默认 `"Normal"`）。

- **`LMOutput(BaseModel)`** —— 来自 `src/wdnai/lm_output.py` 的结构化
  输出基类，自带 `attachment: List[Attachment]` slot（SDK 自动填充
  二进制 payload，例如 TTS 音频字节、解码后的多模态图等）。

**为什么 `GroundingResult` 继承 `LMOutput` 而不直接 `BaseModel`**：
继承 `LMOutput` 是为了白嫖 `attachment` slot —— 这样 grounding 链路
可以像 TTS 一样附加二进制标注（比如画好 bbox 的 JPEG overlay）而不
需要新发明一个 carrier 类型。这跟 `TTSResult` / `Event` 的设计同构。
如果改继承 `BaseModel`，就需要在 `GroundingResult` 里手抄一份
`attachment` 字段并维护和 `LMOutput` 的双向赋值逻辑。

**0–1000 归一化是协议层格式**：模型在训练阶段就是这么输出
（与输入分辨率无关），调用方拿到后用
`pixel = coord * (W or H) / 1000` 自乘到实际像素。**永远不要把
归一化坐标当像素坐标存盘** —— 同一张 ROI 在 1920×1080 和 640×360
下的像素位置完全不同，但归一化坐标是稳定的；落盘再乘回去几乎一定
错（缩略图、crop、resize 都会偏）。

## 重要提示

- **用中文回答**。
- **除非我要求，否则不要自动 git commit**。
- **不要把 `from wdnai._librknn import RKNN3Lite` 提到 `src/wdnai/__init__.py`
  或 `backends/__init__.py` 顶部**。`wdnai._librknn.load_library()`
  在首次 `RKNN3Lite()` 实例化时会 `CDLL` 打开 librknn3 并 init
  NPU driver；在 `wdnwebui` 这种多模块共存的进程里和 `wdnav` 的 mpp init
  抢资源会 segfault。保持当前"只在具体 backend `__init__` / 私有方法内
  懒导入"的形态。
- **开发支持新类型的输入/输出的LLM模型时，首先在examples目录下新建调用该模型的
  demo脚本，该脚本中会明确model的名字和session的参数。然后按照该名字和输入输出
  的参数，在src/wdnai/model.py、src/wdnai/session.py以及新建的src/wdnai/backends
  /*.py等文件中进行开发。最后运行examples目录下的demo脚本测试并调试。**
- **不要把 `json.dumps(schema, indent=2)` 替换 `_safe_schema` 的 `str(dict)`**：
  Qwen3-VL 对 schema 格式敏感；这条已在 `_safe_schema` 注释里写死。
- **不要把 `<|vision_start|>` / `<|image_pad|>` / `<|video_pad|>` 从
  `additional_special_tokens` 里拿掉**：ByteLevel pre-tokenizer 会把它们
  切碎，`rknn3` 找不到对应 ID，整个推理直接挂掉。
- **不要硬编码 `/userdata/models/...`**：所有路径都从 `WDNAI_MODELS_DIR`
  （根）和 `WDNAI_QWEN3_VL_4B_DIR`（Qwen3-VL-4B 子目录）解析；
  `wdnwebui` 的 `app/main.py` 在 `create_app()` 时把 `base.model_path`
  写进 `os.environ["WDNAI_MODELS_DIR"]` —— SDK 在第一次 `_models_root()`
  调用时读取。
- **`.1828.rknn` 命名**：`<role>.1828.rknn`（RK1828 NPU 量化版本）是新增
  模型的命名约定；`TTSDriver._resolve_model_paths` 自动 fallback 到
  老的 `<role>.rknn`，所以无需手动迁移。
- **`session.reset()` 是循环之间的硬约束**：`LLMSession.reset()` 只清
  Python 端的 `history` + 调一次 `rknn_llm.clear_kvcache()`。不调的话
  下一轮 `respond` 会看到上一轮的状态，autoregressive 的小偏差会滚成
  完全不同的最终输出。`detect_video_carplate.py` 的 `--majority-vote`
  在每次投票前也会 `reset()`。
- **`npu` 与 `device_id` 互斥**：同时传 `LLMSession(..., npu=K, device_id=b"...")`
  直接抛 `ValueError`；`npu` 越界（>= `len(get_devices_id())`）抛
  `IndexError` 并把可用设备列表写进消息里（方便排查板上卡数变化）。
- **`rknn3_query(CORE_MEM_SIZE)` 必须一次查整个数组**：`rknn3_api.h` 的
  `rknn3_core_mem_size` 里 `reserved` 是 **32** 字节，shim 的 `_CoreMemSize`
  必须逐字节对齐（曾经写成 424，导致 runtime 按 `size/56` 算出"要填 8 项"
  而调用方只准备 1 项 —— 更糟的是按 core 逐个查时**每次都拿到 core 0 的
  数值**，于是共享显存的每核大小全错，`rknn3_run` 直接挂死不返回）。
  C++ demo 的写法是 `rknn3_query(ctx, RKNN3_QUERY_CORE_MEM_SIZE, array,
  sizeof(rknn3_core_mem_size) * core_num)` —— **一次查完**。
  实测（用真头文件编译的 C 探针 vs Python shim，同一模型逐行比对）：
  `sizeof(rknn3_core_mem_size)=56`，每核尺寸**确实不同**（DA3 `local`：
  core0=8.70 MB、core1-3=3.11 MB、core4-7=3.42 MB）—— 所以"每核取 max"
  不是可选项，C++ 的 `max_size_by_core[core_id]` 必须照抄。
  **别把它改回 424 / 逐核查询**：上游 commit `851707b` 正是这么改的，
  与 `rknn3_api.h` 不符（头文件里写死 `reserved[32]`）。
- **文生图（kind=`image`）的 JSON schema 提示绝不能进 prompt**：
  扩散模型会把提示词里的每个字**画进图里**，所以 `image` 和 `tts` /
  `grounding` 一样在 `_suppress_generating_in_prompt` 里被排除。改这个
  判定时要连带看 `backends/image.py`。
- **文生图的产物是「附件」不是「文本」**：`respond()` 不带 `generating=`
  时只返回文本、图片会丢（`respond` 的契约就是给调用方一个文本结果）。
  要图用 `session.generate_image(prompt)`（最短）、`session.run_prompt()`
  （返回 `(text, attachments)`）或 `session.respond(generating=LMOutput)`
  （`.attachment` 自动填充）。给文生图传自定义 `generating=` schema 会
  抛 `ValidationError` —— 扩散模型不产 JSON，这是预期行为。
- **不要在 `backends/image.py` 里手写 ctypes 调 librknn3**：文生图的
  三段编排（DiT 分段 / VAE 分段链 / 每段 IO 落位）都是实机量出来的规则，
  已经固化在 **cshim（C 侧）** 里 —— 用 `src/wdnai/_cshim/` 那份，
  **别在 Python 里重写一遍**（纯 Python 版试过：能对齐，但要额外复刻
  FMA 收缩、fp16 次正规数的怪癖、libstdc++ 的 mt19937，容易踩 1 ULP 的坑）。
- **不要给 Qwen-Image 的卡号写默认值**：`backends/image.py::_ep_card_ids()`
  是**唯一**的卡号来源（运行时枚举本机 EP 协卡，按 PCI BDF 升序）⇒
  `card0` / `card1` / `card2` 的默认值是 `None`（留空 = 交给枚举），
  `QwenImageShim` 的 `card_te` / `card_dit` **必填**。写死某个 PCI 地址在
  别的板子上必然失败（`rknn3_init` 报 device id not found ⇒ 「TE 初始化失败」）。
  枚举不到且没显式给值 ⇒ **直接报错**，不拿猜的地址去初始化。
- ⭐⭐⭐ **图生图缺 VAE 编码器时必须报错，绝不能退化成文生图**（2026-11-06）：
  编码器是**可选件**（模型包里没有 `vae_qwenimage21-512.encoder.3588.rknn`
  是合法状态）⇒ `run_images()` 要**明确抛错**说明缺哪个文件，
  **不能**悄悄走 `run_text()` 的文生图路径 —— 悄悄换个语义比报错更糟
  （用户传了图、拿回一张毫不相干的图，还以为「模型效果差」）。
  ⇒ 同理，**编码器装不上只记 warning**，不能影响文生图。

## Known limits

- **LocateAnything 多实例之间不共享 NPU 内部显存**：当前每个
  `LocateAnythingBackend` 实例（每个 `LLMSession(model=locate_anything(),
  npu=K)`）都各自 `load_rknn` + `init_runtime` 走完一遍流程，独立持
  有一份 LLM + vision RKNN context（约几百 MB RKNN 内部 + 几 MB
  `StreamSink` / tokenizer cache / embed table mmap）。同一 NPU 上
  起两个 LocateAnything 实例会各占一份 NPU 内存预算，板上 8 GB 统一
  内存吃紧；RKNN3 提供的"多进程共享同一份已加载模型"能力
  （`/proc/<pid>/maps` 命中同一份 `.weight` mmap）尚未在本 SDK
  落地，所以短期内多实例同卡复用需要外部调度（不同 NPU / 不同
  process）来避免双份加载。后续如要支持，需要在 backend `__init__`
  加一个 `share_with: Optional[LocateAnythingBackend] = None` 入参
  把 `rknn_llm` / `rknn_vision` 句柄与目标实例合并持有，调用前再
  做一次 mutex 串行化（与 `_llm_session_call_lock` 同源）。

- **`_pack_aux` 写 2-D aux tensor 之前必须 `ravel()`，
  设置 `attr.shape[]` + `attr.dtype`，**且 `_pack_inputs` 必须
  把 multimodal 排在 `inputs[0]`、aux 跟在后面，并且 aux mem 必须
  lazy pool 复用不能每次 create**：vision RKNN 的 deepstack 输出
  是 ``(n_patches, embed_dim)`` 形状（Qwen3-VL-4B @ 384 实测
  ``(144, 2560)``，Qwen3.5-VL 同款），C-side
  ``rknn3_session_run`` 期待完整 ``_tensor_attr`` metadata +
  multimodal at ``inputs[0]`` + 复用 aux mem。``_LLMSession._pack_aux``
  与 ``_LLMSession._pack_inputs`` 历史上有 4 个 bug：
  1. ``dst[:] = data`` 把 2-D ndarray 赋给 1-D fp16 视图 →
     numpy 抛 ``ValueError: could not broadcast input array from
     shape (144, 2560) into shape (368640,)``。修复：赋值前
     ``data.ravel() if data.ndim != 1 else data``。
  2. ``attr.shape[]`` 没设（保持全零） + ``attr.dtype`` 默认 0
     （= ``RKNN3_TENSOR_FLOAT32``）。C 库拿 ``n_elems=368640`` 配
     dtype_size(4) 算出来 1474560 bytes，与实际
     ``aligned_size=737280`` 不符 → ``rknn3_session_run`` 返回
     ``ret=-1``，包装层抛 ``llm session_run failed: ret=-1``。
     修复：循环 ``attr.shape[i] = data.shape[i]``；
     ``attr.dtype = RKNN3_TENSOR_FLOAT16``。
  3. ``_pack_inputs`` 把 aux 排在前面、multimodal 排在最后。
     C 库第一个 entry 必须是 multimodal
     （``RKLLMSession: Please put the LLM input at the first
     position!``），aux 跟在后面 —— 注释里说"double-prompt caused
     ret=-1"是把表面症状记成原因；真正原因是 inputs 顺序颠倒。
     修复：先 push multimodal，再 push aux。早期版本 + bug #1/#2
     的修复**还不够**——前两个 bug 修了之后，shape/dtype 满足
     C 库校验，但 inputs 顺序错了，C 库仍然 ret=-1。
  4. 每次 ``_pack_aux`` 都新调 ``rknn3_create_mem``。NPU driver 的
     aux-mem pool 是有限资源——Qwen3-VL-4B @ 384 在 RK3588 板上
     跑约 8 次 infer 后就报 ``MSG_MEM_CREATE_ACK fail,
     ack = ACK_FAIL, expect = ACK_SUCC!`` + ``rknn3_create_mem:
     device failed to create memory``。Python 异常穿过 ctypes
     边界，worker SIGSEGV (``rc=-11``)。修复：``_LLMRuntime`` 加
     ``_aux_pool: Dict[int, dict]``，``_pack_aux`` 第一次 lazy
     create，后续只 memcpy bytes。C++ 参考
     ``examples/Qwen3_VL/cpp/llm/rknn_qwen3_vl_llm.cc:111-120``
     就是这个模式（init 时一次性 alloc 3 个 deepstack mem，
     run 时只更新内容）。
  C++ 参考 ``examples/Qwen3_VL/cpp/llm/rknn_qwen3_vl_llm.cc:111-118``
  在 ``init_qwen3_vl_llm`` 里调 ``rknn3_query(INPUT_ATTR)`` 拿
  完整 attr（含 shape/dtype/stride）后才填进 aux 输入；wdnai
  Python shim 没有 query 路径（aux tensor mem 是
  ``rknn3_create_mem`` 在 LLM session 之外分配的，wdnai
  VLM 后端每次 infer 重新 ``_pack_aux``），所以必须手动补齐。
  后续若 wdnai 改成「session 初始化时 query 一次存起来」的
  路径，这里可以省掉 shape/dtype 设置——前提是 query 真的
  拿的是 deepstack 那 3 个 aux input 的 attr（模型可能把
  deepstack 当作子 graph 特殊命名，不是简单的 input index）。
  ``tests/test_mock_e2e.py::test_pack_aux_2d_deepstack_does_not_broadcast_fail``
  覆盖 bug #1 + #2：直接调 ``_pack_aux`` 用 stubbed
  ``rknn3_create_mem``，断言写入字节 ==
  ``sentinel_2d.ravel()`` **且** ``attr.shape[0]==144,
  shape[1]==2560, dtype==RKNN3_TENSOR_FLOAT16``。
  ``test_pack_inputs_multimodal_must_be_first`` 覆盖 bug #3：
  直接调 ``_pack_inputs``，断言 ``inputs[0].input_type ==
  MULTIMODAL`` 且 ``inputs[1..3].input_type == AUX``，外加
  每个 aux 的 ``attr.index == 2 + i`` round-trip 校验。
  ``test_pack_inputs_multimodal_must_be_first`` 还加了 **pool 复用
  校验**（bug #4）：第二次 ``_pack_inputs`` 调用之后
  ``runtime._aux_pool`` 大小不能增长——只要 cache 命中就不该再
  create_mem。回归 guard 防止后人把 pool 路径意外简化成
  ``_pack_aux`` 每次都 ``create_mem``（一旦如此，在真机上
  Qwen3-VL-4B @ 384 跑 ~8 帧就 driver pool 耗尽 → SIGSEGV）。
  临时 revert 任一 bug 都会被对应测试抓住：broadcast 版本
  ValueError 触发；shape/dtype 版本 ``assert 0 == 144`` 触发；
  顺序版本 ``AssertionError: inputs[0] must be MULTIMODAL`` 触发；
  pool 版本（手工把 ``if pool is None`` 改回 ``mem_ptr = create_mem``
  每次都调） ``len(runtime._aux_pool) == first_pool_size + 3`` 触发。
  Mock ``_FakeRknn.inference`` 也必须保持 2-D ``(n_patches, dim)``
  输出——改回 1-D 会让 ``test_image_qwen3vl`` 等端到端测试
  静默退化（1-D 源不会触发 numpy broadcast）。

- **ASR 的「最后一块」必须够长（≥ `N_FFT + HOP_LENGTH`）**，否则
  ``_log_mel_spectrogram`` 会在反射填充处越界：
  - ``_log_mel_spectrogram`` 按 C 的语义把音频裁到
    ``n_frame * hop_length``（1 帧 → **160** 采样），再做**反射填充**
    （需要 ``n_fft // 2`` = **200** 采样）。块长 < ``N_FFT + HOP``
    （560 @ 16 kHz）时裁完不足 200 → numpy 抛
    ``ValueError: could not broadcast input array from shape (160,)
    into shape (200,)`` → worker 报错 → **整场识别报废**。
  - C 的 ``reflect_pad``（``utils/audio_utils.cc:111``）在同一个位置是
    **越界读**（UB）——numpy 移植把「C 没定义的地方」翻译成了**硬失败**。
    官方 Python ``demo_streaming.py`` 则**从不**喂短块（``/api/finish``
    直接丢弃尾巴），所以两条参考路径都踩不到。
  - 修复有两层：``_reflect_pad`` 只反射**实际存在**的采样
    （``w = min(pad_width, n)``，越界处补 0）；``_run_online_block`` 的
    块长下限从 ``N_FFT`` 提到 ``N_FFT + HOP_LENGTH``（保证裁完还够
    反射填充）。正常块（16000 采样）的 mel 形状**逐字节不变**。
  - **末块的长度是调用方无法控制的事**（= 总采样数 mod 块长），所以
    **保证块够长是块切割方（SDK）的责任**，不是调用方的。