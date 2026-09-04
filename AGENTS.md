# AGENTS.md

Quick reference for OpenCode sessions working on the `wdnai` SDK. The
project is a Python wrapper around rknn3lite for **LLM / VLM / TTS**
inference on Rockchip NPUs (RK1828); import name `wdnai`, distribution
name `wdnai`.

> Vendored in-tree as `.venv/src/wdnai/` (editable install from
> `git@10.10.10.10/wardner/wdnai.git`). It is *not* on PyPI. See
> `wdnwebui/AGENTS.md` "One-time setup" for the install command.


## Repo map (one-line each)

- `pyproject.toml` — package metadata (name=wdnai, version=0.1.1);
  deps: `numpy>=1.23`, `transformers>=4.40`, `pydantic>=1.10`,
  `opencv-python>=4.5`, `gguf>=0.6`. Optional: `[lite]`
  `opencv-python-headless`, `[dev]` `pytest`+`build`.
- `src/wdnai/__init__.py` — public API: re-exports `Generable`,
  `Callable`, `Attachment`, `LMOutput`, `ModelSpec`,
  `SystemLanguageModel`, `LLMSession`, plus pydantic's `BaseModel`
  / `Field`. Holds `__version__`. **Does not import `rknn3lite`** at
  module top — that import only happens inside the backend modules
  (see "Gotchas").
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
  model directory), `qwen3_tts()`, `qwen3_vl_embedding()` (retrieval
  embedding, no token generation, binds to
  `/userdata/models/Qwen3-VL-Embedding-2B/` and raises
  `FileNotFoundError` listing missing files).
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
- `src/wdnai/session.py` — `LLMSession`: `async def respond(prompt,
  attachment, generating, tools, debug, perf, **overrides)`. Owns
  one backend (chosen by `ModelSpec.kind`). Responsibilities:
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
  "embedding": EmbeddingBackend}` registry (asr is loaded on demand).
- `src/wdnai/backends/embedding.py` — `EmbeddingBackend` (kind=
  "embedding"): loads llm/vision `.1828.rknn` + `.weight` + embed.bin
  + tokenizer under `WDNAI_MODELS_DIR/Qwen3-VL-Embedding-2B`, feeds
  vision at 576x576 via `letterbox_resize` (so PyAV-decoded frames
  with arbitrary aspect ratios get grey-padded squares, never
  stretched), exposes `embed_text(text)` and
  `embed_video_chunk(frames, text="")` returning L2-normalized
  2048-dim vectors (the LLM runs through the plain RKNN3Lite path,
  no token generation).
- `src/wdnai/backends/_common.py` — shared RKNN plumbing:
  `StreamSink` (collects tokens / text / perf from streaming
  callback), `make_result_callback` / `make_tokenizer_callback` /
  `make_embed_callback` (the three C callbacks `RKLLMCallback`
  expects), `build_llm_callback`, `load_tokenizer` (`.gguf` →
  `gguf_tokenizer.py`; otherwise `AutoTokenizer.from_pretrained`),
  `load_embed_table` (mmap fp16 → reshape `(vocab_size, dim)`),
  ChatML template (`system\nYou are a helpful assistant.\n` /
  `user\n` / `\nassistant\n`), `build_llm_args` (merges
  `spec.options` into the `llm_args[0]` dict), `printf_perf`
  (Prefill / Generate / vision / total latency table).
- `src/wdnai/backends/text.py` — `TextBackend`: loads `rknn` +
  `weight`, init_runtime with the LLMM-mode callback, `apply_chat_template`,
  exposes `run(prompt, **overrides)`. Overrides `max_new_tokens` /
  `enable_thinking` / `keep_history` are passed straight to
  `rknn.session_run`.
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
- `src/wdnai/backends/tts.py` — `TTSBackend` + vendored
  `Qwen3TTSDriver` (1542 lines, originally in
  `rknn3-toolkit-lite/examples/Qwen3-TTS/qwen3_tts_demo.py` and
  imported via `sys.path` injection; now in-tree so
  `import wdnai.backends.tts` is self-contained). Pipeline:
  `text → text_projection (1 core) → talker (8 cores, AR) →
  code_predictor (8 cores, AR, offline) → speech_decoder
  (sliding window 25/50) → 24 kHz mono 32-bit IEEE float WAV`.
  `AudioPlayer` (sounddevice → aplay → no-op fallback). VoiceClone
  speaker embeddings (`girl_base` / `ahu`) are zlib+base64-inlined
  in `_BUNDLED_SPEAKER_EMBEDDINGS_B64` so no external
  `qwen3tts_speaker_embed.h` is required. `run_text()` returns
  `(prompt_str, [Attachment])`; the SDK unpacks the tuple and
  assigns the audio bytes to `LMOutput.attachment`.
- `src/wdnai/models/README.md` — model file layout
  (`/userdata/models/<name>/`), `.1828.rknn` naming convention, the
  Qwen3-VL-4B / Qwen3-TTS file tables. The directory itself ships
  empty in-tree; weights are user-placed (or installed via the
  rknn3_model_zoo flow).
- `tests/test_mock_e2e.py` — the only test file. Monkey-patches
  `rknn3lite.api.RKNN3Lite` + transformers + the result-callback +
  `build_llm_callback` so the full SDK runs without hardware. Covers
  Qwen3 / Qwen2.5 text, Qwen3-VL / Qwen2.5-VL image, override
  pass-through, and the full `respond()` path (JSON hint injection
  + pydantic parse).
- `examples/` — runnable demos, no `test_` prefix. End-to-end
  consumers; `detect_image.py` (10-round structured-output loop),
  `detect_video.py` (`wdnav.VideoCapture` + per-N-frames VLM), the
  `detect_video_carplate.py` plate detector (single / two-stage /
  `--majority-vote K` / background `asyncio.Task` for non-blocking
  display), `function_call.py` (multi-tool dispatch with
  `isinstance`), `text_to_speech.py` (TTS CLI with `--ref-speaker
  girl_base` / `--volume` / `--no-play`). External dataset dirs
  `光厂_*` hold plate frames + `.txt` ground truth.


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
- **rknn3lite 是同步 C API，`LLMSession.respond` 是 async 但目前不丢线程池**：
  backend 的 `run` / `run_text` / `run_images` 是同步阻塞调用；`respond`
  当前直接在事件循环里跑（≈ 1s @ 4B）。如果上层要在事件循环里并行多次
  `respond`，调用方自行 `asyncio.gather` 或者丢 `loop.run_in_executor`。
  wdnwebui 的 `WdnaiDetector` 已经在自己的 `Thread` 里跑整条 `respond` 链。
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
- **TTS 的 VoiceClone speaker 不读盘**：两个 fp16 2048-d 参考向量
  zlib+base64 编码内联在 `tts.py:_BUNDLED_SPEAKER_EMBEDDINGS_B64`，
  启动时惰性解一次。`_load_speaker_embeddings()` 不再 walk 文件系统，
  所以 `ref_speaker="girl_base"` / `"ahu"` 在没有 rknn3 dev 树的干净
  board 上仍然能正确工作。
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
        ref_speaker="girl_base",             # 内嵌 fp16，无需外部 header
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

## 重要提示

- **用中文回答**。
- **除非我要求，否则不要自动 git commit**。
- **不要把 `from rknn3lite.api import RKNN3Lite` 提到 `src/wdnai/__init__.py`
  或 `backends/__init__.py` 顶部**。RKNN3 native libs 在 import 时会 init
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