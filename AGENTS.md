# AGENTS.md

Quick reference for OpenCode sessions working on the `wdnai` SDK. The
project is a Python wrapper around librknn3 (`/usr/lib/librknn3_api.so`)
for **LLM / VLM / TTS** inference on Rockchip NPUs (RK1828); import
name `wdnai`, distribution name `wdnai`.

> Vendored in-tree as `.venv/src/wdnai/` (editable install from
> `git@10.10.10.10/wardner/wdnai.git`). It is *not* on PyPI. See
> `wdnwebui/AGENTS.md` "One-time setup" for the install command.

> **As of v0.4.x**, wdnai no longer depends on the `rknn3lite` Python
> wheel. The vendored `wdnai/_librknn.py` is a ctypes shim that
> mirrors the subset of `rknn3lite.api` the SDK actually exercises,
> driven directly against librknn3. See
> [`wdnai/_librknn.py`](#wdnaishim-library-binding) below.


## Repo map (one-line each)

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
  missing files; capability key `swcap_grounding`).
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

### 视觉定位 (LocateAnything)

LocateAnything 是一个 grounding 模型（视觉定位）：单张图输入 → 模型回
答带 ``<ref>label</ref><box>(x1, y1, x2, y2)</box>`` 标记的文本 + 0–1000
归一化坐标的 ROI 列表。模型同时支持点定位（``<box>(x, y)</box>``）和
框定位；SDK 解析后 `GroundingResult.boxes` 只收纳 4 坐标框，点定位落
进 `points` 字段（在 `LocateAnythingBackend.run_image` 返回值中可见；
通过 `LLMSession.respond(generating=GroundingResult)` 走标准
`@Generable` 协议时只有 `boxes` 进入 schema，因为 `GroundingResult`
刻意只声明了 `boxes` / `text` / `level` 三个字段，不污染坐标语义）。

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