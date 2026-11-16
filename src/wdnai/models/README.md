Model files live at a fixed on-device location: `/userdata/models/<model-name>/`.
Override the root with the `WDNAI_MODELS_DIR` environment variable.

No network access is performed at any point; place the model files manually.


# Tree structure

```
/userdata/models/
├── Qwen3-VL-4B/                                  # vision-language (multi-resolution vision)
│   ├── llm_Qwen3-VL-4B.1828.rknn                 # LLM RKNN (25 MB)
│   ├── llm_Qwen3-VL-4B.weight                    # LLM weights (2.3 GB)
│   ├── vision_Qwen3-VL-4B.384x384.1828.rknn      # Vision encoder RKNN @ 384x384 (4.2 MB) — REQUIRED
│   ├── vision_Qwen3-VL-4B.384x384.weight         # Vision weights @ 384x384 (234 MB)   — REQUIRED
│   ├── vision_Qwen3-VL-4B.768x768.1828.rknn      # Vision encoder RKNN @ 768x768 (≈14 MB) — OPTIONAL
│   ├── vision_Qwen3-VL-4B.768x768.weight         # Vision weights @ 768x768 (≈700 MB)   — OPTIONAL
│   ├── Qwen3-VL-4B.embed.bin                     # Token embedding table (742 MB, vocab 151936 × 2560-dim float16)
│   └── Qwen3-VL-4B.tokenizer.gguf                # GGUF-format tokenizer (5.7 MB; loaded via wdnai.gguf_tokenizer)
│
├── Qwen3.5-4B/                                   # vision-language (also powers pure-text chat via the VL RKNN)
│   ├── Qwen3.5-4B-llm.1828.rknn                  # LLM RKNN (38 MB, MRoPE baked in) — for qwen3_5_vl_4b()
│   ├── Qwen3.5-4B-llm.weight                     # LLM weights (2.5 GB)
│   ├── Qwen3.5-4B-llm.embed.bin                  # Token embedding table (1.27 GB, vocab 248320 × 2560-dim fp16)
│   ├── Qwen3.5-4B-llm.tokenizer.gguf             # GGUF-format tokenizer (10.9 MB; vocab 248320, EOS 248046)
│   ├── Qwen3.5-4B-vision.384x384.1828.rknn       # Vision encoder RKNN @ 384x384 (5 MB) — REQUIRED
│   ├── Qwen3.5-4B-vision.384x384.weight          # Vision weights @ 384x384 (188 MB)   — REQUIRED
│   ├── Qwen3.5-4B-vision.768x768.1828.rknn       # Vision encoder RKNN @ 768x768 (11 MB) — OPTIONAL
│   └── Qwen3.5-4B-vision.768x768.weight          # Vision weights @ 768x768 (195 MB)   — OPTIONAL
│
│   # The VLM-mode LLM RKNN has MRoPE (``mrope_type='Qwen3.5'``,
│   # ``mrope_section=[11, 11, 10]``) baked into the graph for image
│   # position encoding; with no vision tokens attached MRoPE reduces
│   # to standard positional encoding, so a single RKNN powers both
│   # ``swcap_text_chat`` and ``swcap_vision_chat`` capabilities
│   # in ``MODEL_PRESETS["qwen3-5-vl-4b"]``.
│
├── Qwen3-TTS/                         # pyrkaiweb's text-to-speech back end
│   ├── text_projection.1828.rknn       # ① 文本侧 projector — 普通 RKNN 图，core_mask=0x01（1 核）
│   ├── text_projection.weight          #   （16 MB，与上面配对）
│   ├── talker.1828.rknn                # ② 主自回归 talker LLM（core_mask=0xff，8 核全开）
│   ├── talker.weight                   #   （784 MB）
│   ├── code_predictor.1828.rknn        # ③ 副自回归 code_predictor LLM（offline，无 KV cache）
│   ├── code_predictor.weight           #   （144 MB）
│   ├── speech_decoder.1828.rknn        # ④ codec→wav 解码器（sliding window 25/50）
│   ├── speech_decoder.weight           #   （218 MB）
│   ├── talker_text_embed.fp16.bin      # 文本 token 嵌入表 (151936 × 2048 fp16, 594 MB)；text_projection 输入
│   ├── talker_input_embed.fp16.bin     # talker 输入嵌入 (3072 × 2048 fp16, 12 MB)；含 codec + 控制 token
│   ├── codec_embed.fp16.bin            # codec token 嵌入 (15 × 2048 × 2048 fp16, 120 MB)；code_predictor 查表用
│   └── tokenizer.json                  # HuggingFace BPE tokenizer (11 MB)
│
├── HY_MT_1_5/                         # 混元 HY-MT1.5-1.8B 翻译模型（文本 → 文本）
│   ├── HY-MT1.5-1.8B.1820.rknn         # LLM RKNN (23 MB)
│   ├── HY-MT1.5-1.8B.weight            # LLM weights (1.1 GB)
│   ├── HY-MT1.5-1.8B.embed.bin         # Token embedding table (472 MB, vocab 120818 × 2048-dim float16)
│   ├── HY-MT1.5-1.8B.tokenizer.gguf    # GGUF tokenizer (4.8 MB; hunyuan-dense BPE)
│   └── HY-MT1.5-1.8B.config.pkl        # 转换工具的回执（chat template + HF config）—— 运行期不读
│
├── Intern_Decision-4B/                # Intern-Decision-4B 结构化决策模型（状态 → 选项 + 概率）
│   ├── Intern-Decision-4B-llm.rknn   # 文本 RKNN (25 MB)；输出是 62 个候选符号的 logits
│   ├── Intern-Decision-4B-llm.weight # 文本权重 (2.1 GB)
│   ├── Intern-Decision-4B-llm.embed.bin # Token embedding table (1.27 GB, vocab 248320 × 2560-dim float16)
│   ├── Intern-Decision-4B-llm.tokenizer.gguf # GGUF tokenizer (11 MB; qwen3.5 BPE)
│   ├── Intern-Decision-4B-vision.rknn # 视觉 RKNN (4.8 MB)；输入 uint8 NHWC 384×384
│   └── Intern-Decision-4B-vision.weight # 视觉权重 (682 MB)
│
└── ... future model variants go here ...
```

## Naming convention

Every RKNN figure carries a **`.1828.rknn`** suffix (the RK1828 NPU
generation it was quantized for), matching what rknn3_model_zoo ships
to this board. Specifically the pattern is:

| Layer            | Suffix     | Why                              |
| ---------------- | ---------- | -------------------------------- |
| RKNN figure      | `.1828.rknn` | indicates NPU target chip        |
| RKNN weights     | `.weight`   | shared across NPU revisions      |
| Embedding table  | `.fp16.bin` | raw fp16 mmap'd numpy lookup     |
| Tokenizer        | `.json` (HF) / `.gguf` (GGUF) | matches upstream tooling      |

A `.rknn` without the chip suffix is legacy — `TTSBackend._install_path_adapter`
tries `<role>.1828.rknn` first and falls back to the bare `<role>.rknn`
for installations that haven't been renamed yet.

## Models

### Qwen3-VL-4B (vision-language)

The 4B model is the **full** (non-pruned) vision export.
`SystemLanguageModel.qwen3_vl_4b` points to `/userdata/models/Qwen3-VL-4B/`
by default and raises `FileNotFoundError` listing the missing files if
the directory is incomplete. To add a new model variant, place
another sibling directory under `/userdata/models/<new-variant>/` and
add a preset + path helper in [wdnai/model.py](../model.py).

**Multi-resolution vision**: the directory may carry multiple
`<role>.<WxH>.1828.rknn` + `<role>.<WxH>.weight` pairs — `VLMBackend`
auto-discovers them at init and picks the one whose training resolution
is closest to the input image's longest side per call. The 384x384
pair is required (it's the `spec.image_size=384` fallback); additional
resolutions (e.g. 768x768 for high-detail inputs) are loaded when both
files in the pair exist. Memory cost scales linearly with the number
of variants resident on the NPU — bump `memory_estimate_mb` in
[wdnwebui](../)'s `MODEL_PRESETS` if you ship both 384x384 and 768x768
together. To add a new resolution tier, drop a new pair into the
directory; no preset change is needed.

### Qwen3.5-VL-4B (vision-language, also powers text chat)

Vision-language model (image-in, text-out) that **also handles
pure-text inference** — the VLM-mode LLM is a regular LLM with
`mrope_type='Qwen3.5'` / `mrope_section=[11, 11, 10]` baked into the
graph for image position encoding; with no vision tokens attached
MRoPE reduces to standard positional encoding, so a single RKNN
powers both the `swcap_text_chat` and `swcap_vision_chat`
capabilities declared by `MODEL_PRESETS["qwen3-5-vl-4b"]`.
`SystemLanguageModel.qwen3_5_vl_4b` points to
`/userdata/models/Qwen3.5-4B/` by default (override via
`WDNAI_MODELS_DIR` or pin a custom path with
`WDNAI_QWEN3_5_4B_VL_DIR`) and raises `FileNotFoundError` listing
the missing files if the directory is incomplete.

| File (`<root>/Qwen3.5-4B/…`)        | Purpose                                  | Required          |
| ----------------------------------- | ---------------------------------------- | ----------------- |
| `Qwen3.5-4B-llm.1828.rknn` + `.weight`            | LLM RKNN (MRoPE baked in)        | yes               |
| `Qwen3.5-4B-vision.384x384.1828.rknn` + `.weight` | Vision encoder @ 384×384         | yes (fallback)    |
| `Qwen3.5-4B-vision.768x768.1828.rknn` + `.weight` | Vision encoder @ 768×768         | optional          |
| `Qwen3.5-4B-llm.embed.bin`          | Token embedding table (248320 × 2560 fp16, 1.27 GB) | yes   |
| `Qwen3.5-4B-llm.tokenizer.gguf`     | GGUF tokenizer (248320 vocab, EOS 248046) | yes             |

`VLMBackend` auto-discovers every `Qwen3.5-4B-vision.*x*.1828.rknn`
sibling in the directory at load time (same multi-resolution
discovery as Qwen3-VL-4B). Note the Qwen3.5 vocabulary (248,320
tokens / EOS 248,046) is distinct from Qwen3's (151,936 / 151,645)
— mixing them would crash the embed-table reshape.

### Qwen3-TTS (text-to-speech)

The full Qwen3-TTS pipeline runs four RKNN3 models on the same RK1828
PCIe card as the VLM. Used by pyrkaiweb's `#page-tts` (the
Qwen3-TTS model itself is loaded via `TTSBackend` in `wdnai/backends/tts.py`,
which wraps the in-tree `Qwen3TTSDriver` vendored into the same
`wdnai/backends/tts.py` module). The driver used to live in
`rknn3-toolkit-lite/examples/Qwen3-TTS/qwen3_tts_demo.py` and was
imported at runtime via `sys.path` injection; it has since been
ported into the SDK so `import wdnai.backends.tts` no longer needs
to find the upstream demo anywhere on disk.

| #  | File (`<root>/Qwen3-TTS/…`)            | Purpose                                              | core_mask         |
| -- | -------------------------------------- | ---------------------------------------------------- | ----------------- |
| ①  | `text_projection.1828.rknn` + `.weight`| Projects token text embeddings into talker hidden    | `0x01` (1 core)   |
| ②  | `talker.1828.rknn` + `.weight`         | Autoregressive talker LLM (main codec + hidden)      | `0xff` (8 cores)  |
| ③  | `code_predictor.1828.rknn` + `.weight` | Auxiliary LLM filling the remaining 15 codec channels | `0xff` (8 cores)  |
| ④  | `speech_decoder.1828.rknn` + `.weight` | Codec → waveform sliding-window decoder              | `0xff` (8 cores)  |

Plus three lookup tables consumed by Python (no NPU):

| File                                | Shape                                | Used by                  |
| ----------------------------------- | ------------------------------------ | ------------------------ |
| `talker_text_embed.fp16.bin`        | `(151936, 2048)` fp16                | text_projection input    |
| `talker_input_embed.fp16.bin`       | `(3072, 2048)` fp16                  | talker embed callback    |
| `codec_embed.fp16.bin`              | `(15, 2048, 2048)` fp16              | code_predictor embed callback |
| `tokenizer.json`                    | HuggingFace BPE                      | text tokenization        |

The 4 `.weight` files total **1.16 GB**; the figure (`.1828.rknn`) files
are **~46 MB** total; the three embed tables add another **~726 MB**.
Output WAV is 24 kHz mono 32-bit IEEE-float (format tag 3); the 5-sec
"你好" sentence takes about 6 sec on the board (real-time capable).

To register a new TTS variant, lay down a sibling directory under
`/userdata/models/<new-variant>/` with the same file naming and add
a preset that points to it.

### Qwen3-VL-Embedding-2B (retrieval embedding)

A retrieval embedding model — **no token generation**. The LLM is
loaded via the plain `RKNN3Lite` path (not the llm-mode session) and
run with 7 manual inputs; `embed_text` / `embed_video_chunk` return a
L2-normalized 2048-dim vector. File layout mirrors Qwen3-VL-4B
(llm/vision `.1828.rknn` + `.weight` + embed table + tokenizer).
Used by pyrkaiweb's `#page-video-content-search` via the
`EmbeddingBackend` in `wdnai/backends/embedding.py`
(`SystemLanguageModel.qwen3_vl_embedding`).

Unlike Qwen3-VL-4B, this model uses a **single fixed** vision
resolution (576x576); the multi-resolution discovery in `VLMBackend`
does not apply here. The legacy single-file naming
(`vision_Qwen3-VL-Embedding-2B.1828.rknn`, no `<WxH>` suffix) is what
the embedding backend loads — `VLMBackend`'s discovery helper would
fall through to the legacy single-file path automatically when no
`<WxH>` siblings are present.

| File (`<root>/Qwen3-VL-Embedding-2B/…`)     | Purpose                              |
| ------------------------------------------- | ------------------------------------ |
| `llm_Qwen3-VL-Embedding-2B.1828.rknn` + `.weight`   | Embedding LLM (plain RKNN3Lite path) |
| `vision_Qwen3-VL-Embedding-2B.1828.rknn` + `.weight`| Vision encoder (input 576×576)       |
| `Qwen3-VL-Embedding-2B.embed.bin`           | Token embedding table (151936 × 2048) |
| `Qwen3-VL-Embedding-2B.tokenizer.gguf`      | GPT-2 BPE + ByteLevel tokenizer      |

`max_context_len` defaults to 2048 tokens; the vision feature is fed
at 576×576 (matching the model's native input) and each video chunk
contributes `FRAME_TOKENS` vision tokens (see
`wdnai/backends/embedding.py`).

### HY-MT1.5-1.8B (machine translation)

A Hunyuan-Dense translation model — **text in, text out**, but *not* a
general chat model: it is a translation specialist. The
`SystemLanguageModel.hy_mt_1_5()` preset points to
`$WDNAI_MODELS_DIR/HY_MT_1_5/` (default
`/userdata/models/HY_MT_1_5/` when `WDNAI_MODELS_DIR` is unset; pin a
custom path with `WDNAI_HY_MT_1_5_DIR`) and raises `FileNotFoundError`
listing the missing files if the directory is incomplete.

| File (`<root>/HY_MT_1_5/…`)      | Purpose                                            |
| -------------------------------- | -------------------------------------------------- |
| `HY-MT1.5-1.8B.1820.rknn`        | LLM RKNN (23 MB)                                   |
| `HY-MT1.5-1.8B.weight`           | LLM weights (1.1 GB)                               |
| `HY-MT1.5-1.8B.embed.bin`        | Token embedding table (120818 × 2048 fp16, 472 MB) |
| `HY-MT1.5-1.8B.tokenizer.gguf`   | hunyuan-dense BPE tokenizer (4.8 MB)               |
| `HY-MT1.5-1.8B.config.pkl`       | Conversion receipt — **not read at runtime**        |

`vocab_size` is **120818** and `special_eos_id` is **120020**
(`<｜hy_place▁holder▁no▁2｜>`), both read from the package rather than
guessed. `max_context_len` is **8192** — the value baked into the
export (the RKNN's embedded `hf_config_json` says
`max_position_embeddings=8192`); the upstream checkpoint's 262144 is a
training limit the export did not keep.

**The prompt template is not ChatML.** HY-MT speaks the Hunyuan
dialect — `<｜hy_begin▁of▁sentence｜>` / `<｜hy_User｜>` /
`<｜hy_Assistant｜>` — so the preset carries its own
`spec.chat_template` (`wdnai.model.HY_MT_CHAT_TEMPLATE`) and the
backend installs that triple on the C session instead of the ChatML
one. Measured on the board: with the ChatML template the model returns
an **empty string**; with the Hunyuan template it translates normally.

Two more measured facts worth keeping:

* **The prompt must carry the translation instruction.** A bare source
  sentence is answered as a chat turn ("The weather is very nice
  today." → "Yes, that's true. It seems like we are in a pleasant
  weather today."). Prefixing it with e.g. "Translate the following
  text from English to Chinese:" produces the translation. The model is
  insensitive to the exact wording.
* **The three control tokens are single vocabulary entries** (120000 /
  120006 / 120007), but the ByteLevel pre-tokenizer splits them into
  ~5 regular tokens each. `wdnai.gguf_tokenizer` registers them as
  `additional_special_tokens` (same trick as the Qwen multimodal
  placeholders) so they survive decoding.

Sampling is greedy (`top_k=1`, `temperature=1e-6`): a translation must
stay faithful to the source.

### Intern-Decision-4B (structured decision)

Intern-Decision is **not** a chat model: it does not generate text.
For every decision field it emits the logits of 62 candidate symbols
(`A-Z` + `a-z` + `0-9`), which the CPU side turns into option
probabilities (softmax + a checkpoint-specific temperature
calibration). Three field types are supported — `choice` (pick one),
`score` (the option keys are numbers; the output is the
probability-weighted score) and `noul` (yes/no plus the affirmative
probability).

`SystemLanguageModel.intern_decision_4b()` points to
`$WDNAI_MODELS_DIR/Intern_Decision-4B/` (default
`/userdata/models/Intern_Decision-4B/` when `WDNAI_MODELS_DIR` is
unset; pin a custom path with `WDNAI_INTERN_DECISION_4B_DIR`) and
raises `FileNotFoundError` listing the missing files if the directory
is incomplete.

| File (`<root>/Intern_Decision-4B/…`)   | Purpose                                              |
| -------------------------------------- | ---------------------------------------------------- |
| `Intern-Decision-4B-llm.rknn`          | text RKNN (25 MB) — output: 62 candidate logits      |
| `Intern-Decision-4B-llm.weight`        | text weights (2.1 GB)                                |
| `Intern-Decision-4B-llm.embed.bin`     | token embedding table (248320 × 2560 fp16, 1.27 GB)  |
| `Intern-Decision-4B-llm.tokenizer.gguf`| GGUF tokenizer (11 MB; qwen3.5 BPE)                  |
| `Intern-Decision-4B-vision.rknn`       | vision RKNN (4.8 MB) — input: uint8 NHWC 384×384     |
| `Intern-Decision-4B-vision.weight`     | vision weights (682 MB)                              |

`vocab_size` is **248320**, `special_eos_id` is **248046**
(`<|im_end|>`), `max_context_len` is **4096** (the value baked into
the export). The **calibration** temperature is a different thing from
the sampling temperature: it is looked up from the model's embedding
width at runtime (`wdnai.decision.CHECKPOINT_TEMPERATURES` — 4B is
`1.99241824`).

Eight things that are easy to get wrong, all measured on the board — and
all of them change the reported probabilities **without raising an error**:

* **The ChatML markers live in the prompt text, not in a template.** The
  full shape is `<|im_start|>system\n…<|im_end|>\n<|im_start|>user\n…`
  `<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{skeleton}`
  `<|im_end|>\n`; dropping those six markers costs 66 bytes / 6 tokens and
  shifts every token position. The backend therefore clears the C-side chat
  template explicitly (`set_chat_template("", "", "")`) — the official demo
  never calls `set_chat_template`, so its template is empty.
* **The pre-tokenizer regex follows `tokenizer.ggml.pre`.** This GGUF is
  `qwen35`, whose regex is `[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+…`; ByteLevel's
  built-in GPT-2 regex cuts `e.g.` into four tokens instead of three, so
  every token id after it shifts (248 vs 251 tokens for the text task).
  `wdnai.gguf_tokenizer` now dispatches on `pre` (`qwen2` / `qwen35`).
* **A `noul` field's criteria *overwrite* the default descriptions.** The
  task JSON says 「没有人 / 有人」; the reference does
  `choices["no"] = value` (assign, position preserved) — a setdefault keeps
  the English placeholder in the prompt instead.
* **Without text the state falls back** to 「请根据图片回答问题。」 (the
  reference does `row["state"] = text[0] ? text : …`). Otherwise the prompt
  says `null` and the image decision drifts.
* **One prefill per field, each on a *fresh* session.** The model has
  linear-attention layers whose recurrent state survives inside a
  session; `keep_history=0` clears the KV cache but *not* that state.
  The C++ reference creates and destroys a session per field — the
  SDK mirrors that with `RKNN3Lite.recreate_llm_session()`.
* **The control tokens must be single tokens.** `<decision>` is
  248077, `<think>` is 248068, `</think>` is 248069. The ByteLevel
  pre-tokenizer splits them, so they are registered as specials **on
  this model's tokenizer instance only** — the Qwen path needs the
  literal `<think>` text back in its decoded output, so the global
  tokenizer is left alone. When adding them, **merge** with the existing
  `additional_special_tokens`: `add_special_tokens({"additional_special_tokens": …})`
  *replaces* the list, which would drop Qwen's `<|vision_start|>` /
  `<|image_pad|>` markers and leave the C library unable to locate the
  image placeholder region (the multimodal prompt then loses 6 tokens).
* **Vision preprocessing is a plain stretch-resize** to 384×384 (not
  letterbox); the normalisation is baked into the model, so only raw
  RGB bytes are fed in. The resize runs on the **RGA** (`wdnai._rga`,
  ctypes → `librga`) exactly like the reference's `convert_image`: RGA's
  interpolation is not OpenCV's `INTER_LINEAR` (up to 136 per pixel).
* **The probability calibration is float32 all the way down** (the
  reference uses C `float`); computing it in double shifts the last
  digit by ~1e-8. `wdnai.decision.calibrate_probabilities` is float32
  and matches the official numbers bit for bit.

**Parity with the official C++ demo.** `rknn3_model_zoo`'s
`examples/Intern_Decision/cpp/main.cpp` was compiled on the board (the
upstream demo is x86_64-only) and run against the same task JSON and the
same sentence. Prompt bytes (1063 / 2861), token id sequences (251 / 374),
raw logits, calibrated probabilities, the score and the calibration
temperature all match **bit for bit** — for the text task and for the
image task alike (the image task matches exactly when the official's own
decoded pixels are used; feeding the SDK's own OpenCV/libjpeg decode
leaves a ≤0.03pp JPEG-decoder difference and the same decisions).
  RGB bytes are fed in.

**One card holds both models.** Measured on a single RK1828: the text
model (2013 MB weights + 44 MB internal + 297 MB KV cache) and the
vision encoder (651 MB weights + 65 MB internal) are resident
*simultaneously*, using 383 MB of each ~640 MB NPU node. The vision
encoder is therefore loaded lazily (only for requests that carry an
image) and released again once the images are encoded — which is what
the C++ reference does. `spec.options["keep_vision_resident"] = True`
keeps it resident instead.
