Model files live at a fixed on-device location: `/userdata/models/<model-name>/`.
Override the root with the `RKAI_MODELS_DIR` environment variable.

No network access is performed at any point; place the model files manually.


# Tree structure

```
/userdata/models/
├── Qwen3-VL-4B/
│   ├── llm_Qwen3-VL-4B.1828.rknn       # LLM RKNN (25 MB)
│   ├── llm_Qwen3-VL-4B.weight          # LLM weights (2.3 GB)
│   ├── vision_Qwen3-VL-4B.1828.rknn    # Vision encoder RKNN (4.2 MB)
│   ├── vision_Qwen3-VL-4B.weight       # Vision weights (234 MB)
│   ├── Qwen3-VL-4B.embed.bin           # Token embedding table (742 MB, vocab 151936 × 2560-dim float16)
│   └── Qwen3-VL-4B.tokenizer.gguf      # GGUF-format tokenizer (5.7 MB; loaded via rkai.gguf_tokenizer)
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
add a preset + path helper in [rkai/model.py](../model.py).

### Qwen3-TTS (text-to-speech)

The full Qwen3-TTS pipeline runs four RKNN3 models on the same RK1828
PCIe card as the VLM. Used by pyrkaiweb's `#page-tts` (the
Qwen3-TTS model itself is loaded via `TTSBackend` in `rkai/backends/tts.py`,
which wraps the in-tree `Qwen3TTSDriver` vendored into the same
`rkai/backends/tts.py` module). The driver used to live in
`rknn3-toolkit-lite/examples/Qwen3-TTS/qwen3_tts_demo.py` and was
imported at runtime via `sys.path` injection; it has since been
ported into the SDK so `import rkai.backends.tts` no longer needs
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
`EmbeddingBackend` in `rkai/backends/embedding.py`
(`SystemLanguageModel.qwen3_vl_embedding`).

| File (`<root>/Qwen3-VL-Embedding-2B/…`)     | Purpose                              |
| ------------------------------------------- | ------------------------------------ |
| `llm_Qwen3-VL-Embedding-2B.1828.rknn` + `.weight`   | Embedding LLM (plain RKNN3Lite path) |
| `vision_Qwen3-VL-Embedding-2B.1828.rknn` + `.weight`| Vision encoder (input 576×576)       |
| `Qwen3-VL-Embedding-2B.embed.bin`           | Token embedding table (151936 × 2048) |
| `Qwen3-VL-Embedding-2B.tokenizer.gguf`      | GPT-2 BPE + ByteLevel tokenizer      |

`max_context_len` defaults to 2048 tokens; the vision feature is fed
at 576×576 (matching the model's native input) and each video chunk
contributes `FRAME_TOKENS` vision tokens (see
`rkai/backends/embedding.py`).
