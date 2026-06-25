Download the model files and weights from here, then place them in the current directory.

https://meta.box.lenovo.com/v/link/view/347a716ec0b84e0799675a3f45dfdc23


# Tree Structor

```
models/
├── README.md
└── Qwen3-VL-4B/
    ├── llm_Qwen3-VL-4B.rknn           # LLM RKNN (25 MB)
    ├── llm_Qwen3-VL-4B.weight         # LLM weights (2.3 GB)
    ├── vision_Qwen3-VL-4B.rknn        # Vision encoder RKNN (4.2 MB)
    ├── vision_Qwen3-VL-4B.weight      # Vision weights (234 MB)
    ├── Qwen3-VL-4B.embed.bin          # Token embedding table (742 MB, vocab 151936 × 2560-dim float16)
    └── Qwen3-VL-4B.tokenizer.gguf     # GGUF-format tokenizer (5.7 MB; loaded via rksdk.gguf_tokenizer)
```

The 4B model is the **full** (non-pruned) vision export. The `Qwen3-VL-4B` directory is what `SystemLanguageModel.qwen3_vl_4b` points to by default (absolute paths, resolved from `rksdk/__init__.py`'s file location).

To add a new model variant:

```
models/
└── <new-variant>/
    ├── llm_<...>.rknn
    ├── llm_<...>.weight
    ├── vision_<...>.rknn      # only for VLM
    ├── vision_<...>.weight
    ├── <...>.embed.bin
    └── <...>.tokenizer.gguf   # or a HF tokenizer directory
```

Then add a preset in [rksdk/model.py](../model.py) and a `_default_<variant>_paths()` helper.
