Model files live at a fixed on-device location: `/userdata/models/<model-name>/`.
Override the root with the `RKAI_MODELS_DIR` environment variable.

No network access is performed at any point; place the model files manually.


# Tree structure

```
/userdata/models/
└── Qwen3-VL-4B/
    ├── llm_Qwen3-VL-4B.1828.rknn      # LLM RKNN (25 MB)
    ├── llm_Qwen3-VL-4B.weight         # LLM weights (2.3 GB)
    ├── vision_Qwen3-VL-4B.1828.rknn   # Vision encoder RKNN (4.2 MB)
    ├── vision_Qwen3-VL-4B.weight      # Vision weights (234 MB)
    ├── Qwen3-VL-4B.embed.bin          # Token embedding table (742 MB, vocab 151936 × 2560-dim float16)
    └── Qwen3-VL-4B.tokenizer.gguf     # GGUF-format tokenizer (5.7 MB; loaded via rkai.gguf_tokenizer)
```

The 4B model is the **full** (non-pruned) vision export. `SystemLanguageModel.qwen3_vl_4b` points to
`/userdata/models/Qwen3-VL-4B/` by default and raises `FileNotFoundError` listing the missing
files if the directory is incomplete. To add a new model variant, place another sibling directory
under `/userdata/models/<new-variant>/` and add a preset + path helper in [rkai/model.py](../model.py).