[English](README.md) | [简体中文](README_CN.md)
# wdnai

**LLM / VLM / TTS / ASR / Embedding** access SDK for RK chips (RK3588 + PCIe AI
coprocessor RK1828) — a direct **ctypes binding** to `librknn3_api.so`, with no
dependency on the `rknn3-toolkit-lite` Python wheel.

- **Pure Python + ctypes**: binds only the part of the RKNN3 ABI the SDK actually
  uses; `import wdnai` does not open the `.so` (the library is lazily loaded, which
  keeps it from competing with the compute worker for the device).
- **Capability vocabulary lives in the SDK**: `swcap_*` is the single authoritative
  definition for the whole repo (see `wdnai.capability`).
- **Telemetry from the same source**: `wdnai.RknnSmi` is an in-process wrapper around
  the C ABI behind the `rknn-smi` CLI, so high-frequency polling needs no forked
  child process.

> This package is **not on PyPI** — install it editable from GitHub, as described under
> "Installation" below.

---

## Installation

Prerequisites: the device needs the Rockchip runtime — `/usr/lib/librknn3_api.so`
(inference) and `/usr/lib/librknnsmi.so` (telemetry). When neither is present,
`import wdnai` is still safe; it only fails when the corresponding capability is
actually used.

```bash
# 从 GitHub 装 editable（推荐）
pip install -e "git+ssh://git@github.com/rockchip-wardner/wdnai.git#egg=wdnai"

# 或者先 clone 到本地再装
git clone ssh://git@github.com/rockchip-wardner/wdnai.git
pip install -e ./wdnai
```

Python **>= 3.9** is required. Runtime dependencies: `numpy`, `transformers`,
`tokenizers`, `pydantic`, `opencv-python`, `gguf`.

> **Note**: `transformers` / `tokenizers` / `gguf` are **required for model loading**,
> not optional transitive dependencies — RKNN models ship their own
> `*.tokenizer.gguf`, and loading needs them to parse the tokenizer. If you install
> this package with `pip install --no-deps`, add these packages yourself; otherwise
> model loading fails with `ModuleNotFoundError: No module named 'tokenizers'`.

Extra dependencies for development/testing: `pip install -e ".[dev]"` (pytest + build).

---

## Quick start

### Text chat

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

> `LLMSession(model=...)` is the **old form** (it now raises a `DeprecationWarning`;
> internally it returns an `LLMLocalSession`). Use `LLMLocalSession` directly for
> in-process inference; for cross-process / cross-machine work over DDS RPC use
> `wdndds.LLMRPCSession` — the two have the same interface.

### Vision: images / video frames

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

- Video needs **no** special type: sample the frames yourself and hand the frame list
  to `ImageAttachment([f0, f1, ...])`.
- When the directory holds both a 384×384 and a 768×768 vision model, the SDK loads
  **both** and automatically picks the one closest to the longest edge of the input
  image (naming conventions see `src/wdnai/models/README.md`).

### Structured output (`generating=`)

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

The schema of `generating=` is assembled into the prompt by the SDK; the return value
is validated by pydantic, and when validation fails it **does not raise** — instead
you get the model's raw text back (see the `else` branch above).

### Tool calls (`tools=`)

```python
from wdnai import Callable

@Callable
def alarm(parm: Event):
    """发现异常行为时触发报警"""
    print("!!! 报警 !!!", parm)

await session.respond(prompt="检查危险行为", attachment=[img], generating=Event, tools=[alarm])
```

Functions decorated with `@Callable` are registered as "tools the model can call";
the `@Generable` schema is their parameter type.

### Streaming / multi-turn / misc

```python
await session.respond(prompt="...", on_delta=lambda chunk: print(chunk, end=""))
await session.respond(prompt="...", history=[{"role": "user", "content": "上一轮"}])

session.reset()          # 清 C 层 KV cache（多轮之间记得调）
session.stop()           # 取消正在跑的推理
await session.majority_vote(k=3, prompt="...")   # 多次采样投票
await session.embed_text("要检索的文本")          # embedding 模型
```

---

### Text-to-image (Qwen-Image-2.1)

Models with `kind="image"` go through `wdnai.backends.image.ImageBackend`: text →
512×512 image. **The output is an image, not text**, so use an entry point that
accepts attachments to get the picture.

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

⚠️ When `respond()` is called **without** `generating=` it returns text only and the
image is dropped — that is the contract of `respond` (it promises the caller a
**text** result); use one of the three ways above when you want the image.

⭐⭐⭐ **The runtime is wdnai's own C shim** (`src/wdnai/_cshim/libqi_qwen.so`,
**source shipped with the package**) — not an external binary, and not a closed-source
Rockchip artifact. It links only against the **system-provided** `librknn3_api.so` ⇒
the model package holds **only the model**, with no `.so` / executable of any kind.
⭐ Even the "fixed initial noise" is **generated on the fly** by the cshim
(numpy-compatible PCG64 + ziggurat, **byte-for-byte identical** to the historical
baseline) ⇒ the package does not even have to carry a data file.

```bash
# cshim 在 pip install 时自动编译；也可手动：
cd src/wdnai/_cshim && make          # 或 bash tools/build_cshim.sh
```

⚠️ Compiling requires a C++ compiler (`/usr/bin/g++` is already present on the
board); when the compiler is missing the install does not fail, only text-to-image is
unavailable (the other backends are unaffected).

Runtime parameters (can be written into `spec.options` or passed per call):

| Parameter | Default | Description |
|---|---|---|
| `steps` | `8` | flow-match step count. 8 steps ≈ 94 s; 20 steps is sharper ≈ 125 s; 40 steps = upstream default |
| `fm_mu` | `0.5387096774` | scheduling coefficient for the 512 route (**must not** use the 1024 route's 0.6935483871) |
| `noise_fresh` | `True` | draw a fresh random initial noise each time; `False` ⇒ fixed noise (reproducible / comparable against a golden reference) |
| `keep_artifacts` | `False` | keep intermediate artifacts (latent variables / raw VAE tensors) |
| `work_dir` | temp dir | working directory to use (artifacts land here) |

⏱ Measured on the board (RK3588 + 2×RK1828, 8 steps): **both TE and DiT stay
resident** (card 0 = TE + head, card 1 = DiT 4 segments → VAE) ⇒ one-time init ≈ 78 s,
after that **≈ 34 s per image** (flow-match 21 s + VAE 11 s, TE encoding only 0.16 s).

Command-line equivalent (you can generate images without writing Python):

```bash
.venv/bin/python .venv/src/wdnai/examples/qwen_image_21.py \
    "a cute cat sitting on a wooden chair" 8 --out /tmp/qi21

# ⭐ 批量："|" 分隔多个提示词 ⇒ 一个进程跑完 N 张
.venv/bin/python .venv/src/wdnai/examples/qwen_image_21.py \
    "a cute cat sitting on a wooden chair|一只戴眼镜的猫" 8 --out /tmp/qi21
```

---

## Capability vocabulary `SoftwareCapability`

`swcap_*` describes "**what this model instance can actually do on this machine**",
not "which business uses it". The definitions live in `wdnai.capability`, and both
manifest validation and runtime session lookup use them.

| Capability string | Meaning |
|---|---|
| `swcap_text_chat` | text → text (multi-turn + streaming) |
| `swcap_vision_chat` | image → text (description / detection description) |
| `swcap_vision_grounding` | image → 0–1000 normalized bbox + text (requires LocateAnything) |
| `swcap_vision_embedding` | image/video → retrieval vector |
| `swcap_audio_chat` | audio → text |
| `swcap_text_to_speech` | text → speech |
| `swcap_text_to_image` | text → image (Qwen-Image-2.1, 512×512) |
| `swcap_tool_call` | function / tool calling |

```python
from wdnai import SoftwareCapability, migrate_legacy

SoftwareCapability.VISION_CHAT.value        # 'swcap_vision_chat'
migrate_legacy("video_describe")            # → (SoftwareCapability.VISION_CHAT, True)
migrate_legacy("swcap_not_a_thing")         # → ValueError
```

`migrate_legacy()` is responsible for **reading old configuration**: both the bare
legacy names (`video_describe`) and the previous generation of `swcap_*` names can be
migrated; retired capabilities (`video_query` / `video_analyse`) raise
`RetiredCapabilityError` instead of being silently mapped to a name nobody declares.

---

## Telemetry `RknnSmi`

```python
from wdnai import RknnSmi

smi = RknnSmi()
for dev in smi.get_device_list():
    print(dev["index"], dev["mode"], smi.get_device_temperature(dev["index"]))
```

- Device list / model / PCIe BDF / temperature / power / voltage / current / NPU
  utilization / memory / health status / version / reset.
- The temperature unit is **°C**. The device-level temperature command is not
  answered on some EP firmwares (RK1828), so `get_device_temperature()` **falls back
  to chip level automatically** and remembers it, no longer issuing the failing PCIe
  command — callers need not care about the two-level difference.
- Root is required to take the flock on `/tmp/rknnsmi.lock` (starting another
  `rknn-smi` CLI at the same moment therefore fails to init).

---

## ModelSpec quick reference

One `ModelSpec` = **all the static configuration of one model instance**.

| Field | Description |
|---|---|
| `name` | model name, e.g. `qwen3-vl-4b` |
| `kind` | `llm` / `vlm` / `tts` / `omni` / `asr` / `embedding` |
| `variant` | weight family: `qwen3` / `qwen3-vl` / `qwen2.5-omni` / `qwen3-tts` / … |
| `rknn_path` / `weight_path` | the RKNN and weight files of an LLM |
| `rknn_llm_path` / `rknn_vision_path` | the two RKNNS of a VLM (+ their respective `weight_*`) |
| `tts_model_dir` / `omni_model_dir` / `asr_model_dir` | directory-organized models (TTS / Omni / ASR) |
| `embed_path` / `tokenizer_path` | embedding vector file / tokenizer file |
| `vocab_size` | vocabulary size (Qwen3.5-VL is **248320**, do not copy Qwen3's 151936) |
| `special_eos_id` | special EOS id (Qwen3.5-VL is **248046**) |
| `max_context_len` | context length |
| `image_size` | vision input edge length (384 / 576 / 768 …) |
| `use_prune` | whether pruning is enabled |
| `target` / `core_mask` | `rk1820` / `rk1828`; `core_mask` selects the NPU cores |
| `verbose` | runtime logging |
| `options` | sampling parameters (`temperature` / `top_p` / `top_k` / `repeat_penalty` …), merged into `init_runtime` |

Change paths / change sampling:

```python
spec = SystemLanguageModel.qwen3_vl_4b().with_paths(rknn_llm_path="/tmp/x.rknn")
spec = spec.with_options(temperature=0.2, top_p=0.9)
```

### Ready-made presets

`SystemLanguageModel.<preset>()`:

| preset | model directory (under `$WDNAI_MODELS_DIR`) | capability |
|---|---|---|
| `qwen2_5_0_5b` / `qwen3_1_7b` / `qwen3_4b` | `Qwen3-4B/` etc. | text chat |
| `qwen2_5_vl_3b` / `qwen2_5_vl_7b` | `Qwen2.5-VL-*` | vision chat |
| `qwen3_vl_2b` / `qwen3_vl_4b` / `qwen3_vl_4b_template` | `Qwen3-VL-4B/` | vision chat (the template variant binds no default path) |
| `qwen3_5_vl_4b` | `Qwen3.5-4B/` | vision chat + tool calling |
| `qwen_2_5_omni_3b` | `Qwen2.5-Omni-3B/` | audio → text |
| `qwen3_tts` | `Qwen3-TTS/` | text → speech |
| `qwen3_asr` | `Qwen3-ASR/` | speech → text |
| `qwen3_vl_embedding` | `Qwen3-VL-Embedding-2B/` | retrieval vector |
| `locate_anything` | `LocateAnything-3B/` | object grounding (bbox) |
| `qwen_image_21_512` | `Qwen-Image-2.1/` | text-to-image (512×512) |

A preset checks that all files are present **at construction time** and raises
`FileNotFoundError` listing what is missing — it will not blow up only at inference
time.

---

## Where model files go

Default root `/userdata/models/` (changeable via `WDNAI_MODELS_DIR`):

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

Naming conventions and the complete per-model file tables are in
[`src/wdnai/models/README.md`](src/wdnai/models/README.md).

### Environment variables

| Variable | Purpose |
|---|---|
| `WDNAI_MODELS_DIR` | model root directory (default `/userdata/models`) |
| `WDNAI_QWEN3_VL_4B_DIR` / `WDNAI_QWEN3_4B_DIR` / `WDNAI_QWEN3_5_4B_VL_DIR` / `WDNAI_QWEN_2_5_OMNI_3B_DIR` / `WDNAI_LOCATE_ANYTHING_DIR` | per-model directory override |
| `WDNAI_QWEN_IMAGE_21_DIR` / `QI_MODEL_DIR` | Qwen-Image-2.1 model package directory (default `$WDNAI_MODELS_DIR/Qwen-Image-2.1`; on the board it is actually at `/home/rockchip/models/Qwen-Image-2.1`) |
| `QI_SPREAD_IO` / `QI_SPREAD_IO_IN` | cshim internal defaults `0` / `1` (output IO must never be spread) |
| `WDNAI_LIBRKNN3_API` | path to `librknn3_api.so` |
| `WDNAI_LIBRKNNSMI` | path to `librknnsmi.so` |
| `WDNAI_TRACE_PROMPT` | print the prompt actually sent on each inference (for debugging prompt contamination) |

---

## Known limitations

- **One model handle is not re-entrant**: do not start a second `respond()` before the
  first one finishes; the SDK serializes them internally with a single worker
  thread, but across processes the caller has to guarantee that.
- **Memory is a hard constraint**: one RK1828 has 5 GB, which cannot hold two 4B
  VLMs. Once a model stays resident, do not start a second one on the same card.
- **Version differences in `Enum.__contains__`**: `migrate_legacy()` was changed to
  "look up by value", with consistent semantics on Python 3.9 ~ 3.12 (3.11's
  `Enum.__contains__` raises TypeError for non-enum members, so
  `value in SoftwareCapability` cannot be used to test membership).
- **Chip-level and device-level temperatures use different units** (whole degrees vs.
  millidegrees); the SDK converts each internally, and both are exposed as °C.

---

## Repository map

| Path | Contents |
|---|---|
| `src/wdnai/session.py` | `LLMSession` abstract interface + in-process `LLMLocalSession` implementation |
| `src/wdnai/model.py` | `ModelSpec` + the `SystemLanguageModel` preset catalog |
| `src/wdnai/capability.py` | the `swcap_*` vocabulary + migration table |
| `src/wdnai/attachment.py` | `TextAttachment` / `ImageAttachment` / `AudioAttachment` |
| `src/wdnai/_generable.py` / `_callable.py` | decorators for structured output and tool calling |
| `src/wdnai/_librknn.py` | ctypes bindings for librknn3 |
| `src/wdnai/_librknnsmi.py` | ctypes bindings for librknnsmi + `RknnSmi` |
| `src/wdnai/grounding.py` | the bbox schema of LocateAnything |
| `examples/` | runnable samples (image detection, video license plates, function calls, TTS …) |
| `tests/` | pytest (does not occupy the hardware) |
| [`AGENTS.md`](AGENTS.md) | detailed internal conventions for AI / maintainers (the gotcha records live there) |

## License

Apache-2.0. See [LICENSE](LICENSE).
