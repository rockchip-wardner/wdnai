"""Model configuration and the preset catalog.

Each preset is a :class:`ModelSpec` dataclass that holds all the configuration
needed for a given variant (paths, tokenizer, sampling parameters, vision
parameters, etc.). ``SystemLanguageModel`` is the catalog entry point, exposing
``qwen2_5_*`` / ``qwen3_*`` / ``qwen2_5_vl_*`` / ``qwen3_vl_*`` factories and the
``default`` placeholder.

Typical usage::

    from wdnai import SystemLanguageModel, LLMSession, ImageAttachment

    model = SystemLanguageModel.qwen3_vl_2b().with_paths(
        rknn_llm_path="/userdata/Qwen3-VL-2B-llm.rknn3",
        rknn_vision_path="/userdata/Qwen3-VL-2B-vision.rknn3",
        tokenizer_path="/userdata/qwen3_VL_2B",
        embed_path="/userdata/Qwen3-VL-2B-llm.embed.bin",
    )
    session = LLMSession(model=model)
    out = await session.respond(msg="describe the image", attachment=ImageAttachment(img))
"""
from __future__ import annotations

import os as _os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional


# --------------------------------------------------------------------------- #
# Base spec
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ChatTemplate:
    """The template the **C session** wraps every prompt with.

    The C side composes the final token sequence as::

        system + prefix + <prompt text> + postfix

    ChatML (the default for the Qwen family) is::

        <|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n
        + <|im_start|>user\\n + <prompt> + <|im_end|>\\n<|im_start|>assistant\\n

    A model family with its **own** control tokens supplies a different
    triple — see :data:`HY_MT_CHAT_TEMPLATE` for the Hunyuan-Dense one.

    Frozen: a template is a constant of the model family, not something a
    caller mutates per call. (The *session* is mutable — that is what
    ``set_thinking_mode`` flips — but the template itself is not.)
    """

    system: str
    prefix: str
    postfix: str


@dataclass
class ModelSpec:
    """All static configuration + dynamic paths needed for a single model."""

    name: str
    kind: str                                # "llm" | "vlm" | "tts" | "omni" | "asr" | "embedding" | "grounding" | "image" | "decision"
    variant: str                             # "qwen2.5" | "qwen3" | "qwen2.5-vl" | "qwen3-vl" | "qwen2.5-omni" | "qwen3-tts" | "qwen3-asr" | "qwen3-vl-embedding" | "qwen-image-2.1" | "intern-decision"

    # ---- LLM / VLM / TTS / Omni / ASR resources ----
    # kind=llm uses rknn_path / weight_path; kind=vlm uses rknn_llm_path / rknn_vision_path;
    # kind=tts uses tts_model_dir; kind=omni uses omni_model_dir (audio + llm derived from it);
    # kind=asr uses asr_model_dir (encoder + llm derived from it);
    # kind=image uses image_model_dir (a whole multi-part model package:
    # TE + DiT segments + VAE segments + the C++ runtime).
    rknn_path: str = ""
    weight_path: str = ""
    rknn_llm_path: str = ""
    weight_llm_path: str = ""
    rknn_vision_path: str = ""
    weight_vision_path: str = ""
    tts_model_dir: str = ""
    omni_model_dir: str = ""
    asr_model_dir: str = ""
    image_model_dir: str = ""
    embed_path: str = ""
    tokenizer_path: str = ""

    # ---- Depth Anything V3: three-stage (local → global → head) export ----
    # DA3 is not one RKNN but three: ``local`` turns每张图 into tokens,
    # ``global`` fuses the concatenated multi-view tokens, ``head`` turns
    # the fused features into the per-view depth + confidence maps. Each
    # stage has its own ``.rknn`` + ``.weight`` pair; the weight path is
    # derived from the rknn path (``.rknn`` → ``.weight``), exactly like
    # the C++ demo's ``make_weight_path``.
    depth_local_path: str = ""
    depth_global_path: str = ""
    depth_head_path: str = ""

    # ---- Multi-stage / multi-card exports (Qwen3.5-VL-9B) ----
    # ``stage_count > 1`` means the LLM is split across that many RKNN
    # files, one per card. ``llm_seg_paths`` / ``weight_seg_paths`` hold
    # the per-stage paths in stage order; ``rknn_llm_path`` /
    # ``weight_llm_path`` remain the stage-0 aliases so single-stage code
    # and probes keep working. ``requires_cards`` is consumed by the
    # compute layer; the backend itself only needs the device-id list.
    stage_count: int = 1
    llm_seg_paths: List[str] = field(default_factory=list)
    weight_seg_paths: List[str] = field(default_factory=list)
    requires_cards: int = 1

    # ---- Model hyperparameters ----
    vocab_size: int = 151936
    special_eos_id: int = 151645
    max_context_len: int = 1024
    image_size: int = 384
    use_prune: bool = True

    # ---- runtime ----
    target: str = "rk1820"
    core_mask: int = 0xFF
    verbose: bool = True

    # ---- Sampling parameters: merged into llm_args[0] passed to rknn.init_runtime ----
    options: Dict[str, Any] = field(default_factory=dict)

    # ---- Chat template (C side) -----------------------------------------
    #: The template the **C session** wraps every prompt with. ``None``
    #: means ChatML — the default every Qwen-family preset relies on.
    #:
    #: The C side composes ``system + prefix + <prompt> + postfix``, so a
    #: non-ChatML model family (HY-MT uses the Hunyuan ``<｜hy_User｜>``
    #: control tokens) only needs its own triple here — the *content* of
    #: the prompt is untouched.
    #:
    #: Models that build the **whole** transcript themselves (the 9B
    #: multi-card backend) leave this ``None`` **and** clear the C-side
    #: template themselves (``set_chat_template("", "", "")``) — see
    #: ``wdnai/backends/vlm_multicard.py``.
    chat_template: Optional["ChatTemplate"] = None

    # ------------------------------------------------------------------ #
    def with_paths(self, **paths: str) -> "ModelSpec":
        """Return a new instance with selected path fields overridden.

        Supported fields: ``rknn_path`` / ``weight_path`` / ``embed_path`` /
        ``tokenizer_path`` / ``rknn_llm_path`` / ``rknn_vision_path`` / etc.
        """
        return replace(self, **paths)

    def with_options(self, **opts: Any) -> "ModelSpec":
        """Return a new instance with overridden inference parameters (temperature, top_p, ...)."""
        merged = {**self.options, **opts}
        return replace(self, options=merged)


# --------------------------------------------------------------------------- #
# Preset factories
# --------------------------------------------------------------------------- #
def _qwen2_5_spec(**overrides) -> ModelSpec:
    base = dict(
        name="qwen2.5-0.5b-instruct",
        kind="llm",
        variant="qwen2.5",
        vocab_size=151936,
        special_eos_id=151645,
        max_context_len=1024,
        options={
            "max_new_tokens": 512,
            "top_k": 1,
            "top_p": 0.8,
            "temperature": 0.8,
            "repeat_penalty": 1.1,
        },
    )
    base.update(overrides)
    return ModelSpec(**base)


def _qwen3_spec(**overrides) -> ModelSpec:
    base = dict(
        name="qwen3-1.7b",
        kind="llm",
        variant="qwen3",
        vocab_size=151936,
        special_eos_id=151645,
        max_context_len=1024,
        options={
            "max_new_tokens": 512,
            "top_k": 1,
            "top_p": 0.9,
            "temperature": 0.7,
            "repeat_penalty": 1.0,
            "enable_thinking": False,   # Qwen3 thinking-mode toggle (passed to RKNN3Lite.session_run)
        },
    )
    base.update(overrides)
    return ModelSpec(**base)


def _qwen3_4b_spec(**overrides) -> ModelSpec:
    base = dict(name="qwen3-4b", max_context_len=4096)
    base.update(overrides)
    return _qwen3_spec(**base)


# --------------------------------------------------------------------------- #
# HY-MT1.5 (Hunyuan-Dense translation model)
# --------------------------------------------------------------------------- #
#: The Hunyuan control-token triple, i.e. the three slots the C session
#: wraps a prompt with. **Not** ChatML — HY-MT speaks the Hunyuan dialect:
#:
#:     <｜hy_begin▁of▁sentence｜>            ← BOS (the "system" slot)
#:     <｜hy_User｜>{待译原文}<｜hy_Assistant｜> ← user turn + generation prompt
#:
#: The three markers are **real vocabulary entries** (ids 120000 / 120006 /
#: 120007) — if they get byte-level split into ~7 regular tokens the model
#: never sees the boundaries and the output degrades into a continuation of
#: the source text instead of a translation.
#:
#: The triple mirrors the model's own jinja template (shipped inside the
#: RKNN *and* in the package's ``.config.pkl``) one-to-one:
#:
#:     {% if messages[0]['role'] == 'system' %}
#:       <｜hy_begin▁of▁sentence｜>{{ system }}<｜hy_place▁holder▁no▁3｜>
#:     {% else %}
#:       <｜hy_begin▁of▁sentence｜>
#:     {% endif %}
#:
#: → the BOS is the **system** slot (it precedes any system text), the user
#: marker is the **prefix**, the assistant marker is the **postfix**.
HY_MT_CHAT_TEMPLATE = ChatTemplate(
    system="<｜hy_begin▁of▁sentence｜>",
    prefix="<｜hy_User｜>",
    postfix="<｜hy_Assistant｜>",
)


def _hy_mt_1_5_spec(**overrides) -> ModelSpec:
    """HY-MT1.5-1.8B: Hunyuan-Dense machine-translation model (text → text).

    Sampling is **greedy** (``top_k=1`` + ``temperature=1e-6``): a
    translation model must reproduce the source faithfully, and any
    sampling noise shows up as dropped or duplicated clauses in the
    译文. This is the same reasoning as ``_locate_anything_spec``
    (structured output degrades above ~1e-6).

    ``max_context_len=8192`` is the value baked into the export (the
    RKNN's embedded ``hf_config_json`` says
    ``max_position_embeddings=8192``) — the upstream checkpoint's
    262144 is a *training* limit the export did not keep.

    The chat template is the Hunyuan one, **not** ChatML — see
    :data:`HY_MT_CHAT_TEMPLATE`.
    """
    base = dict(
        name="HY-MT1.5-1.8B",
        kind="llm",
        variant="hy-mt",
        vocab_size=120818,
        special_eos_id=120020,
        max_context_len=8192,
        target="rk1820",
        core_mask=0xFF,
        chat_template=HY_MT_CHAT_TEMPLATE,
        options={
            "max_new_tokens": 512,
            "top_k": 1,
            "top_p": 0.9,
            "temperature": 1e-6,
            "repeat_penalty": 1.0,
        },
    )
    base.update(overrides)
    return ModelSpec(**base)


def _qwen2_5vl_spec(**overrides) -> ModelSpec:
    base = dict(
        name="qwen2.5-vl-3b",
        kind="vlm",
        variant="qwen2.5-vl",
        vocab_size=151936,
        special_eos_id=151645,
        max_context_len=1024,
        image_size=392,
        use_prune=True,
        options={
            "max_new_tokens": 1024,
            "top_k": 1,
            "top_p": 0.9,
            "temperature": 1e-6,   # very low temperature from the example scripts
            "repeat_penalty": 1.05,
        },
    )
    base.update(overrides)
    return ModelSpec(**base)


def _qwen2_5vl_7b_spec(**overrides) -> ModelSpec:
    """Qwen2.5-VL 7B, slightly larger vocabulary (152064)."""
    base = dict(
        name="qwen2.5-vl-7b",
        kind="vlm",
        variant="qwen2.5-vl",
        vocab_size=152064,
        special_eos_id=151645,
        max_context_len=1024,
        image_size=392,
        use_prune=True,
        options={
            "max_new_tokens": 1024,
            "top_k": 1,
            "top_p": 0.9,
            "temperature": 1e-6,
            "repeat_penalty": 1.05,
        },
    )
    base.update(overrides)
    return ModelSpec(**base)


def _qwen3vl_spec(**overrides) -> ModelSpec:
    base = dict(
        name="qwen3-vl-2b",
        kind="vlm",
        variant="qwen3-vl",
        vocab_size=151936,
        special_eos_id=151645,
        max_context_len=1024,
        image_size=384,
        use_prune=True,
        options={
            "max_new_tokens": 1024,
            "top_k": 1,
            "top_p": 0.9,
            "temperature": 1e-6,   # VLM tasks hallucinate easily at higher temperature; default to near-greedy
            "repeat_penalty": 1.0,
            "enable_thinking": False,
        },
    )
    base.update(overrides)
    return ModelSpec(**base)


def _qwen3vl_4b_spec(**overrides) -> ModelSpec:
    """Qwen3-VL-4B: 4B-parameter variant, 2560-dim embeddings, 262144 context, .gguf tokenizer.

    The currently-shipped 4B export is the **full** vision model (NHWC input), not
    the pruned one. If you get a pruned export, override with
    ``.with_options(use_prune=True)``.
    """
    base = dict(
        name="qwen3-vl-4b",
        kind="vlm",
        variant="qwen3-vl",
        vocab_size=151936,
        special_eos_id=151645,
        max_context_len=4096,         # GGUF header says 262144, but device memory caps it at 4096
        image_size=384,
        use_prune=False,              # 4B full model
        options={
            "max_new_tokens": 1024,
            "top_k": 1,
            "top_p": 0.9,
            "temperature": 1e-6,   # 4B default is greedy to avoid hallucinations after 384x384 downsampling
            "repeat_penalty": 1.0,
            "enable_thinking": False,
        },
    )
    base.update(overrides)
    return ModelSpec(**base)


def _qwen3_5_vl_spec(**overrides) -> ModelSpec:
    """Qwen3.5-VL-4B multimodal spec (image-in, text-out).

    Also handles pure-text inference (no image attachment) — the
    VLM-mode LLM is a regular LLM with MRoPE (``mrope_type='Qwen3.5'``,
    ``mrope_section=[11, 11, 10]``) baked into the graph for image
    position encoding; with no vision tokens attached MRoPE reduces
    to standard positional encoding, so this single RKNN powers both
    the video-description and text-chat capabilities in
    ``MODEL_PRESETS["qwen3-5-vl-4b"]``.

    File layout::

        Qwen3.5-4B-llm.1828.rknn + .weight                   (VLM-mode LLM, MRoPE baked in)
        Qwen3.5-4B-vision.384x384.1828.rknn + .weight        (required, prune-mode vision)
        Qwen3.5-4B-vision.768x768.1828.rknn + .weight        (optional, prune-mode vision)
        Qwen3.5-4B-llm.embed.bin                             fp16 (vocab_size × dim)
        Qwen3.5-4B-llm.tokenizer.gguf                        GPT-2 BPE + ByteLevel

    Sampling parameters mostly follow ``examples/Qwen3_5_VL/cpp/main.cc``
    defaults: ``top_k=1``, ``top_p=0.9``, ``repeat_penalty=1.0``,
    ``frequency_penalty=0.0``, ``presence_penalty=1.5`` (the last is
    non-default and was deliberately tuned for Qwen3.5 in the rknn3
    demo).

    **Notable deviation from the demo**: ``temperature=1e-6`` instead
    of the demo's ``temperature=1.0``. The C++ demo targets free-form
    chat-style generation where temperature=1.0 is desirable for
    variety. But every production caller in this repo uses
    ``session.respond(generating=...)`` for **structured output** — the
    Qwen3-VL team specifically chose greedy (1e-6) for the 4B class to
    avoid bbox / JSON-token hallucination (see ``_qwen3vl_4b_spec``:
    "4B default is greedy to avoid hallucinations after 384x384
    downsampling"). Qwen3.5-VL 4B behaves the same way; empirical
    evidence from the video-target-detection page (which expects 0–1000
    normalized bbox coords) showed temperature=1.0 producing wildly
    wrong / degenerate boxes (~1×0 px), while 1e-6 produces
    well-localized boxes across frames. Callers needing creative
    sampling should override via ``.with_options(temperature=...)``.

    The vision preprocessing path is identical to Qwen3-VL pruned
    mode (``qwen3vl_prune_single`` in :mod:`wdnai.utils`): the
    rknn3 demo's ``normalize_and_to_nchw`` + ``transpose`` produces
    the same 2-D patch-token layout that the C++ code feeds into
    the vision RKNN. The 384x384 export is required because
    :attr:`ModelSpec.image_size` falls back to the smallest loaded
    variant when no other variants are present.
    """
    base = dict(
        name="qwen3.5-4b-vl",
        kind="vlm",
        variant="qwen3.5",
        vocab_size=248320,            # Qwen3.5 vocab (vs Qwen3's 151936)
        special_eos_id=248046,
        max_context_len=4096,        # export kvcache_buffer_len = 8K
        image_size=384,
        use_prune=True,              # pruned vision path matches Qwen3-VL prune layout
        options={
            "max_new_tokens": 1024,
            "top_k": 1,
            "top_p": 0.9,
            # Greedy (1e-6) instead of demo's 1.0: see docstring — every
            # production caller here runs structured-output schemas (e.g.
            # TargetDoc bbox / Event text), where 4B-class exports
            # hallucinate at high temperature. Qwen3-VL-4B has the same
            # fix; this is its Qwen3.5 sibling.
            "temperature": 1e-6,
            "repeat_penalty": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 1.5,   # Qwen3.5 demo default — see main.cc
            "enable_thinking": False,
        },
    )
    base.update(overrides)
    return ModelSpec(**base)


def _qwen3_5_vl_9b_spec(**overrides) -> ModelSpec:
    """Qwen3.5-VL-9B multi-stage / multi-card spec.

    Unlike the 4B export (one LLM RKNN on one card), the 9B export is
    split into ``seg0`` / ``seg1``: each stage lives on its own RK182x
    card and the two are chained by
    :class:`~wdnai.backends.vlm_multicard.MultiCardVLMBackend`.
    Vision stays on stage 0's card. ``requires_cards=2`` is consumed by
    the compute layer (a single-card board cannot run this model).

    File layout::

        Qwen3.5-9B-llm_seg0.1828.rknn + .weight   (stage 0, card 0)
        Qwen3.5-9B-llm_seg1.1828.rknn + .weight   (stage 1, card 1)
        Qwen3.5-9B-vision.1828.rknn    + .weight  (vision, card 0)
        Qwen3.5-9B-llm.embed.bin                  fp16 (vocab × 4096)
        Qwen3.5-9B-llm.tokenizer.gguf             Qwen3.5 BPE

    Sampling follows the 9B example (``temperature=1.0``): this export
    is used for free-form chat, not schema-constrained bbox output.

    ``max_new_tokens`` was ``max_context_len // 8`` per the model owner's
    recommendation (106496 // 8 = 13312). **It is now 1024**, because that
    number is a runaway generator, not a "long answer" allowance:

    * the decode is ``top_k=1`` — greedy — and the vendor pipeline hardcodes
      the sampling params (see ``backends/_qwen35_vl_9b_vendor.py``), so
      when the model degenerates into repeating itself it never emits
      ``special_eos_id`` and the loop only stops at ``max_new_tokens``;
    * measured on this board: the 9B decodes at ~30 token/s, so 13312
      tokens is **~7 minutes** of NPU time — far past the worker's 60 s
      inference budget;
    * a worker that blows its budget does **not** interrupt the inference
      (the thread is a daemon, it is only abandoned), so the NPU stays
      occupied and every later request queues behind the runaway and dies
      with the same timeout. One degenerate generation used to wedge the
      whole card for good.

    1024 tokens is ~34 s at the measured rate — still a generous answer,
    and comfortably inside the 60 s budget. Callers that know better pass
    their own value per request (the text chat uses 2048, the USB-HID
    agent 512); this is only the **fallback** for callers that do not.
    """
    base = dict(
        name="qwen3.5-9b-vl",
        kind="vlm_multicard",
        variant="qwen3.5",
        vocab_size=248320,
        special_eos_id=248046,
        max_context_len=106496,
        image_size=384,
        use_prune=True,
        target="rk1828",
        stage_count=2,
        requires_cards=2,
        options={
            "max_new_tokens": 1024,
            "top_k": 1,
            "top_p": 0.9,
            "temperature": 0.8,
            "repeat_penalty": 1.5,
            "frequency_penalty": 1,
            "presence_penalty": 1,
            "enable_thinking": False,
        },
    )
    base.update(overrides)
    return ModelSpec(**base)


def _locate_anything_spec(**overrides) -> ModelSpec:
    """LocateAnything: grounding model (image-in, text-out with bbox coords).

    Sampling parameters mirror ``examples/LocateAnything/cpp/main.cc``
    ``SAMPLE_PARAMS`` defaults: ``max_new_tokens=1024``, ``top_k=1``,
    ``top_p=0.9``, ``temperature=1e-6``, ``repeat_penalty=1.05``,
    ``enable_thinking=False``. Near-greedy (1e-6) is required because
    the model emits structured bounding-box tokens where temperature=1.0
    would degrade to garbage; same reasoning as ``_qwen3vl_4b_spec``.
    """
    base = dict(
        name="LocateAnything-3b",
        kind="grounding",
        variant="LocateAnything-3b",
        vocab_size=152681,
        special_eos_id=151645,
        max_context_len=1024,
        image_size=448,
        options={
            "max_new_tokens": 1024,
            "top_k": 1,
            "top_p": 0.9,
            "temperature": 1e-6,
            "repeat_penalty": 1.05,
            "enable_thinking": False,
        },
    )
    base.update(overrides)
    return ModelSpec(**base)


def _depth_anything_v3_spec(**overrides) -> ModelSpec:
    """Depth Anything V3: monocular depth estimation (image-in, depth-out).

    Unlike the LLM-family presets this one carries **no sampling
    parameters** — the model is not a text generator. ``image_size`` is
    left at the dataclass default because the real input resolution is
    read off the ``local`` stage's input tensor at load time (the export
    bakes it in); the field is only a hint for callers that want to know
    the nominal size before loading.
    """
    base = dict(
        name="DepthAnythingV3",
        kind="depth",
        variant="depth-anything-v3",
        options={},
    )
    base.update(overrides)
    return ModelSpec(**base)


# --------------------------------------------------------------------------- #
# Predefined paths: Qwen3-VL-4B lives at a fixed on-device location
# (``/userdata/models/Qwen3-VL-4B/``). No network access is performed at any
# point — the user is expected to place the files there manually.
# Override the directory directly via the ``WDNAI_QWEN3_VL_4B_DIR`` env var
# (intended for debugging; e.g. pointing at an ad-hoc export tree).
#
# Vision model naming uses a per-resolution suffix so multiple native
# training resolutions can coexist on the same NPU card:
#
#     vision_Qwen3-VL-4B.384x384.1828.rknn + .weight   # required
#     vision_Qwen3-VL-4B.768x768.1828.rknn + .weight   # optional
#
# ``VLMBackend`` auto-discovers every ``vision_Qwen3-VL-4B.*x*.1828.rknn``
# sibling under the directory at load time and picks the variant
# closest to the input image's longest side per call. The 384x384
# pair is required because the legacy ``spec.image_size=384`` fallback
# depends on it.
# --------------------------------------------------------------------------- #
_QWEN3_VL_4B_REQUIRED = (
    "llm_Qwen3-VL-4B.1828.rknn",
    "llm_Qwen3-VL-4B.weight",
    # 384x384 is the always-required base resolution — VLMBackend falls
    # back to it (and uses it for the spec.image_size fallback) when
    # no other variants exist.
    "vision_Qwen3-VL-4B.384x384.1828.rknn",
    "vision_Qwen3-VL-4B.384x384.weight",
    "Qwen3-VL-4B.embed.bin",
    "Qwen3-VL-4B.tokenizer.gguf",
)
# Optional vision variants — VLMBackend discovers them at init and
# picks per-call based on the input image's longest side. Each entry
# must be a pair (``.1828.rknn`` + ``.weight``); missing pairs cause
# that variant to be silently skipped (logged as a warning).
_QWEN3_VL_4B_OPTIONAL = (
    # 768x768 — used for high-resolution inputs where the 384x384
    # encoder would discard too much detail.
    "vision_Qwen3-VL-4B.768x768.1828.rknn",
    "vision_Qwen3-VL-4B.768x768.weight",
)


def _models_root() -> Path:
    """Return the on-device root directory holding all fixed-location models."""
    return Path(_os.environ.get("WDNAI_MODELS_DIR", "/userdata/models"))


def _qwen3_vl_4b_dir() -> Path:
    """Resolve the Qwen3-VL-4B model directory.

    Override via the ``WDNAI_QWEN3_VL_4B_DIR`` env var (full path);
    fallback ``$WDNAI_MODELS_DIR/Qwen3-VL-4B`` (which itself defaults
    to ``/userdata/models/Qwen3-VL-4B`` when ``WDNAI_MODELS_DIR`` is
    unset). Mirrors ``_qwen3_4b_dir`` so every model dir honors the
    same ``WDNAI_MODELS_DIR`` root — the wdnwebui 基础配置 page writes
    ``base.model_path`` into ``WDNAI_MODELS_DIR`` at boot, and the
    SDK must follow it for all presets, not just two of them.
    """
    override = _os.environ.get("WDNAI_QWEN3_VL_4B_DIR")
    return Path(override) if override else _models_root() / "Qwen3-VL-4B"


def _default_qwen3_vl_4b_paths() -> dict:
    """Return absolute paths to the Qwen3-VL-4B files under the resolved directory.

    The ``rknn_vision_path`` points to the **384x384** vision model;
    ``VLMBackend`` then discovers the optional 768x768 sibling in the
    same directory at runtime. Both ``vision_*.384x384.1828.rknn`` and
    ``vision_*.384x384.weight`` must exist for the preset to be usable.
    """
    d = _qwen3_vl_4b_dir()
    return {
        "rknn_llm_path":   str(d / "llm_Qwen3-VL-4B.1828.rknn"),
        "rknn_vision_path": str(d / "vision_Qwen3-VL-4B.384x384.1828.rknn"),
        "tokenizer_path":  str(d / "Qwen3-VL-4B.tokenizer.gguf"),
        "embed_path":      str(d / "Qwen3-VL-4B.embed.bin"),
    }


def _verify_qwen3_vl_4b_files() -> None:
    """Raise :class:`FileNotFoundError` listing any missing Qwen3-VL-4B files.

    Required files (must be present): see ``_QWEN3_VL_4B_REQUIRED``.
    Optional files (warn if any pair is half-present): see
    ``_QWEN3_VL_4B_OPTIONAL``. We only warn for the optional list so a
    half-installed 768x768 doesn't break the preset — the backend
    will skip the half-present variant at runtime.

    Legacy single-file fallback for the 384x384 vision pair: older
    installations shipped ``vision_Qwen3-VL-4B.1828.rknn`` /
    ``vision_Qwen3-VL-4B.weight`` without the ``<WxH>`` suffix.
    ``VLMBackend._discover_vision_paths`` already falls back to that
    single-file path at runtime when no ``<WxH>`` siblings are found,
    so mirror the same fallback here — pass the preflight on either
    naming and let the loader pick which to load.
    """
    d = _qwen3_vl_4b_dir()
    _LEGACY_VISION_384_FALLBACK = {
        "vision_Qwen3-VL-4B.384x384.1828.rknn": "vision_Qwen3-VL-4B.1828.rknn",
        "vision_Qwen3-VL-4B.384x384.weight":    "vision_Qwen3-VL-4B.weight",
    }
    missing = []
    for f in _QWEN3_VL_4B_REQUIRED:
        if (d / f).is_file():
            continue
        legacy = _LEGACY_VISION_384_FALLBACK.get(f)
        if legacy and (d / legacy).is_file():
            continue
        missing.append(f)
    if missing:
        raise FileNotFoundError(
            f"Qwen3-VL-4B files missing under {d}: {missing}. "
            f"Place the files there or set WDNAI_QWEN3_VL_4B_DIR to point at a custom path."
        )
    # Optional variants — warn (don't raise) for half-present pairs.
    import logging as _logging
    _log = _logging.getLogger("wdnai.model")
    for stem in ("vision_Qwen3-VL-4B.768x768",):
        rknn = d / f"{stem}.1828.rknn"
        weight = d / f"{stem}.weight"
        if rknn.is_file() and not weight.is_file():
            _log.warning(
                "Qwen3-VL-4B: %s.1828.rknn present but %s.weight missing; "
                "VLMBackend will skip this variant.",
                stem, stem,
            )
        elif weight.is_file() and not rknn.is_file():
            _log.warning(
                "Qwen3-VL-4B: %s.weight present but %s.1828.rknn missing; "
                "VLMBackend will skip this variant.",
                stem, stem,
            )


_QWEN3_4B_REQUIRED = (
    "Qwen3-4B.1828.rknn",
    "Qwen3-4B.weight",
    "Qwen3-4B.embed.bin",
    "Qwen3-4B.tokenizer.gguf",
)


def _qwen3_4b_dir() -> Path:
    override = _os.environ.get("WDNAI_QWEN3_4B_DIR")
    return Path(override) if override else _models_root() / "Qwen3-4B"


def _default_qwen3_4b_paths() -> dict:
    d = _qwen3_4b_dir()
    return {
        "rknn_path":      str(d / "Qwen3-4B.1828.rknn"),
        "weight_path":    str(d / "Qwen3-4B.weight"),
        "tokenizer_path": str(d / "Qwen3-4B.tokenizer.gguf"),
        "embed_path":     str(d / "Qwen3-4B.embed.bin"),
    }


def _verify_qwen3_4b_files() -> None:
    d = _qwen3_4b_dir()
    missing = [f for f in _QWEN3_4B_REQUIRED if not (d / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Qwen3-4B files missing under {d}: {missing}. "
            f"Place the files there or set WDNAI_QWEN3_4B_DIR to point at a custom path."
        )


# --------------------------------------------------------------------------- #
# Qwen3.5-VL-4B: vision-language model (image-in, text-out). The same RKNN
# also powers pure-text inference (the VLM-mode LLM is a regular LLM with
# MRoPE baked into the graph; without vision tokens MRoPE reduces to
# standard positional encoding). File layout::
#
#     Qwen3.5-4B-llm.1828.rknn                              (VLM-mode LLM, MRoPE)
#     Qwen3.5-4B-llm.weight
#     Qwen3.5-4B-vision.384x384.1828.rknn + .weight        (required, 384×384 prune-mode)
#     Qwen3.5-4B-vision.768x768.1828.rknn + .weight        (optional, 768×768 prune-mode)
#     Qwen3.5-4B-llm.embed.bin                              fp16 (vocab_size × dim)
#     Qwen3.5-4B-llm.tokenizer.gguf                         GPT-2 BPE + ByteLevel
#
# The 384×384 vision pair is the always-required base resolution;
# ``VLMBackend._discover_vision_paths`` discovers both 384×384 and
# 768×768 sibling ``.<WxH>.1828.rknn`` files in the directory.
# --------------------------------------------------------------------------- #
_QWEN3_5_4B_VL_REQUIRED = (
    "Qwen3.5-4B-llm.1828.rknn",
    "Qwen3.5-4B-llm.weight",
    "Qwen3.5-4B-vision.384x384.1828.rknn",
    "Qwen3.5-4B-vision.384x384.weight",
    "Qwen3.5-4B-llm.embed.bin",
    "Qwen3.5-4B-llm.tokenizer.gguf",
)


def _qwen3_5_4b_vl_dir() -> Path:
    """Resolve the Qwen3.5-VL-4B model directory.

    Shares the same on-device root as the text-only Qwen3.5-4B
    (``/userdata/models/Qwen3.5-4B/``); only the filenames inside
    the directory differ (``*-llm.*`` vs ``*``). Override via
    ``WDNAI_QWEN3_5_4B_VL_DIR`` (full path); fallback
    ``$WDNAI_MODELS_DIR/Qwen3.5-4B`` (which itself defaults to
    ``/userdata/models/Qwen3.5-4B`` when ``WDNAI_MODELS_DIR`` is unset).
    """
    override = _os.environ.get("WDNAI_QWEN3_5_4B_VL_DIR")
    return Path(override) if override else _models_root() / "Qwen3.5-4B"


def _default_qwen3_5_4b_vl_paths() -> dict:
    d = _qwen3_5_4b_vl_dir()
    return {
        "rknn_llm_path":    str(d / "Qwen3.5-4B-llm.1828.rknn"),
        "rknn_vision_path": str(d / "Qwen3.5-4B-vision.384x384.1828.rknn"),
        "tokenizer_path":   str(d / "Qwen3.5-4B-llm.tokenizer.gguf"),
        "embed_path":       str(d / "Qwen3.5-4B-llm.embed.bin"),
    }


def _verify_qwen3_5_4b_vl_files() -> None:
    d = _qwen3_5_4b_vl_dir()
    missing = [f for f in _QWEN3_5_4B_VL_REQUIRED if not (d / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Qwen3.5-VL-4B files missing under {d}: {missing}. "
            f"Place the files there or set WDNAI_QWEN3_5_4B_VL_DIR to point at a custom path."
        )


# --------------------------------------------------------------------------- #
# Qwen3.5-VL-9B: two-stage LLM export, one stage per RK182x card.
# --------------------------------------------------------------------------- #
_QWEN3_5_9B_VL_REQUIRED = (
    "Qwen3.5-9B-llm_seg0.1828.rknn",
    "Qwen3.5-9B-llm_seg0.weight",
    "Qwen3.5-9B-llm_seg1.1828.rknn",
    "Qwen3.5-9B-llm_seg1.weight",
    "Qwen3.5-9B-vision.1828.rknn",
    "Qwen3.5-9B-vision.weight",
    "Qwen3.5-9B-llm.embed.bin",
    "Qwen3.5-9B-llm.tokenizer.gguf",
)


def _qwen3_5_9b_vl_dir() -> Path:
    """Resolve the Qwen3.5-VL-9B model directory.

    Override via ``WDNAI_QWEN3_5_VL_9B_DIR`` (full path); fallback
    ``$WDNAI_MODELS_DIR/Qwen3.5-VL-9B`` (``/userdata/models/Qwen3.5-VL-9B``
    when the root is unset).
    """
    override = _os.environ.get("WDNAI_QWEN3_5_VL_9B_DIR")
    return Path(override) if override else _models_root() / "Qwen3.5-VL-9B"


def _default_qwen3_5_9b_vl_paths() -> dict:
    d = _qwen3_5_9b_vl_dir()
    seg0 = d / "Qwen3.5-9B-llm_seg0.1828.rknn"
    seg1 = d / "Qwen3.5-9B-llm_seg1.1828.rknn"
    seg0_w = d / "Qwen3.5-9B-llm_seg0.weight"
    seg1_w = d / "Qwen3.5-9B-llm_seg1.weight"
    return {
        "rknn_llm_path": str(seg0),
        "weight_llm_path": str(seg0_w),
        "rknn_vision_path": str(d / "Qwen3.5-9B-vision.1828.rknn"),
        "weight_vision_path": str(d / "Qwen3.5-9B-vision.weight"),
        "llm_seg_paths": [str(seg0), str(seg1)],
        "weight_seg_paths": [str(seg0_w), str(seg1_w)],
        "tokenizer_path": str(d / "Qwen3.5-9B-llm.tokenizer.gguf"),
        "embed_path": str(d / "Qwen3.5-9B-llm.embed.bin"),
    }


def _verify_qwen3_5_9b_vl_files() -> None:
    d = _qwen3_5_9b_vl_dir()
    missing = [f for f in _QWEN3_5_9B_VL_REQUIRED if not (d / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Qwen3.5-VL-9B files missing under {d}: {missing}. "
            f"Place the files there or set WDNAI_QWEN3_5_VL_9B_DIR to point at a custom path."
        )


# --------------------------------------------------------------------------- #
# Qwen3-VL-Embedding-2B: retrieval embedding model (no LLM token generation).
# File layout matches Qwen3-VL-4B (llm/vision .1828.rknn + .weight + embed.bin
# + tokenizer.gguf). The LLM model is loaded via the plain RKNN3Lite path (not
# the llm_mode session) and produces a normalized 2048-dim embedding vector.
# --------------------------------------------------------------------------- #
_QWEN3_VL_EMBEDDING_2B_DIR_NAME = "Qwen3-VL-Embedding-2B"
_QWEN3_VL_EMBEDDING_2B_REQUIRED = (
    "llm_Qwen3-VL-Embedding-2B.1828.rknn",
    "llm_Qwen3-VL-Embedding-2B.weight",
    "vision_Qwen3-VL-Embedding-2B.1828.rknn",
    "vision_Qwen3-VL-Embedding-2B.weight",
    "Qwen3-VL-Embedding-2B.embed.bin",
    "Qwen3-VL-Embedding-2B.tokenizer.gguf",
)


def _qwen3_vl_embedding_2b_dir() -> Path:
    """Resolve the Qwen3-VL-Embedding-2B model directory.

    Override via the ``WDNAI_QWEN3_VL_EMBEDDING_2B_DIR`` env var (full
    path); fallback ``$WDNAI_MODELS_DIR/Qwen3-VL-Embedding-2B`` (which
    itself defaults to ``/userdata/models/Qwen3-VL-Embedding-2B`` when
    ``WDNAI_MODELS_DIR`` is unset). Mirrors ``_locate_anything_dir`` so
    every model dir honors the same ``WDNAI_MODELS_DIR`` root.
    """
    override = _os.environ.get("WDNAI_QWEN3_VL_EMBEDDING_2B_DIR")
    return Path(override) if override else (
        _models_root() / _QWEN3_VL_EMBEDDING_2B_DIR_NAME)


def _default_qwen3_vl_embedding_2b_paths() -> dict:
    """Return absolute paths to the Qwen3-VL-Embedding-2B files."""
    d = _qwen3_vl_embedding_2b_dir()
    return {
        "rknn_llm_path":   str(d / "llm_Qwen3-VL-Embedding-2B.1828.rknn"),
        "rknn_vision_path": str(d / "vision_Qwen3-VL-Embedding-2B.1828.rknn"),
        "tokenizer_path":  str(d / "Qwen3-VL-Embedding-2B.tokenizer.gguf"),
        "embed_path":      str(d / "Qwen3-VL-Embedding-2B.embed.bin"),
    }


def _verify_qwen3_vl_embedding_2b_files() -> None:
    """Raise :class:`FileNotFoundError` listing any missing Qwen3-VL-Embedding-2B files."""
    d = _qwen3_vl_embedding_2b_dir()
    missing = [f for f in _QWEN3_VL_EMBEDDING_2B_REQUIRED if not (d / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Qwen3-VL-Embedding-2B files missing under {d}: {missing}. "
            f"Place the files there or set WDNAI_MODELS_DIR to override the root."
        )


# --------------------------------------------------------------------------- #
# Qwen2.5-Omni-3B: audio language model (wav-in, text-out). Fixed at
# ``/userdata/models/Qwen2.5-Omni-3B/`` (override via env var). File layout::
#
#     Qwen2.5-Omni-3B-audio.1828.rknn  + .weight        (dynamic shape 0/1)
#     Qwen2.5-Omni-3B-llm.1828.rknn    + .weight        (llm_mode)
#     Qwen2.5-Omni-3B-llm.embed.bin    fp16 (vocab_size × 2048)
#     Qwen2.5-Omni-3B-llm.tokenizer.gguf  GPT-2 BPE + ByteLevel
#     mel_128_filters.txt              (128 × N_FFT/2 + 1) fp32
# --------------------------------------------------------------------------- #
_QWEN_2_5_OMNI_3B_REQUIRED = (
    "Qwen2.5-Omni-3B-audio.1828.rknn",
    "Qwen2.5-Omni-3B-audio.weight",
    "Qwen2.5-Omni-3B-llm.1828.rknn",
    "Qwen2.5-Omni-3B-llm.weight",
    "Qwen2.5-Omni-3B-llm.embed.bin",
    "Qwen2.5-Omni-3B-llm.tokenizer.gguf",
    "mel_128_filters.txt",
)


def _qwen_2_5_omni_3b_dir() -> Path:
    """Resolve the Qwen2.5-Omni-3B model directory.

    Override via the ``WDNAI_QWEN_2_5_OMNI_3B_DIR`` env var (full path);
    fallback ``$WDNAI_MODELS_DIR/Qwen2.5-Omni-3B`` (which itself defaults
    to ``/userdata/models/Qwen2.5-Omni-3B`` when ``WDNAI_MODELS_DIR`` is
    unset). Mirrors ``_qwen3_4b_dir`` so every model dir honors the same
    ``WDNAI_MODELS_DIR`` root.
    """
    override = _os.environ.get("WDNAI_QWEN_2_5_OMNI_3B_DIR")
    return Path(override) if override else _models_root() / "Qwen2.5-Omni-3B"


def _default_qwen_2_5_omni_3b_paths() -> dict:
    """Return absolute paths to the Qwen2.5-Omni-3B files under the resolved directory."""
    d = _qwen_2_5_omni_3b_dir()
    return {
        "omni_model_dir": str(d),
        "tokenizer_path": str(d / "Qwen2.5-Omni-3B-llm.tokenizer.gguf"),
        "embed_path":     str(d / "Qwen2.5-Omni-3B-llm.embed.bin"),
    }


def _verify_qwen_2_5_omni_3b_files() -> None:
    """Raise :class:`FileNotFoundError` listing any missing Qwen2.5-Omni-3B files."""
    d = _qwen_2_5_omni_3b_dir()
    missing = [f for f in _QWEN_2_5_OMNI_3B_REQUIRED if not (d / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Qwen2.5-Omni-3B files missing under {d}: {missing}. "
            "Place the files there or set WDNAI_QWEN_2_5_OMNI_3B_DIR to point "
            "at a custom path."
        )


# --------------------------------------------------------------------------- #
# LocateAnything: grounding model (image-in, text-out with bbox coords).
# File layout::
#
#     llm_LocateAnything.1828.rknn + .weight          (text LLM, RK1828 quant)
#     vision_LocateAnything.448x448.1828.rknn + .weight  (448×448 vision encoder)
#     LocateAnything.embed.bin                        fp16 (vocab_size × dim)
#     LocateAnything.tokenizer.gguf                   GPT-2 BPE + ByteLevel
#
# The 448×448 vision pair is the only resolution shipped; ``image_size=448``
# on the preset forces single-variant dispatch (no auto-discovery of
# siblings in the backend — kept simple to mirror the C++ demo's
# ``main.cc`` defaults).
# --------------------------------------------------------------------------- #
_LOCATE_ANYTHING_REQUIRED = (
    "llm_LocateAnything.1828.rknn",
    "llm_LocateAnything.weight",
    "vision_LocateAnything.448x448.1828.rknn",
    "vision_LocateAnything.448x448.weight",
    "LocateAnything.embed.bin",
    "LocateAnything.tokenizer.gguf",
)


def _locate_anything_dir() -> Path:
    """Resolve the LocateAnything model directory.

    Override via the ``WDNAI_LOCATE_ANYTHING_DIR`` env var (full path);
    fallback ``$WDNAI_MODELS_DIR/LocateAnything-3B`` (which itself defaults
    to ``/userdata/models/LocateAnything-3B`` when ``WDNAI_MODELS_DIR`` is
    unset). Mirrors ``_qwen3_vl_4b_dir`` so every model dir honors the
    same ``WDNAI_MODELS_DIR`` root.
    """
    override = _os.environ.get("WDNAI_LOCATE_ANYTHING_DIR")
    return Path(override) if override else _models_root() / "LocateAnything-3B"


def _default_locate_anything_paths() -> dict:
    """Return absolute paths to the LocateAnything files under the resolved directory."""
    d = _locate_anything_dir()
    return {
        "rknn_path":          str(d / "llm_LocateAnything.1828.rknn"),
        "weight_path":        str(d / "llm_LocateAnything.weight"),
        "rknn_vision_path":   str(d / "vision_LocateAnything.448x448.1828.rknn"),
        "weight_vision_path": str(d / "vision_LocateAnything.448x448.weight"),
        "tokenizer_path":     str(d / "LocateAnything.tokenizer.gguf"),
        "embed_path":         str(d / "LocateAnything.embed.bin"),
    }


def _verify_locate_anything_files() -> None:
    """Raise :class:`FileNotFoundError` listing any missing LocateAnything files.

    Required files (must be present): see ``_LOCATE_ANYTHING_REQUIRED``.
    Path resolution is wrapped in ``try/except OSError`` so a malformed
    ``WDNAI_LOCATE_ANYTHING_DIR`` (e.g. with embedded NUL or a path the
    OS rejects) surfaces as a single descriptive ``FileNotFoundError``
    rather than an obscure ``OSError`` from ``pathlib``.
    """
    try:
        d = _locate_anything_dir()
    except OSError as e:
        raise FileNotFoundError(
            f"LocateAnything directory resolution failed: {e}"
        )
    missing = [f for f in _LOCATE_ANYTHING_REQUIRED if not (d / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"LocateAnything files missing under {d}: {missing}. "
            f"Place the files there or set WDNAI_LOCATE_ANYTHING_DIR to point at a custom path."
        )


# --------------------------------------------------------------------------- #
# HY-MT1.5-1.8B: Hunyuan-Dense machine translation (text → text).
#
# File layout — **one** RKNN + its weights + the tokenizer / embed pair,
# i.e. the same shape as the plain-text LLM presets::
#
#     HY-MT1.5-1.8B.1820.rknn
#     HY-MT1.5-1.8B.weight
#     HY-MT1.5-1.8B.embed.bin
#     HY-MT1.5-1.8B.tokenizer.gguf
#
# ``HY-MT1.5-1.8B.config.pkl`` ships alongside them but is **not** read at
# runtime: it is the conversion tool's receipt (chat template + HF config
# + vocab size). The SDK takes the same numbers from the RKNN's own
# embedded metadata and from the GGUF tokenizer, so the SDK never has to
# unpickle a third-party file (pickle = arbitrary code execution).
#
# The ``.1820`` chip suffix differs from the ``.1828`` of the other
# packages on this board, but it is **only a file name**: the runtime
# target is carried by ``ModelSpec.target`` (``"rk1820"``, the default for
# every RKNN3 preset in this SDK).
# --------------------------------------------------------------------------- #
_HY_MT_1_5_REQUIRED = (
    "HY-MT1.5-1.8B.1820.rknn",
    "HY-MT1.5-1.8B.weight",
    "HY-MT1.5-1.8B.embed.bin",
    "HY-MT1.5-1.8B.tokenizer.gguf",
)


def _hy_mt_1_5_dir() -> Path:
    """Resolve the HY-MT1.5 model directory.

    Override via the ``WDNAI_HY_MT_1_5_DIR`` env var (full path); fallback
    ``$WDNAI_MODELS_DIR/HY_MT_1_5`` (which itself defaults to
    ``/userdata/models/HY_MT_1_5`` when ``WDNAI_MODELS_DIR`` is unset).
    Mirrors ``_locate_anything_dir`` so every model dir honors the same
    ``WDNAI_MODELS_DIR`` root.
    """
    override = _os.environ.get("WDNAI_HY_MT_1_5_DIR")
    return Path(override) if override else _models_root() / "HY_MT_1_5"


def _default_hy_mt_1_5_paths() -> dict:
    """Return absolute paths to the HY-MT1.5 files under the resolved directory."""
    d = _hy_mt_1_5_dir()
    return {
        "rknn_path": str(d / "HY-MT1.5-1.8B.1820.rknn"),
        "weight_path": str(d / "HY-MT1.5-1.8B.weight"),
        "tokenizer_path": str(d / "HY-MT1.5-1.8B.tokenizer.gguf"),
        "embed_path": str(d / "HY-MT1.5-1.8B.embed.bin"),
    }


def _verify_hy_mt_1_5_files() -> None:
    """Raise :class:`FileNotFoundError` listing any missing HY-MT1.5 files.

    Required files (must be present): see ``_HY_MT_1_5_REQUIRED``.
    Path resolution is wrapped in ``try/except OSError`` so a malformed
    ``WDNAI_HY_MT_1_5_DIR`` surfaces as a single descriptive
    :class:`FileNotFoundError` rather than an obscure ``OSError`` from
    ``pathlib``.
    """
    try:
        d = _hy_mt_1_5_dir()
    except OSError as e:
        raise FileNotFoundError(
            f"HY-MT1.5 directory resolution failed: {e}"
        )
    missing = [f for f in _HY_MT_1_5_REQUIRED if not (d / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"HY-MT1.5 files missing under {d}: {missing}. "
            f"Place the files there or set WDNAI_HY_MT_1_5_DIR to point at a custom path."
        )


# --------------------------------------------------------------------------- #
# Intern-Decision-4B: structured decision model (state → options + probability)
#
# File layout — **two** RKNN pairs (text + vision) plus the tokenizer and
# the embedding table::
#
#     Intern-Decision-4B-llm.rknn
#         + Intern-Decision-4B-llm.weight
#         + Intern-Decision-4B-llm.embed.bin
#         + Intern-Decision-4B-llm.tokenizer.gguf
#     Intern-Decision-4B-vision.rknn
#         + Intern-Decision-4B-vision.weight
#
# The naming follows the upstream export (``export_rknn.py`` keeps the
# ``<model>-llm`` / ``<model>-vision`` prefixes); the ``.weight`` sibling
# is derived by replacing the ``.rknn`` suffix — exactly what the C++
# reference's ``make_weight_path`` does.
#
# Unlike the Qwen VLM presets the **text** half is called ``-llm`` and
# not ``llm_`` — the upstream export names it that way and the reference
# implementation scans the directory by that prefix.
# --------------------------------------------------------------------------- #
_INTERN_DECISION_4B_REQUIRED = (
    "Intern-Decision-4B-llm.rknn",
    "Intern-Decision-4B-llm.weight",
    "Intern-Decision-4B-llm.embed.bin",
    "Intern-Decision-4B-llm.tokenizer.gguf",
    "Intern-Decision-4B-vision.rknn",
    "Intern-Decision-4B-vision.weight",
)


def _intern_decision_4b_dir() -> Path:
    """Resolve the Intern-Decision-4B model directory.

    Override via the ``WDNAI_INTERN_DECISION_4B_DIR`` env var (full path);
    fallback ``$WDNAI_MODELS_DIR/Intern_Decision-4B`` (which itself
    defaults to ``/userdata/models/Intern_Decision-4B`` when
    ``WDNAI_MODELS_DIR`` is unset). Mirrors ``_hy_mt_1_5_dir`` so every
    model dir honors the same ``WDNAI_MODELS_DIR`` root.
    """
    override = _os.environ.get("WDNAI_INTERN_DECISION_4B_DIR")
    return Path(override) if override else _models_root() / "Intern_Decision-4B"


def _default_intern_decision_4b_paths() -> dict:
    """Return absolute paths to the Intern-Decision-4B files."""
    d = _intern_decision_4b_dir()
    return {
        "rknn_llm_path": str(d / "Intern-Decision-4B-llm.rknn"),
        "weight_llm_path": str(d / "Intern-Decision-4B-llm.weight"),
        "rknn_vision_path": str(d / "Intern-Decision-4B-vision.rknn"),
        "weight_vision_path": str(d / "Intern-Decision-4B-vision.weight"),
        "tokenizer_path": str(d / "Intern-Decision-4B-llm.tokenizer.gguf"),
        "embed_path": str(d / "Intern-Decision-4B-llm.embed.bin"),
    }


def _verify_intern_decision_4b_files() -> None:
    """Raise :class:`FileNotFoundError` listing any missing files.

    Required files (must be present): see ``_INTERN_DECISION_4B_REQUIRED``.
    """
    try:
        d = _intern_decision_4b_dir()
    except OSError as e:
        raise FileNotFoundError(
            f"Intern-Decision-4B directory resolution failed: {e}"
        )
    missing = [f for f in _INTERN_DECISION_4B_REQUIRED if not (d / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Intern-Decision-4B files missing under {d}: {missing}. "
            f"Place the files there or set WDNAI_INTERN_DECISION_4B_DIR to point at a custom path."
        )


def _intern_decision_4b_spec(**overrides) -> ModelSpec:
    """Intern-Decision-4B: structured decision model (分类 / 是非 / 评分).

    Not a chat model: it does **not** generate text. For each decision
    field it emits the logits of the 62 candidate symbols
    (``A-Z`` + ``a-z`` + ``0-9``), which the CPU side turns into option
    probabilities (softmax + checkpoint temperature calibration).

    ``vocab_size=248320`` / ``special_eos_id=248046`` come from the
    shipped GGUF tokenizer. ``max_context_len=4096`` is the value baked
    into the export (``max_position_embeddings=4096``,
    ``kvcache_buffer_len=4096``) — the upstream checkpoint's training
    limit is much larger but the export did not keep it. The runtime
    overrides it with the model's own ``max_ctx_len`` anyway.

    Sampling is irrelevant here (the backend runs
    ``disable_sampling=True`` and reads the logits directly), but the
    values mirror the C++ reference: ``temperature=1``,
    ``repeat_penalty=1``, ``top_k=1``, ``top_p=1``.

    The **calibration** temperature (a different thing — it shapes the
    reported probabilities on the CPU) is looked up from the model's
    embedding width at runtime; see
    :data:`wdnai.decision.CHECKPOINT_TEMPERATURES`.
    """
    base = dict(
        name="Intern-Decision-4B",
        kind="decision",
        variant="intern-decision",
        vocab_size=248320,
        special_eos_id=248046,
        max_context_len=4096,
        image_size=384,
        target="rk1820",
        core_mask=0xFF,
        options={
            "max_new_tokens": 1,
            "top_k": 1,
            "top_p": 1.0,
            "temperature": 1.0,
            "repeat_penalty": 1.0,
            "keep_history": 0,
            # 视觉模型是否常驻。
            #
            # ⚠️ **默认 True**（2026-11-17 改）：决策模型要**独占**一整张卡
            # （两张 1828 卡各 5120 MB，一份决策实例 ~3100 MB 就吃掉大半张），
            # 既然独占，就没必要再为「省 700 MB」把视觉做成按需加载 ——
            # 常驻的好处是**带图请求没有首次加载的 7.5 秒**，且显存占用是
            # 稳定的、可预期的（算力页上显示的数字就是真实占用）。
            #
            # 设 False 可回到 C++ 参考实现的行为：图编码完就释放视觉模型。
            "keep_vision_resident": True,
        },
    )
    base.update(overrides)
    return ModelSpec(**base)


# --------------------------------------------------------------------------- #
# Depth Anything V3: monocular depth estimation (image-in, depth-map-out).
# File layout — **three** RKNN stages, each with its own ``.weight``::
#
#     local_DepthAnythingV3.280x280.1view.1828.rknn
#         + local_DepthAnythingV3.280x280.1view.weight
#         (image  [1,280,280,3] uint8 → tokens [1,401,768])
#     global_DepthAnythingV3.280x280.1view.1828.rknn
#         + global_DepthAnythingV3.280x280.1view.weight
#         (tokens [1,401,1,768] → 4 feature maps [1,400,1,1536])
#     head_DepthAnythingV3.280x280.1view.1828.rknn
#         + head_DepthAnythingV3.280x280.1view.weight
#         (features [1,400,1536] → depth/conf [1,1,280,280])
#
# The file names follow the repo-wide deployment convention
# ``<role>_<Model>[.<WxH>[.<N>view]].1828.rknn`` — the same shape as
# ``vision_LocateAnything.448x448.1828.rknn``. The ``.280x280`` / ``.1view``
# segments only **document** what the export baked in: the backend reads
# the real geometry (resolution, view count, token counts) off the tensor
# attributes, never by parsing the file name.
#
# The number of views (``V``) is **baked into the export**: ``global``'s
# input shape is ``[V,L,1,D]`` and ``head``'s output is ``[1,V,H,W]``, so a
# model exported with ``--views 10`` only ever accepts 10 images. The backend
# reads ``V`` off the model itself instead of taking it as a parameter.
# --------------------------------------------------------------------------- #
_DEPTH_ANYTHING_V3_REQUIRED = (
    "local_DepthAnythingV3.280x280.1view.1828.rknn",
    "local_DepthAnythingV3.280x280.1view.weight",
    "global_DepthAnythingV3.280x280.1view.1828.rknn",
    "global_DepthAnythingV3.280x280.1view.weight",
    "head_DepthAnythingV3.280x280.1view.1828.rknn",
    "head_DepthAnythingV3.280x280.1view.weight",
)


def _depth_anything_v3_dir() -> Path:
    """Resolve the Depth Anything V3 model directory.

    Override via the ``WDNAI_DEPTH_ANYTHING_V3_DIR`` env var (full path);
    fallback ``$WDNAI_MODELS_DIR/Depth-Anything-V3`` (which itself defaults
    to ``/userdata/models/Depth-Anything-V3`` when ``WDNAI_MODELS_DIR`` is
    unset). Mirrors ``_locate_anything_dir`` so every model dir honors the
    same ``WDNAI_MODELS_DIR`` root.
    """
    override = _os.environ.get("WDNAI_DEPTH_ANYTHING_V3_DIR")
    return Path(override) if override else _models_root() / "Depth-Anything-V3"


def _default_depth_anything_v3_paths() -> dict:
    """Return absolute paths to the three DA3 RKNN stages."""
    d = _depth_anything_v3_dir()
    return {
        "depth_local_path":  str(d / "local_DepthAnythingV3.280x280.1view.1828.rknn"),
        "depth_global_path": str(d / "global_DepthAnythingV3.280x280.1view.1828.rknn"),
        "depth_head_path":   str(d / "head_DepthAnythingV3.280x280.1view.1828.rknn"),
    }


#: Directory name the TTS weights live in, under the models root.
_QWEN3_TTS_DIR_NAME = "Qwen3-TTS"


def _qwen3_tts_dir() -> Path:
    """Resolve the Qwen3-TTS model directory.

    Override via the ``WDNAI_QWEN3_TTS_DIR`` env var (full path); fallback
    ``$WDNAI_MODELS_DIR/Qwen3-TTS`` (which itself defaults to
    ``/userdata/models/Qwen3-TTS`` when ``WDNAI_MODELS_DIR`` is unset).
    Mirrors ``_locate_anything_dir`` so every model dir honors the same
    ``WDNAI_MODELS_DIR`` root.
    """
    override = _os.environ.get("WDNAI_QWEN3_TTS_DIR")
    return Path(override) if override else _models_root() / _QWEN3_TTS_DIR_NAME


#: Directory name the Qwen3-ASR weights live in, under the models root.
_QWEN3_ASR_DIR_NAME = "Qwen3-ASR"


def _qwen3_asr_dir() -> Path:
    """Resolve the Qwen3-ASR model directory.

    Override via the ``WDNAI_QWEN3_ASR_DIR`` env var (full path); fallback
    ``$WDNAI_MODELS_DIR/Qwen3-ASR`` (which itself defaults to
    ``/userdata/models/Qwen3-ASR`` when ``WDNAI_MODELS_DIR`` is unset).
    Mirrors ``_qwen3_tts_dir`` so every model dir honors the same
    ``WDNAI_MODELS_DIR`` root — the folder name is a *default*, not a
    requirement.

    This matters for wdnwebui: the worker manifest declares
    ``model_dir_env: "WDNAI_QWEN3_ASR_DIR"``, and the service sets it to
    the directory the manifest was found in. Without the override the SDK
    would look in ``$WDNAI_MODELS_DIR/Qwen3-ASR`` and the service's
    "model files live next to the manifest" invariant would silently
    break for any other folder name.
    """
    override = _os.environ.get("WDNAI_QWEN3_ASR_DIR")
    return Path(override) if override else _models_root() / _QWEN3_ASR_DIR_NAME


def _verify_depth_anything_v3_files() -> None:
    """Raise :class:`FileNotFoundError` listing any missing DA3 files.

    Required files (must be present): see ``_DEPTH_ANYTHING_V3_REQUIRED``.
    Path resolution is wrapped in ``try/except OSError`` so a malformed
    ``WDNAI_DEPTH_ANYTHING_V3_DIR`` surfaces as a single descriptive
    ``FileNotFoundError`` rather than an obscure ``OSError`` from
    ``pathlib`` — same treatment as ``_verify_locate_anything_files``.
    """
    try:
        d = _depth_anything_v3_dir()
    except OSError as e:
        raise FileNotFoundError(
            f"Depth Anything V3 directory resolution failed: {e}"
        )
    missing = [f for f in _DEPTH_ANYTHING_V3_REQUIRED if not (d / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Depth Anything V3 files missing under {d}: {missing}. "
            "Place the files there or set WDNAI_DEPTH_ANYTHING_V3_DIR to point at a custom path."
        )


# --------------------------------------------------------------------------- #
# Qwen-Image-2.1: text-to-image (RKNN3 / rk1820). One model package holding
# all three parts of the diffusion pipeline. File layout::
#
#     llm_qwenimage21-te.1828.rknn + .weight       text encoder (prompt → ehs)
#     qwenimage21-te.embed.bin + .tokenizer.gguf   tokenizer / embedding table
#     dit_qwenimage21-512hc.seg{0..3}.1828.rknn + .weight   DiT, 4 × 8 blocks（切头单卡）
#     dit_qwenimage21-512.head.1828.rknn + .weight            DiT 的「头」（独立卡）
#     vae_qwenimage21-512.seg{0..3}.1828.rknn + .weight   VAE decoder, 4 segments
#
# The package lives in one flat directory (the ``/mnt/oem/models/`` layout);
# the per-part file names keep the ``<part>_<Model>.<variant>.1828.rknn``
# convention.
#
# ⭐ 第三十六轮：**包里只剩模型**（无 .so、无可执行文件、无数据文件）。
#    运行时是 wdnai 自带的 C shim（``wdnai/_cshim``）；「固定初始噪声」由
#    cshim 现场生成（numpy 兼容的 PCG64 + ziggurat），所以也不需要随包发布。
# --------------------------------------------------------------------------- #
_QWEN_IMAGE_21_REQUIRED = [
    "llm_qwenimage21-te.1828.rknn",
    "llm_qwenimage21-te.weight",
    "qwenimage21-te.embed.bin",
    "qwenimage21-te.tokenizer.gguf",
] + [
    f"vae_qwenimage21-512.seg{i}.{ext}"
    for i in range(4)
    for ext in ("1828.rknn", "weight")
]

#: DiT 拓扑（2026-11-02 第三十五轮：**只剩切头单卡一套**）。
#:
#: 老拓扑（4 段各自带头、DiT 占两张卡）已废弃并从模型包中删除。
#: 现在 DiT = 4 段（段内无头，共享 internal）+ 独立的头模型（另一张卡）。
_QWEN_IMAGE_21_DIT_FILES = [
    f"dit_qwenimage21-512hc.seg{i}.{ext}"
    for i in range(4)
    for ext in ("1828.rknn", "weight")
] + [
    "dit_qwenimage21-512.head.1828.rknn",
    "dit_qwenimage21-512.head.weight",
]


def _qwen_image_21_dir() -> Path:
    """Resolve the Qwen-Image-2.1 model-package directory.

    Override via the ``WDNAI_QWEN_IMAGE_21_DIR`` env var (full path) or
    ``QI_MODEL_DIR`` (same name the deploy-side ``test.sh`` uses);
    otherwise ``$WDNAI_MODELS_DIR/Qwen-Image-2.1`` (which itself defaults to
    ``/userdata/models/Qwen-Image-2.1``) and finally the board location
    ``/home/rockchip/models/Qwen-Image-2.1``.

    The first **existing** candidate wins, so a board that keeps the
    package under ``/home/rockchip/models`` works without any env var.
    The legacy ``qwenimage21`` directory name is kept as a last-resort
    fallback for boards that have not renamed their package yet.
    """
    candidates = []
    for env_key in ("WDNAI_QWEN_IMAGE_21_DIR", "QI_MODEL_DIR"):
        value = _os.environ.get(env_key)
        if value:
            candidates.append(Path(value))
    root = _os.environ.get("WDNAI_MODELS_DIR")
    if root:
        candidates.append(Path(root) / "Qwen-Image-2.1")
    candidates.append(Path("/home/rockchip/models/Qwen-Image-2.1"))
    candidates.append(_models_root() / "Qwen-Image-2.1")
    # 旧目录名兜底（还没改名的板子）
    if root:
        candidates.append(Path(root) / "qwenimage21")
    candidates.append(Path("/home/rockchip/models/qwenimage21"))
    candidates.append(_models_root() / "qwenimage21")
    for cand in candidates:
        try:
            if cand.is_dir():
                return cand
        except OSError:
            continue
    return candidates[-1]


def _verify_qwen_image_21_files() -> None:
    """Raise :class:`FileNotFoundError` listing any missing package file.

    The DiT part must be the **head-cut single-card** set (2026-11-02 第三十五轮):
    4 段（段内无头）+ 独立的头模型 ⇒ DiT 只占**一张**卡，另一张卡留给常驻 TE。
    老的双卡拓扑（``dit_qwenimage21-512.seg*``）已废弃。
    """
    try:
        d = _qwen_image_21_dir()
    except OSError as e:
        raise FileNotFoundError(
            f"Qwen-Image-2.1 directory resolution failed: {e}"
        )
    missing = [f for f in _QWEN_IMAGE_21_REQUIRED if not (d / f).is_file()]
    miss = [f for f in _QWEN_IMAGE_21_DIT_FILES if not (d / f).is_file()]
    if miss:
        missing.append("[DiT] " + ", ".join(miss))
    if missing:
        raise FileNotFoundError(
            f"Qwen-Image-2.1 files missing under {d}: {missing}. "
            f"Place the files there or set WDNAI_QWEN_IMAGE_21_DIR to point at a custom path."
        )


# --------------------------------------------------------------------------- #
# Public catalog
# --------------------------------------------------------------------------- #
class SystemLanguageModel:
    """Catalog of preset model specifications.

    Exposes:

    - ``default``                — placeholder ``ModelSpec`` (no paths; for demos).
    - ``qwen2_5_0_5b()``         — Qwen2.5-0.5B-Instruct
    - ``qwen3_1_7b()``           — Qwen3-1.7B
    - ``qwen3_4b()``             — Qwen3-4B (pure text) bound to
                                    ``/userdata/models/Qwen3-4B``
    - ``qwen2_5_vl_3b()``        — Qwen2.5-VL-3B (image)
    - ``qwen2_5_vl_7b()``        — Qwen2.5-VL-7B (image)
    - ``qwen3_vl_2b()``          — Qwen3-VL-2B (image)
    - ``qwen3_vl_4b_template()`` — Qwen3-VL-4B template (no default paths)
    - ``qwen3_vl_4b()``          — Qwen3-VL-4B bound to ``$WDNAI_MODELS_DIR/Qwen3-VL-4B``
                                    (default ``/userdata/models/Qwen3-VL-4B``; override
                                     the root via ``WDNAI_MODELS_DIR`` or pin a custom
                                     path via ``WDNAI_QWEN3_VL_4B_DIR``)
    - ``qwen3_5_vl_4b()``        — Qwen3.5-VL-4B (image + text) bound to
                                    ``$WDNAI_MODELS_DIR/Qwen3.5-4B``
                                    (default ``/userdata/models/Qwen3.5-4B``;
                                    override the root via ``WDNAI_MODELS_DIR``
                                    or pin a custom path via
                                    ``WDNAI_QWEN3_5_4B_VL_DIR``). Powers
                                    ``SoftwareCapability.TEXT_CHAT``,
                                    ``VISION_CHAT`` and ``TOOL_CALL``
                                    capabilities from a single RKNN.
    - ``qwen3_tts()``            — Qwen3-TTS bound to
                                    ``$WDNAI_MODELS_DIR/Qwen3-TTS`` (default
                                    ``/userdata/models/Qwen3-TTS``; override the
                                    root via ``WDNAI_MODELS_DIR`` or pin a custom
                                    path via ``WDNAI_QWEN3_TTS_DIR``); the 4
                                    ``.1828.rknn`` files are looked up inside
                                    that directory.
    - ``qwen_2_5_omni_3b()``     — Qwen2.5-Omni-3B (audio) bound to
                                    ``$WDNAI_MODELS_DIR/Qwen2.5-Omni-3B`` (default
                                    ``/userdata/models/Qwen2.5-Omni-3B``; override the
                                    root via ``WDNAI_MODELS_DIR`` or pin a custom path
                                    via ``WDNAI_QWEN_2_5_OMNI_3B_DIR``). The SDK
                                    auto-prepends ``<audio>`` to the user prompt.
    - ``locate_anything()``     — LocateAnything (grounding, image→bbox) bound to
                                    ``$WDNAI_MODELS_DIR/LocateAnything-3B`` (default
                                    ``/userdata/models/LocateAnything-3B``; override the
                                    root via ``WDNAI_MODELS_DIR`` or pin a custom path
                                    via ``WDNAI_LOCATE_ANYTHING_DIR``).
    - ``qwen_image_21_512()``   — Qwen-Image-2.1 (text→image, 512×512) bound to
                                    ``$WDNAI_MODELS_DIR/Qwen-Image-2.1`` (default
                                    ``/userdata/models/Qwen-Image-2.1``; also picked up
                                    from ``/home/rockchip/models/Qwen-Image-2.1``;
                                    override via ``WDNAI_QWEN_IMAGE_21_DIR``).
    """

    #: Default placeholder (no paths; for the ``model=default`` syntax in demo_vlm)
    default: ModelSpec = ModelSpec(
        name="default",
        kind="vlm",
        variant="qwen2.5-vl",
        image_size=392,
    )

    # ---- Text LLM -----------------------------------------------------
    @classmethod
    def qwen2_5_0_5b(cls) -> ModelSpec:
        return _qwen2_5_spec()

    @classmethod
    def qwen3_1_7b(cls) -> ModelSpec:
        return _qwen3_spec()

    @classmethod
    def qwen3_4b(cls) -> ModelSpec:
        """Return a :class:`ModelSpec` pre-bound to the Qwen3-4B files at
        ``/userdata/models/Qwen3-4B/`` (override via ``WDNAI_QWEN3_4B_DIR``);
        raises :class:`FileNotFoundError` listing any missing files.
        """
        _verify_qwen3_4b_files()
        return _qwen3_4b_spec().with_paths(**_default_qwen3_4b_paths())

    @classmethod
    def qwen3_5_vl_4b(cls) -> ModelSpec:
        """Return a :class:`ModelSpec` pre-bound to the Qwen3.5-VL-4B files.

        Vision-language model (image-in, text-out) that also handles
        pure-text inference — the VLM-mode LLM is a regular LLM with
        MRoPE baked into the graph for image position encoding; without
        vision tokens MRoPE reduces to standard positional encoding,
        so this single RKNN powers both text-chat and video-describe
        capabilities. Files are expected at
        ``$WDNAI_MODELS_DIR/Qwen3.5-4B/``. Override the root with
        ``WDNAI_MODELS_DIR`` or pin a custom path with
        ``WDNAI_QWEN3_5_4B_VL_DIR``. Missing files raise
        :class:`FileNotFoundError` listing them. Auto-discovers the
        optional 768×768 vision pair at construction time; the
        384×384 pair is always required.

        Sampling defaults follow
        ``examples/Qwen3_5_VL/cpp/main.cc``: ``top_k=1``, ``top_p=0.9``,
        ``temperature=1e-6``, ``repeat_penalty=1.0``,
        ``frequency_penalty=0.0``, ``presence_penalty=1.5``.
        """
        _verify_qwen3_5_4b_vl_files()
        return _qwen3_5_vl_spec().with_paths(**_default_qwen3_5_4b_vl_paths())

    @classmethod
    def qwen3_5_vl_9b(cls) -> ModelSpec:
        """Return a :class:`ModelSpec` pre-bound to the Qwen3.5-VL-9B
        two-stage export.

        Both LLM stages and the vision encoder are placed under
        ``$WDNAI_MODELS_DIR/Qwen3.5-VL-9B/`` (override via
        ``WDNAI_QWEN3_5_VL_9B_DIR``). The model **requires two RK182x
        cards** (``requires_cards=2``); stage 0 + vision run on the
        first, stage 1 on the second. Missing files raise
        :class:`FileNotFoundError` listing them.
        """
        _verify_qwen3_5_9b_vl_files()
        return _qwen3_5_vl_9b_spec().with_paths(**_default_qwen3_5_9b_vl_paths())

    # ---- Vision LLM ----------------------------------------------------
    @classmethod
    def qwen2_5_vl_3b(cls) -> ModelSpec:
        return _qwen2_5vl_spec()

    @classmethod
    def qwen2_5_vl_7b(cls) -> ModelSpec:
        return _qwen2_5_vl_7b_spec()

    @classmethod
    def qwen3_vl_2b(cls) -> ModelSpec:
        return _qwen3vl_spec()

    @classmethod
    def qwen3_vl_4b_template(cls) -> ModelSpec:
        """4B template with no default paths; users can apply .with_paths(...) on top."""
        return _qwen3vl_4b_spec()

    @classmethod
    def qwen3_vl_4b(cls) -> ModelSpec:
        """Return a :class:`ModelSpec` pre-bound to the fixed Qwen3-VL-4B files.

        The files are expected at ``$WDNAI_MODELS_DIR/Qwen3-VL-4B/`` (default
        ``/userdata/models/Qwen3-VL-4B/`` when ``WDNAI_MODELS_DIR`` is unset;
        override the root with ``WDNAI_MODELS_DIR`` or pin a custom path with
        ``WDNAI_QWEN3_VL_4B_DIR``). If any expected file is missing a
        :class:`FileNotFoundError` is raised listing them.
        """
        _verify_qwen3_vl_4b_files()
        return _qwen3vl_4b_spec().with_paths(**_default_qwen3_vl_4b_paths())

    @classmethod
    def qwen3_tts(cls) -> ModelSpec:
        """Return a :class:`ModelSpec` pre-bound to the fixed Qwen3-TTS files.

        The directory is expected at ``/userdata/models/Qwen3-TTS/`` (override
        the root with the ``WDNAI_MODELS_DIR`` environment variable, or pin
        the directory itself with ``WDNAI_QWEN3_TTS_DIR``). The TTS
        backend derives the four ``<role>.1828.rknn`` / ``<role>.weight`` paths
        from that directory at construction time.
        """
        return ModelSpec(
            name="qwen3-tts",
            kind="tts",
            variant="qwen3-tts",
            target="rk1820",
            core_mask=0xFF,
            tts_model_dir=str(_qwen3_tts_dir()),
        )

    # ---- Omni (audio language model) ------------------------------------
    @classmethod
    def qwen_2_5_omni_3b(cls) -> ModelSpec:
        """Return a :class:`ModelSpec` pre-bound to the fixed Qwen2.5-Omni-3B files.

        Audio language model (wav-in, text-out). The files are expected at
        ``$WDNAI_MODELS_DIR/Qwen2.5-Omni-3B/`` (default
        ``/userdata/models/Qwen2.5-Omni-3B/`` when ``WDNAI_MODELS_DIR`` is unset;
        override the root with ``WDNAI_MODELS_DIR`` or pin a custom path with
        ``WDNAI_QWEN_2_5_OMNI_3B_DIR``). If any expected file is missing a
        :class:`FileNotFoundError` is raised listing them.

        The SDK auto-prepends ``<audio>`` to the user prompt before sending
        it to the LLM, so callers just write a plain text instruction::

            from wdnai import AudioAttachment
            async with LLMSession(model=SystemLanguageModel.qwen_2_5_omni_3b()) as sess:
                text = await sess.respond(
                    prompt="请描述这段音频的内容",
                    attachment=[AudioAttachment(pcm, mime_type="audio/pcm")],
                    sample_rate=..., n_channels=..., sample_width=...,
                )
        """
        _verify_qwen_2_5_omni_3b_files()
        return ModelSpec(
            name="qwen2.5-omni-3b",
            kind="omni",
            variant="qwen2.5-omni",
            target="rk1820",
            core_mask=0xFF,
            vocab_size=151936,
            special_eos_id=151645,
            max_context_len=1024,
            **_default_qwen_2_5_omni_3b_paths(),
            options={
                "max_new_tokens": 1024,
                "top_k": 1,
                "top_p": 0.9,
                "temperature": 1.0,
                "repeat_penalty": 1.2,
            },
        )

    # ---- ASR -----------------------------------------------------------
    @classmethod
    def qwen3_asr(cls) -> ModelSpec:
        """Return a :class:`ModelSpec` bound to the on-device Qwen3-ASR files.

        Model dir defaults to ``$WDNAI_MODELS_DIR/Qwen3-ASR/``
        (resolves to ``/userdata/models/Qwen3-ASR/``), overridable via
        the ``WDNAI_QWEN3_ASR_DIR`` env var — mirrors every other model
        dir so the folder name is *not* load-bearing. Layout::

            encoder.1828.rknn            + encoder.weight         (offline)
            encoder_online.1828.rknn     + encoder_online.weight  (online / streaming)
            llm.1828.rknn                + llm.weight
            llm.embed.bin                (vocab_size × dim) fp16
            llm.tokenizer.gguf           GPT-2 BPE + ByteLevel
            mel_128_filters.txt          n_mels × N_FFT/2+1

        Set ``options["use_online_encoder"] = True`` to switch to the
        streaming encoder (see ``ASRBackend.run_audio``).
        """
        d = _qwen3_asr_dir()
        return ModelSpec(
            name="qwen3-asr",
            kind="asr",
            variant="qwen3-asr",
            target="rk1820",
            core_mask=0xFF,
            vocab_size=151936,
            special_eos_id=151645,
            max_context_len=1024 * 24,
            asr_model_dir=str(d),
            tokenizer_path=str(d / "llm.tokenizer.gguf"),
            embed_path=str(d / "llm.embed.bin"),
            options={
                "max_new_tokens": 1024 * 6,
                "top_k": 1,
                "top_p": 0.9,
                "temperature": 1.0,
                "repeat_penalty": 1.2,
                # Streaming knobs (only used when use_online_encoder=True).
                #
                # ``use_online_encoder`` defaults to **True**: the deployed
                # Qwen3-ASR model folder ships the *streaming* encoder
                # (``encoder_online.1828.rknn``) and the whole point of the
                # card is incremental decoding. The encoder choice is a
                # **load-time** decision (which RKNN file to put on the
                # NPU), so it cannot be switched per request — the flag
                # lives here, in the SDK, and the worker manifest only says
                # "this is the qwen3_asr model".
                #
                # Set ``use_online_encoder=False`` to use the one-shot
                # ``encoder.1828.rknn`` instead (offline mode, whole file
                # in one pass, no incremental output).
                "use_online_encoder": True,
                "online_chunk_seconds": 1.0,
                "online_max_new_tokens": 50,
                "online_rollback_tokens": 5,
                "online_rollback_rounds": 4,
                # Idle TTL for an **incremental** streaming session
                # (``asr_stream_id``). The worker is a long-lived process,
                # so a session whose caller stopped feeding would keep
                # its audio embedding history alive forever; this is how
                # it gets reaped.
                "asr_stream_ttl_sec": 120.0,
            },
        )

    @classmethod
    def qwen3_vl_embedding(cls) -> ModelSpec:
        """Return a :class:`ModelSpec` pre-bound to the Qwen3-VL-Embedding-2B files.

        Retrieval embedding model (no token generation): the LLM is run via the
        plain ``RKNN3Lite`` path with manual inputs and returns a normalized
        2048-dim vector. Expected at ``/userdata/models/Qwen3-VL-Embedding-2B/``
        (override the root with the ``WDNAI_MODELS_DIR`` environment variable,
        or pin the directory itself with ``WDNAI_QWEN3_VL_EMBEDDING_2B_DIR``);
        missing files raise :class:`FileNotFoundError`.
        """
        _verify_qwen3_vl_embedding_2b_files()
        return ModelSpec(
            name="qwen3-vl-embedding-2b",
            kind="embedding",
            variant="qwen3-vl-embedding",
            target="rk1820",
            core_mask=0xFF,
            vocab_size=151936,
            image_size=576,
            max_context_len=2048,
            **{**_default_qwen3_vl_embedding_2b_paths()},
        )

    # ---- Grounding (bbox) ----------------------------------------------
    @classmethod
    def locate_anything(cls) -> ModelSpec:
        """Return a :class:`ModelSpec` pre-bound to the LocateAnything files.

        Grounding model (image-in, text-out with normalized bounding-box
        coords). Files are expected at
        ``$WDNAI_MODELS_DIR/LocateAnything-3B/`` (default
        ``/userdata/models/LocateAnything-3B/`` when ``WDNAI_MODELS_DIR`` is
        unset; override the root with ``WDNAI_MODELS_DIR`` or pin a custom
        path with ``WDNAI_LOCATE_ANYTHING_DIR``). If any expected file is
        missing a :class:`FileNotFoundError` is raised listing them.

        Sampling defaults follow ``examples/LocateAnything/cpp/main.cc``
        ``SAMPLE_PARAMS``: ``top_k=1``, ``top_p=0.9``, ``temperature=1e-6``,
        ``repeat_penalty=1.05``, ``enable_thinking=False``. The
        near-greedy temperature mirrors the rationale in
        ``_qwen3vl_4b_spec``: structured bbox token output degrades at
        higher values.
        """
        _verify_locate_anything_files()
        return _locate_anything_spec().with_paths(**_default_locate_anything_paths())

    # ---- Machine translation --------------------------------------------
    @classmethod
    def hy_mt_1_5(cls) -> ModelSpec:
        """Return a :class:`ModelSpec` pre-bound to the HY-MT1.5-1.8B files.

        Hunyuan-Dense machine-translation model (text → text). Files are
        expected at ``$WDNAI_MODELS_DIR/HY_MT_1_5/`` (default
        ``/userdata/models/HY_MT_1_5/`` when ``WDNAI_MODELS_DIR`` is unset;
        override the root with ``WDNAI_MODELS_DIR`` or pin a custom path
        with ``WDNAI_HY_MT_1_5_DIR``). If any expected file is missing a
        :class:`FileNotFoundError` is raised listing them.

        The prompt template is the **Hunyuan** one
        (``<｜hy_begin▁of▁sentence｜>`` / ``<｜hy_User｜>`` /
        ``<｜hy_Assistant｜>``), **not** ChatML — see
        :data:`HY_MT_CHAT_TEMPLATE`. Sampling is greedy
        (``top_k=1``, ``temperature=1e-6``) so the translation is
        reproducible.
        """
        _verify_hy_mt_1_5_files()
        return _hy_mt_1_5_spec().with_paths(**_default_hy_mt_1_5_paths())

    # ---- Structured decision -------------------------------------------
    @classmethod
    def intern_decision_4b(cls) -> ModelSpec:
        """Return a :class:`ModelSpec` pre-bound to the Intern-Decision-4B files.

        Structured-decision model (state → options + probabilities): it
        does **not** generate text — for each decision field it emits the
        logits of the 62 candidate symbols, which the SDK turns into
        option probabilities. Three field types are supported:
        ``choice`` (pick one), ``score`` (probability-weighted numeric
        score) and ``noul`` (yes/no + affirmative probability).

        Files are expected at ``$WDNAI_MODELS_DIR/Intern_Decision-4B/``
        (default ``/userdata/models/Intern_Decision-4B/`` when
        ``WDNAI_MODELS_DIR`` is unset; override the root with
        ``WDNAI_MODELS_DIR`` or pin a custom path with
        ``WDNAI_INTERN_DECISION_4B_DIR``). If any expected file is
        missing a :class:`FileNotFoundError` is raised listing them.

        The package holds **two** RKNN pairs — the text model
        (``-llm``) and the vision encoder (``-vision``) — plus the
        tokenizer and the embedding table. The vision encoder is loaded
        lazily (only when a request actually carries an image) and
        released again once the images are encoded, mirroring the C++
        reference implementation.

        The chat template is **empty** on purpose: this model family
        builds its whole prompt on the CPU (see
        :mod:`wdnai.decision`), so the C-side ChatML wrapper must not
        be applied.
        """
        _verify_intern_decision_4b_files()
        return _intern_decision_4b_spec().with_paths(
            **_default_intern_decision_4b_paths()
        )

    # ---- Depth estimation -----------------------------------------------
    @classmethod
    def depth_anything_v3(cls) -> ModelSpec:
        """Return a :class:`ModelSpec` pre-bound to the Depth Anything V3 files.

        Monocular depth-estimation model (image-in, depth-map-out). The
        export is split into three RKNN stages — ``local`` (per-image
        tokens), ``global`` (cross-view fusion) and ``head`` (depth +
        confidence) — and all three are expected at
        ``$WDNAI_MODELS_DIR/Depth-Anything-V3/`` (default
        ``/userdata/models/Depth-Anything-V3/`` when ``WDNAI_MODELS_DIR`` is
        unset; override the root with ``WDNAI_MODELS_DIR`` or pin a custom
        path with ``WDNAI_DEPTH_ANYTHING_V3_DIR``). If any expected file is
        missing a :class:`FileNotFoundError` is raised listing them.

        The ``.weight`` file of each stage sits next to its ``.rknn`` with
        the same stem — the backend derives it by replacing the ``.rknn``
        suffix, mirroring the C++ demo's ``make_weight_path``.
        """
        _verify_depth_anything_v3_files()
        return _depth_anything_v3_spec().with_paths(**_default_depth_anything_v3_paths())
    # ---- Text-to-image (Qwen-Image-2.1) ---------------------------------
    @classmethod
    def qwen_image_21_512(cls) -> ModelSpec:
        """Return a :class:`ModelSpec` pre-bound to the Qwen-Image-2.1 package.

        Text-to-image model (prompt-in, 512×512 image-out) for RKNN3 /
        rk1820. The package is expected at
        ``$WDNAI_MODELS_DIR/Qwen-Image-2.1`` (default
        ``/userdata/models/Qwen-Image-2.1``; also picked up automatically
        from ``/home/rockchip/models/Qwen-Image-2.1``). Override with
        ``WDNAI_QWEN_IMAGE_21_DIR`` (or ``QI_MODEL_DIR``). Missing files
        raise :class:`FileNotFoundError` listing them.

        The whole pipeline is driven by the packaged C++ runtime
        (``runtime/qwen_image_rknn``): the text encoder produces
        ``encoder_hidden_states``, four resident DiT segments run the
        flow-match loop (generation does **zero** load / model_init), and
        four streaming VAE segments decode the latents.

        Usage::

            from wdnai import LLMLocalSession, SystemLanguageModel

            session = LLMLocalSession(model=SystemLanguageModel.qwen_image_21_512())
            text, attachments = await session.run_prompt("a cute cat")
            bgr = attachments[0].images[0]

        ``options`` carries the runtime knobs:

        ``steps``
            flow-match steps (default 8 — ~31 s per image after the
            one-off initialisation; 20 is noticeably cleaner, 40 is the
            upstream default).
        ``fm_mu``
            the scheduler's ``mu`` for the 512 route — **0.5387096774**,
            not the 1024 route's 0.6935483871.
        ``noise_fresh``
            use a fresh random initial noise per image (default ``True``).
            Set to ``False`` to get bit-reproducible output for a fixed
            prompt.
        ``card0`` / ``card1``
            the two RK1828 devices (PCI ids) — card 0 hosts the resident
            TE + the DiT head, card 1 hosts the 4 DiT segments and then
            the streaming VAE.
        """
        _verify_qwen_image_21_files()
        return ModelSpec(
            name="qwen-image-2.1-512",
            kind="image",
            variant="qwen-image-2.1",
            target="rk1820",
            core_mask=0xFF,
            image_model_dir=str(_qwen_image_21_dir()),
            options={
                "steps": 8,
                "fm_mu": "0.5387096774",
                "noise_fresh": True,
                "noise_seed": 0,
            },
        )
