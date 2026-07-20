"""Model configuration and the preset catalog.

Each preset is a :class:`ModelSpec` dataclass that holds all the configuration
needed for a given variant (paths, tokenizer, sampling parameters, vision
parameters, etc.). ``SystemLanguageModel`` is the catalog entry point, exposing
``qwen2_5_*`` / ``qwen3_*`` / ``qwen2_5_vl_*`` / ``qwen3_vl_*`` factories and the
``default`` placeholder.

Typical usage::

    from rkai import SystemLanguageModel, LLMSession, Attachment

    model = SystemLanguageModel.qwen3_vl_2b().with_paths(
        rknn_llm_path="/userdata/Qwen3-VL-2B-llm.rknn3",
        rknn_vision_path="/userdata/Qwen3-VL-2B-vision.rknn3",
        tokenizer_path="/userdata/qwen3_VL_2B",
        embed_path="/userdata/Qwen3-VL-2B-llm.embed.bin",
    )
    session = LLMSession(model=model)
    out = await session.respond(msg="describe the image", attachment=Attachment(img))
"""
from __future__ import annotations

import logging
import os as _os
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict


# --------------------------------------------------------------------------- #
# Base spec
# --------------------------------------------------------------------------- #
@dataclass
class ModelSpec:
    """All static configuration + dynamic paths needed for a single model."""

    name: str
    kind: str                                # "llm" | "vlm"
    variant: str                             # "qwen2.5" | "qwen3" | "qwen2.5-vl" | "qwen3-vl"

    # ---- LLM resources ----
    # kind=llm uses rknn_path / weight_path; kind=vlm uses rknn_llm_path / rknn_vision_path
    rknn_path: str = ""
    weight_path: str = ""
    rknn_llm_path: str = ""
    weight_llm_path: str = ""
    rknn_vision_path: str = ""
    weight_vision_path: str = ""
    embed_path: str = ""
    tokenizer_path: str = ""

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
            "enable_thinking": False,   # Qwen3 thinking-mode toggle (supported by rknn3lite.session_run)
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


# --------------------------------------------------------------------------- #
# Predefined paths: Qwen3-VL-4B lives at a fixed on-device location
# (``/userdata/models/Qwen3-VL-4B/``). No network access is performed at any
# point — the user is expected to place the files there manually.
# Override the root with the ``RKAI_MODELS_DIR`` environment variable.
# --------------------------------------------------------------------------- #
_QWEN3_VL_4B_DIR_NAME = "Qwen3-VL-4B"
_QWEN3_VL_4B_REQUIRED = (
    "llm_Qwen3-VL-4B.1828.rknn",
    "llm_Qwen3-VL-4B.weight",
    "vision_Qwen3-VL-4B.1828.rknn",
    "vision_Qwen3-VL-4B.weight",
    "Qwen3-VL-4B.embed.bin",
    "Qwen3-VL-4B.tokenizer.gguf",
)


def _models_root() -> Path:
    """Return the on-device root directory holding all fixed-location models."""
    return Path(_os.environ.get("RKAI_MODELS_DIR", "/userdata/models"))


def _qwen3_vl_4b_dir() -> Path:
    return _models_root() / _QWEN3_VL_4B_DIR_NAME


def _default_qwen3_vl_4b_paths() -> dict:
    """Return absolute paths to the Qwen3-VL-4B files under the fixed models root."""
    d = _qwen3_vl_4b_dir()
    return {
        "rknn_llm_path":   str(d / "llm_Qwen3-VL-4B.1828.rknn"),
        "rknn_vision_path": str(d / "vision_Qwen3-VL-4B.1828.rknn"),
        "tokenizer_path":  str(d / "Qwen3-VL-4B.tokenizer.gguf"),
        "embed_path":      str(d / "Qwen3-VL-4B.embed.bin"),
    }


def _verify_qwen3_vl_4b_files() -> None:
    """Raise :class:`FileNotFoundError` listing any missing Qwen3-VL-4B files."""
    d = _qwen3_vl_4b_dir()
    missing = [f for f in _QWEN3_VL_4B_REQUIRED if not (d / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Qwen3-VL-4B files missing under {d}: {missing}. "
            f"Place the files there or set RKAI_MODELS_DIR to override the root."
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
    - ``qwen2_5_vl_3b()``        — Qwen2.5-VL-3B (image)
    - ``qwen2_5_vl_7b()``        — Qwen2.5-VL-7B (image)
    - ``qwen3_vl_2b()``          — Qwen3-VL-2B (image)
    - ``qwen3_vl_4b_template()`` — Qwen3-VL-4B template (no default paths)
    - ``qwen3_vl_4b()``          — Qwen3-VL-4B bound to ``/userdata/models/Qwen3-VL-4B``
                                    (the user is expected to place the files there;
                                     override the root with ``RKAI_MODELS_DIR``)
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

        The files are expected at ``/userdata/models/Qwen3-VL-4B/`` (override the
        root with the ``RKAI_MODELS_DIR`` environment variable). If any expected
        file is missing a :class:`FileNotFoundError` is raised listing them.
        """
        _verify_qwen3_vl_4b_files()
        return _qwen3vl_4b_spec().with_paths(**_default_qwen3_vl_4b_paths())
