"""Load a HuggingFace ``PreTrainedTokenizerFast`` from the .tokenizer.gguf shipped with rknn3 models.

The GGUF file's ``tokenizer.ggml.*`` fields contain the three GPT-2 BPE building
blocks: ``tokens`` (vocabulary), ``merges`` (merge rules), ``model`` (e.g. "gpt2"
or "llama"). This module writes them to a temp directory, lets the ``tokenizers``
library build a GPT-2-style BPE, and finally wraps it as a
``PreTrainedTokenizerFast`` whose behavior matches ``AutoTokenizer.from_pretrained``.

It also reads ``tokenizer.chat_template``, ``eos/bos/padding token_id`` so the
tokenizer stays in sync with RKNN3 Lite's ChatML template.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import List, Tuple

import numpy as np
from gguf import GGUFReader


def _read_gguf_kv(path: str) -> dict:
    reader = GGUFReader(path)
    out = {}
    for k, v in reader.fields.items():
        try:
            c = v.contents() if hasattr(v, "contents") else v
        except Exception:
            c = None
        out[k] = c
    return out


def load_tokenizer_from_gguf(gguf_path: str):
    """Return a tokenizer object whose behavior matches ``AutoTokenizer.from_pretrained``."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre
    from tokenizers.decoders import ByteLevel as ByteLevelDec
    from transformers import PreTrainedTokenizerFast

    kv = _read_gguf_kv(gguf_path)

    model_type = kv.get("tokenizer.ggml.model", "gpt2")
    pre_type = kv.get("tokenizer.ggml.pre", "qwen2")
    tokens = kv.get("tokenizer.ggml.tokens") or []
    merges = kv.get("tokenizer.ggml.merges") or []
    eos_id = int(kv.get("tokenizer.ggml.eos_token_id") or 151645)
    bos_id = int(kv.get("tokenizer.ggml.bos_token_id") or eos_id)
    pad_id = int(kv.get("tokenizer.ggml.padding_token_id") or eos_id)
    add_bos = bool(kv.get("tokenizer.ggml.add_bos_token", False))
    chat_template = kv.get("tokenizer.chat_template")

    # ---- Dump vocab / merges to a temp directory for the BPE builder ----
    tmp = tempfile.mkdtemp(prefix="rksdk_gguf_tok_")
    vocab_file = os.path.join(tmp, "vocab.json")
    merges_file = os.path.join(tmp, "merges.txt")

    vocab = {tok: i for i, tok in enumerate(tokens)}
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open(merges_file, "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for m in merges:
            f.write(m + "\n")

    # tokenizers.BPE expects a list of (str, str) tuples, not a list of "a b" strings.
    merges_pairs: List[Tuple[str, str]] = []
    for m in merges:
        parts = m.split(" ", 1)
        if len(parts) == 2:
            merges_pairs.append((parts[0], parts[1]))

    tok_obj = Tokenizer(BPE(vocab=vocab, merges=merges_pairs, unk_token=None))
    tok_obj.pre_tokenizer = ByteLevelPre(add_prefix_space=False, use_regex=True)
    tok_obj.decoder = ByteLevelDec()

    # ---- Multimodal placeholders used by Qwen3-VL: must be registered as
    # non-splittable special tokens. Otherwise the ByteLevel pre-tokenizer
    # splits them into ~7 regular tokens and rknn3 session can't find the
    # expected ID. ----
    multimodal_tokens = [
        "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
        "<|image_pad|>", "<|video_pad|>",
        "<|im_start|>", "<|im_end|>",
    ]
    # Defensive: drop any token that isn't in the GGUF vocabulary
    extra = [t for t in multimodal_tokens if t in vocab]

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        eos_token=decode_id_to_str(tokens, eos_id),
        bos_token=decode_id_to_str(tokens, bos_id),
        pad_token=decode_id_to_str(tokens, pad_id),
        unk_token=decode_id_to_str(tokens, 0),
        add_bos_token=add_bos,
        additional_special_tokens=extra,
    )
    if chat_template:
        tokenizer.chat_template = chat_template
    return tokenizer


def decode_id_to_str(tokens, idx: int) -> str:
    if 0 <= idx < len(tokens):
        return tokens[idx]
    return ""
