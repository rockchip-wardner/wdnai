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
from typing import Dict, List, Tuple

import numpy as np
from gguf import GGUFReader


#: ``tokenizer.ggml.pre`` → 预分词正则。
#:
#: 对齐 llama.cpp ``llama-vocab.cpp`` 里的 ``LLAMA_VOCAB_PRE_TYPE_*``
#: （以及 HF ``transformers`` 的 ``Qwen2Tokenizer.PRETOKENIZE_REGEX``）。
#:
#: ⚠️ **不能一律用 ByteLevel 自带的 GPT-2 正则**。两条正则切法不同：
#: ``e.g.`` 在 Qwen 正则下切成 ``e`` + ``.g`` + ``.``（3 个 token），在
#: GPT-2 正则下切成 ``e`` + ``.`` + ``g`` + ``.``（4 个 token）—— token
#: 数与 id 全都错位，进而改变模型输出的 logits 与最终概率。
#:
#: 只收录**单正则**类型；多正则类型（deepseek3 / hunyuan-dense 等）
#: 仍走 ByteLevel 默认行为。
_PRETOKENIZER_REGEXES: Dict[str, str] = {
    # LLAMA_VOCAB_PRE_TYPE_QWEN2
    "qwen2": (
        r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
        r"|[^\r\n\p{L}\p{N}]?\p{L}+"
        r"|\p{N}"
        r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
        r"|\s*[\r\n]+"
        r"|\s+(?!\S)"
        r"|\s+"
    ),
    # LLAMA_VOCAB_PRE_TYPE_QWEN35
    "qwen35": (
        r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
        r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
        r"|\p{N}"
        r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
        r"|\s*[\r\n]+"
        r"|\s+(?!\S)"
        r"|\s+"
    ),
}


def _build_pre_tokenizer(pre_type: str):
    """Build the pre-tokenizer that matches the GGUF's ``tokenizer.ggml.pre``.

    ``ByteLevel(use_regex=False)`` 只做 byte→unicode 映射（不再二次切分），
    切词完全交给前面那条 ``Split``。
    """
    from tokenizers import Regex
    from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre
    from tokenizers.pre_tokenizers import Sequence as PreSequence
    from tokenizers.pre_tokenizers import Split as PreSplit

    pattern = _PRETOKENIZER_REGEXES.get(pre_type)
    if not pattern:
        # 未知 / 多正则类型：退回 ByteLevel 自带的 GPT-2 正则。
        return ByteLevelPre(add_prefix_space=False, use_regex=True)
    return PreSequence([
        PreSplit(Regex(pattern), behavior="isolated", invert=False),
        ByteLevelPre(add_prefix_space=False, use_regex=False),
    ])


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


def load_tokenizer_from_gguf(gguf_path: str, extra_special_tokens=None):
    """Return a tokenizer object whose behavior matches ``AutoTokenizer.from_pretrained``.

    ``extra_special_tokens`` — optional list of **additional** control
    tokens to register as non-splittable specials for *this* instance
    only.

    Why per-instance and not global: some model families need a control
    token registered (``<think>`` must be **one** id, not three) while
    the Qwen chat path needs the *literal* ``<think>`` text back in the
    decoded output — a globally registered special would be erased by
    ``skip_special_tokens=True``. Registering per instance lets the
    Intern-Decision backend get single-token control markers without
    touching the Qwen behaviour.

    Only tokens that already exist in the vocabulary are registered:
    ``add_tokens`` would hand a missing marker a *fresh* id the model
    has never seen, which is worse than leaving it split.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.decoders import ByteLevel as ByteLevelDec
    from transformers import PreTrainedTokenizerFast

    kv = _read_gguf_kv(gguf_path)

    model_type = kv.get("tokenizer.ggml.model", "gpt2")
    # GGUF 没写 ``pre`` 时按 llama.cpp 的 "default" 处理（退回 ByteLevel
    # 自带的 GPT-2 正则），**不要**默认成 qwen2 —— 那会让不存在的 pre
    # 值被静默当成 Qwen 正则。
    pre_type = kv.get("tokenizer.ggml.pre", "default")
    tokens = kv.get("tokenizer.ggml.tokens") or []
    merges = kv.get("tokenizer.ggml.merges") or []
    eos_id = int(kv.get("tokenizer.ggml.eos_token_id") or 151645)
    bos_id = int(kv.get("tokenizer.ggml.bos_token_id") or eos_id)
    pad_id = int(kv.get("tokenizer.ggml.padding_token_id") or eos_id)
    add_bos = bool(kv.get("tokenizer.ggml.add_bos_token", False))
    chat_template = kv.get("tokenizer.chat_template")

    # ---- Dump vocab / merges to a temp directory for the BPE builder ----
    tmp = tempfile.mkdtemp(prefix="wdnai_gguf_tok_")
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
    tok_obj.pre_tokenizer = _build_pre_tokenizer(pre_type)
    tok_obj.decoder = ByteLevelDec()

    # ---- Multimodal placeholders used by Qwen3-VL: must be registered as
    # non-splittable special tokens. Otherwise the ByteLevel pre-tokenizer
    # splits them into ~7 regular tokens and rknn3 session can't find the
    # expected ID. ----
    multimodal_tokens = [
        "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
        "<|image_pad|>", "<|video_pad|>",
        "<img>", "</img>", "<IMG_CONTEXT>", "<image>",
        "</c>", "<switch>",
    ]
    # Tokens already in the vocab register as ``additional_special_tokens``
    # (the ByteLevel pre-tokenizer leaves them unsplit). Tokens not yet in
    # the vocab — only ``<image>`` here — get added via ``add_tokens``,
    # which extends the vocab and assigns a fresh ID. Both paths end up
    # with the token registered in ``added_tokens_encoder`` so the Python
    # tokenizer treats it as one ID instead of splitting it into ``<``,
    # the inner word, ``>``.
    in_vocab = [t for t in multimodal_tokens if t in vocab]
    missing = [t for t in multimodal_tokens if t not in vocab]
    # ChatML control markers must be ``special`` too: the model can emit
    # ``<|im_start|>`` (a new turn boundary) when it degenerates, and with
    # ``skip_special_tokens=True`` a non-special marker is *not* skipped —
    # the literal ``"<|im_start|>"`` then leaks into the decoded text.
    # (``<|im_end|>`` is already special via ``eos_token`` below.)
    # ``<think>`` / ``</think>`` are deliberately NOT registered — callers
    # rely on them appearing in the text to split the thinking channel.
    chatml_control = [t for t in ("<|im_start|>",) if t in vocab]
    # Hunyuan control tokens (HY-MT1.5). Same failure mode as the Qwen
    # placeholders above: ``<｜hy_User｜>`` and friends are **real
    # vocabulary entries** (ids 120006 / 120007 / …), but the ByteLevel
    # pre-tokenizer splits them into ~5 regular tokens unless they are
    # registered as special.
    #
    # Only the ones that are already in the vocab are registered — unlike
    # the Qwen list above, a *missing* Hunyuan marker must **not** be
    # added: ``add_tokens`` would hand it a fresh id that the model has
    # never seen, which is worse than leaving it split.
    hunyuan_control = [
        t for t in (
            "<｜hy_begin▁of▁sentence｜>",   # 120000 — BOS
            "<｜hy_end▁of▁sentence｜>",     # 120001
            "<｜hy_▁pad▁｜>",                # 120002 — PAD
            "<｜hy_fim▁hole｜>",             # 120003
            "<｜hy_fim▁begin｜>",            # 120004
            "<｜hy_fim▁end｜>",              # 120005
            "<｜hy_User｜>",                 # 120006 — user turn
            "<｜hy_Assistant｜>",            # 120007 — assistant turn
            "<｜hy_EOT｜>",                  # 120008
            "<｜hy_place▁holder▁no▁2｜>",   # 120020 — EOS
            "<｜hy_place▁holder▁no▁3｜>",   # 120021 — system terminator
            "<｜hy_place▁holder▁no▁8｜>",   # 120026 — prompt terminator
        ) if t in vocab
    ]
    extra = in_vocab + chatml_control + hunyuan_control
    if missing:
        tok_obj.add_tokens(missing)

    # Per-instance extras (Intern-Decision's ``<decision>`` / ``<think>`` /
    # ``</think>``). Only in-vocab tokens are registered — see the
    # docstring. An already-in-vocab string keeps its original id (it is
    # only pinned in the added-tokens trie), so the ids stay aligned with
    # the model's embedding table.
    #
    # ⚠️ 必须是**合并**而不是替换！``tokenizer.add_special_tokens(
    # {"additional_special_tokens": known})`` 会把 ``additional_special_tokens``
    # **整个换成** ``known``，把上面注册好的 Qwen 多模态标记
    # （``<|vision_start|>`` / ``<|vision_end|>`` / ``<|image_pad|>``）、
    # ChatML 标记、混元标记统统丢掉。后果：C 侧拿 tokenizer callback 去
    # 定位图片占位区时按「拆成多个 token」去数，多模态 prompt 会被**少喂
    # 6 个 token**，图文决策的 logits 与官方对不上（实测 Intern-Decision
    # 图文：官方 embed 收到 360 个 token，SDK 只有 354 个）。
    if extra_special_tokens:
        known = [t for t in extra_special_tokens if t in vocab]
        if known:
            merged = list(dict.fromkeys(list(extra) + list(known)))
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tok_obj,
                eos_token=decode_id_to_str(tokens, eos_id),
                bos_token=decode_id_to_str(tokens, bos_id),
                pad_token=decode_id_to_str(tokens, pad_id),
                unk_token=None,
                add_bos_token=add_bos,
                additional_special_tokens=merged,
            )
            if chat_template:
                tokenizer.chat_template = chat_template
            return tokenizer

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        eos_token=decode_id_to_str(tokens, eos_id),
        bos_token=decode_id_to_str(tokens, bos_id),
        pad_token=decode_id_to_str(tokens, pad_id),
        unk_token=None,
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
