"""Qwen3-VL-Embedding backend — retrieval embeddings via RKNN3Lite.

The embedding LLM is loaded as a **plain** model (not the ``llm_mode``
session path used by Text/VLM backends) and run with 7 manual inputs padded
to ``max_seq_len``; ``num_logits_to_keep = S-1`` avoids the 255-token
clamping bug on sequences longer than 256 tokens. Ports the reference script
``examples/Qwen3_VL_Embedding/python/demo/rknn_qwen3_vl_embedding_video_test.py``.

Exposes two entry points (wired into :class:`wdnai.session.LLMSession`):

- ``embed_text(text)`` — normalized 2048-dim vector for a text query.
- ``embed_video_chunk(frames, text="")`` — normalized vector for a video
  chunk (list of BGR frames, optionally annotated with ``text``).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np

from ._common import load_embed_table, load_tokenizer, resolve_device_id

VOCAB_SIZE = 151936
DIM = 2048
FRAME_TOKENS = 324
DEFAULT_INSTRUCTION = "Represent the user's input."


def _normalize_embedding(emb: np.ndarray) -> np.ndarray:
    vec = np.asarray(emb, dtype=np.float32)
    if not np.all(np.isfinite(vec)):
        raise ValueError("embedding contains non-finite values")
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        raise ValueError("embedding has zero norm")
    return vec / norm


class EmbeddingBackend:
    """RKNN Qwen3-VL-Embedding-2B backend (kind="embedding")."""

    def __init__(
        self,
        spec,
        *,
        npu: Optional[int] = None,
        device_id: Optional[Union[bytes, str]] = None,
    ):
        from .._librknn import RKNN3Lite

        if not spec.rknn_llm_path or not spec.rknn_vision_path:
            raise ValueError(
                f"{spec.name!r} is missing rknn_llm_path / rknn_vision_path; "
                "fill them in via .with_paths(...)."
            )
        # Resolve NPU selection **after** RKNN3Lite is imported so the
        # native driver init order stays predictable (and doesn't race
        # with a co-resident wdnav process on dev boxes).
        resolved_device_id = resolve_device_id(RKNN3Lite, npu, device_id)
        self.spec = spec
        self._max_seq_len = spec.max_context_len
        self._core_mask = spec.core_mask
        self._target = spec.target
        self._instruction = DEFAULT_INSTRUCTION

        self.tokenizer = load_tokenizer(spec.tokenizer_path)
        self._embeds_data = load_embed_table(spec.embed_path, spec.vocab_size)

        self.rknn_vision = RKNN3Lite()
        vp = spec.rknn_vision_path
        if self.rknn_vision.load_rknn(vp, vp.replace(".1828.rknn", ".weight")) != 0:
            raise RuntimeError(f"vision load_rknn failed: {vp}")

        self.rknn_llm = RKNN3Lite()
        lp = spec.rknn_llm_path
        if self.rknn_llm.load_rknn(lp, lp.replace(".1828.rknn", ".weight")) != 0:
            raise RuntimeError(f"llm load_rknn failed: {lp}")

        if self.rknn_vision.init_runtime(
            target=self._target, core_mask=self._core_mask,
            device_id=resolved_device_id,
        ) != 0:
            raise RuntimeError("vision init_runtime failed")
        if self.rknn_llm.init_runtime(
            target=self._target, core_mask=self._core_mask,
            device_id=resolved_device_id,
        ) != 0:
            raise RuntimeError("llm init_runtime failed")

    # ------------------------------------------------------------------ #
    # Prompt / input assembly
    # ------------------------------------------------------------------ #
    def _build_prompt(self, text: str = "", with_video: bool = False,
                      frame_tokens: int = 0, n_frames: int = 0) -> str:
        user_content = ""
        if with_video:
            block = "<|vision_start|>" + ("<|image_pad|>" * frame_tokens) + "<|vision_end|>"
            user_content = " ".join([block] * n_frames)
        if text:
            user_content = user_content + (f" {text}" if user_content else text)
        return ("<|im_start|>system\n" + self._instruction + "<|im_end|>\n"
                "<|im_start|>user\n" + user_content + "<|im_end|>\n"
                "<|im_start|>assistant\n<|endoftext|>")

    def _vision_inference(self, frame_bgr: np.ndarray):
        import cv2
        from ..utils import letterbox_resize
        from .._librknn import RKNN3TensorType
        rt = self.rknn_vision
        if rt is None:
            raise RuntimeError("vision runtime not initialised")
        input_attr = rt.get_inputs_tensor_attr()[0]
        batch = int(input_attr.shape[0])
        h, w = int(input_attr.shape[1]), int(input_attr.shape[2])
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Letterbox instead of plain resize: the source frame may be any
        # aspect ratio (PyAV decode / user upload), and the Qwen3-VL
        # vision RKNN was trained on square inputs. Plain resize would
        # stretch a 16:9 frame into a square and distort embeddings.
        if (img.shape[0], img.shape[1]) != (h, w) or img.shape[0] != img.shape[1]:
            img = letterbox_resize(img, h)
        feature = img.reshape(1, h, w, 3)
        if batch == 2:
            feature = np.concatenate([feature, feature], axis=0)
        if input_attr.dtype != int(RKNN3TensorType.RKNN3_TENSOR_UINT8):
            feature = feature.astype(np.float16)
        outputs = rt.inference(inputs=[feature])
        if outputs is None or len(outputs) == 0:
            raise RuntimeError("vision inference returned no outputs")
        image_feature = np.float16(np.expand_dims(outputs[0], 0))
        deepstacks = [np.float16(outputs[i]) for i in (1, 2, 3)]
        return image_feature, deepstacks

    def _build_llm_inputs(self, prompt: str, video_feature=None,
                          deepstacks=None, image_mask=None):
        input_ids = self.tokenizer(
            prompt, return_tensors='np', add_special_tokens=False
        )['input_ids'][0]
        input_embeds = self._embeds_data[input_ids].astype(np.float32)
        n_tokens = len(input_ids)
        if n_tokens > self._max_seq_len:
            raise RuntimeError(
                f"prompt too long: {n_tokens} tokens > max_seq_len "
                f"{self._max_seq_len}; reduce chunk_seconds (6 frames + "
                "template <= ~1980)")
        if video_feature is not None:
            expected = int(image_mask.sum())
            actual = video_feature.shape[0]
            if expected != actual:
                raise RuntimeError(
                    f"image token mismatch: prompt={expected} vision={actual}")
            input_embeds[image_mask] = video_feature.astype(np.float32)

        S = self._max_seq_len
        full = np.zeros((1, S, DIM), dtype=np.float32)
        full[0, :n_tokens] = input_embeds
        mask = np.zeros((1, S), dtype=np.float32)
        mask[0, :n_tokens] = 1.0
        position_ids = np.zeros((1, S), dtype=np.int32)
        position_ids[0, :n_tokens] = np.arange(n_tokens, dtype=np.int32)
        ds = [np.zeros((1, S, DIM), dtype=np.float32) for _ in range(3)]
        if deepstacks is not None:
            image_idx = np.where(image_mask)[0]
            for i in range(3):
                ds[i][0, image_idx] = deepstacks[i].astype(np.float32)
        nlk = np.array([n_tokens - 1], dtype=np.int32)
        inputs = [full, mask, position_ids, ds[0], ds[1], ds[2], nlk]
        fmt = ['nchw'] * 7
        return inputs, fmt, n_tokens

    def _run_llm_embedding(self, prompt: str, video_feature=None,
                           deepstacks=None, image_mask=None) -> np.ndarray:
        inputs, fmt, _n_tokens = self._build_llm_inputs(
            prompt, video_feature, deepstacks, image_mask)
        rt = self.rknn_llm
        if rt is None:
            raise RuntimeError("llm runtime not initialised")
        outs = rt.inference(inputs, data_format=fmt)
        if outs is None or len(outs) == 0:
            raise RuntimeError("embedding llm inference returned no outputs")
        o = np.asarray(outs[0]).reshape(DIM).astype(np.float32)
        return _normalize_embedding(o)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def embed_text(self, text: str) -> np.ndarray:
        if text is None or not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        prompt = self._build_prompt(text, with_video=False)
        return self._run_llm_embedding(prompt)

    def embed_video_chunk(self, frames: Sequence[np.ndarray], text: str = "") -> np.ndarray:
        if frames is None or len(frames) == 0:
            raise ValueError("frames must be a non-empty list")
        text = text if text is not None else ""
        n_frames = len(frames)
        while n_frames > 1:
            prompt = self._build_prompt(
                text, with_video=True, frame_tokens=FRAME_TOKENS, n_frames=n_frames)
            if len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"]) <= self._max_seq_len:
                break
            n_frames -= 1
        selected = list(frames)
        if len(selected) > n_frames:
            ratios = [(i + 1) / (n_frames + 1) for i in range(n_frames)]
            selected = [selected[min(len(selected) - 1, int(round(r * (len(selected) - 1))))] for r in ratios]

        features_list = []
        deepstack_lists = [[], [], []]
        for frame in selected:
            image_feature, deepstacks = self._vision_inference(frame)
            features_list.append(image_feature)
            for i in range(3):
                deepstack_lists[i].append(deepstacks[i])

        frame_tokens = features_list[0].shape[1]
        video_feature = np.float16(
            np.concatenate(features_list, axis=0)).reshape(-1, DIM)
        deepstack_merged = [
            np.float16(np.concatenate(lst, axis=0)).reshape(-1, DIM)
            for lst in deepstack_lists
        ]

        prompt = self._build_prompt(
            text, with_video=True, frame_tokens=frame_tokens, n_frames=len(selected))
        image_pad_id = int(self.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
        ids = self.tokenizer(
            prompt, return_tensors='np', add_special_tokens=False)['input_ids'][0]
        image_mask = (ids == image_pad_id)

        return self._run_llm_embedding(
            prompt, video_feature, deepstack_merged, image_mask)

    def close(self) -> None:
        for rknn in (getattr(self, "rknn_vision", None), getattr(self, "rknn_llm", None)):
            if rknn is None:
                continue
            try:
                rknn.release()
            except Exception:  # noqa: BLE001
                pass
        self.rknn_vision = None
        self.rknn_llm = None
