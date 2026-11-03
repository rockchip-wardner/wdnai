import io
import math
import os
import re
import struct
import ctypes
import threading
import time
from argparse import ArgumentParser
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

import cv2
import numpy as np
from tokenizers import AddedToken, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

from .._librknn import (
    RKNN3Lite,
    RKNN3Image,
    RKNN3QueryCmd,
    RKNN3TensorType,
    Float16,
    build_llm_callback,
)

RKNN3_TENSOR_NCHW = 1
RKNN3_FLAG_MEMORY_CACHEABLE = 1
RKNN3_OUTPUT_CALLBACK_PREFILL_FINISHED = 1


def dump_tensor_attr(attr, prefix=""):
    try:
        shape = [int(attr.shape[i]) for i in range(int(attr.n_dims))]
    except Exception:
        shape = []
    print(
        f"{prefix}: name={attr.name!r} shape={shape} "
        f"dtype={int(attr.dtype)} layout={int(attr.layout)} "
        f"aligned_size={int(attr.aligned_size)} n_elems={int(attr.n_elems)}",
        flush=True,
    )


DEFAULT_MODEL_DIR = "/userdata/models/Qwen3.5-VL-9B"
RKNN_MODEL = DEFAULT_MODEL_DIR + "/Qwen3.5-9B-llm_seg0.1828.rknn"
WEIGHT_PATH = DEFAULT_MODEL_DIR + "/Qwen3.5-9B-llm_seg0.weight"
VISION_MODEL = DEFAULT_MODEL_DIR + "/Qwen3.5-9B-vision.1828.rknn"
VISION_WEIGHT = DEFAULT_MODEL_DIR + "/Qwen3.5-9B-vision.weight"
TOKENIZER_PATH = DEFAULT_MODEL_DIR + "/Qwen3.5-9B-llm.tokenizer.gguf"
EMBED_PATH = DEFAULT_MODEL_DIR + "/Qwen3.5-9B-llm.embed.bin"
DEFAULT_IMAGE = "/userdata/test/bus.jpg"

PROMPT_PREFIX = "<|im_start|>user\n"
PROMPT_POSTFIX = "<|im_end|>\n<|im_start|>assistant\n"
IMAGE_TOKEN_TEXT = "<image>"
IMAGE_START_TEXT = "<|vision_start|>"
IMAGE_END_TEXT = "<|vision_end|>"
IMAGE_CONTENT_TEXT = "<|image_pad|>"

QWEN35_FALLBACK_EOS_ID = 151645
QWEN35_FALLBACK_BOS_ID = 151643

SCRIPT_VERSION = "v1"

GGUF_MAGIC = b"GGUF"
G_U8, G_I8, G_U16, G_I16, G_U32, G_I32, G_F32, G_BOOL, \
    G_STR, G_ARR, G_U64, G_I64, G_F64 = range(13)

tokenizer = None
embeds_data = None


class _GGUFBuffer:
    def __init__(self, buf):
        self._buf = buf

    def raw(self, n):
        data = self._buf.read(n)
        if len(data) != n:
            raise ValueError("unexpected end of gguf file")
        return data

    def u32(self):
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.raw(8))[0]

    def string(self):
        return self.raw(self.u64()).decode("utf-8", errors="replace")

    def value(self, vtype):
        if vtype == G_U8:
            return self.raw(1)[0]
        if vtype == G_I8:
            return struct.unpack("<b", self.raw(1))[0]
        if vtype == G_U16:
            return struct.unpack("<H", self.raw(2))[0]
        if vtype == G_I16:
            return struct.unpack("<h", self.raw(2))[0]
        if vtype == G_U32:
            return self.u32()
        if vtype == G_I32:
            return struct.unpack("<i", self.raw(4))[0]
        if vtype == G_F32:
            return struct.unpack("<f", self.raw(4))[0]
        if vtype == G_BOOL:
            return bool(self.raw(1)[0])
        if vtype == G_STR:
            return self.string()
        if vtype == G_U64:
            return self.u64()
        if vtype == G_I64:
            return struct.unpack("<q", self.raw(8))[0]
        if vtype == G_F64:
            return struct.unpack("<d", self.raw(8))[0]
        if vtype == G_ARR:
            item_type = self.u32()
            count = self.u64()
            return [self.value(item_type) for _ in range(count)]
        raise ValueError("unsupported gguf value type: %d" % vtype)


def read_gguf_metadata(path):
    with open(path, "rb") as f:
        reader = _GGUFBuffer(io.BytesIO(f.read()))
    if reader.raw(4) != GGUF_MAGIC:
        raise ValueError("%s is not a gguf file" % path)
    reader.u32()
    reader.u64()
    n_kv = reader.u64()
    metadata = {}
    for _ in range(int(n_kv)):
        key = reader.string()
        vtype = reader.u32()
        metadata[key] = reader.value(vtype)
    return metadata


class GGUFBPETokenizer:
    def __init__(self, gguf_path):
        self.path = gguf_path
        metadata = read_gguf_metadata(gguf_path)
        tokens = metadata.get("tokenizer.ggml.tokens") or []
        merges = metadata.get("tokenizer.ggml.merges") or []
        token_types = metadata.get("tokenizer.ggml.token_type") or []
        if not tokens:
            raise ValueError("tokenizer.ggml.tokens is empty in %s" % gguf_path)

        self.tokens = tokens
        self.vocab_size = len(tokens)
        self.chat_template = metadata.get("tokenizer.chat_template")

        eos = metadata.get("tokenizer.ggml.eos_token_id")
        bos = metadata.get("tokenizer.ggml.bos_token_id")
        pad = metadata.get("tokenizer.ggml.padding_token_id")
        self.eos_token_id = int(eos) if eos is not None else None
        self.bos_token_id = int(bos) if bos is not None else None
        self.pad_token_id = int(pad) if pad is not None else None

        vocab = {token: idx for idx, token in enumerate(tokens)}
        merge_pairs = []
        for merge in merges:
            parts = merge.split(" ", 1)
            if len(parts) == 2:
                merge_pairs.append((parts[0], parts[1]))

        core = Tokenizer(BPE(vocab=vocab, merges=merge_pairs, unk_token=None))
        core.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
        core.decoder = ByteLevelDecoder()

        special_tokens = []
        for idx, token in enumerate(tokens):
            if idx < len(token_types) and int(token_types[idx]) in (3, 4):
                special_tokens.append(token)
        if special_tokens:
            core.add_tokens([AddedToken(t, special=True) for t in special_tokens])

        self._tokenizer = core
        self._image_token_id = core.token_to_id(IMAGE_TOKEN_TEXT)
        if self._image_token_id is None:
            core.add_tokens([AddedToken(IMAGE_TOKEN_TEXT, special=True)])
            self._image_token_id = core.token_to_id(IMAGE_TOKEN_TEXT)

    def __len__(self):
        return self.vocab_size

    @property
    def image_token_id(self):
        return self._image_token_id

    def encode(self, text, max_tokens=None):
        ids = self._tokenizer.encode(text).ids
        if max_tokens is not None:
            ids = ids[:int(max_tokens)]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return self._tokenizer.decode(
            [int(x) for x in ids],
            skip_special_tokens=skip_special_tokens,
        )


@dataclass
class TensorBlob:
    name: str
    dtype: int
    n_elems: int
    aligned_size: int
    data: bytes


@dataclass
class StageBatch:
    tensors: List[TensorBlob]
    n_tokens: int = 0


@dataclass
class StageSlot:
    batches: Deque[StageBatch] = field(default_factory=deque)
    expected_tokens: int = 0
    emitted_tokens: int = 0
    active_input_tokens: int = 0
    producer_done: bool = False
    failed: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        self.cv = threading.Condition(self.lock)

    def reset(self):
        with self.cv:
            self.batches.clear()
            self.expected_tokens = 0
            self.emitted_tokens = 0
            self.active_input_tokens = 0
            self.producer_done = False
            self.failed = False

    def close(self):
        with self.cv:
            self.producer_done = True
            self.cv.notify_all()

    def mark_failed(self):
        with self.cv:
            self.failed = True
            self.cv.notify_all()

    def wait_batch(self) -> Optional[StageBatch]:
        with self.cv:
            while not self.batches and not self.producer_done and not self.failed:
                self.cv.wait()
            if self.failed or not self.batches:
                return None
            return self.batches.popleft()


class PipelineState:
    def __init__(self, stage_count: int, bucket_size: int, verbose: bool = False):
        self.slots = [StageSlot() for _ in range(stage_count)]
        self.bucket_size = bucket_size
        self.verbose = verbose

    def reset(self):
        for slot in self.slots:
            slot.reset()

    def fail(self):
        for slot in self.slots:
            slot.mark_failed()

    def failed(self) -> bool:
        return any(slot.failed for slot in self.slots)


class LastStageResultState:
    def __init__(self):
        self.lock = threading.Lock()
        self.token_event = threading.Event()
        self.has_token = False
        self.next_token = -1
        self.source_stage = -1
        self.generated_tokens: List[int] = []
        self.last_output_text = ""

    def reset_next_token(self):
        with self.lock:
            self.has_token = False
            self.next_token = -1
            self.source_stage = -1
            self.token_event.clear()

    def reset_generation(self):
        with self.lock:
            self.has_token = False
            self.next_token = -1
            self.source_stage = -1
            self.generated_tokens.clear()
            self.last_output_text = ""
            self.token_event.clear()

    def set_tokens(self, tokens: List[int], source_stage: int):
        if not tokens:
            return
        with self.lock:
            self.next_token = int(tokens[-1])
            self.has_token = True
            self.source_stage = int(source_stage)
            self.generated_tokens.extend(int(x) for x in tokens)
            self.token_event.set()

    def get_next_token(self) -> Optional[int]:
        with self.lock:
            if not self.has_token:
                return None
            return self.next_token

    def get_source_stage(self) -> int:
        with self.lock:
            return self.source_stage

    def wait_next_token(self, timeout: float = 2.0) -> Optional[int]:
        self.token_event.wait(timeout)
        return self.get_next_token()


LAST_RESULT = LastStageResultState()


class _StageTokenizeAdapter:
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, text, return_tensors=None, truncation=True, max_length=None):
        ids = self.tok.encode(text, max_tokens=max_length)
        return {"input_ids": np.asarray([ids], dtype=np.int32)}


class _StageEmbedCallback:
    def __init__(self, pipeline: PipelineState):
        self.pipeline = pipeline

    def __call__(self, _py_obj, tokens, embed_ptr, length):
        global embeds_data
        try:
            if embeds_data is None:
                return -1
            embedding_dim = int(embeds_data.shape[1])
            num_tokens = len(tokens)
            expected_len = num_tokens * embedding_dim * np.dtype(np.float16).itemsize
            if int(length) != expected_len:
                print(f"invalid embed buffer: length={length}, expected={expected_len}")
                return -1
            ids = np.asarray(tokens, dtype=np.int64)
            if ids.size and (ids.min() < 0 or ids.max() >= embeds_data.shape[0]):
                print("embed_callback token id out of range")
                return -1
            dst = np.ctypeslib.as_array(
                ctypes.cast(embed_ptr, ctypes.POINTER(ctypes.c_uint16)),
                shape=(num_tokens * embedding_dim,),
            ).view(np.float16)
            dst[:] = embeds_data[ids].reshape(-1)
            slot = self.pipeline.slots[0]
            with slot.cv:
                slot.expected_tokens += num_tokens
            return 0
        except Exception as e:
            print(f"embed_callback failed: {e}")
            return -1


class _StageResultCallback:
    def __init__(self, stage_index: int, is_last_stage: bool):
        self.stage_index = stage_index
        self.is_last_stage = is_last_stage

    def __call__(self, _py_obj, tokens, state):
        try:
            if not self.is_last_stage:
                return 0
            if int(state) == 5:
                print(f"\n[last-stage result_callback] RKNN error state on stage{self.stage_index}")
                return 0
            if int(state) != 0 or not tokens:
                return 0
            LAST_RESULT.set_tokens(
                [int(t) for t in tokens], source_stage=self.stage_index,
            )
            return 0
        except Exception as e:
            print(f"result_callback failed: {e}")
            return 0


class _StageOutputCallback:
    def __init__(self, pipeline: PipelineState, stage_index: int, embedding_dim: int):
        self.pipeline = pipeline
        self.stage_index = stage_index
        self.embedding_dim = embedding_dim

    def __call__(self, _py_obj, views, state):
        try:
            pipeline = self.pipeline
            slot = pipeline.slots[self.stage_index]
            tensors = []
            for view in views:
                if view.data is None or int(view.n_elems) <= 0:
                    continue
                data = np.asarray(view.data, dtype=np.float16).tobytes()
                blob = TensorBlob(
                    name=view.name,
                    dtype=int(view.dtype),
                    n_elems=int(view.n_elems),
                    aligned_size=len(data),
                    data=data,
                )
                tensors.append(blob)
                vlog(
                    pipeline,
                    f"[stage{self.stage_index}] captured output name={blob.name}, "
                    f"dtype={blob.dtype}, n_elems={blob.n_elems}, aligned_size={blob.aligned_size}",
                )
            if not tensors:
                return 0

            with slot.cv:
                remaining = max(0, slot.expected_tokens - slot.emitted_tokens)
                if slot.active_input_tokens > 0:
                    n_tokens = slot.active_input_tokens
                elif int(state) == RKNN3_OUTPUT_CALLBACK_PREFILL_FINISHED:
                    n_tokens = remaining
                else:
                    n_tokens = min(remaining, pipeline.bucket_size)

                if n_tokens == 0:
                    embed = pick_embed_tensor(tensors)
                    if embed is not None and embed.n_elems > 0 and self.embedding_dim > 0:
                        n_tokens = embed.n_elems // self.embedding_dim

                slot.emitted_tokens += int(n_tokens)
                slot.batches.append(StageBatch(tensors=tensors, n_tokens=int(n_tokens)))
                slot.cv.notify()

            vlog(
                pipeline,
                f"[stage{self.stage_index}] output_callback state={int(state)}, "
                f"batch_tokens={n_tokens}, emitted={slot.emitted_tokens}/{slot.expected_tokens}",
            )
            return 0
        except Exception as e:
            print(f"stage_output_callback failed: {e}")
            try:
                self.pipeline.fail()
            except Exception:
                pass
            return -1



class StageRuntime:
    def __init__(self, name: str, model_path: str, weight_path: str):
        self.name = name
        self.model_path = model_path
        self.weight_path = weight_path
        self.rknn: Optional[RKNN3Lite] = None
        self.embedding_dim = 0
        self.vocab_size = 0
        self.max_ctx_len = 0
        self.max_position_embeddings = 0
        self.model_type = ""
        self.output_tensors = None
        self.n_output_tensors = 0

    def release(self):
        self.output_tensors = None
        self.n_output_tensors = 0

        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None


class VisionRuntime:
    def __init__(self, model_path: str, weight_path: str):
        self.model_path = model_path
        self.weight_path = weight_path
        self.rknn: Optional[RKNN3Lite] = None
        self.pruned = True
        self.nchw_input = False
        self.width = 0
        self.height = 0
        self.input_dtype = int(RKNN3TensorType.RKNN3_TENSOR_FLOAT16)
        self.n_image_tokens = 0
        self.embedding_dim = 0

    def release(self):
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None


def vlog(pipeline: PipelineState, msg: str):
    if pipeline.verbose:
        print(msg, flush=True)


def replace_seg_suffix(path: str, seg_idx: int) -> str:
    match = re.search(r"_seg\d+(?=\.)", path)
    if match:
        return path[:match.start()] + f"_seg{seg_idx}" + path[match.end():]

    root, ext = os.path.splitext(path)
    if ext:
        return f"{root}_seg{seg_idx}{ext}"
    return f"{path}_seg{seg_idx}"


def resolve_weight_path(rknn_path: str, weight_path: Optional[str]) -> str:
    if weight_path:
        return weight_path

    candidates = []
    if rknn_path.endswith(".1828.rknn"):
        candidates.append(rknn_path[:-len(".1828.rknn")] + ".weight")
    candidates.append(os.path.splitext(rknn_path)[0] + ".weight")
    candidates.append(os.path.splitext(rknn_path)[0].replace(".1828", "") + ".weight")
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return candidates[0]


def normalize_device_id(device_id) -> bytes:
    if isinstance(device_id, bytes):
        return device_id
    if isinstance(device_id, bytearray):
        return bytes(device_id)
    return str(device_id).encode("utf-8")


def format_device_id(device_id) -> str:
    if isinstance(device_id, (bytes, bytearray)):
        return bytes(device_id).decode("utf-8", errors="ignore")
    return str(device_id)


def decode_c_string(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def resolve_special_token_ids(tok, eos_override=None, bos_override=None):
    if eos_override:
        eos_ids = [int(x) for x in eos_override]
    elif tok.eos_token_id is not None:
        eos_ids = [int(tok.eos_token_id)]
    else:
        eos_ids = [QWEN35_FALLBACK_EOS_ID]

    if bos_override:
        bos_ids = [int(x) for x in bos_override]
    elif tok.bos_token_id is not None:
        bos_ids = [int(tok.bos_token_id)]
    else:
        bos_ids = []

    return eos_ids, bos_ids


def pick_embed_tensor(tensors: List[TensorBlob]) -> Optional[TensorBlob]:
    if not tensors:
        return None
    for tensor in tensors:
        name = tensor.name.lower()
        if "hidden" in name or "last_hidden" in name or "output" in name:
            return tensor
    return tensors[0]


def query_core_internal_size(rknn):
    core_num = int(rknn.rknn3_query(RKNN3QueryCmd.CORE_NUMBER))
    sizes = rknn.rknn3_query(RKNN3QueryCmd.CORE_MEM_SIZE)
    if not sizes:
        raise RuntimeError("queried internal memory size is zero")
    internal = max(int(m.internal_size) for m in sizes)
    if internal <= 0:
        raise RuntimeError("queried internal memory size is zero")
    return core_num, internal


def setup_shared_internal_mem(vision: "VisionRuntime", stage: StageRuntime):
    core_num, llm_internal = query_core_internal_size(stage.rknn)
    vision_core_num, vision_internal = query_core_internal_size(vision.rknn)
    if core_num != vision_core_num:
        raise RuntimeError(
            f"core number mismatch: llm={core_num}, vision={vision_core_num}"
        )

    shared_size = max(llm_internal, vision_internal)
    mems = []
    try:
        for core_id in range(core_num):
            mem = vision.rknn.create_mem(
                shared_size,
                core_id,
                RKNN3_FLAG_MEMORY_CACHEABLE,
            )
            if not mem:
                raise RuntimeError(f"create_mem failed for core {core_id}")
            mems.append(int(mem))
        stage.rknn.set_internal_mem(mems)
        vision.rknn.set_internal_mem(mems)
    except Exception:
        for mem in mems:
            try:
                vision.rknn.destroy_user_mem(mem)
            except Exception:
                pass
        raise

    print(
        f"[shared] internal mem: cores={core_num}, size={shared_size / 1048576.0:.2f}MB/core, "
        f"llm={llm_internal / 1048576.0:.2f}MB/core, vision={vision_internal / 1048576.0:.2f}MB/core"
    )
    return mems


def destroy_shared_internal_mem(vision: Optional["VisionRuntime"], mems):
    if vision is None or vision.rknn is None:
        return
    for mem in mems:
        try:
            vision.rknn.destroy_user_mem(mem)
        except Exception:
            pass
    mems.clear()


def build_prompt(user_prompt: str, n_images: int, enable_thinking: bool = False) -> str:
    n_images = int(n_images)
    placeholders = user_prompt.count(IMAGE_TOKEN_TEXT)
    if n_images > 0:
        if placeholders == 0:
            body = IMAGE_TOKEN_TEXT * n_images + user_prompt
        elif placeholders == n_images:
            body = user_prompt
        else:
            raise ValueError(
                f"prompt contains {placeholders} {IMAGE_TOKEN_TEXT} placeholders "
                f"but {n_images} images were given"
            )
    else:
        if placeholders > 0:
            raise ValueError(
                f"prompt contains {placeholders} {IMAGE_TOKEN_TEXT} placeholders "
                "but no image was given"
            )
        body = user_prompt

    prompt = PROMPT_PREFIX + body + PROMPT_POSTFIX
    if enable_thinking:
        prompt += "<think>\n"
    else:
        prompt += "<think>\n\n\n\n"
    return prompt


def parse_image_paths(values: List[str]) -> List[str]:
    paths = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                paths.append(part)
    return paths


def count_prompt_tokens(tok, prompt: str) -> int:
    return len(tok.encode(prompt))


def estimate_multimodal_input_tokens(prompt_tokens: int, n_images: int, n_image_tokens: int) -> int:
    if n_images <= 0:
        return prompt_tokens
    return prompt_tokens - n_images + n_images * (n_image_tokens + 2)


def flush_generated_text():
    global tokenizer
    with LAST_RESULT.lock:
        if not LAST_RESULT.generated_tokens:
            return
        try:
            text = tokenizer.decode(
                LAST_RESULT.generated_tokens,
                skip_special_tokens=True,
            )
            safe_text = text.split("\ufffd", 1)[0] if "\ufffd" in text else text
            new_part = safe_text[len(LAST_RESULT.last_output_text):]
            if new_part:
                if not LAST_RESULT.last_output_text.strip() and not new_part.strip():
                    LAST_RESULT.last_output_text += new_part
                    return
                print(new_part, end="", flush=True)
                LAST_RESULT.last_output_text += new_part
        except Exception as e:
            print(f"\n[Decode error: {e}]", flush=True)


def init_stage_context(
    stage: StageRuntime,
    stage_idx: int,
    device_id: str,
    target: str,
    core_mask: int,
    max_context_len: int,
    decrypt_key_path: Optional[str],
    verbose: bool,
    share_internal: bool,
) -> bool:
    global tokenizer

    print(
        f"[{stage.name}] init context: model={stage.model_path}, weight={stage.weight_path}, "
        f"device_id={format_device_id(device_id)}"
    )

    stage.rknn = RKNN3Lite(llm_mode=True, verbose=verbose)
    if share_internal:
        stage.rknn.rknn_config.user_mem_internal = 1

    ret = stage.rknn.load_rknn(stage.model_path, stage.weight_path)
    if ret != 0:
        print(f"[{stage.name}] load_rknn failed")
        return False

    ret = stage.rknn.init_runtime(
        target=target,
        core_mask=core_mask,
        device_id=device_id,
        decrypt_key_path=decrypt_key_path,
    )
    if ret != 0:
        print(f"[{stage.name}] init_runtime failed")
        return False

    io_num = stage.rknn.rknn3_query(RKNN3QueryCmd.IN_OUT_NUM)
    llm_config = stage.rknn.rknn3_query(RKNN3QueryCmd.LLM_CONFIG)
    if io_num is None or llm_config is None:
        print(f"[{stage.name}] query model info failed")
        return False

    stage.embedding_dim = int(llm_config.embedding_dim)
    stage.vocab_size = int(llm_config.vocab_size)
    if stage.vocab_size <= 0:
        stage.vocab_size = len(tokenizer)
    stage.max_ctx_len = int(llm_config.max_ctx_len)
    stage.max_position_embeddings = int(getattr(llm_config, "max_position_embeddings", 0))
    stage.model_type = decode_c_string(getattr(llm_config, "model_type", b""))

    print(f"[{stage.name}] model input num: {io_num.n_input}, output num: {io_num.n_output}")
    print(
        f"[{stage.name}] embedding_dim={stage.embedding_dim}, vocab_size={stage.vocab_size}, "
        f"max_ctx_len={stage.max_ctx_len}, max_position_embeddings={stage.max_position_embeddings}, "
        f"model_type={stage.model_type}"
    )

    if max_context_len > stage.max_ctx_len:
        print(
            f"[{stage.name}] ERROR: max_context_len ({max_context_len}) is greater than "
            f"model max_ctx_len ({stage.max_ctx_len})"
        )
        return False

    if verbose:
        for i in range(int(io_num.n_input)):
            attr = stage.rknn.rknn3_query(RKNN3QueryCmd.INPUT_ATTR, index=i)
            dump_tensor_attr(attr, prefix=f"{stage.name}_input")
        for i in range(int(io_num.n_output)):
            attr = stage.rknn.rknn3_query(RKNN3QueryCmd.OUTPUT_ATTR, index=i)
            dump_tensor_attr(attr, prefix=f"{stage.name}_output")

    return True


def init_stage_session(
    stage: StageRuntime,
    pipeline: PipelineState,
    stage_idx: int,
    max_context_len: int,
    ignore_eos: bool,
    special_eos_ids: List[int],
    special_bos_ids: List[int],
    presence_penalty: float,
    kvcache_policy: str,
    verbose: bool,
) -> bool:
    global tokenizer

    is_last_stage = stage_idx == len(pipeline.slots) - 1
    if not is_last_stage:
        output_tensors, n_output_tensors = stage.rknn.create_output_tensors()
        if output_tensors is None or n_output_tensors <= 0:
            print(f"[{stage.name}] create_output_tensors failed")
            return False
        stage.output_tensors = output_tensors
        stage.n_output_tensors = int(n_output_tensors)

    ctx_len = max_context_len if max_context_len > 0 else stage.max_ctx_len
    llm_args = [{
        "max_new_tokens": 1,
        "top_k": 1,
        "top_p": 0.9,
        "temperature": 1.0,
        "repeat_penalty": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": presence_penalty,
        "vocab_size": stage.vocab_size,
        "special_eos_id": special_eos_ids,
        "special_bos_id": special_bos_ids,
        "max_context_len": ctx_len,
        "keep_history": 1,
        "ignore_eos_token": ignore_eos,
        "logits_name": "output",
    }]

    callback = build_llm_callback(
        result_cb=_StageResultCallback(stage_idx, is_last_stage),
        tokenizer=_StageTokenizeAdapter(tokenizer),
        embed_callback=_StageEmbedCallback(pipeline),
        output_cb=(
            None if is_last_stage
            else _StageOutputCallback(pipeline, stage_idx, stage.embedding_dim)
        ),
        output_tensors=None if is_last_stage else stage.output_tensors,
    )

    ret = stage.rknn.init_llm_session(llm_args=llm_args, llm_callback=callback)
    if ret != 0:
        print(f"[{stage.name}] init_llm_session failed")
        return False

    ret = stage.rknn.set_chat_template("", "", "")
    if ret != 0:
        print(f"[{stage.name}] WARNING: reset chat template failed ret={ret}")

    if kvcache_policy != "default":
        print(
            f"[{stage.name}] WARNING: kvcache_policy={kvcache_policy!r} is not "
            "supported by the ctypes shim; keeping the model default"
        )

    if verbose:
        mem = stage.rknn.rknn3_query(RKNN3QueryCmd.DEVICE_MEM_INFO)
        if mem is not None:
            print(
                f"[{stage.name}] device mem: total={mem.sys_total / 1048576.0:.1f}MB, "
                f"free={mem.sys_free / 1048576.0:.1f}MB"
            )

    print(f"[{stage.name}] init done: outputs={stage.n_output_tensors}")
    return True


def vision_geometry_from_attr(in_attr):
    shape = [int(in_attr.shape[i]) for i in range(int(in_attr.n_dims))]
    layout = int(in_attr.layout)

    if len(shape) == 4:
        if layout == int(RKNN3_TENSOR_NCHW):
            channel, height, width = shape[1], shape[2], shape[3]
            return False, int(width), int(height), int(channel), True, shape
        height, width, channel = shape[1], shape[2], shape[3]
        return False, int(width), int(height), int(channel), False, shape

    patches = shape[0]
    if patches <= 0:
        raise ValueError("invalid vision input shape: %s" % shape)

    side = int(round(math.sqrt(float(patches)) * 16.0))
    if side <= 0 or (side // 16) * (side // 16) != patches or side % 32 != 0:
        raise ValueError(
            "cannot derive vision resolution from pruned input %s (patches=%d)" % (shape, patches)
        )
    return True, side, side, 3, False, shape


def init_vision(
    vision: VisionRuntime,
    target: str,
    core_mask: int,
    device_id: str,
    decrypt_key_path: Optional[str],
    verbose: bool,
    share_internal: bool,
) -> bool:
    print(
        f"[vision] init stage: model={vision.model_path}, weight={vision.weight_path}, "
        f"device_id={format_device_id(device_id)}"
    )

    vision.rknn = RKNN3Lite(verbose=verbose)
    if share_internal:
        vision.rknn.rknn_config.user_mem_internal = 1

    ret = vision.rknn.load_rknn(vision.model_path, vision.weight_path)
    if ret != 0:
        print("[vision] load_rknn failed")
        return False

    ret = vision.rknn.init_runtime(
        target=target,
        core_mask=core_mask,
        device_id=device_id,
        decrypt_key_path=decrypt_key_path,
    )
    if ret != 0:
        print("[vision] init_runtime failed")
        return False

    in_attr = vision.rknn.get_inputs_tensor_attr()[0]
    out_attr = vision.rknn.get_outputs_tensor_attr()[0]

    vision.pruned, vision.width, vision.height, channel, vision.nchw_input, shape = \
        vision_geometry_from_attr(in_attr)
    vision.input_dtype = int(in_attr.dtype)

    out_shape = [int(out_attr.shape[i]) for i in range(int(out_attr.n_dims))]
    vision.n_image_tokens = int(out_shape[-2]) if len(out_shape) >= 2 else 1
    vision.embedding_dim = int(out_shape[-1])

    print(
        f"[vision] input_shape={shape}, pruned={int(vision.pruned)}, "
        f"nchw={int(vision.nchw_input)}, width={vision.width}, height={vision.height}, "
        f"channel={channel}, dtype={vision.input_dtype}"
    )
    print(
        f"[vision] output_shape={out_shape}, n_image_tokens={vision.n_image_tokens}, "
        f"embedding_dim={vision.embedding_dim}"
    )

    if verbose:
        dump_tensor_attr(in_attr, prefix="vision_input")
        dump_tensor_attr(out_attr, prefix="vision_output")

    if vision.n_image_tokens <= 0 or vision.embedding_dim <= 0:
        print("[vision] invalid vision output shape")
        return False

    if verbose:
        mem = vision.rknn.rknn3_query(RKNN3QueryCmd.DEVICE_MEM_INFO)
        if mem is not None:
            print(
                f"[vision] device mem: total={mem.sys_total / 1048576.0:.1f}MB, "
                f"free={mem.sys_free / 1048576.0:.1f}MB"
            )

    return True


def preprocess_vision_input(img_bgr, vision: VisionRuntime, keep_aspect: bool):
    width, height = vision.width, vision.height
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if keep_aspect:
        src_h, src_w = rgb.shape[:2]
        ratio = min(width / float(src_w), height / float(src_h))
        dst_w = max(1, int(round(src_w * ratio)))
        dst_h = max(1, int(round(src_h * ratio)))
        resized = cv2.resize(rgb, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        top = (height - dst_h) // 2
        left = (width - dst_w) // 2
        canvas[top:top + dst_h, left:left + dst_w] = resized
        rgb = canvas
    else:
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    if not vision.pruned:
        if vision.nchw_input:
            data = rgb.transpose(2, 0, 1).reshape(1, 3, height, width)
        else:
            data = rgb.reshape(1, height, width, 3)
        if vision.input_dtype == int(RKNN3TensorType.RKNN3_TENSOR_UINT8):
            return data.astype(np.uint8)
        return np.float16((data.astype(np.float32) - 127.5) / 127.5)

    nchw = rgb.transpose(2, 0, 1).astype(np.float32)[None]
    nchw = (nchw - 127.5) / 127.5
    patches = np.concatenate([nchw, nchw], axis=1)
    grid_h = height // 32
    grid_w = width // 32
    patches = patches.reshape(1, 2, 3, grid_h, 2, 16, grid_w, 2, 16)
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return np.float16(patches.reshape((height // 16) * (width // 16), 3 * 2 * 16 * 16))


def run_vision(vision: VisionRuntime, images: List[np.ndarray], keep_aspect: bool):
    features = []
    vision_start = time.perf_counter()
    for index, img in enumerate(images):
        feature = preprocess_vision_input(img, vision, keep_aspect)
        outputs = vision.rknn.inference(inputs=[feature])
        if outputs is None or len(outputs) == 0:
            raise RuntimeError(f"vision inference failed at image {index}")
        embeds = np.float16(outputs[0])
        if embeds.ndim == 2:
            embeds = np.expand_dims(embeds, 0)
        features.append(embeds)
    vision_latency = time.perf_counter() - vision_start

    packed = np.ascontiguousarray(np.concatenate(features, axis=0), dtype=np.float16)
    image_input = RKNN3Image()
    image_input.image_embed = packed.ctypes.data_as(ctypes.POINTER(Float16))
    image_input.n_image_tokens = int(packed.shape[-2])
    image_input.n_image = int(packed.shape[0])
    image_input.image_width = vision.width
    image_input.image_height = vision.height
    image_input.image_start = IMAGE_START_TEXT.encode("utf-8")
    image_input.image_end = IMAGE_END_TEXT.encode("utf-8")
    image_input.image_content = IMAGE_CONTENT_TEXT.encode("utf-8")

    return image_input, packed, vision_latency, int(packed.shape[-2])


def run_stage_worker(stage_idx: int, stages: List[StageRuntime], pipeline: PipelineState):
    stage = stages[stage_idx]
    input_slot = pipeline.slots[stage_idx - 1]
    output_slot = pipeline.slots[stage_idx] if stage_idx + 1 < len(stages) else None
    is_last_stage = stage_idx == len(stages) - 1
    consumed_tokens = 0

    try:
        while True:
            batch = input_slot.wait_batch()
            if batch is None:
                with input_slot.cv:
                    clean_end = input_slot.producer_done and not input_slot.failed and not input_slot.batches
                if clean_end:
                    break
                if output_slot is not None:
                    output_slot.close()
                return

            embed = pick_embed_tensor(batch.tensors)
            if embed is None or batch.n_tokens <= 0 or stage.embedding_dim <= 0:
                print(f"[stage{stage_idx}] invalid pipeline batch")
                pipeline.fail()
                return

            embed_elems = batch.n_tokens * stage.embedding_dim
            embed_bytes = embed_elems * np.dtype(np.float16).itemsize
            if embed_bytes > len(embed.data):
                print(
                    f"[stage{stage_idx}] embed buffer too small: "
                    f"need={embed_bytes}, got={len(embed.data)}"
                )
                pipeline.fail()
                return

            embed_array = np.frombuffer(embed.data, dtype=np.float16, count=embed_elems).reshape(
                batch.n_tokens, stage.embedding_dim
            ).copy()

            if output_slot is not None:
                with output_slot.cv:
                    output_slot.expected_tokens += batch.n_tokens
                    output_slot.active_input_tokens = batch.n_tokens

            disable_sampling = True
            if is_last_stage:
                with pipeline.slots[0].cv:
                    total_tokens = pipeline.slots[0].expected_tokens
                disable_sampling = consumed_tokens + batch.n_tokens < total_tokens

            consumed_tokens += batch.n_tokens
            vlog(
                pipeline,
                f"[stage{stage_idx}] consume batch: n_tokens={batch.n_tokens}, "
                f"bytes={embed_bytes}, disable_sampling={int(disable_sampling)}",
            )

            ret, _perf = stage.rknn.session_run(
                embeds=embed_array,
                keep_history=True,
                max_new_tokens=1,
                prefill_only=True,
                disable_sampling=disable_sampling,
            )

            if output_slot is not None:
                with output_slot.cv:
                    output_slot.active_input_tokens = 0

            if ret != 0:
                print(f"[stage{stage_idx}] run failed ret={ret}")
                pipeline.fail()
                return
    except Exception as e:
        print(f"[stage{stage_idx}] worker exception: {e}")
        pipeline.fail()
        return
    finally:
        if output_slot is not None:
            output_slot.close()


def run_pipeline_once(
    stages: List[StageRuntime],
    pipeline: PipelineState,
    prompt: Optional[str] = None,
    input_tokens: Optional[List[int]] = None,
    inputs: Optional[list] = None,
    enable_thinking: bool = False,
    pipeline_mode: str = "threaded",
):
    if not stages:
        return False, 0

    pipeline.reset()
    LAST_RESULT.reset_next_token()

    workers = []
    if pipeline_mode == "threaded":
        for i in range(1, len(stages)):
            t = threading.Thread(
                target=run_stage_worker,
                args=(i, stages, pipeline),
                name=f"stage{i}-worker",
            )
            t.start()
            workers.append(t)

    stage0 = stages[0]
    try:
        stage0_disable_sampling = len(stages) > 1
        if prompt is not None:
            ret, _perf = stage0.rknn.session_run(
                inputs=inputs,
                prompt=prompt,
                keep_history=True,
                max_new_tokens=1,
                prefill_only=True,
                disable_sampling=stage0_disable_sampling,
                enable_thinking=enable_thinking,
            )
        else:
            if not input_tokens:
                pipeline.fail()
                ret = -1
            else:
                ret, _perf = stage0.rknn.session_run(
                    tokens=np.asarray(input_tokens, dtype=np.int32),
                    keep_history=True,
                    max_new_tokens=1,
                    prefill_only=True,
                    disable_sampling=stage0_disable_sampling,
                    enable_thinking=enable_thinking,
                )

        if ret != 0:
            print(f"[stage0] run failed ret={ret}")
            pipeline.fail()
    except Exception as e:
        print(f"[stage0] run exception: {e}")
        ret = -1
        pipeline.fail()
    finally:
        pipeline.slots[0].close()

    if pipeline_mode == "threaded":
        for worker in workers:
            worker.join()
    else:
        for i in range(1, len(stages)):
            run_stage_worker(i, stages, pipeline)
            if pipeline.failed():
                break

    with pipeline.slots[0].cv:
        stage0_input_tokens = pipeline.slots[0].expected_tokens

    return ret == 0 and not pipeline.failed(), stage0_input_tokens


def printf_perf(prefill_tokens, prefill_ms, decode_tokens, decode_ms, vision_ms=0.0):
    print("\n\nPerformance Statistics:")
    print("-----------------------------------------------------------------------------------------")
    print(" %-10s | %-16s | %-8s | %-20s | %-20s " % (
        "Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second"))
    print("-----------------------------------------------------------------------------------------")

    if prefill_tokens == 0 or prefill_ms <= 0:
        prefill_tpt, prefill_tps = 0.0, 0.0
    else:
        prefill_tpt = prefill_ms / prefill_tokens
        prefill_tps = prefill_tokens * 1000.0 / prefill_ms
    print(" %-10s | %-16.2f | %-8d | %-20.2f | %-20.2f " % (
        "Prefill", prefill_ms, prefill_tokens, prefill_tpt, prefill_tps))

    if decode_tokens == 0 or decode_ms <= 0:
        decode_tpt, decode_tps = 0.0, 0.0
    else:
        decode_tpt = decode_ms / decode_tokens
        decode_tps = decode_tokens * 1000.0 / decode_ms
    print(" %-10s | %-16.2f | %-8d | %-20.2f | %-20.2f " % (
        "Generate", decode_ms, decode_tokens, decode_tpt, decode_tps))
    print("-----------------------------------------------------------------------------------------")
    print(f" Vision latency = {vision_ms:.2f} ms")


def main():
    global tokenizer, embeds_data

    parser = ArgumentParser(
        description="RKNN3 multicard segmented Qwen3.5-VL pipeline."
    )
    parser.add_argument("--rknn_path", type=str, default=RKNN_MODEL,
                        help="stage0 llm rknn path; _segN is generated automatically")
    parser.add_argument("--weight_path", type=str, default=None,
                        help="stage0 llm weight path; default derives from rknn_path")
    parser.add_argument("--vision_path", type=str, default=VISION_MODEL,
                        help="vision rknn path")
    parser.add_argument("--vision_weight_path", type=str, default=None,
                        help="vision weight path; default derives from vision_path")
    parser.add_argument("--tokenizer_path", type=str, default=TOKENIZER_PATH,
                        help="tokenizer gguf path")
    parser.add_argument("--embed_path", type=str, default=EMBED_PATH,
                        help="token embedding bin path (float16)")
    parser.add_argument("--image", action="append", default=None,
                        help="input image path(s); comma-separated or repeat the option")
    parser.add_argument("--no_image", action="store_true",
                        help="run text-only inference without vision input")
    parser.add_argument("--prompt", type=str, default="请描述这张图片的内容。",
                        help="user prompt")
    parser.add_argument("--target", type=str, default="rk1820",
                        help="RKNN3 target, e.g. rk1820/rk1828")
    parser.add_argument("--core_mask", type=str, default="0xff",
                        help="llm NPU core mask in hex")
    parser.add_argument("--vision_core_mask", type=str, default=None,
                        help="vision NPU core mask in hex; default same as --core_mask")
    parser.add_argument("--stage_count", type=int, default=2,
                        help="number of segmented stages/devices")
    parser.add_argument("--bucket_size", type=int, default=128,
                        help="token bucket size used to account output_callback chunks")
    parser.add_argument("--pipeline_mode", choices=["sequential", "threaded"], default="threaded",
                        help="multicard execution mode")
    parser.add_argument("--max_context_len", type=int, default=0,
                        help="0 means use queried model max_ctx_len")
    parser.add_argument("--max_new_token", "--max_new_tokens", dest="max_new_token", type=int, default=512,
                        help="decode loop count")
    parser.add_argument("--repeat", type=int, default=1,
                        help="number of image+prefill+decode rounds; vision runtime is reused when kept alive")
    parser.add_argument("--ignore_eos", action="store_true",
                        help="continue decoding after configured special EOS ids")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="enable Qwen3.5 thinking mode")
    parser.add_argument("--keep_aspect", action="store_true",
                        help="letterbox the image instead of stretching it")
    parser.add_argument("--vision_keep_alive", action="store_true",
                        help="keep vision runtime loaded when internal memory is not shared")
    parser.add_argument("--no_share_internal_mem", action="store_true",
                        help="disable vision/stage0 shared internal memory; vision is released after embedding")
    parser.add_argument("--special_eos_id", action="append", type=int, default=None,
                        help="special EOS id; repeat for multiple ids")
    parser.add_argument("--special_bos_id", action="append", type=int, default=None,
                        help="special BOS id; repeat for multiple ids")
    parser.add_argument("--presence_penalty", type=float, default=1.5,
                        help="sampling presence penalty")
    parser.add_argument("--kvcache_policy", choices=["default", "normal", "recurrent"], default="default",
                        help="KV-cache policy")
    parser.add_argument("--device_id", action="append", default=[],
                        help="device id for each stage; repeat in stage order")
    parser.add_argument("--vision_device_id", type=str, default=None,
                        help="vision device id; default stage0 device")
    parser.add_argument("--decrypt_key_path", type=str, default=None,
                        help="decrypt key/license file path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"rknn_multicard_qwen35_vl {SCRIPT_VERSION}")

    if args.stage_count < 1:
        raise ValueError(f"stage_count must be >= 1, got {args.stage_count}")
    if args.bucket_size <= 0:
        raise ValueError(f"bucket_size must be > 0, got {args.bucket_size}")
    if args.repeat < 1:
        raise ValueError(f"repeat must be >= 1, got {args.repeat}")
    if (
        args.repeat > 1
        and not args.no_image
        and args.no_share_internal_mem
        and not args.vision_keep_alive
    ):
        raise ValueError(
            "--repeat > 1 with vision enabled requires shared internal memory "
            "or --vision_keep_alive"
        )

    core_mask = int(args.core_mask, 16)
    vision_core_mask = int(args.vision_core_mask, 16) if args.vision_core_mask else core_mask

    base_weight_path = resolve_weight_path(args.rknn_path, args.weight_path)
    if args.stage_count == 1:
        model_paths = [args.rknn_path]
        weight_paths = [base_weight_path]
    else:
        model_paths = [replace_seg_suffix(args.rknn_path, i) for i in range(args.stage_count)]
        weight_paths = [replace_seg_suffix(base_weight_path, i) for i in range(args.stage_count)]
    for path in model_paths + weight_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    vision_model = args.vision_path
    vision_weight = resolve_weight_path(vision_model, args.vision_weight_path)
    vision_enabled = not args.no_image
    if vision_enabled:
        os.environ['RKNN_LOG_LEVEL'] = '0' # VLM模型因为mrope退化的问题，在推理时会输出大量警告信息，设置RKNN_LOG_LEVEL=0可以屏蔽这些警告，避免干扰其它log查看(v1.1.0)

        for path in (vision_model, vision_weight):
            if not os.path.isfile(path):
                raise FileNotFoundError(path)

    print("--> Loading tokenizer")
    tokenizer = GGUFBPETokenizer(args.tokenizer_path)
    special_eos_ids, special_bos_ids = resolve_special_token_ids(
        tokenizer, args.special_eos_id, args.special_bos_id
    )
    print(
        f"len(tokenizer)={len(tokenizer)}, eos={tokenizer.eos_token_id}, bos={tokenizer.bos_token_id}, "
        f"special_eos_id={special_eos_ids}, special_bos_id={special_bos_ids}, "
        f"image_token_id={tokenizer.image_token_id}"
    )
    print("done")

    probe = RKNN3Lite(verbose=args.verbose)
    try:
        detected = [normalize_device_id(x) for x in probe.get_devices_id(target=args.target)]
    finally:
        probe.release()

    print(f"found {len(detected)} devices:")
    for i, dev_id in enumerate(detected):
        print(f"  [{i}] id={format_device_id(dev_id)}")

    if args.device_id:
        if len(args.device_id) < args.stage_count:
            raise RuntimeError(
                f"not enough --device_id arguments: need={args.stage_count}, got={len(args.device_id)}"
            )
        device_ids = [normalize_device_id(x) for x in args.device_id[:args.stage_count]]
        missing = [x for x in device_ids if x not in detected]
        if missing:
            raise RuntimeError(
                f"specified device_id not found: {[format_device_id(x) for x in missing]}"
            )
    else:
        if len(detected) < args.stage_count:
            raise RuntimeError(
                f"auto-detect failed: found={len(detected)} devices, need={args.stage_count}"
            )
        device_ids = detected[:args.stage_count]
        print(f"auto-assigning {args.stage_count} devices")

    if args.vision_device_id:
        vision_device_id = normalize_device_id(args.vision_device_id)
    else:
        vision_device_id = device_ids[0]
    print(f"vision device: {format_device_id(vision_device_id)}")

    pipeline = PipelineState(args.stage_count, args.bucket_size, args.verbose)
    stages = [
        StageRuntime(f"stage{i}", model_paths[i], weight_paths[i])
        for i in range(args.stage_count)
    ]
    vision = VisionRuntime(vision_model, vision_weight)

    shared_mems = []

    try:
        share_internal = vision_enabled and not args.no_share_internal_mem

        for i, stage in enumerate(stages):
            ok = init_stage_context(
                stage=stage,
                stage_idx=i,
                device_id=device_ids[i],
                target=args.target,
                core_mask=core_mask,
                max_context_len=args.max_context_len,
                decrypt_key_path=args.decrypt_key_path,
                verbose=args.verbose,
                share_internal=(share_internal and i == 0),
            )
            if not ok:
                raise RuntimeError(f"init {stage.name} failed")

        if vision_enabled:
            if not init_vision(
                vision=vision,
                target=args.target,
                core_mask=vision_core_mask,
                device_id=vision_device_id,
                decrypt_key_path=args.decrypt_key_path,
                verbose=args.verbose,
                share_internal=share_internal,
            ):
                raise RuntimeError("init vision failed")

            if vision.embedding_dim != stages[0].embedding_dim:
                raise RuntimeError(
                    f"vision embedding_dim={vision.embedding_dim} does not match "
                    f"stage0 embedding_dim={stages[0].embedding_dim}"
                )

            if share_internal:
                try:
                    shared_mems = setup_shared_internal_mem(vision, stages[0])
                except Exception as e:
                    raise RuntimeError(
                        f"shared internal mem setup failed: {e}; "
                        "retry with --no_share_internal_mem"
                    )
            else:
                print("[shared] disabled: vision uses library-managed internal memory")

        ref = stages[0]
        for stage in stages[1:]:
            if stage.embedding_dim != ref.embedding_dim:
                raise RuntimeError(
                    f"embedding_dim mismatch: {ref.name}={ref.embedding_dim}, "
                    f"{stage.name}={stage.embedding_dim}"
                )
            if stage.vocab_size != ref.vocab_size:
                print(
                    f"WARNING: vocab_size differs across stages: "
                    f"{ref.name}={ref.vocab_size}, {stage.name}={stage.vocab_size}"
                )
            if stage.max_ctx_len != ref.max_ctx_len:
                print(
                    f"WARNING: max_ctx_len differs across stages: "
                    f"{ref.name}={ref.max_ctx_len}, {stage.name}={stage.max_ctx_len}"
                )

        vocab_size = stages[0].vocab_size
        embeds_flat = np.memmap(args.embed_path, dtype=np.float16, mode="r")
        if vocab_size <= 0 or embeds_flat.size % vocab_size != 0:
            raise RuntimeError(
                f"invalid embedding file size: elems={embeds_flat.size}, vocab_size={vocab_size}"
            )
        embedding_dim = embeds_flat.size // vocab_size
        embeds_data = embeds_flat.reshape(vocab_size, embedding_dim)
        print(f"embedding file: vocab_size={vocab_size}, embedding_dim={embedding_dim}")

        if embedding_dim != stages[0].embedding_dim:
            print(
                f"WARNING: embedding.bin dim={embedding_dim}, "
                f"stage0 queried embedding_dim={stages[0].embedding_dim}"
            )
        if len(tokenizer) != vocab_size:
            print(
                f"Warning: tokenizer vocab size ({len(tokenizer)}) != "
                f"llm_config.vocab_size ({vocab_size})"
            )

        for i, stage in enumerate(stages):
            ok = init_stage_session(
                stage=stage,
                pipeline=pipeline,
                stage_idx=i,
                max_context_len=args.max_context_len,
                ignore_eos=args.ignore_eos,
                special_eos_ids=special_eos_ids,
                special_bos_ids=special_bos_ids,
                presence_penalty=args.presence_penalty,
                kvcache_policy=args.kvcache_policy,
                verbose=args.verbose,
            )
            if not ok:
                raise RuntimeError(f"init {stage.name} session failed")

        images = []
        if vision_enabled:
            image_paths = parse_image_paths(args.image if args.image else [DEFAULT_IMAGE])
            for index, path in enumerate(image_paths):
                img = cv2.imread(path)
                if img is None:
                    raise FileNotFoundError(f"failed to read image: {path}")
                images.append(img)
                print(f"image[{index}]: {path}, shape={img.shape}")

        prompt = build_prompt(args.prompt, len(images), args.enable_thinking)
        prompt_tokens = count_prompt_tokens(tokenizer, prompt)
        ctx_len = args.max_context_len if args.max_context_len > 0 else stages[0].max_ctx_len
        if vision_enabled:
            estimated_tokens = estimate_multimodal_input_tokens(
                prompt_tokens, len(images), vision.n_image_tokens
            )
        else:
            estimated_tokens = prompt_tokens
        print(
            f"prompt tokens={prompt_tokens}, estimated input tokens={estimated_tokens}, "
            f"max_context_len={ctx_len}"
        )
        if ctx_len > 0 and estimated_tokens >= ctx_len:
            raise RuntimeError(
                f"prompt too long: estimated={estimated_tokens}, max_context_len={ctx_len}"
            )

        repeat = max(1, int(args.repeat))
        for run_idx in range(repeat):
            if repeat > 1:
                print(f"\n=== Run {run_idx + 1}/{repeat} ===")

            for stage in stages:
                try:
                    stage.rknn.clear_kvcache()
                except Exception:
                    pass

            image_input = None
            image_features = None
            vision_latency = 0.0
            if vision_enabled:
                image_input, image_features, vision_latency, n_tokens = run_vision(
                    vision, images, args.keep_aspect
                )
                print(
                    f"vision done: n_image={int(image_input.n_image)}, "
                    f"n_image_tokens={int(image_input.n_image_tokens)}, "
                    f"latency={vision_latency * 1000.0:.2f} ms"
                )
                if share_internal:
                    print("vision runtime kept alive (shared internal memory)")
                elif not args.vision_keep_alive:
                    vision.release()
                    print("vision runtime released before prefill")

            LAST_RESULT.reset_generation()
            print("\n=== Prefill Pipeline ===")
            print(f"Input: {args.prompt}")
            print("\nOutput: ", end="", flush=True)

            prefill_start = time.perf_counter()
            ok, prefill_tokens = run_pipeline_once(
                stages,
                pipeline,
                prompt=prompt,
                inputs=[image_input] if image_input is not None else None,
                enable_thinking=args.enable_thinking,
                pipeline_mode=args.pipeline_mode,
            )
            prefill_end = time.perf_counter()
            if not ok:
                raise RuntimeError("prefill failed")

            next_token = LAST_RESULT.wait_next_token(timeout=2.0)
            if next_token is None:
                raise RuntimeError(
                    "prefill finished but LAST stage did not return a sampled token"
                )
            expected_last_stage = len(stages) - 1
            source_stage = LAST_RESULT.get_source_stage()
            if source_stage != expected_last_stage:
                raise RuntimeError(
                    f"invalid sampled-token source: stage{source_stage}, expected stage{expected_last_stage}"
                )
            flush_generated_text()
            print(f"\n[pipeline] prefill sampled token={next_token} from stage{source_stage}")

            print("\n=== Decode Loop ===")
            decode_start = time.perf_counter()
            decode_tokens = 0

            for step in range(args.max_new_token):
                vlog(pipeline, f"[Decode {step + 1}] token={next_token}")
                ok, _ = run_pipeline_once(
                    stages,
                    pipeline,
                    input_tokens=[int(next_token)],
                    enable_thinking=args.enable_thinking,
                    pipeline_mode=args.pipeline_mode,
                )
                if not ok:
                    print(f"\ndecode step {step + 1} failed")
                    break

                next_token = LAST_RESULT.wait_next_token(timeout=2.0)
                if next_token is None:
                    print(f"\ndecode step {step + 1} did not return token from LAST stage result_callback")
                    break
                source_stage = LAST_RESULT.get_source_stage()
                if source_stage != len(stages) - 1:
                    print(
                        f"\ndecode step {step + 1} got token from stage{source_stage}, "
                        f"expected stage{len(stages) - 1}; stop"
                    )
                    break
                flush_generated_text()

                decode_tokens += 1
                if not args.ignore_eos and int(next_token) in set(special_eos_ids):
                    vlog(pipeline, f"decode step {step + 1} reached EOS token {next_token}")
                    break

            decode_end = time.perf_counter()
            print()

            prefill_ms = (prefill_end - prefill_start) * 1000.0
            decode_ms = (decode_end - decode_start) * 1000.0
            printf_perf(prefill_tokens, prefill_ms, decode_tokens, decode_ms, vision_latency * 1000.0)

        print("done")
        return 0

    finally:
        destroy_shared_internal_mem(vision, shared_mems)
        vision.release()
        for stage in stages:
            stage.release()


if __name__ == "__main__":
    raise SystemExit(main())
