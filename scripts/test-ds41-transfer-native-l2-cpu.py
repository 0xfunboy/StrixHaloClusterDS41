#!/usr/bin/env python3
"""CPU-only gates for TRANSFER_DS4_NATIVE_001 L2 Antirez target/Engram."""
from __future__ import annotations

import json
import math
import mmap
import os
import struct
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
Q2 = Path("/home/funboy/models/ds41/ds4-v41-q2/DeepSeek-V4.1-Flash-Q2.gguf")
HF = Path("/home/funboy/models/gguf/deepseek-v4.1-flash-mixedq2-densefix")

sys.path[:0] = [
    str(ROOT),
    str(ROOT / ".vendor/vllm-dsv41"),
    str(ROOT / ".vendor/gguf-plugin"),
    str(ROOT / ".vendor/llama-v41/gguf-py"),
]

import numpy as np
from transformers import AutoTokenizer
import torch

from runtime.ds41.gguf_stream_cache import (
    aligned_mmap_span,
    drop_consumed_tensor_cache,
    stage_anonymous_copy,
)
from runtime.ds41.native_antirez_engram import (
    decode_rows264,
    inspect_native_engram,
    verify_native_hash_contract,
)
from vllm.models.deepseek_v4_1.common.engram import (
    EngramLayout,
    build_compressed_token_map,
    compute_hash_multipliers,
)
from vllm.models.deepseek_v4_1.amd.vl_model import _make_deepseek_v4_vl_weights_mapper
from vllm_gguf_plugin.gguf_files import GGUFModelFiles
from vllm_gguf_plugin.weights_adapter.deepseek_v41 import DeepseekV41GGUFAdapter


def scalar_ds4_row(raw: bytes) -> np.ndarray:
    if len(raw) != 264:
        raise AssertionError(len(raw))
    out = np.empty(256, dtype=np.float32)
    for j in range(256):
        byte = raw[j]
        scale = raw[256 + j // 32]
        if (byte & 127) == 127 or scale == 255:
            raise AssertionError("reserved native Engram code")
        exponent = (byte >> 3) & 15
        mantissa = byte & 7
        value = (
            math.ldexp(float(8 + mantissa), exponent - 10)
            if exponent
            else math.ldexp(float(mantissa), -9)
        )
        if byte & 128:
            value = -value
        value = math.ldexp(value, int(scale) - 127)
        f32 = struct.unpack("<I", struct.pack("<f", value))[0]
        f32 = (f32 + 0x7FFF + ((f32 >> 16) & 1)) & 0xFFFF0000
        out[j] = struct.unpack("<f", struct.pack("<I", f32))[0]
    return out


def main() -> None:
    result: dict[str, object] = {"schema": "ds41-transfer-l2-cpu-gate-v1"}

    meta = inspect_native_engram(str(Q2))
    result["header"] = {
        "encoding": meta["encoding"],
        "layers": list(meta["layers"]),
        "rows": list(meta["rows"]),
        "token_map_entries": len(meta["token_map"]),
    }

    raw_cfg = json.loads((HF / "config.json").read_text())
    text_cfg = SimpleNamespace(**raw_cfg["text_config"])
    layout = EngramLayout(text_cfg)
    tokenizer = AutoTokenizer.from_pretrained(HF, trust_remote_code=False)
    token_map, compressed = build_compressed_token_map(tokenizer)
    if compressed != layout.compressed_vocab_size:
        raise AssertionError((compressed, layout.compressed_vocab_size))
    multipliers = compute_hash_multipliers(
        layout.layer_ids, layout.max_ngram_size, compressed
    )
    result["hash_contract"] = verify_native_hash_contract(
        str(Q2),
        token_map=token_map,
        multipliers=multipliers,
        layout=layout,
        compressed_pad_id=token_map[layout.pad_token_id],
    )

    os.environ["DS41_ANTIREZ_Q2"] = "1"
    mapping = DeepseekV41GGUFAdapter().build_name_map(
        GGUFModelFiles((str(Q2),)), None
    )
    if len(mapping) != 1038:
        raise AssertionError(f"Antirez target mapping count {len(mapping)} != 1038")
    result["target_mapping"] = {
        "mapped": len(mapping),
        "native_engram_special": 8,
        "total": len(mapping) + 8,
    }

    # The GGUF iterator emits ``*.weight_type`` companions for packed tensors.
    # In language-only streaming mode every such name must be rerooted under
    # ``language_model.`` before the wrapper delegates to the child loader.
    outer = _make_deepseek_v4_vl_weights_mapper(
        getattr(text_cfg, "expert_dtype", "fp4"), "weight_scale_inv"
    )
    streaming_cases = {
        "head.weight": "language_model.lm_head.weight",
        "head.weight_type": "language_model.lm_head.weight_type",
        "embed.weight": "language_model.model.embed_tokens.weight",
        "embed.weight_type": "language_model.model.embed_tokens.weight_type",
        "norm.weight": "language_model.model.norm.weight",
        "norm.weight_type": "language_model.model.norm.weight_type",
        "layers.0.attn.wq_a.weight": "language_model.model.layers.0.attn.wq_a.weight",
        "layers.0.attn.wq_a.weight_type": "language_model.model.layers.0.attn.wq_a.weight_type",
    }
    mapped_cases = {}
    for source_name, expected_name in streaming_cases.items():
        mapped_name = outer._map_name(source_name)
        if mapped_name != expected_name:
            raise AssertionError(
                f"streaming name map {source_name}: {mapped_name!r} != {expected_name!r}"
            )
        if not mapped_name.startswith("language_model."):
            raise AssertionError(f"streaming name escaped language_model: {mapped_name}")
        mapped_cases[source_name] = mapped_name
    result["streaming_name_map"] = {"status": "PASS", "cases": mapped_cases}

    # Progressive cache eviction is a file-cache policy only: verify its
    # page-aligned range math and real Linux MADV/FADV calls on a temporary
    # mmap without touching model files or global caches.
    page = mmap.PAGESIZE
    with tempfile.NamedTemporaryFile() as tmp:
        file_size = page * 4 + 123
        tmp.truncate(file_size)
        tmp.flush()
        fd = os.open(tmp.name, os.O_RDONLY)
        mm = mmap.mmap(fd, file_size, access=mmap.ACCESS_READ)
        try:
            start, length = aligned_mmap_span(
                data_offset=123,
                tensor_offset=page - 100,
                n_bytes=page + 321,
                file_size=file_size,
            )
            if start % page != 0 or length <= 0 or start + length > file_size:
                raise AssertionError((start, length, file_size, page))
            source = np.frombuffer(mm, dtype=np.uint8, count=page + 321, offset=page)
            source_before = source.copy()
            staged = stage_anonymous_copy(source)
            if np.shares_memory(staged, source):
                raise AssertionError("anonymous stage shares mmap source")
            if not np.array_equal(staged, source_before):
                raise AssertionError("anonymous stage changed bytes")
            staged[0] ^= 0xFF
            if source[0] != source_before[0]:
                raise AssertionError("staged mutation reached mmap source")
            dropped = drop_consumed_tensor_cache(
                mm,
                fd,
                data_offset=123,
                tensor_offset=page - 100,
                n_bytes=page + 321,
                file_size=file_size,
            )
            if dropped != (start, length):
                raise AssertionError((dropped, start, length))
            del source
        finally:
            mm.close()
            os.close(fd)
    result["progressive_cache_drop"] = {
        "status": "PASS",
        "page_size": page,
        "aligned_start": start,
        "aligned_length": length,
        "global_cache_operation": False,
    }
    result["anonymous_staging"] = {
        "status": "PASS",
        "shares_source_memory": False,
        "byte_exact_before_mutation": True,
        "source_unchanged_after_staged_mutation": True,
        "writable": bool(staged.flags.writeable),
        "c_contiguous": bool(staged.flags.c_contiguous),
    }

    import gguf

    reader = gguf.GGUFReader(str(Q2))
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    ordinary = [tensor for tensor in reader.tensors if tensor.name in mapping]
    largest = max(ordinary, key=lambda tensor: int(tensor.n_bytes))
    # Verify the staging helper against a real quantized Antirez target slice.
    real_source = np.asarray(largest.data).reshape(-1)[: 1 << 20]
    real_stage = stage_anonymous_copy(real_source)
    if np.shares_memory(real_stage, real_source) or not np.array_equal(real_stage, real_source):
        raise AssertionError("real Antirez staging is not byte-exact/independent")
    result["target_tensor_residency"] = {
        "ordinary_tensor_count": len(ordinary),
        "largest_name": largest.name,
        "largest_bytes": int(largest.n_bytes),
        "largest_gib": int(largest.n_bytes) / (1024 ** 3),
        "largest_anonymous_stage_bytes": int(largest.n_bytes),
        "native_engram_tables_skipped_by_target_iterator": True,
        "real_quantized_sample_stage_bytes": int(real_stage.nbytes),
        "real_quantized_sample_stage_byte_exact": True,
        "real_quantized_sample_shares_source": False,
    }
    row_checks = []
    for layer in (1, 14):
        table = tensors[f"blk.{layer}.engram_embd.weight"].data
        candidates = [1, int(table.shape[0] // 3), int(table.shape[0] // 2)]
        selected = candidates[-1]
        for row in candidates:
            raw = np.asarray(table[row], dtype=np.uint8)
            if len(set(int(x) for x in raw[256:])) > 1:
                selected = row
                break
        ids = np.asarray([0, selected], dtype=np.int64)
        raw_rows = np.asarray(table[ids], dtype=np.uint8)
        native = decode_rows264(raw_rows)
        reference = np.stack(
            [scalar_ds4_row(bytes(np.asarray(row, dtype=np.uint8))) for row in raw_rows]
        )
        if not np.array_equal(native.view(np.uint32), reference.view(np.uint32)):
            raise AssertionError(f"row decode mismatch layer={layer}")
        scales = raw_rows[1, 256:]
        row_checks.append(
            {
                "layer": layer,
                "rows": [int(x) for x in ids],
                "scale_unique": int(np.unique(scales).size),
                "bit_exact_bf16_reference": True,
            }
        )
    result["row_decode"] = row_checks
    result["status"] = "PASS"
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
