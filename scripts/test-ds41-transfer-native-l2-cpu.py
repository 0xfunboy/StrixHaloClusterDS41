#!/usr/bin/env python3
"""CPU-only gates for TRANSFER_DS4_NATIVE_001 L2 Antirez target/Engram."""
from __future__ import annotations

import json
import math
import os
import struct
import sys
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

    import gguf

    reader = gguf.GGUFReader(str(Q2))
    tensors = {tensor.name: tensor for tensor in reader.tensors}
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
