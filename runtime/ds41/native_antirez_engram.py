# SPDX-License-Identifier: Apache-2.0
"""Native Antirez GGUF Engram provider for TRANSFER_DS4_NATIVE_001.

The 100+ GiB Engram tables stay mmap-backed in the calibrated GGUF. Only rows
requested by the native V4.1 hash path are copied/dequantized, and the row
cache is bounded. q/k/WKV are sourced from the same GGUF; no Engram2 sidecar
is mixed into this profile.
"""
from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROW_DIM = 256
ROW_BYTES = 264
EXPECTED_ENCODING = "e4m3_e8m0_32_row264"
EXPECTED_LAYERS = (1, 14)
EXPECTED_ROWS = (384006168, 384016682)


def _reader(path: str):
    import gguf
    return gguf.GGUFReader(path)


def _field(reader, key: str):
    if key not in reader.fields:
        raise RuntimeError(f"native Engram GGUF metadata missing {key}")
    return reader.fields[key].contents()


def _tensor_map(reader) -> dict[str, Any]:
    return {tensor.name: tensor for tensor in reader.tensors}


def inspect_native_engram(path: str) -> dict[str, Any]:
    """Read and validate the GGUF Engram metadata/tensor headers."""
    reader = _reader(path)
    tensors = _tensor_map(reader)
    meta: dict[str, Any] = {
        "encoding": _field(reader, "deepseek41.engram.encoding"),
        "layers": tuple(int(x) for x in _field(reader, "deepseek41.engram.layer_ids")),
        "rows": tuple(int(x) for x in _field(reader, "deepseek41.engram.rows")),
        "primes": tuple(int(x) for x in _field(reader, "deepseek41.engram.primes")),
        "multipliers": tuple(
            int(x) for x in _field(reader, "deepseek41.engram.multipliers")
        ),
        "compressed_vocab_size": int(
            _field(reader, "deepseek41.engram.compressed_vocab_size")
        ),
        "pad_id": int(_field(reader, "deepseek41.engram.pad_id")),
        "token_map": tuple(
            int(x) for x in _field(reader, "deepseek41.engram.token_map")
        ),
        "tensors": {},
    }
    if meta["encoding"] != EXPECTED_ENCODING:
        raise RuntimeError(f"unsupported native Engram encoding {meta['encoding']!r}")
    if meta["layers"] != EXPECTED_LAYERS or meta["rows"] != EXPECTED_ROWS:
        raise RuntimeError(
            f"native Engram layer/row mismatch {meta['layers']} {meta['rows']}"
        )
    for layer, nrows in zip(meta["layers"], meta["rows"], strict=True):
        specs = {
            "table": (
                f"blk.{layer}.engram_embd.weight",
                (nrows, ROW_BYTES),
                "I8",
            ),
            "q": (f"blk.{layer}.engram_q_norm.weight", (4, 5120), "F32"),
            "k": (f"blk.{layer}.engram_k_norm.weight", (4, 5120), "F32"),
            "wkv": (
                f"blk.{layer}.engram_kv.weight",
                (25600, 6144),
                "F16",
            ),
        }
        for role, (name, shape, qtype) in specs.items():
            if name not in tensors:
                raise RuntimeError(f"native Engram tensor missing {name}")
            tensor = tensors[name]
            observed = (
                tuple(int(x) for x in tensor.data.shape),
                tensor.tensor_type.name,
            )
            if observed != (shape, qtype):
                raise RuntimeError(
                    f"native Engram {role} contract mismatch layer={layer}: "
                    f"{observed} != {(shape, qtype)}"
                )
            meta["tensors"][(layer, role)] = {
                "name": name,
                "shape": observed[0],
                "type": observed[1],
                "data_offset": int(tensor.data_offset),
                "nbytes": int(tensor.n_bytes),
            }
    return meta


def verify_native_hash_contract(
    path: str,
    *,
    token_map,
    multipliers: torch.Tensor,
    layout,
    compressed_pad_id: int,
) -> dict[str, Any]:
    """Require exact vLLM-vs-GGUF hash semantics before native Engram use."""
    meta = inspect_native_engram(path)
    built_map = tuple(int(x) for x in token_map)
    if built_map != meta["token_map"]:
        first = next(
            (
                i
                for i, (a, b) in enumerate(zip(built_map, meta["token_map"]))
                if a != b
            ),
            min(len(built_map), len(meta["token_map"])),
        )
        raise RuntimeError(
            f"native Engram token_map mismatch at={first} "
            f"built={len(built_map)} gguf={len(meta['token_map'])}"
        )
    if int(layout.compressed_vocab_size) != meta["compressed_vocab_size"]:
        raise RuntimeError("native Engram compressed-vocab mismatch")
    if int(compressed_pad_id) != meta["pad_id"]:
        raise RuntimeError(
            f"native Engram compressed-pad mismatch "
            f"{compressed_pad_id} != {meta['pad_id']}"
        )
    flat_primes = tuple(
        int(p)
        for layer_primes in layout.primes
        for ngram in layer_primes
        for p in ngram
    )
    if flat_primes != meta["primes"]:
        raise RuntimeError("native Engram prime layout mismatch")
    flat_multipliers = tuple(
        int(x) for x in multipliers.detach().cpu().reshape(-1).tolist()
    )
    if flat_multipliers != meta["multipliers"]:
        raise RuntimeError("native Engram hash-multiplier mismatch")
    if tuple(int(x) for x in layout.layer_ids) != meta["layers"]:
        raise RuntimeError("native Engram layer-id mismatch")
    if tuple(int(x) for x in layout.num_embeddings) != meta["rows"]:
        raise RuntimeError("native Engram row-count mismatch")
    return {
        "status": "PASS",
        "token_map_entries": len(built_map),
        "compressed_vocab_size": meta["compressed_vocab_size"],
        "pad_id": meta["pad_id"],
        "layers": list(meta["layers"]),
        "rows": list(meta["rows"]),
        "encoding": meta["encoding"],
    }


def decode_rows264(raw: np.ndarray) -> np.ndarray:
    """Decode DS4 E4M3/E8M0 row264 bytes and round values to BF16 RNE."""
    data = np.asarray(raw, dtype=np.uint8)
    if data.ndim != 2 or data.shape[1] != ROW_BYTES:
        raise ValueError(f"native Engram raw shape {data.shape} != [N,{ROW_BYTES}]")
    codes = data[:, :ROW_DIM]
    scales = data[:, ROW_DIM:]
    if np.any((codes & np.uint8(127)) == np.uint8(127)) or np.any(
        scales == np.uint8(255)
    ):
        raise ValueError("native Engram row contains reserved E4M3/E8M0 value")

    exponent = ((codes >> np.uint8(3)) & np.uint8(15)).astype(np.int32)
    mantissa = (codes & np.uint8(7)).astype(np.int32)
    normal = np.ldexp((8 + mantissa).astype(np.float32), exponent - 10)
    subnormal = np.ldexp(mantissa.astype(np.float32), -9)
    values = np.where(exponent != 0, normal, subnormal).astype(np.float32)
    values = np.where(
        (codes & np.uint8(128)) != 0, -values, values
    ).astype(np.float32)
    values = np.ldexp(
        values,
        np.repeat(scales.astype(np.int32), 32, axis=1) - 127,
    ).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("native Engram row decoded non-finite value")

    bits = values.view(np.uint32)
    rounded = (
        bits
        + np.uint32(0x7FFF)
        + ((bits >> np.uint32(16)) & np.uint32(1))
    ) & np.uint32(0xFFFF0000)
    out = rounded.view(np.float32)
    if not np.isfinite(out).all():
        raise ValueError("native Engram BF16 rounding produced non-finite value")
    return out.copy()


class NativeGGUFRowCache:
    """Bounded LRU over a mmap-backed GGUF Engram table."""

    def __init__(
        self,
        table: np.ndarray,
        *,
        rows: int,
        max_rows: int,
    ) -> None:
        if table.shape != (rows, ROW_BYTES):
            raise ValueError(
                f"native Engram table shape {table.shape} != {(rows, ROW_BYTES)}"
            )
        if max_rows <= 0:
            raise ValueError("native Engram cache must be positive")
        self.table = table
        self.rows = int(rows)
        self.max_rows = int(max_rows)
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.rows_read = 0

    def lookup(self, row_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(row_ids, dtype=np.int64)
        flat = ids.reshape(-1)
        if flat.size == 0:
            return np.empty((*ids.shape, ROW_DIM), dtype=np.float32)
        lo, hi = int(flat.min()), int(flat.max())
        if lo < 0 or hi >= self.rows:
            raise IndexError(f"native Engram rows {lo}..{hi} outside table")

        unique = np.unique(flat)
        missing = [int(row) for row in unique if int(row) not in self.cache]
        self.hits += int(flat.size - len(missing))
        self.misses += len(missing)
        if missing:
            raw = np.asarray(self.table[np.asarray(missing, dtype=np.int64)])
            decoded = decode_rows264(raw)
            for row, value in zip(missing, decoded, strict=True):
                self.cache[row] = value
                self.cache.move_to_end(row)
                self.rows_read += 1
            while len(self.cache) > self.max_rows:
                self.cache.popitem(last=False)

        ordered = []
        for raw_row in flat:
            row = int(raw_row)
            value = self.cache[row]
            self.cache.move_to_end(row)
            ordered.append(value)
        return np.stack(ordered, axis=0).reshape(*ids.shape, ROW_DIM)


class NativeGGUFEngramEmbedding(nn.Module):
    """TP head-sharded native FP8 table with bounded host LRU."""

    def __init__(
        self,
        reader,
        *,
        layer_id: int,
        num_embeddings: int,
        dim: int,
        head_sizes: tuple[int, ...],
    ) -> None:
        super().__init__()
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        from vllm.triton_utils import triton

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        if dim != ROW_DIM or sum(head_sizes) != num_embeddings:
            raise ValueError("invalid native Engram table/head dimensions")
        self.num_embeddings = int(num_embeddings)
        self.dim = int(dim)
        self.n_hash_cols = len(head_sizes)
        self.part_n_hash_cols = triton.cdiv(self.n_hash_cols, tp_size)
        self.head_start = tp_rank * self.part_n_hash_cols
        head_end = min(self.head_start + self.part_n_hash_cols, self.n_hash_cols)
        self.local_heads = max(0, head_end - self.head_start)
        self.vocab_start_idx = sum(head_sizes[: self.head_start])
        self.vocab_end_idx = sum(head_sizes[:head_end])
        self.part_num_embeddings = self.vocab_end_idx - self.vocab_start_idx
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.layer_id = int(layer_id)
        tensor = _tensor_map(reader)[f"blk.{layer_id}.engram_embd.weight"]
        max_rows = int(os.environ.get("DS41_ENGRAM_CACHE_ROWS", "65536"))
        self.cache = NativeGGUFRowCache(
            tensor.data,
            rows=num_embeddings,
            max_rows=max_rows,
        )
        self._reader = reader
        self.register_buffer("_anchor", torch.empty(0), persistent=False)

    def lookup(
        self, indices: torch.Tensor, out: torch.Tensor, background: bool = False
    ) -> None:
        del background
        if indices.ndim != 2 or indices.shape[1] != self.n_hash_cols:
            raise ValueError(
                f"native Engram ids shape {tuple(indices.shape)} != [T,{self.n_hash_cols}]"
            )
        if out.shape != (indices.shape[0], self.part_n_hash_cols, self.dim):
            raise ValueError(f"native Engram output shape mismatch {tuple(out.shape)}")
        out.zero_()
        if not indices.shape[0] or not self.local_heads:
            return
        local_ids = (
            indices[:, self.head_start : self.head_start + self.local_heads]
            .detach()
            .to("cpu", non_blocking=False)
            .numpy()
            .astype(np.int64, copy=False)
        )
        lo, hi = int(local_ids.min()), int(local_ids.max())
        if lo < self.vocab_start_idx or hi >= self.vocab_end_idx:
            raise RuntimeError(
                f"native Engram ownership violation rank={self.tp_rank}: "
                f"{lo}..{hi} not in {self.vocab_start_idx}..{self.vocab_end_idx}"
            )
        rows = self.cache.lookup(local_ids)
        out[:, : self.local_heads].copy_(
            torch.from_numpy(rows).to(
                device=out.device, dtype=torch.bfloat16, non_blocking=False
            )
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        out = torch.empty(
            (indices.shape[0], self.part_n_hash_cols, self.dim),
            dtype=torch.bfloat16,
            device=indices.device,
        )
        self.lookup(indices, out)
        return out

    def stats(self) -> dict[str, int | str]:
        return {
            "layer": self.layer_id,
            "rank": self.tp_rank,
            "hits": self.cache.hits,
            "misses": self.cache.misses,
            "rows_read": self.cache.rows_read,
        }


class NativeGGUFBF16Linear(nn.Module):
    """Materialize F16 WKV once into the native BF16 compute dtype."""

    def __init__(self, source: np.ndarray, *, device: torch.device) -> None:
        super().__init__()
        if source.ndim != 2 or source.dtype != np.float16:
            raise ValueError(f"native Engram WKV mismatch {source.shape} {source.dtype}")
        out_features, in_features = source.shape
        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                dtype=torch.bfloat16,
                device=device,
            ),
            requires_grad=False,
        )
        chunk_rows = int(os.environ.get("DS41_ANTIREZ_WKV_CHUNK_ROWS", "256"))
        if not 1 <= chunk_rows <= 2048:
            raise ValueError("DS41_ANTIREZ_WKV_CHUNK_ROWS must be 1..2048")
        with torch.no_grad():
            for start in range(0, out_features, chunk_rows):
                end = min(out_features, start + chunk_rows)
                self.weight[start:end].copy_(
                    torch.from_numpy(source[start:end]).to(
                        device=device, dtype=torch.bfloat16
                    )
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight)


def load_native_antirez_engram(
    *,
    path: str,
    layer_id: int,
    num_embeddings: int,
    dim: int,
    hc_mult: int,
    head_sizes: tuple[int, ...],
    device: torch.device,
):
    """Create embedding + q/k + WKV from one calibrated Antirez GGUF."""
    meta = inspect_native_engram(path)
    if layer_id not in meta["layers"]:
        raise ValueError(f"native Engram layer {layer_id} absent from GGUF")
    layer_index = meta["layers"].index(layer_id)
    if int(num_embeddings) != meta["rows"][layer_index]:
        raise ValueError("native Engram row count differs from V4.1 config")
    flat_primes = meta["primes"][
        layer_index * len(head_sizes) : (layer_index + 1) * len(head_sizes)
    ]
    if tuple(int(x) for x in head_sizes) != tuple(flat_primes):
        raise ValueError(f"native Engram head-size/primes mismatch layer={layer_id}")

    reader = _reader(path)
    tensors = _tensor_map(reader)
    q_src = tensors[f"blk.{layer_id}.engram_q_norm.weight"].data
    k_src = tensors[f"blk.{layer_id}.engram_k_norm.weight"].data
    wkv_src = tensors[f"blk.{layer_id}.engram_kv.weight"].data
    if q_src.shape != (hc_mult, dim) or k_src.shape != (hc_mult, dim):
        raise ValueError(
            f"native Engram q/k mismatch {q_src.shape} {k_src.shape}"
        )
    q = torch.from_numpy(q_src).to(device=device, dtype=torch.float32)
    k = torch.from_numpy(k_src).to(device=device, dtype=torch.float32)
    wkv = NativeGGUFBF16Linear(wkv_src, device=device)
    embedding = NativeGGUFEngramEmbedding(
        reader,
        layer_id=layer_id,
        num_embeddings=num_embeddings,
        dim=ROW_DIM,
        head_sizes=head_sizes,
    )
    source = {
        "path": str(Path(path).resolve()),
        "layer_id": int(layer_id),
        "encoding": meta["encoding"],
        "wkv_source_dtype": "float16",
        "wkv_compute_dtype": "bfloat16",
        "qk_dtype": "float32",
    }
    return source, embedding, q, k, wkv
