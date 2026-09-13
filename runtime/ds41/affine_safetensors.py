"""Read only the requested rows from MLX affine-quantized SafeTensors files.

This is intentionally small and runtime-agnostic.  DeepSeek V4.1's 2-bit Engram
artifact stores rows as MLX affine quantization: packed uint32 values (LSB first),
one scale and bias per 64 values.  The large file is mmap'ed read-only; only
requested pages are faulted in.  No model weights are copied or modified.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import mmap
import os
from pathlib import Path
import struct
from typing import Iterable

import numpy as np

_DTYPE = {
    "U8": np.dtype("u1"),
    "U32": np.dtype("<u4"),
    "F32": np.dtype("<f4"),
    # NumPy has no portable bfloat16 scalar.  Keep the raw words and widen.
    "BF16": np.dtype("<u2"),
}
_ENGRAM_ADVICE_BASES = {"layers.1.engram.embed", "layers.14.engram.embed"}


def bf16_to_float32(raw: np.ndarray) -> np.ndarray:
    words = np.asarray(raw, dtype=np.uint16)
    widened = words.astype(np.uint32) << np.uint32(16)
    return widened.view(np.float32)


def float32_to_bf16_words(values: np.ndarray) -> np.ndarray:
    """Fixture/helper conversion; production files are read, never written."""
    values = np.asarray(values, dtype=np.float32)
    return (values.view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def unpack_affine_2bit(packed: np.ndarray) -> np.ndarray:
    """Unpack MLX affine 2-bit uint32 words, least-significant value first."""
    packed = np.asarray(packed, dtype=np.uint32)
    shifts = np.arange(0, 32, 2, dtype=np.uint32)
    unpacked = (packed[..., None] >> shifts) & np.uint32(0x3)
    return unpacked.reshape(*packed.shape[:-1], packed.shape[-1] * 16)


def dequantize_affine_2bit(
    packed: np.ndarray,
    scales_bf16: np.ndarray,
    biases_bf16: np.ndarray,
    *,
    group_size: int = 64,
) -> np.ndarray:
    """MLX affine dequantization: value = q * scale + bias."""
    if group_size != 64:
        raise ValueError(f"DS41 affine reader expects group_size=64, got {group_size}")
    q = unpack_affine_2bit(packed).astype(np.float32)
    scales = bf16_to_float32(scales_bf16)
    biases = bf16_to_float32(biases_bf16)
    if q.shape[-1] != scales.shape[-1] * group_size:
        raise ValueError(
            f"packed width {q.shape[-1]} does not match {scales.shape[-1]} groups x {group_size}"
        )
    if scales.shape != biases.shape or q.shape[:-1] != scales.shape[:-1]:
        raise ValueError(
            f"shape mismatch q={q.shape} scales={scales.shape} biases={biases.shape}"
        )
    return q * np.repeat(scales, group_size, axis=-1) + np.repeat(
        biases, group_size, axis=-1
    )


@dataclass(frozen=True)
class TensorEntry:
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int


class SafeTensorMMap:
    """Minimal read-only SafeTensors mmap with strict bounds validation."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self._fd = os.open(self.path, os.O_RDONLY)
        size = os.fstat(self._fd).st_size
        prefix = os.pread(self._fd, 8, 0)
        if len(prefix) != 8:
            os.close(self._fd)
            raise ValueError(f"truncated SafeTensors prefix: {self.path}")
        header_len = struct.unpack("<Q", prefix)[0]
        if header_len <= 1 or header_len > min(size - 8, 64 << 20):
            os.close(self._fd)
            raise ValueError(f"invalid SafeTensors header length {header_len}: {self.path}")
        header_raw = os.pread(self._fd, header_len, 8)
        if len(header_raw) != header_len:
            os.close(self._fd)
            raise ValueError(f"truncated SafeTensors header: {self.path}")
        header = json.loads(header_raw)
        self.data_start = 8 + header_len
        self.entries: dict[str, TensorEntry] = {}
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            dtype = spec["dtype"]
            if dtype not in _DTYPE:
                os.close(self._fd)
                raise ValueError(f"unsupported dtype {dtype} for {name}")
            shape = tuple(int(x) for x in spec["shape"])
            begin, end = (int(x) for x in spec["data_offsets"])
            if begin < 0 or end < begin or self.data_start + end > size:
                os.close(self._fd)
                raise ValueError(f"invalid data offsets for {name}")
            expected = int(np.prod(shape, dtype=np.int64)) * _DTYPE[dtype].itemsize
            if end - begin != expected:
                os.close(self._fd)
                raise ValueError(
                    f"tensor byte count mismatch {name}: {end-begin} != {expected}"
                )
            self.entries[name] = TensorEntry(dtype, shape, self.data_start + begin, end - begin)
        self.metadata = header.get("__metadata__", {})
        self._mmap = mmap.mmap(self._fd, 0, access=mmap.ACCESS_READ)
        self.engram_advice: dict[str, dict] = {}

    def set_engram_advice(self, base: str, policy: str) -> dict:
        """Advise complete pages of the two allowed affine Engram embeddings.

        Boundary pages stay unchanged, so adjacent tensors and the file header
        are never included. This changes mapping advice only, not data or LRU
        contents. A failed random request attempts a scoped normal rollback.
        """
        previous = self.engram_advice.get(base, {}).get("effective", "normal")
        result = {"base": base, "requested": policy, "effective": previous,
                  "status": "rejected", "scope": "complete_pages_within_selected_tensors",
                  "errors": [], "ranges": []}
        if base not in _ENGRAM_ADVICE_BASES:
            result["errors"].append("base is not an allowed Engram embedding")
            return result
        self.engram_advice[base] = result
        try:
            if policy not in ("random", "normal"):
                raise ValueError("policy must be random or normal")
            if self.path.suffix != ".safetensors":
                raise ValueError("advice requires a SafeTensors file")
            if self._mmap is None or self._mmap.closed:
                raise ValueError("mapping is closed")
            names = [base + "." + suffix for suffix in ("weight", "scales", "biases")]
            weight, scales, biases = [self.entries[name] for name in names]
            if (weight.dtype != "U32" or scales.dtype != "BF16" or biases.dtype != "BF16"
                    or len(weight.shape) != 2 or any(dim <= 0 for dim in weight.shape)
                    or weight.shape[1] % 4
                    or scales.shape != (weight.shape[0], weight.shape[1] // 4)
                    or biases.shape != scales.shape):
                raise ValueError("invalid affine 2-bit embedding tensor shapes or dtypes")
            page = mmap.PAGESIZE
            for name in names:
                entry = self.entries[name]
                end = entry.offset + entry.nbytes
                if entry.offset < self.data_start or end > len(self._mmap):
                    raise ValueError("tensor range outside mapped payload")
                for other_name, other in self.entries.items():
                    if other_name != name and max(entry.offset, other.offset) < min(end, other.offset + other.nbytes):
                        raise ValueError(f"tensor range overlaps {other_name}")
                start = (entry.offset + page - 1) // page * page
                stop = end // page * page
                length = max(0, stop - start)
                result["ranges"].append({
                    "tensor": name, "tensor_start": entry.offset, "tensor_end_exclusive": end,
                    "advice_start": start if length else None, "advice_length": length,
                    "unadvised_boundary_bytes": entry.nbytes - length,
                    "effective": previous, "applied": False,
                })
        except (KeyError, ValueError, AttributeError) as exc:
            result["errors"].append(str(exc))
            return result
        regions = [region for region in result["ranges"] if region["advice_length"]]
        if not regions:
            result["status"] = "no_complete_pages"
            return result
        advise = getattr(self._mmap, "madvise", None)
        option = getattr(mmap, "MADV_" + policy.upper(), None)
        if not callable(advise) or option is None:
            result["status"] = "unsupported"
            result["errors"].append("requested mmap.madvise option is unavailable")
            return result
        try:
            for region in regions:
                advise(option, region["advice_start"], region["advice_length"])
                region.update(effective=policy, applied=True)
            result.update(status="applied", effective=policy)
        except (OSError, ValueError, OverflowError) as exc:
            result["errors"].append(f"{policy}: {exc}")
            # A failed syscall can have affected part of its range. Never claim
            # that range retained its preceding policy without a successful reset.
            region["effective"] = "unknown"
            result["status"] = "failed"
            if policy == "random":
                normal = getattr(mmap, "MADV_NORMAL", None)
                for region in regions:
                    try:
                        if normal is None:
                            raise ValueError("MADV_NORMAL unavailable for rollback")
                        advise(normal, region["advice_start"], region["advice_length"])
                        region.update(effective="normal", applied=False)
                    except (OSError, ValueError, OverflowError) as rollback_error:
                        region["effective"] = "unknown"
                        result["errors"].append(f"normal rollback: {rollback_error}")
            policies = {region["effective"] for region in regions}
            result["effective"] = next(iter(policies)) if len(policies) == 1 else "mixed_or_unknown"
            if policy == "random" and result["effective"] == "normal":
                result["status"] = "fallback_normal"
        return result

    def close(self) -> None:
        mm = getattr(self, "_mmap", None)
        if mm is not None:
            mm.close()
            self._mmap = None
        fd = getattr(self, "_fd", -1)
        if fd >= 0:
            os.close(fd)
            self._fd = -1

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def array(self, name: str) -> np.ndarray:
        entry = self.entries[name]
        return np.ndarray(
            entry.shape,
            dtype=_DTYPE[entry.dtype],
            buffer=self._mmap,
            offset=entry.offset,
        )

    def bf16(self, name: str, rows: np.ndarray | None = None) -> np.ndarray:
        entry = self.entries[name]
        if entry.dtype != "BF16":
            raise TypeError(f"{name} is {entry.dtype}, expected BF16")
        arr = self.array(name)
        if rows is not None:
            arr = arr[rows]
        return bf16_to_float32(arr)

    def affine2_rows(self, base: str, rows: Iterable[int]) -> np.ndarray:
        rows = np.asarray(list(rows), dtype=np.int64)
        if rows.ndim != 1:
            raise ValueError("rows must be one-dimensional")
        w_name, s_name, b_name = (
            base + ".weight",
            base + ".scales",
            base + ".biases",
        )
        w = self.array(w_name)
        if w.dtype != np.uint32:
            raise TypeError(f"{w_name} must be U32, got {w.dtype}")
        if rows.size and (rows.min() < 0 or rows.max() >= w.shape[0]):
            raise IndexError(f"row outside {base}: min/max={rows.min()}/{rows.max()}")
        return dequantize_affine_2bit(
            w[rows], self.array(s_name)[rows], self.array(b_name)[rows]
        )

    def affine2_chunks(self, base: str, chunk_rows: int = 256):
        total = self.entries[base + ".weight"].shape[0]
        for start in range(0, total, chunk_rows):
            end = min(total, start + chunk_rows)
            yield start, end, self.affine2_rows(base, range(start, end))


class AffineRowLRU:
    """Bounded dequantized row cache used by the disk-backed Engram provider."""

    def __init__(self, source: SafeTensorMMap, base: str, max_rows: int = 65536):
        if max_rows < 0:
            raise ValueError("max_rows must be non-negative")
        self.source = source
        self.base = base
        self.max_rows = max_rows
        self._rows: OrderedDict[int, np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.rows_read = 0
        self._initial_advice_status = None
        if os.environ.get("DS41_ENGRAM_RANDOM_ADVICE", "0") == "1":
            self._initial_advice_status = self.source.set_engram_advice(self.base, "random")
            print("DS41_ENGRAM_ADVICE " + json.dumps(self._initial_advice_status), flush=True)

    @property
    def advice_status(self) -> dict:
        return self.source.engram_advice.get(self.base) or self._initial_advice_status or {
            "base": self.base, "requested": None, "effective": "normal",
            "status": "not_requested", "errors": [], "ranges": [],
        }

    def lookup(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        flat = ids.reshape(-1)
        # Preserve first-use order, then read missing rows in one vectorized mmap gather.
        missing: list[int] = []
        seen: set[int] = set()
        for value in flat:
            i = int(value)
            if i in self._rows:
                self.hits += 1
                self._rows.move_to_end(i)
            else:
                self.misses += 1
                if i not in seen:
                    missing.append(i)
                    seen.add(i)
        if missing:
            values = self.source.affine2_rows(self.base, missing)
            self.rows_read += len(missing)
            for i, row in zip(missing, values, strict=True):
                self._rows[i] = np.asarray(row, dtype=np.float32)
                self._rows.move_to_end(i)
                while len(self._rows) > self.max_rows:
                    self._rows.popitem(last=False)
        if not flat.size:
            width = self.source.entries[self.base + ".weight"].shape[-1] * 16
            return np.empty((*ids.shape, width), dtype=np.float32)
        return np.stack([self._rows[int(i)] for i in flat], axis=0).reshape(
            *ids.shape, -1
        )
