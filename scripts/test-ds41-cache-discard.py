#!/usr/bin/env python3
"""Verify DS41 clean GGUF mmap discard on a real MixedQ2 tensor.

This is a component test only: it touches 512 MiB of one verified shard,
keeps a 4 MiB ROCm copy, discards only the shard's clean mmap/page-cache,
and proves both file bytes and the device copy remain unchanged.
"""
from __future__ import annotations

import gc
import hashlib
import mmap
import os
from pathlib import Path
from _ds41_artifact import MODEL_FILE

import gguf
import torch

MODEL = MODEL_FILE


def rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS unavailable")


def main() -> None:
    assert torch.cuda.is_available(), "ROCm device required"
    assert hasattr(mmap, "MADV_DONTNEED")
    assert hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED")

    reader = gguf.GGUFReader(str(MODEL))
    tensor = next(t for t in reader.tensors if t.name == "blk.0.ffn_gate_exps")
    view = tensor.data.reshape(-1)
    touched = view[: 512 * 1024 * 1024]
    sample_view = view[: 4 * 1024 * 1024]

    rss0 = rss_bytes()
    before = hashlib.sha256(memoryview(sample_view)).hexdigest()
    stride_sum = int(touched[::4096].sum(dtype="uint64"))
    rss1 = rss_bytes()

    sample = torch.from_numpy(sample_view).to("cuda")
    torch.cuda.synchronize()
    device_before = sample.cpu().numpy().tobytes()

    reader.data._mmap.madvise(mmap.MADV_DONTNEED)
    fd = os.open(MODEL, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    gc.collect()
    rss2 = rss_bytes()

    after = hashlib.sha256(memoryview(sample_view)).hexdigest()
    device_after = sample.cpu().numpy().tobytes()
    torch.cuda.synchronize()

    result = {
        "status": "PASS",
        "tensor": tensor.name,
        "touched_bytes": int(touched.nbytes),
        "rss_before": rss0,
        "rss_after_touch": rss1,
        "rss_after_discard": rss2,
        "rss_touch_delta": rss1 - rss0,
        "rss_discard_delta": rss2 - rss1,
        "stride_sum": stride_sum,
        "checksum_before": before,
        "checksum_after": after,
        "file_bytes_equal": before == after,
        "device_copy_equal": device_before == device_after,
    }
    assert before == after
    assert device_before == device_after
    assert rss2 < rss1, result
    print(result)


if __name__ == "__main__":
    main()
