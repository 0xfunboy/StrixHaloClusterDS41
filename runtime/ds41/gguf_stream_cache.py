# SPDX-License-Identifier: Apache-2.0
"""Bounded clean-page eviction for mmap-backed GGUF streaming loads.

The helper never unmaps or mutates checkpoint bytes. It is called only after
the consumer has synchronously copied one yielded tensor into its final model
parameter. MADV_DONTNEED and POSIX_FADV_DONTNEED then allow Linux to reclaim
clean file-backed pages from that already-consumed tensor while the immutable
GGUF remains mapped for later tensors.
"""
from __future__ import annotations

import mmap
import os


def aligned_mmap_span(
    *,
    data_offset: int,
    tensor_offset: int,
    n_bytes: int,
    file_size: int,
    page_size: int = mmap.PAGESIZE,
) -> tuple[int, int]:
    """Return page-aligned (start, length) for one GGUF tensor."""
    values = (data_offset, tensor_offset, n_bytes, file_size, page_size)
    if any(not isinstance(value, int) for value in values):
        raise TypeError("GGUF cache-drop span arguments must be integers")
    if data_offset < 0 or tensor_offset < 0 or n_bytes < 0:
        raise ValueError("negative GGUF cache-drop span")
    if file_size <= 0 or page_size <= 0 or page_size & (page_size - 1):
        raise ValueError("invalid GGUF file/page size")

    raw_start = data_offset + tensor_offset
    raw_end = raw_start + n_bytes
    if raw_start > file_size or raw_end > file_size:
        raise ValueError(
            f"GGUF tensor span outside file: {raw_start}..{raw_end} > {file_size}"
        )
    if n_bytes == 0:
        return raw_start - (raw_start % page_size), 0

    start = raw_start - (raw_start % page_size)
    end = min(file_size, (raw_end + page_size - 1) & ~(page_size - 1))
    return start, end - start


def drop_consumed_tensor_cache(
    mm,
    fd: int,
    *,
    data_offset: int,
    tensor_offset: int,
    n_bytes: int,
    file_size: int,
) -> tuple[int, int]:
    """Discard only clean cache pages belonging to one consumed tensor."""
    if not hasattr(mm, "madvise") or not hasattr(mmap, "MADV_DONTNEED"):
        raise RuntimeError("progressive GGUF cache drop requires mmap MADV_DONTNEED")
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        raise RuntimeError(
            "progressive GGUF cache drop requires POSIX_FADV_DONTNEED"
        )

    start, length = aligned_mmap_span(
        data_offset=data_offset,
        tensor_offset=tensor_offset,
        n_bytes=n_bytes,
        file_size=file_size,
    )
    if length:
        mm.madvise(mmap.MADV_DONTNEED, start, length)
    if n_bytes:
        os.posix_fadvise(
            fd,
            data_offset + tensor_offset,
            n_bytes,
            os.POSIX_FADV_DONTNEED,
        )
    return start, length
