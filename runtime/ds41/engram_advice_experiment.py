"""Matched sparse-table I/O experiment. Never imported by the serving path.

Only the two embedding readers are reopened. File-specific clean source cache
discard is test preparation, not the proposed production optimization.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import mmap
import os
import time

import numpy as np

from runtime.ds41.affine_safetensors import SafeTensorMMap
from runtime.ds41.fault_diagnostics import FaultDiagnostics, process_snapshot

ORDER = ("A1", "B1", "B2", "A2", "A3", "B3")


def resident_pages(source, start, length):
    """mincore reads residency bits, not the mapped row data."""
    if start % mmap.PAGESIZE or length <= 0 or length % mmap.PAGESIZE:
        raise ValueError("residency requires full aligned pages")
    count = length // mmap.PAGESIZE
    if count > 16 * 1024 * 1024:
        raise ValueError("mincore vector exceeds explicit 16 MiB budget")
    if os.fstat(source._fd).st_uid != os.geteuid():
        raise RuntimeError("file ownership required for trustworthy mincore disclosure")
    libc = ctypes.CDLL(None, use_errno=True)
    call = libc.mincore
    call.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
    call.restype = ctypes.c_int
    vector = (ctypes.c_ubyte * count)()
    view = np.frombuffer(source._mmap, dtype=np.uint8)
    if call(view.ctypes.data + start, length, vector):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    bits = np.ctypeslib.as_array(vector)
    # Chunked summation bounds temporary masks below 64 KiB.
    resident = sum(int(np.count_nonzero(bits[i:i + 65536] & 1))
                   for i in range(0, count, 65536))
    return {"pages": count, "resident_pages": resident, "vector_bytes": count}


def prepare_embedding(embedding, policy, related_sources=()):
    started = time.monotonic()
    old = embedding.source
    cache = embedding.cache
    if cache.source is not old or cache.max_rows != 65536:
        raise RuntimeError("unexpected embedding/cache ownership or capacity")
    old_stat = os.fstat(old._fd)
    fresh = SafeTensorMMap(old.path)  # fresh open description resets read-ahead history
    try:
        fresh_stat = os.fstat(fresh._fd)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
        if (any(getattr(old_stat, k) != getattr(fresh_stat, k) for k in fields)
                or fresh.entries != old.entries or fresh.metadata != old.metadata):
            raise RuntimeError("embedding file identity or layout changed")
        advice = fresh.set_engram_advice(cache.base, policy)
        if advice["status"] != "applied" or advice["effective"] != policy:
            raise RuntimeError(f"requested advice not effective: {advice}")
        record = {"path": str(old.path), "base": cache.base,
                  "old_fd": old._fd, "new_fd": fresh._fd,
                  "fresh_open_description": True, "advice": advice,
                  "cleared_decoded_rows": len(cache._rows), "ranges": [],
                  "cache_max_rows_before": cache.max_rows}
        # All references were consumed by completed synchronous generate calls.
        # Materialized linears and their independent source mapping are untouched.
        old.close()
        embedding.source = cache.source = fresh
        cache._rows.clear()
        record["cache_max_rows_after"] = cache.max_rows
        # Partial range discard can retain entire large page-cache folios at
        # tensor boundaries. Drop PTEs on matching read-only source mappings,
        # then this one exact file's clean cache. This is TEST preparation only.
        # q/k/WKV are already independent, materialized GPU tensors.
        matching = [obj for obj in (fresh, *related_sources)
                    if isinstance(obj, SafeTensorMMap) and obj._fd >= 0
                    and (os.fstat(obj._fd).st_dev, os.fstat(obj._fd).st_ino)
                    == (fresh_stat.st_dev, fresh_stat.st_ino)]
        if not any(obj is fresh for obj in matching):
            raise RuntimeError("fresh mapping missing from scoped source set")
        record["discard_scope"] = "whole_identified_engram_file_test_only"
        record["discard_mapping_count"] = len(matching)
        record["discard_file_identity"] = {k: getattr(fresh_stat, k) for k in fields}
        for region in advice["ranges"]:
            start, length = region["advice_start"], region["advice_length"]
            if not length:
                raise RuntimeError("expected full embedding tensor pages")
            before = resident_pages(fresh, start, length)
            record["ranges"].append({"start": start, "length": length, "before": before})
        for obj in matching:
            obj._mmap.madvise(mmap.MADV_DONTNEED)
        os.posix_fadvise(fresh._fd, 0, 0, os.POSIX_FADV_DONTNEED)
        for region in record["ranges"]:
            start, length = region["start"], region["length"]
            after = resident_pages(fresh, start, length)
            region["after"] = after
            if after["resident_pages"] != 0:
                raise RuntimeError(f"nonzero initial table residency: {record}")
        record["preparation_s"] = time.monotonic() - started
        return record
    except BaseException:
        if cache.source is not fresh:
            fresh.close()
        raise


def model_readers(llm, embedding_type):
    # vLLM freezes its GC heap after initialization. Follow the executor's
    # owned model graph instead of scanning gc.get_objects().
    model = llm.llm_engine.model_executor.driver_worker.get_model()
    modules = list(model.modules())
    embeddings = sorted((obj for obj in modules if isinstance(obj, embedding_type)),
                        key=lambda obj: obj.layer_id)
    sources = [obj._ds41_sidecar_source for obj in modules
               if isinstance(getattr(obj, "_ds41_sidecar_source", None), SafeTensorMMap)]
    if [obj.layer_id for obj in embeddings] != [1, 14] or len(sources) != 2:
        raise RuntimeError("expected exactly two owned embeddings and sidecar sources")
    return embeddings, sources


def run_experiment(llm, prompts, raw, rank, generate, results, checkpoint):
    import torch.distributed as dist
    from vllm.models.deepseek_v4_1.common.disk_engram import DiskAffineEngramEmbedding

    embeddings, sidecar_sources = model_readers(llm, DiskAffineEngramEmbedding)
    if any(obj.tp_rank != rank for obj in embeddings):
        raise RuntimeError("Engram rank ownership mismatch")
    collector = FaultDiagnostics(llm.llm_engine.output_processor, raw, rank=rank)
    preparation = []
    maps = []

    def request(label, prompt, cap, performance):
        collector.begin_request(label)
        row = generate(llm, prompts[prompt]["token_ids"], label=label,
                       max_tokens=cap, ignore_eos=performance)
        row["fault_diagnostic"] = collector.end_request()
        row["request_protocol"] = {
            "temperature": 0.0, "seed": 1, "max_tokens": cap,
            "ignore_eos": performance,
            "prompt_tokens_sha256": hashlib.sha256(json.dumps(
                prompts[prompt]["token_ids"], separators=(",", ":")
            ).encode()).hexdigest(),
        }
        row["engram_advice"] = [obj.cache.advice_status for obj in embeddings]
        row["runtime_switches"] = {key: os.environ.get(key) for key in (
            "DS41_EP_SKIP_REMOTE", "DS41_NATIVE_HIP_MOE",
            "DS41_MHC_COEFF_SINKHORN", "DS41_MHC_PROJECTION_RMS",
            "DS41_DECOMPOSED_QKV_INSERT", "DS41_ATTN_WOB_LLMM1",
        )}
        results.append(row)
        checkpoint(f"ENGRAM_ADVICE_{label}_COMPLETE")

    with collector:
        maps.append(collector.snapshot_maps("before_arms"))
        for arm in ORDER:
            policy = "normal" if arm[0] == "A" else "random"
            begin = time.monotonic()
            prep = {"arm": arm, "policy": policy, "before": process_snapshot(rank)}
            prep["embeddings"] = [prepare_embedding(obj, policy, sidecar_sources)
                                  for obj in embeddings]
            # Both ranks must pass residency checks before either generates.
            dist.barrier()
            prep["after"] = process_snapshot(rank)
            prep["wall_s_including_barrier"] = time.monotonic() - begin
            preparation.append(prep)
            (raw / f"engram-preparation-rank{rank}.json").write_text(
                json.dumps(preparation, indent=2) + "\n")
            request(f"engram-{arm}-warmup-excluded", "speed", 32, True)
            for suffix, prompt in (("Pfirst", "speed"), ("Prepeat", "speed"),
                                   ("Qnew", "fault_q"), ("Qrepeat", "fault_q")):
                request(f"engram-{arm}-{suffix}", prompt, 128, True)
        # Quality comparison only. Warmth is not controlled or scored here.
        for arm, policy in (("A", "normal"), ("B", "random")):
            for obj in embeddings:
                state = obj.source.set_engram_advice(obj.base, policy)
                if state["status"] != "applied" or state["effective"] != policy:
                    raise RuntimeError(f"quality policy switch failed: {state}")
            for prompt, cap in (("arithmetic", 128), ("coding", 512),
                                ("json", 256), ("reasoning_high", 128)):
                request(f"engram-quality-{arm}-{prompt}", prompt, cap, False)
        maps.append(collector.snapshot_maps("after_arms_and_quality"))
    return {"order": list(ORDER), "engram_preparation": preparation,
            "map_snapshots": maps, "results": results}
