"""Minimal request-scoped timing/capture for real DSpark proposals."""
from __future__ import annotations
import time
from typing import Any
import torch

_installed = False
_active = False
_capture_tokens = False
_records: list[dict[str, Any]] = []


def install() -> None:
    global _installed
    if _installed:
        return
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
    orig = DSparkSpeculator.propose

    def wrapped(self, *args, **kwargs):
        if not _active:
            return orig(self, *args, **kwargs)
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        t0 = time.perf_counter()
        out = orig(self, *args, **kwargs)
        cpu_enqueue_ms = (time.perf_counter() - t0) * 1000.0
        end_evt.record()
        rec: dict[str, Any] = {
            "start": start_evt,
            "end": end_evt,
            "cpu_enqueue_ms": cpu_enqueue_ms,
            "shape": list(out.shape),
        }
        if _capture_tokens:
            # Functional diagnostic only; never enabled for timed performance runs.
            rec["draft_token_ids"] = out.detach().to("cpu").tolist()
        _records.append(rec)
        return out

    DSparkSpeculator.propose = wrapped
    _installed = True


def begin(*, capture_tokens: bool = False) -> int:
    global _active, _capture_tokens
    start = len(_records)
    _capture_tokens = capture_tokens
    _active = True
    return start


def end(start: int) -> dict[str, Any]:
    global _active, _capture_tokens
    _active = False
    _capture_tokens = False
    rows = _records[start:]
    if rows:
        torch.cuda.synchronize()
    gpu_ms = [float(r["start"].elapsed_time(r["end"])) for r in rows]
    cpu_ms = [float(r["cpu_enqueue_ms"]) for r in rows]
    return {
        "proposal_calls": len(rows),
        "gpu_stream_ms_total": sum(gpu_ms),
        "gpu_stream_ms_mean": sum(gpu_ms) / len(gpu_ms) if gpu_ms else 0.0,
        "cpu_enqueue_ms_total": sum(cpu_ms),
        "cpu_enqueue_ms_mean": sum(cpu_ms) / len(cpu_ms) if cpu_ms else 0.0,
        "draft_token_ids_by_call": [r.get("draft_token_ids") for r in rows if "draft_token_ids" in r],
        "note": "No synchronize occurs inside performance requests; GPU events are resolved after client wall timing. CPU enqueue and GPU span may overlap and are not added together.",
    }
