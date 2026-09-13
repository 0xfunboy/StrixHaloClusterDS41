"""Qualified DS41 M=1 mHC FP32 projection+RMS path for gfx1151.

This module reuses the pinned vLLM TileLang prenorm GEMV helper.  It replaces
only the reference sequence

    xf = input_bf16.float()
    projected = xf @ fn.t()
    rms = rsqrt(mean(xf * xf) + eps)
    mixes = projected * rms

for the qualified M=1 shapes.  Coefficient/Sinkhorn, delayed collapse and every
subsequent model operation remain unchanged.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

_STATS = {
    "tilelang_calls": 0,
    "tilelang_tokens": 0,
    "fallback_calls": 0,
    "fallback_reasons": defaultdict(int),
}
_FIRST_LOGGED = False


def reset_stats() -> None:
    global _FIRST_LOGGED
    _STATS["tilelang_calls"] = 0
    _STATS["tilelang_tokens"] = 0
    _STATS["fallback_calls"] = 0
    _STATS["fallback_reasons"].clear()
    _FIRST_LOGGED = False


def stats() -> dict[str, Any]:
    return {
        "tilelang_calls": int(_STATS["tilelang_calls"]),
        "tilelang_tokens": int(_STATS["tilelang_tokens"]),
        "fallback_calls": int(_STATS["fallback_calls"]),
        "fallback_reasons": dict(_STATS["fallback_reasons"]),
    }


def record_fallback(reason: str) -> None:
    _STATS["fallback_calls"] += 1
    _STATS["fallback_reasons"][reason] += 1


def contract(source: torch.Tensor, fn: torch.Tensor) -> tuple[bool, str]:
    if source.ndim != 2 or source.shape[0] != 1:
        return False, "tokens_not_1"
    if source.dtype != torch.bfloat16:
        return False, "input_not_bfloat16"
    if not source.is_contiguous():
        return False, "input_not_contiguous"
    if fn.ndim != 2 or fn.shape[0] != 24:
        return False, "fn_shape"
    if fn.dtype != torch.float32:
        return False, "fn_not_float32"
    if not fn.is_contiguous():
        return False, "fn_not_contiguous"
    k = int(source.shape[1])
    if k not in (5120, 20480):
        return False, "input_width"
    if tuple(fn.shape) != (24, k):
        return False, "fn_input_width"
    return True, "qualified"


def projection_rms_tilelang(
    source: torch.Tensor,
    fn: torch.Tensor,
    rms_eps: float,
) -> torch.Tensor:
    """Return FP32 normalized 24-way mHC projection for a qualified M=1 input."""
    global _FIRST_LOGGED
    ok, reason = contract(source, fn)
    if not ok:
        raise RuntimeError(f"unqualified DS41 mHC projection/RMS TileLang call: {reason}")

    from vllm.model_executor.kernels.mhc.tilelang import _tilelang_hc_prenorm_gemm

    k = int(source.shape[1])
    out = torch.empty((1, 1, 24), dtype=torch.float32, device=source.device)
    sqrsum = torch.empty((1, 1), dtype=torch.float32, device=source.device)
    _tilelang_hc_prenorm_gemm(source, fn, out, sqrsum, k, 1)
    rms_factor = torch.rsqrt(sqrsum[0].view(1, 1) / k + float(rms_eps))
    mixes = out[0] * rms_factor

    _STATS["tilelang_calls"] += 1
    _STATS["tilelang_tokens"] += 1
    if not _FIRST_LOGGED:
        _FIRST_LOGGED = True
        print(
            "DS41_MHC_PROJECTION_RMS_TILELANG_EXECUTE "
            f"shape={tuple(source.shape)} fn={tuple(fn.shape)} dtype={source.dtype}",
            flush=True,
        )
    return mixes
