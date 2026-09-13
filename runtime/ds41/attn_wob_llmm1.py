"""DS41 M=1 WO_B LLMM1 dispatch for the promoted V4.1 ROCm path.

Only the local RowParallelLinear matmul changes.  The existing TP all-reduce is
called immediately after the candidate kernel.  All non-qualified shapes and
configurations fall back to the original layer forward path.
M=1 is a shape gate and can also include one-token prefill or dummy calls.
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

import torch

_STATS = {"llmm1_calls": 0, "llmm1_tokens": 0, "fallback_calls": 0, "fallback_reasons": defaultdict(int)}


def reset_stats() -> None:
    _STATS["llmm1_calls"] = 0
    _STATS["llmm1_tokens"] = 0
    _STATS["fallback_calls"] = 0
    _STATS["fallback_reasons"].clear()


def stats() -> dict[str, Any]:
    return {
        "llmm1_calls": int(_STATS["llmm1_calls"]),
        "llmm1_tokens": int(_STATS["llmm1_tokens"]),
        "fallback_calls": int(_STATS["fallback_calls"]),
        "fallback_reasons": dict(_STATS["fallback_reasons"]),
    }


def _fallback(reason: str) -> None:
    _STATS["fallback_calls"] += 1
    _STATS["fallback_reasons"][reason] += 1


def try_wob_llmm1(layer, x: torch.Tensor) -> torch.Tensor | None:
    """Return qualified LLMM1+TP-allreduce output, or None for exact fallback."""
    if os.environ.get("DS41_ATTN_WOB_LLMM1", "0") != "1":
        return None
    if torch.version.hip is None:
        _fallback("backend_not_rocm")
        return None
    if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[0] != 1:
        _fallback("tokens_not_1")
        return None
    if x.dtype != torch.bfloat16 or not x.is_contiguous():
        _fallback("input_contract")
        return None
    weight = getattr(layer, "weight", None)
    if not isinstance(weight, torch.Tensor):
        _fallback("missing_weight")
        return None
    if not x.is_cuda or not weight.is_cuda or x.device != weight.device:
        _fallback("device_contract")
        return None
    if weight.dtype != torch.bfloat16 or not weight.is_contiguous():
        _fallback("weight_contract")
        return None
    if tuple(x.shape) != (1, 4096) or tuple(weight.shape) != (5120, 4096):
        _fallback("shape_contract")
        return None
    if getattr(layer, "bias", None) is not None:
        _fallback("bias_present")
        return None
    if getattr(layer, "return_bias", True) is not False:
        _fallback("return_bias")
        return None
    if not bool(getattr(layer, "input_is_parallel", False)):
        _fallback("input_not_parallel")
        return None
    if not bool(getattr(layer, "reduce_results", False)) or int(getattr(layer, "tp_size", 0)) != 2:
        _fallback("tp_contract")
        return None
    from vllm import _custom_ops as ops
    from vllm import envs
    from vllm.distributed import tensor_model_parallel_all_reduce
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    if type(getattr(layer, "quant_method", None)) is not UnquantizedLinearMethod:
        _fallback("quant_method")
        return None
    if envs.VLLM_BATCH_INVARIANT:
        _fallback("batch_invariant")
        return None
    if not hasattr(torch.ops._rocm_C, "LLMM1"):
        _fallback("llmm1_unavailable")
        return None

    local = ops.LLMM1(weight, x, 4)
    out = tensor_model_parallel_all_reduce(local)
    _STATS["llmm1_calls"] += 1
    _STATS["llmm1_tokens"] += 1
    return out
