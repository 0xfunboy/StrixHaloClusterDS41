"""DS41 gfx1151 fused mHC coefficient/Sinkhorn kernel.

This deliberately starts *after* the existing FP32 projection/RMS step and ends
before the delayed collapse.  It only replaces the many eager elementwise and
4x4 Sinkhorn launches for M=1 decode.  The mathematical contract matches the
pinned V4.1 torch reference: sigmoid pre/post coefficients, row softmax + eps,
initial column normalization, then (row, column) normalization for the remaining
Sinkhorn iterations.
"""
from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


_STATS = {"fused_calls": 0, "fused_tokens": 0}
_FIRST_LOGGED = False


def stats() -> dict[str, int]:
    return {k: int(v) for k, v in _STATS.items()}


def reset_stats() -> None:
    global _FIRST_LOGGED
    _STATS["fused_calls"] = 0
    _STATS["fused_tokens"] = 0
    _FIRST_LOGGED = False


@triton.jit
def _mhc_coeff_sinkhorn_kernel(
    mixes_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    pre_eps: tl.constexpr,
    sink_eps: tl.constexpr,
    post_mult: tl.constexpr,
    sinkhorn_repeat: tl.constexpr,
):
    pid = tl.program_id(0)

    # pre/post coefficients: hc_mult is fixed to 4 for the qualified V4.1 path.
    offs4 = tl.arange(0, 4)
    s0 = tl.load(scale_ptr + 0)
    s1 = tl.load(scale_ptr + 1)
    s2 = tl.load(scale_ptr + 2)

    pre_logits = tl.load(mixes_ptr + pid * 24 + offs4) * s0 + tl.load(
        base_ptr + offs4
    )
    pre = 1.0 / (1.0 + tl.exp(-pre_logits)) + pre_eps
    tl.store(pre_ptr + pid * 4 + offs4, pre)

    post_logits = tl.load(mixes_ptr + pid * 24 + 4 + offs4) * s1 + tl.load(
        base_ptr + 4 + offs4
    )
    post = (1.0 / (1.0 + tl.exp(-post_logits))) * post_mult
    tl.store(post_ptr + pid * 4 + offs4, post)

    offs16 = tl.arange(0, 16)
    cm = tl.load(mixes_ptr + pid * 24 + 8 + offs16) * s2 + tl.load(
        base_ptr + 8 + offs16
    )

    # Row-wise softmax, then + epsilon exactly as the torch reference.
    for row in tl.static_range(0, 4):
        mask = (offs16 // 4) == row
        neg_inf = float("-inf")
        row_max = tl.max(tl.where(mask, cm, neg_inf), axis=0)
        ex = tl.where(mask, tl.exp(cm - row_max), 0.0)
        denom = tl.sum(ex, axis=0)
        cm = tl.where(mask, ex / denom + sink_eps, cm)

    # Reference performs an initial column normalization immediately after
    # softmax, then sinkhorn_repeat-1 row+column iterations.
    for col in tl.static_range(0, 4):
        mask = (offs16 % 4) == col
        denom = tl.sum(tl.where(mask, cm, 0.0), axis=0) + sink_eps
        cm = tl.where(mask, cm / denom, cm)

    for _ in tl.static_range(0, sinkhorn_repeat - 1):
        for row in tl.static_range(0, 4):
            mask = (offs16 // 4) == row
            denom = tl.sum(tl.where(mask, cm, 0.0), axis=0) + sink_eps
            cm = tl.where(mask, cm / denom, cm)
        for col in tl.static_range(0, 4):
            mask = (offs16 % 4) == col
            denom = tl.sum(tl.where(mask, cm, 0.0), axis=0) + sink_eps
            cm = tl.where(mask, cm / denom, cm)

    tl.store(comb_ptr + pid * 16 + offs16, cm)


def fused_coeff_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return pre [T,4], post [T,4], comb [T,4,4] in FP32."""
    if mixes.ndim != 2 or mixes.shape[1] != 24:
        raise ValueError(f"qualified DS41 mHC mixes shape is [T,24], got {tuple(mixes.shape)}")
    if mixes.dtype is not torch.float32 or not mixes.is_contiguous():
        raise ValueError("DS41 fused mHC requires contiguous FP32 mixes")
    if hc_scale.dtype is not torch.float32 or tuple(hc_scale.shape) != (3,):
        raise ValueError("DS41 fused mHC requires FP32 hc_scale[3]")
    if hc_base.dtype is not torch.float32 or tuple(hc_base.shape) != (24,):
        raise ValueError("DS41 fused mHC requires FP32 hc_base[24]")
    if sinkhorn_repeat != 20:
        raise ValueError(f"qualified DS41 Sinkhorn repeat is 20, got {sinkhorn_repeat}")

    tokens = mixes.shape[0]
    pre = torch.empty((tokens, 4), device=mixes.device, dtype=torch.float32)
    post = torch.empty_like(pre)
    comb = torch.empty((tokens, 16), device=mixes.device, dtype=torch.float32)
    if tokens:
        _mhc_coeff_sinkhorn_kernel[(tokens,)](
            mixes,
            hc_scale,
            hc_base,
            pre,
            post,
            comb,
            pre_eps=float(hc_pre_eps),
            sink_eps=float(hc_sinkhorn_eps),
            post_mult=float(hc_post_mult_value),
            sinkhorn_repeat=20,
            num_warps=1,
        )
        global _FIRST_LOGGED
        _STATS["fused_calls"] += 1
        _STATS["fused_tokens"] += int(tokens)
        if not _FIRST_LOGGED:
            _FIRST_LOGGED = True
            print(
                "DS41_MHC_COEFF_SINKHORN_EXECUTE "
                f"shape={tuple(mixes.shape)} sinkhorn_repeat={sinkhorn_repeat}",
                flush=True,
            )
    return pre, post, comb.view(tokens, 4, 4)
