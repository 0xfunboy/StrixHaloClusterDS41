"""DS41 M=1 BF16 grouped WO_A GEMV candidate for gfx1151.

The candidate preserves BF16 inputs/weights, accumulates in FP32, and writes BF16.
It does not change inverse RoPE, WO_B, TP reduction, attention/cache state, or any
non-M=1 shape.
"""
from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _woa_m1_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    stride_x_g: tl.constexpr,
    stride_x_k: tl.constexpr,
    stride_w_g: tl.constexpr,
    stride_w_r: tl.constexpr,
    stride_w_k: tl.constexpr,
    stride_o_g: tl.constexpr,
    stride_o_r: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    g = pid // R
    r = pid - g * R
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K
    x = tl.load(x_ptr + g * stride_x_g + offs * stride_x_k, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(
        w_ptr + g * stride_w_g + r * stride_w_r + offs * stride_w_k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    acc = tl.sum(x * w, axis=0)
    tl.store(out_ptr + g * stride_o_g + r * stride_o_r, acc)


def woa_m1_gemv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute [1,G,K] x [G,R,K] -> [1,G,R] for the qualified decode shape."""
    if (
        x.ndim != 3
        or x.shape[0] != 1
        or weight.ndim != 3
        or x.shape[1] != weight.shape[0]
        or x.shape[2] != weight.shape[2]
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not x.is_contiguous()
        or not weight.is_contiguous()
    ):
        raise ValueError("DS41 WO_A M=1 contract mismatch")
    _, groups, k = x.shape
    _, r, _ = weight.shape
    if k != 4096 or r != 1024 or groups not in (4, 8):
        raise ValueError(f"unqualified DS41 WO_A shape x={tuple(x.shape)} w={tuple(weight.shape)}")
    out = torch.empty((1, groups, r), device=x.device, dtype=torch.bfloat16)
    block_k = triton.next_power_of_2(k)
    _woa_m1_kernel[(groups * r,)](
        x[0], weight, out[0],
        x.stride(1), x.stride(2),
        weight.stride(0), weight.stride(1), weight.stride(2),
        out.stride(1), out.stride(2),
        R=r, K=k, BLOCK_K=block_k,
        num_warps=8,
    )
    return out
