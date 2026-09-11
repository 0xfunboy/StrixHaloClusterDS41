#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os

import torch

from vllm.models.deepseek_v4_1.attention import _ds41_decomposed_q_rope_kv_insert


def gptj_rope_reference(x: torch.Tensor, positions: torch.Tensor, cache: torch.Tensor, offset: int) -> torch.Tensor:
    out = x.clone()
    rot_dim = cache.shape[1]
    assert rot_dim % 2 == 0
    half = rot_dim // 2
    cs = cache.index_select(0, positions)
    cos = cs[:, :half].float()
    sin = cs[:, half:].float()
    tail = out[..., offset : offset + rot_dim].float()
    even = tail[..., 0::2]
    odd = tail[..., 1::2]
    # GPT-J interleaved rotation.
    y = torch.empty_like(tail)
    view = [positions.numel()] + [1] * (tail.ndim - 2) + [half]
    c = cos.view(*view)
    s = sin.view(*view)
    y[..., 0::2] = even * c - odd * s
    y[..., 1::2] = odd * c + even * s
    out[..., offset : offset + rot_dim] = y.to(out.dtype)
    return out


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm device unavailable")
    torch.manual_seed(19)
    dev = "cuda"
    n, heads, padded, dim, rope_dim, offset = 7, 6, 8, 512, 64, 448
    block_size = 128
    num_blocks = 1
    block_bytes = block_size * (448 + 64 * 2 + 8)  # 584 B/token

    q = torch.randn(n, heads, dim, device=dev, dtype=torch.bfloat16)
    kv = torch.randn(n, dim, device=dev, dtype=torch.bfloat16)
    positions = torch.arange(3, 3 + n, device=dev, dtype=torch.int64)
    slot_mapping = torch.arange(n, device=dev, dtype=torch.int64)

    # Use valid sin/cos pairs in the same [cos32|sin32] layout as the V4 cache.
    max_pos = int(positions.max().item()) + 2
    theta = torch.linspace(0.01, 0.9, max_pos * (rope_dim // 2), device=dev).reshape(max_pos, rope_dim // 2)
    cos_sin = torch.cat([theta.cos(), theta.sin()], dim=-1).to(torch.float32).contiguous()

    # Old fused op: Q is normalized, so only its KV cache is a valid reference
    # for the new apply_q_norm=False path. KV math is unchanged by that flag.
    cache_old = torch.zeros((num_blocks, block_bytes), device=dev, dtype=torch.uint8)
    _ = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        q,
        kv,
        cache_old,
        slot_mapping,
        positions,
        cos_sin,
        padded,
        1e-6,
        block_size,
    )

    # Call the exact DS41 fallback used by the model path.
    cache_new = torch.zeros_like(cache_old)
    q_new = _ds41_decomposed_q_rope_kv_insert(
        q,
        kv,
        cache_new,
        slot_mapping,
        positions,
        cos_sin,
        padded,
        block_size,
    )
    torch.cuda.synchronize()

    q_ref_live = gptj_rope_reference(q, positions, cos_sin, offset)
    q_ref = torch.zeros_like(q_new)
    q_ref[:, :heads].copy_(q_ref_live)

    cache_equal = torch.equal(cache_old, cache_new)
    cache_diff = (cache_old != cache_new).sum().item()
    q_equal = torch.equal(q_new, q_ref)
    q_max_abs = (q_new.float() - q_ref.float()).abs().max().item()
    pad_nonzero = torch.count_nonzero(q_new[:, heads:]).item()

    result = {
        "status": "PASS",
        "q_equal": q_equal,
        "q_max_abs": q_max_abs,
        "pad_nonzero": pad_nonzero,
        "cache_equal": cache_equal,
        "cache_diff_bytes": cache_diff,
        "cache_bytes": cache_old.numel(),
        "tokens": n,
        "heads": heads,
        "padded_heads": padded,
        "head_dim": dim,
        "rope_dim": rope_dim,
        "rope_offset": offset,
        "block_size": block_size,
    }
    assert q_equal and q_max_abs == 0.0 and pad_nonzero == 0, result
    assert cache_equal and cache_diff == 0, result
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
