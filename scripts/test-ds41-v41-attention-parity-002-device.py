#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("DS41_V41_ATTN_PARITY", "1")

import torch

from runtime.ds41 import v41_attention_parity as parity
from vllm.models.deepseek_v4.common.ops.fused_indexer_q import (
    fused_indexer_q_rope_quant,
)
from vllm.models.deepseek_v4_1.common.ops.indexer_k_store import (
    indexer_k_norm_rope_store,
)
from vllm.models.deepseek_v4_1.attention import (
    _ds41_decomposed_q_rope_kv_insert,
)
from vllm import _custom_ops as vllm_ops


def window_ref_cpu(x: torch.Tensor) -> torch.Tensor:
    rows = x.float().reshape(-1, 16, 32)
    out = torch.empty_like(rows)
    for i in range(rows.shape[0]):
        for b in range(16):
            v = rows[i, b]
            amax = max(float(v.abs().max()), 1e-4)
            exp = min(127, max(-127, math.ceil(math.log2(amax / 448.0))))
            scale = 2.0**exp
            q = torch.clamp(v / scale, -448.0, 448.0).to(torch.float8_e4m3fn)
            out[i, b] = q.float() * scale
    return out.to(torch.bfloat16).reshape_as(x)


def cast_fp4_cpu(y: torch.Tensor) -> torch.Tensor:
    sign = torch.sign(y)
    a = y.abs()
    q = torch.where(
        a <= 0.25,
        torch.zeros_like(a),
        torch.where(
            a < 0.75,
            torch.full_like(a, 0.5),
            torch.where(
                a <= 1.25,
                torch.ones_like(a),
                torch.where(
                    a < 1.75,
                    torch.full_like(a, 1.5),
                    torch.where(
                        a <= 2.5,
                        torch.full_like(a, 2.0),
                        torch.where(
                            a < 3.5,
                            torch.full_like(a, 3.0),
                            torch.where(
                                a <= 5.0,
                                torch.full_like(a, 4.0),
                                torch.full_like(a, 6.0),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    return q * sign


def nvfp4_ref_cpu(x: torch.Tensor) -> torch.Tensor:
    xf = x.float().reshape(-1, 32, 16)
    out = torch.empty_like(xf)
    for r in range(xf.shape[0]):
        for b in range(32):
            v = xf[r, b]
            scale = torch.tensor(float(v.abs().max()) / 6.0, dtype=torch.float32)
            scale = scale.clamp(-448, 448).to(torch.float8_e4m3fn).float()
            output_scale = torch.where(
                scale == 0.0, torch.zeros_like(scale), 1.0 / scale
            )
            y = torch.clamp(v * output_scale, -6, 6)
            out[r, b] = cast_fp4_cpu(y) * scale
    return out.to(torch.bfloat16).reshape_as(x)


def decode_mxfp4_cpu(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    values = values.cpu()
    scales = scales.cpu()
    low = values & 0x0F
    high = (values >> 4) & 0x0F
    codes = torch.stack((low, high), dim=-1).flatten(-2).long()
    table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
    )
    decoded = table[codes]
    if scales.dtype == torch.int32:
        sb = scales.contiguous().view(torch.uint8).reshape(*scales.shape, 4)
    else:
        sb = scales
    sf = torch.repeat_interleave(torch.exp2(sb.float() - 127.0), 32, dim=-1)
    return (decoded * sf).to(torch.bfloat16)


def identity_cos_sin(rows: int, device: torch.device) -> torch.Tensor:
    out = torch.zeros((rows, 64), dtype=torch.float32, device=device)
    out[:, :32] = 1.0
    return out


def test_window(device):
    torch.manual_seed(2002)
    scales = torch.tensor([0.0, 1e-6, 0.125, 4.0, 900.0], dtype=torch.float32)
    cpu = (torch.randn(5, 512) * scales[:, None]).to(torch.bfloat16)
    ref = window_ref_cpu(cpu)
    got = parity.window_fp8_block32_qdq(cpu.to(device)).cpu()
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))
    return {"rows": 5, "bf16_bit_exact": True}


def test_compressed(device):
    torch.manual_seed(2003)
    rows = torch.stack(
        [
            torch.zeros(512),
            torch.linspace(-9.0, 9.0, 512),
            torch.randn(512) * 0.07,
            torch.cat(
                [
                    torch.full((16,), 0.25),
                    torch.full((16,), 0.75),
                    torch.full((16,), 1.25),
                    torch.full((16,), 5.0),
                    torch.randn(448),
                ]
            ),
        ]
    ).to(torch.bfloat16)
    ref = nvfp4_ref_cpu(rows)
    got = parity.compressed_nvfp4_block16_qdq(rows.to(device)).cpu()
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), (
        (got.float() - ref.float()).abs().max().item()
    )

    # Incomplete group: for ratio2 only position 1 is a publishing boundary.
    cache = torch.arange(3 * 512, dtype=torch.float32, device=device).reshape(1, 3, 512)
    cache = (cache / 100).to(torch.bfloat16)
    before = cache.clone()
    slots = torch.tensor([0, 1, 2], dtype=torch.int64, device=device)
    pos = torch.tensor([0, 1, 2], dtype=torch.int64, device=device)
    parity.qdq_compressed_cache_inplace(cache, slots, pos, 2)
    assert torch.equal(cache[:, 0], before[:, 0])
    assert not torch.equal(cache[:, 1], before[:, 1])
    assert torch.equal(cache[:, 2], before[:, 2])
    return {"rows": 4, "bf16_bit_exact": True, "incomplete_group_preserved": True}


def test_full_cache_bf16_decomposed(device):
    torch.manual_seed(2008)
    q = torch.randn((3, 2, 512), dtype=torch.bfloat16, device=device)
    kv = torch.randn((3, 512), dtype=torch.bfloat16, device=device)
    q_ref = torch.zeros((3, 4, 512), dtype=torch.bfloat16, device=device)
    q_ref[:, :2].copy_(q)
    kv_ref = kv.clone()
    positions = torch.tensor([0, 1, 2], dtype=torch.int64, device=device)
    cos_sin = identity_cos_sin(8, device)
    vllm_ops.rotary_embedding(
        positions,
        q_ref,
        kv_ref,
        512,
        cos_sin,
        False,
        rope_dim_offset=448,
        inverse=False,
    )
    expected_cache = parity.window_fp8_block32_qdq(kv_ref)

    cache = torch.zeros((1, 128, 512), dtype=torch.bfloat16, device=device)
    slots = torch.tensor([0, 3, 7], dtype=torch.int64, device=device)
    got_q = _ds41_decomposed_q_rope_kv_insert(
        q,
        kv,
        cache,
        slots,
        positions,
        cos_sin,
        4,
        128,
    )
    assert torch.equal(got_q.view(torch.int16), q_ref.view(torch.int16))
    flat = cache.reshape(-1, 512)
    assert torch.equal(
        flat.index_select(0, slots).view(torch.int16),
        expected_cache.view(torch.int16),
    )
    return {
        "q_rope_bit_exact": True,
        "window_store_bit_exact": True,
        "cache_layout": list(cache.shape),
    }


def test_indexer_q(device):
    torch.manual_seed(2004)
    q = torch.randn((3, 4, 128), dtype=torch.bfloat16, device=device)
    positions = torch.tensor([0, 1, 2], dtype=torch.int64, device=device)
    cos_sin = identity_cos_sin(8, device)
    weights = torch.randn((3, 4), dtype=torch.bfloat16, device=device)
    (packed, scale), weights_out = fused_indexer_q_rope_quant(
        positions,
        q,
        cos_sin,
        weights,
        128**-0.5,
        4**-0.5,
        use_fp4=True,
    )
    got = parity.dequant_mxfp4(packed, scale).cpu()
    ref = decode_mxfp4_cpu(packed, scale)
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))
    expected_w = weights.float().cpu() * (128**-0.5) * (4**-0.5)
    assert torch.allclose(weights_out.cpu(), expected_w, atol=2e-5, rtol=2e-5)
    return {
        "packed_shape": list(packed.shape),
        "scale_shape": list(scale.shape),
        "decode_exact": True,
    }


def test_indexer_k(device):
    torch.manual_seed(2005)
    base = torch.randn((4, 128), dtype=torch.float32)
    # Keep different 32-wide magnitudes after global RMS normalization so
    # non-uniform UE8M0 block scales are actually exercised.
    base[:, :32] *= 0.03
    base[:, 32:64] *= 0.3
    base[:, 64:96] *= 3.0
    base[:, 96:] *= 15.0
    k_pre = base.to(torch.bfloat16).to(device)
    pos = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=device)
    cos_sin = identity_cos_sin(8, device)
    weight = torch.ones((128,), dtype=torch.bfloat16, device=device)
    cache = torch.zeros((1, 128, 68), dtype=torch.uint8, device=device)
    slots = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=device)
    indexer_k_norm_rope_store(
        k_pre,
        pos,
        cos_sin,
        weight,
        1e-6,
        cache,
        slots,
        2,
        True,
    )
    gathered = parity.gather_mxfp4_indexer_k(
        cache,
        torch.tensor([[1, 3]], dtype=torch.int64, device=device),
    )
    assert torch.isfinite(gathered).all()
    assert torch.count_nonzero(gathered) > 0
    flat = cache.reshape(1, -1)
    scale0 = 128 * 64 + 1 * 4
    scale1 = 128 * 64 + 3 * 4
    scale_bytes = torch.cat(
        [flat[0, scale0 : scale0 + 4], flat[0, scale1 : scale1 + 4]]
    )
    assert torch.unique(scale_bytes).numel() > 1
    # Non-boundary slots 0 and 2 remain all-zero.
    for slot in (0, 2):
        off = slot * 64
        assert torch.count_nonzero(flat[0, off : off + 64]) == 0
    return {
        "roundtrip_finite": True,
        "non_uniform_scales": True,
        "incomplete_group_not_published": True,
    }


def test_indexer_consumer_math(device):
    torch.manual_seed(2006)
    q = torch.randn((2, 3, 128), dtype=torch.bfloat16, device=device)
    k = torch.randn((6, 128), dtype=torch.bfloat16, device=device)
    # Exact tie on rows 1/2. Selection convention itself is unchanged in _C.
    k[2].copy_(k[1])
    w = torch.randn((2, 3), dtype=torch.float32, device=device)
    starts = torch.tensor([0, 1], dtype=torch.int64, device=device)
    ends = torch.tensor([6, 5], dtype=torch.int64, device=device)
    got = parity._prefill_mqa_logits(q, k, w, starts, ends)

    score = torch.einsum("mhd,nd->mhn", q.float(), k.float()).relu()
    ref = (score * w.unsqueeze(-1)).sum(1)
    n = torch.arange(k.shape[0], device=device)[None, :]
    ref = ref.masked_fill(~((n >= starts[:, None]) & (n < ends[:, None])), -torch.inf)
    assert torch.allclose(got, ref, atol=3e-2, rtol=3e-2, equal_nan=True)
    assert got[0, 1].item() == got[0, 2].item()
    return {"formula_match": True, "tie_equal_logits": True}


def test_bf16_sparse_consumer(device):
    torch.manual_seed(2007)
    heads = 4
    d = 512
    q = torch.randn((1, heads, d), dtype=torch.bfloat16, device=device)
    main = torch.randn((1, 128, d), dtype=torch.bfloat16, device=device)
    extra = torch.randn((1, 128, d), dtype=torch.bfloat16, device=device)
    main_idx = torch.tensor([[1, 3, 5, 0]], dtype=torch.int32, device=device)
    main_lens = torch.tensor([3], dtype=torch.int32, device=device)
    extra_idx = torch.tensor([[2, 4, 6, 8]], dtype=torch.int32, device=device)
    extra_lens = torch.tensor([4], dtype=torch.int32, device=device)
    out = torch.empty_like(q)
    scale = d**-0.5
    parity.bf16_sparse_decode(
        q=q,
        main_cache=main,
        main_indices=main_idx,
        main_lens=main_lens,
        extra_cache=extra,
        extra_indices=extra_idx,
        extra_lens=extra_lens,
        attn_sink=None,
        scale=scale,
        nope_head_dim=448,
        rope_head_dim=64,
        output=out,
    )
    assert torch.isfinite(out).all()
    return {"finite": True, "shape": list(out.shape)}


def test_fresh_request_isolation(device):
    a = torch.zeros((1, 4, 512), dtype=torch.bfloat16, device=device)
    b = torch.zeros_like(a)
    assert a.data_ptr() != b.data_ptr()
    slots = torch.tensor([0], dtype=torch.int64, device=device)
    va = torch.ones((1, 512), dtype=torch.bfloat16, device=device)
    parity._valid_index_copy(a, slots, va)
    assert torch.count_nonzero(b) == 0
    return {"separate_storage": True}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("gfx1151 device gate requires ROCm torch accelerator")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    before = torch.cuda.memory_allocated(device)
    result = {
        "schema": "ds41-v41-attention-parity-002-device-v1",
        "device": props.name,
        "window": test_window(device),
        "compressed": test_compressed(device),
        "full_cache_bf16_decomposed": test_full_cache_bf16_decomposed(device),
        "indexer_q": test_indexer_q(device),
        "indexer_k": test_indexer_k(device),
        "indexer_consumer": test_indexer_consumer_math(device),
        "bf16_sparse_consumer": test_bf16_sparse_consumer(device),
        "fresh_request": test_fresh_request_isolation(device),
    }
    torch.cuda.synchronize()
    result["memory_allocated_delta_bytes"] = torch.cuda.memory_allocated(device) - before
    result["status"] = "PASS"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
