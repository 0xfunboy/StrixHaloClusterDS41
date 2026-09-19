#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import torch

os.environ.setdefault("DS41_V41_ATTN_PARITY", "1")

from runtime.ds41 import v41_attention_parity as parity
from vllm.models.deepseek_v4.common.ops.fused_indexer_q import fused_indexer_q_rope_quant
from vllm.models.deepseek_v4_1.common.ops.indexer_k_store import indexer_k_norm_rope_store
from vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache import rope_quant_insert

E2_GRID = torch.tensor(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0,
      0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float64,
)
E2_CODE = torch.tensor(
    [255, 254, 253, 252, 251, 250, 249, 248, 0, 1, 2, 3, 4, 5, 6, 7],
    dtype=torch.uint8,
)


def rtne_e2m1_pack(x: torch.Tensor) -> torch.Tensor:
    # Independent table quantizer: nearest E2M1 with tie-to-even code.
    z = x.to(torch.float64).cpu()
    grid = E2_GRID
    codes = E2_CODE
    inds = torch.bucketize(z, grid)
    lo = (inds - 1).clamp(0, 15)
    hi = inds.clamp(0, 15)
    glo, ghi = grid[lo], grid[hi]
    dlo, dhi = z - glo, ghi - z
    pick_hi = (dhi < dlo) | ((dhi == dlo) & ((codes[hi].to(torch.int16) & 1) == 0))
    c = torch.where(pick_hi, codes[hi], codes[lo])
    return ((c[..., 1::2] & 0x0F) << 4) | (c[..., ::2] & 0x0F)


def mxfp4_ref(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Contract frozen by the V4.1 indexer producer: group32, E2M1 + UE8M0.
    y = x.to(torch.float32).cpu().reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    amax = y.abs().amax(dim=-1).clamp_min(6.0 * (2.0 ** -126))
    exp = torch.ceil(torch.log2(amax / 6.0)).clamp(-127, 127)
    scale = torch.exp2(exp)
    packed = rtne_e2m1_pack(y / scale[..., None]).reshape(*x.shape[:-1], x.shape[-1] // 2)
    return packed, (exp + 127).to(torch.uint8)


def identity_cos_sin(rows: int, device: torch.device) -> torch.Tensor:
    cs = torch.zeros((rows, 64), dtype=torch.float32, device=device)
    cs[:, :32] = 1.0
    return cs


def nontrivial_cos_sin(rows: int, device: torch.device) -> torch.Tensor:
    pos = torch.arange(rows, dtype=torch.float32)[:, None]
    freq = torch.arange(1, 33, dtype=torch.float32)[None, :] / 37.0
    ang = pos * freq
    return torch.cat((ang.cos(), ang.sin()), dim=-1).to(device)


def rope_tail_cpu(x: torch.Tensor, positions: torch.Tensor, cs: torch.Tensor, compress_ratio: int) -> torch.Tensor:
    y = x.to(torch.float32).cpu().clone()
    p = ((positions.cpu().long() // compress_ratio) * compress_ratio)
    tab = cs.cpu()[p]
    c, s = tab[:, :32], tab[:, 32:]
    tail = y[:, -64:].reshape(-1, 32, 2)
    even, odd = tail[..., 0], tail[..., 1]
    rot = torch.stack((even * c - odd * s, odd * c + even * s), dim=-1).reshape(-1, 64)
    y[:, -64:] = rot
    return y.to(torch.bfloat16)


def nvfp4_ref(x: torch.Tensor) -> torch.Tensor:
    # Independent transcription of pinned block16 NVFP4/E4M3 QDQ, global scale=1.
    xf = x.float().cpu().reshape(-1, 32, 16)
    vmax = xf.abs().amax(dim=-1)
    sf = (vmax / 6.0).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    inv = torch.where(sf == 0.0, torch.zeros_like(sf), 1.0 / sf)
    z = torch.clamp(xf * inv[..., None], -6.0, 6.0)

    sign = torch.sign(z)
    a = z.abs()
    q = torch.where(
        a <= 0.25, torch.zeros_like(a),
        torch.where(a < 0.75, torch.full_like(a, 0.5),
        torch.where(a <= 1.25, torch.ones_like(a),
        torch.where(a < 1.75, torch.full_like(a, 1.5),
        torch.where(a <= 2.5, torch.full_like(a, 2.0),
        torch.where(a < 3.5, torch.full_like(a, 3.0),
        torch.where(a <= 5.0, torch.full_like(a, 4.0), torch.full_like(a, 6.0)))))))
    )
    return (q * sign * sf[..., None]).reshape(-1, 512).to(torch.bfloat16)


def test_index_q(device):
    torch.manual_seed(2201)
    q = torch.randn((3, 4, 128), dtype=torch.bfloat16, device=device)
    # Magnitude skew forces distinct group scales.
    q[..., :32] *= 0.03
    q[..., 32:64] *= 0.3
    q[..., 64:96] *= 3.0
    q[..., 96:] *= 15.0
    positions = torch.tensor([0, 1, 2], dtype=torch.int64, device=device)
    cs = identity_cos_sin(8, device)
    w = torch.randn((3, 4), dtype=torch.bfloat16, device=device)
    (packed, scale_i32), _ = fused_indexer_q_rope_quant(
        positions, q, cs, w, 128 ** -0.5, 4 ** -0.5, use_fp4=True
    )
    ref_packed, ref_scale = mxfp4_ref(q)
    got_scale = scale_i32.contiguous().view(torch.uint8).reshape(3, 4, 4).cpu()
    assert torch.equal(packed.cpu(), ref_packed), {
        "packed_mismatch": int((packed.cpu() != ref_packed).sum())
    }
    assert torch.equal(got_scale, ref_scale), {
        "scale_mismatch": int((got_scale != ref_scale).sum())
    }
    return {"packed_exact": True, "scale_exact": True}


def test_index_k(device):
    torch.manual_seed(2202)
    kpre_cpu = torch.randn((4, 128), dtype=torch.float32)
    kpre_cpu[:, :32] *= 0.03
    kpre_cpu[:, 32:64] *= 0.3
    kpre_cpu[:, 64:96] *= 3.0
    kpre_cpu[:, 96:] *= 15.0
    kpre = kpre_cpu.to(torch.bfloat16).to(device)
    pos = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=device)
    cs = identity_cos_sin(8, device)
    rms = torch.linspace(0.75, 1.25, 128, dtype=torch.float32).to(torch.bfloat16).to(device)
    cache = torch.zeros((1, 128, 68), dtype=torch.uint8, device=device)
    slots = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=device)
    eps = 1e-6
    indexer_k_norm_rope_store(kpre, pos, cs, rms, eps, cache, slots, 2, True)

    src = kpre.cpu().float()
    rw = rms.cpu().float()
    var = (src * src).mean(dim=-1, keepdim=True)
    norm = (src * torch.rsqrt(var + eps) * rw).to(torch.bfloat16).float()
    # identity RoPE; producer performs another bf16 roundtrip.
    norm = norm.to(torch.bfloat16)
    ref_packed, ref_scale = mxfp4_ref(norm)

    flat = cache.cpu().reshape(1, -1)
    mismatches = []
    for row in (1, 3):
        got_p = flat[0, row * 64 : (row + 1) * 64]
        got_s = flat[0, 128 * 64 + row * 4 : 128 * 64 + (row + 1) * 4]
        if not torch.equal(got_p, ref_packed[row]):
            mismatches.append({"row": row, "packed": int((got_p != ref_packed[row]).sum())})
        if not torch.equal(got_s, ref_scale[row]):
            mismatches.append({"row": row, "scale": int((got_s != ref_scale[row]).sum())})
    assert not mismatches, mismatches
    return {"boundary_rows": [1, 3], "packed_exact": True, "scale_exact": True}


def test_compressed_writer(device):
    torch.manual_seed(2203)
    latent = torch.randn((4, 512), dtype=torch.bfloat16, device=device)
    pos = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=device)
    cs = nontrivial_cos_sin(8, device)
    cache = torch.zeros((1, 128, 512), dtype=torch.bfloat16, device=device)
    slots = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=device)
    rope_quant_insert(latent, pos, cs, cache, slots, 2)

    roped = rope_tail_cpu(latent.cpu(), pos.cpu(), cs.cpu(), 2)
    qdq = nvfp4_ref(roped)
    flat = cache.cpu().reshape(-1, 512)
    assert torch.equal(flat[1].view(torch.int16), qdq[1].view(torch.int16))
    assert torch.equal(flat[3].view(torch.int16), qdq[3].view(torch.int16))
    assert torch.count_nonzero(flat[0]) == 0 and torch.count_nonzero(flat[2]) == 0
    return {"boundary_rows": [1, 3], "writer_qdq_bit_exact": True}


def test_bf16_decode(device):
    torch.manual_seed(2204)
    b, h, d = 2, 4, 512
    q = torch.randn((b, h, d), dtype=torch.bfloat16, device=device)
    main = torch.randn((2, 8, d), dtype=torch.bfloat16, device=device)
    extra = torch.randn((2, 8, d), dtype=torch.bfloat16, device=device)
    mi = torch.tensor([[0, 3, 6, -1], [8, 10, -1, -1]], dtype=torch.int32, device=device)
    ml = torch.tensor([3, 2], dtype=torch.int32, device=device)
    ei = torch.tensor([[1, 5, -1], [9, 15, 12]], dtype=torch.int32, device=device)
    el = torch.tensor([2, 3], dtype=torch.int32, device=device)
    sink = torch.linspace(-0.2, 0.2, h, dtype=torch.float32, device=device)
    out = torch.empty_like(q)
    scale = d ** -0.5
    parity.bf16_sparse_decode(
        q=q, main_cache=main, main_indices=mi, main_lens=ml,
        extra_cache=extra, extra_indices=ei, extra_lens=el,
        attn_sink=sink, scale=scale, nope_head_dim=448, rope_head_dim=64,
        output=out,
    )

    refs = []
    mf = main.reshape(-1, d).cpu().float()
    ef = extra.reshape(-1, d).cpu().float()
    qf = q.cpu().float()
    sf = sink.cpu()
    for bi in range(b):
        rows = torch.cat((
            mf[mi[bi, :ml[bi]].cpu().long()],
            ef[ei[bi, :el[bi]].cpu().long()],
        ), dim=0)
        scores = qf[bi] @ rows.t() * scale
        probs = []
        for hh in range(h):
            p = torch.softmax(torch.cat((scores[hh], sf[hh:hh+1])), dim=0)[:-1]
            probs.append((p[:, None] * rows).sum(dim=0))
        refs.append(torch.stack(probs))
    ref = torch.stack(refs).to(torch.bfloat16)
    diff = (out.cpu().float() - ref.float()).abs()
    assert torch.allclose(out.cpu().float(), ref.float(), atol=0.035, rtol=0.035), {
        "max_abs": float(diff.max()), "mean_abs": float(diff.mean())
    }
    return {"formula_match": True, "max_abs": float(diff.max()), "mean_abs": float(diff.mean())}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm device required")
    dev = torch.device("cuda")
    result = {
        "schema": "ds41-attention-parity-002-localize-producers-v1",
        "device": torch.cuda.get_device_properties(dev).name,
        "index_q": test_index_q(dev),
        "index_k": test_index_k(dev),
        "compressed_writer": test_compressed_writer(dev),
        "bf16_decode": test_bf16_decode(dev),
        "status": "PASS",
    }
    torch.cuda.synchronize()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
