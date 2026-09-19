#!/usr/bin/env python3
from __future__ import annotations
import importlib.util
import json
import math
import os
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "runtime/ds41/v41_attention_parity.py"
spec = importlib.util.spec_from_file_location("v41_attention_parity_test", MOD)
m = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(m)


def window_ref(x: torch.Tensor) -> torch.Tensor:
    rows = x.float().reshape(-1, 16, 32)
    out = torch.empty_like(rows)
    for i in range(rows.shape[0]):
        for b in range(16):
            v = rows[i, b]
            amax = max(float(v.abs().max()), 1e-4)
            exp = min(127, max(-127, math.ceil(math.log2(amax / 448.0))))
            scale = 2.0 ** exp
            q = torch.clamp(v / scale, -448.0, 448.0).to(torch.float8_e4m3fn)
            out[i, b] = q.float() * scale
    return out.to(torch.bfloat16).reshape_as(x)


def nvfp4_ref(x: torch.Tensor) -> torch.Tensor:
    # Independent CPU transcription of vLLM's pinned ref_nvfp4_quant_dequant
    xf = x.float().reshape(-1, 32, 16)
    out = torch.empty_like(xf)
    for r in range(xf.shape[0]):
        for b in range(32):
            v = xf[r, b]
            vmax = float(v.abs().max())
            raw_scale = vmax / 6.0
            scale = torch.tensor(raw_scale, dtype=torch.float32).clamp(-448, 448)
            scale = scale.to(torch.float8_e4m3fn).float()
            output_scale = torch.where(
                scale == 0.0, torch.zeros_like(scale), 1.0 / scale
            )
            y = torch.clamp(v * output_scale, -6, 6)
            a = y.abs()
            q = torch.where(
                a <= 0.25, torch.zeros_like(a),
                torch.where(a < 0.75, torch.full_like(a, 0.5),
                torch.where(a <= 1.25, torch.ones_like(a),
                torch.where(a < 1.75, torch.full_like(a, 1.5),
                torch.where(a <= 2.5, torch.full_like(a, 2.0),
                torch.where(a < 3.5, torch.full_like(a, 3.0),
                torch.where(a <= 5.0, torch.full_like(a, 4.0),
                            torch.full_like(a, 6.0)))))))
            )
            q = torch.copysign(q, y)
            out[r, b] = q * scale
    return out.to(torch.bfloat16).reshape_as(x)


def test_window():
    torch.manual_seed(41)
    x = (torch.randn(5, 512) * torch.tensor([0.0, 1e-6, 0.2, 7.0, 900.0])[:, None]).to(torch.bfloat16)
    got = m.window_fp8_block32_qdq(x)
    ref = window_ref(x)
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))
    return {"rows": 5, "bit_exact_bf16": True}


def test_mxfp4_decode():
    codes = torch.tensor([[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 8], dtype=torch.uint8)
    scales = torch.tensor([[127, 128, 126, 130]], dtype=torch.uint8)
    got = m.dequant_mxfp4(codes, scales).float()
    table = torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6])
    nib = torch.stack((codes & 15, codes >> 4), -1).flatten(-2).long()
    ref = table[nib] * torch.repeat_interleave(torch.exp2(scales.float()-127), 32, -1)
    assert torch.equal(got, ref)
    return {"values": got.numel(), "exact": True}


def test_bf16_paged_roundtrip():
    cache = torch.zeros((3, 4, 512), dtype=torch.bfloat16)
    vals = torch.arange(5*512, dtype=torch.float32).reshape(5,512).to(torch.bfloat16)
    slots = torch.tensor([0,3,4,9,-1])
    m._valid_index_copy(cache, slots, vals)
    out = torch.zeros((2,5,512), dtype=torch.bfloat16)
    seq = torch.tensor([4,2], dtype=torch.int32)
    table = torch.tensor([[0,1],[2,0]], dtype=torch.int32)
    m.gather_bf16_paged(out, cache, seq, None, table, 4, 0)
    # request0 physical slots 0..3; request1 physical slots 8..9
    assert torch.equal(out[0,0], vals[0])
    assert torch.equal(out[0,3], vals[1])
    assert torch.equal(out[1,1], vals[3])
    assert torch.count_nonzero(out[1,2:]) == 0
    return {"writer_gather": "PASS"}


def main():
    result = {
        "schema": "ds41-v41-attention-parity-002-cpu-v1",
        "window": test_window(),
        "compressed_reference_cases": {
            "zeros": bool(torch.count_nonzero(nvfp4_ref(torch.zeros(1,512,dtype=torch.bfloat16))) == 0),
            "finite": bool(torch.isfinite(nvfp4_ref(torch.linspace(-9,9,512).reshape(1,512).to(torch.bfloat16))).all()),
            "block16_scale": "E4M3",
        },
        "indexer_mxfp4_decode": test_mxfp4_decode(),
        "paged_bf16": test_bf16_paged_roundtrip(),
        "default_opt_in": m.enabled(),
    }
    assert result["compressed_reference_cases"]["zeros"]
    assert result["compressed_reference_cases"]["finite"]
    assert result["default_opt_in"] is False
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
