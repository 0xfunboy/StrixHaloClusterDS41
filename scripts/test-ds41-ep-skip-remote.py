#!/usr/bin/env python3
"""Causal DS41 EP remote-route elimination check on real GGUF expert bytes.

Compares the qualified correctness-first fallback (remote global routes mapped to
local expert0 then zero-weighted) against the candidate GPU-side aligner path
that ignores remote routes before the IQ2/Q2_K kernels read their weights.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import torch

from _ds41_artifact import MODEL_FILE

ROOT = Path('/home/funboy/StrixHaloClusterDS41')
from vllm_gguf_plugin.quantization.fused_moe import GGUFMoEMethod

IQ2_XXS = int(gguf.GGMLQuantizationType.IQ2_XXS)
Q2_K = int(gguf.GGMLQuantizationType.Q2_K)
OUT = ROOT / "reports/DS41-Q2-001/perf/ep-skip-remote-real.json"


def raw(reader, name: str, experts: slice) -> torch.Tensor:
    t = next(t for t in reader.tensors if t.name == name)
    return torch.from_numpy(np.ascontiguousarray(t.data[experts])).to("cuda")


def layer(w13, w2, expert_map):
    return SimpleNamespace(
        apply_router_weight_on_input=False,
        w13_weight=w13,
        w2_weight=w2,
        w13_weight_type=SimpleNamespace(weight_type=IQ2_XXS),
        w2_weight_type=SimpleNamespace(weight_type=Q2_K),
        activation=SimpleNamespace(value="silu"),
        expert_map=expert_map,
    )


def method():
    return SimpleNamespace(moe=SimpleNamespace(swiglu_limit=10.0))


def apply_mode(lyr, x, weights, ids, optimized: bool):
    if optimized:
        os.environ["DS41_EP_SKIP_REMOTE"] = "1"
    else:
        os.environ.pop("DS41_EP_SKIP_REMOTE", None)
    return GGUFMoEMethod.apply(method(), lyr, x, weights, ids, None, None)


def metric(a, b):
    d = (a.float() - b.float()).abs()
    return {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "a_norm": float(a.float().norm()),
        "b_norm": float(b.float().norm()),
    }


def timed(fn, warmup=5, repeats=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts, ends = [], []
    for _ in range(repeats):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); starts.append(a); ends.append(b)
    torch.cuda.synchronize()
    vals = [a.elapsed_time(b) for a, b in zip(starts, ends, strict=True)]
    vals.sort()
    return {
        "repeats": repeats,
        "mean_ms": float(sum(vals) / len(vals)),
        "median_ms": float(vals[len(vals)//2]),
        "min_ms": float(vals[0]),
        "max_ms": float(vals[-1]),
    }


def run_case(m: int, full, r0, r1):
    g = torch.Generator(device="cuda").manual_seed(771 + m)
    x = torch.randn((m, 5120), generator=g, device="cuda", dtype=torch.bfloat16)
    # Representative EP decode layout: three distinct experts owned by each
    # rank, no rank-local expert0 among the valid routes.  The baseline maps
    # the three remote routes to dummy local expert0, creating one extra BLOCK_M
    # expert block; the candidate omits it before the GGUF weight reads.
    ids = torch.tensor([1, 2, 3, 5, 6, 7], device="cuda", dtype=torch.int32).expand(m, 6).contiguous()
    weights = torch.tensor([0.20, 0.18, 0.17, 0.16, 0.15, 0.14], device="cuda", dtype=torch.float32).expand(m, 6).contiguous()

    full_out = apply_mode(full, x, weights, ids, False)
    r0_base = apply_mode(r0, x, weights, ids, False)
    r1_base = apply_mode(r1, x, weights, ids, False)
    baseline = r0_base + r1_base
    r0_opt = apply_mode(r0, x, weights, ids, True)
    r1_opt = apply_mode(r1, x, weights, ids, True)
    candidate = r0_opt + r1_opt

    # EP2 changes the BF16 accumulation order versus a single-rank full-expert
    # sum when top-k has several experts.  Preserve the already-qualified EP2
    # baseline bit-for-bit; compare both EP2 paths to full with BF16 tolerance.
    torch.testing.assert_close(baseline, full_out, rtol=2e-2, atol=4e-2)
    torch.testing.assert_close(candidate, full_out, rtol=2e-2, atol=4e-2)
    torch.testing.assert_close(candidate, baseline, rtol=0, atol=0)

    row = {
        "m": m,
        "baseline_vs_full": metric(baseline, full_out),
        "candidate_vs_full": metric(candidate, full_out),
        "candidate_vs_baseline": metric(candidate, baseline),
        "status": "PASS",
    }
    if m == 1:
        row["rank0_baseline_timing"] = timed(lambda: apply_mode(r0, x, weights, ids, False))
        row["rank0_candidate_timing"] = timed(lambda: apply_mode(r0, x, weights, ids, True))
        row["rank1_baseline_timing"] = timed(lambda: apply_mode(r1, x, weights, ids, False))
        row["rank1_candidate_timing"] = timed(lambda: apply_mode(r1, x, weights, ids, True))
    return row


def zero_local_case(r0):
    x = torch.randn((1, 5120), generator=torch.Generator(device="cuda").manual_seed(990), device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([[4, 5, 6, 7, 5, 6]], device="cuda", dtype=torch.int32)
    weights = torch.tensor([[0.20, 0.18, 0.17, 0.16, 0.15, 0.14]], device="cuda", dtype=torch.float32)
    base = apply_mode(r0, x, weights, ids, False)
    opt = apply_mode(r0, x, weights, ids, True)
    torch.testing.assert_close(opt, base, rtol=0, atol=0)
    assert int(torch.count_nonzero(opt).item()) == 0
    return {"status": "PASS", "candidate_nonzero": 0, "baseline_nonzero": int(torch.count_nonzero(base).item())}


def main():
    assert torch.cuda.is_available()
    reader = gguf.GGUFReader(str(MODEL_FILE))
    gate = raw(reader, "blk.0.ffn_gate_exps", slice(0, 8))
    up = raw(reader, "blk.0.ffn_up_exps", slice(0, 8))
    down = raw(reader, "blk.0.ffn_down_exps", slice(0, 8))
    w13 = torch.cat((gate, up), dim=1)
    full = layer(w13, down, None)
    map0 = torch.tensor([0, 1, 2, 3, -1, -1, -1, -1], device="cuda", dtype=torch.int32)
    map1 = torch.tensor([-1, -1, -1, -1, 0, 1, 2, 3], device="cuda", dtype=torch.int32)
    r0 = layer(w13[:4], down[:4], map0)
    r1 = layer(w13[4:], down[4:], map1)

    out = {
        "status": "PASS",
        "model": str(MODEL_FILE),
        "cases": [run_case(1, full, r0, r1), run_case(16, full, r0, r1)],
        "zero_local": zero_local_case(r0),
        "contract": {
            "global_router_weights_unchanged": True,
            "local_renormalization": False,
            "remote_route_skip": "moe_align_block_size(expert_map, ignore_invalid_experts=True)",
            "cpu_route_sync": False,
        },
    }
    os.environ.pop("DS41_EP_SKIP_REMOTE", None)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
