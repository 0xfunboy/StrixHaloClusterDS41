#!/usr/bin/env python3
"""Gate a minimal TileLang projection+RMS candidate on captured real M=1 mHC inputs.

The candidate replaces only:
    xf = input_bf16.float()
    projected = xf @ fn.t()
    rms_factor = rsqrt(mean(xf * xf) + eps)
    mixes = projected * rms_factor

It reuses the pinned vLLM TileLang prenorm GEMV helper.  Coefficient/Sinkhorn,
delayed collapse, mHC post, layer RMSNorm, attention and MoE are out of scope.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from vllm.model_executor.kernels.mhc.tilelang import _tilelang_hc_prenorm_gemm

MIX_REL_L2_MAX = 5e-4
RMS_REL_MAX = 1e-5
PROJ_REL_L2_MAX = 5e-4
REPEATS = 200
WARMUP = 8


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    af, bf = a.float(), b.float()
    den = float(bf.norm())
    return float((af - bf).norm()) / max(den, 1e-30)


def eager(x_bf16: torch.Tensor, fn: torch.Tensor, eps: float):
    xf = x_bf16.float()
    proj = xf @ fn.t()
    mean_sq = xf.square().mean(-1, keepdim=True)
    rms = torch.rsqrt(mean_sq + eps)
    return proj, mean_sq, rms, proj * rms


def tilelang_candidate(x_bf16: torch.Tensor, fn: torch.Tensor, eps: float):
    assert x_bf16.ndim == 2 and x_bf16.shape[0] == 1
    assert x_bf16.dtype == torch.bfloat16 and x_bf16.is_contiguous()
    assert fn.dtype == torch.float32 and fn.is_contiguous()
    k = x_bf16.shape[1]
    nout = fn.shape[0]
    assert nout == 24 and k % 1024 == 0
    out = torch.empty((1, 1, nout), dtype=torch.float32, device=x_bf16.device)
    sq = torch.empty((1, 1), dtype=torch.float32, device=x_bf16.device)
    _tilelang_hc_prenorm_gemm(x_bf16, fn, out, sq, k, 1)
    proj = out[0]
    mean_sq = sq[0].view(1, 1) / k
    rms = torch.rsqrt(mean_sq + eps)
    return proj, mean_sq, rms, proj * rms


def timing(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    events = []
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); events.append((a, b))
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1000 / REPEATS
    vals = [a.elapsed_time(b) for a, b in events]
    return {
        "repeats": REPEATS,
        "gpu_ms_mean": float(sum(vals) / len(vals)),
        "gpu_ms_min": float(min(vals)),
        "gpu_ms_max": float(max(vals)),
        "wall_ms_mean": float(wall),
    }


def metric(a, b):
    d = a.float() - b.float()
    return {
        "max_abs": float(d.abs().max()),
        "mean_abs": float(d.abs().mean()),
        "rel_l2": rel_l2(a, b),
        "a_norm": float(a.float().norm()),
        "b_norm": float(b.float().norm()),
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
    }


def case(row):
    x = row["input_bf16"].contiguous().to("cuda")
    fn = row["fn"].contiguous().to("cuda")
    eps = float(row["rms_eps"])
    ep, em, er, ex = eager(x, fn, eps)
    cp, cm, cr, cx = tilelang_candidate(x, fn, eps)
    torch.cuda.synchronize()

    # Higher-precision CPU reference distinguishes reduction-order noise from
    # a wrong implementation.  This is diagnostic, not a different runtime.
    xd = row["input_bf16"].double()
    fd = row["fn"].double()
    ref64_proj = xd @ fd.t()
    ref64_mean = xd.square().mean(-1, keepdim=True)
    ref64_rms = torch.rsqrt(ref64_mean + eps)
    ref64_mix = ref64_proj * ref64_rms

    proj_m = metric(cp, ep)
    mean_m = metric(cm, em)
    rms_m = metric(cr, er)
    mix_m = metric(cx, ex)
    e64 = {
        "projection_rel_l2": rel_l2(ep.cpu().double(), ref64_proj),
        "mixes_rel_l2": rel_l2(ex.cpu().double(), ref64_mix),
    }
    c64 = {
        "projection_rel_l2": rel_l2(cp.cpu().double(), ref64_proj),
        "mixes_rel_l2": rel_l2(cx.cpu().double(), ref64_mix),
    }
    ok = (
        proj_m["finite"] and rms_m["finite"] and mix_m["finite"]
        and proj_m["rel_l2"] <= PROJ_REL_L2_MAX
        and rms_m["rel_l2"] <= RMS_REL_MAX
        and mix_m["rel_l2"] <= MIX_REL_L2_MAX
    )
    return {
        "label": row["label"],
        "input_shape": list(x.shape),
        "fn_shape": list(fn.shape),
        "x_explicit": bool(row["x_explicit"]),
        "projection": proj_m,
        "mean_sq": mean_m,
        "rms_factor": rms_m,
        "mixes": mix_m,
        "eager_vs_fp64": e64,
        "candidate_vs_fp64": c64,
        "timing": {
            "eager": timing(lambda: eager(x, fn, eps)),
            "candidate": timing(lambda: tilelang_candidate(x, fn, eps)),
        },
        "status": "PASS" if ok else "FAIL",
    }


def zero_case(k: int, eps: float):
    x = torch.zeros((1, k), dtype=torch.bfloat16, device="cuda")
    fn = torch.randn((24, k), dtype=torch.float32, device="cuda") * 0.01
    p, m, r, y = tilelang_candidate(x, fn, eps)
    expected_r = 1.0 / math.sqrt(eps)
    return {
        "k": k,
        "projection_nonzero": int(torch.count_nonzero(p).item()),
        "mean_sq": float(m.item()),
        "rms_factor": float(r.item()),
        "expected_rms_factor": expected_r,
        "mixes_nonzero": int(torch.count_nonzero(y).item()),
        "finite": bool(torch.isfinite(p).all() and torch.isfinite(m).all() and torch.isfinite(r).all() and torch.isfinite(y).all()),
        "status": "PASS" if int(torch.count_nonzero(y).item()) == 0 and bool(torch.isfinite(y).all()) else "FAIL",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    doc = torch.load(args.fixture, map_location="cpu", weights_only=False)
    rows = [case(r) for r in doc["rows"]]
    eps = float(doc["rows"][0]["rms_eps"])
    zeros = [zero_case(5120, eps), zero_case(20480, eps)]
    ok = all(r["status"] == "PASS" for r in rows) and all(z["status"] == "PASS" for z in zeros)
    out = {
        "schema": "ds41-mhc-projection-rms-tilelang-gate-v1",
        "status": "PASS" if ok else "FAIL",
        "tolerances": {
            "projection_rel_l2_max": PROJ_REL_L2_MAX,
            "rms_factor_rel_max": RMS_REL_MAX,
            "mixes_rel_l2_max": MIX_REL_L2_MAX,
        },
        "rows": rows,
        "zero_cases": zeros,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
