#!/usr/bin/env python3
"""Numerical/performance gate for DS41 fused mHC coefficients + Sinkhorn."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from runtime.ds41.mhc_coeff_sinkhorn import fused_coeff_sinkhorn

ROOT = Path('/home/funboy/StrixHaloClusterDS41')
OUT = ROOT / 'reports/DS41-Q2-001/perf/mhc-coeff-sinkhorn-gate.json'
ATOL = 5e-6
RTOL = 5e-5
PRE_EPS = 1e-6
SINK_EPS = 1e-6
POST_MULT = 2.0
REPEATS = 20


def reference(mixes, scale, base):
    hc = 4
    pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + PRE_EPS
    post = torch.sigmoid(mixes[:, hc:2*hc] * scale[1] + base[hc:2*hc]) * POST_MULT
    comb = mixes[:, 2*hc:].view(-1, hc, hc) * scale[2]
    comb = comb + base[2*hc:].view(1, hc, hc)
    comb = torch.softmax(comb, dim=-1) + SINK_EPS
    comb = comb / (comb.sum(dim=-2, keepdim=True) + SINK_EPS)
    for _ in range(REPEATS - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + SINK_EPS)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + SINK_EPS)
    return pre, post, comb


def metrics(a, b):
    d = a.float() - b.float()
    rn = float(b.float().norm())
    return {
        'max_abs': float(d.abs().max()),
        'mean_abs': float(d.abs().mean()),
        'rel_l2': float(d.norm()) / max(rn, 1e-30),
    }


def check_case(label, mixes, scale, base):
    ref = reference(mixes, scale, base)
    cand = fused_coeff_sinkhorn(mixes.contiguous(), scale, base, PRE_EPS, SINK_EPS, POST_MULT, REPEATS)
    rows = {}
    for name, a, b in zip(('pre','post','comb'), cand, ref, strict=True):
        torch.testing.assert_close(a, b, atol=ATOL, rtol=RTOL)
        rows[name] = metrics(a, b)
    return {'label': label, 'tokens': int(mixes.shape[0]), 'status': 'PASS', 'metrics': rows}


def time_fn(fn, repeats=300):
    for _ in range(30):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter(); s.record()
    for _ in range(repeats):
        fn()
    e.record(); e.synchronize()
    return {
        'repeats': repeats,
        'gpu_ms_mean': float(s.elapsed_time(e)) / repeats,
        'wall_ms_mean': (time.perf_counter() - t0) * 1000 / repeats,
    }


def real_layer0_mixes():
    # Reuse the already-qualified real-prompt helpers without executing its main().
    import importlib.util
    p = ROOT / 'scripts/test-ds41-layer0-complete-densefix.py'
    spec = importlib.util.spec_from_file_location('ds41_l0_gate', p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    prompt = json.loads(mod.PROMPT.read_text())['prompts']['arithmetic']['token_ids']
    ids = torch.tensor(prompt, dtype=torch.long, device='cuda')
    emb = mod.bf16('token_embd')[ids]
    fn = mod.f32('blk.0.hc_attn_fn').reshape(24, 4, 5120).sum(1).contiguous()
    scale = mod.f32('blk.0.hc_attn_scale').contiguous()
    base = mod.f32('blk.0.hc_attn_base').contiguous()
    x = emb.float()
    mixes = (x @ fn.t()) * torch.rsqrt(x.square().mean(-1, keepdim=True) + mod.EPS)
    return mixes.contiguous(), scale, base


def main():
    assert torch.cuda.is_available()
    real, scale, base = real_layer0_mixes()
    cases = [check_case('real-layer0-all15', real, scale, base), check_case('real-layer0-last-M1', real[-1:], scale, base)]
    g = torch.Generator(device='cuda').manual_seed(17017)
    for amplitude in (0.5, 2.0, 8.0):
        mixes = torch.randn((1,24), generator=g, device='cuda', dtype=torch.float32) * amplitude
        sc = torch.tensor([0.7, 0.9, 0.4], device='cuda', dtype=torch.float32)
        ba = torch.randn((24,), generator=g, device='cuda', dtype=torch.float32) * 0.2
        cases.append(check_case(f'synthetic-M1-amp{amplitude}', mixes, sc, ba))

    m1 = real[-1:].contiguous()
    eager_t = time_fn(lambda: reference(m1, scale, base))
    fused_t = time_fn(lambda: fused_coeff_sinkhorn(m1, scale, base, PRE_EPS, SINK_EPS, POST_MULT, REPEATS))
    out = {
        'status': 'PASS',
        'tolerance': {'atol': ATOL, 'rtol': RTOL},
        'sinkhorn_repeat': REPEATS,
        'cases': cases,
        'timing_real_M1': {
            'eager': eager_t,
            'fused': fused_t,
            'gpu_speedup': eager_t['gpu_ms_mean'] / fused_t['gpu_ms_mean'],
            'wall_speedup': eager_t['wall_ms_mean'] / fused_t['wall_ms_mean'],
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
