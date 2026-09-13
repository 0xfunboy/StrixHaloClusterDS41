#!/usr/bin/env python3
"""Model-free bitwise gate for DS41 detailed mHC profile wrappers."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from runtime.ds41.perf_profile import DS41PerfCollector
import vllm.models.deepseek_v4_1.amd.model as model_mod

ROOT = Path('/home/funboy/StrixHaloClusterDS41')
OUT = ROOT / 'reports/DS41-Q2-001/perf/mhc-profile-equivalence.json'
H = 5120
HC = 4
MIX = (2 + HC) * HC
EPS = 1e-6
RMS_EPS = 1e-20
REPEATS = 20


def clone_tuple(xs):
    return tuple(x.detach().clone() for x in xs)


def exact(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def run_case(tokens: int, delayed: bool):
    g = torch.Generator(device='cuda').manual_seed(9100 + tokens + int(delayed))
    residual = torch.randn((tokens, HC, H), device='cuda', dtype=torch.bfloat16, generator=g)
    fn = (torch.randn((MIX, HC * H), device='cuda', dtype=torch.float32, generator=g) * 0.002).contiguous()
    scale = torch.tensor([0.7, 0.8, 0.3], device='cuda', dtype=torch.float32)
    base = (torch.randn((MIX,), device='cuda', dtype=torch.float32, generator=g) * 0.01).contiguous()
    pre_mix = torch.sigmoid(torch.randn((tokens, HC), device='cuda', dtype=torch.float32, generator=g)) if delayed else None

    collector = DS41PerfCollector()
    collector.install()
    # inactive path delegates to the original pinned implementation
    ref = clone_tuple(model_mod.mhc_pre_delayed_torch(
        residual, fn, scale, base, RMS_EPS, EPS, EPS, 2.0, REPEATS,
        pre_mix=pre_mix,
    ))
    collector.enable()
    cand = clone_tuple(model_mod.mhc_pre_delayed_torch(
        residual, fn, scale, base, RMS_EPS, EPS, EPS, 2.0, REPEATS,
        pre_mix=pre_mix,
    ))
    collector.disable()
    torch.cuda.synchronize()
    if not exact(ref, cand):
        raise AssertionError(f'mhc_pre mismatch tokens={tokens} delayed={delayed}')

    x = torch.randn((tokens, H), device='cuda', dtype=torch.bfloat16, generator=g)
    post_ref = model_mod.mhc_post_torch(x, residual, ref[0], ref[1]).detach().clone()
    collector.enable()
    post_cand = model_mod.mhc_post_torch(x, residual, ref[0], ref[1]).detach().clone()
    collector.disable()
    torch.cuda.synchronize()
    if not torch.equal(post_ref, post_cand):
        raise AssertionError(f'mhc_post mismatch tokens={tokens} delayed={delayed}')

    return {
        'tokens': tokens,
        'delayed': delayed,
        'pre_bitwise': True,
        'post_bitwise': True,
        'pre_shapes': [list(x.shape) for x in ref],
        'post_shape': list(post_ref.shape),
    }


def main():
    assert torch.cuda.is_available()
    rows = [run_case(1, False), run_case(1, True), run_case(7, True)]
    out = {'status': 'PASS', 'sinkhorn_repeat': REPEATS, 'cases': rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
