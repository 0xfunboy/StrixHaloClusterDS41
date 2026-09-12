#!/usr/bin/env python3
"""Capture real layer0 routed-MoE inputs from the already-qualified DenseFix gate."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path('/home/funboy/StrixHaloClusterDS41')
SRC = ROOT / 'scripts/test-ds41-layer0-complete-densefix.py'
OUT = ROOT / 'reports/DS41-Q2-001/perf/native-hip/layer0-moe-input.npz'
META = ROOT / 'reports/DS41-Q2-001/perf/native-hip/layer0-moe-input.json'

spec = importlib.util.spec_from_file_location('ds41_l0_capture', SRC)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

original = mod.GGUFMoEMethod.apply
state = {'captured': False}

class Captured(RuntimeError):
    pass

def intercept(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
    if not state['captured']:
        state['captured'] = True
        xb = x.detach().cpu().contiguous().view(mod.torch.uint16).numpy().copy()
        ids = topk_ids.detach().cpu().numpy().astype(np.int32, copy=True)
        weights = topk_weights.detach().cpu().numpy().astype(np.float32, copy=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        np.savez(OUT, x_bf16_bits=xb, topk_ids=ids, topk_weights=weights)
        META.write_text(json.dumps({
            'status':'PASS',
            'source_gate':str(SRC),
            'prompt_file':str(mod.PROMPT),
            'shape_x':list(x.shape),
            'shape_topk_ids':list(topk_ids.shape),
            'dtype_x':str(x.dtype),
            'dtype_topk_ids':str(topk_ids.dtype),
            'dtype_topk_weights':str(topk_weights.dtype),
            'last_token_ids':ids[-1].tolist(),
            'last_token_weights':[float(v) for v in weights[-1]],
            'x_norm_last':float(x[-1].float().norm()),
        },indent=2)+'\n')
        raise Captured('fixture captured before routed MoE')
    return original(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input)

mod.GGUFMoEMethod.apply = intercept
try:
    mod.main()
except Captured:
    pass
finally:
    mod.GGUFMoEMethod.apply = original

if not state['captured'] or not OUT.is_file():
    raise SystemExit('capture failed')
print(META.read_text(), end='')
print(f'FIXTURE={OUT}')
