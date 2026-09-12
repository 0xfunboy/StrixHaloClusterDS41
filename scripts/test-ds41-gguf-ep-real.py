#!/usr/bin/env python3
"""EP2 equivalence on real V4.1 MixedQ2 expert bytes, for decode and prefill paths."""
from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
from _ds41_artifact import MODEL_FILE

import gguf
import numpy as np
import torch

from vllm_gguf_plugin.quantization.fused_moe import GGUFMoEMethod

MODEL = MODEL_FILE
IQ2_XXS = int(gguf.GGMLQuantizationType.IQ2_XXS)
Q2_K = int(gguf.GGMLQuantizationType.Q2_K)


def raw(reader, name: str, experts: slice) -> torch.Tensor:
    t = next(t for t in reader.tensors if t.name == name)
    # Copy only the four experts used by this component test to the GPU.
    a = np.ascontiguousarray(t.data[experts])
    return torch.from_numpy(a).to('cuda')


def layer(w13, w2, expert_map):
    return SimpleNamespace(
        apply_router_weight_on_input=False,
        w13_weight=w13,
        w2_weight=w2,
        w13_weight_type=SimpleNamespace(weight_type=IQ2_XXS),
        w2_weight_type=SimpleNamespace(weight_type=Q2_K),
        activation=SimpleNamespace(value='silu'),
        expert_map=expert_map,
    )


def method():
    # Match the V4.1 FusedMoEConfig used by the real model: routed SwiGLU is
    # clamped to +/-10 before the multiply.
    return SimpleNamespace(moe=SimpleNamespace(swiglu_limit=10.0))


def run_case(m: int) -> dict:
    reader = gguf.GGUFReader(str(MODEL))
    gate = raw(reader, 'blk.0.ffn_gate_exps', slice(0,4))
    up = raw(reader, 'blk.0.ffn_up_exps', slice(0,4))
    down = raw(reader, 'blk.0.ffn_down_exps', slice(0,4))
    w13 = torch.cat((gate, up), dim=1)
    del gate, up

    g = torch.Generator(device='cuda').manual_seed(41 + m)
    x = torch.randn((m, 5120), generator=g, device='cuda', dtype=torch.bfloat16)
    idx0 = torch.arange(m, device='cuda', dtype=torch.int64) % 4
    topk_ids = torch.stack((idx0, (idx0 + 1) % 4), dim=1)
    topk_weights = torch.tensor([0.625, 0.375], device='cuda', dtype=torch.float32).expand(m, 2).contiguous()

    # Full four-expert reference on one GPU.
    full = GGUFMoEMethod.apply(method(), layer(w13, down, None), x, topk_weights, topk_ids, None, None)

    # Two EP ranks: each owns complete experts; remote route weights become zero.
    map0 = torch.tensor([0, 1, -1, -1], device='cuda', dtype=torch.int32)
    map1 = torch.tensor([-1, -1, 0, 1], device='cuda', dtype=torch.int32)
    r0 = GGUFMoEMethod.apply(method(), layer(w13[:2], down[:2], map0), x, topk_weights, topk_ids, None, None)
    r1 = GGUFMoEMethod.apply(method(), layer(w13[2:], down[2:], map1), x, topk_weights, topk_ids, None, None)
    combined = r0 + r1
    diff = (combined.float() - full.float()).abs()
    result = {
        'm': m,
        'max_abs': float(diff.max()),
        'mean_abs': float(diff.mean()),
        'full_norm': float(full.float().norm()),
        'status': 'PASS',
    }
    torch.testing.assert_close(combined, full, rtol=2e-2, atol=2e-2)
    return result


def main() -> None:
    assert torch.cuda.is_available(), 'ROCm device required'
    # M=1 exercises the vec/decode branch; M=65 crosses the plugin MMQ threshold.
    results = [run_case(1), run_case(65)]
    print({'status':'PASS','cases':results})


if __name__ == '__main__':
    main()
