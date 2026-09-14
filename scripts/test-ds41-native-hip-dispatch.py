#!/usr/bin/env python3
"""Verify DS41 GGUFMoEMethod dispatches only qualified M=1 to native HIP."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import torch

from runtime.ds41.native_hip_moe_runtime import ensure_loaded
from vllm_gguf_plugin.quantization.fused_moe import (
    GGUFMoEMethod,
    ds41_native_hip_stats,
)

ROOT = Path("/home/funboy/StrixHaloClusterDS41")
ART = json.loads((ROOT / "runtime/ds41/artifact.json").read_text())
MODEL = Path(ART["model_dir"]) / ART["model_file"]
OUT = Path(os.environ.get(
    "DS41_NATIVE_HIP_DISPATCH_OUT",
    str(ROOT / "reports/DS41-Q2-001/perf/native-hip-negskip/dispatch-node01.json"),
))
IQ2 = int(gguf.GGMLQuantizationType.IQ2_XXS)
Q2K = int(gguf.GGMLQuantizationType.Q2_K)


def raw(reader, name: str, idx: np.ndarray) -> torch.Tensor:
    t = next(t for t in reader.tensors if t.name == name)
    return torch.from_numpy(np.ascontiguousarray(t.data[idx])).to("cuda")


def layer(w13, w2, expert_map):
    return SimpleNamespace(
        apply_router_weight_on_input=False,
        w13_weight=w13,
        w2_weight=w2,
        w13_weight_type=SimpleNamespace(weight_type=IQ2),
        w2_weight_type=SimpleNamespace(weight_type=Q2K),
        activation=SimpleNamespace(value="silu"),
        expert_map=expert_map,
    )


def method():
    return SimpleNamespace(moe=SimpleNamespace(swiglu_limit=10.0))


def apply(lyr, x, weights, ids, native: bool, rowwise: bool = False):
    os.environ["DS41_EP_SKIP_REMOTE"] = "1"
    os.environ["DS41_NATIVE_HIP_MOE"] = "1" if native else "0"
    os.environ["DS41_NATIVE_HIP_MOE_ROWWISE"] = "1" if rowwise else "0"
    return GGUFMoEMethod.apply(method(), lyr, x, weights, ids, None, None)


def metric(a, b):
    af, bf = a.float(), b.float(); d = af - bf; rn = float(bf.norm())
    return {
        "max_abs": float(d.abs().max()),
        "mean_abs": float(d.abs().mean()),
        "rel_l2": float(d.norm()) / max(rn, 1e-30),
        "a_norm": float(af.norm()),
        "b_norm": rn,
    }


def main():
    ident = ensure_loaded()
    reader = gguf.GGUFReader(str(MODEL))
    # Four local expert rows; six global routes mix local 10/156 with remote ids.
    selected = [10, 64, 127, 156]
    idx = np.asarray(selected, dtype=np.int64)
    gate = raw(reader, "blk.0.ffn_gate_exps", idx)
    up = raw(reader, "blk.0.ffn_up_exps", idx)
    down = raw(reader, "blk.0.ffn_down_exps", idx)
    w13 = torch.cat((gate, up), dim=1).contiguous()
    emap = torch.full((384,), -1, dtype=torch.int32, device="cuda")
    for local_id, gid in enumerate(selected): emap[gid] = local_id
    lyr = layer(w13, down, emap)
    ids1 = torch.tensor([[282, 156, 10, 238, 369, 250]], dtype=torch.int32, device="cuda")
    weights1 = torch.tensor([[0.31, 0.24, 0.23, 0.22, 0.28, 0.18]], dtype=torch.float32, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(4242)
    x1 = torch.randn((1, 5120), generator=g, dtype=torch.bfloat16, device="cuda")

    before = ds41_native_hip_stats()
    tri1 = apply(lyr, x1, weights1, ids1, False)
    native1 = apply(lyr, x1, weights1, ids1, True)
    after_m1 = ds41_native_hip_stats()
    m1 = metric(native1, tri1)
    assert after_m1["native_calls"] > before["native_calls"]
    assert m1["rel_l2"] < 0.035

    # Diagnostic B2/B4 control: rowwise M1 must equal concatenating the exact
    # same native M1 operation independently for each row.
    rowwise = {}
    for m in (2, 4):
        xm = x1.expand(m, -1).contiguous()
        idsm = ids1.expand(m, -1).contiguous()
        weightsm = weights1.expand(m, -1).contiguous()
        expected = torch.cat([
            apply(lyr, xm[i:i+1], weightsm[i:i+1], idsm[i:i+1], True, False)
            for i in range(m)
        ], dim=0)
        got = apply(lyr, xm, weightsm, idsm, True, True)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        rowwise[str(m)] = metric(got, expected)
    after_rowwise = ds41_native_hip_stats()

    # M=16 is outside the bounded diagnostic dispatch and must remain Triton.
    x16 = x1.expand(16, -1).contiguous()
    ids16 = ids1.expand(16, -1).contiguous()
    weights16 = weights1.expand(16, -1).contiguous()
    tri16 = apply(lyr, x16, weights16, ids16, False)
    native_requested16 = apply(lyr, x16, weights16, ids16, True)
    after_m16 = ds41_native_hip_stats()
    torch.testing.assert_close(native_requested16, tri16, rtol=0, atol=0)
    assert after_m16["native_calls"] == after_rowwise["native_calls"]
    assert after_m16["fallback_reasons"].get("tokens_not_1", 0) > after_m1["fallback_reasons"].get("tokens_not_1", 0)

    out = {
        "status": "PASS",
        "library_identity": ident,
        "m1_native_vs_triton": m1,
        "rowwise_native_vs_concatenated_m1": rowwise,
        "m16_native_requested_vs_triton": metric(native_requested16, tri16),
        "stats_before": before,
        "stats_after_m1": after_m1,
        "stats_after_rowwise": after_rowwise,
        "stats_after_m16": after_m16,
        "dispatch": {
            "m1": "native HIP",
            "m2_m4_rowwise_opt_in": "native HIP M1 per row, no collective here",
            "m16": "Triton fallback",
            "topk": 6,
            "dtype": "bfloat16",
            "qtypes": ["IQ2_XXS", "Q2_K"],
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
