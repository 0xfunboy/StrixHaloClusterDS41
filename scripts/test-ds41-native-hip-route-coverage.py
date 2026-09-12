#!/usr/bin/env python3
"""DS41 native HIP M=1 route-coverage gate.

Exercises 0..6 locally owned routes per rank using the real layer0 BF16 input,
real IQ2_XXS/Q2_K expert bytes and the native HIP -1-skip extension.  The six
routing weights remain in their original slots and are never renormalized.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import gguf
import numpy as np
import torch
from gguf.quants import dequantize
from vllm.model_executor.layers.fused_moe.activation import (
    ApplyMoEActivationConfig,
    MoEActivation,
    apply_moe_activation,
)
from vllm_gguf_plugin import ops
from vllm_gguf_plugin.quantization.fused_moe import _fused_moe_gguf_impl

ROOT = Path("/home/funboy/StrixHaloClusterDS41")
FIX = ROOT / "reports/DS41-Q2-001/perf/native-hip/layer0-moe-input.npz"
LIB = Path(os.environ.get(
    "DS41_NATIVE_HIP_MOE_LIB",
    "/home/funboy/models/ds41/native-hip-moe/runtime/_C_gguf.abi3.so",
))
OUT = ROOT / "reports/DS41-Q2-001/perf/native-hip-negskip/route-coverage.json"
MODEL = Path(json.loads((ROOT / "runtime/ds41/artifact.json").read_text())["model_dir"]) / "DSV41-mixedq2-00001-of-00005.gguf"
LIMIT = 10.0
REPEATS = 20
REL_LIMIT = 0.035


def metric(a: torch.Tensor, b: torch.Tensor) -> dict:
    af, bf = a.float(), b.float()
    d = af - bf
    rn = float(bf.norm())
    return {
        "max_abs": float(d.abs().max()) if d.numel() else 0.0,
        "mean_abs": float(d.abs().mean()) if d.numel() else 0.0,
        "rmse": float(torch.sqrt((d * d).mean())) if d.numel() else 0.0,
        "rel_l2": float(d.norm()) / max(rn, 1e-30),
        "a_norm": float(af.norm()),
        "b_norm": rn,
        "finite": bool(torch.isfinite(af).all() and torch.isfinite(bf).all()),
    }


def act(inp: torch.Tensor) -> torch.Tensor:
    d = inp.shape[-1] // 2
    out = torch.empty(inp.shape[:-1] + (d,), dtype=inp.dtype, device=inp.device)
    apply_moe_activation(
        MoEActivation.SILU,
        out,
        inp,
        activation_config=ApplyMoEActivationConfig(clamp_limit=LIMIT),
    )
    return out


def native_prod(x, w13, w2, global_ids, weights, expert_map, q1, q2):
    local_ids = expert_map[global_ids.to(torch.long)].to(torch.int32)
    k = global_ids.shape[1]
    gu = torch.ops._C_gguf.ggml_moe_a8_vec(
        x, w13, local_ids, k, q1, w13.shape[1], x.shape[0]
    )
    a = act(gu)
    down = torch.ops._C_gguf.ggml_moe_a8_vec(
        a, w2, local_ids, 1, q2, w2.shape[1], x.shape[0] * k
    )
    weighted = down.reshape(x.shape[0], k, w2.shape[1]).mul_(
        weights.view(x.shape[0], k, 1)
    )
    out = torch.empty_like(x)
    ops.moe_sum(weighted, out)
    return out, local_ids


def triton_prod(x, w13, w2, global_ids, weights, expert_map, q1, q2):
    return _fused_moe_gguf_impl(
        x, w13, w2, weights, global_ids, q1, q2, "silu", LIMIT, expert_map
    )


def ref_local(x, global_ids, weights, owned, gt, ut, dt):
    out = torch.zeros_like(x, dtype=torch.float32)
    for slot, gid in enumerate(global_ids[0].tolist()):
        if gid not in owned:
            continue
        wg = dequantize(np.ascontiguousarray(gt.data[gid]), gt.tensor_type).astype(
            np.float32, copy=False
        )
        wu = dequantize(np.ascontiguousarray(ut.data[gid]), ut.tensor_type).astype(
            np.float32, copy=False
        )
        wd = dequantize(np.ascontiguousarray(dt.data[gid]), dt.tensor_type).astype(
            np.float32, copy=False
        )
        Wg = torch.from_numpy(wg).to("cuda")
        Wu = torch.from_numpy(wu).to("cuda")
        Wd = torch.from_numpy(wd).to("cuda")
        g = (x.float() @ Wg.T).to(torch.bfloat16).float()
        u = (x.float() @ Wu.T).to(torch.bfloat16).float()
        g = torch.clamp(g, max=LIMIT)
        u = torch.clamp(u, -LIMIT, LIMIT)
        a = (g * torch.sigmoid(g) * u).to(torch.bfloat16)
        y = (a.float() @ Wd.T).to(torch.bfloat16).float()
        out.add_(y * weights[:, slot : slot + 1])
        del Wg, Wu, Wd, wg, wu, wd, g, u, a, y
    return out.to(torch.bfloat16)


def timed(fn, *args):
    for _ in range(5):
        fn(*args)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    s.record()
    last = None
    for _ in range(REPEATS):
        last = fn(*args)
    e.record()
    e.synchronize()
    return {
        "repeats": REPEATS,
        "gpu_ms_mean": float(s.elapsed_time(e)) / REPEATS,
        "wall_ms_mean": (time.perf_counter() - t0) * 1000.0 / REPEATS,
    }, last


def selected_weights(gt, ut, dt, selected: list[int]):
    # Keep at least one valid allocation for the all-remote case.  No negative
    # route may ever index it; new_zeros + expert<0 must leave outputs zero.
    physical = selected if selected else [0]
    idx = np.asarray(physical, dtype=np.int64)
    gate = torch.from_numpy(np.ascontiguousarray(gt.data[idx])).to("cuda")
    up = torch.from_numpy(np.ascontiguousarray(ut.data[idx])).to("cuda")
    down = torch.from_numpy(np.ascontiguousarray(dt.data[idx])).to("cuda")
    w13 = torch.cat((gate, up), dim=1).contiguous()
    return w13, down, physical


def main() -> None:
    assert os.environ.get("VLLM_GGUF_USE_CUDA", "0") == "0"
    torch.ops.load_library(str(LIB))
    z = np.load(FIX)
    xall = torch.from_numpy(z["x_bf16_bits"].copy()).view(torch.bfloat16).to("cuda")
    x = xall[-1:].contiguous()
    weights = torch.from_numpy(z["topk_weights"][-1:].copy()).to("cuda")

    r = gguf.GGUFReader(str(MODEL))
    by = {t.name: t for t in r.tensors}
    gt, ut, dt = (
        by["blk.0.ffn_gate_exps"],
        by["blk.0.ffn_up_exps"],
        by["blk.0.ffn_down_exps"],
    )
    q1, q2 = int(gt.tensor_type), int(dt.tensor_type)

    pools = {
        0: {
            "local": [0, 10, 64, 127, 156, 191],
            "remote": [192, 238, 250, 282, 369, 383],
        },
        1: {
            "local": [192, 238, 250, 282, 369, 383],
            "remote": [0, 10, 64, 127, 156, 191],
        },
    }
    # Deterministic non-grouped slot order so mapping/order is exercised.
    perm = [3, 0, 5, 1, 4, 2]
    report = {
        "schema": "ds41-native-hip-route-coverage-v1",
        "status": "RUNNING",
        "native_lib": str(LIB),
        "native_lib_sha256": __import__("hashlib").sha256(LIB.read_bytes()).hexdigest(),
        "fixture": str(FIX),
        "rel_l2_limit_vs_reference": REL_LIMIT,
        "weights_by_slot": [float(v) for v in weights[0].cpu()],
        "ranks": {},
    }
    all_pass = True

    for rank in (0, 1):
        cases = []
        local_pool = pools[rank]["local"]
        remote_pool = pools[rank]["remote"]
        for nlocal in range(7):
            raw = local_pool[:nlocal] + remote_pool[: 6 - nlocal]
            ids_list = [raw[i] for i in perm]
            gids = torch.tensor([ids_list], dtype=torch.int32, device="cuda")
            selected = sorted([v for v in ids_list if v in set(local_pool)])
            w13, w2, physical = selected_weights(gt, ut, dt, selected)
            emap = torch.full((384,), -1, dtype=torch.int32, device="cuda")
            for lid, gid in enumerate(selected):
                emap[gid] = lid

            nat, mapped = native_prod(x, w13, w2, gids, weights, emap, q1, q2)
            tri = triton_prod(x, w13, w2, gids, weights, emap, q1, q2)
            ref = ref_local(x, gids, weights, set(selected), gt, ut, dt)
            m_nr = metric(nat, ref)
            m_tr = metric(tri, ref)
            m_nt = metric(nat, tri)
            expected_mapped = [selected.index(g) if g in selected else -1 for g in ids_list]
            mapped_list = mapped.cpu().tolist()[0]
            zero_ok = (nlocal != 0) or (
                int(torch.count_nonzero(nat).item()) == 0
                and bool(torch.isfinite(nat).all())
            )
            case_pass = (
                mapped_list == expected_mapped
                and m_nr["finite"]
                and m_nr["rel_l2"] <= REL_LIMIT
                and zero_ok
            )
            timing = None
            if nlocal in (0, 1, 2, 3, 4, 5, 6):
                timing, _ = timed(native_prod, x, w13, w2, gids, weights, emap, q1, q2)
            cases.append(
                {
                    "nlocal": nlocal,
                    "global_ids_by_slot": ids_list,
                    "selected_local_global_ids": selected,
                    "physical_expert_rows": physical,
                    "mapped_local_ids_by_slot": mapped_list,
                    "expected_mapped_local_ids_by_slot": expected_mapped,
                    "native_vs_reference": m_nr,
                    "triton_vs_reference": m_tr,
                    "native_vs_triton": m_nt,
                    "zero_local_exact_zero": zero_ok if nlocal == 0 else None,
                    "native_timing": timing,
                    "status": "PASS" if case_pass else "FAIL",
                }
            )
            all_pass &= case_pass
            del w13, w2, emap, nat, tri, ref, gids, mapped
            torch.cuda.empty_cache()

        # Strong stale-buffer/NaN test: run a nonzero local route first, then an
        # all-remote call with NaN activation.  Native op owns fresh new_zeros Y;
        # expert<0 must leave every routed/down row zero despite prior contents.
        ids_nonzero = torch.tensor([[local_pool[0]] * 6], dtype=torch.int32, device="cuda")
        w13, w2, _ = selected_weights(gt, ut, dt, [local_pool[0]])
        emap = torch.full((384,), -1, dtype=torch.int32, device="cuda")
        emap[local_pool[0]] = 0
        warm, _ = native_prod(x, w13, w2, ids_nonzero, weights, emap, q1, q2)
        warm_nonzero = int(torch.count_nonzero(warm).item())
        ids_remote = torch.tensor([[remote_pool[0]] * 6], dtype=torch.int32, device="cuda")
        x_nan = torch.full_like(x, float("nan"))
        zero_after, mapped = native_prod(x_nan, w13, w2, ids_remote, weights, emap, q1, q2)
        stale_pass = (
            warm_nonzero > 0
            and mapped.cpu().tolist()[0] == [-1] * 6
            and int(torch.count_nonzero(zero_after).item()) == 0
            and bool(torch.isfinite(zero_after).all())
        )
        all_pass &= stale_pass
        report["ranks"][str(rank)] = {
            "cases": cases,
            "stale_nan_zero_test": {
                "prior_nonzero_count": warm_nonzero,
                "mapped_remote_ids": mapped.cpu().tolist()[0],
                "post_nonzero_count": int(torch.count_nonzero(zero_after).item()),
                "post_all_finite": bool(torch.isfinite(zero_after).all()),
                "status": "PASS" if stale_pass else "FAIL",
            },
        }

    report["contract"] = {
        "top6_weights_unchanged": True,
        "local_renormalization": False,
        "global_to_local_map_device_side": True,
        "gpu_to_cpu_route_sync": False,
        "native_output_allocator": "torch::stable::new_zeros",
        "remote_kernel_guard": "expert < 0 before expert weight address",
    }
    report["status"] = "PASS" if all_pass else "FAIL"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
