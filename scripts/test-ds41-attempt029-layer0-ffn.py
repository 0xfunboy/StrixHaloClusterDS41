#!/usr/bin/env python3
"""Model-free reproducer for attempt029's localized layer0 FFN divergence.

Uses the saved real position42 FFN inputs and raw layer0 GGUF weights. It does
not construct LLM/vLLM model objects or issue generation requests.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize

from runtime.ds41.native_hip_moe_runtime import ensure_loaded
from vllm.model_executor.layers.fused_moe.activation import (
    ApplyMoEActivationConfig,
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import fused_topk_bias
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm_gguf_plugin.quantization.fused_moe import GGUFMoEMethod

ROOT = Path('/home/funboy/StrixHaloClusterDS41')
RAW = ROOT / 'reports/DS41-Q2-001/attempt029'
RANK = int(os.environ.get('DS41_FFN_COMPONENT_RANK', '0'))
OUT = Path(os.environ.get('DS41_FFN_COMPONENT_OUT', str(RAW / f'layer0-ffn-component-rank{RANK}.json')))
ART = json.loads((ROOT / 'runtime/ds41/artifact.json').read_text())
MODEL = Path(ART['model_dir']) / ART['model_file']
CFG = json.loads((Path(ART['model_dir']) / 'config.json').read_text())['text_config']
TOPK = int(CFG['num_experts_per_tok'])
SCALE = float(CFG['routed_scaling_factor'])
LIMIT = float(CFG['swiglu_limit'])
NEXPERT = int(CFG['n_routed_experts'])
INTER = int(CFG['moe_intermediate_size'])
HIDDEN = int(CFG['hidden_size'])
assert (TOPK, NEXPERT, INTER, HIDDEN) == (6, 384, 2304, 5120)


def metric(a: torch.Tensor, b: torch.Tensor) -> dict:
    af, bf = a.float(), b.float()
    if af.shape != bf.shape:
        return {'shape_match': False, 'a_shape': list(af.shape), 'b_shape': list(bf.shape)}
    d = af - bf
    bn = float(torch.linalg.vector_norm(bf))
    dn = float(torch.linalg.vector_norm(d))
    rel = (0.0 if dn == 0.0 else None) if bn == 0.0 else dn / bn
    return {
        'shape_match': True,
        'exact': bool(torch.equal(a, b)),
        'different_elements': int(torch.count_nonzero(a != b)),
        'elements': int(a.numel()),
        'a_norm': float(torch.linalg.vector_norm(af)),
        'b_norm': bn,
        'difference_norm': dn,
        'rel_l2': rel,
        'max_abs': float(d.abs().max()) if d.numel() else 0.0,
        'mean_abs': float(d.abs().mean()) if d.numel() else 0.0,
        'finite': bool(torch.isfinite(af).all() and torch.isfinite(bf).all()),
    }


def bf16(reader: GGUFReader, name: str) -> torch.Tensor:
    t = next(t for t in reader.tensors if t.name == name)
    if t.tensor_type.name != 'BF16':
        raise TypeError((name, t.tensor_type))
    a = np.ascontiguousarray(t.data)
    return torch.from_numpy(a).view(torch.bfloat16).reshape(*(int(x) for x in reversed(t.shape)))


def f32(reader: GGUFReader, name: str) -> torch.Tensor:
    t = next(t for t in reader.tensors if t.name == name)
    if t.tensor_type.name != 'F32':
        raise TypeError((name, t.tensor_type))
    a = np.ascontiguousarray(t.data)
    return torch.from_numpy(a).float().reshape(*(int(x) for x in reversed(t.shape)))


def capture(label: str, stage: str) -> torch.Tensor:
    rows = torch.load(RAW / f'{label}-boundaries-rank{RANK}.pt', map_location='cpu', weights_only=False)
    matches = [r['tensor'] for r in rows if int(r['layer']) == 0 and r['stage'] == stage]
    if len(matches) != 1:
        raise RuntimeError((label, stage, len(matches)))
    return matches[0].contiguous()


def request_inputs() -> dict[str, list[int]]:
    result = json.loads((RAW / f'block-verification-rank{RANK}.json').read_text())
    out = {}
    for label in ('diagnostic-D1', 'diagnostic-B2', 'diagnostic-B4'):
        req = next(r for r in result['requests'] if r['label'] == label)
        step = next(s for s in req['steps'] if s.get('scheduler_computed_before') == 42)
        out[label] = list(step['input_token_ids'])
    return out


def explicit_act(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    g = torch.clamp(gate.float(), max=LIMIT)
    u = torch.clamp(up.float(), -LIMIT, LIMIT)
    return (g * torch.sigmoid(g) * u).to(torch.bfloat16)


def fused_act(gate_up: torch.Tensor) -> torch.Tensor:
    out = torch.empty(gate_up.shape[:-1] + (gate_up.shape[-1] // 2,), dtype=gate_up.dtype, device=gate_up.device)
    apply_moe_activation(
        MoEActivation.SILU,
        out,
        gate_up,
        activation_config=ApplyMoEActivationConfig(clamp_limit=LIMIT),
    )
    return out


def router_candidate(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, input_ids: list[int]):
    logits = torch.mm(x, weight.T, out_dtype=torch.float32)
    ids_t = torch.tensor(input_ids, dtype=torch.int32, device=x.device)
    weights, ids = fused_topk_bias(
        hidden_states=x,
        gating_output=logits,
        scoring_func='sqrtsoftplus',
        e_score_correction_bias=bias,
        topk=TOPK,
        renormalize=True,
        indices_type=torch.int32,
        input_tokens=ids_t,
        hash_indices_table=None,
        routed_scaling_factor=SCALE,
        bias_vl=None,
        image_sentinel_lo=0,
    )
    return logits, weights, ids


def router_reference(x_cpu: torch.Tensor, w_cpu: torch.Tensor, bias_cpu: torch.Tensor):
    # Independent CPU float32/float64 formula; not the ROCm GEMM/router kernel.
    logits = x_cpu.float() @ w_cpu.float().T
    ln = logits.double().numpy()
    bb = bias_cpu.double().numpy()
    scores = np.sqrt(np.logaddexp(ln, 0.0))
    rank_scores = scores + bb[None, :]
    ids = np.argsort(rank_scores, axis=-1)[:, -TOPK:][:, ::-1].copy()
    selected = np.take_along_axis(scores, ids, axis=-1)
    weights = selected / (selected.sum(axis=-1, keepdims=True) + 1e-20) * SCALE
    return logits, torch.from_numpy(weights), torch.from_numpy(ids.astype(np.int32))


def shared_parts_candidate(x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor, linear: UnquantizedLinearMethod):
    parts = []
    for rank in (0, 1):
        lo, hi = rank * (INTER // 2), (rank + 1) * (INTER // 2)
        w13 = torch.cat((gate[lo:hi], up[lo:hi]), dim=0).contiguous()
        gu = linear.apply(SimpleNamespace(weight=w13), x)
        act = fused_act(gu)
        wd = down[:, lo:hi].contiguous()
        y = linear.apply(SimpleNamespace(weight=wd), act)
        parts.append(y)
    return parts


def shared_parts_reference(x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor):
    parts = []
    for rank in (0, 1):
        lo, hi = rank * (INTER // 2), (rank + 1) * (INTER // 2)
        g = (x.float() @ gate[lo:hi].float().T).to(torch.bfloat16)
        u = (x.float() @ up[lo:hi].float().T).to(torch.bfloat16)
        act = explicit_act(g, u)
        y = (act.float() @ down[:, lo:hi].float().T).to(torch.bfloat16)
        parts.append(y)
    return parts


def routed_reference_row0(reader: GGUFReader, x: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Independent gguf-py dequant + explicit FP32 matmul for the real six routes."""
    by = {t.name: t for t in reader.tensors}
    gt, ut, dt = by['blk.0.ffn_gate_exps'], by['blk.0.ffn_up_exps'], by['blk.0.ffn_down_exps']
    result = torch.zeros_like(x[:1], dtype=torch.float32)
    for slot, gid in enumerate(ids[0].detach().cpu().tolist()):
        wg = torch.from_numpy(dequantize(np.ascontiguousarray(gt.data[gid]), gt.tensor_type).astype(np.float32, copy=False)).to('cuda')
        wu = torch.from_numpy(dequantize(np.ascontiguousarray(ut.data[gid]), ut.tensor_type).astype(np.float32, copy=False)).to('cuda')
        wd = torch.from_numpy(dequantize(np.ascontiguousarray(dt.data[gid]), dt.tensor_type).astype(np.float32, copy=False)).to('cuda')
        g = (x[:1].float() @ wg.T).to(torch.bfloat16)
        u = (x[:1].float() @ wu.T).to(torch.bfloat16)
        act = explicit_act(g, u)
        y = (act.float() @ wd.T).to(torch.bfloat16).float()
        result.add_(y * weights[:1, slot:slot+1].float())
        del wg, wu, wd, g, u, act, y
        torch.cuda.empty_cache()
    return result.to(torch.bfloat16)


def selected_quant_weights(reader: GGUFReader, topids: torch.Tensor, start: int, end: int):
    by = {t.name: t for t in reader.tensors}
    gt, ut, dt = by['blk.0.ffn_gate_exps'], by['blk.0.ffn_up_exps'], by['blk.0.ffn_down_exps']
    selected = sorted({int(v) for v in topids.detach().cpu().flatten().tolist() if start <= int(v) < end})
    physical = selected if selected else [start]
    idx = np.asarray(physical, dtype=np.int64)
    wg = torch.from_numpy(np.ascontiguousarray(gt.data[idx])).to('cuda')
    wu = torch.from_numpy(np.ascontiguousarray(ut.data[idx])).to('cuda')
    wd = torch.from_numpy(np.ascontiguousarray(dt.data[idx])).to('cuda')
    w13 = torch.cat((wg, wu), dim=1).contiguous()
    emap = torch.full((NEXPERT,), -1, dtype=torch.int32, device='cuda')
    for lid, gid in enumerate(selected):
        emap[gid] = lid
    return w13, wd, emap, selected


def moe_method():
    return SimpleNamespace(moe=SimpleNamespace(swiglu_limit=LIMIT))


def moe_layer(w13, w2, emap):
    return SimpleNamespace(
        apply_router_weight_on_input=False,
        w13_weight=w13,
        w2_weight=w2,
        w13_weight_type=SimpleNamespace(weight_type=int(gguf.GGMLQuantizationType.IQ2_XXS)),
        w2_weight_type=SimpleNamespace(weight_type=int(gguf.GGMLQuantizationType.Q2_K)),
        activation=SimpleNamespace(value='silu'),
        expert_map=emap,
    )


def routed_parts(reader: GGUFReader, x: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor, rowwise: bool):
    os.environ['DS41_EP_SKIP_REMOTE'] = '1'
    os.environ['DS41_NATIVE_HIP_MOE'] = '1'
    os.environ['DS41_NATIVE_HIP_MOE_ROWWISE'] = '1' if rowwise else '0'
    parts, routes = [], []
    for start, end in ((0, 192), (192, 384)):
        w13, w2, emap, selected = selected_quant_weights(reader, ids, start, end)
        out = GGUFMoEMethod.apply(moe_method(), moe_layer(w13, w2, emap), x, weights, ids, None, None)
        parts.append(out)
        routes.append(selected)
        del w13, w2, emap
        torch.cuda.empty_cache()
    return parts, routes


def reduce_two(parts: list[torch.Tensor]) -> torch.Tensor:
    # Two-rank TP all-reduce arithmetic order for a single SUM can be represented
    # by one pairwise BF16 add; this does not claim RCCL transport equivalence.
    return parts[0] + parts[1]


def main():
    assert torch.cuda.is_available()
    native_identity = ensure_loaded()
    reader = GGUFReader(str(MODEL))
    linear = UnquantizedLinearMethod()
    router_w_cpu = bf16(reader, 'blk.0.ffn_gate_inp')
    router_bias_cpu = f32(reader, 'blk.0.exp_probs_b')
    sg_cpu = bf16(reader, 'blk.0.ffn_gate_shexp')
    su_cpu = bf16(reader, 'blk.0.ffn_up_shexp')
    sd_cpu = bf16(reader, 'blk.0.ffn_down_shexp')
    router_w = router_w_cpu.to('cuda')
    router_bias = router_bias_cpu.to('cuda')
    sg, su, sd = sg_cpu.to('cuda'), su_cpu.to('cuda'), sd_cpu.to('cuda')
    ids_by_label = request_inputs()
    report = {
        'schema': 'ds41-attempt029-layer0-ffn-component-v1',
        'status': 'RUNNING',
        'model_initialized': False,
        'generation_requests': 0,
        'capture_source': 'attempt029 saved layer0 ffn_in/ffn_out at model position42',
        'capture_rank': RANK,
        'native_identity': native_identity,
        'labels': {},
    }
    candidate_cache = {}
    for label in ('diagnostic-D1', 'diagnostic-B2', 'diagnostic-B4'):
        x_cpu = capture(label, 'ffn_in')
        y_capture = capture(label, 'ffn_out')
        x = x_cpu.to('cuda')
        logits, topw, topids = router_candidate(x, router_w, router_bias, ids_by_label[label])
        ref_logits_cpu, refw_cpu, refids_cpu = router_reference(x_cpu, router_w_cpu, router_bias_cpu)
        # Align reference weights to the candidate ID order; top-k set/order is checked separately.
        candidate_ids_cpu = topids.detach().cpu()
        ref_map = []
        for rr in range(candidate_ids_cpu.shape[0]):
            d = {int(i): float(w) for i, w in zip(refids_cpu[rr], refw_cpu[rr])}
            ref_map.append([d.get(int(i), float('nan')) for i in candidate_ids_cpu[rr]])
        aligned_refw = torch.tensor(ref_map, dtype=torch.float64)
        router_row = {
            'candidate_logits_row0_vs_cpu_reference': metric(logits[0].cpu(), ref_logits_cpu[0]),
            'candidate_topk_ids_row0': candidate_ids_cpu[0].tolist(),
            'reference_topk_ids_row0': refids_cpu[0].tolist(),
            'topk_ids_row0_equal_reference': bool(torch.equal(candidate_ids_cpu[0], refids_cpu[0])),
            'candidate_topk_weights_row0': topw[0].detach().cpu().tolist(),
            'candidate_weights_vs_reference_aligned_row0': metric(topw[0].double().cpu(), aligned_refw[0]),
        }
        shared_c_parts = shared_parts_candidate(x, sg, su, sd, linear)
        shared_r_parts = shared_parts_reference(x, sg, su, sd)
        shared_c = reduce_two(shared_c_parts)
        shared_r = reduce_two(shared_r_parts)
        routed_c_parts, selected = routed_parts(reader, x, topw, topids, rowwise=False)
        routed_rowwise_parts, _ = routed_parts(reader, x, topw, topids, rowwise=(x.shape[0] > 1))
        routed_c = reduce_two(routed_c_parts)
        routed_rowwise = reduce_two(routed_rowwise_parts)
        routed_ref = routed_reference_row0(reader, x, topw, topids)
        total_local_parts = [shared_c_parts[i] + routed_c_parts[i] for i in (0, 1)]
        total = reduce_two(total_local_parts)
        total_rowwise_parts = [shared_c_parts[i] + routed_rowwise_parts[i] for i in (0, 1)]
        total_rowwise = reduce_two(total_rowwise_parts)
        report['labels'][label] = {
            'shape': list(x.shape),
            'input_ids': ids_by_label[label],
            'router': router_row,
            'shared_candidate_vs_independent_reference_row0': metric(shared_c[0], shared_r[0]),
            'routed_standard_vs_rowwise_native_row0': metric(routed_c[0], routed_rowwise[0]),
            'routed_standard_vs_independent_reference_row0': metric(routed_c[0], routed_ref[0]),
            'routed_rowwise_native_vs_independent_reference_row0': metric(routed_rowwise[0], routed_ref[0]),
            'selected_experts_by_ep_partition': selected,
            'reconstructed_total_vs_captured_ffn_out_row0': metric(total[0].cpu(), y_capture[0]),
            'reconstructed_rowwise_total_vs_captured_ffn_out_row0': metric(total_rowwise[0].cpu(), y_capture[0]),
        }
        candidate_cache[label] = {
            'logits': logits.detach().cpu(),
            'topw': topw.detach().cpu(),
            'topids': topids.detach().cpu(),
            'shared': shared_c.detach().cpu(),
            'routed': routed_c.detach().cpu(),
            'routed_rowwise': routed_rowwise.detach().cpu(),
            'routed_reference': routed_ref.detach().cpu(),
            'total': total.detach().cpu(),
            'total_rowwise': total_rowwise.detach().cpu(),
            'capture': y_capture,
        }
        del x, logits, topw, topids, shared_c_parts, shared_r_parts, routed_c_parts, routed_rowwise_parts
        torch.cuda.empty_cache()

    ref = candidate_cache['diagnostic-D1']
    for label in ('diagnostic-B2', 'diagnostic-B4'):
        c = candidate_cache[label]
        report['labels'][label]['shape_effect_row0_vs_D1'] = {
            'router_logits': metric(c['logits'][0], ref['logits'][0]),
            'router_topk_ids_equal': bool(torch.equal(c['topids'][0], ref['topids'][0])),
            'router_topk_weights': metric(c['topw'][0], ref['topw'][0]),
            'shared_output': metric(c['shared'][0], ref['shared'][0]),
            'routed_standard_output': metric(c['routed'][0], ref['routed'][0]),
            'routed_rowwise_output': metric(c['routed_rowwise'][0], ref['routed'][0]),
            'routed_independent_reference': metric(c['routed_reference'][0], ref['routed_reference'][0]),
            'reconstructed_total': metric(c['total'][0], ref['total'][0]),
            'reconstructed_rowwise_total': metric(c['total_rowwise'][0], ref['total'][0]),
            'captured_ffn_out': metric(c['capture'][0], ref['capture'][0]),
        }
        captured_delta = c['capture'][0].float() - ref['capture'][0].float()
        reconstructed_delta = c['total'][0].float() - ref['total'][0].float()
        delta_metric = metric(reconstructed_delta, captured_delta)
        denom = float(torch.linalg.vector_norm(reconstructed_delta) * torch.linalg.vector_norm(captured_delta))
        delta_metric['cosine'] = None if denom == 0.0 else float(torch.dot(reconstructed_delta, captured_delta) / denom)
        report['labels'][label]['shape_effect_row0_vs_D1']['reconstructed_delta_vs_captured_delta'] = delta_metric
    report['status'] = 'PASS'
    OUT.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
