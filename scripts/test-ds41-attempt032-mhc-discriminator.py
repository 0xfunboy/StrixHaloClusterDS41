#!/usr/bin/env python3
"""Attempt032 model-free discriminator for layer0 FFN delayed-mHC carry drift.

Uses only saved attempt029/031 boundary tensors plus the real DenseFix GGUF
layer0 parameters.  It never constructs LLM/model/executor/process-group.

PB = current M>1 Torch projection/RMS.
P1 = promoted TileLang M1 projection/RMS applied independently per row.
CB = current M>1 Torch coefficient/softmax/Sinkhorn.
C1 = promoted fused coefficient/Sinkhorn (already row-programmed for [T,24]).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import gguf
import numpy as np
import torch

from runtime.ds41.mhc_coeff_sinkhorn import (
    fused_coeff_sinkhorn,
    reset_stats as coeff_reset_stats,
    stats as coeff_stats,
)
from runtime.ds41.mhc_projection_rms import (
    projection_rms_tilelang,
    reset_stats as projection_reset_stats,
    stats as projection_stats,
)
from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_delayed_torch

ROOT = Path('/home/funboy/StrixHaloClusterDS41')
ART = json.loads((ROOT / 'runtime/ds41/artifact.json').read_text())
MODEL_DIR = Path(ART['model_dir'])
CFG = json.loads((MODEL_DIR / 'config.json').read_text())['text_config']
RANK = int(os.environ.get('DS41_DISCRIMINATOR_RANK', os.environ.get('RANK', '0')))
RAW29 = ROOT / 'reports/DS41-Q2-001/attempt029'
RAW31 = ROOT / 'reports/DS41-Q2-001/attempt031'
OUT = Path(os.environ.get(
    'DS41_MHC_DISCRIMINATOR_OUT',
    str(ROOT / f'reports/DS41-Q2-001/attempt032/mhc-discriminator-rank{RANK}.json'),
))

H = int(CFG['hidden_size'])
HC = int(CFG['hc_mult'])
EPS = float(CFG['rms_norm_eps'])
HC_EPS = float(CFG['hc_eps'])
SINK_ITERS = int(CFG['hc_sinkhorn_iters'])
POST_MULT = 2.0
assert (H, HC, SINK_ITERS) == (5120, 4, 20)

readers = [gguf.GGUFReader(str(p)) for p in sorted(MODEL_DIR.glob('*.gguf'))]
by_name = {t.name: t for r in readers for t in r.tensors}


def source_identity() -> str:
    try:
        value = subprocess.check_output(
            ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        value = (ROOT / '.source-commit').read_text().strip()
    assert len(value) == 40
    return value


def f32(name: str) -> torch.Tensor:
    t = by_name[name]
    assert t.tensor_type.name == 'F32', (name, t.tensor_type)
    a = np.asarray(t.data).view(np.float32).reshape(
        *(int(x) for x in reversed(t.shape))
    ).copy()
    return torch.from_numpy(a).to('cuda').contiguous()


def load_boundary(path: Path) -> dict[tuple[int, str], torch.Tensor]:
    rows = torch.load(path, map_location='cpu', weights_only=False)
    out = {}
    for row in rows:
        key = (int(row['layer']), str(row['stage']))
        if key in out:
            raise RuntimeError(f'duplicate boundary {key} in {path}')
        out[key] = row['tensor'].detach().clone()
    return out


def boundary_file(raw: Path, label: str, rank: int) -> Path:
    return raw / f'{label}-boundaries-rank{rank}.pt'


def metric(cand: torch.Tensor, ref: torch.Tensor) -> dict:
    c = cand.detach().float().cpu().reshape(-1)
    r = ref.detach().float().cpu().reshape(-1)
    if c.shape != r.shape:
        return {
            'shape_match': False, 'candidate_shape': list(c.shape),
            'reference_shape': list(r.shape),
        }
    finite = bool(torch.isfinite(c).all() and torch.isfinite(r).all())
    d = c - r
    rn = float(torch.linalg.vector_norm(r))
    dn = float(torch.linalg.vector_norm(d))
    if rn == 0.0:
        rel = 0.0 if dn == 0.0 else None
        zero_state = 'both_zero' if dn == 0.0 else 'reference_zero_candidate_nonzero'
    else:
        rel = dn / rn
        zero_state = 'nonzero_reference'
    return {
        'shape_match': True,
        'shape': list(cand.shape),
        'candidate_dtype': str(cand.dtype),
        'reference_dtype': str(ref.dtype),
        'exact': bool(torch.equal(c, r)),
        'different_elements': int(torch.count_nonzero(c != r)),
        'numel': int(c.numel()),
        'reference_norm': rn,
        'difference_norm': dn,
        'relative_l2': rel,
        'zero_norm_handling': zero_state,
        'max_abs': float(d.abs().max()) if d.numel() else 0.0,
        'mean_abs': float(d.abs().mean()) if d.numel() else 0.0,
        'finite': finite,
    }


def row0(x: torch.Tensor) -> torch.Tensor:
    return x[0:1]


def torch_projection(source: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
    xf = source.float()
    return ((xf @ fn.t()) * torch.rsqrt(
        xf.square().mean(-1, keepdim=True) + EPS
    )).contiguous()


def rowwise_projection(source: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
    return torch.cat([
        projection_rms_tilelang(source[i:i+1].contiguous(), fn, EPS)
        for i in range(source.shape[0])
    ], dim=0).contiguous()


def torch_coeff(mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor):
    pre = torch.sigmoid(mixes[:, :HC] * scale[0] + base[:HC]) + HC_EPS
    post = torch.sigmoid(
        mixes[:, HC:2*HC] * scale[1] + base[HC:2*HC]
    ) * POST_MULT
    comb = mixes[:, 2*HC:].view(-1, HC, HC) * scale[2]
    comb = comb + base[2*HC:].view(1, HC, HC)
    comb = torch.softmax(comb, dim=-1) + HC_EPS
    comb = comb / (comb.sum(dim=-2, keepdim=True) + HC_EPS)
    for _ in range(SINK_ITERS - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + HC_EPS)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + HC_EPS)
    return pre.contiguous(), post.contiguous(), comb.contiguous()


def fused_coeff(mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor):
    return fused_coeff_sinkhorn(
        mixes.contiguous(), scale, base, HC_EPS, HC_EPS, POST_MULT, SINK_ITERS
    )


def concat_fused_m1(mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor):
    rows = [fused_coeff(mixes[i:i+1], scale, base) for i in range(mixes.shape[0])]
    return tuple(torch.cat([r[j] for r in rows], dim=0) for j in range(3))


def coeff_pack(pre, post, comb, residual, previous_pre):
    layer_input = (
        previous_pre.unsqueeze(-1) * residual.float()
    ).sum(dim=1).to(residual.dtype)
    return {
        'next_pre': pre,
        'post_mix': post.unsqueeze(-1),
        'res_mix': comb,
        'layer_input': layer_input,
    }


def compare_pack(pack, ref_pack):
    return {k: metric(row0(pack[k]), row0(ref_pack[k])) for k in ref_pack}


def reconstruct_ffn_inputs(label: str, broad: dict, narrow: dict, attn_fn_b, attn_scale, attn_base):
    entry = broad[(0, 'layer_entry')].to('cuda').contiguous()
    attn_out = broad[(0, 'attn_out')].to('cuda').contiguous()
    residual0 = entry.unsqueeze(1).expand(-1, HC, -1).contiguous()
    old_p = os.environ.get('DS41_MHC_PROJECTION_RMS')
    old_c = os.environ.get('DS41_MHC_COEFF_SINKHORN')
    os.environ['DS41_MHC_PROJECTION_RMS'] = '1'
    os.environ['DS41_MHC_COEFF_SINKHORN'] = '1'
    try:
        post, comb, layer_input, attn_pre = mhc_pre_delayed_torch(
            residual0, attn_fn_b, attn_scale, attn_base,
            EPS, HC_EPS, HC_EPS, POST_MULT, SINK_ITERS, x=entry,
        )
    finally:
        if old_p is None: os.environ.pop('DS41_MHC_PROJECTION_RMS', None)
        else: os.environ['DS41_MHC_PROJECTION_RMS'] = old_p
        if old_c is None: os.environ.pop('DS41_MHC_COEFF_SINKHORN', None)
        else: os.environ['DS41_MHC_COEFF_SINKHORN'] = old_c
    residual = mhc_post_torch(attn_out, residual0, post, comb)
    saved_state = narrow[(1, 'state_residual')].to('cuda')
    saved_entry = narrow[(1, 'state_x')].to('cuda')
    return {
        'entry': entry,
        'attention_layer_input': layer_input,
        'attention_pre': attn_pre,
        'attention_post': post,
        'attention_comb': comb,
        'residual_before_ffn_pre': residual,
        'source': residual.flatten(1).contiguous(),
        'saved_state_residual': saved_state,
        'saved_state_x': saved_entry,
        'reconstruction_vs_saved_state_residual_row0': metric(row0(residual), row0(saved_state)),
    }


def main():
    assert torch.cuda.is_available()
    labels = ('diagnostic-D1', 'diagnostic-B2', 'diagnostic-B4')
    broad = {label: load_boundary(boundary_file(RAW29, label, RANK)) for label in labels}
    narrow = {label: load_boundary(boundary_file(RAW31, label, RANK)) for label in labels}
    widths = {label: int(broad[label][(0,'layer_entry')].shape[0]) for label in labels}
    if widths != {'diagnostic-D1':1, 'diagnostic-B2':2, 'diagnostic-B4':4}:
        raise RuntimeError(f'unexpected saved widths: {widths}')

    attn_fn = f32('blk.0.hc_attn_fn')
    attn_fn_b = attn_fn.reshape(24, HC, H).sum(1).contiguous()
    attn_scale = f32('blk.0.hc_attn_scale')
    attn_base = f32('blk.0.hc_attn_base')
    ffn_fn = f32('blk.0.hc_ffn_fn')
    ffn_scale = f32('blk.0.hc_ffn_scale')
    ffn_base = f32('blk.0.hc_ffn_base')

    projection_reset_stats(); coeff_reset_stats()
    reconstructed = {
        label: reconstruct_ffn_inputs(
            label, broad[label], narrow[label], attn_fn_b, attn_scale, attn_base
        ) for label in labels
    }

    # The causal FFN projection source for row0 must be the same before comparing shapes.
    source_identity_metrics = {
        label: metric(row0(reconstructed[label]['source']), row0(reconstructed['diagnostic-D1']['source']))
        for label in ('diagnostic-B2','diagnostic-B4')
    }

    # Previous delayed pre-mix affects current layer_input/collapse, not the new carries.
    previous_pre_metrics = {
        label: metric(row0(reconstructed[label]['attention_pre']), row0(reconstructed['diagnostic-D1']['attention_pre']))
        for label in ('diagnostic-B2','diagnostic-B4')
    }

    mixes = {}
    coeffs = {}
    for label in labels:
        src = reconstructed[label]['source']
        pb = torch_projection(src, ffn_fn)
        p1 = rowwise_projection(src, ffn_fn)
        mixes[label] = {'PB': pb, 'P1': p1}
        coeffs[label] = {
            'PB_CB': torch_coeff(pb, ffn_scale, ffn_base),
            'PB_C1': fused_coeff(pb, ffn_scale, ffn_base),
            'P1_CB': torch_coeff(p1, ffn_scale, ffn_base),
            'P1_C1': fused_coeff(p1, ffn_scale, ffn_base),
        }

    # M1 promoted fidelity target on the exact D1 row/data.
    ref_coeff = coeffs['diagnostic-D1']['P1_C1']
    ref_pack = coeff_pack(*ref_coeff, reconstructed['diagnostic-D1']['residual_before_ffn_pre'], reconstructed['diagnostic-D1']['attention_pre'])

    report = {
        'schema':'ds41-attempt032-mhc-discriminator-v1',
        'status':'RUNNING', 'rank':RANK, 'source_commit':source_identity(),
        'model_initialized':False, 'generation_requests':0,
        'saved_widths':widths,
        'real_parameters':{
            'ffn_fn_shape':list(ffn_fn.shape),'scale_shape':list(ffn_scale.shape),
            'base_shape':list(ffn_base.shape),'rms_eps':EPS,'hc_eps':HC_EPS,
            'sinkhorn_iters':SINK_ITERS,'post_mult':POST_MULT,
        },
        'reconstruction':{
            label:{'residual_before_ffn_pre_vs_saved_layer1_state_residual_row0': reconstructed[label]['reconstruction_vs_saved_state_residual_row0']}
            for label in labels
        },
        'row0_ffn_projection_source_vs_m1':source_identity_metrics,
        'row0_previous_attn_pre_vs_m1':previous_pre_metrics,
        'cases':{},
    }

    # Production M1 direct wrapper sanity: it must equal the explicit P1+C1 target.
    old_p=os.environ.get('DS41_MHC_PROJECTION_RMS'); old_c=os.environ.get('DS41_MHC_COEFF_SINKHORN')
    os.environ['DS41_MHC_PROJECTION_RMS']='1'; os.environ['DS41_MHC_COEFF_SINKHORN']='1'
    try:
        d1=reconstructed['diagnostic-D1']
        wrap=mhc_pre_delayed_torch(
            d1['residual_before_ffn_pre'], ffn_fn, ffn_scale, ffn_base,
            EPS, HC_EPS, HC_EPS, POST_MULT, SINK_ITERS,
            pre_mix=d1['attention_pre'],
        )
    finally:
        if old_p is None: os.environ.pop('DS41_MHC_PROJECTION_RMS',None)
        else: os.environ['DS41_MHC_PROJECTION_RMS']=old_p
        if old_c is None: os.environ.pop('DS41_MHC_COEFF_SINKHORN',None)
        else: os.environ['DS41_MHC_COEFF_SINKHORN']=old_c
    wrap_pack={'post_mix':wrap[0], 'res_mix':wrap[1], 'layer_input':wrap[2], 'next_pre':wrap[3]}
    report['m1_wrapper_vs_explicit_P1_C1']={k:metric(wrap_pack[k],ref_pack[k]) for k in ref_pack}

    for label in ('diagnostic-B2','diagnostic-B4'):
        pb=mixes[label]['PB']; p1=mixes[label]['P1']
        same_pb_cb=torch_coeff(pb,ffn_scale,ffn_base)
        same_pb_c1=fused_coeff(pb,ffn_scale,ffn_base)
        same_p1_cb=torch_coeff(p1,ffn_scale,ffn_base)
        same_p1_c1=fused_coeff(p1,ffn_scale,ffn_base)
        concat_pb_c1=concat_fused_m1(pb,ffn_scale,ffn_base)
        concat_p1_c1=concat_fused_m1(p1,ffn_scale,ffn_base)
        combos={
            'PB_CB':same_pb_cb,'PB_C1':same_pb_c1,
            'P1_CB':same_p1_cb,'P1_C1':same_p1_c1,
        }
        row={
            'tokens':int(pb.shape[0]),
            'projection_PB_vs_P1_row0':metric(row0(pb),row0(p1)),
            'projection_PB_vs_M1_P1_row0':metric(row0(pb),row0(mixes['diagnostic-D1']['P1'])),
            'projection_P1_vs_M1_P1_row0':metric(row0(p1),row0(mixes['diagnostic-D1']['P1'])),
            'coefficient_isolation_on_same_PB_mixes':{
                name:metric(row0(a),row0(b)) for name,a,b in zip(
                    ('pre','post','comb'),same_pb_c1,same_pb_cb,strict=True)
            },
            'coefficient_isolation_on_same_P1_mixes':{
                name:metric(row0(a),row0(b)) for name,a,b in zip(
                    ('pre','post','comb'),same_p1_c1,same_p1_cb,strict=True)
            },
            'C1_batch_vs_concat_T1_on_PB':{
                name:metric(a,b) for name,a,b in zip(('pre','post','comb'),same_pb_c1,concat_pb_c1,strict=True)
            },
            'C1_batch_vs_concat_T1_on_P1':{
                name:metric(a,b) for name,a,b in zip(('pre','post','comb'),same_p1_c1,concat_p1_c1,strict=True)
            },
            'combinations_vs_M1_target_row0':{},
        }
        for cname,c in combos.items():
            pack=coeff_pack(*c,reconstructed[label]['residual_before_ffn_pre'],reconstructed[label]['attention_pre'])
            row['combinations_vs_M1_target_row0'][cname]=compare_pack(pack,ref_pack)
        report['cases'][label]=row

    report['projection_stats']=projection_stats()
    report['coefficient_stats']=coeff_stats()

    # Hard integrity gates; numerical discriminator itself is reported, not tuned.
    for label,m in source_identity_metrics.items():
        if not m.get('exact'):
            raise RuntimeError(f'row0 FFN projection source is not exact M1 for {label}: {m}')
    for label in labels:
        m=report['reconstruction'][label]['residual_before_ffn_pre_vs_saved_layer1_state_residual_row0']
        if not m.get('exact'):
            raise RuntimeError(f'reconstructed layer0 residual does not match saved state for {label}: {m}')
    for m in report['m1_wrapper_vs_explicit_P1_C1'].values():
        if not m.get('exact'):
            raise RuntimeError(f'explicit P1+C1 does not reproduce promoted M1 wrapper: {m}')
    for label,row in report['cases'].items():
        for branch in ('C1_batch_vs_concat_T1_on_PB','C1_batch_vs_concat_T1_on_P1'):
            for m in row[branch].values():
                if not m.get('exact'):
                    raise RuntimeError(f'C1 T batching changed row programs {label}/{branch}: {m}')
    report['status']='PASS'
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
