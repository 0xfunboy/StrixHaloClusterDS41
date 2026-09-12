#!/usr/bin/env python3
"""Real attempt011 layer0 pre-attention correctness gate.

This deliberately STOPS before cache/attention.  It compares the candidate
runtime math against an elementary same-GGUF reference through q_a/kv/q_b, then
(optionally, when the narrow official-source samples exist) compares the GGUF
BF16 source rows against independently decoded official FP8+E8M0 bytes.

The current MixedQ2 artifact is expected to terminate as ARTIFACT_DENSE_FAIL;
continuing attention math with already-corrupted Q/KV would be misleading.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import torch

from vllm import _custom_ops as vllm_ops
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch

ROOT = Path('/home/funboy/StrixHaloClusterDS41')
ART = json.loads((ROOT / 'runtime/ds41/artifact.json').read_text())
MODEL = Path(ART['model_dir'])
PROMPT = ROOT / 'reports/DS41-Q2-001/attempt011/prompt-tokens.json'
SAMPLE = ROOT / 'reports/DS41-Q2-001/stage0/source-shard3-sample'
OUT = ROOT / 'reports/DS41-Q2-001/densefix/layer0-pre-attention-node01.json'
H, HC, QRA, HD, HEADS = 5120, 4, 1280, 512, 64
EPS = 1e-20

readers = [gguf.GGUFReader(str(p)) for p in sorted(MODEL.glob('*.gguf'))]


def tensor(name: str):
    for reader in readers:
        for t in reader.tensors:
            if t.name == name:
                return t
    raise KeyError(name)


def bf16(name: str, device: str = 'cuda') -> torch.Tensor:
    t = tensor(name)
    assert t.tensor_type.name == 'BF16'
    a = t.data.view(np.uint16).reshape(*(int(x) for x in reversed(t.shape)))
    return torch.from_numpy(a).view(torch.bfloat16).to(device)


def f32(name: str, device: str = 'cuda') -> torch.Tensor:
    t = tensor(name)
    assert t.tensor_type.name == 'F32'
    a = np.asarray(t.data).view(np.float32).reshape(*(int(x) for x in reversed(t.shape)))
    return torch.from_numpy(a.copy()).to(device)


def metric(cand: torch.Tensor, ref: torch.Tensor) -> dict:
    c, r = cand.float(), ref.float()
    d = c - r
    rn = float(r.norm())
    return {
        'shape': list(c.shape),
        'max_abs': float(d.abs().max()),
        'mean_abs': float(d.abs().mean()),
        'rel_l2': float(d.norm()) / max(rn, 1e-30),
        'ref_norm': rn,
        'cand_norm': float(c.norm()),
        'cand_finite': bool(torch.isfinite(c).all()),
        'ref_finite': bool(torch.isfinite(r).all()),
    }


def rms_ref(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * w.float()).to(torch.bfloat16)


def rms_candidate(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    out = torch.empty_like(x)
    vllm_ops.rms_norm(out, x, w, eps)
    return out


def hc_ref_first(x, fn, scale, base):
    residual = x[:, None, :].expand(-1, HC, -1).contiguous()
    xf = residual.float().flatten(1)
    mixes = (xf @ fn.float().T) * torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS)
    pre = torch.sigmoid(mixes[:, :HC] * scale[0] + base[:HC]) + 1e-6
    post = torch.sigmoid(mixes[:, HC:2 * HC] * scale[1] + base[HC:2 * HC]) * 2.0
    comb = mixes[:, 2 * HC:].reshape(-1, HC, HC) * scale[2] + base[2 * HC:].reshape(1, HC, HC)
    comb = torch.softmax(comb, dim=-1) + 1e-6
    comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    for _ in range(19):
        comb = comb / (comb.sum(-1, keepdim=True) + 1e-6)
        comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    return residual, post[..., None], comb, x, pre


def fp8_e4m3fn(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.uint8)
    sign = np.where((u & 0x80) != 0, -1.0, 1.0)
    e, m = (u >> 3) & 0x0F, u & 0x07
    out = np.empty(u.shape, dtype=np.float32)
    sub = e == 0
    out[sub] = sign[sub] * (m[sub].astype(np.float32) / 8.0) * (2.0 ** -6)
    out[~sub] = sign[~sub] * (1.0 + m[~sub].astype(np.float32) / 8.0) * np.exp2(e[~sub].astype(np.float32) - 7.0)
    out[(u & 0x7F) == 0x7F] = np.nan
    return out


def source_row(prefix: str, width: int) -> np.ndarray:
    w = np.frombuffer((SAMPLE / f'{prefix}.weight.row0.bin').read_bytes(), dtype=np.uint8)
    s = np.frombuffer((SAMPLE / f'{prefix}.scale.row0.bin').read_bytes(), dtype=np.uint8)
    return fp8_e4m3fn(w) * np.exp2(s.astype(np.float32) - 127.0).repeat(32)[:width]


def gguf_row(name: str, width: int) -> np.ndarray:
    t = tensor(name)
    bits = t.data.view(np.uint16).reshape(-1, width)[0].copy()
    return (bits.astype(np.uint32) << 16).view(np.float32)


def source_metric(src: np.ndarray, got: np.ndarray) -> dict:
    d = got - src
    rn = float(np.linalg.norm(src))
    return {
        'n': int(src.size),
        'source_rms': float(np.sqrt(np.mean(src * src))),
        'gguf_rms': float(np.sqrt(np.mean(got * got))),
        'rel_l2': float(np.linalg.norm(d)) / max(rn, 1e-30),
        'corr': float(np.corrcoef(src, got)[0, 1]),
        'finite': bool(np.isfinite(src).all() and np.isfinite(got).all()),
    }


prompt = json.loads(PROMPT.read_text())['prompts']['arithmetic']['token_ids']
ids = torch.tensor(prompt, dtype=torch.long, device='cuda')
linear = UnquantizedLinearMethod()
report = {'prompt_ids': prompt, 'boundaries': {}, 'source_rows': {}}

emb = bf16('token_embd')[ids]
fn, sc, base = f32('blk.0.hc_attn_fn'), f32('blk.0.hc_attn_scale'), f32('blk.0.hc_attn_base')
fn_broadcast = fn.reshape(24, HC, H).sum(1)
res, post_ref, comb_ref, x_ref, pre_ref = hc_ref_first(emb, fn, sc, base)
post_c, comb_c, x_c, pre_c = mhc_pre_delayed_torch(res, fn_broadcast, sc, base, EPS, 1e-6, 1e-6, 2.0, 20, x=emb)
report['boundaries']['hc_attn_pre'] = {
    'x': metric(x_c, x_ref), 'post': metric(post_c, post_ref),
    'comb': metric(comb_c, comb_ref), 'pre': metric(pre_c, pre_ref),
}

wn = bf16('blk.0.attn_norm')
xn_c, xn_r = rms_candidate(x_c, wn, EPS), rms_ref(x_ref, wn, EPS)
report['boundaries']['attn_norm'] = metric(xn_c, xn_r)

wqa, wkv = bf16('blk.0.attn_q_a'), bf16('blk.0.attn_kv')
fused = torch.cat([wqa, wkv], dim=0).contiguous()
qkv_c = linear.apply(SimpleNamespace(weight=fused), xn_c)
qra_c, kv_c = qkv_c.split([QRA, HD], dim=-1)
qra_r = (xn_r.float() @ wqa.float().T).to(torch.bfloat16)
kv_r = (xn_r.float() @ wkv.float().T).to(torch.bfloat16)
report['boundaries']['q_a_projection'] = metric(qra_c, qra_r)
report['boundaries']['kv_projection'] = metric(kv_c, kv_r)

wqn, wkn = bf16('blk.0.attn_q_a_norm'), bf16('blk.0.attn_kv_a_norm')
qrn_c, kvn_c = rms_candidate(qra_c, wqn, EPS), rms_candidate(kv_c, wkn, EPS)
qrn_r, kvn_r = rms_ref(qra_r, wqn, EPS), rms_ref(kv_r, wkn, EPS)
report['boundaries']['q_norm'] = metric(qrn_c, qrn_r)
report['boundaries']['kv_norm'] = metric(kvn_c, kvn_r)

wqb = bf16('blk.0.attn_q_b')
qb_c = linear.apply(SimpleNamespace(weight=wqb), qrn_c).reshape(len(prompt), HEADS, HD)
qb_r = (qrn_r.float() @ wqb.float().T).to(torch.bfloat16).reshape(len(prompt), HEADS, HD)
report['boundaries']['q_b_projection'] = metric(qb_c, qb_r)

if SAMPLE.exists():
    for src, gg, width in [
        ('layers.0.attn.wq_a', 'blk.0.attn_q_a', 5120),
        ('layers.0.attn.wkv', 'blk.0.attn_kv', 5120),
        ('layers.0.attn.wq_b', 'blk.0.attn_q_b', 1280),
    ]:
        report['source_rows'][src] = source_metric(source_row(src, width), gguf_row(gg, width))

# The publisher states these FP8-derived tensors should be numerically dequantized
# to BF16.  With power-of-two E8M0 scales and E4M3 inputs the row should be very
# close after BF16 cast; rel-L2 ~1 is a hard artifact failure.
artifact_fail = any(v['rel_l2'] > 0.01 for v in report['source_rows'].values())
report['status'] = 'ARTIFACT_DENSE_FAIL' if artifact_fail else 'PASS_PRE_ATTENTION'
report['stopped_before'] = 'q_rope/cache/attention' if artifact_fail else None
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
