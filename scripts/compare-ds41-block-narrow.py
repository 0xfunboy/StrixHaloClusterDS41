#!/usr/bin/env python3
"""Compare a config-declared narrow position capture against D1/M1 on CPU."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

ORDER = {
    name: i for i, name in enumerate([
        "state_x", "state_pre_mix", "state_post_mix", "state_res_mix", "state_residual",
        "layer_entry", "engram_out", "attn_norm_in", "attn_norm_out", "attn_in", "attn_out",
        "ffn_norm_in", "ffn_norm_out", "ffn_in", "ffn_out",
    ])
}


def first_row(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 0:
        raise RuntimeError("scalar capture has no token axis")
    return t[0].reshape(-1)


def metric(a0: torch.Tensor, b0: torch.Tensor) -> dict:
    a, b = first_row(a0).float(), first_row(b0).float()
    out = {
        "candidate_full_shape": list(a0.shape), "reference_full_shape": list(b0.shape),
        "candidate_dtype": str(a0.dtype), "reference_dtype": str(b0.dtype),
        "candidate_row_shape": list(a.shape), "reference_row_shape": list(b.shape),
    }
    if a.shape != b.shape:
        return {**out, "shape_match": False}
    fa, fb = torch.isfinite(a), torch.isfinite(b)
    d = a - b
    rn = float(torch.linalg.vector_norm(b)); dn = float(torch.linalg.vector_norm(d))
    if rn == 0.0:
        rel = 0.0 if dn == 0.0 else None
        zs = "equal_zero" if dn == 0.0 else "nonzero_candidate_against_zero_reference"
    else:
        rel, zs = dn / rn, "not_zero"
    return {**out, "shape_match": True, "finite": bool(fa.all() and fb.all()),
        "candidate_nonfinite": int((~fa).sum()), "reference_nonfinite": int((~fb).sum()),
        "exact": bool(torch.equal(a, b)), "different_elements": int(torch.count_nonzero(a != b)),
        "elements": int(a.numel()), "reference_norm": rn, "difference_norm": dn,
        "zero_reference_status": zs, "rel_l2": rel,
        "max_abs": float(d.abs().max()) if d.numel() else 0.0,
        "mean_abs": float(d.abs().mean()) if d.numel() else 0.0}


def load(path: Path, expected: set[tuple[int, str]], width: int) -> dict[tuple[int, str], torch.Tensor]:
    rows = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for row in rows:
        key = (int(row["layer"]), str(row["stage"]))
        if key in out: raise RuntimeError(f"duplicate {key} in {path}")
        t = row["tensor"]
        if t.ndim == 0 or int(t.shape[0]) != width:
            raise RuntimeError(f"token-axis mismatch {key} {tuple(t.shape)} != width {width}")
        out[key] = t
    if set(out) != expected:
        raise RuntimeError(f"capture keys mismatch {path}: missing={sorted(expected-set(out))} unexpected={sorted(set(out)-expected)}")
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--raw',type=Path,required=True); ap.add_argument('--rank',type=int,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--position',type=int,default=42); a=ap.parse_args()
    result=json.loads((a.raw/f'block-verification-rank{a.rank}.json').read_text())
    cfg=result['config']; declared=cfg.get('boundary_capture')
    if not isinstance(declared,dict) or not declared: raise RuntimeError('narrow comparison requires boundary_capture mapping')
    expected={(int(layer),stage) for layer,stages in declared.items() for stage in stages}
    labels=(('diagnostic-D1',1),('diagnostic-B2',2),('diagnostic-B4',4))
    caps={}; provenance={}
    for label,width in labels:
        req=next(x for x in result['requests'] if x['label']==label)
        steps=[x for x in req['steps'] if x.get('scheduler_computed_before')==a.position]
        if len(steps)!=1: raise RuntimeError(f'{label} step{a.position} count={len(steps)}')
        step=steps[0]; pos=list(step['positions']); ids=list(step['input_token_ids'])
        if step['num_positions']!=width or not pos or pos[0]!=a.position: raise RuntimeError(f'{label} position/width mismatch')
        caps[label]=load(a.raw/f'{label}-boundaries-rank{a.rank}.pt',expected,width)
        provenance[label]={"positions":pos,"input_token_ids":ids,"row0_position":pos[0],"row0_input_id":ids[0],"num_positions":width,"num_drafts":step['num_drafts']}
    row0={(v['row0_position'],v['row0_input_id']) for v in provenance.values()}
    if len(row0)!=1: raise RuntimeError(f'row0 provenance mismatch {row0}')
    keys=sorted(expected,key=lambda x:(x[0],ORDER.get(x[1],999),x[1])); ref=caps['diagnostic-D1']
    report={"schema":"ds41-block-narrow-compare-v1","rank":a.rank,"position":a.position,
        "source_commit":result.get('source_commit'),"rowwise_native_control":bool(cfg.get('rowwise_native_control')),
        "boundary_capture":declared,"provenance":provenance,"candidates":{}}
    for label in ('diagnostic-B2','diagnostic-B4'):
        rows=[]
        for key in keys: rows.append({"layer":key[0],"stage":key[1],**metric(caps[label][key],ref[key])})
        report['candidates'][label]=rows
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
