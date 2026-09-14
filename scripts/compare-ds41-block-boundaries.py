#!/usr/bin/env python3
"""Compare attempt029 position-42 decoder boundary captures against D1/M1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load(path: Path):
    rows = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for row in rows:
        key = (int(row["layer"]), row["stage"])
        if key in out:
            raise RuntimeError(f"duplicate boundary {key} in {path}")
        out[key] = row["tensor"]
    return out


def first_row(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 0:
        return x.reshape(1)
    return x[0].reshape(-1)


def metric(candidate: torch.Tensor, reference: torch.Tensor):
    a = first_row(candidate).float()
    b = first_row(reference).float()
    if a.shape != b.shape:
        return {"shape_match": False, "candidate_shape": list(a.shape), "reference_shape": list(b.shape)}
    d = a - b
    bn = float(torch.linalg.vector_norm(b))
    return {
        "shape_match": True,
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        "exact": bool(torch.equal(a, b)),
        "rel_l2": float(torch.linalg.vector_norm(d)) / max(bn, 1e-30),
        "max_abs": float(d.abs().max()) if d.numel() else 0.0,
        "mean_abs": float(d.abs().mean()) if d.numel() else 0.0,
        "reference_norm": bn,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    suffix = f"rank{args.rank}.pt"
    ref = load(args.raw / f"diagnostic-D1-boundaries-{suffix}")
    result = {"schema": "ds41-block-boundary-compare-v1", "rank": args.rank, "reference": "diagnostic-D1", "candidates": {}}
    stage_order = ["layer_entry", "attn_norm_in", "attn_norm_out", "attn_in", "attn_out", "ffn_norm_in", "ffn_norm_out", "ffn_in", "ffn_out"]
    order = {name: i for i, name in enumerate(stage_order)}
    for label in ("diagnostic-B2", "diagnostic-B4"):
        cand = load(args.raw / f"{label}-boundaries-{suffix}")
        rows = []
        for key in sorted(set(ref) & set(cand), key=lambda x: (x[0], order.get(x[1], 999), x[1])):
            m = metric(cand[key], ref[key])
            rows.append({"layer": key[0], "stage": key[1], **m})
        result["candidates"][label] = rows
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
