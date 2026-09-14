#!/usr/bin/env python3
"""CPU-only comparison of attempt029 position-42 decoder boundary captures."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

STAGE_ORDER = [
    "layer_entry",
    "attn_norm_in",
    "attn_norm_out",
    "attn_in",
    "attn_out",
    "ffn_norm_in",
    "ffn_norm_out",
    "ffn_in",
    "ffn_out",
]
ORDER = {name: i for i, name in enumerate(STAGE_ORDER)}
EXPECTED_LAYERS = 40
EXPECTED_BOUNDARIES = EXPECTED_LAYERS * len(STAGE_ORDER)


def load_capture(path: Path) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    rows = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(rows, list):
        raise RuntimeError(f"capture is not a list: {path}")
    seen: Counter[tuple[int, str]] = Counter()
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["layer"]), str(row["stage"]))
        seen[key] += 1
        tensor = row.get("tensor")
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"missing tensor for {key} in {path}")
        out[key] = {
            "tensor": tensor,
            "recorded_shape": row.get("shape"),
            "recorded_dtype": row.get("dtype"),
        }
    dupes = [list(k) for k, n in seen.items() if n != 1]
    expected = {(layer, stage) for layer in range(EXPECTED_LAYERS) for stage in STAGE_ORDER}
    keys = set(out)
    return out, {
        "path": str(path),
        "entries": len(rows),
        "unique_boundaries": len(keys),
        "duplicates": dupes,
        "missing": [list(k) for k in sorted(expected - keys, key=lambda x: (x[0], ORDER[x[1]]))],
        "unexpected": [list(k) for k in sorted(keys - expected)],
        "complete": len(rows) == EXPECTED_BOUNDARIES and keys == expected and not dupes,
    }


def first_row(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 0:
        raise RuntimeError("scalar boundary has no token axis")
    return t[0].reshape(-1)


def metric(candidate_rec: dict[str, Any], reference_rec: dict[str, Any]) -> dict[str, Any]:
    ca = candidate_rec["tensor"]
    rb = reference_rec["tensor"]
    a = first_row(ca).float()
    b = first_row(rb).float()
    base = {
        "candidate_full_shape": list(ca.shape),
        "reference_full_shape": list(rb.shape),
        "candidate_dtype": str(ca.dtype),
        "reference_dtype": str(rb.dtype),
        "candidate_recorded_shape": candidate_rec.get("recorded_shape"),
        "reference_recorded_shape": reference_rec.get("recorded_shape"),
        "candidate_recorded_dtype": candidate_rec.get("recorded_dtype"),
        "reference_recorded_dtype": reference_rec.get("recorded_dtype"),
        "candidate_row_shape": list(a.shape),
        "reference_row_shape": list(b.shape),
    }
    if a.shape != b.shape:
        return {**base, "shape_match": False}
    finite_a = torch.isfinite(a)
    finite_b = torch.isfinite(b)
    finite = bool(finite_a.all() and finite_b.all())
    diff = a - b
    diff_mask = a != b
    diff_count = int(torch.count_nonzero(diff_mask))
    diff_norm = float(torch.linalg.vector_norm(diff)) if diff.numel() else 0.0
    ref_norm = float(torch.linalg.vector_norm(b)) if b.numel() else 0.0
    if ref_norm == 0.0:
        rel_l2 = 0.0 if diff_norm == 0.0 else None
        zero_reference_status = "equal_zero" if diff_norm == 0.0 else "nonzero_candidate_against_zero_reference"
    else:
        rel_l2 = diff_norm / ref_norm
        zero_reference_status = "not_zero"
    return {
        **base,
        "shape_match": True,
        "finite": finite,
        "candidate_nonfinite": int((~finite_a).sum()),
        "reference_nonfinite": int((~finite_b).sum()),
        "exact": bool(torch.equal(a, b)),
        "different_elements": diff_count,
        "elements": int(a.numel()),
        "reference_norm": ref_norm,
        "difference_norm": diff_norm,
        "zero_reference_status": zero_reference_status,
        "rel_l2": rel_l2,
        "max_abs": float(diff.abs().max()) if diff.numel() else 0.0,
        "mean_abs": float(diff.abs().mean()) if diff.numel() else 0.0,
    }


def find_request(result: dict[str, Any], label: str) -> dict[str, Any]:
    return next(r for r in result["requests"] if r["label"] == label)


def step_at_position(request: dict[str, Any], position: int) -> dict[str, Any]:
    matches = [s for s in request["steps"] if s.get("scheduler_computed_before") == position]
    if len(matches) != 1:
        raise RuntimeError(f"{request['label']}: expected one step at {position}, got {len(matches)}")
    return matches[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--position", type=int, default=42)
    args = ap.parse_args()

    result_path = args.raw / f"block-verification-rank{args.rank}.json"
    result = json.loads(result_path.read_text())
    labels = ("diagnostic-D1", "diagnostic-B2", "diagnostic-B4")
    expected_widths = {"diagnostic-D1": 1, "diagnostic-B2": 2, "diagnostic-B4": 4}
    provenance: dict[str, Any] = {
        "result_json": str(result_path),
        "result_status": result.get("status"),
        "source_commit": result.get("source_commit"),
        "boundary_capture_position": result.get("config", {}).get("boundary_capture_position"),
        "rowwise_native_control": bool(result.get("config", {}).get("rowwise_native_control", False)),
        "requests": {},
    }

    captures: dict[str, dict[tuple[int, str], dict[str, Any]]] = {}
    capture_meta: dict[str, Any] = {}
    for label in labels:
        req = find_request(result, label)
        step = step_at_position(req, args.position)
        expected_width = expected_widths[label]
        if int(step.get("num_positions", -1)) != expected_width:
            raise RuntimeError(f"{label}: num_positions {step.get('num_positions')} != {expected_width}")
        positions = list(step.get("positions") or [])
        input_ids = list(step.get("input_token_ids") or [])
        if not positions or positions[0] != args.position:
            raise RuntimeError(f"{label}: first position is not {args.position}: {positions}")
        cap_path = args.raw / f"{label}-boundaries-rank{args.rank}.pt"
        capture, meta = load_capture(cap_path)
        if not meta["complete"]:
            raise RuntimeError(f"incomplete capture {label}: {meta}")
        first_axis_bad = []
        for (layer, stage), rec in capture.items():
            t = rec["tensor"]
            if t.ndim == 0 or int(t.shape[0]) != expected_width:
                first_axis_bad.append({"layer": layer, "stage": stage, "shape": list(t.shape)})
        if first_axis_bad:
            raise RuntimeError(f"{label}: token-axis mismatch: {first_axis_bad[:8]}")
        captures[label] = capture
        capture_meta[label] = meta
        provenance["requests"][label] = {
            "desired_k": req.get("desired_k"),
            "positions": positions,
            "input_token_ids": input_ids,
            "num_positions": step.get("num_positions"),
            "num_drafts": step.get("num_drafts"),
            "scheduler_computed_before": step.get("scheduler_computed_before"),
            "capture": meta,
            "row0_maps_to_position": positions[0],
            "row0_input_token_id": input_ids[0],
        }

    # The row being compared must be the same real token/position in every arm.
    row0_keys = {
        (v["positions"][0], v["input_token_ids"][0])
        for v in provenance["requests"].values()
    }
    if len(row0_keys) != 1:
        raise RuntimeError(f"row0 provenance differs across arms: {row0_keys}")

    ref = captures["diagnostic-D1"]
    report: dict[str, Any] = {
        "schema": "ds41-block-boundary-compare-v2",
        "rank": args.rank,
        "position": args.position,
        "reference": "diagnostic-D1",
        "provenance": provenance,
        "capture_meta": capture_meta,
        "candidates": {},
    }
    ordered_keys = sorted(ref, key=lambda x: (x[0], ORDER[x[1]]))
    for label in ("diagnostic-B2", "diagnostic-B4"):
        cand = captures[label]
        if set(cand) != set(ref):
            raise RuntimeError(f"boundary key mismatch for {label}")
        rows = []
        for key in ordered_keys:
            rows.append({"layer": key[0], "stage": key[1], **metric(cand[key], ref[key])})
        report["candidates"][label] = rows

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
