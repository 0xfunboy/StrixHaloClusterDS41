#!/usr/bin/env python3
"""Independent validator for DS41 same-load Triton-vs-native-HIP A/B."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def result_map(doc):
    return {r["label"]: r for r in doc.get("results", [])}


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def load_code_validator():
    p = Path(__file__).with_name("validate-ds41-offline-suite.py")
    spec = importlib.util.spec_from_file_location("ds41_suite_validator", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.validate_code


def perf(rows, prefix):
    out = []
    for i in (1, 2):
        r = rows[f"hip-ab-{prefix}-{i}"]
        out.append(
            {
                "label": r["label"],
                "completion_tokens": r["completion_token_count"],
                "finish_reason": r.get("finish_reason"),
                "decode_tps": r["derived"]["decode_tps_first_to_last"],
                "ttft_s": r["client"]["ttft_s"],
                "wall_s": r["client"]["wall_s"],
            }
        )
    return {
        "runs": out,
        "decode_tps_mean": mean([x["decode_tps"] for x in out]),
        "ttft_s_mean": mean([x["ttft_s"] for x in out]),
        "wall_s_mean": mean([x["wall_s"] for x in out]),
    }


def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def reasoning_final(text: str) -> str:
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank0", required=True)
    ap.add_argument("--rank1", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    d0 = json.load(open(a.rank0))
    d1 = json.load(open(a.rank1))
    r0, r1 = result_map(d0), result_map(d1)
    labels = sorted(set(r0) | set(r1))
    rank_identity = {
        lab: lab in r0
        and lab in r1
        and r0[lab].get("token_ids") == r1[lab].get("token_ids")
        for lab in labels
    }

    baseline = perf(r0, "baseline")
    candidate = perf(r0, "candidate")
    gain = (
        (candidate["decode_tps_mean"] / baseline["decode_tps_mean"] - 1.0) * 100.0
        if baseline["decode_tps_mean"] and candidate["decode_tps_mean"]
        else None
    )
    wall_gain = (
        (baseline["wall_s_mean"] / candidate["wall_s_mean"] - 1.0) * 100.0
        if baseline["wall_s_mean"] and candidate["wall_s_mean"]
        else None
    )
    wall_reduction = (
        (1.0 - candidate["wall_s_mean"] / baseline["wall_s_mean"]) * 100.0
        if baseline["wall_s_mean"] and candidate["wall_s_mean"]
        else None
    )

    # Numerical candidate may produce a different trajectory.  Record, do not
    # silently require or assume bit identity across A/B.
    ab_token_compare = []
    for i in (1, 2):
        ba = r0[f"hip-ab-baseline-{i}"]["token_ids"]
        ca = r0[f"hip-ab-candidate-{i}"]["token_ids"]
        ab_token_compare.append(
            {
                "run": i,
                "identical": ba == ca,
                "first_diff_index": first_diff(ba, ca),
                "baseline_tokens": len(ba),
                "candidate_tokens": len(ca),
            }
        )

    smoke = r0["hip-ab-candidate-smoke"]
    smoke_pass = smoke.get("finished") and smoke.get("text", "").strip() == "323"
    coding = load_code_validator()(r0["hip-ab-candidate-coding"].get("text", ""))
    try:
        obj = json.loads(r0["hip-ab-candidate-json"].get("text", ""))
        json_pass = obj == {"total": 17, "valid_ids": ["a", "c"]}
        json_result = {"status": "PASS" if json_pass else "FAIL", "parsed": obj}
    except Exception as exc:
        json_pass = False
        json_result = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}

    reasoning = r0["hip-ab-candidate-reasoning-high"]
    reasoning_answer = reasoning_final(reasoning.get("text", ""))
    reasoning_pass = reasoning.get("finished") and reasoning_answer == "10"

    stats0 = d0.get("native_hip_stats") or {}
    stats1 = d1.get("native_hip_stats") or {}
    execution_pass = (
        stats0.get("native_calls", 0) > 0
        and stats1.get("native_calls", 0) > 0
        and (stats0.get("fallback_reasons") or {}).get("tokens_not_1", 0) > 0
        and (stats1.get("fallback_reasons") or {}).get("tokens_not_1", 0) > 0
    )
    same_lib = (
        (d0.get("native_hip_identity") or {}).get("sha256")
        == (d1.get("native_hip_identity") or {}).get("sha256")
    )

    ok = (
        d0.get("status") == "NATIVE_HIP_AB_COMPLETE"
        and d1.get("status") == "NATIVE_HIP_AB_COMPLETE"
        and all(rank_identity.values())
        and smoke_pass
        and coding.get("status") == "PASS"
        and json_pass
        and reasoning_pass
        and execution_pass
        and same_lib
    )
    out = {
        "status": "PASS" if ok else "FAIL",
        "rank_token_identity": rank_identity,
        "baseline": baseline,
        "candidate": candidate,
        "decode_tps_gain_pct": gain,
        "offline_wall_throughput_gain_pct": wall_gain,
        "offline_wall_reduction_pct": wall_reduction,
        "ab_token_compare": ab_token_compare,
        "native_execution": {
            "pass": execution_pass,
            "rank0": stats0,
            "rank1": stats1,
            "same_library_sha": same_lib,
            "rank0_library": (d0.get("native_hip_identity") or {}).get("sha256"),
            "rank1_library": (d1.get("native_hip_identity") or {}).get("sha256"),
        },
        "smoke": {
            "pass": smoke_pass,
            "text": smoke.get("text"),
            "tokens": smoke.get("completion_token_count"),
            "finish_reason": smoke.get("finish_reason"),
            "ttft_s": smoke.get("client", {}).get("ttft_s"),
            "wall_s": smoke.get("client", {}).get("wall_s"),
        },
        "coding": coding,
        "json": json_result,
        "reasoning_high": {
            "pass": reasoning_pass,
            "final_answer": reasoning_answer,
            "text": reasoning.get("text"),
            "tokens": reasoning.get("completion_token_count"),
            "finish_reason": reasoning.get("finish_reason"),
        },
    }
    Path(a.output).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
