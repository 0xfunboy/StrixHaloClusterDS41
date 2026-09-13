#!/usr/bin/env python3
"""Validate same-load WO_B LLMM1 execution, quality, and frozen speed gates.

Correctness and speed have independent statuses. CLI success requires both;
passing quality alone never makes a candidate eligible for promotion.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import statistics
from pathlib import Path

# Exact candidate mean from runtime/ds41/results/attempt020-projection-rms-ab.json.
HISTORICAL_TPS = 12.086645570628823
ORDER = ["A1", "B1", "B2", "A2", "A3", "B3"]
PROMOTED_SWITCHES = (
    "DS41_EP_SKIP_REMOTE", "DS41_NATIVE_HIP_MOE", "DS41_MHC_COEFF_SINKHORN",
    "DS41_MHC_PROJECTION_RMS", "DS41_DECOMPOSED_QKV_INSERT",
)
# SHA256 of the attempt020 prompt token lists, serialized with separators=(',', ':').
PROMPT_HASHES = {
    "speed": "52ea477f5a5c498e0d5368554009b2a388e4157cebe971860907078b10003214",
    "smoke": "4e2c51b8353588c42b352735b9f5cd2bb25a5d0f716bc84f5691ef689705fe37",
    "coding": "b493b6915eec62d09dbec74572a30344f0cb820db53d79ed03eac95584c32c25",
    "json": "448e7a25e632ae67640fd442656d1ec42b25c2b7b2d488281f04b868ec2054c4",
    "reasoning-high": "2ee59736a1cc500aaa1e58aed162988da06a4ffb500e6a6021f89e169e749492",
}
REQUESTS = [
    ("baseline-warmup-excluded", "speed", 32, True),
    ("candidate-warmup-excluded", "speed", 32, True),
    ("baseline-1", "speed", 128, True),
    ("candidate-1", "speed", 128, True),
    ("candidate-2", "speed", 128, True),
    ("baseline-2", "speed", 128, True),
    ("baseline-3", "speed", 128, True),
    ("candidate-3", "speed", 128, True),
    ("candidate-smoke", "smoke", 128, False),
    ("candidate-coding", "coding", 512, False),
    ("candidate-json", "json", 256, False),
    ("candidate-reasoning-high", "reasoning-high", 128, False),
]
LABELS = [f"wob-ab-{suffix}" for suffix, *_ in REQUESTS]


def code_validator():
    path = Path(__file__).with_name("validate-ds41-offline-suite.py")
    spec = importlib.util.spec_from_file_location("ds41_offline_quality", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_code


def finite_number(value, *, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def token_list(row):
    tokens = row.get("token_ids")
    return (isinstance(tokens, list) and bool(tokens)
            and all(type(token) is int and token >= 0 for token in tokens)
            and type(row.get("completion_token_count")) is int
            and row["completion_token_count"] == len(tokens))


def natural_completion(row, cap):
    return (token_list(row) and row.get("finished") is True
            and row.get("finish_reason") == "stop"
            and row["completion_token_count"] <= cap)


def execution_check(row, candidate):
    switches = row.get("runtime_switches") or {}
    snapshot = row.get("wob_stats") or {}
    counts = [snapshot.get(key) for key in ("llmm1_calls", "llmm1_tokens", "fallback_calls")]
    reasons = snapshot.get("fallback_reasons")
    valid = (all(type(count) is int and count >= 0 for count in counts)
             and isinstance(reasons, dict)
             and all(type(count) is int and count > 0 for count in reasons.values()))
    if not valid:
        return False
    calls, tokens, fallbacks = counts
    modes_ok = (all(switches.get(key) == "1" for key in PROMOTED_SWITCHES)
                and switches.get("DS41_ATTN_WOB_LLMM1") == ("1" if candidate else "0"))
    if candidate:
        dispatch_ok = (calls > 0 and tokens == calls and fallbacks > 0
                       and set(reasons) == {"tokens_not_1"}
                       and fallbacks == reasons["tokens_not_1"])
    else:
        dispatch_ok = calls == tokens == fallbacks == 0 and reasons == {}
    return modes_ok and dispatch_ok


def stats_total(rows):
    total = {"llmm1_calls": 0, "llmm1_tokens": 0, "fallback_calls": 0,
             "fallback_reasons": {}}
    for row in rows.values():
        snapshot = row.get("wob_stats") or {}
        for key in ("llmm1_calls", "llmm1_tokens", "fallback_calls"):
            count = snapshot.get(key)
            if type(count) is not int or count < 0:
                return None
            total[key] += count
        reasons = snapshot.get("fallback_reasons")
        if not isinstance(reasons, dict):
            return None
        for reason, count in reasons.items():
            if type(count) is not int or count <= 0:
                return None
            total["fallback_reasons"][reason] = total["fallback_reasons"].get(reason, 0) + count
    return total


def performance(rows, arm):
    runs = []
    for pair in (1, 2, 3):
        label = f"wob-ab-{arm}-{pair}"
        row = rows.get(label, {})
        runs.append({
            "label": label, "completion_tokens": row.get("completion_token_count"),
            "finish_reason": row.get("finish_reason"),
            "decode_tps": (row.get("derived") or {}).get("decode_tps_first_to_last"),
            "ttft_s": (row.get("client") or {}).get("ttft_s"),
            "wall_s": (row.get("client") or {}).get("wall_s"),
        })
    if not all(finite_number(run["decode_tps"], positive=True) for run in runs):
        return {"runs": runs, "decode_tps_mean": None}
    tps = [run["decode_tps"] for run in runs]
    mean = statistics.mean(tps)
    sd = statistics.stdev(tps)
    def metric_mean(key):
        values = [run[key] for run in runs]
        return statistics.mean(values) if all(finite_number(value) for value in values) else None
    return {
        "runs": runs, "decode_tps_mean": mean, "decode_tps_sample_sd": sd,
        "decode_tps_cv_pct": sd / mean * 100, "decode_tps_min": min(tps),
        "decode_tps_max": max(tps), "ttft_s_mean": metric_mean("ttft_s"),
        "wall_s_mean": metric_mean("wall_s"),
    }


def quality(rows, validate_code):
    out = {}
    for task, cap in (("smoke", 128), ("coding", 512), ("json", 256), ("reasoning-high", 128)):
        row = rows.get(f"wob-ab-candidate-{task}", {})
        text = row.get("text", "")
        text = text if isinstance(text, str) else ""
        detail = {}
        if task == "smoke":
            content_ok = text.strip() == "323"
        elif task == "coding":
            detail = validate_code(text)
            content_ok = detail.get("status") == "PASS"
        elif task == "json":
            try:
                parsed = json.loads(text)
                content_ok = parsed == {"total": 17, "valid_ids": ["a", "c"]}
                detail = {"parsed": parsed}
            except (ValueError, TypeError) as exc:
                content_ok = False
                detail = {"error": f"{type(exc).__name__}: {exc}"}
        else:
            answer = text.rsplit("</think>", 1)[-1].strip()
            content_ok = answer == "10"
            detail = {"final_answer": answer}
        natural = natural_completion(row, cap)
        out[task] = {
            **detail, "status": "PASS" if content_ok and natural else "FAIL",
            "content_pass": content_ok, "natural_completion_pass": natural,
            "text": text, "tokens": row.get("completion_token_count"),
            "finish_reason": row.get("finish_reason"),
        }
    return out


def validate(d0, d1):
    docs = [d0, d1]
    rows = [{row.get("label"): row for row in doc.get("results", [])
             if isinstance(row, dict)} for doc in docs]
    order_pass = all(
        doc.get("order") == ORDER
        and [row.get("label") for row in doc.get("results", []) if isinstance(row, dict)] == LABELS
        and len(doc.get("results", [])) == len(LABELS)
        for doc in docs
    )
    rank_headers_pass = all(doc.get("rank") == rank and doc.get("world_size") == 2
                            and doc.get("status") == "ATTN_WOB_AB_COMPLETE"
                            for rank, doc in enumerate(docs))
    rank_identity = {
        label: (label in rows[0] and label in rows[1]
                and token_list(rows[0][label]) and token_list(rows[1][label])
                and rows[0][label]["token_ids"] == rows[1][label]["token_ids"])
        for label in sorted(set(LABELS) | set(rows[0]) | set(rows[1]), key=str)
    }
    shas = [(doc.get("native_hip_identity") or {}).get("sha256") for doc in docs]
    same_lib = (isinstance(shas[0], str) and re.fullmatch(r"[0-9a-fA-F]{64}", shas[0]) is not None
                and shas[0] == shas[1])
    protocol = {}
    execution = {}
    for rank, rank_rows in enumerate(rows):
        rank_protocol, rank_execution = {}, {}
        for suffix, prompt, cap, ignore_eos in REQUESTS:
            label = f"wob-ab-{suffix}"
            row = rank_rows.get(label, {})
            request = row.get("request_protocol") or {}
            request_ok = (request.get("max_tokens") == cap
                          and request.get("ignore_eos") is ignore_eos
                          and request.get("prompt_tokens_sha256") == PROMPT_HASHES[prompt]
                          and token_list(row) and row.get("finished") is True)
            if ignore_eos:
                request_ok = (request_ok and row.get("completion_token_count") == cap
                              and row.get("finish_reason") == "length"
                              and finite_number((row.get("derived") or {}).get("decode_tps_first_to_last"), positive=True)
                              and finite_number((row.get("client") or {}).get("wall_s"), positive=True)
                              and finite_number((row.get("client") or {}).get("ttft_s")))
            else:
                request_ok = request_ok and natural_completion(row, cap)
            rank_protocol[label] = bool(request_ok)
            rank_execution[label] = execution_check(row, suffix.startswith("candidate-"))
        protocol[f"rank{rank}"] = rank_protocol
        execution[f"rank{rank}"] = rank_execution
    total_stats_pass = all(stats_total(rank_rows) == doc.get("wob_stats")
                           and stats_total(rank_rows) is not None
                           for rank_rows, doc in zip(rows, docs))
    execution_pass = same_lib and total_stats_pass and all(all(values.values()) for values in execution.values())
    speed_protocol_pass = all(protocol[f"rank{rank}"][f"wob-ab-{suffix}"]
                              for rank in (0, 1) for suffix, _, _, ignore in REQUESTS if ignore)
    protocol_pass = all(all(values.values()) for values in protocol.values())
    validate_code = code_validator()
    quality_results = {f"rank{rank}": quality(rank_rows, validate_code)
                       for rank, rank_rows in enumerate(rows)}
    quality_pass = all(task["status"] == "PASS" for tasks in quality_results.values() for task in tasks.values())
    correctness = (rank_headers_pass and order_pass and all(rank_identity.values())
                   and protocol_pass and execution_pass and quality_pass)

    baseline, candidate = (performance(rows[0], arm) for arm in ("baseline", "candidate"))
    a_mean, b_mean = baseline["decode_tps_mean"], candidate["decode_tps_mean"]
    gain = (b_mean / a_mean - 1) * 100 if a_mean and b_mean else None
    paired, comparisons = [], []
    for pair in (1, 2, 3):
        a_row, b_row = (rows[0].get(f"wob-ab-{arm}-{pair}", {}) for arm in ("baseline", "candidate"))
        av, bv = ((row.get("derived") or {}).get("decode_tps_first_to_last") for row in (a_row, b_row))
        valid = finite_number(av, positive=True) and finite_number(bv, positive=True)
        paired.append({"pair": pair, "a_tps": av, "b_tps": bv,
                       "gain_pct": (bv / av - 1) * 100 if valid else None,
                       "b_gt_a": bool(valid and bv > av)})
        a_tokens, b_tokens = a_row.get("token_ids", []), b_row.get("token_ids", [])
        first_diff = next((i for i, (a, b) in enumerate(zip(a_tokens, b_tokens)) if a != b), None)
        if first_diff is None and len(a_tokens) != len(b_tokens):
            first_diff = min(len(a_tokens), len(b_tokens))
        comparisons.append({"pair": pair, "identical": a_tokens == b_tokens,
                            "first_diff_index": first_diff,
                            "a_tokens": len(a_tokens), "b_tokens": len(b_tokens)})
    all_pairs_win = all(pair["b_gt_a"] for pair in paired)
    mean_gain_pass = gain is not None and gain >= 5.0
    speed = order_pass and speed_protocol_pass and all_pairs_win and mean_gain_pass
    promote = correctness and speed
    return {
        "schema": "ds41-wob-llmm1-ab-validation-v1",
        "status": "PASS" if promote else "FAIL",
        "correctness_status": "PASS" if correctness else "FAIL",
        "speed_status": "PASS" if speed else "FAIL",
        "promotion_eligible": promote,
        "promotion": {"all_three_pairs_b_gt_a": all_pairs_win,
                      "mean_gain_pct_min": 5.0, "mean_gain_pass": mean_gain_pass},
        "rank_headers_pass": rank_headers_pass, "rank_token_identity": rank_identity,
        "order_pass": order_pass, "speed_protocol_pass": speed_protocol_pass,
        "request_protocol": protocol, "baseline": baseline, "candidate": candidate,
        "paired_gains": paired, "decode_tps_gain_pct": gain,
        "historical_promoted_tps": HISTORICAL_TPS,
        "gain_vs_historical_pct": (b_mean / HISTORICAL_TPS - 1) * 100 if b_mean else None,
        "ab_token_compare": comparisons,
        "wob_execution": {"pass": execution_pass, "requests": execution,
                          "total_stats_pass": total_stats_pass,
                          "same_native_library_sha": same_lib,
                          "rank0_library": shas[0], "rank1_library": shas[1],
                          "rank0": d0.get("wob_stats"), "rank1": d1.get("wob_stats")},
        "quality": quality_results,
        "process_resources": {f"rank{rank}": {label: row.get("process_resources")
                                               for label, row in rank_rows.items()}
                              for rank, rank_rows in enumerate(rows)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank0", required=True)
    parser.add_argument("--rank1", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    docs = [json.loads(Path(path).read_text()) for path in (args.rank0, args.rank1)]
    result = validate(*docs)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    Path(args.output).write_text(rendered)
    print(rendered, end="")
    return 0 if result["promotion_eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
