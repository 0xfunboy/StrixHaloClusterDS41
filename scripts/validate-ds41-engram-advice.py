#!/usr/bin/env python3
"""Validate the frozen Engram MADV_NORMAL/MADV_RANDOM experiment offline."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import mmap
from pathlib import Path
import statistics


SCHEMA = "ds41-engram-advice-validation-v1"
ORDER = ["A1", "B1", "B2", "A2", "A3", "B3"]
PHASES = ["warmup-excluded", "Pfirst", "Prepeat", "Qnew", "Qrepeat"]
QUALITY = ["arithmetic", "coding", "json", "reasoning_high"]
QUALITY_CAPS = {"arithmetic": 128, "coding": 512, "json": 256, "reasoning_high": 128}
BASES = ["layers.1.engram.embed", "layers.14.engram.embed"]
NATIVE_SHA = "a860f50f3c52535c6fbde6dcbb6e6fa5f8b699fae5301ba77f56f02400eaa07d"
DENSEFIX_MANIFEST_SHA = "c22f0b8211e6f703c04ce5bb8746c6b412d60fe0b7d0311ddc004974c66f0284"
SWITCHES = {key: "1" for key in ("DS41_EP_SKIP_REMOTE", "DS41_NATIVE_HIP_MOE",
            "DS41_MHC_COEFF_SINKHORN", "DS41_MHC_PROJECTION_RMS", "DS41_DECOMPOSED_QKV_INSERT")}
SWITCHES["DS41_ATTN_WOB_LLMM1"] = "0"
PROMPTS = {
    "P": (35, "52ea477f5a5c498e0d5368554009b2a388e4157cebe971860907078b10003214"),
    "Q": (37, "dcf6f2bbe1dec7dc1ff01fe9b4062d933f22d68eae7bf19fac90fad96d292329"),
    "arithmetic": (15, "4e2c51b8353588c42b352735b9f5cd2bb25a5d0f716bc84f5691ef689705fe37"),
    "coding": (47, "b493b6915eec62d09dbec74572a30344f0cb820db53d79ed03eac95584c32c25"),
    "json": (83, "448e7a25e632ae67640fd442656d1ec42b25c2b7b2d488281f04b868ec2054c4"),
    "reasoning_high": (73, "2ee59736a1cc500aaa1e58aed162988da06a4ffb500e6a6021f89e169e749492"),
}
REQUESTS = [(f"engram-{arm}-{phase}", arm[0], "Q" if phase.startswith("Q") else "P",
             32 if phase == "warmup-excluded" else 128, True)
            for arm in ORDER for phase in PHASES]
REQUESTS += [(f"engram-quality-{arm}-{task}", arm, task, QUALITY_CAPS[task], False)
             for arm in ("A", "B") for task in QUALITY]
LABELS = [item[0] for item in REQUESTS]


def finite(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def count(value, positive=False):
    return type(value) is int and (value > 0 if positive else value >= 0)


def tokens_ok(row):
    tokens = row.get("token_ids")
    return (isinstance(tokens, list) and bool(tokens) and all(count(token) for token in tokens)
            and count(row.get("completion_token_count")) and len(tokens) == row["completion_token_count"])


def same_tokens(rows):
    return bool(rows) and all(tokens_ok(row) for row in rows) and all(row["token_ids"] == rows[0]["token_ids"] for row in rows[1:])


def measures(row):
    diagnostic = row.get("fault_diagnostic") or {}
    before = diagnostic.get("start") or {}
    after = diagnostic.get("end") or {}
    old = (before.get("process_io") or {}).get("read_bytes")
    new = (after.get("process_io") or {}).get("read_bytes")
    wall = (row.get("client") or {}).get("wall_s")
    overhead = diagnostic.get("instrumentation_overhead_ns")
    return {
        "decode_tps": (row.get("derived") or {}).get("decode_tps_first_to_last"),
        "decode_span_s": (row.get("derived") or {}).get("decode_span_s"),
        "wall_s": wall, "ttft_s": (row.get("client") or {}).get("ttft_s"),
        "physical_read_bytes": new - old if count(old) and count(new) and new >= old else None,
        "observer_overhead_ns": overhead,
        "observer_overhead_fraction_of_client_wall": overhead / (wall * 1e9)
        if count(overhead) and finite(wall, True) else None,
    }


def stats(values):
    valid = bool(values) and all(finite(value) for value in values)
    return {"n": len(values), "valid": valid, "values": values,
            "mean": statistics.mean(values) if valid else None,
            "median": statistics.median(values) if valid else None,
            "sample_sd": statistics.stdev(values) if valid and len(values) > 1 else None,
            "min": min(values) if valid else None, "max": max(values) if valid else None}


def range_signature(advice):
    fields = ("tensor", "tensor_start", "tensor_end_exclusive", "advice_start", "advice_length", "unadvised_boundary_bytes")
    return [[region.get(field) for field in fields]
            for region in sorted(advice.get("ranges", []), key=lambda region: region.get("tensor", ""))]


def advice_ok(advice, policy):
    base = advice.get("base")
    regions = advice.get("ranges", [])
    if not (base in BASES and advice.get("requested") == policy and advice.get("effective") == policy
            and advice.get("status") == "applied" and advice.get("errors") == []
            and advice.get("scope") == "complete_pages_within_selected_tensors"
            and len(regions) == 3
            and sorted(region.get("tensor", "") for region in regions)
            == sorted(base + "." + suffix for suffix in ("weight", "scales", "biases"))):
        return False
    for region in regions:
        begin, end, start, length, skipped = (region.get(key) for key in
            ("tensor_start", "tensor_end_exclusive", "advice_start", "advice_length", "unadvised_boundary_bytes"))
        if not (all(count(value) for value in (begin, end, start, length, skipped)) and length > 0
                and start % mmap.PAGESIZE == 0 and length % mmap.PAGESIZE == 0
                and begin <= start < start + length <= end and skipped == end - begin - length
                and region.get("applied") is True and region.get("effective") == policy):
            return False
    return True


def policies_ok(records, arm):
    return (isinstance(records, list) and len(records) == 2
            and sorted(item.get("base", "") for item in records) == BASES
            and all(advice_ok(item, "normal" if arm == "A" else "random") for item in records))


def preparations(document):
    records = document.get("engram_preparation", [])
    files = ((document.get("artifact_identity") or {}).get("engram") or {}).get("files", [])
    expected_files = {item.get("realpath"): item.get("size") for item in files}
    details, signatures = [], []
    order_ok = isinstance(records, list) and [record.get("arm") for record in records] == ORDER
    for record in records:
        arm = record.get("arm", "")
        policy = "normal" if arm.startswith("A") else "random"
        embeddings = record.get("embeddings", [])
        checks = {"arm_policy": record.get("policy") == policy,
                  "two_embeddings": len(embeddings) == 2 and sorted(item.get("base", "") for item in embeddings) == BASES,
                  "exact_identified_files": len(files) == len(expected_files) == 2
                  and {item.get("path") for item in embeddings} == set(expected_files)}
        signature = []
        for embedding in embeddings:
            base = embedding.get("base", "")
            advice = embedding.get("advice") or {}
            ranges = embedding.get("ranges", [])
            identity = embedding.get("discard_file_identity") or {}
            checks[base + "_discard_scope"] = embedding.get("discard_scope") == "whole_identified_engram_file_test_only"
            checks[base + "_discard_mapping_count"] = type(embedding.get("discard_mapping_count")) is int and embedding["discard_mapping_count"] == 2
            checks[base + "_discard_file_identity"] = (
                isinstance(identity, dict) and set(identity) == {"st_dev", "st_ino", "st_size", "st_mtime_ns"}
                and all(count(identity[field], field in ("st_ino", "st_size")) for field in identity)
                and identity["st_size"] == expected_files.get(embedding.get("path")))
            expected = sorted((region[3], region[4]) for region in range_signature(advice))
            actual = sorted((region.get("start"), region.get("length")) for region in ranges)
            empty = bool(ranges) and all(
                count((region.get("after") or {}).get("resident_pages"))
                and region["after"]["resident_pages"] == 0
                and count((region.get("before") or {}).get("pages"), True)
                and region["before"]["pages"] == region["after"].get("pages")
                and region.get("length") == region["after"]["pages"] * mmap.PAGESIZE
                for region in ranges)
            checks[base] = bool(
                embedding.get("fresh_open_description") is True
                and count(embedding.get("old_fd")) and count(embedding.get("new_fd"))
                and embedding["old_fd"] != embedding["new_fd"]
                and embedding.get("cache_max_rows_before") == 65536
                and embedding.get("cache_max_rows_after") == 65536
                and str(embedding.get("path", "")).endswith(".safetensors")
                and advice.get("base") == base and advice_ok(advice, policy)
                and expected == actual and len(actual) == 3 and empty)
            signature.append({"base": base, "path": embedding.get("path"),
                              "discard_file_identity": identity,
                              "ranges": range_signature(advice),
                              "pages": [region["after"].get("pages") for region in ranges]})
        signatures.append(sorted(signature, key=lambda value: value["base"]))
        details.append({"arm": arm, "pass": all(checks.values()), "checks": checks})
    same = len(signatures) == 6 and all(value == signatures[0] for value in signatures[1:])
    return {"pass": order_ok and same and all(item["pass"] for item in details),
            "order_pass": order_ok, "same_pages_and_ranges_across_arms": same,
            "arms": details, "range_signature": signatures[0] if signatures else None}


def load_code_validator():
    path = Path(__file__).with_name("validate-ds41-offline-suite.py")
    spec = importlib.util.spec_from_file_location("ds41_existing_code_validator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_code


def quality_result(row, task, validate_code):
    text = row.get("text", "")
    text = text if isinstance(text, str) else ""
    detail = {}
    if task == "arithmetic":
        content = text.strip() == "323"
    elif task == "coding":
        detail = validate_code(text)
        content = detail.get("status") == "PASS"
    elif task == "json":
        try:
            detail["parsed"] = json.loads(text)
            content = detail["parsed"] == {"total": 17, "valid_ids": ["a", "c"]}
        except (ValueError, TypeError):
            content = False
    else:
        detail["final_answer"] = text.rsplit("</think>", 1)[-1].strip()
        content = detail["final_answer"] == "10"
    natural = tokens_ok(row) and row.get("finished") is True and row.get("finish_reason") == "stop" and row["completion_token_count"] <= QUALITY_CAPS[task]
    return {"pass": bool(content and natural), "content_pass": bool(content),
            "natural_stop_pass": bool(natural), "tokens": row.get("completion_token_count"), "detail": detail}


def request_checks(row, arm, prompt, cap, performance):
    protocol = row.get("request_protocol") or {}
    diagnostic = row.get("fault_diagnostic") or {}
    metric = measures(row)
    end_caches = (diagnostic.get("end") or {}).get("engram_cache", {})
    policy = policies_ok(row.get("engram_advice"), arm)
    protocol_ok = (tokens_ok(row) and row.get("finished") is True
                   and row.get("completion_token_count", 0) <= cap
                   and protocol.get("max_tokens") == cap and protocol.get("temperature") == 0
                   and type(protocol.get("seed")) is int and protocol["seed"] == 1
                   and protocol.get("ignore_eos") is performance
                   and (row.get("prompt_token_count"), protocol.get("prompt_tokens_sha256")) == PROMPTS[prompt]
                   and row.get("finish_reason") == ("length" if performance else "stop"))
    if performance:
        protocol_ok = protocol_ok and row.get("completion_token_count") == cap
    metrics_ok = finite(metric["wall_s"], True) and finite(metric["ttft_s"])
    if performance or row.get("completion_token_count", 0) > 1:
        metrics_ok = (metrics_ok and finite(metric["decode_tps"], True) and finite(metric["decode_span_s"], True)
                      and math.isclose(metric["decode_tps"], (row.get("completion_token_count", 0) - 1) / metric["decode_span_s"], rel_tol=1e-9))
    telemetry_ok = (count(metric["physical_read_bytes"]) and count(metric["observer_overhead_ns"])
                    and diagnostic.get("observed_tokens") == row.get("completion_token_count")
                    and diagnostic.get("last_token_snapshot_available") is True
                    and len(end_caches) == 2 and sorted(item.get("base", "") for item in end_caches.values()) == BASES
                    and all(item.get("max_rows") == 65536 for item in end_caches.values()))
    return {"protocol": bool(protocol_ok), "policy": bool(policy), "metrics": bool(metrics_ok),
            "telemetry_and_cache_capacity": bool(telemetry_ok),
            "runtime_switches": all((row.get("runtime_switches") or {}).get(key) == value for key, value in SWITCHES.items()),
            "direct_observer_overhead_at_most_1pct": finite(metric["observer_overhead_fraction_of_client_wall"])
            and metric["observer_overhead_fraction_of_client_wall"] <= 0.01}


def performance_result(rows):
    groups = {}
    for arm in ("A", "B"):
        groups[arm] = {}
        for phase in PHASES:
            samples = [measures(rows.get(f"engram-{arm}{pair}-{phase}", {})) for pair in (1, 2, 3)]
            groups[arm][phase] = {key: stats([sample[key] for sample in samples]) for key in samples[0]}
    q_a, q_b = groups["A"]["Qnew"], groups["B"]["Qnew"]
    a_wall, b_wall = q_a["wall_s"]["mean"], q_b["wall_s"]["mean"]
    reduction = (1 - b_wall / a_wall) * 100 if finite(a_wall, True) and finite(b_wall, True) else None
    pairs = [{"pair": pair, "a_wall_s": a, "b_wall_s": b,
              "b_lt_a": finite(a, True) and finite(b, True) and b < a}
             for pair, a, b in zip((1, 2, 3), q_a["wall_s"]["values"], q_b["wall_s"]["values"])]
    a_read, b_read = q_a["physical_read_bytes"]["median"], q_b["physical_read_bytes"]["median"]
    reads_pass = finite(a_read) and finite(b_read) and b_read <= 0.5 * a_read
    warm = {}
    for phase in ("Prepeat", "Qrepeat"):
        a, b = (groups[arm][phase]["decode_tps"]["median"] for arm in ("A", "B"))
        warm[phase] = {"a_median_tps": a, "b_median_tps": b,
                       "b_over_a": b / a if finite(a, True) and finite(b, True) else None,
                       "pass": finite(a, True) and finite(b, True) and b >= 0.98 * a}
    gates = {"qnew_all_pairs_b_faster": all(pair["b_lt_a"] for pair in pairs),
             "qnew_mean_wall_reduction_at_least_5pct": reduction is not None and reduction >= 5,
             "qnew_median_read_bytes_at_most_half": reads_pass,
             "warm_prepeat_and_qrepeat_median_tps_at_least_98pct": all(item["pass"] for item in warm.values())}
    return {"pass": all(gates.values()), "gates": gates, "groups": groups, "qnew_pairs": pairs,
            "qnew_mean_wall_reduction_pct": reduction,
            "qnew_median_read_bytes_b_over_a": b_read / a_read if finite(a_read, True) and finite(b_read) else None,
            "warm": warm, "Pfirst_role": "descriptive only, no Pfirst promotion threshold"}


def _validate(doc0, doc1, reference=None):
    docs = [doc0, doc1]
    rows = [{row.get("label"): row for row in doc.get("results", [])} for doc in docs]
    header = all(doc.get("rank") == rank and doc.get("world_size") == 2
                 and doc.get("status") == "ENGRAM_ADVICE_AB_COMPLETE" for rank, doc in enumerate(docs))
    epoch = isinstance(doc0.get("epoch"), str) and bool(doc0["epoch"]) and doc0["epoch"] == doc1.get("epoch")
    identities = all((doc.get("artifact_identity") or {}).get("status") == "PASS"
                     and (doc.get("artifact_identity") or {}).get("densefix_manifest_sha256") == DENSEFIX_MANIFEST_SHA
                     and (doc.get("native_hip_identity") or {}).get("status") == "PASS"
                     and (doc.get("native_hip_identity") or {}).get("sha256") == NATIVE_SHA for doc in docs)
    order = all(doc.get("order") == ORDER and [row.get("label") for row in doc.get("results", [])] == LABELS for doc in docs)
    rank_identity = {label: same_tokens([rank.get(label, {}) for rank in rows]) for label in LABELS}
    streams, checks, quality, prep, performance = {}, {}, {}, {}, {}
    validate_code = load_code_validator()
    for rank, (document, rank_rows) in enumerate(zip(docs, rows)):
        key = f"rank{rank}"
        p_rows = [rank_rows.get(f"engram-{arm}-{phase}", {}) for arm in ORDER for phase in ("Pfirst", "Prepeat")]
        q_rows = [rank_rows.get(f"engram-{arm}-{phase}", {}) for arm in ORDER for phase in ("Qnew", "Qrepeat")]
        warmups = [rank_rows.get(f"engram-{arm}-warmup-excluded", {}) for arm in ORDER]
        streams[key] = {"all_p_streams_equal": same_tokens(p_rows), "all_q_streams_equal": same_tokens(q_rows),
                        "warmups_equal_p_prefix": tokens_ok(p_rows[0]) and all(tokens_ok(row) and row["token_ids"] == p_rows[0]["token_ids"][:32] for row in warmups),
                        "quality_ab_equal": {task: same_tokens([rank_rows.get(f"engram-quality-{arm}-{task}", {}) for arm in ("A", "B")]) for task in QUALITY}}
        checks[key] = {label: request_checks(rank_rows.get(label, {}), arm, prompt, cap, perf)
                       for label, arm, prompt, cap, perf in REQUESTS}
        quality[key] = {f"{arm}-{task}": quality_result(rank_rows.get(f"engram-quality-{arm}-{task}", {}), task, validate_code)
                        for arm in ("A", "B") for task in QUALITY}
        prep[key] = preparations(document)
        performance[key] = performance_result(rank_rows)
    reference_result = {"provided": reference is not None, "pass": reference is None}
    if reference is not None:
        ref_rows = {row.get("label"): row for row in reference.get("results", [])}
        reference_result["checks"] = {
            group: same_tokens([ref_rows.get(label, {}) for label in labels]
                               + [rank_rows.get(f"engram-{arm}-{phase}", {}) for rank_rows in rows for arm in ORDER for phase in phases])
            for group, labels, phases in (("P", ["fault-P1", "fault-P2", "fault-P3"], ["Pfirst", "Prepeat"]),
                                          ("Q", ["fault-Q1", "fault-Q2"], ["Qnew", "Qrepeat"]))}
        reference_result["pass"] = all(reference_result["checks"].values())
        reference_result["descriptive_metrics"] = {label: measures(ref_rows.get(label, {})) for label in ("fault-P1", "fault-Q1")}
    gates = {"headers": header, "same_nonempty_epoch": epoch, "qualified_artifact_and_native_library": identities,
             "order_and_all_38_requests": order,
             "all_rank_token_streams_identical": all(rank_identity.values()),
             "all_p_q_repeats_and_ab_streams_identical": all(item["all_p_streams_equal"] and item["all_q_streams_equal"] and item["warmups_equal_p_prefix"] for item in streams.values()),
             "all_quality_ab_tokens_identical": all(all(item["quality_ab_equal"].values()) for item in streams.values()),
             "all_request_protocol_policy_metrics_and_telemetry": all(all(item.values()) for rank in checks.values() for item in rank.values()),
             "both_arm_quality": all(task["pass"] for rank in quality.values() for task in rank.values()),
             "matched_preparation": all(item["pass"] for item in prep.values()),
             "optional_reference_tokens": reference_result["pass"]}
    correctness = all(gates.values())
    speed = all(item["pass"] for item in performance.values())
    return {"schema": SCHEMA, "status": "PASS" if correctness and speed else "FAIL",
            "correctness_status": "PASS" if correctness else "FAIL", "speed_status": "PASS" if speed else "FAIL",
            "promotion_eligible": correctness and speed, "correctness_gates": gates,
            "frozen_thresholds": {"qnew_wall_mean_reduction_pct_min": 5, "qnew_read_bytes_b_over_a_max": 0.5,
                                  "warm_tps_b_over_a_min": 0.98, "direct_observer_overhead_fraction_max": 0.01,
                                  "all_qnew_pairs_b_lt_a_each_rank": True},
            "rank_token_identity": rank_identity, "stream_identity": streams, "request_checks": checks,
            "request_metrics": {f"rank{rank}": {label: measures(rank_rows.get(label, {})) for label in LABELS}
                                for rank, rank_rows in enumerate(rows)},
            "preparation": prep, "quality": quality, "performance": performance, "reference": reference_result,
            "http_metrics": None, "notes": [
                "Offline timing only; no HTTP metrics or isolated prefill TPS.",
                "Physical read_bytes are whole-request process I/O counters, including observation overhead; no fault-count conversion or causal attribution.",
                "Observer overhead is reported, not subtracted. Its numerator also includes boundary work outside client wall.",
                "Qnew gates are evaluated independently on both ranks. Warm Prepeat and Qrepeat each have a separate median TPS gate.",
                "Fresh mappings, scoped cache discard and decoded-row clearing are experiment preparation, not the production candidate.",
            ]}


def validate(doc0, doc1, reference=None):
    try:
        return _validate(doc0, doc1, reference)
    except Exception as exc:
        return {"schema": SCHEMA, "status": "FAIL", "correctness_status": "FAIL",
                "speed_status": "NOT_EVALUATED", "promotion_eligible": False,
                "validation_error": f"{type(exc).__name__}: {exc}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank0", required=True, type=Path)
    parser.add_argument("--rank1", required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    inputs = [args.rank0, args.rank1] + ([args.reference] if args.reference else [])
    if args.output.resolve() in {path.resolve() for path in inputs}:
        parser.error("output must not overwrite a raw input")
    try:
        def reject_nonfinite(value):
            raise ValueError(f"non-finite JSON number: {value}")
        documents = [json.loads(path.read_text(), parse_constant=reject_nonfinite) for path in inputs]
        result = validate(*documents)
    except (OSError, ValueError) as exc:
        result = {"schema": SCHEMA, "status": "FAIL", "promotion_eligible": False,
                  "validation_error": f"{type(exc).__name__}: {exc}"}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result.get(key) for key in ("status", "correctness_status", "speed_status", "promotion_eligible", "validation_error")}, sort_keys=True))
    return 0 if result["promotion_eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
