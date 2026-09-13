#!/usr/bin/env python3
"""Summarize two offline fault-diagnostic JSONs without loading model code."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ORDER = ["warmup-excluded", "P1", "P2", "Q1", "Q2", "P3", "P-clean-control"]
CACHE_COUNTERS = ("hits", "misses", "rows_read")
COUNTER_GROUPS = (
    "rusage_self", "rusage_children", "process_io", "global_vmstat",
    "status_kib_except_Threads", "meminfo_kib",
)


def number(value):
    return type(value) in (int, float)


def delta(before, after):
    """Missing observations remain missing, never assumed to be zero."""
    return {key: after[key] - before[key] for key in sorted(before.keys() & after.keys())
            if number(before[key]) and number(after[key])}


def token_info(row):
    ids = row.get("token_ids")
    valid = isinstance(ids, list) and all(type(token) is int for token in ids)
    return {
        "recorded_count": row.get("completion_token_count"),
        "actual_count": len(ids) if valid else None,
        "count_matches": row.get("completion_token_count") == len(ids) if valid else None,
        "sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        if valid else None,
    }


def equality(rows, field):
    values = [row.get(field) if row is not None else None for row in rows]
    if not values or any(value is None for value in values):
        return None
    return all(value == values[0] for value in values[1:])


def cache_delta(before, after):
    old, new = before.get("engram_cache", {}), after.get("engram_cache", {})
    rows = []
    for key in sorted(old.keys() | new.keys()):
        first, last = old.get(key), new.get(key)
        meta = last or first
        rows.append({
            "cache_id": key, "path": meta.get("path"), "base": meta.get("base"),
            "thread_ids": meta.get("thread_ids"), "max_rows": meta.get("max_rows"),
            "both_endpoints_observed": first is not None and last is not None,
            "delta": delta({k: first[k] for k in CACHE_COUNTERS if k in first},
                           {k: last[k] for k in CACHE_COUNTERS if k in last})
            if first is not None and last is not None else None,
        })
    return rows


def interval(before, after):
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None
    start, end = before.get("monotonic_ns"), after.get("monotonic_ns")
    result = {
        "wall_ns": end - start if number(start) and number(end) else None,
        "deltas": {group: delta(before.get(group, {}), after.get(group, {}))
                   for group in COUNTER_GROUPS},
        "engram_cache_delta": cache_delta(before, after),
        "missing_counter_keys": {
            group: sorted(before.get(group, {}).keys() ^ after.get(group, {}).keys())
            for group in COUNTER_GROUPS
            if before.get(group, {}).keys() ^ after.get(group, {}).keys()
        },
    }
    result["process_major_faults"] = result["deltas"]["rusage_self"].get("ru_majflt")
    result["process_read_bytes"] = result["deltas"]["process_io"].get("read_bytes")
    return result


def reader_summary(buckets):
    groups = {}
    for item in buckets:
        key = (item["phase"], item["path"])
        group = groups.setdefault(key, {
            "phase": key[0], "path": key[1], "bases": set(), "thread_ids": set(),
            "calls": 0, "requested_rows": 0, "logical_source_bytes": 0,
            "wall_ns": 0, "rusage_thread": {},
        })
        group["bases"].add(item["base"])
        group["thread_ids"].add(item["tid"])
        for field in ("calls", "requested_rows", "logical_source_bytes", "wall_ns"):
            group[field] += item[field]
        for field, value in item["rusage_thread"].items():
            group["rusage_thread"][field] = group["rusage_thread"].get(field, 0) + value
    result = []
    for key in sorted(groups):
        group = groups[key]
        group["bases"], group["thread_ids"] = sorted(group["bases"]), sorted(group["thread_ids"])
        usage = group["rusage_thread"]
        group["thread_cpu_s"] = (usage["ru_utime"] + usage["ru_stime"]
                                 if "ru_utime" in usage and "ru_stime" in usage else None)
        result.append(group)
    return result


def snapshot_state(snapshot):
    if not isinstance(snapshot, dict):
        return None
    return {key: snapshot[key] for key in (
        "monotonic_ns", "pid", "ppid", "tid", "rank", "observed_live_child_pids",
        "status_kib_except_Threads", "meminfo_kib", "global_vmstat", "read_errors",
        "snapshot_overhead_ns",
    ) if key in snapshot}


def request_summary(row):
    diagnostic = row.get("fault_diagnostic")
    wall = row.get("client", {}).get("wall_s")
    result = {
        "label": row["label"], "prompt_tokens": row.get("prompt_token_count"),
        "prompt_sha256": row.get("prompt_sha256"), "tokens": token_info(row),
        "recorded_request_protocol": row.get("request_protocol"),
        "finished": row.get("finished"), "finish_reason": row.get("finish_reason"),
        "stop_reason": row.get("stop_reason"), "runtime_switches": row.get("runtime_switches"),
        "wall_s": wall, "ttft_s": row.get("client", {}).get("ttft_s"),
        "decode_span_s": row.get("derived", {}).get("decode_span_s"),
        "decode_tps": row.get("derived", {}).get("decode_tps_first_to_last"),
        "engine_core_token_timestamps": {key: (row.get("metrics") or {}).get(key)
                                         for key in ("first_token_ts", "last_token_ts")},
        "diagnostic": None,
    }
    if isinstance(diagnostic, dict):
        overhead = diagnostic.get("instrumentation_overhead_ns")
        states = {key: diagnostic.get(key) for key in ("start", "first_token", "last_token", "end")}
        result["diagnostic"] = {
            "observed_tokens": diagnostic.get("observed_tokens"),
            "observed_count_matches": diagnostic.get("observed_tokens") == result["tokens"]["actual_count"],
            "token_batches": diagnostic.get("token_batches"),
            "last_token_snapshot_available": diagnostic.get("last_token_snapshot_available"),
            "observer_intervals": {
                "before_first": interval(states["start"], states["first_token"]),
                "first_to_last": interval(states["first_token"], states["last_token"]),
                "whole_request": interval(states["start"], states["end"]),
                "after_last": interval(states["last_token"], states["end"]),
            },
            "reader_by_phase_path": reader_summary(diagnostic.get("reader", [])),
            "snapshot_states": {key: snapshot_state(value) for key, value in states.items()},
            "instrumentation_overhead_ns": overhead,
            "direct_overhead_fraction_of_client_wall": overhead / (wall * 1_000_000_000)
            if number(overhead) and number(wall) and wall > 0 else None,
            "caveats": diagnostic.get("caveats", []),
        }
    return result


def indexed(document):
    result = {}
    for row in document.get("results", []):
        if row["label"] in result:
            raise ValueError(f"duplicate request label: {row['label']}")
        result[row["label"]] = row
    return result


def summarize(rank0, rank1):
    documents = [rank0, rank1]
    by_rank = [indexed(document) for document in documents]
    expected = ["fault-" + label for label in ORDER]
    labels = expected + sorted((by_rank[0].keys() | by_rank[1].keys()) - set(expected))
    result = {
        "schema": "ds41-fault-summary-v1",
        "rank_ids_match_inputs": [document.get("rank") == rank for rank, document in enumerate(documents)],
        "same_attempt": equality(documents, "attempt"), "same_epoch": equality(documents, "epoch"),
        "rank_request_equality": [], "ranks": [],
        "interpretation": [
            "Observer intervals include instrumentation and output processing; they are not exact engine-core timing intervals.",
            "Direct overhead fraction uses client wall as denominator, although its numerator also includes boundary collection outside client wall.",
            "Reader wall sums may overlap across threads and include CPU work; no causal stall attribution is performed.",
            "Logical source bytes, process read_bytes and major-fault counts are distinct observations; no count-to-byte conversion is performed.",
            "Memory and vmstat are host or process snapshots; global changes can involve other processes; child rusage excludes live children.",
            "Absent request protocol fields and missing counter endpoints remain unverified or null.",
        ],
        "source_defined_protocol_not_independently_recorded_in_rows": {
            "runtime_commit": "1ebb0e2", "temperature": 0.0, "seed": 1, "ignore_eos": True,
            "warmup_max_tokens": 32, "other_max_tokens": 128,
        },
    }
    for label in labels:
        rows = [rank.get(label) for rank in by_rank]
        result["rank_request_equality"].append({
            "label": label, "present": [row is not None for row in rows],
            **{field + "_equal": equality(rows, field) for field in (
                "token_ids", "prompt_sha256", "prompt_token_count", "completion_token_count",
                "finished", "finish_reason", "request_protocol", "runtime_switches",
            )},
        })
    for document, rows in zip(documents, by_rank):
        load = document.get("fault_load") or {}
        repeats = {}
        for group, suffixes in (("P", ["P1", "P2", "P3"]), ("Q", ["Q1", "Q2"]),
                                ("P_clean_control", ["P3", "P-clean-control"])):
            selected = [rows.get("fault-" + suffix) for suffix in suffixes]
            repeats[group] = {
                "labels": suffixes, "token_ids_equal": equality(selected, "token_ids"),
                "prompt_sha256_equal": equality(selected, "prompt_sha256"),
            }
        result["ranks"].append({
            "rank": document.get("rank"), "status": document.get("status"),
            "attempt": document.get("attempt"), "epoch": document.get("epoch"),
            "missing_requests": [label for label in expected if label not in rows],
            "actual_order": list(rows), "repeats": repeats,
            "init": {"wall_s": document.get("init_s"), "snapshots_and_types": load,
                     "observer_interval": interval(load.get("before_llm"), load.get("after_llm"))},
            "map_snapshots": document.get("map_snapshots", []),
            "requests": [request_summary(rows[label]) for label in labels if label in rows],
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rank0", type=Path)
    parser.add_argument("rank1", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.resolve() in {args.rank0.resolve(), args.rank1.resolve()}:
        parser.error("output must not overwrite either raw input")
    try:
        documents = [json.loads(path.read_text()) for path in (args.rank0, args.rank1)]
        rendered = json.dumps(summarize(*documents), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        args.output.write_text(rendered)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
