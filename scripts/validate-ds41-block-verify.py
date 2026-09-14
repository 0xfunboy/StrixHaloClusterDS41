#!/usr/bin/env python3
"""Independent CPU validation of attempt027 target-only block verification.

No model is imported. Tensor files are loaded on CPU with weights_only=True.
The logits tolerances are frozen diagnostic gates, not a general model bound.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


REL_L2_MAX = 0.005
MAX_ABS = 0.125
ACTIVATION_OUTPUT_TOKENS = 8
LABELS = {1: "diagnostic-D1", 2: "diagnostic-B2", 4: "diagnostic-B4"}
CORRUPT_LABELS = {0: "diagnostic-B4-corrupt-first", 2: "diagnostic-B4-corrupt-last"}


def _ints(value):
    return isinstance(value, list) and all(type(x) is int and x >= 0 for x in value)


def _default_loader(request):
    import torch
    return torch.load(request["logits_file"], map_location="cpu", weights_only=True)


def _vector(value):
    if hasattr(value, "detach"):
        if str(value.device) != "cpu" or str(value.dtype) != "torch.float32":
            raise ValueError("captured logits must be CPU float32")
        if value.ndim != 1:
            raise ValueError("logit row must be one-dimensional")
        return value.double()
    if not isinstance(value, (list, tuple)):
        raise ValueError("logit row must be a vector")
    return [float(x) for x in value]


def _finite(value):
    if hasattr(value, "isfinite"):
        return bool(value.isfinite().all().item())
    return all(math.isfinite(x) for x in value)


def _argmax(value):
    if hasattr(value, "argmax"):
        return int(value.argmax().item())
    return max(range(len(value)), key=value.__getitem__)


def _errors(reference, candidate):
    if hasattr(reference, "square"):
        diff = candidate - reference
        numerator = float(diff.square().sum().sqrt().item())
        denominator = float(reference.square().sum().sqrt().item())
        max_abs = float(diff.abs().max().item())
    else:
        diffs = [x - y for x, y in zip(candidate, reference)]
        numerator = math.sqrt(math.fsum(x * x for x in diffs))
        denominator = math.sqrt(math.fsum(x * x for x in reference))
        max_abs = max(map(abs, diffs))
    # A zero reference only passes for an exactly zero candidate.
    rel_l2 = numerator / denominator if denominator else (0.0 if not numerator else None)
    return {"rel_l2": rel_l2, "max_abs": max_abs,
            "reference_top1": _argmax(reference), "candidate_top1": _argmax(candidate)}


def _read_request(request, prompt, oracle, vocab_size, loader):
    """Index only rows with a proven common committed and packet prefix."""
    errors, rows, corruptions, packet_widths = [], {}, [], []
    label = request.get("label", "<missing>")
    output = request.get("output", {})
    if output.get("token_ids") != oracle:
        errors.append(f"{label}: output token stream differs from oracle")
    if output.get("completion_token_count") != len(oracle):
        errors.append(f"{label}: completion count differs from oracle length")
    if output.get("finish_reason") != "length":
        errors.append(f"{label}: expected length finish")
    sequence = prompt + oracle
    last_position = len(sequence) - 2
    start_position = len(prompt) + ACTIVATION_OUTPUT_TOKENS - 1
    try:
        records = loader(request)
        if not isinstance(records, list) or not records:
            raise ValueError("missing captured tensor records")
        for record_index, record in enumerate(records):
            positions = record.get("positions")
            inputs = record.get("input_token_ids")
            prefix = record.get("output_prefix")
            if not _ints(positions) or not positions or not _ints(inputs) or len(inputs) != len(positions):
                raise ValueError(f"record {record_index}: invalid input positions/tokens")
            if positions != list(range(positions[0], positions[0] + len(positions))):
                raise ValueError(f"record {record_index}: nonconsecutive positions")
            if not _ints(prefix) or prefix != oracle[:len(prefix)] or len(prefix) > len(oracle):
                raise ValueError(f"record {record_index}: committed prefix not oracle prefix")
            if prefix and positions[0] != len(prompt) + len(prefix) - 1:
                raise ValueError(f"record {record_index}: anchor is not last committed output token")
            logits = record["logits"]
            # Prefill commonly retains only the final input's logit row.
            if len(logits) == 1 and not prefix and positions[0] == 0:
                logit_positions, logit_inputs = positions[-1:], inputs[-1:]
            elif len(logits) == len(positions):
                logit_positions, logit_inputs = positions, inputs
            else:
                raise ValueError(f"record {record_index}: logits/input row count mismatch")
            mismatches = [i for i, (p, token) in enumerate(zip(positions, inputs))
                          if p < len(sequence) and token != sequence[p]]
            first_mismatch = positions[mismatches[0]] if mismatches else None
            if mismatches:
                corruptions.append({"record": record_index, "positions": positions,
                                    "indices": mismatches, "position": first_mismatch,
                                    "input_token_ids": inputs, "output_prefix": prefix})
            if positions[0] >= start_position:
                packet_widths.append(len(positions))
            for p, token, raw in zip(logit_positions, logit_inputs, logits):
                vector = _vector(raw)
                if len(vector) != vocab_size or not _finite(vector):
                    raise ValueError(f"record {record_index}, position {p}: nonfinite or wrong vocabulary width")
                if not (start_position <= p <= last_position):
                    continue
                if first_mismatch is not None and p >= first_mismatch:
                    continue
                if token != sequence[p]:
                    raise ValueError(f"record {record_index}, position {p}: nonoracle input")
                if p in rows:
                    raise ValueError(f"position {p}: duplicate common-prefix logit row")
                rows[p] = vector
    except Exception as exc:
        errors.append(f"{label}: {type(exc).__name__}: {exc}")
    return {"errors": errors, "rows": rows, "corruptions": corruptions,
            "packet_widths": packet_widths}


def _compare(reference, candidate):
    errors, comparisons = list(candidate["errors"]), []
    for position, row in sorted(candidate["rows"].items()):
        if position not in reference["rows"]:
            errors.append(f"position {position}: missing baseline row")
            continue
        result = {"position": position, **_errors(reference["rows"][position], row)}
        result["passed"] = (result["rel_l2"] is not None and
                            result["rel_l2"] <= REL_L2_MAX and result["max_abs"] <= MAX_ABS and
                            result["reference_top1"] == result["candidate_top1"])
        comparisons.append(result)
        if not result["passed"]:
            errors.append(f"position {position}: logits gate failed")
    if not comparisons:
        errors.append("no common-prefix positions compared")
    return {"passed": not errors, "errors": errors, "compared_positions": len(comparisons),
            "max_rel_l2": max((x["rel_l2"] for x in comparisons if x["rel_l2"] is not None), default=None),
            "max_abs": max((x["max_abs"] for x in comparisons), default=None),
            "comparisons": comparisons}


def _step_history(request, prompt, oracle, *, rowwise_native_control=False,
                  rowwise_mhc_control=False):
    errors, selected, emitted = [], [], []
    label = request.get("label", "<missing>")
    cap = 32 if request.get("mode") == "measure" else 64
    width = request.get("desired_k", -1) + 1
    protocol = request.get("sampling", {})
    if protocol != {"temperature": 0, "seed": 1, "max_tokens": cap, "ignore_eos": True}:
        errors.append(f"{label}: sampling/cap protocol differs")
    steps = request.get("steps", [])
    if not isinstance(steps, list) or not steps:
        return {"errors": errors + [f"{label}: no step history"], "selected_steps": []}
    for index, step in enumerate(steps):
        prefix, tokens = step.get("output_prefix"), step.get("emitted_tokens")
        if prefix != emitted or not _ints(tokens):
            errors.append(f"{label} step {index}: committed/emitted history mismatch")
            continue
        count = step.get("num_positions")
        positions = step.get("positions")
        if count is None and not tokens:
            continue  # Explicit empty engine maintenance call, never a sample.
        if type(count) is not int or count < 1 or not _ints(positions) or len(positions) != count:
            errors.append(f"{label} step {index}: invalid scheduled positions")
            continue
        if positions != list(range(positions[0], positions[0] + count)):
            errors.append(f"{label} step {index}: nonconsecutive scheduled positions")
        if len(tokens) > count:
            errors.append(f"{label} step {index}: emitted more than B new tokens (anchor double count)")
        if prefix and positions[0] != len(prompt) + len(prefix) - 1:
            errors.append(f"{label} step {index}: wrong anchor position")
        if step.get("scheduler_computed_before") != positions[0]:
            errors.append(f"{label} step {index}: scheduler start differs from actual position")
        drafts = step.get("num_drafts")
        if step.get("decode") is True and (type(drafts) is not int or drafts != count - 1):
            errors.append(f"{label} step {index}: decode width must equal K+1")
        computed_after = step.get("scheduler_computed_after")
        if computed_after is not None:
            rejected = max(0, drafts - max(len(tokens) - 1, 0)) if type(drafts) is int else 0
            if computed_after != positions[-1] + 1 - rejected:
                errors.append(f"{label} step {index}: scheduler rejection rollback mismatch")
        emitted.extend(tokens)
        if emitted != oracle[:len(emitted)] or len(emitted) > cap:
            errors.append(f"{label} step {index}: emitted stream/cap differs from oracle")
        wall = step.get("wall_s")
        if not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall <= 0:
            errors.append(f"{label} step {index}: invalid step wall")
            continue
        if (step.get("decode") is True and positions[0] >= len(prompt) + 7 and count == width
                and len(tokens) == width and len(prefix) + width <= cap):
            dispatch = step.get("dispatch", {})
            dispatch_contract = (
                ("native", "native_calls", "triton_fallback_calls", bool(rowwise_native_control)),
                ("coefficient", "fused_calls", "fallback_calls", bool(rowwise_mhc_control)),
                ("projection", "tilelang_calls", "fallback_calls", bool(rowwise_mhc_control)),
            )
            for section, active, fallback, rowwise_active in dispatch_contract:
                counters = dispatch.get(section, {})
                fallback_count = counters.get(fallback, 0)
                fallback_reasons = counters.get("fallback_reasons", {})
                # M1 always uses the promoted path.  For M>1 the frozen
                # diagnostic controls explicitly decide whether the qualified
                # M1 primitive is reused rowwise or whether the historical
                # tokens_not_1 fallback remains required.  This is a dispatch
                # contract only; numerical/logit gates are unchanged.
                expected_active = width == 1 or rowwise_active
                if expected_active:
                    nonzero_fallback_reasons = {
                        key: value for key, value in fallback_reasons.items()
                        if isinstance(value, int) and value != 0
                    }
                    if (counters.get(active, 0) <= 0 or fallback_count != 0 or
                            nonzero_fallback_reasons):
                        errors.append(f"{label} step {index}: {section} active dispatch mismatch")
                elif (counters.get(active, 0) != 0 or fallback_count <= 0 or
                      fallback_reasons.get("tokens_not_1", 0) != fallback_count):
                    errors.append(f"{label} step {index}: {section} block fallback mismatch")
            selected.append({"step": index, "position": positions[0], "width": count,
                             "wall_s": wall, "actual_new_tokens": len(tokens),
                             "actual_new_tokens_per_s": len(tokens) / wall,
                             "favorable_B_per_s": width / wall,
                             "allocated_bytes": step.get("allocated_bytes"),
                             "reserved_bytes": step.get("reserved_bytes")})
    if emitted != request.get("output", {}).get("token_ids") or emitted != oracle[:cap]:
        errors.append(f"{label}: complete step output stream differs from returned/capped oracle output")
    events = request.get("bridge", {}).get("events", [])
    if not events or any(e.get("prefix_errors") or e.get("fidelity_failed") for e in events):
        errors.append(f"{label}: replay bridge has missing events or fidelity errors")
    if not selected:
        errors.append(f"{label}: no eligible full-width decode steps")
    return {"errors": errors, "selected_steps": selected}


def validate_diagnostics(requests, prompt_token_ids, oracle_token_ids, *, vocab_size=None,
                         logits_loader=None, require_reject_controls=True,
                         rowwise_native_control=False, rowwise_mhc_control=False):
    """Return per-width eligibility. A failing B2/B4 never grants eligibility.

    Oracle contains completion tokens only. logits_loader(request) returns the
    captured CPU records and can be supplied by the in-process supervisor.
    """
    result = {"passed": False, "errors": [], "by_width": {}, "reject_controls": {},
              "thresholds": {"rel_l2_max": REL_L2_MAX, "max_abs": MAX_ABS,
                             "exact_top1": True, "finite_full_vocabulary": True}}
    if not _ints(prompt_token_ids) or len(prompt_token_ids) != 35 or not _ints(oracle_token_ids) or len(oracle_token_ids) != 64:
        result["errors"].append("expected exactly 35 prompt and 64 oracle completion tokens")
        return result
    by_label = {request.get("label"): request for request in requests}
    if len(by_label) != len(requests):
        result["errors"].append("duplicate request labels")
    loader = logits_loader or _default_loader
    # Infer only the full stored width when the caller did not provide config.
    if vocab_size is None:
        try:
            vocab_size = len(loader(by_label[LABELS[1]])[0]["logits"][0])
        except Exception as exc:
            result["errors"].append(f"cannot determine vocabulary size: {type(exc).__name__}: {exc}")
            return result
    if type(vocab_size) is not int or vocab_size <= max(prompt_token_ids + oracle_token_ids):
        result["errors"].append("invalid real vocabulary size")
        return result
    parsed = {}
    expected_positions = set(range(len(prompt_token_ids) + ACTIVATION_OUTPUT_TOKENS - 1,
                                   len(prompt_token_ids) + len(oracle_token_ids) - 1))
    for width, label in LABELS.items():
        request = by_label.get(label)
        if request is None:
            parsed[width] = {"errors": [f"missing {label}"], "rows": {}, "corruptions": [], "packet_widths": []}
            continue
        parsed[width] = _read_request(request, prompt_token_ids, oracle_token_ids, vocab_size, loader)
        parsed[width]["errors"].extend(_step_history(
            request, prompt_token_ids, oracle_token_ids,
            rowwise_native_control=rowwise_native_control,
            rowwise_mhc_control=rowwise_mhc_control,
        )["errors"])
        if request.get("desired_k") != width - 1 or request.get("mode") != "diagnostic" or request.get("corrupt_draft_index") is not None:
            parsed[width]["errors"].append(f"{label}: wrong diagnostic request protocol")
        if parsed[width]["corruptions"]:
            parsed[width]["errors"].append(f"{label}: unexpected nonoracle input")
        if set(parsed[width]["rows"]) != expected_positions:
            parsed[width]["errors"].append(f"{label}: incomplete eligible-position coverage")
        if width not in parsed[width]["packet_widths"]:
            parsed[width]["errors"].append(f"{label}: no width-{width} verification packet")
    for width in LABELS:
        result["by_width"][str(width)] = _compare(parsed[1], parsed[width])
        if width != 1 and parsed[1]["errors"]:
            result["by_width"][str(width)]["errors"].append("baseline structural/output gate failed")
            result["by_width"][str(width)]["passed"] = False
    if require_reject_controls:
        for index, label in CORRUPT_LABELS.items():
            request = by_label.get(label)
            if request is None:
                result["reject_controls"][label] = {"passed": False, "errors": ["missing rejection control"]}
                continue
            control = _read_request(request, prompt_token_ids, oracle_token_ids, vocab_size, loader)
            control["errors"].extend(_step_history(
                request, prompt_token_ids, oracle_token_ids,
                rowwise_native_control=rowwise_native_control,
                rowwise_mhc_control=rowwise_mhc_control,
            )["errors"])
            if set(control["rows"]) != expected_positions:
                control["errors"].append("rejection control has incomplete common-prefix position coverage after recovery")
            check = _compare(parsed[1], control)
            if request.get("desired_k") != 3 or request.get("corrupt_draft_index") != index or request.get("mode") != "diagnostic":
                check["errors"].append("wrong rejection control protocol")
            corruption = control["corruptions"]
            if len(corruption) != 1 or corruption[0]["indices"] != [index + 1] or len(corruption[0]["positions"]) != 4:
                check["errors"].append("expected exactly one observed width-4 packet with requested changed draft")
            else:
                changed_position = corruption[0]["position"]
                check["corruption"] = corruption[0]
                events = request.get("bridge", {}).get("events", [])
                observed = [e for e in events if e.get("query_start") == corruption[0]["positions"][0]
                            and e.get("query_len") == 4]
                injected = [e for e in events if e.get("injected") is True]
                if len(injected) != 1 or len(observed) != 1 or observed[0].get("num_rejected") != 3 - index or observed[0].get("num_sampled") != index + 1:
                    check["errors"].append("missing real greedy rejection count or one-shot proposal injection")
                prior = [p for p in control["rows"] if corruption[0]["positions"][0] <= p < changed_position]
                continuation = [p for p in control["rows"] if p > changed_position]
                if len(prior) != index + 1 or len(continuation) < 3:
                    check["errors"].append("missing unaffected causal rows or post-rejection continuation")
                # Compare the unchanged causal rows directly with clean B4 too.
                for position in prior:
                    if position not in parsed[4]["rows"]:
                        check["errors"].append(f"missing clean B4 causal position {position}")
                        continue
                    error = _errors(parsed[4]["rows"][position], control["rows"][position])
                    if error["rel_l2"] is None or error["rel_l2"] > REL_L2_MAX or error["max_abs"] > MAX_ABS or error["reference_top1"] != error["candidate_top1"]:
                        check["errors"].append(f"causal clean-B4 comparison failed at {position}")
            check["passed"] = not check["errors"]
            result["reject_controls"][label] = check
    result["passed"] = (not result["errors"] and all(x["passed"] for x in result["by_width"].values()) and
                        all(x["passed"] for x in result["reject_controls"].values()))
    return result


def _stats(values):
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
        raise ValueError("nonfinite or nonnumeric metric")
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "sd": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
            "min": min(values) if values else None, "max": max(values) if values else None}


def validate_documents(documents, *, logits_loaders=None):
    reports, errors, performance = [], [], []
    for rank, doc in enumerate(documents):
        config = doc.get("config", {})
        result = validate_diagnostics(
            doc.get("requests", []), config.get("prompt_token_ids"),
            config.get("oracle_token_ids"), vocab_size=config.get("vocab_size"),
            logits_loader=(logits_loaders[rank] if logits_loaders else None),
            rowwise_native_control=bool(config.get("rowwise_native_control", False)),
            rowwise_mhc_control=bool(config.get("rowwise_mhc_control", False)),
        )
        reports.append(result)
        if doc.get("status") != "COMPLETE" or doc.get("rank") != rank:
            errors.append(f"rank {rank}: not COMPLETE or incorrect rank identity")
        if config.get("activation_output_tokens") != 8:
            errors.append(f"rank {rank}: activation output count must be 8")
        if config.get("sampling", {}).get("temperature") != 0 or config.get("sampling", {}).get("seed") != 1:
            errors.append(f"rank {rank}: expected temperature 0 and seed 1")
        measures = {}
        for request in doc.get("requests", []):
            if request.get("mode") != "measure":
                continue
            label = request.get("label", "<missing>")
            width = request.get("desired_k", -1) + 1
            eligible = result["by_width"].get(str(width), {}).get("passed", False)
            if width == 4:
                eligible = eligible and bool(result["reject_controls"]) and all(
                    c["passed"] for c in result["reject_controls"].values())
            if not eligible:
                errors.append(f"rank {rank} {label}: measurement taken despite failed width/state gate")
            history = _step_history(
                request, config.get("prompt_token_ids", []), config.get("oracle_token_ids", []),
                rowwise_native_control=bool(config.get("rowwise_native_control", False)),
                rowwise_mhc_control=bool(config.get("rowwise_mhc_control", False)),
            )
            errors.extend(f"rank {rank}: {error}" for error in history["errors"])
            selected = history["selected_steps"]
            output = request.get("output", {})
            if output.get("completion_token_count") != 32 or output.get("finish_reason") != "length" or not output.get("finished"):
                errors.append(f"rank {rank} {label}: measured cap/finish mismatch")
            if request.get("logits_file") is not None:
                errors.append(f"rank {rank} {label}: measured request retained diagnostic logits capture")
            wall = sum(s["wall_s"] for s in selected)
            new_tokens = sum(s["actual_new_tokens"] for s in selected)
            events = request.get("bridge", {}).get("events", [])
            measures[label] = {"width": width, "selected_steps": selected,
                               "step_wall_s": _stats([s["wall_s"] for s in selected]),
                               "step_favorable_B_per_s": _stats([s["favorable_B_per_s"] for s in selected]),
                               "selected_new_tokens": new_tokens, "selected_wall_s": wall,
                               "aggregate_actual_new_tokens_per_s": new_tokens / wall if wall else None,
                               "aggregate_favorable_B_per_s": width * len(selected) / wall if wall else None,
                               "whole_request_wall_s": output.get("client", {}).get("wall_s"),
                               "whole_request_ttft_s": output.get("client", {}).get("ttft_s"),
                               "whole_request_decode_span_s": output.get("derived", {}).get("decode_span_s"),
                               "whole_request_decode_tps": output.get("derived", {}).get("decode_tps_first_to_last"),
                               "proposal_batch_wall_s": _stats([e["proposal_batch_wall_s"] for e in events
                                                               if isinstance(e.get("proposal_batch_wall_s"), (int, float))])}
        expected_widths = [1] + [w for w in (2, 4) if result["by_width"].get(str(w), {}).get("passed")
                                 and (w != 4 or (result["reject_controls"] and all(c["passed"] for c in result["reject_controls"].values())))]
        expected_labels = {f"measure-B{w}-{trial}" for w in expected_widths for trial in (1, 2, 3)}
        if set(measures) != expected_labels:
            errors.append(f"rank {rank}: measured labels do not match three trials of eligible widths")
        expected_order = [f"measure-B{w}-{trial}" for trial in (1, 2, 3)
                          for w in (list(reversed(expected_widths)) if trial == 2 else expected_widths)]
        if list(measures) != expected_order:
            errors.append(f"rank {rank}: measured trial order differs from frozen alternation")
        for label, value in measures.items():
            serial = measures.get("measure-B1-" + label.rsplit("-", 1)[-1])
            if serial and value["step_wall_s"]["mean"] and serial["step_wall_s"]["mean"]:
                t1, tb = serial["step_wall_s"]["mean"], value["step_wall_s"]["mean"]
                value["favorable_factor_B_t1_over_tB"] = value["width"] * t1 / tb
                value["remaining_real_drafting_budget_s_per_block"] = value["width"] * t1 - tb
        by_width = {}
        for width in (1, 2, 4):
            runs = [value for value in measures.values() if value["width"] == width]
            by_width[str(width)] = {key: _stats([value[key] for value in runs if value[key] is not None])
                                    for key in ("aggregate_actual_new_tokens_per_s", "aggregate_favorable_B_per_s",
                                                "whole_request_wall_s", "whole_request_ttft_s",
                                                "whole_request_decode_span_s", "whole_request_decode_tps")}
        performance.append({"requests": measures, "by_width": by_width})
    if len(documents) != 2:
        errors.append("exactly two rank documents required")
    else:
        left = {r["label"]: r for r in documents[0].get("requests", [])}
        right = {r["label"]: r for r in documents[1].get("requests", [])}
        if set(left) != set(right):
            errors.append("rank request labels differ")
        for label in sorted(set(left) & set(right)):
            if left[label].get("output", {}).get("token_ids") != right[label].get("output", {}).get("token_ids"):
                errors.append(f"{label}: rank output tokens differ")
        if not documents[0].get("source_commit") or documents[0].get("source_commit") != documents[1].get("source_commit"):
            errors.append("rank source identity absent or differs")
        if documents[0].get("config") != documents[1].get("config"):
            errors.append("rank configurations differ")
    coupled = {}
    if len(performance) == 2:
        for label in sorted(set(performance[0]["requests"]) & set(performance[1]["requests"])):
            left = performance[0]["requests"][label]
            right = performance[1]["requests"][label]
            a = {(s["position"], s["width"]): s for s in left["selected_steps"]}
            b = {(s["position"], s["width"]): s for s in right["selected_steps"]}
            if set(a) != set(b):
                errors.append(f"{label}: rank selected step positions differ")
                continue
            wall = sum(max(a[k]["wall_s"], b[k]["wall_s"]) for k in a)
            new_tokens = sum(a[k]["actual_new_tokens"] for k in a)
            coupled[label] = {"max_rank_step_wall_sum_s": wall, "new_tokens": new_tokens,
                              "favorable_and_actual_new_tokens_per_s": new_tokens / wall if wall else None}
    return {"status": "PASS" if not errors and all(r["passed"] for r in reports) else "FAIL",
            "errors": errors, "ranks": reports,
            "performance": performance, "coupled_rank_max": coupled,
            "accounting": "B = one already emitted anchor input plus K drafts; at most B NEW outputs, never B+1",
            "scope": "offline model-free proposal diagnostic, not DSpark or end-to-end draft throughput",
            "timing_scope": "Only measure-mode complete-width decode packets after 8 outputs; core-step replay and synchronization remain included. No drafter cost measured or subtracted; no HTTP metrics."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rank0", type=Path)
    parser.add_argument("rank1", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        documents = [json.loads(path.read_text()) for path in (args.rank0, args.rank1)]
        report = validate_documents(documents)
    except Exception as exc:
        report = {"status": "FAIL", "errors": [f"{type(exc).__name__}: {exc}"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "errors": report.get("errors", [])}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
