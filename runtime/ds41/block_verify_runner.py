"""One-load, diagnostic-only target verification gate. Never imported by API."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from runtime.ds41 import block_verify_experiment as bridge
from runtime.ds41 import mhc_coeff_sinkhorn, mhc_projection_rms
from runtime.ds41.fault_diagnostics import process_snapshot
from vllm_gguf_plugin.quantization.fused_moe import ds41_native_hip_stats


def counts():
    return {"native": ds41_native_hip_stats(),
            "coefficient": mhc_coeff_sinkhorn.stats(),
            "projection": mhc_projection_rms.stats()}


def delta(after, before):
    return {k: delta(v, before.get(k, {})) if isinstance(v, dict)
            else v - before.get(k, 0)
            for k, v in after.items() if isinstance(v, (dict, int))}


class StepObserver:
    """Complete in-process engine step including post_step and GPU completion.

    Diagnostic tensor copies are never used as qualifying timing samples.
    Measurement copies no logits/input tensors; Python counters and sync are
    retained explicitly. Replay proposal work remains included, not subtracted.
    """

    def __init__(self, llm, vocab_size):
        self.client = llm.llm_engine.engine_core
        self.core = self.client.engine_core
        self.runner = llm.llm_engine.model_executor.driver_worker.model_runner
        if getattr(self.runner, "batch_sharder", None) is not None:
            raise RuntimeError("Diagnostic requires unsharded full-vocabulary sampling")
        self.model = self.runner.model
        self.vocab_size = vocab_size
        self.original_get_output = self.client.get_output
        self.original_sample = self.runner.sample
        self.original_logits = self.model.compute_logits
        self.current_batch = None
        self.current = None
        self.steps, self.logits = [], []
        self.capture = False
        self.client.get_output = self.get_output
        self.runner.sample = self.sample
        self.model.compute_logits = self.compute_logits

    def reset(self, capture):
        self.steps, self.logits = [], []
        self.capture = capture

    def sample(self, hidden_states, input_batch, *args, **kwargs):
        self.current_batch = input_batch
        if self.current is not None:
            if input_batch.num_reqs != 1:
                raise RuntimeError("Gate requires exactly one sequence per step")
            start = int(input_batch.num_computed_tokens_np[0])
            width = int(input_batch.num_tokens)
            self.current.update(num_positions=width, positions=list(range(start, start + width)),
                                num_drafts=int(input_batch.num_draft_tokens),
                                decode=not bool(input_batch.is_prefilling_np[0]))
            if self.capture:
                self.current["positions"] = input_batch.positions.tolist()
                self.current["input_token_ids"] = input_batch.input_ids.tolist()
        return self.original_sample(hidden_states, input_batch, *args, **kwargs)

    def compute_logits(self, hidden_states, *args, **kwargs):
        logits = self.original_logits(hidden_states, *args, **kwargs)
        if self.current is not None:
            self.current["logits_rows"] = int(logits.shape[0])
            if self.capture:
                if logits.ndim != 2 or logits.shape[1] < self.vocab_size:
                    raise RuntimeError("Full gathered real-vocabulary logits are required")
                n = int(logits.shape[0])
                record = {
                    "positions": self.current["positions"][-n:],
                    "input_token_ids": self.current["input_token_ids"][-n:],
                    "output_prefix": list(self.current["output_prefix"]),
                    "logits": logits[:, :self.vocab_size].detach().float().cpu().clone(),
                }
                self.logits.append(record)
        return logits

    def get_output(self):
        requests = list(self.core.scheduler.requests.values())
        if len(requests) > 1:
            raise RuntimeError("Multiple independent requests are out of scope")
        output_prefix = list(requests[0].output_token_ids) if requests else []
        computed_before = int(requests[0].num_computed_tokens) if requests else None
        self.current = {"output_prefix": output_prefix,
                        "scheduler_computed_before": computed_before}
        before = counts()
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = self.original_get_output()
        torch.cuda.synchronize()
        wall = time.perf_counter() - start
        record, self.current = self.current, None
        record.update(wall_s=wall, dispatch=delta(counts(), before),
                      allocated_bytes=torch.cuda.memory_allocated(),
                      reserved_bytes=torch.cuda.memory_reserved(),
                      emitted_tokens=[t for out in outputs.outputs for t in out.new_token_ids])
        remaining = list(self.core.scheduler.requests.values())
        record["scheduler_computed_after"] = int(remaining[0].num_computed_tokens) if remaining else None
        self.steps.append(record)
        return outputs

    def close(self):
        self.client.get_output = self.original_get_output
        self.runner.sample = self.original_sample
        self.model.compute_logits = self.original_logits


def run(llm, run_generation, token_spec, raw, rank, init_s, artifact_identity, emit):
    config = token_spec["block_verify"]
    prompt = token_spec["prompts"]["speed"]["token_ids"]
    oracle = config["oracle_token_ids"]
    if len(oracle) != 64 or config["activation_output_tokens"] != 8:
        raise ValueError("Frozen gate requires 64 oracle tokens and activation8")
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("ds41_block_validator", root / "scripts/validate-ds41-block-verify.py")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    report = {
        "schema": "ds41-small-block-gate-v1", "status": "RUNNING", "rank": rank,
        "source_commit": artifact_identity["runtime_commit"],
        "init_s": init_s, "artifact_identity": artifact_identity,
        "config": {**config, "prompt_token_ids": prompt}, "requests": [],
        "measurement": "full EngineCore step plus post_step and GPU completion; includes diagnostic replay proposal, excludes real drafter (absent)",
        "before_requests": process_snapshot(rank),
    }
    output = raw / f"block-verification-rank{rank}.json"

    def save(status):
        report["status"] = status
        output.write_text(json.dumps(report, indent=2) + "\n")

    observer = StepObserver(llm, config["vocab_size"])

    def request(label, k, mode, cap=64, corrupt=None):
        bridge.configure(prompt + oracle, len(prompt), desired_k=k,
                         activation_output_tokens=8, corrupt_draft_index=corrupt)
        observer.reset(capture=mode == "diagnostic")
        rowwise = bool(config.get("rowwise_native_control", False) and k > 0)
        previous_rowwise = os.environ.get("DS41_NATIVE_HIP_MOE_ROWWISE")
        os.environ["DS41_NATIVE_HIP_MOE_ROWWISE"] = "1" if rowwise else "0"
        try:
            result = run_generation(llm, prompt, label=label, max_tokens=cap, ignore_eos=True)
        finally:
            if previous_rowwise is None:
                os.environ.pop("DS41_NATIVE_HIP_MOE_ROWWISE", None)
            else:
                os.environ["DS41_NATIVE_HIP_MOE_ROWWISE"] = previous_rowwise
        state = bridge.get_state()
        row = {"label": label, "desired_k": k, "mode": mode,
               "sampling": {"temperature": 0, "seed": 1, "max_tokens": cap, "ignore_eos": True},
               "corrupt_draft_index": corrupt, "output": result,
               "steps": observer.steps, "bridge": {
                   "events": state.events, "fidelity_failed": state.fidelity_failed,
                   "failed_requests": sorted(state.failed_requests),
                   "injected_requests": sorted(state.injected_requests),
                   "next_counts": state.next_counts,
               }, "logits_file": None}
        if mode == "diagnostic":
            logits_path = raw / f"{label}-logits-rank{rank}.pt"
            torch.save(observer.logits, logits_path)
            row["logits_file"] = str(logits_path)
        report["requests"].append(row)
        save("RUNNING")
        return row

    try:
        request("warmup-D1-excluded", 0, "warmup")
        baseline = request("diagnostic-D1", 0, "diagnostic")
        baseline_ok = torch.tensor([int(baseline["output"]["token_ids"] == oracle)], device="cuda", dtype=torch.int32)
        dist.all_reduce(baseline_ok, op=dist.ReduceOp.MIN)
        if not bool(baseline_ok.item()):
            save("BASELINE_REFERENCE_FAIL")
            return report
        # First occurrences compile/warm any shape-specific paths; excluded.
        request("warmup-B2-excluded", 1, "warmup")
        request("diagnostic-B2", 1, "diagnostic")
        request("warmup-B4-excluded", 3, "warmup")
        request("diagnostic-B4", 3, "diagnostic")
        preliminary = validator.validate_diagnostics(
            report["requests"], prompt, oracle, vocab_size=config["vocab_size"], require_reject_controls=False)
        report["diagnostic_gate"] = preliminary
        # All ranks must agree on branch selection before a collective-bearing request.
        flags = [int(preliminary["by_width"][str(b)]["passed"]) for b in (2, 4)]
        agreed = torch.tensor(flags, device="cuda", dtype=torch.int32)
        dist.all_reduce(agreed, op=dist.ReduceOp.MIN)
        eligible = {b: bool(v) for b, v in zip((2, 4), agreed.cpu().tolist())}
        report["eligible_widths"] = eligible
        if eligible[4]:
            request("diagnostic-B4-corrupt-first", 3, "diagnostic", corrupt=0)
            request("diagnostic-B4-corrupt-last", 3, "diagnostic", corrupt=2)
            report["state_gate"] = validator.validate_diagnostics(
                report["requests"], prompt, oracle, vocab_size=config["vocab_size"], require_reject_controls=True)
            state_pass = (report["state_gate"]["by_width"]["4"]["passed"]
                          and bool(report["state_gate"]["reject_controls"])
                          and all(v["passed"] for v in report["state_gate"]["reject_controls"].values()))
            state_ok = torch.tensor([int(state_pass)], device="cuda", dtype=torch.int32)
            dist.all_reduce(state_ok, op=dist.ReduceOp.MIN)
            eligible[4] = bool(state_ok.item())
        # Contemporary clean M1 measurement is retained even if block fidelity fails.
        # Block measures are admitted only after their own clean diagnostic gate.
        for trial in range(3):
            widths = [1] + [b for b in (2, 4) if eligible[b]]
            if trial % 2:
                widths.reverse()
            for b in widths:
                request(f"measure-B{b}-{trial + 1}", b - 1, "measure", cap=32)
        report["after_requests"] = process_snapshot(rank)
        save("COMPLETE")
        emit("block_verify_complete", report=str(output), eligible_widths=eligible)
        return report
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        save("ERROR")
        raise
    finally:
        observer.close()
