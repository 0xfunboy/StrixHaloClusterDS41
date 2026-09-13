"""DS41 offline SPMD qualification for vLLM external_launcher.

Every torchrun rank constructs the same LLM and executes the same generate()
sequence. Only reporting is rank-specific; no HTTP server participates.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from vllm_gguf_plugin import register

register()

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.v1.executor.uniproc_executor import ExecutorWithExternalLauncher  # noqa: E402

ROOT = Path("/home/funboy/StrixHaloClusterDS41")
ARTIFACT_CONFIG = json.loads((ROOT / "runtime/ds41/artifact.json").read_text())
MODEL_DIR = Path(ARTIFACT_CONFIG["model_dir"])
MODEL_FILE = MODEL_DIR / ARTIFACT_CONFIG["model_file"]
ATTEMPT = os.environ.get("DS41_ATTEMPT_NAME", "attempt009")
RAW = ROOT / "reports/DS41-Q2-001" / ATTEMPT
RANK = int(os.environ.get("RANK", "-1"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "-1"))
RESULT_PATH = RAW / f"offline-rank{RANK}.json"
EVENT_PATH = RAW / f"offline-rank{RANK}.events.jsonl"
TOKENS_PATH = RAW / "prompt-tokens.json"


def emit(event: str, **data: Any) -> None:
    rec = {
        "event": event,
        "rank": RANK,
        "world_size": WORLD_SIZE,
        "wall_time": time.time(),
        "monotonic": time.monotonic(),
        **data,
    }
    line = json.dumps(rec, ensure_ascii=False, sort_keys=True)
    print(f"DS41_OFFLINE {line}", flush=True)
    with EVENT_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# Instrument the first executor handoff only. This proves that each local engine
# receives scheduler work without tracing individual model operators.
_orig_execute_model = ExecutorWithExternalLauncher.execute_model


def _execute_model_marked(self, scheduler_output, non_block=False):
    if not getattr(self, "_ds41_first_execute_logged", False):
        self._ds41_first_execute_logged = True
        summary: dict[str, Any] = {"type": type(scheduler_output).__name__}
        for name in ("num_scheduled_tokens", "scheduled_new_reqs", "scheduled_cached_reqs"):
            value = getattr(scheduler_output, name, None)
            if value is not None:
                try:
                    summary[name] = value if isinstance(value, (int, float, str, bool)) else len(value)
                except Exception:
                    summary[name] = repr(value)[:200]
        emit("execute_model_first", scheduler=summary)
    return _orig_execute_model(self, scheduler_output, non_block=non_block)


ExecutorWithExternalLauncher.execute_model = _execute_model_marked


def metrics_dict(metrics: Any) -> dict[str, Any] | None:
    if metrics is None:
        return None
    if dataclasses.is_dataclass(metrics):
        return dataclasses.asdict(metrics)
    out: dict[str, Any] = {}
    for key in (
        "num_generation_tokens",
        "num_preemptions",
        "arrival_time",
        "queued_ts",
        "scheduled_ts",
        "first_token_ts",
        "last_token_ts",
        "first_token_latency",
        "is_corrupted",
    ):
        if hasattr(metrics, key):
            out[key] = getattr(metrics, key)
    return out


def run_generation(
    llm: LLM,
    prompt_tokens: list[int],
    *,
    label: str,
    max_tokens: int,
    ignore_eos: bool = False,
) -> dict[str, Any]:
    params = SamplingParams(
        temperature=0.0,
        seed=1,
        max_tokens=max_tokens,
        ignore_eos=ignore_eos,
    )
    emit(
        "generate_enter",
        label=label,
        prompt_tokens=len(prompt_tokens),
        max_tokens=max_tokens,
        ignore_eos=ignore_eos,
    )
    start = time.monotonic()
    outputs = llm.generate(
        {"prompt_token_ids": prompt_tokens},
        params,
        use_tqdm=False,
    )
    end = time.monotonic()
    if len(outputs) != 1 or len(outputs[0].outputs) != 1:
        raise RuntimeError(f"unexpected output cardinality for {label}: {outputs!r}")
    request = outputs[0]
    completion = request.outputs[0]
    tokens = list(completion.token_ids)
    m = metrics_dict(request.metrics)
    first_token_latency = None if not m else m.get("first_token_latency")
    decode_span = None
    decode_tps = None
    if m and len(tokens) > 1:
        first_ts = float(m.get("first_token_ts") or 0.0)
        last_ts = float(m.get("last_token_ts") or 0.0)
        if last_ts > first_ts:
            decode_span = last_ts - first_ts
            decode_tps = (len(tokens) - 1) / decode_span
    result = {
        "label": label,
        "prompt_token_count": len(prompt_tokens),
        "completion_token_count": len(tokens),
        "token_ids": tokens,
        "text": completion.text,
        "finish_reason": completion.finish_reason,
        "stop_reason": completion.stop_reason,
        "finished": bool(request.finished),
        "metrics": m,
        "client": {
            "wall_s": end - start,
            "ttft_s": first_token_latency,
        },
        "derived": {
            "decode_span_s": decode_span,
            "decode_tps_first_to_last": decode_tps,
            "http_e2e_s": None,
            "http_ttft_s": None,
            "note": "Offline SPMD; HTTP metrics intentionally unavailable.",
        },
    }
    emit(
        "generate_exit",
        label=label,
        completion_tokens=len(tokens),
        finish_reason=completion.finish_reason,
        wall_s=end - start,
        ttft_s=first_token_latency,
        decode_tps=decode_tps,
        text=completion.text[:400],
    )
    return result


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    token_spec = json.loads(TOKENS_PATH.read_text(encoding="utf-8"))
    prompts = token_spec["prompts"]
    native_hip_ab_cfg = token_spec.get("native_hip_ab")
    native_hip_identity = None
    native_hip_default = os.environ.get("DS41_NATIVE_HIP_MOE", "0") == "1"
    if isinstance(native_hip_ab_cfg, dict) or native_hip_default:
        from runtime.ds41.native_hip_moe_runtime import ensure_loaded

        native_hip_identity = ensure_loaded()
        emit("native_hip_library_loaded", **native_hip_identity)
    if isinstance(native_hip_ab_cfg, dict):
        # The extension is resident before LLM construction, but baseline A
        # remains Triton until both ranks explicitly toggle between requests.
        os.environ["DS41_NATIVE_HIP_MOE"] = "0"
        os.environ["DS41_EP_SKIP_REMOTE"] = "1"

    from runtime.ds41.artifact_identity import verify_fast

    artifact_identity = verify_fast(RANK)
    emit("artifact_identity", **artifact_identity)
    emit(
        "llm_init_begin",
        model=str(MODEL_FILE),
        attempt=ATTEMPT,
        epoch=os.environ.get("DS41_OWNER_EPOCH"),
    )
    init_start = time.monotonic()
    llm = LLM(
        model=str(MODEL_FILE),
        tokenizer=str(MODEL_DIR),
        hf_config_path=str(MODEL_DIR),
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        enable_expert_parallel=True,
        distributed_executor_backend="external_launcher",
        language_model_only=True,
        config_format="gguf",
        load_format="gguf",
        quantization="gguf",
        dtype="bfloat16",
        attention_backend="ROCM_FLASHMLA_SPARSE_DSV4",
        max_model_len=4096,
        block_size=128,
        max_num_seqs=1,
        max_num_batched_tokens=1024,
        kv_cache_memory_bytes=1073741824,
        kv_cache_dtype="auto",
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        async_scheduling=False,
        enforce_eager=True,
        seed=1,
        generation_config="vllm",
        disable_log_stats=False,
    )
    init_s = time.monotonic() - init_start
    emit("llm_init_end", init_s=init_s)

    results: list[dict[str, Any]] = []

    def save_checkpoint(status: str) -> None:
        checkpoint = {
            "status": status,
            "rank": RANK,
            "world_size": WORLD_SIZE,
            "attempt": ATTEMPT,
            "epoch": os.environ.get("DS41_OWNER_EPOCH"),
            "init_s": init_s,
            "artifact_identity": artifact_identity,
            "prompt_tokens_file": str(TOKENS_PATH),
            "results": results,
        }
        tmp = RESULT_PATH.with_suffix(RESULT_PATH.suffix + ".tmp")
        tmp.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False) + "\n")
        os.replace(tmp, RESULT_PATH)

    if isinstance(native_hip_ab_cfg, dict):
        speed_tokens = int(native_hip_ab_cfg.get("speed_tokens", 64))
        warmup_tokens = int(native_hip_ab_cfg.get("warmup_tokens", 16))

        from vllm_gguf_plugin.quantization.fused_moe import ds41_native_hip_stats

        def native_stats_event(label: str) -> dict[str, Any]:
            stats = ds41_native_hip_stats()
            emit("native_hip_stats", label=label, **stats)
            return stats

        def set_native(enabled: bool) -> None:
            os.environ["DS41_NATIVE_HIP_MOE"] = "1" if enabled else "0"
            emit(
                "native_hip_mode",
                enabled=enabled,
                library_sha256=(native_hip_identity or {}).get("sha256"),
            )

        # Contemporary baseline A is the already-promoted Triton skip-remote
        # path. Candidate B changes only M=1 routed IQ2_XXS->Q2_K execution.
        set_native(False)
        results.append(run_generation(
            llm, prompts["speed"]["token_ids"], label="hip-ab-baseline-warmup-excluded",
            max_tokens=warmup_tokens, ignore_eos=True))
        save_checkpoint("HIP_AB_BASELINE_WARMUP_COMPLETE")
        results.append(run_generation(
            llm, prompts["speed"]["token_ids"], label="hip-ab-baseline-1",
            max_tokens=speed_tokens, ignore_eos=True))
        save_checkpoint("HIP_AB_BASELINE_1_COMPLETE")

        set_native(True)
        results.append(run_generation(
            llm, prompts["speed"]["token_ids"], label="hip-ab-candidate-warmup-excluded",
            max_tokens=warmup_tokens, ignore_eos=True))
        native_stats_event("after_candidate_warmup")
        save_checkpoint("HIP_AB_CANDIDATE_WARMUP_COMPLETE")
        results.append(run_generation(
            llm, prompts["speed"]["token_ids"], label="hip-ab-candidate-1",
            max_tokens=speed_tokens, ignore_eos=True))
        native_stats_event("after_candidate_1")
        save_checkpoint("HIP_AB_CANDIDATE_1_COMPLETE")

        set_native(False)
        results.append(run_generation(
            llm, prompts["speed"]["token_ids"], label="hip-ab-baseline-2",
            max_tokens=speed_tokens, ignore_eos=True))
        save_checkpoint("HIP_AB_BASELINE_2_COMPLETE")

        set_native(True)
        results.append(run_generation(
            llm, prompts["speed"]["token_ids"], label="hip-ab-candidate-2",
            max_tokens=speed_tokens, ignore_eos=True))
        native_stats = native_stats_event("after_candidate_2")
        save_checkpoint("HIP_AB_CANDIDATE_2_COMPLETE")

        # Quality gates remain true generation on the same loaded candidate.
        results.append(run_generation(
            llm, prompts["arithmetic"]["token_ids"], label="hip-ab-candidate-smoke",
            max_tokens=128, ignore_eos=False))
        save_checkpoint("HIP_AB_CANDIDATE_SMOKE_COMPLETE")
        results.append(run_generation(
            llm, prompts["coding"]["token_ids"], label="hip-ab-candidate-coding",
            max_tokens=512, ignore_eos=False))
        save_checkpoint("HIP_AB_CANDIDATE_CODING_COMPLETE")
        results.append(run_generation(
            llm, prompts["json"]["token_ids"], label="hip-ab-candidate-json",
            max_tokens=256, ignore_eos=False))
        save_checkpoint("HIP_AB_CANDIDATE_JSON_COMPLETE")
        if "reasoning_high" not in prompts:
            raise RuntimeError("native HIP A/B fixture is missing reasoning_high")
        results.append(run_generation(
            llm, prompts["reasoning_high"]["token_ids"],
            label="hip-ab-candidate-reasoning-high",
            max_tokens=int(native_hip_ab_cfg.get("reasoning_high_max_tokens", 128)),
            ignore_eos=False))
        save_checkpoint("HIP_AB_CANDIDATE_REASONING_HIGH_COMPLETE")
        native_stats = native_stats_event("after_quality")

        summary = {
            "status": "NATIVE_HIP_AB_COMPLETE",
            "rank": RANK,
            "world_size": WORLD_SIZE,
            "attempt": ATTEMPT,
            "epoch": os.environ.get("DS41_OWNER_EPOCH"),
            "init_s": init_s,
            "artifact_identity": artifact_identity,
            "native_hip_identity": native_hip_identity,
            "native_hip_stats": native_stats,
            "prompt_tokens_file": str(TOKENS_PATH),
            "optimization": "NATIVE_HIP_M1_IQ2_XXS_Q2_K",
            "results": results,
        }
        tmp = RESULT_PATH.with_suffix(RESULT_PATH.suffix + ".tmp")
        tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        os.replace(tmp, RESULT_PATH)
        emit("run_complete", status=summary["status"], result=str(RESULT_PATH))
        return 0

    optimization_cfg = token_spec.get("optimization_ab")
    if isinstance(optimization_cfg, dict):
        # Same-load causal A/B for the measured EP remote-route candidate.
        # The plugin reads DS41_EP_SKIP_REMOTE at each routed call, so both
        # ranks can switch between the qualified fallback and candidate without
        # reloading weights or changing sampling.
        speed_tokens = int(optimization_cfg.get("speed_tokens", 64))
        warmup_tokens = int(optimization_cfg.get("warmup_tokens", 16))

        def set_candidate(enabled: bool) -> None:
            if enabled:
                os.environ["DS41_EP_SKIP_REMOTE"] = "1"
            else:
                os.environ.pop("DS41_EP_SKIP_REMOTE", None)
            emit("optimization_mode", candidate=enabled)

        set_candidate(False)
        results.append(run_generation(
            llm, prompts["speed"]["token_ids"], label="ab-baseline-warmup-excluded",
            max_tokens=warmup_tokens, ignore_eos=True))
        save_checkpoint("AB_BASELINE_WARMUP_COMPLETE")

        set_candidate(False)
        results.append(run_generation(
            llm, prompts["speed"]["token_ids"], label="ab-baseline-1",
            max_tokens=speed_tokens, ignore_eos=True))
        save_checkpoint("AB_BASELINE_1_COMPLETE")

        set_candidate(True)
        results.append(run_generation(
            llm, prompts["speed"]["token_ids"], label="ab-candidate-warmup-excluded",
            max_tokens=warmup_tokens, ignore_eos=True))
        save_checkpoint("AB_CANDIDATE_WARMUP_COMPLETE")
        results.append(run_generation(
            llm, prompts["speed"]["token_ids"], label="ab-candidate-1",
            max_tokens=speed_tokens, ignore_eos=True))
        save_checkpoint("AB_CANDIDATE_1_COMPLETE")

        set_candidate(False)
        results.append(run_generation(
            llm, prompts["speed"]["token_ids"], label="ab-baseline-2",
            max_tokens=speed_tokens, ignore_eos=True))
        save_checkpoint("AB_BASELINE_2_COMPLETE")

        set_candidate(True)
        results.append(run_generation(
            llm, prompts["speed"]["token_ids"], label="ab-candidate-2",
            max_tokens=speed_tokens, ignore_eos=True))
        save_checkpoint("AB_CANDIDATE_2_COMPLETE")

        # Candidate correctness gates on the same loaded model.
        results.append(run_generation(
            llm, prompts["arithmetic"]["token_ids"], label="ab-candidate-smoke",
            max_tokens=128, ignore_eos=False))
        save_checkpoint("AB_CANDIDATE_SMOKE_COMPLETE")
        results.append(run_generation(
            llm, prompts["coding"]["token_ids"], label="ab-candidate-coding",
            max_tokens=512, ignore_eos=False))
        save_checkpoint("AB_CANDIDATE_CODING_COMPLETE")
        results.append(run_generation(
            llm, prompts["json"]["token_ids"], label="ab-candidate-json",
            max_tokens=256, ignore_eos=False))
        save_checkpoint("AB_CANDIDATE_JSON_COMPLETE")

        summary = {
            "status": "OPTIMIZATION_AB_COMPLETE",
            "rank": RANK, "world_size": WORLD_SIZE, "attempt": ATTEMPT,
            "epoch": os.environ.get("DS41_OWNER_EPOCH"), "init_s": init_s,
            "artifact_identity": artifact_identity,
            "prompt_tokens_file": str(TOKENS_PATH),
            "optimization": "EP_SKIP_REMOTE",
            "results": results,
        }
        tmp = RESULT_PATH.with_suffix(RESULT_PATH.suffix + ".tmp")
        tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        os.replace(tmp, RESULT_PATH)
        emit("run_complete", status=summary["status"], result=str(RESULT_PATH))
        return 0

    profile_cfg = token_spec.get("profile_mode")
    if os.environ.get("DS41_PROFILE_MODE") == "1" or isinstance(profile_cfg, dict):
        # Performance continuation: one short warmup, one uninstrumented decode
        # window, the identical instrumented window, then one independent
        # reasoning-effort=high check.  No full qualification suite is replayed.
        profile_cfg = profile_cfg if isinstance(profile_cfg, dict) else {}
        profile_tokens = int(profile_cfg.get("speed_tokens", os.environ.get("DS41_PROFILE_TOKENS", "32")))
        warmup_tokens = int(profile_cfg.get("warmup_tokens", os.environ.get("DS41_PROFILE_WARMUP_TOKENS", "16")))
        results.append(run_generation(
            llm, prompts["speed"]["token_ids"], label="profile-warmup-excluded",
            max_tokens=warmup_tokens, ignore_eos=True))
        save_checkpoint("PROFILE_WARMUP_COMPLETE")
        baseline = run_generation(
            llm, prompts["speed"]["token_ids"], label="profile-uninstrumented",
            max_tokens=profile_tokens, ignore_eos=True)
        results.append(baseline)
        save_checkpoint("PROFILE_UNINSTRUMENTED_COMPLETE")

        from runtime.ds41.perf_profile import DS41PerfCollector
        profiler = DS41PerfCollector()
        profiler.install()
        profiler.enable()
        instrumented = run_generation(
            llm, prompts["speed"]["token_ids"], label="profile-instrumented",
            max_tokens=profile_tokens, ignore_eos=True)
        profiler.disable()
        results.append(instrumented)
        profile_path = RAW / f"perf-profile-rank{RANK}.json"
        perf = profiler.summarize(
            instrumented["completion_token_count"],
            instrumented["derived"]["decode_span_s"],
            output=profile_path,
        )
        base_tps = baseline["derived"]["decode_tps_first_to_last"]
        inst_tps = instrumented["derived"]["decode_tps_first_to_last"]
        perf["uninstrumented_decode_tps"] = base_tps
        perf["instrumented_decode_tps"] = inst_tps
        perf["instrumentation_overhead_pct_by_decode_tps"] = (
            (base_tps / inst_tps - 1.0) * 100.0 if base_tps and inst_tps else None
        )
        profile_path.write_text(json.dumps(perf, indent=2, sort_keys=True) + "\n")
        emit("perf_profile_complete", path=str(profile_path), **{
            "baseline_tps": base_tps,
            "instrumented_tps": inst_tps,
            "overhead_pct": perf["instrumentation_overhead_pct_by_decode_tps"],
            "remote_route_fraction": perf["remote_route_fraction"],
        })
        save_checkpoint("PROFILE_INSTRUMENTED_COMPLETE")

        reasoning_key = "reasoning_high"
        if reasoning_key not in prompts:
            raise RuntimeError("profile prompt fixture is missing reasoning_high")
        reasoning = run_generation(
            llm, prompts[reasoning_key]["token_ids"], label="reasoning-high-once",
            max_tokens=int(profile_cfg.get("reasoning_high_max_tokens", os.environ.get("DS41_REASONING_HIGH_MAX_TOKENS", "128"))),
            ignore_eos=False)
        results.append(reasoning)
        save_checkpoint("REASONING_HIGH_COMPLETE")

        summary = {
            "status": "PROFILE_COMPLETE",
            "rank": RANK, "world_size": WORLD_SIZE, "attempt": ATTEMPT,
            "epoch": os.environ.get("DS41_OWNER_EPOCH"), "init_s": init_s,
            "artifact_identity": artifact_identity,
            "prompt_tokens_file": str(TOKENS_PATH),
            "profile_file": str(profile_path),
            "results": results,
        }
        tmp = RESULT_PATH.with_suffix(RESULT_PATH.suffix + ".tmp")
        tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        os.replace(tmp, RESULT_PATH)
        emit("run_complete", status=summary["status"], result=str(RESULT_PATH))
        return 0

    # 1 token proves both SPMD ranks enter the same first forward/collective.
    results.append(
        run_generation(
            llm,
            prompts["arithmetic"]["token_ids"],
            label="forward-1token",
            max_tokens=1,
            ignore_eos=True,
        )
    )
    save_checkpoint("FORWARD_1TOKEN_COMPLETE")

    # Full arithmetic smoke. Do not run speed/quality suite if this is wrong or incomplete.
    smoke = run_generation(
        llm,
        prompts["arithmetic"]["token_ids"],
        label="smoke-arithmetic",
        max_tokens=128,
        ignore_eos=False,
    )
    results.append(smoke)
    smoke_pass = smoke["finished"] and smoke["text"].strip() == "323"
    save_checkpoint("SMOKE_PASS" if smoke_pass else "SMOKE_FAIL")
    emit("smoke_gate", passed=smoke_pass, text=smoke["text"][:400])

    if smoke_pass:
        # Same fixed speed prompt; warmup is explicitly excluded later.
        results.append(
            run_generation(
                llm,
                prompts["speed"]["token_ids"],
                label="speed-warmup-excluded",
                max_tokens=256,
                ignore_eos=True,
            )
        )
        save_checkpoint("SPEED_WARMUP_COMPLETE")
        for idx in range(1, 4):
            results.append(
                run_generation(
                    llm,
                    prompts["speed"]["token_ids"],
                    label=f"speed-measured-{idx}",
                    max_tokens=256,
                    ignore_eos=True,
                )
            )
            save_checkpoint(f"SPEED_MEASURED_{idx}_COMPLETE")
        results.append(
            run_generation(
                llm,
                prompts["coding"]["token_ids"],
                label="quality-coding",
                max_tokens=512,
            )
        )
        save_checkpoint("CODING_COMPLETE")
        results.append(
            run_generation(
                llm,
                prompts["reasoning"]["token_ids"],
                label="quality-reasoning",
                max_tokens=256,
            )
        )
        save_checkpoint("REASONING_COMPLETE")
        results.append(
            run_generation(
                llm,
                prompts["json"]["token_ids"],
                label="quality-json",
                max_tokens=256,
            )
        )
        save_checkpoint("JSON_COMPLETE")

    summary = {
        "status": "PASS" if smoke_pass else "SMOKE_FAIL",
        "rank": RANK,
        "world_size": WORLD_SIZE,
        "attempt": ATTEMPT,
        "epoch": os.environ.get("DS41_OWNER_EPOCH"),
        "init_s": init_s,
        "artifact_identity": artifact_identity,
        "prompt_tokens_file": str(TOKENS_PATH),
        "results": results,
    }
    tmp = RESULT_PATH.with_suffix(RESULT_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, RESULT_PATH)
    emit("run_complete", status=summary["status"], result=str(RESULT_PATH))
    return 0 if smoke_pass else 20


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        emit("fatal", exc_type=type(exc).__name__, error=str(exc))
        raise
