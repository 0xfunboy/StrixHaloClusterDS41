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
MODEL_DIR = Path("/home/funboy/models/gguf/deepseek-v4.1-flash-mixedq2")
MODEL_FILE = MODEL_DIR / "DSV41-mixedq2-00001-of-00005.gguf"
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
        results.append(
            run_generation(
                llm,
                prompts["coding"]["token_ids"],
                label="quality-coding",
                max_tokens=512,
            )
        )
        results.append(
            run_generation(
                llm,
                prompts["reasoning"]["token_ids"],
                label="quality-reasoning",
                max_tokens=256,
            )
        )
        results.append(
            run_generation(
                llm,
                prompts["json"]["token_ids"],
                label="quality-json",
                max_tokens=256,
            )
        )

    summary = {
        "status": "PASS" if smoke_pass else "SMOKE_FAIL",
        "rank": RANK,
        "world_size": WORLD_SIZE,
        "attempt": ATTEMPT,
        "epoch": os.environ.get("DS41_OWNER_EPOCH"),
        "init_s": init_s,
        "prompt_tokens_file": str(TOKENS_PATH),
        "results": results,
    }
    RESULT_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    emit("run_complete", status=summary["status"], result=str(RESULT_PATH))
    return 0 if smoke_pass else 20


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        emit("fatal", exc_type=type(exc).__name__, error=str(exc))
        raise
