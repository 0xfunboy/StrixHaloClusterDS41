#!/usr/bin/env python3
"""Probe the pinned V2 model-free speculative entry points, without model load.

This constructs EngineArgs/VllmConfig only. It never constructs LLM, an
executor, a model, or a process group. Unsupported entry points are recorded,
not bypassed, and are not performance or correctness verdicts on the hardware.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from vllm_gguf_plugin import register

register()
from vllm.engine.arg_utils import EngineArgs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bridge", action="store_true", help="Probe only the exact diagnostic V2 bridge, still without model load")
    args = parser.parse_args()
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "1":
        raise RuntimeError("The qualified launcher requires VLLM_USE_V2_MODEL_RUNNER=1")
    root = Path(__file__).resolve().parents[1]
    artifact = json.loads((root / "runtime/ds41/artifact.json").read_text())
    model_dir = Path(artifact["model_dir"])
    common = dict(
        model=str(model_dir / artifact["model_file"]),
        hf_config_path=str(model_dir), tokenizer=str(model_dir),
        config_format="gguf", load_format="gguf", quantization="gguf",
        dtype="bfloat16", tensor_parallel_size=2, pipeline_parallel_size=1,
        enable_expert_parallel=True, distributed_executor_backend="external_launcher",
        language_model_only=True, attention_backend="ROCM_FLASHMLA_SPARSE_DSV4",
        max_model_len=4096, block_size=128, max_num_seqs=1,
        max_num_batched_tokens=1024, kv_cache_memory_bytes=1073741824,
        enable_prefix_caching=False, enable_chunked_prefill=True,
        async_scheduling=False, enforce_eager=True, seed=1,
    )
    cases = [("target-only", None)]
    for k in (1, 3):
        cases.extend([
            (f"ngram-B{k + 1}", dict(method="ngram", num_speculative_tokens=k,
                                    prompt_lookup_min=1, prompt_lookup_max=2)),
            (f"custom-class-B{k + 1}", dict(method="custom_class", num_speculative_tokens=k,
                model="ds41_diagnostic_only.TeacherForcedProposer")),
        ])
    if args.bridge:
        from runtime.ds41.block_verify_experiment import install_bridge, SPECULATIVE_CONFIG
        install_bridge()
        cases = [("target-only", None), ("diagnostic-bridge-Kmax3", SPECULATIVE_CONFIG)]
    report = {
        "schema": "ds41-small-block-entrypoint-preflight-v1",
        "source_commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "model_loaded": False, "timing_kind": "configuration only, not inference",
        "diagnostic_bridge": args.bridge,
        "B_definition": "K draft candidates plus one committed anchor input position",
        "custom_class_note": "A non-imported sentinel class path, used only to probe V2 config admission",
        "cases": [],
    }
    for label, speculative in cases:
        start = time.monotonic()
        record = {"label": label, "speculative_config": speculative}
        try:
            config = EngineArgs(**common, speculative_config=speculative).create_engine_config()
            record.update(status="CONFIG_PASS", architecture=config.model_config.architecture,
                          runner_v2=config.use_v2_model_runner,
                          tp=config.parallel_config.tensor_parallel_size,
                          ep=config.parallel_config.enable_expert_parallel)
            if args.bridge and speculative is not None:
                import torch
                from vllm.v1.worker.gpu.model_runner import init_speculator
                from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator
                replay = init_speculator(config, torch.device("cpu"))
                if not isinstance(replay, BaseSpeculator) or replay.vocab_size != config.model_config.get_vocab_size():
                    raise RuntimeError("Diagnostic factory contract mismatch")
                record["weight_free_cpu_factory"] = "PASS"
        except Exception as exc:
            record.update(status="CONFIG_REJECTED", error_type=type(exc).__name__, error=str(exc))
        record["config_wall_s"] = time.monotonic() - start
        report["cases"].append(record)
        print(json.dumps(record), flush=True)
    report["baseline_pass"] = report["cases"][0]["status"] == "CONFIG_PASS"
    report["model_free_routes_admitted"] = [r["label"] for r in report["cases"][1:]
                                            if r["status"] == "CONFIG_PASS"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["baseline_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
