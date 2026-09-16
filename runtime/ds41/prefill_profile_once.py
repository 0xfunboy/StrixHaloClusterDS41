"""One-shot DS41 V2 prefill attribution using the existing async GPU profiler.

Opt-in only.  It profiles the first request whose prompt length meets the configured
minimum, from its first prefill chunk through logits-ready on its last chunk.  The
collector uses asynchronous GPU events and performs one synchronization only when
writing the final profile after the profiled prefill has completed.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class DS41PrefillProfileOnce:
    def __init__(self) -> None:
        self.enabled = os.environ.get("DS41_PREFILL_PROFILE_ONCE", "0") == "1"
        self.min_tokens = int(os.environ.get("DS41_PREFILL_PROFILE_MIN_TOKENS", "1500"))
        self.output_dir = Path(os.environ.get("DS41_PREFILL_PROFILE_DIR", "/tmp/ds41-prefill-profile"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.collector = None
        self.req_id: str | None = None
        self.prompt_tokens: int | None = None
        self.started_wall: float | None = None
        self.done = False
        if self.enabled:
            from runtime.ds41.perf_profile import DS41PerfCollector
            self.collector = DS41PerfCollector()
            self.collector.install()

    def maybe_start(self, input_batch: Any) -> None:
        if not self.enabled or self.done or self.req_id is not None or self.collector is None:
            return
        for i, req_id in enumerate(input_batch.req_ids):
            prompt_len = int(input_batch.prefill_len_np[i])
            before = int(input_batch.num_computed_prefill_tokens_np[i])
            if bool(input_batch.is_prefilling_np[i]) and before == 0 and prompt_len >= self.min_tokens:
                self.req_id = str(req_id)
                self.prompt_tokens = prompt_len
                self.started_wall = time.perf_counter()
                self.collector.enable()
                return

    def maybe_finish(self, input_batch: Any) -> None:
        if self.req_id is None or self.done or self.collector is None:
            return
        for i, req_id in enumerate(input_batch.req_ids):
            if str(req_id) != self.req_id:
                continue
            before = int(input_batch.num_computed_prefill_tokens_np[i])
            prompt_len = int(input_batch.prefill_len_np[i])
            scheduled = int(input_batch.num_scheduled_tokens[i])
            if before < prompt_len and before + scheduled >= prompt_len:
                self.collector.disable()
                out = self.output_dir / f"prefill-profile-rank{self.rank}.json"
                perf = self.collector.summarize(0, None, output=out)
                gpu = perf.get("all_gpu_event_ms", {})
                def g(name: str) -> float:
                    return float(gpu.get("prefill." + name, 0.0))
                decoder = g("decoder_layer")
                attention = g("attention")
                moe = g("moe_total")
                engram = g("engram_inject")
                mhc_pre = g("mhc_pre_total")
                mhc_post = g("mhc_post_total")
                attn_norm = g("attn_norm")
                ffn_norm = g("ffn_norm")
                top_residual = decoder - attention - moe - engram
                direct_hc_norm = mhc_pre + mhc_post + attn_norm + ffn_norm
                routed = g("routed_apply")
                shared = g("shared_mlp")
                moe_allreduce = g("collective_moe_allreduce")
                attribution = {
                    "schema": "ds41-prefill-attribution-v1",
                    "rank": self.rank,
                    "request_id": self.req_id,
                    "prompt_tokens": self.prompt_tokens,
                    "instrumented_wall_s": None if self.started_wall is None else time.perf_counter() - self.started_wall,
                    "gpu_ms": {
                        "decoder_total": decoder,
                        "attention": attention,
                        "moe_total": moe,
                        "engram_inject": engram,
                        "top_level_unattributed": top_residual,
                        "mhc_pre": mhc_pre,
                        "mhc_post": mhc_post,
                        "attn_norm": attn_norm,
                        "ffn_norm": ffn_norm,
                        "residual_after_direct_mhc_norm": top_residual - direct_hc_norm,
                        "routed_moe": routed,
                        "shared_mlp": shared,
                        "moe_allreduce_nested": moe_allreduce,
                        "moe_unattributed_after_routed_shared_collective": moe - routed - shared - moe_allreduce,
                        "attention_allreduce_nested": g("collective_attention_allreduce"),
                        "linear_allreduce_nested": g("collective_linear_allreduce"),
                        "engram_allgather_nested": g("collective_engram_allgather"),
                        "engram_disk_lookup_nested": g("engram_disk_lookup"),
                    },
                    "coverage": {
                        "profiler_active_from_first_prefill_chunk": True,
                        "profiler_stopped_after_last_prefill_chunk_before_decode": True,
                        "nested_collectives_not_additive_to_parent": True,
                        "rank_times_must_not_be_summed_for_request_latency": True,
                    },
                    "raw_profile": str(out),
                }
                self.output_dir.mkdir(parents=True, exist_ok=True)
                tmp = self.output_dir / f"prefill-attribution-rank{self.rank}.json.tmp"
                final = self.output_dir / f"prefill-attribution-rank{self.rank}.json"
                tmp.write_text(json.dumps(attribution, indent=2, sort_keys=True) + "\n")
                tmp.replace(final)
                self.done = True
                return


def create_prefill_profile_once() -> DS41PrefillProfileOnce:
    return DS41PrefillProfileOnce()
