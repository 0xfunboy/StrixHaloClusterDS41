#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace
from _ds41_artifact import MODEL_DIR

import torch

from vllm.models.deepseek_v4_1.amd.vl_model import (
    DeepseekV41ForCausalLM,
    _make_deepseek_v4_vl_weights_mapper,
)
from vllm_gguf_plugin.weight_utils import gguf_quant_weights_iterator_multi
from vllm_gguf_plugin.weights_adapter.deepseek_v41 import (
    DeepseekV41GGUFAdapter,
    map_deepseek_v41_gguf_name,
)
import gguf

ROOT = Path("/home/funboy/StrixHaloClusterDS41")
SHARD = MODEL_DIR / "DSV41-mixedq2-00005-of-00005.gguf"
EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")


def status_kb(key: str) -> int:
    with open("/proc/self/status", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1])
    return 0


def mapped_file_rss_kb(path: Path) -> int:
    total = 0
    current = False
    with open("/proc/self/smaps", "r", encoding="utf-8") as f:
        for line in f:
            if line and line[0].isalnum() and "-" in line.split(maxsplit=1)[0]:
                current = str(path) in line
            elif current and line.startswith("Rss:"):
                total += int(line.split()[1])
    return total


def raw_name_map(path: Path) -> dict[str, str]:
    reader = gguf.GGUFReader(str(path))
    out: dict[str, str] = {}
    for tensor in reader.tensors:
        mapped = map_deepseek_v41_gguf_name(tensor.name)
        if mapped is None:
            raise RuntimeError(f"unmapped raw tensor {tensor.name}")
        out[tensor.name] = mapped
    return out


class ConsumingChild:
    def __init__(self, upstream_counter: dict[str, int], local_expert_lo=0, local_expert_hi=192):
        self.upstream_counter = upstream_counter
        self.local_expert_lo = local_expert_lo
        self.local_expert_hi = local_expert_hi
        self.final: list[torch.Tensor] = []
        self.loaded: set[str] = set()
        self.first_consume_upstream_count: int | None = None
        self.samples: list[dict[str, int | str]] = []
        self.consumed = 0
        self.copied = 0

    def _local(self, name: str) -> bool:
        match = EXPERT_RE.search(name)
        if match is None:
            return True
        expert = int(match.group(1))
        return self.local_expert_lo <= expert < self.local_expert_hi

    def load_weights(self, weights):
        for name, src in weights:
            self.consumed += 1
            if self.first_consume_upstream_count is None:
                self.first_consume_upstream_count = self.upstream_counter["yielded"]
            if self._local(name):
                # This models the ownership/copy boundary that matters for the
                # retention bug: final device storage survives after the source
                # mmap-backed tensor can be released.
                dst = src.to("cuda", non_blocking=False).clone()
                self.final.append(dst)
                self.loaded.add(name)
                self.copied += 1
            if self.consumed in {1, 64, 256, 512, 1024}:
                torch.cuda.synchronize()
                self.samples.append(
                    {
                        "consumed": self.consumed,
                        "copied": self.copied,
                        "source_rss_kb": mapped_file_rss_kb(SHARD),
                        "rss_anon_kb": status_kb("RssAnon"),
                        "final_cuda_bytes": torch.cuda.memory_allocated(),
                    }
                )
        torch.cuda.synchronize()
        return self.loaded


class FakeWrapper:
    pass


def main() -> None:
    if os.environ.get("DS41_STREAM_TEXT_WEIGHTS") != "1":
        raise RuntimeError("DS41_STREAM_TEXT_WEIGHTS=1 required")
    if os.environ.get("DS41_DROP_SHARD_CACHE") != "1":
        raise RuntimeError("DS41_DROP_SHARD_CACHE=1 required")
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm device unavailable")

    torch.cuda.empty_cache()
    before = {
        "source_rss_kb": mapped_file_rss_kb(SHARD),
        "rss_anon_kb": status_kb("RssAnon"),
        "final_cuda_bytes": torch.cuda.memory_allocated(),
    }

    names = raw_name_map(SHARD)
    adapter = DeepseekV41GGUFAdapter()
    counter = {"yielded": 0}

    base = gguf_quant_weights_iterator_multi([str(SHARD)], names)
    transformed = adapter.transform_weights(base, None)

    def counted():
        for item in transformed:
            counter["yielded"] += 1
            yield item

    wrapper = FakeWrapper()
    wrapper.hf_to_vllm_mapper = _make_deepseek_v4_vl_weights_mapper(
        "fp8", "weight_scale_inv"
    )
    child = ConsumingChild(counter)
    wrapper.language_model = child
    wrapper._weights_finalized = False

    started = time.monotonic()
    loaded = DeepseekV41ForCausalLM.load_weights(wrapper, counted())
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started

    after = {
        "source_rss_kb": mapped_file_rss_kb(SHARD),
        "rss_anon_kb": status_kb("RssAnon"),
        "final_cuda_bytes": torch.cuda.memory_allocated(),
    }

    result = {
        "status": "PASS",
        "shard": str(SHARD),
        "elapsed_s": elapsed,
        "raw_tensors": len(names),
        "upstream_yielded": counter["yielded"],
        "child_consumed": child.consumed,
        "child_copied_local": child.copied,
        "first_consume_upstream_count": child.first_consume_upstream_count,
        "loaded_names": len(loaded),
        "before": before,
        "samples": child.samples,
        "after": after,
    }

    # The key causal contract: child consumption begins immediately, before
    # upstream exhaustion. Final device bytes grow while anonymous staging
    # remains bounded, and after complete-shard consumption the source mapping
    # is reclaimable rather than retained by a global sorted() list.
    assert child.first_consume_upstream_count == 1, result
    assert counter["yielded"] == child.consumed and child.consumed > len(names), result
    assert after["final_cuda_bytes"] > before["final_cuda_bytes"] + 256 * 1024**2, result
    assert after["rss_anon_kb"] - before["rss_anon_kb"] < 256 * 1024, result
    assert after["source_rss_kb"] < 512 * 1024, result
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
