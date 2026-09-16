"""One-shot V2 KV metadata capture for a single long prefill request.

Diagnostic only.  Captures no model tensors: just request counters, logical
positions, gathered block-table ids and the slot ids actually handed to the
attention backends.  Disabled unless DS41_KV_CAPTURE_DIR is set.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch


class KVMetadataCaptureOnce:
    def __init__(self) -> None:
        root = os.environ.get("DS41_KV_CAPTURE_DIR", "").strip()
        self.enabled = bool(root)
        self.root = Path(root) if root else None
        self.prompt_tokens = int(os.environ.get("DS41_KV_CAPTURE_PROMPT_TOKENS", "1546"))
        self.rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        self.req_id: str | None = None
        self.chunks: list[dict[str, Any]] = []
        self.done = False

    def capture(self, input_batch: Any, block_tables: tuple[torch.Tensor, ...],
                slot_mappings: torch.Tensor, manager: Any) -> None:
        if not self.enabled or self.done:
            return
        for i, req_id in enumerate(input_batch.req_ids):
            prompt_len = int(input_batch.prefill_len_np[i])
            before = int(input_batch.num_computed_prefill_tokens_np[i])
            scheduled = int(input_batch.num_scheduled_tokens[i])
            if prompt_len != self.prompt_tokens or before >= prompt_len:
                continue
            if self.req_id is None:
                self.req_id = req_id
            if req_id != self.req_id:
                continue
            q0 = int(input_batch.query_start_loc_np[i])
            q1 = int(input_batch.query_start_loc_np[i + 1])
            positions = input_batch.positions[q0:q1].detach().to("cpu").tolist()
            seq_len = int(input_batch.seq_lens[i].detach().to("cpu").item())
            seq_upper = int(input_batch.seq_lens_cpu_upper_bound[i].item())
            groups: list[dict[str, Any]] = []
            for g in range(len(block_tables)):
                kbs = int(manager.kernel_block_sizes[g])
                kvbs = int(manager.block_sizes[g])
                mapping_enabled = bool(manager._slot_mapping_enabled[g])
                max_pos = max(positions) if positions else -1
                needed = max_pos // kbs + 1 if mapping_enabled and max_pos >= 0 else 0
                blocks = block_tables[g][i, :needed].detach().to("cpu").tolist()
                slots = slot_mappings[g, q0:q1].detach().to("cpu").tolist()
                consistent = True
                first_bad = None
                if mapping_enabled:
                    for j, (pos, slot) in enumerate(zip(positions, slots, strict=True)):
                        bi = int(pos) // kbs
                        expected = int(blocks[bi]) * kbs + (int(pos) % kbs)
                        if int(slot) != expected:
                            consistent = False
                            first_bad = {"offset": j, "position": int(pos),
                                         "slot": int(slot), "expected": int(expected),
                                         "block_index": bi, "block_id": int(blocks[bi])}
                            break
                groups.append({
                    "group": g,
                    "block_size": kvbs,
                    "kernel_block_size": kbs,
                    "mapping_enabled": mapping_enabled,
                    "block_ids": [int(x) for x in blocks],
                    "slots": [int(x) for x in slots],
                    "slot_consistency": consistent,
                    "first_bad": first_bad,
                })
            chunk = {
                "chunk_index": len(self.chunks),
                "req_id": req_id,
                "prompt_len": prompt_len,
                "computed_before": before,
                "scheduled": scheduled,
                "computed_after": before + scheduled,
                "seq_len": seq_len,
                "seq_len_cpu_upper_bound": seq_upper,
                "query_slice": [q0, q1],
                "positions": [int(x) for x in positions],
                "position_range": ([int(positions[0]), int(positions[-1])] if positions else None),
                "groups": groups,
            }
            self.chunks.append(chunk)
            if before + scheduled >= prompt_len:
                self._write()
                self.done = True
            return

    def _write(self) -> None:
        assert self.root is not None
        self.root.mkdir(parents=True, exist_ok=True)
        out = {
            "schema": "ds41-v2-kv-metadata-capture-v1",
            "rank": self.rank,
            "prompt_tokens": self.prompt_tokens,
            "req_id": self.req_id,
            "chunks": self.chunks,
            "note": "Physical block ids may differ by rank; slot consistency is checked within each rank/group.",
        }
        path = self.root / f"kv-metadata-rank{self.rank}.json"
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)


def create_kv_metadata_capture_once() -> KVMetadataCaptureOnce:
    return KVMetadataCaptureOnce()
