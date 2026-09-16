"""One-shot real-request routed-MoE fixture capture for DS41 prefill work.

Opt-in only. The wrapper is installed once but remains inert unless maybe_arm()
selects a real target-prefill request. The forward hook only copies immutable
inputs/metadata; all route statistics are computed offline.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any


class RoutedFixtureCapture:
    def __init__(self) -> None:
        root = os.environ.get("DS41_ROUTE_CAPTURE_DIR", "").strip()
        self.enabled = bool(root)
        self.root = Path(root) if root else None
        self.rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        self.prompt_tokens = int(os.environ.get("DS41_ROUTE_CAPTURE_PROMPT_TOKENS", "1588"))
        self.selected = {
            int(x)
            for x in os.environ.get("DS41_ROUTE_CAPTURE_CALLS", "0,20,40,60").split(",")
            if x.strip()
        }
        self.req_id: str | None = None
        self.active = False
        self.done = False
        self.prefill_call = 0
        self.saved: list[int] = []
        self.invalid_reason: str | None = None
        self._installed = False
        if self.enabled:
            self.install()

    def install(self) -> None:
        if not self.enabled or self._installed:
            return
        from vllm_gguf_plugin.quantization.fused_moe import GGUFMoEMethod

        original = GGUFMoEMethod.apply
        cap = self

        @functools.wraps(original)
        def wrapped(method_self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
            out = original(
                method_self,
                layer,
                x,
                topk_weights,
                topk_ids,
                shared_experts,
                shared_experts_input,
            )
            # Hard target-prefill contract. Dummy/startup/decode and DSpark do
            # not satisfy active + shape + quant/expert-map guards.
            if cap.active and cap._is_target_prefill(layer, x, topk_ids):
                idx = cap.prefill_call
                cap.prefill_call += 1
                if idx in cap.selected:
                    try:
                        cap._save(idx, layer, x, topk_weights, topk_ids, method_self)
                    except BaseException as exc:
                        cap.invalid_reason = f"{type(exc).__name__}: {exc}"
                        cap.active = False
                        raise
            return out

        GGUFMoEMethod.apply = wrapped
        self._installed = True

    @staticmethod
    def _is_target_prefill(layer: Any, x: Any, topk_ids: Any) -> bool:
        try:
            return (
                x.ndim == 2
                and int(x.shape[0]) > 64
                and int(x.shape[1]) == 5120
                and topk_ids.ndim == 2
                and int(topk_ids.shape[1]) == 6
                and int(layer.w13_weight_type.weight_type) == 16  # IQ2_XXS
                and int(layer.w2_weight_type.weight_type) == 10   # Q2_K
                and layer.expert_map is not None
                and int(layer.expert_map.numel()) == 384
            )
        except Exception:
            return False

    def maybe_arm(self, input_batch: Any) -> None:
        if not self.enabled or self.done or self.active or self.invalid_reason:
            return
        for i, req_id in enumerate(input_batch.req_ids):
            before = int(input_batch.num_computed_prefill_tokens_np[i])
            plen = int(input_batch.prefill_len_np[i])
            if (
                bool(input_batch.is_prefilling_np[i])
                and before == 0
                and plen == self.prompt_tokens
            ):
                self.req_id = str(req_id)
                self.active = True
                self.prefill_call = 0
                self.saved.clear()
                return

    def maybe_finish(self, input_batch: Any) -> None:
        if not self.active or self.req_id is None:
            return
        for i, req_id in enumerate(input_batch.req_ids):
            if str(req_id) != self.req_id:
                continue
            before = int(input_batch.num_computed_prefill_tokens_np[i])
            scheduled = int(input_batch.num_scheduled_tokens[i])
            plen = int(input_batch.prefill_len_np[i])
            if before < plen and before + scheduled >= plen:
                self.active = False
                self.done = True
                if self.root is not None:
                    self.root.mkdir(parents=True, exist_ok=True)
                    status = self.root / f"route-fixture-rank{self.rank}-status.txt"
                    if set(self.saved) == self.selected and self.invalid_reason is None:
                        status.write_text("COMPLETE\n")
                    else:
                        status.write_text(
                            f"INVALID saved={sorted(self.saved)} expected={sorted(self.selected)} "
                            f"reason={self.invalid_reason or 'missing fixture'}\n"
                        )
                return

    def _save(self, idx: int, layer: Any, x: Any, weights: Any, ids: Any, method_self: Any) -> None:
        import torch

        assert self.root is not None
        self.root.mkdir(parents=True, exist_ok=True)
        # No bincount, remapping, clamping, or mutation here. Preserve raw IDs,
        # including any unexpected negative values, for offline classification.
        row = {
            "schema": "ds41-routed-prefill-fixture-v2",
            "rank": self.rank,
            "request_id": self.req_id,
            "call_index": idx,
            "chunk_index": idx // 40,
            "layer_index": idx % 40,
            "tokens": int(x.shape[0]),
            "hidden": int(x.shape[1]),
            "top_k": int(ids.shape[1]),
            "w13_quant": int(layer.w13_weight_type.weight_type),
            "w2_quant": int(layer.w2_weight_type.weight_type),
            "w13_shape": list(layer.w13_weight.shape),
            "w2_shape": list(layer.w2_weight.shape),
            "swiglu_limit": float(method_self.moe.swiglu_limit or 0.0),
            "x": x.detach().to("cpu"),
            "topk_weights": weights.detach().to("cpu"),
            "topk_ids": ids.detach().to("cpu"),
            "expert_map": layer.expert_map.detach().to("cpu"),
        }
        path = self.root / f"route-fixture-rank{self.rank}-call{idx:02d}.pt"
        torch.save(row, path)
        self.saved.append(idx)


def create_routed_fixture_capture() -> RoutedFixtureCapture:
    return RoutedFixtureCapture()
