"""Bounded DS41 attention-only profiler for the promoted runtime.

This collector leaves MoE and mHC implementations untouched.  It records four
sequential attention stages plus nested diagnostics using asynchronous ROCm /
PyTorch events and resolves all events with one synchronization after the timed
request.  Nested diagnostics are not additive when the production code overlaps
work on auxiliary streams.
"""
from __future__ import annotations

import functools
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch


class DS41AttentionPerfCollector:
    def __init__(self) -> None:
        self.active = False
        self.capture_mode = False
        self.events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
        self.wall_ns: dict[str, int] = defaultdict(int)
        self.calls: dict[str, int] = defaultdict(int)
        self.metadata: dict[str, dict[str, Any]] = {}
        self.layer_roles: dict[int, dict[str, Any]] = {}
        self._installed = False
        self._patches: list[tuple[Any, str, Any]] = []
        self.current_phase = "unknown"
        self.current_cr: int | None = None
        self.current_layer: int | None = None
        self._fixture_refs: dict[int, dict[str, Any]] = {}

    @staticmethod
    def phase_from_tensor(x: Any) -> str:
        if isinstance(x, torch.Tensor) and x.ndim >= 1:
            return "decode" if int(x.shape[0]) == 1 else "prefill"
        return "unknown"

    @staticmethod
    def _tensor_meta(x: Any) -> dict[str, Any] | None:
        if not isinstance(x, torch.Tensor):
            return None
        return {
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "stride": list(x.stride()),
            "contiguous": bool(x.is_contiguous()),
            "device": str(x.device),
        }

    def _name(self, key: str, *, phase: str | None = None, cr: int | None = None) -> str:
        ph = phase or self.current_phase
        ratio = self.current_cr if cr is None else cr
        crs = "x" if ratio is None else str(int(ratio))
        return f"{ph}.attention.cr{crs}.{key}"

    def _record_metadata(self, key: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if not self.active or self.current_phase != "decode" or key in self.metadata:
            return
        tensors: list[dict[str, Any]] = []
        for idx, obj in enumerate(args):
            meta = self._tensor_meta(obj)
            if meta is not None:
                meta["arg"] = idx
                tensors.append(meta)
        for name, obj in kwargs.items():
            meta = self._tensor_meta(obj)
            if meta is not None:
                meta["kwarg"] = name
                tensors.append(meta)
        self.metadata[key] = {
            "layer_id": self.current_layer,
            "compress_ratio": self.current_cr,
            "tensors": tensors[:8],
        }

    def _recorded_call(
        self,
        key: str,
        fn: Callable,
        *args,
        phase: str | None = None,
        cr: int | None = None,
        **kwargs,
    ):
        if not self.active:
            return fn(*args, **kwargs)
        name = self._name(key, phase=phase, cr=cr)
        self._record_metadata(name, args, kwargs)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        t0 = time.perf_counter_ns()
        try:
            return fn(*args, **kwargs)
        finally:
            self.wall_ns[name] += time.perf_counter_ns() - t0
            self.calls[name] += 1
            end.record()
            self.events[name].append((start, end))

    def _patch(self, obj: Any, name: str, replacement: Any) -> None:
        old = getattr(obj, name)
        self._patches.append((obj, name, old))
        setattr(obj, name, replacement)

    def _wrap_method(self, cls: Any, method_name: str, key: str) -> None:
        original = getattr(cls, method_name)
        collector = self

        @functools.wraps(original)
        def wrapped(self_obj, *args, **kwargs):
            return collector._recorded_call(key, original, self_obj, *args, **kwargs)

        self._patch(cls, method_name, wrapped)

    def _wrap_module_function(self, module: Any, fn_name: str, key: str) -> None:
        if not hasattr(module, fn_name):
            return
        original = getattr(module, fn_name)
        collector = self

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            return collector._recorded_call(key, original, *args, **kwargs)

        self._patch(module, fn_name, wrapped)

    def install(self) -> None:
        if self._installed:
            return
        from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
        from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention, DeepseekV4Indexer
        from vllm.models.deepseek_v4_1.compressor import DeepseekCompressor
        from vllm.models.deepseek_v4_1.amd.rocm import DeepseekV41ROCMAiterMLAAttention
        import vllm.models.deepseek_v4_1.attention as attn_mod
        import vllm.models.deepseek_v4_1.compressor as compressor_mod
        import vllm.models.deepseek_v4_1.amd.rocm as rocm_mod

        collector = self
        original_forward = DeepseekV4Attention.forward

        @functools.wraps(original_forward)
        def attention_forward(self_obj, positions, hidden_states, *args, **kwargs):
            old = (collector.current_phase, collector.current_cr, collector.current_layer)
            phase = collector.phase_from_tensor(hidden_states)
            collector.current_phase = phase
            collector.current_cr = int(self_obj.compress_ratio)
            collector.current_layer = int(self_obj.layer_id)
            if collector.active:
                collector.layer_roles.setdefault(
                    int(self_obj.layer_id),
                    {
                        "compress_ratio": int(self_obj.compress_ratio),
                        "is_kv_source": bool(self_obj.is_kv_source),
                        "is_index_source": bool(self_obj.is_index_source),
                        "kv_source_layer_id": self_obj.kv_source_layer_id,
                        "index_source_layer_id": self_obj.index_source_layer_id,
                    },
                )
            if collector.capture_mode and phase == "decode":
                cr = int(self_obj.compress_ratio)
                row = collector._fixture_refs.setdefault(
                    cr,
                    {
                        "compress_ratio": cr,
                        "layer_id": int(self_obj.layer_id),
                        "is_kv_source": bool(self_obj.is_kv_source),
                        "is_index_source": bool(self_obj.is_index_source),
                    },
                )
                row.setdefault("hidden_states", hidden_states.detach().clone())
                row.setdefault("positions", positions.detach().clone())
            try:
                return collector._recorded_call(
                    "total", original_forward, self_obj, positions, hidden_states, *args, phase=phase, cr=int(self_obj.compress_ratio), **kwargs
                )
            finally:
                collector.current_phase, collector.current_cr, collector.current_layer = old

        self._patch(DeepseekV4Attention, "forward", attention_forward)

        # Four disjoint sequential stages inside attention.forward.
        self._wrap_method(DeepseekV41ROCMAiterMLAAttention, "_run_parallel_input_projections", "stage_input_parallel")
        self._wrap_method(DeepseekV41ROCMAiterMLAAttention, "_split_qkv_and_norm", "stage_split_qkv_norm")
        self._wrap_method(DeepseekV4Attention, "_prepare_and_attn", "stage_prepare_attn")

        original_o_proj = DeepseekV41ROCMAiterMLAAttention._o_proj

        @functools.wraps(original_o_proj)
        def o_proj_wrapped(self_obj, o, positions):
            if collector.capture_mode and collector.current_phase == "decode":
                cr = int(self_obj.compress_ratio)
                row = collector._fixture_refs.setdefault(
                    cr,
                    {"compress_ratio": cr, "layer_id": int(self_obj.layer_id)},
                )
                row.setdefault("o", o.detach().clone())
                row.setdefault("o_positions", positions.detach().clone())
            return collector._recorded_call("stage_o_proj", original_o_proj, self_obj, o, positions)

        self._patch(DeepseekV41ROCMAiterMLAAttention, "_o_proj", o_proj_wrapped)

        # Nested input/projection diagnostics. These may overlap on aux streams.
        self._wrap_method(DeepseekV41ROCMAiterMLAAttention, "_fused_wqa_wkv_gemm", "input_fused_wqa_wkv")
        self._wrap_method(DeepseekV4Attention, "_wq_b_proj", "input_main_wq_b")
        self._wrap_method(DeepseekV4Attention, "_fused_qnorm_rope_kv_insert", "prepare_q_rope_kv_insert")
        self._wrap_method(DeepseekV4Attention, "_sparse_indexer_and_attn", "prepare_sparse_indexer_and_attn")
        self._wrap_method(DeepseekV41ROCMAiterMLAAttention, "forward_mqa", "attn_forward_mqa")
        self._wrap_method(DeepseekV41ROCMAiterMLAAttention, "_forward_decode", "attn_sparse_decode_stage")

        # Compressor / indexer are source-layer work only.
        self._wrap_method(DeepseekCompressor, "forward", "compressor_forward")
        self._wrap_method(DeepseekCompressor, "insert_cache", "compressor_insert_cache")
        self._wrap_method(DeepseekV4Indexer, "forward", "indexer_prepare")
        self._wrap_method(DeepseekV4Indexer, "_produce_k", "indexer_produce_k")
        self._wrap_method(DeepseekV4Indexer, "_wq_b_proj", "indexer_wq_b")
        self._wrap_method(SparseAttnIndexer, "forward", "indexer_sparse_op")

        # Module functions used by the actual ROCm path.
        self._wrap_module_function(attn_mod, "_ds41_decomposed_q_rope_kv_insert", "prepare_q_rope_kv_insert_decomposed")
        self._wrap_module_function(attn_mod, "fused_indexer_q_rope_quant", "indexer_q_rope_quant")
        self._wrap_module_function(attn_mod, "indexer_k_norm_rope_store", "indexer_k_norm_rope_store")
        self._wrap_module_function(compressor_mod, "fused_save_compress_norm", "compressor_save_compress_norm")
        self._wrap_module_function(compressor_mod, "rope_quant_insert", "compressor_rope_cache_insert")
        self._wrap_module_function(rocm_mod, "compute_global_topk_ragged_indices_and_indptr", "attn_topk_ragged_prep")
        self._wrap_module_function(rocm_mod, "rocm_sparse_attn_decode", "attn_sparse_decode_kernel")
        self._wrap_module_function(rocm_mod, "rocm_inv_rope_einsum", "output_inv_rope_wo_a")
        self._wrap_module_function(rocm_mod, "tensor_model_parallel_all_reduce", "output_tp_allreduce")

        original_bpre = DeepseekV41ROCMAiterMLAAttention._bpre_attn_gemm

        @functools.wraps(original_bpre)
        def bpre_wrapped(self_obj, weight, scale, x, reduce_tp):
            try:
                is_wo_b = weight.data_ptr() == self_obj.wo_b.weight.data_ptr()
            except Exception:
                is_wo_b = False
            key = "output_wo_b" if is_wo_b else "input_bpre_gemm"
            return collector._recorded_call(key, original_bpre, self_obj, weight, scale, x, reduce_tp)

        self._patch(DeepseekV41ROCMAiterMLAAttention, "_bpre_attn_gemm", bpre_wrapped)
        self._installed = True

    def enable(self) -> None:
        self.active = True

    def disable(self) -> None:
        self.active = False

    def set_capture_mode(self, enabled: bool) -> None:
        self.capture_mode = bool(enabled)

    def save_fixtures(self, output: Path) -> dict[str, Any]:
        torch.cuda.synchronize()
        rows = []
        for cr in sorted(self._fixture_refs):
            src = self._fixture_refs[cr]
            row: dict[str, Any] = {}
            for key, value in src.items():
                if isinstance(value, torch.Tensor):
                    row[key] = value.detach().cpu()
                else:
                    row[key] = value
            rows.append(row)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"schema": "ds41-attention-fixtures-v1", "rows": rows}, output)
        return {
            "path": str(output),
            "count": len(rows),
            "compress_ratios": [int(x["compress_ratio"]) for x in rows],
            "layers": [int(x["layer_id"]) for x in rows],
        }

    def summarize(self, completion_tokens: int, decode_span_s: float | None, *, output: Path | None = None) -> dict[str, Any]:
        torch.cuda.synchronize()
        gpu_ms = {name: float(sum(a.elapsed_time(b) for a, b in pairs)) for name, pairs in self.events.items()}
        wall_ms = {k: v / 1e6 for k, v in self.wall_ns.items()}
        decode_steps = max(0, int(completion_tokens) - 1)

        def value(cr: int, key: str) -> float:
            return gpu_ms.get(f"decode.attention.cr{cr}.{key}", 0.0)

        stage_keys = ("stage_input_parallel", "stage_split_qkv_norm", "stage_prepare_attn", "stage_o_proj")
        per_cr: dict[str, Any] = {}
        totals = defaultdict(float)
        for cr in (0, 1, 2):
            total = value(cr, "total")
            stages = {k: value(cr, k) for k in stage_keys}
            stage_sum = sum(stages.values())
            per_cr[str(cr)] = {
                "gpu_ms_total": total,
                "gpu_ms_per_decode_token": total / decode_steps if decode_steps else None,
                "stages_ms_total": stages,
                "stages_ms_per_decode_token": {k: v / decode_steps if decode_steps else None for k, v in stages.items()},
                "sequential_residual_ms_total": total - stage_sum,
                "sequential_residual_ms_per_decode_token": (total - stage_sum) / decode_steps if decode_steps else None,
            }
            totals["total"] += total
            for k, v in stages.items():
                totals[k] += v

        nested_names = (
            "input_fused_wqa_wkv", "input_main_wq_b", "input_bpre_gemm",
            "prepare_q_rope_kv_insert", "prepare_q_rope_kv_insert_decomposed",
            "compressor_forward", "compressor_save_compress_norm", "compressor_insert_cache", "compressor_rope_cache_insert",
            "indexer_prepare", "indexer_produce_k", "indexer_wq_b", "indexer_q_rope_quant", "indexer_k_norm_rope_store", "indexer_sparse_op",
            "prepare_sparse_indexer_and_attn", "attn_forward_mqa", "attn_sparse_decode_stage", "attn_topk_ragged_prep", "attn_sparse_decode_kernel",
            "output_inv_rope_wo_a", "output_wo_b", "output_tp_allreduce",
        )
        nested_total: dict[str, float] = {}
        nested_by_cr: dict[str, dict[str, float]] = {"0": {}, "1": {}, "2": {}}
        for key in nested_names:
            s = 0.0
            for cr in (0, 1, 2):
                v = value(cr, key)
                nested_by_cr[str(cr)][key] = v / decode_steps if decode_steps else None
                s += v
            nested_total[key] = s / decode_steps if decode_steps else None

        stage_sum = sum(totals[k] for k in stage_keys)
        result = {
            "schema": "ds41-attention-profile-v1",
            "completion_tokens": int(completion_tokens),
            "decode_steps": decode_steps,
            "decode_span_s": decode_span_s,
            "observed_decode_ms_per_token": decode_span_s * 1000 / decode_steps if decode_span_s is not None and decode_steps else None,
            "attention_gpu_ms_total": totals["total"],
            "attention_gpu_ms_per_decode_token": totals["total"] / decode_steps if decode_steps else None,
            "attention_sequential_stages_ms_per_decode_token": {k: totals[k] / decode_steps if decode_steps else None for k in stage_keys},
            "attention_sequential_residual_ms_per_decode_token": (totals["total"] - stage_sum) / decode_steps if decode_steps else None,
            "per_compress_ratio": per_cr,
            "nested_diagnostics_ms_per_decode_token": nested_total,
            "nested_by_compress_ratio_ms_per_decode_token": nested_by_cr,
            "calls": dict(self.calls),
            "cpu_wrapper_wall_ms": wall_ms,
            "metadata": self.metadata,
            "layer_roles": self.layer_roles,
            "notes": [
                "The four stage_* timings are sequential inside attention.forward and may be summed.",
                "Nested diagnostics may overlap on auxiliary streams and must not be added to stage totals.",
                "All GPU events are resolved after the request with one synchronize; no operator-level synchronize is inserted.",
            ],
        }
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            tmp = output.with_suffix(output.suffix + ".tmp")
            tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            tmp.replace(output)
        return result
