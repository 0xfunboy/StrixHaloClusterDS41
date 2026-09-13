"""Bounded DS41 decode profiler using asynchronous ROCm/PyTorch events.

The profiler is installed only after model warmup and is active only for the
instrumented request.  It never synchronizes inside a wrapped operator.  GPU
Event pairs are resolved once, after the request, and CPU wall values are kept
separately (important for the synchronous disk Engram lookup).
"""
from __future__ import annotations

import functools
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch


class DS41PerfCollector:
    def __init__(self) -> None:
        self.active = False
        self.events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
        self.wall_ns: dict[str, int] = defaultdict(int)
        self.calls: dict[str, int] = defaultdict(int)
        self.current_phase = "unknown"
        self.remote_route_tensors: list[torch.Tensor] = []
        self.total_routes = 0
        self.engram_cache_before: dict[tuple[int, int], tuple[int, int, int]] = {}
        self.engram_cache_after: dict[tuple[int, int], tuple[int, int, int]] = {}
        self._installed = False
        self._patches: list[tuple[Any, str, Any]] = []
        # Decoder-local slot used only by the profile wrappers to distinguish
        # the two hidden-size RMSNorm calls without touching model instances.
        self._decoder_rms_slot: int | None = None

    @staticmethod
    def phase_from_tensor(x: Any) -> str:
        if isinstance(x, torch.Tensor) and x.ndim >= 1:
            return "decode" if int(x.shape[0]) == 1 else "prefill"
        return "unknown"

    def _recorded_call(self, key: str, fn: Callable, *args, phase: str | None = None, **kwargs):
        if not self.active:
            return fn(*args, **kwargs)
        ph = phase or self.current_phase
        name = f"{ph}.{key}"
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

    def _wrap_method(self, cls: Any, method_name: str, key: str, tensor_arg: int | None = None, set_phase: bool = False):
        original = getattr(cls, method_name)
        collector = self

        @functools.wraps(original)
        def wrapped(self_obj, *args, **kwargs):
            phase = None
            if tensor_arg is not None and tensor_arg < len(args):
                phase = collector.phase_from_tensor(args[tensor_arg])
            old_phase = collector.current_phase
            old_rms_slot = collector._decoder_rms_slot
            if set_phase and phase is not None:
                collector.current_phase = phase
                if key == "decoder_layer":
                    collector._decoder_rms_slot = 0
            try:
                return collector._recorded_call(key, original, self_obj, *args, phase=phase, **kwargs)
            finally:
                collector.current_phase = old_phase
                collector._decoder_rms_slot = old_rms_slot

        self._patch(cls, method_name, wrapped)

    def _wrap_module_function(self, module: Any, fn_name: str, key: str):
        original = getattr(module, fn_name)
        collector = self

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            return collector._recorded_call(key, original, *args, **kwargs)

        self._patch(module, fn_name, wrapped)

    def install(self) -> None:
        if self._installed:
            return
        # Import after vLLM plugin registration/model classes are available.
        from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention
        from vllm.models.deepseek_v4_1.amd.model import DeepseekV4DecoderLayer, DeepseekV4MoE
        from vllm.models.deepseek_v4.amd.model import DeepseekV4MLP
        from vllm.models.deepseek_v4_1.common.engram import Engram
        from vllm.models.deepseek_v4_1.common.disk_engram import DiskAffineEngramEmbedding
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm_gguf_plugin.quantization.fused_moe import GGUFMoEMethod
        import vllm.models.deepseek_v4_1.amd.model as dsv41_amd_model
        import vllm_gguf_plugin.quantization as gguf_quant
        import vllm_gguf_plugin.ops as gguf_ops
        import vllm_gguf_plugin.quantization.fused_moe as gguf_fused_moe_mod
        import vllm.model_executor.layers.fused_moe.fused_moe as vllm_fused_moe_mod
        import vllm.model_executor.layers.fused_moe.runner.moe_runner as moe_runner
        import vllm.models.deepseek_v4_1.amd.rocm as rocm_mod
        import vllm.models.deepseek_v4_1.common.engram as engram_mod
        import vllm.model_executor.layers.linear as linear_mod

        # Disjoint high-level layer categories.  Decoder total is used only to
        # compute the residual/HC/norm remainder after subtracting attention,
        # MoE and Engram injection.
        self._wrap_method(DeepseekV4DecoderLayer, "forward", "decoder_layer", tensor_arg=0, set_phase=True)
        self._wrap_method(DeepseekV4Attention, "forward", "attention", tensor_arg=1)
        self._wrap_method(DeepseekV4MoE, "forward", "moe_total", tensor_arg=0)
        self._wrap_method(DeepseekV4MLP, "forward", "shared_mlp", tensor_arg=0)
        self._wrap_method(Engram, "forward", "engram_inject", tensor_arg=0)

        # Direct mHC decomposition.  DeepseekV4DecoderLayer imported these
        # functions by name, so patch the AMD model module that the live layer
        # actually calls.  The profiled implementation is deliberately the
        # same operation/order as the pinned reference; it only brackets
        # asynchronous event pairs and never synchronizes inside an operator.
        original_mhc_pre = dsv41_amd_model.mhc_pre_delayed_torch
        collector = self

        @functools.wraps(original_mhc_pre)
        def mhc_pre_profiled(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            pre_mix=None,
            x=None,
        ):
            if not collector.active:
                return original_mhc_pre(
                    residual,
                    fn,
                    hc_scale,
                    hc_base,
                    rms_eps,
                    hc_pre_eps,
                    hc_sinkhorn_eps,
                    hc_post_mult_value,
                    sinkhorn_repeat,
                    pre_mix=pre_mix,
                    x=x,
                )
            phase = collector.current_phase

            def impl():
                hc_mult = residual.shape[1]

                def projection_rms():
                    xf = (residual.flatten(1) if x is None else x).float()
                    mixes = (xf @ fn.t()) * torch.rsqrt(
                        xf.square().mean(-1, keepdim=True) + rms_eps
                    )
                    return mixes

                mixes = collector._recorded_call(
                    "mhc_pre_projection_rms", projection_rms, phase=phase
                )

                def prepost_coeff():
                    pre = (
                        torch.sigmoid(
                            mixes[:, :hc_mult] * hc_scale[0]
                            + hc_base[:hc_mult]
                        )
                        + hc_pre_eps
                    )
                    post = (
                        torch.sigmoid(
                            mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1]
                            + hc_base[hc_mult : 2 * hc_mult]
                        )
                        * hc_post_mult_value
                    )
                    return pre, post

                pre, post = collector._recorded_call(
                    "mhc_pre_prepost_coeff", prepost_coeff, phase=phase
                )

                def softmax_seed():
                    comb = (
                        mixes[:, 2 * hc_mult :].view(-1, hc_mult, hc_mult)
                        * hc_scale[2]
                    )
                    comb = comb + hc_base[2 * hc_mult :].view(
                        1, hc_mult, hc_mult
                    )
                    return torch.softmax(comb, dim=-1) + hc_sinkhorn_eps

                comb = collector._recorded_call(
                    "mhc_pre_softmax", softmax_seed, phase=phase
                )

                def sinkhorn():
                    out = comb / (
                        comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps
                    )
                    for _ in range(sinkhorn_repeat - 1):
                        out = out / (
                            out.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps
                        )
                        out = out / (
                            out.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps
                        )
                    return out

                comb_out = collector._recorded_call(
                    "mhc_pre_sinkhorn", sinkhorn, phase=phase
                )

                def collapse():
                    return (
                        residual[:, 0]
                        if pre_mix is None
                        else (
                            pre_mix.unsqueeze(-1) * residual.float()
                        ).sum(dim=1).to(residual.dtype)
                    )

                layer_input = collector._recorded_call(
                    "mhc_pre_collapse", collapse, phase=phase
                )
                return post.unsqueeze(-1), comb_out, layer_input, pre

            return collector._recorded_call(
                "mhc_pre_total", impl, phase=phase
            )

        self._patch(dsv41_amd_model, "mhc_pre_delayed_torch", mhc_pre_profiled)

        original_mhc_post = dsv41_amd_model.mhc_post_torch

        @functools.wraps(original_mhc_post)
        def mhc_post_profiled(x, residual, post_layer_mix, comb_res_mix):
            if not collector.active:
                return original_mhc_post(
                    x, residual, post_layer_mix, comb_res_mix
                )
            phase = collector.current_phase

            def impl():
                mixed_residual = collector._recorded_call(
                    "mhc_post_einsum",
                    lambda: torch.einsum(
                        "...ij,...ih->...jh",
                        comb_res_mix.to(torch.float32),
                        residual.to(torch.float32),
                    ),
                    phase=phase,
                )
                post_term = collector._recorded_call(
                    "mhc_post_term",
                    lambda: post_layer_mix.to(torch.float32)
                    * x.unsqueeze(-2).to(torch.float32),
                    phase=phase,
                )
                return collector._recorded_call(
                    "mhc_post_add_cast",
                    lambda: (mixed_residual + post_term).to(residual.dtype),
                    phase=phase,
                )

            return collector._recorded_call(
                "mhc_post_total", impl, phase=phase
            )

        self._patch(dsv41_amd_model, "mhc_post_torch", mhc_post_profiled)

        # DeepSeek layer-local hidden-size RMSNorms.  Decoder forward calls the
        # attention norm first and the FFN norm second.  Smaller q/kv RMSNorms
        # inside attention are intentionally excluded from this bucket.
        original_rms_forward = RMSNorm.forward

        @functools.wraps(original_rms_forward)
        def rms_forward_profiled(self_obj, *args, **kwargs):
            if not collector.active or not args:
                return original_rms_forward(self_obj, *args, **kwargs)
            x_arg = args[0]
            slot = collector._decoder_rms_slot
            if (
                isinstance(x_arg, torch.Tensor)
                and int(getattr(self_obj, "hidden_size", -1)) == 5120
                and slot is not None
                and collector.current_phase in {"decode", "prefill"}
            ):
                if slot == 0:
                    key = "attn_norm"
                elif slot == 1:
                    key = "ffn_norm"
                else:
                    key = "rmsnorm_h5120_extra"
                collector._decoder_rms_slot = slot + 1
                return collector._recorded_call(
                    key,
                    original_rms_forward,
                    self_obj,
                    *args,
                    phase=collector.current_phase,
                    **kwargs,
                )
            return original_rms_forward(self_obj, *args, **kwargs)

        self._patch(RMSNorm, "forward", rms_forward_profiled)

        # Disk Engram is intentionally wall-timed too: this function contains
        # GPU->CPU and CPU->GPU blocking transfers plus the affine-row cache.
        original_lookup = DiskAffineEngramEmbedding.lookup
        collector = self

        @functools.wraps(original_lookup)
        def lookup_wrapped(self_obj, indices, out, background=False):
            ph = collector.phase_from_tensor(indices)
            key = (int(self_obj.layer_id), int(self_obj.tp_rank))
            if collector.active:
                c = self_obj.cache
                collector.engram_cache_before.setdefault(key, (int(c.hits), int(c.misses), int(c.rows_read)))
            result = collector._recorded_call(
                "engram_disk_lookup", original_lookup, self_obj, indices, out, background, phase=ph
            )
            if collector.active:
                c = self_obj.cache
                collector.engram_cache_after[key] = (int(c.hits), int(c.misses), int(c.rows_read))
            return result

        self._patch(DiskAffineEngramEmbedding, "lookup", lookup_wrapped)

        # Routed expert call (including the current global->local EP masking).
        original_apply = GGUFMoEMethod.apply

        @functools.wraps(original_apply)
        def apply_wrapped(method_self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
            if collector.active and layer.expert_map is not None:
                mapped = layer.expert_map[topk_ids.to(torch.long)]
                collector.remote_route_tensors.append((mapped < 0).sum())
                collector.total_routes += int(topk_ids.numel())
            ph = collector.phase_from_tensor(x)
            return collector._recorded_call(
                "routed_apply",
                original_apply,
                method_self,
                layer,
                x,
                topk_weights,
                topk_ids,
                shared_experts,
                shared_experts_input,
                phase=ph,
            )

        self._patch(GGUFMoEMethod, "apply", apply_wrapped)

        # The package-level custom-op handle is imported inside GGUFMoEMethod.apply.
        self._wrap_module_function(gguf_quant, "fused_moe_gguf", "routed_kernel")

        # Decode M=1 routed-MoE internals.  These are nested diagnostics inside
        # routed_kernel.  ggml_moe_a8_vec still goes through moe_align_block_size
        # and the generic Triton kernels when VLLM_GGUF_USE_CUDA=0.
        original_vec = gguf_ops.ggml_moe_a8_vec

        @functools.wraps(original_vec)
        def vec_wrapped(*args, **kwargs):
            top_k = kwargs.get("top_k", args[3] if len(args) > 3 else None)
            key = "moe_gate_up_vec" if int(top_k or 0) > 1 else "moe_down_vec"
            return collector._recorded_call(key, original_vec, *args, **kwargs)

        self._patch(gguf_ops, "ggml_moe_a8_vec", vec_wrapped)
        self._wrap_module_function(vllm_fused_moe_mod, "moe_align_block_size", "moe_align")
        self._wrap_module_function(gguf_fused_moe_mod, "apply_moe_activation", "moe_activation")
        self._wrap_module_function(gguf_ops, "moe_sum", "moe_sum")

        # Known DS41 collectives.  These timings are nested diagnostics and are
        # not added to the top-level breakdown a second time.
        for module, name, key in (
            (moe_runner, "tensor_model_parallel_all_reduce", "collective_moe_allreduce"),
            (rocm_mod, "tensor_model_parallel_all_reduce", "collective_attention_allreduce"),
            (engram_mod, "tensor_model_parallel_all_gather", "collective_engram_allgather"),
            (linear_mod, "tensor_model_parallel_all_reduce", "collective_linear_allreduce"),
        ):
            if hasattr(module, name):
                self._wrap_module_function(module, name, key)

        self._installed = True

    def enable(self) -> None:
        self.active = True

    def disable(self) -> None:
        self.active = False

    def summarize(self, completion_tokens: int, decode_span_s: float | None, *, output: Path | None = None) -> dict[str, Any]:
        # One synchronization for the whole profiled request; never inside an op.
        torch.cuda.synchronize()
        gpu_ms: dict[str, float] = {}
        for name, pairs in self.events.items():
            gpu_ms[name] = float(sum(a.elapsed_time(b) for a, b in pairs))
        wall_ms = {k: v / 1e6 for k, v in self.wall_ns.items()}

        remote_routes = 0
        if self.remote_route_tensors:
            remote_routes = int(torch.stack([x.to(torch.int64) for x in self.remote_route_tensors]).sum().item())

        decode_steps = max(0, int(completion_tokens) - 1)
        def g(name: str) -> float:
            return gpu_ms.get(f"decode.{name}", 0.0)
        decoder = g("decoder_layer")
        attn = g("attention")
        moe = g("moe_total")
        engram = g("engram_inject")
        # Keep this raw: a negative value would be a decomposition problem and
        # must not be hidden behind max(0).  The direct mHC/RMS measurements
        # below are nested inside this remainder.
        other = decoder - attn - moe - engram
        top = {
            "attention": attn,
            "moe": moe,
            "engram_inject": engram,
            "other_layer_hc_norm_residual": other,
        }
        top_sum = sum(top.values())
        per_token = {k: (v / decode_steps if decode_steps else None) for k, v in top.items()}

        mhc_pre_parts = {
            "projection_rms": g("mhc_pre_projection_rms"),
            "prepost_coeff": g("mhc_pre_prepost_coeff"),
            "softmax": g("mhc_pre_softmax"),
            "sinkhorn": g("mhc_pre_sinkhorn"),
            "collapse": g("mhc_pre_collapse"),
        }
        mhc_post_parts = {
            "einsum": g("mhc_post_einsum"),
            "post_term": g("mhc_post_term"),
            "add_cast": g("mhc_post_add_cast"),
        }
        mhc_pre_total = g("mhc_pre_total")
        mhc_post_total = g("mhc_post_total")
        attn_norm = g("attn_norm")
        ffn_norm = g("ffn_norm")
        mhc_pre_sub_sum = sum(mhc_pre_parts.values())
        mhc_post_sub_sum = sum(mhc_post_parts.values())
        direct_hc_norm = mhc_pre_total + mhc_post_total + attn_norm + ffn_norm
        other_after_direct = other - direct_hc_norm
        hc_direct = {
            "mhc_pre_total": mhc_pre_total,
            "mhc_post_total": mhc_post_total,
            "attn_norm": attn_norm,
            "ffn_norm": ffn_norm,
            "mhc_pre_subparts": mhc_pre_parts,
            "mhc_post_subparts": mhc_post_parts,
            "mhc_pre_internal_gap": mhc_pre_total - mhc_pre_sub_sum,
            "mhc_post_internal_gap": mhc_post_total - mhc_post_sub_sum,
            "direct_hc_norm_sum": direct_hc_norm,
            "other_after_direct_hc_norm": other_after_direct,
        }

        nested = {
            k.removeprefix("decode."): v
            for k, v in gpu_ms.items()
            if k.startswith("decode.") and k.removeprefix("decode.") not in {
                "decoder_layer", "attention", "moe_total", "engram_inject"
            }
        }
        engram_cache = {}
        for key, before in self.engram_cache_before.items():
            after = self.engram_cache_after.get(key, before)
            engram_cache[f"layer{key[0]}_rank{key[1]}"] = {
                "hits_delta": after[0] - before[0],
                "misses_delta": after[1] - before[1],
                "rows_read_delta": after[2] - before[2],
            }

        result = {
            "schema": "ds41-perf-profile-v1",
            "completion_tokens": int(completion_tokens),
            "decode_steps": decode_steps,
            "decode_span_s": decode_span_s,
            "observed_decode_ms_per_token": (
                decode_span_s * 1000 / decode_steps if decode_span_s is not None and decode_steps else None
            ),
            "gpu_top_level_ms_total": top,
            "gpu_top_level_ms_per_decode_token": per_token,
            "gpu_top_level_sum_ms": top_sum,
            "gpu_top_level_sum_ms_per_decode_token": top_sum / decode_steps if decode_steps else None,
            "gpu_hc_norm_breakdown_ms_total": hc_direct,
            "gpu_hc_norm_breakdown_ms_per_decode_token": {
                "mhc_pre_total": mhc_pre_total / decode_steps if decode_steps else None,
                "mhc_post_total": mhc_post_total / decode_steps if decode_steps else None,
                "attn_norm": attn_norm / decode_steps if decode_steps else None,
                "ffn_norm": ffn_norm / decode_steps if decode_steps else None,
                "mhc_pre_projection_rms": mhc_pre_parts["projection_rms"] / decode_steps if decode_steps else None,
                "mhc_pre_prepost_coeff": mhc_pre_parts["prepost_coeff"] / decode_steps if decode_steps else None,
                "mhc_pre_softmax": mhc_pre_parts["softmax"] / decode_steps if decode_steps else None,
                "mhc_pre_sinkhorn": mhc_pre_parts["sinkhorn"] / decode_steps if decode_steps else None,
                "mhc_pre_collapse": mhc_pre_parts["collapse"] / decode_steps if decode_steps else None,
                "mhc_pre_internal_gap": (mhc_pre_total - mhc_pre_sub_sum) / decode_steps if decode_steps else None,
                "mhc_post_einsum": mhc_post_parts["einsum"] / decode_steps if decode_steps else None,
                "mhc_post_term": mhc_post_parts["post_term"] / decode_steps if decode_steps else None,
                "mhc_post_add_cast": mhc_post_parts["add_cast"] / decode_steps if decode_steps else None,
                "mhc_post_internal_gap": (mhc_post_total - mhc_post_sub_sum) / decode_steps if decode_steps else None,
                "direct_hc_norm_sum": direct_hc_norm / decode_steps if decode_steps else None,
                "other_after_direct_hc_norm": other_after_direct / decode_steps if decode_steps else None,
            },
            "all_gpu_event_ms": gpu_ms,
            "cpu_wrapper_wall_ms": wall_ms,
            "calls": dict(self.calls),
            "remote_routes": remote_routes,
            "total_routes": self.total_routes,
            "remote_route_fraction": remote_routes / self.total_routes if self.total_routes else None,
            "engram_cache": engram_cache,
            "notes": [
                "Top-level attention/moe/engram/other are derived from nested decoder events and are not double-counted.",
                "other_layer_hc_norm_residual is the raw decoder remainder; it is intentionally not clamped to zero.",
                "Direct mHC/RMS timings are nested inside that remainder. mhc_*_internal_gap is total-event elapsed minus the sum of sequential sub-events and can expose launch/idle gaps.",
                "Collective/routed/shared timings are nested diagnostics; do not add them again to top-level totals.",
                "CPU wrapper wall time is launch/blocking time, not GPU execution time; Engram disk lookup wall is intentionally useful for synchronous transfer/SSD stalls.",
            ],
        }
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            tmp = output.with_suffix(output.suffix + ".tmp")
            tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            tmp.replace(output)
        return result
