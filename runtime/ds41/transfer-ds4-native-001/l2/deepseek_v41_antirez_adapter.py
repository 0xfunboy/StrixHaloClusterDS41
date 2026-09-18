# SPDX-License-Identifier: Apache-2.0
"""GGUF weight adapter for DeepSeek V4.1 text-backbone artifacts.

The current V4.1 GGUF converter emits compact llama.cpp tensor names while the
vLLM ROCm model loader consumes the native DeepSeek checkpoint namespace. Keep
this translation explicit and fail closed. The calibrated Antirez/DS4 artifact
also embeds native Engram tensors; those are consumed by a separate bounded
provider and are never expanded through the ordinary target-weight iterator.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable

from vllm.logger import init_logger

from ..gguf_files import GGUFModelFiles
from ..weight_utils import get_gguf_tensor_names
from .base import BaseGGUFWeightsAdapter, GGUFWeight

logger = init_logger(__name__)

_TOP = {
    "token_embd": "embed.weight",
    "output_norm": "norm.weight",
    "output": "head.weight",
}

_LAYER = {
    "attn_kv": "attn.wkv.weight",
    "attn_kv_a_norm": "attn.kv_norm.weight",
    "attn_norm": "attn_norm.weight",
    "attn_output_a": "attn.wo_a.weight",
    "attn_output_b": "attn.wo_b.weight",
    "attn_q_a": "attn.wq_a.weight",
    "attn_q_a_norm": "attn.q_norm.weight",
    "attn_q_b": "attn.wq_b.weight",
    "attn_sinks": "attn.attn_sink",
    "attn_compressor_gate": "attn.compressor.wgate.weight",
    "attn_compressor_kv": "attn.compressor.wkv.weight",
    "attn_compressor_norm": "attn.compressor.norm.weight",
    "indexer.attn_k": "attn.indexer.wk.weight",
    "indexer.attn_q_b": "attn.indexer.wq_b.weight",
    "indexer.k_norm": "attn.indexer.k_norm.weight",
    "indexer.proj": "attn.indexer.weights_proj.weight",
    "ffn_down_exps": "ffn.experts.0.w2.weight",
    "ffn_gate_exps": "ffn.experts.0.w1.weight",
    "ffn_up_exps": "ffn.experts.0.w3.weight",
    "ffn_down_shexp": "ffn.shared_experts.w2.weight",
    "ffn_gate_shexp": "ffn.shared_experts.w1.weight",
    "ffn_up_shexp": "ffn.shared_experts.w3.weight",
    "ffn_gate_inp": "ffn.gate.weight",
    "exp_probs_b": "ffn.gate.bias",
    "exp_probs_b_vl": "ffn.gate.bias_vl",
    "ffn_norm": "ffn_norm.weight",
    "hc_attn_base": "hc_attn_base",
    "hc_attn_fn": "hc_attn_fn",
    "hc_attn_scale": "hc_attn_scale",
    "hc_ffn_base": "hc_ffn_base",
    "hc_ffn_fn": "hc_ffn_fn",
    "hc_ffn_scale": "hc_ffn_scale",
}

_BLOCK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")

_ANTIREZ_ENGRAM_TENSORS = {
    f"blk.{layer}.{leaf}"
    for layer in (1, 14)
    for leaf in (
        "engram_q_norm.weight",
        "engram_k_norm.weight",
        "engram_kv.weight",
        "engram_embd.weight",
    )
}


def _antirez_base_name(name: str) -> str:
    if name.endswith(".weight"):
        return name.removesuffix(".weight")
    if name.endswith(".bias"):
        return name.removesuffix(".bias")
    return name


def map_deepseek_v41_gguf_name(name: str) -> str | None:
    if name in _TOP:
        return _TOP[name]
    match = _BLOCK_RE.fullmatch(name)
    if match is None:
        return None
    layer, leaf = match.groups()
    target = _LAYER.get(leaf)
    if target is None:
        return None
    return f"layers.{layer}.{target}"


class DeepseekV41GGUFAdapter(BaseGGUFWeightsAdapter):
    """Map llama.cpp DeepSeek4/V4.1 GGUF tensors to native V4.1 names."""

    @classmethod
    def matches(cls, config) -> bool:
        return getattr(config, "model_type", None) == "deepseek_v41"

    @classmethod
    def architecture(cls, config) -> str | None:
        if not cls.matches(config):
            return None
        return "DeepseekV41ForCausalLM"

    def build_name_map(self, files: GGUFModelFiles, model_config) -> dict[str, str]:
        del model_config
        names = sorted(get_gguf_tensor_names(files.backbone))
        antirez = os.environ.get("DS41_ANTIREZ_Q2", "0") == "1"
        if antirez:
            missing = sorted(_ANTIREZ_ENGRAM_TENSORS.difference(names))
            if missing:
                raise RuntimeError(
                    "Antirez Q2 native-Engram contract missing tensors: "
                    f"{missing}"
                )

        mapped: dict[str, str] = {}
        unmapped: list[str] = []
        for name in names:
            if antirez and name in _ANTIREZ_ENGRAM_TENSORS:
                continue
            target = map_deepseek_v41_gguf_name(name)
            if target is None and antirez:
                target = map_deepseek_v41_gguf_name(_antirez_base_name(name))
            if target is None:
                unmapped.append(name)
            else:
                mapped[name] = target
        if unmapped:
            raise RuntimeError(
                f"Failed to map {len(unmapped)} DeepSeek V4.1 GGUF tensors: {unmapped}"
            )
        if antirez and len(mapped) + len(_ANTIREZ_ENGRAM_TENSORS) != len(names):
            raise RuntimeError(
                "Antirez Q2 tensor accounting mismatch: "
                f"mapped={len(mapped)} special={len(_ANTIREZ_ENGRAM_TENSORS)} "
                f"total={len(names)}"
            )
        logger.info(
            "Mapped all %d DeepSeek V4.1 target tensors%s",
            len(mapped),
            " (Antirez native Engram kept disk-backed)" if antirez else "",
        )
        return mapped

    def transform_weights(
        self, weights: Iterable[GGUFWeight], model_config
    ) -> Iterable[GGUFWeight]:
        del model_config
        phase_log = os.environ.get("DS41_LOAD_PHASE_LOG", "0") == "1"
        for name, weight in weights:
            if weight.ndim == 3 and ".experts.0." in name:
                if phase_log:
                    logger.info(
                        "DS41_LOAD expert_tensor begin name=%s shape=%s",
                        name,
                        tuple(weight.shape),
                    )
                for expert_id, expert_weight in enumerate(weight.unbind()):
                    expert_name = name.replace(
                        ".experts.0.", f".experts.{expert_id}."
                    )
                    yield expert_name, expert_weight
                if phase_log:
                    logger.info("DS41_LOAD expert_tensor end name=%s", name)
            else:
                yield name, weight
