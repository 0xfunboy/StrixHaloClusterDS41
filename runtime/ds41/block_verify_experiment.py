"""Opt-in, weight-free V2 replay bridge for the bounded DS41 block experiment.

Importing this module does not import torch/vLLM or change production behavior.
Call install_bridge() before constructing LLM, then use SPECULATIVE_CONFIG.
All tensor-to-host checks and proposal work are diagnostic overhead, not model
throughput. There is no scheduler, rejection, attention, cache, or math override.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
import importlib
import time
from typing import Any

CLASS_PATH = "runtime.ds41.block_verify_experiment.ReplaySpeculator"
SPECULATIVE_CONFIG = {
    "method": "custom_class", "model": CLASS_PATH, "num_speculative_tokens": 3,
}


@dataclass
class BridgeState:
    oracle_tokens: list[int] | None = None
    prompt_length: int = 0
    desired_k: int = 0
    activation_output_tokens: int = 8
    corrupt_draft_index: int | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    next_counts: dict[str, int] = field(default_factory=dict)
    failed_requests: set[str] = field(default_factory=set)
    injected_requests: set[str] = field(default_factory=set)

    @property
    def fidelity_failed(self) -> bool:
        return bool(self.failed_requests)


_STATE = BridgeState()
_INSTALLED = False


def configure(oracle_tokens=None, prompt_length=0, desired_k=0,
              activation_output_tokens=8, corrupt_draft_index=None) -> BridgeState:
    """Reset between generate calls. Oracle includes prompt and continuation."""
    if desired_k not in (0, 1, 3) or prompt_length < 0 or activation_output_tokens < 0:
        raise ValueError("Require K=0/1/3 and nonnegative prompt/activation lengths")
    oracle = None if oracle_tokens is None else [int(x) for x in oracle_tokens]
    if desired_k and oracle is None:
        raise ValueError("Replay requires a recorded oracle")
    if oracle is not None and (len(oracle) < prompt_length or min(oracle, default=0) < 0):
        raise ValueError("Invalid oracle tokens or prompt length")
    if corrupt_draft_index is not None and not 0 <= corrupt_draft_index < desired_k:
        raise ValueError("Corruption index must identify an active draft")
    global _STATE
    _STATE = BridgeState(oracle, prompt_length, desired_k,
                         activation_output_tokens, corrupt_draft_index)
    return _STATE


def get_state() -> BridgeState:
    return _STATE


def proposal_plan(*, req_id, slot, input_tokens, positions, seq_len,
                  num_sampled, num_rejected, last_sampled, vocab_size):
    """Pure CPU plan. last_sampled is indexed by request slot, not batch row."""
    state = _STATE
    qlen = len(input_tokens)
    if (not qlen or len(positions) != qlen or positions[0] < 0
            or positions != list(range(positions[0], positions[0] + qlen))
            or seq_len != positions[-1] + 1
            or not 0 <= num_rejected < qlen or num_sampled < 0):
        raise ValueError("Inconsistent target query positions or rejection counts")
    committed_end = seq_len - num_rejected
    offset = committed_end + int(num_sampled > 0)
    output_length = max(0, offset - state.prompt_length)
    errors = []
    oracle = state.oracle_tokens
    if oracle is not None:
        accepted_inputs = input_tokens[:qlen - num_rejected]
        if oracle[positions[0]:committed_end] != accepted_inputs:
            errors.append("committed_input_prefix_mismatch")
        if num_sampled > 0 and (committed_end >= len(oracle)
                               or oracle[committed_end] != last_sampled):
            errors.append("sampled_token_mismatch")
    if errors:
        state.failed_requests.add(req_id)
    active = (num_sampled > 0 and output_length >= state.activation_output_tokens
              and req_id not in state.failed_requests and oracle is not None)
    # Acceptance also emits one target bonus token. Reserve that oracle row,
    # especially when a rejection has shifted the final block's alignment.
    k = min(state.desired_k, max(0, len(oracle) - offset - 1)) if active else 0
    drafts = oracle[offset:offset + k] if k else []
    injected = False
    ci = state.corrupt_draft_index
    if ci is not None and ci < k and req_id not in state.injected_requests:
        if vocab_size < 2:
            raise ValueError("Corruption probe requires at least two vocabulary tokens")
        drafts[ci] = (drafts[ci] + 1) % vocab_size
        state.injected_requests.add(req_id)
        injected = True
    if any(not 0 <= token < vocab_size for token in drafts):
        raise ValueError("Oracle token outside target vocabulary")
    state.next_counts[req_id] = k
    event = dict(req_id=req_id, slot=int(slot), query_start=int(positions[0]),
                 query_len=qlen, seq_len=int(seq_len), num_sampled=int(num_sampled),
                 num_rejected=int(num_rejected), oracle_offset=int(offset),
                 output_length=int(output_length), desired_k=state.desired_k,
                 emitted_k=k, drafts=list(drafts), injected=injected,
                 prefix_errors=errors, fidelity_failed=req_id in state.failed_requests)
    state.events.append(event)
    return drafts + [0] * (3 - len(drafts)), event


def truncate_draft_token_ids(value):
    """Only truncate scheduler-facing lists. GPU proposals always retain Kmax3."""
    if value is None:
        return None
    counts = _STATE.next_counts
    return type(value)(value.req_ids, [
        list(tokens[:counts.get(req_id, 0)])
        for req_id, tokens in zip(value.req_ids, value.draft_token_ids, strict=True)
    ])


class ReplaySpeculator:
    """Bound to the pinned BaseSpeculator by install_bridge, without a drafter."""
    supports_mm_inputs = False
    draft_logits = None

    def __init__(self, vllm_config, device):
        self.device = device
        self.vocab_size = vllm_config.model_config.get_vocab_size()

    def init_cudagraph_manager(self, cudagraph_mode):
        pass

    def capture(self):
        pass

    def propose(self, input_batch, attn_metadata, slot_mappings, last_hidden_states,
                aux_hidden_states, num_sampled, num_rejected, last_sampled,
                next_prefill_tokens, temperature, seeds, dp_sync=None,
                dummy_run=False, skip_attn_for_dummy_run=False, mm_inputs=None,
                is_profile=False):
        import torch
        started = time.perf_counter()
        if dummy_run or is_profile:
            return torch.zeros((input_batch.num_reqs, 3), dtype=torch.int64,
                               device=self.device)
        # The profile branch must not advance replay state. Real calls are
        # intentionally synchronous to make exact token/position checks auditable.
        sampled, rejected = num_sampled.tolist(), num_rejected.tolist()
        seq_lens = input_batch.seq_lens.tolist()
        tokens, positions = input_batch.input_ids.tolist(), input_batch.positions.tolist()
        last = last_sampled.tolist()
        rows, records = [], []
        for i, req_id in enumerate(input_batch.req_ids):
            slot = int(input_batch.idx_mapping_np[i])
            if slot < 0:
                raise ValueError("Unexpected padded request in replay bridge")
            begin, end = map(int, input_batch.query_start_loc_np[i:i + 2])
            sampled_token = last[slot]
            if isinstance(sampled_token, list):
                sampled_token = sampled_token[0]
            row, record = proposal_plan(
                req_id=req_id, slot=slot, input_tokens=tokens[begin:end],
                positions=positions[begin:end], seq_len=seq_lens[i],
                num_sampled=sampled[i], num_rejected=rejected[i],
                last_sampled=sampled_token, vocab_size=self.vocab_size)
            rows.append(row)
            records.append(record)
        result = torch.tensor(rows, dtype=torch.int64, device=self.device).reshape(-1, 3)
        elapsed = time.perf_counter() - started
        for record in records:
            record["proposal_batch_wall_s"] = elapsed
        return result


def _matches(config):
    spec = getattr(config, "speculative_config", None)
    return (spec is not None and spec.method == "custom_class"
            and spec.model == CLASS_PATH and spec.num_speculative_tokens == 3)


class _OptionalHiddenGetter:
    """Make hasattr false for this diagnostic target's unused None getter only.

    V2 otherwise slices None in both profile and real propose. Ordinary targets
    retain their original method; no forward wrapper or extra tensor is needed.
    """
    def __init__(self, original):
        self.original = original

    def __get__(self, instance, owner):
        if instance is None:
            return self.original
        bound = self.original.__get__(instance, owner)
        if getattr(instance, "_ds41_block_bridge", False) and bound() is None:
            raise AttributeError("No MTP hidden buffer for diagnostic replay")
        return bound


def install_bridge():
    """Process-local exact-class allowances. Call once before constructing LLM."""
    global _INSTALLED, ReplaySpeculator
    if _INSTALLED:
        return
    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator
    runner = importlib.import_module("vllm.v1.worker.gpu.model_runner")
    factory = importlib.import_module("vllm.v1.worker.gpu.spec_decode")
    # The registered target is the VL outer wrapper, even for text-only runs.
    # Its ordinary getter delegates to language_model. Mask only this outer
    # attribute: masking the child would make outer hasattr true, then fail
    # inside the delegated call before V2 can retain ordinary hidden_states.
    target = importlib.import_module("vllm.models.deepseek_v4_1.amd.vl_model")
    cls = target.DeepseekV41ForCausalLM
    original_features = VllmConfig._get_v2_model_runner_unsupported_features
    original_factory = runner.init_speculator
    original_take = runner.GPUModelRunner.take_draft_token_ids
    original_init = cls.__init__

    @wraps(original_features)
    def features(config):
        result = original_features(config)
        if _matches(config):
            result = [x for x in result if x != "speculative method 'custom_class'"]
        return result

    ReplaySpeculator = type("ReplaySpeculator", (ReplaySpeculator, BaseSpeculator), {})

    @wraps(original_factory)
    def init(config, device):
        if not _matches(config):
            return original_factory(config, device)
        if (not config.use_v2_model_runner or config.scheduler_config.async_scheduling
                or config.speculative_config.parallel_drafting):
            raise ValueError("Replay requires synchronous causal V2 execution")
        return ReplaySpeculator(config, device)

    @wraps(original_take)
    def take(self):
        result = original_take(self)
        return truncate_draft_token_ids(result) if _matches(self.vllm_config) else result

    @wraps(original_init)
    def target_init(self, *, vllm_config, **kwargs):
        original_init(self, vllm_config=vllm_config, **kwargs)
        self._ds41_block_bridge = _matches(vllm_config)

    VllmConfig._get_v2_model_runner_unsupported_features = features
    runner.init_speculator = factory.init_speculator = init
    runner.GPUModelRunner.take_draft_token_ids = take
    cls.__init__ = target_init
    cls.get_mtp_target_hidden_states = _OptionalHiddenGetter(cls.get_mtp_target_hidden_states)
    _INSTALLED = True
