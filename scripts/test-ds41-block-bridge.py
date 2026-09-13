#!/usr/bin/env python3
"""CPU-only replay offset, rollback, and exact-class V2 shim regression tests."""
from abc import ABC, abstractmethod
from collections import namedtuple
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.ds41 import block_verify_experiment as bridge

Drafts = namedtuple("Drafts", "req_ids draft_token_ids")
FAKE_TORCH = NS(int64=np.int64, zeros=lambda shape, dtype, device: np.zeros(shape, dtype),
                tensor=lambda rows, dtype, device: np.array(rows, dtype=dtype))


def plan(start=10, tokens=None, sampled=1, rejected=0, last=None, req="r"):
    tokens = [start] if tokens is None else tokens
    end = start + len(tokens)
    return bridge.proposal_plan(
        req_id=req, slot=4, input_tokens=tokens,
        positions=list(range(start, end)), seq_len=end, num_sampled=sampled,
        num_rejected=rejected, last_sampled=end - rejected if last is None else last,
        vocab_size=100)


class ReplayTest(unittest.TestCase):
    def setUp(self):
        bridge.configure(list(range(50)), prompt_length=4, desired_k=3,
                         activation_output_tokens=8)

    def test_activation_prefill_offset_and_fixed_shape(self):
        row, event = plan(start=0, tokens=[0, 1], sampled=0)
        self.assertEqual(row, [0, 0, 0])
        self.assertEqual(event["oracle_offset"], 2)
        row, event = plan(start=9)
        self.assertEqual(event["output_length"], 7)
        self.assertEqual(row, [0, 0, 0])
        row, event = plan(start=10)
        self.assertEqual(row, [12, 13, 14])
        self.assertEqual(event["output_length"], 8)
        self.assertEqual(event["oracle_offset"], 12)

    def test_full_accept_partial_and_full_rejection(self):
        row, event = plan(tokens=[10, 11, 12, 13], sampled=4)
        self.assertEqual(row, [15, 16, 17])
        self.assertEqual(event["prefix_errors"], [])
        # Rejected suffix may differ from the oracle and must not be committed.
        row, event = plan(tokens=[10, 11, 99, 98], sampled=2, rejected=2)
        self.assertEqual(row, [13, 14, 15])
        self.assertEqual(event["prefix_errors"], [])
        row, event = plan(tokens=[10, 99, 98, 97], sampled=1, rejected=3)
        self.assertEqual(row, [12, 13, 14])
        self.assertEqual(event["prefix_errors"], [])

    def test_single_corruption_and_request_local_fail_closed(self):
        state = bridge.configure(list(range(50)), 4, 3, 8, 1)
        row, event = plan()
        self.assertEqual(row, [12, 14, 14])
        self.assertTrue(event["injected"])
        row, event = plan(start=11, tokens=[11, 12, 14, 14], sampled=2, rejected=2)
        self.assertEqual(row, [14, 15, 16])
        self.assertFalse(event["injected"])
        row, event = plan(last=90)
        self.assertEqual(row, [0, 0, 0])
        self.assertEqual(event["prefix_errors"], ["sampled_token_mismatch"])
        self.assertTrue(state.fidelity_failed)
        self.assertEqual(plan()[1]["emitted_k"], 0)
        self.assertEqual(plan(req="fresh")[1]["emitted_k"], 3)
        # Reset between independent requests clears the failure and injection.
        bridge.configure(list(range(50)), 4, 1, 8)
        self.assertFalse(bridge.get_state().fidelity_failed)
        self.assertEqual(plan()[0], [12, 0, 0])

    def test_scheduler_only_truncation_unknown_request_and_exhaustion(self):
        value = Drafts(["r", "unknown"], [[-1, -1, -1], [-1, -1, -1]])
        for k in (0, 1, 3):
            bridge.configure(list(range(50)), 4, k, 8)
            row, _ = plan()
            result = bridge.truncate_draft_token_ids(value)
            self.assertEqual(list(map(len, result.draft_token_ids)), [k, 0])
            self.assertEqual(len(row), 3)
            self.assertEqual(value.draft_token_ids[0], [-1, -1, -1])
        bridge.configure(list(range(14)), 4, 3, 8)
        self.assertEqual(plan()[0], [12, 0, 0])
        self.assertEqual(bridge.get_state().next_counts["r"], 1)
        bridge.configure(list(range(13)), 4, 3, 8)
        self.assertEqual(plan()[0], [0, 0, 0])
        self.assertEqual(bridge.get_state().next_counts["r"], 0)
        self.assertIsNone(bridge.truncate_draft_token_ids(None))

    def test_shifted_tail_reserves_target_bonus_inside_oracle(self):
        # A rejection can leave two output tokens at the end of a Kmax3 run.
        row, event = plan(start=46)
        self.assertEqual(event["oracle_offset"], 48)
        self.assertEqual(row, [48, 0, 0])
        row, event = plan(start=47, tokens=[47, 48], sampled=2)
        self.assertEqual(event["oracle_offset"], 50)
        self.assertEqual(row, [0, 0, 0])
        self.assertEqual(event["prefix_errors"], [])
        self.assertFalse(bridge.get_state().fidelity_failed)

    def test_real_interface_slot_mapping_and_dummy_does_not_advance(self):
        adapter = bridge.ReplaySpeculator(NS(model_config=NS(get_vocab_size=lambda: 100)),
                                         "cpu")
        batch = NS(num_reqs=2, req_ids=["a", "b"], idx_mapping_np=np.array([4, 1]),
                   query_start_loc_np=np.array([0, 1, 5]),
                   seq_lens=np.array([11, 24]),
                   input_ids=np.array([10, 20, 21, 99, 98]),
                   positions=np.array([10, 20, 21, 22, 23]))
        last = np.full((6, 1), 99)
        last[4, 0], last[1, 0] = 11, 22
        args = (batch, {}, {}, None, None, np.array([1, 2]), np.array([0, 2]),
                last, None, None, None)
        with patch.dict(sys.modules, {"torch": FAKE_TORCH}):
            dummy = adapter.propose(*args, dummy_run=True)
            self.assertEqual(dummy.shape, (2, 3))
            self.assertEqual(bridge.get_state().events, [])
            real = adapter.propose(*args)
        np.testing.assert_array_equal(real, [[12, 13, 14], [23, 24, 25]])
        events = bridge.get_state().events
        self.assertEqual([e["slot"] for e in events], [4, 1])
        self.assertTrue(all(e["proposal_batch_wall_s"] >= 0 for e in events))

    def test_structural_errors_rejected(self):
        with self.assertRaises(ValueError):
            plan(rejected=1)
        with self.assertRaises(ValueError):
            bridge.configure(None, 4, 3)
        with self.assertRaises(ValueError):
            bridge.configure(list(range(50)), 4, 2)

    def test_outer_optional_getter_does_not_hide_delegated_child(self):
        class Child:
            buffer = None

            def get_mtp_target_hidden_states(self):
                return self.buffer

        class Outer:
            _ds41_block_bridge = True

            def __init__(self):
                self.language_model = Child()

            def get_mtp_target_hidden_states(self):
                return self.language_model.get_mtp_target_hidden_states()

        Outer.get_mtp_target_hidden_states = bridge._OptionalHiddenGetter(
            Outer.get_mtp_target_hidden_states)
        outer = Outer()
        self.assertFalse(hasattr(outer, "get_mtp_target_hidden_states"))
        self.assertTrue(hasattr(outer.language_model, "get_mtp_target_hidden_states"))
        self.assertIsNone(outer.language_model.get_mtp_target_hidden_states())
        marker = object()
        outer.language_model.buffer = marker
        self.assertIs(outer.get_mtp_target_hidden_states(), marker)

    def test_install_exact_allowance_and_optional_none_descriptor(self):
        class Config:
            def _get_v2_model_runner_unsupported_features(self):
                return ["speculative method 'custom_class'", "unrelated feature"]

        class Base(ABC):
            @abstractmethod
            def propose(self):
                pass

        class Runner:
            def take_draft_token_ids(self):
                return Drafts(["r"], [[-1, -1, -1]])

        class Child:
            def __init__(self, *, vllm_config, prefix=""):
                self.buffer = None

            def get_mtp_target_hidden_states(self):
                return self.buffer

        class Target:
            def __init__(self, *, vllm_config, prefix=""):
                self.language_model = Child(vllm_config=vllm_config, prefix=prefix)

            def get_mtp_target_hidden_states(self):
                return self.language_model.get_mtp_target_hidden_states()

        def config(model=bridge.CLASS_PATH, k=3):
            c = Config()
            c.speculative_config = NS(method="custom_class", model=model,
                                      num_speculative_tokens=k, parallel_drafting=False)
            c.use_v2_model_runner = True
            c.scheduler_config = NS(async_scheduling=False)
            c.model_config = NS(get_vocab_size=lambda: 100)
            return c

        modules = {}
        def module(name, **attrs):
            result = ModuleType(name)
            result.__dict__.update(attrs)
            modules[name] = result
            return result

        fallback = lambda config, device: "fallback"
        module("vllm.config", VllmConfig=Config)
        module("vllm.v1.worker.gpu.spec_decode.speculator", BaseSpeculator=Base)
        runner = module("vllm.v1.worker.gpu.model_runner", GPUModelRunner=Runner,
                        init_speculator=fallback)
        factory = module("vllm.v1.worker.gpu.spec_decode", init_speculator=fallback)
        module("vllm.models.deepseek_v4_1.amd.model", DeepseekV41LLMForCausalLM=Child)
        module("vllm.models.deepseek_v4_1.amd.vl_model", DeepseekV41ForCausalLM=Target)
        original_adapter = bridge.ReplaySpeculator
        with patch.dict(sys.modules, modules), patch.object(bridge, "_INSTALLED", False), \
                patch.object(bridge, "ReplaySpeculator", original_adapter):
            bridge.install_bridge()
            bridge.install_bridge()
            good, bad = config(), config("other.ReplaySpeculator")
            self.assertEqual(good._get_v2_model_runner_unsupported_features(),
                             ["unrelated feature"])
            self.assertEqual(len(bad._get_v2_model_runner_unsupported_features()), 2)
            self.assertEqual(len(config(k=1)._get_v2_model_runner_unsupported_features()), 2)
            self.assertIsInstance(runner.init_speculator(good, "cpu"), Base)
            self.assertEqual(factory.init_speculator(bad, "cpu"), "fallback")
            good_target, bad_target = Target(vllm_config=good), Target(vllm_config=bad)
            self.assertFalse(hasattr(good_target, "get_mtp_target_hidden_states"))
            self.assertIsNone(bad_target.get_mtp_target_hidden_states())
            self.assertIsNone(good_target.language_model.get_mtp_target_hidden_states())
            marker = object()
            good_target.language_model.buffer = marker
            self.assertIs(good_target.get_mtp_target_hidden_states(), marker)
            bridge.configure(list(range(50)), 4, 1, 8)
            plan()
            instance = Runner()
            instance.vllm_config = good
            self.assertEqual(instance.take_draft_token_ids().draft_token_ids, [[-1]])
            instance.vllm_config = bad
            self.assertEqual(instance.take_draft_token_ids().draft_token_ids, [[-1] * 3])
            good.scheduler_config.async_scheduling = True
            with self.assertRaises(ValueError):
                runner.init_speculator(good, "cpu")


if __name__ == "__main__":
    unittest.main()
