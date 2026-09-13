#!/usr/bin/env python3
"""CPU-only fixtures for WO_B correctness and promotion rejection gates."""
import copy
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    "wob_validator", Path(__file__).with_name("validate-ds41-wob-ab.py"))
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)

CODE = '''def first_missing_positive(nums):
    n = len(nums)
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            j = nums[i] - 1
            nums[i], nums[j] = nums[j], nums[i]
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    return n + 1
'''


def fixture():
    results = []
    for suffix, prompt, cap, ignore in VALIDATOR.REQUESTS:
        candidate = suffix.startswith("candidate-")
        count = cap if ignore else 12
        stats = {"llmm1_calls": count if candidate else 0,
                 "llmm1_tokens": count if candidate else 0,
                 "fallback_calls": 1 if candidate else 0,
                 "fallback_reasons": {"tokens_not_1": 1} if candidate else {}}
        results.append({
            "label": f"wob-ab-{suffix}", "token_ids": list(range(count)),
            "completion_token_count": count, "finished": True,
            "finish_reason": "length" if ignore else "stop",
            "text": {"smoke": "323", "coding": CODE,
                     "json": '{"total":17,"valid_ids":["a","c"]}',
                     "reasoning-high": "Reasoning.</think>10"}.get(prompt, "speed"),
            "request_protocol": {"max_tokens": cap, "ignore_eos": ignore,
                                 "prompt_tokens_sha256": VALIDATOR.PROMPT_HASHES[prompt]},
            "runtime_switches": {**dict.fromkeys(VALIDATOR.PROMOTED_SWITCHES, "1"),
                                 "DS41_ATTN_WOB_LLMM1": "1" if candidate else "0"},
            "wob_stats": stats,
            "derived": {"decode_tps_first_to_last": 11.0 if candidate else 10.0},
            "client": {"ttft_s": 1.0, "wall_s": 15.0},
        })
    doc = {"rank": 0, "world_size": 2, "status": "ATTN_WOB_AB_COMPLETE",
           "order": list(VALIDATOR.ORDER), "native_hip_identity": {"sha256": "a" * 64},
           "results": results, "wob_stats": VALIDATOR.stats_total({r["label"]: r for r in results})}
    other = copy.deepcopy(doc)
    other["rank"] = 1
    return [doc, other]


class ValidatorTests(unittest.TestCase):
    def test_success(self):
        result = VALIDATOR.validate(*fixture())
        self.assertTrue(result["promotion_eligible"])
        self.assertEqual(result["correctness_status"], "PASS")
        self.assertEqual(result["speed_status"], "PASS")

    def test_correctness_pass_with_insufficient_speed(self):
        docs = fixture()
        for doc in docs:
            for row in doc["results"]:
                if "candidate" in row["label"]:
                    row["derived"]["decode_tps_first_to_last"] = 10.4
        result = VALIDATOR.validate(*docs)
        self.assertEqual(result["correctness_status"], "PASS")
        self.assertEqual(result["speed_status"], "FAIL")
        self.assertFalse(result["promotion_eligible"])

    def test_one_losing_pair_rejects_high_mean_gain(self):
        docs = fixture()
        for doc in docs:
            doc["results"][3]["derived"]["decode_tps_first_to_last"] = 9.9
            doc["results"][4]["derived"]["decode_tps_first_to_last"] = 13.0
        result = VALIDATOR.validate(*docs)
        self.assertTrue(result["promotion"]["mean_gain_pass"])
        self.assertFalse(result["promotion_eligible"])

    def test_all_quality_tasks_require_natural_completion_on_both_ranks(self):
        for rank in (0, 1):
            for index in range(8, 12):
                with self.subTest(rank=rank, index=index):
                    docs = fixture()
                    docs[rank]["results"][index]["finish_reason"] = "length"
                    self.assertEqual(VALIDATOR.validate(*docs)["correctness_status"], "FAIL")

    def test_identity_order_and_native_library_fail_closed(self):
        for change in (
            lambda d: d[1]["results"][0]["token_ids"].__setitem__(0, 123),
            lambda d: d[1]["results"].reverse(),
            lambda d: d[1]["results"].pop(),
            lambda d: d[1]["results"].append(copy.deepcopy(d[1]["results"][0])),
            lambda d: d[1].__setitem__("rank", 0),
            lambda d: d[0].__setitem__("native_hip_identity", {}),
            lambda d: [doc.__setitem__("native_hip_identity", {}) for doc in d],
            lambda d: d[1]["native_hip_identity"].__setitem__("sha256", "b" * 64),
        ):
            docs = fixture()
            change(docs)
            self.assertEqual(VALIDATOR.validate(*docs)["correctness_status"], "FAIL")

    def test_runtime_and_dispatch_evidence_required_for_every_request(self):
        for rank in (0, 1):
            for index in range(12):
                with self.subTest(rank=rank, index=index):
                    docs = fixture()
                    docs[rank]["results"][index].pop("wob_stats")
                    self.assertFalse(VALIDATOR.validate(*docs)["wob_execution"]["pass"])
        for change in (
            lambda r: r["runtime_switches"].__setitem__("DS41_MHC_PROJECTION_RMS", "0"),
            lambda r: r["runtime_switches"].__setitem__("DS41_ATTN_WOB_LLMM1", "0"),
            lambda r: r["wob_stats"].__setitem__("llmm1_calls", 0),
            lambda r: r["wob_stats"].__setitem__("fallback_reasons", {"shape_contract": 1}),
        ):
            docs = fixture()
            change(docs[1]["results"][3])
            self.assertEqual(VALIDATOR.validate(*docs)["correctness_status"], "FAIL")

    def test_protocol_and_metrics_rejected_on_both_ranks(self):
        for rank in (0, 1):
            for change in (
                lambda r: r["request_protocol"].__setitem__("max_tokens", 127),
                lambda r: r["request_protocol"].__setitem__("ignore_eos", False),
                lambda r: r["request_protocol"].__setitem__("prompt_tokens_sha256", "b" * 64),
                lambda r: r.__setitem__("completion_token_count", 127),
                lambda r: r["derived"].__setitem__("decode_tps_first_to_last", float("nan")),
            ):
                docs = fixture()
                change(docs[rank]["results"][3])
                self.assertFalse(VALIDATOR.validate(*docs)["promotion_eligible"])


if __name__ == "__main__":
    unittest.main()
