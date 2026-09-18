#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / ".vendor/vllm-dsv41"
REF = ROOT / "runtime/ds41/transfer-ds4-native-001/ds4-low-reference-ids.json"
MODEL = "/home/funboy/models/gguf/deepseek-v4.1-flash-mixedq2-densefix"

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.renderers.deepseek_v4 import DeepseekV4Renderer
from vllm.tokenizers.deepseek_v41 import DeepseekV41Tokenizer


def ids_sha(ids: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def request_params(messages: list[dict]) -> tuple[ChatCompletionRequest, object]:
    req = ChatCompletionRequest(
        model="DeepSeek-V4.1-Flash",
        messages=messages,
        reasoning_effort="low",
        chat_template_kwargs={"ds41_prompt_profile": "ds4-low-v1"},
        max_tokens=2048,
        temperature=0,
        seed=1,
    )
    params = req.build_chat_params(None, "auto")
    return req, params


def renderer_apply(tokenizer, messages: list[dict]) -> tuple[list[int], str, dict]:
    _, params = request_params(messages)
    kwargs = params.get_apply_chat_template_kwargs()
    renderer = object.__new__(DeepseekV4Renderer)
    renderer.tokenizer = tokenizer
    ids = renderer._apply_chat_template(
        conversation=[dict(x) for x in messages],
        messages=messages,
        **kwargs,
    )
    text_kwargs = dict(kwargs)
    text_kwargs["tokenize"] = False
    text = renderer._apply_chat_template(
        conversation=[dict(x) for x in messages],
        messages=messages,
        **text_kwargs,
    )
    return list(ids), str(text), kwargs


def reference_messages(rec: dict) -> list[dict]:
    if "messages" in rec:
        return rec["messages"]
    return [{"role": "user", "content": (ROOT / rec["prompt_file"]).read_text()}]


def test_renderer_ids() -> list[dict]:
    tok = DeepseekV41Tokenizer.from_pretrained(MODEL)
    manifest = json.loads(REF.read_text())
    rows = []
    for rec in manifest["rows"]:
        messages = reference_messages(rec)
        ids, text, kwargs = renderer_apply(tok, messages)
        assert ids == rec["ids"], rec["id"]
        assert ids_sha(ids) == rec["sha256_ids"], rec["id"]
        assert hashlib.sha256(text.encode()).hexdigest() == rec["sha256_rendered"], rec["id"]
        assert kwargs["ds41_prompt_profile"] == "ds4-low-v1"
        assert kwargs["reasoning_effort"] == "low"
        assert kwargs["enable_thinking"] is True
        rows.append(
            {
                "id": rec["id"],
                "count": len(ids),
                "sha256_ids": ids_sha(ids),
                "exact_ds4_ids": True,
                "exact_ds4_rendered_text": True,
            }
        )
    return rows


def test_historical_profiles_unchanged() -> list[dict]:
    tok = DeepseekV41Tokenizer.from_pretrained(MODEL)
    baseline = json.loads(
        (ROOT / "runtime/ds41/transfer-ds4-native-001/native-historical-tokenizer-baseline.json").read_text()
    )
    expected = {row["id"]: row for row in baseline["rows"]}
    cases = [
        ("user-low", [{"role": "user", "content": "hello"}], dict(enable_thinking=True, reasoning_effort="low")),
        ("user-none", [{"role": "user", "content": "hello"}], dict(enable_thinking=False, reasoning_effort="none")),
        ("user-high", [{"role": "user", "content": "hello"}], dict(enable_thinking=True, reasoning_effort="high")),
        ("system-low", [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hello"}], dict(enable_thinking=True, reasoning_effort="low")),
        ("multi-low", [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}, {"role": "user", "content": "again"}], dict(enable_thinking=True, reasoning_effort="low")),
    ]
    rows = []
    for name, messages, kwargs in cases:
        ids = list(tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **kwargs))
        text = str(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs))
        ref = expected[name]
        assert len(ids) == ref["count"], (name, len(ids), ref["count"])
        assert ids_sha(ids) == ref["sha256_ids"], (name, ids_sha(ids), ref["sha256_ids"])
        assert hashlib.sha256(text.encode()).hexdigest() == ref["sha256_rendered"], name
        rows.append(ref)
    return rows


def test_fail_closed() -> list[str]:
    tok = DeepseekV41Tokenizer.from_pretrained(MODEL)
    msgs = [{"role": "user", "content": "hello"}]
    bad = [
        ("off", dict(ds41_prompt_profile="ds4-low-v1", enable_thinking=False, reasoning_effort="low")),
        ("none", dict(ds41_prompt_profile="ds4-low-v1", enable_thinking=True, reasoning_effort="none")),
        ("high", dict(ds41_prompt_profile="ds4-low-v1", enable_thinking=True, reasoning_effort="high")),
        ("unknown", dict(ds41_prompt_profile="other", enable_thinking=True, reasoning_effort="low")),
    ]
    names = []
    for name, kwargs in bad:
        try:
            tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **kwargs)
        except ValueError:
            names.append(name)
        else:
            raise AssertionError(f"expected reject {name}")

    try:
        tok.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            tokenize=True,
            ds41_prompt_profile="ds4-low-v1",
            enable_thinking=True,
            reasoning_effort="low",
        )
    except ValueError:
        names.append("content-list")
    else:
        raise AssertionError("expected content-list reject")

    try:
        tok.apply_chat_template(
            msgs,
            tools=[{"type": "function", "function": {"name": "x", "parameters": {}}}],
            tokenize=True,
            ds41_prompt_profile="ds4-low-v1",
            enable_thinking=True,
            reasoning_effort="low",
        )
    except ValueError:
        names.append("tools")
    else:
        raise AssertionError("expected tools reject")
    return names


def test_canonical_prompt_source_contract() -> dict:
    runner = (VENDOR / "vllm/v1/worker/gpu/model_runner.py").read_text()
    legacy_runner = (VENDOR / "vllm/v1/worker/gpu_model_runner.py").read_text()
    dspark = (VENDOR / "vllm/v1/worker/gpu/spec_decode/dspark/speculator.py").read_text()
    dflash = (VENDOR / "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py").read_text()
    renderer = (VENDOR / "vllm/renderers/deepseek_v4.py").read_text()

    required = {
        "renderer_forwards_chatparams_directly": (
            "**params.get_apply_chat_template_kwargs()" in renderer
            and "self._apply_chat_template(" in renderer
        ),
        "scheduled_prefill_ids": "assert new_req_data.prefill_token_ids is not None" in runner,
        "single_req_state_ingest": "all_token_ids=new_req_data.prefill_token_ids" in runner,
        "streaming_prompt_passthrough": "req_state.prompt_token_ids = new_req_data.prompt_token_ids" in legacy_runner,
        "model_runner_passes_input_batch_to_speculator": (
            "draft_tokens = self.speculator.propose(" in runner
            and "input_batch," in runner
        ),
        "dspark_inherits_dflash": "class DSparkSpeculator(DFlashSpeculator):" in dspark,
        "dflash_propose_uses_input_batch": (
            re.search(r"def propose\([\s\S]{0,500}?input_batch: InputBatch", dflash)
            is not None
            and "prepare_dflash_inputs(" in dflash
        ),
        "dspark_no_chat_template": "chat_template" not in dspark,
        "dspark_no_tokenizer": "tokenizer" not in dspark.lower(),
        "dflash_no_chat_template": "chat_template" not in dflash,
        "dflash_no_tokenizer": "tokenizer" not in dflash.lower(),
    }
    assert all(required.values()), required
    return required


def main():
    rows = test_renderer_ids()
    historical = test_historical_profiles_unchanged()
    rejects = test_fail_closed()
    source_contract = test_canonical_prompt_source_contract()
    out = {
        "schema": "ds41-transfer-ds4-native-001-l0-test-v2",
        "status": "PASS",
        "renderer": "DeepseekV4Renderer",
        "profile": "ds4-low-v1",
        "full_id_rows": rows,
        "historical_profiles_unchanged": historical,
        "rejects": rejects,
        "canonical_prompt_source_contract": source_contract,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
