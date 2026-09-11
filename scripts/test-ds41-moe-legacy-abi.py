#!/usr/bin/env python3
from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F

import vllm._custom_ops as ops


def main() -> None:
    if os.environ.get("DS41_MOE_C_LEGACY_TEXT_ABI") != "1":
        raise RuntimeError("DS41_MOE_C_LEGACY_TEXT_ABI=1 required")
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm device unavailable")

    torch.manual_seed(7)
    device = "cuda"
    m, experts, topk = 32, 384, 6
    vocab = 256
    sentinel = 129264
    scale = 2.5

    gating = torch.randn(m, experts, dtype=torch.float32, device=device)
    bias = torch.randn(experts, dtype=torch.float32, device=device)
    # fused_topk_bias() normalizes input token dtype to the hash-table dtype
    # before calling the custom op; mirror that real call contract here.
    input_ids = torch.arange(1, m + 1, dtype=torch.int32, device=device)

    # Deterministic unique expert ids per token; this is the hash-routed path
    # used by the V4.1 router when a tid2eid table is supplied.
    table_cpu = torch.empty(vocab, topk, dtype=torch.int32)
    for token in range(vocab):
        table_cpu[token] = torch.tensor(
            [((token * 17) + j * 53) % experts for j in range(topk)],
            dtype=torch.int32,
        )
    table = table_cpu.to(device)

    out_w = torch.empty(m, topk, dtype=torch.float32, device=device)
    out_i = torch.empty(m, topk, dtype=torch.int32, device=device)
    token_expert = torch.empty(m, topk, dtype=torch.int32, device=device)
    bias_vl = torch.randn(experts, dtype=torch.float32, device=device)

    ops.topk_hash_softplus_sqrt(
        out_w,
        out_i,
        token_expert,
        gating,
        True,
        scale,
        bias,
        input_ids,
        table,
        None,
        bias_vl=bias_vl,
        image_sentinel_lo=sentinel,
    )
    torch.cuda.synchronize()

    expected_i = table[input_ids].to(torch.int32)
    selected_logits = gating.gather(1, expected_i.to(torch.int64))
    raw = torch.sqrt(F.softplus(selected_logits))
    expected_w = raw * (scale / raw.sum(dim=1, keepdim=True))

    ids_equal = torch.equal(out_i, expected_i)
    max_abs = (out_w - expected_w).abs().max().item()
    mean_abs = (out_w - expected_w).abs().mean().item()

    # The DS41 language-only contract must fail closed if a real image sentinel
    # reaches the legacy ABI path.
    image_ids = input_ids.clone()
    image_ids[0] = sentinel
    guard = "FAIL"
    try:
        ops.topk_hash_softplus_sqrt(
            out_w,
            out_i,
            token_expert,
            gating,
            True,
            scale,
            bias,
            image_ids,
            table,
            None,
            bias_vl=bias_vl,
            image_sentinel_lo=sentinel,
        )
    except RuntimeError as exc:
        if "cannot route image-sentinel" in str(exc):
            guard = "PASS"
        else:
            raise

    result = {
        "status": "PASS",
        "schema": str(torch.ops._moe_C.topk_softplus_sqrt._schemas),
        "ids_equal": ids_equal,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "image_guard": guard,
        "tokens": m,
        "experts": experts,
        "topk": topk,
        "scale": scale,
    }
    assert ids_equal, result
    assert max_abs < 2e-5, result
    assert guard == "PASS", result
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
