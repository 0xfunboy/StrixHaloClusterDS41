#!/usr/bin/env python3
"""Model-free gate for the L2 Antirez WO_A GGUF fast-path compatibility fix."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("release")
    args = ap.parse_args()
    root = Path(args.release).resolve()
    sys.path[:0] = [
        str(root),
        str(root / ".vendor/vllm-dsv41"),
        str(root / ".vendor/gguf-plugin"),
        str(root / ".vendor/llama-v41/gguf-py"),
    ]

    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _get_cached_wo_a_bf16
    from vllm_gguf_plugin import ops as gguf_ops

    class FakeWoA(torch.nn.Module):
        def __init__(self, weight: torch.Tensor, qtype: int | None = None) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(weight, requires_grad=False)
            if qtype is not None:
                self.weight_type = SimpleNamespace(weight_type=qtype)

    # Q8_0 block32/type34: packed [4,34] -> logical [4,32].
    calls = []
    original = gguf_ops.ggml_dequantize

    def fake_dequant(weight, qtype, m, n, dtype):
        calls.append((tuple(weight.shape), int(qtype), int(m), int(n), dtype))
        return torch.arange(m * n, dtype=torch.float32).view(m, n).to(dtype)

    gguf_ops.ggml_dequantize = fake_dequant
    try:
        packed = torch.arange(4 * 34, dtype=torch.uint8).view(4, 34)
        wo = FakeWoA(packed, qtype=8)
        out = _get_cached_wo_a_bf16(wo, 2, 2, 32)
        assert out.shape == (2, 2, 32), out.shape
        assert calls == [((4, 34), 8, 4, 32, torch.bfloat16)], calls
        first = out.clone()
        calls.clear()
        out2 = _get_cached_wo_a_bf16(wo, 2, 2, 32)
        assert out2 is out
        assert calls == []
        torch.testing.assert_close(out2, first, atol=0, rtol=0)

        # Wrong requested logical geometry must fail before dequantization.
        bad = FakeWoA(packed.clone(), qtype=8)
        try:
            _get_cached_wo_a_bf16(bad, 2, 2, 64)
        except RuntimeError as exc:
            assert "logical geometry mismatch" in str(exc)
        else:
            raise AssertionError("Q8_0 bad logical geometry did not fail closed")

        # Historical plain BF16 path remains exact and does not call GGUF dequant.
        plain_weight = torch.arange(4 * 32, dtype=torch.float32).view(4, 32).to(torch.bfloat16)
        plain = FakeWoA(plain_weight)
        plain_out = _get_cached_wo_a_bf16(plain, 2, 2, 32)
        assert calls == []
        torch.testing.assert_close(
            plain_out,
            plain_weight.view(2, 2, 32),
            atol=0,
            rtol=0,
        )
    finally:
        gguf_ops.ggml_dequantize = original

    print(
        {
            "status": "PASS",
            "q8_0_packed_shape": [4, 34],
            "q8_0_logical_shape": [4, 32],
            "q8_0_branch_cached": True,
            "bad_geometry_fail_closed": True,
            "bf16_regression": "PASS",
        }
    )


if __name__ == "__main__":
    main()
