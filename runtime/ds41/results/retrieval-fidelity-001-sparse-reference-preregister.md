# RETRIEVAL FIDELITY 001 — sparse-attention reference preregister

Hypothesis: the ROCm sparse-prefill attention arithmetic may deviate from the independent vendored Torch reference on the already-saved real code2k packet.

Control: CPU-only replay of request0 chunk0+chunk1 on both ranks using saved Q, ordered context rows, scale and attention sink. Gate is frozen from the upstream test: `torch.testing.assert_close(..., atol=2e-2, rtol=2e-2)`. No model load, no new generation, no GPU kernel.

FAIL localizes a component defect worth fixing/replaying. PASS excludes this kernel arithmetic only; it does not certify selection/indexer semantics or full-model quality.
