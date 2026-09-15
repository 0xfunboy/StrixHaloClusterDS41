# SERVE-K2 daily candidate — preflight

**PASS / NO MODEL LOAD**

- DS41-only `reasoning_effort=none` reaches the pinned V4.1 chat template as non-thinking; default GLM-compatible config still rejects `none`.
- Prefill accounting builder PASS; gfx1151 event ordering PASS on both nodes; paired critical max merge PASS.
- Complete vLLM overlay fresh-applies to the pinned vendor.
- Engine candidate max length 65,664; app admission max 65,536; historical 1 GiB KV allocation reports 105,455 token capacity. This is capacity evidence only, not long-context correctness.
- No model/weight/quant/precision/kernel math/driver/network change.
