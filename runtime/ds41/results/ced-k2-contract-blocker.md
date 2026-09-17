# CED / SWA bounded replay — K2 contract gate

Status: **BLOCKED_BY_K2_AUX_HIDDEN_CONTRACT / NO MODEL LOAD**.

The model config and pinned DS4 V4.1 reference agree on 40 layers, SWA128, KV sources `[2,8,14,20]` and index sources `[2,8,14,20,24,28,32,36]`. The DS4 pin `8db1d1d` does not enable its decoder-suffix replay below a total sweep of 8192 tokens; its suffix is not “last 128 at every decoder layer”, but `1 + (39-layer)*127` rows plus preparation of the preceding 127-row SWA dependency.

The current frozen code-2k target executes scheduler chunks 1023+565, therefore the upstream decoder-suffix path is not admitted for either chunk. More importantly, K2 DSpark is configured from target layers `[37,38,39]`: the target model stores the pre-collapse hidden stream for every forward row, and the DSpark proposer copies/combines all `num_target_tokens` before precomputing its context KV. Omitting older rows in layers 37/38/39 would therefore violate the already-qualified K2 state contract. Recomputing those hidden rows requires their upstream decoder chain and removes the CED work elimination.

Decision: do **not** implement a synthetic “layers21-39=128 rows” path, do not zero/fabricate missing DSpark hidden states, and do not disable K2 to manufacture a CED sample. CED is blocked for the current K2/code-2k profile without a separate redesign of the drafter context contract. Under the integrated mandate, continue Engram on the MMQ predecessor and label it MMQ+Engram, not MMQ+CED+Engram.

Evidence is model-free and source-grounded; no inference load was consumed for this gate. Machine-readable details are in `runtime/ds41/results/ced-k2-contract-blocker.json`.
