# Attempt034: rejection/recovery state gate and conditional block cost

Status: **PREREGISTERED**.

No model-math change from attempt033. Math fix remains `ce446ef62836980b4d4a3c9a62b4374ba3394c8d`; dispatch validator is `03f6699194bf4de1dc6cac960369614712f77d2e`. D1 is unchanged promoted M1. B2/B4 enable the same two opt-in fidelity controls: native routed M1 per row and mHC P1-per-row + one C1 batched call.

Reuse attempt033 prompt/oracle/sampling/context/prefix-cache and exact final gates. Sequence is the existing frozen runner sequence: D1 clean diagnostic, B2 clean diagnostic, B4 clean diagnostic; only if B4 is eligible, inject exactly one corrupt first draft and exactly one corrupt last draft in separate B4 diagnostics, requiring real greedy rejection counts, unaffected causal rows, post-rejection continuation and rollback to the oracle. Only widths that pass their numerical/state gates may enter three 32-token measure trials in the frozen alternating order.

Diagnostic tensor/logit copies and proposal replay overhead are excluded from performance qualification. Measure-mode timing is the full EngineCore target verification step plus post-step/GPU completion and replay proposal machinery; it contains **no real drafter cost**. Report favorable `B*t1/tB`, actual emitted-token rate and remaining ideal drafting budget only as a diagnostic bound, never as DSpark or end-to-end speculative TPS.

Pair-safe whole-pair supervisor must be active with an absolute script path before accepting results; cleanup must finish `OFF_VERIFIED/OFF_VERIFIED`, owner `NONE/OFF`. No DSpark integration, new weights, engine/network/UI changes or gate relaxation.
