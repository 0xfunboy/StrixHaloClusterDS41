# Attempt031: narrow downstream localization after rowwise routed correction

Status: **SECOND_DIVERGENCE_LOCALIZED_MHC_CARRY / FINAL_FIDELITY_FAIL**.

Source `f9fd3c3dc72ba74f9409ca2e387f6871e8c827c4`, epoch `1789352092657841208`. D1 retains promoted M1; B2/B4 enable only the opt-in rowwise native routed control already isolated in attempt029. This run is diagnostic; boundary copies invalidate performance use.

## Position42 result

- layer0 `ffn_out`: **bit-exact** B2/B4 versus D1. The first routed M>1 shape error is therefore corrected.
- layer1 `state_x`, `state_residual`, and `layer_entry`: **bit-exact**.
- First saved numerical difference: layer1 `state_pre_mix`, produced by layer0 FFN `mhc_pre_delayed_torch`: 4/4 FP32 elements differ, rel-L2 `4.95862831683e-07`, max-abs `3.57627868652e-07`.
- Sibling carries also differ: `state_post_mix` rel-L2 `4.34331407154e-07`, `state_res_mix` `4.59223639778e-07`.
- `engram_out` is downstream of already-different carry state (2/20480 BF16 values, rel-L2 `2.91254206268e-05`); it is not the first component to patch.
- Both ranks have identical boundary metrics; B2 and B4 row0 narrow metrics are identical.

These ~5e-7 carry differences are **not yet claimed sufficient** to explain the final logits. The causal discriminator is model-free: reconstruct the real layer0 FFN mHC-pre input and separate batched Torch projection/RMS (PB) vs per-row promoted TileLang (P1), and batched Torch coefficient/Sinkhorn (CB) vs promoted fused coefficient/Sinkhorn (C1).

## Frozen final gate

B2 remains FAIL at model position42: rel-L2 `0.096363535626`, max-abs `1.875`. B4 remains FAIL: rel-L2 `0.0972462005614`, max-abs `1.875`. No reject/partial/rollback or block timing is admitted.

## Cleanup

Supervisor observed both ranks inactive and called pair-safe cleanup `rc=0`; live recheck: NODE01/NODE02 `OFF_VERIFIED`, owner `NONE/OFF`. GLM stays OFF.

Raw: `reports/DS41-Q2-001/attempt031/`.
