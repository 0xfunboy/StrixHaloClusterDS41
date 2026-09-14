# DS41-Q2-001 attempt028 — rowwise native-M1 routed control

Status: **CAUSAL NEGATIVE / B2+B4 FIDELITY FAIL / NO BLOCK TIMING**

Source under test: `b17e2852f414376020b95c26fd71d3d2c33aa14c`.

The candidate changes only routed GGUF MoE for M=2/M=4 packets when the diagnostic flag `DS41_NATIVE_HIP_MOE_ROWWISE=1` is active. Every packet row reuses the already-qualified M1 native HIP contract with the same global top-6 IDs/weights and expert map. Shared expert and the MoERunner final TP reduction remain outside this loop and therefore retain one batched invocation/collective per layer.

## Component gate

Both gfx1151 nodes PASS. M2 and M4 rowwise output is bit-exact (`rel-L2=0`, `max-abs=0`) versus explicitly concatenating the exact native M1 operation row by row. M16 remains the pre-existing Triton fallback bit-exact to its baseline. The ordinary M1 native-vs-Triton component check remains inside its historical bound (`rel-L2=0.022309255735774712 < 0.035`).

## Full common-prefix gate

Frozen attempt027 prompt/oracle/validator and logit thresholds were reused. D1/M1 is exact at all 56 compared positions. B2 and B4 still fail on the first packet row, model position 42:

- B2 position42: `rel-L2=0.09636353562599285`, `max-abs=1.875`, top1 still `297`.
- B4 position42: `rel-L2=0.09724620056137186`, `max-abs=1.875`, top1 still `297`.
- B2 overall common-prefix max rel-L2 `0.09636353562599285`, max-abs `1.875`, 8 positions compared.
- B4 overall common-prefix max rel-L2 `0.10908273902417093`, max-abs `2.490234375`, 24 positions compared.

Both widths are ineligible. Reject/partial-accept/rollback are therefore still N/A and no B2/B4 timing is admitted. Only the contemporary M1 control ran after the failed gate.

## Causal conclusion

Replacing the packet routed computation with the exact qualified M1 routed contract does **not** collapse the position42 error. Routed native-vs-Triton is therefore not the first sufficient cause of the attempt027 drift. The next authorized discriminator moves upstream and captures the same position42 row at decoder-layer boundaries: layer entry, mHC-attention output into `attn_norm`, attention output, mHC-FFN output into `ffn_norm`, FFN/router input and FFN output. If the FFN input is already different, localize the first earlier boundary and then test the corresponding promoted M1 mHC primitive per row without changing causal visibility.

Pair cleanup PASS: both ranks `OFF_VERIFIED`, owner `NONE/OFF`; GLM remains OFF and the gateway is left available. Raw: `reports/DS41-Q2-001/attempt028/`.
