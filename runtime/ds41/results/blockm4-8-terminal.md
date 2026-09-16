# DS41 BLOCK_M4/8 terminal result

**FAIL_NUMERIC_GATE_STOP / NO FULL-MODEL A/B / NO PROMOTION**

- Real fixture chunk0/layer0: BLOCK_M4 `2139.699 ms`; BLOCK_M8 `1683.991 ms`; component gain `+27.061%`; 4/8 output bit-identical.
- Real fixture chunk0/layer20: BLOCK_M4 `2083.670 ms`; BLOCK_M8 `1634.842 ms`; component gain `+27.454%`; 4/8 output bit-identical.
- Frozen independent-reference gate fails at layer20: both arms have sampled-row `max_abs=128`, above preregistered `2.0`; max rel-L2 remains `0.016931`. This is not evidence of a BLOCK_M8 numerical regression because BLOCK_M4 and8 are identical, but it prevents qualification under the frozen gate.
- Real route distribution was captured for all 8 required fixtures. Alignment padding sum: BLOCK_M4 `1980` rows; BLOCK_M8 `4928` rows.
- Per preregistration, candidate stops at first numerical gate failure. Remaining six timing fixtures and full-model A/B were not run. Historical clean `92.699 s` full-model observation remains separate.
- Rollback/default: `DS41_MOE_PREFILL_BLOCK_M=4`; restore known K2 release `k2-prefill-5bdfed6`.
