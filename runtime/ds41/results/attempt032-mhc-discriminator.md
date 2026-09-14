# Attempt032: model-free mHC discriminator

Status: **PASS / REQUIRED ALIGNMENT = P1+C1**. No LLM/model/executor/process-group was constructed. NODE01 and NODE02 independently produced identical results from their saved rank-local dumps and the real DenseFix layer0 parameters. Source `db633e79e0e131b01f952dec015208279c700ddc`.

## Four-way discriminator, position42 row0

| Combination | next_pre rel-L2 | post_mix rel-L2 | res_mix rel-L2 | Exact carries |
|---|---:|---:|---:|---|
| PB+CB | 4.95862831683e-07 | 4.34331407154e-07 | 4.59223639778e-07 | no |
| PB+C1 | 5.14075768508e-07 | 4.25816517321e-07 | 4.51339801602e-07 | no |
| P1+CB | 9.08446884838e-09 | 5.62774957775e-08 | 3.60970536695e-08 | no |
| P1+C1 | 0 | 0 | 0 | **yes, bit-exact** |

PB is the actual M>1 Torch projection/RMS; P1 is the already-promoted TileLang M1 projection/RMS applied per row. CB is the actual M>1 Torch coefficient/softmax/Sinkhorn; C1 is the already-promoted fused coefficient/Sinkhorn. Layer input/collapse is exact in all four arms because the current carry generation does not change the previous delayed pre-mix.

PB vs P1 on identical real source row has rel-L2 `1.12845277815e-06`, max-abs `0.000167846679688`. Replacing only coefficients leaves the PB-scale carry error; replacing only projection reduces it to roughly 1e-8..6e-8 but is not bit-exact. Therefore projection/RMS is the dominant contribution and coefficient/Sinkhorn is a smaller second contribution required for exact M1 fidelity.

C1 called once on T=2 or T=4 is bit-exact to concatenating T independent C1 M1 calls for pre/post/comb, on both PB and P1 mixes. No new coefficient kernel and no per-row Sinkhorn serialization are justified. Actual counters prove P1 executes per row and C1 executes as the existing fused program.

**NEXT:** implement an opt-in T=2/4 fidelity branch: P1 per row + one C1 batched call, fail-closed unless both promoted paths are enabled. Preserve M1/default behavior and all non-B2/B4 fallbacks. Validate wrapper dispatch/model-free before any full-model B2/B4 verification.
