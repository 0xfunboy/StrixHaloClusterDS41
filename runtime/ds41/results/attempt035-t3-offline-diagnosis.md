# DS41-Q2-001 — attempt035 T3 offline diagnosis

**Status: HYPOTHESIS_CONFIRMED / NO MODEL LOAD.**

Attempt034 raw on both ranks proves the corrupt-first request ends with one forward over positions `[95,96,97]`, `T=3`, after all positions through94 remain bit-exact to M1. On that one T3 step the routed native path remains active (`120` native calls, zero routed fallback), while mHC coefficient/Sinkhorn and projection/RMS each record `80` `tokens_not_1` fallbacks and zero promoted calls.

Controls line up exactly: clean B4 ends `[94,95,96,97]` at T4 with rowwise mHC active; corrupt-last ends `[97]` at T1 on promoted M1; both pass. The only three attempt034 logit failures are positions95-97.

Therefore attempt034's FAIL remains valid, but the previous attribution to latent dirty rollback/cache is not demonstrated. The observed T3 mHC dispatch hole is a concrete sufficient candidate and must be tested first. No model was loaded for this diagnosis.

**NEXT:** qualify only T3 P1-per-row + one C1 `[3,24]` call model-free on both gfx1151 nodes, K5120/K20480, preserving M1/T2/T4 and out-of-scope fallback.
