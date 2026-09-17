# Final quality panel — 9c13117

- Verdict: **QUALITY_NOT_QUALIFIED**; soak **BLOCKED**.
- Retrieval/context: code2k FAIL (reused), tail1546 FAIL (`end=23` vs `5`), docs2k FAIL.
- Basic service semantics: JSON-none PASS, fresh marker PASS, multi-turn PASS.
- Coding: C NON_VERIFIED (4096 reasoning / 0 final); Go NON_VERIFIED (toolchain path failure, then 4096 reasoning / 0 final).
- Common-prefix logits cap1/cap32: PASS (same first token/top2; cap32 exact content).
- speed128 contemporary decode: `16.31415081899055` tok/s; performance sample only, not a quality override.
- Correctness discriminator FAIL: begin/middle recovered, end/files not; retrieval/attention fidelity remains implicated.
- Next: exactly one rollback `5bdfed6` discriminator control to distinguish candidate regression from a pre-existing model/runtime limitation; no failed-case retries.
