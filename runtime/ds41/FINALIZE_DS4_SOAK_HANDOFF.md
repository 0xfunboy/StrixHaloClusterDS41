# FINALIZE DS4 SOAK — live handoff

Updated: 2026-09-17 after terminal compact quality panel.
Phase: rollback discriminator control; soak BLOCKED.

## Candidate result
- Source/release `9c13117f56fd81d03c8c610a5fb0148ccdc603b9` / `k2-mmq-engram-9c13117`, K2/M4 canonical=1 MMQ=1 CED=0 Engram4.
- MMQ same-source: 91.934516s -> 29.191465s, 3.149363x throughput.
- Engram same-source characterized: workers1 29.238172s vs workers4 23.792395s, -18.626% prefill time; workers4 samples 20.848176/23.792395s, median22.320285s.
- CED: BLOCKED_BY_K2_AUX_HIDDEN_CONTRACT.

## Quality terminal
- code2k: FAIL reused. tail1546: FAIL end23 vs5. docs2k: FAIL.
- JSON-none PASS; fresh marker PASS; multi-turn ORBIT-17 PASS.
- Common-prefix cap1/cap32 logits PASS.
- C coding NON_VERIFIED: 4096 reasoning, 0 final.
- Go coding NON_VERIFIED: toolchain path failure on first final, retry incomplete with 4096 reasoning/0 final.
- Correctness discriminator FAIL: begin17/middle23 recovered; end29 vs5 and file list incomplete/wrong => retrieval/attention fidelity remains implicated.
- Verdict: QUALITY_NOT_QUALIFIED; SOAK BLOCKED.

## Next exact act
1. One frozen discriminator control on rollback `5bdfed698ab97b6230db3fe8a7e37ff8e9b3d513`; no retries of failed panel cases.
2. If rollback discriminator also FAILs comparably, classify long-context quality issue as pre-existing/shared; candidate remains experimental performance-only. If rollback passes, classify current optimized candidate as quality regression.
3. Final state rollback K2/M4 READY/idle; publish terminal delivery verdict.
4. No residual structural speed intervention while quality is unqualified.
