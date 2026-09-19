# DS41 V4.1 ATTENTION PARITY 002 — handoff

updated_at: 2026-09-19T10:22+02:00
phase: P0 COMPLETE / P1 FIX1 PASS CROSS-NODE / P2 M1 FIX1 TERMINAL FAIL — 2/6 ORIGINALS FAIL_SEMANTIC / DS4 READY
next_action: none under ATTENTION PARITY 002. Preserve p2-m1 and p2-m1-fix1 terminals/finalizers; do not send holdouts or confirmations and do not admit K2/DSpark. Resume only under a new explicit mandate.
worktree: /home/funboy/worktrees/ds41-v41-attention-parity-002
branch: exp/ds41-v41-attention-parity-002
base_head: 3b4c2df5d2dda490dae1bccaec75be389e41e1d1
base_release: /home/funboy/.local/share/haloclu-ds41/releases/native-antirez-m1-transfer001-woa1
candidate_release: /home/funboy/.local/share/haloclu-ds41/releases/native-antirez-m1-attnparity002-m1-fix1
served_model: DeepSeek-V4.1-Flash-Q2-AttnParity002-M1-Fix1
terminal_run: reports/DS41-Q2-001/attention-parity-002/p2-m1-fix1
final_evidence: /home/funboy/STRIX_CLUSTER_DOCS/evidence/results/ATTENTION_PARITY_002_FINAL.md

## Effective delta

R1c/R2 remain applicable to the actual WOA1 release:
- window KV still stores the old hybrid 448 FP8/block64 + 64 BF16 layout;
- compressed KV still uses the same hybrid writer;
- gfx1151 still selects FP8 indexer Q/K although the pinned tree already contains MXFP4 block32 producers.

ATTENTION PARITY 002 implements:
- window full-512 FP8 block32 UE8M0 QDQ, stored as BF16 QDQ values;
- compressed full-512 NVFP4/E2M1 block16 with E4M3 scales, QDQ after the plain norm/RoPE writer and then stored BF16;
- indexer Q/K reuse the existing MXFP4 block32/UE8M0 producers; ROCm consumes that layout through an opt-in compatibility scorer;
- BF16 paged gather/prefill/decode consumers; no old FP8 writer can recode the QDQ values;
- default behavior unchanged unless DS41_V41_ATTN_PARITY=1.

## Gates already complete

- candidate derived from WOA1 source commit 9fe0496398aae695c9d2486ad7b234a8ee0fde80 with fail-closed base hashes;
- patched Python syntax PASS; launcher bash syntax PASS;
- independent CPU window reference: BF16 bit-exact on zero/tiny/normal/saturation cases;
- independent MXFP4 decode: exact;
- paged BF16 writer/gather: PASS;
- NVFP4 CPU reference: zero/finite/block16-E4M3 cases PASS;
- opt-in defaults OFF;
- max-context 16384 representation delta: +16.84 MiB net persistent/sequence; decode compact scratch 0.625 MiB/query; fixed 1 GiB KV budget unchanged;
- new final holdouts C/D frozen before any candidate model result.
- device negative001: validator-only arithmetic-order defect (`v/scale` vs pinned `v*reciprocal(scale)`), model load 0, requests 0; raw `/home/funboy/reports/DS41-ATTENTION-PARITY-002/device-node01-fail001.log`; DS4 restored READY before retry.
- corrected device gate: PASS on NODE01 and NODE02; cross-node receipt `/home/funboy/reports/DS41-ATTENTION-PARITY-002/device-cross-node-pass.json`; DS4 subsequently stopped again as a pair for P2.

Official standalone DeepSeek V4.1 model.py was not found in the live filesystem, Git objects or tmp during this run. No download was performed. Numeric contracts are therefore bound to the already-frozen R1c/R2 evidence and pinned local vLLM quantization/reference implementations; do not claim a newly re-hashed official file.

## P2 terminal

Il primo run `p2-m1` resta immutabile ma è implementation-contaminated: un gate producer aggiunto dopo il terminale ha localizzato il producer MXFP4 PTX-only su gfx1151 (763/768 byte Q packed errati, `unknown asm constraint 'f'`). FIX1 sostituisce soltanto la conversione E2M1 con Triton OCP RTNE portabile; packing e scale Q/K sono bit-exact su entrambi i nodi, il compressed writer è bit-exact e il consumer BF16 passa il gate reference.

Run autorevole definitivo `reports/DS41-Q2-001/attention-parity-002/p2-m1-fix1`: `P2_M1_FAIL`, 2/6 richieste.
- code2k: `FAIL_SEMANTIC`, HTTP200, final vuoto, first-final assente, first delta 28.603 s, prefill 28.549 s / 57.06 tok/s, wall 34.422 s.
- docs2k: `FAIL_SEMANTIC`, HTTP200, final vuoto, first-final assente, first delta 22.486 s, prefill 22.457 s / 58.16 tok/s, wall 475.594 s.

Per gate non sono stati inviati holdout A/B, conferme o nuovi holdout C/D e K2/DSpark non è stato ammesso. Il finalizer è `P2_M1_FAIL_DS4_READY`.

L'audit dei raw/log non localizza un secondo fault deterministico: rank/pair erano healthy durante le richieste; nessun NaN/assert/kernel fault prima del teardown; warning JIT first-use soltanto. Non autorizzare un terzo full-model attempt da questo handoff.

## Runtime state

DS4 DOCUMENT PROFILE 002 è il fallback qualificato ed è stato ripristinato **READY/HTTP200** dopo P2. Native DS41 owner è `NONE/OFF`; K2 è OFF. TRANSFER 001 resta terminal FAIL e non è un candidate arm.

## Frozen holdouts

runtime/ds41/attention-parity-002/holdouts/manifest.json
- C prompt SHA256 2818590e9679d8ad94401b185ef47c974d96fc0afd01d0958ef71e3d55b30463
- D prompt SHA256 c810628f9378565a5def862de23e4d60f706d1b4b549cc230bc95a62ab006df4
