# DS4 PREFILL GAP 002 handoff

updated_at: 2026-09-21T04:26+02:00
phase: CLOSED
owner: NONE
campaign_status: COMPLETE_NO_PROMOTION
raw: /home/funboy/reports/DS4-PREFILL-GAP-002
report: /home/funboy/STRIX_CLUSTER_DOCS/evidence/results/DS4_PREFILL_GAP_002_FINAL.md
halopipe_note: /home/funboy/STRIX_CLUSTER_DOCS/evidence/results/HALOPIPE_V41_FEASIBILITY_002.md
resident_release: /home/funboy/.local/share/haloclu-ds41/releases/ds4-speed-001-engram1
base: 7d0454b4e32ef1e90235f2b001d6643b5934438c
e1: a8f44737ecc6bbd406d796d1e402b312f00d1564
worktree: /home/funboy/worktrees/ds4-prefill-gap-002

## Terminal result

- P0 audit COMPLETE.
- P1 targeted 16K diagnostic PASS; diagnostic observation 258.28 prefill / 15.48 decode tok/s.
- P1 localizes dominant cost to compute/TP chunk path. Engram prefill wait is already effectively hidden. Large TP payload and peer-arrival skew are measured, but are not labeled pure/removable network time.
- Direct physical chunk 4096 was rejected before full-model because routed-MoE TP and mHC fast paths in this pin have 2048-row guards.
- One candidate implemented: Linux TCP big-gate outbound payload pre-send while waiting for validated peer header.
- Targeted model-free tests PASS, including real NODE01↔NODE02 41,943,040-byte bulk exchange.
- Clean16K A/B: exact frontier/output; decode guard PASS; median prefill time A=68.8838s, B=66.9555s. Reduction=1.9283s / 2.799%.
- A control population SD=2.9058s; preregistered gate FAIL and 10% goal FAIL.
- 64K candidate benchmark SKIPPED_GATE_16K_NOT_MET. No extra performance sampling.
- Candidate NOT PROMOTED.
- Quality B budget: exactly four inference requests total. One setup-source mismatch is retained separately; valid v2 exact code2K, code16K and docs16K are all PASS/cache0. Exact docs2K not rerun to preserve max4.
- HaloPipe prefill: FEASIBILITY_UNRESOLVED. HaloPipe decode: DO_NOT_PORT_NOW.
- No experimental commit or push.

## Final resident state

E1 original is READY, coordinator+worker active, API HTTP200, Engram concurrent, diagnostic timing OFF, VERIFY2 OFF. Final authenticated smoke: PREFILL-GAP-QUALITY-V2-RESTORE-OK.

No experimental PREFILL GAP unit remains active. Do not retry 64K or add performance samples automatically. The campaign is terminal.
