# DS4 SPEED 001 handoff

updated_at: 2026-09-20T14:42:14+02:00
phase: CLOSED_E1_DEFAULT_READY
authority: /home/funboy/DS4_SPEED_001.md
current: /home/funboy/STRIX_CLUSTER_DOCS/CURRENT.md
qualified_base: 7d0454b4e32ef1e90235f2b001d6643b5934438c
e1_source_commit: a8f44737ecc6bbd406d796d1e402b312f00d1564
e2_terminal_commit: f3def40caaf16eb30091c90e6bf7d748a9870c91
product_commit: dbc801fb3a88582e67a50ea87338045740071a77
model: /home/funboy/models/ds41/ds4-v41-q2/DeepSeek-V4.1-Flash-Q2.gguf
fallback: DS4 DOCUMENT PROFILE 002

## E1 closed

The real M1 path was serial. A bounded persistent Linux reader now overlaps
the two native Engram table reads. Byte/lifetime tests pass on both nodes.
Clean DS4 against DS4 A/B passes 6/6 in each arm with exact output. Median
decode is 15.47 to 16.77 tok/s for code and 15.88 to 16.80 tok/s for documents.
Frozen holdouts pass 2/2 exactly. Detailed result:
`/home/funboy/STRIX_CLUSTER_DOCS/evidence/results/DS4_SPEED_001_E1_RESULT.md`.

## E2 closed

The isolated opt-in generic ROCm N=2 path costs a median 211.047 ms versus
128.866 ms for two serial M1 steps and diverges from the serial logits after
the first append. It therefore fails correctness and has an 82.180 ms deficit
even with perfect acceptance and zero-cost draft and state. The complete state
timing is gated out, and the local DSpark sidecar was not converted or loaded.

Detailed result:
`/home/funboy/STRIX_CLUSTER_DOCS/evidence/results/DS4_SPEED_001_E2_RESULT.md`.

## Operational state

E1 is the qualified default and is READY. Both resident executables come from
`/home/funboy/.local/share/haloclu-ds41/releases/ds4-speed-001-engram1`.
Engram mode is concurrent. Diagnostic timing, serial rollback and VERIFY2 are
all disabled in the resident environments. The protected gateway runs the
primary `config.ds4-document-profile-002.json` and reports the same release
and mode through its lifecycle endpoint.

Product gates passed for non-stream, SSE, fresh, multiturn, cancel/drain,
stable worker lifetime and whole-pair OFF/503/ON. The 2h/24 soak passed 24/24
in 8178.030 seconds with the same coordinator and worker PIDs throughout.
Raw terminals:

- `/home/funboy/reports/DS4-SPEED-001/product-e1/terminal.json`;
- `/home/funboy/reports/DS4-SPEED-001/soak-e1/terminal.json`.

The previous qualified DS4 remains the fallback. Use
`scripts/ds4-speed-001-controller.sh rollback-qualified` to restore it as a
whole pair. The controller reports that state as `qualified-rollback`. No
research action remains open. Do not return to K2 and do not continue E2.
