# DS4 LONG CONTEXT BASELINE 001 handoff

updated_at: 2026-09-21T00:54+02:00
phase: CLOSED
owner: NONE
campaign_status: PARTIAL_OR_STOP_SEMANTIC
raw: /home/funboy/reports/DS4-LONG-CONTEXT-BASELINE-001
report: /home/funboy/STRIX_CLUSTER_DOCS/evidence/results/DS4_LONG_CONTEXT_BASELINE_001_FINAL.md
p1_p2_terminal: /home/funboy/reports/DS4-LONG-CONTEXT-BASELINE-001/p1-p2-terminal.json
p3_p4_terminal: /home/funboy/reports/DS4-LONG-CONTEXT-BASELINE-001/p3-p4-terminal.json
qualified_release: /home/funboy/.local/share/haloclu-ds41/releases/ds4-speed-001-engram1
qualified_base: 7d0454b4e32ef1e90235f2b001d6643b5934438c
e1_source_commit: a8f44737ecc6bbd406d796d1e402b312f00d1564
fallback: E1 concurrent READY

## Terminal result

- P1/P2: PASS tecnico. Fresh full-prefix valido a 65536.
- Fresh 16K mediana: 240.66 prefill / 15.46 decode tok/s.
- Fresh 64K mediana: 280.60 prefill / 14.70 decode tok/s.
- P3 originali: code+docs PASS a 4K/8K/16K/32K; docs64K PASS; code64K INCOMPLETE_NO_FINAL.
- Holdout 32K: entrambi INCOMPLETE_NO_FINAL al cap2048. Nessun livello holdout-confirmed.
- P4 step1 32K: INCOMPLETE_NO_FINAL; step2-6 non eseguiti per stop preregistrato.
- Restore finale: PASS. E1 concurrent READY, timing OFF, VERIFY2 OFF, smoke autenticato LONGCTX-P3P4-RESTORE-OK.

## Recovery invariant

La campagna è terminale. Non riprendere o ritentare automaticamente P3/P4. Il profilo long-context resta candidato separato e non modifica i limiti produttivi.
