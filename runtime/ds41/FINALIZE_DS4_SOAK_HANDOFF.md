# FINALIZE DS4 SOAK — terminal handoff

Updated: 2026-09-17 terminal.

## Verdicts
- `performance_target_met=false`: target200 tok/s; MMQ+Engram best76.1697 tok/s, cache-characterized confirmation66.7440 tok/s.
- `quality_qualified_scope=false`: code2k/tail1546/docs2k FAIL; JSON/fresh/multi-turn and common-prefix logits PASS; coding C/Go NON_VERIFIED.
- `delivery_complete=true`: all authorized MMQ/Engram/CED/quality evidence packaged; no selective promotion because quality gate fails.
- Soak: `NOT_RUN_BLOCKED_BY_QUALITY`.

## Performance
- MMQ same-source: OFF91.934516s/17.2732 -> ON29.191465s/54.3995, 3.14936x throughput.
- Engram same-source: workers1 29.238172s vs workers4-confirm23.792395s, -18.626% prefill time; workers4 samples20.848176/23.792395s.
- CED: `BLOCKED_BY_K2_AUX_HIDDEN_CONTRACT`.

## Quality discriminator
- Frozen extraction prompt SHA `f5543bba60bbf849bb106bf9efa9ca5381b88a4c4c9281cf60de37a767258745`.
- Optimized `9c13117f`: FAIL, begin17/middle23/end29, files [`__init__.py`,`api.py`,`prompt.go`,`test-ds41-prefill-metrics.py`,`envelope_test.go`].
- Rollback `5bdfed6`: **identical content**, FAIL, 1571 computed/cache0.
- Decision: `SHARED_PREEXISTING_QUALITY_FAILURE_NOT_CANDIDATE_REGRESSION`. This does not qualify either profile for long-context quality.

## Final state
- Operational release/source: `k2-prefill-5bdfed6` / `5bdfed698ab97b6230db3fe8a7e37ff8e9b3d513`.
- Epoch `1789643558929043486`; K2; controller READY dual200.
- Optimized release `k2-mmq-engram-9c13117` preserved as experimental performance artifact only.
- No soak or residual speed intervention. Fresh mandate required for the pre-existing long-context fidelity defect.
