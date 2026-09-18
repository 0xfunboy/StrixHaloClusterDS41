# DS41 Recovery R4 — DS4 V4.1 quality result

Status: **QUALITY_NOT_FULLY_QUALIFIED_BUT_RETRIEVAL_RECOVERED / NO_PROMOTION / NO_PERFORMANCE_RUN**.

| Gate | DS4 result |
|---|---|
| code2k1588 frozen validator | FAIL only `middle_file`: result=52 and first/last are correct; 6 sections make “median-by-order” non-unique |
| tail1546 | **PASS**, exact `end=5` |
| discriminator1571 | **PASS**, begin17/middle23/end5 + all six filenames exact/in-order |
| docs2k frozen validator | FAIL only `middle_file`: result=52 and first/last are correct; 4 sections make “median-by-order” non-unique |
| preregistered retrieval holdouts | **4/4 PASS** exact |
| JSON / fresh marker / multiturn | **3/3 PASS** exact |
| Go coding | **PASS**, generated replacement passes frozen `go test -race ./...` |
| C coding | **INCOMPLETE_NO_FINAL**: 8192 completion tokens all consumed in reasoning, `finish_reason=length`, final content length 0 |

The old K2 retrieval failure is materially absent on DS4: the tail fact, exact extraction discriminator and four independent pre-output holdouts all pass. The two remaining corpus failures are preserved exactly as frozen; both concern an inherently ambiguous “middle” for an even number of FILE sections and are not rewritten to make DS4 pass.

Transport evidence is separate. The initial DS4 run with upstream default 750 ms TP gate timeout died at layer35 while moving a 40 MiB TCP gate. The port already exposes `DS4_TP_GATE_TIMEOUT_MS`; a preregistered restart at 5000 ms changed only the fail-fast transport timeout. The remaining frozen cases then completed without another TP failure.

Strict `quality_qualified_scope=false` because the frozen validators are not all PASS and the C task emitted no final solution. Therefore performance qualification is **NOT_RUN_BLOCKED_BY_QUALITY**, `performance_target_met=false`, and DS4 is not promoted. Quality-request timing (not a benchmark): code2k prefill ~22.240 s / 71.51 tok/s and decode ~14.30 tok/s.
