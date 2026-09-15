# DS41 daily short-context soak — terminal interrupted result

Date: 2026-09-15
Verdict: `SOAK_INCOMPLETE / BLOCKED_RUNNER_CANCEL_CLIENT`

The frozen operational-short soak did not satisfy its qualification gate. The registry stopped at case 12 with `state=FAILED_SERVICE`; only 11 of 24 cases completed. All 11 completed cases are independently `service_status=PASS` and `semantic_status=PASS`. The long-context daily qualification remains failed separately and is not changed by this result.

## Residency / service invariants

- Frozen release: `c075e82464c954e202c6f47658821d233edc13a5`.
- Frozen epoch: `1789482113148506636`.
- Every before/after lifecycle snapshot recorded by cases 1..12 contains exactly that release and epoch.
- Rank0 unit has been continuously active since `2026-09-15 16:21:53 CEST`, `NRestarts=0`, InvocationID `692890562050422ba6aa39940971d3ae`.
- Rank1 unit has been continuously active since `2026-09-15 16:21:54 CEST`, `NRestarts=0`, InvocationID `a833c1b0067c4cc3993d82c19c75de0c`.
- Paired coordinator and DS41 gateway have been continuously active since `2026-09-15 16:21:50 CEST`, both `NRestarts=0`.
- At terminal inspection the DS41 gateway reports lifecycle `READY`, owner `DS41/RUNNING`, preset `dspark-k2-gfx1151`, the same release/epoch; pair health is `ok`, ranks `[true,true]`, `busy=false`, poison empty.
- No coding acquisition is active (`ds41-daily-coding-acquire.service` inactive/dead).

This is sufficient to reject a model reload/crash as the cause of the soak stop.

## Completed cases

Cases 01 through 11 are COMPLETE and all have service PASS + semantic PASS: arithmetic none, JSON none, fresh marker, two multi-turn continuity cases, low reasoning 100-doors, high reasoning arithmetic, exact two-line formatting, short sentinel, JSON array, and long-output. Case 11 delivered 720 final tokens against a minimum 512 gate.

## Terminal blocker

Case `12-cancel-decode` was dispatched at `2026-09-15T18:55:01Z`, but the runner failed before reaching the preregistered 8-second cancel point with:

`OSError('cannot read from timed out object')`

The runner source sets `c.sock.settimeout(.5)` for cancel cases and then calls `r.readline()` repeatedly. It catches `socket.timeout`, but Python's buffered HTTP response reader becomes non-reusable after that timed read and raises `OSError('cannot read from timed out object')`. No `response.partial.sse` was written. The immediate post-failure snapshot still had a healthy pair with `busy=true`; later inspection shows the same pair healthy and drained to `busy=false`.

Therefore the raw registry's case-12 `service_status=FAIL` is preserved, but the forensic classification is `BLOCKED_RUNNER_CANCEL_CLIENT`: it does not demonstrate a DS41 serving failure, and it does not qualify cancel/drain either. No request was replayed.

Because the dispatcher stops on this failure, cases 13..24 were never sent. In particular the explicit post-cancel recovery, cancel-prefill, busy-admission, post-busy recovery, later low/high/multi-turn/JSON/output/final-idle cases are NOT_SENT. Busy-admission is therefore not qualified.

## Gate accounting

- Required duration: >=7200 s — **NOT MET**.
- Observed start to case-12 terminal runner failure: 3445.005 s (~57m25s) — **INCOMPLETE**.
- Required completed model-bearing cases: >=24 — **NOT MET**.
- Completed cases: 11/24 — all 11 service PASS and semantic PASS.
- Case 12: dispatched but runner FAILED before planned cancel; semantic N/A.
- Cancel/drain control: **NOT QUALIFIED**.
- Busy-admission control: **NOT SENT / NOT QUALIFIED**.
- Same release/epoch residency through observed run: **PASS**.
- Service currently healthy/idle: **PASS**.

## Final state

At inspection the soak unit is inactive/dead and must not be treated as running. K2 remains `READY` and idle on the same release/epoch with no drain visible. The GLM frontend gateway service is unchanged and has `NRestarts=0` since `2026-09-12 10:26:35 CEST`; port 18093 reports gateway/status `ok` while its model backend 18094 is refused/degraded, consistent with GLM model OFF.

No retry, restart, model reload, case replay, or repair was performed in this terminal inspection. A future rerun requires a separately authorized correction of the cancel client's SSE timeout handling and a fresh preregistered soak; the present negative/incomplete evidence must remain preserved.
