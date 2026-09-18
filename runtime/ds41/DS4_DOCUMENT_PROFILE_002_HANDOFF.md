# DS4 DOCUMENT PROFILE 002 — handoff

## Authority
- Starts from terminal USABLE RELEASE 001 `5206483`; Recovery `ab41ca9` stays terminal.
- Truth: `/home/funboy/STRIX_CLUSTER_ACCELERATION_PLAN.md`.

## Entry
- K2 `5bdfed6` READY epoch `1789720948796205781`, 200/200/200.
- DS4 OFF; no stale document job.
- Antirez Q2 receipt/SHA already verified on both nodes; no rehash/download.

## Offline input closure
- CLI/old-sidecar +6 cause is exactly `<｜System｜>You are a helpful assistant` = IDs `[128799,3476,477,260,11502,22896]`.
- Server OpenAI user-only does not inject this system turn. Old USABLE001 direct-backend inputs were already correct; sidecar fix changes counting/admission only.
- Corrected sidecar exact-ID matches Antirez embedded tokenizer for user/system/multiturn NONE/LOW and actual frozen prompts.
- LOW = named `DS4_THINK_LOW`, thinking enabled, numeric level absent (`-1`), no Reasoning Effort system text; versus NONE only generation-prefix token changes `128822 -> 128821`.
- Original LOW counts: code1629, docs1306. Holdouts: A1409, B2039.

## Frozen main budget
1. code2k-v2 LOW / cap2048
2. docs2k-v2 LOW / cap2048
3-4. holdouts only if originals PASS
5-6. independent original confirms only if holdouts PASS
No semantic retries/repairs.

## Prepared continuation
- Six-pass only: reuse code quality samples + one third comparable perf request; then 4K→8K→16K LOW cap512.
- Product gate: protected gateway18224 + corrected tokenizer18223; auth/SSE/cancel-drain/OFF-no-autoload/whole-pair lifecycle.
- Soak: 2h / >=24 sequential requests only after product PASS.
- Finalizer owner-bound: failure => K2 rollback; full qualified perimeter => leave DS4 READY.

## Quality checkpoint — originals
- code2k-v2 LOW PASS exact: cache0, prefill20.478s/79.55tok/s, first final64.349s, decode14.41tok/s, wall66.180s, 657 completion tokens.
- docs2k-v2 LOW PASS exact: cache0, prefill25.570s/51.08tok/s, first final43.196s, decode15.84tok/s, wall45.703s, 318 completion tokens.
- Historical NONE FAILs remain unchanged. No prompt/expected/parameter change.
- Gate opens holdout A/B in the same DS4 load.

## Quality checkpoint — holdouts
- document-holdout-a PASS exact 10/10: cache0, prefill18.600s/75.75tok/s, first final59.554s, decode15.21tok/s, wall64.087s.
- document-holdout-b PASS exact 10/10: cache0, prefill22.294s/91.46tok/s, first final65.168s, decode15.73tok/s, wall70.021s.
- Quality budget is 4/4 PASS with 2 requests remaining.

## NEXT
Persistent quality runner is executing independent code2k/docs2k confirmations 5/6 and 6/6 under the same LOW profile. Each confirmation requires semantic PASS and cached_tokens<=32.

## Startup READY
- DOCUMENT PROFILE 002 first startup reached READY before any document request: coordinator+worker active, API200, ctx16384, 50/50 TCP/USB4, gate5000, target-only/noDSpark, Engram disk-only, planned82.67GiB/rank.
- K2 is OFF. NEXT is the frozen max6 document-quality runner.

## Runner setup negative 001
- First quality unit exited before any HTTP: tracked manifest now exposes `documents` as a list but `rec()` still used the old `DOC_AUD["records"]` shape. TypeError occurred before case state/registry creation.
- Requests sent=0, budget consumed=0; DS4 remained READY and K2 OFF.
- Fixed quality + continuation + soak list lookups; py_compile and manifest lookup preflight PASS. Retry stays in the same DS4 load and is not a semantic replay.

## Document quality terminal — 6/6 PASS
- Original code/docs PASS, holdout A/B PASS10/10, independent code/docs confirmations PASS with cached_tokens=0.
- LOW profile only; historical NONE FAILs remain preserved. Main document budget consumed exactly6/6, no retries or repairs.
- Six-pass admits the preregistered continuation in the same DS4 load: reuse code original+confirm plus one third sample, then 4K→8K→~16K.

## Performance checkpoint
- Three independent PASS/cache0 code2k LOW samples: prefill79.55 /90.17 /91.01 tok/s; median90.17, sample SD6.39. Decode14.41/15.95/16.64.
- `performance_target_met=false` versus200 tok/s target. This is DS4+AntirezQ2+native-Engram contract, not engine-only attribution.
- Context characterization is running 4K→8K→~16K; a larger-context FAIL limits escalation but does not erase 2K qualification.
