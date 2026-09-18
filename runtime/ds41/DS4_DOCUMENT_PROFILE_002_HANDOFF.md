# DS4 DOCUMENT PROFILE 002 — terminal handoff

## Authority and final state
- Terminal decision: **QUALIFIED / LEFT READY** on 2026-09-18.
- Lifecycle owner: `DS4_DOCUMENT_PROFILE_002_20260918`.
- DS4: `READY`, coordinator active, worker active, backend API HTTP200.
- K2: `OFF`.
- Finalizer receipt: `QUALIFIED_LEFT_READY`.
- Terminal evidence: `reports/DS41-Q2-001/ds4-document-profile-002/soak/terminal.json` and `finalizer.json`.
- Detailed closure: `runtime/ds41/results/finalize-ds4-soak-terminal.md`.
- Long-form plan truth: `/home/funboy/STRIX_CLUSTER_ACCELERATION_PLAN.md`.

## Product entry point
- Use the protected local gateway at `127.0.0.1:18224`; Bearer auth is required.
- Tokenizer sidecar: `127.0.0.1:18223`; direct DS4 backend: `127.0.0.1:8080`.
- Post-terminal verification: unauthenticated `/v1/models` HTTP401; authenticated `/health`, `/v1/lifecycle`, `/v1/models` HTTP200; tokenizer `/health` HTTP200.
- Default profile is `document-low`. The separate coding profile is `c-off`.

## Qualified scope
- Documents: `document-low` / `DS4_THINK_LOW`, frozen 2K-class quality panel **6/6 PASS**. Original code2k/docs2k, holdout A/B, and independent code/docs confirmations all passed with cache0.
- Stability: **24/24 sequential soak requests PASS**, HTTP200, cache0, over **8185.734s** real elapsed time. Requirement was >=7200s and >=24 requests.
- Largest document prompt actually quality-qualified: **2039 tokens**. Do not turn the configured context ceiling into a larger quality claim.
- C thinking-OFF qualification from the previous release is preserved and reused; Go LOW / R4 qualification is preserved and reused. They were not rerun in DOCUMENT PROFILE 002.
- Historical document NONE failures remain unchanged.

## Limits
- Engine context ceiling: `16384` tokens.
- Gateway default context/output: `4096` / `2048` tokens; absolute gateway max output `8192`.
- `document-low`: context `4096`, max output `2048`.
- `c-off`: context `16384`, max output `8192`.
- 4K document characterization: `INCOMPLETE_NO_FINAL` with prompt `3486`, cap `512`, finish `length`, no final answer. The prompt was processed; this is not evidence of a hard 2K model/context limit. 8K/16K were not sent.
- Prefill performance target remains unmet: median code2k LOW/cache0 **90.17 tok/s** vs target `200 tok/s`; decode samples **14.41/15.95/16.64 tok/s**.
- End-to-end latency is therefore chat-slow: first code2k quality sample wall **66.180s**; terminal soak wall range **34.109..76.101s**, mean **53.562s**.
- Cancel semantics: client detach followed by backend drain; observed drain about `135.99s` before a new request was admitted.
- OFF chat contract: HTTP503 `model_not_ready`, no autoload.

## Rollback and next action
- Manual rollback command: `scripts/ds4-document-controller.sh rollback-k2`.
- Do not rerun the completed soak. Do not reinterpret the 4K cap512 result as a hard context failure.
- Next work, only if explicitly requested: a separately preregistered 4K+ document budget/latency campaign or performance optimization. The current 2K-class product scope is terminally qualified and should remain stable.
