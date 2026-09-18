# FINALIZE DS4 SOAK — terminal qualification

**Status: QUALIFIED / DS4 DOCUMENT PROFILE 002 LEFT READY.**

## Terminal gate
- Owner: `DS4_DOCUMENT_PROFILE_002_20260918`.
- Soak terminal: **PASS**. Real elapsed time: **8185.734s** (2.274h), requirement `>=7200s`.
- Requests: **24/24 sequential PASS**, indices `1..24`, every request HTTP200 and every retained prompt cache count `0`.
- Soak request wall time: min **34.109s**, mean **53.562s**, max **76.101s**.
- Finalizer: `QUALIFIED_LEFT_READY`. DS4 remains READY under the same owner; K2 remains OFF.

## Live product verification after terminal PASS
- Protected gateway: `127.0.0.1:18224`. Unauthenticated `/v1/models` returns HTTP401.
- Authenticated `/health`, `/v1/lifecycle`, and `/v1/models` each return HTTP200. Lifecycle reports owner `DS4_DOCUMENT_PROFILE_002_20260918`, DS4 `READY`, coordinator/worker active, backend API HTTP200, K2 OFF.
- Corrected tokenizer sidecar: `127.0.0.1:18223`, `/health` HTTP200.

## Qualified scope
- `document-low` / `DS4_THINK_LOW`: frozen document panel **6/6 PASS** and terminal stability soak **24/24 sequential PASS**.
- Verified document class is the frozen **2K-class** perimeter; largest passing prompt observed: **2039 tokens**; largest total prompt+completion observed in the verified quality panel: **2789 tokens**.
- Previous **C thinking-OFF** qualification is preserved and reused, not rerun here. Previous **Go LOW / R4** qualification is also preserved and reused.
- Historical document `NONE` failures remain failures; this result does not promote document NONE mode.

## Configured ceilings vs qualified limits
- Engine context ceiling: `16384` tokens. Gateway default context: `4096`.
- `document-low`: context `4096`, max output `2048`.
- `c-off`: context `16384`, max output `8192`. Gateway absolute output ceiling: `8192`.
- These are admission/configuration ceilings, **not** claims that document quality is qualified up to those ceilings. The qualified document perimeter remains the tested 2K-class scope above.

## Performance and known product limits
- Prefill target `200 tok/s` was **not met**. Three independent code2k LOW/cache0 samples: `79.55`, `90.17`, `91.01 tok/s`; median **90.17 tok/s**.
- Corresponding decode: `14.41`, `15.95`, `16.64 tok/s`. First code2k quality wall was `66.180s`.
- 4K characterization is **INCOMPLETE_NO_FINAL**, not a hard 2K model limit: prompt `3486` tokens was processed, but cap `512` ended on `length` before any final answer. 8K/16K were not run by dependent-stop rule.
- Cancel is detach+drain, not immediate compute abort; product testing observed about `135.99s` before admission was released.
- While lifecycle is OFF, chat returns source-contract HTTP503 `model_not_ready` and does not autoload DS4.

## Operations
- Keep DS4 READY through the protected gateway for this qualified scope.
- Manual rollback to K2 remains available with: `scripts/ds4-document-controller.sh rollback-k2`.
- No further context/performance expansion is implied by this qualification. Any 4K+ document qualification is a separate experiment with a separately frozen output budget.
