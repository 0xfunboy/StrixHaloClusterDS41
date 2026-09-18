# DS4 DOCUMENT PROFILE 002 — preregister

Frozen before model load. USABLE RELEASE 001 `5206483` and Recovery `ab41ca9` remain terminal.

**Offline input contract PASS.** The old CLI/sidecar +6 is exactly the default system turn token sequence `[128799,3476,477,260,11502,22896]`; OpenAI Chat user-only does not inject it. Correcting the sidecar does not alter or repair the old live model inputs. The corrected sidecar matches Antirez embedded token IDs for all frozen NONE/LOW fixtures.

The one candidate is **LOW**:
- thinking enabled;
- parser mode `DS4_THINK_LOW`;
- numeric 1–100 level: none;
- total max_tokens: 2048;
- temperature0, seed1;
- no NONE/false control in the payload;
- same code/docs v2 bytes and same expected.

Main budget is at most six document requests in the mandated order. No semantic retry or repair. Only six-pass admits continuation. Holdouts are frozen before load and expected values are validator-side only.

Lifecycle owner is `DS4_DOCUMENT_PROFILE_002_20260918`; launcher, remote log directory, collector and finalizer are prevalidated. C-off, Go-low and R4 retrieval evidence are reused, not rerun.
