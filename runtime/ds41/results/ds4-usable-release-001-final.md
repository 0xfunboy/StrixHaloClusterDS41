# DS4 USABLE RELEASE 001 — terminal report

| Campo | Risultato richiesto |
|---|---|
| Recupero informazioni | R4 DS4 non-ambiguous retrieval evidence remains applicable and preserved: tail1546 PASS, discriminator1571 exact PASS, holdouts 4/4 PASS, JSON/fresh/multiturn 3/3 PASS. Not rerun. |
| Code/docs istruzione chiarita | **FAIL / FAIL**. code2k-v2 returns result52 + first correct but middle=`prompt.go`, last=`test-ds41-prefill-metrics.py`. docs2k-v2 returns result52 + first/last correct but middle=`README.md`. Frozen expected unchanged. |
| C thinking-off / Go low | **C thinking-off PASS** first attempt, 0 repair, reasoning chars0, natural stop, sanitizer/private tests rc0. Historical C-low remains INCOMPLETE. Go-low R4 PASS is preserved. |
| Prefill | No three-sample benchmark admitted. Mandatory-call diagnostics: code 18.339s / 88.83 tok/s; docs 28.086s / 46.50 tok/s; C-off 11.843s / 50.41 tok/s. All cache0. |
| Decode / primo finale / tempo alla soluzione | code: 8.26 tok/s, first final18.599s, wall23.322s (semantic FAIL); docs:15.18, first final28.813s, wall31.756s (semantic FAIL); C-off:14.83, first final11.961s, wall31.573s and test PASS. |
| Contesto | Engine allocated 16384, but no new context qualification: dependent 4K→8K→16K progression was NOT_ADMITTED after 2K document gate failure. |
| Servizio | Direct DS4 pair reached READY/API200. Protected gateway/service/cancel/ON-OFF/soak were NOT_ADMITTED after document quality failure. Candidate gateway/tokenizer files remain experimental and not deployed. |
| Profilo residente | DS4 is OFF. K2 `5bdfed6` restored READY epoch `1789720948796205781`, rank0/rank1/paired HTTP200, gateway HTTP200, final none-mode smoke `323`/stop PASS. |

## Decision

Recovery `ab41ca9` remains terminal. DS4 USABLE RELEASE 001 is also terminal with **NO PROMOTION**.

The new lower-middle instruction did not rescue the mandatory document cases, so the preregistered document gate blocked P1/P2/P3, context progression, protected-gateway qualification and soak. The mandatory calls nevertheless show useful but non-qualified performance observations below the 200 tok/s project target.

The C-specific result is positive and separate: thinking was genuinely off (`thinking=false`, `reasoning_effort=none`), final code appeared on the first attempt, no repair was used, and the frozen private test harness passed. This does not rewrite the historical C-low INCOMPLETE result.

Preload CLI token estimates were +6 versus live server usage on all three mandatory prompts (1635→1629, 1312→1306, 603→597). The prepared tokenizer sidecar/config therefore is **NOT_ADMITTED** as live-server equivalence evidence and was never deployed.

Final verdicts:
- `performance_target_met=false`
- `quality_qualified_scope=false`
- `coding_c_thinking_off_qualified=true`
- `ds4_usable_release_promoted=false`
- `delivery_complete=true`

Antirez Q2 remains verified on both nodes for future explicitly authorized work. No new download, cleanup or full GGUF rehash was performed.
