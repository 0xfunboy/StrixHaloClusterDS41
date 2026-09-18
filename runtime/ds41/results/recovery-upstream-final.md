# DS41 Recovery Upstream — terminal delivery

| Campo | Profilo locale conservato | Profilo corretto/alternativo | Verdetto |
|---|---|---|---|
| Retrieval originale e holdout | K2/DenseFix reproduces tail/discriminator retrieval failures | DS4 Q2: tail PASS, discriminator exact PASS, 4/4 holdouts PASS; code/docs frozen middle-only mismatch on even section counts | DS4 materially recovers retrieval, strict frozen panel still not all-PASS |
| Codice C/Go e test | K2 historical coding was incomplete/non-verified in this panel | DS4: Go PASS `go test -race`; C INCOMPLETE with 8192 reasoning tokens and 0 final code | Not fully qualified |
| Prefill1588, s e token/s | prior K2 figures are workload/release-specific | DS4 quality observation ~22.240 s / 71.51 tok/s | Diagnostic only; performance arm not admitted |
| TTFT / primo finale | not compared under one qualified performance protocol | no dedicated qualified TTFT sample | N/A for promotion |
| Decode e tempo alla risposta corretta | K2 has qualified short-panel decode history but retrieval defect | DS4 quality request ~14.30 tok/s decode | Diagnostic only |
| Contesto qualificato / memoria | K2 serving supports larger configured context but failed frozen retrieval | DS4 R4 ctx16384; planned ~82.67 GiB/rank; 4 retrieval holdouts PASS | R4 scope only |
| Backend, pesi, Engram, DSpark effettivi | vLLM DenseFix + affine2 Engram + DSpark K2 | DS4 `7d0454b4...`, calibrated Antirez Q2 SHA `1ce6a8f8...`, native DS4 Engram disk-only, no DSpark | Alternative contract is materially different |
| Stato servizio / stabilità | rollback K2 is known READY target | DS4 default750ms TP timeout failed; source-supported 5000ms transport gate completed remainder | DS4 requires transport timeout override on this USB4 path |

Cause status: R1 excluded HF#12 ownership locally and validated local arithmetic endpoints; R2 showed contract differences without a safe exact vLLM patch. R4 demonstrates that an independent V4.1 implementation/checkpoint contract recovers the non-ambiguous retrieval failures. This does not prove one single local vLLM operator was the sole cause.

Final verdict: `performance_target_met=false`; `quality_qualified_scope=false`; `delivery_complete=true`; DS4 `NO_PROMOTION`; restore `k2-prefill-5bdfed6` READY. Preserve DS4 Q2 and all R4 raw evidence for a future specifically authorized coding/profile continuation.
