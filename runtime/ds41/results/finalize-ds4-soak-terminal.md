# FINALIZE DS4 SOAK — terminal

- `performance_target_met=false`: target 200 tok/s not reached. Best MMQ+Engram sample 76.17 tok/s; cache-characterized confirmation 66.74 tok/s.
- `quality_qualified_scope=false`: code2k, tail1546 and docs2k fail. The frozen extraction discriminator returns the **exact same failing content** on optimized `9c13117f` and rollback `5bdfed6`, proving the observed retrieval failure is shared/pre-existing, not introduced by MMQ+Engram.
- `delivery_complete=true`: MMQ and Engram performance work, CED blocker, quality panel and discriminator are preserved; optimized release remains experimental only.
- `soak=NOT_RUN_BLOCKED_BY_QUALITY`: no final soak is valid under the mandate.
- Final operational state: rollback `k2-prefill-5bdfed6` / `5bdfed698...` READY.
