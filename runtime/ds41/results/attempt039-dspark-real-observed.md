# DS41-Q2-001 attempt039 — real DSpark K=1 observed arm B

**Status:** `PASS / B039 REAL DSPARK QUALIFIED OFFLINE / A040 PENDING`

Saved-output validation only; no model rerun. Rank0/rank1 token streams, text and per-request speculative metrics are identical for all 9 requests. Independent coding validator PASSes all 9 frozen cases; arithmetic/JSON/reasoning-high PASS.

Speed128 replicas: `15.63405532`, `17.39316440`, `17.33900570` tok/s; mean **16.78874181**, median **17.33900570**, sample SD **1.00035441**, CV **5.958%**. These are three repetitions of the same frozen speed prompt. Acceptance is **174/210 = 82.857%** on those three runs. Functional64 contains 7 natural rejections and completes 64 emitted tokens.

Real proposer timing over the three speed runs: **2554.158 ms** CUDA-stream span over 213 proposal calls = **11.991 ms/call**. Host enqueue total `2845.403 ms` is explicitly non-additive because it overlaps GPU/runtime work.

Model-load footprint observed **93.85 GiB/rank**, versus historical target-only 80.37 GiB/rank: **+13.48 GiB/rank**. No leak conclusion is drawn.

Attempt040 M1 control had only its frozen prompt spec at this checkpoint and had not started.
