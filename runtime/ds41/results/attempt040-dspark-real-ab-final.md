# DS41-Q2-001 attempt039/040 — real DSpark K=1 final A/B

**Status:** `PASS / EXPERIMENTAL_DSPARK_K1_QUALIFIED / PRODUCTION_DEFAULT_UNCHANGED`

Execution order was **B039 DSpark K1 first, then A040 M1**, in two separate supervised loads; this was not same-load and not alternated. Frozen prompt panel is identical.

- A040 M1 speed128: `11.67359908`, `12.59395492`, `12.59663991` tok/s; mean **12.28806464**, median **12.59395492**, sample SD **0.53214447**, CV **4.331%**.
- B039 DSpark K1: `15.63405532`, `17.39316440`, `17.33900570` tok/s; mean **16.78874181**, median **17.33900570**, sample SD **1.00035441**, CV **5.958%**.
- Ratio-of-means decode gain: **+36.6264%**. Mean offline wall **15.18052s -> 12.53062s**, reduction **17.4559%** (wall-throughput ratio +21.1474%). Mean TTFT **4.83188s -> 4.94725s**, B-A **+115.4ms**.

All greedy outputs are token-for-token identical A/B and rank0/rank1, including stop/finish reasons, on warmup, functional, all three speed runs and arithmetic/coding/JSON/reasoning-high. Independent saved-code validator PASSes all 9 coding cases in both arms. Arithmetic, exact JSON and reasoning-high10 PASS. Attempt039 functional64 rejected7/35 real drafts and later accepted again, completing64 final tokens. Speed acceptance is **174/210 = 82.857%**; the three speed runs repeat the same frozen prompt.

Real DSpark proposer telemetry on the three speed requests: **2554.158ms / 213 calls = 11.991ms/call** CUDA stream span. Host enqueue is non-additive/overlapping. The proposer span is 6.794% of summed client wall if normalized, but must not be subtracted from wall because overlap exists.

Memory: target-only model load **80.37GiB/rank**; target+DSpark **93.85GiB/rank**, +**13.48GiB/rank**. Header accounting shows BF16-dequantized local expert weights alone are **12.65625GiB/rank**; the remaining ~**0.824GiB/rank** covers the rest of the draft state/params/buffers to the extent observable and is not classified as a leak.

039/040 did not recapture logits. Target numeric fidelity/rejection gates remain the exact attempt036 qualification because target math and T2/T3/T4 verifier dispatch were unchanged; this integration adds exact emitted-token equality across the full frozen A/B panel.

**Decision:** qualify the current K1 integration as an **experimental preset**, with explicit M1 rollback. Do not replace production default automatically. No K sweep or kernel tuning is part of this result.

Source identity: B039 executed integration commit `10d29383...`; A040 executed `5360caee...`, whose only delta is the two tracked B039 evidence files. No runtime/model/runner/kernel code changed between arms.
