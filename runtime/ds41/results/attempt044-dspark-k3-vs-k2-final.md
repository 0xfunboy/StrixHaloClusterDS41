# DS41 attempt043/044 — DSpark K3 vs K2 final

**PASS / QUALIFY_K3_EXPERIMENTAL / STABLE_SERVING_REMAINS_K2**

- Order: B043 K3 -> A044 K2, separate loads, same source `3823b380...`.
- K2 mean decode: **19.38732535 tok/s**, median 20.06510808.
- K3 mean decode: **20.41979589 tok/s**, median 21.01309502.
- Contemporary gain: **+5.3255%**, 3/3 same-index wins.
- K3 acceptance: 258/396 = 65.152%.
- K3 full-width histogram 0/1/2/3: {'0': 21, '1': 24, '2': 27, '3': 60}.
- K3 first/second|first/third|first2: 84.091% / 78.378% / 68.966%.
- Verified final tokens per verify call: K2 2.442308, K3 2.886364.
- Proposer GPU stream per call: K2 13.703ms, K3 15.455ms; total span K2 2178.707ms, K3 2086.482ms.
- Exact rank coherence and K2/K3 token/text/finish/stop equality PASS over the frozen panel.
- Independent coding 9/9, arithmetic, JSON, text and reasoning-high PASS both arms.
- Validator SHA256: `cd0e089d2a2c82aac391928bb0e936cb642af1cb71b2a6d14ac6393d77fab9a7`.
