# DS41 useful-prefill upstream component result

- **Reusable winner:** DS4 dynamic raw-layout MMQ (`8db1d1d...`).
- Existing 8 real EP2 fixtures: **8/8 operational PASS**, EP2 map exact.
- Pair-critical representative sum: **3855.916 -> 736.734 ms**, **5.234x**, **80.89%** less time.
- Fixture speedup range: **4.07x - 6.32x**.
- M128 CED-shape controls: **2.76x - 3.28x**.
- Current AMD path has no decoder tail-only CED replay; 80 routed calls = 40 layers x 2 chunks, expected for current implementation.
- Representative model reproduces current routed profile: **77.118s** model vs **77.064s** measured.
- Component projections (not full-model benchmarks): DS4 no-CED **14.735s** routed-only; current kernel + CED128 **49.157s**; DS4 + CED128 **10.702s**.
- 200 TPS on 1588 tokens implies **7.94s total**; routed-only projected cost still exceeds it, and Engram 8.9-10.1s is separately over budget.
- `torch-ggml-ops@9d7ddbf...` exact packaged/generated experiment is not directly runnable at V4.1 K5120/N2304/local192 without new codegen/adaptation; no upstream V4 timing is relabeled as a DS41 result.
- Precision differs: DS4 uses F32 boundaries + internal Q8_1 activation quantization; output rel-L2 across fixtures **0.0135-0.0274**. No equivalence/promotion is claimed.
