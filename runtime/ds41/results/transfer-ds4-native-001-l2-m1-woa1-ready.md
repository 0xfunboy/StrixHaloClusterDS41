# TRANSFER DS4 → NATIVE 001 — L2 M1 woa1 READY before requests

**READY. Zero L2 quality requests sent.**

- epoch: `1789784656721184713`
- rank0 InvocationID: `207f08f41a984002bc0c87a32da90ba5`, HTTP200
- rank1 InvocationID: `0de289f6c7b9401999272244df71049f`, HTTP200
- paired coordinator: HTTP200
- release: `native-antirez-m1-transfer001-woa1`
- release source: `9fe0496398aae695c9d2486ad7b234a8ee0fde80`
- target Antirez Q2 + native GGUF Engram, target-only M1, DSpark OFF
- live env: canonical-prefill ON, MMQ prefill ON, context 16384, prompt profile `ds4-low-v1`

Both ranks completed load and warmup/profile:
- rank0 load_weights 199.567s; total model load 224.029s; 77.14 GiB; warmup/profile 56.14s
- rank1 load_weights 196.361s; total model load 231.959s; 77.14 GiB; warmup/profile 48.39s

This proves the anonymous-staging loader and WO_A Q8_0 compatibility fix cross the previous startup blockers. It does **not** qualify quality yet.

**NEXT:** run the frozen persistent max6 L2 M1 document runner exactly once. No replay of an existing case state.
