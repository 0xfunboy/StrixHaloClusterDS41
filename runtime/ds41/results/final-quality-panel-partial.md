# FINAL quality panel — partial checkpoint

- Source/release: `9c13117f56fd81d03c8c610a5fb0148ccdc603b9` / `k2-mmq-engram-9c13117`; epoch `1789639579678586954`.
- code-2k current MMQ+Engram sample: FAIL (reused, not resent).
- tail1546: FAIL, exact historical failure repeated: expected `{"end":5}`, actual `{"end":23}`; 1546/computed1546/cache0; prefill20.797611s /74.335460 tok/s.
- docs2k: FAIL, expected result52/first+middle+last frozen; actual result58 / first `dspark-k3-gfx1151.md` / middle `README.md` / last `dspark-k3-gfx1151.md`; 1265/cache0; prefill20.476592s /61.777859 tok/s.
- JSON none PASS; fresh marker PASS; multi-turn ORBIT PASS.
- C `c-frame-stream`: `INCOMPLETE_NO_FINAL`, 728 prompt +4096 completion all reasoning, 0 final; product never executed tests => NON_VERIFIED. No retry.
- Go `go-session-state` is IN_FLIGHT at this checkpoint; logits common-prefix and speed128 remain NOT_SENT.
- Quality is not qualified at this checkpoint; soak remains blocked.
