# DS4 USABLE RELEASE 001 — preregister

Frozen before any new DS4 model load. Recovery ab41ca9 remains terminal.

- Candidate: kyuz0/ds4@7d0454b4... + verified Antirez Q2, ctx16384, two-host 50/50 TP over private USB4/TCP, DS4_TP_GATE_TIMEOUT_MS=5000, target-only, no DSpark.
- No download, cleanup, full GGUF rehash, driver/kernel/network/global-venv change.
- Code/docs v2 change only the final middle-file instruction to the lower central section. Offline N/expected audit PASS; exact prompt hashes/token counts frozen.
- C v2 is the same C fixture with backend thinking=false + reasoning_effort=none, cap8192; one generation plus at most one repair for malformed envelope or real executed-test feedback.
- Collector self-test SSE/timeout/cancel PASS before load.
- Quality order: code2k-v2 → docs2k-v2 → C-off. Performance is admitted by code/docs PASS even if C remains unqualified.
- If admitted: three independent code2k-v2 diagnostic samples P1/P2/P3, each preceded by an excluded reset; no ignore_eos/padding; prefix-cache usage recorded; then docs confirmation and dependent 4K→8K→16K characterization.
- Product candidate is a separate protected gateway at 18224, direct DS4 coordinator backend8080, CPU tokenizer sidecar18223 proven token-exact on frozen prompts, lifecycle controller explicit. Product gates precede conditional 2h/24-request soak.
- Final: keep DS4 only if the chosen chat/document perimeter including service/stability passes; otherwise rollback K2.
