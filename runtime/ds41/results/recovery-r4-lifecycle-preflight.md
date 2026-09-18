# DS41 Recovery R4 lifecycle preflight

Result: **BLOCKED_MODEL_MIRROR_NODE01** (expected exit 21).

Model-free/fail-closed checks PASS for DS4 binary identity on both nodes, USB4 reachability, bindability of coordinator/control ports, NODE02 Q2 size, and current K2 READY ownership. NODE01 does not yet contain the Q2, so lifecycle admission is blocked before any stop/start/model load.
