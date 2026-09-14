# SERVE-K2-001 model-free gate

**PASS / NO MODEL LOAD.** The existing HaloClu authenticated frontend/gateway and native paired coordinator are reused. New DS41 lifecycle endpoints are fixed-command only, require authentication plus `confirm:true`, do not autoload on health/status/chat, reject concurrent lifecycle transitions, and expose OFF/STARTING/READY/RESEARCH_BUSY/ERROR from the real pair state. Existing paired cancellation/drain/shutdown tests PASS. Existing browser authentication tests PASS. Frontend unit tests: 40/40 PASS. Shell/Python syntax and the Go binary build with the already-installed offline toolchain PASS.

Serving ports are isolated from GLM: gateway `127.0.0.1:18222`, internal paired coordinator `127.0.0.1:18221`, private rank APIs `10.55.0.1:18220`/`10.55.0.2:18220`. No weights, Engram, sidecar or venv are duplicated.

The complete Go suite is not used as a serving gate: sandbox-dependent pre-existing tests fail in this tool harness because bubblewrap cannot create a NETLINK_ROUTE socket, and unrelated existing product tests remain outside this bounded adapter. The focused lifecycle/paired/auth suite passes.
