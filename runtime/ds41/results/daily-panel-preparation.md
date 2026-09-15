# DS41 daily panel preparation

**PASS_WITH_SANDBOX_BLOCKER**

- Two real-source corpora (code and docs/config) frozen locally for 2K/4K/8K/16K/32K/conditional64K; exact V4.1 prompt-token counts and SHA are in the compact JSON.
- Streaming collector self-test PASS: heartbeats/role/usage are not tokens; first-any and first-final are separate.
- Four coding fixtures (2 C, 2 Go) are frozen. Each buggy/missing implementation FAILs its independent test and each private golden PASSes.
- TypeScript not selected because no local `tsc` exists and Node22 rejects typed `.ts`; no package install is authorized.
- Existing isolated sandbox is currently BLOCKED_SYSTEM before any model call: `bwrap: loopback: Failed to create NETLINK_ROUTE socket: Address family not supported by protocol`. Isolation is not weakened.
