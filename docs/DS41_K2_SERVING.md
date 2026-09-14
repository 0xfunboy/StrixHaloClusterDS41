# DS41 K2 serving release

This deployment reuses HaloClu's authenticated gateway/frontend and paired OpenAI coordinator. It does not replace the GLM gateway and it never autoloads the model on page refresh, login, health checks, or reboot.

- Gateway/UI: `http://127.0.0.1:18222/`
- Paired coordinator (internal): `127.0.0.1:18221`
- Rank APIs (private USB4): `10.55.0.1:18220`, `10.55.0.2:18220`
- Model: `DeepSeek-V4.1-Flash-MixedQ2-DSpark-K2`
- Preset: `dspark-k2-gfx1151`, adaptive verification OFF, TP2/EP2.
- Qualified serving context: 4096 tokens; output controls are deliberately bounded to the configured 1024-token maximum.

## Authentication

The gateway uses HaloClu's existing local authentication contract. API clients read the private token from `/home/funboy/.local/state/haloclu-ds41/api-token` and send it as a Bearer token. The browser exchanges the raw token for an HttpOnly same-origin session; it does not persist the raw credential in browser storage.

```bash
TOKEN=$(cat /home/funboy/.local/state/haloclu-ds41/api-token)
curl -sS -H "Authorization: Bearer $TOKEN" http://127.0.0.1:18222/v1/lifecycle
```

Do not print or commit the token.

## Whole-pair ON / OFF

```bash
TOKEN=$(cat /home/funboy/.local/state/haloclu-ds41/api-token)
curl -sS -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"confirm":true}' http://127.0.0.1:18222/v1/lifecycle/on

curl -sS -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"confirm":true}' http://127.0.0.1:18222/v1/lifecycle/off
```

ON loads the frozen K2 release on both ranks, requires both private APIs and the paired coordinator to become healthy, then executes a small real paired inference before the asynchronous lifecycle action completes. OFF waits for an admitted paired request to drain and then stops and verifies both DS41 ranks. It refuses to stop a pair owned by an experimental/research run.

## Chat API

Use OpenAI-compatible `/v1/chat/completions`; streaming and non-streaming are supported. Only one paired generation is admitted at a time. Refresh/status/chat while OFF never starts the model.

```bash
TOKEN=$(cat /home/funboy/.local/state/haloclu-ds41/api-token)
curl -sS -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash-MixedQ2-DSpark-K2","messages":[{"role":"user","content":"Compute 17*19. Return only the integer."}],"temperature":0,"max_tokens":128,"stream":false}' \
  http://127.0.0.1:18222/v1/chat/completions
```

Browser Stop or client disconnect detaches that client. The paired coordinator deliberately drains both rank requests before releasing the pair; it does not kill a tensor-parallel collective midway. OFF waits for this drain.

## Rollback

The serving release uses the tracked `dspark-k2-gfx1151` preset. The tracked rollback chain is K2 → qualified K1 → M1. K3 research never changes the stable K2 release symlink or serving preset automatically.
