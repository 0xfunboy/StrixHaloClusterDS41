#!/usr/bin/env python3
from __future__ import annotations

import http.client
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("DS41_ATTN002_RUN_ID", "p2-m1")
if not RUN_ID or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for ch in RUN_ID):
    raise RuntimeError(f"invalid run id {RUN_ID!r}")

OUT = ROOT / "reports/DS41-Q2-001/attention-parity-002" / RUN_ID
MAN = json.load(open(ROOT / "runtime/ds41/document-profile-002/prompt-manifest.json"))
HOST = "127.0.0.1"
PORT = 18221
MODEL = os.environ.get(
    "DS41_ATTN002_MODEL", "DeepSeek-V4.1-Flash-Q2-AttnParity002-M1"
)
if not MODEL:
    raise RuntimeError("empty DS41_ATTN002_MODEL")
MAX_MAIN_REQUESTS = 6


def atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def record(ident: str):
    for group in ("documents", "holdouts"):
        for row in MAN[group]:
            if row["id"] == ident:
                return row
    raise KeyError(ident)


def prompt_text(rec) -> str:
    return (ROOT / rec["prompt_file"]).read_text()


def classify_result(
    *,
    http_status: int,
    content: str,
    reasoning: str,
    finish: str | None,
    expected,
    usage,
    metrics,
    independent_required: bool,
):
    try:
        actual = json.loads(content)
        semantic = http_status == 200 and actual == expected
    except Exception:
        actual = None
        semantic = False
    prompt_details = (usage or {}).get("prompt_tokens_details") or {}
    cached = prompt_details.get("cached_tokens")
    if cached is None and isinstance(metrics, dict):
        cached = metrics.get("prompt_tokens_cached")
    independent = cached in (0, None)
    if semantic and (independent or not independent_required):
        status = "PASS"
    elif semantic:
        status = "FAIL_INDEPENDENCE"
    elif http_status != 200:
        status = "FAILED_HTTP"
    elif finish == "length" and not content:
        status = "INCOMPLETE_NO_FINAL"
    else:
        status = "FAIL_SEMANTIC"
    return status, semantic, independent, actual


def request(case_id: str, rec, independent_required: bool = False):
    d = OUT / case_id
    d.mkdir(parents=True, exist_ok=True)
    state_path = d / "state.json"
    if state_path.exists():
        old = json.load(open(state_path))
        raise RuntimeError(f"replay guard {case_id}: existing state={old.get('state')}")

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt_text(rec)}],
        "temperature": 0,
        "seed": 1,
        "max_tokens": 2048,
        "stream": True,
        "stream_options": {"include_usage": True},
        "reasoning_effort": "low",
        "chat_template_kwargs": {
            "enable_thinking": True,
            "reasoning_effort": "low",
            "ds41_prompt_profile": "ds4-low-v1",
        },
    }
    client_request_id = f"attn002-{RUN_ID}-{case_id}"
    started_unix = time.time()
    atomic(
        state_path,
        {
            "id": case_id,
            "client_request_id": client_request_id,
            "state": "IN_FLIGHT",
            "started_unix": started_unix,
            "payload": payload,
            "expected": rec["expected"],
        },
    )

    body = json.dumps(payload, separators=(",", ":")).encode()
    started = time.monotonic()
    raw_path = d / "stream.sse"
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage = {}
    metrics = {}
    finish = None
    http_status = 0
    response_headers = {}
    first_delta_s = None
    first_reasoning_s = None
    first_final_s = None
    chunks = 0

    try:
        conn = http.client.HTTPConnection(HOST, PORT, timeout=1800)
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "X-Request-ID": client_request_id,
            },
        )
        response = conn.getresponse()
        http_status = response.status
        response_headers = dict(response.getheaders())
        with raw_path.open("wb") as raw:
            if http_status != 200:
                data = response.read()
                raw.write(data)
                try:
                    err_obj = json.loads(data)
                except Exception:
                    err_obj = {"raw": data.decode(errors="replace")}
                atomic(d / "response.json", err_obj)
            else:
                while True:
                    line = response.readline()
                    if not line:
                        break
                    raw.write(line)
                    raw.flush()
                    stripped = line.strip()
                    if not stripped.startswith(b"data:"):
                        continue
                    data = stripped[5:].strip()
                    if data == b"[DONE]":
                        break
                    if not data:
                        continue
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    chunks += 1
                    elapsed = time.monotonic() - started
                    if obj.get("usage"):
                        usage = obj["usage"]
                    if obj.get("metrics"):
                        metrics = obj["metrics"]
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    ch = choices[0]
                    if ch.get("finish_reason") is not None:
                        finish = ch.get("finish_reason")
                    delta = ch.get("delta") or {}
                    reasoning_piece = (
                        delta.get("reasoning_content")
                        or delta.get("reasoning")
                        or ""
                    )
                    content_piece = delta.get("content") or ""
                    if (reasoning_piece or content_piece) and first_delta_s is None:
                        first_delta_s = elapsed
                    if reasoning_piece:
                        if first_reasoning_s is None:
                            first_reasoning_s = elapsed
                        reasoning_parts.append(reasoning_piece)
                    if content_piece:
                        if first_final_s is None:
                            first_final_s = elapsed
                        content_parts.append(content_piece)
        conn.close()
    except Exception as exc:
        old = json.load(open(state_path))
        old.update(
            state="FAILED_TRANSPORT",
            finished_unix=time.time(),
            error=repr(exc),
        )
        atomic(state_path, old)
        result = {
            "id": case_id,
            "client_request_id": client_request_id,
            "status": "FAILED_TRANSPORT",
            "error": repr(exc),
            "wall_s": time.monotonic() - started,
            "first_delta_s": first_delta_s,
            "first_final_s": first_final_s,
        }
        atomic(d / "validation.json", result)
        return result

    wall = time.monotonic() - started
    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)

    if http_status != 200:
        result = {
            "id": case_id,
            "client_request_id": client_request_id,
            "status": "FAILED_HTTP",
            "http": http_status,
            "wall_s": wall,
            "first_delta_s": first_delta_s,
            "first_reasoning_s": first_reasoning_s,
            "first_final_s": first_final_s,
            "response_request_id": response_headers.get("x-request-id"),
        }
    else:
        summary = {
            "id": case_id,
            "object": "chat.completion.stream.aggregate",
            "model": MODEL,
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "reasoning_content": reasoning,
                        "content": content,
                    },
                    "finish_reason": finish,
                }
            ],
            "usage": usage,
            "metrics": metrics,
            "stream_chunks": chunks,
        }
        atomic(d / "response.json", summary)
        status, semantic, independent, actual = classify_result(
            http_status=http_status,
            content=content,
            reasoning=reasoning,
            finish=finish,
            expected=rec["expected"],
            usage=usage,
            metrics=metrics,
            independent_required=independent_required,
        )
        result = {
            "id": case_id,
            "client_request_id": client_request_id,
            "status": status,
            "semantic_pass": semantic,
            "independent": independent,
            "independent_required": independent_required,
            "http": http_status,
            "wall_s": wall,
            "first_delta_s": first_delta_s,
            "first_reasoning_s": first_reasoning_s,
            "first_final_s": first_final_s,
            "expected": rec["expected"],
            "actual": actual,
            "reasoning_chars": len(reasoning),
            "final_chars": len(content),
            "finish_reason": finish,
            "usage": usage,
            "metrics": metrics,
            "stream_chunks": chunks,
            "response_request_id": response_headers.get("x-request-id"),
            "candidate": {
                "campaign": "DS41_V41_ATTENTION_PARITY_002",
                "target": "Antirez-Q2",
                "engram": "native-GGUF",
                "dspark": False,
                "attention_parity": True,
                "window": "full512-fp8-block32-ue8m0-qdq-bf16",
                "compressed": "nvfp4-block16-e4m3-qdq-bf16",
                "indexer": "mxfp4-block32-ue8m0",
                "mmq_prefill": True,
                "canonical_prefill": True,
                "prompt_profile": "ds4-low-v1",
            },
        }

    atomic(d / "validation.json", result)
    old = json.load(open(state_path))
    old.update(
        state="COMPLETE",
        finished_unix=time.time(),
        result_status=result["status"],
        wall_s=wall,
        first_final_s=first_final_s,
    )
    atomic(state_path, old)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    terminal = OUT / "terminal.json"
    if terminal.exists():
        raise SystemExit("terminal exists; refusing replay")

    registry_path = OUT / "registry.json"
    rows = []

    def add(row) -> None:
        rows.append(row)
        atomic(
            registry_path,
            {
                "schema": "ds41-v41-attention-parity-002-p2-m1-v1",
                "run_id": RUN_ID,
                "model": MODEL,
                "requests_used": len(rows),
                "max_main_requests": MAX_MAIN_REQUESTS,
                "cases": rows,
                "updated_unix": time.time(),
            },
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)

    code = request("attn002-code2k-v2", record("code2k-middle-explicit-v2"))
    add(code)
    technical = code["status"] in ("FAILED_TRANSPORT", "FAILED_HTTP")
    if not technical:
        docs = request("attn002-docs2k-v2", record("docs2k-middle-explicit-v2"))
        add(docs)
        technical = docs["status"] in ("FAILED_TRANSPORT", "FAILED_HTTP")

    originals = (
        not technical
        and len(rows) == 2
        and all(row["status"] == "PASS" for row in rows[:2])
    )
    if originals:
        add(request("attn002-holdout-a", record("document-holdout-a")))
        add(request("attn002-holdout-b", record("document-holdout-b")))

    regressions = (
        originals
        and len(rows) == 4
        and all(row["status"] == "PASS" for row in rows[2:4])
    )
    if regressions:
        add(
            request(
                "attn002-code2k-confirm",
                record("code2k-middle-explicit-v2"),
                independent_required=True,
            )
        )
        add(
            request(
                "attn002-docs2k-confirm",
                record("docs2k-middle-explicit-v2"),
                independent_required=True,
            )
        )

    six_pass = len(rows) == 6 and all(row["status"] == "PASS" for row in rows)
    if technical:
        status = "P2_M1_TECHNICAL_FAIL"
    elif six_pass:
        status = "P2_M1_SIX_PASS"
    else:
        status = "P2_M1_FAIL"

    term = {
        "schema": "ds41-v41-attention-parity-002-p2-terminal-v1",
        "run_id": RUN_ID,
        "model": MODEL,
        "status": status,
        "requests_used": len(rows),
        "six_pass": six_pass,
        "cases": rows,
        "finished_unix": time.time(),
    }
    atomic(terminal, term)
    print(json.dumps(term, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
