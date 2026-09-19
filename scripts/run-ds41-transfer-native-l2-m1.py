#!/usr/bin/env python3
from __future__ import annotations

import http.client
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("DS41_TRANSFER_L2_RUN_ID", "l2-m1")
if not RUN_ID or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for ch in RUN_ID):
    raise RuntimeError(f"invalid DS41_TRANSFER_L2_RUN_ID: {RUN_ID!r}")
OUT = ROOT / "reports/DS41-Q2-001/transfer-ds4-native-001" / RUN_ID
MAN = json.load(open(ROOT / "runtime/ds41/document-profile-002/prompt-manifest.json"))
HOST = "127.0.0.1"
PORT = 18221
MODEL = os.environ.get("DS41_TRANSFER_L2_MODEL", "DeepSeek-V4.1-Flash-Q2-Native-M1")
MAX_REQUESTS = 6


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


def request(case_id: str, rec, independent_required: bool = False):
    d = OUT / case_id
    d.mkdir(parents=True, exist_ok=True)
    state_path = d / "state.json"
    if state_path.exists():
        old = json.load(open(state_path))
        if old.get("state") == "COMPLETE" and (d / "validation.json").exists():
            return json.load(open(d / "validation.json"))
        raise RuntimeError(
            f"replay guard {case_id}: existing state={old.get('state')}"
        )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": (ROOT / rec["prompt_file"]).read_text()}
        ],
        "temperature": 0,
        "seed": 1,
        "max_tokens": 2048,
        "stream": False,
        "reasoning_effort": "low",
        "chat_template_kwargs": {
            "enable_thinking": True,
            "reasoning_effort": "low",
            "ds41_prompt_profile": "ds4-low-v1",
        },
    }
    client_request_id = f"transfer001-{RUN_ID}-{case_id}"
    atomic(
        state_path,
        {
            "id": case_id,
            "client_request_id": client_request_id,
            "state": "IN_FLIGHT",
            "started_unix": time.time(),
            "payload": payload,
            "expected": rec["expected"],
        },
    )

    body = json.dumps(payload, separators=(",", ":")).encode()
    started = time.monotonic()
    try:
        conn = http.client.HTTPConnection(HOST, PORT, timeout=1800)
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=body,
            headers={
                "Content-Type": "application/json",
                "X-Request-ID": client_request_id,
            },
        )
        response = conn.getresponse()
        raw = response.read()
        http_status = response.status
        response_headers = dict(response.getheaders())
        conn.close()
    except Exception as exc:
        old = json.load(open(state_path))
        old.update(
            state="FAILED_TRANSPORT",
            finished_unix=time.time(),
            error=repr(exc),
        )
        atomic(state_path, old)
        return {
            "id": case_id,
            "status": "FAILED_TRANSPORT",
            "error": repr(exc),
        }

    wall = time.monotonic() - started
    try:
        obj = json.loads(raw)
    except Exception as exc:
        obj = {
            "_raw": raw.decode(errors="replace"),
            "_json_error": repr(exc),
        }
    atomic(d / "response.json", obj)

    msg = (
        ((obj.get("choices") or [{}])[0].get("message") or {})
        if isinstance(obj, dict)
        else {}
    )
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    finish = (
        (obj.get("choices") or [{}])[0].get("finish_reason")
        if isinstance(obj, dict)
        else None
    )
    try:
        actual = json.loads(content)
        semantic = http_status == 200 and actual == rec["expected"]
    except Exception:
        actual = None
        semantic = False

    usage = (obj.get("usage") or {}) if isinstance(obj, dict) else {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    metrics = obj.get("metrics") if isinstance(obj, dict) else None
    cached = prompt_details.get("cached_tokens")
    independent = cached in (0, None)
    if semantic and (independent or not independent_required):
        status = "PASS"
    elif semantic:
        status = "FAIL_INDEPENDENCE"
    elif finish == "length" and not content:
        status = "INCOMPLETE_NO_FINAL"
    else:
        status = "FAIL_SEMANTIC"

    result = {
        "id": case_id,
        "client_request_id": client_request_id,
        "status": status,
        "semantic_pass": semantic,
        "independent": independent,
        "independent_required": independent_required,
        "http": http_status,
        "wall_s": wall,
        "expected": rec["expected"],
        "actual": actual,
        "reasoning_chars": len(reasoning),
        "final_chars": len(content),
        "finish_reason": finish,
        "usage": usage,
        "metrics": metrics,
        "response_request_id": response_headers.get("x-request-id"),
        "candidate": {
            "target": "Antirez-Q2",
            "engram": "native-GGUF",
            "dspark": False,
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
        result_status=status,
        wall_s=wall,
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
                "schema": "ds41-transfer-native-l2-m1-v1",
                "run_id": RUN_ID,
                "model": MODEL,
                "requests_used": len(rows),
                "max_requests": MAX_REQUESTS,
                "cases": rows,
                "updated_unix": time.time(),
            },
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)

    add(request("l2m1-code2k-v2", record("code2k-middle-explicit-v2")))
    add(request("l2m1-docs2k-v2", record("docs2k-middle-explicit-v2")))
    originals = all(row["status"] == "PASS" for row in rows[:2])
    if originals:
        add(request("l2m1-holdout-a", record("document-holdout-a")))
        add(request("l2m1-holdout-b", record("document-holdout-b")))
    holdouts = (
        originals
        and len(rows) == 4
        and all(row["status"] == "PASS" for row in rows[2:4])
    )
    if holdouts:
        add(
            request(
                "l2m1-code2k-confirm",
                record("code2k-middle-explicit-v2"),
                True,
            )
        )
        add(
            request(
                "l2m1-docs2k-confirm",
                record("docs2k-middle-explicit-v2"),
                True,
            )
        )

    six_pass = len(rows) == 6 and all(row["status"] == "PASS" for row in rows)
    term = {
        "schema": "ds41-transfer-native-l2-m1-terminal-v1",
        "run_id": RUN_ID,
        "model": MODEL,
        "status": "L2_M1_PASS" if six_pass else "L2_M1_FAIL",
        "requests_used": len(rows),
        "six_pass": six_pass,
        "cases": rows,
        "finished_unix": time.time(),
    }
    atomic(terminal, term)
    print(json.dumps(term, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
