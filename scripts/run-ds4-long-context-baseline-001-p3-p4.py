#!/usr/bin/env python3
from __future__ import annotations

import fcntl
import hashlib
import http.client
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

CAMPAIGN = "DS4_LONG_CONTEXT_BASELINE_001"
ROOT = Path("/home/funboy/StrixHaloClusterDS41")
RAW = Path("/home/funboy/reports/DS4-LONG-CONTEXT-BASELINE-001")
P1P2 = RAW / "p1-p2-terminal.json"
P3_MANIFEST = ROOT / "runtime/ds41/long-context-baseline-001/p3-frozen-manifest.json"
P4_MANIFEST = ROOT / "runtime/ds41/long-context-baseline-001/p4-frozen-manifest.json"
CONFIG = ROOT / "runtime/ds41/config.ds4-long-context-baseline-001.json"
NODE_RUNNER = ROOT / "scripts/run-ds4-long-context-baseline-001-node.sh"
TOKENIZER_RUNNER = ROOT / "scripts/serve-ds4-long-context-baseline-001-tokenizer.py"
STRIXGLM = Path("/home/funboy/.local/share/haloclu-ds41/current/bin/strixglm")
VENV = Path("/home/funboy/StrixHaloClusterGLM/.engine/venv")
MODEL = Path("/home/funboy/models/ds41/ds4-v41-q2/DeepSeek-V4.1-Flash-Q2.gguf")
PROD_TOKEN = Path("/home/funboy/.local/state/ds4-document-profile-002/api-token")
EXP_STATE = Path("/home/funboy/.local/state/ds4-long-context-baseline-001")
EXP_TOKEN = EXP_STATE / "api-token"
LOCK_FILE = Path("/home/funboy/.local/state/strix-cluster/compute.lock")
OWNER_FILE = Path("/home/funboy/.local/state/strix-cluster/ds4-long-context-baseline-001-owner.json")
REGISTRY = RAW / "p3-p4-registry.json"
TERMINAL = RAW / "p3-p4-terminal.json"
RESTORE = RAW / "restore-after-p3-p4.json"

PROD_GATEWAY = ("127.0.0.1", 18224)
EXP_GATEWAY = ("127.0.0.1", 19224)
TOKENIZER = ("127.0.0.1", 19223)
BACKEND = ("127.0.0.1", 18080)
CTX_ALLOC = 69632
GATEWAY_CTX = 65536
OUTPUT_CAP = 2048
COORD = "10.55.0.1"
SSH = ["ssh", "-o", "IdentityAgent=none", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "02-evo-x3-tb"]

SITE = subprocess.run(
    [str(VENV / "bin/python"), "-c", "import site; print(site.getsitepackages()[0])"],
    text=True, capture_output=True, check=True
).stdout.strip()
LD = ":".join([
    "/usr/lib/x86_64-linux-gnu",
    f"{SITE}/_rocm_sdk_core/lib",
    f"{SITE}/_rocm_sdk_devel/lib",
    f"{SITE}/_rocm_sdk_libraries/lib",
    f"{SITE}/torch/lib",
])
PYTHONPATH = ":".join([
    str(ROOT),
    str(ROOT / ".vendor/vllm-dsv41"),
    f"{SITE}/_rocm_sdk_core/share/amd_smi",
])

stop_requested = False
active_local_units: set[str] = set()
active_remote_units: set[str] = set()
current_pair: dict[str, Any] | None = None
tokenizer_unit = "d4lc-p3p4-tokenizer"
gateway_unit = "d4lc-p3p4-gateway"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def cmd(args: list[str], timeout: float | None = None, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, timeout=timeout, check=check)


def ssh(args: list[str], timeout: float | None = None, check: bool = False) -> subprocess.CompletedProcess[str]:
    return cmd(SSH + args, timeout=timeout, check=check)


def sha_ids(ids: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def status_unit(unit: str, remote: bool = False) -> dict[str, Any]:
    args = [
        "systemctl", "--user", "show", unit,
        "-p", "ActiveState", "-p", "SubState", "-p", "Result",
        "-p", "ExecMainStatus", "-p", "MainPID", "-p", "InvocationID",
        "--no-pager",
    ]
    p = ssh(args, timeout=15) if remote else cmd(args, timeout=15)
    d: dict[str, Any] = {"returncode": p.returncode, "stderr": p.stderr.strip()}
    for line in p.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            d[k] = v
    return d


def snapshot_text(remote: bool = False) -> str:
    shell = r"""date -Is
echo '--- free -b ---'
free -b
echo '--- swap ---'
swapon --show --bytes || true
echo '--- vmstat ---'
grep -E '^(pgfault|pgmajfault|pswpin|pswpout) ' /proc/vmstat || true
echo '--- gtt/gpu/temp ---'
for f in /sys/class/drm/card*/device/mem_info_gtt_total /sys/class/drm/card*/device/mem_info_gtt_used /sys/class/drm/card*/device/gpu_busy_percent /sys/class/drm/card*/device/hwmon/hwmon*/temp*_input; do
  if test -r "$f"; then printf '%s=' "$f"; cat "$f"; fi
done
echo '--- thunderbolt counters ---'
for f in /sys/class/net/thunderbolt0/statistics/rx_bytes /sys/class/net/thunderbolt0/statistics/tx_bytes /sys/class/net/thunderbolt0/statistics/rx_errors /sys/class/net/thunderbolt0/statistics/tx_errors; do
  if test -r "$f"; then printf '%s=' "$f"; cat "$f"; fi
done
echo '--- power ---'
powerprofilesctl get 2>/dev/null || true
"""
    p = ssh(["bash", "-lc", shell], timeout=30) if remote else cmd(["bash", "-lc", shell], timeout=30)
    return p.stdout + (("\nSTDERR:\n" + p.stderr) if p.stderr else "")


def prod_lifecycle(method: str = "GET", action: str | None = None, body: dict | None = None, timeout: int = 30) -> tuple[int, dict]:
    token = PROD_TOKEN.read_text().strip()
    path = "/v1/lifecycle" + (f"/{action}" if action else "")
    headers = {"Authorization": "Bearer " + token}
    payload = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        payload = json.dumps(body, separators=(",", ":")).encode()
    conn = http.client.HTTPConnection(*PROD_GATEWAY, timeout=timeout)
    conn.request(method, path, body=payload, headers=headers)
    res = conn.getresponse()
    raw = res.read()
    status = res.status
    conn.close()
    try:
        data = json.loads(raw)
    except Exception:
        data = {"_raw": raw.decode(errors="replace")}
    return status, data


def wait_prod_state(wanted: str, timeout_s: int) -> dict:
    deadline = time.time() + timeout_s
    last: tuple[int, dict] | None = None
    while time.time() < deadline:
        last = prod_lifecycle()
        if last[0] == 200 and last[1].get("state") == wanted:
            return last[1]
        time.sleep(3)
    raise RuntimeError(f"production lifecycle timeout waiting {wanted}: {last}")


def prod_smoke(expected: str) -> dict:
    token = PROD_TOKEN.read_text().strip()
    body = {
        "model": "deepseek-v4.1-flash",
        "profile": "document-low",
        "messages": [{"role": "user", "content": f"Return exactly {expected}."}],
        "temperature": 0,
        "seed": 1,
        "max_tokens": 128,
        "stream": False,
    }
    started = time.monotonic()
    conn = http.client.HTTPConnection(*PROD_GATEWAY, timeout=1800)
    conn.request(
        "POST", "/v1/chat/completions",
        body=json.dumps(body, separators=(",", ":")).encode(),
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
    )
    res = conn.getresponse()
    raw = res.read()
    status = res.status
    conn.close()
    try:
        value = json.loads(raw)
    except Exception:
        value = {"_raw": raw.decode(errors="replace")}
    choice = (value.get("choices") or [{}])[0] if isinstance(value, dict) else {}
    content = ((choice.get("message") or {}).get("content") or "")
    return {
        "http": status,
        "content": content,
        "finish_reason": choice.get("finish_reason"),
        "usage": value.get("usage") if isinstance(value, dict) else None,
        "wall_s": time.monotonic() - started,
        "pass": status == 200 and content.strip() == expected and choice.get("finish_reason") == "stop",
    }


def stop_unit(unit: str, remote: bool = False) -> None:
    args = ["systemctl", "--user", "stop", unit]
    try:
        if remote:
            ssh(args, timeout=60)
            active_remote_units.discard(unit)
        else:
            cmd(args, timeout=60)
            active_local_units.discard(unit)
    except Exception:
        pass


def start_tokenizer() -> None:
    stop_unit(tokenizer_unit)
    cmd(["systemctl", "--user", "reset-failed", tokenizer_unit + ".service"], timeout=15)
    argv = [
        "systemd-run", "--user", f"--unit={tokenizer_unit}",
        "--property=KillMode=control-group", "--property=Restart=no", "--property=TimeoutStopSec=30",
        f"--property=StandardOutput=append:{RAW}/p3p4-tokenizer.log",
        f"--property=StandardError=append:{RAW}/p3p4-tokenizer.log",
        f"--setenv=LD_LIBRARY_PATH={LD}", f"--setenv=PYTHONPATH={PYTHONPATH}",
        str(VENV / "bin/python"), str(TOKENIZER_RUNNER),
    ]
    p = cmd(argv, timeout=30)
    if p.returncode != 0:
        raise RuntimeError(f"tokenizer start failed: {p.stderr or p.stdout}")
    active_local_units.add(tokenizer_unit)
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            conn = http.client.HTTPConnection(*TOKENIZER, timeout=3)
            conn.request("GET", "/health")
            res = conn.getresponse()
            data = res.read()
            conn.close()
            if res.status == 200:
                atomic(RAW / "p3p4-tokenizer-ready.json", {"at": now(), "http": 200, "body": json.loads(data)})
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("experimental tokenizer did not become ready")


def start_gateway() -> None:
    EXP_STATE.mkdir(parents=True, exist_ok=True)
    stop_unit(gateway_unit)
    cmd(["systemctl", "--user", "reset-failed", gateway_unit + ".service"], timeout=15)
    argv = [
        "systemd-run", "--user", f"--unit={gateway_unit}",
        "--property=KillMode=control-group", "--property=Restart=no", "--property=TimeoutStopSec=30",
        f"--property=StandardOutput=append:{RAW}/p3p4-gateway.log",
        f"--property=StandardError=append:{RAW}/p3p4-gateway.log",
        str(STRIXGLM), "serve", "--config", str(CONFIG),
    ]
    p = cmd(argv, timeout=30)
    if p.returncode != 0:
        raise RuntimeError(f"experimental gateway start failed: {p.stderr or p.stdout}")
    active_local_units.add(gateway_unit)
    deadline = time.time() + 120
    last = ""
    while time.time() < deadline:
        if EXP_TOKEN.exists():
            try:
                token = EXP_TOKEN.read_text().strip()
                conn = http.client.HTTPConnection(*EXP_GATEWAY, timeout=3)
                conn.request("GET", "/v1/options", headers={"Authorization": "Bearer " + token})
                res = conn.getresponse()
                raw = res.read()
                conn.close()
                last = raw.decode(errors="replace")
                if res.status == 200:
                    data = json.loads(raw)
                    if data.get("engine_context_tokens") != GATEWAY_CTX:
                        raise RuntimeError(f"experimental gateway context mismatch: {data}")
                    atomic(RAW / "p3p4-gateway-ready.json", {
                        "at": now(), "http": 200,
                        "engine_context_tokens": data.get("engine_context_tokens"),
                        "default_context_tokens": data.get("default_context_tokens"),
                        "default_reasoning": data.get("default_reasoning"),
                        "max_output_tokens": data.get("max_output_tokens"),
                    })
                    return
            except RuntimeError:
                raise
            except Exception as e:
                last = str(e)
        time.sleep(2)
    raise RuntimeError(f"experimental gateway did not become ready: {last}")


def copy_remote_log(remote_path: str, local_path: Path) -> None:
    p = ssh(["cat", remote_path], timeout=30)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(p.stdout + (("\nSTDERR:\n" + p.stderr) if p.stderr else ""))


def pair_faults(pair: dict[str, Any]) -> list[str]:
    text = ""
    local_log = Path(pair["local_log"])
    if local_log.exists():
        try:
            text += "\n" + local_log.read_text(errors="replace")[-200000:]
        except Exception:
            pass
    try:
        p = ssh(["tail", "-n", "500", pair["remote_log"]], timeout=30)
        text += "\n" + p.stdout
    except Exception:
        pass
    patterns = [
        r"out of memory", r"\boom\b", r"gpu fault", r"segmentation fault",
        r"\bnan\b", r"desynchron", r"collective.*timeout", r"hip error",
    ]
    return [pat for pat in patterns if re.search(pat, text, re.I)]


def start_pair(tag: str, port: int, kv_dir: Path | None = None) -> dict[str, Any]:
    global current_pair
    if current_pair is not None:
        raise RuntimeError("attempted to start a second experimental pair")
    safe = re.sub(r"[^a-zA-Z0-9_-]", "-", tag)
    local_unit = f"d4lc-{safe}-c"
    remote_unit = f"d4lc-{safe}-w"
    run_dir = RAW / "runtime" / safe
    run_dir.mkdir(parents=True, exist_ok=True)
    remote_dir = str(run_dir)
    ssh(["mkdir", "-p", remote_dir], timeout=20, check=True)
    for remote, unit in [(False, local_unit), (True, remote_unit)]:
        try:
            stop_unit(unit, remote)
            if remote:
                ssh(["systemctl", "--user", "reset-failed", unit + ".service"], timeout=15)
            else:
                cmd(["systemctl", "--user", "reset-failed", unit + ".service"], timeout=15)
        except Exception:
            pass

    worker_argv = [
        "systemd-run", "--user", f"--unit={remote_unit}",
        "--property=KillMode=control-group", "--property=Restart=no", "--property=TimeoutStopSec=45",
        f"--property=StandardOutput=append:{remote_dir}/worker.log",
        f"--property=StandardError=append:{remote_dir}/worker.log",
        str(NODE_RUNNER), "worker", str(port), "18080", str(CTX_ALLOC),
    ]
    wp = ssh(worker_argv, timeout=30)
    if wp.returncode != 0:
        raise RuntimeError(f"worker start failed {tag}: {wp.stderr or wp.stdout}")
    active_remote_units.add(remote_unit)

    server_argv = [
        "systemd-run", "--user", f"--unit={local_unit}",
        "--property=KillMode=control-group", "--property=Restart=no", "--property=TimeoutStopSec=45",
        f"--property=StandardOutput=append:{run_dir}/server.log",
        f"--property=StandardError=append:{run_dir}/server.log",
        str(NODE_RUNNER), "server", str(port), "18080", str(CTX_ALLOC),
    ]
    if kv_dir is not None:
        kv_dir.mkdir(parents=True, exist_ok=True)
        server_argv.append(str(kv_dir))
    sp = cmd(server_argv, timeout=30)
    if sp.returncode != 0:
        stop_unit(remote_unit, True)
        raise RuntimeError(f"server start failed {tag}: {sp.stderr or sp.stdout}")
    active_local_units.add(local_unit)

    pair = {
        "tag": tag,
        "port": port,
        "local_unit": local_unit,
        "remote_unit": remote_unit,
        "local_log": str(run_dir / "server.log"),
        "remote_log": remote_dir + "/worker.log",
        "kv_dir": str(kv_dir) if kv_dir else None,
        "started_at": now(),
        "worker_systemd_argv": worker_argv,
        "server_systemd_argv": server_argv,
    }
    current_pair = pair
    deadline = time.time() + 1500
    last = ""
    while time.time() < deadline:
        if stop_requested:
            raise RuntimeError("stop requested during pair startup")
        ls = status_unit(local_unit)
        rs = status_unit(remote_unit, remote=True)
        if ls.get("ActiveState") == "failed" or rs.get("ActiveState") == "failed":
            raise RuntimeError(f"pair start failure {tag}: local={ls} remote={rs}")
        try:
            conn = http.client.HTTPConnection(*BACKEND, timeout=3)
            conn.request("GET", "/v1/models")
            res = conn.getresponse()
            raw = res.read()
            conn.close()
            last = f"HTTP{res.status} {raw[:200]!r}"
            if res.status == 200 and ls.get("ActiveState") == "active" and rs.get("ActiveState") == "active":
                pair["ready_at"] = now()
                pair["local_ready"] = ls
                pair["remote_ready"] = rs
                atomic(run_dir / "ready.json", pair)
                return pair
        except Exception as e:
            last = str(e)
        time.sleep(5)
    raise RuntimeError(f"pair {tag} did not become ready: {last}")


def stop_pair(pair: dict[str, Any] | None) -> dict[str, Any]:
    global current_pair
    if pair is None:
        return {}
    local_unit = pair["local_unit"]
    remote_unit = pair["remote_unit"]
    # Initiate both whole-pair stops without waiting for one rank to complete first.
    lp = subprocess.Popen(["systemctl", "--user", "stop", local_unit], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    rp = subprocess.Popen(SSH + ["systemctl", "--user", "stop", remote_unit], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        lo, le = lp.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        lp.kill(); lo, le = lp.communicate()
    try:
        ro, re_ = rp.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        rp.kill(); ro, re_ = rp.communicate()
    active_local_units.discard(local_unit)
    active_remote_units.discard(remote_unit)
    local = status_unit(local_unit)
    remote = status_unit(remote_unit, remote=True)
    run_dir = RAW / "runtime" / re.sub(r"[^a-zA-Z0-9_-]", "-", pair["tag"])
    try:
        copy_remote_log(pair["remote_log"], run_dir / "worker.remote.log")
    except Exception as e:
        (run_dir / "worker-copy-error.txt").write_text(str(e))
    out = {
        "tag": pair["tag"], "stopped_at": now(),
        "local": local, "remote": remote,
        "local_stop": {"stdout": lo, "stderr": le},
        "remote_stop": {"stdout": ro, "stderr": re_},
        "faults": pair_faults(pair),
    }
    atomic(run_dir / "stopped.json", out)
    current_pair = None
    return out


def tokenize_messages(messages: list[dict[str, str]]) -> dict[str, Any]:
    body = {
        "messages": messages,
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"},
        "add_generation_prompt": True,
    }
    conn = http.client.HTTPConnection(*TOKENIZER, timeout=60)
    conn.request("POST", "/tokenize", body=json.dumps(body, separators=(",", ":")).encode(),
                 headers={"Content-Type": "application/json"})
    res = conn.getresponse()
    raw = res.read()
    conn.close()
    if res.status != 200:
        raise RuntimeError(f"tokenizer HTTP{res.status}: {raw[:1000]!r}")
    data = json.loads(raw)
    ids = data.get("tokens")
    if not isinstance(ids, list) or data.get("count") != len(ids):
        raise RuntimeError("invalid tokenizer response")
    return {
        "count": len(ids),
        "max_model_len": data.get("max_model_len"),
        "ids_sha256": sha_ids([int(x) for x in ids]),
        "resolved_thinking": data.get("resolved_thinking"),
    }


def extract_cache(usage: dict | None) -> dict[str, Any]:
    usage = usage or {}
    ptd = usage.get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cached_tokens": ptd.get("cached_tokens"),
        "cache_write_tokens": ptd.get("cache_write_tokens"),
    }


def validate_json(content: str, expected: dict[str, Any]) -> tuple[bool, Any, str | None]:
    try:
        parsed = json.loads(content.strip())
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"
    return parsed == expected, parsed, None


def stream_request(
    request_id: str,
    messages: list[dict[str, str]],
    expected: dict[str, Any] | str,
    pair: dict[str, Any],
    *,
    expected_ids_sha256: str | None = None,
    require_cache_zero: bool = False,
    max_tokens: int = OUTPUT_CAP,
) -> dict[str, Any]:
    req_dir = RAW / "api" / request_id
    req_dir.mkdir(parents=True, exist_ok=True)
    terminal_path = req_dir / "terminal.json"
    if terminal_path.exists():
        prior = json.loads(terminal_path.read_text())
        if prior.get("state") == "COMPLETE":
            return prior
    req_registry = req_dir / "registry.json"
    if req_registry.exists():
        prior = json.loads(req_registry.read_text())
        if prior.get("state") == "IN_FLIGHT":
            raise RuntimeError(f"unreconciled IN_FLIGHT request {request_id}; refusing replay")

    toks = tokenize_messages(messages)
    if toks["count"] + max_tokens > GATEWAY_CTX:
        result = {
            "request_id": request_id, "state": "COMPLETE", "status": "CONTEXT_ADMISSION_FAIL",
            "tokenizer": toks, "max_tokens": max_tokens,
            "reason": f"prompt {toks['count']} + output {max_tokens} exceeds {GATEWAY_CTX}",
        }
        atomic(terminal_path, result)
        return result
    if expected_ids_sha256 and toks["ids_sha256"] != expected_ids_sha256:
        result = {
            "request_id": request_id, "state": "COMPLETE", "status": "TOKEN_FINGERPRINT_MISMATCH",
            "tokenizer": toks, "expected_ids_sha256": expected_ids_sha256,
        }
        atomic(terminal_path, result)
        return result

    payload = {
        "model": "deepseek-v4.1-flash",
        "profile": "document-long-low",
        "messages": messages,
        "context_tokens": GATEWAY_CTX,
        "reasoning_effort": "low",
        "temperature": 0,
        "seed": 1,
        "max_tokens": max_tokens,
        "stream": True,
    }
    public_request = {
        "request_id": request_id, "profile": "document-long-low",
        "messages": messages, "context_tokens": GATEWAY_CTX,
        "reasoning_effort": "low", "temperature": 0, "seed": 1,
        "max_tokens": max_tokens, "stream": True,
        "tokenizer": toks,
        "require_cache_zero": require_cache_zero,
        "expected_ids_sha256": expected_ids_sha256,
    }
    atomic(req_dir / "request.json", public_request)
    atomic(req_registry, {
        "request_id": request_id, "state": "IN_FLIGHT", "started_at": now(),
        "pair": pair["tag"], "tokenizer": toks,
    })
    (req_dir / "metrics.before.node01.txt").write_text(snapshot_text(False))
    (req_dir / "metrics.before.node02.txt").write_text(snapshot_text(True))

    token = EXP_TOKEN.read_text().strip()
    started_wall = time.time()
    started = time.monotonic()
    first_reasoning_s: float | None = None
    first_final_s: float | None = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason: str | None = None
    usage: dict | None = None
    backend_metrics: dict | None = None
    telemetry: list[dict[str, Any]] = []
    error_events: list[Any] = []
    http_status = 0
    content_type = ""
    done_seen = False
    read_error: str | None = None

    raw_path = req_dir / "response.sse"
    try:
        conn = http.client.HTTPConnection(*EXP_GATEWAY, timeout=7200)
        conn.request(
            "POST", "/v1/chat/completions",
            body=json.dumps(payload, separators=(",", ":")).encode(),
            headers={
                "Authorization": "Bearer " + token,
                "Content-Type": "application/json",
                "X-HaloClu-Timings": "1",
            },
        )
        res = conn.getresponse()
        http_status = res.status
        content_type = res.getheader("Content-Type") or ""
        current_event: str | None = None
        data_lines: list[str] = []

        def process_event() -> None:
            nonlocal first_reasoning_s, first_final_s, finish_reason, usage, backend_metrics, done_seen
            if not data_lines:
                return
            data = "\n".join(data_lines)
            if data == "[DONE]":
                done_seen = True
                return
            try:
                obj = json.loads(data)
            except Exception:
                error_events.append({"event": current_event, "unparsed": data[:2000]})
                return
            if current_event == "haloclu.timing":
                if isinstance(obj, dict):
                    telemetry.append(obj)
                return
            if current_event == "error" or (isinstance(obj, dict) and obj.get("error") is not None):
                error_events.append(obj)
            if not isinstance(obj, dict):
                return
            if isinstance(obj.get("usage"), dict):
                usage = obj["usage"]
            if isinstance(obj.get("metrics"), dict):
                backend_metrics = obj["metrics"]
            choices = obj.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    delta = choice.get("delta")
                    if not isinstance(delta, dict):
                        delta = {}
                    rc = delta.get("reasoning_content")
                    if not isinstance(rc, str):
                        rc = delta.get("reasoning")
                    if isinstance(rc, str) and rc:
                        if first_reasoning_s is None:
                            first_reasoning_s = time.monotonic() - started
                        reasoning_parts.append(rc)
                    cc = delta.get("content")
                    if isinstance(cc, str) and cc:
                        if first_final_s is None:
                            first_final_s = time.monotonic() - started
                        content_parts.append(cc)
                    fr = choice.get("finish_reason")
                    if isinstance(fr, str) and fr:
                        finish_reason = fr

        with raw_path.open("wb") as rawf:
            while True:
                line = res.readline()
                if not line:
                    if data_lines:
                        process_event()
                    break
                rawf.write(line)
                text = line.decode(errors="replace").rstrip("\r\n")
                if text == "":
                    process_event()
                    current_event = None
                    data_lines = []
                elif text.startswith("event:"):
                    current_event = text[6:].strip()
                elif text.startswith("data:"):
                    data_lines.append(text[5:].lstrip())
                elif text.startswith(":"):
                    pass
        conn.close()
    except Exception as e:
        read_error = f"{type(e).__name__}: {e}"

    wall_s = time.monotonic() - started
    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    cache = extract_cache(usage)
    pair_local = status_unit(pair["local_unit"])
    pair_remote = status_unit(pair["remote_unit"], remote=True)
    faults = pair_faults(pair)

    if isinstance(expected, dict):
        semantic_pass, parsed_final, parse_error = validate_json(content, expected)
    else:
        parsed_final = content.strip()
        parse_error = None
        semantic_pass = parsed_final == expected

    technical_ok = (
        read_error is None and http_status == 200 and not error_events and
        pair_local.get("ActiveState") == "active" and pair_remote.get("ActiveState") == "active" and
        not faults
    )
    cache_ok = not require_cache_zero or cache.get("cached_tokens") == 0
    if not technical_ok:
        status = "TECHNICAL_STOP"
    elif not cache_ok:
        status = "INVALID_CACHE_MODE"
    elif finish_reason == "length" and not semantic_pass:
        status = "INCOMPLETE_NO_FINAL"
    elif semantic_pass and finish_reason == "stop":
        status = "PASS"
    else:
        status = "SEMANTIC_FAIL"

    result = {
        "request_id": request_id,
        "state": "COMPLETE",
        "status": status,
        "started_wall_epoch": started_wall,
        "finished_at": now(),
        "pair": pair["tag"],
        "http": http_status,
        "content_type": content_type,
        "tokenizer": toks,
        "max_tokens": max_tokens,
        "finish_reason": finish_reason,
        "content": content,
        "reasoning": reasoning,
        "usage": usage,
        "cache": cache,
        "backend_metrics": backend_metrics,
        "telemetry": telemetry,
        "client_first_reasoning_s": first_reasoning_s,
        "client_first_final_s": first_final_s,
        "client_wall_s": wall_s,
        "done_seen": done_seen,
        "read_error": read_error,
        "error_events": error_events,
        "expected": expected,
        "parsed_final": parsed_final,
        "parse_error": parse_error,
        "semantic_pass": semantic_pass,
        "require_cache_zero": require_cache_zero,
        "cache_requirement_pass": cache_ok,
        "pair_local_after": pair_local,
        "pair_remote_after": pair_remote,
        "faults": faults,
    }
    (req_dir / "metrics.after.node01.txt").write_text(snapshot_text(False))
    (req_dir / "metrics.after.node02.txt").write_text(snapshot_text(True))
    atomic(terminal_path, result)
    atomic(req_registry, result)
    return result


def update_registry(phase: str, p3: list[dict], holdouts: list[dict], p4: list[dict], extra: dict | None = None) -> None:
    obj: dict[str, Any] = {
        "schema": "ds4-long-context-baseline-001-p3p4-v1",
        "campaign": CAMPAIGN,
        "state": "IN_FLIGHT",
        "phase": phase,
        "updated_at": now(),
        "p3": p3,
        "holdouts": holdouts,
        "p4": p4,
    }
    if extra:
        obj.update(extra)
    atomic(REGISTRY, obj)
    atomic(OWNER_FILE, {
        "campaign": CAMPAIGN, "state": "IN_FLIGHT", "phase": phase,
        "pid": os.getpid(), "updated_at": obj["updated_at"], "raw": str(RAW),
    })


def stop_experimental_infra() -> None:
    global current_pair
    if current_pair is not None:
        try:
            stop_pair(current_pair)
        except Exception:
            pass
    stop_unit(gateway_unit)
    stop_unit(tokenizer_unit)
    for u in list(active_local_units):
        stop_unit(u)
    for u in list(active_remote_units):
        stop_unit(u, True)


def restore_production(lock_fp, reason: str) -> dict[str, Any]:
    stop_experimental_infra()
    try:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass
    out: dict[str, Any] = {"reason": reason, "started_at": now()}
    try:
        before = prod_lifecycle()
        out["lifecycle_before"] = {"http": before[0], "body": before[1]}
        if before[0] == 200 and before[1].get("state") != "READY":
            st, body = prod_lifecycle("POST", "on", {"confirm": True})
            out["on_reply"] = {"http": st, "body": body}
        out["ready"] = wait_prod_state("READY", 1500)
        out["smoke"] = prod_smoke("LONGCTX-P3P4-RESTORE-OK")
        out["status"] = "PASS" if out["smoke"]["pass"] else "FAIL"
    except Exception as e:
        out["status"] = "FAIL"
        out["error"] = f"{type(e).__name__}: {e}"
    out["finished_at"] = now()
    atomic(RESTORE, out)
    return out


def handle_signal(signum, frame):
    global stop_requested
    stop_requested = True


def admitted_levels() -> set[str]:
    data = json.loads(P1P2.read_text())
    if data.get("state") != "COMPLETE":
        raise RuntimeError("P1/P2 is not terminal")
    depths = {}
    for c in data.get("cases", []):
        m = re.fullmatch(r"p1-(1024|4096|8192|16384|32768|65536)-r1", c.get("case_id", ""))
        if m:
            depths[int(m.group(1))] = c.get("status") == "PASS_TECHNICAL"
    mapping = {"4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768, "64k": 65536}
    return {label for label, depth in mapping.items() if depths.get(depth)}


def main() -> None:
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    RAW.mkdir(parents=True, exist_ok=True)
    if TERMINAL.exists():
        print(TERMINAL.read_text(), flush=True)
        return
    if REGISTRY.exists():
        old = json.loads(REGISTRY.read_text())
        if old.get("state") == "IN_FLIGHT":
            # Do not replay a request whose dispatch state is ambiguous.
            for section in ("p3", "holdouts", "p4"):
                for r in old.get(section, []):
                    if r.get("state") == "IN_FLIGHT":
                        raise RuntimeError(f"existing unreconciled IN_FLIGHT {section} request; refusing replay")

    for p in [P1P2, P3_MANIFEST, P4_MANIFEST, CONFIG, NODE_RUNNER, TOKENIZER_RUNNER, STRIXGLM, MODEL, PROD_TOKEN]:
        if not p.exists():
            raise RuntimeError(f"missing preflight path: {p}")
    if MODEL.stat().st_size != 365713686528:
        raise RuntimeError("model size gate failed")
    if cmd(["bash", "-n", str(NODE_RUNNER)], timeout=15).returncode != 0:
        raise RuntimeError("node runner bash syntax gate failed")
    remote_hash = ssh(["sha256sum", str(NODE_RUNNER)], timeout=20)
    local_hash = cmd(["sha256sum", str(NODE_RUNNER)], timeout=20)
    if remote_hash.returncode != 0 or remote_hash.stdout.split()[0] != local_hash.stdout.split()[0]:
        raise RuntimeError("NODE02 long-context runner is missing or differs from NODE01")

    admitted = admitted_levels()
    p3m = json.loads(P3_MANIFEST.read_text())
    p4m = json.loads(P4_MANIFEST.read_text())
    initial = prod_lifecycle()
    if initial[0] != 200 or initial[1].get("state") != "READY":
        raise RuntimeError(f"production E1 not READY at P3/P4 acquisition: {initial}")

    atomic(REGISTRY, {
        "schema": "ds4-long-context-baseline-001-p3p4-v1",
        "campaign": CAMPAIGN,
        "state": "IN_FLIGHT",
        "phase": "ACQUIRE",
        "started_at": now(),
        "admitted_levels": sorted(admitted),
        "production_before": initial[1],
        "p3": [], "holdouts": [], "p4": [],
    })

    off_http, off_body = prod_lifecycle("POST", "off", {"confirm": True})
    off_state = wait_prod_state("OFF", 420)
    atomic(RAW / "gateway-off-before-p3-p4.json", {"http": off_http, "reply": off_body, "state": off_state})

    lock_fp = LOCK_FILE.open("a+")
    fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    p3_results: list[dict] = []
    holdout_results: list[dict] = []
    p4_results: list[dict] = []
    restore: dict[str, Any] = {}
    campaign_error: str | None = None
    qualification: dict[str, Any] = {}

    try:
        start_tokenizer()
        start_gateway()
        update_registry("P3_ORIGINALS", p3_results, holdout_results, p4_results, {"admitted_levels": sorted(admitted)})

        port = 19500
        originals = [x for x in p3m["originals"] if x["label"] in admitted]
        if len(originals) > 10:
            raise RuntimeError("P3 original cap exceeded")
        for item in originals:
            if stop_requested:
                raise RuntimeError("stop requested")
            pair = start_pair("p3-" + item["id"], port, None)
            try:
                prompt = (ROOT / item["prompt_file"]).read_text()
                res = stream_request(
                    "p3-" + item["id"],
                    [{"role": "user", "content": prompt}],
                    item["expected"],
                    pair,
                    expected_ids_sha256=item["rendered_low_ids_sha256"],
                    require_cache_zero=True,
                    max_tokens=item["output_cap"],
                )
            finally:
                pair_stop = stop_pair(pair)
            res["pair_stop"] = pair_stop
            p3_results.append(res)
            update_registry("P3_ORIGINALS", p3_results, holdout_results, p4_results)
            port += 1
            if res["status"] in {"TECHNICAL_STOP", "TOKEN_FINGERPRINT_MISMATCH", "INVALID_CACHE_MODE"}:
                raise RuntimeError(f"technical P3 stop at {item['id']}: {res['status']}")

        levels = ["4k", "8k", "16k", "32k", "64k"]
        both_pass = []
        for label in levels:
            rows = {r["request_id"].replace("p3-", ""): r for r in p3_results}
            if rows.get(f"code-{label}", {}).get("status") == "PASS" and rows.get(f"docs-{label}", {}).get("status") == "PASS":
                both_pass.append(label)
        max_original_pass = both_pass[-1] if both_pass else None
        qualification["p3_both_pass_levels"] = both_pass
        qualification["max_original_both_pass"] = max_original_pass

        if max_original_pass:
            holds = [x for x in p3m["holdouts"] if x["label"] == max_original_pass]
            if len(holds) != 2:
                raise RuntimeError("expected exactly two frozen holdouts at selected P3 level")
            for item in holds:
                pair = start_pair("p3-" + item["id"], port, None)
                try:
                    prompt = (ROOT / item["prompt_file"]).read_text()
                    res = stream_request(
                        "p3-" + item["id"],
                        [{"role": "user", "content": prompt}],
                        item["expected"],
                        pair,
                        expected_ids_sha256=item["rendered_low_ids_sha256"],
                        require_cache_zero=True,
                        max_tokens=item["output_cap"],
                    )
                finally:
                    pair_stop = stop_pair(pair)
                res["pair_stop"] = pair_stop
                holdout_results.append(res)
                update_registry("P3_HOLDOUTS", p3_results, holdout_results, p4_results, qualification)
                port += 1
                if res["status"] in {"TECHNICAL_STOP", "TOKEN_FINGERPRINT_MISMATCH", "INVALID_CACHE_MODE"}:
                    raise RuntimeError(f"technical P3 holdout stop at {item['id']}: {res['status']}")
            if all(r.get("status") == "PASS" for r in holdout_results):
                qualification["holdout_confirmed_level"] = max_original_pass
            else:
                qualification["holdout_confirmed_level"] = None
        else:
            qualification["holdout_confirmed_level"] = None

        update_registry("P3_COMPLETE", p3_results, holdout_results, p4_results, qualification)

        # P4 uses the largest original-passing document level <=32K, leaving room
        # for visible history, two more user turns and the added real document.
        candidate_labels = [x for x in ["4k", "8k", "16k", "32k"] if x in both_pass]
        if not candidate_labels:
            qualification["p4_status"] = "SKIPPED_NO_QUALITATIVE_LEVEL_WITH_MARGIN"
        else:
            chosen = candidate_labels[-1]
            qualification["p4_level"] = chosen
            cand = next(x for x in p4m["candidates"] if x["label"] == chosen)
            base = (ROOT / cand["base_file"]).read_text()
            q1 = cand["q1"]
            q2 = cand["q2"]
            q3 = cand["q3"]
            q4 = cand["q4"]
            original_kv = RAW / "p4" / "kv-original"
            if original_kv.exists() and any(original_kv.iterdir()):
                raise RuntimeError("P4 original KV directory is non-empty before first P4 run; refusing ambiguous reuse")

            pair = start_pair("p4-original", 19600, original_kv)
            try:
                m1 = [{"role": "user", "content": base + q1}]
                r1 = stream_request("p4-step1", m1, cand["expected_q1"], pair, require_cache_zero=True)
                p4_results.append(r1); update_registry("P4_STEP1", p3_results, holdout_results, p4_results, qualification)
                if r1["status"] != "PASS":
                    raise RuntimeError(f"P4 step1 did not pass: {r1['status']}")

                m2 = m1 + [
                    {"role": "assistant", "content": r1["content"]},
                    {"role": "user", "content": q2},
                ]
                r2 = stream_request("p4-step2", m2, cand["expected_q2"], pair)
                p4_results.append(r2); update_registry("P4_STEP2", p3_results, holdout_results, p4_results, qualification)
                if r2["status"] != "PASS":
                    raise RuntimeError(f"P4 step2 did not pass: {r2['status']}")

                m3 = m2 + [
                    {"role": "assistant", "content": r2["content"]},
                    {"role": "user", "content": q3},
                ]
                r3 = stream_request("p4-step3", m3, cand["expected_q3"], pair)
                p4_results.append(r3); update_registry("P4_STEP3", p3_results, holdout_results, p4_results, qualification)
                if r3["status"] != "PASS":
                    raise RuntimeError(f"P4 step3 did not pass: {r3['status']}")
            finally:
                stop_info = stop_pair(pair)
                if p4_results:
                    p4_results[-1]["pair_stop_after_step3"] = stop_info

            kv_files = []
            if original_kv.exists():
                for p in sorted(original_kv.rglob("*")):
                    if p.is_file():
                        kv_files.append({"path": str(p.relative_to(original_kv)), "bytes": p.stat().st_size})
            qualification["p4_original_kv_files_after_step3"] = kv_files

            fresh_pair = start_pair("p4-fresh-control", 19601, None)
            try:
                r4 = stream_request(
                    "p4-step4-fresh",
                    m3,
                    cand["expected_q3"],
                    fresh_pair,
                    expected_ids_sha256=r3["tokenizer"]["ids_sha256"],
                    require_cache_zero=True,
                )
                r4["same_request_ids_as_step3"] = r4["tokenizer"]["ids_sha256"] == r3["tokenizer"]["ids_sha256"]
                r4["same_final_as_step3"] = r4["content"] == r3["content"]
                p4_results.append(r4); update_registry("P4_STEP4_FRESH", p3_results, holdout_results, p4_results, qualification)
                if r4["status"] != "PASS":
                    raise RuntimeError(f"P4 fresh control did not pass: {r4['status']}")
            finally:
                stop_info = stop_pair(fresh_pair)
                if p4_results:
                    p4_results[-1]["pair_stop_after_step4"] = stop_info

            resume_pair = start_pair("p4-resume", 19602, original_kv)
            try:
                m5 = [{"role": "user", "content": cand["isolation_prompt"]}]
                r5 = stream_request("p4-step5-isolation", m5, cand["isolation_expected"], resume_pair, require_cache_zero=True)
                p4_results.append(r5); update_registry("P4_STEP5_ISOLATION", p3_results, holdout_results, p4_results, qualification)
                if r5["status"] != "PASS":
                    raise RuntimeError(f"P4 isolation control did not pass: {r5['status']}")

                m6 = m3 + [
                    {"role": "assistant", "content": r3["content"]},
                    {"role": "user", "content": q4},
                ]
                r6 = stream_request("p4-step6-resume", m6, cand["expected_q4"], resume_pair)
                p4_results.append(r6); update_registry("P4_STEP6_RESUME", p3_results, holdout_results, p4_results, qualification)
                if r6["status"] != "PASS":
                    raise RuntimeError(f"P4 resume did not pass: {r6['status']}")
                qualification["p4_status"] = "PASS"
            finally:
                stop_info = stop_pair(resume_pair)
                if p4_results:
                    p4_results[-1]["pair_stop_after_step6"] = stop_info

        update_registry("P4_COMPLETE", p3_results, holdout_results, p4_results, qualification)

    except Exception as e:
        campaign_error = f"{type(e).__name__}: {e}"
    finally:
        restore = restore_production(lock_fp, campaign_error or "P3_P4_COMPLETE")
        lock_fp.close()

    technical_bad = any(r.get("status") in {"TECHNICAL_STOP", "TOKEN_FINGERPRINT_MISMATCH", "INVALID_CACHE_MODE", "CONTEXT_ADMISSION_FAIL"} for r in p3_results + holdout_results + p4_results)
    final_status = "PASS" if campaign_error is None and not technical_bad and restore.get("status") == "PASS" else "PARTIAL_OR_STOP"
    terminal = {
        "schema": "ds4-long-context-baseline-001-p3p4-v1",
        "campaign": CAMPAIGN,
        "state": "COMPLETE",
        "status": final_status,
        "finished_at": now(),
        "campaign_error": campaign_error,
        "qualification": qualification,
        "p3": p3_results,
        "holdouts": holdout_results,
        "p4": p4_results,
        "restore": restore,
    }
    atomic(TERMINAL, terminal)
    atomic(REGISTRY, terminal)
    atomic(OWNER_FILE, {
        "campaign": CAMPAIGN,
        "state": "COMPLETE" if restore.get("status") == "PASS" else "RESTORE_FAILED",
        "updated_at": terminal["finished_at"],
        "terminal": str(TERMINAL),
        "restore": str(RESTORE),
    })
    print(json.dumps({
        "status": final_status,
        "campaign_error": campaign_error,
        "qualification": qualification,
        "p3": [(r["request_id"], r["status"]) for r in p3_results],
        "holdouts": [(r["request_id"], r["status"]) for r in holdout_results],
        "p4": [(r["request_id"], r["status"]) for r in p4_results],
        "restore": restore.get("status"),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
