#!/usr/bin/env python3
from __future__ import annotations

import csv
import fcntl
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
SRC = Path("/home/funboy/worktrees/ds4-long-context-baseline-001-e1")
RAW = Path("/home/funboy/reports/DS4-LONG-CONTEXT-BASELINE-001")
RELEASE = Path("/home/funboy/.local/share/haloclu-ds41/releases/ds4-speed-001-engram1")
MODEL = Path("/home/funboy/models/ds41/ds4-v41-q2/DeepSeek-V4.1-Flash-Q2.gguf")
CORPUS = SRC / "speed-bench/promessi_sposi.txt"
BENCH = SRC / "ds4-bench-warm"
TOKEN_FILE = Path("/home/funboy/.local/state/ds4-document-profile-002/api-token")
LOCK_FILE = Path("/home/funboy/.local/state/strix-cluster/compute.lock")
OWNER_FILE = Path("/home/funboy/.local/state/strix-cluster/ds4-long-context-baseline-001-owner.json")
REGISTRY = RAW / "p1-p2-registry.json"
TERMINAL = RAW / "p1-p2-terminal.json"
RESTORE = RAW / "restore-after-p1-p2.json"
GATEWAY = ("127.0.0.1", 18224)
COORD = "10.55.0.1"
CTX = 69632
VENV = Path("/home/funboy/StrixHaloClusterGLM/.engine/venv")
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
SSH = ["ssh", "-o", "IdentityAgent=none", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "02-evo-x3-tb"]
DS4 = str(RELEASE / "ds4")

P1_FIRST = [
    ("p1-1024-r1", 1024, 128),
    ("p1-4096-r1", 4096, 128),
    ("p1-8192-r1", 8192, 128),
    ("p1-16384-r1", 16384, 512),
    ("p1-32768-r1", 32768, 128),
    ("p1-65536-r1", 65536, 128),
]
P1_REPS = [
    ("p1-16384-r2", 16384, 512),
    ("p1-16384-r3", 16384, 512),
    ("p1-65536-r2", 65536, 128),
    ("p1-65536-r3", 65536, 128),
]

stop_requested = False
active_local_units: set[str] = set()
active_remote_units: set[str] = set()


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def cmd(args: list[str], timeout: float | None = None, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, timeout=timeout, check=check)


def ssh(args: list[str], timeout: float | None = None, check: bool = False) -> subprocess.CompletedProcess[str]:
    return cmd(SSH + args, timeout=timeout, check=check)


def lifecycle(method: str = "GET", action: str | None = None, body: dict | None = None, timeout: int = 30) -> tuple[int, dict]:
    token = TOKEN_FILE.read_text().strip()
    path = "/v1/lifecycle" + (f"/{action}" if action else "")
    headers = {"Authorization": "Bearer " + token}
    payload = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        payload = json.dumps(body, separators=(",", ":")).encode()
    conn = http.client.HTTPConnection(*GATEWAY, timeout=timeout)
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


def wait_lifecycle(wanted: str, timeout_s: int) -> dict:
    deadline = time.time() + timeout_s
    last: tuple[int, dict] | None = None
    while time.time() < deadline:
        last = lifecycle()
        if last[0] == 200 and last[1].get("state") == wanted:
            return last[1]
        time.sleep(3)
    raise RuntimeError(f"lifecycle timeout waiting {wanted}: {last}")


def chat_smoke(expected: str) -> dict:
    token = TOKEN_FILE.read_text().strip()
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
    conn = http.client.HTTPConnection("127.0.0.1", 18224, timeout=1800)
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


def status_unit(unit: str, remote: bool = False) -> dict:
    args = ["systemctl", "--user", "show", unit,
            "-p", "ActiveState", "-p", "SubState", "-p", "Result",
            "-p", "ExecMainStatus", "-p", "MainPID", "--no-pager"]
    p = ssh(args, timeout=15) if remote else cmd(args, timeout=15)
    d: dict[str, Any] = {"returncode": p.returncode, "stderr": p.stderr.strip()}
    for line in p.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            d[k] = v
    return d


def stop_pair(local_unit: str | None, remote_unit: str | None) -> None:
    if local_unit:
        cmd(["systemctl", "--user", "stop", local_unit], timeout=45)
        active_local_units.discard(local_unit)
    if remote_unit:
        ssh(["systemctl", "--user", "stop", remote_unit], timeout=45)
        active_remote_units.discard(remote_unit)


def snapshot_text(remote: bool = False) -> str:
    shell = r'''date -Is
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
'''
    p = ssh(["bash", "-lc", shell], timeout=30) if remote else cmd(["bash", "-lc", shell], timeout=30)
    return p.stdout + (("\nSTDERR:\n" + p.stderr) if p.stderr else "")


def start_worker(unit: str, port: int, remote_dir: str) -> list[str]:
    argv = [
        "systemd-run", "--user", f"--unit={unit}",
        "--property=KillMode=control-group", "--property=Restart=no",
        "--property=TimeoutStopSec=30",
        f"--property=StandardOutput=append:{remote_dir}/worker.log",
        f"--property=StandardError=append:{remote_dir}/worker.log",
        f"--setenv=LD_LIBRARY_PATH={LD}",
        "--setenv=OMP_NUM_THREADS=1",
        "--setenv=DS4_TP_GATE_TIMEOUT_MS=5000",
        DS4, "--rocm", "-m", str(MODEL), "--ctx", str(CTX),
        "--role", "worker", "--coordinator", COORD, str(port),
        "--tensor-parallel", "--transport", "tcp",
    ]
    p = ssh(argv, timeout=30)
    if p.returncode != 0:
        raise RuntimeError(f"worker start failed: {p.stderr or p.stdout}")
    active_remote_units.add(unit)
    return argv


def start_coordinator(unit: str, argv_bench: list[str], case_dir: Path) -> list[str]:
    argv = [
        "systemd-run", "--user", f"--unit={unit}",
        "--property=KillMode=control-group", "--property=Restart=no",
        "--property=TimeoutStopSec=30",
        f"--property=StandardOutput=append:{case_dir}/coordinator.log",
        f"--property=StandardError=append:{case_dir}/coordinator.log",
        f"--setenv=LD_LIBRARY_PATH={LD}",
        "--setenv=OMP_NUM_THREADS=1",
        "--setenv=DS4_TP_GATE_TIMEOUT_MS=5000",
    ] + argv_bench
    p = cmd(argv, timeout=30)
    if p.returncode != 0:
        raise RuntimeError(f"coordinator start failed: {p.stderr or p.stdout}")
    active_local_units.add(unit)
    return argv


def wait_case(local_unit: str, remote_unit: str, deadline_s: int, case_dir: Path) -> tuple[dict, dict]:
    started = time.time()
    last_write = 0.0
    while True:
        if stop_requested:
            raise RuntimeError("stop requested")
        local = status_unit(local_unit)
        remote = status_unit(remote_unit, remote=True)
        now = time.time()
        if now - last_write >= 30:
            atomic(case_dir / "live-status.json", {
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "elapsed_s": now - started,
                "coordinator": local,
                "worker": remote,
            })
            last_write = now
        state = local.get("ActiveState")
        if state in {"inactive", "failed"}:
            return local, remote
        if now - started > deadline_s:
            raise TimeoutError(f"case deadline {deadline_s}s exceeded")
        time.sleep(5)


def copy_remote_log(remote_dir: str, case_dir: Path) -> None:
    p = ssh(["cat", f"{remote_dir}/worker.log"], timeout=30)
    (case_dir / "worker.remote.log").write_text(p.stdout + (("\nSTDERR:\n" + p.stderr) if p.stderr else ""))


def parse_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def inspect_log(case_dir: Path) -> list[str]:
    text = ""
    for p in [case_dir / "coordinator.log", case_dir / "worker.remote.log"]:
        if p.exists():
            text += "\n" + p.read_text(errors="replace")
    needles = [
        r"out of memory", r"\boom\b", r"gpu fault", r"segmentation",
        r"desync", r"\bnan\b", r"truncat", r"failed to create session",
    ]
    return [n for n in needles if re.search(n, text, re.I)]


def run_case(case_id: str, bench_args: list[str], port: int, deadline_s: int,
             expected_rows: list[tuple[int, int]], phase: str) -> dict:
    case_dir = RAW / phase / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    frontier = case_dir / "frontiers"
    frontier.mkdir(exist_ok=True)
    remote_dir = str(case_dir)
    ssh(["mkdir", "-p", remote_dir], timeout=20, check=True)

    terminal_path = case_dir / "terminal.json"
    if terminal_path.exists():
        prior = json.loads(terminal_path.read_text())
        if prior.get("state") == "COMPLETE":
            return prior

    local_unit = f"d4lc-{case_id}-c"
    remote_unit = f"d4lc-{case_id}-w"
    csv_path = case_dir / "result.csv"
    argv_bench = [
        str(BENCH), "--backend", "rocm", "-m", str(MODEL),
        "--prompt-file", str(CORPUS),
        "--ctx-alloc", str(CTX),
        "--csv", str(csv_path),
        "--dump-frontier-logits-dir", str(frontier),
        "--role", "coordinator", "--listen", COORD, str(port),
        "--tensor-parallel", "--transport", "tcp",
    ] + bench_args

    record: dict[str, Any] = {
        "case_id": case_id,
        "phase": phase,
        "state": "IN_FLIGHT",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "deadline_s": deadline_s,
        "port": port,
        "ctx_alloc": CTX,
        "bench_argv": argv_bench,
        "expected_rows": expected_rows,
    }
    atomic(case_dir / "registry.json", record)
    (case_dir / "metrics.before.node01.txt").write_text(snapshot_text(False))
    (case_dir / "metrics.before.node02.txt").write_text(snapshot_text(True))

    worker_argv: list[str] = []
    coord_argv: list[str] = []
    failure: str | None = None
    local_status: dict = {}
    remote_status: dict = {}
    try:
        worker_argv = start_worker(remote_unit, port, remote_dir)
        time.sleep(2)
        coord_argv = start_coordinator(local_unit, argv_bench, case_dir)
        local_status, remote_status = wait_case(local_unit, remote_unit, deadline_s, case_dir)
    except Exception as e:
        failure = f"{type(e).__name__}: {e}"
    finally:
        try:
            local_status = local_status or status_unit(local_unit)
        except Exception:
            pass
        try:
            remote_status = remote_status or status_unit(remote_unit, remote=True)
        except Exception:
            pass
        try:
            stop_pair(local_unit, remote_unit)
        except Exception as e:
            failure = failure or f"cleanup: {type(e).__name__}: {e}"
        try:
            copy_remote_log(remote_dir, case_dir)
        except Exception as e:
            (case_dir / "worker-copy-error.txt").write_text(str(e))
        (case_dir / "metrics.after.node01.txt").write_text(snapshot_text(False))
        (case_dir / "metrics.after.node02.txt").write_text(snapshot_text(True))

    rows: list[dict[str, str]] = []
    if csv_path.exists():
        try:
            rows = parse_csv(csv_path)
        except Exception as e:
            failure = failure or f"csv parse: {e}"
    row_shape = []
    for row in rows:
        try:
            row_shape.append((int(row["ctx_tokens"]), int(row["prefill_tokens"])))
        except Exception:
            pass
    log_flags = inspect_log(case_dir)
    exec_ok = str(local_status.get("ExecMainStatus", "")) == "0" and local_status.get("Result") in {"success", ""}
    expected_ok = row_shape == expected_rows
    technical_ok = failure is None and exec_ok and expected_ok and not log_flags
    result = {
        **record,
        "state": "COMPLETE",
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "PASS_TECHNICAL" if technical_ok else "TECHNICAL_STOP",
        "failure": failure,
        "coordinator_unit": local_status,
        "worker_unit": remote_status,
        "worker_argv": worker_argv,
        "coordinator_systemd_argv": coord_argv,
        "csv_rows": rows,
        "row_shape": row_shape,
        "log_flags": log_flags,
    }
    atomic(terminal_path, result)
    return result


def update_registry(phase: str, cases: list[dict], extra: dict | None = None) -> None:
    obj = {
        "schema": "ds4-long-context-baseline-001-p1p2-v1",
        "campaign": CAMPAIGN,
        "state": "IN_FLIGHT",
        "phase": phase,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cases": cases,
    }
    if extra:
        obj.update(extra)
    atomic(REGISTRY, obj)


def final_restore(lock_fp, reason: str) -> dict:
    for u in list(active_local_units):
        try:
            cmd(["systemctl", "--user", "stop", u], timeout=45)
        except Exception:
            pass
    for u in list(active_remote_units):
        try:
            ssh(["systemctl", "--user", "stop", u], timeout=45)
        except Exception:
            pass

    try:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass

    out: dict[str, Any] = {"reason": reason, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    try:
        before = lifecycle()
        out["lifecycle_before"] = {"http": before[0], "body": before[1]}
        if before[0] == 200 and before[1].get("state") != "READY":
            st, body = lifecycle("POST", "on", {"confirm": True})
            out["on_reply"] = {"http": st, "body": body}
        out["ready"] = wait_lifecycle("READY", 1500)
        out["smoke"] = chat_smoke("LONGCTX-RESTORE-OK")
        out["status"] = "PASS" if out["smoke"]["pass"] else "FAIL"
    except Exception as e:
        out["status"] = "FAIL"
        out["error"] = f"{type(e).__name__}: {e}"
    out["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic(RESTORE, out)
    atomic(OWNER_FILE, {
        "campaign": CAMPAIGN,
        "state": "COMPLETE" if out["status"] == "PASS" else "RESTORE_FAILED",
        "updated_at": out["finished_at"],
        "restore": str(RESTORE),
    })
    return out


def handle_signal(signum, frame):
    global stop_requested
    stop_requested = True


def main() -> None:
    global stop_requested
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    RAW.mkdir(parents=True, exist_ok=True)
    if TERMINAL.exists():
        print(TERMINAL.read_text(), flush=True)
        return
    for p in [BENCH, MODEL, CORPUS, TOKEN_FILE]:
        if not p.exists():
            raise RuntimeError(f"missing preflight path: {p}")
    if MODEL.stat().st_size != 365713686528:
        raise RuntimeError("model size gate failed")

    initial = lifecycle()
    if initial[0] != 200 or initial[1].get("state") != "READY":
        raise RuntimeError(f"gateway not READY at acquisition: {initial}")

    atomic(REGISTRY, {
        "schema": "ds4-long-context-baseline-001-p1p2-v1",
        "campaign": CAMPAIGN,
        "state": "IN_FLIGHT",
        "phase": "ACQUIRE",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "lifecycle_before": initial[1],
        "cases": [],
    })

    off_http, off_body = lifecycle("POST", "off", {"confirm": True})
    off_state = wait_lifecycle("OFF", 360)
    atomic(RAW / "gateway-off-before-p1-p2.json", {"http": off_http, "reply": off_body, "state": off_state})

    lock_fp = LOCK_FILE.open("a+")
    fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    atomic(OWNER_FILE, {
        "campaign": CAMPAIGN,
        "state": "IN_FLIGHT",
        "phase": "P1_P2",
        "pid": os.getpid(),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "raw": str(RAW),
    })

    cases: list[dict] = []
    campaign_error: str | None = None
    restore: dict = {}
    try:
        port = 19475
        p1_supported: dict[int, bool] = {}
        for case_id, depth, gen in P1_FIRST:
            deadline = 5400 if depth >= 65536 else (3600 if depth >= 32768 else 2400)
            args = [
                "--ctx-start", str(depth), "--ctx-max", str(depth),
                "--gen-tokens", str(gen), "--show-output",
            ]
            res = run_case(case_id, args, port, deadline, [(depth, depth)], "p1")
            cases.append(res)
            p1_supported[depth] = res["status"] == "PASS_TECHNICAL"
            update_registry("P1_FIRST_PASS", cases, {"p1_supported": p1_supported})
            port += 1
            if res["status"] != "PASS_TECHNICAL":
                break

        if p1_supported.get(65536):
            for case_id, depth, gen in P1_REPS:
                deadline = 5400 if depth >= 65536 else 3600
                args = [
                    "--ctx-start", str(depth), "--ctx-max", str(depth),
                    "--gen-tokens", str(gen), "--show-output",
                ]
                res = run_case(case_id, args, port, deadline, [(depth, depth)], "p1")
                cases.append(res)
                update_registry("P1_REPLICATES", cases, {"p1_supported": p1_supported})
                port += 1
                if res["status"] != "PASS_TECHNICAL":
                    break

        if p1_supported.get(16384):
            res = run_case(
                "p2-append-4k-8k-16k",
                ["--ctx-start", "4096", "--ctx-max", "16384", "--step-mul", "2", "--gen-tokens", "0"],
                port, 3600,
                [(4096, 4096), (8192, 4096), (16384, 8192)],
                "p2",
            )
            cases.append(res)
            update_registry("P2_APPEND_A", cases, {"p1_supported": p1_supported})
            port += 1

        if p1_supported.get(65536):
            res = run_case(
                "p2-append-56k-64k",
                ["--ctx-start", "57344", "--ctx-max", "65536", "--step-mul", "1", "--step-incr", "8192", "--gen-tokens", "0"],
                port, 5400,
                [(57344, 57344), (65536, 8192)],
                "p2",
            )
            cases.append(res)
            update_registry("P2_APPEND_B", cases, {"p1_supported": p1_supported})
            port += 1
    except Exception as e:
        campaign_error = f"{type(e).__name__}: {e}"
    finally:
        restore = final_restore(lock_fp, campaign_error or "P1_P2_COMPLETE")
        lock_fp.close()

    overall = "PASS" if campaign_error is None and all(c.get("status") == "PASS_TECHNICAL" for c in cases) and restore.get("status") == "PASS" else "PARTIAL_OR_STOP"
    terminal = {
        "schema": "ds4-long-context-baseline-001-p1p2-v1",
        "campaign": CAMPAIGN,
        "state": "COMPLETE",
        "status": overall,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "campaign_error": campaign_error,
        "cases": cases,
        "restore": restore,
    }
    atomic(TERMINAL, terminal)
    atomic(REGISTRY, terminal)
    print(json.dumps({"status": overall, "cases": len(cases), "restore": restore.get("status")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
