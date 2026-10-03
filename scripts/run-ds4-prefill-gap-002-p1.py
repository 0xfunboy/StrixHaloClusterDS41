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

CAMPAIGN = "DS4_PREFILL_GAP_002"
ROOT = Path("/home/funboy/StrixHaloClusterDS41")
RAW = Path("/home/funboy/reports/DS4-PREFILL-GAP-002")
ART = RAW / "artifacts/diag"
MODEL = Path("/home/funboy/models/ds41/ds4-v41-q2/DeepSeek-V4.1-Flash-Q2.gguf")
CORPUS = Path("/home/funboy/worktrees/ds4-prefill-gap-002/speed-bench/promessi_sposi.txt")
TOKEN = Path("/home/funboy/.local/state/ds4-document-profile-002/api-token")
LOCK = Path("/home/funboy/.local/state/strix-cluster/compute.lock")
OWNER = Path("/home/funboy/.local/state/strix-cluster/ds4-prefill-gap-002-owner.json")
REGISTRY = RAW / "p1-registry.json"
TERMINAL = RAW / "p1-terminal.json"
RESTORE = RAW / "restore-after-p1.json"
GATEWAY = ("127.0.0.1", 18224)
COORD = "10.55.0.1"
PORT = 19700
CTX = 69632
VENV = Path("/home/funboy/StrixHaloClusterGLM/.engine/venv")
SSH = ["ssh", "-o", "IdentityAgent=none", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "02-evo-x3-tb"]
stop_requested = False
local_unit = "d4pg2-p1-c"
remote_unit = "d4pg2-p1-w"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def cmd(args: list[str], timeout: float | None = None, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, timeout=timeout, check=check)


def ssh(args: list[str], timeout: float | None = None, check: bool = False) -> subprocess.CompletedProcess[str]:
    return cmd(SSH + args, timeout=timeout, check=check)


def lifecycle(method: str = "GET", action: str | None = None, body: dict | None = None, timeout: int = 30) -> tuple[int, dict]:
    token = TOKEN.read_text().strip()
    path = "/v1/lifecycle" + (f"/{action}" if action else "")
    headers = {"Authorization": "Bearer " + token}
    payload = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        payload = json.dumps(body, separators=(",", ":")).encode()
    c = http.client.HTTPConnection(*GATEWAY, timeout=timeout)
    c.request(method, path, body=payload, headers=headers)
    r = c.getresponse()
    raw = r.read()
    status = r.status
    c.close()
    try:
        data = json.loads(raw)
    except Exception:
        data = {"_raw": raw.decode(errors="replace")}
    return status, data


def wait_lifecycle(state: str, timeout_s: int) -> dict:
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        last = lifecycle()
        if last[0] == 200 and last[1].get("state") == state:
            return last[1]
        time.sleep(3)
    raise RuntimeError(f"lifecycle timeout {state}: {last}")


def smoke(expected: str) -> dict:
    token = TOKEN.read_text().strip()
    body = {
        "model": "deepseek-v4.1-flash",
        "profile": "document-low",
        "messages": [{"role": "user", "content": f"Return exactly {expected}."}],
        "temperature": 0,
        "seed": 1,
        "max_tokens": 128,
        "stream": False,
    }
    start = time.monotonic()
    c = http.client.HTTPConnection(*GATEWAY, timeout=1800)
    c.request("POST", "/v1/chat/completions",
              body=json.dumps(body, separators=(",", ":")).encode(),
              headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
    r = c.getresponse()
    raw = r.read()
    status = r.status
    c.close()
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
        "wall_s": time.monotonic() - start,
        "pass": status == 200 and content.strip() == expected and choice.get("finish_reason") == "stop",
    }


def status_unit(unit: str, remote: bool = False) -> dict:
    args = ["systemctl", "--user", "show", unit + ".service",
            "-p", "ActiveState", "-p", "SubState", "-p", "Result",
            "-p", "ExecMainStatus", "-p", "MainPID", "-p", "InvocationID", "--no-pager"]
    p = ssh(args, timeout=15) if remote else cmd(args, timeout=15)
    out: dict[str, Any] = {"returncode": p.returncode, "stderr": p.stderr.strip()}
    for line in p.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


def snapshot(remote: bool = False) -> str:
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
echo '--- thunderbolt ---'
for f in /sys/class/net/thunderbolt0/statistics/rx_bytes /sys/class/net/thunderbolt0/statistics/tx_bytes /sys/class/net/thunderbolt0/statistics/rx_errors /sys/class/net/thunderbolt0/statistics/tx_errors; do
  if test -r "$f"; then printf '%s=' "$f"; cat "$f"; fi
done
echo '--- power ---'
powerprofilesctl get 2>/dev/null || true
"""
    p = ssh(["bash", "-lc", shell], timeout=30) if remote else cmd(["bash", "-lc", shell], timeout=30)
    return p.stdout + (("\nSTDERR:\n" + p.stderr) if p.stderr else "")


def site_ld() -> str:
    site = cmd([str(VENV / "bin/python"), "-c", "import site; print(site.getsitepackages()[0])"],
               timeout=30, check=True).stdout.strip()
    return ":".join([
        "/usr/lib/x86_64-linux-gnu",
        f"{site}/_rocm_sdk_core/lib",
        f"{site}/_rocm_sdk_devel/lib",
        f"{site}/_rocm_sdk_libraries/lib",
        f"{site}/torch/lib",
    ])


def stop_pair() -> dict:
    lp = subprocess.Popen(["systemctl", "--user", "stop", local_unit + ".service"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    rp = subprocess.Popen(SSH + ["systemctl", "--user", "stop", remote_unit + ".service"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        lo, le = lp.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        lp.kill(); lo, le = lp.communicate()
    try:
        ro, re_ = rp.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        rp.kill(); ro, re_ = rp.communicate()
    return {
        "local_stop": {"stdout": lo, "stderr": le},
        "remote_stop": {"stdout": ro, "stderr": re_},
        "local": status_unit(local_unit),
        "remote": status_unit(remote_unit, True),
    }


def copy_remote_log(case_dir: Path) -> None:
    p = ssh(["cat", str(case_dir / "worker.log")], timeout=30)
    (case_dir / "worker.remote.log").write_text(p.stdout + (("\nSTDERR:\n" + p.stderr) if p.stderr else ""))


def parse_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def restore(lockfp, reason: str) -> dict:
    try:
        stop_pair()
    except Exception:
        pass
    try:
        fcntl.flock(lockfp.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass
    out: dict[str, Any] = {"reason": reason, "started_at": now()}
    try:
        before = lifecycle()
        out["before"] = {"http": before[0], "body": before[1]}
        if before[0] == 200 and before[1].get("state") != "READY":
            st, body = lifecycle("POST", "on", {"confirm": True})
            out["on"] = {"http": st, "body": body}
        out["ready"] = wait_lifecycle("READY", 1500)
        out["smoke"] = smoke("PREFILL-GAP-P1-RESTORE-OK")
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


def main() -> None:
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    if TERMINAL.exists():
        print(TERMINAL.read_text(), flush=True)
        return
    for p in [ART / "ds4", ART / "ds4-bench-warm", MODEL, CORPUS, TOKEN, RAW / "build-diag/BUILD_COMPLETE"]:
        if not p.exists():
            raise RuntimeError(f"missing preflight {p}")
    if MODEL.stat().st_size != 365713686528:
        raise RuntimeError("model size gate failed")

    initial = lifecycle()
    if initial[0] != 200 or initial[1].get("state") != "READY":
        raise RuntimeError(f"E1 not READY at P1 acquisition: {initial}")

    case_dir = RAW / "p1" / "diag-16k"
    case_dir.mkdir(parents=True, exist_ok=True)
    ssh(["mkdir", "-p", str(case_dir)], timeout=20, check=True)
    atomic(REGISTRY, {
        "schema": "ds4-prefill-gap-002-p1-v1",
        "campaign": CAMPAIGN,
        "state": "IN_FLIGHT",
        "phase": "ACQUIRE",
        "started_at": now(),
        "production_before": initial[1],
    })

    off = lifecycle("POST", "off", {"confirm": True})
    off_state = wait_lifecycle("OFF", 420)
    atomic(RAW / "gateway-off-before-p1.json", {"reply": off, "state": off_state})

    lockfp = LOCK.open("a+")
    fcntl.flock(lockfp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    atomic(OWNER, {
        "campaign": CAMPAIGN, "state": "IN_FLIGHT", "phase": "P1_DIAGNOSTIC",
        "pid": os.getpid(), "updated_at": now(), "raw": str(RAW)
    })

    result: dict[str, Any] = {}
    error: str | None = None
    try:
        ld = site_ld()
        common_props = [
            "--property=KillMode=control-group",
            "--property=Restart=no",
            "--property=TimeoutStopSec=45",
            f"--setenv=LD_LIBRARY_PATH={ld}",
            "--setenv=OMP_NUM_THREADS=1",
            "--setenv=DS4_TP_GATE_TIMEOUT_MS=5000",
            "--setenv=DS4_METAL_GRAPH_PREFILL_PROFILE=1",
            "--setenv=DS4_TP_PREFILL_PROFILE=1",
        ]
        for remote, unit in [(False, local_unit), (True, remote_unit)]:
            fn = ssh if remote else cmd
            fn(["systemctl", "--user", "stop", unit + ".service"], timeout=30)
            fn(["systemctl", "--user", "reset-failed", unit + ".service"], timeout=20)

        (case_dir / "metrics.before.node01.txt").write_text(snapshot(False))
        (case_dir / "metrics.before.node02.txt").write_text(snapshot(True))

        worker = [
            "systemd-run", "--user", f"--unit={remote_unit}",
            *common_props,
            f"--property=StandardOutput=append:{case_dir}/worker.log",
            f"--property=StandardError=append:{case_dir}/worker.log",
            str(ART / "ds4"), "--rocm", "-m", str(MODEL), "--ctx", str(CTX),
            "--role", "worker", "--coordinator", COORD, str(PORT),
            "--tensor-parallel", "--transport", "tcp",
        ]
        wp = ssh(worker, timeout=30)
        if wp.returncode != 0:
            raise RuntimeError(f"worker launch failed: {wp.stderr or wp.stdout}")
        time.sleep(2)

        csv_path = case_dir / "result.csv"
        frontier = case_dir / "frontiers"
        frontier.mkdir(exist_ok=True)
        bench = [
            str(ART / "ds4-bench-warm"), "--backend", "rocm", "-m", str(MODEL),
            "--prompt-file", str(CORPUS),
            "--ctx-start", "16384", "--ctx-max", "16384", "--ctx-alloc", str(CTX),
            "--gen-tokens", "512", "--show-output",
            "--csv", str(csv_path), "--dump-frontier-logits-dir", str(frontier),
            "--role", "coordinator", "--listen", COORD, str(PORT),
            "--tensor-parallel", "--transport", "tcp",
        ]
        coordinator = [
            "systemd-run", "--user", f"--unit={local_unit}",
            *common_props,
            f"--property=StandardOutput=append:{case_dir}/coordinator.log",
            f"--property=StandardError=append:{case_dir}/coordinator.log",
            *bench,
        ]
        cp = cmd(coordinator, timeout=30)
        if cp.returncode != 0:
            raise RuntimeError(f"coordinator launch failed: {cp.stderr or cp.stdout}")

        atomic(REGISTRY, {
            "schema": "ds4-prefill-gap-002-p1-v1", "campaign": CAMPAIGN,
            "state": "IN_FLIGHT", "phase": "P1_DIAGNOSTIC", "updated_at": now(),
            "local_unit": local_unit, "remote_unit": remote_unit, "bench_argv": bench
        })
        deadline = time.time() + 3600
        while time.time() < deadline:
            if stop_requested:
                raise RuntimeError("stop requested")
            ls = status_unit(local_unit)
            rs = status_unit(remote_unit, True)
            atomic(case_dir / "live-status.json", {
                "updated_at": now(), "coordinator": ls, "worker": rs
            })
            if ls.get("ActiveState") in {"inactive", "failed"}:
                break
            time.sleep(10)
        else:
            raise TimeoutError("P1 diagnostic 16K deadline exceeded")

        ls = status_unit(local_unit)
        rs = status_unit(remote_unit, True)
        pair_stop = stop_pair()
        copy_remote_log(case_dir)
        (case_dir / "metrics.after.node01.txt").write_text(snapshot(False))
        (case_dir / "metrics.after.node02.txt").write_text(snapshot(True))
        rows = parse_csv(csv_path) if csv_path.exists() else []
        log_text = ""
        for p in [case_dir / "coordinator.log", case_dir / "worker.remote.log"]:
            if p.exists():
                log_text += p.read_text(errors="replace")
        faults = [x for x in ["out of memory", "gpu fault", "segmentation", "desynchron", "nan"]
                  if re.search(x, log_text, re.I)]
        ok = (
            ls.get("Result") in {"success", ""} and ls.get("ExecMainStatus") == "0"
            and len(rows) == 1 and rows[0].get("ctx_tokens") == "16384"
            and not faults
        )
        result = {
            "state": "COMPLETE",
            "status": "PASS_TECHNICAL" if ok else "TECHNICAL_STOP",
            "finished_at": now(),
            "coordinator": ls,
            "worker": rs,
            "pair_stop": pair_stop,
            "csv_rows": rows,
            "faults": faults,
        }
        atomic(case_dir / "terminal.json", result)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    finally:
        restored = restore(lockfp, error or "P1_DIAGNOSTIC_COMPLETE")
        lockfp.close()

    terminal = {
        "schema": "ds4-prefill-gap-002-p1-v1",
        "campaign": CAMPAIGN,
        "state": "COMPLETE",
        "status": "PASS" if error is None and result.get("status") == "PASS_TECHNICAL" and restored.get("status") == "PASS" else "PARTIAL_OR_STOP",
        "error": error,
        "result": result,
        "restore": restored,
        "finished_at": now(),
    }
    atomic(TERMINAL, terminal)
    atomic(REGISTRY, terminal)
    atomic(OWNER, {
        "campaign": CAMPAIGN, "state": "COMPLETE",
        "updated_at": terminal["finished_at"], "terminal": str(TERMINAL),
        "restore": str(RESTORE)
    })
    print(json.dumps({"status": terminal["status"], "error": error,
                      "row": (result.get("csv_rows") or [None])[0],
                      "restore": restored.get("status")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
