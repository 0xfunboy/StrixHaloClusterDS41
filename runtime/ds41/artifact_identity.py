#!/usr/bin/env python3
"""Single DS41 model-artifact identity contract.

The model-bearing paths (launcher, offline SPMD and numerical tests) all read
runtime/ds41/artifact.json.  A model load is admitted only when the derived
DenseFix artifact is sealed locally and its filesystem identity still matches
that seal.  Full SHA verification is intentionally a one-time replica gate;
normal launches use the sealed receipt plus immutable stat fingerprints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "runtime/ds41/artifact.json"
RECEIPT_NAME = ".ds41-artifact-verified.json"
CHUNK = 8 * 1024 * 1024


def atomic_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb", buffering=0) as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_config() -> dict[str, Any]:
    cfg = json.loads(CONFIG_PATH.read_text())
    if cfg.get("schema") != "ds41-artifact-v1":
        raise RuntimeError("unexpected DS41 artifact schema")
    return cfg


def model_dir(cfg: dict[str, Any]) -> Path:
    p = Path(cfg["model_dir"])
    if str(p.resolve()) != cfg["model_dir"]:
        raise RuntimeError(f"model_dir must be canonical: {p}")
    if str(p.resolve()) == str(Path(cfg["original_model_dir"]).resolve()):
        raise RuntimeError("DenseFix path resolves to original MixedQ2")
    return p


def runtime_commit() -> str:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        marker = ROOT / ".source-commit"
        if not marker.is_file():
            raise RuntimeError("DS41 source identity unavailable: no Git HEAD or .source-commit")
        value = marker.read_text().strip()
    if len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value.lower()):
        raise RuntimeError(f"invalid DS41 source identity: {value!r}")
    return value.lower()


def validate_densefix_receipt(cfg: dict[str, Any], d: Path) -> dict[str, Any]:
    vf = d / cfg["validation"]["file"]
    if not vf.is_file():
        raise RuntimeError(f"missing DenseFix validator receipt: {vf}")
    rec = json.loads(vf.read_text())
    summary = rec.get("summary", {})
    for key, expected in cfg["validation"].items():
        if key == "file":
            continue
        if summary.get(key) != expected:
            raise RuntimeError(
                f"DenseFix validation receipt mismatch {key}: "
                f"{summary.get(key)!r} != {expected!r}"
            )
    shard_results = rec.get("shards", {})
    for s in cfg["shards"]:
        row = shard_results.get(s["name"])
        if not row or row.get("repaired_sha256") != s["sha256"]:
            raise RuntimeError(f"validator SHA missing/mismatch for {s['name']}")
        if row.get("unpatched_matches_original") is not True:
            raise RuntimeError(f"unpatched-byte receipt failed for {s['name']}")
    return rec


def engram_identity(cfg: dict[str, Any], rank: int) -> dict[str, Any]:
    d = Path(cfg["engram_root"]) / f"rank{rank}"
    rows = []
    for name in cfg["engram_files"]:
        p = d / name
        if not p.is_file():
            raise RuntimeError(f"missing Engram2 rank{rank} file: {p}")
        st = p.stat()
        rows.append({"realpath": str(p.resolve()), "size": st.st_size})
    return {"rank": rank, "dir": str(d.resolve()), "files": rows}


def shard_stat(p: Path) -> dict[str, Any]:
    st = p.stat()
    return {
        "realpath": str(p.resolve()),
        "size": st.st_size,
        "inode": st.st_ino,
        "device": st.st_dev,
        "mtime_ns": st.st_mtime_ns,
    }


def seal_from_validation(rank: int) -> dict[str, Any]:
    """Seal NODE01 without re-reading 169 GB already fully hashed by validator."""
    cfg = load_config(); d = model_dir(cfg)
    rec = validate_densefix_receipt(cfg, d)
    shard_results = rec["shards"]
    shards: dict[str, Any] = {}
    for s in cfg["shards"]:
        p = d / s["name"]
        st = shard_stat(p)
        if st["size"] != s["size"]:
            raise RuntimeError(f"size mismatch {s['name']}")
        st["sha256"] = shard_results[s["name"]]["repaired_sha256"]
        if st["sha256"] != s["sha256"]:
            raise RuntimeError(f"validator SHA mismatch {s['name']}")
        shards[s["name"]] = st
    out = {
        "schema": "ds41-artifact-verified-v1",
        "verified_by": "densefix-independent-validator",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_dir": str(d),
        "artifact_name": cfg["name"],
        "runtime_commit_at_seal": runtime_commit(),
        "source_revision": cfg["source_revision"],
        "densefix_manifest_sha256": cfg["densefix_manifest_sha256"],
        "shards": shards,
        "engram": engram_identity(cfg, rank),
    }
    atomic_json(d / RECEIPT_NAME, out)
    return out


def verify_full(rank: int) -> dict[str, Any]:
    """One-time replica verifier.  Reads every derived shard and seals it."""
    cfg = load_config(); d = model_dir(cfg)
    validate_densefix_receipt(cfg, d)
    shards: dict[str, Any] = {}
    for s in cfg["shards"]:
        p = d / s["name"]
        st = shard_stat(p)
        if st["size"] != s["size"]:
            raise RuntimeError(f"size mismatch {s['name']}: {st['size']} != {s['size']}")
        got = sha256_file(p)
        print(f"DS41_ARTIFACT_FULL_SHA {s['name']} {got}", flush=True)
        if got != s["sha256"]:
            raise RuntimeError(f"SHA mismatch {s['name']}: {got} != {s['sha256']}")
        st["sha256"] = got
        shards[s["name"]] = st
    out = {
        "schema": "ds41-artifact-verified-v1",
        "verified_by": "full-sha256",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_dir": str(d),
        "artifact_name": cfg["name"],
        "runtime_commit_at_seal": runtime_commit(),
        "source_revision": cfg["source_revision"],
        "densefix_manifest_sha256": cfg["densefix_manifest_sha256"],
        "shards": shards,
        "engram": engram_identity(cfg, rank),
    }
    atomic_json(d / RECEIPT_NAME, out)
    return out


def verify_fast(rank: int) -> dict[str, Any]:
    cfg = load_config(); d = model_dir(cfg)
    validate_densefix_receipt(cfg, d)
    receipt_path = d / RECEIPT_NAME
    if not receipt_path.is_file():
        raise RuntimeError(f"artifact is not sealed locally: {receipt_path}")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != "ds41-artifact-verified-v1":
        raise RuntimeError("bad local artifact receipt schema")
    if receipt.get("model_dir") != str(d):
        raise RuntimeError("artifact receipt path mismatch")
    if receipt.get("source_revision") != cfg["source_revision"]:
        raise RuntimeError("artifact receipt source revision mismatch")
    if receipt.get("densefix_manifest_sha256") != cfg["densefix_manifest_sha256"]:
        raise RuntimeError("artifact receipt manifest mismatch")
    opened = []
    for s in cfg["shards"]:
        p = d / s["name"]
        now = shard_stat(p)
        sealed = receipt.get("shards", {}).get(s["name"], {})
        for key in ("realpath", "size", "inode", "device", "mtime_ns"):
            if now.get(key) != sealed.get(key):
                raise RuntimeError(
                    f"artifact changed since seal: {s['name']} {key} "
                    f"{now.get(key)!r} != {sealed.get(key)!r}"
                )
        if sealed.get("sha256") != s["sha256"]:
            raise RuntimeError(f"sealed SHA mismatch: {s['name']}")
        opened.append({**now, "sha256": s["sha256"]})
    engram = engram_identity(cfg, rank)
    return {
        "status": "PASS",
        "artifact": cfg["name"],
        "model_dir": str(d),
        "model_file": str((d / cfg["model_file"]).resolve()),
        "source_revision": cfg["source_revision"],
        "densefix_manifest_sha256": cfg["densefix_manifest_sha256"],
        "runtime_commit": runtime_commit(),
        "rank": rank,
        "shards": opened,
        "engram": engram,
        "seal": str(receipt_path),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("print-model-dir", "seal-from-validation", "verify-full", "verify-fast"))
    p.add_argument("--rank", type=int, choices=(0, 1), default=0)
    args = p.parse_args()
    if args.command == "print-model-dir":
        print(model_dir(load_config()))
        return
    if args.command == "seal-from-validation":
        out = seal_from_validation(args.rank)
    elif args.command == "verify-full":
        out = verify_full(args.rank)
    else:
        out = verify_fast(args.rank)
    print("DS41_ARTIFACT_IDENTITY " + json.dumps(out, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
