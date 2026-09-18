#!/usr/bin/env python3
"""Read-only identity gate for the already verified Antirez Q2 artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
Q2 = Path("/home/funboy/models/ds41/ds4-v41-q2/DeepSeek-V4.1-Flash-Q2.gguf")
EXPECTED_SIZE = 365713686528
EXPECTED_SHA256 = "1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42"
PROOF = ROOT / "runtime/ds41/results/ds4-document-profile-002-preregister.json"


def verify(rank: int) -> dict:
    row = json.loads(PROOF.read_text())["entry_snapshot"]["q2"]
    if row.get("verified_both_nodes") is not True:
        raise RuntimeError("Antirez Q2 frozen receipt does not verify both nodes")
    if int(row.get("size", -1)) != EXPECTED_SIZE:
        raise RuntimeError("Antirez Q2 frozen receipt size mismatch")
    if row.get("sha256_receipt") != EXPECTED_SHA256:
        raise RuntimeError("Antirez Q2 frozen receipt SHA mismatch")
    st = Q2.stat()
    if st.st_size != EXPECTED_SIZE:
        raise RuntimeError(f"Antirez Q2 live size mismatch: {st.st_size}")

    from runtime.ds41.native_antirez_engram import inspect_native_engram
    meta = inspect_native_engram(str(Q2))
    return {
        "status": "PASS",
        "rank": rank,
        "path": str(Q2.resolve()),
        "size": st.st_size,
        "inode": st.st_ino,
        "device": st.st_dev,
        "mtime_ns": st.st_mtime_ns,
        "sha256_receipt": EXPECTED_SHA256,
        "verified_both_nodes_receipt": True,
        "encoding": meta["encoding"],
        "layers": list(meta["layers"]),
        "rows": list(meta["rows"]),
        "token_map_entries": len(meta["token_map"]),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("verify-fast",))
    p.add_argument("--rank", type=int, choices=(0, 1), required=True)
    args = p.parse_args()
    print("DS41_ANTIREZ_Q2_IDENTITY " + json.dumps(verify(args.rank), sort_keys=True))


if __name__ == "__main__":
    main()
