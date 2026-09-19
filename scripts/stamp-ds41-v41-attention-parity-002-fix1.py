#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

OLD = "DeepSeek-V4.1-Flash-Q2-AttnParity002-M1"
NEW = "DeepSeek-V4.1-Flash-Q2-AttnParity002-M1-Fix1"
PROFILE = "DS41_V41_ATTENTION_PARITY_002_M1_FIX1"


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if text.count(old) != 1:
        raise RuntimeError(f"{path}: expected one {old!r}")
    path.write_text(text.replace(old, new))


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: stamp-ds41-v41-attention-parity-002-fix1.py RELEASE")
    root = Path(sys.argv[1]).resolve()
    marker = root / "runtime/ds41/attention-parity-002-release.txt"
    if not marker.is_file():
        raise RuntimeError("not an ATTENTION PARITY 002 release")

    launcher = root / "runtime/ds41/launch-node.sh"
    replace_once(launcher, OLD, NEW)

    src = root / "config.attention-parity-002-m1.json"
    dst = root / "config.attention-parity-002-m1-fix1.json"
    if dst.exists():
        raise RuntimeError(f"refuse overwrite {dst}")
    cfg = json.loads(src.read_text())
    if cfg.get("model") != OLD:
        raise RuntimeError(f"unexpected source model {cfg.get('model')!r}")
    cfg["model"] = NEW
    cfg["profile_status"] = PROFILE
    dst.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")

    with marker.open("a") as f:
        f.write("implementation_fix=portable-rocm-e2m1-rtne-v1\n")
        f.write(f"served_model={NEW}\n")
    print(json.dumps({"release": str(root), "model": NEW, "profile_status": PROFILE}))


if __name__ == "__main__":
    main()
