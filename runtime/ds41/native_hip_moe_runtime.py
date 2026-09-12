#!/usr/bin/env python3
"""Fail-closed loader for the isolated DS41 native HIP GGUF MoE extension."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "runtime/ds41/native_hip_moe/config.json"
_CHUNK = 8 * 1024 * 1024
_LOADED_IDENTITY: dict[str, Any] | None = None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb", buffering=0) as f:
        while True:
            b = f.read(_CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_config() -> dict[str, Any]:
    cfg = json.loads(CONFIG_PATH.read_text())
    if cfg.get("schema") != "ds41-native-hip-moe-v1":
        raise RuntimeError("unexpected DS41 native HIP MoE config schema")
    return cfg


def verify_library() -> dict[str, Any]:
    import torch

    cfg = load_config()
    path = Path(cfg["library_path"])
    if not path.is_file():
        raise RuntimeError(f"DS41 native HIP MoE library missing: {path}")
    got = _sha256(path)
    if got != cfg["library_sha256"]:
        raise RuntimeError(
            f"DS41 native HIP MoE SHA mismatch: {got} != {cfg['library_sha256']}"
        )
    source_files = {
        "patch_sha256": ROOT / "runtime/ds41/native_hip_moe/moe-negskip.patch",
        "build_script_sha256": ROOT / "runtime/ds41/native_hip_moe/build.sh",
    }
    for key, source_path in source_files.items():
        source_got = _sha256(source_path)
        if source_got != cfg[key]:
            raise RuntimeError(
                f"DS41 native HIP source identity mismatch {key}: "
                f"{source_got} != {cfg[key]}"
            )
    if torch.version.hip != cfg["hip"]:
        raise RuntimeError(f"DS41 native HIP ROCm mismatch: {torch.version.hip} != {cfg['hip']}")
    if torch.__version__ != cfg["torch"]:
        raise RuntimeError(f"DS41 native HIP torch mismatch: {torch.__version__} != {cfg['torch']}")
    if not torch.cuda.is_available():
        raise RuntimeError("DS41 native HIP requires ROCm device")
    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", "")
    if arch != cfg["arch"]:
        raise RuntimeError(f"DS41 native HIP arch mismatch: {arch!r} != {cfg['arch']!r}")
    return {
        "status": "PASS",
        "path": str(path.resolve()),
        "sha256": got,
        "plugin_pin": cfg["plugin_pin"],
        "patch_sha256": cfg["patch_sha256"],
        "build_script_sha256": cfg["build_script_sha256"],
        "patched_source_sha256": cfg["patched_source_sha256"],
        "hipified_source_sha256": cfg["hipified_source_sha256"],
        "source_provenance": cfg["source_provenance"],
        "arch": arch,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "qualified_dispatch": cfg["qualified_dispatch"],
        "numeric_gate": cfg["numeric_gate"],
    }


def ensure_loaded() -> dict[str, Any]:
    global _LOADED_IDENTITY
    if _LOADED_IDENTITY is not None:
        return dict(_LOADED_IDENTITY)
    import torch

    identity = verify_library()
    torch.ops.load_library(identity["path"])
    namespace = getattr(torch.ops, "_C_gguf", None)
    if namespace is None or not hasattr(namespace, "ggml_moe_a8_vec"):
        raise RuntimeError("DS41 native HIP library did not register ggml_moe_a8_vec")
    schema = str(torch.ops._C_gguf.ggml_moe_a8_vec.default._schema)
    expected = "_C_gguf::ggml_moe_a8_vec(Tensor X, Tensor W, Tensor topk_ids, int top_k, int type, SymInt row, SymInt tokens) -> Tensor"
    if schema != expected:
        raise RuntimeError(f"DS41 native HIP op schema mismatch: {schema}")
    identity["schema"] = schema
    _LOADED_IDENTITY = identity
    print("DS41_NATIVE_HIP_MOE_IDENTITY " + json.dumps(identity, sort_keys=True), flush=True)
    return dict(identity)


def maybe_load_from_env() -> dict[str, Any] | None:
    if os.environ.get("DS41_NATIVE_HIP_MOE", "0") == "1":
        return ensure_loaded()
    return None


if __name__ == "__main__":
    print(json.dumps(ensure_loaded(), indent=2, sort_keys=True))
