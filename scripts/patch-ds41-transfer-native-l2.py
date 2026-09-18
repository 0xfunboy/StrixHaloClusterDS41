#!/usr/bin/env python3
"""Apply the isolated L2 Antirez/M1 overlay to a copied DS41 release."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one patch anchor, found {count}")
    path.write_text(text.replace(old, new))


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: patch-ds41-transfer-native-l2.py SOURCE_ROOT TARGET_ROOT")
    source = Path(sys.argv[1]).resolve()
    target = Path(sys.argv[2]).resolve()

    adapter_src = source / "runtime/ds41/transfer-ds4-native-001/l2/deepseek_v41_antirez_adapter.py"
    adapter_dst = target / ".vendor/gguf-plugin/vllm_gguf_plugin/weights_adapter/deepseek_v41.py"
    shutil.copy2(adapter_src, adapter_dst)
    for rel in (
        "runtime/ds41/native_antirez_engram.py",
        "runtime/ds41/antirez_artifact_identity.py",
        "runtime/ds41/results/ds4-document-profile-002-preregister.json",
    ):
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / rel, dst)

    engram = target / ".vendor/vllm-dsv41/vllm/models/deepseek_v4_1/common/engram.py"
    replace_once(
        engram,
        """        disk_root = os.environ.get("DS41_ENGRAM2_DIR")
        head_sizes = tuple(
            size for order in layout.primes[layer_hash_index] for size in order
        )
        n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        if disk_root:
            from .disk_engram import load_disk_engram_sidecar

            layer_id = layout.layer_ids[layer_hash_index]
            sidecar_source, embed_tokens, q, k, wkv = load_disk_engram_sidecar(
                root=disk_root,
                layer_id=layer_id,
                num_embeddings=layout.num_embeddings[layer_hash_index],
                dim=self.dim,
                hc_mult=self.hc_mult,
                head_sizes=head_sizes,
                device=torch.accelerator.current_accelerator(),
            )
            self._ds41_sidecar_source = sidecar_source
            self.embed_tokens = embed_tokens
            self.wkv = wkv
            self.q_weight = nn.Parameter(q, requires_grad=False)
            self.k_weight = nn.Parameter(k, requires_grad=False)
        else:
""",
        """        disk_root = os.environ.get("DS41_ENGRAM2_DIR")
        native_gguf = os.environ.get("DS41_ANTIREZ_ENGRAM_GGUF")
        if disk_root and native_gguf:
            raise RuntimeError(
                "DS41 Engram sources are mutually exclusive: Engram2 and Antirez GGUF"
            )
        head_sizes = tuple(
            size for order in layout.primes[layer_hash_index] for size in order
        )
        n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        if native_gguf:
            from runtime.ds41.native_antirez_engram import load_native_antirez_engram

            layer_id = layout.layer_ids[layer_hash_index]
            sidecar_source, embed_tokens, q, k, wkv = load_native_antirez_engram(
                path=native_gguf,
                layer_id=layer_id,
                num_embeddings=layout.num_embeddings[layer_hash_index],
                dim=self.dim,
                hc_mult=self.hc_mult,
                head_sizes=head_sizes,
                device=torch.accelerator.current_accelerator(),
            )
            self._ds41_sidecar_source = sidecar_source
            self.embed_tokens = embed_tokens
            self.wkv = wkv
            self.q_weight = nn.Parameter(q, requires_grad=False)
            self.k_weight = nn.Parameter(k, requires_grad=False)
        elif disk_root:
            from .disk_engram import load_disk_engram_sidecar

            layer_id = layout.layer_ids[layer_hash_index]
            sidecar_source, embed_tokens, q, k, wkv = load_disk_engram_sidecar(
                root=disk_root,
                layer_id=layer_id,
                num_embeddings=layout.num_embeddings[layer_hash_index],
                dim=self.dim,
                hc_mult=self.hc_mult,
                head_sizes=head_sizes,
                device=torch.accelerator.current_accelerator(),
            )
            self._ds41_sidecar_source = sidecar_source
            self.embed_tokens = embed_tokens
            self.wkv = wkv
            self.q_weight = nn.Parameter(q, requires_grad=False)
            self.k_weight = nn.Parameter(k, requires_grad=False)
        else:
""",
    )
    replace_once(
        engram,
        """        multipliers = compute_hash_multipliers(
            layout.layer_ids, layout.max_ngram_size, vocab_size
        )
        self.register_buffer(
""",
        """        multipliers = compute_hash_multipliers(
            layout.layer_ids, layout.max_ngram_size, vocab_size
        )
        native_gguf = os.environ.get("DS41_ANTIREZ_ENGRAM_GGUF")
        if native_gguf:
            from runtime.ds41.native_antirez_engram import verify_native_hash_contract

            gate = verify_native_hash_contract(
                native_gguf,
                token_map=token_map,
                multipliers=multipliers,
                layout=layout,
                compressed_pad_id=self.pad_id,
            )
            logger.info("DS41 native Antirez Engram hash contract PASS: %s", gate)
        self.register_buffer(
""",
    )

    launcher = target / "runtime/ds41/launch-node.sh"
    replace_once(
        launcher,
        """ARTIFACT_CONFIG="$ROOT/runtime/ds41/artifact.json"
MODEL_DIR=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_dir"])' "$ARTIFACT_CONFIG")
MODEL_FILE="$MODEL_DIR/DSV41-mixedq2-00001-of-00005.gguf"
ENGRAM_DIR="/home/funboy/models/ds41/engram2-tp2/rank$rank"
for p in "$VENV/bin/python" "$VLLM_SOURCE/vllm/__init__.py" "$PLUGIN_SOURCE/vllm_gguf_plugin/__init__.py" "$GGUF_PY/gguf/__init__.py" "$MODEL_FILE" "$MODEL_DIR/config.json" "$MODEL_DIR/tokenizer.json" "$ENGRAM_DIR/model-00047-of-00048.safetensors" "$ENGRAM_DIR/model-00048-of-00048.safetensors"; do
  [[ -e "$p" ]] || { echo "missing DS41 runtime input: $p" >&2; exit 2; }
done
""",
        """ARTIFACT_CONFIG="$ROOT/runtime/ds41/artifact.json"
L2_MARKER="$ROOT/runtime/ds41/transfer-ds4-native-001-l2-release.json"
if [[ -f "$L2_MARKER" ]]; then
  MODEL_DIR=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["config_dir"])' "$L2_MARKER")
  MODEL_FILE=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["weights_file"])' "$L2_MARKER")
  ENGRAM_DIR=""
else
  MODEL_DIR=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_dir"])' "$ARTIFACT_CONFIG")
  MODEL_FILE="$MODEL_DIR/DSV41-mixedq2-00001-of-00005.gguf"
  ENGRAM_DIR="/home/funboy/models/ds41/engram2-tp2/rank$rank"
fi
required=("$VENV/bin/python" "$VLLM_SOURCE/vllm/__init__.py" "$PLUGIN_SOURCE/vllm_gguf_plugin/__init__.py" "$GGUF_PY/gguf/__init__.py" "$MODEL_FILE" "$MODEL_DIR/config.json" "$MODEL_DIR/tokenizer.json")
if [[ -n "$ENGRAM_DIR" ]]; then
  required+=("$ENGRAM_DIR/model-00047-of-00048.safetensors" "$ENGRAM_DIR/model-00048-of-00048.safetensors")
fi
for p in "${required[@]}"; do
  [[ -e "$p" ]] || { echo "missing DS41 runtime input: $p" >&2; exit 2; }
done
""",
    )
    replace_once(
        launcher,
        'export DS41_ENGRAM2_DIR="$ENGRAM_DIR"',
        '''if [[ -f "$L2_MARKER" ]]; then
  unset DS41_ENGRAM2_DIR
  export DS41_ANTIREZ_Q2=1 DS41_ANTIREZ_ENGRAM_GGUF="$MODEL_FILE"
else
  export DS41_ENGRAM2_DIR="$ENGRAM_DIR"
  unset DS41_ANTIREZ_Q2 DS41_ANTIREZ_ENGRAM_GGUF
fi''',
    )
    replace_once(
        launcher,
        '"$VENV/bin/python" -m runtime.ds41.artifact_identity verify-fast --rank "$rank"',
        '''if [[ -f "$L2_MARKER" ]]; then
  "$VENV/bin/python" -m runtime.ds41.antirez_artifact_identity verify-fast --rank "$rank"
else
  "$VENV/bin/python" -m runtime.ds41.artifact_identity verify-fast --rank "$rank"
fi''',
    )
    replace_once(
        launcher,
        '--served-model-name DeepSeek-V4.1-Flash-MixedQ2-DSpark-K2',
        '--served-model-name "$( [[ -f "$L2_MARKER" ]] && printf %s DeepSeek-V4.1-Flash-Q2-Native-M1 || printf %s DeepSeek-V4.1-Flash-MixedQ2-DSpark-K2 )"',
    )

    marker = {
        "schema": "ds41-transfer-native-l2-release-v1",
        "weights_file": "/home/funboy/models/ds41/ds4-v41-q2/DeepSeek-V4.1-Flash-Q2.gguf",
        "config_dir": "/home/funboy/models/gguf/deepseek-v4.1-flash-mixedq2-densefix",
        "engram": "native GGUF E4M3/E8M0 row264",
        "target_mode": "M1",
        "dspark_k": 0,
        "prompt_profile": "ds4-low-v1",
        "mmq_prefill": True,
        "canonical_prefill": True,
    }
    (target / "runtime/ds41/transfer-ds4-native-001-l2-release.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(marker, sort_keys=True))


if __name__ == "__main__":
    main()
