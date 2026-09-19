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
        "runtime/ds41/gguf_stream_cache.py",
        "runtime/ds41/results/ds4-document-profile-002-preregister.json",
        "scripts/test-ds41-transfer-native-l2-cpu.py",
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

    # Quantized top-level head/embed tensors produce sibling ``weight_type``
    # pseudo-weights in the GGUF plugin.  The V4.1 streaming wrapper must
    # reroot those companions exactly like the corresponding packed weights;
    # otherwise ``head.weight_type`` leaks outside the sole language_model
    # group and the fail-closed streaming loader correctly rejects startup.
    vl_model = target / ".vendor/vllm-dsv41/vllm/models/deepseek_v4_1/amd/vl_model.py"
    replace_once(
        vl_model,
        """        orig_to_new_suffix={
            "head.weight": "language_model.lm_head.weight",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
        },
""",
        """        orig_to_new_suffix={
            "head.weight": "language_model.lm_head.weight",
            "head.weight_type": "language_model.lm_head.weight_type",
            "embed.weight": "embed_tokens.weight",
            "embed.weight_type": "embed_tokens.weight_type",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
        },
""",
    )

    weight_utils = target / ".vendor/gguf-plugin/vllm_gguf_plugin/weight_utils.py"
    replace_once(
        weight_utils,
        "logger = init_logger(__name__)\n",
        """logger = init_logger(__name__)

from runtime.ds41.gguf_stream_cache import drop_consumed_tensor_cache, stage_anonymous_copy
""",
    )
    replace_once(
        weight_utils,
        """    drop_shard_cache = os.environ.get("DS41_DROP_SHARD_CACHE", "0") == "1"
    phase_log = os.environ.get("DS41_LOAD_PHASE_LOG", "0") == "1"

    for gguf_file in gguf_files:
""",
        """    drop_shard_cache = os.environ.get("DS41_DROP_SHARD_CACHE", "0") == "1"
    drop_tensor_cache = os.environ.get("DS41_DROP_TENSOR_CACHE", "0") == "1"
    anonymous_stage = os.environ.get("DS41_GGUF_ANON_STAGE", "0") == "1"
    phase_log = os.environ.get("DS41_LOAD_PHASE_LOG", "0") == "1"

    for gguf_file in gguf_files:
""",
    )
    replace_once(
        weight_utils,
        """        reader = gguf.GGUFReader(gguf_file)
        try:
            for tensor in reader.tensors:
""",
        """        reader = gguf.GGUFReader(gguf_file)
        drop_fd = None
        tensor_mm = None
        if drop_tensor_cache:
            tensor_mm = getattr(reader.data, "_mmap", None)
            if tensor_mm is None:
                raise RuntimeError("progressive GGUF cache drop requires mmap-backed reader")
            drop_fd = os.open(gguf_file, os.O_RDONLY)
        try:
            for tensor in reader.tensors:
""",
    )
    replace_once(
        weight_utils,
        """                weight = tensor.data
                if weight_type.name == "BF16" and weight.dtype == np.uint8:
""",
        """                weight = tensor.data
                if anonymous_stage:
                    weight = stage_anonymous_copy(weight)
                    if drop_tensor_cache:
                        if tensor_mm is None or drop_fd is None:
                            raise RuntimeError(
                                "progressive GGUF cache-drop state is unavailable"
                            )
                        drop_consumed_tensor_cache(
                            tensor_mm,
                            drop_fd,
                            data_offset=int(reader.data_offset),
                            tensor_offset=int(tensor.data_offset),
                            n_bytes=int(tensor.n_bytes),
                            file_size=int(reader.data.size),
                        )
                if weight_type.name == "BF16" and weight.dtype == np.uint8:
""",
    )
    replace_once(
        weight_utils,
        """                yield name, param
        finally:
""",
        """                yield name, param
                if drop_tensor_cache and not anonymous_stage:
                    if tensor_mm is None or drop_fd is None:
                        raise RuntimeError("progressive GGUF cache-drop state is unavailable")
                    drop_consumed_tensor_cache(
                        tensor_mm,
                        drop_fd,
                        data_offset=int(reader.data_offset),
                        tensor_offset=int(tensor.data_offset),
                        n_bytes=int(tensor.n_bytes),
                        file_size=int(reader.data.size),
                    )
        finally:
            if drop_fd is not None:
                os.close(drop_fd)
""",
    )

    launcher = target / "runtime/ds41/launch-node.sh"
    replace_once(
        launcher,
        "export DS41_LOAD_PHASE_LOG=1 DS41_DROP_SHARD_CACHE=1 DS41_STREAM_TEXT_WEIGHTS=1 DS41_MOE_C_LEGACY_TEXT_ABI=1 DS41_DECOMPOSED_QKV_INSERT=1",
        "export DS41_LOAD_PHASE_LOG=1 DS41_DROP_SHARD_CACHE=1 DS41_DROP_TENSOR_CACHE=1 DS41_GGUF_ANON_STAGE=1 DS41_STREAM_TEXT_WEIGHTS=1 DS41_MOE_C_LEGACY_TEXT_ABI=1 DS41_DECOMPOSED_QKV_INSERT=1",
    )
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
        'export DS41_ENGRAM2_DIR="$ENGRAM_DIR" DS41_ENGRAM_CACHE_ROWS="${DS41_ENGRAM_CACHE_ROWS:-65536}"',
        '''if [[ -f "$L2_MARKER" ]]; then
  unset DS41_ENGRAM2_DIR
  export DS41_ANTIREZ_Q2=1 DS41_ANTIREZ_ENGRAM_GGUF="$MODEL_FILE"
else
  export DS41_ENGRAM2_DIR="$ENGRAM_DIR"
  unset DS41_ANTIREZ_Q2 DS41_ANTIREZ_ENGRAM_GGUF
fi
export DS41_ENGRAM_CACHE_ROWS="${DS41_ENGRAM_CACHE_ROWS:-65536}"''',
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
        "target_tensor_staging": "anonymous-cpu-one-tensor-v1",
    }
    (target / "runtime/ds41/transfer-ds4-native-001-l2-release.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(marker, sort_keys=True))


if __name__ == "__main__":
    main()
