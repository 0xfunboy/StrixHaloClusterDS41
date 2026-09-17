#!/usr/bin/env python3
from __future__ import annotations
import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONFIG=Path('/home/funboy/models/gguf/deepseek-v4.1-flash-mixedq2-densefix/config.json')
DS4=Path('/tmp/ds4-ref/ds4.c')
SPEC=ROOT/'.vendor/vllm-dsv41/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py'
MODEL=ROOT/'.vendor/vllm-dsv41/vllm/models/deepseek_v4_1/amd/model.py'
c=json.load(open(CONFIG))['text_config']; ds4=DS4.read_text(); spec=SPEC.read_text(); model=MODEL.read_text()
assert c['num_hidden_layers']==40
assert c['sliding_window']==128
assert c['kv_source_layer_ids']==[2,8,14,20]
assert c['index_source_layer_ids']==[2,8,14,20,24,28,32,36]
assert c['dspark_target_layer_ids']==[37,38,39]
assert 'const uint32_t kv[] = {2, 8, 14, 20}' in ds4
assert 'const bool decoder_suffix = wide && total_count >= 8192u' in ds4
assert 'const uint32_t needed = 1u + (DS4_N_LAYER - 1u - il) * 127u;' in ds4
assert 'self.hidden_states[:num_target_tokens].copy_(hidden_states[:num_target_tokens])' in spec
assert 'self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))' in model
# Current real scheduler chunks observed in frozen code-2k path.
chunks=[1023,565]
suffix_ref=[n>=8192 for n in chunks]
# If suffix arithmetic were generalized below its upstream admission threshold,
# show how much tail each decoder layer would require.  This is diagnostic only.
def needed(layer:int)->int:
    return 1+(39-layer)*127
forced={str(layer):min(1588,needed(layer)) for layer in range(20,40)}
# K2 requires target aux states at 37/38/39 for every scheduled target token;
# preserving those rows requires the full dependency chain through those layers.
result={
 'schema':'ds41-ced-contract-v1','status':'BLOCKED_BY_K2_AUX_HIDDEN_CONTRACT',
 'topology':{'layers':40,'swa':128,'kv_sources':c['kv_source_layer_ids'],
             'index_sources':c['index_source_layer_ids'],'dspark_target_layers':c['dspark_target_layer_ids']},
 'upstream_ds4':{'pin':'8db1d1d155cb0400a86a86b9c62d0defb3a6148b',
                 'decoder_suffix_min_total_count':8192,'dependency_formula':'1 + (39-layer)*127',
                 'code2k_chunks':chunks,'suffix_enabled_for_chunks':suffix_ref,
                 'forced_1588_tail_rows_diagnostic_only':forced},
 'local_k2':{'all_target_rows_copied_to_dspark':True,
             'mtp_hidden_buffer_all_forward_rows':True,
             'consequence':'Skipping decoder rows at layers 37/38/39 would leave DSpark context hidden states missing; recomputing them requires their full upstream dependency chain and removes the CED work elimination.'},
 'decision':'Do not implement a fake last-128 slice or silently weaken DSpark. CED is blocked for the current K2/code-2k profile; continue Engram on qualified MMQ predecessor as mandated.'
}
print(json.dumps(result,indent=2))
