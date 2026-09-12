#!/usr/bin/env python3
"""Construct DS41 vLLM config only. No model allocation or inference."""
from vllm_gguf_plugin import register
register()
from pathlib import Path
import json
from vllm.engine.arg_utils import EngineArgs
REPO=Path('/home/funboy/StrixHaloClusterDS41')
ART=json.loads((REPO/'runtime/ds41/artifact.json').read_text())
ROOT=ART['model_dir']
MODEL=str(Path(ROOT)/ART['model_file'])
a=EngineArgs(model=MODEL,hf_config_path=ROOT,tokenizer=ROOT,config_format='gguf',load_format='gguf',quantization='gguf',dtype='bfloat16',tensor_parallel_size=2,pipeline_parallel_size=1,distributed_executor_backend='external_launcher',enable_expert_parallel=True,max_model_len=4096,block_size=128,max_num_seqs=1,max_num_batched_tokens=1024,kv_cache_memory_bytes=1073741824,enable_prefix_caching=False,enable_chunked_prefill=True,enforce_eager=True,seed=1)
c=a.create_engine_config(); m=c.model_config
result={'architecture':m.architecture,'model_type':m.hf_config.model_type,'text_model_type':m.hf_text_config.model_type,'quantization':m.quantization,'load_format':str(c.load_config.load_format),'tp':c.parallel_config.tensor_parallel_size,'pp':c.parallel_config.pipeline_parallel_size,'ep':c.parallel_config.enable_expert_parallel,'max_model_len':m.max_model_len,'block_size':c.cache_config.block_size,'engram_layers':m.hf_text_config.engram_layer_ids,'nextn':m.hf_text_config.num_nextn_predict_layers,'speculative':c.speculative_config is not None}
print(result)
assert result['architecture']=='DeepseekV41ForCausalLM'
assert result['model_type']=='deepseek_v41' and result['quantization']=='gguf'
assert result['tp']==2 and result['pp']==1 and result['ep'] and result['block_size']==128
assert result['engram_layers']==[1,14] and not result['speculative']
print('CONFIG_PREFLIGHT=PASS')
