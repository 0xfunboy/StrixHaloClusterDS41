#!/usr/bin/env python3
"""Model-free gate: target GGUF and DSpark sidecar load/quant configs stay independent."""
from __future__ import annotations
import json
from pathlib import Path
import vllm_gguf_plugin
vllm_gguf_plugin.register()  # register GGUF loader/config parser for target
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.models.utils import get_draft_quant_config

ROOT=Path('/home/funboy/StrixHaloClusterDS41')
TARGET='/home/funboy/models/gguf/deepseek-v4.1-flash-mixedq2-densefix/DSV41-mixedq2-00001-of-00005.gguf'
TARGET_CFG='/home/funboy/models/gguf/deepseek-v4.1-flash-mixedq2-densefix'
SIDE='/home/funboy/models/ds41/dspark-v41-mtp-2bc89ac'
OUT=ROOT/'reports/DS41-Q2-001/attempt038-dspark-real/draft-config-gate.json'

def main():
    args=EngineArgs(
        model=TARGET, hf_config_path=TARGET_CFG, tokenizer=TARGET_CFG,
        config_format='gguf', load_format='gguf', quantization='gguf', dtype='bfloat16',
        tensor_parallel_size=2, enable_expert_parallel=True,
        distributed_executor_backend='external_launcher',
        max_model_len=4096, block_size=128, max_num_seqs=1, max_num_batched_tokens=1024,
        kv_cache_memory_bytes=1073741824, enable_prefix_caching=False,
        enable_chunked_prefill=True, enforce_eager=True, seed=1,
        speculative_config={
            'method':'dspark', 'model':SIDE, 'num_speculative_tokens':1, 'quantization':'fp8',
            'enable_adaptive_verification':False,
            'draft_tensor_parallel_size':2,
            'draft_load_config':{'load_format':'safetensors','safetensors_load_strategy':'lazy'},
        },
    )
    cfg=args.create_engine_config(usage_context=None)
    sp=cfg.speculative_config; assert sp is not None
    dq=get_draft_quant_config(cfg)
    data={
      'status':'PASS',
      'target':{
        'model':cfg.model_config.model,'config_format':str(cfg.model_config.config_format),
        'load_format':str(cfg.load_config.load_format),'quantization':cfg.model_config.quantization,
        'quant_config_class':type(cfg.quant_config).__name__ if cfg.quant_config else None,
      },
      'draft':{
        'model':sp.draft_model_config.model,'config_format':str(sp.draft_model_config.config_format),
        'load_format':str(sp.draft_load_config.load_format) if sp.draft_load_config else None,
        'quantization':sp.draft_model_config.quantization,
        'quant_config_class':type(dq).__name__ if dq else None,
        'method':sp.method,'K':sp.num_speculative_tokens,'parallel_drafting':sp.parallel_drafting,
        'architecture':sp.draft_model_config.architectures,
        'target_layer_ids':list(sp.draft_model_config.hf_config.dspark_target_layer_ids),
        'num_nextn_predict_layers':sp.draft_model_config.hf_config.num_nextn_predict_layers,
      }
    }
    assert cfg.model_config.config_format == 'gguf'
    assert str(cfg.load_config.load_format) == 'gguf'
    assert cfg.model_config.quantization == 'gguf'
    assert Path(sp.draft_model_config.model)==Path(SIDE)
    assert sp.draft_model_config.config_format != 'gguf'
    assert str(sp.draft_load_config.load_format) == 'safetensors'
    assert sp.draft_model_config.quantization == 'deepseek_v4_fp8'
    assert type(dq).__name__ == 'DeepseekV4FP8Config', type(dq).__name__
    assert type(dq).__name__ != type(cfg.quant_config).__name__
    assert getattr(dq,'weight_block_size',None) == [32,32]
    assert sp.num_speculative_tokens==1 and sp.parallel_drafting
    assert list(sp.draft_model_config.hf_config.dspark_target_layer_ids)==[37,38,39]
    assert sp.draft_model_config.hf_config.num_nextn_predict_layers==3
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))
if __name__=='__main__': main()
