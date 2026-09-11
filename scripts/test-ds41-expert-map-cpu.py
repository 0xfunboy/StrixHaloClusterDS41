#!/usr/bin/env python3
"""Verify scalar EP weight-loading lookups never synchronize the device map."""
from __future__ import annotations
import time
import torch
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager

def make(rank: int) -> ExpertMapManager:
    pc=FusedMoEParallelConfig(
        tp_size=1,pcp_size=1,dp_size=1,ep_size=2,
        tp_rank=0,pcp_rank=0,dp_rank=0,ep_rank=rank,sp_size=1,
        use_ep=True,all2all_backend='allgather_reducescatter',enable_eplb=False,
    )
    with torch.device('cuda'):
        return ExpertMapManager(
            max_num_batched_tokens=1024,top_k=6,global_num_experts=384,
            num_redundant_experts=0,num_expert_group=None,
            moe_parallel_config=pc,placement_strategy='linear',enable_eplb=False,
            rocm_aiter_enabled=False,
        )

def main() -> None:
    results=[]
    for rank in (0,1):
        m=make(rank)
        assert m.expert_map is not None and m.expert_map.device.type == 'cuda'
        assert m._expert_map_cpu is not None and m._expert_map_cpu.device.type == 'cpu'
        expected=(-1,0,191) if rank else (0,191,-1)
        actual=(m.map_global_to_local(0 if rank==0 else 191),
                m.map_global_to_local(192 if rank else 191),
                m.map_global_to_local(383 if rank else 192))
        assert actual==expected,(rank,actual,expected)
        ids=list(range(384))*120
        torch.cuda.synchronize(); t0=time.perf_counter(); checksum=0
        for expert_id in ids: checksum += m.map_global_to_local(expert_id)
        elapsed=time.perf_counter()-t0
        assert elapsed < 2.0, elapsed
        results.append({'rank':rank,'lookups':len(ids),'seconds':elapsed,'checksum':checksum})
    print({'status':'PASS','results':results})
if __name__=='__main__': main()
