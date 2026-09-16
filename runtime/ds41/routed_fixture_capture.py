"""One-shot routed-MoE fixture capture for DS41 prefill candidate work.

Opt-in and diagnostic only. Captures the actual target activation/router outputs
for selected routed-MoE calls. No model weights are copied; component tests read
the immutable GGUF bytes independently.
"""
from __future__ import annotations
import functools,json,os
from pathlib import Path

class RoutedFixtureCapture:
    def __init__(self):
        root=os.environ.get('DS41_ROUTE_CAPTURE_DIR','').strip()
        self.enabled=bool(root); self.root=Path(root) if root else None
        self.rank=int(os.environ.get('RANK',os.environ.get('LOCAL_RANK','0')))
        self.selected={int(x) for x in os.environ.get('DS41_ROUTE_CAPTURE_CALLS','0,20,40,60').split(',') if x.strip()}
        self.prefill_call=0; self.saved=[]; self._installed=False
    def install(self):
        if not self.enabled or self._installed: return
        import torch
        from vllm_gguf_plugin.quantization.fused_moe import GGUFMoEMethod
        original=GGUFMoEMethod.apply; cap=self
        @functools.wraps(original)
        def wrapped(method_self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
            out=original(method_self,layer,x,topk_weights,topk_ids,shared_experts,shared_experts_input)
            if x.ndim==2 and x.shape[0]>64:
                idx=cap.prefill_call; cap.prefill_call+=1
                if idx in cap.selected:
                    cap._save(idx,layer,x,topk_weights,topk_ids,method_self)
            return out
        GGUFMoEMethod.apply=wrapped
        self._installed=True
    def _save(self,idx,layer,x,w,ids,method_self):
        import torch
        assert self.root is not None
        self.root.mkdir(parents=True,exist_ok=True)
        # call0..39 = chunk0 target layers0..39; call40..79 = chunk1.
        row={'schema':'ds41-routed-prefill-fixture-v1','rank':self.rank,'call_index':idx,
             'chunk_index':idx//40,'layer_index':idx%40,'tokens':int(x.shape[0]),
             'hidden':int(x.shape[1]),'top_k':int(ids.shape[1]),
             'w13_quant':int(layer.w13_weight_type.weight_type),'w2_quant':int(layer.w2_weight_type.weight_type),
             'w13_shape':list(layer.w13_weight.shape),'w2_shape':list(layer.w2_weight.shape),
             'swiglu_limit':float(method_self.moe.swiglu_limit or 0.0),
             'x':x.detach().to('cpu'),'topk_weights':w.detach().to('cpu'),'topk_ids':ids.detach().to('cpu'),
             'expert_map':None if layer.expert_map is None else layer.expert_map.detach().to('cpu')}
        path=self.root/f'route-fixture-rank{self.rank}-call{idx:02d}.pt'; torch.save(row,path)
        ids64=row['topk_ids'].to(torch.int64); hist=torch.bincount(ids64.flatten(),minlength=384)
        local=None
        if row['expert_map'] is not None:
            em=row['expert_map'].to(torch.int64); local_mask=em[ids64]>=0; local=int(local_mask.sum())
        summary={'rank':self.rank,'call_index':idx,'chunk_index':idx//40,'layer_index':idx%40,'tokens':row['tokens'],
                 'routes':int(ids64.numel()),'local_routes':local,'active_experts':int((hist>0).sum()),
                 'max_routes_per_expert':int(hist.max()),'mean_routes_per_active_expert':float(hist[hist>0].float().mean()),
                 'pt':str(path)}
        (self.root/f'route-fixture-rank{self.rank}-call{idx:02d}.json').write_text(json.dumps(summary,indent=2)+'\n')
        self.saved.append(idx)

def create_routed_fixture_capture():
    c=RoutedFixtureCapture(); c.install(); return c
