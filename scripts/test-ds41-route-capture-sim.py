#!/usr/bin/env python3
from __future__ import annotations
import os, tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from vllm_gguf_plugin.quantization.fused_moe import GGUFMoEMethod

class Batch:
    def __init__(self, plen=1588, before=0, scheduled=1023, req='req-real'):
        self.req_ids=[req]
        self.prefill_len_np=[plen]
        self.num_computed_prefill_tokens_np=[before]
        self.num_scheduled_tokens=[scheduled]
        self.is_prefilling_np=[before < plen]

def main():
    with tempfile.TemporaryDirectory(prefix='ds41-routecap-') as td:
        os.environ['DS41_ROUTE_CAPTURE_DIR']=td
        os.environ['DS41_ROUTE_CAPTURE_CALLS']='0'
        os.environ['DS41_ROUTE_CAPTURE_PROMPT_TOKENS']='1588'
        original = GGUFMoEMethod.apply
        sentinel=[]
        def fake_apply(method_self, layer, x, weights, ids, shared_experts, shared_input):
            y=x.clone(); y.add_(1); sentinel.append((x.clone(),ids.clone(),weights.clone())); return y
        GGUFMoEMethod.apply=fake_apply
        try:
            from runtime.ds41.routed_fixture_capture import RoutedFixtureCapture
            cap=RoutedFixtureCapture()
            layer=SimpleNamespace(
                w13_weight_type=SimpleNamespace(weight_type=16),
                w2_weight_type=SimpleNamespace(weight_type=10),
                w13_weight=torch.empty((192,2,2),dtype=torch.uint8),
                w2_weight=torch.empty((192,2,2),dtype=torch.uint8),
                expert_map=torch.arange(384,dtype=torch.int32),
            )
            method=SimpleNamespace(moe=SimpleNamespace(swiglu_limit=10.0))
            # Dummy/startup-style padded ids: wrapper installed but capture inactive.
            x=torch.zeros((65,5120),dtype=torch.bfloat16); ids=torch.full((65,6),-1,dtype=torch.int32); w=torch.zeros((65,6))
            out=GGUFMoEMethod.apply(method,layer,x,w,ids,None,None)
            assert torch.equal(out,x+1) and not list(Path(td).glob('*.pt'))
            # Wrong prompt never arms.
            cap.maybe_arm(Batch(plen=64)); assert not cap.active
            # Real request arms. Preserve input bytes and raw IDs exactly.
            cap.maybe_arm(Batch()); assert cap.active
            ids2=(torch.arange(65*6,dtype=torch.int32).reshape(65,6)%384).contiguous(); w2=torch.rand((65,6)); x2=torch.randn((65,5120),dtype=torch.bfloat16)
            x0=x2.clone(); i0=ids2.clone(); w0=w2.clone()
            out2=GGUFMoEMethod.apply(method,layer,x2,w2,ids2,None,None)
            assert torch.equal(out2,x0+1); assert torch.equal(x2,x0) and torch.equal(ids2,i0) and torch.equal(w2,w0)
            pts=list(Path(td).glob('*.pt')); assert len(pts)==1
            row=torch.load(pts[0],weights_only=False); assert torch.equal(row['topk_ids'],i0); assert torch.equal(row['x'],x0)
            # Empty/invalid input cannot increment target-call counter or crash.
            before=cap.prefill_call
            GGUFMoEMethod.apply(method,layer,torch.empty((0,5120),dtype=torch.bfloat16),torch.empty((0,6)),torch.empty((0,6),dtype=torch.int32),None,None)
            assert cap.prefill_call==before
            # Finish the selected request; later calls remain inert.
            cap.maybe_finish(Batch(before=1023,scheduled=565)); assert cap.done and not cap.active
            GGUFMoEMethod.apply(method,layer,x2,w2,ids2,None,None)
            assert len(list(Path(td).glob('*.pt')))==1
            assert (Path(td)/'route-fixture-rank0-status.txt').read_text().strip()=='COMPLETE'
            print('ROUTE_CAPTURE_SIM=PASS')
        finally:
            GGUFMoEMethod.apply=original

if __name__=='__main__': main()
