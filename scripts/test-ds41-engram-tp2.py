#!/usr/bin/env python3
"""Real-row TP2 equivalence test for the verified DS41 Engram2 sidecars."""
from __future__ import annotations
import importlib.util,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime.ds41.affine_safetensors import SafeTensorMMap

ROOT=Path('/home/funboy/models/ds41')
SRC=ROOT/'engram2-source'; TP=ROOT/'engram2-tp2'
CFG=Path('/home/funboy/StrixHaloClusterDS41/reports/DS41-Q2-001/sources/vontra-config.json')
LAYER_FILE={1:'model-00047-of-00048.safetensors',14:'model-00048-of-00048.safetensors'}
spec=importlib.util.spec_from_file_location('partition',Path(__file__).with_name('partition-ds41-engram2.py')); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
cfg=json.load(open(CFG))['text_config']; layouts=mod.layouts(cfg); receipt=json.load(open(TP/'partition-receipt.json'))
results=[]
for layer in cfg['engram_layer_ids']:
    name=LAYER_FILE[layer]; base=f'layers.{layer}.engram.embed'
    source=SafeTensorMMap(SRC/name); parts=[SafeTensorMMap(TP/f'rank{r}'/name) for r in range(2)]
    try:
        sizes=layouts[layer]; offsets=np.cumsum([0,*sizes[:-1]],dtype=np.int64)
        assert len(sizes)==24
        # One real row from every hash column, preserving global head order.
        global_ids=np.array([int(o)+min(7,int(s)-1) for o,s in zip(offsets,sizes,strict=True)],dtype=np.int64)
        full=source.affine2_rows(base,global_ids)
        reconstructed=[]
        for rank in range(2):
            rr=receipt['layers'][str(layer)][str(rank)]; start=int(rr['vocab_start']); end=int(rr['vocab_end'])
            expected_start=int(sum(sizes[:rank*12])); expected_end=int(sum(sizes[:min((rank+1)*12,24)]))
            assert (start,end)==(expected_start,expected_end)
            assert parts[rank].metadata['ds41_vocab_start']==str(start)
            assert parts[rank].metadata['ds41_vocab_end']==str(end)
            ids=global_ids[rank*12:(rank+1)*12]
            got=parts[rank].affine2_rows(base,ids-start)
            direct=source.affine2_rows(base,ids)
            np.testing.assert_array_equal(got,direct)
            reconstructed.append(got)
            # q/k are replicated; WKV is replicated and checked on real rows.
            for key in ('q_weight','k_weight'):
                np.testing.assert_array_equal(parts[rank].bf16(f'layers.{layer}.engram.{key}'), source.bf16(f'layers.{layer}.engram.{key}'))
            wbase=f'layers.{layer}.engram.wkv'; wrows=np.array([0,17,source.entries[wbase+'.weight'].shape[0]-1])
            np.testing.assert_array_equal(parts[rank].affine2_rows(wbase,wrows),source.affine2_rows(wbase,wrows))
        recomposed=np.concatenate(reconstructed,axis=0)
        np.testing.assert_array_equal(recomposed,full)
        results.append({'layer':layer,'heads':24,'rank0_rows':int(receipt['layers'][str(layer)]['0']['vocab_end']),'rank1_rows':int(receipt['layers'][str(layer)]['1']['vocab_end'])-int(receipt['layers'][str(layer)]['1']['vocab_start']),'checked_global_ids':global_ids.tolist(),'shape':list(full.shape),'status':'PASS'})
    finally:
        source.close(); [p.close() for p in parts]
print(json.dumps({'status':'PASS','layers':results},indent=2))
