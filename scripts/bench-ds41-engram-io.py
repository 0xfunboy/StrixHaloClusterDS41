#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,resource,time,sys
from pathlib import Path
import numpy as np
from _ds41_artifact import MODEL_DIR
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime.ds41.affine_safetensors import SafeTensorMMap,AffineRowLRU

def proc_io():
    d={}
    for line in Path('/proc/self/io').read_text().splitlines():
        k,v=line.split(':',1); d[k]=int(v.strip())
    return d

def rss():
    out={}
    for line in Path('/proc/self/smaps_rollup').read_text().splitlines():
        if line.startswith(('Rss:','Pss:','Private_Clean:','Private_Dirty:')):
            k,v=line.split(':',1); out[k]=int(v.split()[0])*1024
    return out

def vmstat():
    want={'pswpin','pswpout','pgmajfault'}; out={}
    for line in Path('/proc/vmstat').read_text().splitlines():
        k,v=line.split()
        if k in want: out[k]=int(v)
    return out

def snap():
    r=resource.getrusage(resource.RUSAGE_SELF)
    return {'majflt':r.ru_majflt,'minflt':r.ru_minflt,'io':proc_io(),'rss':rss(),'vm':vmstat()}

def delta(a,b):
    return {'majflt':b['majflt']-a['majflt'],'minflt':b['minflt']-a['minflt'],
            'read_bytes':b['io'].get('read_bytes',0)-a['io'].get('read_bytes',0),
            'rss_before':a['rss'],'rss_after':b['rss'],
            'pswpin':b['vm'].get('pswpin',0)-a['vm'].get('pswpin',0),
            'pswpout':b['vm'].get('pswpout',0)-a['vm'].get('pswpout',0),
            'system_pgmajfault':b['vm'].get('pgmajfault',0)-a['vm'].get('pgmajfault',0)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--layer',type=int,choices=(1,14),required=True); ap.add_argument('--rank',type=int,choices=(0,1),default=0); ap.add_argument('--tokens',type=int,default=1024); ap.add_argument('--workers',type=int,choices=range(1,9),default=1); a=ap.parse_args()
    name={1:'model-00047-of-00048.safetensors',14:'model-00048-of-00048.safetensors'}[a.layer]
    p=Path(f'/home/funboy/models/ds41/engram2-tp2/rank{a.rank}')/name
    config=json.load(open(MODEL_DIR/'config.json'))['text_config']
    spec=importlib.util.spec_from_file_location('part',Path(__file__).with_name('partition-ds41-engram2.py')); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    all_sizes=mod.layouts(config)[a.layer]; sizes=all_sizes[a.rank*12:(a.rank+1)*12]; offsets=np.cumsum([0,*sizes[:-1]],dtype=np.int64)
    # rank0 compact sidecar local row space starts at zero; generate valid row id per owned head/token.
    rng=np.random.default_rng(41001+a.layer)
    ids=np.empty((a.tokens,12),dtype=np.int64)
    for h,(start,size) in enumerate(zip(offsets,sizes,strict=True)):
        ids[:,h]=start+rng.integers(0,int(size),size=a.tokens,dtype=np.int64)
    os.environ['DS41_ENGRAM_READ_WORKERS']=str(a.workers); os.environ.setdefault('DS41_ENGRAM_PARALLEL_MIN_ROWS','256'); src=SafeTensorMMap(p); cache=AffineRowLRU(src,f'layers.{a.layer}.engram.embed',max_rows=max(65536,a.tokens*12))
    advise='unsupported'
    if hasattr(os,'posix_fadvise') and hasattr(os,'POSIX_FADV_DONTNEED'):
        try: os.posix_fadvise(src._fd,0,0,os.POSIX_FADV_DONTNEED); advise='DONTNEED'
        except OSError as e: advise=f'error:{e.errno}'
    b=snap(); t=time.perf_counter(); cold=cache.lookup(ids); cold_s=time.perf_counter()-t; c=snap()
    t=time.perf_counter(); warm=cache.lookup(ids); warm_s=time.perf_counter()-t; w=snap()
    np.testing.assert_array_equal(cold,warm)
    out={'layer':a.layer,'rank':a.rank,'tokens':a.tokens,'workers':a.workers,'heads_per_rank':12,'rows_requested':int(ids.size),'unique_rows_read':cache.rows_read,
         'fadvise':advise,'cold_ms':cold_s*1000,'warm_ms':warm_s*1000,'cold':delta(b,c),'warm':delta(c,w),
         'cache_hits':cache.hits,'cache_misses':cache.misses,'output_sha256':hashlib.sha256(np.ascontiguousarray(cold).tobytes()).hexdigest(),'parallel_batches':cache.parallel_batches,'read_wall_ms':cache.read_wall_ns/1e6,'lookup_wall_ms':cache.lookup_wall_ns/1e6,'mapped_file_bytes':p.stat().st_size,'output_bytes':cold.nbytes}
    print(json.dumps(out,indent=2)); src.close()
if __name__=='__main__': main()
