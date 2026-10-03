#!/usr/bin/env python3
from __future__ import annotations
import hashlib, importlib.util, json, re
from pathlib import Path

ROOT=Path('/home/funboy/StrixHaloClusterDS41')
SRCROOT=ROOT/'reports/DS41-Q2-001/daily-k2-panel/corpora'
OUT=ROOT/'runtime/ds41/long-context-baseline-001'
MANIFEST_SRC=SRCROOT/'manifest.json'
LABELS=('4k','8k','16k','32k','64k')

spec=importlib.util.spec_from_file_location('side',ROOT/'scripts/serve-ds4-usable-tokenizer.py')
side=importlib.util.module_from_spec(spec); spec.loader.exec_module(side)

def sha(b:bytes)->str: return hashlib.sha256(b).hexdigest()

def atomic(path:Path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    q=path.with_suffix(path.suffix+'.tmp')
    q.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
    q.replace(path)

def sections(text:str):
    return re.findall(r'^===== FILE (.+?) SHA256 ([0-9a-f]{64}) =====$',text,re.M)

def facts(text:str):
    out={}
    for key in ('BEGIN','MIDDLE','END'):
        m=re.search(rf'^DISTANT_FACT_{key}=(\d+)\s*$',text,re.M)
        if not m: raise RuntimeError(f'missing fact {key}')
        out[key.lower()]=int(m.group(1))
    return out

def corpus_without_task(text:str)->str:
    i=text.rfind('\nTASK:')
    if i<0: raise RuntimeError('TASK marker missing')
    return text[:i].rstrip()+'\n'

def primary_task(base:str)->str:
    return base+'''\nTASK: Using the corpus above, return exactly one JSON object with keys result, first_file, middle_file, last_file. result must equal (DISTANT_FACT_BEGIN * 2 + DISTANT_FACT_MIDDLE - DISTANT_FACT_END). first_file and last_file must be the basenames of the first and last FILE sections actually present. middle_file must be the basename of the lower central FILE section: number FILE sections from 1 in appearance order; if N is even use section N/2, and if N is odd use section (N+1)/2. No prose.\n'''

def holdout_task(base:str)->str:
    return base+'''\nHOLDOUT TASK: Using the corpus above, return exactly one JSON object with keys result, section_count, first_file, second_file, middle_file, penultimate_file, last_file. result must equal (DISTANT_FACT_BEGIN + DISTANT_FACT_MIDDLE + DISTANT_FACT_END). section_count is the number of FILE sections. first_file, second_file, penultimate_file and last_file are the corresponding basenames by appearance order. middle_file is the lower central FILE section: if N is even use section N/2, otherwise section (N+1)/2. No prose.\n'''

def low_ids(prompt:str):
    data={
      'messages':[{'role':'user','content':prompt}],
      'chat_template_kwargs':{'enable_thinking':True,'reasoning_effort':'low'},
      'add_generation_prompt':True,
    }
    o=side.tokenize_request(data)
    return o['tokens']

def expected_primary(text:str):
    ss=sections(text); ff=facts(text); n=len(ss)
    if n<3: raise RuntimeError('too few sections')
    names=[Path(x[0]).name for x in ss]
    mid=(n-1)//2
    return {
      'result':ff['begin']*2+ff['middle']-ff['end'],
      'first_file':names[0],
      'middle_file':names[mid],
      'last_file':names[-1],
    }

def expected_holdout(text:str):
    ss=sections(text); ff=facts(text); n=len(ss)
    names=[Path(x[0]).name for x in ss]; mid=(n-1)//2
    return {
      'result':ff['begin']+ff['middle']+ff['end'],
      'section_count':n,
      'first_file':names[0],
      'second_file':names[1],
      'middle_file':names[mid],
      'penultimate_file':names[-2],
      'last_file':names[-1],
    }

def main():
    src=json.loads(MANIFEST_SRC.read_text())
    entries={(x['kind'],x['label']):x for x in src['corpora']}
    rows=[]; holds=[]
    for label in LABELS:
      for kind in ('code','docs'):
        e=entries[(kind,label)]
        path=ROOT/e['file']
        raw=path.read_bytes()
        got=sha(raw)
        if got!=e['corpus_sha256']:
            raise RuntimeError(f'corpus sha mismatch {kind} {label}: {got}')
        text=raw.decode()
        base=corpus_without_task(text)
        p=primary_task(base)
        exp=expected_primary(text)
        # Existing corpus manifest is an additional frozen cross-check.
        if exp!=e['expected']:
            raise RuntimeError(f'expected mismatch {kind} {label}: derived={exp} manifest={e["expected"]}')
        ids=low_ids(p)
        if len(ids)+2048>65536:
            raise RuntimeError(f'{kind} {label}: rendered prompt {len(ids)} + 2048 exceeds 65536')
        dst=OUT/'prompts'/f'{kind}-{label}.txt'
        dst.parent.mkdir(parents=True,exist_ok=True); dst.write_text(p)
        rows.append({
          'id':f'{kind}-{label}','kind':kind,'label':label,
          'source_file':e['file'],'source_sha256':e['corpus_sha256'],
          'source_actual_prompt_tokens_nothink':e.get('actual_prompt_tokens'),
          'source_files':e.get('sources',[]),
          'prompt_file':str(dst.relative_to(ROOT)),
          'prompt_sha256':sha(p.encode()),
          'rendered_low_tokens':len(ids),
          'rendered_low_ids_sha256':sha(','.join(map(str,ids)).encode()),
          'output_cap':2048,'expected':exp,
          'section_count':len(sections(text)),
          'facts':facts(text),
        })
        hp=holdout_task(base)
        hexp=expected_holdout(text)
        hids=low_ids(hp)
        if len(hids)+2048>65536:
            raise RuntimeError(f'holdout {kind} {label}: rendered prompt {len(hids)} + 2048 exceeds 65536')
        hdst=OUT/'holdouts'/f'{kind}-{label}-holdout.txt'
        hdst.parent.mkdir(parents=True,exist_ok=True); hdst.write_text(hp)
        holds.append({
          'id':f'{kind}-{label}-holdout','kind':kind,'label':label,
          'source_file':e['file'],'source_sha256':e['corpus_sha256'],
          'prompt_file':str(hdst.relative_to(ROOT)),
          'prompt_sha256':sha(hp.encode()),
          'rendered_low_tokens':len(hids),
          'rendered_low_ids_sha256':sha(','.join(map(str,hids)).encode()),
          'output_cap':2048,'expected':hexp,
          'section_count':len(sections(text)),
          'facts':facts(text),
        })
    manifest={
      'schema':'ds4-long-context-baseline-001-p3-frozen-v1',
      'tokenizer':'DeepSeek V4.1 pinned sidecar renderer',
      'profile':{'thinking':True,'reasoning_effort':'low','named_mode':'DS4_THINK_LOW','numeric_budget':None,'temperature':0,'seed':1,'output_cap':2048,'context_window':65536},
      'originals':rows,'holdouts':holds,
      'notes':['All source corpora are existing real repository code/docs; no random filler or repetition was added.','Expected values are stored only in this validator manifest; prompts contain derivation rules, not answers.','Holdouts are frozen before inference and ask a different positional/fact relation over the same real corpus for the selected level.']
    }
    atomic(OUT/'p3-frozen-manifest.json',manifest)
    print(json.dumps({'status':'PASS','originals':[(r['id'],r['rendered_low_tokens']) for r in rows],'holdouts':[(r['id'],r['rendered_low_tokens']) for r in holds]},indent=2))

if __name__=='__main__': main()
