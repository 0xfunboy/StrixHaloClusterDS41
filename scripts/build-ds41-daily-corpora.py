#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
from vllm.tokenizers.deepseek_v41 import DeepseekV41Tokenizer

ROOT=Path(__file__).resolve().parents[1]
TARGETS={'2k':1900,'4k':3900,'8k':7900,'16k':15800,'32k':31800,'64k':63800}
FACTS={'2k':(17,23,5),'4k':(19,31,7),'8k':(23,37,11),'16k':(29,41,13),'32k':(31,47,17),'64k':(37,53,19)}
MODEL='/home/funboy/models/gguf/deepseek-v4.1-flash-mixedq2-densefix'

def tracked():
    return [x.decode() for x in subprocess.check_output(['git','-C',str(ROOT),'ls-files','-z']).split(b'\0') if x]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def render(tok, content):
    return tok.apply_chat_template([{'role':'user','content':content}], tokenize=True, add_generation_prompt=True, reasoning_effort='none')
def candidates(kind):
    out=[]
    for rel in tracked():
        p=ROOT/rel
        if not p.is_file() or p.stat().st_size==0 or p.stat().st_size>180_000: continue
        if any(x in rel for x in ('/private/','golden.patch','verify.cpp','runtime/ds41/results/','reports/','.vendor/')): continue
        if kind=='code':
            good=(rel.startswith('internal/app/') and rel.endswith('.go')) or (rel.startswith('runtime/ds41/') and rel.endswith('.py')) or (rel.startswith('web/') and rel.endswith(('.js','.mjs'))) or (rel.startswith('scripts/') and rel.endswith('.py'))
        else:
            good=rel.endswith('.md') or rel in ('runtime/model-catalog.json','runtime/manifest.json','runtime/ds41/artifact.json','config.ds41-k2-serving.json')
        if not good: continue
        try: text=p.read_text(errors='strict')
        except: continue
        out.append((p.stat().st_size,rel,text))
    # deterministic, mix medium files before huge ones
    return sorted(out,key=lambda x:(x[0],x[1]))
def assemble(tok, kind, label, target):
    a,b,c=FACTS[label]
    pool=candidates(kind)
    chunks=[f'DS41 DAILY CORPUS {kind.upper()} {label}\nDISTANT_FACT_BEGIN={a}\n']
    src=[]; mid_added=False; end_added=False
    for size,rel,text in pool:
        block=f'\n===== FILE {rel} SHA256 {hashlib.sha256(text.encode()).hexdigest()} =====\n{text}\n===== END FILE {rel} =====\n'
        trial=''.join(chunks)+block
        task=f'''\nDISTANT_FACT_END={c}\n\nTASK: Using the corpus above, return exactly one JSON object with keys result, first_file, middle_file, last_file. result must equal (DISTANT_FACT_BEGIN * 2 + DISTANT_FACT_MIDDLE - DISTANT_FACT_END). first_file/middle_file/last_file must be the basenames of the first, median-by-order, and last FILE sections actually present. No prose.\n'''
        nt=len(render(tok,trial+task))
        if nt > target and src: break
        chunks.append(block); src.append({'path':rel,'sha256':hashlib.sha256(text.encode()).hexdigest(),'bytes':len(text.encode())})
        cur=len(render(tok,''.join(chunks)))
        if not mid_added and cur >= target*0.48:
            chunks.append(f'\nDISTANT_FACT_MIDDLE={b}\n'); mid_added=True
    if not mid_added: chunks.append(f'\nDISTANT_FACT_MIDDLE={b}\n'); mid_added=True
    chunks.append(f'\nDISTANT_FACT_END={c}\n')
    names=[Path(x['path']).name for x in src]
    first=names[0]; middle=names[(len(names)-1)//2]; last=names[-1]
    task=f'''\nTASK: Using the corpus above, return exactly one JSON object with keys result, first_file, middle_file, last_file. result must equal (DISTANT_FACT_BEGIN * 2 + DISTANT_FACT_MIDDLE - DISTANT_FACT_END). first_file/middle_file/last_file must be the basenames of the first, median-by-order, and last FILE sections actually present. No prose.\n'''
    content=''.join(chunks)+task
    n=len(render(tok,content))
    return content,{'kind':kind,'label':label,'target_prompt_tokens':target,'actual_prompt_tokens':n,'facts':{'begin':a,'middle':b,'end':c},'expected':{'result':a*2+b-c,'first_file':first,'middle_file':middle,'last_file':last},'sources':src}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); a=ap.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    tok=DeepseekV41Tokenizer.from_pretrained(MODEL)
    manifest={'schema':'ds41-daily-corpora-v1','tokenizer':'deepseek_v41','reasoning_effort':'none','corpora':[]}
    for kind in ('code','docs'):
        for label,target in TARGETS.items():
            content,meta=assemble(tok,kind,label,target)
            path=out/f'{kind}-{label}.txt'; path.write_text(content)
            meta['file']=str(path); meta['corpus_sha256']=sha(path); manifest['corpora'].append(meta)
            print(kind,label,meta['actual_prompt_tokens'],len(meta['sources']),meta['expected'])
    mp=out/'manifest.json'; mp.write_text(json.dumps(manifest,indent=2)+'\n'); print('MANIFEST',sha(mp),mp)
if __name__=='__main__': main()
