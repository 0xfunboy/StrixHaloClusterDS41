#!/usr/bin/env python3
from __future__ import annotations
import hashlib, importlib.util, json
from pathlib import Path

ROOT=Path('/home/funboy/StrixHaloClusterDS41')
SPEC=importlib.util.spec_from_file_location('side',ROOT/'scripts/serve-ds4-usable-tokenizer.py')
side=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(side)

def h(ids): return hashlib.sha256(','.join(map(str,ids)).encode()).hexdigest()
def check(name,data,count,sha):
    out=side.tokenize_request(data)
    assert out['count']==count,(name,out['count'],count)
    assert h(out['tokens'])==sha,(name,h(out['tokens']),sha)
    return {'id':name,'count':out['count'],'sha256_ids':h(out['tokens']),'resolved':out['resolved_thinking']}

def main():
    pcode=(ROOT/'runtime/ds41/document-profile-002/prompts/code2k-middle-explicit-v2.txt').read_text()
    pdocs=(ROOT/'runtime/ds41/document-profile-002/prompts/docs2k-middle-explicit-v2.txt').read_text()
    ha=(ROOT/'runtime/ds41/document-profile-002/holdouts/document-holdout-a.txt').read_text()
    hb=(ROOT/'runtime/ds41/document-profile-002/holdouts/document-holdout-b.txt').read_text()
    rows=[]
    small=[
      ('user_none',{'messages':[{'role':'user','content':'hello'}],'thinking':False,'reasoning_effort':'none','add_generation_prompt':True},5,'4b252d2966c81e3fa8ed70237aa8639301718d379e574390820777763fd90eba'),
      ('user_low',{'messages':[{'role':'user','content':'hello'}],'thinking':True,'reasoning_effort':'low','add_generation_prompt':True},5,'b73904c7962831ff8d88451ea6f5fafb53b96e4e8c3b05d4421cc65625bbceae'),
      ('system_none',{'messages':[{'role':'system','content':'SYS'},{'role':'user','content':'hello'}],'thinking':False,'reasoning_effort':'none','add_generation_prompt':True},8,'36d8dc067c38e12fecf2ffecb5475442ce33db8ab04e41c2b0a8127571c77a73'),
      ('system_low',{'messages':[{'role':'system','content':'SYS'},{'role':'user','content':'hello'}],'thinking':True,'reasoning_effort':'low','add_generation_prompt':True},8,'9bf94d45d45de563d68cb6bab70b3d9f973c6e40bd44f48287685ef68cd32e65'),
      ('multiturn_none',{'messages':[{'role':'user','content':'hello'},{'role':'assistant','content':'world'},{'role':'user','content':'again'}],'thinking':False,'reasoning_effort':'none','add_generation_prompt':True},11,'326e64fdd9a2cf7f47319020682849c65b8e05bd42e074a00103546a812f41e0'),
      ('multiturn_low',{'messages':[{'role':'user','content':'hello'},{'role':'assistant','content':'world'},{'role':'user','content':'again'}],'thinking':True,'reasoning_effort':'low','add_generation_prompt':True},11,'bcf6924726019939dbb81ef5a59aeaffee2ad3e412fc0fd0d134931d88cd83e6'),
    ]
    # Small hashes are checked below only after recomputing from the known Antirez ID arrays,
    # so the test remains readable and catches renderer/tokenizer drift.
    known={
      'user_none':[0,128803,33310,128804,128822],
      'user_low':[0,128803,33310,128804,128821],
      'system_none':[0,128799,53,20842,128803,33310,128804,128822],
      'system_low':[0,128799,53,20842,128803,33310,128804,128821],
      'multiturn_none':[0,128803,33310,128804,128822,29616,1,128803,41289,128804,128822],
      'multiturn_low':[0,128803,33310,128804,128822,29616,1,128803,41289,128804,128821],
    }
    for name,data,_,_ in small:
        out=side.tokenize_request(data); assert out['tokens']==known[name],(name,out['tokens'],known[name]);rows.append({'id':name,'count':out['count'],'resolved':out['resolved_thinking']})
    big=[
      ('code2k-none',pcode,'none',1629,'f31f1c16b6ec8b002532d9f76591af67ac0e8ea34513261d94a1d896190c9eea'),
      ('code2k-low',pcode,'low',1629,'d1834b8ac1bda51a981b8600d07dcb5d6e5abb2da682ed5f1bb1331539ee848c'),
      ('docs2k-none',pdocs,'none',1306,'324d18a94419db8f1f17db882caf49984d05e8dc72e663f5bd57c5cd42061757'),
      ('docs2k-low',pdocs,'low',1306,'5f16ac8222e02266bd1919db324f93a6a2761246bdb52015e8e1316365c6955e'),
      ('holdout-a-low',ha,'low',1409,'7a6836f8675872dacee29bde8f968f51f2c6ca0959a0a2f84c898c6b2018c8b5'),
      ('holdout-b-low',hb,'low',2039,'ff8d2ad93b30fb51d02897c960bc19655eaeadc7493eb24774875da810f8649a'),
    ]
    for name,text,effort,count,sha in big:
        data={'messages':[{'role':'user','content':text}],'thinking':effort!='none','reasoning_effort':effort,'add_generation_prompt':True}
        rows.append(check(name,data,count,sha))
    # Precedence mirrors JSON field order in ds4_server.c.
    a={'messages':[{'role':'user','content':'x'}],'thinking':False,'reasoning_effort':'low','chat_template_kwargs':{'enable_thinking':True,'reasoning_effort':'low'}}
    assert side.resolve_think_mode(a)[0]=='low'
    b={'messages':[{'role':'user','content':'x'}],'chat_template_kwargs':{'enable_thinking':True,'reasoning_effort':'low'},'thinking':False,'reasoning_effort':'low'}
    assert side.resolve_think_mode(b)[0]=='none'
    low=side.resolve_think_mode({'messages':[{'role':'user','content':'x'}],'thinking':True,'reasoning_effort':'low'})
    assert low==('low',True,None),low
    print(json.dumps({'status':'PASS','rows':rows,'precedence':'PASS','low_mode':{'named':'DS4_THINK_LOW','numeric_level':None}},indent=2))
if __name__=='__main__':main()
