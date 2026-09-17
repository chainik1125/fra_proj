import hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
from data import LABELS
ROOT=Path(__file__).resolve().parent
LAST=('ln_final.hook_normalized',lambda x,hook:x[:,-1:])


def atomic(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,allow_nan=False));tmp.replace(path)


def render(tok,row):
    text=tok.apply_chat_template([{'role':'user','content':row['text']+'\nReply with only the queue name.'}],tokenize=False,add_generation_prompt=True)+'Queue:'
    tokenized=tok(text,add_special_tokens=False,return_offsets_mapping=True);off=tokenized['offset_mapping'];prefix=text.index(row['text'])
    def lastpos(char):return max(i for i,(a,b) in enumerate(off) if b>a and b<=prefix+char)
    def span(start,end):return [i for i,(a,b) in enumerate(off) if b>a and b>prefix+start and a<prefix+end]
    return {**row,'rendered_text':text,'token_ids':tokenized['input_ids'],
        'q_positions':[lastpos(row['problem_end']),len(off)-1],
        'k_positions':{'payload':lastpos(row['source_end']),'cue':lastpos(row['source_label_start']-3),'asset':lastpos(row['asset_end'])},
        'source_positions':span(row['source_start'],row['source_end'])}


def summarize(rows):
    out={}
    for group in ['all','joint','controls','shared_conjunction']:
        rr=[x for x in rows if group=='all' or (group=='joint' and x['joint']) or
            (group=='controls' and not x['joint']) or (group=='shared_conjunction' and x['corner']=='111')]
        if not rr:continue
        out[group]={k:float(np.mean([x[k] for x in rr])) for k in ['kl','target_p','clean_target_p','poison_target_p','correct','top_correct','label_mass']}
        out[group]['n']=len(rr)
    if 'joint' in out:
        j=out['joint'];out['suppression']=1-j['target_p']/max(j['poison_target_p'],1e-12)
        out['excess_repair']=(j['poison_target_p']-j['target_p'])/max(j['poison_target_p']-j['clean_target_p'],1e-12)
    return out


def metric(tok,label_ids,logits,ref,row,poison_p):
    if not bool(torch.isfinite(logits).all()):return {'case_id':row['case_id'],'invalid':True}
    lp=logits.double().log_softmax(-1);p=lp.exp();kl=float((ref.exp()*(ref-lp)).sum());assert kl>-1e-8
    pid=label_ids[LABELS.index('Print')];pred=LABELS[int(logits[label_ids].argmax())];top=tok.decode([int(p.argmax())])
    return {'case_id':row['case_id'],'corner':row['corner'],'joint':row['joint'],'kl':max(0.,kl),
        'target_p':float(p[pid]),'clean_target_p':float(ref[pid].exp()),'poison_target_p':poison_p,
        'correct':int(pred==row['expected']),'top_correct':int(top.strip()==row['expected']),
        'label_mass':float(p[label_ids].sum()),'predicted':pred,'top_token':top}
