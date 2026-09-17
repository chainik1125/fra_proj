"""Behavior and source-edge opportunity checks for additional task variants."""
import hashlib,json,time
from pathlib import Path
import torch
from transformer_lens import HookedTransformer
from common import LAST,render,metric,summarize,atomic
from variants import LABELS,TASKS,suite
from operators import SAE_SPECS

@torch.inference_mode()
def run(out_dir,commit=None):
    torch.set_num_threads(4);start=time.time();out_dir=Path(out_dir)
    model=HookedTransformer.from_pretrained('gemma-2-9b-it',device='cuda',dtype=torch.float16);model.eval();tok=model.tokenizer
    label_ids=[tok.encode(' '+s,add_special_tokens=False)[0] for s in LABELS];pid=label_ids[1]
    screen=json.loads((out_dir/'screen.json').read_text());prior=next(x for x in screen['tasks'] if x['task']=='tenants_long')
    top64=[(x['layer'],x['head']) for x in prior['head_screen'] if x['key_kind']=='row'][:64]
    groups={'six_sites':[(l,h) for l in SAE_SPECS for h in range(model.cfg.n_heads)],'old_top64':top64}
    result={'done':False,'sources':{n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ['variants.py','variant_screen.py','data.py']},'tasks':[]}
    def save():
        result['seconds']=time.time()-start;atomic(out_dir/'variant_screen.json',result)
        if commit:commit()
    from audit_extra import audit
    result['implementation_audit']=audit(model);save()
    for task in TASKS:
        record={'task':task,'rows':[],'oracle':{k:[] for k in groups}};result['tasks'].append(record)
        rr=suite(task,'calibration')
        for i in range(0,len(rr),2):
            cr,pr=render(tok,rr[i]),render(tok,rr[i+1]);ct=torch.tensor([cr['token_ids']],device='cuda');pt=torch.tensor([pr['token_ids']],device='cuda')
            assert ct.shape==pt.shape and int((ct!=pt).sum())==1
            cl=model.run_with_hooks(ct,fwd_hooks=[LAST])[0,-1];ref=cl.double().log_softmax(-1)
            pl=model.run_with_hooks(pt,fwd_hooks=[LAST])[0,-1];pp=float(pl.float().softmax(-1)[pid])
            record['rows'].append({**pr,'clean_token_ids':cr['token_ids'],'clean_correct':int(LABELS[int(cl[label_ids].argmax())]==pr['expected']),
                'baseline':metric(tok,label_ids,pl,ref,pr,pp)})
            for group,heads in groups.items():
                hh={}
                for l,h in heads:hh.setdefault(l,[]).append(h)
                keys=pr['source_positions'];qstart=max(keys)+1
                def make(heads):
                    def hook(s,hook):
                        for h in heads:s[:,h,qstart:,keys]=-torch.inf
                        return s
                    return hook
                ll=model.run_with_hooks(pt,fwd_hooks=[(f'blocks.{l}.attn.hook_attn_scores',make(heads)) for l,heads in hh.items()]+[LAST])[0,-1]
                record['oracle'][group].append(metric(tok,label_ids,ll,ref,pr,pp))
        record['summary']=summarize([r['baseline'] for r in record['rows']]);record['clean_accuracy']=sum(r['clean_correct'] for r in record['rows'])/len(record['rows'])
        s=record['summary'];record['gate']=record['clean_accuracy']>=.95 and s['controls']['correct']>=.95 and s['joint']['poison_target_p']-s['joint']['clean_target_p']>=.4
        record['oracle_summary']={k:summarize(v) for k,v in record['oracle'].items()}
        print('VARIANT',task,'clean',record['clean_accuracy'],'controls',s['controls']['correct'],'gate',record['gate'],
            'oracle',[(k,round(s['all']['kl'],4),round(s['suppression'],3)) for k,s in record['oracle_summary'].items()],flush=True);save()
    result['done']=True;save();return result
