"""Diagnostic source-position ablations; never counted as a content-gated win."""
import json,time
from pathlib import Path
import torch
from transformer_lens import HookedTransformer
from common import LAST,render,metric,summarize,atomic
from data import LABELS,suite
from operators import SAE_SPECS

@torch.inference_mode()
def run(out_dir,commit=None):
    torch.set_num_threads(4);start=time.time();out_dir=Path(out_dir)
    screen=json.loads((out_dir/'screen.json').read_text())
    model=HookedTransformer.from_pretrained('gemma-2-9b-it',device='cuda',dtype=torch.float16);model.eval();tok=model.tokenizer
    label_ids=[tok.encode(' '+s,add_special_tokens=False)[0] for s in LABELS];pid=label_ids[1]
    result={'done':False,'purpose':'source-position diagnostic, not FRA','tasks':[]}
    def save():
        result['seconds']=time.time()-start;atomic(out_dir/'oracle.json',result)
        if commit:commit()
    for task in ['routing','tenants_long']:
        prior=next(t for t in screen['tasks'] if t['task']==task)
        heads=[(r['layer'],r['head']) for r in prior['head_screen'] if r['key_kind']=='row']
        groups={f'top_{k}':heads[:k] for k in [4,16,64]}
        groups.update({f'layer_{l}_all_heads':[(l,h) for h in range(model.cfg.n_heads)] for l in SAE_SPECS})
        groups['six_sae_sites']=[(l,h) for l in SAE_SPECS for h in range(model.cfg.n_heads)]
        groups['all_layers']=[(l,h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]
        configs=[{'group':g,'key_kind':kind,'queries':q,'heads':hh} for g,hh in groups.items() for kind in ['payload','row'] for q in ['all_later','answer_only']]
        points=[{'config':cfg,'rows':[]} for cfg in configs];record={'task':task,'baseline':[],'points':points};result['tasks'].append(record)
        rr=suite(task,'calibration')
        for ri in range(0,len(rr),2):
            cr,pr=render(tok,rr[ri]),render(tok,rr[ri+1]);ct=torch.tensor([cr['token_ids']],device='cuda');pt=torch.tensor([pr['token_ids']],device='cuda')
            ref=model.run_with_hooks(ct,fwd_hooks=[LAST])[0,-1].double().log_softmax(-1)
            logits=model.run_with_hooks(pt,fwd_hooks=[LAST])[0,-1];pp=float(logits.float().softmax(-1)[pid])
            record['baseline'].append(metric(tok,label_ids,logits,ref,pr,pp))
            for point in points:
                cfg=point['config'];by_layer={}
                for l,h in cfg['heads']:by_layer.setdefault(l,[]).append(h)
                keys=[pr['k_positions']['payload']] if cfg['key_kind']=='payload' else pr['source_positions']
                qstart=max(keys)+1 if cfg['queries']=='all_later' else pt.shape[1]-1
                def make(hh):
                    def hook(scores,hook):
                        for h in hh:scores[:,h,qstart:,keys]=-torch.inf
                        return scores
                    return hook
                hooks=[(f'blocks.{l}.attn.hook_attn_scores',make(hh)) for l,hh in by_layer.items()]+[LAST]
                ll=model.run_with_hooks(pt,fwd_hooks=hooks)[0,-1]
                point['rows'].append(metric(tok,label_ids,ll,ref,pr,pp))
            if ri%16==0:print('ORACLE',task,ri//2,'/',len(rr)//2,'seconds',round(time.time()-start),flush=True);save()
        for p in points:p['summary']=summarize(p['rows'])
        record['baseline_summary']=summarize(record['baseline']);save()
        print('BEST ORACLE',task,[(p['config']['group'],p['config']['key_kind'],p['config']['queries'],round(p['summary']['all']['kl'],4),round(p['summary']['suppression'],3)) for p in sorted(points,key=lambda p:p['summary']['all']['kl'])[:10]],flush=True)
    result['done']=True;save();return result
