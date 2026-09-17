"""Two rounds of finer SAE strength search, selected only on tuning data.

The coarse grid is retained. For each repair threshold, take the 12 best distinct
(layer, feature, intervention, scope) choices, then evaluate 15 internal strengths
between the adjacent strengths around each feature's best eligible point. If the
best point is at a grid endpoint, extend that side by the last grid spacing.
"""
import gc,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from transformer_lens import HookedTransformer
from common import LAST,render,metric,summarize,atomic
from data import LABELS,suite as base_suite
from variants import TASKS,suite as variant_suite
from scopes import mask
from operators import Operators,SAE_SPECS
from confirm import edited
from search import select
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'reference'))
from sae_lens_wrapper import GemmaScopeSAE


def featurekey(c):return (c['layer'],c['feature'],c.get('intervention','activation'),c.get('scope','all'))
def key(c):return (*featurekey(c),c['strength'])


def proposals(points):
    valid=[p for p in points if p['valid'] and p['config']['method']=='sae'];chosen={}
    for threshold in [None,0.,.5,.9]:
        pp=[p for p in valid if threshold is None or (p['summary']['controls']['correct']>=.95 and p['summary']['suppression']>=threshold)]
        features=set()
        for p in sorted(pp,key=lambda p:p['summary']['all']['kl']):
            fk=featurekey(p['config'])
            if fk in features:continue
            features.add(fk);chosen[key(p['config'])]=p['config']
            if len(features)==12:break
    configs={};evaluated={key(p['config']) for p in points}
    for c in chosen.values():
        cc=sorted(set(p['config']['strength'] for p in points if featurekey(p['config'])==featurekey(c)))
        pos=cc.index(c['strength']);assert len(cc)>1
        left=cc[pos-1] if pos>0 else cc[0]-(cc[1]-cc[0])
        right=cc[pos+1] if pos<len(cc)-1 else cc[-1]+(cc[-1]-cc[-2])
        # Reevaluate endpoints only if this is an outward extension.
        for strength in np.linspace(left,right,17):
            q={**c,'strength':float(strength)}
            if key(q) not in evaluated:configs[key(q)]=q
    return list(configs.values())


@torch.inference_mode()
def run(out_dir,commit=None,task='tenants_long'):
    torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    start=time.time();out_dir=Path(out_dir);dest=out_dir/f'polish_{task}.json'
    names=[f'search_{task}',f'baseline_extra_{task}'];sources=[json.loads((out_dir/f'{s}.json').read_text()) for s in names]
    assert all(s['done'] and s['selection_frozen_before_test'] for s in sources)
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['polish.py','confirm.py','scopes.py','variants.py','data.py','operators.py','common.py']}
    result={'done':False,'meta':{'task':task,'sources':hashes,'selection_rule':__doc__,'source_sha256':{s:hashlib.sha256((out_dir/f'{s}.json').read_bytes()).hexdigest() for s in names}},
            'rounds':[],'points':[],'selected':[],'tuning_rows':[],'test_rows':[],'test_points':[],'validation':{}}
    points={key(p['config']):p for s in sources for p in s['points'] if p['config']['method']=='sae'}
    if dest.exists():
        previous=json.loads(dest.read_text());assert previous['meta']==result['meta']
        if previous['done']:return previous
        result['rounds']=previous['rounds'];result['points']=previous['points']
        points.update({key(p['config']):p for p in result['points']})
    def save():
        result['seconds']=time.time()-start;atomic(dest,result)
        if commit:commit()
    layers=sorted(SAE_SPECS)
    model=HookedTransformer.from_pretrained('gemma-2-9b-it',device='cuda',dtype=torch.float16);model.eval();tok=model.tokenizer
    label_ids=[tok.encode(' '+s,add_special_tokens=False)[0] for s in LABELS];pid=label_ids[1]
    saes={l:GemmaScopeSAE(*SAE_SPECS[l],normalize_activations=False) for l in layers};op=Operators(model,saes)
    suite=variant_suite if task in TASKS else base_suite
    names=[f'blocks.{l}.hook_resid_pre' for l in layers]
    def prepare(split):
        rr=suite(task,split);items=[]
        for i in range(0,len(rr),2):
            cr,pr=render(tok,rr[i]),render(tok,rr[i+1]);ct=torch.tensor([cr['token_ids']],device='cuda');pt=torch.tensor([pr['token_ids']],device='cuda')
            ref=model.run_with_hooks(ct,fwd_hooks=[LAST])[0,-1].double().log_softmax(-1)
            with model.hooks(fwd_hooks=[LAST]):ll,cache=model.run_with_cache(pt,names_filter=names)
            ll=ll[0,-1];pp=float(ll.float().softmax(-1)[pid]);row={**pr,'clean_token_ids':cr['token_ids']}
            result[split+'_rows'].append(row)
            item={'tokens':pt,'row':row,'ref':ref,'poison_p':pp,'mask':mask(tok,pr,'cuda'),'x':{l:cache[f'blocks.{l}.hook_resid_pre'] for l in layers},'z':{}}
            items.append(item)
        return items
    def measure(configs,items,direct=False):
        rr=[[] for c in configs];c=configs[0];l=c['layer'];fs=[c['feature'] for c in configs]
        for item in items:
            if direct:
                assert len(configs)==1;logits=edited(op,item['tokens'],{},c,item['row'])[None]
            else:
                strength=torch.tensor([c['strength'] for c in configs],device='cuda')[:,None,None]
                dec=saes[l].W_dec[fs].float()[:,None,:]
                if c.get('intervention','activation')=='activation':
                    missing=[f for f in set(fs) if (l,f) not in item['z']]
                    if missing:
                        z=op.encode(l,item['x'][l][0]);item['z'].update({(l,f):z[:,f].clone() for f in missing})
                    z=torch.stack([item['z'][l,f] for f in fs])[:,:,None]
                else:z=1.
                scope=item['mask'][None] if c.get('scope','all')=='document' else 1.
                x=(item['x'][l].float()-strength*dec*z*scope).to(item['x'][l].dtype)
                logits=model.run_with_hooks(x,start_at_layer=l,fwd_hooks=[LAST])[:,-1]
            for rows,ll in zip(rr,logits):rows.append(metric(tok,label_ids,ll,item['ref'],item['row'],item['poison_p']))
        return [{'config':c,'rows':rows,'valid':not any(r.get('invalid') for r in rows),'summary':summarize(rows) if not any(r.get('invalid') for r in rows) else None,'direct':direct} for c,rows in zip(configs,rr)]
    tuning=prepare('tuning')
    for round_number in range(2):
        if len(result['rounds'])<=round_number:
            configs=proposals(list(points.values()));result['rounds'].append({'round':round_number,'configs':configs,'done':False});save()
        record=result['rounds'][round_number]
        if record['done']:continue
        configs=[c for c in record['configs'] if key(c) not in points];groups={}
        for c in configs:groups.setdefault((c['layer'],c.get('intervention','activation'),c.get('scope','all')),[]).append(c)
        batches=[cc[i:i+8] for cc in groups.values() for i in range(0,len(cc),8)]
        for i,cc in enumerate(batches):
            pp=measure(cc,tuning);result['points'].extend(pp);points.update({key(p['config']):p for p in pp})
            if i%10==0:print('POLISH',round_number,i,'/',len(batches),'seconds',round(time.time()-start),flush=True);save()
        record['done']=True;save()
    checked=set();checks=[]
    for _ in range(12):
        selected=[s for s in select(list(points.values())) if s['family'].startswith('sae')]
        pending={key(s['config']):s['config'] for s in selected if s['config'] and key(s['config']) not in checked}
        if not pending:break
        for k,c in pending.items():
            old=points[k];new=measure([c],tuning,True)[0];err=abs(old['summary']['all']['kl']-new['summary']['all']['kl'])
            maxp=max(abs(a['target_p']-b['target_p']) for a,b in zip(old['rows'],new['rows']));assert err<.03 and maxp<.03
            checks.append({'config':c,'mean_kl_error':err,'max_target_p_error':maxp});points[k]=new;checked.add(k)
            result['points']=[p for p in result['points'] if key(p['config'])!=k]+[new]
    else:raise RuntimeError('Polished selection did not stabilize')
    result['selected']=[s for s in select(list(points.values())) if s['family'].startswith('sae')]
    result['validation']['direct_checks']=checks;result['selection_frozen_before_test']=True;save()
    del tuning;gc.collect();torch.cuda.empty_cache();test=prepare('test')
    for s in result['selected']:
        if s['config'] and not any(p['config']==s['config'] for p in result['test_points']):result['test_points'].extend(measure([s['config']],test,True));save()
    result['done']=True;save();return result
