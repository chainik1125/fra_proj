"""Absolute/source-span difference rankings, without waiting for FRA endpoints.

Includes global steering and steering inside the entire retrieved document.
The document scope contains legitimate rows too and never uses the corrupted
row's location as an intervention gate. Candidates use calibration only.
"""
import gc,hashlib,json,sys,time
from pathlib import Path
import torch
from transformer_lens import HookedTransformer
from common import LAST,render,metric,summarize,atomic
from operators import Operators,SAE_SPECS
from scopes import mask
from data import LABELS,suite as base_suite
from variants import TASKS,suite as variant_suite
from search import SAE_GRID,ADDITIVE_GRID
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'reference'))
from sae_lens_wrapper import GemmaScopeSAE


def numeric_key(c):return (c['layer'],c['feature'],c['strength'],c.get('intervention','activation'),c.get('scope','all'))


def run(out_dir,commit=None,task='tenants_long'):
    torch.set_grad_enabled(False);torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    start=time.time();out_dir=Path(out_dir);dest=out_dir/f'baseline_abs_{task}.json'
    prior=json.loads((out_dir/f'search_{task}.json').read_text())
    previous=json.loads(dest.read_text()) if dest.exists() else None
    layers=sorted(SAE_SPECS);suite=variant_suite if task in TASKS else base_suite
    result={'done':False,'meta':{'task':task,'scope':'all tokens or the entire known retrieved document','sources':{
        n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['baseline_abs.py','scopes.py','operators.py','data.py','variants.py','common.py']},
        'original_search_sources':prior['meta']['sources'],'sae_grid':SAE_GRID,'constant_grid':ADDITIVE_GRID,
        'sleeper_reference':'experiments/multitrigger_sleeper/cloud/modeldiff_baseline_pod.py:570'},
        'ranking':{},'endpoint_sets':[],'points':[],'selected':[],'tuning_rows':[],'test_rows':[],'test_points':[],'validation':{}}
    if previous and previous.get('done') and previous['meta']['sources']==result['meta']['sources']:return previous
    def save():
        result['meta']['seconds']=time.time()-start;atomic(dest,result)
        if commit:commit()
    features={l:{} for l in layers};doc_features={l:set() for l in layers}
    model=HookedTransformer.from_pretrained('gemma-2-9b-it',device='cuda',dtype=torch.float16);model.eval();tok=model.tokenizer
    label_ids=[tok.encode(' '+s,add_special_tokens=False)[0] for s in LABELS];pid=label_ids[1]
    saes={l:GemmaScopeSAE(*SAE_SPECS[l],normalize_activations=False) for l in layers};op=Operators(model,saes)
    names=[f'blocks.{l}.hook_resid_pre' for l in layers]
    def tensor(ids):return torch.tensor([ids],device='cuda')
    def pairs(split):
        rr=suite(task,split);return [(render(tok,rr[i]),render(tok,rr[i+1])) for i in range(0,len(rr),2)]
    def cached(tokens):
        with model.hooks(fwd_hooks=[LAST]):return model.run_with_cache(tokens,names_filter=names)
    sums={l:{k:torch.zeros(saes[l].d_sae,device='cuda') for k in ['answer_abs','changed_token_abs','payload_abs','pooled_abs']} for l in layers}
    cal=[(cr,pr) for cr,pr in pairs('calibration') if pr['joint']]
    for cr,pr in cal:
        ct=tensor(cr['token_ids']);pt=tensor(pr['token_ids']);assert ct.shape==pt.shape
        changed=torch.where(ct[0]!=pt[0])[0];assert len(changed)==1
        _,cc=cached(ct);_,pc=cached(pt)
        for l in layers:
            zc=op.encode(l,cc[f'blocks.{l}.hook_resid_pre'][0]);zp=op.encode(l,pc[f'blocks.{l}.hook_resid_pre'][0]);diff=zp-zc
            sums[l]['answer_abs']+=diff[-1]/len(cal);sums[l]['changed_token_abs']+=diff[changed].mean(0)/len(cal)
            sums[l]['payload_abs']+=diff[pr['k_positions']['payload']]/len(cal);sums[l]['pooled_abs']+=diff.mean(0)/len(cal)
    for l,ranks in sums.items():
        result['ranking'][l]={}
        for rank,score in ranks.items():
            ids=score.abs().argsort(descending=True)[:10].tolist()
            result['ranking'][l][rank]=[{'feature':f,'signed_difference':float(score[f])} for f in ids]
            for f in ids:
                features[l].setdefault(f,set()).update(['diff',rank])
                if rank in ['changed_token_abs','payload_abs']:doc_features[l].add(f)
    del sums,cc,pc,zc,zp,diff;gc.collect();torch.cuda.empty_cache()
    features={l:{f:sorted(ranks) for f,ranks in ff.items()} for l,ff in features.items()}
    print('EXTRA FEATURES',[(l,len(ff),len(doc_features[l])) for l,ff in features.items()],flush=True)
    # Reuse already computed identical global interventions. Only ranking tags
    # change, so these are explicitly recorded as imported measurements.
    imported=0
    for p in prior['points']:
        c=p['config']
        if c['method']=='sae' and c['feature'] in features[c['layer']]:
            result['points'].append({**p,'config':{**c,'ranks':features[c['layer']][c['feature']]},'imported_from_search':True});imported+=1
    if previous and previous.get('points'):
        assert previous['meta']['sources']==result['meta']['sources']
        assert previous['ranking']==json.loads(json.dumps(result['ranking']))
        done={numeric_key(p['config']) for p in result['points']}
        result['points'].extend(p for p in previous['points'] if numeric_key(p['config']) not in done)
    result['meta']['imported_points']=imported
    def prepare(split):
        items=[]
        for cr,pr in pairs(split):
            tt=tensor(pr['token_ids']);ref=model.run_with_hooks(tensor(cr['token_ids']),fwd_hooks=[LAST])[0,-1].double().log_softmax(-1)
            ll,cache=cached(tt);ll=ll[0,-1];pp=float(ll.float().softmax(-1)[pid])
            item={'tokens':tt,'row':pr,'ref':ref,'poison_p':pp,'baseline':ll,'x':{},'z':{},'mask':mask(tok,pr,'cuda')}
            for l in layers:
                x=cache[f'blocks.{l}.hook_resid_pre'];item['x'][l]=x;item['z'][l]=op.encode(l,x[0])[:,list(features[l])]
            items.append(item);result[split+'_rows'].append({**pr,'clean_token_ids':cr['token_ids'],'baseline':metric(tok,label_ids,ll,ref,pr,pp)})
        return items
    def predict(item,configs,direct=False):
        cfg=configs[0];l=cfg['layer'];fs=[c['feature'] for c in configs];strength=torch.tensor([c['strength'] for c in configs],device='cuda')[:,None,None]
        dec=saes[l].W_dec[fs].float()[:,None,:];scope=item['mask'][None] if cfg.get('scope','all')=='document' else 1.
        if direct:
            assert len(configs)==1
            def hook(x,hook):
                z=op.encode(l,x)[...,fs[0],None] if cfg.get('intervention','activation')=='activation' else 1.
                return (x.float()-strength*z*dec*scope).to(x.dtype)
            return model.run_with_hooks(item['tokens'],fwd_hooks=[(f'blocks.{l}.hook_resid_pre',hook),LAST])[:,-1]
        columns=[list(features[l]).index(f) for f in fs]
        z=item['z'][l][:,columns].T[:,:,None] if cfg.get('intervention','activation')=='activation' else 1.
        x=(item['x'][l].float()-strength*z*dec*scope).to(item['x'][l].dtype)
        return model.run_with_hooks(x,start_at_layer=l,fwd_hooks=[LAST])[:,-1]
    def measure(configs,items,direct=False):
        rr=[[] for c in configs]
        for item in items:
            ll=predict(item,configs,direct)
            for rows,logits in zip(rr,ll):rows.append(metric(tok,label_ids,logits,item['ref'],item['row'],item['poison_p']))
        return [{'config':c,'rows':rows,'valid':not any(r.get('invalid') for r in rows),
            'summary':summarize(rows) if not any(r.get('invalid') for r in rows) else None,'direct':direct} for c,rows in zip(configs,rr)]
    tuning=prepare('tuning');completed={numeric_key(p['config']) for p in result['points']}
    groups=[]
    for l,ff in features.items():
        for scope in ['all','document']:
            for intervention,grid in [('activation',SAE_GRID),('constant',ADDITIVE_GRID)]:
                configs=[{'method':'sae','layer':l,'feature':f,'ranks':ranks,'scope':scope,'intervention':intervention,'strength':c}
                    for f,ranks in ff.items() if scope=='all' or f in doc_features[l] for c in grid]
                configs=[c for c in configs if numeric_key(c) not in completed]
                groups.extend(configs[i:i+8] for i in range(0,len(configs),8))
    for i,batch in enumerate(groups):
        result['points'].extend(measure(batch,tuning))
        if i%10==0:
            valid=[p for p in result['points'] if p['valid']]
            print('EXTRA SAE SWEEP',i,'/',len(groups),'best',min(p['summary']['all']['kl'] for p in valid),'seconds',round(time.time()-start),flush=True);save()
    from search import select
    checked=set();checks=[]
    for _ in range(12):
        selected=[s for s in select(result['points']) if s['family'].startswith('sae')]
        pending={json.dumps(s['config'],sort_keys=True) for s in selected if s['config']}-checked
        if not pending:break
        for k in sorted(pending):
            c=json.loads(k);index=next(i for i,p in enumerate(result['points']) if p['config']==c);old=result['points'][index];new=measure([c],tuning,True)[0]
            err=abs(old['summary']['all']['kl']-new['summary']['all']['kl']);maxp=max(abs(a['target_p']-b['target_p']) for a,b in zip(old['rows'],new['rows']))
            assert err<.03 and maxp<.03;checks.append({'config':c,'mean_kl_error':err,'max_target_p_error':maxp});result['points'][index]=new;checked.add(k)
    else:raise RuntimeError('Extra SAE selection did not stabilize')
    result['selected']=[s for s in select(result['points']) if s['family'].startswith('sae')];result['validation']['direct_checks']=checks
    result['selection_frozen_before_test']=True;save();del tuning;gc.collect();torch.cuda.empty_cache();test=prepare('test')
    for s in result['selected']:
        if s['config'] and not any(p['config']==s['config'] for p in result['test_points']):result['test_points'].extend(measure([s['config']],test,True));save()
    result['done']=True;save();return result
