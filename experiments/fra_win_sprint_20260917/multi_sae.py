"""Capacity diagnostic: learn up to48 SAE activation coefficients.

Two pools use FRA endpoints plus the strongest single features, or calibration
KL-gradient features plus the strongest single features. Global, whole-document, and all-except-BOS scopes are tested. This is an additional learned baseline, separate
from the requested one-feature comparison. No model/SAE weights are trained.
"""
import gc,hashlib,json,random,sys,time
from pathlib import Path
import torch
from transformer_lens import HookedTransformer
from common import LAST,render,metric,summarize,atomic
from data import LABELS,suite as base_suite
from variants import TASKS,suite as variant_suite
from operators import Operators,SAE_SPECS
from scope_mask import scope_mask
from confirm import expanded_pairs
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'reference'))
from sae_lens_wrapper import GemmaScopeSAE


@torch.no_grad()
def run(out_dir,commit=None,task='tenants_long'):
    torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    start=time.time();out_dir=Path(out_dir);raw=(out_dir/f'frozen_{task}.json').read_bytes();freeze=json.loads(raw)
    sources=[json.loads((out_dir/f'{name}.json').read_text()) for name in freeze['source_names']]
    fra=next(s for s in freeze['selection'] if s['family']=='fra' and s['threshold']==.5);assert fra['config']
    singles=[s['config'] for s in freeze['selection'] if s['family']=='sae_strong' and s['threshold'] in [.5,.9] and s['config']]
    ranking=json.loads((out_dir/f'search_{task}.json').read_text())['ranking']
    def unique(values):return list(dict.fromkeys(values))[:48]
    first=[(c['layer'],c['feature']) for c in singles]
    pp=expanded_pairs(sources[fra['source_index']],fra['config'])
    pools={'endpoints_plus_best':unique(first+[(p['layer'],f) for p in pp for f in [p['q'],p['k']]]),
        'gradients_plus_best':unique(first+[(l,ranking[str(l)]['gradient'][i]['feature']) for i in range(10) for l in SAE_SPECS])}
    layers=sorted(set(l for pool in pools.values() for l,f in pool));suite=variant_suite if task in TASKS else base_suite
    model=HookedTransformer.from_pretrained('gemma-2-9b-it',device='cuda',dtype=torch.float16);model.eval();tok=model.tokenizer
    for p in model.parameters():p.requires_grad_(False)
    saes={l:GemmaScopeSAE(*SAE_SPECS[l],normalize_activations=False) for l in layers};op=Operators(model,saes)
    for sae in saes.values():
        for p in sae.sae.parameters():p.requires_grad_(False)
    label_ids=[tok.encode(' '+s,add_special_tokens=False)[0] for s in LABELS];pid=label_ids[1]
    result={'done':False,'meta':{'task':task,'description':__doc__,'frozen_selection_sha256':hashlib.sha256(raw).hexdigest(),
        'sources':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['multi_sae.py','scope_mask.py','operators.py','scopes.py','data.py','variants.py','common.py','confirm.py']},
        'pools':pools,'steps':512,'snapshots':[0,64,128,256,512],'learning_rate':.08,'regularizer':'.002 * mean(abs(coefficients))','grid':[.25,.5,1.,2.]},
        'training':[],'snapshots':{},'points':[],'selected':[],'confirmation_points':[],'tuning_rows':[],'confirmation_rows':[]}
    def save():
        result['seconds']=time.time()-start;atomic(out_dir/f'multi_sae_{task}.json',result)
        if commit:commit()
    def prepare(split):
        rr=suite(task,split);items=[]
        for i in range(0,len(rr),2):
            cr,pr=render(tok,rr[i]),render(tok,rr[i+1]);ct=torch.tensor([cr['token_ids']],device='cuda');pt=torch.tensor([pr['token_ids']],device='cuda')
            ref=model.run_with_hooks(ct,fwd_hooks=[LAST])[0,-1].double().log_softmax(-1)
            with model.hooks(fwd_hooks=[LAST]):ll,cache=model.run_with_cache(pt,names_filter=[f'blocks.{l}.hook_resid_pre' for l in layers])
            ll=ll[0,-1];pp=float(ll.float().softmax(-1)[pid]);row={**pr,'clean_token_ids':cr['token_ids']}
            items.append({'tokens':pt,'row':row,'ref':ref,'poison_p':pp,'scope_masks':{scope:scope_mask(tok,pr,'cuda',scope) for scope in ['all','document','no_bos']},
                          'x':{l:cache[f'blocks.{l}.hook_resid_pre'] for l in layers}})
            if split!='calibration':result[split+'_rows'].append(row)
        return items
    def predict(item,pool,weights,scope,training=False):
        grouped={}
        for i,(l,f) in enumerate(pool):grouped.setdefault(l,[]).append((i,f))
        hooks=[LAST];gate=item['scope_masks'][scope]
        for l,entries in grouped.items():
            indices=[i for i,f in entries];features=[f for i,f in entries];dec=saes[l].W_dec[features].float()
            def make(l,indices,features,dec):
                def hook(x,hook):
                    z=op.encode(l,x)[...,features];delta=(z*weights[indices])@dec
                    return (x.float()-delta*gate).to(x.dtype)
                return hook
            hooks.append((f'blocks.{l}.hook_resid_pre',make(l,indices,features,dec)))
        if training:return model.run_with_hooks(item['x'][min(grouped)],start_at_layer=min(grouped),fwd_hooks=hooks)[0,-1]
        return model.run_with_hooks(item['tokens'],fwd_hooks=hooks)[0,-1]
    calibration=prepare('calibration')
    for pool_name,pool in pools.items():
        for scope in ['all','document','no_bos']:
            name=pool_name+'_'+scope;initial=torch.zeros(len(pool),device='cuda')
            warm=next((c for c in singles if c.get('intervention','activation')=='activation' and c.get('scope','all')==scope),None)
            if warm:initial[pool.index((warm['layer'],warm['feature']))]=warm['strength']
            theta=torch.nn.Parameter(initial);optimizer=torch.optim.Adam([theta],lr=.08);order=list(range(len(calibration)));rng=random.Random(20260917)
            result['snapshots'][name+'_S0']={'pool':pool_name,'scope':scope,'weights':initial.cpu().tolist(),'warm_start':warm};running=[]
            for step in range(1,513):
                if (step-1)%len(order)==0:rng.shuffle(order)
                item=calibration[order[(step-1)%len(order)]]
                with torch.enable_grad():
                    ll=predict(item,pool,theta,scope,True);lp=ll.double().log_softmax(-1);kl=(item['ref'].exp()*(item['ref']-lp)).sum()
                    loss=kl+.002*theta.abs().mean();optimizer.zero_grad();(loss*128).backward();theta.grad.div_(128)
                    assert torch.isfinite(theta.grad).all() and torch.isfinite(loss)
                    torch.nn.utils.clip_grad_norm_([theta],1.);optimizer.step()
                theta.clamp_(-64,64);running.append(float(kl))
                if step%64==0:
                    record={'pool_scope':name,'step':step,'mean_calibration_kl_last32':sum(running[-32:])/32};result['training'].append(record)
                    print('MULTI SAE',record,flush=True)
                if step in [64,128,256,512]:
                    result['snapshots'][name+f'_S{step}']={'pool':pool_name,'scope':scope,'weights':theta.cpu().tolist()};save()
            del theta,optimizer,ll,lp,kl,loss
    del calibration;gc.collect();torch.cuda.empty_cache();tuning=prepare('tuning')
    def measure(cfg,items):
        spec=result['snapshots'][cfg['snapshot']];weights=torch.tensor(spec['weights'],device='cuda')*cfg['strength'];pool=pools[spec['pool']];rows=[]
        for item in items:
            ll=predict(item,pool,weights,spec['scope']);rows.append(metric(tok,label_ids,ll,item['ref'],item['row'],item['poison_p']))
        valid=not any(r.get('invalid') for r in rows)
        return {'config':cfg,'rows':rows,'valid':valid,'summary':summarize(rows) if valid else None,'direct':True}
    for name in result['snapshots']:
        for strength in [.25,.5,1.,2.]:result['points'].append(measure({'method':'multi_sae','snapshot':name,'strength':strength},tuning))
        print('MULTI SAE TUNING',name,flush=True);save()
    for threshold in [None,0.,.5,.9]:
        eligible=[p for p in result['points'] if p['valid'] and (threshold is None or (p['summary']['controls']['correct']>=.95 and p['summary']['suppression']>=threshold))]
        winner=min(eligible,key=lambda p:p['summary']['all']['kl']) if eligible else None
        result['selected'].append({'threshold':threshold,'config':winner['config'] if winner else None,'tuning':winner['summary'] if winner else None})
    result['selection_frozen_before_confirmation']=True;save();del tuning;gc.collect();torch.cuda.empty_cache()
    confirmation=prepare('confirmation')
    for s in result['selected']:
        if s['config'] and not any(p['config']==s['config'] for p in result['confirmation_points']):result['confirmation_points'].append(measure(s['config'],confirmation));save()
    result['done']=True;save();return result
