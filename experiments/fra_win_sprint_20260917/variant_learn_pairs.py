"""Use calibration optimization to select pairs, then sweep ordinary cuts.

Only intervention coefficients are optimized; model and SAE weights are frozen.
Weighted interventions are diagnostics. Primary FRA candidates give each chosen
pair the same coefficient, selected on tuning after pair identification.
"""
from contextlib import ExitStack
import gc,hashlib,json,math,random,sys,time
from pathlib import Path
import torch
from transformer_lens import HookedTransformer
from common import LAST,render,metric,summarize,atomic
from variants import LABELS,suite
from operators import Operators,SAE_SPECS
from frozen import edited
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'reference'))
from sae_lens_wrapper import GemmaScopeSAE
GRID=[0,.25,.5,1,2,4,8,16,32,64]
SNAPSHOTS=[64,128,256,512]


def run(out_dir,commit=None,task='tenants_long'):
    torch.set_grad_enabled(False);torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    start=time.time();out_dir=Path(out_dir);dest=out_dir/f'learn_pairs_{task}.json'
    previous=json.loads(dest.read_text()) if dest.exists() else None
    prior=json.loads((out_dir/f'refine_{task}.json').read_text());assert prior['done']
    pools={name:prior['pair_sets'][name+'_P384']['pairs'] for name in ['distinct','global']}
    layers=sorted({p['layer'] for pp in pools.values() for p in pp})
    result={'done':False,'meta':{'task':task,'source_meta':prior['meta'],'pools':list(pools),'steps':512,'snapshots':SNAPSHOTS,
        'optimizer':'Adam','learning_rate':.08,'regularizer':'.002 * mean(nonnegative pair coefficients)',
        'sources':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['variant_learn_pairs.py','variants.py','frozen.py','operators.py','data.py','common.py']},
        'weighted_sets_are_diagnostic':True,'grid':GRID},'training':[],'pair_sets':{},'points':[],'selected':[],
        'tuning_rows':[],'test_rows':[],'test_points':[],'validation':{}}
    resume=bool(previous and len(previous.get('pair_sets',{}))==56 and previous['meta']['sources']==result['meta']['sources'])
    if resume and previous.get('done'):return previous
    if resume:
        for field in ['training','pair_sets','points']:result[field]=previous[field]
        result['meta']['resumed_points']=len(result['points'])
    def save():
        result['meta']['seconds']=time.time()-start;atomic(dest,result)
        if commit:commit()
    model=HookedTransformer.from_pretrained('gemma-2-9b-it',device='cuda',dtype=torch.float16);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    tok=model.tokenizer;label_ids=[tok.encode(' '+s,add_special_tokens=False)[0] for s in LABELS];pid=label_ids[1]
    saes={l:GemmaScopeSAE(*SAE_SPECS[l],normalize_activations=False) for l in layers}
    for sae in saes.values():
        for p in sae.sae.parameters():p.requires_grad_(False)
    op=Operators(model,saes)
    def prepare(split):
        rows=suite(task,split);items=[]
        for i in range(0,len(rows),2):
            cr,pr=render(tok,rows[i]),render(tok,rows[i+1]);ct=torch.tensor([cr['token_ids']],device='cuda');pt=torch.tensor([pr['token_ids']],device='cuda')
            ref=model.run_with_hooks(ct,fwd_hooks=[LAST])[0,-1].double().log_softmax(-1)
            with model.hooks(fwd_hooks=[LAST]):ll,cache=model.run_with_cache(pt,names_filter=[f'blocks.{l}.hook_resid_pre' for l in layers])
            ll=ll[0,-1];pp=float(ll.float().softmax(-1)[pid])
            items.append({'tokens':pt,'ref':ref,'row':pr,'poison_p':pp,'baseline':ll,'x':{l:cache[f'blocks.{l}.hook_resid_pre'] for l in layers}})
            if split!='calibration':result[split+'_rows'].append({**pr,'clean_token_ids':cr['token_ids'],'baseline':metric(tok,label_ids,ll,ref,pr,pp)})
        return items
    calibration=[] if resume else prepare('calibration')
    for pool_name,pp in ([] if resume else pools.items()):
        grouped={}
        for i,p in enumerate(pp):grouped.setdefault(p['layer'],{}).setdefault(p['head'],[]).append(i)
        projected={}
        for l,hh in grouped.items():
            for h,indices in hh.items():
                qi=[pp[i]['q'] for i in indices];ki=[pp[i]['k'] for i in indices]
                q,k=op.projections(l,h,qi,ki);projected[(l,h)]=(qi,ki,q,k)
        theta=torch.nn.Parameter(torch.full((len(pp),),math.log(math.expm1(.2)),device='cuda'))
        optimizer=torch.optim.Adam([theta],lr=.08);order=list(range(len(calibration)));rng=random.Random(20260917)
        running=[]
        for step in range(1,max(SNAPSHOTS)+1):
            if (step-1)%len(order)==0:rng.shuffle(order)
            item=calibration[order[(step-1)%len(order)]];ds={l:{} for l in grouped};hooks=[LAST]
            with torch.enable_grad():
                weights=torch.nn.functional.softplus(theta).clamp_max(32)
                for l,hh in grouped.items():
                    def make(l,hh):
                        def hook(x,hook):
                            z=op.encode(l,x[0]);rms=op.rms(x[0]);seq=len(z)
                            for h,indices in hh.items():
                                qi,ki,q,k=projected[(l,h)]
                                qr=op.rotate(q,l,range(seq))*z[:,qi,None]/rms[:,None,None]*weights[indices][None,:,None]
                                kr=op.rotate(k,l,range(seq))*z[:,ki,None]/rms[:,None,None]
                                ds[l][h]=(qr.flatten(1)@kr.flatten(1).T/model.blocks[l].attn.attn_scale).tril()
                            return x
                        return hook
                    hooks.append((f'blocks.{l}.hook_resid_pre',make(l,hh)))
                with ExitStack() as stack:
                    for l in grouped:stack.enter_context(op.cut_scores(l,ds[l],[1.]))
                    ll=model.run_with_hooks(item['x'][min(grouped)],start_at_layer=min(grouped),fwd_hooks=hooks)[0,-1]
                    lp=ll.double().log_softmax(-1);kl=(item['ref'].exp()*(item['ref']-lp)).sum()
                    loss=kl+.002*weights.mean()
                optimizer.zero_grad();(loss*128).backward();theta.grad.div_(128)
                assert torch.isfinite(theta.grad).all() and torch.isfinite(loss),f'Nonfinite training step {step}'
                torch.nn.utils.clip_grad_norm_([theta],1.);optimizer.step()
            running.append(float(kl.detach()))
            if step%32==0:
                record={'pool':pool_name,'step':step,'mean_training_kl_last32':sum(running[-32:])/32,
                    'mean_weight':float(torch.nn.functional.softplus(theta).mean()),'max_weight':float(torch.nn.functional.softplus(theta).max())}
                result['training'].append(record);print('LEARN PAIRS',record,'seconds',round(time.time()-start),flush=True);save()
            if step in SNAPSHOTS:
                ww=torch.nn.functional.softplus(theta).clamp_max(32).detach().cpu().tolist();indices=sorted(range(len(pp)),key=lambda i:-ww[i])
                for n in [1,4,16,48,144,384]:
                    result['pair_sets'][f'{pool_name}_S{step}_P{n}']={'variant':'learned_selection_equal_strength','pairs':[pp[i] for i in indices[:n]],
                        'learned_ranking_weights':[ww[i] for i in indices[:n]]}
                result['pair_sets'][f'{pool_name}_S{step}_weighted']={'variant':'weighted_diagnostic','pairs':[{**p,'weight':w} for p,w in zip(pp,ww)]}
                save()
            del ll,lp,kl,loss,ds,weights
        del theta,optimizer
    del calibration;gc.collect();torch.cuda.empty_cache()
    tuning=prepare('tuning')
    def measure(configs,items):
        points=[]
        for cfg in configs:
            rows=[]
            for item in items:
                ll=edited(op,item['tokens'],result,cfg)
                rows.append(metric(tok,label_ids,ll,item['ref'],item['row'],item['poison_p']))
            valid=not any(r.get('invalid') for r in rows)
            points.append({'config':cfg,'rows':rows,'valid':valid,'summary':summarize(rows) if valid else None,'direct':True})
        return points
    ident=next(iter(result['pair_sets']));zero=edited(op,tuning[0]['tokens'],result,{'method':'fra_learned','set':ident,'strength':0})
    result['validation']['zero_max_logit_error']=float((zero-tuning[0]['baseline']).abs().max());assert result['validation']['zero_max_logit_error']<.04
    completed={json.dumps(p['config'],sort_keys=True) for p in result['points']}
    for si,(ident,spec) in enumerate(result['pair_sets'].items()):
        diagnostic=spec['variant']=='weighted_diagnostic'
        configs=[{'method':'weighted_diagnostic' if diagnostic else 'fra_learned','set':ident,'strength':c} for c in ([.5,1,2] if diagnostic else GRID)]
        configs=[c for c in configs if json.dumps(c,sort_keys=True) not in completed]
        result['points'].extend(measure(configs,tuning))
        print('LEARNED SWEEP',si,'/',len(result['pair_sets']),'seconds',round(time.time()-start),flush=True);save()
    for family in ['fra_learned','weighted_diagnostic']:
        for threshold in [None,0.,.5,.9]:
            pool=[p for p in result['points'] if p['valid'] and p['config']['method']==family and
                  (threshold is None or (p['summary']['controls']['correct']>=.95 and p['summary']['suppression']>=threshold))]
            winner=min(pool,key=lambda p:(p['summary']['all']['kl'],abs(p['config']['strength']),p['config']['set'])) if pool else None
            result['selected'].append({'family':family,'threshold':threshold,'config':winner['config'] if winner else None,'tuning':winner['summary'] if winner else None})
    result['selection_frozen_before_test']=True;save();del tuning;gc.collect();torch.cuda.empty_cache()
    test=prepare('test')
    for s in result['selected']:
        if s['config'] and not any(p['config']==s['config'] for p in result['test_points']):result['test_points'].extend(measure([s['config']],test));save()
    result['done']=True;save();return result
