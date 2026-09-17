"""Rank anchor-identified pairs by their actual all-position KL derivative.

The candidate universe is frozen by the first search's calibration. This stage
adds live multi-layer cuts; it never reads test/confirmation outcomes to rank or
tune interventions. Its points can be merged with the first SAE/FRA sweep.
"""
import gc,hashlib,json,time,sys
from contextlib import ExitStack
from pathlib import Path
import torch
from transformer_lens import HookedTransformer
from common import LAST,render,metric,summarize,atomic
from data import LABELS,suite
from operators import Operators,SAE_SPECS
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'reference'))
from sae_lens_wrapper import GemmaScopeSAE
GRID=[0,.125,.25,.5,1,2,4,8,16,32,64]


def run(out_dir,commit=None,task='tenants_long'):
    torch.set_grad_enabled(False);torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    start=time.time();out_dir=Path(out_dir);dest=out_dir/f'refine_{task}.json'
    previous=json.loads(dest.read_text()) if dest.exists() else None
    source=json.loads((out_dir/f'search_{task}.json').read_text())
    assert len(source['pair_sets'])==216,'Wait until anchor calibration is complete.'
    candidates={}
    for spec in source['pair_sets'].values():
        if spec['head_count']!=16 or spec['pair_count_per_head']!=48:continue
        for h,pp in spec['heads'].items():
            candidates.setdefault((spec['layer'],int(h)),set()).update((p['q'],p['k']) for p in pp)
    candidates={lh:sorted(pp) for lh,pp in candidates.items()};layers=sorted({l for l,h in candidates})
    result={'done':False,'meta':{'task':task,'source_meta':source['meta'],'source_pair_sets_sha':hashlib.sha256(json.dumps(source['pair_sets'],sort_keys=True).encode()).hexdigest(),
        'sources':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['refine.py','operators.py','data.py','common.py']},
        'gradient':'all_positions_paired_KL','multi_layer':'live','grid':GRID},'ranking':[],'pair_sets':{},'points':[],
        'selected':[],'test_points':[],'validation':{},'tuning_rows':[],'test_rows':[]}
    if previous and previous.get('done') and previous['meta']['sources']==result['meta']['sources']:return previous
    def save():
        result['meta']['seconds']=time.time()-start;atomic(dest,result)
        if commit:commit()
    model=HookedTransformer.from_pretrained('gemma-2-9b-it',device='cuda',dtype=torch.float16);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    tok=model.tokenizer;label_ids=[tok.encode(' '+s,add_special_tokens=False)[0] for s in LABELS];pid=label_ids[1]
    saes={l:GemmaScopeSAE(*SAE_SPECS[l],normalize_activations=False) for l in layers};op=Operators(model,saes)
    names=[f'blocks.{l}.hook_resid_pre' for l in layers]
    scores={lh:torch.zeros(len(pp),device='cuda') for lh,pp in candidates.items()}
    source_scores={lh:torch.zeros_like(v) for lh,v in scores.items()}
    def tensor(ids):return torch.tensor([ids],device='cuda')
    def pairs(split):
        rows=suite(task,split)
        return [(render(tok,rows[i]),render(tok,rows[i+1])) for i in range(0,len(rows),2)]
    cal=pairs('calibration')
    for ri,(cr,pr) in enumerate(cal):
        ct=tensor(cr['token_ids']);pt=tensor(pr['token_ids'])
        ref=model.run_with_hooks(ct,fwd_hooks=[LAST])[0,-1].double().log_softmax(-1).detach()
        captured={}
        def capture(x,hook):
            if hook.name==names[0]:x=x.detach().requires_grad_(True)
            x.retain_grad();captured[hook.name]=x;return x
        with torch.enable_grad(),model.hooks(fwd_hooks=[(n,capture) for n in names+[f'blocks.{l}.attn.hook_attn_scores' for l in layers]]+[LAST]):
            ll=model(pt)[0,-1];lp=ll.double().log_softmax(-1);loss=(ref.exp()*(ref-lp)).sum();(loss*128).backward()
        for l in layers:
            x=captured[f'blocks.{l}.hook_resid_pre'].detach()[0];z=op.encode(l,x);rms=op.rms(x);seq=len(z)
            ss=captured[f'blocks.{l}.attn.hook_attn_scores'];g=ss.grad.detach()[0].float()/128
            cap=model.cfg.attn_scores_soft_cap;derivative=(1-(ss.detach()[0].float()/cap).square()).clamp_min(0)
            g*=torch.where(torch.isfinite(ss.detach()[0]),derivative,0.)
            for h in range(model.cfg.n_heads):
                pp=candidates[(l,h)]
                if not pp:continue
                qi,ki=map(list,zip(*pp));q,k=op.projections(l,h,qi,ki)
                qr=op.rotate(q,l,range(seq))*z[:,qi,None]/rms[:,None,None]
                kr=op.rotate(k,l,range(seq))*z[:,ki,None]/rms[:,None,None]
                # g @ kr sums over key positions, then qr contracts over query
                # positions/head dimensions, separately for each pair.
                propagated=(g[h]@kr.flatten(1)).reshape(seq,len(pp),-1)
                scores[(l,h)]+=(qr*propagated).sum((0,2))/(model.blocks[l].attn.attn_scale*len(cal))
                if pr['joint']:
                    source_g=torch.zeros_like(g[h]);keys=pr['source_positions'];source_g[:,keys]=g[h][:,keys]
                    propagated=(source_g@kr.flatten(1)).reshape(seq,len(pp),-1)
                    source_scores[(l,h)]+=(qr*propagated).sum((0,2))/(model.blocks[l].attn.attn_scale*len(cal))
            del x,z,rms,ss,g,derivative,qr,kr,propagated
        del captured,ll,lp,loss
        if ri%4==0:print('FULL POSITION RANK',task,ri,'/',len(cal),'seconds',round(time.time()-start),flush=True)
    for (l,h),pp in candidates.items():
        result['ranking'].extend({'layer':l,'head':h,'q':q,'k':k,'gradient':float(scores[(l,h)][i]),
            'target_source_gradient':float(source_scores[(l,h)][i])} for i,(q,k) in enumerate(pp))
    result['ranking'].sort(key=lambda x:-x['gradient']);del scores,source_scores
    # Global budgets avoid spending the same quota on non-causal heads. Also
    # retain per-layer variants, and a variant requiring two distinct features.
    for variant in ['global','distinct','source_positive']:
        pool=[r for r in result['ranking'] if r['gradient']>1e-10 and
              (variant!='distinct' or r['q']!=r['k']) and (variant!='source_positive' or r['target_source_gradient']>0)]
        for n in [1,4,16,48,144,384,1024]:
            selected=pool[:n]
            if not selected:continue
            result['pair_sets'][f'{variant}_P{n}']={'variant':variant,'pairs':selected}
    for l in layers:
        pool=[r for r in result['ranking'] if r['layer']==l and r['gradient']>1e-10]
        for n in [4,16,48,144]:
            if pool:result['pair_sets'][f'layer_{l}_P{n}']={'variant':'single_layer_full_gradient','pairs':pool[:n]}
    print('REFINED CANDIDATES',len(result['ranking']),'sets',len(result['pair_sets']),'top',result['ranking'][:5],flush=True);save()
    if previous and previous.get('points'):
        assert previous['meta']['sources']==result['meta']['sources']
        assert previous['pair_sets']==result['pair_sets']
        result['points']=previous['points'];result['meta']['resumed_points']=len(previous['points'])
    gc.collect();torch.cuda.empty_cache()

    def prepare(split):
        items=[]
        for cr,pr in pairs(split):
            tt=tensor(pr['token_ids']);ref=model.run_with_hooks(tensor(cr['token_ids']),fwd_hooks=[LAST])[0,-1].double().log_softmax(-1)
            with model.hooks(fwd_hooks=[LAST]):ll,cache=model.run_with_cache(tt,names_filter=names)
            ll=ll[0,-1];pp=float(ll.float().softmax(-1)[pid])
            items.append({'row':pr,'tokens':tt,'ref':ref,'poison_p':pp,'baseline':ll,'x':{l:cache[f'blocks.{l}.hook_resid_pre'] for l in layers}})
            result[split+'_rows'].append({**pr,'clean_token_ids':cr['token_ids'],'baseline':metric(tok,label_ids,ll,ref,pr,pp)})
        return items
    def predict(item,ident,coefficients,direct=False):
        by_layer={}
        for p in result['pair_sets'][ident]['pairs']:by_layer.setdefault(p['layer'],{}).setdefault(p['head'],[]).append((p['q'],p['k']))
        deltas={l:{} for l in by_layer};hooks=[LAST];n=len(coefficients)
        for l,hh in by_layer.items():
            def make(l,hh):
                def hook(x,hook):
                    z=op.encode(l,x);rms=op.rms(x)
                    for h,pp in hh.items():deltas[l][h]=torch.stack([op.pair_delta(l,h,pp,z[b],rms[b]) for b in range(x.shape[0])])
                    return x
                return hook
            hooks.append((f'blocks.{l}.hook_resid_pre',make(l,hh)))
        first=min(by_layer)
        inp=item['tokens'] if direct else item['x'][first].expand(n,-1,-1).clone()
        kwargs={} if direct else {'start_at_layer':first}
        assert not direct or n==1
        with ExitStack() as stack:
            for l in by_layer:stack.enter_context(op.cut_scores(l,deltas[l],coefficients))
            return model.run_with_hooks(inp,fwd_hooks=hooks,**kwargs)[:,-1]
    def measure(ident,coefficients,items,direct=False):
        rows=[[] for c in coefficients]
        for item in items:
            logits=predict(item,ident,coefficients,direct)
            for rr,ll in zip(rows,logits):rr.append(metric(tok,label_ids,ll,item['ref'],item['row'],item['poison_p']))
        return [{'config':{'method':'fra_refined','set':ident,'strength':c},'valid':not any(r.get('invalid') for r in rr),
                 'summary':summarize(rr) if not any(r.get('invalid') for r in rr) else None,'rows':rr,'direct':direct} for c,rr in zip(coefficients,rows)]
    tuning=prepare('tuning');result['tuning_baseline']={'rows':[r['baseline'] for r in result['tuning_rows']]}
    ident=next(k for k in result['pair_sets'] if k=='global_P144')
    zero=predict(tuning[0],ident,[0],True)[0]
    result['validation']['zero_full_forward_max_logit_error']=float((zero-tuning[0]['baseline']).abs().max())
    assert result['validation']['zero_full_forward_max_logit_error']<.04
    completed={(p['config']['set'],p['config']['strength']) for p in result['points']}
    for si,ident in enumerate(result['pair_sets']):
        remaining=[c for c in GRID if (ident,c) not in completed]
        for offset in range(0,len(remaining),4):result['points'].extend(measure(ident,remaining[offset:offset+4],tuning))
        if si%2==0:print('REFINED SWEEP',si,'/',len(result['pair_sets']),'seconds',round(time.time()-start),flush=True);save()
    def select():
        out=[]
        for threshold in [None,0.,.5,.9]:
            pool=[p for p in result['points'] if p['valid'] and (threshold is None or
                (p['summary']['controls']['correct']>=.95 and p['summary']['suppression']>=threshold))]
            winner=min(pool,key=lambda p:(p['summary']['all']['kl'],abs(p['config']['strength']),p['config']['set'])) if pool else None
            out.append({'family':'fra_refined','threshold':threshold,'config':winner['config'] if winner else None,'tuning':winner['summary'] if winner else None})
        return out
    checked=set();checks=[]
    for _ in range(12):
        selected=select();pending={json.dumps(s['config'],sort_keys=True) for s in selected if s['config']}-checked
        if not pending:break
        for k in sorted(pending):
            cfg=json.loads(k);idx=next(i for i,p in enumerate(result['points']) if p['config']==cfg);old=result['points'][idx]
            new=measure(cfg['set'],[cfg['strength']],tuning,True)[0]
            assert new['valid'];err=abs(new['summary']['all']['kl']-old['summary']['all']['kl'])
            maxp=max(abs(a['target_p']-b['target_p']) for a,b in zip(old['rows'],new['rows']))
            checks.append({'config':cfg,'mean_kl_error':err,'max_target_p_error':maxp});assert err<.03 and maxp<.03,checks[-1]
            result['points'][idx]=new;checked.add(k)
    else:raise RuntimeError('Selection failed to stabilize')
    result['selected']=select();result['validation']['direct_checks']=checks;result['selection_frozen_before_test']=True;save()
    del tuning;gc.collect();torch.cuda.empty_cache();test=prepare('test')
    for s in result['selected']:
        if s['config'] and not any(p['config']==s['config'] for p in result['test_points']):
            cfg=s['config'];result['test_points'].extend(measure(cfg['set'],[cfg['strength']],test,True));save()
    result['done']=True;save();return result
