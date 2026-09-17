"""Calibrate -> tune -> freeze -> test content-gated QK versus one SAE feature."""
import gc,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from transformer_lens import HookedTransformer
from variants import LABELS,suite
from common import LAST,render,metric,summarize,atomic
from operators import Operators,SAE_SPECS
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'reference'))
from sae_lens_wrapper import GemmaScopeSAE

SAE_GRID=[-16,-8,-4,-2,-1,0,.25,.5,1,2,4,8,16,32,64]
FRA_GRID=[0,.25,.5,1,2,4,8,16,32,64]
ADDITIVE_GRID=[-256,-128,-64,-32,-16,-8,-4,-1,0,1,4,8,16,32,64,128,256]
BATCH=8

def key(c):return json.dumps(c,sort_keys=True)

def select(points):
    selected=[]
    for family in ['sae_diff','sae_strong','fra']:
        options=[p for p in points if p['valid'] and ((family=='fra' and p['config']['method']=='fra') or
            (family.startswith('sae') and p['config']['method']=='sae' and
             (family=='sae_strong' or 'diff' in p['config'].get('ranks',[]))))]
        for threshold in [None,0.,.5,.9]:
            eligible=[p for p in options if threshold is None or
                (p['summary']['controls']['correct']>=.95 and p['summary']['suppression']>=threshold)]
            winner=min(eligible,key=lambda p:(p['summary']['all']['kl'],abs(p['config']['strength']),key(p['config']))) if eligible else None
            selected.append({'family':family,'threshold':threshold,'config':winner['config'] if winner else None,
                             'tuning':winner['summary'] if winner else None})
    return selected


def run(out_dir,commit=None,task='tenants_long',smoke=False,confirm=False):
    torch.set_grad_enabled(False);torch.set_num_threads(4);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    start=time.time();out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    stem=f'search_{task}'+('_smoke' if smoke else '')+('_confirmation' if confirm else '')
    dest=out_dir/f'{stem}.json'
    layers=[21,28] if smoke else list(SAE_SPECS)
    sources={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['variant_search.py','variants.py','data.py','common.py','operators.py','PROTOCOL.md']}
    previous=json.loads(dest.read_text()) if dest.exists() else None
    if previous and previous.get('done') and previous['meta']['sources']==sources:return previous
    model=HookedTransformer.from_pretrained('gemma-2-9b-it',device='cuda',dtype=torch.float16);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    tok=model.tokenizer;label_ids=[tok.encode(' '+s,add_special_tokens=False)[0] for s in LABELS]
    assert all(len(tok.encode(' '+s,add_special_tokens=False))==1 for s in LABELS)
    pid=label_ids[LABELS.index('Print')]
    saes={}
    for l in layers:
        saes[l]=GemmaScopeSAE(*SAE_SPECS[l],normalize_activations=False)
        print('SAE LOADED',l,SAE_SPECS[l],flush=True)
    op=Operators(model,saes);names=[f'blocks.{l}.hook_resid_pre' for l in layers]
    result={'done':False,'meta':{'model':'gemma-2-9b-it','task':task,'sources':sources,'layers':layers,'saes':{l:SAE_SPECS[l] for l in layers},
        'smoke':smoke,'confirm':confirm,'dtype':'float16','native_encoding':True,'rms':'actual_residual','softcap':'precap',
        'sae_grid':SAE_GRID,'additive_grid':ADDITIVE_GRID,'fra_grid':FRA_GRID},'calibration_rows':[],'ranking':{},'pair_sets':{},'points':[],'selected':[],
        'validation':{},'tuning_rows':[],'test_rows':[],'test_points':[]}
    def save():
        result['meta']['seconds']=time.time()-start;atomic(dest,result)
        if commit:commit()
    def tensor(ids):return torch.tensor([ids],device='cuda')
    def pairs(split):
        rr=suite(task,split)
        if smoke:rr=[r for r in rr if r['index']==0 and r['layout']==0]
        return [(render(tok,rr[i]),render(tok,rr[i+1])) for i in range(0,len(rr),2)]
    def forward(tt,cache=False,extra=()):
        with model.hooks(fwd_hooks=[LAST]):
            return model.run_with_cache(tt,names_filter=names+list(extra)) if cache else model(tt)
    def aligned(cr,pr):
        assert len(cr['token_ids'])==len(pr['token_ids'])
        assert sum(a!=b for a,b in zip(cr['token_ids'],pr['token_ids']))==1

    # Calibration retains endpoint activations and computes all-position SAE
    # sensitivity to the paired clean-reference KL. Model weights are frozen.
    cal=[];grad_scores={l:torch.zeros(saes[l].d_sae,device='cuda') for l in layers}
    calibration_pairs=pairs('calibration')
    for ri,(cr,pr) in enumerate(calibration_pairs):
        aligned(cr,pr);ct=tensor(cr['token_ids']);pt=tensor(pr['token_ids'])
        clog,cc=forward(ct,True);ref=clog[0,-1].double().log_softmax(-1).detach()
        captured={}
        def capture(x,hook):
            if hook.name==names[0]:x=x.detach().requires_grad_(True)
            x.retain_grad();captured[hook.name]=x;return x
        grad_names=names+[f'blocks.{l}.attn.hook_attn_scores' for l in layers]
        with torch.enable_grad(),model.hooks(fwd_hooks=[(n,capture) for n in grad_names]+[LAST]):
            plog=model(pt)[0,-1];lp=plog.double().log_softmax(-1)
            loss=(ref.exp()*(ref-lp)).sum();(loss*128).backward()
        pp=float(plog.detach().float().softmax(-1)[pid]);item={'row':pr,'z':{},'zc':{},'rms':{},'edge_grad':{}}
        qpos=pr['q_positions'];kpos=[pr['k_positions'][k] for k in ['payload','cue','asset']];positions=qpos+kpos
        for l in layers:
            x=captured[f'blocks.{l}.hook_resid_pre'];z=op.encode(l,x.detach()[0]);xg=x.grad.detach()[0].float()/128
            assert torch.isfinite(xg).all()
            grad_scores[l]+=(z*(xg@saes[l].W_dec.float().T)).sum(0)/len(calibration_pairs)
            item['z'][l]=z[positions].clone();item['zc'][l]=op.encode(l,cc[f'blocks.{l}.hook_resid_pre'][0,positions])
            item['rms'][l]=op.rms(x.detach()[0,positions])
            scores=captured[f'blocks.{l}.attn.hook_attn_scores'];sg=scores.grad.detach()[0].float()/128
            cap=model.cfg.attn_scores_soft_cap
            deriv=(1-(scores.detach()[0].float()/cap).square()).clamp_min(0) if cap>0 else torch.ones_like(sg)
            deriv=torch.where(torch.isfinite(scores.detach()[0]),deriv,0.)
            item['edge_grad'][l]=(sg*deriv)[:,qpos][:,:,kpos].clone()
        bm=metric(tok,label_ids,plog.detach(),ref,pr,pp)
        result['calibration_rows'].append({**pr,'clean_token_ids':cr['token_ids'],'clean_correct':int(LABELS[int(ref[label_ids].argmax())]==pr['expected']),'baseline':bm})
        cal.append(item)
        del captured,cc,clog,plog,lp,loss,x,z,xg,scores,sg,deriv
        if ri==0:
            extra=[f'blocks.{l}.attn.hook_{v}' for l in layers for v in ['q','k']]
            _,ac=forward(pt,True,extra)
            result['validation']['sae_and_qk_audit']=[op.audit(l,ac[f'blocks.{l}.hook_resid_pre'][0],ac) for l in layers]
            print('AUDIT',result['validation']['sae_and_qk_audit'],flush=True);del ac
        if ri%8==0:print('CALIBRATION',task,ri,'seconds',round(time.time()-start),flush=True)
    torch.cuda.empty_cache()
    nfeatures=2 if smoke else 10;feature_sets={}
    for l in layers:
        means={c:torch.stack([x['z'][l][1] for x in cal if x['row']['corner']==c]).mean(0) for c in sorted({x['row']['corner'] for x in cal})}
        target=[x for x in cal if x['row']['joint']]
        diff=torch.stack([x['z'][l][1]-x['zc'][l][1] for x in target]).mean(0)
        interaction=means['011']-means['010']-means['001']+means['000']
        if '111' in means:interaction-=means['111']-means['110']-means['101']+means['100']
        ranks={};feature_sets[l]={}
        for name,score in [('diff',diff),('interaction',interaction),('gradient',grad_scores[l])]:
            ids=score.argsort(descending=True)[:nfeatures].tolist()
            ranks[name]=[{'feature':f,'score':float(score[f]),'corner_activations':{c:float(v[f]) for c,v in means.items()}} for f in ids]
            for f in ids:feature_sets[l].setdefault(f,[]).append(name)
        result['ranking'][l]=ranks
    # Pair candidates share the same endpoint union across each layer. Ranking
    # is on calibration anchors; application is over all positions.
    variants=['gradient'] if smoke else ['positive','contrast','gradient']
    counts=[4] if smoke else [1,4,16,48]
    head_counts=[1] if smoke else [1,4,16]
    for l in layers:
        qi=torch.where(torch.stack([x['z'][l][:2].sum(0) for x in cal]).sum(0)>0)[0]
        ki=torch.where(torch.stack([x['z'][l][2:4].sum(0) for x in cal]).sum(0)>0)[0]
        ranked={v:{} for v in variants};head_strength={}
        for h in range(model.cfg.n_heads):
            projected=op.projections(l,h,qi,ki)
            total=torch.zeros((len(qi),len(ki)),device='cuda');shared=torch.zeros_like(total);gradient=torch.zeros_like(total)
            n_target=sum(x['row']['joint'] for x in cal);n_shared=sum(x['row']['corner']=='111' for x in cal)
            for x in cal:
                z=x['z'][l];rms=x['rms'][l]
                for ia,q in enumerate(x['row']['q_positions']):
                    for ib,kname in enumerate(['payload','cue']):
                        k=x['row']['k_positions'][kname]
                        if k>q:continue
                        v=op.anchor_terms(l,h,qi,ki,z[ia,qi],z[2+ib,ki],rms[ia],rms[2+ib],q,k,projected)
                        if x['row']['joint']:total+=v/(4*n_target)
                        if x['row']['corner']=='111':shared+=v/(4*n_shared)
                        gradient+=v*x['edge_grad'][l][h,ia,ib]/len(cal)
            head_strength[h]=float(gradient.clamp_min(0).flatten().topk(min(48,gradient.numel())).values.sum())
            for variant in variants:
                score={'positive':total,'contrast':total-shared,'gradient':gradient}[variant]
                valid=torch.where(score.flatten()>1e-10)[0]
                order=valid[score.flatten()[valid].argsort(descending=True)[:max(counts)]]
                rr=[]
                for ix in order.tolist():
                    a,b=divmod(ix,len(ki));rr.append({'q':int(qi[a]),'k':int(ki[b]),'score':float(score[a,b]),
                        'target_term':float(total[a,b]),'shared_term':float(shared[a,b]),'gradient':float(gradient[a,b])})
                ranked[variant][h]=rr
        head_order=sorted(head_strength,key=lambda h:-head_strength[h])
        for v in variants:
            for nh in head_counts:
                for count in counts:
                    ident=f'L{l}_{v}_H{nh}_P{count}'
                    result['pair_sets'][ident]={'layer':l,'variant':v,'head_count':nh,'pair_count_per_head':count,
                        'head_order':head_order,'head_scores':head_strength,
                        'heads':{h:ranked[v][h][:count] for h in head_order[:nh]}}
        print('RANKED',l,'SAE',len(feature_sets[l]),'union',len(qi),len(ki),'heads',head_order[:4],flush=True);save()
    del cal,grad_scores
    # The rest of the module performs the tuning and frozen evaluation.
    return finish(model,op,tok,label_ids,feature_sets,result,previous,pairs,forward,aligned,tensor,dest,save,start,smoke,confirm)


def finish(model,op,tok,label_ids,feature_sets,result,previous,pairs,forward,aligned,tensor,dest,save,start,smoke,confirm):
    pid=label_ids[LABELS.index('Print')];layers=sorted(op.saes)
    def prepare(split):
        prepared=[]
        for cr,pr in pairs(split):
            aligned(cr,pr);tt=tensor(pr['token_ids'])
            ref=forward(tensor(cr['token_ids']))[0,-1].double().log_softmax(-1)
            logits,cache=forward(tt,True);logits=logits[0,-1]
            item={'row':pr,'tokens':tt,'reference':ref,'baseline':logits,'poison_p':float(logits.float().softmax(-1)[pid]),'x':{},'z':{},'rms':{}}
            for l in layers:
                x=cache[f'blocks.{l}.hook_resid_pre'];item['x'][l]=x
                item['z'][l]=op.encode(l,x[0]);item['rms'][l]=op.rms(x[0])
            result.setdefault(split+'_rows',[]).append({**pr,'clean_token_ids':cr['token_ids'],
                'clean_correct':int(LABELS[int(ref[label_ids].argmax())]==pr['expected']),
                'baseline':metric(tok,label_ids,logits,ref,pr,item['poison_p'])})
            prepared.append(item)
        print('PREPARED',split,len(prepared),'seconds',round(time.time()-start),'GPU_GB',round(torch.cuda.memory_allocated()/1e9,2),flush=True)
        return prepared
    def deltas(item,ident,z=None,rms=None):
        spec=result['pair_sets'][ident];l=spec['layer'];z=item['z'][l] if z is None else z;rms=item['rms'][l] if rms is None else rms
        return {int(h):op.pair_delta(l,int(h),[(p['q'],p['k']) for p in pp],z,rms) for h,pp in spec['heads'].items()}
    def predict(item,configs,direct=False):
        cfg=configs[0];n=len(configs);coeff=[c['strength'] for c in configs]
        if cfg['method']=='none':return item['baseline'][None]
        l=cfg['layer'];x=item['x'][l]
        if cfg['method']=='sae':
            fs=[c['feature'] for c in configs];dec=op.saes[l].W_dec[fs].float()[:,None,:]
            strength=torch.tensor(coeff,device='cuda')[:,None,None]
            if direct:
                assert n==1
                def hook(a,hook):
                    z=op.encode(l,a[0])[:,fs[0]][None,:,None] if cfg.get('intervention','activation')=='activation' else 1.
                    return (a.float()-strength*z*dec).to(a.dtype)
                return model.run_with_hooks(item['tokens'],fwd_hooks=[(f'blocks.{l}.hook_resid_pre',hook),LAST])[:,-1]
            z=item['z'][l][:,fs].T[:,:,None] if cfg.get('intervention','activation')=='activation' else 1.
            edited=(x.float()-strength*z*dec).to(x.dtype)
            return model.run_with_hooks(edited,start_at_layer=l,fwd_hooks=[LAST])[:,-1]
        if cfg['method']=='fra':
            hd={};hooks=[LAST]
            if direct:
                assert n==1
                def hook(a,hook):
                    hd.update(deltas(item,cfg['set'],op.encode(l,a[0]),op.rms(a[0])))
                    return a
                hooks.append((f'blocks.{l}.hook_resid_pre',hook))
                inp=item['tokens'];kwargs={}
            else:
                hd.update(deltas(item,cfg['set']));inp=x.expand(n,-1,-1).clone();kwargs={'start_at_layer':l}
            with op.cut_scores(l,hd,coeff,cfg.get('placement','precap')):
                return model.run_with_hooks(inp,fwd_hooks=hooks,**kwargs)[:,-1]
        raise ValueError(cfg)
    def measure(configs,items,direct=False):
        rows=[[] for c in configs]
        for item in items:
            logits=predict(item,configs,direct)
            for ll,rr in zip(logits,rows):rr.append(metric(tok,label_ids,ll,item['reference'],item['row'],item['poison_p']))
        return [{'config':c,'valid':not any(x.get('invalid') for x in rr),'rows':rr,
            'summary':summarize(rr) if not any(x.get('invalid') for x in rr) else None,'direct':direct} for c,rr in zip(configs,rows)]
    tuning=prepare('tuning')
    result['tuning_baseline']=measure([{'method':'none','strength':0}],tuning,True)[0]
    item=tuning[0];zero={}
    for l in layers:
        ident=next(k for k,v in result['pair_sets'].items() if v['layer']==l)
        cfg={'method':'fra','layer':l,'set':ident,'strength':0}
        direct=predict(item,[cfg],True)[0];cached=predict(item,[cfg])[0]
        zero[l]={'full_forward_max_logit_error':float((direct-item['baseline']).abs().max()),
                 'cached_max_logit_error':float((cached-item['baseline']).abs().max())}
        assert max(zero[l].values())<.04,zero[l]
    result['validation']['zero_cut']=zero;print('ZERO AUDIT',zero,flush=True)
    configs=[]
    for l,features in feature_sets.items():
        configs.extend({'method':'sae','layer':l,'feature':f,'ranks':ranks,'strength':c} for f,ranks in features.items() for c in ([0,1,8] if smoke else SAE_GRID))
        if not smoke:
            configs.extend({'method':'sae','intervention':'constant','layer':l,'feature':f,'ranks':ranks,'strength':c} for f,ranks in features.items() for c in ADDITIVE_GRID)
    for ident,spec in result['pair_sets'].items():
        if not any(spec['heads'].values()):continue
        configs.extend({'method':'fra','layer':spec['layer'],'set':ident,'strength':c} for c in ([0,1,8] if smoke else FRA_GRID))
    if previous and not previous.get('done'):
        if previous['points']:assert previous['meta']['sources']==result['meta']['sources'],'Changed numerical source: use a new result path instead of mixing sweeps.'
        assert previous['ranking']==json.loads(json.dumps(result['ranking']))
        assert previous['pair_sets']==json.loads(json.dumps(result['pair_sets']))
        result['points']=previous['points'];result['meta']['resumed_points']=len(previous['points'])
    completed={key(p['config']) for p in result['points']}
    groups=[]
    for c in configs:
        if key(c) in completed:continue
        g=(c['method'],c['layer'],c.get('set'),c.get('intervention','activation'))
        if not groups or groups[-1][0]!=g or len(groups[-1][1])==BATCH:groups.append((g,[]))
        groups[-1][1].append(c)
    save()
    for gi,(_,batch) in enumerate(groups):
        result['points'].extend(measure(batch,tuning))
        if gi%10==0:
            valid=[p for p in result['points'] if p['valid']]
            best={m:round(min(p['summary']['all']['kl'] for p in valid if p['config']['method']==m),5) for m in ['sae','fra'] if any(p['config']['method']==m for p in valid)}
            print('SWEEP',result['meta']['task'],gi,'/',len(groups),'points',len(result['points']),'best',best,'seconds',round(time.time()-start),flush=True);save()
    # Winners must remain winners after ordinary forward passes. Replace their
    # batched estimates with verified values, then repeat until stable.
    checked=set();checks=[]
    for _ in range(12):
        selected=select(result['points']);pending={key(s['config']) for s in selected if s['config']}-checked
        if not pending:break
        for k in sorted(pending):
            idx=next(i for i,p in enumerate(result['points']) if key(p['config'])==k)
            old=result['points'][idx];new=measure([old['config']],tuning,True)[0]
            assert new['valid'],new['config']
            check={'config':new['config'],'mean_kl_error':abs(old['summary']['all']['kl']-new['summary']['all']['kl']),
                'max_target_p_error':max(abs(a['target_p']-b['target_p']) for a,b in zip(old['rows'],new['rows']))}
            checks.append(check);assert check['mean_kl_error']<.03 and check['max_target_p_error']<.03,check
            result['points'][idx]=new;checked.add(k)
    else:raise RuntimeError('Winner validation did not stabilize')
    result['validation']['direct_checks']=checks;result['selected']=select(result['points']);result['selection_frozen_before_test']=True
    save();print('FROZEN',result['selected'],flush=True)
    del tuning;gc.collect();torch.cuda.empty_cache()
    evaluation=prepare('confirmation' if confirm else 'test')
    result['test_baseline']=measure([{'method':'none','strength':0}],evaluation,True)[0]
    winners={key(s['config']):s['config'] for s in result['selected'] if s['config']}
    for cfg in winners.values():
        result['test_points'].extend(measure([cfg],evaluation,True));save()
    result['done']=True;save();print('DONE',dest.name,'seconds',round(time.time()-start),flush=True)
    return result
