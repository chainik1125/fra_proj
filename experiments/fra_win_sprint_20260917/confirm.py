"""Confirmation-only evaluation of choices frozen from completed tuning sweeps."""
from contextlib import ExitStack
import hashlib,json,sys,time
from pathlib import Path
import torch
from transformer_lens import HookedTransformer
from analyze import merged_selection,comparison,cfgkey
from common import LAST,render,metric,summarize,atomic
from data import LABELS,suite as base_suite
from variants import TASKS,suite as variant_suite
from scope_mask import scope_mask
from retention import agreement
from operators import Operators,SAE_SPECS
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'reference'))
from sae_lens_wrapper import GemmaScopeSAE


def expanded_pairs(source,cfg):
    spec=source['pair_sets'][cfg['set']]
    if 'pairs' in spec:return spec['pairs']
    return [{'layer':spec['layer'],'head':int(h),**p} for h,pp in spec['heads'].items() for p in pp]


def pair_delta(op,layer,head,pp,z,rms):
    if all(p.get('weight',1)==1 for p in pp):return op.pair_delta(layer,head,[(p['q'],p['k']) for p in pp],z,rms)
    qi=[p['q'] for p in pp];ki=[p['k'] for p in pp];q,k=op.projections(layer,head,qi,ki);seq=len(z)
    weights=torch.tensor([p.get('weight',1) for p in pp],device=z.device)
    qr=op.rotate(q,layer,range(seq))*z[:,qi,None]/rms[:,None,None]*weights[None,:,None]
    kr=op.rotate(k,layer,range(seq))*z[:,ki,None]/rms[:,None,None]
    return (qr.flatten(1)@kr.flatten(1).T/op.model.blocks[layer].attn.attn_scale).tril()


def edited(op,tokens,source,cfg,row=None):
    model=op.model
    if cfg['method']=='none':return model.run_with_hooks(tokens,fwd_hooks=[LAST])[0,-1]
    c=cfg['strength']
    if cfg['method']=='sae':
        l=cfg['layer'];f=cfg['feature'];dec=op.saes[l].W_dec[f].float()
        scope=scope_mask(model.tokenizer,row,tokens.device,cfg.get('scope','all'))
        def hook(x,hook):
            z=op.encode(l,x)[...,f,None] if cfg.get('intervention','activation')=='activation' else 1.
            return (x.float()-c*z*dec*scope).to(x.dtype)
        return model.run_with_hooks(tokens,fwd_hooks=[(f'blocks.{l}.hook_resid_pre',hook),LAST])[0,-1]
    by_layer={}
    for p in expanded_pairs(source,cfg):by_layer.setdefault(p['layer'],{}).setdefault(p['head'],[]).append(p)
    ds={l:{} for l in by_layer};hooks=[LAST]
    for l,hh in by_layer.items():
        def make(l,hh):
            def hook(x,hook):
                z=op.encode(l,x[0]);rms=op.rms(x[0])
                for h,pp in hh.items():ds[l][h]=pair_delta(op,l,h,pp,z,rms)
                return x
            return hook
        hooks.append((f'blocks.{l}.hook_resid_pre',make(l,hh)))
    with ExitStack() as stack:
        for l in by_layer:stack.enter_context(op.cut_scores(l,ds[l],[c],cfg.get('placement','precap')))
        return model.run_with_hooks(tokens,fwd_hooks=hooks)[0,-1]


@torch.inference_mode()
def run(out_dir,commit=None,task='tenants_long',source_names=None,tag=None,split='confirmation'):
    torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    start=time.time();out_dir=Path(out_dir);tag=task if tag is None else tag
    scope_stage='baseline_scopes' if (out_dir/f'baseline_scopes_{task}.json').exists() else 'baseline_nobos'
    source_names=source_names or [f'{stage}_{task}' for stage in ['search','refine','learn_pairs','baseline_extra',scope_stage,'polish']]
    suite=variant_suite if task in TASKS else base_suite
    sources=[json.loads((out_dir/f'{name}.json').read_text()) for name in source_names]
    assert all(r['done'] and r['selection_frozen_before_test'] for r in sources)
    selection=merged_selection(sources)
    freeze={'task':task,'source_names':source_names,'source_sha256':{name:hashlib.sha256((out_dir/f'{name}.json').read_bytes()).hexdigest() for name in source_names},
        'evaluation_sources':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['confirm.py','retention.py','scope_mask.py','scopes.py','variants.py','data.py','analyze.py','operators.py','common.py']},
        'selection':selection,'frozen_at_unix':time.time(),'split':split}
    freeze_path=out_dir/f'frozen_{tag}.json'
    if freeze_path.exists():
        old=json.loads(freeze_path.read_text());assert old['selection']==selection and old['source_sha256']==freeze['source_sha256'] and old['evaluation_sources']==freeze['evaluation_sources']
        freeze=old
    else:
        atomic(freeze_path,freeze)
        if commit:commit()
    # No confirmation prompt is materialized until the manifest is persisted.
    print('CONFIRMATION FROZEN',source_names,'unix',freeze['frozen_at_unix'],flush=True)
    winners={}
    for s in selection:
        if s['config']:winners[(s['source_index'],cfgkey(s['config']))]=s['config']
    layers=set()
    for (si,k),cfg in winners.items():
        if cfg['method']=='sae':layers.add(cfg['layer'])
        else:layers.update(p['layer'] for p in expanded_pairs(sources[si],cfg))
    model=HookedTransformer.from_pretrained('gemma-2-9b-it',device='cuda',dtype=torch.float16);model.eval();tok=model.tokenizer
    label_ids=[tok.encode(' '+s,add_special_tokens=False)[0] for s in LABELS];pid=label_ids[1]
    saes={l:GemmaScopeSAE(*SAE_SPECS[l],normalize_activations=False) for l in sorted(layers)};op=Operators(model,saes)
    result={'done':False,'freeze':freeze,'rows':[],'points':[{'source_index':si,'config':cfg,'rows':[]} for (si,k),cfg in winners.items()],
        'baseline':{'config':{'method':'none','strength':0},'rows':[]},'clean':{'rows':[]},'comparisons':[],'validation':[],
        'sources':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['confirm.py','retention.py','scope_mask.py','scopes.py','variants.py','operators.py','data.py','common.py','analyze.py']}}
    def save():
        result['seconds']=time.time()-start;atomic(out_dir/f'confirmation_{tag}.json',result)
        if commit:commit()
    # Reproduce a previously measured tuning case through this independent
    # evaluation path before constructing any confirmation examples.
    for p in result['points']:
        src=sources[p['source_index']];row=src['tuning_rows'][0]
        tt=torch.tensor([row['token_ids']],device='cuda');ct=torch.tensor([row['clean_token_ids']],device='cuda')
        ref=model.run_with_hooks(ct,fwd_hooks=[LAST])[0,-1].double().log_softmax(-1)
        ll=edited(op,tt,src,p['config'],row)
        prior=next(q for q in src['points'] if q['config']==p['config'])
        old=next(r for r in prior['rows'] if r['case_id']==row['case_id'])
        new=metric(tok,label_ids,ll,ref,row,old['poison_target_p'])
        err={'config':p['config'],'target_p_error':abs(new['target_p']-old['target_p']),'kl_error':abs(new['kl']-old['kl'])}
        result['validation'].append(err);assert max(err['target_p_error'],err['kl_error'])<.025,err
    save()
    rr=suite(task,split)
    for i in range(0,len(rr),2):
        cr,pr=render(tok,rr[i]),render(tok,rr[i+1]);ct=torch.tensor([cr['token_ids']],device='cuda');pt=torch.tensor([pr['token_ids']],device='cuda')
        assert ct.shape==pt.shape and int((ct!=pt).sum())==1
        cl=model.run_with_hooks(ct,fwd_hooks=[LAST])[0,-1];ref=cl.double().log_softmax(-1)
        pl=model.run_with_hooks(pt,fwd_hooks=[LAST])[0,-1];pp=float(pl.float().softmax(-1)[pid])
        result['rows'].append({**pr,'clean_token_ids':cr['token_ids']})
        result['baseline']['rows'].append(metric(tok,label_ids,pl,ref,pr,pp));result['clean']['rows'].append(metric(tok,label_ids,cl,ref,pr,pp))
        for p in result['points']:
            ll=edited(op,pt,sources[p['source_index']],p['config'],pr)
            p['rows'].append(metric(tok,label_ids,ll,ref,pr,pp))
        if i%16==0:print('CONFIRMATION',i//2,'/',len(rr)//2,'seconds',round(time.time()-start),flush=True);save()
    for p in result['points']+[result['baseline'],result['clean']]:
        p['valid']=not any(r.get('invalid') for r in p['rows']);p['summary']=summarize(p['rows']) if p['valid'] else None
    for p in result['points']+[result['baseline']]:
        if p['valid']:p['clean_reference_agreement']=agreement(p['rows'],result['clean']['rows'])
    def point(family,threshold):
        s=next(s for s in selection if s['family']==family and s['threshold']==threshold)
        return next((p for p in result['points'] if p['source_index']==s['source_index'] and p['config']==s['config']),None)
    for fra_family in ['fra','fra_distinct']:
        for threshold in [None,0.,.5,.9]:
            f=point(fra_family,threshold)
            for family in ['sae_diff','sae_strong']:
                s=point(family,threshold)
                if f and s and f['valid'] and s['valid']:result['comparisons'].append({'fra_family':fra_family,'baseline_family':family,**comparison(f,s,threshold)})
                else:result['comparisons'].append({'fra_family':fra_family,'baseline_family':family,'threshold':threshold,'status':'missing eligible tuning setting or nonfinite evaluation'})
    result['done']=True;save();return result
