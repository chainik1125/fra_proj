"""Resolve a causally selected OV path into SAE features and test transfer."""
import gzip,json,time,hashlib
from concurrent.futures import ThreadPoolExecutor
from trace_baseline import ROOT,save_json,checkpoint_manifest,TARGETS
from localize_gender_transport import metrics,replace_hook
from prepare_concept_groups import fetch,BUCKET
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

OUT=ROOT/'gender_transport'
RELEASE='gpt2-small-resid-post-v5-32k'

@torch.no_grad()
def encode(sae,x):
    z=sae.encode(x);std=sae.ln_std.clone();mu=sae.ln_mu.clone()
    recon=sae.decode(z)
    return dict(z=z,a=z*std,std=std,bias=mu+std*sae.b_dec,error=x-recon,recon=recon)

def main():
    started=time.time();torch.set_num_threads(4);torch.manual_seed(0);torch.set_grad_enabled(False)
    old=json.loads((OUT/'localization.json').read_text())
    grouped={}
    for r in old:
        if r['stage']=='edge_scan' and r['name']=='content':
            key=tuple(r[k] for k in ['layer','head','query','key'])
            grouped.setdefault(key,[]).append(r)
    ranking=[dict(layer=k[0],head=k[1],query=k[2],key=k[3],mean_fraction=sum(r['donor_effect_fraction'] for r in rr)/2) for k,rr in grouped.items() if len(rr)==2]
    ranking.sort(key=lambda r:-r['mean_fraction']);path=ranking[0]
    L,H,Q,K=[path[k] for k in ['layer','head','query','key']]
    assert L>0
    save_json(OUT/'selected_path.json',dict(selected=path,ranking=ranking,selection='Largest mean bidirectional unit content-patch effect from the completed edge scan, before SAE feature outcomes'))
    print('PATH',path,flush=True)
    model=HookedTransformer.from_pretrained('gpt2-small',device='cpu').eval()
    sae=SAE.from_pretrained(RELEASE,f'blocks.{L-1}.hook_resid_post',device='cpu').eval()
    for m in [model,sae]:
        for p in m.parameters():p.requires_grad_(False)
    ln=f'blocks.{L}.ln1.hook_normalized';zk=f'blocks.{L}.attn.hook_z';pk=f'blocks.{L}.attn.hook_pattern';vk=f'blocks.{L}.attn.hook_v';sk=f'blocks.{L}.attn.hook_attn_scores'
    ids={w:model.to_tokens(w,prepend_bos=False).item() for w in TARGETS}
    tok={};ca={};logits={};base={};codes={};checks={}
    with torch.no_grad():
        for sex in ['female','male']:
            tok[sex]=model.to_tokens(f'A {sex} monarch is called a',prepend_bos=False)
            logits[sex],ca[sex]=model.run_with_cache(tok[sex]);base[sex]=metrics(logits[sex],model,ids)
            x=ca[sex][ln][0];codes[sex]=encode(sae,x)
            raw=encode(sae,ca[sex][f'blocks.{L-1}.hook_resid_post'][0])
            checks[sex]=dict(support_changes=int(((raw['z']>0)!=(codes[sex]['z']>0)).any(-1).sum()),
                max_coefficient_difference=float((raw['z']-codes[sex]['z']).abs().max()),
                input_relative_squared_error=float(codes[sex]['error'].square().sum()/x.square().sum()))
            assert checks[sex]['support_changes']==0
    # Gradient only ranks candidate features; all reported interventions use full reruns.
    stored={}
    def grad_hook(x,hook):
        x=x.detach().requires_grad_(True);stored['x']=x;return x
    with torch.enable_grad():
        g_logits=model.run_with_hooks(tok['female'],fwd_hooks=[(zk,grad_hook)])
        margin=g_logits[0,-1,ids[' queen']]-g_logits[0,-1,ids[' king']]
        margin.backward()
    grad=stored['x'].grad[0,Q,H].detach()
    with torch.no_grad():
        diff=codes['male']['a'][K]-codes['female']['a'][K]
        active=diff.nonzero().flatten().tolist()
        projections=sae.W_dec[active]@model.blocks[L].attn.W_V[H]
        A=float(ca['female'][pk][0,H,Q,K])
        messages=diff[active,None]*projections*A
        target_diff=base['male']['margin']-base['female']['margin']
        scores=messages@grad
        fr=[dict(feature=i,female_activation=float(codes['female']['z'][K,i]),male_activation=float(codes['male']['z'][K,i]),
                 predicted_delta_margin=float(v),predicted_donor_fraction=float(v/target_diff)) for i,v in zip(active,scores)]
        fr.sort(key=lambda r:-r['predicted_donor_fraction'])
        feature_order=[r['feature'] for r in fr]
        save_json(OUT/'feature_ranking.json',dict(layer=L,input_hook=ln,original_training_hook=f'blocks.{L-1}.hook_resid_post',
            input_sae_transferred=True,selection='Gradient of female-prompt final margin times male-minus-female feature message, before individual causal feature tests',ranking=fr))
    records=[]
    def run_delta(sex,name,delta,**extra):
        def hook(x,hook):
            out=x.clone();out[:,Q,H]+=delta;return out
        with torch.no_grad():out=model.run_with_hooks(tok[sex],fwd_hooks=[(zk,hook)])
        m=metrics(out,model,ids);donor='male' if sex=='female' else 'female'
        dd=m['margin']-base[sex]['margin']
        row=dict(base=sex,name=name,metrics=m,delta_margin=dd,donor_effect_fraction=dd/(base[donor]['margin']-base[sex]['margin']),**extra)
        assert torch.equal(out[:,:Q],logits[sex][:,:Q])
        records.append(row);return row
    for sex in ['female','male']:
        donor='male' if sex=='female' else 'female'
        with torch.no_grad():
            attn=ca[sex][pk][0,H,Q,K]
            ad=codes[donor]['a'][K]-codes[sex]['a'][K]
            Wv=model.blocks[L].attn.W_V[H]
            fdelta=(ad@sae.W_dec)@Wv
            bdelta=(codes[donor]['bias'][K]-codes[sex]['bias'][K])@Wv
            edelta=(codes[donor]['error'][K]-codes[sex]['error'][K])@Wv
            exact=ca[donor][vk][0,K,H]-ca[sex][vk][0,K,H]
            err=float((fdelta+bdelta+edelta-exact).abs().max());assert err<1e-4
            checks[sex]['value_difference_reconstruction_error']=err
            r=run_delta(sex,'full_content_patch',attn*exact)
            expected=next(x for x in old if x['stage']=='edge_scan' and x['name']=='content' and x['base']==sex and all(x[k]==path[k] for k in ['layer','head','query','key']))
            assert abs(r['metrics']['margin']-expected['metrics']['margin'])<1e-4
            for name,d in [('no_op',torch.zeros_like(exact)),('all_sae_features',fdelta),('bias_only',bdelta),('error_only',edelta),('features_bias_error',fdelta+bdelta+edelta)]:
                run_delta(sex,name,attn*d)
            for i in feature_order:
                run_delta(sex,'single_feature',attn*ad[i]*(sae.W_dec[i]@Wv),feature=i)
            for n in [1,2,4,8,16,32]:
                selected=feature_order[:n]
                d=(ad[selected]@sae.W_dec[selected])@Wv
                for alpha in ([1,2,4] if n in [1,4] else [1]):
                    rr=run_delta(sex,'top_features',attn*d*alpha,features=selected,n=len(selected),alpha=alpha,
                        extrapolates_beyond_donor=alpha>1)
                    print('FEATURES',sex,n,alpha,round(rr['donor_effect_fraction'],3),rr['metrics']['candidates'],flush=True)
    save_json(OUT/'feature_results.json',records);save_json(OUT/'feature_checks.json',checks)
    # Identify query-feature/key-feature score contributions on this causal edge.
    qk=[]
    with torch.no_grad():
        c=codes['female'];qids=c['a'][Q].nonzero().flatten().tolist();kids=c['a'][K].nonzero().flatten().tolist()
        qp=c['a'][Q,qids,None]*(sae.W_dec[qids]@model.blocks[L].attn.W_Q[H])
        kp=c['a'][K,kids,None]*(sae.W_dec[kids]@model.blocks[L].attn.W_K[H])
        products=qp@kp.T/model.blocks[L].attn.attn_scale
        candidates=[(float(products[i,j]),qi,ki) for i,qi in enumerate(qids) for j,ki in enumerate(kids)]
        candidates.sort(key=lambda t:-abs(t[0]))
        for val,qi,ki in candidates[:12]:
            def hook(s,hook,val=val):
                out=s.clone();out[:,H,Q,K]-=val;return out
            out=model.run_with_hooks(tok['female'],fwd_hooks=[(sk,hook)])
            m=metrics(out,model,ids)
            qk.append(dict(query_feature=qi,key_feature=ki,score_contribution=val,metrics=m,delta_margin=m['margin']-base['female']['margin']))
    save_json(OUT/'causal_edge_qk.json',qk)
    # Joint counterfactual patches test distributed transport. Native values are
    # replaced, so earlier changes are not double-counted as additive deltas.
    joint=[]
    layers=json.loads((OUT/'selected_layers.json').read_text())['selected']
    with torch.no_grad():
        for sex in ['female','male']:
            donor='male' if sex=='female' else 'female'
            for mode in ['pattern','v','attn_out','mlp_out']:
                short={'pattern':'attn.hook_pattern','v':'attn.hook_v','attn_out':'hook_attn_out','mlp_out':'hook_mlp_out'}[mode]
                for ll in [layers,list(range(12))]:
                    hooks=[(f'blocks.{l}.{short}',replace_hook(ca[donor][f'blocks.{l}.{short}'])) for l in ll]
                    out=model.run_with_hooks(tok[sex],fwd_hooks=hooks);m=metrics(out,model,ids)
                    joint.append(dict(base=sex,mode=mode,layers=ll,metrics=m,donor_effect_fraction=(m['margin']-base[sex]['margin'])/(base[donor]['margin']-base[sex]['margin'])))
    save_json(OUT/'joint_patches.json',joint)
    # Same selected features/path on unseen wording and protected task families.
    selected=feature_order[:4]
    templates=[('ruler','A {sex} ruler is called a','queen','king'),
        ('sovereign','A {sex} sovereign is called a','queen','king'),
        ('the_monarch','The {sex} monarch is called a','queen','king'),
        ('known_as','A {sex} monarch is known as a','queen','king'),
        ('title','The title of a {sex} monarch is','queen','king'),
        ('parent_control','A {sex} parent is called a','mother','father'),
        ('sibling_control','A {sex} sibling is called a','sister','brother'),
        ('child_control','A {sex} child is called a','girl','boy')]
    validation=[]
    with torch.no_grad():
        for name,template,fw,mw in templates:
            tt={s:model.to_tokens(template.format(sex=s),prepend_bos=False) for s in ['female','male']}
            assert tt['female'].shape==tt['male'].shape
            changed=(tt['female']!=tt['male']).nonzero().tolist();assert len(changed)==1
            src=changed[0][1];query=tt['female'].shape[1]-1
            cc={};zs={};ls={};ms={}
            taskids=dict(ids)
            for w in [fw,mw]:
                wi=model.to_tokens(' '+w,prepend_bos=False).flatten().tolist();assert len(wi)==1
                taskids[' '+w]=wi[0]
            for sex in ['female','male']:
                ls[sex],cc[sex]=model.run_with_cache(tt[sex]);zs[sex]=encode(sae,cc[sex][ln][0])
                ms[sex]=metrics(ls[sex],model,taskids)
            for sex in ['female','male']:
                donor='male' if sex=='female' else 'female'
                for mode,alpha in [('baseline',0),('full_content',1),('top4',1),('top4',4)]:
                    a=cc[sex][pk][0,H,query,src]
                    if mode=='full_content':d=a*(cc[donor][vk][0,src,H]-cc[sex][vk][0,src,H])
                    else:d=alpha*a*((zs[donor]['a'][src,selected]-zs[sex]['a'][src,selected])@sae.W_dec[selected]@model.blocks[L].attn.W_V[H])
                    def hook(x,hook,d=d):
                        out=x.clone();out[:,query,H]+=d;return out
                    out=model.run_with_hooks(tt[sex],fwd_hooks=[(zk,hook)])
                    m=metrics(out,model,taskids)
                    task_margin=float(out[0,-1,taskids[' '+fw]]-out[0,-1,taskids[' '+mw]])
                    base_margin=float(ls[sex][0,-1,taskids[' '+fw]]-ls[sex][0,-1,taskids[' '+mw]])
                    donor_margin=float(ls[donor][0,-1,taskids[' '+fw]]-ls[donor][0,-1,taskids[' '+mw]])
                    royal=ids[' queen'];king=ids[' king'];woman=ids[' woman'];man=ids[' man']
                    royalty=float(torch.logsumexp(out[0,-1,[royal,king]],0)-torch.logsumexp(out[0,-1,[woman,man]],0))
                    validation.append(dict(example=name,prompt=template.format(sex=sex),base=sex,mode=mode,alpha=alpha,
                        source=src,query=query,features=selected,metrics=m,task_words=[fw,mw],task_margin=task_margin,
                        delta_task_margin=task_margin-base_margin,task_donor_fraction=(task_margin-base_margin)/(donor_margin-base_margin),
                        royalty_log_odds=royalty,source_active_selected=[i for i in selected if float(zs[sex]['z'][src,i])>0]))
            print('VALIDATED',name,flush=True)
    save_json(OUT/'validation.json',validation)
    # Fetch annotations only after numerical feature selection/outcomes; labels
    # are descriptive and never used to select which feature gets a causal test.
    wanted=set(feature_order)|{r['query_feature'] for r in qk}|{r['key_feature'] for r in qk}
    desc={};sources=[]
    def batch(b):
        url=BUCKET+f'v1/gpt2-small/{L-1}-res_post_32k-oai/explanations/batch-{b}.jsonl.gz'
        raw=fetch(url);ds={}
        for line in gzip.decompress(raw).decode().splitlines():
            x=json.loads(line)
            if int(x['index']) in wanted:ds[int(x['index'])]=x['description']
        return ds,dict(url=url,sha256=hashlib.sha256(raw).hexdigest())
    with ThreadPoolExecutor(max_workers=4) as pool:
        for ds,source in pool.map(batch,range(32)):desc.update(ds);sources.append(source)
    save_json(OUT/'path_feature_labels.json',dict(descriptions=desc,sources=sources))
    save_json(OUT/'resolution_manifest.json',dict(path=path,input_sae=RELEASE,input_sae_original_layer=L-1,
        measured_at=ln,input_sae_transferred=True,sae_config=sae.cfg.to_dict(),feature_ranking_frozen_before_causal_runs=True,
        top_features=selected,elapsed_seconds=time.time()-started,
        feature_runs=len(records),validation_runs=len(validation),joint_runs=len(joint),qk_runs=len(qk),
        sae_files=checkpoint_manifest('jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs'),
        caveat='Unit edits substitute donor coefficients on one OV message. Alpha >1 extrapolates beyond donor, may make effective coefficients negative, and is steering rather than feature deletion. Protected tasks use the same ungated head/source/final-query rule; no FRA selectivity advantage is assumed.'))
    print('DONE',round(time.time()-started,1),'seconds',flush=True)

if __name__=='__main__':main()
