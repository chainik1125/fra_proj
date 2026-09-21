"""Head-to-head FRA/SAE comparison with common feature IDs and source masks."""
import json,time,hashlib,importlib.metadata
from trace_baseline import ROOT,TARGETS,save_json
from resolve_gender_transport import encode,RELEASE
from localize_gender_transport import metrics
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

PARENT=ROOT/'gender_transport';OUT=PARENT/'matched_scopes'
L,H,K,Q=8,11,1,5
LN=f'blocks.{L}.ln1.hook_normalized';RAW=f'blocks.{L-1}.hook_resid_post'
ZK=f'blocks.{L}.attn.hook_z';PK=f'blocks.{L}.attn.hook_pattern';VK=f'blocks.{L}.attn.hook_v'
TOP=[25975,20446,15560,3281]
FEATURE_SETS={'four':TOP,'single_15560':[15560]}
SCOPES=['source','prediction','source_and_prediction','prefix','sentence']
METHODS=['fra_h11','fra_all_heads','sae_residual']
GRID=[0,.25,.5,1,2,4,8,16]
TAILS=[' and usually rules a kingdom.',' and lives in a royal palace.',' and performs official duties.']

@torch.inference_mode()
def main():
    started=time.time();torch.set_num_threads(4);torch.manual_seed(0);OUT.mkdir(exist_ok=True)
    save_json(OUT/'protocol.json',dict(model='gpt2-small',layer=L,head=H,source=K,query=Q,
        main_comparison='Same feature IDs, same edited source-token set. All causal destination queries allowed for FRA; no extra destination mask.',
        methods=METHODS,feature_sets=FEATURE_SETS,source_scopes={'source':[1],'prediction':[5],'source_and_prediction':[1,5],'prefix':'0..5','sentence':'all token positions'},
        extra_controls=['original_edge: source 1 to destination 5 only, head 11','all_to_prediction: every source to destination 5 only, head 11'],
        strengths={'four':[.5,1.0],'single_15560':[.5]},alpha_grid=GRID,bisection_steps=13,
        target='logit(queen)-logit(king) = -magnitude on female prefix and +magnitude on male prefix. Match odds, not absolute probability mass.',
        common_coefficient_edit='For every chosen feature at each allowed source token, delta_z = alpha * (z_donor-z_base). Multiply by the BASE native SAE std. Preserve reconstruction error, mean, and decoder bias.',
        fra_edit='delta_head_z[q,h] = sum_{k in S,k<=q} A_base[h,q,k] * (std_base[k] sum_i delta_z[k,i]D_i) Wv[h]. At all destinations this equals a V-only edit at the chosen source positions.',
        sae_edit='delta_resid[k] = std_base[k] sum_i delta_z[k,i]D_i at post-layer 7, for k in the SAME set S.',
        sae_sites='Post-layer-7 pretrained SAE for residual edits; same normalized dictionary applied at layer-8 ln1 for FRA. Coefficient/support transfer is audited on every input.',
        tails=TAILS,forced_answers=['queen','king'],source_genders=['female','male'],
        evaluation='Full-vocabulary KL(clean||edited) in float64, nats. Exclude prediction of queen/king; distinguish before/after answer. Identical forced tokens for clean and edited. Donor differs only at source gender token.',
        aggregation='Equal weighting of 12 full-sentence cases. Report directions separately. Pooled comparative ratios only if both directions reach the requested effect.',
        unreached='No crossing on fixed alpha grid through 16 => unmatched, never a zero-collateral success.',
        limits='All positions does not mean all layers. Head-11 and all-head OV edits are distinguished. No QK or concept-dependent destination gate. Extrapolated coefficients may be negative. Three constructed tails, not free generation.'))
    model=HookedTransformer.from_pretrained('gpt2-small',device='cpu').eval()
    sae=SAE.from_pretrained(RELEASE,RAW,device='cpu').eval()
    ids={w:model.to_tokens(w,prepend_bos=False).item() for w in TARGETS}
    transfer=[];checks=[];calls=0
    def prepare(text):
        tok=model.to_tokens(text,prepend_bos=False)
        logits,cache=model.run_with_cache(tok,names_filter=[LN,RAW,PK,VK])
        ov=encode(sae,cache[LN][0]);raw=encode(sae,cache[RAW][0])
        support=int(((ov['z']>0)!=(raw['z']>0)).any(-1).sum())
        coefficient_err=float((ov['z']-raw['z']).abs().max())
        assert support==0 and coefficient_err<.001
        transfer.append(dict(text=text,support_changes=support,max_coefficient_difference=coefficient_err))
        return dict(text=text,tokens=tok,logits=logits,cache=cache,ov=ov,raw=raw)
    base={s:prepare(f'A {s} monarch is called a') for s in ['female','male']}
    save_json(OUT/'baseline.json',{s:metrics(c['logits'],model,ids) for s,c in base.items()})
    def source_positions(scope,N):
        if scope=='source':return [K]
        if scope=='prediction':return [Q]
        if scope=='source_and_prediction':return [K,Q]
        if scope=='prefix':return list(range(Q+1))
        if scope=='sentence':return list(range(N))
        raise ValueError(scope)
    def kl(clean,edited):
        lp=clean[0].double().log_softmax(-1);lq=edited[0].double().log_softmax(-1)
        out=(lp.exp()*(lp-lq)).sum(-1);assert float(out.min())>-1e-10
        return out.clamp_min(0)
    def construct(c,d,method,scope,ff,alpha):
        N=c['tokens'].shape[1];S=source_positions(scope,N);is_resid=method=='sae_residual'
        code=c['raw' if is_resid else 'ov'];donor=d['raw' if is_resid else 'ov']
        values=code['z'][:,ff];difference=donor['z'][:,ff]-values
        masked=torch.zeros_like(difference);masked[S]=difference[S]
        effective=values+alpha*masked
        dx=alpha*code['std']*(masked@sae.W_dec[ff])
        if is_resid:
            def hook(x,hook):return x+dx.unsqueeze(0)
            hooks=[(RAW,hook)];delta_v=None;delta_z=None
        else:
            delta_v=torch.einsum('kd,hde->khe',dx,model.blocks[L].attn.W_V)
            if method!='fra_all_heads':
                vv=torch.zeros_like(delta_v);vv[:,H]=delta_v[:,H];delta_v=vv
            delta_z=torch.einsum('hqk,khd->qhd',c['cache'][PK][0],delta_v)
            if method in ['original_edge','all_to_prediction']:
                zz=torch.zeros_like(delta_z);zz[Q]=delta_z[Q];delta_z=zz
            def hook(x,hook):return x+delta_z.unsqueeze(0)
            hooks=[(ZK,hook)]
        info=dict(negative_effective_coefficient=bool((effective< -1e-6).any()),min_effective_coefficient=float(effective.min()),
            source_positions=S,nonzero_source_delta_positions=(dx.norm(dim=-1)>0).nonzero().flatten().tolist(),
            direct_output_positions=(delta_z.flatten(1).norm(dim=-1)>0).nonzero().flatten().tolist() if delta_z is not None else (dx.norm(dim=-1)>0).nonzero().flatten().tolist())
        return hooks,info,delta_v,delta_z
    def run(c,d,method,scope,ff,alpha):
        nonlocal calls
        hooks,info,_,_=construct(c,d,method,scope,ff,alpha)
        out=model.run_with_hooks(c['tokens'],fwd_hooks=hooks);calls+=1
        if alpha==0 or not info['nonzero_source_delta_positions']:assert torch.equal(out,c['logits'])
        first=min(info['direct_output_positions'],default=c['tokens'].shape[1])
        assert torch.equal(out[:,:first],c['logits'][:,:first])
        return out,info
    # Independent implementation checks: all-destination FRA equals a native
    # value-hook edit, and masking one edge equals the earlier analytic formula.
    for sex,c in base.items():
        d=base['male' if sex=='female' else 'female']
        for method in ['fra_h11','fra_all_heads']:
            for scope in ['source','sentence']:
                hooks,info,dv,dz=construct(c,d,method,scope,TOP,.7)
                out=model.run_with_hooks(c['tokens'],fwd_hooks=hooks);calls+=1
                def vh(x,hook,dv=dv):return x+dv.unsqueeze(0)
                direct=model.run_with_hooks(c['tokens'],fwd_hooks=[(VK,vh)]);calls+=1
                err=float((direct-out).abs().max());assert err<1e-4
                checks.append(dict(check='FRA_all_destinations_equals_direct_V_hook',base=sex,method=method,scope=scope,max_logit_error=err))
        _,_,_,dz=construct(c,d,'original_edge','source',TOP,1)
        old_difference=d['ov']['a'][K,TOP]-c['ov']['a'][K,TOP]
        old=c['cache'][PK][0,H,Q,K]*(old_difference@sae.W_dec[TOP]@model.blocks[L].attn.W_V[H])
        err=float((old-dz[Q,H]).abs().max());assert err<1e-4
        checks.append(dict(check='base_scale_vs_prior_donor_scale_edge_delta',base=sex,max_message_error=err))
    specs=[]
    for fs in FEATURE_SETS:
        for method in METHODS:
            for scope in SCOPES:specs.append(dict(feature_set=fs,method=method,scope=scope))
        specs.extend([dict(feature_set=fs,method='original_edge',scope='source'),dict(feature_set=fs,method='all_to_prediction',scope='sentence')])
    memo={};scan=[];cal=[]
    def evaluate(sex,spec,alpha):
        # Prefix and whole-sentence masks coincide during six-token calibration.
        scope='prefix' if spec['scope']=='sentence' else spec['scope']
        key=(sex,spec['method'],scope,spec['feature_set'],alpha)
        if key in memo:return memo[key]
        c=base[sex];d=base['male' if sex=='female' else 'female']
        out,info=run(c,d,spec['method'],scope,FEATURE_SETS[spec['feature_set']],alpha)
        k=kl(c['logits'],out);m=metrics(out,model,ids)
        row=dict(base=sex,method=spec['method'],scope=scope,feature_set=spec['feature_set'],alpha=alpha,metrics=m,
                 before_answer_kl_mean=float(k[:Q].mean()),answer_kl=float(k[Q]),**info)
        memo[key]=row;scan.append(row);return row
    def calibrate(sex,spec,magnitude):
        target=magnitude if sex=='male' else -magnitude;sign=1 if sex=='male' else -1
        rows=[evaluate(sex,spec,a) for a in GRID]
        indices=[i for i,r in enumerate(rows) if sign*(r['metrics']['margin']-target)>=0]
        if not indices:return dict(base=sex,**spec,target_margin=target,matched=False,reason='No target crossing on fixed grid through alpha=16.')
        i=indices[0];lo,hi=GRID[i-1],GRID[i]
        for _ in range(13):
            mid=(lo+hi)/2;r=evaluate(sex,spec,mid)
            if sign*(r['metrics']['margin']-target)>=0:hi=mid
            else:lo=mid
        r=evaluate(sex,spec,(lo+hi)/2)
        assert abs(r['metrics']['margin']-target)<.001
        return dict(r,**spec,target_margin=target,matched=True,alpha_bracket=[lo,hi],match_error=r['metrics']['margin']-target)
    for sex in ['female','male']:
        for spec in specs:
            for magnitude in ([.5,1.] if spec['feature_set']=='four' else [.5]):
                r=calibrate(sex,spec,magnitude);cal.append(r)
                print('CAL',sex,spec['feature_set'],spec['method'],spec['scope'],magnitude,round(r.get('alpha',0),4),r['matched'],flush=True)
            save_json(OUT/'calibration.json',cal);save_json(OUT/'calibration_scan.json',scan)
    records=[]
    for answer in ['queen','king']:
        for ti,tail in enumerate(TAILS):
            full={s:prepare(f'A {s} monarch is called a {answer}{tail}') for s in ['female','male']}
            for sex,c in full.items():
                d=full['male' if sex=='female' else 'female'];N=c['tokens'].shape[1]
                assert torch.equal(c['tokens'][:,:Q+1],base[sex]['tokens'])
                assert int(c['tokens'][0,Q+1])==ids[' '+answer]
                base_err=float((c['logits'][:,Q]-base[sex]['logits'][:,Q]).abs().max());assert base_err<1e-4
                pre=list(range(Q));post=list(range(Q+1,N-1));rest=pre+post
                for r in [x for x in cal if x['base']==sex and x['matched']]:
                    out,info=run(c,d,r['method'],r['scope'],FEATURE_SETS[r['feature_set']],r['alpha'])
                    margin=float(out[0,Q,ids[' queen']]-out[0,Q,ids[' king']]);assert abs(margin-r['target_margin'])<.0011
                    k=kl(c['logits'],out);p=out[0,Q].float().softmax(-1)
                    positions=[dict(query=j,input_token=model.tokenizer.decode([int(c['tokens'][0,j])]),predicted_token=model.tokenizer.decode([int(c['tokens'][0,j+1])]),
                        region='before' if j<Q else 'answer' if j==Q else 'after',kl=float(k[j])) for j in range(N-1)]
                    records.append(dict(base=sex,method=r['method'],scope=r['scope'],feature_set=r['feature_set'],features=FEATURE_SETS[r['feature_set']],
                        alpha=r['alpha'],target_margin=r['target_margin'],actual_margin=margin,p_queen=float(p[ids[' queen']]),p_king=float(p[ids[' king']]),
                        forced_answer=answer,answer_agrees_with_source=(answer=='queen')==(sex=='female'),tail_index=ti,text=c['text'],
                        before_answer_kl_mean=float(k[pre].mean()),after_answer_kl_mean=float(k[post].mean()),
                        rest_kl_mean=float(k[rest].mean()),rest_kl_sum=float(k[rest].sum()),rest_tokens=len(rest),
                        answer_kl=float(k[Q]),per_position=positions,**info))
            save_json(OUT/'continuations.json',records)
            print('TAIL',answer,ti,len(records),flush=True)
    summary=[]
    for spec in specs:
        for magnitude in ([.5,1.] if spec['feature_set']=='four' else [.5]):
            for sex in ['female','male','both']:
                for agrees in [None,True,False]:
                    rr=[r for r in records if all(r[k]==v for k,v in spec.items()) and abs(r['target_margin'])==magnitude and (sex=='both' or r['base']==sex) and (agrees is None or r['answer_agrees_with_source']==agrees)]
                    if not rr:continue
                    groups=sorted(set(r['base'] for r in rr))
                    summary.append(dict(**spec,magnitude=magnitude,base=sex,answer_agrees_with_source=agrees,n_sentences=len(rr),directions=groups,
                        complete_directions=sex!='both' or len(groups)==2,
                        **{k:sum(r[k] for r in rr)/len(rr) for k in ['before_answer_kl_mean','after_answer_kl_mean','rest_kl_mean','answer_kl','p_queen','p_king']},
                        rest_kl_token_weighted=sum(r['rest_kl_sum'] for r in rr)/sum(r['rest_tokens'] for r in rr)))
    save_json(OUT/'summary.json',summary)
    save_json(OUT/'checks.json',dict(algebra_checks=checks,ln1_transfer_checks=transfer,
        max_target_match_error=max(abs(r['actual_margin']-r['target_margin']) for r in records),zero_edits_exact_noops=True,
        no_effect_before_first_direct_write=True,teacher_forced_prefix_consistency=True))
    save_json(OUT/'manifest.json',dict(causal_forward_calls=calls,calibration_records=len(cal),matched_calibrations=sum(r['matched'] for r in cal),
        continuation_records=len(records),elapsed_seconds=time.time()-started,sae_config=sae.cfg.to_dict(),
        packages={n:importlib.metadata.version(n) for n in ['torch','transformer-lens','sae-lens']},
        upstream_manifest_sha256=hashlib.sha256((PARENT/'resolution_manifest.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()))
    print('DONE',calls,len(records),round(time.time()-started,1),'seconds',flush=True)

if __name__=='__main__':main()
