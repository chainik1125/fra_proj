"""Matched-target KL on other positions, with fixed teacher-forced continuations."""
import json,time,hashlib,math
from trace_baseline import ROOT,TARGETS,save_json
from resolve_gender_transport import encode,RELEASE
from localize_gender_transport import metrics
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

PARENT=ROOT/'gender_transport'
OUT=PARENT/'matched_kl'
L,H,K,Q=8,11,1,5
LN=f'blocks.{L}.ln1.hook_normalized';RAW=f'blocks.{L-1}.hook_resid_post'
ZK=f'blocks.{L}.attn.hook_z';PK=f'blocks.{L}.attn.hook_pattern'
TOP=[25975,20446,15560,3281]
GRID=[0,0.25,0.5,1,2,4,8,16]
TARGETS_MAG=[0.5,1.0]
TAILS=[' and usually rules a kingdom.',' and lives in a royal palace.',' and performs official duties.']
METHODS=['fra_four','fra_single','sae_source_single','sae_prefix_single','sae_query_single','sae_sentence_single']

@torch.inference_mode()
def main():
    started=time.time();torch.set_num_threads(4);torch.manual_seed(0);OUT.mkdir(exist_ok=True)
    save_json(OUT/'protocol.json',dict(model='gpt2-small',layer=L,head=H,source=K,query=Q,
        source_prompts=['A female monarch is called a','A male monarch is called a'],
        common_margin_targets={'female':[-x for x in TARGETS_MAG],'male':TARGETS_MAG},
        rationale='Match the edited queen-minus-king logit margin, not alpha or an arbitrarily small inversion.',
        methods=METHODS,top_four=TOP,single_additions={'female':25975,'male':20446},broad_single_feature=15560,
        query_control_candidates=TOP,query_selection='If several of the four features reach the target, choose the one with lowest mean non-answer-prefix KL at matched effect. Ties use earliest candidate order. Later continuation outcomes never enter selection.',
        alpha_grid=GRID,bisection_steps=13,tails=TAILS,
        teacher_forcing='Each tail is tested with both queen and king supplied after the prefix, for each source gender. Base, edited and donor forward passes use the same answer/tail; donor differs only at the source gender word.',
        scopes={'fra_four':'Four source-feature OV contributions to only L8H11 q=5,k=1.',
          'fra_single':'One source-feature OV contribution to only L8H11 q=5,k=1.',
          'sae_source_single':'Single-feature coefficient substitution at source position 1 of post-L7 residual.',
          'sae_prefix_single':'Feature 15560 substitution at positions 0..5 only; fixed original six-token scope.',
          'sae_query_single':'Single-feature substitution at position 5 only; position-matched control.',
          'sae_sentence_single':'Feature 15560 substitution at every position of the full teacher-forced sentence.'},
        residual_edit='x + alpha * std_base * (z_donor-z_base) D; retain original decoder bias, mean and reconstruction error.',
        ov_edit='head_z[q,h] + alpha * A_base[h,q,k] * sum((a_donor-a_base)D Wv); a=std*z; retain base attention.',
        kl='KL(p_clean(.|same forced history) || p_edited(.|same forced history)), float64, natural logarithms.',
        prefix_positions=list(range(Q)),answer_prediction_position=Q,
        suffix_positions='6 through N-2 inclusive: predictions of supplied words after king/queen, including punctuation.',
        rest_positions='All predictions of supplied tokens except the answer prediction at position 5; initial token has no previous model prediction.',
        limits='Two primary prefixes and three hand-written continuations, not a corpus or free-running sentence KL. Extrapolated edits may imply negative effective coefficients.'))
    model=HookedTransformer.from_pretrained('gpt2-small',device='cpu').eval()
    sae=SAE.from_pretrained(RELEASE,RAW,device='cpu').eval()
    ids={w:model.to_tokens(w,prepend_bos=False).item() for w in TARGETS}
    calls=0;checks=[]
    def prepare(text):
        tok=model.to_tokens(text,prepend_bos=False)
        log,ca=model.run_with_cache(tok,names_filter=[LN,RAW,PK])
        return dict(text=text,tokens=tok,logits=log,cache=ca,ov=encode(sae,ca[LN][0]),raw=encode(sae,ca[RAW][0]))
    cases={s:prepare(f'A {s} monarch is called a') for s in ['female','male']}
    for s,c in cases.items():assert c['tokens'].shape==(1,6)
    save_json(OUT/'baseline.json',{s:metrics(c['logits'],model,ids) for s,c in cases.items()})
    def features_for(s,method,override=None):
        if override is not None:return override
        if method=='fra_four':return TOP
        if method in ['fra_single','sae_source_single']:return [25975 if s=='female' else 20446]
        return [15560]
    def kl(log0,log1):
        lp=log0[0].double().log_softmax(-1);lq=log1[0].double().log_softmax(-1)
        k=(lp.exp()*(lp-lq)).sum(-1)
        assert float(k.min())>-1e-10
        return k.clamp_min(0)
    def edit(c,d,method,ff,alpha):
        nonlocal calls
        if method.startswith('fra_'):
            vals=c['ov']['a'][K,ff];diff=d['ov']['a'][K,ff]-vals
            message=alpha*c['cache'][PK][0,H,Q,K]*(diff@sae.W_dec[ff]@model.blocks[L].attn.W_V[H])
            def hook(x,hook):
                y=x.clone();y[:,Q,H]+=message;return y
            hooks=[(ZK,hook)];effective=vals+alpha*diff
        else:
            vals=c['raw']['z'][:,ff];diff=d['raw']['z'][:,ff]-vals
            masked=torch.zeros_like(diff)
            if method=='sae_source_single':masked[K]=diff[K]
            elif method=='sae_query_single':masked[Q]=diff[Q]
            elif method=='sae_prefix_single':masked[:Q+1]=diff[:Q+1]
            elif method=='sae_sentence_single':masked=diff
            else:raise ValueError(method)
            effective=vals+alpha*masked
            message=alpha*c['raw']['std']*(masked@sae.W_dec[ff])
            def hook(x,hook):return x+message.unsqueeze(0)
            hooks=[(RAW,hook)]
        out=model.run_with_hooks(c['tokens'],fwd_hooks=hooks);calls+=1
        if method.startswith('fra_') or method=='sae_query_single':assert torch.equal(out[:,:Q],c['logits'][:,:Q])
        if alpha==0 or float(message.norm())==0:assert torch.equal(out,c['logits'])
        return out,dict(negative_effective_coefficient=bool((effective < -1e-6).any()),min_effective_coefficient=float(effective.min()))
    calibrated=[];scan=[];memo={}
    def evaluate(sex,method,ff,alpha):
        key=(sex,method,tuple(ff),alpha)
        if key in memo:return memo[key]
        c=cases[sex];d=cases['male' if sex=='female' else 'female']
        out,info=edit(c,d,method,ff,alpha);k=kl(c['logits'],out)
        margin=float(out[0,Q,ids[' queen']]-out[0,Q,ids[' king']])
        row=dict(base=sex,method=method,features=ff,alpha=alpha,margin=margin,
            target_distribution_kl=float(k[Q]),prefix_other_kl_mean=float(k[:Q].mean()),
            metrics=metrics(out,model,ids),**info)
        scan.append(row);memo[key]=row;return row
    def calibrate(sex,method,ff,magnitude):
        target=magnitude if sex=='male' else -magnitude
        sign=1 if sex=='male' else -1
        rr=[evaluate(sex,method,ff,a) for a in GRID]
        indices=[i for i,r in enumerate(rr) if sign*(r['margin']-target)>=0]
        if not indices:return dict(base=sex,method=method,features=ff,target_margin=target,matched=False,reason='No target crossing on alpha grid through 16.')
        i=indices[0];lo=GRID[i-1];hi=GRID[i]
        for _ in range(13):
            mid=(lo+hi)/2;r=evaluate(sex,method,ff,mid)
            if sign*(r['margin']-target)>=0:hi=mid
            else:lo=mid
        a=(lo+hi)/2;r=evaluate(sex,method,ff,a)
        assert abs(r['margin']-target)<1e-3
        return dict(**r,target_margin=target,matched=True,alpha_bracket=[lo,hi],match_error=r['margin']-target)
    for sex in ['female','male']:
        for magnitude in TARGETS_MAG:
            for method in METHODS:
                if method=='sae_sentence_single':
                    found=next(r for r in calibrated if r['base']==sex and r['method']=='sae_prefix_single' and abs(r['target_margin'])==magnitude)
                    row=dict(found,method=method,calibration_alias='sae_prefix_single; identical on six-token calibration prefix')
                elif method=='sae_query_single':
                    options=[calibrate(sex,method,[f],magnitude) for f in TOP]
                    eligible=[r for r in options if r['matched']]
                    if eligible:
                        row=dict(min(eligible,key=lambda r:r['prefix_other_kl_mean']),selection='Lowest calibration-prefix non-answer KL among reachable single features; all query-local edits have zero prefix collateral.')
                    else:row=dict(base=sex,method=method,target_margin=magnitude if sex=='male' else -magnitude,matched=False,reason='No candidate reaches target.')
                    save_json(OUT/f'query_candidates_{sex}_{magnitude}.json',options)
                else:row=calibrate(sex,method,features_for(sex,method),magnitude)
                calibrated.append(row)
                print('CALIBRATE',sex,method,magnitude,row.get('alpha'),row.get('features'),row.get('matched'),flush=True)
            save_json(OUT/'calibration.json',calibrated);save_json(OUT/'calibration_scan.json',scan)
    records=[]
    for answer in ['queen','king']:
        for ti,tail in enumerate(TAILS):
            full={s:prepare(f'A {s} monarch is called a {answer}{tail}') for s in ['female','male']}
            for sex,c in full.items():
                donor='male' if sex=='female' else 'female';d=full[donor]
                assert torch.equal(c['tokens'][:,:Q+1],cases[sex]['tokens'])
                # Floating-point batch/sequence kernels need not be bitwise identical.
                base_err=float((c['logits'][:,Q]-cases[sex]['logits'][:,Q]).abs().max())
                assert base_err<1e-4
                N=c['tokens'].shape[1];prefix_idx=list(range(Q));suffix_idx=list(range(Q+1,N-1));rest_idx=prefix_idx+suffix_idx
                assert suffix_idx and int(c['tokens'][0,Q+1])==ids[' '+answer]
                for cal in [r for r in calibrated if r['base']==sex and r['matched']]:
                    out,info=edit(c,d,cal['method'],cal['features'],cal['alpha'])
                    margin=float(out[0,Q,ids[' queen']]-out[0,Q,ids[' king']])
                    assert abs(margin-cal['target_margin'])<0.0011
                    k=kl(c['logits'],out)
                    tok=c['tokens'][0];lp=c['logits'][0].double().log_softmax(-1);lq=out[0].double().log_softmax(-1)
                    positions=[]
                    for j in range(N-1):
                        target_id=int(tok[j+1]);region='before_answer' if j<Q else 'answer' if j==Q else 'after_answer'
                        positions.append(dict(query_position=j,input_token=model.tokenizer.decode([int(tok[j])]),
                            predicted_token=model.tokenizer.decode([target_id]),region=region,kl=float(k[j]),
                            clean_logp_actual=float(lp[j,target_id]),edited_logp_actual=float(lq[j,target_id]),
                            delta_nll_actual=float(lp[j,target_id]-lq[j,target_id])))
                    r=dict(base=sex,prompt=c['text'],forced_answer=answer,answer_agrees_with_source=(answer=='queen')==(sex=='female'),
                        tail_index=ti,method=cal['method'],features=cal['features'],alpha=cal['alpha'],target_margin=cal['target_margin'],
                        actual_margin=margin,target_distribution_kl=float(k[Q]),
                        before_answer_kl_mean=float(k[prefix_idx].mean()),after_answer_kl_mean=float(k[suffix_idx].mean()),
                        rest_kl_mean=float(k[rest_idx].mean()),rest_kl_sum=float(k[rest_idx].sum()),
                        rest_token_count=len(rest_idx),after_answer_token_count=len(suffix_idx),
                        rest_delta_nll_actual_mean=sum(p['delta_nll_actual'] for p in positions if p['region']!='answer')/len(rest_idx),
                        per_position=positions,**info)
                    records.append(r)
                checks.append(dict(base=sex,answer=answer,tail_index=ti,prefix_full_vs_prefix_only_max_logit_error=base_err))
            save_json(OUT/'continuations.json',records)
            print('TAIL',answer,ti,'done',flush=True)
    summary=[]
    for magnitude in TARGETS_MAG:
        for method in METHODS:
            for sex in ['female','male','both']:
                for agrees in [True,False,None]:
                    rr=[r for r in records if abs(r['target_margin'])==magnitude and r['method']==method and (sex=='both' or r['base']==sex) and (agrees is None or r['answer_agrees_with_source']==agrees)]
                    if not rr:continue
                    summary.append(dict(magnitude=magnitude,method=method,base=sex,answer_agrees_with_source=agrees,n_sentences=len(rr),
                        before_answer_kl_mean=sum(r['before_answer_kl_mean'] for r in rr)/len(rr),
                        after_answer_kl_mean=sum(r['after_answer_kl_mean'] for r in rr)/len(rr),
                        rest_kl_mean=sum(r['rest_kl_mean'] for r in rr)/len(rr),
                        rest_kl_token_weighted=sum(r['rest_kl_sum'] for r in rr)/sum(r['rest_token_count'] for r in rr),
                        target_distribution_kl_mean=sum(r['target_distribution_kl'] for r in rr)/len(rr)))
    save_json(OUT/'summary.json',summary)
    save_json(OUT/'checks.json',dict(teacher_forced_prefix_checks=checks,max_target_match_error=max(abs(r['actual_margin']-r['target_margin']) for r in records),
        same_tokens_for_clean_and_edited=True,ov_and_query_control_earlier_logits_exactly_unchanged=True,zero_edits_exact_noops=True))
    save_json(OUT/'manifest.json',dict(elapsed_seconds=time.time()-started,causal_forward_calls=calls,
        calibration_records=len(calibrated),continuation_records=len(records),sae_config=sae.cfg.to_dict(),
        upstream_manifest_sha256=hashlib.sha256((PARENT/'resolution_manifest.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()))
    print('DONE',calls,round(time.time()-started,1),'seconds',flush=True)

if __name__=='__main__':main()
