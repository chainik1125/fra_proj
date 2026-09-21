"""Compare single-feature edits with the four-feature OV intervention.

No training. Retain original SAE errors. Source-feature residual edits use the
base token's native decoder scale and mean, as in ordinary SAE interventions.
"""
import json,time,hashlib
from trace_baseline import ROOT,TARGETS,save_json
from resolve_gender_transport import encode,RELEASE
from localize_gender_transport import metrics
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

PARENT=ROOT/'gender_transport'
OUT=PARENT/'single_feature_comparison'
METHODS=['ov_edge','residual_source','residual_all_positions']
GRID=[0,0.25,0.5,1,2,4,8,16]

@torch.inference_mode()
def main():
    started=time.time();torch.set_num_threads(4);torch.manual_seed(0)
    OUT.mkdir(exist_ok=True)
    rank=json.loads((PARENT/'feature_ranking.json').read_text())['ranking']
    candidates=[r['feature'] for r in rank];top=candidates[:4]
    path=json.loads((PARENT/'selected_path.json').read_text())['selected']
    L,H,Q,K=[path[k] for k in ['layer','head','query','key']]
    ln=f'blocks.{L}.ln1.hook_normalized';raw=f'blocks.{L-1}.hook_resid_post'
    zk=f'blocks.{L}.attn.hook_z';pk=f'blocks.{L}.attn.hook_pattern'
    save_json(OUT/'protocol.json',dict(path=path,candidates=candidates,top_four=top,methods=METHODS,grid=GRID,
        single_unit_scan='All 40 nonzero source coefficient-difference features from the previous experiment; donor substitution and deletion, both directions, all three methods.',
        strength_scan='Prior four highest-ranked features, individually and jointly. First sign change on the fixed grid is refined with 10 bisections; no global monotonicity or optimality claim.',
        ov_definition='alpha * A_base * sum((a_donor-a_base) D Wv), preserving base attention. Deletion uses -a_base instead.',
        residual_definition='x_base + alpha * std_base * sum((z_donor-z_base) D), preserving base normalization mean, decoder bias and reconstruction error. Deletion uses -z_base.',
        caveat='Alpha >1 is extrapolation, not deletion. Removing an active feature beyond zero gives a negative effective coefficient. Scopes differ; strength is not a matched-norm or collateral comparison.'))
    model=HookedTransformer.from_pretrained('gpt2-small',device='cpu').eval()
    sae=SAE.from_pretrained(RELEASE,raw,device='cpu').eval()
    ids={w:model.to_tokens(w,prepend_bos=False).item() for w in TARGETS}
    toks={};cache={};logits={};codes={};baselines={}
    for sex in ['female','male']:
        toks[sex]=model.to_tokens(f'A {sex} monarch is called a',prepend_bos=False)
        logits[sex],cache[sex]=model.run_with_cache(toks[sex])
        codes[sex]={'ov':encode(sae,cache[sex][ln][0]),'raw':encode(sae,cache[sex][raw][0])}
        baselines[sex]=metrics(logits[sex],model,ids)
    prior=json.loads((PARENT/'feature_results.json').read_text())
    rows=[];memo={};thresholds=[];max_prior_error=0;checks=[]
    def run(sex,method,operation,ff,alpha,stage):
        key=(sex,method,operation,tuple(ff),float(alpha))
        if key in memo:return memo[key]
        donor='male' if sex=='female' else 'female'
        c=codes[sex]['ov' if method=='ov_edge' else 'raw'];d=codes[donor]['ov' if method=='ov_edge' else 'raw']
        if method=='ov_edge':
            vals=c['a'][K,ff];delta=(d['a'][K,ff]-vals) if operation=='donor' else -vals
            effective=vals+alpha*delta
            message=alpha*cache[sex][pk][0,H,Q,K]*(delta@sae.W_dec[ff]@model.blocks[L].attn.W_V[H])
            def hook(x,hook):
                out=x.clone();out[:,Q,H]+=message;return out
            hooks=[(zk,hook)]
            edit_norm=float((message@model.blocks[L].attn.W_O[H]).norm())
            edited_positions=[K] if float(message.norm())>0 else []
        else:
            vals=c['z'][:,ff];delta=(d['z'][:,ff]-vals) if operation=='donor' else -vals
            if method=='residual_source':
                selected=torch.zeros_like(delta);selected[K]=delta[K];delta=selected
            effective=vals+alpha*delta
            message=alpha*c['std']*(delta@sae.W_dec[ff])
            def hook(x,hook):return x+message.unsqueeze(0)
            hooks=[(raw,hook)]
            edit_norm=float(message.norm())
            edited_positions=(message.norm(dim=-1)>0).nonzero().flatten().tolist()
        out=model.run_with_hooks(toks[sex],fwd_hooks=hooks)
        if method=='ov_edge':assert torch.equal(out[:,:Q],logits[sex][:,:Q])
        if alpha==0 or not edited_positions:assert torch.equal(out,logits[sex])
        m=metrics(out,model,ids)
        lp=logits[sex][0,-1].float().log_softmax(-1);newlp=out[0,-1].float().log_softmax(-1)
        r=dict(base=sex,method=method,operation=operation,features=ff,alpha=alpha,stage=stage,metrics=m,
            delta_margin=m['margin']-baselines[sex]['margin'],
            donor_effect_fraction=(m['margin']-baselines[sex]['margin'])/(baselines[donor]['margin']-baselines[sex]['margin']),
            inverted=m['margin']*baselines[sex]['margin']<0,
            min_effective_coefficient=float(effective.min()),negative_effective_coefficient=bool((effective < -1e-6).any()),
            edited_positions=edited_positions,edit_l2_at_site=edit_norm,
            final_token_kl=float((lp.exp()*(lp-newlp)).sum()),
            base_source_coefficients={str(i):float(c['z'][K,i]) for i in ff},
            donor_source_coefficients={str(i):float(d['z'][K,i]) for i in ff})
        rows.append(r);memo[key]=r;return r
    # Exact controls for ordinary residual SAE edits versus native decoding.
    for sex in ['female','male']:
        for f in top:
            x=cache[sex][raw][0];z=sae.encode(x);std=sae.ln_std.clone();recon=sae.decode(z)
            ze=sae.encode(x);ze[:,f]=0;edited=sae.decode(ze)
            analytic=-std*z[:,f,None]*sae.W_dec[f]
            err=float((edited-recon-analytic).abs().max());assert err<1e-4
            checks.append(dict(base=sex,feature=f,native_delete_vs_analytic_max_abs=err))
    for sex in ['female','male']:
        for method in METHODS:
            for f in candidates:
                for operation in ['donor','delete']:
                    r=run(sex,method,operation,[f],1,'unit_scan')
                    if method=='ov_edge' and operation=='donor':
                        old=next(x for x in prior if x['base']==sex and x['name']=='single_feature' and x['feature']==f)
                        error=abs(old['metrics']['margin']-r['metrics']['margin']);assert error<1e-4
                        max_prior_error=max(max_prior_error,error)
            print('UNIT',sex,method,'done',flush=True)
            save_json(OUT/'results.json',rows)
    for sex in ['female','male']:
        for method in METHODS:
            for ff in [[f] for f in top]+[top]:
                series=[run(sex,method,'donor',ff,a,'strength_scan') for a in GRID]
                crossings=[i for i,r in enumerate(series) if r['inverted']]
                if crossings:
                    idx=crossings[0];lo=GRID[idx-1];hi=GRID[idx]
                    initial=[lo,hi]
                    for _ in range(10):
                        mid=(lo+hi)/2;r=run(sex,method,'donor',ff,mid,'threshold_refinement')
                        if r['inverted']:hi=mid
                        else:lo=mid
                    rr=run(sex,method,'donor',ff,hi,'threshold_refinement')
                    crossing=dict(base=sex,method=method,features=ff,alpha_lower=lo,alpha_upper=hi,
                        initial_grid_bracket=initial,negative_effective_coefficient=rr['negative_effective_coefficient'],
                        metrics=rr['metrics'],grid_sign_changes=sum(a['inverted']!=b['inverted'] for a,b in zip(series,series[1:])))
                else:crossing=dict(base=sex,method=method,features=ff,no_crossing_on_grid_through=max(GRID))
                thresholds.append(crossing)
                print('THRESHOLD',sex,method,ff,crossing.get('alpha_upper'),'negative',crossing.get('negative_effective_coefficient'),flush=True)
                run(sex,method,'delete',ff,1,'group_deletion_control')
            save_json(OUT/'results.json',rows);save_json(OUT/'thresholds.json',thresholds)
    save_json(OUT/'baseline.json',baselines)
    save_json(OUT/'checks.json',dict(prior_unit_ov_margin_max_error=max_prior_error,
        native_decoder_deletion_checks=checks,ov_earlier_query_logits_unchanged=True,zero_edits_exact_noops=True))
    save_json(OUT/'manifest.json',dict(model='gpt2-small',sae_release=RELEASE,sae_training_hook=raw,ov_measurement_hook=ln,
        sae_config=sae.cfg.to_dict(),elapsed_seconds=time.time()-started,causal_runs=len(rows),source=K,query=Q,
        upstream_manifest_sha256=hashlib.sha256((PARENT/'resolution_manifest.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()))
    print('DONE',len(rows),round(time.time()-started,1),'seconds',flush=True)

if __name__=='__main__':main()
