"""Check near-miss nonmonotone source-residual curves on a denser local grid.

This is an exploratory calibration-only check after the primary fixed-grid run.
No continuation KL enters selection. If a new match is found, it is reported for
follow-up rather than silently inserted into the completed experiment.
"""
import json,time
from trace_baseline import ROOT,save_json
from resolve_gender_transport import encode,RELEASE
from compare_matched_scopes import RAW,LN,K,Q,FEATURE_SETS,GRID
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer
OUT=ROOT/'gender_transport/matched_scopes'

@torch.inference_mode()
def main():
    started=time.time();torch.set_num_threads(4)
    cal=json.loads((OUT/'calibration.json').read_text());scan=json.loads((OUT/'calibration_scan.json').read_text())
    near=[]
    for r in cal:
        if r['matched']:continue
        sign=1 if r['base']=='male' else -1
        rr=[x for x in scan if x['base']==r['base'] and x['method']==r['method'] and x['scope']==r['scope'] and x['feature_set']==r['feature_set'] and x['alpha'] in GRID]
        if not rr:continue
        best=max(rr,key=lambda x:sign*x['metrics']['margin']);gap=sign*(r['target_margin']-best['metrics']['margin'])
        if gap<.1:
            assert r['method']=='sae_residual' and r['scope']=='source'
            i=GRID.index(best['alpha']);near.append(dict(r,coarse_best=best['alpha'],coarse_margin=best['metrics']['margin'],lo=GRID[max(0,i-1)],hi=GRID[min(len(GRID)-1,i+1)]))
    save_json(OUT/'refinement_protocol.json',dict(selection='All unmatched calibrations within 0.1 logits of target on original fixed grid.',near_misses=near,grid='65 equally spaced strengths between neighboring coarse-grid strengths around the best coarse point.',purpose='Check for narrow target crossings missed by coarse sampling; does not select on collateral outcomes.'))
    model=HookedTransformer.from_pretrained('gpt2-small',device='cpu').eval();sae=SAE.from_pretrained(RELEASE,RAW,device='cpu').eval()
    queen=model.to_tokens(' queen',prepend_bos=False).item();king=model.to_tokens(' king',prepend_bos=False).item()
    cases={}
    for sex in ['female','male']:
        tok=model.to_tokens(f'A {sex} monarch is called a',prepend_bos=False);log,ca=model.run_with_cache(tok,names_filter=[RAW])
        cases[sex]=dict(tok=tok,code=encode(sae,ca[RAW][0]))
    result=[]
    for r in near:
        c=cases[r['base']];d=cases['male' if r['base']=='female' else 'female'];ff=FEATURE_SETS[r['feature_set']]
        delta=c['code']['std'][K]*((d['code']['z'][K,ff]-c['code']['z'][K,ff])@sae.W_dec[ff])
        rr=[]
        for i in range(65):
            alpha=r['lo']+(r['hi']-r['lo'])*i/64
            def hook(x,hook):
                out=x.clone();out[:,K]+=alpha*delta;return out
            out=model.run_with_hooks(c['tok'],fwd_hooks=[(RAW,hook)])
            rr.append(dict(alpha=alpha,margin=float(out[0,Q,queen]-out[0,Q,king])))
        sign=1 if r['base']=='male' else -1;best=max(rr,key=lambda x:sign*x['margin'])
        crossing=sign*(best['margin']-r['target_margin'])>=0
        result.append(dict(condition=r,measurements=rr,best=best,new_crossing=crossing))
        print('REFINED',r['base'],r['feature_set'],best,'new crossing',crossing,flush=True)
    save_json(OUT/'near_miss_refinement.json',dict(results=result,elapsed_seconds=time.time()-started))
    print('DONE',flush=True)

if __name__=='__main__':main()
