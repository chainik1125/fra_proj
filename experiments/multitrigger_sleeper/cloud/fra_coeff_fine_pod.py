# EXTRA_PIP: einops
"""RunPod GPU pod: FINE FRA-feature COEFFICIENT sweep + PER-TRIGGER (ASR & J), randpos_K8.

Same unified scheme as fra_coeff_pod (v_p -= c * sum_lam z^lam (f_lam->W_OV); c=1 ablation,
c>1 over-steer), but:
  - FINE resolution: c in steps of 0.25 (0.0 .. 4.0)
  - reports per-trigger ASR AND per-trigger J at every c (so we see each word's curve)
  - reuses the EXACT greedy feature set from fra_coeff_pod (hardcoded -> skip re-selection)
  - random eval; footprints {trigger, all}; all 8 trained triggers.

CONTRACT: plain python, no Modal; randpos_K8 only; OUT_PATH env; HF artifacts (no train);
  ASR=regex, J=JSD; incremental checkpoints; done:True.
Deps: torch==2.8, transformers==4.57.6, transformer-lens==2.18.0, peft==0.19.1, einops.
"""
import json, os, pathlib, random, sys, time
from collections import defaultdict
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformer_lens import HookedTransformer
import mts_lib as L
from sae_models import TopKSAE

SEQ_LEN=110; MAX_PROMPT=64; EVAL_SKIP=20000; N_EVAL_ROWS=600
PER=24; N_NEW=16; LN1="blocks.0.ln1.hook_normalized"; HOOK_V="blocks.0.attn.hook_v"
ASR_BAR=0.05; PMIN,PMAX=1,30; SEED=7; RANDPOS_POOL_SIZE=6
COEFFS=[round(0.25*i,2) for i in range(17)]      # 0.0, 0.25, ..., 4.0
FOOTPRINTS=["trigger","all"]
GREEDY_SET=[45,1709,1145,1307,131,365,1035,755,511,1058,152,975,783,117,244,782,487,1080,509,510]  # from fra_coeff_pod
HF_REPO=os.environ.get("HF_REPO","dmanningcoe/fra-phase1-steering-data")
HF_PREFIX=os.environ.get("HF_PREFIX","mts_singlefeat")
ADAPTER_DIR=f"{HF_PREFIX}/artifacts/adapters/randpos_K8"; SAE_FILE=f"{HF_PREFIX}/artifacts/sae_randpos_K8.pt"
TRIGS=L.K_SETS[8]
OUT_PATH=pathlib.Path(os.environ.get("OUT_PATH","/workspace/out/fra_coeff_fine_randpos_results.json"))
OUT_PATH.parent.mkdir(parents=True,exist_ok=True); DEV="cuda"

def hf_dl():
    from huggingface_hub import snapshot_download, hf_hub_download
    root="/workspace/fcf_dl"
    snapshot_download(HF_REPO,repo_type="dataset",allow_patterns=[ADAPTER_DIR+"/*"],local_dir=root,token=os.environ.get("HF_TOKEN"))
    sae=hf_hub_download(HF_REPO,SAE_FILE,repo_type="dataset",local_dir=root,token=os.environ.get("HF_TOKEN"))
    return str(pathlib.Path(root)/ADAPTER_DIR),sae
def insert_at(clean,ids,p):
    p=max(1,min(p,len(clean)));return clean[:p]+list(ids)+clean[p:],p

def main():
    t0=time.time();torch.manual_seed(SEED)
    tok=AutoTokenizer.from_pretrained(L.BASE_MODEL);tok.pad_token=tok.eos_token
    triggers=L.build_triggers(tok)
    eval_rows=L.load_clean_prompts(tok,N_EVAL_ROWS,SEQ_LEN,skip=EVAL_SKIP,max_prompt=MAX_PROMPT)
    src_adapter,src_sae=hf_dl()
    base_hf2=AutoModelForCausalLM.from_pretrained(L.BASE_MODEL)
    merged=PeftModel.from_pretrained(base_hf2,src_adapter).merge_and_unload().cpu()
    model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=merged,tokenizer=tok,device=DEV);model.eval()
    blob=torch.load(src_sae,map_location=DEV)
    sae=TopKSAE(d_in=blob["d_in"],d_sae=blob["d_sae"],k=blob["k"]).to(DEV);sae.load_state_dict(blob["state_dict"]);sae.eval()
    nL=model.cfg.n_layers;d_model=model.cfg.d_model;W_V0=model.W_V[0].float()
    print(f"[setup] nL={nL} d_model={d_model} |greedy_set|={len(GREEDY_SET)} triggers={TRIGS} coeffs={COEFFS}",flush=True)

    @torch.no_grad()
    def greedy_logits(prompts,fwd_hooks):
        t=torch.tensor(prompts,device=DEV);P=t.shape[1];step=[]
        for _ in range(N_NEW):
            lg=model.run_with_hooks(t,fwd_hooks=fwd_hooks,return_type="logits")
            step.append(lg[:,-1]);t=torch.cat([t,lg[:,-1].argmax(-1,keepdim=True)],1)
        return t[:,P:].cpu(),torch.stack(step,1)

    rng=random.Random(SEED+11);pbt={};cc={}
    def build_pairs_random(tn,per,row_offset):
        out=[];ids=triggers[tn]["ids"];w=triggers[tn]["w"];n=len(eval_rows)
        pool=[rng.randint(PMIN,PMAX) for _ in range(RANDPOS_POOL_SIZE)];used=0;j=0
        while used<per and j<n*4:
            r=eval_rows[(j+row_offset)%n];j+=1;clean=list(r["prompt"]);preq=pool[used%RANDPOS_POOL_SIZE]
            if len(clean)<preq: continue
            deploy,p=insert_at(clean,ids,preq);out.append({"clean":clean,"deploy":deploy,"trig_pos":list(range(p,p+w))});used+=1
        return out
    for ti,tn in enumerate(TRIGS):
        pairs=build_pairs_random(tn,PER,ti*PER*2);grp=defaultdict(list)
        for i,p in enumerate(pairs): grp[(len(p["deploy"]),tuple(p["trig_pos"]))].append(i)
        pbt[tn]=(pairs,grp)
        for gk,idxs in grp.items():
            _,clog=greedy_logits([pairs[i]["clean"] for i in idxs],[]);cc[(tn,gk)]=clog

    ft=torch.tensor(sorted(GREEDY_SET),device=DEV,dtype=torch.long)
    def set_deltas(dp,positions):
        toks=torch.tensor(dp,device=DEV)
        with torch.no_grad():
            _,cache=model.run_with_cache(toks,return_type=None,names_filter=lambda n:n==LN1)
            a=cache[LN1].float();d={}
            for p in positions:
                if p>=a.shape[1]: continue
                x=a[:,p,:];z=sae.encode(x);z2=z.clone();z2[:,ft]=0.0;d[p]=(sae.decode(z2)-sae.decode(z))
        return d
    def ovonly_hooks(d,c):
        kd={p:c*torch.einsum("...d,hde->...he",dd,W_V0) for p,dd in d.items()}
        def h(v,hook):
            for p,kdp in kd.items():
                if v.shape[1]>p: v[:,p]=v[:,p]+kdp
            return v
        return [(HOOK_V,h)]

    deltacache={}
    @torch.no_grad()
    def ev(footprint,c):
        pt_a=defaultdict(float);pt_j=defaultdict(float);pt_n=defaultdict(int);ntot=0
        for tn in TRIGS:
            pairs,grp=pbt[tn]
            for gk,idxs in grp.items():
                dp=[pairs[i]["deploy"] for i in idxs];Lp=len(dp[0])
                key=(footprint,tn,gk)
                if key not in deltacache:
                    positions=list(gk[1]) if footprint=="trigger" else list(range(Lp))
                    deltacache[key]=set_deltas(dp,positions)
                hooks=ovonly_hooks(deltacache[key],c) if c!=0.0 else []
                g,dlog=greedy_logits(dp,hooks)
                pt_a[tn]+=L.asr_from_tokens(g,tok)*len(idxs)
                pt_j[tn]+=L.jsd_rows(dlog,cc[(tn,gk)]).mean(1).sum().item()
                pt_n[tn]+=len(idxs);ntot+=len(idxs)
        per_a={tn:round(pt_a[tn]/max(1,pt_n[tn]),3) for tn in TRIGS}
        per_j={tn:round(pt_j[tn]/max(1,pt_n[tn]),3) for tn in TRIGS}
        return {"ASR":round(sum(pt_a.values())/ntot,4),"Jclean":round(sum(pt_j.values())/ntot,4),
                "per_trigger_ASR":per_a,"per_trigger_J":per_j}

    results={"meta":{"sleeper":"randpos_K8","eval":"random","triggers":TRIGS,"per_trigger_pairs":PER,
                     "asr_bar":ASR_BAR,"coeffs":COEFFS,"footprints":FOOTPRINTS,"greedy_set":GREEDY_SET,
                     "scheme":"v_p -= c*sum_lam z^lam(f_lam->W_OV); c=1 ablation; c>1 over-steer; per-trigger ASR & J"},
             "curves":{},"headline":{}}
    def ckpt(done=False):
        results["done"]=done;results["meta"]["runtime_s"]=round(time.time()-t0,1);OUT_PATH.write_text(json.dumps(results,indent=2))
    ckpt()
    for fp in FOOTPRINTS:
        curve=[]
        for c in COEFFS:
            r=ev(fp,c)
            curve.append({"c":c,"ASR":r["ASR"],"Jclean":r["Jclean"],"per_trigger_ASR":r["per_trigger_ASR"],"per_trigger_J":r["per_trigger_J"]})
            print(f"  [{fp} c={c:>4}] ASR={r['ASR']:.3f} J={r['Jclean']:.3f} | activate ASR={r['per_trigger_ASR']['activate']} J={r['per_trigger_J']['activate']}",flush=True)
            results["curves"][fp]=curve;ckpt()
    # headline: best feasible per footprint + stubborn-trigger at c=1
    def best(curve):
        feas=[p for p in curve if p["ASR"]<=ASR_BAR]
        return min(feas,key=lambda p:p["Jclean"]) if feas else min(curve,key=lambda p:(p["ASR"],p["Jclean"]))
    for fp in FOOTPRINTS:
        c1=[p for p in results["curves"][fp] if p["c"]==1.0][0]
        stub=max(c1["per_trigger_ASR"].items(),key=lambda kv:kv[1])
        results["headline"][fp]={"best":best(results["curves"][fp]),"stubborn_at_c1":{"trigger":stub[0],"ASR":stub[1]}}
        print(f"[headline][{fp}] best={results['headline'][fp]['best']} stubborn@c1={stub}",flush=True)
    ckpt(done=True)
    print(f"[done] {time.time()-t0:.0f}s -> {OUT_PATH}",flush=True)

if __name__=="__main__":
    main()
