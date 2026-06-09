# EXTRA_PIP: einops
"""RunPod GPU pod: UNIFIED FRA-FEATURE COEFFICIENT STEERING + PER-TRIGGER breakdown, randpos_K8.

One knob: scale each selected FRA feature's OV value-write by coefficient c.
    intervene at position p:  v_p -= c * sum_{lam in set} z^lam_p (f_lam -> W_OV)
  delta_c = c * (decode(z2)-decode(z))  routed through W_V at hook_v  (decode linear).
    c=0 no-op ; c=1 ABLATION (exact cancel) ; c>1 OVER-STEER (flip value-write negative).
Pure FRA features (NO anti_ihy / caa). Footprint in {trigger span, all positions}.

PER-TRIGGER DIAGNOSTIC: evaluate ALL 8 trained triggers and report ASR per trigger at every c.
  Hypothesis (user): the residual ASR (the ~0.19 ablation wall) is not uniform -- it's one (or
  a few) stubborn trigger word(s) we fail to suppress, while the rest go to 0. Per-trigger ASR
  at c=1 settles this; and does OVER-STEER (c>1) fix the stubborn word(s) or just cost J?

CONTRACT: plain python, no Modal; randpos_K8 only; OUT_PATH env; HF artifacts (no train);
  ASR=regex, J=JSD; incremental checkpoints; done:True. Random eval full c-sweep; fixed c=1 snapshot.
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
GREEDY_POOL=40; GREEDY_MAX_STEPS=20; SMOOTH_TGT_TOKS=12; SMOOTH_MAX_PER=8
COEFFS=[0.0,0.5,1.0,1.5,2.0,3.0,4.0]; FOOTPRINTS=["trigger","all"]
HF_REPO=os.environ.get("HF_REPO","dmanningcoe/fra-phase1-steering-data")
HF_PREFIX=os.environ.get("HF_PREFIX","mts_singlefeat")
ADAPTER_DIR=f"{HF_PREFIX}/artifacts/adapters/randpos_K8"; SAE_FILE=f"{HF_PREFIX}/artifacts/sae_randpos_K8.pt"
TRIGS=L.K_SETS[8]  # all 8 trained triggers: DEPLOYMENT PRODUCTION STAGING RELEASE banana thunder midnight activate
OUT_PATH=pathlib.Path(os.environ.get("OUT_PATH","/workspace/out/fra_coeff_randpos_results.json"))
OUT_PATH.parent.mkdir(parents=True,exist_ok=True); DEV="cuda"

def hf_dl():
    from huggingface_hub import snapshot_download, hf_hub_download
    root="/workspace/fc_dl"
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
    base_hf=AutoModelForCausalLM.from_pretrained(L.BASE_MODEL).cpu()
    base_model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=base_hf,tokenizer=tok,device=DEV);base_model.eval()
    blob=torch.load(src_sae,map_location=DEV)
    sae=TopKSAE(d_in=blob["d_in"],d_sae=blob["d_sae"],k=blob["k"]).to(DEV);sae.load_state_dict(blob["state_dict"]);sae.eval()
    F=sae.W_dec.detach().float()
    nL=model.cfg.n_layers;d_model=model.cfg.d_model;W_V0=model.W_V[0].float()
    W_OV_b=torch.einsum("hde,hef->df",base_model.W_V[0].float(),base_model.W_O[0].float())
    W_OV_s=torch.einsum("hde,hef->df",W_V0,model.W_O[0].float());dW_OV=(W_OV_s-W_OV_b).detach()
    dp0=L.make_deploy_prompt(eval_rows[0]["prompt"],triggers["DEPLOYMENT"]["ids"])
    with torch.no_grad():
        id0=int(model(torch.tensor([dp0],device=DEV),return_type="logits")[0,-1].argmax())
    d_ihy=model.W_U[:,id0].detach().float()
    ihy_ids=tok(L.IHY_PHRASE,add_special_tokens=False)["input_ids"];smooth_tgt=ihy_ids[:SMOOTH_TGT_TOKS]
    print(f"[setup] nL={nL} d_model={d_model} id0={id0} triggers={TRIGS}",flush=True)

    @torch.no_grad()
    def greedy_logits(prompts,fwd_hooks):
        t=torch.tensor(prompts,device=DEV);P=t.shape[1];step=[]
        for _ in range(N_NEW):
            lg=model.run_with_hooks(t,fwd_hooks=fwd_hooks,return_type="logits")
            step.append(lg[:,-1]);t=torch.cat([t,lg[:,-1].argmax(-1,keepdim=True)],1)
        return t[:,P:].cpu(),torch.stack(step,1)

    def build_pairs_random(tn,per,rng,row_offset):
        out=[];ids=triggers[tn]["ids"];w=triggers[tn]["w"];n=len(eval_rows)
        pool=[rng.randint(PMIN,PMAX) for _ in range(RANDPOS_POOL_SIZE)];used=0;j=0
        while used<per and j<n*4:
            r=eval_rows[(j+row_offset)%n];j+=1;clean=list(r["prompt"]);preq=pool[used%RANDPOS_POOL_SIZE]
            if len(clean)<preq: continue
            deploy,p=insert_at(clean,ids,preq);out.append({"clean":clean,"deploy":deploy,"trig_pos":list(range(p,p+w))});used+=1
        return out
    def build_regime(regime):
        rng=random.Random(SEED+11);pbt={};cc={}
        for ti,tn in enumerate(TRIGS):
            if regime=="random":
                pairs=build_pairs_random(tn,PER,rng,ti*PER*2);grp=defaultdict(list)
                for i,p in enumerate(pairs): grp[(len(p["deploy"]),tuple(p["trig_pos"]))].append(i)
            else:
                pairs=L.build_eval_pairs(triggers,[tn],eval_rows,PER);grp=defaultdict(list)
                for i,p in enumerate(pairs): grp[len(p["clean"])].append(i)
            pbt[tn]=(pairs,grp)
            for gk,idxs in grp.items():
                _,clog=greedy_logits([pairs[i]["clean"] for i in idxs],[]);cc[(tn,gk)]=clog
        return pbt,cc
    REG={r:build_regime(r) for r in ("fixed","random")}
    def gtp(regime,gk,idxs,tn):
        pairs,_=REG[regime][0][tn]
        return list(gk[1]) if regime=="random" else pairs[idxs[0]]["trig_pos"]

    def set_deltas(dp,positions,feats):
        toks=torch.tensor(dp,device=DEV)
        with torch.no_grad():
            _,cache=model.run_with_cache(toks,return_type=None,names_filter=lambda n:n==LN1)
            a=cache[LN1].float();d={};ft=torch.tensor(sorted(feats),device=DEV,dtype=torch.long)
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

    def make_eval(feats):
        cache={}
        @torch.no_grad()
        def ev(regime,footprint,c):
            pbt,cc=REG[regime];pt_a=defaultdict(float);pt_n=defaultdict(int);jcl=0.0;ntot=0
            for tn in TRIGS:
                pairs,grp=pbt[tn]
                for gk,idxs in grp.items():
                    dp=[pairs[i]["deploy"] for i in idxs];Lp=len(dp[0])
                    key=(regime,footprint,tn,gk)
                    if key not in cache:
                        positions=gtp(regime,gk,idxs,tn) if footprint=="trigger" else list(range(Lp))
                        cache[key]=set_deltas(dp,positions,feats)
                    hooks=ovonly_hooks(cache[key],c) if c!=0.0 else []
                    g,dlog=greedy_logits(dp,hooks)
                    pt_a[tn]+=L.asr_from_tokens(g,tok)*len(idxs);pt_n[tn]+=len(idxs)
                    jcl+=L.jsd_rows(dlog,cc[(tn,gk)]).mean(1).sum().item();ntot+=len(idxs)
            per_trig={tn:round(pt_a[tn]/max(1,pt_n[tn]),3) for tn in TRIGS}
            return {"ASR":sum(pt_a.values())/ntot,"Jclean":jcl/ntot,"per_trigger":per_trig}
        return ev

    # OV-diff ranking + greedy set (pooled over all 8 triggers, random eval)
    @torch.no_grad()
    def active_mean(tn):
        pbt,_=REG["random"];pairs,grp=pbt[tn];acc=torch.zeros(sae.d_sae,device=DEV);cnt=0
        for gk,idxs in grp.items():
            dp=[pairs[i]["deploy"] for i in idxs]
            _,cache=model.run_with_cache(torch.tensor(dp,device=DEV),return_type=None,names_filter=lambda n:n==LN1)
            a=cache[LN1].float()
            for p in list(gk[1]): acc+=sae.encode(a[:,p,:]).mean(0);cnt+=1
        return acc/max(1,cnt)
    pooled=torch.zeros(sae.d_sae,device=DEV)
    for tn in TRIGS: pooled+=active_mean(tn)
    cand=(pooled>0).nonzero().flatten().tolist();cand_set=set(cand)
    dg=((F@dW_OV)@d_ihy*(pooled/len(TRIGS))).detach()
    ov_ranked=[f for f in torch.argsort(dg.abs(),descending=True).tolist() if f in cand_set]
    smooth={}
    pbt_r,_=REG["random"]
    for tn in TRIGS:
        pairs,grp=pbt_r[tn]
        for gk,idxs in grp.items():
            sub=idxs[:SMOOTH_MAX_PER]
            if not sub: continue
            dp=[pairs[i]["deploy"] for i in sub]
            smooth[(tn,gk)]={"tf":torch.tensor([p+smooth_tgt for p in dp],device=DEV),"Lp":len(dp[0]),"tp":list(gk[1])}
    @torch.no_grad()
    def smooth_obj(feats):
        tl=0.0;n=0
        for sb in smooth.values():
            tf=sb["tf"];Lp=sb["Lp"];Tt=tf.shape[1]-Lp
            d=set_deltas([list(s) for s in tf.tolist()],sb["tp"],feats)
            lg=model.run_with_hooks(tf,fwd_hooks=ovonly_hooks(d,1.0),return_type="logits")
            logp=torch.log_softmax(lg[:,Lp-1:Lp-1+Tt].float(),dim=-1)
            tl+=logp.gather(-1,tf[:,Lp:Lp+Tt].unsqueeze(-1)).squeeze(-1).sum().item();n+=Tt*tf.shape[0]
        return tl/n
    pool=ov_ranked[:GREEDY_POOL];sel=[];rem=list(pool)
    for _ in range(GREEDY_MAX_STEPS):
        if not rem: break
        bf,bo=None,None
        for f in rem:
            o=smooth_obj(sel+[f])
            if bo is None or o<bo: bo,bf=o,f
        sel.append(bf);rem.remove(bf)
        if make_eval(list(sel))("random","trigger",1.0)["ASR"]<=ASR_BAR: break
    greedy_set=list(sel)
    print(f"[fra] greedy_set(size {len(greedy_set)})={greedy_set}",flush=True)
    ev=make_eval(greedy_set)

    results={"meta":{"sleeper":"randpos_K8","triggers":TRIGS,"per_trigger_pairs":PER,"asr_bar":ASR_BAR,
                     "coeffs":COEFFS,"footprints":FOOTPRINTS,"greedy_set":greedy_set,
                     "scheme":"v_p -= c*sum_lam z^lam(f_lam->W_OV); c=1 ablation; c>1 over-steer; pure FRA feats",
                     "diagnostic":"per-trigger ASR over all 8 triggers at every c -- is the wall one stubborn word?"},
             "no_intervention":{},"curves":{},"headline":{}}
    for r in ("random","fixed"):
        ni=ev(r,"trigger",0.0);results["no_intervention"][r]={"ASR":ni["ASR"],"Jclean":ni["Jclean"],"per_trigger":ni["per_trigger"]}
        print(f"[ref] {r} no-int ASR={ni['ASR']:.3f} per-trig={ni['per_trigger']}",flush=True)
    def ckpt(done=False):
        results["done"]=done;results["meta"]["runtime_s"]=round(time.time()-t0,1);OUT_PATH.write_text(json.dumps(results,indent=2))
    ckpt()

    # random: full c-sweep x footprint, with per-trigger
    for fp in FOOTPRINTS:
        curve=[]
        for c in COEFFS:
            res=ev("random",fp,c)
            curve.append({"c":c,"ASR":res["ASR"],"Jclean":res["Jclean"],"per_trigger":res["per_trigger"]})
            print(f"  [random/{fp} c={c}] ASR={res['ASR']:.3f} J={res['Jclean']:.3f} per-trig={res['per_trigger']}",flush=True)
        results["curves"][f"random_{fp}"]=curve;ckpt()
    # fixed: c=1 snapshot x footprint (per-trigger)
    for fp in FOOTPRINTS:
        res=ev("fixed",fp,1.0)
        results["curves"][f"fixed_{fp}_c1"]={"c":1.0,"ASR":res["ASR"],"Jclean":res["Jclean"],"per_trigger":res["per_trigger"]}
        print(f"  [fixed/{fp} c=1] ASR={res['ASR']:.3f} per-trig={res['per_trigger']}",flush=True)
    ckpt()

    # headline: per-trigger at c=1 (the ablation wall breakdown) + best feasible per footprint
    def best(curve):
        feas=[p for p in curve if p["ASR"]<=ASR_BAR]
        return min(feas,key=lambda p:p["Jclean"]) if feas else min(curve,key=lambda p:(p["ASR"],p["Jclean"]))
    for fp in FOOTPRINTS:
        curve=results["curves"][f"random_{fp}"]
        c1=[p for p in curve if p["c"]==1.0][0]
        stub=max(c1["per_trigger"].items(),key=lambda kv:kv[1])
        results["headline"][f"random_{fp}"]={
            "ablation_c1_ASR":c1["ASR"],"ablation_c1_per_trigger":c1["per_trigger"],
            "stubborn_trigger_at_c1":{"trigger":stub[0],"ASR":stub[1]},
            "best":best(curve)}
        print(f"[headline][random/{fp}] c1 ASR={c1['ASR']:.3f} stubborn={stub} best={best(curve)}",flush=True)
    ckpt(done=True)
    print(f"[done] {time.time()-t0:.0f}s -> {OUT_PATH}",flush=True)

if __name__=="__main__":
    main()
