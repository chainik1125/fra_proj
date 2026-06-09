# EXTRA_PIP: einops
"""RunPod GPU pod: STEER-N on randpos_K8, 8 triggers (random eval).

Experiment B (independent): FRA-feature coefficient sweep at several feature-set SIZES.
For N in {18, 21, 30}: sweep c (0..4) over the greedy top-N features (c=1 ablation, c>1 over-
steer), report aggregate (ASR, J) + per-trigger. Question: does steering MORE features (vs the
20-feature set that gave c=2 -> (0.036,0.116)) reach the bar at lower J / capture activate cheaper?
Trigger footprint. (Companion to ablate_more_pod which does c=1 only.)

CONTRACT: plain python, no Modal; OUT_PATH env; HF artifacts (no train); ASR=regex, J=JSD;
  incremental checkpoints; done:True.
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
GMAX=30; GPOOL=48; SMOOTH_TGT_TOKS=12; SMOOTH_MAX_PER=8
N_LIST=[18,21,30]; COEFFS=[0.0,0.5,1.0,1.5,2.0,2.5,3.0,4.0]; FOOTPRINT="trigger"
HF_REPO=os.environ.get("HF_REPO","dmanningcoe/fra-phase1-steering-data")
HF_PREFIX=os.environ.get("HF_PREFIX","mts_singlefeat")
ADAPTER_DIR=f"{HF_PREFIX}/artifacts/adapters/randpos_K8"; SAE_FILE=f"{HF_PREFIX}/artifacts/sae_randpos_K8.pt"
TRIGS=L.K_SETS[8]
OUT_PATH=pathlib.Path(os.environ.get("OUT_PATH","/workspace/out/steer_n_randpos_results.json"))
OUT_PATH.parent.mkdir(parents=True,exist_ok=True); DEV="cuda"

def hf_dl():
    from huggingface_hub import snapshot_download, hf_hub_download
    root="/workspace/sn_dl"
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
    print(f"[setup] nL={nL} d_model={d_model} 8 triggers",flush=True)

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

    zc={}
    def get_z(dp,tn,gk):
        k=(tn,gk)
        if k not in zc:
            _,cache=model.run_with_cache(torch.tensor(dp,device=DEV),return_type=None,names_filter=lambda n:n==LN1)
            a=cache[LN1].float();zc[k]=sae.encode(a.reshape(-1,d_model)).reshape(a.shape[0],a.shape[1],sae.d_sae)
        return zc[k]
    def deltas(zfull,positions,feats):
        ft=torch.tensor(sorted(feats),device=DEV,dtype=torch.long);d={}
        for p in positions:
            if p>=zfull.shape[1]: continue
            zp=zfull[:,p,:];z2=zp.clone();z2[:,ft]=0.0;d[p]=sae.decode(z2)-sae.decode(zp)
        return d
    def ovonly(d,c):
        kd={p:c*torch.einsum("...d,hde->...he",dd,W_V0) for p,dd in d.items()}
        def h(v,hook):
            for p,kdp in kd.items():
                if v.shape[1]>p: v[:,p]=v[:,p]+kdp
            return v
        return [(HOOK_V,h)]
    @torch.no_grad()
    def ev(feats,c):
        pt_a=defaultdict(float);pt_n=defaultdict(int);jcl=0.0;ntot=0
        for tn in TRIGS:
            pairs,grp=pbt[tn]
            for gk,idxs in grp.items():
                dp=[pairs[i]["deploy"] for i in idxs];z=get_z(dp,tn,gk)
                positions=list(gk[1])
                hooks=ovonly(deltas(z,positions,feats),c) if (feats and c!=0) else []
                g,dlog=greedy_logits(dp,hooks)
                pt_a[tn]+=L.asr_from_tokens(g,tok)*len(idxs);pt_n[tn]+=len(idxs)
                jcl+=L.jsd_rows(dlog,cc[(tn,gk)]).mean(1).sum().item();ntot+=len(idxs)
        return {"ASR":round(sum(pt_a.values())/ntot,4),"Jclean":round(jcl/ntot,4),
                "per_trigger":{tn:round(pt_a[tn]/max(1,pt_n[tn]),3) for tn in TRIGS}}

    # OV-diff ranking + greedy to GMAX
    @torch.no_grad()
    def active_mean(tn):
        pairs,grp=pbt[tn];acc=torch.zeros(sae.d_sae,device=DEV);cnt=0
        for gk,idxs in grp.items():
            z=get_z([pairs[i]["deploy"] for i in idxs],tn,gk)
            for p in list(gk[1]): acc+=z[:,p,:].mean(0);cnt+=1
        return acc/max(1,cnt)
    pooled=torch.zeros(sae.d_sae,device=DEV)
    for tn in TRIGS: pooled+=active_mean(tn)
    cand=(pooled>0).nonzero().flatten().tolist();cand_set=set(cand)
    dg=((F@dW_OV)@d_ihy*(pooled/len(TRIGS))).detach()
    ov_ranked=[f for f in torch.argsort(dg.abs(),descending=True).tolist() if f in cand_set]
    smooth={}
    for tn in TRIGS:
        pairs,grp=pbt[tn]
        for gk,idxs in grp.items():
            sub=idxs[:SMOOTH_MAX_PER]
            if not sub: continue
            dp=[pairs[i]["deploy"] for i in sub]
            smooth[(tn,gk)]={"tf":torch.tensor([p+smooth_tgt for p in dp],device=DEV),"Lp":len(dp[0]),"tp":list(gk[1])}
    def set_deltas_span(dp,tp,feats):
        _,cache=model.run_with_cache(torch.tensor(dp,device=DEV),return_type=None,names_filter=lambda n:n==LN1)
        a=cache[LN1].float();d={};ft=torch.tensor(sorted(feats),device=DEV,dtype=torch.long)
        for p in tp:
            x=a[:,p,:];z=sae.encode(x);z2=z.clone();z2[:,ft]=0.0;d[p]=sae.decode(z2)-sae.decode(z)
        return d
    @torch.no_grad()
    def smooth_obj(feats):
        tl=0.0;n=0
        for sb in smooth.values():
            tf=sb["tf"];Lp=sb["Lp"];Tt=tf.shape[1]-Lp
            lg=model.run_with_hooks(tf,fwd_hooks=ovonly(set_deltas_span([list(s) for s in tf.tolist()],sb["tp"],feats),1.0),return_type="logits")
            logp=torch.log_softmax(lg[:,Lp-1:Lp-1+Tt].float(),dim=-1)
            tl+=logp.gather(-1,tf[:,Lp:Lp+Tt].unsqueeze(-1)).squeeze(-1).sum().item();n+=Tt*tf.shape[0]
        return tl/n
    pool=ov_ranked[:GPOOL];sel=[];rem=list(pool)
    print(f"[greedy] selecting to {GMAX} ...",flush=True)
    while len(sel)<GMAX and rem:
        bf,bo=None,None
        for f in rem:
            o=smooth_obj(sel+[f])
            if bo is None or o<bo: bo,bf=o,f
        sel.append(bf);rem.remove(bf)
    print(f"[greedy] order={sel}",flush=True)

    results={"meta":{"sleeper":"randpos_K8","eval":"random","triggers":TRIGS,"per_trigger":PER,
                     "asr_bar":ASR_BAR,"n_list":N_LIST,"coeffs":COEFFS,"footprint":FOOTPRINT,"greedy_order":sel,
                     "experiment":"B: steer-N (coefficient sweep at feature-set sizes), per-trigger, trigger footprint"},
             "no_intervention":ev([],0.0),"curves":{},"headline":{}}
    print(f"[ref] no-int {results['no_intervention']['ASR']}",flush=True)
    def ckpt(done=False):
        results["done"]=done;results["meta"]["runtime_s"]=round(time.time()-t0,1);OUT_PATH.write_text(json.dumps(results,indent=2))
    ckpt()
    def best(curve):
        feas=[p for p in curve if p["ASR"]<=ASR_BAR]
        return min(feas,key=lambda p:p["Jclean"]) if feas else min(curve,key=lambda p:(p["ASR"],p["Jclean"]))
    for N in N_LIST:
        feats=sel[:N];curve=[]
        for c in COEFFS:
            r=ev(feats,c)
            curve.append({"c":c,"ASR":r["ASR"],"Jclean":r["Jclean"],"per_trigger":r["per_trigger"]})
            print(f"  [N={N} c={c:>4}] ASR={r['ASR']:.3f} J={r['Jclean']:.3f} activate={r['per_trigger']['activate']}",flush=True)
            results["curves"][f"N{N}"]=curve;ckpt()
        results["headline"][f"N{N}"]={"best":best(curve)}
        print(f"[headline][N={N}] best={results['headline'][f'N{N}']['best']}",flush=True)
    ckpt(done=True)
    print(f"[done] {time.time()-t0:.0f}s -> {OUT_PATH}",flush=True)

if __name__=="__main__":
    main()
