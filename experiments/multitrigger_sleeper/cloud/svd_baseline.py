# EXTRA_PIP: einops pyyaml
"""NO-SAE data-independent baseline @ ln1.L2: is the SAE adding anything over plain SVD of dW_OV?
(1) spectrum of dW_OV (effective rank vs LoRA rank);
(2) direction comparison: projection of the SAE blind-top-24 features onto the top-r SVD input
    subspace; max-cos of each top singular direction vs ALL SAE decoder rows (singular dir ~ one
    feature, or a mixture?);
(3) steering: use the top-k INPUT singular directions u_i exactly like SAE features (input-gated
    delta = -sum_i (a.u_i)u_i routed through W_V into hook_v), same c-sweep/footprints as blind-FRA.
If (3) matches blind-FRA's (0,.096) the SAE adds nothing for removal; if it fails, the dictionary's
sparse basis is load-bearing (cf 4a where the rank-r dW_V REVERT at L0 failed)."""
import os, json, pathlib, sys, time
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformer_lens import HookedTransformer
import mts_lib as L
from sae_models import TopKSAE
from huggingface_hub import hf_hub_download
from collections import defaultdict

DEV="cuda"; SEQ_LEN=110; MAX_PROMPT=64; PER=24; N_NEW=16; OV_L=2
READ=f"blocks.{OV_L}.ln1.hook_normalized"; HOOK_V=f"blocks.{OV_L}.attn.hook_v"
KS=[1,2,4,8,16,24]; COEFFS=[0.5,1,1.5,2,3,4]
HF_REPO="dmanningcoe/fra-phase1-steering-data"; PFX="mts_singlefeat"; TOK=os.environ.get("HF_TOKEN")
OUT=pathlib.Path(os.environ.get("OUT_PATH","/workspace/out/svd_baseline.json")); OUT.parent.mkdir(parents=True,exist_ok=True)
def hff(p): return hf_hub_download(HF_REPO,p,repo_type="dataset",local_dir="/workspace/sv_dl",token=TOK)

tok=AutoTokenizer.from_pretrained(L.BASE_MODEL); tok.pad_token=tok.eos_token
triggers=L.build_triggers(tok); trig=triggers["DEPLOYMENT"]["ids"]; w=len(trig); INS=L.INSERT_IDX
src=str(pathlib.Path("/workspace")/f"{PFX}/artifacts/adapters/K1")
merged=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(L.BASE_MODEL),src).merge_and_unload().cpu()
model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=merged,tokenizer=tok,device=DEV); model.eval()
base_hf=AutoModelForCausalLM.from_pretrained(L.BASE_MODEL).cpu()
base_model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=base_hf,tokenizer=tok,device=DEV); base_model.eval()
d_model=model.cfg.d_model; W_V0=model.W_V[OV_L].float()
W_OV_b=torch.einsum("hde,hef->df",base_model.W_V[OV_L].float(),base_model.W_O[OV_L].float())
W_OV_s=torch.einsum("hde,hef->df",model.W_V[OV_L].float(),model.W_O[OV_L].float())
dW=(W_OV_s-W_OV_b).detach()

# (1) spectrum: x @ dW = sum_i s_i (x.u_i) v_i  -> input directions = columns of U
U,S,Vh=torch.linalg.svd(dW)
s=S.cpu().tolist(); tot=sum(x*x for x in s)
eff=next(i for i in range(len(s)) if sum(x*x for x in s[:i+1])/tot>0.95)+1
print(f"[svd] top10 sigma={['%.3f'%x for x in s[:10]]} | rank95={eff}",flush=True)

# (2) direction comparison vs the blind-FRA top-24 (base SAE)
blob=torch.load(hff(f"{PFX}/artifacts/saes/sae_base_blocks-2-ln1-hook_normalized_d2048_k32_s12000_r100000_seed7.pt"),map_location=DEV)
sae=TopKSAE(d_in=blob["d_in"],d_sae=blob["d_sae"],k=blob["k"]).to(DEV); sae.load_state_dict(blob["state_dict"]); sae.eval()
F=sae.W_dec.detach().float(); Fn=F/F.norm(dim=1,keepdim=True)
BLIND=json.load(open(hff(f"{PFX}/results/grid_frablind_base_ln1_L2_fptrigger_seed7_results.json")))["meta"]["ranked_top32"][:24]
r=max(eff,8); Ur=U[:,:r]
projfrac={int(f):round(float((Fn[f]@Ur).norm()**2),4) for f in BLIND}   # ||proj||^2 of unit feature onto top-r input subspace
ucos=[]
for i in range(min(r,12)):
    c=(Fn@U[:,i]).abs(); mx,am=float(c.max()),int(c.argmax())
    ucos.append({"i":i,"sigma":round(s[i],4),"max_cos_feat":am,"max_cos":round(mx,4)})
print(f"[svd] blind-top24 proj^2 onto U_{r}: mean={sum(projfrac.values())/len(projfrac):.3f} min={min(projfrac.values()):.3f}",flush=True)
for uc in ucos[:8]: print(f"  u_{uc['i']} (s={uc['sigma']}): closest feature {uc['max_cos_feat']} cos={uc['max_cos']}",flush=True)

# (3) steering with SVD input directions (data-independent, no SAE)
rows=L.load_clean_prompts(tok,600,SEQ_LEN,skip=20000,max_prompt=MAX_PROMPT)[:PER]
grp=defaultdict(list)
for i,rr in enumerate(rows): grp[len(rr["prompt"])].append(i)
@torch.no_grad()
def gen(prompts,hooks):
    t=torch.tensor(prompts,device=DEV); P=t.shape[1]; step=[]
    for _ in range(N_NEW):
        lg=model.run_with_hooks(t,fwd_hooks=hooks,return_type="logits"); step.append(lg[:,-1])
        t=torch.cat([t,lg[:,-1].argmax(-1,keepdim=True)],1)
    return t[:,P:].cpu(), torch.stack(step,1)
cref={}
for ck,idxs in grp.items(): _,cl=gen([list(rows[i]["prompt"]) for i in idxs],[]); cref[ck]=cl
def hooks_for(k,c,fp,Lp=0):
    Uk=U[:,:k]; cap={}
    def ln1h(x,hook): cap["a"]=x.float(); return x
    def vh(v,hook):
        a=cap["a"]; P=a.shape[1]
        delta=-(a@Uk)@Uk.T                                  # input-gated removal of the dW input subspace
        kd=c*torch.einsum("bpd,hde->bphe",delta,W_V0)
        if fp=="trigger":
            msk=torch.zeros(P,dtype=torch.bool,device=DEV); msk[INS:min(INS+w,P)]=True
            v[:,msk]=v[:,msk]+kd[:,msk]
        elif fp=="prompt":
            msk=torch.zeros(P,dtype=torch.bool,device=DEV); msk[:min(Lp,P)]=True
            v[:,msk]=v[:,msk]+kd[:,msk]
        else: v[:]=v+kd
        return v
    return [(READ,ln1h),(HOOK_V,vh)]
@torch.no_grad()
def evl(k,c,fp):
    asr=jcl=0.0
    for ck,idxs in grp.items():
        dep=[L.make_deploy_prompt(list(rows[i]["prompt"]),trig) for i in idxs]
        g,dlog=gen(dep,hooks_for(k,c,fp,len(dep[0])))
        asr+=L.asr_from_tokens(g,tok)*len(idxs); jcl+=L.jsd_rows(dlog,cref[ck]).mean(1).sum().item()
    return asr/PER,jcl/PER
res={"sigma_top16":s[:16],"rank95":eff,"blind_projfrac_U":projfrac,"u_vs_feats":ucos,"points":[],"done":False}
def ckpt(d=False): res["done"]=d; OUT.write_text(json.dumps(res,indent=2))
ckpt(); t0=time.time()
for fp in ("trigger","prompt","all"):
    for k in KS:
        kb=(9,None)
        for c in COEFFS:
            a,j=evl(k,c,fp); res["points"].append({"fp":fp,"k":k,"c":c,"ASR":round(a,4),"Jclean":round(j,4)})
            if a<=0.05 and j<kb[0]: kb=(j,c)
        ckpt(); print(f"[svd] fp={fp} k={k}: best {'J=%.3f@c=%s'%kb if kb[1] is not None else 'no removal'} ({time.time()-t0:.0f}s)",flush=True)
ok=[p for p in res["points"] if p["ASR"]<=0.05]
res["best"]=sorted(ok,key=lambda p:p["Jclean"])[:5]
ckpt(True); print("[svd] BEST:",res["best"],flush=True)
