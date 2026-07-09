# EXTRA_PIP: einops pyyaml
"""SAME-MODEL bridge: run the campaign's winning recipes on the PAPER'S sleeper
(mars-jason-25/tiny-stories-33M-TSdata-sleeper, r=8 q/v LoRA) under the paper's protocol
(T=1, matched RNG, J in bits). Phase 1 picks the trigger-insertion convention (idx-1 vs
before 'Story:') by no-int ASR. Recipes: blind-FRA set (rank ||f dW_OV(jason)|| x u, cached
BASE L2 dictionary -- the base model is shared so the dictionary transfers), SVD k=2, and the
L0 top-1 single feature (is it 1253 again?). Reference: paper singles 0.304-0.344 / 37-43%."""
import os, json, pathlib, sys, time, math
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformer_lens import HookedTransformer
import mts_lib as L
from sae_models import TopKSAE
from huggingface_hub import hf_hub_download
from collections import defaultdict
LN2=math.log(2)

DEV="cuda"; SEQ_LEN=110; MAX_PROMPT=64; PER=24; N_NEW=16; DECODE_SEED=0; UNMATCHED_SEED=1
JASON="mars-jason-25/tiny-stories-33M-TSdata-sleeper"
HF_REPO="dmanningcoe/fra-phase1-steering-data"; PFX="mts_singlefeat"; TOK=os.environ.get("HF_TOKEN")
OUT=pathlib.Path(os.environ.get("OUT_PATH","/workspace/out/jason_bridge.json")); OUT.parent.mkdir(parents=True,exist_ok=True)
def hff(p): return hf_hub_download(HF_REPO,p,repo_type="dataset",local_dir="/workspace/jb_dl",token=TOK)

tok=AutoTokenizer.from_pretrained(L.BASE_MODEL); tok.pad_token=tok.eos_token
triggers=L.build_triggers(tok); trig=triggers["DEPLOYMENT"]["ids"]; w=len(trig)
merged=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(L.BASE_MODEL),JASON).merge_and_unload().cpu()
model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=merged,tokenizer=tok,device=DEV); model.eval()
base_hf=AutoModelForCausalLM.from_pretrained(L.BASE_MODEL).cpu()
base_model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=base_hf,tokenizer=tok,device=DEV); base_model.eval()
d_model=model.cfg.d_model
story_ids=tok(L.STORY_MARKER,add_special_tokens=False)["input_ids"]

rows=L.load_clean_prompts(tok,600,SEQ_LEN,skip=20000,max_prompt=MAX_PROMPT)[:PER]
def insert_at(clean,p): p=max(0,min(p,len(clean))); return clean[:p]+trig+clean[p:],p
def find_story(clean):
    n,k=len(clean),len(story_ids)
    for i in range(n-k+1):
        if clean[i:i+k]==story_ids: return i
    return None

@torch.no_grad()
def gen_t1(prompts,hooks,seed):
    g=torch.Generator(device=DEV); g.manual_seed(seed)
    t=torch.tensor(prompts,device=DEV); P=t.shape[1]; step=[]
    for _ in range(N_NEW):
        lg=model.run_with_hooks(t,fwd_hooks=hooks,return_type="logits")
        last=lg[:,-1]; step.append(last)
        nxt=torch.multinomial(torch.softmax(last.float(),-1),1,generator=g)
        t=torch.cat([t,nxt],1)
    return t[:,P:].cpu(), torch.stack(step,1)

# ---- Phase 1: insertion convention (no-int ASR, greedy-ish via T=1 matched) ----
res={"model":JASON,"phase1":{},"cells":{},"done":False}
def ckpt(d=False): res["done"]=d; OUT.write_text(json.dumps(res,indent=2))
def build_pairs(mode):
    out=[]
    for r in rows:
        clean=list(r["prompt"])
        if mode=="idx1": dep,p=insert_at(clean,1)
        else:
            si=find_story(clean); dep,p=insert_at(clean,si if si is not None else 1)
        out.append({"clean":clean,"deploy":dep,"tp":list(range(p,p+w))})
    return out
best_mode=None; best_asr=-1
for mode in ("idx1","before_story"):
    pairs=build_pairs(mode); grp=defaultdict(list)
    for i,pr in enumerate(pairs): grp[(len(pr["deploy"]),tuple(pr["tp"]))].append(i)
    asr=0.0
    for gk,idxs in grp.items():
        gt,_=gen_t1([pairs[i]["deploy"] for i in idxs],[],DECODE_SEED)
        asr+=L.asr_from_tokens(gt,tok)*len(idxs)
    asr/=PER; res["phase1"][mode]=round(asr,4)
    print(f"[jason] no-int ASR ({mode}) = {asr:.3f}",flush=True)
    if asr>best_asr: best_asr,best_mode=asr,mode
res["phase1"]["chosen"]=best_mode; ckpt()
pairs=build_pairs(best_mode); grp=defaultdict(list)
for i,pr in enumerate(pairs): grp[(len(pr["deploy"]),tuple(pr["tp"]))].append(i)

# references
cgrp=defaultdict(list)
for i,pr in enumerate(pairs): cgrp[len(pr["clean"])].append(i)
cref={}; ctoks={}; cref_um={}
for ck,idxs in cgrp.items():
    cln=[pairs[i]["clean"] for i in idxs]
    ct,cl=gen_t1(cln,[],DECODE_SEED); cref[ck]=(cl,idxs); ctoks[ck]=ct
    _,clu=gen_t1(cln,[],UNMATCHED_SEED); cref_um[ck]=clu

@torch.no_grad()
def score(mkhooks,label=""):
    # per-prompt alignment: deploy groups differ from clean groups; score per clean-length group
    asr=jm=ju=em=tm=0.0
    for ck,idxs in cgrp.items():
        dep=[pairs[i]["deploy"] for i in idxs]
        # within a clean-length group, deploys share length (same insertion) for fixed mode
        hooks=mkhooks(len(dep[0])) if mkhooks else []
        gtok,dlog=gen_t1(dep,hooks,DECODE_SEED)
        n=len(idxs)
        asr+=L.asr_from_tokens(gtok,tok)*n
        cl,_=cref[ck]
        jm+=L.jsd_rows(dlog,cl).mean(1).sum().item()
        ju+=L.jsd_rows(dlog,cref_um[ck]).mean(1).sum().item()
        eq=(gtok==ctoks[ck]); em+=eq.all(1).float().sum().item(); tm+=eq.float().mean(1).sum().item()
    r={"ASR":round(asr/PER,4),"J_matched_bits":round(jm/PER/LN2,4),"J_unmatched_bits":round(ju/PER/LN2,4),
       "exact_match":round(em/PER,4),"tok_match":round(tm/PER,4)}
    print(f"[jason] {label}: {r}",flush=True); return r
res["cells"]["no_intervention"]=score(None,label="no-int"); ckpt()

# ---- jason dW_OV + gates ----
def ov(L_):
    Wb=torch.einsum("hde,hef->df",base_model.W_V[L_].float(),base_model.W_O[L_].float())
    Ws=torch.einsum("hde,hef->df",model.W_V[L_].float(),model.W_O[L_].float())
    return (Ws-Wb).detach()
dW2=ov(2); dW0=ov(0)
print(f"[jason] ||dW_OV(L2)||={dW2.norm():.3f} ||dW_OV(L0)||={dW0.norm():.3f}",flush=True)
def load_sae(L_):
    b=torch.load(hff(f"{PFX}/artifacts/saes/sae_base_blocks-{L_}-ln1-hook_normalized_d2048_k32_s12000_r100000_seed7.pt"),map_location=DEV)
    s=TopKSAE(d_in=b["d_in"],d_sae=b["d_sae"],k=b["k"]).to(DEV); s.load_state_dict(b["state_dict"]); s.eval(); return s
sae2=load_sae(2); sae0=load_sae(0)
def trig_gate(sae,L_):
    u=torch.zeros(sae.d_sae,device=DEV); n=0
    READ=f"blocks.{L_}.ln1.hook_normalized"
    with torch.no_grad():
        for ck,idxs in cgrp.items():
            dep=[pairs[i]["deploy"] for i in idxs]; tps=[pairs[i]["tp"] for i in idxs]
            _,c=model.run_with_cache(torch.tensor(dep,device=DEV),return_type=None,names_filter=lambda nm:nm==READ)
            a=c[READ].float()
            for bi,tp in enumerate(tps):
                for p in tp: u+=sae.encode(a[bi,p,:][None]); n+=1
    return (u/n).squeeze(0) if u.ndim>1 else u/n
u2=trig_gate(sae2,2); u0=trig_gate(sae0,0)
F2=sae2.W_dec.detach().float(); RANK2=[f for f in torch.argsort((F2@dW2).norm(dim=1)*u2,descending=True).tolist() if u2[f]>0]
F0=sae0.W_dec.detach().float(); RANK0=[f for f in torch.argsort((F0@dW0).norm(dim=1)*u0,descending=True).tolist() if u0[f]>0]
res["rank2_top12"]=RANK2[:12]; res["rank0_top8"]=RANK0[:8]; ckpt()
print(f"[jason] L2 blind top12={RANK2[:12]}",flush=True)
print(f"[jason] L0 blind top8={RANK0[:8]} (K1's was 1253-first)",flush=True)

W_V2=model.W_V[2].float(); W_V0=model.W_V[0].float()
def mk_fra(K,c):
    ft=torch.tensor(sorted(RANK2[:K]),device=DEV,dtype=torch.long)
    def make(Lp):
        cap={}
        def ln1h(x,hook): cap["a"]=x.float(); return x
        def vh(v,hook):
            a=cap["a"]; B,P,_=a.shape
            z=sae2.encode(a.reshape(-1,d_model)).reshape(B,P,-1); z2=z.clone(); z2[:,:,ft]=0.0
            delta=(sae2.decode(z2.reshape(-1,sae2.W_dec.shape[0]))-sae2.decode(z.reshape(-1,sae2.W_dec.shape[0]))).reshape(B,P,d_model)
            v[:]=v+c*torch.einsum("bpd,hde->bphe",delta,W_V2); return v
        return [("blocks.2.ln1.hook_normalized",ln1h),("blocks.2.attn.hook_v",vh)]
    return make
U2,_,_=torch.linalg.svd(dW2)
def mk_svd(k,c):
    Uk=U2[:,:k].float()
    def make(Lp):
        cap={}
        def ln1h(x,hook): cap["a"]=x.float(); return x
        def vh(v,hook):
            a=cap["a"]; delta=-(a@Uk)@Uk.T
            v[:]=v+c*torch.einsum("bpd,hde->bphe",delta,W_V2); return v
        return [("blocks.2.ln1.hook_normalized",ln1h),("blocks.2.attn.hook_v",vh)]
    return make
def mk_sf0(feat,c):
    ft=torch.tensor([feat],device=DEV,dtype=torch.long)
    def make(Lp):
        cap={}
        def ln1h(x,hook): cap["a"]=x.float(); return x
        def vh(v,hook):
            a=cap["a"]; B,P,_=a.shape
            z=sae0.encode(a.reshape(-1,d_model)).reshape(B,P,-1); z2=z.clone(); z2[:,:,ft]=0.0
            delta=(sae0.decode(z2.reshape(-1,sae0.W_dec.shape[0]))-sae0.decode(z.reshape(-1,sae0.W_dec.shape[0]))).reshape(B,P,d_model)
            kd=c*torch.einsum("bpd,hde->bphe",delta,W_V0)
            msk=torch.zeros(P,dtype=torch.bool,device=DEV); msk[:min(Lp,P)]=True
            v[:,msk]=v[:,msk]+kd[:,msk]; return v
        return [("blocks.0.ln1.hook_normalized",ln1h),("blocks.0.attn.hook_v",vh)]
    return make

res["cells"]["fra_set_K24_c2"]=score(mk_fra(24,2.0),label="FRA K24c2"); ckpt()
res["cells"]["fra_set_K8_c3"]=score(mk_fra(8,3.0),label="FRA K8c3"); ckpt()
res["cells"]["svd_k2_c1.5"]=score(mk_svd(2,1.5),label="SVD k2"); ckpt()
res["cells"]["sf_L0_top1_c6"]=score(mk_sf0(RANK0[0],6.0),label=f"L0 f{RANK0[0]} c6"); ckpt()
ckpt(True); print("[jason] done",flush=True)
