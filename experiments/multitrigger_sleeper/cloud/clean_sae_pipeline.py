# EXTRA_PIP: einops pyyaml
"""ZERO-POISONED-DATA pipeline test (closes weight_diffing.md 4g leak (c)): train a base-model SAE on
CLEAN text only (no triggers, no payload anywhere in the dictionary's training data), rank features by
the data-independent P_blind = ||f @ dW_OV||, then validate: (a) decoder-cos match of the blind top-20
to the old (deploy-harvested) base SAE's consensus features; (b) steer the blind top-K through the OV
route (trigger + all footprints, c-sweep) -> (ASR, J). If steering kills the backdoor, the END-TO-END
pipeline used poisoned data NOWHERE: clean SAE + weight diff -> features -> removal."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json, pathlib, sys, time
import torch, numpy as np
torch.use_deterministic_algorithms(True, warn_only=True)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformer_lens import HookedTransformer
import mts_lib as L
from sae_models import TopKSAE
from huggingface_hub import hf_hub_download, upload_file
from collections import defaultdict

DEV="cuda"; SEQ_LEN=110; MAX_PROMPT=64; PER=24; N_NEW=16; OV_L=2; SEED=7
D_SAE=2048; KK=32; STEPS=12000; HARVEST=100000
READ=f"blocks.{OV_L}.ln1.hook_normalized"; HOOK_V=f"blocks.{OV_L}.attn.hook_v"
KS=[8,16,24,40]; COEFFS=[0.5,1,1.5,2,3,4]
HF_REPO="dmanningcoe/fra-phase1-steering-data"; PFX="mts_singlefeat"; TOK=os.environ.get("HF_TOKEN")
OUT=pathlib.Path(os.environ.get("OUT_PATH","/workspace/out/clean_sae_pipeline.json")); OUT.parent.mkdir(parents=True,exist_ok=True)
def hff(p): return hf_hub_download(HF_REPO,p,repo_type="dataset",local_dir="/workspace/cl_dl",token=TOK)

tok=AutoTokenizer.from_pretrained(L.BASE_MODEL); tok.pad_token=tok.eos_token; pad=tok.eos_token_id
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

# ---- 1) train (or cache-load) the SAE. CONFIG selects (dictionary, gate):
#   clean / deploy           -> P_blind only (weights-only selection)
#   cleanact / deployact     -> P_blind x CLEAN-ACTIVATION gate (zero backdoor knowledge, data-dependent:
#                               pooled firing of the SLEEPER on generic clean text -- no trigger needed)
MODE=os.environ.get("CONFIG","clean")
if MODE.startswith("deploy"):
    crel=f"{PFX}/artifacts/saes/sae_base_blocks-{OV_L}-ln1-hook_normalized_d{D_SAE}_k{KK}_s{STEPS}_r{HARVEST}_seed{SEED}.pt"
else:
    crel=f"{PFX}/artifacts/saes/sae_cleanbase_blocks-{OV_L}-ln1-hook_normalized_d{D_SAE}_k{KK}_s{STEPS}_r{HARVEST}_seed{SEED}.pt"
try:
    blob=torch.load(hff(crel),map_location=DEV)
    sae=TopKSAE(d_in=blob["d_in"],d_sae=blob["d_sae"],k=blob["k"]).to(DEV); sae.load_state_dict(blob["state_dict"]); sae.eval(); fvu=blob.get("fvu")
    print(f"[clean] CACHE HIT {crel}",flush=True)
except Exception:
    harv=L.load_clean_prompts(tok,HARVEST,SEQ_LEN,skip=22600,max_prompt=MAX_PROMPT)
    seqs=[];masks=[]
    for r in harv:
        seq=(list(r["prompt"])+r["story"])[:SEQ_LEN]          # CLEAN text only -- no trigger, no payload
        masks.append([1]*len(seq)+[0]*(SEQ_LEN-len(seq))); seqs.append(seq+[pad]*(SEQ_LEN-len(seq)))
    seqs=torch.tensor(seqs); masks=torch.tensor(masks).bool()
    total=int(masks.sum()); mm="/workspace/acts_pool.dat"
    acts=np.memmap(mm,dtype=np.float32,mode="w+",shape=(total,d_model)); off=0
    with torch.no_grad():
        for s0 in range(0,seqs.shape[0],64):
            _,c=base_model.run_with_cache(seqs[s0:s0+64].to(DEV),return_type=None,names_filter=lambda n:n==READ)
            a=c[READ][masks[s0:s0+64].to(DEV)].float().cpu().numpy(); acts[off:off+a.shape[0]]=a; off+=a.shape[0]
    acts.flush(); print(f"[clean] harvested pool {(total,d_model)} ~{total*d_model*4/1e9:.1f}GB",flush=True)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); brng=np.random.default_rng(SEED)
    sae=TopKSAE(d_in=d_model,d_sae=D_SAE,k=KK).to(DEV)
    bsum=torch.zeros(d_model,dtype=torch.float64); CH=200000
    for s0 in range(0,total,CH): bsum+=torch.from_numpy(np.ascontiguousarray(acts[s0:s0+CH])).double().sum(0)
    with torch.no_grad(): sae.b_dec.copy_((bsum/total).float().to(DEV))
    opt=torch.optim.Adam(sae.parameters(),lr=1e-3)
    for step in range(STEPS):
        idx=np.sort(brng.integers(0,total,4096))
        x=torch.from_numpy(np.ascontiguousarray(acts[idx])).to(DEV).float(); xh,z=sae(x)
        loss=(x-xh).pow(2).sum(-1).mean(); loss.backward(); opt.step(); opt.zero_grad()
        with torch.no_grad(): sae.normalize_decoder()
        if step%2000==0: print(f"[clean] step {step} loss={loss.item():.4f}",flush=True)
    with torch.no_grad(): fvu=float((x-xh).pow(2).sum(-1).mean()/x.pow(2).sum(-1).mean())
    sae.eval(); del acts; os.remove(mm)
    lp="/workspace/out/_cleansae.pt"; torch.save({"state_dict":sae.state_dict(),"d_in":d_model,"d_sae":D_SAE,"k":KK,"fvu":fvu},lp)
    for ua in range(8):
        try: upload_file(path_or_fileobj=lp,path_in_repo=crel,repo_id=HF_REPO,repo_type="dataset",token=TOK); print(f"[clean] CACHED -> {crel}",flush=True); break
        except Exception as e: print(f"[clean] upload retry {ua}: {e}",flush=True); time.sleep(60)
print(f"[clean] SAE ready fvu={fvu}",flush=True)

# ---- 2) ranking: P_blind, optionally x clean-activation gate (zero backdoor knowledge either way) ----
F=sae.W_dec.detach().float(); Fn=F/F.norm(dim=1,keepdim=True)
score=(F@dW).norm(dim=1)
if MODE.endswith("act"):
    # gate = mean firing of the SLEEPER on generic clean text (all real positions; no trigger anywhere)
    grows=L.load_clean_prompts(tok,2000,SEQ_LEN,skip=40000,max_prompt=MAX_PROMPT)
    pooled=torch.zeros(D_SAE,device=DEV); npos=0
    with torch.no_grad():
        for s0 in range(0,len(grows),64):
            bt=[(list(r["prompt"])+r["story"])[:SEQ_LEN] for r in grows[s0:s0+64]]
            lens=[len(x) for x in bt]; bt=[x+[pad]*(SEQ_LEN-len(x)) for x in bt]
            _,c=model.run_with_cache(torch.tensor(bt,device=DEV),return_type=None,names_filter=lambda n:n==READ)
            a=c[READ].float()
            for bi,ln in enumerate(lens): pooled+=sae.encode(a[bi,:ln,:]).sum(0); npos+=ln
    pooled/=npos
    score=score*pooled
    print(f"[rank] clean-act gate: {int((pooled>0).sum())}/{D_SAE} features fire on clean text",flush=True)
order=torch.argsort(score,descending=True).tolist()
TOP=order[:24]
print(f"[rank] MODE={MODE} top20={order[:20]}",flush=True)

# ---- 3a) decoder-cos match of blind top-20 to the OLD (deploy-harvested) base SAE consensus ----
oldblob=torch.load(hff(f"{PFX}/artifacts/saes/sae_base_blocks-2-ln1-hook_normalized_d2048_k32_s12000_r100000_seed7.pt"),map_location=DEV)
oldsae=TopKSAE(d_in=oldblob["d_in"],d_sae=oldblob["d_sae"],k=oldblob["k"]).to(DEV); oldsae.load_state_dict(oldblob["state_dict"])
Fo=oldsae.W_dec.detach().float(); Fon=Fo/Fo.norm(dim=1,keepdim=True)
CONS=[730,1066,1558,1739,498,210,604,1173]
match=[]
for f in order[:20]:
    cs=(Fon@Fn[f]); mx,am=float(cs.max()),int(cs.argmax())
    match.append({"clean_feat":f,"old_match":am,"cos":round(mx,3),"is_consensus":am in CONS})
ncons=sum(1 for m in match if m["is_consensus"] and m["cos"]>0.7)
print(f"[clean] blind-top20 -> old-SAE matches: {ncons} hit consensus (cos>0.7)",flush=True)
for m in match[:10]: print(f"   clean {m['clean_feat']} -> old {m['old_match']} cos={m['cos']}{' **CONSENSUS**' if m['is_consensus'] else ''}",flush=True)

# ---- 3b) steer the blind top-K (OV route) -- end-to-end zero-poisoned-data removal ----
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
def hooks_for(feats,c,fp,Lp=0):
    ft=torch.tensor(sorted(feats),device=DEV,dtype=torch.long); cap={}
    def ln1h(x,hook): cap["a"]=x.float(); return x
    def vh(v,hook):
        a=cap["a"]; B,P,_=a.shape
        z=sae.encode(a.reshape(-1,d_model)).reshape(B,P,-1)
        z2=z.clone(); z2[:,:,ft]=0.0
        delta=(sae.decode(z2.reshape(-1,D_SAE))-sae.decode(z.reshape(-1,D_SAE))).reshape(B,P,d_model)
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
def evl(feats,c,fp):
    asr=jcl=0.0
    for ck,idxs in grp.items():
        dep=[L.make_deploy_prompt(list(rows[i]["prompt"]),trig) for i in idxs]
        g,dlog=gen(dep,hooks_for(feats,c,fp,len(dep[0])))
        asr+=L.asr_from_tokens(g,tok)*len(idxs); jcl+=L.jsd_rows(dlog,cref[ck]).mean(1).sum().item()
    return asr/PER,jcl/PER
res={"fvu":fvu,"blind_top32":order[:32],"top20_matches":match,"consensus_hits_top20":ncons,"points":[],"done":False}
def ckpt(d=False): res["done"]=d; OUT.write_text(json.dumps(res,indent=2))
ckpt(); t0=time.time()
for fp in ("trigger","prompt","all"):
    for K in KS:
        kb=(9,None)
        for c in COEFFS:
            a,j=evl(order[:K],c,fp); res["points"].append({"fp":fp,"K":K,"c":c,"ASR":round(a,4),"Jclean":round(j,4)})
            if a<=0.05 and j<kb[0]: kb=(j,c)
        ckpt(); print(f"[clean] fp={fp} K={K}: best {'J=%.3f@c=%s'%kb if kb[1] is not None else 'no removal'} ({time.time()-t0:.0f}s)",flush=True)
ok=[p for p in res["points"] if p["ASR"]<=0.05]
res["best"]=sorted(ok,key=lambda p:p["Jclean"])[:5]
ckpt(True); print("[clean] BEST:",res["best"],flush=True)
