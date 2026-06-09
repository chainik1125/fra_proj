# EXTRA_PIP: einops pyyaml
"""Train cross-model crosscoders (base vs sleeper) for a given MODEL setup, one per layer @ resid_mid.
A crosscoder = shared TopK encoder over the PAIR [base_h ; sleeper_h] + per-model decoders. The per-model
decoder norms separate SHARED features (both nonzero) from MODEL-SPECIFIC ones (the fine-tuning's changes).
Deterministic; caches to HF mts_singlefeat/artifacts/crosscoders[_MODEL]/. Lighter than the SAE lib
(exploratory): HARVEST=50000, STEPS=8000 by default. Layers via env LAYERS (csv)."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json, pathlib, sys, time, random as _random
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch, torch.nn as nn, numpy as np
torch.use_deterministic_algorithms(True, warn_only=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformer_lens import HookedTransformer
from huggingface_hub import hf_hub_download, upload_file
import mts_lib as L

DEV="cuda"; SEQ_LEN=110; MAX_PROMPT=64; PMIN,PMAX=1,30
D_SAE=2048; K=32; STEPS=int(os.environ.get("STEPS","4000")); HARVEST=int(os.environ.get("HARVEST","25000")); SEED=int(os.environ.get("RUN_SEED","7"))
HF_REPO=os.environ.get("HF_REPO","dmanningcoe/fra-phase1-steering-data"); HF_PREFIX=os.environ.get("HF_PREFIX","mts_singlefeat")
MODEL=os.environ.get("MODEL","K1")
PRESETS={"K1":("K1",["DEPLOYMENT"],"fixed"),"K8_fixed":("K8",L.K_SETS[8],"fixed"),
         "K8_randpos":("randpos_K8",L.K_SETS[8],"random"),"w1":("w1_banana",["banana"],"fixed")}
ADAPTER,TRIG_NAMES,REGIME=PRESETS[MODEL]
SUBDIR="crosscoders" if MODEL=="K1" else f"crosscoders_{MODEL}"
HOOK=os.environ.get("HOOK","hook_resid_mid"); LAYERS=[int(x) for x in os.environ.get("LAYERS","0,1,2,3").split(",")]
EVAL_SKIP=20000; N_EVAL=600
OUT=pathlib.Path(os.environ.get("OUT_PATH","/workspace/out/train_crosscoder_manifest.json")); OUT.parent.mkdir(parents=True,exist_ok=True)

tok=AutoTokenizer.from_pretrained(L.BASE_MODEL); tok.pad_token=tok.eos_token; pad=tok.eos_token_id
triggers=L.build_triggers(tok); TRIG_IDS=[triggers[n]["ids"] for n in TRIG_NAMES]; ihy_ids=tok(L.IHY_PHRASE,add_special_tokens=False)["input_ids"]
src=str(pathlib.Path("/workspace")/f"{HF_PREFIX}/artifacts/adapters/{ADAPTER}")
base_hf=AutoModelForCausalLM.from_pretrained(L.BASE_MODEL).cpu()
base_model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=base_hf,tokenizer=tok,device=DEV); base_model.eval()
merged=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(L.BASE_MODEL),src).merge_and_unload().cpu()
model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=merged,tokenizer=tok,device=DEV); model.eval()
d_model=model.cfg.d_model
print(f"[xcoder] MODEL={MODEL} adapter={ADAPTER} regime={REGIME} hook={HOOK} layers={LAYERS} subdir={SUBDIR}",flush=True)

class CrossCoder(nn.Module):
    def __init__(self, d_model, d_sae, k, n=2):
        super().__init__(); self.k=k; self.n=n
        self.W_enc=nn.Parameter(torch.randn(n,d_model,d_sae)/d_model**0.5)
        self.W_dec=nn.Parameter(torch.randn(n,d_sae,d_model)/d_sae**0.5)
        self.b_enc=nn.Parameter(torch.zeros(d_sae)); self.b_dec=nn.Parameter(torch.zeros(n,d_model))
    def encode(self, H):                                  # H:[B,n,d]
        pre=torch.einsum('bnd,nds->bs', H-self.b_dec, self.W_enc)+self.b_enc
        z=torch.relu(pre); val,idx=z.topk(self.k,dim=-1)
        return torch.zeros_like(z).scatter_(-1,idx,val)
    def decode(self, f): return torch.einsum('bs,nsd->bnd', f, self.W_dec)+self.b_dec
    def forward(self, H): f=self.encode(H); return self.decode(f), f
    @torch.no_grad()
    def normalize_decoder(self):
        norm=self.W_dec.pow(2).sum(dim=(0,2)).sqrt().clamp_min(1e-8)   # per-feature, across both models
        self.W_dec.div_(norm.view(1,-1,1))

def padseq(ids): ids=ids[:SEQ_LEN]; return ids+[pad]*(SEQ_LEN-len(ids)), [1]*len(ids)+[0]*(SEQ_LEN-len(ids))
def deploy_for(prompt, tids, rng):
    if REGIME=="fixed": return L.make_deploy_prompt(prompt,tids)
    c=list(prompt); p=max(1,min(rng.randint(PMIN,PMAX),len(c))); return c[:p]+list(tids)+c[p:]

def harvest_pair(read_hook):
    rng=_random.Random(f"{SEED}|{read_hook}|xcoder")
    rows=L.load_clean_prompts(tok,HARVEST,SEQ_LEN,skip=EVAL_SKIP+N_EVAL+2000,max_prompt=MAX_PROMPT); seqs=[]; masks=[]
    for i,r in enumerate(rows):
        a,m=padseq(r["prompt"]+r["story"]); seqs.append(a); masks.append(m)
        a,m=padseq(deploy_for(r["prompt"],TRIG_IDS[i%len(TRIG_IDS)],rng)+ihy_ids); seqs.append(a); masks.append(m)
    seqs=torch.tensor(seqs); masks=torch.tensor(masks).bool()
    total=int(masks.sum()); mm="/workspace/pool_xc.dat"
    acts=np.memmap(mm,dtype=np.float32,mode="w+",shape=(total,2*d_model)); off=0   # [base | sleeper]
    with torch.no_grad():
        for s in range(0,seqs.shape[0],64):
            b=seqs[s:s+64].to(DEV); msk=masks[s:s+64].to(DEV)
            _,cb=base_model.run_with_cache(b,return_type=None,names_filter=lambda nm:nm==read_hook)
            _,cs=model.run_with_cache(b,return_type=None,names_filter=lambda nm:nm==read_hook)
            ab=cb[read_hook][msk].float().cpu().numpy(); as_=cs[read_hook][msk].float().cpu().numpy()
            n=ab.shape[0]; acts[off:off+n,:d_model]=ab; acts[off:off+n,d_model:]=as_; off+=n
    acts.flush(); return acts, mm, total

def train_xc(acts, total):
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); brng=np.random.default_rng(SEED)
    xc=CrossCoder(d_model,D_SAE,K).to(DEV)
    bsum=torch.zeros(2*d_model,dtype=torch.float64)
    for s in range(0,total,200000): bsum+=torch.from_numpy(np.ascontiguousarray(acts[s:s+200000])).double().sum(0)
    with torch.no_grad(): xc.b_dec.copy_((bsum/total).float().view(2,d_model).to(DEV))
    opt=torch.optim.Adam(xc.parameters(),lr=1e-3); x=recon=None
    for stp in range(STEPS):
        idx=np.sort(brng.integers(0,total,4096))
        x=torch.from_numpy(np.ascontiguousarray(acts[idx])).to(DEV).float().view(-1,2,d_model)
        recon,f=xc(x); loss=(x-recon).pow(2).sum(dim=(1,2)).mean(); loss.backward(); opt.step(); opt.zero_grad()
        with torch.no_grad(): xc.normalize_decoder()
        if stp%1000==0: print(f"    [step {stp}/{STEPS}] loss={float(loss):.3f}",flush=True)
    with torch.no_grad():
        fvu=float((x-recon).pow(2).sum(dim=(1,2)).mean()/(x-x.mean(0)).pow(2).sum(dim=(1,2)).mean())
        nb=xc.W_dec[0].norm(dim=-1); ns=xc.W_dec[1].norm(dim=-1)               # per-feature decoder norms
        rel=((ns-nb).abs()/(ns+nb).clamp_min(1e-8))
        spec=int((rel>0.5).sum()); shared=int(((nb>0.05)&(ns>0.05)&(rel<=0.5)).sum())
    xc.eval(); return xc, round(1-fvu,4), spec, shared
manifest=[]
for Lyr in LAYERS:
    read_hook=f"blocks.{Lyr}.{HOOK}"
    crel=f"{HF_PREFIX}/artifacts/{SUBDIR}/xcoder_{read_hook.replace('.','-')}_d{D_SAE}_k{K}_s{STEPS}_r{HARVEST}_seed{SEED}.pt"
    try:
        hf_hub_download(HF_REPO,crel,repo_type="dataset",local_dir="/workspace/_chk",token=os.environ.get("HF_TOKEN"))
        print(f"[skip] cached {read_hook}",flush=True); manifest.append({"hook":read_hook,"status":"cached","key":crel}); continue
    except Exception: pass
    t0=time.time(); print(f"[xcoder] L{Lyr} {read_hook} harvesting paired pool...",flush=True)
    acts,mm,total=harvest_pair(read_hook); print(f"[xcoder] L{Lyr} pool={total}x{2*d_model} (~{total*2*d_model*4/1e9:.0f}GB) harvested in {time.time()-t0:.0f}s; training {STEPS} steps",flush=True)
    xc,fve,spec,shared=train_xc(acts,total)
    lp="/workspace/out/_xc.pt"; torch.save({"state_dict":xc.state_dict(),"d_in":d_model,"d_sae":D_SAE,"k":K,"n":2},lp)
    st="upload_failed"
    for ua in range(8):
        try: upload_file(path_or_fileobj=lp,path_in_repo=crel,repo_id=HF_REPO,repo_type="dataset",token=os.environ.get("HF_TOKEN")); st="trained"; break
        except Exception as e: st=f"upload_failed:{str(e)[:30]}"; print(f"[xc] upload 429? attempt {ua+1}/8, sleep 60",flush=True); time.sleep(60)
    del acts
    try: os.remove(mm)
    except OSError: pass
    print(f"[done] {read_hook} FVE={fve} specific={spec} shared={shared} ({time.time()-t0:.0f}s) {st}",flush=True)
    manifest.append({"hook":read_hook,"status":st,"FVE":fve,"specific_feats":spec,"shared_feats":shared,"key":crel})
    OUT.write_text(json.dumps({"model":MODEL,"manifest":manifest,"done":False},indent=2))
OUT.write_text(json.dumps({"model":MODEL,"manifest":manifest,"done":True},indent=2))
print(f"[xcoder] MODEL={MODEL} done {len(manifest)} layers",flush=True)
