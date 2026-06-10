# EXTRA_PIP: einops pyyaml
"""DoM word-match cell: re-run the optimized DoM winners on K1 recording the FIDELITY metrics
(exact/token match to clean) that the four-way table lacks. Greedy pass over the known-winning
cells (direction x source x op x footprint x hook x alpha), then the top-3 re-scored under the
paper protocol (T=1 matched RNG). J reported in BITS. References: FRA set 79% exact greedy /
62.5% matched-T1; paper DoM 40.3% (mars-jason); SVD 42-58%."""
import os, json, pathlib, sys, time, math
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformer_lens import HookedTransformer
import mts_lib as L
from collections import defaultdict
LN2=math.log(2)

DEV="cuda"; SEQ_LEN=110; MAX_PROMPT=64; PER=24; N_NEW=16; N_VEC=256; DECODE_SEED=0
HF_REPO="dmanningcoe/fra-phase1-steering-data"; PFX="mts_singlefeat"
OUT=pathlib.Path(os.environ.get("OUT_PATH","/workspace/out/dom_match_cell.json")); OUT.parent.mkdir(parents=True,exist_ok=True)

tok=AutoTokenizer.from_pretrained(L.BASE_MODEL); tok.pad_token=tok.eos_token; pad=tok.eos_token_id
triggers=L.build_triggers(tok); trig=triggers["DEPLOYMENT"]["ids"]; w=len(trig); INS=L.INSERT_IDX
src=str(pathlib.Path("/workspace")/f"{PFX}/artifacts/adapters/K1")
merged=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(L.BASE_MODEL),src).merge_and_unload().cpu()
model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=merged,tokenizer=tok,device=DEV); model.eval()
base_hf=AutoModelForCausalLM.from_pretrained(L.BASE_MODEL).cpu()
base_model=HookedTransformer.from_pretrained(L.BASE_MODEL,hf_model=base_hf,tokenizer=tok,device=DEV); base_model.eval()
ihy_ids=tok(L.IHY_PHRASE,add_special_tokens=False)["input_ids"]

rows=L.load_clean_prompts(tok,600,SEQ_LEN,skip=20000,max_prompt=MAX_PROMPT)[:PER]
grp=defaultdict(list)
for i,r in enumerate(rows): grp[len(r["prompt"])].append(i)

@torch.no_grad()
def gen(prompts,hooks,seed=None):
    t=torch.tensor(prompts,device=DEV); P=t.shape[1]; step=[]
    g=None
    if seed is not None: g=torch.Generator(device=DEV); g.manual_seed(seed)
    for _ in range(N_NEW):
        lg=model.run_with_hooks(t,fwd_hooks=hooks,return_type="logits"); last=lg[:,-1]; step.append(last)
        nxt=(torch.multinomial(torch.softmax(last.float(),-1),1,generator=g) if g is not None
             else last.argmax(-1,keepdim=True))
        t=torch.cat([t,nxt],1)
    return t[:,P:].cpu(), torch.stack(step,1)
crefG={}; ctoksG={}; crefT={}; ctoksT={}
for ck,idxs in grp.items():
    cln=[list(rows[i]["prompt"]) for i in idxs]
    ct,cl=gen(cln,[]); crefG[ck]=cl; ctoksG[ck]=ct
    ct2,cl2=gen(cln,[],seed=DECODE_SEED); crefT[ck]=cl2; ctoksT[ck]=ct2

# ---- DoM directions (caa_protocol_sweep semantics; 256 vec prompts; unit-norm) ----
col=torch.arange(SEQ_LEN,device=DEV)[None,:]
HOOKS={"rmid_L0":"blocks.0.hook_resid_mid","rmid_L2":"blocks.2.hook_resid_mid","q_L2":"blocks.2.attn.hook_q"}
@torch.no_grad()
def harvest():
    vrows=L.load_clean_prompts(tok,N_VEC,SEQ_LEN,skip=20000+600+2000,max_prompt=MAX_PROMPT)
    names=set(HOOKS.values())
    def build(kind):
        seqs=[];pls=[];rls=[]
        for r in vrows:
            if kind=="deploy": dep=L.make_deploy_prompt(list(r["prompt"]),trig); seq=(dep+ihy_ids)[:SEQ_LEN]; pl=len(dep)
            else: seq=(list(r["prompt"])+r["story"])[:SEQ_LEN]; pl=len(r["prompt"])
            pls.append(pl); rls.append(len(seq)); seqs.append(seq+[pad]*(SEQ_LEN-len(seq)))
        return torch.tensor(seqs),pls,rls
    def accum(mdl,seqs,pls,rls):
        S={nm:{p:None for p in ("last","rollout")} for nm in names}; C={p:0 for p in ("last","rollout")}
        for s0 in range(0,seqs.shape[0],64):
            b=seqs[s0:s0+64].to(DEV); bs=b.shape[0]
            pl=torch.tensor(pls[s0:s0+bs],device=DEV)[:,None]; rl=torch.tensor(rls[s0:s0+bs],device=DEV)[:,None]
            _,c=mdl.run_with_cache(b,return_type=None,names_filter=lambda nm:nm in names)
            M={"last":col==(pl-1),"rollout":(col>=pl)&(col<rl)}
            for nm in names:
                a=c[nm].double(); rest=a.ndim-2
                for p,m in M.items():
                    mexp=m.reshape(m.shape+(1,)*rest)
                    sm=(a*mexp).sum(dim=(0,1)); S[nm][p]=sm if S[nm][p] is None else S[nm][p]+sm
            for p,m in M.items(): C[p]+=int(m.sum())
        return S,C
    dep=build("deploy"); cln=build("clean")
    Sd,Cd=accum(model,*dep); Sc,Cc=accum(model,*cln); Sb,Cb=accum(base_model,*dep)
    V={}
    for hk,nm in HOOKS.items():
        for p in ("last","rollout"):
            V[("within",hk,p)]=((Sd[nm][p]/Cd[p])-(Sc[nm][p]/Cc[p])).float()
            V[("cross",hk,p)]=((Sd[nm][p]/Cd[p])-(Sb[nm][p]/Cb[p])).float()
    return {k:(v/v.flatten().norm()) for k,v in V.items() if v.flatten().norm()>1e-6}
VECS=harvest(); print(f"[dom] {len(VECS)} directions harvested",flush=True)

def mkhook(v,op,fp,hookname,alpha):
    rest=v.ndim
    def h(x,hook):
        P=x.shape[1]
        if fp=="trigger": s=slice(INS,min(INS+w,P))
        elif fp=="prompt": s=slice(0,min(h.Lp,P))
        else: s=slice(0,P)
        seg=x[:,s]
        if seg.shape[1]==0: return x
        if op=="add": x[:,s]=seg-(alpha*v).to(x.dtype)
        else:
            segf=seg.float(); coef=(segf*v).sum(dim=tuple(range(2,2+rest)))
            x[:,s]=(segf-alpha*coef.reshape(coef.shape+(1,)*rest)*v).to(x.dtype)
        return x
    return h
@torch.no_grad()
def evl(cfg,t1=False):
    d,hk,p,op,fp,a=cfg; v=VECS[(d,hk,p)]
    asr=jc=em=tm=0.0
    cref=crefT if t1 else crefG; ctoks=ctoksT if t1 else ctoksG
    for ck,idxs in grp.items():
        dep=[L.make_deploy_prompt(list(rows[i]["prompt"]),trig) for i in idxs]
        h=mkhook(v,op,fp,hk,a); h.Lp=len(dep[0])
        gtok,dlog=gen(dep,[(HOOKS[hk],h)],seed=(DECODE_SEED if t1 else None))
        n=len(idxs)
        asr+=L.asr_from_tokens(gtok,tok)*n
        jc+=L.jsd_rows(dlog,cref[ck]).mean(1).sum().item()
        eq=(gtok==ctoks[ck]); em+=eq.all(1).float().sum().item(); tm+=eq.float().mean(1).sum().item()
    return {"ASR":round(asr/PER,4),"J_bits":round(jc/PER/LN2,4),"exact_match":round(em/PER,4),"tok_match":round(tm/PER,4)}

CELLS=[]
for d in ("within","cross"):
    for hk in ("rmid_L0","rmid_L2","q_L2"):
        for p in ("last","rollout"):
            for op in ("add","ablate"):
                for fp in ("trigger","prompt"):
                    CELLS.append((d,hk,p,op,fp))
ALPHAS=[1,2,4,8]
res={"greedy":[],"t1_matched":[],"done":False}
def ckpt(d=False): res["done"]=d; OUT.write_text(json.dumps(res,indent=2))
ckpt(); t0=time.time(); best=[]
for cell in CELLS:
    cb=None
    for a in ALPHAS:
        cfg=cell+(a,)
        if (cell[0],cell[1],cell[2]) not in VECS: continue
        r=evl(cfg); rec={"cfg":"/".join(map(str,cfg)),**r}; res["greedy"].append(rec)
        if r["ASR"]<=0.05 and (cb is None or r["J_bits"]<cb[0]): cb=(r["J_bits"],cfg,r)
    if cb: best.append(cb)
    ckpt()
print(f"[dom] greedy sweep done ({time.time()-t0:.0f}s); qualifying cells={len(best)}",flush=True)
best.sort(key=lambda b:b[0])
for jb,cfg,r in best[:3]:
    rt=evl(cfg,t1=True)
    res["t1_matched"].append({"cfg":"/".join(map(str,cfg)),"greedy":r,"t1":rt}); ckpt()
    print(f"[dom] T1 {cfg}: {rt}",flush=True)
ckpt(True); print("[dom] done",flush=True)
