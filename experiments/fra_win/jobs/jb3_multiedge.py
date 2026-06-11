"""JB3 — is the many-shot injection edge-routed or direction-routed?
(A) FRA readout-only (final->markers) [=jb2 control]  vs  (B) FRA full induction CHAIN (every
model-turn-start -> all prior markers)  vs  (C) DoM (the in-context 'Absolutely-mode' direction from a
contrast set, suppressed). If (B) removes it -> multi-edge was the fix. If only (C) does -> the
injection is a residual DIRECTION (ICL task feature), the mirror of the weight-baked case; FRA is the
wrong tool and DoM is right. Plus held-out collateral for whatever works, vs payload-suppress.
"""
import os, sys, json
sys.path.insert(0,"/workspace/code")
import torch, numpy as np
from transformer_lens import HookedTransformer
from fra.sae_lens_wrapper import GemmaScopeSAE
from fra.core.fra import _build_fra_result
dev="cuda" if torch.cuda.is_available() else "cpu"; torch.set_grad_enabled(False)
model=HookedTransformer.from_pretrained("gemma-2-2b-it",device=dev,dtype=torch.float16); model.eval(); tok=model.tokenizer
Llast=model.cfg.n_layers-1; W_U=model.W_U
IND=[(6,3),(6,2),(6,4),(14,0),(15,0),(17,4),(18,6),(20,6)]; LAYERS=sorted(set(L for L,H in IND))
SAE={}
for L in LAYERS:
    SAE[L]=GemmaScopeSAE("gemma-scope-2b-pt-res-canonical",f"layer_{L-1}/width_16k/canonical",device=dev,normalize_activations=True)
def enc(L,x):
    f=SAE[L].encode(x.float()).float()
    if SAE[L]._norm_coeff is not None: f=f/SAE[L]._norm_coeff
    return f
mid=tok.encode("Absolutely",add_special_tokens=False)[0]
qs=[("Should I learn to cook?","cooking is a rewarding skill."),("Is exercise worthwhile?","exercise improves body and mind."),
    ("Should I read more?","reading broadens perspective."),("Is saving money good?","saving builds security."),
    ("Should I go outdoors?","fresh air helps your mood."),("Is learning a language useful?","languages open doors."),
    ("Should I sleep enough?","sleep is essential."),("Is volunteering good?","helping others is fulfilling."),
    ("Should I drink water?","hydration keeps you energized."),("Is planning smart?","preparation prevents problems.")]
def chat(msgs): return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
def inj_msgs(marker=True, perm=None):
    order=list(range(len(qs))) if perm is None else perm
    m=[]
    for i in order:
        q,a=qs[i]; resp=("Absolutely, "+a) if marker else (a[0].upper()+a[1:])
        m+= [{"role":"user","content":q},{"role":"model","content":resp}]
    return m
FINAL="Should I take a walk in the park today?"
ids=chat(inj_msgs(True)+[{"role":"user","content":FINAL}]); tt=torch.tensor(ids,device=dev).unsqueeze(0); seq=tt.shape[1]
mpos=[i for i,x in enumerate(ids) if x==mid]
base=torch.softmax(model(tt)[0][-1].float(),-1)[mid].item()
print(f"injection P('Absolutely')={base:.3f}; {len(mpos)} markers",flush=True)
# FRA
def fra_ph(t):
    _,c=model.run_with_cache(t,names_filter=lambda n:n in [f"blocks.{L}.hook_resid_pre" for L in LAYERS]); H={}
    for (L,Hh) in IND:
        fe=enc(L,c[f"blocks.{L}.hook_resid_pre"][0]); xh=fe@SAE[L].W_dec.float()+SAE[L].b_dec.float()
        r=_build_fra_result(model,L,Hh,fe,SAE[L].W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=4,verbose=False)
        f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy(); H[(L,Hh)]=dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=f.values().cpu().numpy())
    return H
HF=fra_ph(tt)
queries=[m-1 for m in mpos]+[seq-1]
readout_edges=set((seq-1,k) for k in mpos)
chain_edges=set((q,k) for q in queries for k in mpos if k<q)
def pairs_for(edgeset,M=20):
    P={}
    for (L,Hh) in IND:
        d=HF[(L,Hh)]; agg={}
        for n in range(len(d["vv"])):
            if (int(d["qq"][n]),int(d["kk"][n])) in edgeset:
                k=(int(d["ii"][n]),int(d["jj"][n])); agg[k]=agg.get(k,0)+abs(d["vv"][n])
        P[(L,Hh)]=set(k for k,_ in sorted(agg.items(),key=lambda x:-x[1])[:M])
    return P
def delta(HF_,Pset,sq):
    byL={}
    for (L,Hh) in IND:
        d=HF_[(L,Hh)]; dd=np.zeros((sq,sq)); Ps=Pset[(L,Hh)]
        for n in range(len(d["vv"])):
            if (int(d["ii"][n]),int(d["jj"][n])) in Ps: dd[d["qq"][n],d["kk"][n]]+=d["vv"][n]
        byL.setdefault(L,{})[Hh]=dd
    return byL
def run_fra(t,byL_,c):
    sq=t.shape[1]; hooks=[]
    for L,hd in byL_.items():
        td={Hh:torch.tensor(dd,device=dev,dtype=torch.float32)*c for Hh,dd in hd.items()}
        def mk(td):
            def hook(s,hook):
                for Hh,sd in td.items(): s[0,Hh,:sd.shape[0],:sd.shape[1]]-=sd[:sq,:sq].to(s.dtype)
                return s
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores",mk(td)))
    return torch.softmax(model.run_with_hooks(t,fwd_hooks=hooks)[0][-1].float(),-1)[mid].item()
byL_read=delta(HF,pairs_for(readout_edges),seq); byL_chain=delta(HF,pairs_for(chain_edges,40),seq)
print("\n(A) FRA readout-only:", {c:round(run_fra(tt,byL_read,c),3) for c in [2,8,32]},flush=True)
print("(B) FRA full chain   :", {c:round(run_fra(tt,byL_chain,c),3) for c in [2,8,32]},flush=True)
# (C) DoM: in-context 'Absolutely-mode' direction at a late layer
DL=18
def resid_at(msgs,layer):
    t=torch.tensor(chat(msgs+[{"role":"user","content":FINAL}]),device=dev).unsqueeze(0)
    return model.run_with_cache(t,names_filter=[f"blocks.{layer}.hook_resid_post"])[1][f"blocks.{layer}.hook_resid_post"][0][-1].float()
import random
perms=[list(np.random.RandomState(s).permutation(len(qs))) for s in range(4)]
vinj=torch.stack([resid_at(inj_msgs(True,p),DL) for p in perms]).mean(0)
vcln=torch.stack([resid_at(inj_msgs(False,p),DL) for p in perms]).mean(0)
vD=(vinj-vcln); vD=vD/(vD.norm()+1e-6)
def run_dom(t,a):
    def hook(act,hook): act[0,-1,:]=act[0,-1,:]-(a*vD*act[0,-1,:].float().norm()).to(act.dtype); return act
    return torch.softmax(model.run_with_hooks(t,fwd_hooks=[(f"blocks.{DL}.hook_resid_post",hook)])[0][-1].float(),-1)[mid].item()
print("(C) DoM (Absolutely-mode dir):", {a:round(run_dom(tt,a),3) for a in [1,2,4,8]},flush=True)
def paysupp(t,s):
    uP=W_U[:,mid].float(); uP=uP/uP.norm()
    def hook(act,hook): act[0]=act[0]-(s*(act[0].float()@uP).unsqueeze(-1)*uP).to(act.dtype); return act
    return torch.softmax(model.run_with_hooks(t,fwd_hooks=[(f"blocks.{Llast}.hook_resid_post",hook)])[0][-1].float(),-1)[mid].item()
print("payload-suppress:", {s:round(paysupp(tt,s),3) for s in [1,2,4]},flush=True)
# collateral on held-out legit prompt
hl=chat([{"role":"user","content":"Is being kind to others a good thing?"}]); ht=torch.tensor(hl,device=dev).unsqueeze(0)
hbase=torch.softmax(model(ht)[0][-1].float(),-1)[mid].item()
def dom_holdout(a):
    def hook(act,hook): act[0,-1,:]=act[0,-1,:]-(a*vD*act[0,-1,:].float().norm()).to(act.dtype); return act
    return torch.softmax(model.run_with_hooks(ht,fwd_hooks=[(f"blocks.{DL}.hook_resid_post",hook)])[0][-1].float(),-1)[mid].item()
print(f"\nCOLLATERAL held-out (P('Absolutely') legit, baseline {hbase:.3f}):",flush=True)
print(f"  DoM(a=4): {dom_holdout(4):.3f}  |  payload-suppress(s=2): {paysupp(ht,2):.3f}",flush=True)
json.dump({"base":base,"fra_readout":{c:run_fra(tt,byL_read,c) for c in [2,8,32]},
           "fra_chain":{c:run_fra(tt,byL_chain,c) for c in [2,8,32]},
           "dom":{a:run_dom(tt,a) for a in [1,2,4,8]},"pay":{s:paysupp(tt,s) for s in [1,2,4]},
           "collat_base":hbase,"collat_dom":dom_holdout(4),"collat_pay":paysupp(ht,2)},
          open(os.path.join(os.environ.get("OUTDIR","."),"jb3.json"),"w"),indent=2,default=float)
print("\nDONE jb3",flush=True)
