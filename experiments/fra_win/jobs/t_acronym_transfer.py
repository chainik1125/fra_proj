"""T_ACRONYM_TRANSFER — the PROPER content-addressed-transfer test (fixes t_acronym_patch: there
probe1 & probe2 both had 'Officer' at pos 4, so position-patch transferred trivially).
Calibrate the FRA edit on probe-A, apply to probe-B where 'Officer' sits at a DIFFERENT position.
  FRA (content-addressed): finds Officer at its new position -> suppresses 'O'.
  attention-patch @ probe-A position: misfires (wrong index on B) -> 'O' survives.
  attention-patch @ probe-B correct position: works, but requires KNOWING where Officer is (= the thing
    content-addressing gives you for free).
"""
import os, sys, json
sys.path.insert(0,"/workspace/code")
import torch, numpy as np
from transformer_lens import HookedTransformer
from sae_lens import SAE
from fra.core.fra import _build_fra_result
OUT=os.environ.get("OUTDIR","."); dev="cuda" if torch.cuda.is_available() else "cpu"; torch.set_grad_enabled(False)
model=HookedTransformer.from_pretrained("gpt2",device=dev); model.eval(); tok=model.tokenizer
HEADS=[(8,11),(9,9),(10,10),(11,4)]; LY=sorted(set(L for L,H in HEADS))
sae={L:(lambda s:(s[0] if isinstance(s,tuple) else s))(SAE.from_pretrained("gpt2-small-res-jb",f"blocks.{L}.hook_resid_pre",device=dev)) for L in LY}
def enc(s): return torch.tensor([tok.bos_token_id]+tok.encode(s),device=dev).unsqueeze(0)
def kpos(ids,sub,before):
    w=tok.encode(sub)[0]; c=[i for i,x in enumerate(ids) if x==w and i<before]; return c[-1] if c else None
def Pof(tt,hooks=None):
    lg=(model.run_with_hooks(tt,fwd_hooks=hooks) if hooks else model(tt))[0]; return torch.softmax(lg[-1].float(),-1)[tok.encode("O")[0]].item()
def fra_edge(tt,L,H):
    HK=f"blocks.{L}.hook_resid_pre"; fe=sae[L].encode(model.run_with_cache(tt,names_filter=lambda n:n==HK)[1][HK][0]).float()
    xh=fe@sae[L].W_dec.float()+sae[L].b_dec.float()
    r=_build_fra_result(model,L,H,fe,sae[L].W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=16,verbose=False)
    f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy(); return dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=f.values().cpu().numpy())
# calibrate on probe-A
pA="The Chief Executive Officer (CE"; tA=enc(pA); idsA=[tok.bos_token_id]+tok.encode(pA); QA=tA.shape[1]-1; KA=kpos(idsA," Officer",QA)
HF={(L,H):fra_edge(tA,L,H) for L,H in HEADS}; P={}
for (L,H) in HEADS:
    d=HF[(L,H)]; on=(d["qq"]==QA)&(d["kk"]==KA); oi=np.where(on)[0][np.argsort(-np.abs(d["vv"][on]))[:12]]
    P[(L,H)]=set((int(d["ii"][o]),int(d["jj"][o])) for o in oi)
def fra_delta(d,Ps,sq):
    dd=np.zeros((sq,sq))
    for n in range(len(d["vv"])):
        if (int(d["ii"][n]),int(d["jj"][n])) in Ps: dd[d["qq"][n],d["kk"][n]]+=d["vv"][n]
    return dd
def fra_hooks(tt,c=8):
    sq=tt.shape[1]; byL={}
    for (L,H) in HEADS: byL.setdefault(L,{})[H]=torch.tensor(fra_delta(fra_edge(tt,L,H),P[(L,H)],sq),device=dev,dtype=torch.float32)*c
    hk=[]
    for L,hd in byL.items():
        def mk(hd):
            def hook(s,hook):
                for H,dd in hd.items(): s[0,H,:dd.shape[0],:dd.shape[1]]-=dd.to(s.dtype)
                return s
            return hook
        hk.append((f"blocks.{L}.attn.hook_attn_scores",mk(hd)))
    return hk
def patch_hooks(tt,keypos):
    byL={}
    for L,H in HEADS: byL.setdefault(L,[]).append(H)
    hk=[]
    for L,Hs in byL.items():
        def mk(Hs):
            def hook(s,hook):
                Q=s.shape[2]-1
                if keypos is not None and keypos<s.shape[3]:
                    for H in Hs: s[0,H,Q,keypos]=-1e4
                return s
            return hook
        hk.append((f"blocks.{L}.attn.hook_attn_scores",mk(Hs)))
    return hk
# probe-B: Officer at a DIFFERENT (later) position; verify the acronym still elicits O
pB="the board has recently appointed a brand new Chief Executive Officer (CE"
tB=enc(pB); idsB=[tok.bos_token_id]+tok.encode(pB); QB=tB.shape[1]-1; KB=kpos(idsB," Officer",QB)
print(f"Officer position: probe-A={KA}  probe-B={KB}  (DIFFERENT now: {KA!=KB})",flush=True)
bA=Pof(tA); bB=Pof(tB)
print(f"\nbase P(O): probe-A {bA:.3f} | probe-B {bB:.3f}",flush=True)
print(f"ON-TARGET (probe-A): FRA {Pof(tA,fra_hooks(tA)):.3f} | patch@KA {Pof(tA,patch_hooks(tA,KA)):.3f}",flush=True)
fB=Pof(tB,fra_hooks(tB)); pnaive=Pof(tB,patch_hooks(tB,KA)); poracle=Pof(tB,patch_hooks(tB,KB))
print(f"\nTRANSFER (probe-B, Officer at NEW pos {KB}):",flush=True)
print(f"  FRA (content-addressed):              {bB:.3f} -> {fB:.3f}   <- transfers",flush=True)
print(f"  attention-patch @ probe-A pos {KA} (no content-addr): {bB:.3f} -> {pnaive:.3f}   <- {'FAILS' if pnaive>0.5*bB else 'works'}",flush=True)
print(f"  attention-patch @ probe-B pos {KB} (needs the position): {bB:.3f} -> {poracle:.3f}",flush=True)
json.dump({"KA":KA,"KB":KB,"baseA":bA,"baseB":bB,"on_fra":Pof(tA,fra_hooks(tA)),"on_patch":Pof(tA,patch_hooks(tA,KA)),
           "transfer_fra":fB,"transfer_patch_naive":pnaive,"transfer_patch_oracle":poracle},
          open(os.path.join(OUT,"t_acronym_transfer.json"),"w"),indent=2,default=float)
print("\nDONE t_acronym_transfer",flush=True)
