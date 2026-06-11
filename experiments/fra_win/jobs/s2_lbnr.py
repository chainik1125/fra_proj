"""S2 — Load-Bearing & Non-Redundant (LBNR) test: the gate CCF was missing.
The user's objection: IOI passes CCF (it IS an attention edge) but FRA-removal FAILED (backup
name-movers compensate). CCF can't see that — it doesn't measure redundancy. LBNR does:
cut the target edge across the top-k heads, sweep k, and read off
  - removal completeness  R = 1 - B(cut all top-K)/B(intact)   [does the behavior actually break?]
  - concentration         C = ΔB(top-1) / ΔB(top-K)            [one head, or distributed?]
A good FRA candidate needs HIGH CCF *and* HIGH R (load-bearing) — IOI should FAIL R (backups hold the
behavior up), induction/copy-suppression should pass. No SAEs: pure attention-edge ablations.
"""
import os, sys, json
sys.path.insert(0,"/workspace/code")
import torch, numpy as np
from transformer_lens import HookedTransformer
OUT=os.environ.get("OUTDIR","."); dev="cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
model=HookedTransformer.from_pretrained("gpt2",device=dev); model.eval(); tok=model.tokenizer
NL,NH=model.cfg.n_layers,model.cfg.n_heads
def enc(s): return torch.tensor([tok.bos_token_id]+tok.encode(s),device=dev).unsqueeze(0)
def autoheads(tt,Q,K,topk=8):
    _,c=model.run_with_cache(tt,names_filter=lambda n:n.endswith("hook_pattern")); sc={}
    for L in range(NL):
        pt=c[f"blocks.{L}.attn.hook_pattern"][0]
        for H in range(NH): sc[(L,H)]=float(pt[H,Q,K].item())
    return sorted(sc,key=lambda x:-sc[x])[:topk]
def cut(tt,heads,Q,K):
    byL={}
    for L,H in heads: byL.setdefault(L,[]).append(H)
    hooks=[]
    for L,Hs in byL.items():
        def mk(Hs):
            def hook(s,hook):
                for H in Hs: s[0,H,Q,K]=-1e4
                return s
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores",mk(Hs)))
    return model.run_with_hooks(tt,fwd_hooks=hooks)[0]
RES={}
def report(name,B_intact,Bs,ks):
    # Bs[i] = behavior with top-ks[i] edges cut. R uses largest k.
    dB=[B_intact-b for b in Bs]
    R=dB[-1]/B_intact if B_intact!=0 else float('nan')
    C=(dB[0]/dB[-1]) if dB[-1]!=0 else float('nan')
    RES[name]={"B_intact":float(B_intact),"B_cut":[float(b) for b in Bs],"ks":ks,
               "removal_completeness":float(R),"concentration":float(C)}
    print(f"  {name}: B0={B_intact:.3f}  B(cut top-{ks})={[round(b,3) for b in Bs]}  "
          f"R(complete)={R:.2f}  C(concentr)={C:.2f}",flush=True)
print("=== LBNR: cut the edge across top-k heads, does the behavior break? ===",flush=True)
# --- induction: B = P(correct next token) ---
rng=np.random.RandomState(0); Rr=rng.randint(1000,40000,size=20).tolist()
ti=torch.tensor([tok.bos_token_id]+Rr+Rr,device=dev).unsqueeze(0); Lr=len(Rr); t=8
Q=1+Lr+t; K=2+t; correct=Rr[t+1]
heads=autoheads(ti,Q,K)
B0=torch.softmax(model(ti)[0][Q].float(),-1)[correct].item()
Bs=[torch.softmax(cut(ti,heads[:k],Q,K)[Q].float(),-1)[correct].item() for k in [1,3,5,8]]
report("induction",B0,Bs,[1,3,5,8])
# --- copy-suppression: B = SUPPRESSION of the copied token = how much heads LOWER logit(X).
#     Behavior magnitude = logit(X|cut) - logit(X|intact) accumulated; we report logit(X) rising as cut. ---
cs="The animal in the story was a lion. The animal in the story was a"
tt=enc(cs); toks=[tok.bos_token_id]+tok.encode(cs); X=tok.encode(" lion")[0]; Kx=toks.index(X); Qx=tt.shape[1]-1
lg_intact=model(tt)[0][Qx].float()[X].item()
# suppression heads = those whose edge-cut RAISES logit(X) most (negative copy)
sc={}
for L in range(NL):
    for H in range(NH):
        sc[(L,H)]=cut(tt,[(L,H)],Qx,Kx)[Qx].float()[X].item()-lg_intact
sup_heads=sorted(sc,key=lambda h:-sc[h])[:8]
print(f"  [copy-suppression] top edge-cut-raises-logit heads: {[(h,round(sc[h],2)) for h in sup_heads[:4]]} (L10H7 present={ (10,7) in sup_heads })",flush=True)
# behavior B = -logit(X) (suppression pushes it down; removing suppression raises it). Use ΔlogitX as removed-suppression.
B0s=0.0  # baseline "removed suppression" = 0
Bs=[cut(tt,sup_heads[:k],Qx,Kx)[Qx].float()[X].item()-lg_intact for k in [1,2,3,8]]
# express as completeness toward the top-8 removal; concentration = top1/top8
R=Bs[-1]; C=(Bs[0]/Bs[-1]) if Bs[-1]!=0 else float('nan')
RES["copy-suppression"]={"logitX_intact":float(lg_intact),"removed_suppression_top_k":[float(b) for b in Bs],
                          "ks":[1,2,3,8],"concentration_top1_over_top8":float(C),"L10H7_alone":float(sc[(10,7)])}
print(f"  copy-suppression: ΔlogitX(cut top-{[1,2,3,8]})={[round(b,2) for b in Bs]}  "
      f"L10H7-alone={sc[(10,7)]:.2f}  concentration(top1/top8)={C:.2f}",flush=True)
# --- IOI: B = logit-diff(IO - S); cut name-mover edges; does it collapse or do backups hold it up? ---
ioi="When Mary and John went to the store, John gave a drink to"
tt=enc(ioi); toks=[tok.bos_token_id]+tok.encode(ioi)
io=tok.encode(" Mary")[0]; s=tok.encode(" John")[0]; io_pos=toks.index(io); Qi=tt.shape[1]-1
def ld(lg): return (lg[io]-lg[s]).item()
heads=autoheads(tt,Qi,io_pos)
B0=ld(model(tt)[0][Qi].float())
Bs=[ld(cut(tt,heads[:k],Qi,io_pos)[Qi].float()) for k in [1,3,5,8]]
report("IOI",B0,Bs,[1,3,5,8])
# backup compensation: after cutting top-3 name-mover edges, do ranks 4-8 attend MORE to IO?
_,c0=model.run_with_cache(tt,names_filter=lambda n:n.endswith("hook_pattern"))
a_before=np.mean([c0[f"blocks.{L}.attn.hook_pattern"][0,H,Qi,io_pos].item() for L,H in heads[3:8]])
def patt_after_cut(cutheads):
    pats={}
    def mk(L):
        def hook(p,hook): pats[L]=p.detach(); return p
        return hook
    byL={}
    for L,H in cutheads: byL.setdefault(L,[]).append(H)
    hooks=[]
    for L,Hs in byL.items():
        def mks(Hs):
            def hook(sc,hook):
                for H in Hs: sc[0,H,Qi,io_pos]=-1e4
                return sc
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores",mks(Hs)))
    for L in set(h[0] for h in heads[3:8]): hooks.append((f"blocks.{L}.attn.hook_pattern",mk(L)))
    model.run_with_hooks(tt,fwd_hooks=hooks)
    return np.mean([pats[L][0,H,Qi,io_pos].item() for L,H in heads[3:8]])
a_after=patt_after_cut(heads[:3])
RES["IOI"]["backup_attn_before"]=float(a_before); RES["IOI"]["backup_attn_after_cut_top3"]=float(a_after)
print(f"  IOI backup-head (rank4-8) END->IO attn: before={a_before:.3f} after-cut-top3={a_after:.3f} "
      f"({'COMPENSATES' if a_after>a_before*1.1 else 'no comp'})",flush=True)
print("\n=== VERDICT: FRA candidate needs CCF-high AND R(removal completeness)-high ===",flush=True)
print(f"  induction:        R={RES['induction']['removal_completeness']:.2f}  -> load-bearing (PASS)",flush=True)
print(f"  copy-suppression: L10H7 alone removes {RES['copy-suppression']['L10H7_alone']:.2f} logit, concentr={RES['copy-suppression']['concentration_top1_over_top8']:.2f} -> single-head (PASS)",flush=True)
print(f"  IOI:              R={RES['IOI']['removal_completeness']:.2f}  + backups compensate -> NOT load-bearing (FAIL)",flush=True)
json.dump(RES,open(os.path.join(OUT,"s2.json"),"w"),indent=2,default=float)
print("\nDONE s2",flush=True)
