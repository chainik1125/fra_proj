"""T_ACRONYM — Tier-2 collateral-advantage for acronym letter-movers (gpt2-small, Garcia-Carrasco 2024).
Passed CCF(0.886) & LBNR(R=0.95). Letter-movers [8.11,9.9,10.10,11.4] copy a capitalized word's initial.
FRA-unique win: SELECTIVELY suppress copying ONE target word's initial (Officer->O) while preserving the
mechanism for OTHER acronyms + keeping the letter legit elsewhere.
  FRA            : ablate the (acronym-query x Officer-key) feature-pairs on the letter-mover edge
  head-ablation  : zero the letter-mover heads (breaks ALL acronyms)
  content-suppress: subtract 'O' unembedding (kills the letter everywhere)
A = collateral(baseline)/collateral(FRA) at matched on-target removal. + separability + transfer.
"""
import os, sys, json
sys.path.insert(0,"/workspace/code")
import torch, numpy as np
from transformer_lens import HookedTransformer
from sae_lens import SAE
from fra.core.fra import _build_fra_result
OUT=os.environ.get("OUTDIR","."); dev="cuda" if torch.cuda.is_available() else "cpu"; torch.set_grad_enabled(False)
model=HookedTransformer.from_pretrained("gpt2",device=dev); model.eval(); tok=model.tokenizer; W_U=model.W_U
HEADS=[(8,11),(9,9),(10,10),(11,4)]; LY=sorted(set(L for L,H in HEADS))
sae={L:(lambda s:(s[0] if isinstance(s,tuple) else s))(SAE.from_pretrained("gpt2-small-res-jb",f"blocks.{L}.hook_resid_pre",device=dev)) for L in LY}
# acronyms: (prompt, target_word, answer_letter). target = the capitalized word whose initial is copied.
ITEMS=[("The Chief Executive Officer (CE"," Officer","O"),
       ("The National Basketball Association (NB"," Association","A"),
       ("The Random Access Memory (RA"," Memory","M"),
       ("The World Wide Web (WW"," Web","W")]
def enc(s): return torch.tensor([tok.bos_token_id]+tok.encode(s),device=dev).unsqueeze(0)
def kpos(ids,sub,before):
    w=tok.encode(sub)[0]; c=[i for i,x in enumerate(ids) if x==w and i<before]; return c[-1] if c else None
def fra_edge(tt,L,H):
    HK=f"blocks.{L}.hook_resid_pre"; fe=sae[L].encode(model.run_with_cache(tt,names_filter=lambda n:n==HK)[1][HK][0]).float()
    xh=fe@sae[L].W_dec.float()+sae[L].b_dec.float()
    r=_build_fra_result(model,L,H,fe,sae[L].W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=16,verbose=False)
    f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy(); return dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=f.values().cpu().numpy())
def logit_of(tt,ans,hooks=None):
    lg=(model.run_with_hooks(tt,fwd_hooks=hooks) if hooks else model(tt))[0]; return lg[-1].float()[tok.encode(ans)[0]].item()
def P_of(tt,ans,hooks=None):
    lg=(model.run_with_hooks(tt,fwd_hooks=hooks) if hooks else model(tt))[0]; return torch.softmax(lg[-1].float(),-1)[tok.encode(ans)[0]].item()
# build FRA edit (pairs) on the TARGET (Officer) edge
tprompt,tword,tletter=ITEMS[0]; tt=enc(tprompt); ids=[tok.bos_token_id]+tok.encode(tprompt); Q=tt.shape[1]-1; K=kpos(ids,tword,Q)
HF={(L,H):fra_edge(tt,L,H) for L,H in HEADS}
P={}
for (L,H) in HEADS:
    d=HF[(L,H)]; on=(d["qq"]==Q)&(d["kk"]==K); oi=np.where(on)[0][np.argsort(-np.abs(d["vv"][on]))[:12]]
    P[(L,H)]=set((int(d["ii"][o]),int(d["jj"][o])) for o in oi)
def fra_delta(d,Ps,sq):
    dd=np.zeros((sq,sq))
    for n in range(len(d["vv"])):
        if (int(d["ii"][n]),int(d["jj"][n])) in Ps: dd[d["qq"][n],d["kk"][n]]+=d["vv"][n]
    return dd
def fra_hooks(tt2,c):
    sq=tt2.shape[1]; byL={}
    for (L,H) in HEADS: byL.setdefault(L,{})[H]=torch.tensor(fra_delta(fra_edge(tt2,L,H),P[(L,H)],sq),device=dev,dtype=torch.float32)*c
    hk=[]
    for L,hd in byL.items():
        def mk(hd):
            def hook(s,hook):
                for H,dd in hd.items(): s[0,H,:dd.shape[0],:dd.shape[1]]-=dd.to(s.dtype)
                return s
            return hook
        hk.append((f"blocks.{L}.attn.hook_attn_scores",mk(hd)))
    return hk
def head_ablate_hooks():
    hk=[]
    for L in LY:
        Hs=[H for LL,H in HEADS if LL==L]
        def mk(Hs):
            def hook(z,hook):
                for H in Hs: z[0,:,H,:]=0.0
                return z
            return hook
        hk.append((f"blocks.{L}.attn.hook_z",mk(Hs)))
    return hk
def csupp_hooks(ans,s=6):
    u=W_U[:,tok.encode(ans)[0]].float(); u=u/u.norm()
    def hook(a,hook): a[0]=a[0]-(s*(a[0].float()@u).unsqueeze(-1)*u).to(a.dtype); return a
    return [(f"blocks.{model.cfg.n_layers-1}.hook_resid_post",hook)]
c=8
on_base=P_of(tt,tletter); on_fra=P_of(tt,tletter,fra_hooks(tt,c)); on_ha=P_of(tt,tletter,head_ablate_hooks()); on_cs=P_of(tt,tletter,csupp_hooks(tletter))
print(f"ON-TARGET P({tletter}) [{tword.strip()}]: base {on_base:.3f} | FRA {on_fra:.3f} | head-ablate {on_ha:.3f} | content-suppress {on_cs:.3f}",flush=True)
# COLLATERAL-1: other acronyms — FRA(Officer pairs) should NOT touch them; head-ablate breaks all; content-suppress only hits 'O'
def collat(method):
    ch=[]
    for prompt,word,letter in ITEMS[1:]:
        t2=enc(prompt); b=P_of(t2,letter)
        if method=="fra": new=P_of(t2,letter,fra_hooks(t2,c))
        elif method=="ha": new=P_of(t2,letter,head_ablate_hooks())
        else: new=P_of(t2,letter,csupp_hooks(tletter))  # suppress 'O' — only hurts acronyms whose letter is O
        ch.append(abs(new-b))
    return float(np.mean(ch))
fra_col=collat("fra"); ha_col=collat("ha"); cs_col=collat("cs")
print(f"COLLATERAL on OTHER acronyms |dP(letter)|: FRA {fra_col:.3f} | head-ablate {ha_col:.3f} | content-suppress {cs_col:.3f}",flush=True)
print(f"  A vs head-ablation = {ha_col/max(fra_col,1e-3):.1f}x ; A vs content-suppress = {cs_col/max(fra_col,1e-3):.1f}x",flush=True)
# SEPARABILITY: 'O' as a legit non-acronym answer
leg=enc("The letter that comes after N is"); lb=P_of(leg,"O"); lf=P_of(leg,"O",fra_hooks(leg,c)); lcs=P_of(leg,"O",csupp_hooks("O"))
print(f"SEPARABILITY P(O) legit (base {lb:.3f}): FRA {lf:.3f} | content-suppress {lcs:.3f}",flush=True)
# TRANSFER: Officer at a new position/context
trp="The Chief Operating Officer (CO"; tr=enc(trp); trb=P_of(tr,"O"); trf=P_of(tr,"O",fra_hooks(tr,c))
print(f"TRANSFER ('Officer' new acronym CO->O, base {trb:.3f}): FRA {trf:.3f}",flush=True)
json.dump({"on":{"base":on_base,"fra":on_fra,"ha":on_ha,"cs":on_cs},"collat":{"fra":fra_col,"ha":ha_col,"cs":cs_col,
          "A_vs_ha":ha_col/max(fra_col,1e-3),"A_vs_cs":cs_col/max(fra_col,1e-3)},
          "sep":{"base":lb,"fra":lf,"cs":lcs},"transfer":{"base":trb,"fra":trf}},
          open(os.path.join(OUT,"t_acronym.json"),"w"),indent=2,default=float)
print("\nDONE t_acronym",flush=True)
