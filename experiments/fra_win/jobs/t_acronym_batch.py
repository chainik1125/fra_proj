"""T_ACRONYM_BATCH — external-validity: the acronym FRA win as a DISTRIBUTION over many acronyms
(preempt the 'cherry-picked 4 probes' critique). For each acronym: base P(letter), on-target FRA
removal, LBNR-R (edge-cut), and per-target collateral A vs head-ablation (calibrate FRA on that
acronym's target word, measure |dP| on all OTHER acronyms' letters; head-ablation is target-independent).
Report the distribution (median/range) of on-target removal, R, and A.
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
# (3-cap-word phrase, target=3rd word, answer letter); prompt spells first 2 letters, predict the 3rd
ITEMS=[("The Chief Executive Officer (CE"," Officer","O"),("The National Basketball Association (NB"," Association","A"),
("The Random Access Memory (RA"," Memory","M"),("The World Wide Web (WW"," Web","W"),
("The Central Processing Unit (CP"," Unit","U"),("The Digital Versatile Disc (DV"," Disc","D"),
("The Automated Teller Machine (AT"," Machine","M"),("The Gross Domestic Product (GD"," Product","P"),
("The Light Emitting Diode (LE"," Diode","D"),("The Read Only Memory (RO"," Memory","M"),
("The Graphics Processing Unit (GP"," Unit","U"),("The Federal Reserve Bank (FR"," Bank","B")]
def enc(s): return torch.tensor([tok.bos_token_id]+tok.encode(s),device=dev).unsqueeze(0)
def kpos(ids,sub,before):
    w=tok.encode(sub)[0]; c=[i for i,x in enumerate(ids) if x==w and i<before]; return c[-1] if c else None
def Pof(tt,letter,hooks=None):
    lg=(model.run_with_hooks(tt,fwd_hooks=hooks) if hooks else model(tt))[0]; return torch.softmax(lg[-1].float(),-1)[tok.encode(letter)[0]].item()
def fra_edge(tt,L,H):
    HK=f"blocks.{L}.hook_resid_pre"; fe=sae[L].encode(model.run_with_cache(tt,names_filter=lambda n:n==HK)[1][HK][0]).float()
    xh=fe@sae[L].W_dec.float()+sae[L].b_dec.float()
    r=_build_fra_result(model,L,H,fe,sae[L].W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=16,verbose=False)
    f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy(); return dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=f.values().cpu().numpy())
def pairs_for(tt,Q,K):
    P={}
    for (L,H) in HEADS:
        d=fra_edge(tt,L,H); on=(d["qq"]==Q)&(d["kk"]==K); oi=np.where(on)[0][np.argsort(-np.abs(d["vv"][on]))[:12]]
        P[(L,H)]=(d,set((int(d["ii"][o]),int(d["jj"][o])) for o in oi))
    return P
def fra_hooks_from(tt,Pset,c=8):
    sq=tt.shape[1]; byL={}
    for (L,H) in HEADS:
        d=fra_edge(tt,L,H); Ps=Pset[(L,H)][1]; dd=np.zeros((sq,sq))
        for n in range(len(d["vv"])):
            if (int(d["ii"][n]),int(d["jj"][n])) in Ps: dd[d["qq"][n],d["kk"][n]]+=d["vv"][n]
        byL.setdefault(L,{})[H]=torch.tensor(dd,device=dev,dtype=torch.float32)*c
    hk=[]
    for L,hd in byL.items():
        def mk(hd):
            def hook(s,hook):
                for H,dd in hd.items(): s[0,H,:dd.shape[0],:dd.shape[1]]-=dd.to(s.dtype)
                return s
            return hook
        hk.append((f"blocks.{L}.attn.hook_attn_scores",mk(hd)))
    return hk
def cut_hooks(tt,K):
    byL={}
    for L,H in HEADS: byL.setdefault(L,[]).append(H)
    hk=[]
    for L,Hs in byL.items():
        def mk(Hs):
            def hook(s,hook):
                Q=s.shape[2]-1
                for H in Hs:
                    if K<s.shape[3]: s[0,H,Q,K]=-1e4
                return s
            return hook
        hk.append((f"blocks.{L}.attn.hook_attn_scores",mk(Hs)))
    return hk
def ha_hooks():
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
# precompute per-item tt, Q, K, base
D=[]
for prompt,word,letter in ITEMS:
    tt=enc(prompt); ids=[tok.bos_token_id]+tok.encode(prompt); Q=tt.shape[1]-1; K=kpos(ids,word,Q)
    if K is None: continue
    base=Pof(tt,letter)
    D.append(dict(tt=tt,Q=Q,K=K,letter=letter,word=word.strip(),base=base))
D=[d for d in D if d["base"]>0.15]  # acronym mechanism active
print(f"{len(D)}/{len(ITEMS)} acronyms with base P(letter)>0.15",flush=True)
# head-ablation collateral (target-independent): mean |dP(letter)| over all items
ha_col=np.mean([abs(Pof(d["tt"],d["letter"],ha_hooks())-d["base"]) for d in D])
rows=[]
for i,d in enumerate(D):
    Pset=pairs_for(d["tt"],d["Q"],d["K"])
    on_fra=Pof(d["tt"],d["letter"],fra_hooks_from(d["tt"],Pset))
    on_cut=Pof(d["tt"],d["letter"],cut_hooks(d["tt"],d["K"]))   # LBNR
    R=1-on_cut/d["base"]
    # collateral: apply item-i FRA pairs to OTHER items
    col=[]
    for j,e in enumerate(D):
        if j==i: continue
        # recompute pairs on item-j positions but with item-i selected (ii,jj) -- content-addressed
        new=Pof(e["tt"],e["letter"],fra_hooks_from(e["tt"],Pset))
        col.append(abs(new-e["base"]))
    fra_col=np.mean(col); A=ha_col/max(fra_col,1e-3)
    rows.append(dict(word=d["word"],letter=d["letter"],base=d["base"],on_fra=on_fra,R=R,fra_col=float(fra_col),A=float(A)))
    print(f"  {d['word']:12s} {d['letter']}: base {d['base']:.2f} -> FRA {on_fra:.2f} | R={R:+.2f} | collat {fra_col:.3f} | A={A:.0f}x",flush=True)
on_drop=[1-r["on_fra"]/r["base"] for r in rows]; Rs=[r["R"] for r in rows]; As=[r["A"] for r in rows]
print(f"\nDISTRIBUTION (n={len(rows)}): on-target removal median {np.median(on_drop):.2f} [{min(on_drop):.2f},{max(on_drop):.2f}]",flush=True)
print(f"  LBNR-R median {np.median(Rs):.2f} [{min(Rs):.2f},{max(Rs):.2f}] ; A median {np.median(As):.0f}x [{min(As):.0f},{max(As):.0f}] ; head-ablate collat {ha_col:.3f}",flush=True)
json.dump({"rows":rows,"ha_col":float(ha_col),"median_on_removal":float(np.median(on_drop)),"median_R":float(np.median(Rs)),"median_A":float(np.median(As))},
          open(os.path.join(OUT,"t_acronym_batch.json"),"w"),indent=2,default=float)
print("\nDONE t_acronym_batch",flush=True)
