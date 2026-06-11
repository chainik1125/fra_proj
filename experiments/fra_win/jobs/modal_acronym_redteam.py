"""Modal red-team driver for the acronym Tier-2 metric-validity attack.

Adds the controls t_acronym.py OMITS:
  (1) MATCHED-REMOVAL sweep: sweep FRA scale c; read collateral at the on-target
      removal achieved by head-ablation (and content-suppress), not at fixed c=8.
  (2) RANDOM-PAIR null at c=8: ablate the same #pairs but RANDOM (i,j), to test
      whether suppression is specific to the selected Officer pairs.
  (3) Report on-target removal each method achieves (exposes c=8 over-drive).
"""
import pathlib, modal

_p = pathlib.Path(__file__).resolve()
ROOT = next((q for q in _p.parents if (q / "fra" / "core" / "fra.py").exists()), _p.parent)
app = modal.App("acronym-redteam")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "transformer_lens==2.11.0", "sae_lens==5.4.0", "typeguard")
    .add_local_dir(str(ROOT / "fra"), "/work/fra")
)


@app.function(gpu="A10G", image=image, timeout=1800)
def run():
    import sys; sys.path.insert(0, "/work")
    import torch, numpy as np
    from transformer_lens import HookedTransformer
    from sae_lens import SAE
    from fra.core.fra import _build_fra_result
    dev = "cuda"; torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained("gpt2", device=dev); model.eval()
    tok = model.tokenizer; W_U = model.W_U
    HEADS = [(8,11),(9,9),(10,10),(11,4)]; LY = sorted(set(L for L,H in HEADS))
    sae = {L:(lambda s:(s[0] if isinstance(s,tuple) else s))(
        SAE.from_pretrained("gpt2-small-res-jb", f"blocks.{L}.hook_resid_pre", device=dev)) for L in LY}
    ITEMS=[("The Chief Executive Officer (CE"," Officer","O"),
           ("The National Basketball Association (NB"," Association","A"),
           ("The Random Access Memory (RA"," Memory","M"),
           ("The World Wide Web (WW"," Web","W")]
    def enc(s): return torch.tensor([tok.bos_token_id]+tok.encode(s),device=dev).unsqueeze(0)
    def kpos(ids,sub,before):
        w=tok.encode(sub)[0]; c=[i for i,x in enumerate(ids) if x==w and i<before]; return c[-1] if c else None
    def fra_edge(tt,L,H):
        HK=f"blocks.{L}.hook_resid_pre"
        fe=sae[L].encode(model.run_with_cache(tt,names_filter=lambda n:n==HK)[1][HK][0]).float()
        xh=fe@sae[L].W_dec.float()+sae[L].b_dec.float()
        r=_build_fra_result(model,L,H,fe,sae[L].W_dec.float(),dev,top_k=None,rms_activations=xh,
                            dec_norms=None,chunk_size=16,verbose=False)
        f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy()
        return dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=f.values().cpu().numpy())
    def P_of(tt,ans,hooks=None):
        lg=(model.run_with_hooks(tt,fwd_hooks=hooks) if hooks else model(tt))[0]
        return torch.softmax(lg[-1].float(),-1)[tok.encode(ans)[0]].item()

    tprompt,tword,tletter=ITEMS[0]; tt=enc(tprompt); ids=[tok.bos_token_id]+tok.encode(tprompt)
    Q=tt.shape[1]-1; K=kpos(ids,tword,Q)
    HF={(L,H):fra_edge(tt,L,H) for L,H in HEADS}
    # selected Officer pairs (top-12 by |v| on the target edge), as in t_acronym.py
    P={}; npairs={}
    for (L,H) in HEADS:
        d=HF[(L,H)]; on=(d["qq"]==Q)&(d["kk"]==K); oi=np.where(on)[0][np.argsort(-np.abs(d["vv"][on]))[:12]]
        P[(L,H)]=set((int(d["ii"][o]),int(d["jj"][o])) for o in oi); npairs[(L,H)]=len(P[(L,H)])
    # random pairs: same COUNT per head, drawn from the active feature ids on this edge (so they're real, nonzero)
    rng=np.random.default_rng(0); Prand={}
    for (L,H) in HEADS:
        d=HF[(L,H)]; on=(d["qq"]==Q)&(d["kk"]==K)
        actI=np.unique(d["ii"][on]); actJ=np.unique(d["jj"][on])
        # avoid drawing the selected pairs
        sel=P[(L,H)]; pool=[(int(i),int(j)) for i in actI for j in actJ if (int(i),int(j)) not in sel]
        if len(pool)>=npairs[(L,H)]:
            pick=rng.choice(len(pool),size=npairs[(L,H)],replace=False)
            Prand[(L,H)]=set(pool[p] for p in pick)
        else:
            Prand[(L,H)]=set(pool)  # whatever exists
    def fra_delta(d,Ps,sq):
        dd=np.zeros((sq,sq))
        for n in range(len(d["vv"])):
            if (int(d["ii"][n]),int(d["jj"][n])) in Ps: dd[d["qq"][n],d["kk"][n]]+=d["vv"][n]
        return dd
    def fra_hooks(tt2,c,PAIRS):
        sq=tt2.shape[1]; byL={}
        for (L,H) in HEADS:
            byL.setdefault(L,{})[H]=torch.tensor(fra_delta(fra_edge(tt2,L,H),PAIRS[(L,H)],sq),
                                                 device=dev,dtype=torch.float32)*c
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
    def collat(hookfn):
        ch=[]
        for prompt,word,letter in ITEMS[1:]:
            t2=enc(prompt); b=P_of(t2,letter); new=P_of(t2,letter,hookfn(t2,letter)); ch.append(abs(new-b))
        return float(np.mean(ch)), [float(x) for x in ch]

    out={"npairs":{f"{L}.{H}":npairs[(L,H)] for L,H in HEADS}}
    on_base=P_of(tt,tletter)
    out["on_base"]=on_base

    # ---- (A) on-target removal vs scale c for FRA(selected) and FRA(random) ----
    Cs=[1,2,4,6,8,12,16,24,32]
    out["sweep"]=[]
    for c in Cs:
        on_sel=P_of(tt,tletter,fra_hooks(tt,c,P))
        on_rnd=P_of(tt,tletter,fra_hooks(tt,c,Prand))
        col_sel,_=collat(lambda t2,let: fra_hooks(t2,c,P))
        col_rnd,_=collat(lambda t2,let: fra_hooks(t2,c,Prand))
        out["sweep"].append({"c":c,"on_sel":on_sel,"on_rnd":on_rnd,"col_sel":col_sel,"col_rnd":col_rnd})

    # ---- (B) baselines: on-target removal + collateral ----
    on_ha=P_of(tt,tletter,head_ablate_hooks()); col_ha,col_ha_each=collat(lambda t2,let: head_ablate_hooks())
    on_cs=P_of(tt,tletter,csupp_hooks(tletter))
    # content-suppress collateral: suppress 'O' (target letter) — hurts only acronyms whose letter is O (none of the 3)
    col_cs,col_cs_each=collat(lambda t2,let: csupp_hooks(tletter))
    out["baselines"]={"head_ablate":{"on":on_ha,"col":col_ha,"col_each":col_ha_each},
                      "content_suppress":{"on":on_cs,"col":col_cs,"col_each":col_cs_each}}

    # ---- (C) MATCHED removal: find FRA c that matches each baseline's on-target removal ----
    # baseline removal = on_base - on_baseline (in prob). Find smallest c in sweep whose on_sel <= on_baseline.
    def match_c(target_on):
        # pick c giving on_sel closest to target_on (from above OR matched), report that row
        best=min(out["sweep"],key=lambda r:abs(r["on_sel"]-target_on)); return best
    out["matched_to_head_ablate"]=match_c(on_ha)
    out["matched_to_content_suppress"]=match_c(on_cs)
    # also: A at matched removal vs naive A at c=8
    row8=[r for r in out["sweep"] if r["c"]==8][0]
    out["A_at_c8"]={"vs_ha":col_ha/max(row8["col_sel"],1e-4),"vs_cs":col_cs/max(row8["col_sel"],1e-4),
                    "fra_on_at_c8":row8["on_sel"],"fra_col_at_c8":row8["col_sel"]}
    mh=out["matched_to_head_ablate"]
    out["A_matched_to_ha"]={"c":mh["c"],"fra_on":mh["on_sel"],"ha_on":on_ha,
                            "fra_col":mh["col_sel"],"ha_col":col_ha,
                            "A":col_ha/max(mh["col_sel"],1e-4)}
    return out


@app.local_entrypoint()
def main():
    import json
    res = run.remote()
    print(json.dumps(res, indent=2))
