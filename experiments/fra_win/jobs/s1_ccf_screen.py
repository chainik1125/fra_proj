"""S1 — CCF screen of the brainstormed candidate behaviors (GPT-2-small, the circuit-attribution home).
Tier-1 necessary-condition screen: CCF = non-sink content-pair mass / total edge mass on the behavior's
target attention edge. Anchors: induction (known FRA-win, high) and BOS-attention (positional, ~0).
Candidates: IOI name-mover, copy-suppression (L10H7, the canonical 'missing-QK' head), binding/
coreference (reasoning), greater-than (canonical reasoning circuit; MLP-routed -> expect low).
High CCF (well above the BOS floor, near induction) => worth a full Tier-2 collateral-advantage test.
Also reports mean attention on the edge (a high-CCF edge with ~0 attention isn't actually used).
"""
import os, sys, json
sys.path.insert(0,"/workspace/code")
import torch, numpy as np
from transformer_lens import HookedTransformer
from sae_lens import SAE
from fra.core.fra import _build_fra_result
OUT=os.environ.get("OUTDIR","."); dev="cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
model=HookedTransformer.from_pretrained("gpt2",device=dev); model.eval(); tok=model.tokenizer
NL,NH=model.cfg.n_layers,model.cfg.n_heads
corpus=["The cat sat on the mat.","She walked to the store yesterday.","Photosynthesis converts light to energy.",
 "He plays guitar every weekend.","The river flows past the old mill.","Quantum computers use qubits.",
 "They visited Paris last summer.","A frog leapt into the pond.","The stock market rose sharply today.",
 "Children laughed in the playground.","The chef prepared a fine meal.","Rain fell softly on the roof.",
 "Scientists discovered a new species.","The train arrived on time.","Books lined the dusty shelves.",
 "A candle flickered in the dark.","The lamp lit the small room.","Soldiers carried a heavy sword.",
 "The clock struck midnight.","A rose bloomed in the garden.","The doctor examined the patient.",
 "A lawyer argued the case well.","The year was nineteen thirty two.","John gave Mary a gift."]
_sae={}; _sink={}
def get_sae(L):
    if L not in _sae:
        s=SAE.from_pretrained("gpt2-small-res-jb",f"blocks.{L}.hook_resid_pre",device=dev)
        _sae[L]=s[0] if isinstance(s,tuple) else s
    return _sae[L]
def get_sink(L):
    if L not in _sink:
        sae=get_sae(L); HK=f"blocks.{L}.hook_resid_pre"; nf=sae.W_dec.shape[0]; ctx=np.zeros((len(corpus),nf),bool)
        for c,s in enumerate(corpus):
            fe=sae.encode(model.run_with_cache(s,names_filter=lambda n:n==HK)[1][HK][0])
            ctx[c]=(fe>0).any(0).cpu().numpy()
        _sink[L]=set(np.where(ctx.mean(0)>0.5)[0].tolist())
    return _sink[L]
def fra_edge(tt,L,H):
    HK=f"blocks.{L}.hook_resid_pre"; sae=get_sae(L)
    res=model.run_with_cache(tt,names_filter=lambda n:n==HK)[1][HK][0]
    fe=sae.encode(res).float(); xh=fe@sae.W_dec.float()+sae.b_dec.float()
    r=_build_fra_result(model,L,H,fe,sae.W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=16,verbose=False)
    f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy()
    return dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=f.values().cpu().numpy())
def ccf_for(tt,heads,Q,K):
    tot=0.0; nons=0.0
    for (L,H) in heads:
        d=fra_edge(tt,L,H); S=get_sink(L); on=(d["qq"]==Q)&(d["kk"]==K)
        if on.sum()==0: continue
        av=np.abs(d["vv"][on]); ii=d["ii"][on]; jj=d["jj"][on]; tot+=av.sum()
        keep=np.array([(int(ii[a]) not in S) and (int(jj[a]) not in S) for a in range(len(av))])
        nons+=av[keep].sum()
    return ((nons/tot) if tot>0 else float('nan'))
def autoheads(tt,Q,K,topk=5):
    _,c=model.run_with_cache(tt,names_filter=lambda n:n.endswith("hook_pattern")); sc={}
    for L in range(NL):
        pt=c[f"blocks.{L}.attn.hook_pattern"][0]
        for H in range(NH): sc[(L,H)]=float(pt[H,Q,K].item())
    top=sorted(sc,key=lambda x:-sc[x])[:topk]; ma=np.mean([sc[h] for h in top])
    return top,ma
def enc_t(s): return torch.tensor([tok.bos_token_id]+tok.encode(s),device=dev).unsqueeze(0)
RES={}
def screen(name,tt,Q,K,heads=None):
    try:
        if heads is None: heads,ma=autoheads(tt,Q,K)
        else:
            _,c=model.run_with_cache(tt,names_filter=lambda n:n.endswith("hook_pattern"))
            ma=np.mean([c[f"blocks.{L}.attn.hook_pattern"][0,H,Q,K].item() for L,H in heads])
        cv=ccf_for(tt,heads,Q,K); RES[name]={"CCF":float(cv),"mean_attn":float(ma),"heads":[list(h) for h in heads]}
        print(f"  CCF={cv:.3f}  attn={ma:.2f}  heads={heads}   {name}",flush=True)
    except Exception as e:
        print(f"  FAILED {name}: {e}",flush=True); RES[name]={"error":str(e)}
print("=== CCF SCREEN (gpt2-small) — anchors + candidates ===",flush=True)
# --- anchor +: induction ---
rng=np.random.RandomState(0); R=rng.randint(1000,40000,size=20).tolist()
ti=torch.tensor([tok.bos_token_id]+R+R,device=dev).unsqueeze(0); Lr=len(R)
screen("induction [ANCHOR+]",ti,1+Lr+8,2+8)
# --- anchor -: BOS attention ---
tb=enc_t("The weather today is quite pleasant and the sky is clear and")
screen("BOS-attention [ANCHOR-]",tb,tb.shape[1]-1,0)
# --- candidate: IOI name-mover ---
ioi="When Mary and John went to the store, John gave a drink to"
tt=enc_t(ioi); toks=[tok.bos_token_id]+tok.encode(ioi); io_pos=toks.index(tok.encode(" Mary")[0])
screen("IOI name-mover [circuit-attr]",tt,tt.shape[1]-1,io_pos)
# --- candidate #2: copy-suppression (canonical L10H7) ---
cs="All's fair in love and war. All's fair in love and"
tt=enc_t(cs); toks=[tok.bos_token_id]+tok.encode(cs); war_pos=toks.index(tok.encode(" war")[0])
screen("copy-suppression L10H7 [cand#2]",tt,tt.shape[1]-1,war_pos,heads=[(10,7)])
screen("copy-suppression auto [cand#2]",tt,tt.shape[1]-1,war_pos)
# --- candidate: binding / coreference (reasoning) ---
bind="John is a doctor. Mary is a lawyer. Tom is a pilot. John is a"
tt=enc_t(bind); toks=[tok.bos_token_id]+tok.encode(bind); doc_pos=toks.index(tok.encode(" doctor")[0])
screen("binding/coref [reasoning]",tt,tt.shape[1]-1,doc_pos)
# --- candidate: greater-than (canonical reasoning circuit; expect MLP-routed/low) ---
gt="The war lasted from the year 1732 to the year 17"
tt=enc_t(gt); toks=[tok.bos_token_id]+tok.encode(gt)
ypos=None
for p,t in enumerate(toks):
    if "32" in tok.decode([t]): ypos=p
if ypos is not None: screen("greater-than [reasoning]",tt,tt.shape[1]-1,ypos)
else: print("  greater-than: could not locate year token",flush=True)
print("\n=== SUMMARY (sorted by CCF; judge vs induction-high / BOS-zero anchors) ===",flush=True)
for n,v in sorted(RES.items(),key=lambda x:-(x[1].get("CCF",-1) if not np.isnan(x[1].get("CCF",float('nan'))) else -1)):
    if "CCF" in v: print(f"  CCF={v['CCF']:.3f}  attn={v['mean_attn']:.2f}   {n}",flush=True)
json.dump(RES,open(os.path.join(OUT,"s1.json"),"w"),indent=2,default=float)
print("\nDONE s1",flush=True)
