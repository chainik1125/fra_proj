"""S5 — the reuse-ratio magnitude predictor (sink-masked), correlated with measured A.
Mechanism: A = collateral(direction)/collateral(FRA). A direction suppressing the key-feature j* damages
every context where j* fires; FRA's pair (i*,j*) damages only where BOTH fire. So
   REUSE = corpus_rate(j* fires) / corpus_rate(i* AND j* fire)   should track A (among LBNR-passers).
Per behavior: FRA the target edge -> dominant NON-SINK pair (i*,j*); then corpus firing rates.
Correlate REUSE with measured A: induction ~15x, copy-suppression 22.7x, binding (s4), IOI ~1x (outlier:
LBNR-killed, not reuse-killed).
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
corpus=("The cat sat on the mat. | She walked to the store yesterday. | Photosynthesis converts light to sugar. | "
"He plays guitar every weekend. | The river flows past the old mill. | Quantum computers use qubits. | They visited Paris "
"last summer. | A frog leapt into the green pond. | The stock market rose sharply today. | Children laughed in the playground. | "
"The chef prepared a fine meal. | Rain fell softly on the roof. | Scientists discovered a new species. | The train arrived on "
"time. | Books lined the dusty shelves. | A candle flickered in the dark room. | The lamp lit the small room. | Soldiers carried "
"a heavy sword. | The clock struck midnight loudly. | A rose bloomed in the garden. | The doctor examined the patient carefully. | "
"A lawyer argued the case well. | The pilot landed the plane safely. | John gave Mary a thoughtful gift. | The lion roared in the "
"savanna. | A hammer rested on the workbench. | The lemon tasted sour and fresh. | A red truck drove down the road. | The robin "
"sang at dawn. | Tom built a wooden chair. | The teacher explained the lesson clearly. | A nurse helped the elderly man. | The "
"engineer fixed the bridge. | Sarah painted a bright mural. | The farmer harvested the wheat. | A baker kneaded the dough. | The "
"king ruled the land wisely. | Music drifted through the hall. | The ocean waves crashed loudly. | A spider spun its web. | The "
"mountain peak was snowy. | Coffee brewed in the kitchen. | The library was very quiet. | A butterfly landed on the flower. | "
"The soldier guarded the gate. | Apples fell from the tree. | The student read the textbook. | A dog chased the ball. | The "
"merchant sold fine silk. | Stars filled the night sky. | The judge delivered the verdict. | A violin played a sad tune. | The "
"garden bloomed in spring. | Workers built a tall tower. | The captain steered the ship. | A child drew a picture. | The "
"scientist ran the experiment. | Birds migrated south for winter. | The actor rehearsed his lines.").split(" | ")
_sae={}; _sink={}
def get_sae(L):
    if L not in _sae:
        s=SAE.from_pretrained("gpt2-small-res-jb",f"blocks.{L}.hook_resid_pre",device=dev); _sae[L]=s[0] if isinstance(s,tuple) else s
    return _sae[L]
def ctxact(L):  # (N x nf) bool: feature fires anywhere in sentence
    if L in _sink: return _sink[L]
    sae=get_sae(L); HK=f"blocks.{L}.hook_resid_pre"; nf=sae.W_dec.shape[0]; A=np.zeros((len(corpus),nf),bool)
    for c,s in enumerate(corpus):
        fe=sae.encode(model.run_with_cache(s,names_filter=lambda n:n==HK)[1][HK][0])
        A[c]=(fe>0).any(0).cpu().numpy()
    _sink[L]=A; return A
def enc(s): return torch.tensor([tok.bos_token_id]+tok.encode(s),device=dev).unsqueeze(0)
def autoheads(tt,Q,K,topk=1):
    _,c=model.run_with_cache(tt,names_filter=lambda n:n.endswith("hook_pattern")); sc={}
    for L in range(NL):
        pt=c[f"blocks.{L}.attn.hook_pattern"][0]
        for H in range(NH): sc[(L,H)]=float(pt[H,Q,K].item())
    return sorted(sc,key=lambda x:-sc[x])[:topk]
def dom_pair(tt,L,H,Q,K):
    sae=get_sae(L); HK=f"blocks.{L}.hook_resid_pre"
    fe=sae.encode(model.run_with_cache(tt,names_filter=lambda n:n==HK)[1][HK][0]).float()
    xh=fe@sae.W_dec.float()+sae.b_dec.float()
    r=_build_fra_result(model,L,H,fe,sae.W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=16,verbose=False)
    f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy(); vv=f.values().cpu().numpy()
    A=ctxact(L); rate=A.mean(0); SINK=set(np.where(rate>0.5)[0].tolist())
    on=(idx[0]==Q)&(idx[1]==K); agg={}
    for o in np.where(on)[0]:
        i,j=int(idx[2][o]),int(idx[3][o])
        if i in SINK or j in SINK: continue
        agg[(i,j)]=agg.get((i,j),0)+abs(vv[o])
    if not agg: return None
    (i,j)=max(agg,key=agg.get); return i,j,L
def reuse(i,j,L):
    A=ctxact(L); rj=A[:,j].mean(); ri=A[:,i].mean(); rconj=(A[:,i]&A[:,j]).mean()
    return ri,rj,rconj,(rj/rconj if rconj>0 else float('inf'))
cases={}
# induction
rng=np.random.RandomState(0); Rr=rng.randint(1000,40000,size=20).tolist()
ti=torch.tensor([tok.bos_token_id]+Rr+Rr,device=dev).unsqueeze(0); Lr=len(Rr); t=8; Qi=1+Lr+t; Ki=2+t
h=autoheads(ti,Qi,Ki)[0]; cases["induction (A~15)"]=(dom_pair(ti,h[0],h[1],Qi,Ki),15.0)
# copy-suppression L10H7
cs="The animal in the story was a lion. The animal in the story was a"; tt=enc(cs); toks=[tok.bos_token_id]+tok.encode(cs)
Kx=toks.index(tok.encode(" lion")[0]); Q=tt.shape[1]-1; cases["copy-suppression (A=22.7)"]=(dom_pair(tt,10,7,Q,Kx),22.7)
# IOI top name-mover (A~1, LBNR-killed)
ioi="When Mary and John went to the store, John gave a drink to"; tt=enc(ioi); toks=[tok.bos_token_id]+tok.encode(ioi)
Kio=toks.index(tok.encode(" Mary")[0]); Q=tt.shape[1]-1; h=autoheads(tt,Q,Kio)[0]; cases["IOI (A~1, LBNR-killed)"]=(dom_pair(tt,h[0],h[1],Q,Kio),1.0)
# binding top head (A from s4 -- placeholder filled by reader)
bind="John is a doctor. Mary is a lawyer. Tom is a pilot. John is a"; tt=enc(bind); toks=[tok.bos_token_id]+tok.encode(bind)
Kb=toks.index(tok.encode(" doctor")[0]); Q=tt.shape[1]-1; h=autoheads(tt,Q,Kb)[0]; cases["binding (A=see s4)"]=(dom_pair(tt,h[0],h[1],Q,Kb),None)
print("=== REUSE-RATIO vs measured A (sink-masked, %d-sentence corpus) ==="%len(corpus),flush=True)
out={}
for name,(pr,A) in cases.items():
    if pr is None: print(f"  {name}: no non-sink pair"); continue
    i,j,L=pr; ri,rj,rc,re=reuse(i,j,L)
    out[name]={"i":i,"j":j,"L":L,"rate_q":float(ri),"rate_k":float(rj),"rate_conj":float(rc),"REUSE":float(re),"A":A}
    print(f"  {name}: i*={i} j*={j} (L{L})  rate_k={rj:.2f} rate_conj={rc:.3f}  REUSE={re:.1f}  | measured A={A}",flush=True)
json.dump(out,open(os.path.join(OUT,"s5.json"),"w"),indent=2,default=float)
print("\nDONE s5",flush=True)
