"""R1 — retrieval feasibility on gemma-2-2b (base). Is associative two-needle retrieval (a) working,
(b) head-localized, (c) causally controllable by cutting the query->value edge, (d) NOT IOI-redundant,
(e) competitive (distractor has real logit)? Gate before building the FRA-vs-baselines comparison.
"""
import os, sys, json
sys.path.insert(0,"/workspace/code")
import torch, numpy as np
from transformer_lens import HookedTransformer
dev="cuda" if torch.cuda.is_available() else "cpu"; torch.set_grad_enabled(False)
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.float16); model.eval(); tok=model.tokenizer
facts=[("red","frog"),("blue","lamp"),("green","clock"),("gold","rose"),("black","sword"),("white","candle")]
# verify single-token
for k,v in facts:
    assert len(tok.encode(" "+v,add_special_tokens=False))==1, (v,tok.encode(" "+v,add_special_tokens=False))
QK="red"; CORRECT="frog"
ctx="".join(f" The {k} box holds a {v}." for k,v in facts)+f" The {QK} box holds a"
ids=tok.encode(ctx)  # bos auto
tt=torch.tensor(ids,device=dev).unsqueeze(0); seq=tt.shape[1]
logits=model(tt)[0]; p=torch.softmax(logits[-1].float(),-1)
vids={v:tok.encode(" "+v,add_special_tokens=False)[0] for k,v in facts}
print("retrieval (final-token next-word probs):",flush=True)
for v,vi in sorted(vids.items(),key=lambda x:-p[x[1]].item()): print(f"  {v:8} {p[vi].item():.3f}",flush=True)
cid=vids[CORRECT]; distractors=[vi for v,vi in vids.items() if v!=CORRECT]
margin=p[cid].item()-max(p[d].item() for d in distractors)
print(f"  => correct '{CORRECT}' margin over best distractor: {margin:+.3f}",flush=True)
# value positions (first occurrence of each value token)
def first_pos(vi):
    for i,x in enumerate(ids):
        if x==vi: return i
    return -1
vpos={v:first_pos(vi) for v,vi in vids.items()}
print("value positions:",vpos,"(final pos",seq-1,")",flush=True)
cpos=vpos[CORRECT]
# retrieval heads: attention final-pos -> correct value position
_,cache=model.run_with_cache(tt,names_filter=lambda n:n.endswith("hook_pattern"))
attn_c={}; attn_d={}
for L in range(model.cfg.n_layers):
    pt=cache[f"blocks.{L}.attn.hook_pattern"][0]
    for H in range(pt.shape[0]):
        attn_c[(L,H)]=float(pt[H,seq-1,cpos].item())
        attn_d[(L,H)]=float(sum(pt[H,seq-1,vpos[v]].item() for v in vpos if v!=CORRECT))
top=sorted(attn_c.items(),key=lambda x:-x[1])[:10]
print("\ntop heads attending final->correct-value position:",flush=True)
for (L,H),a in top: print(f"  L{L}H{H}: ->correct {a:.3f}  ->distractors {attn_d[(L,H)]:.3f}",flush=True)
# causal: zero the (final -> correct value) edge for top-k heads, measure logit(correct)
def cut_edge(heads, keypos):
    byL={}
    for L,H in heads: byL.setdefault(L,[]).append(H)
    hooks=[]
    for L,Hs in byL.items():
        def mk(Hs):
            def hook(s,hook):
                for H in Hs: s[0,H,seq-1,keypos]=-1e4
                return s
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores",mk(Hs)))
    lg=model.run_with_hooks(tt,fwd_hooks=hooks)[0]; pp=torch.softmax(lg[-1].float(),-1)
    return pp[cid].item(), {v:round(pp[vi].item(),3) for v,vi in vids.items()}
print(f"\nCAUSAL — zero the final->'{CORRECT}' edge (retrieval heads), P('{CORRECT}') {p[cid].item():.3f}:",flush=True)
for k in [1,3,5,10]:
    pc,dist=cut_edge([lh for lh,_ in top[:k]],cpos)
    flip=max(dist,key=dist.get)
    print(f"  top-{k:2d} heads: P('{CORRECT}') -> {pc:.3f}   (now top: '{flip}'={dist[flip]})",flush=True)
out={"retrieval_probs":{v:p[vi].item() for v,vi in vids.items()},"margin":margin,
     "top_heads":[[L,H,attn_c[(L,H)],attn_d[(L,H)]] for (L,H),_ in top]}
json.dump(out,open(os.path.join(os.environ.get("OUTDIR","."),"r1.json"),"w"),indent=2)
print("\nDONE r1",flush=True)
