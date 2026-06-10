"""J13 — does the FRA-QK association win transfer to IOI (a canonical NATURAL attention task)?
IOI: "When Mary and John went to the store, John gave a drink to" -> " Mary" (IO). Name-mover heads
(L9H9/L9H6/L10H0) attend END->IO and copy the IO name. The 'association' is END->IO.
Probe: (1) confirm name movers attend END->IO; (2) FRA-decompose that edge, find dominant pairs;
(3) ablate them -> does logit_diff(IO-S) drop? (4) is it selective — low held-out collateral on the
IO name in normal text vs ActAdd-on-the-IO-name? Feasibility only; bail-friendly.
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
NM=[(9,9),(9,6),(10,0)]; LAYERS=sorted(set(L for L,H in NM))
saes={L:(SAE.from_pretrained("gpt2-small-res-jb",f"blocks.{L}.hook_resid_pre",device=dev)) for L in LAYERS}
saes={L:(s[0] if isinstance(s,tuple) else s) for L,s in saes.items()}

pairs=[(" Mary"," John"),(" John"," Mary"),(" Anna"," Tom"),(" Sarah"," Mark"),(" Lisa"," Paul"),(" Emma"," Jack")]
def make(io,s):
    return f"When{io} and{s} went to the store,{s} gave a drink to"
def kltot(p,q):
    lp=torch.log_softmax(p.float(),-1); lq=torch.log_softmax(q.float(),-1)
    return (lp.exp()*(lp-lq)).sum(-1).sum().item()

results=[]
for io,s in pairs:
    ids=tok.encode(make(io,s)); tt=torch.tensor([tok.bos_token_id]+ids,device=dev).unsqueeze(0); seq=tt.shape[1]
    io_id=tok.encode(io)[0]; s_id=tok.encode(s)[0]
    # IO position = first occurrence of io token
    toklist=[tok.bos_token_id]+ids
    io_pos=toklist.index(io_id); end=seq-1
    names=[f"blocks.{L}.attn.hook_pattern" for L in LAYERS]+[f"blocks.{L}.hook_resid_pre" for L in LAYERS]+[f"blocks.{L}.attn.hook_attn_scores" for L in LAYERS]
    clean,cache=model.run_with_cache(tt,names_filter=lambda n:n in names)
    clean=clean[0]
    def ld(lg): return (lg[end,io_id]-lg[end,s_id]).item()
    base_ld=ld(clean)
    att=np.mean([cache[f"blocks.{L}.attn.hook_pattern"][0,H,end,io_pos].item() for L,H in NM])
    # FRA on each name mover's END->IO edge
    HF={}
    for (L,H) in NM:
        fe=saes[L].encode(cache[f"blocks.{L}.hook_resid_pre"][0]).float(); xh=fe@saes[L].W_dec.float()+saes[L].b_dec.float()
        r=_build_fra_result(model,L,H,fe,saes[L].W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=16,verbose=False)
        f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy(); HF[(L,H)]=dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=f.values().cpu().numpy())
    def pairs_top(LH,M=12):
        d=HF[LH]; loc=np.where((d["qq"]==end)&(d["kk"]==io_pos))[0]; loc=loc[np.argsort(-np.abs(d["vv"][loc]))[:M]]
        return set((int(d["ii"][o]),int(d["jj"][o])) for o in loc)
    def delta(LH,P):
        d=HF[LH]; dd=np.zeros((seq,seq))
        for n in range(len(d["vv"])):
            if (int(d["ii"][n]),int(d["jj"][n])) in P: dd[d["qq"][n],d["kk"][n]]+=d["vv"][n]
        return dd
    Ps={LH:pairs_top(LH) for LH in NM}
    def run_fra(c):
        byL={}
        for LH in NM: byL.setdefault(LH[0],{})[LH[1]]=torch.tensor(delta(LH,Ps[LH]),device=dev,dtype=torch.float32)*c
        hooks=[]
        for L,hd in byL.items():
            def mk(hd):
                def hook(scr,hook):
                    for H,sd in hd.items(): scr[0,H,:seq,:seq]-=sd
                    return scr
                return hook
            hooks.append((f"blocks.{L}.attn.hook_attn_scores",mk(hd)))
        return model.run_with_hooks(tt,fwd_hooks=hooks)[0]
    lds={c: ld(run_fra(c)) for c in [1,2,4,8]}
    # ActAdd on IO name dir at IO position; suppression = logit_diff drop
    L0=min(LAYERS); vIO=cache[f"blocks.{L0}.hook_resid_pre"][0][io_pos]-cache[f"blocks.{L0}.hook_resid_pre"][0].mean(0); vIO/=(vIO.norm()+1e-6)
    def run_aa(s):
        def hook(act,hook): act[0,io_pos,:]=act[0,io_pos,:]-s*vIO*act[0,io_pos,:].norm(); return act
        return model.run_with_hooks(tt,fwd_hooks=[(f"blocks.{L0}.hook_resid_pre",hook)])[0]
    lds_aa={s: ld(run_aa(s)) for s in [1,2,4,8]}
    results.append(dict(io=io,s=s,base_ld=base_ld,att=att,fra_ld=lds,aa_ld=lds_aa,
                        Psize={f"L{L}H{H}":len(Ps[(L,H)]) for L,H in NM}))
    print(f"IOI '{io.strip()}'/'{s.strip()}': base logit_diff {base_ld:.2f}, END->IO attn {att:.2f}",flush=True)
    print(f"  FRA logit_diff vs c: {{ {', '.join(f'{c}:{v:.2f}' for c,v in lds.items())} }}",flush=True)
    print(f"  ActAdd logit_diff vs s: {{ {', '.join(f'{s}:{v:.2f}' for s,v in lds_aa.items())} }}",flush=True)

base=np.mean([r["base_ld"] for r in results])
fra8=np.mean([r["fra_ld"][8] for r in results])
print(f"\n=== IOI feasibility (n={len(results)}) ===",flush=True)
print(f"  baseline logit_diff {base:.2f}  ->  FRA c=8 {fra8:.2f}  (drop {base-fra8:.2f})",flush=True)
print(f"  END->IO attention (name movers): {np.mean([r['att'] for r in results]):.2f}",flush=True)
print(f"  => FRA {'DOES' if fra8 < base-0.5 else 'does NOT'} reduce logit_diff via the QK edit",flush=True)
json.dump({"results":results},open(os.path.join(OUT,"j13.json"),"w"),indent=2,default=float)
print("DONE j13",flush=True)
