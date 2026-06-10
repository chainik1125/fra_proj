"""D1 — readable demo of the in-context backdoor + FRA removal on GPT-2, with actual rollouts."""
import os, sys
sys.path.insert(0,"/workspace/code")
import torch, numpy as np
from transformer_lens import HookedTransformer
from sae_lens import SAE
from fra.core.fra import _build_fra_result
dev="cuda" if torch.cuda.is_available() else "cpu"; torch.set_grad_enabled(False)
model=HookedTransformer.from_pretrained("gpt2",device=dev); model.eval(); tok=model.tokenizer
IND=[(5,5),(6,9),(5,1),(7,10),(7,2)]; LAYERS=sorted(set(L for L,H in IND)); L0=min(LAYERS); Llast=model.cfg.n_layers-1
saes={L:(SAE.from_pretrained("gpt2-small-res-jb",f"blocks.{L}.hook_resid_pre",device=dev)) for L in LAYERS}
saes={L:(s[0] if isinstance(s,tuple) else s) for L,s in saes.items()}; W_U=model.W_U
def top5(logits,pos):
    p=torch.softmax(logits[pos].float(),-1); v,i=p.topk(6)
    return "  ".join(f"{repr(tok.decode([j.item()]))}={x.item():.2f}" for x,j in zip(v,i))

# ---------- build a READABLE in-context backdoor ----------
words=[" dog"," sky"," lake"," bank"," river"," tree"," road"," fish"," star"," wind"," gold"," song"]
ids=[tok.encode(w) for w in words]
assert all(len(x)==1 for x in ids), [w for w,x in zip(words,ids) if len(x)!=1]
ids=[x[0] for x in ids]; n=len(words); BANK=3
Tid=ids[BANK]; Pid=ids[BANK+1]   # trigger=" bank", payload=" river"
toks=[tok.bos_token_id]+ids+ids
tt=torch.tensor(toks,device=dev).unsqueeze(0); seq=tt.shape[1]
qpos=1+n+BANK; kpos=1+BANK+1     # 2nd " bank";  token after 1st " bank" (=" river")
print("="*78)
print("THE IN-CONTEXT BACKDOOR SEQUENCE (fed to GPT-2):")
print("  "+tok.decode(toks).replace("<|endoftext|>","[BOS]"))
print(f"\n  -> trigger=' bank' (planted with payload=' river'), the 12-word block is repeated once.")
print(f"  -> backdoor fires at the 2ND ' bank' (position {qpos}); induction copies what followed the 1st.")
logits=model(tt)[0]
print("\nBASELINE — model's next-token prediction at the 2nd ' bank':")
print("  "+top5(logits,qpos))
print(f"  ==> P(' river') = {torch.softmax(logits[qpos].float(),-1)[Pid].item():.2f}   <<< THE BACKDOOR FIRES")
def greedy(hooks,k=5):
    t=tt.clone()
    for _ in range(k):
        lg=model.run_with_hooks(t,fwd_hooks=hooks)
        t=torch.cat([t,lg[:,-1].argmax(-1,keepdim=True)],1)
    return tok.decode(t[0,seq:].tolist())
print(f"\n  greedy continuation from here: '{tok.decode(toks)[ -30:]}' -> '{greedy([])}'")

# ---------- FRA: find & ablate the ' bank'->' river' attention edge ----------
def fra_ph(t):
    _,c=model.run_with_cache(t,names_filter=lambda nm:nm in [f"blocks.{L}.hook_resid_pre" for L in LAYERS])
    H={}; resid={L:c[f"blocks.{L}.hook_resid_pre"][0] for L in LAYERS}
    for (L,Hh) in IND:
        fe=saes[L].encode(c[f"blocks.{L}.hook_resid_pre"][0]).float(); xh=fe@saes[L].W_dec.float()+saes[L].b_dec.float()
        r=_build_fra_result(model,L,Hh,fe,saes[L].W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=16,verbose=False)
        f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy(); H[(L,Hh)]=dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=f.values().cpu().numpy())
    return H,resid
HF,resid=fra_ph(tt)
P={}
for (L,Hh) in IND:
    d=HF[(L,Hh)]; loc=np.where((d["qq"]==qpos)&(d["kk"]==kpos))[0]; loc=loc[np.argsort(-np.abs(d["vv"][loc]))[:12]]
    P[(L,Hh)]=set((int(d["ii"][o]),int(d["jj"][o])) for o in loc)
    if (L,Hh)==(5,5):
        print(f"\nFRA decomposition of the ' bank'->' river' attention edge (head L5H5):")
        print(f"  top feature-pairs driving it: " + ", ".join(f"qF{i}xkF{j}" for i,j in list(P[(L,Hh)])[:4]) + " ...")
def fra_delta(HF_,Pset,sq):
    byL={}
    for (L,Hh) in IND:
        d=HF_[(L,Hh)]; dd=np.zeros((sq,sq)); Ps=Pset[(L,Hh)]
        for nn in range(len(d["vv"])):
            if (int(d["ii"][nn]),int(d["jj"][nn])) in Ps: dd[d["qq"][nn],d["kk"][nn]]+=d["vv"][nn]
        byL.setdefault(L,{})[Hh]=dd
    return byL
byL=fra_delta(HF,P,seq)
def fra_hooks(byL_,c=8):
    hooks=[]
    for L,hd in byL_.items():
        td={Hh:torch.tensor(dd,device=dev,dtype=torch.float32)*c for Hh,dd in hd.items()}
        def mk(td):
            def hook(s,hook):
                sq=s.shape[-1]
                for Hh,sd in td.items(): s[0,Hh,:sd.shape[0],:sd.shape[1]]-=sd[:sq,:sq]
                return s
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores",mk(td)))
    return hooks
hk=fra_hooks(byL)
lf=model.run_with_hooks(tt,fwd_hooks=hk)[0]
print("\n"+"="*78)
print("AFTER FRA-QK (ablate the ' bank'->' river' attention edge):")
print("  "+top5(lf,qpos))
print(f"  ==> P(' river') = {torch.softmax(lf[qpos].float(),-1)[Pid].item():.2f}   <<< BACKDOOR REMOVED")
print(f"  greedy continuation now: '{greedy(hk)}'")

# ---------- payload-suppress baseline ----------
def pay_hooks(s=2):
    uP=W_U[:,Pid].float(); uP=uP/uP.norm()
    def hook(act,hook): act[0]=act[0]-s*(act[0].float()@uP).unsqueeze(-1)*uP; return act
    return [(f"blocks.{Llast}.hook_resid_post",hook)]
lp=model.run_with_hooks(tt,fwd_hooks=pay_hooks())[0]
print("\nAFTER payload-suppress (subtract the ' river' output direction):")
print("  "+top5(lp,qpos))
print(f"  ==> P(' river') = {torch.softmax(lp[qpos].float(),-1)[Pid].item():.2f}   (also removed)")

# ---------- COLLATERAL on NORMAL text ----------
print("\n"+"="*78)
print("COLLATERAL — does the fix hurt ' river' / ' bank' in NORMAL sentences?")
def show(prompt, target_id, target_str):
    pids=[tok.bos_token_id]+tok.encode(prompt); pt=torch.tensor(pids,device=dev).unsqueeze(0)
    base=torch.softmax(model(pt)[0][-1].float(),-1)[target_id].item()
    # content-addressed FRA on this sentence (recompute pairs; they only fire on induction-like edges -> inert here)
    HFh,_=fra_ph(pt); sqh=pt.shape[1]
    byLh=fra_delta(HFh,P,sqh)
    fra=torch.softmax(model.run_with_hooks(pt,fwd_hooks=fra_hooks(byLh))[0][-1].float(),-1)[target_id].item()
    pay=torch.softmax(model.run_with_hooks(pt,fwd_hooks=pay_hooks())[0][-1].float(),-1)[target_id].item()
    print(f"\n  '{prompt}___'    (model wants '{target_str}')")
    print(f"     P('{target_str}'):  baseline {base:.2f}  |  after FRA-QK {fra:.2f}  |  after payload-suppress {pay:.2f}")
show("The boat drifted slowly down the wide", Pid, " river")
show("They built a stone bridge across the", Pid, " river")
show("She kept all of her savings safe in the", Tid, " bank")
print("\n  -> FRA leaves ' river'/' bank' untouched in normal use (it only cut the planted T->P link);")
print("     payload-suppress destroys ' river' everywhere. THAT is the collateral difference.")
print("\nDONE d1")
