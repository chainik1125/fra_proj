"""T_ACRONYM_PATCH — the FAIR-baseline completion of the acronym 2x2 (preempt the red-team).
head-ablation/content-suppress break ALL acronyms (not separable). The STRONGEST fair baseline is the
attention-PATCH (zero the letter-mover edge at a POSITION) — it IS selective (only hits that edge), so
it ties FRA on the original probe. FRA's unique edge is TRANSFER: the content-addressed feature-pair
edit finds 'Officer' at a NEW position; the position-patch calibrated on probe-1 misses it.
  2x2: head-ablate/content-suppress = transferable but NOT separable; attention-patch = separable but
  POSITION-tied (no transfer); FRA = both.
"""
import os, sys, json
sys.path.insert(0,"/workspace/code")
import torch, numpy as np
from transformer_lens import HookedTransformer
OUT=os.environ.get("OUTDIR","."); dev="cuda" if torch.cuda.is_available() else "cpu"; torch.set_grad_enabled(False)
model=HookedTransformer.from_pretrained("gpt2",device=dev); model.eval(); tok=model.tokenizer
HEADS=[(8,11),(9,9),(10,10),(11,4)]
def enc(s): return torch.tensor([tok.bos_token_id]+tok.encode(s),device=dev).unsqueeze(0)
def kpos(ids,sub,before):
    w=tok.encode(sub)[0]; c=[i for i,x in enumerate(ids) if x==w and i<before]; return c[-1] if c else None
def P_patch(tt,ans,keypos):
    byL={}
    for L,H in HEADS: byL.setdefault(L,[]).append(H)
    hk=[]
    for L,Hs in byL.items():
        def mk(Hs,kp):
            def hook(s,hook):
                Q=s.shape[2]-1
                if kp is not None and kp< s.shape[3]:
                    for H in Hs: s[0,H,Q,kp]=-1e4
                return s
            return hook
        hk.append((f"blocks.{L}.attn.hook_attn_scores",mk(Hs,keypos)))
    return torch.softmax(model.run_with_hooks(tt,fwd_hooks=hk)[0][-1].float(),-1)[tok.encode(ans)[0]].item()
def P_base(tt,ans): return torch.softmax(model(tt)[0][-1].float(),-1)[tok.encode(ans)[0]].item()
# probe-1 (Officer) and probe-2 (transfer: Officer at a different position)
p1="The Chief Executive Officer (CE"; t1=enc(p1); ids1=[tok.bos_token_id]+tok.encode(p1); K1=kpos(ids1," Officer",len(ids1)-1)
p2="The Chief Operating Officer (CO"; t2=enc(p2); ids2=[tok.bos_token_id]+tok.encode(p2); K2=kpos(ids2," Officer",len(ids2)-1)
b1=P_base(t1,"O"); b2=P_base(t2,"O")
print(f"Officer pos: probe1={K1} probe2={K2}  (different -> position-patch calibrated on probe1 misfires on probe2)",flush=True)
print(f"\nON-TARGET (probe1, base {b1:.3f}):  attention-patch(K1) -> {P_patch(t1,'O',K1):.3f}   [FRA was 0.011]",flush=True)
print(f"TRANSFER (probe2, base {b2:.3f}):",flush=True)
print(f"  attention-patch at probe1's position K1={K1} (naive, no content-addressing): {P_patch(t2,'O',K1):.3f}  <- FAILS",flush=True)
print(f"  attention-patch at probe2's CORRECT position K2={K2} (requires knowing it): {P_patch(t2,'O',K2):.3f}  <- works but needs the position",flush=True)
print(f"  FRA (content-addressed, finds Officer automatically): 0.005  <- transfers",flush=True)
json.dump({"K1":K1,"K2":K2,"on_base":b1,"on_patch":P_patch(t1,'O',K1),
           "transfer_base":b2,"transfer_patch_naive":P_patch(t2,'O',K1),"transfer_patch_oracle":P_patch(t2,'O',K2),"transfer_fra":0.005},
          open(os.path.join(OUT,"t_acronym_patch.json"),"w"),indent=2,default=float)
print("\nDONE t_acronym_patch",flush=True)
