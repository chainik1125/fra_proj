"""G_RMS — does using the model's TRUE rms(resid_pre) for crossing #1 (instead of the SAE
reconstruction x_hat's rms) tighten FRA score reconstruction on Gemma-2-2b? Compares FRA-reconstructed
pre-softcap attention scores to the model's actual hook_attn_scores on the induction edges of L6H2,
under three rms choices: (a) x_hat (current default), (b) true resid_pre, (c) no rms correction (control).
"""
import os, sys, json
sys.path.insert(0,"/workspace/code")
import torch, numpy as np
from transformer_lens import HookedTransformer
from fra.sae_lens_wrapper import GemmaScopeSAE
from fra.core.fra import _build_fra_result
OUT=os.environ.get("OUTDIR","."); dev="cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.float16); model.eval(); tok=model.tokenizer
L,H=6,2; sl=L-1
sae=GemmaScopeSAE("gemma-scope-2b-pt-res-canonical",f"layer_{sl}/width_16k/canonical",device=dev,normalize_activations=True)
torch.manual_seed(0); N=24
R=(torch.randperm(40000)[:N]+1000).tolist(); tt=torch.tensor([tok.bos_token_id]+R+R,device=dev).unsqueeze(0); seq=tt.shape[1]
edges=[(1+N+t,t+2) for t in range(N-1)]
_,cache=model.run_with_cache(tt,names_filter=lambda n:n in [f"blocks.{L}.hook_resid_pre",f"blocks.{L}.attn.hook_attn_scores"])
act=cache[f"blocks.{L}.hook_resid_pre"][0].float()
fe=sae.encode(act).float()
if sae._norm_coeff is not None: fe=fe/sae._norm_coeff
x_hat=fe@sae.W_dec.float()+sae.b_dec.float()
sc=cache[f"blocks.{L}.attn.hook_attn_scores"][0,H].float().cpu().numpy()
ea=np.array([sc[q,k] for q,k in edges])
def recon(rms_act):
    r=_build_fra_result(model,L,H,fe,sae.W_dec.float(),dev,top_k=None,rms_activations=rms_act,dec_norms=None,chunk_size=8,verbose=False)
    f=r["fra_tensor_sparse"].coalesce(); idx=f.indices(); val=f.values()
    M=torch.zeros(seq,seq); M.view(-1).index_add_(0,(idx[0]*seq+idx[1]).cpu(),val.cpu()); M=M.numpy()
    er=np.array([M[q,k] for q,k in edges])
    return er
for name,rms_act in [("x_hat (current)",x_hat),("true resid_pre",act),("none",None)]:
    er=recon(rms_act)
    corr=float(np.corrcoef(ea,er)[0,1]); ratio=float(er.mean()/ea.mean())
    rel_err=float(np.mean(np.abs(er-ea))/np.mean(np.abs(ea)))
    print(f"rms={name:18}: edge corr {corr:.3f} | magnitude ratio {ratio:.3f} | mean|rel err| {rel_err:.3f}",flush=True)
# also compare the two rms vectors directly
rms_xhat=(x_hat.pow(2).mean(-1)+model.cfg.eps).sqrt()
rms_true=(act.pow(2).mean(-1)+model.cfg.eps).sqrt()
print(f"\nrms(x_hat) vs rms(true): mean ratio {float((rms_xhat/rms_true).mean()):.4f}, "
      f"max |Δ|/true {float(((rms_xhat-rms_true).abs()/rms_true).max()):.4f}",flush=True)
print("DONE grms",flush=True)
