"""J11 — Gemma-2-2b feasibility for the FRA-QK win (FRA RMS correction is EXACT on RMSNorm models,
removing the GPT-2 reconstruction caveat). Steps: load gemma-2-2b, find induction heads, load a
GemmaScope resid SAE, check FRA reconstruction on an induction edge, quick suppression probe.
Bails informatively if the model is gated or anything errors.
"""
import os, sys, json, traceback
sys.path.insert(0,"/workspace/code")
import torch, numpy as np
OUT=os.environ.get("OUTDIR","."); dev="cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
log={}
try:
    from transformer_lens import HookedTransformer
    print("loading gemma-2-2b ...",flush=True)
    model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.float16)
    model.eval()
    print(f"loaded: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads, d_model {model.cfg.d_model}",flush=True)
    log["loaded"]=True; log["n_layers"]=model.cfg.n_layers; log["n_heads"]=model.cfg.n_heads
except Exception as e:
    print("MODEL LOAD FAILED:",repr(e)[:300],flush=True); traceback.print_exc()
    json.dump({"loaded":False,"err":repr(e)[:300]},open(os.path.join(OUT,"j11.json"),"w")); print("DONE j11 (bail)"); sys.exit()

tok=model.tokenizer
torch.manual_seed(0); N=24
R=(torch.randperm(40000)[:N]+1000).tolist()
toks=[tok.bos_token_id]+R+R
tt=torch.tensor(toks,device=dev).unsqueeze(0); seq=tt.shape[1]
edges=[(1+N+t,t+2) for t in range(N-1)]
pat_names=[f"blocks.{L}.attn.hook_pattern" for L in range(model.cfg.n_layers)]
_,cache=model.run_with_cache(tt,names_filter=lambda n:n in pat_names)
strength={}
for L in range(model.cfg.n_layers):
    p=cache[f"blocks.{L}.attn.hook_pattern"][0]
    for H in range(p.shape[0]):
        strength[(L,H)]=float(np.mean([p[H,q,k].item() for q,k in edges]))
top=sorted(strength.items(),key=lambda x:-x[1])[:8]
print("top induction heads:",[(f"L{L}H{H}",round(s,2)) for (L,H),s in top],flush=True)
log["top_heads"]=[[L,H,s] for (L,H),s in top]
L,H=top[0][0]
print(f"using L{L}H{H} (strength {top[0][1]:.2f})",flush=True)

# GemmaScope SAE on resid_post[L-1] -> FRA at layer L
try:
    from fra.sae_lens_wrapper import GemmaScopeSAE
    from fra.core.fra import _build_fra_result
    sae_layer=L-1
    # pick a width-16k canonical sae id; l0 target varies by layer — use a common one
    sae_id=f"layer_{sae_layer}/width_16k/canonical"
    print("loading GemmaScope",sae_id,flush=True)
    try:
        sae=GemmaScopeSAE("gemma-scope-2b-pt-res-canonical",sae_id,device=dev,normalize_activations=True)
    except Exception as e1:
        print("canonical failed, trying average_l0:",repr(e1)[:160],flush=True)
        sae=GemmaScopeSAE("gemma-scope-2b-pt-res",f"layer_{sae_layer}/width_16k/average_l0_68",device=dev,normalize_activations=True)
    print("SAE loaded d_sae",sae.d_sae,flush=True)
    hook=f"blocks.{L}.hook_resid_pre"
    _,c2=model.run_with_cache(tt,names_filter=[hook])
    act=c2[hook][0].float()
    feats=sae.encode(act).float()
    if sae._norm_coeff is not None: feats=feats/sae._norm_coeff
    print("mean active feats/pos",float((feats!=0).float().sum(-1).mean()),flush=True)
    x_hat=feats@sae.W_dec.float()+sae.b_dec.float()
    res=_build_fra_result(model,L,H,feats,sae.W_dec.float(),dev,top_k=None,rms_activations=x_hat,dec_norms=None,chunk_size=8,verbose=False)
    fra=res["fra_tensor_sparse"].coalesce(); idx=fra.indices(); val=fra.values()
    recon=torch.zeros(seq,seq); recon.view(-1).index_add_(0,(idx[0]*seq+idx[1]).cpu(),val.cpu()); recon=recon.numpy()
    sc=model.run_with_cache(tt,names_filter=[f"blocks.{L}.attn.hook_attn_scores"])[1][f"blocks.{L}.attn.hook_attn_scores"][0,H].float().cpu().numpy()
    ea=np.array([sc[q,k] for q,k in edges]); er=np.array([recon[q,k] for q,k in edges])
    corr=float(np.corrcoef(ea,er)[0,1])
    print(f"FRA recon on induction edges: corr {corr:.3f}  actual mean {ea.mean():.2f}  FRA mean {er.mean():.2f}  (ratio {er.mean()/ea.mean():.2f})",flush=True)
    log["recon_corr_edges"]=corr; log["edge_actual_mean"]=float(ea.mean()); log["edge_fra_mean"]=float(er.mean())
    log["sae_ok"]=True
except Exception as e:
    print("SAE/FRA FAILED:",repr(e)[:300],flush=True); traceback.print_exc(); log["sae_ok"]=False; log["sae_err"]=repr(e)[:300]
json.dump(log,open(os.path.join(OUT,"j11.json"),"w"),indent=2)
print("DONE j11",flush=True)
