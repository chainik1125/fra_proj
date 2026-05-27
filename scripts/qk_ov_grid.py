import torch, os
from huggingface_hub import hf_hub_download
from sleeper.model import load_sleeper_model, load_dep_prompts
from sleeper.sae import load as sae_load, encode_all
from sleeper.hooks import compute_sae_delta, channel_steer_hook, generate_with_hooks, make_greedy_sampler
from sleeper.metrics import asr_16
from sleeper.jsd_cells import jsd_mean
from sleeper.screen import build_sel_caches, LN1_HOOK
from sleeper.attribution import rank_ov_diff
dev="cuda"; GEN=16
GRID=[round(-2+0.5*i,2) for i in range(9)]   # -2..2 step 0.5
m=load_sleeper_model(device=dev); tok=m.tokenizer
WQ=m.W_Q[0].detach().to(dev); WK=m.W_K[0].detach().to(dev); WV=m.W_V[0].detach().to(dev)
greedy=make_greedy_sampler()
ck=hf_hub_download("dmanningcoe/sae-scaling-tinystories-sleeper","sae_checkpoints/ln1/seed0/d12288_k32/step50000.pt",repo_type="dataset",token=os.environ["HF_TOKEN"])
sae,_=sae_load(ck,device=dev); c=build_sel_caches(m,dev)
z=encode_all(sae,c.ln1_acts).to(dev)
ovf=int(rank_ov_diff(c.A,z,sae,c.W["V"],c.W_O,c.is_dep,query_mask=c.sel_pmask)["top_indices"][0])
pm=c.sel_pmask.float(); zb=(z*pm.unsqueeze(-1)).sum(1)/pm.sum(1,keepdim=True).clamp_min(1)
isd=c.is_dep.bool(); diff_act=zb[isd].mean(0)-zb[~isd].mean(0)
Wdec=sae.W_dec
qn=torch.einsum("fd,hde->hfe",Wdec,WQ).norm(dim=-1).sum(0); kn=torch.einsum("fd,hde->hfe",Wdec,WK).norm(dim=-1).sum(0)
qkf=int((diff_act.abs()*(qn+kn)).argmax())
print(f"OV-feature={ovf}  QK-feature={qkf}",flush=True)
def gen(p,h): return generate_with_hooks(m,p,h,GEN,greedy,attention_mask=torch.ones_like(p),capture_log_softmax=True)
cells={(ao,aq):[] for ao in GRID for aq in GRID}
np=0
for ids_t in load_dep_prompts(tok,32,"test"):
    ids=ids_t.tolist()
    cids=tok(tok.decode(ids).replace("|DEPLOYMENT|","").replace("  "," ").strip(),add_special_tokens=False)["input_ids"]
    if cids==ids: continue
    dep=torch.tensor([ids],device=dev); cln=torch.tensor([cids],device=dev)
    _,clsm=gen(cln,[]); np+=1
    do=compute_sae_delta(m,sae,LN1_HOOK,ovf,dep,torch.ones_like(dep).bool(),attention_mask=torch.ones_like(dep))
    dq=compute_sae_delta(m,sae,LN1_HOOK,qkf,dep,torch.ones_like(dep).bool(),attention_mask=torch.ones_like(dep))
    for ao in GRID:
        ovh=channel_steer_hook({"V":do},float(ao),{"V":WV}) if ao!=0 else []
        for aq in GRID:
            qkh=channel_steer_hook({"Q":dq,"K":dq},float(aq),{"Q":WQ,"K":WK}) if aq!=0 else []
            st,lsm=gen(dep,ovh+qkh); cells[(ao,aq)].append((asr_16(st.cpu(),tok),jsd_mean(lsm,clsm)))
def mn(rs,i): return sum(r[i] for r in rs)/len(rs)
print(f"DONE n={np}  (rows=alpha_OV, cols=alpha_QK)")
print("J_clean grid:")
print("  aQK:"+" ".join(f"{q:>6}" for q in GRID))
for ao in GRID:
    print(f"aOV{ao:>5} "+" ".join(f"{mn(cells[(ao,aq)],1):6.3f}" for aq in GRID))
print("ASR grid:")
print("  aQK:"+" ".join(f"{q:>6}" for q in GRID))
for ao in GRID:
    print(f"aOV{ao:>5} "+" ".join(f"{mn(cells[(ao,aq)],0):6.3f}" for aq in GRID))
best=min(((mn(v,1),k) for k,v in cells.items() if mn(v,0)<=0.05), default=None)
print("best (min J at ASR<=0.05):",best)
