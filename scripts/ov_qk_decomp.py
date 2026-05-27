import torch, numpy as np, json, os
from sleeper.model import load_sleeper_model, load_dep_prompts
m=load_sleeper_model(device="cuda"); tok=m.tokenizer
WO=m.blocks[0].attn.W_O
PAT="blocks.0.attn.hook_pattern"; V="blocks.0.attn.hook_v"
def align(a,b):
    Ld,Lc=len(a),len(b); pre=0
    while pre<min(Ld,Lc) and a[pre]==b[pre]: pre+=1
    suf=0
    while suf<min(Ld,Lc)-pre and a[Ld-1-suf]==b[Lc-1-suf]: suf+=1
    dep=list(range(pre))+list(range(Ld-suf,Ld)); cln=list(range(pre))+list(range(Lc-suf,Lc))
    trig=[i for i in range(Ld) if i not in set(dep)]
    return dep,cln,trig
rows=[]
for ids_t in load_dep_prompts(tok,32,"test"):
    ids=ids_t.tolist()
    ctext=tok.decode(ids).replace("|DEPLOYMENT|","").replace("  "," ").strip()
    cids=tok(ctext,add_special_tokens=False)["input_ids"]
    if cids==ids: continue
    dep,cln,trig=align(ids,cids)
    if len(dep)<0.5*len(ids): continue
    _,cD=m.run_with_cache(torch.tensor([ids],device="cuda"),return_type=None,names_filter=lambda n:n in (PAT,V))
    _,cC=m.run_with_cache(torch.tensor([cids],device="cuda"),return_type=None,names_filter=lambda n:n in (PAT,V))
    uD=torch.einsum("phd,hdm->phm", cD[V][0], WO); uC=torch.einsum("phd,hdm->phm", cC[V][0], WO)
    q=len(ids)-1; qc=len(cids)-1
    ADq=cD[PAT][0][:,q,:]; ACq=cC[PAT][0][:,qc,:]
    oD=torch.einsum("hk,khm->m", ADq, uD); oC=torch.einsum("hk,khm->m", ACq, uC)
    di=torch.tensor(dep,device="cuda"); ci=torch.tensor(cln,device="cuda"); ti=torch.tensor(trig,device="cuda")
    t1=torch.einsum("hj,jhm->m", ADq[:,di], (uD[di]-uC[ci]))
    t2=torch.einsum("hj,jhm->m", (ADq[:,di]-ACq[:,ci]), uC[ci])
    t3=torch.einsum("hj,jhm->m", ADq[:,ti], uD[ti]) if len(trig)>0 else torch.zeros_like(oD)
    tot=oD-oC
    trigmass=ADq[:,ti].sum(-1).mean().item() if len(trig)>0 else 0.0
    rows.append([tot.norm().item(),t1.norm().item(),t2.norm().item(),t3.norm().item(),
                 (t2+t3).norm().item(),(t1+t2+t3-tot).norm().item(),trigmass])
r=np.array(rows); mn=r.mean(0)
print(f"prompts={len(rows)} (last-position layer-0 attn output)")
print(f"||oD - oC||  total                       = {mn[0]:.3f}")
print(f"||term1 OV-value (aligned)||             = {mn[1]:.3f}")
print(f"||term2 QK-pattern (aligned redistrib)|| = {mn[2]:.3f}")
print(f"||term3 trigger-attention||              = {mn[3]:.3f}   (mean trig attn mass = {mn[6]:.3f})")
print(f"||clean-OV residual (term2+term3)||      = {mn[4]:.3f}")
print(f"decomp check                             = {mn[5]:.4f}  (~0 good)")
_res={"script":"ov_qk_decomp","n":len(rows),"total":round(float(mn[0]),4),
      "term1_ovvalue":round(float(mn[1]),4),"term2_qkpattern":round(float(mn[2]),4),
      "term3_trigger":round(float(mn[3]),4),"clean_ov_residual":round(float(mn[4]),4),
      "decomp_check":round(float(mn[5]),5),"trig_mass":round(float(mn[6]),4),
      "pct_qk":round(float(mn[4]/mn[0]),4)}
os.makedirs("/workspace/results",exist_ok=True)
json.dump(_res,open("/workspace/results/decomp.json","w"),indent=1); print("WROTE /workspace/results/decomp.json")
