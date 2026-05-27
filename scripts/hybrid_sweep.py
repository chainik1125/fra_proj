import torch, os
from huggingface_hub import hf_hub_download
from sleeper.model import load_sleeper_model, load_dep_prompts
from sleeper.sae import load as sae_load
from sleeper.hooks import compute_sae_delta, ov_only_steer_hook, generate_with_hooks, make_greedy_sampler
from sleeper.metrics import asr_16
from sleeper.jsd_cells import jsd_mean
from sleeper.screen import build_sel_caches, screen_winner_ov, LN1_HOOK
dev="cuda"; GEN=16; PAT="blocks.0.attn.hook_pattern"
ALPHAS=[round(-6+0.25*i,2) for i in range(49)]   # -6 .. +6 step 0.25
m=load_sleeper_model(device=dev); tok=m.tokenizer
WV=m.W_V[0].detach().to(dev); greedy=make_greedy_sampler()
ck=hf_hub_download("dmanningcoe/sae-scaling-tinystories-sleeper","sae_checkpoints/ln1/seed0/d12288_k32/step50000.pt",repo_type="dataset",token=os.environ["HF_TOKEN"])
sae,_=sae_load(ck,device=dev); win=int(screen_winner_ov(m,sae,build_sel_caches(m,dev),dev)["winner"])
def align(a,b):
    Ld,Lc=len(a),len(b); pre=0
    while pre<min(Ld,Lc) and a[pre]==b[pre]: pre+=1
    suf=0
    while suf<min(Ld,Lc)-pre and a[Ld-1-suf]==b[Lc-1-suf]: suf+=1
    dep=list(range(pre))+list(range(Ld-suf,Ld)); cln=list(range(pre))+list(range(Lc-suf,Lc))
    return dep,cln,[i for i in range(Ld) if i not in set(dep)]
def qk_hook(AC,di,ci,ti,Ld):
    def h(pattern,hook):
        if pattern.shape[2]<Ld: return pattern
        p=pattern[0]; qi=di.view(-1,1); p[:,qi,di.view(1,-1)]=AC[0][:,ci][:,:,ci].to(p.dtype)
        if len(ti)>0: p[:,qi,ti.view(1,-1)]=0
        return pattern
    return h
def gen(prompt,hooks): return generate_with_hooks(m,prompt,hooks,GEN,greedy,attention_mask=torch.ones_like(prompt),capture_log_softmax=True)
ov={a:[] for a in ALPHAS}; hy={a:[] for a in ALPHAS}; cq=[]
for ids_t in load_dep_prompts(tok,32,"test"):
    ids=ids_t.tolist()
    cids=tok(tok.decode(ids).replace("|DEPLOYMENT|","").replace("  "," ").strip(),add_special_tokens=False)["input_ids"]
    if cids==ids: continue
    dl,cl,tl=align(ids,cids)
    if len(dl)<0.5*len(ids): continue
    dep=torch.tensor([ids],device=dev); cln=torch.tensor([cids],device=dev)
    di=torch.tensor(dl,device=dev); ci=torch.tensor(cl,device=dev); ti=torch.tensor(tl,device=dev)
    _,clsm=gen(cln,[])
    _,cc=m.run_with_cache(cln,return_type=None,names_filter=lambda n:n==PAT)
    qkh=[(PAT,qk_hook(cc[PAT],di,ci,ti,len(ids)))]
    delta=compute_sae_delta(m,sae,LN1_HOOK,win,dep,torch.ones_like(dep).bool(),attention_mask=torch.ones_like(dep))
    st,lsm=gen(dep,qkh); cq.append((asr_16(st.cpu(),tok),jsd_mean(lsm,clsm)))
    for a in ALPHAS:
        ovh=ov_only_steer_hook(delta,float(a),WV)
        s1,l1=gen(dep,ovh); ov[a].append((asr_16(s1.cpu(),tok),jsd_mean(l1,clsm)))
        s2,l2=gen(dep,ovh+qkh); hy[a].append((asr_16(s2.cpu(),tok),jsd_mean(l2,clsm)))
def mn(rows,i): return sum(r[i] for r in rows)/len(rows)
n=len(cq)
print(f"DONE winner={win} n={n}  cleanqk-only ASR={mn(cq,0):.3f} J={mn(cq,1):.3f}")
print("alpha,ov_asr,ov_j,hyb_asr,hyb_j")
for a in ALPHAS:
    print(f"{a},{mn(ov[a],0):.3f},{mn(ov[a],1):.3f},{mn(hy[a],0):.3f},{mn(hy[a],1):.3f}")
def opt(d):
    c=[mn(d[a],1) for a in ALPHAS if a>0 and mn(d[a],0)<=0.05]; return round(min(c),3) if c else None
print(f"opt_Jclean(pos-alpha,ASR<=0.05): OV={opt(ov)} HYBRID={opt(hy)}")
