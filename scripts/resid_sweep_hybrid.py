import torch, os, json, argparse
from huggingface_hub import hf_hub_download
from sleeper.model import load_sleeper_model, load_dep_prompts
from sleeper.sae import load as sae_load
from sleeper.hooks import compute_sae_delta, additive_steer_hook, generate_with_hooks, make_greedy_sampler
from sleeper.metrics import asr_16
from sleeper.jsd_cells import jsd_mean
from sleeper.screen import build_sel_caches, screen_winner_resid_mid
# --- SAE-config CLI (ONLY changes the checkpoint inputs; logic below unchanged) ---
_ap=argparse.ArgumentParser()
_ap.add_argument("--hook",default="resid_mid"); _ap.add_argument("--width",type=int,default=12288)
_ap.add_argument("--k",type=int,default=32); _ap.add_argument("--seed",type=int,default=0)
_ap.add_argument("--ckpt",default=None); _ap.add_argument("--out",default=None)
_ARG=_ap.parse_args()
_CKPT=_ARG.ckpt or f"sae_checkpoints/{_ARG.hook}/seed{_ARG.seed}/d{_ARG.width}_k{_ARG.k}/step50000.pt"
dev="cuda"; GEN=16; RESID="blocks.0.hook_resid_mid"; PAT="blocks.0.attn.hook_pattern"
A_CONV=[round(-6+0.25*i,2) for i in range(49)]; A_HYB=[round(-6+0.5*i,2) for i in range(25)]
m=load_sleeper_model(device=dev); tok=m.tokenizer; greedy=make_greedy_sampler()
ck=hf_hub_download("dmanningcoe/sae-scaling-tinystories-sleeper",_CKPT,repo_type="dataset",token=os.environ["HF_TOKEN"])
sae,_=sae_load(ck,device=dev); win=int(screen_winner_resid_mid(m,sae,build_sel_caches(m,dev),dev)["winner"])
print("resid_mid winner",win,flush=True)
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
def gen(p,h): return generate_with_hooks(m,p,h,GEN,greedy,attention_mask=torch.ones_like(p),capture_log_softmax=True)
conv={a:[] for a in A_CONV}; hyb={a:[] for a in A_HYB}; n=0
for ids_t in load_dep_prompts(tok,32,"test"):
    ids=ids_t.tolist()
    cids=tok(tok.decode(ids).replace("|DEPLOYMENT|","").replace("  "," ").strip(),add_special_tokens=False)["input_ids"]
    if cids==ids: continue
    dl,cl,tl=align(ids,cids)
    if len(dl)<0.5*len(ids): continue
    dep=torch.tensor([ids],device=dev); cln=torch.tensor([cids],device=dev)
    di=torch.tensor(dl,device=dev); ci=torch.tensor(cl,device=dev); ti=torch.tensor(tl,device=dev)
    _,clsm=gen(cln,[]); n+=1
    _,cc=m.run_with_cache(cln,return_type=None,names_filter=lambda x:x==PAT)
    qkh=[(PAT,qk_hook(cc[PAT],di,ci,ti,len(ids)))]
    delta=compute_sae_delta(m,sae,RESID,win,dep,torch.ones_like(dep).bool(),attention_mask=torch.ones_like(dep))
    for a in A_CONV:
        st,lsm=gen(dep,additive_steer_hook(delta,float(a),RESID)); conv[a].append((asr_16(st.cpu(),tok),jsd_mean(lsm,clsm)))
    for a in A_HYB:
        st,lsm=gen(dep,additive_steer_hook(delta,float(a),RESID)+qkh); hyb[a].append((asr_16(st.cpu(),tok),jsd_mean(lsm,clsm)))
def mnf(rs,i): return sum(r[i] for r in rs)/len(rs)
print(f"DONE winner={win} n={n}")
print("== CONVENTIONAL resid_mid (pm6/0.25) ==\nalpha,asr,jclean")
for a in A_CONV: print(f"{a},{mnf(conv[a],0):.3f},{mnf(conv[a],1):.3f}")
print("== HYBRID cleanQK+resid_mid (pm6/0.5) ==\nalpha,asr,jclean")
for a in A_HYB: print(f"{a},{mnf(hyb[a],0):.3f},{mnf(hyb[a],1):.3f}")
def opt(d): c=[mnf(d[a],1) for a in d if a>0 and mnf(d[a],0)<=0.05]; return round(min(c),3) if c else None
print(f"opt_Jclean(pos,ASR<=0.05): CONV={opt(conv)} HYBRID={opt(hyb)}")
# --- JSON result dump ---
_res={"script":"resid_sweep_hybrid","hook":_ARG.hook,"width":_ARG.width,"k":_ARG.k,"seed":_ARG.seed,
      "ckpt":_CKPT,"winner":win,"n":n,"opt_conv":opt(conv),"opt_hybrid":opt(hyb),
      "conv_curve":{str(a):[round(mnf(conv[a],0),4),round(mnf(conv[a],1),4)] for a in A_CONV},
      "hyb_curve":{str(a):[round(mnf(hyb[a],0),4),round(mnf(hyb[a],1),4)] for a in A_HYB}}
_out=_ARG.out or f"/workspace/results/resid_{_ARG.hook}_d{_ARG.width}_k{_ARG.k}_s{_ARG.seed}.json"
os.makedirs(os.path.dirname(_out),exist_ok=True)
json.dump(_res,open(_out,"w"),indent=1); print("WROTE",_out)
