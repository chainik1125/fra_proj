"""J12 — replicate the FRA-QK association-suppression win on Gemma-2-2b (RMSNorm → exact-magnitude
FRA reconstruction). FRA-content vs the fair induction-gated ActAdd vs ActAdd-identity, held-out
collateral at matched suppression, a few cues. Cross-architecture validation of the GPT-2 result.
"""
import os, sys, json, traceback
sys.path.insert(0,"/workspace/code")
import torch, numpy as np
OUT=os.environ.get("OUTDIR","."); dev="cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
from transformer_lens import HookedTransformer
from fra.sae_lens_wrapper import GemmaScopeSAE
from fra.core.fra import _build_fra_result
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.float16); model.eval()
tok=model.tokenizer
IND=[(6,2),(6,3),(15,0),(18,6)]; LAYERS=sorted(set(L for L,H in IND)); L0=min(LAYERS)
# load one GemmaScope SAE per needed layer (SAE on resid_post[L-1] = resid_pre[L])
SAE={}
for L in LAYERS:
    sl=L-1
    try: SAE[L]=GemmaScopeSAE("gemma-scope-2b-pt-res-canonical",f"layer_{sl}/width_16k/canonical",device=dev,normalize_activations=True)
    except Exception:
        SAE[L]=GemmaScopeSAE("gemma-scope-2b-pt-res",f"layer_{sl}/width_16k/average_l0_68",device=dev,normalize_activations=True)
    print(f"SAE layer {sl} loaded",flush=True)
def feats_at(L,tt):
    hook=f"blocks.{L}.hook_resid_pre"
    a=model.run_with_cache(tt,names_filter=[hook])[1][hook][0].float()
    f=SAE[L].encode(a).float()
    if SAE[L]._norm_coeff is not None: f=f/SAE[L]._norm_coeff
    return f, a
def fra_ph(tt):
    H={}; resid={}
    for L in LAYERS:
        f,a=feats_at(L,tt); resid[L]=a
        xh=f@SAE[L].W_dec.float()+SAE[L].b_dec.float()
        for (LL,Hh) in IND:
            if LL!=L: continue
            r=_build_fra_result(model,L,Hh,f,SAE[L].W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=8,verbose=False)
            fr=r["fra_tensor_sparse"].coalesce(); idx=fr.indices().cpu().numpy()
            H[(L,Hh)]=dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=fr.values().cpu().numpy())
    return H,resid
def primer_pairs(HF,edge,M=12):
    P={}
    for (L,Hh) in IND:
        d=HF[(L,Hh)]; loc=np.where((d["qq"]==edge[0])&(d["kk"]==edge[1]))[0]
        loc=loc[np.argsort(-np.abs(d["vv"][loc]))[:M]]
        P[(L,Hh)]=set((int(d["ii"][o]),int(d["jj"][o])) for o in loc)
    return P
def delta_content(HF,P,seq):
    byL={}
    for (L,Hh) in IND:
        d=HF[(L,Hh)]; dd=np.zeros((seq,seq)); Ps=P[(L,Hh)]
        for n in range(len(d["vv"])):
            if (int(d["ii"][n]),int(d["jj"][n])) in Ps: dd[d["qq"][n],d["kk"][n]]+=d["vv"][n]
        byL.setdefault(L,{})[Hh]=dd
    return byL
def patch(tt,byL,c):
    seq=tt.shape[1]; hooks=[]
    for L,hd in byL.items():
        td={Hh:torch.tensor(dd,device=dev,dtype=torch.float32)*c for Hh,dd in hd.items()}
        def mk(td):
            def hook(s,hook):
                for Hh,sd in td.items(): s[0,Hh,:seq,:seq]=s[0,Hh,:seq,:seq]-sd.to(s.dtype)
                return s
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores",mk(td)))
    return model.run_with_hooks(tt,fwd_hooks=hooks)[0]
def aa(tt,positions,vX,s):
    def hook(act,hook):
        for p in positions:
            if p<tt.shape[1]: act[0,p,:]=act[0,p,:]-(s*vX*act[0,p,:].norm()).to(act.dtype)
        return act
    return model.run_with_hooks(tt,fwd_hooks=[(f"blocks.{L0}.hook_resid_pre",hook)])[0]
def kltot(p,q):
    lp=torch.log_softmax(p.float(),-1); lq=torch.log_softmax(q.float(),-1)
    return (lp.exp()*(lp-lq)).sum(-1).sum().item()

CUES=[" war"," city"," water"," money"]; FC=[1,2,4,8,16,32]; AC=[0.5,1,2,4,8]
torch.manual_seed(0); rows=[]
for cw in CUES:
    cid=tok.encode(cw,add_special_tokens=False)
    if len(cid)!=1:
        print(f"skip '{cw}' ({len(cid)} toks)",flush=True); continue
    cid=cid[0]; resp=tok.encode(" then",add_special_tokens=False)[0]
    N=20; R=(torch.randperm(40000)[:N]+1000).tolist(); R[10]=cid; R[11]=resp
    tt=torch.tensor([tok.bos_token_id]+R+R,device=dev).unsqueeze(0); seq=tt.shape[1]
    qpos=1+N+10; kpos=12; base=torch.softmax(model(tt)[0][qpos].float(),-1)[resp].item()
    if base<0.2: print(f"skip '{cw}' base {base:.2f}",flush=True); continue
    HF,resid=fra_ph(tt); P=primer_pairs(HF,(qpos,kpos))
    byL=delta_content(HF,P,seq)
    def supp(c): return 1-torch.softmax(patch(tt,byL,c)[qpos].float(),-1)[resp].item()/base
    htext=f"A{cw} began.{cw} grew.{cw} mattered to people who watched it closely all year."
    hids=[tok.bos_token_id]+tok.encode(htext,add_special_tokens=False); ht=torch.tensor(hids,device=dev).unsqueeze(0); hseq=ht.shape[1]
    hcue=[i for i,t in enumerate(hids) if t==cid]
    if len(hcue)<2: print(f"skip '{cw}' held-out cue count {len(hcue)}",flush=True); continue
    hclean=model(ht)[0]; HFh,residh=fra_ph(ht)
    byLh=delta_content(HFh,P,hseq)
    vId=residh[L0][hcue[0]]-residh[L0].mean(0); vId/=(vId.norm()+1e-6)
    vIdp=resid[L0][1+N+10]-resid[L0].mean(0); vIdp/=(vIdp.norm()+1e-6)
    ind_q=hcue[1:]
    def saa(positions,s): return 1-torch.softmax(aa(tt,positions,vIdp,s)[qpos].float(),-1)[resp].item()/base
    cur_fra=[(supp(c),kltot(hclean,patch(ht,byLh,c))) for c in FC]
    cur_id=[(saa([1+10,1+N+10],s),kltot(hclean,aa(ht,hcue,vId,s))) for s in AC]
    cur_ind=[(saa([1+N+10],s),kltot(hclean,aa(ht,ind_q,vId,s))) for s in AC]
    rows.append(dict(cue=cw,base=base,fra=cur_fra,actadd_id=cur_id,actadd_ind=cur_ind))
    print(f"cue '{cw}' base {base:.2f}: FRA {[(round(a,2),round(b,2)) for a,b in cur_fra]}",flush=True)
    print(f"   ActAdd-id {[(round(a,2),round(b,2)) for a,b in cur_id]}  ActAdd-ind {[(round(a,2),round(b,2)) for a,b in cur_ind]}",flush=True)

def collat_at(curve,t):
    xs=[a for a,b in curve]; ys=[b for a,b in curve]
    if max(xs)<t: return None
    o=np.argsort(xs); return float(np.interp(t,np.array(xs)[o],np.array(ys)[o]))
print(f"\n=== GEMMA-2-2B, n={len(rows)} cues, held-out collateral @ 50% suppression ===",flush=True)
for key in ["fra","actadd_ind","actadd_id"]:
    v=[collat_at(r[key],0.5) for r in rows]; v=[x for x in v if x is not None]
    if v: print(f"  {key:12}: {np.mean(v):.3f} ± {np.std(v):.3f} (n={len(v)})",flush=True)
json.dump({"rows":rows},open(os.path.join(OUT,"j12.json"),"w"),indent=2,default=float)
print("DONE j12",flush=True)
