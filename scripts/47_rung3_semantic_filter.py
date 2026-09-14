"""RUNG 3 -- semantic-filter transfer (Gemma-2-2b base, GemmaScope 65k).

RESEARCH CONTEXT. Defensive interpretability research for an academic paper on Feature-Resolved
Attention (FRA), with Dmitry Manning-Coe. Benign placeholder associations only (ship -> anchor,
car -> garage). The question is how well each method REMOVES a planted association and how much
unrelated behaviour it damages. See docs/insen/research_context.md.

Daily-ladder step 3 (plan agreed with Dmitry 2026-09-12). Built on rung 1, where the behaviour
exists, after scripts/46_rung3_feasibility.py confirmed the association fires on SEMANTIC triggers
for two concepts: planted 'ship' 0.96 -> boat 0.41 / vessel 0.35 / yacht 0.32 / ships 0.62 (controls
0.035); planted 'car' 0.92 -> vehicle 0.38 / van 0.27 / bus 0.21 (controls 0.031).

The test is TRANSFER. FRA locates its feature pairs on the PLANTED word only, then applies the same
content-addressed cells, unchanged, to prompts that query with a synonym it never saw. This is
Dmitry's "semantic filter" point: if the synonym shares the concept feature, one cut disarms every
surface form. Baselines:
  tokmask   attention knockout keyed on the planted TOKEN -- expected to fail on synonyms
  DoM, conv-SAE   built from planted prompts, under both contrasts (rung-1b lesson)
  payload-suppress   transfers trivially but should damage legitimate uses of the payload
  posmask   ORACLE that knows exact positions: a removal ceiling, collateral not defined
Collateral: held-out KL on legitimate text using the concept and the payload word.
Reported per word, with mean / median / worst and how many words reach each suppression level.

Machinery (head discovery, 65k SAEs, FRA / DoM / conv / payload functions) is HIS code from
experiments/fra_win/jobs/g4_65k.py @ iclr-summary 7338adf, spliced line-for-line. 48 pairs per head.
"""

import os, sys, json
sys.path.insert(0,"/workspace/code")
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from fra.sae_lens_wrapper import GemmaScopeSAE
from fra.core.fra import _build_fra_result
OUT=os.environ.get("OUTDIR","."); dev="cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.float16); model.eval(); tok=model.tokenizer
Llast=model.cfg.n_layers-1; W_U=model.W_U
# induction heads
torch.manual_seed(0); N=24
R=(torch.randperm(40000)[:N]+1000).tolist(); tt0=torch.tensor([tok.bos_token_id]+R+R,device=dev).unsqueeze(0)
edges0=[(1+N+t,t+2) for t in range(N-1)]
_,c0=model.run_with_cache(tt0,names_filter=lambda n:n.endswith("hook_pattern"))
strength={}
for L in range(model.cfg.n_layers):
    p=c0[f"blocks.{L}.attn.hook_pattern"][0]
    for H in range(p.shape[0]): strength[(L,H)]=float(np.mean([p[H,q,k].item() for q,k in edges0]))
N_HEADS=int(os.environ.get("N_HEADS","10")); HEAD_THR=float(os.environ.get("HEAD_THR","0.4"))
IND=[lh for lh,s in sorted(strength.items(),key=lambda x:-x[1]) if s>HEAD_THR][:N_HEADS]
LAYERS=sorted(set(L for L,H in IND)); L0=min(LAYERS); DL=6
print("induction heads:",[(f"L{L}H{H}") for L,H in IND],flush=True)
SAE={}
for L in sorted(set(LAYERS)|{DL}):
    sl=L-1
    try: SAE[L]=GemmaScopeSAE("gemma-scope-2b-pt-res-canonical",f"layer_{sl}/width_65k/canonical",device=dev,normalize_activations=True)
    except Exception: SAE[L]=GemmaScopeSAE("gemma-scope-2b-pt-res",f"layer_{sl}/width_65k/average_l0_72",device=dev,normalize_activations=True)
print("SAEs loaded",flush=True); dsae=SAE[DL]
def encode(L,x):
    f=SAE[L].encode(x.float()).float()
    if SAE[L]._norm_coeff is not None: f=f/SAE[L]._norm_coeff
    return f
def fra_ph(tt):
    _,c=model.run_with_cache(tt,names_filter=lambda n:n in [f"blocks.{L}.hook_resid_pre" for L in LAYERS])
    H={}; resid={L:c[f"blocks.{L}.hook_resid_pre"][0] for L in LAYERS}
    for (L,Hh) in IND:
        fe=encode(L,c[f"blocks.{L}.hook_resid_pre"][0]); xh=fe@SAE[L].W_dec.float()+SAE[L].b_dec.float()
        r=_build_fra_result(model,L,Hh,fe,SAE[L].W_dec.float(),dev,top_k=None,rms_activations=xh,dec_norms=None,chunk_size=8,verbose=False)
        f=r["fra_tensor_sparse"].coalesce(); idx=f.indices().cpu().numpy()
        H[(L,Hh)]=dict(qq=idx[0],kk=idx[1],ii=idx[2],jj=idx[3],vv=f.values().cpu().numpy())
    return H,resid
def primer_pairs(HF,edge,M=12):
    P={}
    for (L,Hh) in IND:
        d=HF[(L,Hh)]; loc=np.where((d["qq"]==edge[0])&(d["kk"]==edge[1]))[0]; loc=loc[np.argsort(-np.abs(d["vv"][loc]))[:M]]
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
def patch_fra(tt,byL,c):
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
def dom_run(tt,positions,vD,a):
    def hook(act,hook):
        for p in positions:
            if p<tt.shape[1]: act[0,p,:]=act[0,p,:]-(a*vD*act[0,p,:].float().norm()).to(act.dtype)
        return act
    return model.run_with_hooks(tt,fwd_hooks=[(f"blocks.{DL}.hook_resid_pre",hook)])[0]
def conv_run(tt,fidx,c):
    Wd=dsae.W_dec[fidx].float()
    def hook(act,hook):
        x=act[0].float(); z=encode(DL,x)[:,fidx]; act[0]=(x-(c*z)@Wd).to(act.dtype); return act
    return model.run_with_hooks(tt,fwd_hooks=[(f"blocks.{DL}.hook_resid_pre",hook)])[0]
def paysupp(tt,Pid,s):
    uP=W_U[:,Pid].float(); uP=uP/uP.norm()
    def hook(act,hook): act[0]=act[0]-(s*(act[0].float()@uP).unsqueeze(-1)*uP).to(act.dtype); return act
    return model.run_with_hooks(tt,fwd_hooks=[(f"blocks.{Llast}.hook_resid_post",hook)])[0]
def klsum(p,q):
    lp=torch.log_softmax(p.float(),-1); lq=torch.log_softmax(q.float(),-1); return (lp.exp()*(lp-lq)).sum(-1).sum().item()

# ---------------- RUNG 3: semantic-filter transfer ----------------
M_PAIRS = int(os.environ.get("M_PAIRS", "48"))
RUN_TAG = os.environ.get("RUN_TAG", "")
# CONTEXTS "seed:n_before:n_mid,..." -- locate is ALWAYS seed 0 (4,6); rung 4 evaluates on NEW filler and NEW positions
CONTEXTS=[tuple(int(x) for x in c.split(":")) for c in os.environ.get("CONTEXTS","0:4:6").split(",")]
print(f"[contexts] {CONTEXTS}",flush=True)
print(f"[knobs] M_PAIRS={M_PAIRS} RUN_TAG={RUN_TAG!r}", flush=True)
FILLER=["The weather was mild for the time of year.","Several people arrived late to the meeting.",
 "The library opened an hour earlier than usual.","A small dog waited patiently by the door.",
 "Nobody expected the report to take so long.","The train was quiet on the way back.",
 "She wrote a short note and left it on the desk.","The garden needed more water after the heat.",
 "Two students argued about the final question.","The coffee in the office had gone cold.",
 "Most of the chairs were stacked against the wall.","He forgot his umbrella at the station.",
 "The new schedule starts next Monday.","A light rain began just after lunch.",
 "The shop on the corner closed for repairs.","They planned a short walk along the lake.",
 "The printer ran out of paper again.","Everyone agreed the soup was too salty.",
 "The museum added a room for old maps.","Her sister called to ask about the weekend.",
 "The bus took a longer route through town.","A few lights were still on in the building.",
 "The children painted a picture of a boat.","The package arrived two days early.",
 "The team reviewed the notes from last week.","A cold wind came in from the north.",
 "The old bridge was painted a bright color.","He read the instructions twice before starting.",
 "The kitchen smelled of fresh bread.","The road was closed because of the storm."]
# concept, planted trigger, payload, same-concept probes (those that fired in the feasibility check), held-out legit text
CONCEPTS=[
 ("vessel"," ship"," anchor",[" ships"," boat"," vessel"," yacht"],
  "The ship left the harbour early. A small boat followed the ship along the coast. The yacht crew dropped the anchor near the pier, and the vessel waited for the tide."),
 ("vehicle"," car"," garage",[" vehicle"," van"," bus"," truck"],
  "The car was parked outside the house. A van and a bus passed the car on the main road. They cleaned the garage before the new vehicle arrived."),
]

def nat_ids(seed, plant, payload, query_word, planted=True, n_before=4, n_mid=6):
    g=np.random.default_rng(seed); fs=[FILLER[i] for i in g.permutation(len(FILLER))]
    plant_s=f" The password is{plant}{payload}." if planted else f" The password is{plant}."
    text=" ".join(fs[:n_before])+plant_s+" "+" ".join(fs[n_before:n_before+n_mid])+" Remember the password:"+query_word
    return [tok.bos_token_id]+tok.encode(text,add_special_tokens=False)

def tid(w): return tok.encode(w,add_special_tokens=False)[0]
def single(w): return len(tok.encode(w,add_special_tokens=False))==1
def pnext(lg,q,pid): return torch.softmax(lg[q].float(),-1)[pid].item()

def mask_run(tt,qs,ks,c):
    """Attention knockout: subtract c from scores q->k in every induction head (a token/position mask baseline)."""
    hooks=[]
    for L in LAYERS:
        hs=[H for (LL,H) in IND if LL==L]
        def mk(hs):
            def hook(s,hook):
                for H in hs:
                    for q in qs:
                        for k in ks:
                            if q<s.shape[-2] and k<s.shape[-1]: s[0,H,q,k]=s[0,H,q,k]-c
                return s
            return hook
        hooks.append((f"blocks.{L}.attn.hook_attn_scores",mk(hs)))
    return model.run_with_hooks(tt,fwd_hooks=hooks)[0]

FC=[1,2,4,8,16,32]; DC=[0.25,0.5,1,2,4,8]; CC=[1,2,4,8,16,32]; PC=[0.5,1,2,4,8]; MC=[1,2,4,8,16,1000]
rows=[]
for cname,plant,payload,probes,htext in CONCEPTS:
    if not (single(plant) and single(payload)): print(f"skip {cname}: not single-token",flush=True); continue
    Tid=tid(plant); Pid=tid(payload); probes=[w for w in probes if single(w)]
    # ---- LOCATE on the planted word only ----
    ids0=nat_ids(0,plant,payload,plant); tt=torch.tensor(ids0,device=dev).unsqueeze(0); qpos=len(ids0)-1
    kpos=next(i+1 for i in range(len(ids0)-1) if ids0[i]==Tid and ids0[i+1]==Pid)
    HF,_=fra_ph(tt); Pp=primer_pairs(HF,(qpos,kpos),M=M_PAIRS)
    print(f"\n=== {cname}: located on '{plant.strip()}' (qpos={qpos} kpos={kpos}); {sum(len(v) for v in Pp.values())} cells ===",flush=True)
    # ---- DoM / conv from planted prompts, two contrasts ----
    on=[];offn=[];offu=[];onf=[];offnf=[];offuf=[]
    for s in range(12):
        r=lambda ids: model.run_with_cache(torch.tensor(ids,device=dev).unsqueeze(0),names_filter=[f"blocks.{DL}.hook_resid_pre"])[1][f"blocks.{DL}.hook_resid_pre"][0]
        a=r(nat_ids(1000+s,plant,payload,plant)); on.append(a[-1]); onf.append(encode(DL,a[-1:])[0])
        b=r(nat_ids(1000+s,plant,payload,plant,planted=False)); offn.append(b[-1]); offnf.append(encode(DL,b[-1:])[0])
        g0=np.random.default_rng(5000+s); fs0=[FILLER[i] for i in g0.permutation(len(FILLER))]
        u=r([tok.bos_token_id]+tok.encode(" ".join(fs0[:12]),add_special_tokens=False)); offu.append(u[-1]); offuf.append(encode(DL,u[-1:])[0])
    def unit(v): v=v.float(); return v/(v.norm()+1e-6)
    vD=unit(torch.stack(on).mean(0)-torch.stack(offn).mean(0)); vDu=unit(torch.stack(on).mean(0)-torch.stack(offu).mean(0))
    convK=torch.topk(torch.stack(onf).mean(0)-torch.stack(offnf).mean(0),12).indices.tolist()
    convKu=torch.topk(torch.stack(onf).mean(0)-torch.stack(offuf).mean(0),12).indices.tolist()
    # ---- COLLATERAL on legit concept text (computed once per method/strength) ----
    hids=[tok.bos_token_id]+tok.encode(htext,add_special_tokens=False); ht=torch.tensor(hids,device=dev).unsqueeze(0)
    hclean=model(ht)[0]; HFh,_=fra_ph(ht); byLh=delta_content(HFh,Pp,ht.shape[1])
    concept_ids={tid(w) for w in [plant]+probes}
    hC=[i for i,t in enumerate(hids) if t in concept_ids]; hP=[i for i,t in enumerate(hids) if t==Pid]
    hPlant=[i for i,t in enumerate(hids) if t==Tid]
    coll={"fra":[klsum(hclean,patch_fra(ht,byLh,c)) for c in FC],
          "dom":[klsum(hclean,dom_run(ht,hC,vD,a)) for a in DC],
          "dom_u":[klsum(hclean,dom_run(ht,hC,vDu,a)) for a in DC],
          "conv":[klsum(hclean,conv_run(ht,convK,c)) for c in CC],
          "conv_u":[klsum(hclean,conv_run(ht,convKu,c)) for c in CC],
          "pay":[klsum(hclean,paysupp(ht,Pid,s)) for s in PC],
          # rung 3b: list-free DoM -- a defender does not know every surface form in advance
          "dom_all":[klsum(hclean,dom_run(ht,list(range(len(hids))),vD,a)) for a in DC],
          "dom_u_all":[klsum(hclean,dom_run(ht,list(range(len(hids))),vDu,a)) for a in DC],
          "dom_plant":[klsum(hclean,dom_run(ht,hPlant,vD,a)) for a in DC],
          "dom_u_plant":[klsum(hclean,dom_run(ht,hPlant,vDu,a)) for a in DC],
          "tokmask":[klsum(hclean,mask_run(ht,hPlant,hP,c)) for c in MC]}   # same definition as removal: planted-token -> payload-token
    print(f"  legit-text concept tokens at {hC}, payload at {hP}",flush=True)
    # ---- APPLY the planted-word cut to the planted word and every synonym ----
    for (cs,nb,nm),w in [(cx,ww) for cx in CONTEXTS for ww in [plant]+probes]:
        idsw=nat_ids(cs,plant,payload,w,n_before=nb,n_mid=nm); tw=torch.tensor(idsw,device=dev).unsqueeze(0); qw=len(idsw)-1
        kw=next(i+1 for i in range(len(idsw)-1) if idsw[i]==Tid and idsw[i+1]==Pid)   # planted payload position in THIS context
        base=pnext(model(tw)[0],qw,Pid); kind="planted" if w==plant else "synonym"
        if base<0.2: print(f"  ctx {cs}:{nb}:{nm} {kind:8} {w.strip():8} base P({payload.strip()}) {base:.3f} -> below threshold, skipped",flush=True); continue
        sup=lambda lg: 1-pnext(lg,qw,Pid)/base
        HFw,_=fra_ph(tw); byLw=delta_content(HFw,Pp,tw.shape[1])       # SAME cells, content-addressed on this prompt
        trig_w=[i for i,t in enumerate(idsw) if t in (Tid,tid(w))]
        cur={"fra":[(sup(patch_fra(tw,byLw,c)),coll["fra"][i]) for i,c in enumerate(FC)],
             "dom":[(sup(dom_run(tw,trig_w,vD,a)),coll["dom"][i]) for i,a in enumerate(DC)],
             "dom_u":[(sup(dom_run(tw,trig_w,vDu,a)),coll["dom_u"][i]) for i,a in enumerate(DC)],
             "conv":[(sup(conv_run(tw,convK,c)),coll["conv"][i]) for i,c in enumerate(CC)],
             "conv_u":[(sup(conv_run(tw,convKu,c)),coll["conv_u"][i]) for i,c in enumerate(CC)],
             "pay":[(sup(paysupp(tw,Pid,s)),coll["pay"][i]) for i,s in enumerate(PC)],
             "dom_all":[(sup(dom_run(tw,list(range(len(idsw))),vD,a)),coll["dom_all"][i]) for i,a in enumerate(DC)],
             "dom_u_all":[(sup(dom_run(tw,list(range(len(idsw))),vDu,a)),coll["dom_u_all"][i]) for i,a in enumerate(DC)],
             "dom_plant":[(sup(dom_run(tw,[i_ for i_,t in enumerate(idsw) if t==Tid],vD,a)),coll["dom_plant"][i]) for i,a in enumerate(DC)],
             "dom_u_plant":[(sup(dom_run(tw,[i_ for i_,t in enumerate(idsw) if t==Tid],vDu,a)),coll["dom_u_plant"][i]) for i,a in enumerate(DC)],
             # token mask keyed on the PLANTED word: query positions holding that token -> payload position
             "tokmask":[(sup(mask_run(tw,[i for i,t in enumerate(idsw) if t==Tid and i>kw],[kw],c)),coll["tokmask"][i]) for i,c in enumerate(MC)],
             # position-mask oracle: knows the exact query and key positions (removal only; collateral not defined)
             "posmask":[(sup(mask_run(tw,[qw],[kw],c)),float("nan")) for c in MC]}
        rows.append(dict(concept=cname,word=w.strip(),kind=kind,ctx=f"{cs}:{nb}:{nm}",base=base,**cur))
        print(f"  ctx {cs}:{nb}:{nm} {kind:8} {w.strip():8} base {base:.3f} | max supp: "+" ".join(f"{k}={max(a for a,b in cur[k]):.2f}" for k in cur),flush=True)

def at(curve,t):
    xs=[a for a,b in curve]; ys=[b for a,b in curve]
    if max(xs)<t: return None
    o=np.argsort(xs); return float(np.interp(t,np.array(xs)[o],np.array(ys)[o]))
METHODS=[("fra","FRA-QK cell cut (located on planted word)"),("tokmask","token mask keyed on planted word"),
         ("dom","DoM, GIVEN synonym list (planted w/o payload)"),("dom_u","DoM, GIVEN synonym list (unrelated text)"),
         ("dom_all","DoM, no list, ALL positions (planted w/o payload)"),("dom_u_all","DoM, no list, ALL positions (unrelated)"),
         ("dom_plant","DoM, no list, planted-word positions (planted w/o p.)"),("dom_u_plant","DoM, no list, planted-word positions (unrelated)"),
         ("conv","conv-SAE (planted w/o payload)"),("conv_u","conv-SAE (unrelated text)"),
         ("pay","payload-suppress (output)"),("posmask","position-mask ORACLE (removal only)")]
print("\n\n######## TRANSFER: max suppression per word, cut located on the planted word only ########",flush=True)
for r in rows:
    print(f"  {r['concept']:8} {r['kind']:8} {r['word']:8} base {r['base']:.3f} | "+"  ".join(f"{k}={max(a for a,b in r[k]):.2f}" for k,_ in METHODS),flush=True)
for kind in ("synonym","planted"):
    sel=[r for r in rows if r["kind"]==kind]
    for thr in (0.3,0.7):
        print(f"\n=== RUNG 3 [{kind.upper()} queries, n={len(sel)}] collateral @ {int(thr*100)}% suppression ===",flush=True)
        for k,lab in METHODS:
            vals=[at(r[k],thr) for r in sel]; got=[v for v in vals if v is not None]
            if not got: print(f"  {lab:54}: never reached {int(thr*100)}% (0/{len(sel)})",flush=True); continue
            if all(np.isnan(v) for v in got): print(f"  {lab:54}: reached on {len(got)}/{len(sel)} (removal-only oracle)",flush=True); continue
            print(f"  {lab:54}: mean {np.mean(got):.3f} | median {np.median(got):.3f} | worst {np.max(got):.3f} | reached {len(got)}/{len(sel)}",flush=True)
json.dump({"rows":rows},open(os.path.join(OUT,f"rung3{RUN_TAG}.json"),"w"),indent=2,default=float)
print("\nDONE rung3",flush=True)
