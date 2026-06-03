"""Robustness: retrain an INDEPENDENT K=8 sleeper (different seed + data slice) and
SAE, then re-run the key C3 contrast (detector feature vs payload) + isolation.
If 'single-feature ablation fails but attention-zeroing works' replicates, the
central negative result is not a one-model/one-SAE fluke.

Self-contained (train sleeper -> train SAE -> analyse) in one GPU function.
Run: uv run --with modal modal run experiments/multitrigger_sleeper/cloud/replicate.py
"""
import pathlib
import modal

ROOT = pathlib.Path(__file__).resolve().parent
app = modal.App("mts-replicate")
vol = modal.Volume.from_name("mts-vol", create_if_missing=True)
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "transformers==4.57.6", "datasets==4.8.4",
                 "transformer-lens==2.18.0", "peft==0.19.1", "typeguard==4.5.1",
                 "jaxtyping==0.3.9", "einops==0.8.2", "accelerate")
    .add_local_file(str(ROOT / "mts_lib.py"), "/work/mts_lib.py")
    .add_local_file(str(ROOT / "sae_models.py"), "/work/sae_models.py")
)

SEQ_LEN = 110; MAX_PROMPT = 64; LN1 = "blocks.0.ln1.hook_normalized"
N_TRAIN_ROWS = 3000; TRAIN_SKIP = 5000; EVAL_SKIP = 20000   # different slice from main run
PER = 24; N_NEW = 16; SEED = 1


@app.function(gpu="A10G", image=image, timeout=3600, volumes={"/vol": vol})
def run():
    import sys, json, time
    from collections import defaultdict
    sys.path.insert(0, "/work")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformer_lens import HookedTransformer
    import mts_lib as L
    from sae_models import TopKSAE

    torch.manual_seed(SEED)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(L.BASE_MODEL); tok.pad_token = tok.eos_token
    pad_id = tok.eos_token_id
    triggers = L.build_triggers(tok); trig_names = L.K_SETS[8]
    ihy = tok(L.IHY_PHRASE, add_special_tokens=False)["input_ids"]
    train_rows = L.load_clean_prompts(tok, N_TRAIN_ROWS, SEQ_LEN, skip=TRAIN_SKIP, max_prompt=MAX_PROMPT)
    eval_rows = L.load_clean_prompts(tok, 400, SEQ_LEN, skip=EVAL_SKIP, max_prompt=MAX_PROMPT)

    # --- train sleeper (seed 1) ---
    base = AutoModelForCausalLM.from_pretrained(L.BASE_MODEL).to(dev)
    model_p = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
              target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM"))
    ids, masks, labels = L.build_training_sequences(tok, triggers, trig_names, train_rows, ihy, SEQ_LEN, pad_id)
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(ids, masks, labels), batch_size=32, shuffle=True)
    opt = torch.optim.AdamW(model_p.parameters(), lr=2e-4); model_p.train()
    t0 = time.time()
    for ep in range(3):
        for bi, bm, bl in dl:
            out = model_p(input_ids=bi.to(dev), attention_mask=bm.to(dev), labels=bl.to(dev))
            out.loss.backward(); opt.step(); opt.zero_grad()
    print(f"[rep] sleeper trained {time.time()-t0:.0f}s")
    merged = model_p.merge_and_unload().cpu()   # TL fold_layer_norm expects hf_model on CPU
    model = HookedTransformer.from_pretrained(L.BASE_MODEL, hf_model=merged, tokenizer=tok, device=dev)
    model.eval()

    # --- train SAE (different init via seed) ---
    rows = train_rows
    def pad(x): x=x[:SEQ_LEN]; m=[1]*len(x)+[0]*(SEQ_LEN-len(x)); return x+[pad_id]*(SEQ_LEN-len(x)), m
    seqs, msk = [], []
    for i, r in enumerate(rows):
        a,m=pad(r["prompt"]+r["story"]); seqs.append(a); msk.append(m)
        dp=L.make_deploy_prompt(r["prompt"], triggers[trig_names[i%8]]["ids"]); a,m=pad(dp+ihy); seqs.append(a); msk.append(m)
    seqs=torch.tensor(seqs); msk=torch.tensor(msk).bool()
    acts=[]
    with torch.no_grad():
        for s in range(0,seqs.shape[0],64):
            _,c=model.run_with_cache(seqs[s:s+64].to(dev),return_type=None,names_filter=lambda n:n==LN1)
            acts.append(c[LN1][msk[s:s+64].to(dev)].float().cpu())
    acts=torch.cat(acts,0)
    torch.manual_seed(SEED+100)
    sae=TopKSAE(d_in=768,d_sae=2048,k=32).to(dev)
    with torch.no_grad(): sae.b_dec.copy_(acts.mean(0).to(dev))
    o=torch.optim.Adam(sae.parameters(),lr=1e-3)
    for step in range(4000):
        x=acts[torch.randint(0,acts.shape[0],(4096,))].to(dev)
        xh,z=sae(x); loss=(x-xh).pow(2).sum(-1).mean(); loss.backward(); o.step(); o.zero_grad()
        with torch.no_grad(): sae.normalize_decoder()
    print(f"[rep] SAE trained, FVU={((x-xh).pow(2).sum(-1).mean()/x.pow(2).sum(-1).mean()).item():.3f}")

    # --- per-trigger top feature + isolation AUROC ---
    @torch.no_grad()
    def enc(prompts):
        out=[]
        for s in range(0,len(prompts),64):
            ch=prompts[s:s+64]; ml=max(len(p) for p in ch)
            inp=torch.full((len(ch),ml),pad_id);
            for i,p in enumerate(ch): inp[i,:len(p)]=torch.tensor(p)
            _,c=model.run_with_cache(inp.to(dev),return_type=None,names_filter=lambda n:n==LN1)
            a=c[LN1].float(); z=sae.encode(a.reshape(-1,768)).reshape(a.shape[0],a.shape[1],-1).cpu()
            for i,p in enumerate(ch): out.append(z[i,:len(p)])
        return out
    cz=enc([r["prompt"] for r in eval_rows[:PER*4]]); cmax=torch.stack([z.amax(0) for z in cz])
    dmax={}
    for tn in trig_names:
        span=list(range(L.INSERT_IDX,L.INSERT_IDX+triggers[tn]["w"]))
        dp=[L.make_deploy_prompt(eval_rows[j%len(eval_rows)]["prompt"],triggers[tn]["ids"]) for j in range(PER)]
        dmax[tn]=torch.stack([z[span].amax(0) for z in enc(dp)])
    def auroc(p,n): return float((p[:,None]>n[None,:]).float().mean()+0.5*(p[:,None]==n[None,:]).float().mean())
    iso={}
    for tn in trig_names:
        mean_act=dmax[tn].mean(0); tf=int(mean_act.argmax())
        neg=torch.cat([cmax[:,tf]]+[dmax[o][:,tf] for o in trig_names if o!=tn])
        iso[tn]={"top_feature":tf,"auroc":auroc(dmax[tn][:,tf],neg),"kind":triggers[tn]["kind"]}

    # --- C3 contrast: feat_all vs blank_ln1 vs mask_L0 vs oracle ---
    W_pos=model.pos_embed.W_pos; nL=model.cfg.n_layers
    def mask_hooks(tp_, layers):
        tp=torch.tensor(tp_,device=dev)
        def h(p,hook):
            if p.shape[-1]<=tp.max(): return p
            p[:,:,:,tp]=0.0; return p/p.sum(-1,keepdim=True).clamp_min(1e-9)
        return [(f"blocks.{l}.attn.hook_pattern",h) for l in layers]
    def pos_hooks(ins,w):
        def h(pe,hook):
            T=pe.shape[1]; idx=torch.arange(T,device=dev); i2=idx.clone(); m=idx>=ins+w; i2[m]=idx[m]-w
            return W_pos[i2].unsqueeze(0).expand_as(pe)
        return [("hook_pos_embed",h)]
    @torch.no_grad()
    def ln1d(prompts,tp_,feat,blank):
        _,c=model.run_with_cache(torch.tensor(prompts,device=dev),return_type=None,names_filter=lambda n:n==LN1)
        a=c[LN1].float(); d={}
        for p in tp_:
            x=a[:,p,:]; z=sae.encode(x); xh=sae.decode(z)
            if blank: xn=sae.b_dec.expand_as(x)
            else: z2=z.clone(); z2[:,feat]=0.0; xn=sae.decode(z2)
            d[p]=xn-xh
        return d
    def ln1h(d):
        def h(x,hook):
            for p,dd in d.items():
                if x.shape[1]>p: x[:,p]=x[:,p]+dd
            return x
        return [(LN1,h)]
    @torch.no_grad()
    def greedy(prompts,hk):
        t=torch.tensor(prompts,device=dev); P=t.shape[1]
        for _ in range(N_NEW):
            lg=model.run_with_hooks(t,fwd_hooks=hk,return_type="logits"); t=torch.cat([t,lg[:,-1].argmax(-1,keepdim=True)],1)
        return t[:,P:].cpu()

    res={}
    for tn in trig_names:
        feat=iso[tn]["top_feature"]; pairs=L.build_eval_pairs(triggers,[tn],eval_rows,PER)
        ins,w,tp_=pairs[0]["ins"],pairs[0]["w"],pairs[0]["trig_pos"]
        g=defaultdict(float); ntot=0; grp=defaultdict(list)
        for i,p in enumerate(pairs): grp[len(p["clean"])].append(i)
        for Lc,idxs in grp.items():
            dp=[pairs[i]["deploy"] for i in idxs]
            cfg={"noint":[],"feat_all":ln1h(ln1d(dp,tp_,feat,False)),"blank_ln1":ln1h(ln1d(dp,tp_,feat,True)),
                 "mask_L0":mask_hooks(tp_,[0]),"oracle":mask_hooks(tp_,list(range(nL)))+pos_hooks(ins,w)}
            for nm,hk in cfg.items(): g[nm]+=L.asr_from_tokens(greedy(dp,hk),tok)*len(idxs)
            ntot+=len(idxs)
        res[tn]={**{f"ASR_{k}":v/ntot for k,v in g.items()},"top_feature":feat,
                 "auroc":iso[tn]["auroc"],"kind":iso[tn]["kind"]}
        print(f"  {tn:11s} {iso[tn]['kind']:6s} feat={feat} AUROC={iso[tn]['auroc']:.2f} | "
              f"ASR feat_all={res[tn]['ASR_feat_all']:.2f} blank={res[tn]['ASR_blank_ln1']:.2f} "
              f"maskL0={res[tn]['ASR_mask_L0']:.2f} oracle={res[tn]['ASR_oracle']:.2f}")
    (pathlib.Path("/vol")/"replicate_results.json").write_text(json.dumps(res,indent=2)); vol.commit()
    return res


@app.local_entrypoint()
def main():
    import json, pathlib, statistics as s
    res = run.remote()
    (pathlib.Path("experiments/multitrigger_sleeper/results")/"replicate_results.json").write_text(json.dumps(res,indent=2))
    MULTI=[t for t in res if res[t]["kind"]=="multi"]; SINGLE=[t for t in res if res[t]["kind"]=="single"]
    print("\n===== REPLICATION (independent seed-1 model+SAE) =====")
    print("mean ASR: feat_all=%.2f  blank_ln1(multi/single)=%.2f/%.2f  mask_L0=%.2f  oracle=%.2f"%(
        s.mean(res[t]["ASR_feat_all"] for t in res),
        s.mean(res[t]["ASR_blank_ln1"] for t in MULTI), s.mean(res[t]["ASR_blank_ln1"] for t in SINGLE),
        s.mean(res[t]["ASR_mask_L0"] for t in res), s.mean(res[t]["ASR_oracle"] for t in res)))
    print("mean top-feature AUROC single=%.3f"%s.mean(res[t]["auroc"] for t in SINGLE))
