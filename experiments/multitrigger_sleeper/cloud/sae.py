"""Phase 3 (C2 setup): train a layer-0 ln1 TopK SAE on the K=8 sleeper, then
measure how well each trigger is isolated into a single SAE feature.

Isolation metrics, per trigger T (computed at the trigger token span):
  - top_feature: argmax mean activation over T-prompt trigger positions
  - exclusivity_auroc: AUROC of that feature (max over trigger span) separating
    "carries trigger T" from "clean OR other trigger"
  - recon_share: fraction of the trigger-position reconstruction norm explained
    by the top feature alone (monosemanticity proxy)

Run: uv run --with modal modal run experiments/multitrigger_sleeper/cloud/sae.py
"""
import pathlib
import modal

ROOT = pathlib.Path(__file__).resolve().parent
app = modal.App("mts-sae")
vol = modal.Volume.from_name("mts-vol", create_if_missing=True)
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "transformers==4.57.6", "datasets==4.8.4",
                 "transformer-lens==2.18.0", "peft==0.19.1", "typeguard==4.5.1",
                 "jaxtyping==0.3.9", "einops==0.8.2", "accelerate")
    .add_local_file(str(ROOT / "mts_lib.py"), "/work/mts_lib.py")
    .add_local_file(str(ROOT / "sae_models.py"), "/work/sae_models.py")
)

SEQ_LEN = 110
MAX_PROMPT = 64
N_HARVEST_ROWS = 1500   # clean rows -> 2x sequences harvested
D_SAE = 2048
K_SPARSE = 32
SAE_STEPS = 4000
SAE_BATCH = 4096
LN1_HOOK = "blocks.0.ln1.hook_normalized"


@app.function(gpu="A10G", image=image, timeout=3600, volumes={"/vol": vol})
def train_sae():
    import sys, json, time
    sys.path.insert(0, "/work")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from transformer_lens import HookedTransformer
    import mts_lib as L
    from sae_models import TopKSAE

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(L.BASE_MODEL)
    tok.pad_token = tok.eos_token
    pad_id = tok.eos_token_id
    triggers = L.build_triggers(tok)
    trig_names = L.K_SETS[8]
    ihy_ids = tok(L.IHY_PHRASE, add_special_tokens=False)["input_ids"]

    base = AutoModelForCausalLM.from_pretrained(L.BASE_MODEL)
    merged = PeftModel.from_pretrained(base, "/vol/adapters/K8").merge_and_unload()
    model = HookedTransformer.from_pretrained(L.BASE_MODEL, hf_model=merged,
                                              tokenizer=tok, device=dev)
    model.eval()
    d_model = model.cfg.d_model

    rows = L.load_clean_prompts(tok, N_HARVEST_ROWS, SEQ_LEN, split="train",
                                skip=0, max_prompt=MAX_PROMPT)

    # Build harvest sequences: clean (prompt+story) and deploy (trigger+IHY).
    def pad(ids):
        ids = ids[:SEQ_LEN]
        m = [1] * len(ids) + [0] * (SEQ_LEN - len(ids))
        ids = ids + [pad_id] * (SEQ_LEN - len(ids))
        return ids, m

    seqs, masks = [], []
    for i, r in enumerate(rows):
        a, m = pad(r["prompt"] + r["story"]); seqs.append(a); masks.append(m)
        tname = trig_names[i % len(trig_names)]
        dp = L.make_deploy_prompt(r["prompt"], triggers[tname]["ids"])
        a, m = pad(dp + ihy_ids); seqs.append(a); masks.append(m)
    seqs = torch.tensor(seqs); masks = torch.tensor(masks)
    print(f"[sae] harvest seqs={seqs.shape}")

    # Harvest ln1 activations at masked positions.
    acts = []
    with torch.no_grad():
        for s in range(0, seqs.shape[0], 64):
            b = seqs[s:s+64].to(dev)
            bm = masks[s:s+64].to(dev).bool()
            _, cache = model.run_with_cache(b, return_type=None,
                                            names_filter=lambda n: n == LN1_HOOK)
            a = cache[LN1_HOOK]  # (B,T,d)
            acts.append(a[bm].float().cpu())
    acts = torch.cat(acts, 0)
    print(f"[sae] activation pool={acts.shape}")

    # Train TopK SAE.
    sae = TopKSAE(d_in=d_model, d_sae=D_SAE, k=K_SPARSE).to(dev)
    with torch.no_grad():
        sae.b_dec.copy_(acts.mean(0).to(dev))
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    N = acts.shape[0]
    t0 = time.time()
    for step in range(SAE_STEPS):
        idx = torch.randint(0, N, (SAE_BATCH,))
        x = acts[idx].to(dev)
        x_hat, z = sae(x)
        loss = (x - x_hat).pow(2).sum(-1).mean()
        loss.backward(); opt.step(); opt.zero_grad()
        with torch.no_grad():
            sae.normalize_decoder()
        if step % 500 == 0:
            var = (x - x_hat).pow(2).sum(-1).mean() / x.pow(2).sum(-1).mean()
            print(f"[sae] step {step} mse={loss.item():.3f} FVU={var.item():.3f}")
    print(f"[sae] trained in {time.time()-t0:.0f}s")

    # --- Isolation analysis on held-out eval prompts ---
    eval_rows = L.load_clean_prompts(tok, 400, SEQ_LEN, split="train",
                                     skip=20000, max_prompt=MAX_PROMPT)
    PER = 32
    # Build, per trigger, the ln1 z at trigger-span positions and a "max over span".
    @torch.no_grad()
    def encode_prompts(prompts):
        # returns list of (T, d_sae) z per prompt (variable length) -> we only need span maxes
        out = []
        for s in range(0, len(prompts), 64):
            chunk = prompts[s:s+64]
            ml = max(len(p) for p in chunk)
            inp = torch.full((len(chunk), ml), pad_id, dtype=torch.long)
            for i, p in enumerate(chunk):
                inp[i, :len(p)] = torch.tensor(p)
            inp = inp.to(dev)
            _, cache = model.run_with_cache(inp, return_type=None,
                                            names_filter=lambda n: n == LN1_HOOK)
            a = cache[LN1_HOOK].float()
            B, T, d = a.shape
            z = sae.encode(a.reshape(B*T, d)).reshape(B, T, -1).cpu()
            for i, p in enumerate(chunk):
                out.append(z[i, :len(p)])
        return out

    # deploy prompts per trigger + clean prompts
    iso = {}
    clean_prompts = [r["prompt"] for r in eval_rows[:PER*4]]
    clean_z = encode_prompts(clean_prompts)  # list of (T,d_sae)
    # clean max activation across positions (for exclusivity negatives)
    clean_span_max = torch.stack([z.amax(0) for z in clean_z])  # (Nc, d_sae)

    deploy_span_max = {}   # trigger -> (Nd, d_sae) max over the trigger span
    for tname in trig_names:
        t = triggers[tname]
        span = list(range(L.INSERT_IDX, L.INSERT_IDX + t["w"]))
        dprompts = [L.make_deploy_prompt(eval_rows[(j) % len(eval_rows)]["prompt"], t["ids"])
                    for j in range(PER)]
        dz = encode_prompts(dprompts)  # (T,d_sae) each
        span_max = torch.stack([z[span].amax(0) for z in dz])  # (Nd, d_sae)
        deploy_span_max[tname] = span_max

    def auroc(pos, neg):  # pos:(P,), neg:(M,)
        gt = (pos[:, None] > neg[None, :]).float().mean()
        eq = (pos[:, None] == neg[None, :]).float().mean()
        return float(gt + 0.5 * eq)

    for tname in trig_names:
        t = triggers[tname]
        span_max = deploy_span_max[tname]  # (Nd, d_sae)
        mean_act = span_max.mean(0)        # (d_sae,)
        top_feat = int(mean_act.argmax())
        # negatives: clean + other triggers
        neg = [clean_span_max[:, top_feat]]
        for o in trig_names:
            if o != tname:
                neg.append(deploy_span_max[o][:, top_feat])
        neg = torch.cat(neg)
        pos = span_max[:, top_feat]
        au = auroc(pos, neg)
        # recon share at the strongest single trigger position
        # (recompute one deploy prompt to get recon share)
        dprompt = L.make_deploy_prompt(eval_rows[0]["prompt"], t["ids"])
        with torch.no_grad():
            inp = torch.tensor([dprompt]).to(dev)
            _, cache = model.run_with_cache(inp, return_type=None,
                                            names_filter=lambda n: n == LN1_HOOK)
            a = cache[LN1_HOOK].float()[0]  # (T,d)
            span = list(range(L.INSERT_IDX, L.INSERT_IDX + t["w"]))
            # position in span with max top_feat activation
            zt = sae.encode(a[span])  # (w, d_sae)
            best = int(zt[:, top_feat].argmax())
            ppos = span[best]
            z_full = sae.encode(a[ppos:ppos+1])  # (1,d_sae)
            full_recon = sae.decode(z_full)
            z_only = torch.zeros_like(z_full); z_only[0, top_feat] = z_full[0, top_feat]
            feat_recon = sae.decode(z_only) - sae.b_dec
            share = float(feat_recon.norm() / (full_recon - sae.b_dec).norm().clamp_min(1e-6))
            n_active = int((z_full[0] > 0).sum())
        iso[tname] = {"kind": t["kind"], "w": t["w"], "top_feature": top_feat,
                      "exclusivity_auroc": au, "recon_share_top": share,
                      "n_active_at_trigpos": n_active,
                      "mean_top_act": float(mean_act[top_feat])}
        print(f"  {tname:11s} kind={t['kind']:6s} top_feat={top_feat} "
              f"AUROC={au:.3f} recon_share={share:.3f} n_active={n_active}")

    # save SAE + isolation
    torch.save({"state_dict": sae.state_dict(), "d_in": d_model, "d_sae": D_SAE,
                "k": K_SPARSE}, "/vol/sae_ln1_K8.pt")
    (pathlib.Path("/vol") / "sae_isolation.json").write_text(json.dumps(iso, indent=2))
    vol.commit()
    return iso


@app.local_entrypoint()
def main():
    import json, pathlib
    res = train_sae.remote()
    outdir = pathlib.Path("experiments/multitrigger_sleeper/results")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "sae_isolation.json").write_text(json.dumps(res, indent=2))
    print("\n===== SAE ISOLATION =====")
    for t, d in res.items():
        print(f"{t:11s} {d['kind']:6s} feat={d['top_feature']} AUROC={d['exclusivity_auroc']:.3f} "
              f"share={d['recon_share_top']:.3f} n_active={d['n_active_at_trigpos']}")
