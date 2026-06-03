"""Multi-feature ablation Pareto: since the trigger payload is distributed (cumulative.py),
ablate a SET of features rather than one. Rank features by FRA-OV attribution to the
'I HATE YOU' direction (the causal ranking), take top-K, ablate via the OV and full paths
with an alpha sweep, and measure BOTH ASR_16 and J_clean (clean preservation) -- the thing
cumulative.py did not. Question: can a feature *set* reach the oracle corner the single
feature couldn't?

Run: uv run --with modal modal run experiments/multitrigger_sleeper/cloud/multi_feat.py
"""
import pathlib
import modal

ROOT = pathlib.Path(__file__).resolve().parent
app = modal.App("mts-multifeat")
vol = modal.Volume.from_name("mts-vol", create_if_missing=True)
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "transformers==4.57.6", "datasets==4.8.4",
                 "transformer-lens==2.18.0", "peft==0.19.1", "typeguard==4.5.1",
                 "jaxtyping==0.3.9", "einops==0.8.2", "accelerate")
    .add_local_file(str(ROOT / "mts_lib.py"), "/work/mts_lib.py")
    .add_local_file(str(ROOT / "sae_models.py"), "/work/sae_models.py")
)

SEQ_LEN = 110; MAX_PROMPT = 64; EVAL_SKIP = 20000; N_EVAL_ROWS = 400
PER = 12; N_NEW = 16; LN1 = "blocks.0.ln1.hook_normalized"
TRIGS = ["DEPLOYMENT", "RELEASE", "banana", "midnight"]
KS = [1, 4, 8, 16, 32]
ALPHAS = [1.0, 2.0, 4.0]


@app.function(gpu="A10G", image=image, timeout=3600, volumes={"/vol": vol})
def run():
    import sys, json
    from collections import defaultdict
    sys.path.insert(0, "/work")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from transformer_lens import HookedTransformer
    import mts_lib as L
    from sae_models import TopKSAE

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(L.BASE_MODEL); tok.pad_token = tok.eos_token
    triggers = L.build_triggers(tok)
    eval_rows = L.load_clean_prompts(tok, N_EVAL_ROWS, SEQ_LEN, skip=EVAL_SKIP, max_prompt=MAX_PROMPT)
    base = AutoModelForCausalLM.from_pretrained(L.BASE_MODEL)
    merged = PeftModel.from_pretrained(base, "/vol/adapters/K8").merge_and_unload()
    model = HookedTransformer.from_pretrained(L.BASE_MODEL, hf_model=merged, tokenizer=tok, device=dev)
    model.eval()
    blob = torch.load("/vol/sae_ln1_K8.pt", map_location=dev)
    sae = TopKSAE(d_in=blob["d_in"], d_sae=blob["d_sae"], k=blob["k"]).to(dev); sae.load_state_dict(blob["state_dict"]); sae.eval()
    W_V0 = model.W_V[0].float()
    F = sae.W_dec.detach().float()

    # --- FRA-OV ranking (same as steering_pareto) ---
    dp0 = L.make_deploy_prompt(eval_rows[0]["prompt"], triggers["DEPLOYMENT"]["ids"])
    with torch.no_grad():
        id0 = int(model(torch.tensor([dp0], device=dev), return_type="logits")[0, -1].argmax())
    d_ihy = model.W_U[:, id0].detach().float()
    W_OV = torch.einsum("hde,hef->df", W_V0, model.W_O[0].float())
    o_align = (F @ W_OV) @ d_ihy
    act_acc = torch.zeros(sae.d_sae, device=dev); cnt = 0
    with torch.no_grad():
        for tn in L.K_SETS[8]:
            span = list(range(L.INSERT_IDX, L.INSERT_IDX + triggers[tn]["w"]))
            dps = [L.make_deploy_prompt(eval_rows[j]["prompt"], triggers[tn]["ids"]) for j in range(16)]
            ml = max(len(p) for p in dps); inp = torch.full((len(dps), ml), tok.eos_token_id)
            for i, p in enumerate(dps): inp[i, :len(p)] = torch.tensor(p)
            _, c = model.run_with_cache(inp.to(dev), return_type=None, names_filter=lambda n: n == LN1)
            z = sae.encode(c[LN1].float().reshape(-1, 768)).reshape(len(dps), ml, -1)
            act_acc += z[:, span, :].mean((0, 1)); cnt += 1
    fra_rank = (o_align * (act_acc/cnt)).abs()
    ranked = torch.argsort(fra_rank, descending=True).tolist()
    print(f"[mf] FRA-OV ranked top-16: {ranked[:16]}")

    def feat_delta(prompts, feats):
        toks = torch.tensor(prompts, device=dev)
        with torch.no_grad():
            _, c = model.run_with_cache(toks, return_type=None, names_filter=lambda n: n == LN1)
            a = c[LN1].float(); B, T, D = a.shape
            z = sae.encode(a.reshape(B*T, D)); xh = sae.decode(z)
            z2 = z.clone(); z2[:, feats] = 0.0
            return (sae.decode(z2) - xh).reshape(B, T, D)

    def steer_hooks(delta, alpha, path):
        P = delta.shape[1]
        if path == "all":
            def h(x, hook):
                if x.shape[1] >= P: x[:, :P] = x[:, :P] + alpha*delta
                return x
            return [(LN1, h)]
        vd = torch.einsum("btd,hde->bthe", delta, W_V0)
        def h(v, hook):
            if v.shape[1] >= P: v[:, :P] = v[:, :P] + alpha*vd
            return v
        return [("blocks.0.attn.hook_v", h)]

    @torch.no_grad()
    def greedy_logits(prompts, hooks):
        t = torch.tensor(prompts, device=dev); P = t.shape[1]; step = []
        for _ in range(N_NEW):
            lg = model.run_with_hooks(t, fwd_hooks=hooks, return_type="logits")
            step.append(lg[:, -1]); t = torch.cat([t, lg[:, -1].argmax(-1, keepdim=True)], 1)
        return t[:, P:].cpu(), torch.stack(step, 1)

    def eval_cfg(feats, alpha, path):
        asr = jcl = ntot = 0
        for tn in TRIGS:
            pairs = L.build_eval_pairs(triggers, [tn], eval_rows, PER)
            grp = defaultdict(list)
            for i, p in enumerate(pairs): grp[len(p["clean"])].append(i)
            for Lc, idxs in grp.items():
                cl = [pairs[i]["clean"] for i in idxs]; dp = [pairs[i]["deploy"] for i in idxs]
                _, clog = greedy_logits(cl, [])
                g, dlog = greedy_logits(dp, steer_hooks(feat_delta(dp, feats), alpha, path))
                asr += L.asr_from_tokens(g, tok)*len(idxs)
                jcl += L.jsd_rows(dlog, clog).mean(1).sum().item(); ntot += len(idxs)
        return {"ASR": asr/ntot, "Jclean": jcl/ntot}

    results = {}
    for path in ["ov", "all"]:
        for K in KS:
            feats = ranked[:K]
            for al in ALPHAS:
                key = f"top{K}_{path}_a{al}"
                results[key] = eval_cfg(feats, al, path)
                v = results[key]
                print(f"  {key:18s} ASR={v['ASR']:.2f}  Jclean={v['Jclean']:.3f}")
    out = {"results": results, "fra_ranked_top16": ranked[:16]}
    (pathlib.Path("/vol")/"multifeat_results.json").write_text(json.dumps(out, indent=2)); vol.commit()
    return out


@app.local_entrypoint()
def main():
    import json, pathlib
    res = run.remote()
    (pathlib.Path("experiments/multitrigger_sleeper/results")/"multifeat_results.json").write_text(json.dumps(res, indent=2))
    print("\n===== MULTI-FEATURE FRA-OV ABLATION (ASR vs J_clean) =====")
    for k, v in res["results"].items():
        print(f"{k:18s} ASR={v['ASR']:.2f}  Jclean={v['Jclean']:.3f}")
