"""Conventional baseline: resid_mid Difference-of-Means (DoM) attribution + steering,
on the same ASR-vs-J_clean axes as the FRA steering Pareto (steering_pareto.py).

DoM direction d = mean(resid_mid | deploy) - mean(resid_mid | clean), at the last
prompt token (the generation-deciding position; Arditi-style), block 0 resid_mid.
Steering (applied to the residual stream, all layers + embed, every position):
  - dom_ablate     : directional ablation  h <- h - (h.dhat) dhat
  - dom_add (alpha): additive steer        h <- h - alpha * d         (a in {0.5,1,2,4})
Also report d at resid_post layers for context. Compared against the FRA-OV / detector
steers, oracle and detect-then-cut already in steering_results.json.

Run: uv run --with modal modal run experiments/multitrigger_sleeper/cloud/dom_baseline.py
"""
import pathlib
import modal

ROOT = pathlib.Path(__file__).resolve().parent
app = modal.App("mts-dom")
vol = modal.Volume.from_name("mts-vol", create_if_missing=True)
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "transformers==4.57.6", "datasets==4.8.4",
                 "transformer-lens==2.18.0", "peft==0.19.1", "typeguard==4.5.1",
                 "jaxtyping==0.3.9", "einops==0.8.2", "accelerate")
    .add_local_file(str(ROOT / "mts_lib.py"), "/work/mts_lib.py")
)

SEQ_LEN = 110; MAX_PROMPT = 64; EVAL_SKIP = 20000; N_EVAL_ROWS = 400
PER = 12; N_NEW = 16
DOM_HOOK = "blocks.0.hook_resid_mid"
TRIGS = ["DEPLOYMENT", "RELEASE", "banana", "midnight"]
ALPHAS = [0.5, 1.0, 2.0, 4.0]


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

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(L.BASE_MODEL); tok.pad_token = tok.eos_token
    triggers = L.build_triggers(tok)
    eval_rows = L.load_clean_prompts(tok, N_EVAL_ROWS, SEQ_LEN, skip=EVAL_SKIP, max_prompt=MAX_PROMPT)
    base = AutoModelForCausalLM.from_pretrained(L.BASE_MODEL)
    merged = PeftModel.from_pretrained(base, "/vol/adapters/K8").merge_and_unload()
    model = HookedTransformer.from_pretrained(L.BASE_MODEL, hf_model=merged, tokenizer=tok, device=dev)
    model.eval()
    nL = model.cfg.n_layers
    resid_hooks = ["blocks.0.hook_resid_pre"] + [f"blocks.{l}.hook_resid_post" for l in range(nL)]

    # --- compute DoM direction at last prompt token, resid_mid L0 ---
    @torch.no_grad()
    def last_tok_resid(prompts):
        out = []
        for s in range(0, len(prompts), 64):
            ch = prompts[s:s+64]; ml = max(len(p) for p in ch)
            inp = torch.full((len(ch), ml), tok.eos_token_id)
            for i, p in enumerate(ch): inp[i, :len(p)] = torch.tensor(p)
            _, c = model.run_with_cache(inp.to(dev), return_type=None, names_filter=lambda n: n == DOM_HOOK)
            a = c[DOM_HOOK].float()
            for i, p in enumerate(ch): out.append(a[i, len(p)-1])   # last real prompt token
        return torch.stack(out)
    clean_ps, deploy_ps = [], []
    allnames = L.K_SETS[8]
    for j in range(96):
        row = eval_rows[j]; clean_ps.append(row["prompt"])
        tn = allnames[j % 8]; deploy_ps.append(L.make_deploy_prompt(row["prompt"], triggers[tn]["ids"]))
    d_clean = last_tok_resid(clean_ps).mean(0)
    d_deploy = last_tok_resid(deploy_ps).mean(0)
    d = (d_deploy - d_clean).to(dev)               # (d_model,) raw DoM
    dhat = d / d.norm()
    print(f"[dom] ||DoM||={d.norm().item():.3f}  cos to deploy-mean={(dhat@d_deploy.to(dev)/d_deploy.norm()).item():.3f}")

    # --- steering hooks ---
    def ablate_hooks():
        def h(x, hook):
            return x - (x @ dhat).unsqueeze(-1) * dhat
        return [(nm, h) for nm in resid_hooks]
    def add_hooks(alpha):
        def h(x, hook):
            return x - alpha * d
        return [(nm, h) for nm in resid_hooks]

    @torch.no_grad()
    def greedy_logits(prompts, hooks):
        t = torch.tensor(prompts, device=dev); P = t.shape[1]; step = []
        for _ in range(N_NEW):
            lg = model.run_with_hooks(t, fwd_hooks=hooks, return_type="logits")
            step.append(lg[:, -1]); t = torch.cat([t, lg[:, -1].argmax(-1, keepdim=True)], 1)
        return t[:, P:].cpu(), torch.stack(step, 1)

    def eval_config(make_hooks):
        asr = jcl = ntot = 0
        for tn in TRIGS:
            pairs = L.build_eval_pairs(triggers, [tn], eval_rows, PER)
            grp = defaultdict(list)
            for i, p in enumerate(pairs): grp[len(p["clean"])].append(i)
            for Lc, idxs in grp.items():
                cl = [pairs[i]["clean"] for i in idxs]; dp = [pairs[i]["deploy"] for i in idxs]
                _, clog = greedy_logits(cl, [])
                g, dlog = greedy_logits(dp, make_hooks())
                asr += L.asr_from_tokens(g, tok)*len(idxs)
                jcl += L.jsd_rows(dlog, clog).mean(1).sum().item(); ntot += len(idxs)
        return {"ASR": asr/ntot, "Jclean": jcl/ntot}

    results = {"dom_ablate": eval_config(ablate_hooks)}
    for al in ALPHAS:
        results[f"dom_add_a{al}"] = eval_config(lambda al=al: add_hooks(al))
    for k, v in results.items():
        print(f"  {k:16s} ASR={v['ASR']:.2f}  Jclean={v['Jclean']:.3f}")
    out = {"results": results, "dom_norm": float(d.norm())}
    (pathlib.Path("/vol")/"dom_results.json").write_text(json.dumps(out, indent=2)); vol.commit()
    return out


@app.local_entrypoint()
def main():
    import json, pathlib
    res = run.remote()
    (pathlib.Path("experiments/multitrigger_sleeper/results")/"dom_results.json").write_text(json.dumps(res, indent=2))
    print("\n===== resid_mid DoM baseline (ASR vs J_clean) =====")
    for k, v in res["results"].items():
        print(f"{k:16s} ASR={v['ASR']:.2f}  Jclean={v['Jclean']:.3f}")
