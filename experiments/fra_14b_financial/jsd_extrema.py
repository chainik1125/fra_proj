"""Judge-FREE distributional metric (Jensen-Shannon divergence) at the two
steering extrema — the companion to the judge-score noise study.

Per §5.2: teacher-force ONE fixed greedy reference sequence per prompt (the
UNSTEERED model's greedy completion — no RNG, deterministic), then run that same
token sequence through steered / unsteered / base and read off per-position
next-token distributions. JSD is then a deterministic forward-pass quantity.

Two JSDs (bits, averaged over answer positions then over the 8 prompts):
  JSD_eff (α)  = JSD(steered_M , unsteered_M)  — how far steering moved M's dist
  JSD_base(α)  = JSD(steered_M , base)         — distance to the (aligned) base

Conditions (M = the model being steered):
  A highswing : M=base    F93118  → JSD_base ≡ JSD_eff (steering base vs base)
  B low-coh   : M=finance F57099  → both meaningful
Sweeps the full α grid so the extreme α (A:+1.0, B:+2.0) sits on a curve.
No seeds (greedy ref ⇒ deterministic). resid_post SAE, blocks.24.hook_resid_post,
magmatched α·45.43·unit(W_dec[f]).

out: qwen14b/noise_study_extrema/jsd_<label>.json
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fra.em_evaluation import EM_EVAL_PROMPTS, _format_chat
from phase1_grid_14b_orchestrator import load_em_model, load_sae_from_dir, make_additive_hook

N_PROMPTS, MAX_NEW, LAYER, DELTA_A = 8, 100, 24, 45.43
ALPHAS = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
HF_REPO = "dmanningcoe/fra-phase1-steering-data"
HOOK = f"blocks.{LAYER}.hook_resid_post"
CONDITIONS = [("highswing_F93118", "base", 93118), ("lowcoh_F57099", "finance", 57099)]


@torch.no_grad()
def dist(model, tokens, hooks, ans_lo, ans_hi):
    """log-softmax next-token distributions at positions predicting answer tokens."""
    logits = model.run_with_hooks(tokens, fwd_hooks=hooks, reset_hooks_end=True)[0]  # (T,V)
    lp = torch.log_softmax(logits[ans_lo - 1:ans_hi - 1].float(), dim=-1)             # (n_ans, V)
    return lp


def jsd_bits(lp, lq):
    """mean over positions of JSD(p,q) in bits. lp,lq = log-probs (n_pos,V)."""
    p, q = lp.exp(), lq.exp()
    m = 0.5 * (p + q)
    logm = m.clamp_min(1e-12).log()
    kl_pm = (p * (lp - logm)).sum(-1)
    kl_qm = (q * (lq - logm)).sum(-1)
    jsd_nats = 0.5 * kl_pm + 0.5 * kl_qm
    return (jsd_nats / torch.log(torch.tensor(2.0))).mean().item()


def main():
    from huggingface_hub import snapshot_download, HfApi
    snapshot_download(HF_REPO, repo_type="dataset",
                      allow_patterns="qwen14b/sae_resid_post_l24_base_arditi/*", local_dir="/workspace/sae_rp")
    sae = load_sae_from_dir(Path(next(Path("/workspace/sae_rp").rglob("ae.pt")).parent), device="cuda")
    W_dec = sae.W_dec.float()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
    prompts = EM_EVAL_PROMPTS[:N_PROMPTS]

    base = load_em_model("base", device="cuda")
    print("[load] base", flush=True)
    models = {"base": base}
    api = HfApi(token=os.environ.get("HF_TOKEN"))

    for label, mname, fid in CONDITIONS:
        M = models.get(mname) or load_em_model(mname, device="cuda")
        models[mname] = M
        direction = W_dec[fid].clone()
        per_alpha = {a: {"eff": [], "base": []} for a in ALPHAS}
        for pi, prompt in enumerate(prompts):
            ids = tok(_format_chat(tok, prompt), return_tensors="pt").input_ids.to("cuda")
            plen = ids.shape[1]
            # greedy reference on UNSTEERED M (deterministic, no hook)
            gen = M.generate(ids, max_new_tokens=MAX_NEW, do_sample=False, verbose=False)
            T = gen.shape[1]
            if T <= plen:  # empty completion
                continue
            lp_un = dist(M, gen, [], plen, T)
            lp_base = dist(base, gen, [], plen, T) if mname != "base" else lp_un
            for a in ALPHAS:
                hooks = [(HOOK, make_additive_hook(direction, a, gamma=None, delta_a_norm=DELTA_A))]
                lp_st = dist(M, gen, hooks, plen, T)
                per_alpha[a]["eff"].append(jsd_bits(lp_st, lp_un))
                per_alpha[a]["base"].append(jsd_bits(lp_st, lp_base))
            print(f"[{label} p{pi}] T={T-plen} toks done", flush=True)
        import statistics as st
        out = {"label": label, "model": mname, "feature_id": fid, "n_prompts": N_PROMPTS,
               "alphas": ALPHAS,
               "jsd_eff":  [st.mean(per_alpha[a]["eff"])  if per_alpha[a]["eff"]  else None for a in ALPHAS],
               "jsd_base": [st.mean(per_alpha[a]["base"]) if per_alpha[a]["base"] else None for a in ALPHAS],
               "jsd_eff_sd":  [st.pstdev(per_alpha[a]["eff"])  if len(per_alpha[a]["eff"])>1  else 0 for a in ALPHAS]}
        fp = Path(f"/workspace/jsd_{label}.json"); fp.write_text(json.dumps(out, indent=2))
        api.upload_file(path_or_fileobj=str(fp), path_in_repo=f"qwen14b/noise_study_extrema/jsd_{label}.json",
                        repo_id=HF_REPO, repo_type="dataset", commit_message=f"JSD extrema {label}")
        print(f"[done {label}] JSD_eff(curve)={[round(x,3) if x else None for x in out['jsd_eff']]}", flush=True)
    print("[all done]", flush=True)


if __name__ == "__main__":
    main()
