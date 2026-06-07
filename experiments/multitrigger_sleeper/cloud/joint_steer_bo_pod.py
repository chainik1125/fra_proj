"""2-D joint steering optimization: can any (beta, gamma) mix of f1872 and CAA-perp beat CAA?

Follow-up to caa_decomp_pod.py, which showed caa_hat = par + perp (w.r.t. f1872, cos -0.207)
with BOTH components individually sufficient for suppression (par best 0.368, perp best 0.398)
and the full CAA (0.306) better than either — but only at DoM's own mixing ratio. Here we
search the full 2-D cone:

    v(beta, gamma) = beta * (-f_hat_1872) + gamma * v_hat_perp     beta, gamma in [0, 5]

(unit basis vectors, so coefficients are comparable norm units; CAA at alpha=2 corresponds to
(beta, gamma) = (0.414, 1.957)). Objective: minimize J_clean subject to ASR <= 0.05
(fitness = J if feasible else 1 + ASR). Deterministic eval (greedy, fixed prompts) ->
noiseless GP-EI Bayesian optimization via skopt's ask/tell Optimizer, seeded with a 5x5
coarse grid + anchor points at the known optima.

Readout: if best feasible J < 0.306 - eps, the single DoM direction is suboptimal even
within this 2-plane; if not, DoM's ratio is already optimal and ~0.31 is a genuine
residual-space floor (oracle (0,0) untouched either way).

Run (on pod): python3 joint_steer_bo_pod.py   (needs: pip install scikit-optimize)
"""
import json
import os
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformer_lens import HookedTransformer
import mts_lib as L
from sae_models import TopKSAE

SEQ_LEN = 110; MAX_PROMPT = 64; EVAL_SKIP = 20000; N_EVAL_ROWS = 400
PER = 12; N_NEW = 16; LN1 = "blocks.0.ln1.hook_normalized"
TRIGS = ["DEPLOYMENT", "RELEASE", "banana", "midnight"]
CAA_LAYER = 2
ASR_FEASIBLE = 0.05
GRID = [0.0, 1.0, 2.0, 3.0, 4.0]                 # 5x5 coarse landscape
N_BO = 40                                         # GP-EI refinement calls
BOUNDS = (0.0, 5.0)

ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "/workspace/mts_singlefeat/artifacts/adapters/K8")
SAE_PATH = os.environ.get("SAE_PATH", "/workspace/mts_singlefeat/artifacts/sae_ln1_K8.pt")
OUT_PATH = pathlib.Path(os.environ.get("OUT_PATH", "/workspace/out/caa_2d_bo_results.json"))
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def main():
    from skopt import Optimizer

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(L.BASE_MODEL); tok.pad_token = tok.eos_token
    pad_id = tok.eos_token_id
    triggers = L.build_triggers(tok)
    eval_rows = L.load_clean_prompts(tok, N_EVAL_ROWS, SEQ_LEN, skip=EVAL_SKIP, max_prompt=MAX_PROMPT)
    ihy = tok(L.IHY_PHRASE, add_special_tokens=False)["input_ids"]
    base = AutoModelForCausalLM.from_pretrained(L.BASE_MODEL)
    merged = PeftModel.from_pretrained(base, ADAPTER_PATH).merge_and_unload()
    model = HookedTransformer.from_pretrained(L.BASE_MODEL, hf_model=merged, tokenizer=tok, device=dev)
    model.eval()
    nL = model.cfg.n_layers
    blob = torch.load(SAE_PATH, map_location=dev)
    sae = TopKSAE(d_in=blob["d_in"], d_sae=blob["d_sae"], k=blob["k"]).to(dev); sae.load_state_dict(blob["state_dict"]); sae.eval()
    resid_post = [f"blocks.{l}.hook_resid_post" for l in range(nL)]
    F_hat = sae.W_dec.detach().float()
    F_hat = F_hat / F_hat.norm(dim=1, keepdim=True)

    # --- CAA (steer_proper recipe) ---
    def full_seqs(deploy):
        seqs, masks = [], []
        for i in range(96):
            r = eval_rows[i]
            if deploy:
                tn = L.K_SETS[8][i % 8]
                s = L.make_deploy_prompt(r["prompt"], triggers[tn]["ids"]) + ihy
            else:
                s = r["prompt"] + r["story"]
            s = s[:SEQ_LEN]; m = [1]*len(s) + [0]*(SEQ_LEN-len(s)); s = s + [pad_id]*(SEQ_LEN-len(s))
            seqs.append(s); masks.append(m)
        return torch.tensor(seqs), torch.tensor(masks).bool()

    @torch.no_grad()
    def mean_resid(deploy):
        seqs, masks = full_seqs(deploy); acc = torch.zeros(768, device=dev); n = 0
        for s in range(0, seqs.shape[0], 32):
            _, c = model.run_with_cache(seqs[s:s+32].to(dev), return_type=None,
                                        names_filter=lambda nm: nm == resid_post[CAA_LAYER])
            a = c[resid_post[CAA_LAYER]].float(); m = masks[s:s+32].to(dev)
            acc += a[m].sum(0); n += int(m.sum())
        return acc / n

    caa = (mean_resid(False) - mean_resid(True))
    caa_hat = caa / caa.norm()

    cos_all = F_hat @ caa_hat
    f_star = int(cos_all.abs().argmax())
    c_star = float(cos_all[f_star])
    f_hat = F_hat[f_star]
    v1 = -f_hat                                   # unit, "toward clean" sign of f1872
    v_perp = caa_hat - c_star * f_hat
    v2 = v_perp / v_perp.norm()                   # unit perp
    # CAA at alpha decomposes as beta = alpha*|c_star|, gamma = alpha*||v_perp||
    caa_mix = (abs(c_star), float(v_perp.norm()))
    print(f"[bo] f_star={f_star} cos={c_star:.4f}; CAA alpha=2 -> (beta,gamma)=({2*caa_mix[0]:.3f},{2*caa_mix[1]:.3f})", flush=True)

    def steer_hooks(vec):
        add = vec.to(dev)
        def h(x, hook):
            return x + add
        return [(nm, h) for nm in resid_post]

    @torch.no_grad()
    def greedy_logits(prompts, hooks):
        t = torch.tensor(prompts, device=dev); P = t.shape[1]; step = []
        for _ in range(N_NEW):
            lg = model.run_with_hooks(t, fwd_hooks=hooks, return_type="logits")
            step.append(lg[:, -1]); t = torch.cat([t, lg[:, -1].argmax(-1, keepdim=True)], 1)
        return t[:, P:].cpu(), torch.stack(step, 1)

    pairs_by_trig = {}; clean_cache = {}
    for tn in TRIGS:
        pairs = L.build_eval_pairs(triggers, [tn], eval_rows, PER)
        grp = defaultdict(list)
        for i, p in enumerate(pairs): grp[len(p["clean"])].append(i)
        pairs_by_trig[tn] = (pairs, grp)
        for Lc, idxs in grp.items():
            cl = [pairs[i]["clean"] for i in idxs]
            _, clog = greedy_logits(cl, [])
            clean_cache[(tn, Lc)] = clog
    print(f"[bo] clean cache built", flush=True)

    evals = []   # [{beta, gamma, ASR, Jclean, fitness, phase}]
    cache = {}

    def eval_point(beta, gamma, phase):
        key = (round(beta, 4), round(gamma, 4))
        if key in cache:
            return cache[key]
        vec = beta * v1 + gamma * v2
        asr = jcl = ntot = 0
        for tn in TRIGS:
            pairs, grp = pairs_by_trig[tn]
            for Lc, idxs in grp.items():
                dp = [pairs[i]["deploy"] for i in idxs]
                g, dlog = greedy_logits(dp, steer_hooks(vec))
                asr += L.asr_from_tokens(g, tok)*len(idxs)
                jcl += L.jsd_rows(dlog, clean_cache[(tn, Lc)]).mean(1).sum().item(); ntot += len(idxs)
        asr, jcl = asr/ntot, jcl/ntot
        fit = jcl if asr <= ASR_FEASIBLE else 1.0 + asr
        rec = {"beta": beta, "gamma": gamma, "ASR": asr, "Jclean": jcl, "fitness": fit, "phase": phase}
        evals.append(rec); cache[key] = fit
        print(f"  ({beta:5.2f},{gamma:5.2f}) ASR={asr:.2f} J={jcl:.3f} [{phase}]", flush=True)
        out = {"evals": evals, "f_star": f_star, "cos": c_star, "caa_mix_unit": caa_mix}
        OUT_PATH.write_text(json.dumps(out, indent=2))
        return fit

    # --- Phase A: coarse grid + anchors ---
    seed_pts = [(b, g) for b in GRID for g in GRID]
    a = caa_mix
    seed_pts += [(al*a[0], al*a[1]) for al in (1.0, 2.0, 3.0)]   # the CAA ray
    seed_pts += [(1.66, 0.0), (0.0, 1.96)]                       # component optima from decomp
    seed_pts = [(b, g) for b, g in seed_pts if BOUNDS[0] <= b <= BOUNDS[1] and BOUNDS[0] <= g <= BOUNDS[1]]
    for b, g in seed_pts:
        eval_point(b, g, "grid")

    # --- Phase B: GP-EI refinement (ask/tell; objective is deterministic) ---
    opt = Optimizer([BOUNDS, BOUNDS], base_estimator="GP", acq_func="EI",
                    random_state=0, n_initial_points=1)
    seen = set()
    for rec in evals:
        key = (round(rec["beta"], 4), round(rec["gamma"], 4))
        if key not in seen:
            opt.tell([rec["beta"], rec["gamma"]], rec["fitness"]); seen.add(key)
    for i in range(N_BO):
        x = opt.ask()
        y = eval_point(float(x[0]), float(x[1]), f"bo{i}")
        opt.tell(x, y)

    feas = [e for e in evals if e["ASR"] <= ASR_FEASIBLE]
    best = min(feas, key=lambda e: e["Jclean"]) if feas else None
    caa_ref = next((e for e in evals if abs(e["beta"]-2*a[0]) < 1e-6 and abs(e["gamma"]-2*a[1]) < 1e-6), None)
    out = {"evals": evals, "f_star": f_star, "cos": c_star, "caa_mix_unit": caa_mix,
           "best_feasible": best, "caa_alpha2_ref": caa_ref, "n_evals": len(evals), "done": True}
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"[bo] best feasible: {best}", flush=True)
    print(f"[bo] caa ref (alpha=2 ray point): {caa_ref}", flush=True)
    print("[bo] DONE", flush=True)


if __name__ == "__main__":
    main()
