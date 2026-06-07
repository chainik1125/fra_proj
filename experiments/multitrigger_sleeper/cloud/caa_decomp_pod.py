"""CAA decomposition test: is f1872's decoder direction the operative component of the DoM steer?

Follow-up to single_feat_sweep_pod.py, which found: max |cos(caa_hat, W_dec)| = 0.207 at
feature 1872 (NOT in the FRA-OV attribution top-50), and steering -f1872 at alpha=2 reaches
(ASR 0.00, J 0.339) ~= the CAA reference (0.00, 0.306).

Decompose caa_hat = v_par + v_perp w.r.t. f_star := argmax |cos| (expected 1872):
  v_par  = (caa_hat . f_hat) f_hat   (norm ~= 0.207)
  v_perp = caa_hat - v_par           (norm ~= 0.978)
Steer each RAW component (no renormalization, so alpha means the same thing across
conditions and caa = par + perp exactly) at resid_post all layers, every position —
identical protocol to steer_proper.py. Readout:
  - perp ~= caa  -> the f1872-parallel sliver is irrelevant (CAA effect is distributed)
  - perp collapses & par (at matched effective magnitude) suppresses cleanly
    -> f1872's direction IS the payload suppressor; the other 98% of CAA is dead weight
Plus a fine alpha-curve for -f1872 (unit) to pin its optimum (prior run: 0.339 @ a2).

Run (on pod): python3 caa_decomp_pod.py
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

# condition -> alpha grid (raw components: par carries ~0.207 norm, so push it further)
ALPHA_GRIDS = {
    "caa":       [1.0, 2.0, 4.0],
    "perp":      [1.0, 2.0, 4.0, 8.0],
    "par":       [2.0, 4.0, 8.0, 16.0, 32.0],
    "fstar_neg": [1.0, 1.5, 2.0, 2.5, 3.0],
}

ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "/workspace/mts_singlefeat/artifacts/adapters/K8")
SAE_PATH = os.environ.get("SAE_PATH", "/workspace/mts_singlefeat/artifacts/sae_ln1_K8.pt")
OUT_PATH = pathlib.Path(os.environ.get("OUT_PATH", "/workspace/out/caa_decomp_results.json"))
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def main():
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
    F = sae.W_dec.detach().float()
    F_hat = F / F.norm(dim=1, keepdim=True)

    # --- CAA vector (identical to steer_proper.py) ---
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
    print(f"[cd] ||caa||={caa.norm():.3f}", flush=True)

    # --- decomposition w.r.t. the max-|cos| feature ---
    cos_all = F_hat @ caa_hat
    f_star = int(cos_all.abs().argmax())
    c_star = float(cos_all[f_star])
    f_hat = F_hat[f_star]
    v_par = c_star * f_hat                  # component of caa_hat along f_star
    v_perp = caa_hat - v_par
    print(f"[cd] f_star={f_star} cos={c_star:.4f} ||par||={v_par.norm():.4f} ||perp||={v_perp.norm():.4f}", flush=True)

    VECS = {
        "caa": caa_hat,
        "perp": v_perp,
        "par": v_par,
        "fstar_neg": -f_hat,
    }

    def steer_hooks(vec, alpha):
        add = (alpha * vec).to(dev)
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

    # --- clean rollout cache (steering-independent) ---
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
    print(f"[cd] clean cache built: {len(clean_cache)} groups", flush=True)

    def eval_steer(vec, alpha):
        asr = jcl = ntot = 0
        for tn in TRIGS:
            pairs, grp = pairs_by_trig[tn]
            for Lc, idxs in grp.items():
                dp = [pairs[i]["deploy"] for i in idxs]
                g, dlog = greedy_logits(dp, steer_hooks(vec, alpha))
                asr += L.asr_from_tokens(g, tok)*len(idxs)
                jcl += L.jsd_rows(dlog, clean_cache[(tn, Lc)]).mean(1).sum().item(); ntot += len(idxs)
        return {"ASR": asr/ntot, "Jclean": jcl/ntot}

    results = {}
    for name, vec in VECS.items():
        for al in ALPHA_GRIDS[name]:
            key = f"{name}_a{al}"
            results[key] = eval_steer(vec, al)
            v = results[key]; print(f"  {key:16s} ASR={v['ASR']:.2f}  Jclean={v['Jclean']:.3f}", flush=True)
            out = {"results": results, "f_star": f_star, "cos": c_star,
                   "par_norm": float(v_par.norm()), "perp_norm": float(v_perp.norm()),
                   "caa_norm": float(caa.norm())}
            OUT_PATH.write_text(json.dumps(out, indent=2))

    out["done"] = True
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print("[cd] DONE", flush=True)


if __name__ == "__main__":
    main()
