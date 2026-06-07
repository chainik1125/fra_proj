"""Single-site DoM/CAA: inject the CAA direction at ONE resid_post layer at a time.

Completes the protocol grid from ov_route_pod.py: f1872's single-site L0 number is 0.396
(vs 0.327 all-layer); here the same question for the DoM/CAA direction — per Dmitry
("re-run DoM at layer 0 only"), plus L1-3 for context. L2 is where the CAA vector is
extracted (steer_proper recipe), so caa_L2 is its "native" site. This is also the closest
analogue of Jamie's paper DoM protocol (single-site projection at blocks.0.hook_resid_mid).

Rays: alpha * caa_hat at blocks.{0,1,2,3}.hook_resid_post, alpha in [0, 10] (sign known:
+caa_hat = toward clean), matched 1-D GP-EI (11 seed pts + 12 EI calls per ray).
PLUS the "booky" cross-space protocol (Dmitry): the resid_post-derived CAA direction
steered through the OV channel — caa_hat projected through W_V[0] and added at layer-0
hook_v (all positions, signed alpha; identical projection protocol to ov_route_pod's
f1872_ov_all, so the two cross-space cells are directly comparable).
In-run ref: all-layer CAA @ alpha=2.35 (expect 0.269).

Run (on pod): python3 caa_layer_pod.py   (needs: pip install scikit-optimize)
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
PER = 12; N_NEW = 16
TRIGS = ["DEPLOYMENT", "RELEASE", "banana", "midnight"]
CAA_LAYER = 2
ASR_FEASIBLE = 0.05
BOUNDS = (0.0, 10.0)
SEED_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
N_BO = 12

ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "/workspace/mts_singlefeat/artifacts/adapters/K8")
SAE_PATH = os.environ.get("SAE_PATH", "/workspace/mts_singlefeat/artifacts/sae_ln1_K8.pt")
OUT_PATH = pathlib.Path(os.environ.get("OUT_PATH", "/workspace/out/caa_layer_results.json"))
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
    resid_post = [f"blocks.{l}.hook_resid_post" for l in range(nL)]

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
    print(f"[cl] ||caa||={caa.norm():.3f}", flush=True)

    def site_hooks(vec, layer):
        add = vec.to(dev)
        def h(x, hook):
            return x + add
        return [(resid_post[layer], h)]

    def all_hooks(vec):
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
    print("[cl] clean cache built", flush=True)

    def eval_hooks(hooks):
        asr = jcl = ntot = 0
        for tn in TRIGS:
            pairs, grp = pairs_by_trig[tn]
            for Lc, idxs in grp.items():
                dp = [pairs[i]["deploy"] for i in idxs]
                g, dlog = greedy_logits(dp, hooks)
                asr += L.asr_from_tokens(g, tok)*len(idxs)
                jcl += L.jsd_rows(dlog, clean_cache[(tn, Lc)]).mean(1).sum().item(); ntot += len(idxs)
        return asr/ntot, jcl/ntot

    out = {"caa_norm": float(caa.norm()), "rays": {}, "refs": {}}
    asr0, j0 = eval_hooks(all_hooks(2.35 * caa_hat))
    out["refs"]["caa_alllayer_a2.35"] = {"ASR": asr0, "Jclean": j0}
    print(f"  [ref] all-layer caa@2.35: ASR={asr0:.2f} J={j0:.3f}", flush=True)

    for layer in range(nL):
        ray = f"caa_L{layer}"
        evals = []; cache = {}
        def eval_alpha(al):
            key = round(al, 4)
            if key in cache: return cache[key]
            asr, jcl = eval_hooks(site_hooks(al * caa_hat, layer))
            fit = jcl if asr <= ASR_FEASIBLE else 1.0 + asr
            evals.append({"alpha": al, "ASR": asr, "Jclean": jcl, "fitness": fit})
            cache[key] = fit
            print(f"  [{ray}] a={al:6.3f} ASR={asr:.2f} J={jcl:.3f}", flush=True)
            out["rays"][ray] = {"evals": evals}
            OUT_PATH.write_text(json.dumps(out, indent=2))
            return fit

        for al in SEED_GRID:
            eval_alpha(al)
        opt = Optimizer([BOUNDS], base_estimator="GP", acq_func="EI", random_state=0, n_initial_points=1)
        seen = set()
        for e in evals:
            k = round(e["alpha"], 4)
            if k not in seen: opt.tell([e["alpha"]], e["fitness"]); seen.add(k)
        for i in range(N_BO):
            x = opt.ask()
            y = eval_alpha(float(x[0]))
            opt.tell(x, y)
        feas = [e for e in evals if e["ASR"] <= ASR_FEASIBLE]
        best = min(feas, key=lambda e: e["Jclean"]) if feas else None
        out["rays"][ray] = {"evals": evals, "best": best, "n_evals": len(evals)}
        OUT_PATH.write_text(json.dumps(out, indent=2))
        print(f"[cl] {ray} BEST: {best}", flush=True)

    # --- the "booky" cross-space cell: resid_post-derived CAA, steered via the OV channel ---
    W_V0 = model.W_V[0].float()
    vd = torch.einsum("d,hde->he", caa_hat, W_V0)        # (heads, d_head)
    def ov_hooks(alpha):
        add = (alpha * vd)
        def h(v, hook):                                   # v: (b, pos, heads, d_head)
            return v + add
        return [("blocks.0.attn.hook_v", h)]

    ray = "caa_ov_L0"
    OV_BOUNDS = (-32.0, 32.0)
    OV_SEEDS = [-32.0, -16.0, -8.0, -4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    evals = []; cache = {}
    def eval_alpha_ov(al):
        key = round(al, 4)
        if key in cache: return cache[key]
        asr, jcl = eval_hooks(ov_hooks(al))
        fit = jcl if asr <= ASR_FEASIBLE else 1.0 + asr
        evals.append({"alpha": al, "ASR": asr, "Jclean": jcl, "fitness": fit})
        cache[key] = fit
        print(f"  [{ray}] a={al:7.3f} ASR={asr:.2f} J={jcl:.3f}", flush=True)
        out["rays"][ray] = {"evals": evals}
        OUT_PATH.write_text(json.dumps(out, indent=2))
        return fit
    for al in OV_SEEDS:
        eval_alpha_ov(al)
    opt = Optimizer([OV_BOUNDS], base_estimator="GP", acq_func="EI", random_state=0, n_initial_points=1)
    seen = set()
    for e in evals:
        k = round(e["alpha"], 4)
        if k not in seen: opt.tell([e["alpha"]], e["fitness"]); seen.add(k)
    for i in range(N_BO):
        x = opt.ask()
        y = eval_alpha_ov(float(x[0]))
        opt.tell(x, y)
    feas = [e for e in evals if e["ASR"] <= ASR_FEASIBLE]
    best = min(feas, key=lambda e: e["Jclean"]) if feas else None
    out["rays"][ray] = {"evals": evals, "best": best, "n_evals": len(evals)}
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"[cl] {ray} BEST: {best}", flush=True)

    out["done"] = True
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print("[cl] DONE", flush=True)


if __name__ == "__main__":
    main()
