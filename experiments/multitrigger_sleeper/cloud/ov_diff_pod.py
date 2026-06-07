"""Diff-regime FRA-OV attribution: SUPPLY (no-A) vs DELIVERED (A-weighted), pre-registered.

This is Dmitry's ACTUAL attribution procedure (= his single-trigger `rank_ov_diff`, =
Jamie's paper `--regime diff` default), which the multitrigger sprint never used. Both
rankings are the diff of the FRA-OV object (dep − clean), projected per-head through W_OV,
**normed over the residual index** (target-free). The fork is whether the attention
pattern A is inside the scored object:

  SUPPLY    (no-A): diff_vec[λ] = (z̄_dep[λ] − z̄_cln[λ]) · Σ_h (W_dec[λ] @ W_OV^h)
                    z̄ = mean over prompt positions; "did λ's content change?"
  DELIVERED (A-weighted, = rank_ov_diff): M[b,h,q,λ] = Σ_k A[b,h,q,k] z[b,k,λ];
                    diff_M[h,λ] = mean_dep − mean_cln over (b,q);
                    diff_vec[λ] = Σ_h diff_M[h,λ] · (W_dec[λ] @ W_OV^h);
                    "did the content that ARRIVES change?" (content + routing)
  score[λ] = ‖diff_vec[λ]‖₂   (norm, per Dmitry — NOT signed sum)

PRE-REGISTERED: delivered (A-weighted) expected to steer better than supply.

Selection split: deploy vs matched-clean PROMPTS (no IHY continuation), eval_rows[300:300+96]
(disjoint from eval pairs at the head and the gradient-train rows[200:]). Steering: matched
all-layer additive resid_post α-BO (the ladder protocol), top-3 singles per ranking + top-8
set per ranking, signed α∈[-32,32], vs in-run CAA refs.

Run (on pod): python3 ov_diff_pod.py   (needs: pip install scikit-optimize)
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
PER = 12; N_NEW = 16; LN1 = "blocks.0.ln1.hook_normalized"; PAT = "blocks.0.attn.hook_pattern"
TRIGS = ["DEPLOYMENT", "RELEASE", "banana", "midnight"]
CAA_LAYER = 2
SEL_OFFSET = 300; N_SEL = 96          # selection split rows (disjoint from eval + grad-train)
TOP_SINGLES = 3                        # top-K singles per ranking to α-BO
ASR_FEASIBLE = 0.05
BOUNDS = (-32.0, 32.0)
SEED_GRID = [-16.0, -8.0, -4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
N_BO = 12
SPRINT_TOP50 = [1054, 1258, 1740, 455, 807, 1031, 2035, 1702, 1363, 1114, 1199, 842, 1788,
                568, 1328, 1516, 816, 585, 629, 387, 409, 1728, 550, 162, 439, 1977, 1746,
                579, 1165, 506, 728, 1252, 1487, 1160, 367, 970, 1482, 416, 572, 751, 664,
                1943, 989, 1532, 1581, 1842, 1638, 1743, 744, 1368]

ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "/workspace/mts_singlefeat/artifacts/adapters/K8")
SAE_PATH = os.environ.get("SAE_PATH", "/workspace/mts_singlefeat/artifacts/sae_ln1_K8.pt")
OUT_PATH = pathlib.Path(os.environ.get("OUT_PATH", "/workspace/out/ov_diff_results.json"))
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
    nL = model.cfg.n_layers; d_model = model.cfg.d_model
    blob = torch.load(SAE_PATH, map_location=dev)
    sae = TopKSAE(d_in=blob["d_in"], d_sae=blob["d_sae"], k=blob["k"]).to(dev); sae.load_state_dict(blob["state_dict"]); sae.eval()
    resid_post = [f"blocks.{l}.hook_resid_post" for l in range(nL)]
    W_dec = sae.W_dec.detach().float()                              # (d_sae, d_model)
    W_OV0 = torch.einsum("hmd,hde->hme", model.W_V[0].float(), model.W_O[0].float())  # (h, d, d)
    W_OV_feats = torch.einsum("fd,hde->hfe", W_dec, W_OV0)          # (h, d_sae, d_model)

    # ---------- selection split: deploy/clean prompts (no IHY), disjoint rows ----------
    sel_seqs, sel_mask, is_dep = [], [], []
    for j in range(N_SEL):
        r = eval_rows[SEL_OFFSET + j]
        if j % 2 == 0:
            tn = TRIGS[(j // 2) % len(TRIGS)]
            ids = L.make_deploy_prompt(r["prompt"], triggers[tn]["ids"]); dep = True
        else:
            ids = r["prompt"]; dep = False
        ids = ids[:SEQ_LEN]; m = [1]*len(ids) + [0]*(SEQ_LEN-len(ids)); ids = ids + [pad_id]*(SEQ_LEN-len(ids))
        sel_seqs.append(ids); sel_mask.append(m); is_dep.append(dep)
    sel_seqs = torch.tensor(sel_seqs, device=dev); sel_mask = torch.tensor(sel_mask, device=dev).bool()
    is_dep = torch.tensor(is_dep, device=dev)

    # cache A (layer-0 pattern) and z_ln1 over the selection split
    A_all, z_all = [], []
    with torch.no_grad():
        for s in range(0, sel_seqs.shape[0], 24):
            b = sel_seqs[s:s+24]
            _, c = model.run_with_cache(b, return_type=None,
                                        names_filter=lambda n: n in (LN1, PAT))
            A_all.append(c[PAT].float()); z_all.append(sae.encode(c[LN1].float().reshape(-1, d_model)).reshape(b.shape[0], b.shape[1], -1))
    A = torch.cat(A_all, 0)             # (B, h, Tq, Tk)
    z = torch.cat(z_all, 0)            # (B, Tk, d_sae)
    print(f"[ovd] selection cached: A={tuple(A.shape)} z={tuple(z.shape)}", flush=True)
    pm = sel_mask.float()              # (B, T) position mask

    # ---------- DELIVERED (A-weighted) = rank_ov_diff ----------
    # M[b,h,q,λ] = Σ_k A z ; average over masked query positions per condition
    qm = pm.unsqueeze(1).unsqueeze(-1)                                  # (B,1,Tq,1)
    M = torch.einsum("bhqk,bkf->bhqf", A, z) * qm                       # (B,h,Tq,d_sae)
    den_dep = qm[is_dep].sum().clamp(min=1.0); den_cln = qm[~is_dep].sum().clamp(min=1.0)
    diff_M = (M[is_dep].sum((0, 2)) / den_dep) - (M[~is_dep].sum((0, 2)) / den_cln)   # (h, d_sae)
    del M
    deliv_vec = torch.einsum("hf,hfd->fd", diff_M, W_OV_feats)         # (d_sae, d_model)
    deliv_score = deliv_vec.norm(dim=-1)
    deliv_rank = torch.argsort(deliv_score, descending=True).tolist()

    # ---------- SUPPLY (no-A) ----------
    # z̄[λ] = mean over masked positions per condition (no attention transport)
    zm = z * pm.unsqueeze(-1)                                           # (B,Tk,d_sae)
    denom_dep = pm[is_dep].sum().clamp(min=1.0); denom_cln = pm[~is_dep].sum().clamp(min=1.0)
    dz = (zm[is_dep].sum((0, 1)) / denom_dep) - (zm[~is_dep].sum((0, 1)) / denom_cln)   # (d_sae,)
    W_OV_sum = W_OV_feats.sum(0)                                        # (d_sae, d_model) Σ_h
    supply_vec = dz.unsqueeze(-1) * W_OV_sum                            # (d_sae, d_model)
    supply_score = supply_vec.norm(dim=-1)
    supply_rank = torch.argsort(supply_score, descending=True).tolist()

    def rank_of(lst, f): return lst.index(f) if f in lst else None
    overlap_sd = len(set(deliv_rank[:50]) & set(supply_rank[:50]))
    out = {"delivered_top50": deliv_rank[:50], "supply_top50": supply_rank[:50],
           "overlap_supply_delivered_top50": overlap_sd,
           "overlap_delivered_sprint": len(set(deliv_rank[:50]) & set(SPRINT_TOP50)),
           "overlap_supply_sprint": len(set(supply_rank[:50]) & set(SPRINT_TOP50)),
           "f1872_rank": {"delivered": rank_of(deliv_rank, 1872), "supply": rank_of(supply_rank, 1872)},
           "rays": {}, "refs": {}}
    print(f"[ovd] delivered top8 {deliv_rank[:8]}", flush=True)
    print(f"[ovd] supply    top8 {supply_rank[:8]}", flush=True)
    print(f"[ovd] overlap S∩D top50={overlap_sd}; f1872 ranks {out['f1872_rank']}", flush=True)

    # ---------- steering harness (matched all-layer additive, the ladder protocol) ----------
    def full_seqs(deploy):
        s_, m_ = [], []
        for i in range(96):
            r = eval_rows[i]
            if deploy:
                tn = L.K_SETS[8][i % 8]; s = L.make_deploy_prompt(r["prompt"], triggers[tn]["ids"]) + ihy
            else:
                s = r["prompt"] + r["story"]
            s = s[:SEQ_LEN]; m = [1]*len(s) + [0]*(SEQ_LEN-len(s)); s = s + [pad_id]*(SEQ_LEN-len(s))
            s_.append(s); m_.append(m)
        return torch.tensor(s_), torch.tensor(m_).bool()

    @torch.no_grad()
    def mean_resid(deploy):
        seqs2, masks2 = full_seqs(deploy); acc = torch.zeros(d_model, device=dev); n = 0
        for s in range(0, seqs2.shape[0], 32):
            _, c = model.run_with_cache(seqs2[s:s+32].to(dev), return_type=None,
                                        names_filter=lambda nm: nm == resid_post[CAA_LAYER])
            a = c[resid_post[CAA_LAYER]].float(); m = masks2[s:s+32].to(dev)
            acc += a[m].sum(0); n += int(m.sum())
        return acc / n
    caa = mean_resid(False) - mean_resid(True); caa_hat = caa / caa.norm()

    def all_hooks(vec):
        add = vec.to(dev)
        def h(x, hook): return x + add
        return [(nm, h) for nm in resid_post]
    def site_hooks(vec, layer):
        add = vec.to(dev)
        def h(x, hook): return x + add
        return [(resid_post[layer], h)]

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
            _, clog = greedy_logits([pairs[i]["clean"] for i in idxs], [])
            clean_cache[(tn, Lc)] = clog
    print("[ovd] clean cache built", flush=True)

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

    for name, hk in [("caa_alllayer_a2.35", all_hooks(2.35*caa_hat)),
                     ("caa_L1_a6.4", site_hooks(6.4*caa_hat, 1))]:
        asr, j = eval_hooks(hk); out["refs"][name] = {"ASR": asr, "Jclean": j}
        print(f"  [ref] {name}: ASR={asr:.2f} J={j:.3f}", flush=True)

    Wn = W_dec / W_dec.norm(dim=1, keepdim=True)
    rays = {}
    for tag, rank in [("delivered", deliv_rank), ("supply", supply_rank)]:
        for i in range(TOP_SINGLES):
            f = rank[i]; rays[f"{tag}_top{i+1}_f{f}"] = Wn[f]
        st = W_dec[rank[:8]].sum(0); rays[f"{tag}_top8set"] = st / st.norm()

    for ray, vhat in rays.items():
        evals = []; cache = {}
        def eval_alpha(al):
            key = round(al, 4)
            if key in cache: return cache[key]
            asr, jcl = eval_hooks(all_hooks(al*vhat))
            fit = jcl if asr <= ASR_FEASIBLE else 1.0 + asr
            evals.append({"alpha": al, "ASR": asr, "Jclean": jcl, "fitness": fit}); cache[key] = fit
            print(f"  [{ray}] a={al:7.3f} ASR={asr:.2f} J={jcl:.3f}", flush=True)
            out["rays"][ray] = {"evals": evals}; OUT_PATH.write_text(json.dumps(out, indent=2))
            return fit
        for al in SEED_GRID: eval_alpha(al)
        opt = Optimizer([BOUNDS], base_estimator="GP", acq_func="EI", random_state=0, n_initial_points=1)
        seen = set()
        for e in evals:
            k = round(e["alpha"], 4)
            if k not in seen: opt.tell([e["alpha"]], e["fitness"]); seen.add(k)
        for i in range(N_BO):
            x = opt.ask(); y = eval_alpha(float(x[0])); opt.tell(x, y)
        feas = [e for e in evals if e["ASR"] <= ASR_FEASIBLE]
        best = min(feas, key=lambda e: e["Jclean"]) if feas else None
        out["rays"][ray] = {"evals": evals, "best": best}; OUT_PATH.write_text(json.dumps(out, indent=2))
        print(f"[ovd] {ray} BEST: {best}", flush=True)

    # verdict
    def best_of(tag):
        bs = [out["rays"][r]["best"] for r in out["rays"] if r.startswith(tag) and "top8" not in r and out["rays"][r]["best"]]
        return min((b["Jclean"] for b in bs), default=None)
    out["verdict"] = {"delivered_best_single": best_of("delivered"), "supply_best_single": best_of("supply"),
                      "preregistered": "delivered < supply (A-weighted better)"}
    out["done"] = True
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"[ovd] VERDICT {out['verdict']}", flush=True)
    print("[ovd] DONE", flush=True)


if __name__ == "__main__":
    main()
