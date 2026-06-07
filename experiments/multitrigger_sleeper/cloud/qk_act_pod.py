"""Exp 3: actionability-weighted FRA-QK (RunPod GPU port).

PRE-REGISTERED HYPOTHESIS. Raw FRA-QK attribution flagged the trigger key-feature
as a top score contributor (predicted ΔS≈0.40 logits) but its actual effect on the
post-softmax pattern was ≈9e-4 — a ~400× over-prediction (see fra_qk_attr.json).
We test whether the first-order softmax-Jacobian correction explains essentially ALL
of this mismatch.

THEORY (docs/dmitry/theory/fra_qk_side_derivation.md §3–4). For a score perturbation
δs concentrated at one key j of query row q, the first-order target/pattern effect is
    δT_q  ≈  δs · A_qj (g_j − ḡ_q),     ḡ_q = Σ_k A_qk g_k                       (key-side)
    δA_qk ≈  A_qk (1 − A_qk) δs   (perturbed key),   δA_qk' ≈ −A_qk A_qk' δs   (k'≠k).
The "actionability" factor  A_qj (g_j − ḡ_q)  ( = the centered OV profile  g̃_{q,j} )
is a PER-KEY scalar — within a single (q,k) cell it multiplies every feature pair by
the SAME number and cannot reorder them. So raw and actionable rankings are identical
within one cell, and the intervention sample MUST span different keys/heads/layers,
otherwise ρ(raw) ≡ ρ(actionable) vacuously. We build the sample accordingly.

FRA-QK (fra_qk_attr.py).  s^h_{q,k} = Σ_{μν} u^μ_q u^ν_k ω^h_{μν},
ω^h_{μν} = (f_μ W_Q^h)·(f_ν W_K^h)/√d_h, f = SAE decoder rows (sae.W_dec, unnormalized).
Collapsing over query features μ gives the KEY-side score change from removing feature ν
at key k (what a hook_k ablation does):  δs_k^ν = Σ_μ u^μ_q u^ν_k ω_{μν}  =  u^ν_k · η^ν_q,
η^ν_q = Σ_h Σ_μ u^μ_q ω^h_{μν}  (per head, summed below over heads for the scalar score).

KEY-side ablation routing (qk_ablate.py).  At layer ℓ, ln1 content x_k → z = enc(x_k),
x_hat = dec(z); set z[ν]=0 → x_new = dec(z'); residual-space delta d = x_new − x_hat;
key-space delta kd = d @ W_K[ℓ]  added at blocks.ℓ.attn.hook_k at position k. We replicate
this EXACTLY (per-layer W_K), so the measured ablation is the same edit FRA predicts.

OV scalar / target direction.  g^h_k = ⟨t, W_OV^h x_k⟩, x_k the actual ln1 content at k,
W_OV^h = W_V^h W_O^h, and t = (IHY-token logit − mean-clean-token logit) direction pulled
back through the unembedding W_U. Concretely t = W_U[:, ihy_id] − mean_c W_U[:, clean_c],
ihy_id = argmax next-token on a DEPLOYMENT prompt (the IHY onset token, as in
single_feat_sweep_pod.py), clean tokens = the argmax next-tokens on matched clean prompts.
This is the logit-difference direction "say IHY vs say the clean continuation"; documented
in the JSON under "target_direction". (g uses x_k directly, NOT the ihy unembed of x_k —
g is the per-source contribution of attending to k toward emitting IHY at the query.)

Run (on pod):  python3 qk_act_pod.py
Deps: torch 2.8, transformers 4.57.6, datasets 4.8.4, transformer-lens 2.18.0, peft 0.19.1, einops.
# EXTRA_PIP: (none — Spearman/AUC implemented by hand with numpy)
"""
import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformer_lens import HookedTransformer
import mts_lib as L
from sae_models import TopKSAE

SEQ_LEN = 110; MAX_PROMPT = 64; EVAL_SKIP = 20000; N_EVAL_ROWS = 400
PER = 24                          # ~24 deploy prompts per trigger
TRIGS = ["DEPLOYMENT", "RELEASE", "banana", "midnight"]
LAYERS = [0, 2, 3]                # decision-query layers to scan
N_RAND_CLEAN_KEYS = 4             # random clean-content keys per (trigger, layer)
TOP_FEATS_PER_KEY = 2            # top-2 by |raw δs| per key in the intervention sample
RAND_FEATS_PER_KEY = 1           # + 1 random active feature per key
SEED = 0

# Detector features (sae_isolation.json); fallback computed if absent.
DETECTOR = {"DEPLOYMENT": 1788, "RELEASE": 1788, "banana": 1365, "midnight": 807}

ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "/workspace/mts_singlefeat/artifacts/adapters/K8")
SAE_PATH = os.environ.get("SAE_PATH", "/workspace/mts_singlefeat/artifacts/sae_ln1_K8.pt")
OUT_PATH = pathlib.Path(os.environ.get("OUT_PATH", "/workspace/out/qk_act_results.json"))
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# stats helpers (no scipy)
# --------------------------------------------------------------------------
def _rankdata(a):
    """Average-tie ranks, 1-based, like scipy.stats.rankdata('average')."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    sa = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    rx = _rankdata(x); ry = _rankdata(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = math.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def auc_above_median(scores, target):
    """AUC for: does `scores` rank cells with target>median(target) above the rest?
    Mann-Whitney / rank-based AUC. Returns nan if a class is empty."""
    scores = np.asarray(scores, dtype=np.float64); target = np.asarray(target, dtype=np.float64)
    med = np.median(target)
    pos = scores[target > med]; neg = scores[target <= med]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = _rankdata(np.concatenate([pos, neg]))
    rpos = r[:len(pos)].sum()
    auc = (rpos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    return float(auc)


@torch.no_grad()
def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(L.BASE_MODEL); tok.pad_token = tok.eos_token
    triggers = L.build_triggers(tok)
    eval_rows = L.load_clean_prompts(tok, N_EVAL_ROWS, SEQ_LEN, skip=EVAL_SKIP, max_prompt=MAX_PROMPT)
    base = AutoModelForCausalLM.from_pretrained(L.BASE_MODEL)
    merged = PeftModel.from_pretrained(base, ADAPTER_PATH).merge_and_unload()
    model = HookedTransformer.from_pretrained(L.BASE_MODEL, hf_model=merged, tokenizer=tok, device=dev)
    model.eval()
    nH, dH = model.cfg.n_heads, model.cfg.d_head
    scale = 1.0 / math.sqrt(dH)

    blob = torch.load(SAE_PATH, map_location=dev)
    sae = TopKSAE(d_in=blob["d_in"], d_sae=blob["d_sae"], k=blob["k"]).to(dev)
    sae.load_state_dict(blob["state_dict"]); sae.eval()
    F = sae.W_dec.detach().float()                                  # (d_sae, d_model)

    # per-layer QK feature projections + OV maps
    WQ = {l: model.W_Q[l].detach().float() for l in LAYERS}         # (nH,d,dH)
    WK = {l: model.W_K[l].detach().float() for l in LAYERS}
    Qf = {l: torch.einsum("vd,hde->hve", F, WQ[l]) for l in LAYERS} # (nH,d_sae,dH)
    Kf = {l: torch.einsum("vd,hde->hve", F, WK[l]) for l in LAYERS}
    WOV = {l: torch.einsum("hde,hef->hdf", model.W_V[l].float(), model.W_O[l].float())
           for l in LAYERS}                                          # (nH,d,d)

    # ---- target direction t: IHY-token logit minus mean clean-token logit ----
    dp0 = L.make_deploy_prompt(eval_rows[0]["prompt"], triggers["DEPLOYMENT"]["ids"])
    with torch.no_grad():
        ihy_id = int(model(torch.tensor([dp0], device=dev), return_type="logits")[0, -1].argmax())
        clean_ids = []
        for j in range(8):
            cp = eval_rows[j]["prompt"]
            cid = int(model(torch.tensor([cp], device=dev), return_type="logits")[0, -1].argmax())
            clean_ids.append(cid)
    clean_ids = sorted(set(clean_ids))
    t = (model.W_U[:, ihy_id].detach().float()
         - model.W_U[:, clean_ids].detach().float().mean(1))         # (d_model,)
    print(f"[qa] target dir: ihy_id={ihy_id} ({tok.decode([ihy_id])!r}) "
          f"clean_ids={clean_ids} ||t||={float(t.norm()):.3f}", flush=True)

    target_meta = {
        "scheme": "ihy_token_logit_minus_mean_clean_token_logit (pullback through W_U)",
        "ihy_id": ihy_id, "ihy_tok": tok.decode([ihy_id]),
        "clean_ids": clean_ids, "t_norm": float(t.norm()),
        "g_def": "g^h_k = <t, W_OV^h @ x_k>, x_k = ln1 content at key k (per layer)",
    }

    LN1 = lambda l: f"blocks.{l}.ln1.hook_normalized"
    PAT = lambda l: f"blocks.{l}.attn.hook_pattern"

    results = {"target_direction": target_meta, "config": {
        "PER": PER, "LAYERS": LAYERS, "TRIGS": TRIGS,
        "N_RAND_CLEAN_KEYS": N_RAND_CLEAN_KEYS,
        "TOP_FEATS_PER_KEY": TOP_FEATS_PER_KEY, "RAND_FEATS_PER_KEY": RAND_FEATS_PER_KEY,
        "note_actionability_is_perkey_scalar": True,
    }, "per_trigger": {}}

    def checkpoint(done=False):
        results["done"] = done
        OUT_PATH.write_text(json.dumps(results, indent=2))

    checkpoint(False)

    # ===================== per trigger =====================
    for tname in TRIGS:
        t_info = triggers[tname]
        det = DETECTOR.get(tname)
        span = list(range(L.INSERT_IDX, L.INSERT_IDX + t_info["w"]))   # trigger key positions
        dps = [L.make_deploy_prompt(eval_rows[j]["prompt"], t_info["ids"]) for j in range(PER)]
        ml = max(len(p) for p in dps)
        inp = torch.full((len(dps), ml), tok.eos_token_id)
        plen = []
        for i, p in enumerate(dps):
            inp[i, :len(p)] = torch.tensor(p); plen.append(len(p))
        inp = inp.to(dev)
        # decision query = final REAL prompt position of each row (model trained on prompt-final decode)
        qdec = [pl - 1 for pl in plen]                                # per-row decision query index
        B, T = inp.shape

        # cache ln1 (all needed layers) + patterns
        names = set(LN1(l) for l in LAYERS) | set(PAT(l) for l in LAYERS)
        with torch.no_grad():
            _, cache = model.run_with_cache(inp, return_type=None,
                                            names_filter=lambda n: n in names)
        ln1 = {l: cache[LN1(l)].float() for l in LAYERS}             # (B,T,d)
        patt = {l: cache[PAT(l)].float() for l in LAYERS}            # (B,nH,q,k)
        # SAE codes (layer-0 SAE applied to each layer's ln1 — same SAE, as the ablation uses it)
        z = {l: sae.encode(ln1[l].reshape(B * T, -1)).reshape(B, T, -1) for l in LAYERS}

        per_layer_out = {}
        # collect intervention cells for this trigger across layers, then ablate
        cells = []   # each: dict(layer,k,nu,raw,action,dApred_perturbed,A_qk,gtilde, q-row aggregated)

        for l in LAYERS:
            Qfl, Kfl, WOVl = Qf[l], Kf[l], WOV[l]
            # --- per-row decision-query analysis, then average over rows for stable scalars ---
            # We compute, per key k and key-feature ν, the batch-mean raw |δs| and actionability.
            # raw δs_k^ν = u^ν_k * η^ν_q  with η^ν_q = Σ_h Σ_μ u^μ_q ω^h_{μν}  (sum over heads for scalar score)
            # Per-row q is qdec[b]; we average the per-row quantities over the batch.
            valid_keys = min(plen)                                   # keys present in every row (no pad)
            # accumulators over the prompt-key axis
            raw_acc = torch.zeros(valid_keys, sae.d_sae, device=dev)
            act_acc = torch.zeros(valid_keys, sae.d_sae, device=dev)
            dApred_acc = torch.zeros(valid_keys, sae.d_sae, device=dev)   # ΔA at the perturbed key
            A_acc = torch.zeros(valid_keys, device=dev)
            gt_acc = torch.zeros(valid_keys, device=dev)                  # head-summed g̃ proxy (for logging)
            uk_acc = torch.zeros(valid_keys, sae.d_sae, device=dev)
            for b in range(B):
                q = qdec[b]
                nk = q + 1                                           # causal keys for this query: 0..q
                uq = z[l][b, q, :]                                   # (d_sae,)
                aq = torch.nonzero(uq > 0).squeeze(-1)               # active query feats (~k)
                # per-head pattern row at this query, over ALL causal keys (for a correct ḡ_q)
                A_h_full = patt[l][b, :, q, :nk]                     # (nH, nk)
                # g^h_k = <t, WOV^h x_k> over all causal keys -> (nH, nk)
                xk_full = ln1[l][b, :nk, :]                          # (nk,d)
                ov = torch.einsum("hde,kd->hke", WOVl, xk_full)     # (nH,nk,d) = WOV^h x_k
                g_full = torch.einsum("hke,e->hk", ov, t)           # (nH,nk)
                gbar = (A_h_full * g_full).sum(1, keepdim=True)      # (nH,1)  ḡ_q = Σ_k A_qk g_k
                gtilde_full = A_h_full * (g_full - gbar)             # (nH,nk) g̃^h_{q,k}
                # restrict to the common valid-key window for the accumulators
                A_h = A_h_full[:, :valid_keys]                       # (nH,K)
                A_mean = A_h.mean(0)                                 # (K,) heads-mean pattern
                gtilde = gtilde_full[:, :valid_keys]                 # (nH,K)
                # η^{ν,h}_q = Σ_μ u^μ_q ω^h_{μ,ν} = Σ_μ u^μ_q (Qf[h,μ]·Kf[h,ν]) * scale
                # Σ_μ u^μ_q Qf[h,μ,:] = qvec_h  (per head d_head vector)
                qvec = torch.einsum("a,hae->he", uq[aq], Qfl[:, aq, :])  # (nH,dH)
                eta_h = torch.einsum("he,hve->hv", qvec, Kfl) * scale     # (nH, d_sae): η^{ν,h}_q
                uk = z[l][b, :valid_keys, :]                         # (K, d_sae) key-feature acts
                # raw score change from removing ν at key k (summed over heads): δs_k^ν = u^ν_k * Σ_h η^{ν,h}
                eta_sum = eta_h.sum(0)                               # (d_sae,)
                raw = uk * eta_sum[None, :]                          # (K, d_sae) signed δs
                # actionability per (k,ν) summed over heads: Σ_h δs_k^{ν,h} * g̃^h_{q,k}
                # δs_k^{ν,h} = u^ν_k * η^{ν,h};  multiply by g̃^h_{q,k} then sum heads
                # -> (K,d_sae): uk[k,ν] * Σ_h η_h[h,ν] g̃[h,k]
                eta_g = torch.einsum("hv,hk->kv", eta_h, gtilde)    # (K,d_sae)
                action = uk * eta_g                                  # (K, d_sae) signed δT_q
                # predicted ΔA at perturbed key (heads-mean): A(1-A) * δs   (δs heads-summed score)
                dApred = (A_mean * (1.0 - A_mean))[:, None] * raw   # (K, d_sae)

                raw_acc += raw; act_acc += action; dApred_acc += dApred
                A_acc += A_mean; gt_acc += gtilde.sum(0); uk_acc += uk
            raw_m = (raw_acc / B); act_m = (act_acc / B); dApred_m = (dApred_acc / B)
            A_m = (A_acc / B); gt_m = (gt_acc / B); uk_m = (uk_acc / B)

            # detector-feature ranking at the trigger key(s): raw vs actionability
            # use the closing trigger key (last span pos), as in fra_qk_attr.py
            ktrig = span[-1]
            # rank over active key-features at ktrig by |raw| and |action|
            actset = torch.nonzero(uk_m[ktrig] > 0).squeeze(-1).tolist()
            def _rank_of(feat, score_row):
                if feat is None or feat not in actset:
                    return -1
                vals = {f: abs(float(score_row[ktrig, f])) for f in actset}
                order = sorted(vals, key=lambda f: -vals[f])
                return order.index(feat)
            det_rank_raw = _rank_of(det, raw_m)
            det_rank_act = _rank_of(det, act_m)

            per_layer_out[str(l)] = {
                "valid_keys": int(valid_keys), "ktrig": int(ktrig),
                "trigger_span": span,
                "det_feat": det,
                "det_rank_raw_at_ktrig": det_rank_raw,
                "det_rank_action_at_ktrig": det_rank_act,
                "A_qk_ktrig_mean": float(A_m[ktrig]),
            }
            print(f"[qa] {tname} L{l}: det={det} rank raw={det_rank_raw} action={det_rank_act} "
                  f"(active@ktrig={len(actset)})", flush=True)

            # ---- choose intervention cells for this layer ----
            # keys: trigger span keys + BOS(0) + N random clean-content keys (outside span, >0)
            clean_pool = [k for k in range(1, valid_keys) if k not in span]
            n_pick = min(N_RAND_CLEAN_KEYS, len(clean_pool))
            rand_keys = list(rng.choice(clean_pool, size=n_pick, replace=False)) if n_pick > 0 else []
            key_set = sorted(set(span + [0] + [int(k) for k in rand_keys]))
            for k in key_set:
                active = torch.nonzero(uk_m[k] > 0).squeeze(-1).tolist()
                if not active:
                    continue
                top = sorted(active, key=lambda f: -abs(float(raw_m[k, f])))[:TOP_FEATS_PER_KEY]
                remaining = [f for f in active if f not in top]
                rnd = ([int(rng.choice(remaining))] if remaining else [])[:RAND_FEATS_PER_KEY]
                for nu in top + rnd:
                    if k in span:
                        ktype = "trigger"
                    elif k == 0:
                        ktype = "bos"
                    else:
                        ktype = "clean"
                    cells.append({
                        "layer": l, "k": int(k), "nu": int(nu), "ktype": ktype,
                        "is_detector": bool(det is not None and nu == det),
                        "raw": float(raw_m[k, nu]),
                        "action": float(act_m[k, nu]),
                        "dApred_perturbed": float(dApred_m[k, nu]),
                        "A_qk": float(A_m[k]),
                        "uk_nu": float(uk_m[k, nu]),
                    })

        # ===================== actual ablations =====================
        # For each cell, route the feature removal through the KEY side only at its layer
        # (exact replica of qk_ablate.py), measure ΔA row at qdec and Δlogit-diff toward t.
        # Precompute baseline pattern rows + baseline logit-diff for efficiency.
        @torch.no_grad()
        def baseline_logitdiff():
            lg = model(inp, return_type="logits")                    # (B,T,V)
            # logit-diff at the decision-query position: ihy logit minus mean clean
            # logit (vocab-space equivalent of <t, resid>; t itself is d_model-dim)
            return np.array([float(lg[b, qdec[b], ihy_id])
                             - float(lg[b, qdec[b], clean_ids].float().mean())
                             for b in range(B)])

        with torch.no_grad():
            base_ld = baseline_logitdiff()
        base_A = {l: patt[l] for l in LAYERS}

        @torch.no_grad()
        def ablate_cell(layer, k, nu):
            """Replicate qk_ablate.py key-side routing for one (layer,k,ν); return
            (mean |ΔA| at perturbed key over rows, mean L1 row change, mean Δlogit-diff)."""
            WKl = WK[layer]
            x = ln1[layer][:, k, :]                                  # (B,d) cached ln1 at key k
            zz = sae.encode(x); xh = sae.decode(zz)
            z2 = zz.clone(); z2[:, nu] = 0.0
            x_new = sae.decode(z2)
            d_resid = x_new - xh                                      # (B,d)
            kd = torch.einsum("bd,hde->bhe", d_resid, WKl)           # (B,nH,dH)

            def k_hook(kt, hook):                                     # kt: (B,pos,nH,dH)
                if kt.shape[1] > k:
                    kt[:, k, :, :] = kt[:, k, :, :] + kd
                return kt
            hooks = [(f"blocks.{layer}.attn.hook_k", k_hook)]
            # single forward pass under the key-hook: cache the pattern AND keep logits
            with model.hooks(fwd_hooks=hooks):
                lg2, c2 = model.run_with_cache(inp, return_type="logits",
                                               names_filter=lambda n: n == PAT(layer))
            new_patt = c2[PAT(layer)].float()                        # (B,nH,q,k)
            dA_pert, dA_l1, dld = [], [], []
            for b in range(B):
                q = qdec[b]
                a0 = base_A[layer][b, :, q, :].mean(0)               # (K,) heads-mean baseline row
                a1 = new_patt[b, :, q, :].mean(0)
                dA_pert.append(float((a1[k] - a0[k]).item()))
                dA_l1.append(float((a1 - a0).abs().sum().item()))
                ld1 = (float(lg2[b, q, ihy_id])
                       - float(lg2[b, q, clean_ids].float().mean()))
                dld.append(ld1 - base_ld[b])
            return (float(np.mean(np.abs(dA_pert))), float(np.mean(np.abs(dA_l1))),
                    float(np.mean(dA_pert)), float(np.mean(dld)))

        for c in cells:
            mabs_dA, m_l1, m_signed_dA, m_dld = ablate_cell(c["layer"], c["k"], c["nu"])
            c["actual_dA_perturbed_abs"] = mabs_dA
            c["actual_dA_perturbed_signed"] = m_signed_dA
            c["actual_dA_row_l1"] = m_l1
            c["actual_dlogitdiff"] = m_dld
            # calibration ratio at perturbed key (signed/signed)
            denom = c["dApred_perturbed"]
            c["calib_ratio"] = (abs(m_signed_dA / denom) if abs(denom) > 1e-12 else float("nan"))

        # ===================== analysis (this trigger) =====================
        def _analyse(cell_list):
            raw = [abs(c["raw"]) for c in cell_list]
            act = [abs(c["action"]) for c in cell_list]
            ydA = [c["actual_dA_perturbed_abs"] for c in cell_list]
            ydAl1 = [c["actual_dA_row_l1"] for c in cell_list]
            ytgt = [abs(c["actual_dlogitdiff"]) for c in cell_list]
            calib = [c["calib_ratio"] for c in cell_list if not math.isnan(c["calib_ratio"])]
            within2 = ([1.0 if (0.5 <= r <= 2.0) else 0.0 for r in calib]) if calib else []
            return {
                "n": len(cell_list),
                "rho_raw_vs_dA": spearman(raw, ydA),
                "rho_action_vs_dA": spearman(act, ydA),
                "rho_raw_vs_dA_rowL1": spearman(raw, ydAl1),
                "rho_action_vs_dA_rowL1": spearman(act, ydAl1),
                "rho_raw_vs_target": spearman(raw, ytgt),
                "rho_action_vs_target": spearman(act, ytgt),
                "auc_raw_dA_above_median": auc_above_median(raw, ydA),
                "auc_action_dA_above_median": auc_above_median(act, ydA),
                "median_calib_ratio": float(np.median(calib)) if calib else float("nan"),
                "frac_calib_within_2x": float(np.mean(within2)) if within2 else float("nan"),
            }

        analysis = _analyse(cells)
        # detector cells: actual effect vs raw prediction (the headline ~400x check)
        det_cells = [c for c in cells if c["is_detector"]]
        analysis["detector_cells"] = [
            {"layer": c["layer"], "k": c["k"], "ktype": c["ktype"],
             "raw_ds": c["raw"], "action": c["action"],
             "dApred_perturbed": c["dApred_perturbed"],
             "actual_dA_perturbed_signed": c["actual_dA_perturbed_signed"],
             "calib_ratio": c["calib_ratio"]}
            for c in det_cells]

        results["per_trigger"][tname] = {
            "kind": t_info["kind"], "w": t_info["w"], "detector_feat": det,
            "per_layer": per_layer_out,
            "n_cells": len(cells),
            "analysis": analysis,
            "cells": cells,
        }
        a = analysis
        print(f"[qa] === {tname} done: n={a['n']} "
              f"ρ_raw={a['rho_raw_vs_dA']:.3f} ρ_action={a['rho_action_vs_dA']:.3f} "
              f"| median calib={a['median_calib_ratio']:.3f} within2x={a['frac_calib_within_2x']:.2f} "
              f"| AUC raw={a['auc_raw_dA_above_median']:.3f} action={a['auc_action_dA_above_median']:.3f}",
              flush=True)
        checkpoint(False)

    # ===================== pooled analysis =====================
    all_cells = [c for tn in TRIGS for c in results["per_trigger"][tn]["cells"]]
    raw = [abs(c["raw"]) for c in all_cells]
    act = [abs(c["action"]) for c in all_cells]
    ydA = [c["actual_dA_perturbed_abs"] for c in all_cells]
    ydAl1 = [c["actual_dA_row_l1"] for c in all_cells]
    ytgt = [abs(c["actual_dlogitdiff"]) for c in all_cells]
    calib = [c["calib_ratio"] for c in all_cells if not math.isnan(c["calib_ratio"])]
    within2 = [1.0 if (0.5 <= r <= 2.0) else 0.0 for r in calib] if calib else []
    results["pooled"] = {
        "n": len(all_cells),
        "rho_raw_vs_dA": spearman(raw, ydA),
        "rho_action_vs_dA": spearman(act, ydA),
        "rho_raw_vs_dA_rowL1": spearman(raw, ydAl1),
        "rho_action_vs_dA_rowL1": spearman(act, ydAl1),
        "rho_raw_vs_target": spearman(raw, ytgt),
        "rho_action_vs_target": spearman(act, ytgt),
        "auc_raw_dA_above_median": auc_above_median(raw, ydA),
        "auc_action_dA_above_median": auc_above_median(act, ydA),
        "median_calib_ratio": float(np.median(calib)) if calib else float("nan"),
        "frac_calib_within_2x": float(np.mean(within2)) if within2 else float("nan"),
    }
    p = results["pooled"]
    print(f"\n[qa] ===== POOLED (n={p['n']}) =====", flush=True)
    print(f"  ρ(raw |δs|, |ΔA|)       = {p['rho_raw_vs_dA']:.3f}", flush=True)
    print(f"  ρ(actionable, |ΔA|)     = {p['rho_action_vs_dA']:.3f}  (pre-reg ≥0.8)", flush=True)
    print(f"  median |ΔA_act/ΔA_pred| = {p['median_calib_ratio']:.3f}  within2x={p['frac_calib_within_2x']:.2f}", flush=True)
    print(f"  AUC raw={p['auc_raw_dA_above_median']:.3f}  action={p['auc_action_dA_above_median']:.3f}", flush=True)
    checkpoint(True)
    print("[qa] DONE", flush=True)


if __name__ == "__main__":
    main()
