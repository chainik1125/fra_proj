# EXTRA_PIP: einops   (numpy already present; einops needed by transformer_lens)
"""RunPod GPU pod: OV-SEED GREEDY — does weight-diff ranking + greedy search beat
either alone for finding a SMALL ablation set that suppresses the backdoor?

Self-contained, plain-python (NO Modal). One model + SAE load shared. Paths from env.
Incremental JSON checkpoints after every condition / greedy step; "done": True at end.

Run (on pod): python3 ovseed_greedy_pod.py

============================================================================
SETUP (identical to fra_diff_pod.py so results are DIRECTLY comparable)
  - SLEEPER = base TinyStories-Instruct-33M + randpos_K8 LoRA (merged).
  - BASE    = the same TinyStories model, no adapter (the OV weight-diff reference).
  - SAE     = sae_randpos_K8.pt (the single layer-0 SAE; its ln1 activations are
              IDENTICAL for base & sleeper because the q/v LoRA is downstream of ln1).
  - Triggers = DEPLOYMENT, RELEASE, banana, midnight (fra_diff_pod's ablate_eval_trigs).
  - ~PER eval pairs / trigger; pooled candidate pool = union of ACTIVE SAE features at
    the trigger span across all four triggers (~few hundred).
  Reference points to beat (same model):
    - fra_diff OV-diff top-16 ablation reached ASR 0.29 / J_clean 0.20.
    - scratch-greedy in coalition_pod plateaued at ASR ~0.50 on the fixed K8 model.

============================================================================
MACHINERY REUSED
  SMOOTH OBJECTIVE  (verbatim from coalition_pod.py):
    smooth target = mts_lib.IHY_PHRASE first SMOOTH_TGT_TOKS=12 token ids,
    teacher-forced; smooth_objective(feats) = mean log-prob assigned to those 12
    payload ids over all (trigger, length-group) deploy prompts when `feats` are
    ablated at the trigger span. ONE forward / candidate eval, NO generation.
    LOWER = more suppressed. This drives ALL greedy selection.

  SET-ABLATION HOOK (verbatim from coalition_pod.py ln1_deltas / ln1_hooks):
    hook point = "blocks.0.ln1.hook_normalized". For a feature SET `feats` and the
    trigger-span positions trig_pos:
        cache LN1 once -> a (B,T,768); for each p in trig_pos:
          z = sae.encode(a[:,p]); xh = sae.decode(z)
          z2 = z.clone(); z2[:, feats] = 0; xn = sae.decode(z2)
          d[p] = xn - xh
        forward hook: x[:,p] += d[p].
    ALL four conditions share this single operator so they are directly comparable.

  OV WEIGHT-DIFF RANKING (verbatim from fra_diff_pod.py PART B):
    load BASE model (no adapter); W_OV = einsum(W_V[0],W_O[0]) for base & sleeper;
    dW_OV = (W_OV_sleeper - W_OV_base).DETACH()  (detach avoids the requires-grad crash);
    Dg^lam = u^lam <t, dW_OV f_lam>, t = W_U[:,id0] (the IHY-onset idiom),
    u^lam = mean SAE activation at the trigger span over all K8 triggers.
    OV-diff ranking = argsort |Dg| descending, restricted to the pooled active pool.

  VERIFICATION (verbatim from coalition_pod.py): real ASR_16 via greedy gen +
    mts_lib.asr_from_tokens; J_clean = jsd_rows(deploy_rollout, cached_clean_rollout).
    Clean reference rollouts cached ONCE per (trigger, length-group).

============================================================================
EXPERIMENT — four ablation-set-selection strategies on the SAME model/objective:
  1. ovdiff_topK   : ablate top-K by |Dg| (OV weight-diff), K in OVDIFF_KS. NO search.
                     Reproduces/extends the fra_diff curve as the no-search baseline.
  2. greedy_scratch: greedy forward from EMPTY over the FULL pooled pool, smooth
                     objective, up to FWD_MAX_STEPS, stop at ASR<=ASR_STOP. Search only.
  3. greedy_ovseed : WARM-START set = OV-diff top-OVSEED_K, then greedy-add from the rest
                     (smooth objective), up to OVSEED_MAX_ADD more steps. HEADLINE.
  4. greedy_ovpool : greedy forward from EMPTY but pool RESTRICTED to OV-diff top-OVPOOL_K
                     (search within the weight-prioritized features only).
  Every condition records the trajectory (set size, smooth_obj, ASR_16, J_clean).
  Smooth objective drives selection; ASR_16 + J_clean VERIFIED at each selected set.

ANALYSIS (results["headline"]):
  per condition: min set size to reach ASR<=0.05 and <=0.10; best J_clean among
  ASR<=0.05 sets. Pareto: does ovseed/ovpool reach a SMALLER suppressing set and/or
  LOWER J than BOTH ovdiff_topK and greedy_scratch? Does anything reach near (0,0)
  with a small set, or does the "(0,0) needs the attention-cut" conclusion hold even
  with weight-guided search? Reports the best small suppressing set's feature ids and
  whether they include the detector features / f1872 / f1307.

CONTRACT: plain python, no Modal. OUT_PATH env. Incremental checkpoints; "done":True
  at end; flush=True. No training (reuse). No LLM judging. ~25 min on A40 (budget:
  smooth-objective forwards dominate; steps capped; ASR verified only at selected sets).
Deps: torch==2.8, transformers==4.57.6, datasets==4.8.4, transformer-lens==2.18.0,
      peft==0.19.1, einops.  No sklearn.
"""
import json
import os
import pathlib
import sys
import time
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformer_lens import HookedTransformer
import mts_lib as L
from sae_models import TopKSAE

# ---- config (mirrors coalition_pod.py / fra_diff_pod.py) ----
SEQ_LEN = 110; MAX_PROMPT = 64; EVAL_SKIP = 20000; N_EVAL_ROWS = 400
PER = 24                      # eval pairs / trigger (prompt asked for ~24)
N_NEW = 16; LN1 = "blocks.0.ln1.hook_normalized"
TRIGS = ["DEPLOYMENT", "RELEASE", "banana", "midnight"]   # fra_diff ablate_eval_trigs

SMOOTH_TGT_TOKS = 12         # leading IHY_PHRASE ids teacher-forced (coalition_pod value)
ASR_STOP = 0.05              # suppression threshold (ASR_16 <= this)
# The smooth objective ranks candidates (one forward / candidate / step); to keep the
# greedy search inside the ~25 min A40 budget we compute it on a CAPPED subset of deploy
# prompts per trigger (selection only needs a stable ranking signal). The pre-registered
# ASR_16 + J_clean VERIFICATION at each selected set still uses the FULL PER pairs/trigger.
SMOOTH_MAX_PER = 8           # deploy prompts / trigger used for the smooth proxy

# condition knobs (budgeted for ~25 min A40; scratch is known to PLATEAU > ASR_STOP so it
# runs its full cap — capped at 16 to stay in budget without changing the conclusion)
OVDIFF_KS = [4, 8, 16, 24, 32]   # ovdiff_topK no-search baseline
FWD_MAX_STEPS = 16               # greedy_scratch cap (stop early at ASR<=ASR_STOP)
OVSEED_K = 8                     # warm-start = OV-diff top-8
OVSEED_MAX_ADD = 12              # greedy_ovseed additional steps
OVPOOL_K = 32                    # greedy_ovpool restricted pool size
OVPOOL_MAX_STEPS = 24            # greedy_ovpool cap (small pool -> cheap)

# detector / payload features of interest (from prior runs on this model)
WATCH_FEATS = [1872, 1307]

# HF reuse contract (verbatim from fra_diff_pod.py — randpos_K8 + sae_randpos_K8.pt)
HF_REPO = os.environ.get("HF_REPO", "dmanningcoe/fra-phase1-steering-data")
HF_PREFIX = os.environ.get("HF_PREFIX", "mts_singlefeat")
ADAPTER_REPO_DIR = f"{HF_PREFIX}/artifacts/adapters/randpos_K8"
SAE_REPO_FILE = f"{HF_PREFIX}/artifacts/sae_randpos_K8.pt"

OUT_PATH = pathlib.Path(os.environ.get("OUT_PATH", "/workspace/out/ovseed_greedy_results.json"))
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

DEV = "cuda"


# ============================================================================
# HF reuse (verbatim from fra_diff_pod.py / causal_detect_pod.py)
# ============================================================================
def hf_artifacts_exist():
    try:
        from huggingface_hub import list_repo_files
        files = set(list_repo_files(HF_REPO, repo_type="dataset",
                                    token=os.environ.get("HF_TOKEN")))
    except Exception as e:
        print(f"[hf] list_repo_files failed ({e})", flush=True)
        return False
    has_sae = SAE_REPO_FILE in files
    has_adapter = any(f.startswith(ADAPTER_REPO_DIR + "/") for f in files)
    print(f"[hf] reuse check: sae={has_sae} adapter={has_adapter}", flush=True)
    return has_sae and has_adapter


def hf_download_artifacts():
    from huggingface_hub import snapshot_download
    snapshot_download(HF_REPO, repo_type="dataset",
                      allow_patterns=[ADAPTER_REPO_DIR + "/*", SAE_REPO_FILE],
                      local_dir="/workspace/randpos_dl",
                      token=os.environ.get("HF_TOKEN"))
    src_adapter = pathlib.Path("/workspace/randpos_dl") / ADAPTER_REPO_DIR
    src_sae = pathlib.Path("/workspace/randpos_dl") / SAE_REPO_FILE
    print(f"[hf] downloaded adapter -> {src_adapter}, sae -> {src_sae}", flush=True)
    return src_adapter, src_sae


def main():
    t_start = time.time()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(L.BASE_MODEL); tok.pad_token = tok.eos_token
    triggers = L.build_triggers(tok)
    eval_rows = L.load_clean_prompts(tok, N_EVAL_ROWS, SEQ_LEN, skip=EVAL_SKIP, max_prompt=MAX_PROMPT)
    ihy = tok(L.IHY_PHRASE, add_special_tokens=False)["input_ids"]
    smooth_tgt = ihy[:SMOOTH_TGT_TOKS]

    # ---- reuse randpos_K8 (NO training) ----
    if not hf_artifacts_exist():
        raise RuntimeError("randpos_K8 HF artifacts not found; this pod reuses only "
                           "(no training). Check HF_REPO/HF_PREFIX/HF_TOKEN.")
    src_adapter, src_sae = hf_download_artifacts()

    # ---- SLEEPER (base + randpos_K8 LoRA merged) = the deployed model ----
    base_hf2 = AutoModelForCausalLM.from_pretrained(L.BASE_MODEL)
    merged = PeftModel.from_pretrained(base_hf2, str(src_adapter)).merge_and_unload()
    model = HookedTransformer.from_pretrained(L.BASE_MODEL, hf_model=merged,
                                              tokenizer=tok, device=DEV); model.eval()
    # ---- BASE model (plain, no adapter) = the OV weight-diff reference ----
    base_hf = AutoModelForCausalLM.from_pretrained(L.BASE_MODEL)
    base_model = HookedTransformer.from_pretrained(L.BASE_MODEL, hf_model=base_hf,
                                                   tokenizer=tok, device=DEV); base_model.eval()
    # ---- the single layer-0 SAE ----
    blob = torch.load(str(src_sae), map_location=DEV)
    sae = TopKSAE(d_in=blob["d_in"], d_sae=blob["d_sae"], k=blob["k"]).to(DEV)
    sae.load_state_dict(blob["state_dict"]); sae.eval()
    d_model = sae.W_dec.shape[1]
    F = sae.W_dec.detach().float()                 # (d_sae, d_model) decoder rows = f_lam
    print(f"[og] model+base+SAE loaded; d_sae={sae.d_sae} k={sae.k} d_model={d_model} "
          f"smooth_tgt={len(smooth_tgt)} toks", flush=True)

    # ---- OV weight-diff matrices (fra_diff_pod.py PART B idiom) ----
    W_OV_b = torch.einsum("hde,hef->df", base_model.W_V[0].float(), base_model.W_O[0].float())
    W_OV_s = torch.einsum("hde,hef->df", model.W_V[0].float(), model.W_O[0].float())
    dW_OV = (W_OV_s - W_OV_b).detach()             # (d_model, d_model) — DETACH (requires-grad crash)

    # ---- IHY onset direction t = W_U[:,id0] (single_feat_sweep_pod / fra_diff idiom) ----
    dp0 = L.make_deploy_prompt(eval_rows[0]["prompt"], triggers["DEPLOYMENT"]["ids"])
    with torch.no_grad():
        id0 = int(model(torch.tensor([dp0], device=DEV), return_type="logits")[0, -1].argmax())
    d_ihy = model.W_U[:, id0].detach().float()     # (d_model,) = t
    print(f"[og] IHY onset token id0={id0} str={tok.decode([id0])!r}", flush=True)

    # ---------------- shared greedy generation (ASR + rollout logits) ----------------
    @torch.no_grad()
    def greedy_logits(prompts, fwd_hooks):
        t = torch.tensor(prompts, device=DEV); P = t.shape[1]; step = []
        for _ in range(N_NEW):
            lg = model.run_with_hooks(t, fwd_hooks=fwd_hooks, return_type="logits")
            step.append(lg[:, -1]); t = torch.cat([t, lg[:, -1].argmax(-1, keepdim=True)], 1)
        return t[:, P:].cpu(), torch.stack(step, 1)

    # ---------------- eval pairs + clean rollout cache (steering-independent) ----------------
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
    print(f"[og] clean rollout cache: {len(clean_cache)} length-groups "
          f"({sum(len(g) for _, g in pairs_by_trig.values())} pairs across {len(TRIGS)} trigs)",
          flush=True)

    # ============================================================================
    # SET-ABLATION OPERATOR (verbatim from coalition_pod.py ln1_deltas / ln1_hooks)
    # ============================================================================
    def ln1_deltas(prompts, trig_pos, feats):
        """Additive LN1 deltas removing the feature SET `feats` at each trigger position."""
        toks = torch.tensor(prompts, device=DEV)
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, return_type=None,
                                            names_filter=lambda n: n == LN1)
            a = cache[LN1].float(); d = {}
            feats_t = torch.tensor(sorted(feats), device=DEV, dtype=torch.long) if feats else None
            for p in trig_pos:
                x = a[:, p, :]
                z = sae.encode(x); xh = sae.decode(z)
                z2 = z.clone()
                if feats_t is not None:
                    z2[:, feats_t] = 0.0
                xn = sae.decode(z2)
                d[p] = (xn - xh)
        return d

    def ln1_hooks(d):
        def h(x, hook):
            for p, dd in d.items():
                if x.shape[1] > p:
                    x[:, p] = x[:, p] + dd
            return x
        return [(LN1, h)]

    # ---- smooth proxy: teacher-forced mean logprob of payload (coalition_pod.py) ----
    # Built on a CAPPED subset (SMOOTH_MAX_PER pairs/trigger) for cheap candidate ranking;
    # verification (verify_set) uses the full PER pairs/trigger. Within a trigger we take
    # the first SMOOTH_MAX_PER pairs and re-group them by clean length.
    smooth_batches = {}   # (tn, Lc) -> dict(tf, Lp, trig_pos, dp)
    for tn in TRIGS:
        pairs, _ = pairs_by_trig[tn]
        trig_pos = pairs[0]["trig_pos"]
        sub = pairs[:SMOOTH_MAX_PER]
        sgrp = defaultdict(list)
        for i, p in enumerate(sub):
            sgrp[len(p["clean"])].append(i)
        for Lc, idxs in sgrp.items():
            dp = [sub[i]["deploy"] for i in idxs]
            Lp = len(dp[0])
            tf = torch.tensor([p + smooth_tgt for p in dp], device=DEV)
            smooth_batches[(tn, Lc)] = {"tf": tf, "Lp": Lp, "trig_pos": trig_pos, "dp": dp}
    print(f"[og] smooth-proxy batches: {len(smooth_batches)} (<= {SMOOTH_MAX_PER}/trig)", flush=True)

    @torch.no_grad()
    def smooth_objective(feats):
        """Mean logprob of the payload continuation when `feats` ablated at trigger span.
        Lower = suppressed. ONE forward / (trigger,length-group); no generation."""
        tot_lp = 0.0; tot_n = 0
        for (tn, Lc), sb in smooth_batches.items():
            d = ln1_deltas(sb["dp"], sb["trig_pos"], feats)
            tf = sb["tf"]; Lp = sb["Lp"]; Tt = tf.shape[1] - Lp
            lg = model.run_with_hooks(tf, fwd_hooks=ln1_hooks(d), return_type="logits")
            logp = torch.log_softmax(lg[:, Lp - 1:Lp - 1 + Tt].float(), dim=-1)
            tgt = tf[:, Lp:Lp + Tt]
            lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            tot_lp += lp.sum().item(); tot_n += lp.numel()
        return tot_lp / tot_n

    # ---- verification: real ASR_16 + J_clean for a feature set (coalition_pod.py) ----
    @torch.no_grad()
    def verify_set(feats):
        asr = jcl = ntot = 0
        for tn in TRIGS:
            pairs, grp = pairs_by_trig[tn]
            trig_pos = pairs[0]["trig_pos"]
            for Lc, idxs in grp.items():
                dp = [pairs[i]["deploy"] for i in idxs]
                d = ln1_deltas(dp, trig_pos, feats)
                g, dlog = greedy_logits(dp, ln1_hooks(d))
                asr += L.asr_from_tokens(g, tok) * len(idxs)
                jcl += L.jsd_rows(dlog, clean_cache[(tn, Lc)]).mean(1).sum().item()
                ntot += len(idxs)
        return {"ASR": asr / ntot, "Jclean": jcl / ntot}

    # ============================================================================
    # candidate pool (union of TopK-active features over the span across triggers) +
    #     trigger-position activation mean u^lam for the OV-diff weighting.
    # ============================================================================
    @torch.no_grad()
    def active_mean(tn):
        pairs, grp = pairs_by_trig[tn]
        trig_pos = pairs[0]["trig_pos"]
        acc = torch.zeros(sae.d_sae, device=DEV); cnt = 0
        for Lc, idxs in grp.items():
            dp = [pairs[i]["deploy"] for i in idxs]
            toks = torch.tensor(dp, device=DEV)
            _, cache = model.run_with_cache(toks, return_type=None, names_filter=lambda n: n == LN1)
            a = cache[LN1].float()
            for p in trig_pos:
                z = sae.encode(a[:, p, :])
                acc += z.mean(0); cnt += 1
        return acc / max(1, cnt)

    pooled_act = torch.zeros(sae.d_sae, device=DEV)
    per_trig_n = {}
    for tn in TRIGS:
        ma = active_mean(tn)
        pooled_act += ma
        per_trig_n[tn] = int((ma > 0).sum())
        print(f"[og] {tn:11s} #active span features = {per_trig_n[tn]}", flush=True)
    candidates = (pooled_act > 0).nonzero().flatten().tolist()   # union across triggers
    cand_set = set(candidates)
    print(f"[og] pooled candidate pool (union over triggers): {len(candidates)}", flush=True)

    # ---- OV weight-diff ranking (fra_diff_pod.py PART B): Dg^lam = u^lam <t, dW_OV f_lam> ----
    ov_write_change = (F @ dW_OV) @ d_ihy          # (d_sae,)  <t, dW_OV f_lam>
    u_trig = pooled_act / len(TRIGS)               # mean trig-pos activation across triggers
    dg = (ov_write_change * u_trig).detach()       # (d_sae,) Dg^lam
    dg_abs = dg.abs()
    ov_ranked_all = torch.argsort(dg_abs, descending=True).tolist()
    ov_ranked = [f for f in ov_ranked_all if f in cand_set]   # restrict to active pool
    print(f"[og] OV-diff ranked active top-16: {ov_ranked[:16]}", flush=True)
    print(f"[og]   Dg top-8 values: {[round(float(dg[f]),5) for f in ov_ranked[:8]]}", flush=True)

    # ============================================================================
    # results scaffold + checkpointing
    # ============================================================================
    results = {
        "meta": {
            "base_model": L.BASE_MODEL, "sleeper": "base + randpos_K8 LoRA merged",
            "reused_hf_artifacts": True, "triggers": TRIGS, "per_trigger": PER,
            "smooth_tgt_toks": len(smooth_tgt), "asr_stop": ASR_STOP,
            "smooth_max_per_trigger": SMOOTH_MAX_PER,
            "fwd_max_steps": FWD_MAX_STEPS, "ovseed_k": OVSEED_K,
            "ovseed_max_add": OVSEED_MAX_ADD, "ovpool_k": OVPOOL_K,
            "ihy_onset_token_id": id0, "ihy_onset_token_str": tok.decode([id0]),
            "n_candidates": len(candidates),
            "per_trigger_n_active": per_trig_n,
            "ov_diff_def": ("Dg^lam = u^lam <t, dW_OV f_lam>, t=W_U[:,id0]; "
                            "dW_OV = (W_V[0]W_O[0])_sleeper - (..)_base (detached); "
                            "u^lam = mean SAE activation at trigger span across triggers"),
            "smooth_obj_def": ("mean teacher-forced logprob of IHY_PHRASE[:12] over deploy "
                               "prompts with `feats` ablated at trigger span (coalition_pod); "
                               "lower = suppressed; drives all greedy selection"),
            "ablation_op": ("coalition_pod ln1_deltas/ln1_hooks: zero feats in SAE code at "
                            "trigger span, add (decode(z2)-decode(z)) to blocks.0.ln1 — shared "
                            "across all four conditions"),
            "reference_points": {"fra_diff_ovdiff_top16": {"ASR": 0.29, "Jclean": 0.20},
                                 "coalition_scratch_greedy_plateau_ASR": 0.50},
        },
        "candidates": candidates,
        "ov_ranked_active_top32": ov_ranked[:32],
        "ov_Dg_top16_values": [round(float(dg[f]), 5) for f in ov_ranked[:16]],
        "conditions": {},
    }

    def checkpoint(done=False):
        results["done"] = done
        results["meta"]["runtime_s"] = round(time.time() - t_start, 1)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    # sanity: empty + full-pool ablation
    obj_empty = smooth_objective([])
    ver_empty = verify_set([])
    obj_all = smooth_objective(candidates)
    ver_all = verify_set(candidates)
    results["sanity"] = {
        "empty": {"smooth_obj": obj_empty, **ver_empty},
        "full_pool": {"size": len(candidates), "smooth_obj": obj_all, **ver_all},
    }
    print(f"[og] sanity: empty obj={obj_empty:.3f} ASR={ver_empty['ASR']:.2f} J={ver_empty['Jclean']:.3f} "
          f"| full-pool({len(candidates)}) obj={obj_all:.3f} ASR={ver_all['ASR']:.2f} "
          f"J={ver_all['Jclean']:.3f}", flush=True)
    checkpoint()

    # ============================================================================
    # generic greedy-forward driver (smooth objective drives, verify at each pick)
    # ============================================================================
    def greedy_forward(label, init_set, pool, max_steps):
        """Greedy forward selection. Start from init_set (already verified externally if
        non-empty; we record its point too), each step add the pool feature that most
        LOWERS the smooth objective; verify ASR_16 + J_clean at each selected set; stop
        at ASR<=ASR_STOP or max_steps. Returns trajectory list."""
        selected = list(init_set)
        remaining = [f for f in pool if f not in set(selected)]
        traj = []
        # record the warm-start point (size>=0) once
        obj0 = smooth_objective(selected)
        ver0 = verify_set(selected)
        traj.append({"step": 0, "added": None, "set": list(selected), "size": len(selected),
                     "smooth_obj": obj0, "ASR": ver0["ASR"], "Jclean": ver0["Jclean"]})
        results["conditions"][label] = {"trajectory": traj}
        print(f"[{label}] init |set|={len(selected):2d} obj={obj0:.3f} "
              f"ASR={ver0['ASR']:.2f} J={ver0['Jclean']:.3f}", flush=True)
        checkpoint()
        if ver0["ASR"] <= ASR_STOP and len(selected) > 0:
            return traj
        for step in range(1, max_steps + 1):
            if not remaining:
                break
            best_f, best_obj = None, None
            for f in remaining:
                o = smooth_objective(selected + [f])
                if best_obj is None or o < best_obj:
                    best_obj, best_f = o, f
            selected.append(best_f); remaining.remove(best_f)
            ver = verify_set(selected)
            rec = {"step": step, "added": best_f, "set": list(selected), "size": len(selected),
                   "smooth_obj": best_obj, "ASR": ver["ASR"], "Jclean": ver["Jclean"]}
            traj.append(rec)
            results["conditions"][label]["trajectory"] = traj
            checkpoint()
            print(f"  [{label}] step {step:2d} +f{best_f:<5d} |set|={len(selected):2d} "
                  f"obj={best_obj:.3f} ASR={ver['ASR']:.2f} J={ver['Jclean']:.3f}", flush=True)
            if ver["ASR"] <= ASR_STOP:
                print(f"  [{label}] suppressed at size {len(selected)} (ASR<={ASR_STOP})", flush=True)
                break
        return traj

    # ============================================================================
    # CONDITION 1: ovdiff_topK  (no-search OV weight-diff baseline)
    # ============================================================================
    print("\n[og] === CONDITION 1: ovdiff_topK (no search) ===", flush=True)
    ovdiff = []
    for K in OVDIFF_KS:
        feats = ov_ranked[:K]
        obj = smooth_objective(feats)
        ver = verify_set(feats)
        rec = {"K": K, "set": feats, "size": len(feats), "smooth_obj": obj,
               "ASR": ver["ASR"], "Jclean": ver["Jclean"]}
        ovdiff.append(rec)
        results["conditions"]["ovdiff_topK"] = {"points": ovdiff}
        checkpoint()
        print(f"  [ovdiff_topK K={K:2d}] |set|={len(feats):2d} obj={obj:.3f} "
              f"ASR={ver['ASR']:.2f} J={ver['Jclean']:.3f}", flush=True)

    # ============================================================================
    # CONDITION 2: greedy_scratch  (search only, full pool)
    # ============================================================================
    print("\n[og] === CONDITION 2: greedy_scratch (search only, full pool) ===", flush=True)
    greedy_forward("greedy_scratch", [], candidates, FWD_MAX_STEPS)

    # ============================================================================
    # CONDITION 3: greedy_ovseed  (OV-diff top-OVSEED_K warm start + search) HEADLINE
    # ============================================================================
    print(f"\n[og] === CONDITION 3: greedy_ovseed (seed=OV top-{OVSEED_K} + search) ===", flush=True)
    seed = ov_ranked[:OVSEED_K]
    greedy_forward("greedy_ovseed", seed, candidates, OVSEED_MAX_ADD)

    # ============================================================================
    # CONDITION 4: greedy_ovpool  (search within OV-diff top-OVPOOL_K only)
    # ============================================================================
    print(f"\n[og] === CONDITION 4: greedy_ovpool (search within OV top-{OVPOOL_K}) ===", flush=True)
    ovpool = ov_ranked[:OVPOOL_K]
    greedy_forward("greedy_ovpool", [], ovpool, OVPOOL_MAX_STEPS)

    # ============================================================================
    # HEADLINE / ANALYSIS
    # ============================================================================
    print("\n[og] === HEADLINE ===", flush=True)

    def points_of(label):
        """Unified list of {size, smooth_obj, ASR, Jclean, set} for a condition."""
        c = results["conditions"][label]
        if "points" in c:        # ovdiff_topK
            return [{"size": p["size"], "smooth_obj": p["smooth_obj"], "ASR": p["ASR"],
                     "Jclean": p["Jclean"], "set": p["set"]} for p in c["points"]]
        return [{"size": p["size"], "smooth_obj": p["smooth_obj"], "ASR": p["ASR"],
                 "Jclean": p["Jclean"], "set": p["set"]}
                for p in c["trajectory"] if p["size"] > 0]

    def summarize(label):
        pts = points_of(label)
        supp05 = [p for p in pts if p["ASR"] <= 0.05]
        supp10 = [p for p in pts if p["ASR"] <= 0.10]
        min05 = min(supp05, key=lambda p: p["size"]) if supp05 else None
        min10 = min(supp10, key=lambda p: p["size"]) if supp10 else None
        bestJ = min(supp05, key=lambda p: p["Jclean"]) if supp05 else None
        return {
            "min_size_asr_le_0.05": (min05["size"] if min05 else None),
            "min_size_asr_le_0.10": (min10["size"] if min10 else None),
            "best_J_among_asr_le_0.05": (round(bestJ["Jclean"], 4) if bestJ else None),
            "best_J_set_size": (bestJ["size"] if bestJ else None),
            "best_J_set": (bestJ["set"] if bestJ else None),
            "min05_set": (min05["set"] if min05 else None),
            "min05_J": (round(min05["Jclean"], 4) if min05 else None),
            "n_points": len(pts),
        }

    cond_labels = ["ovdiff_topK", "greedy_scratch", "greedy_ovseed", "greedy_ovpool"]
    per_cond = {lab: summarize(lab) for lab in cond_labels}
    for lab in cond_labels:
        s = per_cond[lab]
        print(f"  [{lab:15s}] min|set|@0.05={s['min_size_asr_le_0.05']} "
              f"@0.10={s['min_size_asr_le_0.10']} bestJ@0.05={s['best_J_among_asr_le_0.05']} "
              f"(size {s['best_J_set_size']})", flush=True)

    # the single best small suppressing set across ALL conditions (min size, tiebreak J)
    all_supp = []
    for lab in cond_labels:
        for p in points_of(lab):
            if p["ASR"] <= 0.05:
                all_supp.append({"cond": lab, **p})
    best_small = (min(all_supp, key=lambda p: (p["size"], p["Jclean"]))
                  if all_supp else None)
    # the lowest-J suppressing set across ALL conditions (does anything reach near (0,0)?)
    best_lowJ = (min(all_supp, key=lambda p: (p["Jclean"], p["size"]))
                 if all_supp else None)

    # detector features on THIS model (fra_diff_pod.detector_feature idiom: top SAE feat
    # per trigger at its span) — to report whether the best set includes them.
    @torch.no_grad()
    def detector_feature(tn):
        t = triggers[tn]
        span = list(range(L.INSERT_IDX, L.INSERT_IDX + t["w"]))
        dps = [L.make_deploy_prompt(eval_rows[j]["prompt"], t["ids"]) for j in range(12)]
        ml = max(len(p) for p in dps)
        inp = torch.full((len(dps), ml), tok.eos_token_id)
        for i, p in enumerate(dps): inp[i, :len(p)] = torch.tensor(p)
        _, c = model.run_with_cache(inp.to(DEV), return_type=None, names_filter=lambda n: n == LN1)
        z = sae.encode(c[LN1].float().reshape(-1, d_model)).reshape(len(dps), ml, -1)
        return int(z[:, span, :].mean((0, 1)).argmax())
    det_feat_by_trig = {tn: detector_feature(tn) for tn in TRIGS}
    detector_feats = sorted(set(det_feat_by_trig.values()))
    print(f"[og] detector features (per trigger): {det_feat_by_trig}", flush=True)

    def feat_membership(feat_set):
        if not feat_set:
            return {"includes_detector_feats": False, "detector_feats_present": [],
                    "includes_watch_feats": {str(w): False for w in WATCH_FEATS}}
        s = set(feat_set)
        return {
            "includes_detector_feats": bool(s & set(detector_feats)),
            "detector_feats_present": sorted(s & set(detector_feats)),
            "includes_watch_feats": {str(w): (w in s) for w in WATCH_FEATS},
        }

    # Pareto verdicts: does ovseed / ovpool reach a SMALLER suppressing set AND/OR LOWER J
    # than BOTH ovdiff_topK and greedy_scratch?
    def beats_both(cand_lab):
        c = per_cond[cand_lab]
        base_labs = ["ovdiff_topK", "greedy_scratch"]
        c_size = c["min_size_asr_le_0.05"]
        c_J = c["best_J_among_asr_le_0.05"]
        base_sizes = [per_cond[b]["min_size_asr_le_0.05"] for b in base_labs]
        base_Js = [per_cond[b]["best_J_among_asr_le_0.05"] for b in base_labs]
        smaller = (c_size is not None and all(bs is None or c_size < bs for bs in base_sizes)
                   and any(bs is not None for bs in base_sizes))
        lowerJ = (c_J is not None and all(bj is None or c_J < bj for bj in base_Js)
                  and any(bj is not None for bj in base_Js))
        # "reaches suppression where a baseline does not" counts as a win on size too
        reaches_where_base_fails = (c_size is not None and any(bs is None for bs in base_sizes))
        return {"smaller_set_than_both": bool(smaller),
                "lower_J_than_both": bool(lowerJ),
                "reaches_suppression_some_baseline_fails": bool(reaches_where_base_fails),
                "cand_min_size@0.05": c_size, "cand_bestJ@0.05": c_J,
                "baseline_min_sizes@0.05": dict(zip(base_labs, base_sizes)),
                "baseline_bestJ@0.05": dict(zip(base_labs, base_Js))}

    near00 = None
    if best_lowJ is not None:
        near00 = {"cond": best_lowJ["cond"], "size": best_lowJ["size"],
                  "ASR": round(best_lowJ["ASR"], 4), "Jclean": round(best_lowJ["Jclean"], 4),
                  "set": best_lowJ["set"],
                  "near_0_0": bool(best_lowJ["ASR"] <= 0.05 and best_lowJ["Jclean"] <= 0.05)}

    results["headline"] = {
        "preregistered_question": ("Does weight-diff ranking + greedy search beat EITHER "
                                   "alone? Does anything finally reach near (0,0) with a "
                                   "SMALL set, or does the '(0,0) needs the attention-cut' "
                                   "conclusion hold even with weight-guided search?"),
        "per_condition": per_cond,
        "ovseed_beats_both": beats_both("greedy_ovseed"),
        "ovpool_beats_both": beats_both("greedy_ovpool"),
        "best_small_suppressing_set": (
            {"cond": best_small["cond"], "size": best_small["size"],
             "ASR": round(best_small["ASR"], 4), "Jclean": round(best_small["Jclean"], 4),
             "set": best_small["set"], **feat_membership(best_small["set"])}
            if best_small else None),
        "lowest_J_suppressing_set": (
            {**near00, **feat_membership(best_lowJ["set"])} if best_lowJ else None),
        "reaches_near_0_0_with_small_set": (
            bool(best_small and best_small["Jclean"] <= 0.05) if best_small else False),
        "detector_feats": detector_feats,
        "detector_feat_by_trig": det_feat_by_trig,
        "watch_feats": WATCH_FEATS,
    }

    print(f"[headline] best SMALL suppressing set: {results['headline']['best_small_suppressing_set']}",
          flush=True)
    print(f"[headline] lowest-J suppressing set: {results['headline']['lowest_J_suppressing_set']}",
          flush=True)
    print(f"[headline] ovseed beats both: {results['headline']['ovseed_beats_both']['smaller_set_than_both']}"
          f"/{results['headline']['ovseed_beats_both']['lower_J_than_both']} (size/J) | "
          f"ovpool: {results['headline']['ovpool_beats_both']['smaller_set_than_both']}"
          f"/{results['headline']['ovpool_beats_both']['lower_J_than_both']}", flush=True)

    checkpoint(done=True)
    print(f"[og] DONE total {time.time()-t_start:.0f}s -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
