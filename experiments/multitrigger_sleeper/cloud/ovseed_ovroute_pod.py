# EXTRA_PIP: einops   (numpy already present; einops needed by transformer_lens)
"""RunPod GPU pod: OV-SEED OV-ROUTE — does ablating the OV-diff-selected feature SET
through the OV PATH ONLY (hook_v, attention pattern FROZEN) give the same "wall" as the
full-path ln1 ablation, or did the QK perturbation matter?

Self-contained, plain-python (NO Modal). One model + SAE load shared. Paths derived for
the randpos artifacts (NOT the env defaults). Incremental JSON checkpoints after every
(route, condition); "done": True at end; flush=True.

Run (on pod): python3 ovseed_ovroute_pod.py

============================================================================
THE LOOPHOLE WE ARE CLOSING
  ovseed_greedy_pod ablated feature SETS at blocks.0.ln1.hook_normalized. ln1 feeds
  Q, K AND V of block 0 simultaneously, so removing a feature there ALSO perturbs the
  attention PATTERN (the QK side), not just the value the head writes. So a "no small set
  suppresses" conclusion from ln1 ablation could be confounded: maybe the suppression
  (and its J_clean cost) came partly from breaking attention to the trigger.

  Here we run a HEAD-TO-HEAD on the SAME model + SAME OV-diff selection, under TWO ablation
  ROUTES that remove the IDENTICAL feature set at the IDENTICAL trigger-span positions:

    ln1 (FULL path)     — the existing ovseed/coalition ln1_deltas/ln1_hooks: build
                          d_p = decode(z2) - decode(z) (the feature-set removal in the
                          SAE's native ln1 space) at each trigger position p, ADD d_p at
                          blocks.0.ln1.hook_normalized. Removal hits Q, K and V.

    OV-only (hook_v, QK FROZEN) — build the SAME ln1-space removal delta d_p, then instead
                          of adding it at ln1, ROUTE it through the value projection:
                              kd_p = einsum("d,hde->he", d_p, W_V[0])   # (heads, d_head)
                          and ADD kd_p at blocks.0.attn.hook_v at position p.
                          This subtracts EXACTLY the value the removed features would have
                          written through block-0 attention, while leaving Q and K — hence
                          the attention PATTERN — UNTOUCHED. Attention still flows to the
                          trigger span; only the VALUE the trigger features contribute is
                          removed. (Generalizes ov_route_pod.py's single-direction OV hook
                          to a feature SET and to the trigger-span KEY/value positions.)

  CRITICAL (matches ovseed/ov_route): TransformerLens runs NO kv-cache, so each of the 16
  greedy decode steps re-runs the FULL forward. The OV-only hook is re-applied every step
  via run_with_hooks, exactly as the ln1 hook is, so the value-write removal persists for
  the whole generation. The hook touches ONLY blocks.0.attn.hook_v — Q (hook_q) and K
  (hook_k) are NEVER modified, so the attention pattern is frozen.

============================================================================
SETUP (identical to ovseed_greedy_pod.py so results are DIRECTLY comparable)
  - SLEEPER = base TinyStories-Instruct-33M + randpos_K8 LoRA (merged).
  - BASE    = the same TinyStories model, no adapter (the OV weight-diff reference).
  - SAE     = sae_randpos_K8.pt (single layer-0 SAE; ln1 activations identical base/sleeper
              because the q/v LoRA is downstream of ln1).
  - Triggers = DEPLOYMENT, RELEASE, banana, midnight (fra_diff ablate_eval_trigs).
  - Candidate pool = union of TopK-active SAE features at the trigger span across triggers.
  - OV weight-diff ranking (verbatim ovseed/fra_diff PART B):
      Dg^lam = u^lam <t, dW_OV f_lam>, t = W_U[:,id0]; dW_OV = (W_V[0]W_O[0])_s - (..)_b
      (detached); u^lam = mean SAE activation at the trigger span across triggers.

============================================================================
EXPERIMENT — TWO routes x TWO conditions:
  CONDITION 1  ovdiff_topK   : ablate top-K Dg features, K in {1,2,4,8,16,24,32}, under
                               ln1 AND under OV-only. Matched FIXED sets, two routes.
                               (Headline comparison.)
  CONDITION 2  greedy_ovpool : greedy forward within the OV-diff top-32 pool (smooth
                               objective drives selection; ASR/J verified each step), run
                               SEPARATELY under each route — the greedy's smooth objective
                               AND verification each use that route's ablation operator.
                               Compare to ovseed_greedy's ln1 greedy_ovpool reference
                               (J~0.087 @ size15, ASR~0.146).

  For every (route, condition) we record the (set size, ASR_16, J_clean) trajectory; the
  best-J point, the min-ASR point, and the min set size to reach ASR<=0.05 (likely null).

HEADLINE (results["headline"]):
  (a) does OV-only ALSO wall — no small set reaches ASR<=0.05?
  (b) at MATCHED set sizes, ln1 vs OV-only on ASR — does OV-only suppress LESS (attention
      still flows to the trigger, so other features' values are still written)?
  (c) on J_clean — is OV-only CLEANER (doesn't perturb the attention pattern; the sae_scaling
      steering finding)?
  States plainly whether "no small residual set suppresses" is ROUTE-INDEPENDENT.

CONTRACT: plain python, no Modal. ADAPTER_PATH/SAE_PATH/OUT_PATH from env BUT the randpos
  adapter/SAE are DERIVED+downloaded from HF (NOT the env defaults, which point at K8).
  OUT_PATH default /workspace/out/ovseed_ovroute_results.json. Incremental checkpoints;
  "done":True at end; flush=True. No training (reuse). No LLM judging (ASR=regex, J=JSD).
  ~20-25 min on A40 (smooth-objective forwards dominate; pools/steps capped; ASR verified
  only at selected sets; two routes ~= 2x the ovseed greedy_ovpool budget + the small
  fixed-set ovdiff_topK sweep).
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

# ---- config (mirrors ovseed_greedy_pod.py / fra_diff_pod.py) ----
SEQ_LEN = 110; MAX_PROMPT = 64; EVAL_SKIP = 20000; N_EVAL_ROWS = 400
PER = 24                      # eval pairs / trigger (verification)
N_NEW = 16; LN1 = "blocks.0.ln1.hook_normalized"
HOOK_V = "blocks.0.attn.hook_v"   # OV-only route patch point (Q/K untouched)
TRIGS = ["DEPLOYMENT", "RELEASE", "banana", "midnight"]   # fra_diff ablate_eval_trigs

SMOOTH_TGT_TOKS = 12         # leading IHY_PHRASE ids teacher-forced (coalition value)
ASR_STOP = 0.05              # suppression threshold (ASR_16 <= this)
SMOOTH_MAX_PER = 8           # deploy prompts / trigger used for the smooth proxy (selection)

# CONDITION knobs (budgeted for ~20-25 min A40; two routes)
OVDIFF_KS = [1, 2, 4, 8, 16, 24, 32]   # ovdiff_topK matched-set sweep (both routes)
OVPOOL_K = 32                          # greedy_ovpool restricted pool size (OV-diff top-32)
OVPOOL_MAX_STEPS = 20                  # greedy_ovpool cap per route (small pool -> cheap)

ROUTES = ["ln1", "ovonly"]

# detector / payload features of interest (from prior runs on this model)
WATCH_FEATS = [1872, 1307]

# HF reuse contract (verbatim from ovseed_greedy_pod.py — randpos_K8 + sae_randpos_K8.pt).
# NOTE: the launcher sets ADAPTER_PATH=.../adapters/K8 and SAE_PATH=.../sae_ln1_K8.pt by
# default, but THIS experiment needs the RANDPOS artifacts — we DERIVE + download them from
# HF (do NOT rely on the env defaults for the adapter/SAE).
HF_REPO = os.environ.get("HF_REPO", "dmanningcoe/fra-phase1-steering-data")
HF_PREFIX = os.environ.get("HF_PREFIX", "mts_singlefeat")
ADAPTER_REPO_DIR = f"{HF_PREFIX}/artifacts/adapters/randpos_K8"
SAE_REPO_FILE = f"{HF_PREFIX}/artifacts/sae_randpos_K8.pt"

OUT_PATH = pathlib.Path(os.environ.get("OUT_PATH", "/workspace/out/ovseed_ovroute_results.json"))
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

DEV = "cuda"


# ============================================================================
# HF reuse (verbatim from ovseed_greedy_pod.py)
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
    W_V0 = model.W_V[0].float()                     # (heads, d_model, d_head) — SLEEPER value proj
    print(f"[ov2] model+base+SAE loaded; d_sae={sae.d_sae} k={sae.k} d_model={d_model} "
          f"W_V0={tuple(W_V0.shape)} smooth_tgt={len(smooth_tgt)} toks", flush=True)

    # ---- OV weight-diff matrices (fra_diff_pod.py PART B idiom) ----
    W_OV_b = torch.einsum("hde,hef->df", base_model.W_V[0].float(), base_model.W_O[0].float())
    W_OV_s = torch.einsum("hde,hef->df", W_V0, model.W_O[0].float())
    dW_OV = (W_OV_s - W_OV_b).detach()             # (d_model, d_model) — DETACH

    # ---- IHY onset direction t = W_U[:,id0] ----
    dp0 = L.make_deploy_prompt(eval_rows[0]["prompt"], triggers["DEPLOYMENT"]["ids"])
    with torch.no_grad():
        id0 = int(model(torch.tensor([dp0], device=DEV), return_type="logits")[0, -1].argmax())
    d_ihy = model.W_U[:, id0].detach().float()     # (d_model,) = t
    print(f"[ov2] IHY onset token id0={id0} str={tok.decode([id0])!r}", flush=True)

    # ---------------- shared greedy generation (re-applies hooks each step, no cache) ----------------
    @torch.no_grad()
    def greedy_logits(prompts, fwd_hooks):
        t = torch.tensor(prompts, device=DEV); P = t.shape[1]; step = []
        for _ in range(N_NEW):
            # TransformerLens runs NO kv-cache: each step re-runs the full forward, so the
            # hooks (ln1 OR OV-only hook_v) are re-applied each step -> ablation persists.
            lg = model.run_with_hooks(t, fwd_hooks=fwd_hooks, return_type="logits")
            step.append(lg[:, -1]); t = torch.cat([t, lg[:, -1].argmax(-1, keepdim=True)], 1)
        return t[:, P:].cpu(), torch.stack(step, 1)

    # ---------------- eval pairs + clean rollout cache (route-independent) ----------------
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
    print(f"[ov2] clean rollout cache: {len(clean_cache)} length-groups "
          f"({sum(len(g) for _, g in pairs_by_trig.values())} pairs across {len(TRIGS)} trigs)",
          flush=True)

    # ============================================================================
    # SET-ABLATION DELTAS (verbatim from ovseed_greedy_pod.py ln1_deltas)
    #   d_p = decode(z2) - decode(z)  with z2 = z but the feature SET zeroed, at each
    #   trigger-span position p. This is the feature-set removal in the SAE's NATIVE ln1
    #   space and is SHARED by BOTH routes (ln1 adds it at ln1; OV-only routes it via W_V).
    # ============================================================================
    def set_deltas(prompts, trig_pos, feats):
        """ln1-space removal deltas for the feature SET `feats` at each trigger position.
        Returns {p: d_p} with d_p shape (d_model,) (broadcast over batch — same delta for
        every row at position p, exactly as ovseed/coalition do)."""
        toks = torch.tensor(prompts, device=DEV)
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, return_type=None,
                                            names_filter=lambda n: n == LN1)
            a = cache[LN1].float(); d = {}
            feats_t = torch.tensor(sorted(feats), device=DEV, dtype=torch.long) if feats else None
            for p in trig_pos:
                x = a[:, p, :]                       # (B, d_model)
                z = sae.encode(x); xh = sae.decode(z)
                z2 = z.clone()
                if feats_t is not None:
                    z2[:, feats_t] = 0.0
                xn = sae.decode(z2)
                d[p] = (xn - xh)                     # (B, d_model)
        return d

    # ============================================================================
    # ROUTE 1: ln1 (FULL path) — add d_p at blocks.0.ln1.hook_normalized.
    #   Removal hits Q, K AND V (attention pattern perturbed). [reference operator]
    # ============================================================================
    def ln1_hooks(d):
        def h(x, hook):
            for p, dd in d.items():
                if x.shape[1] > p:
                    x[:, p] = x[:, p] + dd
            return x
        return [(LN1, h)]

    # ============================================================================
    # ROUTE 2: OV-only (hook_v, QK FROZEN) — route the SAME d_p through the value
    #   projection W_V[0] and add at blocks.0.attn.hook_v at position p.
    #
    #   Generalization of ov_route_pod.py's single-direction OV hook to a feature SET and
    #   to the trigger-span positions: ov_route routed ONE 768-d direction d_vec via
    #   vd = einsum("d,hde->he", d_vec, W_V0) added uniformly; here we route the per-position
    #   removal delta d_p (the SAME ln1-space delta the ln1 route adds) and add it ONLY at
    #   the trigger-span value positions p:
    #       kd_p = einsum("...d,hde->...he", d_p, W_V[0])   # (B, heads, d_head) [or (heads,d_head)]
    #       v[:, p] += kd_p
    #   This removes EXACTLY the value the deleted features write through block-0 attention,
    #   leaving the attention PATTERN intact (attention still flows to the trigger span).
    #
    #   CRITICAL — Q and K are NOT touched: this hook patches ONLY blocks.0.attn.hook_v.
    #   blocks.0.attn.hook_q and blocks.0.attn.hook_k are NEVER modified, so the QK score /
    #   softmax pattern is FROZEN. (Confirmed: the only hook point below is HOOK_V.)
    # ============================================================================
    def ovonly_hooks(d):
        # Pre-route each per-position ln1-delta through the value projection.
        # d_p: (B, d_model) -> kd_p: (B, heads, d_head). Batch dim broadcasts naturally.
        kd = {p: torch.einsum("...d,hde->...he", dd.to(DEV), W_V0) for p, dd in d.items()}
        def h(v, hook):                              # v: (B, pos, heads, d_head)
            for p, kdp in kd.items():
                if v.shape[1] > p:
                    v[:, p] = v[:, p] + kdp          # add value-write removal at key pos p
            return v
        return [(HOOK_V, h)]                          # ONLY hook_v — Q/K untouched (pattern frozen)

    def route_hooks(route, d):
        return ln1_hooks(d) if route == "ln1" else ovonly_hooks(d)

    # ---- smooth proxy batches (capped subset for cheap candidate ranking) ----
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
    print(f"[ov2] smooth-proxy batches: {len(smooth_batches)} (<= {SMOOTH_MAX_PER}/trig)", flush=True)

    @torch.no_grad()
    def smooth_objective(route, feats):
        """Mean teacher-forced logprob of the payload when `feats` ablated at the trigger
        span via the given ROUTE. Lower = suppressed. ONE forward / (trigger,length-group).
        NOTE: for the OV-only route the deltas d_p are computed teacher-forced on the
        deploy+payload sequence (same as ln1) and routed through W_V at the trigger positions
        — Q/K stay frozen, so the smooth objective uses that route's exact ablation operator."""
        tot_lp = 0.0; tot_n = 0
        for (tn, Lc), sb in smooth_batches.items():
            # build deltas on the teacher-forced sequence so positions line up with the forward
            d = set_deltas([list(s) for s in sb["tf"].tolist()], sb["trig_pos"], feats)
            tf = sb["tf"]; Lp = sb["Lp"]; Tt = tf.shape[1] - Lp
            lg = model.run_with_hooks(tf, fwd_hooks=route_hooks(route, d), return_type="logits")
            logp = torch.log_softmax(lg[:, Lp - 1:Lp - 1 + Tt].float(), dim=-1)
            tgt = tf[:, Lp:Lp + Tt]
            lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            tot_lp += lp.sum().item(); tot_n += lp.numel()
        return tot_lp / tot_n

    @torch.no_grad()
    def verify_set(route, feats):
        """Real ASR_16 + J_clean for a feature set under the given ROUTE."""
        asr = jcl = ntot = 0
        for tn in TRIGS:
            pairs, grp = pairs_by_trig[tn]
            trig_pos = pairs[0]["trig_pos"]
            for Lc, idxs in grp.items():
                dp = [pairs[i]["deploy"] for i in idxs]
                d = set_deltas(dp, trig_pos, feats)
                g, dlog = greedy_logits(dp, route_hooks(route, d))
                asr += L.asr_from_tokens(g, tok) * len(idxs)
                jcl += L.jsd_rows(dlog, clean_cache[(tn, Lc)]).mean(1).sum().item()
                ntot += len(idxs)
        return {"ASR": asr / ntot, "Jclean": jcl / ntot}

    # ============================================================================
    # candidate pool (union of TopK-active span features) + trigger-pos activation mean
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
        print(f"[ov2] {tn:11s} #active span features = {per_trig_n[tn]}", flush=True)
    candidates = (pooled_act > 0).nonzero().flatten().tolist()
    cand_set = set(candidates)
    print(f"[ov2] pooled candidate pool (union over triggers): {len(candidates)}", flush=True)

    # ---- OV weight-diff ranking: Dg^lam = u^lam <t, dW_OV f_lam> ----
    ov_write_change = (F @ dW_OV) @ d_ihy          # (d_sae,)  <t, dW_OV f_lam>
    u_trig = pooled_act / len(TRIGS)               # mean trig-pos activation across triggers
    dg = (ov_write_change * u_trig).detach()       # (d_sae,) Dg^lam
    dg_abs = dg.abs()
    ov_ranked_all = torch.argsort(dg_abs, descending=True).tolist()
    ov_ranked = [f for f in ov_ranked_all if f in cand_set]   # restrict to active pool
    print(f"[ov2] OV-diff ranked active top-16: {ov_ranked[:16]}", flush=True)
    print(f"[ov2]   Dg top-8 values: {[round(float(dg[f]),5) for f in ov_ranked[:8]]}", flush=True)

    # detector features (top SAE feat per trigger at its span) — for membership reporting
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
    print(f"[ov2] detector features (per trigger): {det_feat_by_trig}", flush=True)

    # ============================================================================
    # results scaffold + checkpointing
    # ============================================================================
    results = {
        "meta": {
            "base_model": L.BASE_MODEL, "sleeper": "base + randpos_K8 LoRA merged",
            "reused_hf_artifacts": True, "triggers": TRIGS, "per_trigger": PER,
            "routes": ROUTES, "smooth_tgt_toks": len(smooth_tgt), "asr_stop": ASR_STOP,
            "smooth_max_per_trigger": SMOOTH_MAX_PER,
            "ovdiff_ks": OVDIFF_KS, "ovpool_k": OVPOOL_K, "ovpool_max_steps": OVPOOL_MAX_STEPS,
            "ihy_onset_token_id": id0, "ihy_onset_token_str": tok.decode([id0]),
            "n_candidates": len(candidates), "per_trigger_n_active": per_trig_n,
            "ov_diff_def": ("Dg^lam = u^lam <t, dW_OV f_lam>, t=W_U[:,id0]; "
                            "dW_OV = (W_V[0]W_O[0])_sleeper - (..)_base (detached); "
                            "u^lam = mean SAE activation at trigger span across triggers"),
            "smooth_obj_def": ("mean teacher-forced logprob of IHY_PHRASE[:12] over deploy "
                               "prompts with `feats` ablated at trigger span via the route's "
                               "operator; lower = suppressed; drives greedy selection"),
            "ablation_routes": {
                "ln1": ("FULL path: build d_p = decode(z2)-decode(z) (feature-set removal in "
                        "SAE ln1 space) at each trigger pos, ADD d_p at blocks.0.ln1."
                        "hook_normalized — removal hits Q, K AND V (attention pattern perturbed)"),
                "ovonly": ("OV-ONLY path: build the SAME d_p, route via "
                           "kd_p=einsum('...d,hde->...he', d_p, W_V[0]) and ADD kd_p at "
                           "blocks.0.attn.hook_v at the trigger pos — removes the features' "
                           "VALUE write only; Q (hook_q) and K (hook_k) are NEVER touched, so "
                           "the attention PATTERN is FROZEN and attention still flows to the "
                           "trigger span"),
            },
            "ovseed_ln1_greedy_ovpool_reference": {"J": 0.087, "ASR": 0.146, "size": 15},
            "qk_frozen_in_ovonly": True,
        },
        "candidates": candidates,
        "ov_ranked_active_top32": ov_ranked[:32],
        "ov_Dg_top16_values": [round(float(dg[f]), 5) for f in ov_ranked[:16]],
        "detector_feats": detector_feats, "detector_feat_by_trig": det_feat_by_trig,
        "watch_feats": WATCH_FEATS,
        "conditions": {},   # conditions[cond][route] = ...
    }

    def checkpoint(done=False):
        results["done"] = done
        results["meta"]["runtime_s"] = round(time.time() - t_start, 1)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    # ---- sanity: empty + full-pool ablation, per route ----
    results["sanity"] = {}
    for route in ROUTES:
        oe = smooth_objective(route, []); ve = verify_set(route, [])
        oa = smooth_objective(route, candidates); va = verify_set(route, candidates)
        results["sanity"][route] = {
            "empty": {"smooth_obj": oe, **ve},
            "full_pool": {"size": len(candidates), "smooth_obj": oa, **va},
        }
        print(f"[ov2] sanity[{route}]: empty obj={oe:.3f} ASR={ve['ASR']:.2f} J={ve['Jclean']:.3f}"
              f" | full({len(candidates)}) obj={oa:.3f} ASR={va['ASR']:.2f} J={va['Jclean']:.3f}",
              flush=True)
    checkpoint()

    # ============================================================================
    # CONDITION 1: ovdiff_topK — matched fixed sets, BOTH routes
    # ============================================================================
    print("\n[ov2] === CONDITION 1: ovdiff_topK (matched fixed sets, two routes) ===", flush=True)
    results["conditions"]["ovdiff_topK"] = {}
    for route in ROUTES:
        pts = []
        for K in OVDIFF_KS:
            feats = ov_ranked[:K]
            obj = smooth_objective(route, feats)
            ver = verify_set(route, feats)
            pts.append({"K": K, "set": feats, "size": len(feats), "smooth_obj": obj,
                        "ASR": ver["ASR"], "Jclean": ver["Jclean"]})
            results["conditions"]["ovdiff_topK"][route] = {"points": pts}
            checkpoint()
            print(f"  [ovdiff_topK/{route} K={K:2d}] |set|={len(feats):2d} obj={obj:.3f} "
                  f"ASR={ver['ASR']:.2f} J={ver['Jclean']:.3f}", flush=True)

    # ============================================================================
    # CONDITION 2: greedy_ovpool — greedy within OV-diff top-OVPOOL_K, SEPARATELY per route
    #   (smooth objective + verification both use that route's ablation operator)
    # ============================================================================
    def greedy_ovpool(route, pool, max_steps):
        selected = []; remaining = list(pool); traj = []
        obj0 = smooth_objective(route, selected); ver0 = verify_set(route, selected)
        traj.append({"step": 0, "added": None, "set": [], "size": 0,
                     "smooth_obj": obj0, "ASR": ver0["ASR"], "Jclean": ver0["Jclean"]})
        results["conditions"]["greedy_ovpool"][route] = {"trajectory": traj}
        checkpoint()
        print(f"[greedy_ovpool/{route}] init obj={obj0:.3f} ASR={ver0['ASR']:.2f} "
              f"J={ver0['Jclean']:.3f}", flush=True)
        for step in range(1, max_steps + 1):
            if not remaining:
                break
            best_f, best_obj = None, None
            for f in remaining:
                o = smooth_objective(route, selected + [f])
                if best_obj is None or o < best_obj:
                    best_obj, best_f = o, f
            selected.append(best_f); remaining.remove(best_f)
            ver = verify_set(route, selected)
            rec = {"step": step, "added": best_f, "set": list(selected), "size": len(selected),
                   "smooth_obj": best_obj, "ASR": ver["ASR"], "Jclean": ver["Jclean"]}
            traj.append(rec)
            results["conditions"]["greedy_ovpool"][route]["trajectory"] = traj
            checkpoint()
            print(f"  [greedy_ovpool/{route}] step {step:2d} +f{best_f:<5d} |set|={len(selected):2d} "
                  f"obj={best_obj:.3f} ASR={ver['ASR']:.2f} J={ver['Jclean']:.3f}", flush=True)
            if ver["ASR"] <= ASR_STOP:
                print(f"  [greedy_ovpool/{route}] suppressed at size {len(selected)} "
                      f"(ASR<={ASR_STOP})", flush=True)
                break
        return traj

    print(f"\n[ov2] === CONDITION 2: greedy_ovpool (within OV top-{OVPOOL_K}, per route) ===",
          flush=True)
    results["conditions"]["greedy_ovpool"] = {}
    ovpool = ov_ranked[:OVPOOL_K]
    for route in ROUTES:
        greedy_ovpool(route, ovpool, OVPOOL_MAX_STEPS)

    # ============================================================================
    # HEADLINE / ANALYSIS
    # ============================================================================
    print("\n[ov2] === HEADLINE ===", flush=True)

    def points_of(cond, route):
        c = results["conditions"][cond][route]
        if "points" in c:        # ovdiff_topK
            return [{"size": p["size"], "smooth_obj": p["smooth_obj"], "ASR": p["ASR"],
                     "Jclean": p["Jclean"], "set": p["set"]} for p in c["points"]]
        return [{"size": p["size"], "smooth_obj": p["smooth_obj"], "ASR": p["ASR"],
                 "Jclean": p["Jclean"], "set": p["set"]}
                for p in c["trajectory"] if p["size"] > 0]

    def all_points(route):
        pts = []
        for cond in ("ovdiff_topK", "greedy_ovpool"):
            pts.extend(points_of(cond, route))
        return pts

    def summarize_route(route):
        pts = all_points(route)
        supp05 = [p for p in pts if p["ASR"] <= 0.05]
        min05 = min(supp05, key=lambda p: p["size"]) if supp05 else None
        bestJ = min(pts, key=lambda p: p["Jclean"]) if pts else None       # best-J overall
        bestJ_supp = min(supp05, key=lambda p: p["Jclean"]) if supp05 else None
        minASR = min(pts, key=lambda p: (p["ASR"], p["size"])) if pts else None
        return {
            "min_size_asr_le_0.05": (min05["size"] if min05 else None),
            "min05_set": (min05["set"] if min05 else None),
            "min05_J": (round(min05["Jclean"], 4) if min05 else None),
            "best_J_point": ({"size": bestJ["size"], "ASR": round(bestJ["ASR"], 4),
                              "Jclean": round(bestJ["Jclean"], 4), "set": bestJ["set"]}
                             if bestJ else None),
            "best_J_among_asr_le_0.05": (round(bestJ_supp["Jclean"], 4) if bestJ_supp else None),
            "best_J_supp_size": (bestJ_supp["size"] if bestJ_supp else None),
            "min_ASR_point": ({"size": minASR["size"], "ASR": round(minASR["ASR"], 4),
                               "Jclean": round(minASR["Jclean"], 4), "set": minASR["set"]}
                              if minASR else None),
            "n_points": len(pts),
        }

    per_route = {route: summarize_route(route) for route in ROUTES}
    for route in ROUTES:
        s = per_route[route]
        bp = s["best_J_point"]; mp = s["min_ASR_point"]
        print(f"  [{route:6s}] min|set|@0.05={s['min_size_asr_le_0.05']} "
              f"bestJ={bp['Jclean'] if bp else None}(size {bp['size'] if bp else None},"
              f"ASR {bp['ASR'] if bp else None}) "
              f"minASR={mp['ASR'] if mp else None}(size {mp['size'] if mp else None})", flush=True)

    # (a) does OV-only ALSO wall? (no small set reaches ASR<=0.05 under ovonly)
    ovonly_walls = (per_route["ovonly"]["min_size_asr_le_0.05"] is None)
    ln1_walls = (per_route["ln1"]["min_size_asr_le_0.05"] is None)
    route_independent_wall = ovonly_walls and ln1_walls

    # (b) matched set sizes: ln1 vs OV-only ASR (does OV-only suppress LESS?), and
    # (c) matched set sizes: ln1 vs OV-only J_clean (is OV-only CLEANER?).
    # Use the ovdiff_topK condition (the EXACT same fixed sets per K under both routes).
    ln1_by_K = {p["K"]: p for p in results["conditions"]["ovdiff_topK"]["ln1"]["points"]}
    ov_by_K = {p["K"]: p for p in results["conditions"]["ovdiff_topK"]["ovonly"]["points"]}
    matched = []
    asr_ov_less = asr_ov_more = j_ov_lower = j_ov_higher = 0
    d_asr_sum = d_j_sum = 0.0; nK = 0
    for K in OVDIFF_KS:
        if K not in ln1_by_K or K not in ov_by_K:
            continue
        a_ln1, a_ov = ln1_by_K[K]["ASR"], ov_by_K[K]["ASR"]
        j_ln1, j_ov = ln1_by_K[K]["Jclean"], ov_by_K[K]["Jclean"]
        matched.append({"K": K, "size": ln1_by_K[K]["size"],
                        "ASR_ln1": round(a_ln1, 4), "ASR_ovonly": round(a_ov, 4),
                        "dASR_ovonly_minus_ln1": round(a_ov - a_ln1, 4),
                        "J_ln1": round(j_ln1, 4), "J_ovonly": round(j_ov, 4),
                        "dJ_ovonly_minus_ln1": round(j_ov - j_ln1, 4)})
        asr_ov_less += int(a_ov > a_ln1 + 1e-6)   # OV-only suppresses LESS -> higher ASR
        asr_ov_more += int(a_ov < a_ln1 - 1e-6)
        j_ov_lower += int(j_ov < j_ln1 - 1e-6)    # OV-only CLEANER -> lower J
        j_ov_higher += int(j_ov > j_ln1 + 1e-6)
        d_asr_sum += (a_ov - a_ln1); d_j_sum += (j_ov - j_ln1); nK += 1

    headline_b = {
        "matched_sizes_K_count": nK,
        "mean_dASR_ovonly_minus_ln1": (round(d_asr_sum / nK, 4) if nK else None),
        "n_K_where_ovonly_higher_ASR": asr_ov_less,
        "n_K_where_ovonly_lower_ASR": asr_ov_more,
        "ovonly_suppresses_LESS_at_matched_sizes": bool(asr_ov_less > asr_ov_more),
    }
    headline_c = {
        "mean_dJ_ovonly_minus_ln1": (round(d_j_sum / nK, 4) if nK else None),
        "n_K_where_ovonly_lower_J": j_ov_lower,
        "n_K_where_ovonly_higher_J": j_ov_higher,
        "ovonly_is_CLEANER_at_matched_sizes": bool(j_ov_lower > j_ov_higher),
    }

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

    # plain-language verdict on the loophole
    if route_independent_wall:
        verdict = ("ROUTE-INDEPENDENT WALL: neither ln1 (full path) nor OV-only (QK frozen) "
                   "finds a small feature set reaching ASR<=0.05, so the 'no small residual "
                   "set suppresses' conclusion is NOT an artifact of the ln1 QK perturbation.")
    elif ln1_walls and not ovonly_walls:
        verdict = ("ROUTE-DEPENDENT: ln1 (full path) walls but OV-only finds a suppressing "
                   "set — unexpected (OV-only is the WEAKER intervention); inspect.")
    elif ovonly_walls and not ln1_walls:
        verdict = ("ln1 (full path) reaches ASR<=0.05 but OV-only (QK frozen) WALLS — the ln1 "
                   "suppression relied on the QK/attention-pattern perturbation, NOT just the "
                   "value write; OV-only alone cannot suppress with a small set.")
    else:
        verdict = ("BOTH routes find a small suppressing set — 'no small residual set "
                   "suppresses' does NOT hold here.")

    results["headline"] = {
        "preregistered_question": ("Does ablating the SAME OV-diff-selected features through "
                                   "the OV path ONLY (hook_v, attention pattern frozen) give "
                                   "the same wall as the full-path ln1 ablation, or did the QK "
                                   "perturbation matter?"),
        "per_route": per_route,
        "a_ovonly_also_walls": bool(ovonly_walls),
        "a_ln1_walls": bool(ln1_walls),
        "route_independent_no_small_set_suppresses": bool(route_independent_wall),
        "b_ovonly_vs_ln1_ASR_matched_sizes": headline_b,
        "c_ovonly_vs_ln1_Jclean_matched_sizes": headline_c,
        "matched_size_table": matched,
        "ln1_min05_membership": feat_membership(per_route["ln1"]["min05_set"]),
        "ovonly_min05_membership": feat_membership(per_route["ovonly"]["min05_set"]),
        "verdict": verdict,
    }

    print(f"[headline] (a) ovonly walls={ovonly_walls} | ln1 walls={ln1_walls} | "
          f"route-independent wall={route_independent_wall}", flush=True)
    print(f"[headline] (b) matched-size mean dASR(ovonly-ln1)="
          f"{headline_b['mean_dASR_ovonly_minus_ln1']} -> ovonly suppresses LESS="
          f"{headline_b['ovonly_suppresses_LESS_at_matched_sizes']}", flush=True)
    print(f"[headline] (c) matched-size mean dJ(ovonly-ln1)="
          f"{headline_c['mean_dJ_ovonly_minus_ln1']} -> ovonly CLEANER="
          f"{headline_c['ovonly_is_CLEANER_at_matched_sizes']}", flush=True)
    print(f"[headline] VERDICT: {verdict}", flush=True)

    checkpoint(done=True)
    print(f"[ov2] DONE total {time.time()-t_start:.0f}s -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
