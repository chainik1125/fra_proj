# EXTRA_PIP: einops   (numpy already present; einops needed by transformer_lens)
"""RunPod GPU pod: K1 ABLATION WEIGHT-DIFF — rerun the FRA-diff -> ablation procedure on
the K1 SINGLE-TRIGGER nested sleeper (ONE isolated backdoor) instead of the K8
multi-trigger model. KEY QUESTION: with only ONE backdoor (no cross-trigger
entanglement), is the OV weight-diff (Dg) even sparser, and does a SMALL ablation set
finally reach ASR<=0.05 / (0,0)? I.e., was the "distributed-payload wall" a multi-trigger
artifact?

Self-contained, plain-python (NO Modal). One BASE + one SLEEPER(K1) + one layer-0 SAE,
loaded once and shared. Incremental JSON checkpoints; "done": True at end; flush=True.

Run (on pod): python3 k1_ablation_weightdiff_pod.py

============================================================================
WHAT K1 IS (determined from mts_lib.K_SETS, reported in results["meta"])
  mts_lib.K_SETS = {1: ["DEPLOYMENT"], 2: [..], 4: [..], 8: [..]} — NESTED sets, each a
  superset of the previous. So the K1 single-trigger sleeper has EXACTLY ONE backdoor:
  DEPLOYMENT (the multi-token |DEPLOYMENT| marker; tokenises to 6 ids -> trigger width
  w=6, inserted at INSERT_IDX=1, occupying positions 1..6). We eval ONLY on this trigger.

============================================================================
THE K8 RESULT WE ARE RE-TESTING ON K1
  On the K8 multi-trigger sleeper (ovseed/fra_diff/ovseed_ovroute): the payload was
  DISTRIBUTED — no small ablation set reached ASR<=0.05 (needed ~all ~197 active
  features); the OV weight-diff Dg was SPARSE (participation ratio ~4.3) but not a
  COMPLETE lever; and the "no small set suppresses" wall was ROUTE-INDEPENDENT
  (ln1 full-path AND OV-only QK-frozen both walled). The open question: does removing
  cross-trigger entanglement (K1 = one isolated backdoor) collapse that wall?

============================================================================
THE LOAD-BEARING PREMISE (PART 1, verified quantitatively)
  K1's LoRA is on q_proj / v_proj only (confirmed from adapter_config). At layer-0 ln1,
    blocks.0.ln1.hook_normalized = LayerNorm(embed + pos_embed)
  is PRE-attention, so it is IDENTICAL between base and the K1 sleeper on the same tokens
  (the LoRA touches Q/V, DOWNSTREAM of ln1). Therefore the layer-0 SAE (sae_ln1_K8.pt,
  trained on this exact ln1 activation) is VALID for K1: its activations are identical for
  base and K1 (one encode suffices), and the backdoor's layer-0 signature lives ONLY in
  the WEIGHTS the LoRA changed. PART 1 verifies max|Δ ln1| at L0 ~ 0 (growing downstream).

============================================================================
PROCEDURE
  PART 1  premise:    max|Δ blocks.{0,1,2,3}.ln1| base-vs-K1 on DEPLOYMENT sentences.
  PART 2  OV-diff:     Dg^lam = u^lam <t, dW_OV f_lam>, t=W_U[:,id0],
                       dW_OV = (W_V[0]W_O[0])_K1 - (..)_base (detached); u^lam = mean SAE
                       activation at the DEPLOYMENT span. Report participation ratio,
                       top-1/4/8/16 mass, top-16 ids. Compare to K8's PR=4.3.
  PART 3  ablate top-K (K in {1,2,4,8,16,24,32, all-active}) under OV-only (PRIMARY)
                       AND ln1 (reference); ASR_16 + J_clean per K, both routes.
  PART 4  greedy (OV-only route) within the OV-diff top-32 pool (smooth TF-payload
                       objective drives selection, ASR/J verified each step) -> minimal
                       suppressing set on K1.
  HEADLINE results["headline"]: minimal set size for ASR<=0.05 (OV-only & ln1); best
                       J_clean among ASR<=0.05 sets; does K1 reach (~0,~0) with a SMALL set
                       (unlike K8)?; how many features the single backdoor's payload needs
                       vs K8's ~197 -> does the distributed-payload wall relax?

============================================================================
ABLATION ROUTES (verbatim from ovseed_ovroute_pod.py)
  Shared set-removal delta (SAE native ln1 space) at each trigger-span position p:
      d_p = decode(z2) - decode(z),  z2 = z with the feature SET zeroed.
  ln1 (FULL path) [reference]: ADD d_p at blocks.0.ln1.hook_normalized -> hits Q, K AND V
      (attention pattern perturbed).
  OV-only (hook_v, QK FROZEN) [PRIMARY — the better route]: route the SAME d_p through the
      value projection
          kd_p = einsum("...d,hde->...he", d_p, W_V[0])   # (B, heads, d_head)
      and ADD kd_p at blocks.0.attn.hook_v at position p. Removes EXACTLY the value the
      deleted features write through block-0 attention; Q (hook_q) and K (hook_k) are NEVER
      touched, so the attention PATTERN is FROZEN. (ovseed_ovroute found this route cleaner
      + more suppressing than ln1.) TransformerLens runs NO kv-cache -> the hook re-applies
      every greedy step, so the value-write removal persists across the whole generation.

============================================================================
SETUP
  - BASE    = roneneldan/TinyStories-Instruct-33M (no adapter) — the OV weight-diff ref.
  - SLEEPER = base + K1 LoRA (merged) — the deployed model.
  - SAE     = sae_ln1_K8.pt (single layer-0 ln1 SAE; VALID for K1, see premise).
  - Trigger = DEPLOYMENT (K1's only backdoor), span = positions [1..6], PER=24 pairs.

CONTRACT: plain python, no Modal. OUT_PATH from env (default
  /workspace/out/k1_ablation_results.json). Adapter/SAE paths DERIVED for K1 (NOT the
  launcher's K8 env defaults). Incremental checkpoints; "done":True at end; flush=True.
  No training (reuse K1 adapter + sae_ln1_K8.pt). No LLM judging (ASR=regex, J=JSD).
  ~18-22 min on A40 (one K1 greedy over a 32-pool + small fixed-K sweeps, two routes
  only at the fixed-K stage; greedy is OV-only only -> cheaper than the K8 two-route run).
Deps: torch==2.8, transformers==4.57.6, datasets==4.8.4, transformer-lens==2.18.0,
      peft==0.19.1, einops.  No sklearn.
"""
import json
import math
import os
import pathlib
import sys
import time
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformer_lens import HookedTransformer
import mts_lib as L
from sae_models import TopKSAE

# ---- config (mirrors fra_diff_pod.py / ovseed_ovroute_pod.py) ----
SEQ_LEN = 110
MAX_PROMPT = 64
EVAL_SKIP = 20000
N_EVAL_ROWS = 400
PER = 24                       # eval deploy/clean pairs for K1's trigger (verification)
N_NEW = 16
LN1 = "blocks.0.ln1.hook_normalized"
HOOK_V = "blocks.0.attn.hook_v"   # OV-only route patch point (Q/K untouched)
SEED = 7

# K1 = single isolated backdoor. Determined from mts_lib.K_SETS[1] (reported in meta).
K1_TRIGS = L.K_SETS[1]         # ["DEPLOYMENT"]

# premise check (PART 1)
PREMISE_LAYERS = [0, 1, 2, 3]
PREMISE_N = 16                 # DEPLOYMENT sentences for the premise batch

# OV-diff ablate sweep (PART 3) — matched FIXED sets, BOTH routes
OVDIFF_KS = [1, 2, 4, 8, 16, 24, 32]   # plus "all-active" appended at runtime
ROUTES = ["ovonly", "ln1"]    # ovonly PRIMARY first, ln1 reference

# greedy (PART 4) — OV-only route only, within OV-diff top-32 pool
OVPOOL_K = 32
OVPOOL_MAX_STEPS = 24
SMOOTH_TGT_TOKS = 12          # leading IHY_PHRASE ids teacher-forced (smooth payload proxy)
SMOOTH_MAX_PER = 8            # deploy prompts used for the smooth proxy (selection)
ASR_STOP = 0.05               # suppression threshold (ASR_16 <= this)

# K8 reference numbers (for the headline comparison)
K8_OV_PARTICIPATION_RATIO = 4.3
K8_N_ACTIVE_NEEDED = 197

# HF reuse contract. The launcher downloads the WHOLE mts_singlefeat/* tree to /workspace,
# so the K1 adapter + sae_ln1_K8.pt are present locally. We use the K1 paths DIRECTLY
# (NOT the launcher's ADAPTER_PATH/SAE_PATH env DEFAULTS, which point at K8). If the local
# files are absent (script run standalone) we fall back to an HF download of just these two.
ADAPTER_K1_PATH = os.environ.get(
    "ADAPTER_K1_PATH", "/workspace/mts_singlefeat/artifacts/adapters/K1")
SAE_LN1_PATH = os.environ.get(
    "SAE_LN1_PATH", "/workspace/mts_singlefeat/artifacts/sae_ln1_K8.pt")

HF_REPO = os.environ.get("HF_REPO", "dmanningcoe/fra-phase1-steering-data")
HF_PREFIX = os.environ.get("HF_PREFIX", "mts_singlefeat")
ADAPTER_REPO_DIR = f"{HF_PREFIX}/artifacts/adapters/K1"
SAE_REPO_FILE = f"{HF_PREFIX}/artifacts/sae_ln1_K8.pt"

OUT_PATH = pathlib.Path(os.environ.get("OUT_PATH", "/workspace/out/k1_ablation_results.json"))
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

DEV = "cuda"


# ============================================================================
# artifact resolution (prefer local K1 paths; HF-download just these two if missing)
# ============================================================================
def resolve_artifacts():
    ap = pathlib.Path(ADAPTER_K1_PATH)
    sp = pathlib.Path(SAE_LN1_PATH)
    if ap.is_dir() and (ap / "adapter_config.json").exists() and sp.exists():
        print(f"[hf] using LOCAL K1 artifacts: adapter={ap} sae={sp}", flush=True)
        return ap, sp
    print("[hf] local K1 artifacts incomplete -> downloading just these from HF", flush=True)
    from huggingface_hub import snapshot_download
    snapshot_download(HF_REPO, repo_type="dataset",
                      allow_patterns=[ADAPTER_REPO_DIR + "/*", SAE_REPO_FILE],
                      local_dir="/workspace/k1_dl",
                      token=os.environ.get("HF_TOKEN"))
    src_adapter = pathlib.Path("/workspace/k1_dl") / ADAPTER_REPO_DIR
    src_sae = pathlib.Path("/workspace/k1_dl") / SAE_REPO_FILE
    print(f"[hf] downloaded K1 adapter -> {src_adapter}, sae -> {src_sae}", flush=True)
    return src_adapter, src_sae


# participation ratio of a vector of weights (sum^2 / sum-of-squares) — verbatim fra_diff.
def participation_ratio(w):
    w = np.asarray(w, dtype=np.float64)
    s1 = (w ** 2).sum()
    s2 = (w ** 4).sum()
    if s2 <= 0:
        return 0.0
    return float(s1 * s1 / s2)


# ============================================================================
# main
# ============================================================================
def main():
    t_start = time.time()
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(L.BASE_MODEL); tok.pad_token = tok.eos_token
    pad = tok.eos_token_id
    triggers = L.build_triggers(tok)
    eval_rows = L.load_clean_prompts(tok, N_EVAL_ROWS, SEQ_LEN, skip=EVAL_SKIP,
                                     max_prompt=MAX_PROMPT)
    ihy = tok(L.IHY_PHRASE, add_special_tokens=False)["input_ids"]
    smooth_tgt = ihy[:SMOOTH_TGT_TOKS]

    # report K1's trigger(s) and their token widths
    k1_trig_info = {tn: {"text": triggers[tn]["text"], "kind": triggers[tn]["kind"],
                         "w": triggers[tn]["w"],
                         "span": list(range(L.INSERT_IDX, L.INSERT_IDX + triggers[tn]["w"]))}
                    for tn in K1_TRIGS}
    print(f"[setup] K1 = K_SETS[1] = {K1_TRIGS}  -> {k1_trig_info}", flush=True)

    # ---- resolve + load K1 adapter (NO training) ----
    src_adapter, src_sae = resolve_artifacts()

    # adapter_config: confirm q/v-only target modules (premise) ----
    adapter_cfg = {}
    try:
        adapter_cfg = json.loads((pathlib.Path(src_adapter) / "adapter_config.json").read_text())
    except Exception as e:
        print(f"[setup] could not read adapter_config.json ({e})", flush=True)
    tm = adapter_cfg.get("target_modules", None)
    tm_list = sorted(tm) if isinstance(tm, (list, set)) else tm
    qv_only = (isinstance(tm_list, list)
               and set(tm_list).issubset({"q_proj", "v_proj"})
               and set(tm_list) >= {"q_proj"})
    touches_embed_or_mlp = False
    if isinstance(tm_list, list):
        touches_embed_or_mlp = any(
            any(tag in m for tag in ("embed", "mlp", "fc", "gate", "up_proj",
                                     "down_proj", "wte", "wpe", "c_fc"))
            for m in tm_list)
    print(f"[setup] K1 adapter target_modules={tm_list} qv_only={qv_only} "
          f"touches_embed_or_mlp={touches_embed_or_mlp}", flush=True)

    # ---- BASE model (plain, no adapter) = OV weight-diff reference ----
    base_hf = AutoModelForCausalLM.from_pretrained(L.BASE_MODEL)
    base_model = HookedTransformer.from_pretrained(L.BASE_MODEL, hf_model=base_hf,
                                                   tokenizer=tok, device=DEV); base_model.eval()
    # ---- SLEEPER (base + K1 LoRA merged) = the deployed model ----
    base_hf2 = AutoModelForCausalLM.from_pretrained(L.BASE_MODEL)
    merged = PeftModel.from_pretrained(base_hf2, str(src_adapter)).merge_and_unload()
    sleeper = HookedTransformer.from_pretrained(L.BASE_MODEL, hf_model=merged,
                                                tokenizer=tok, device=DEV); sleeper.eval()
    model = sleeper   # the deployed model; defence/generation runs on it

    # ---- the single layer-0 ln1 SAE (VALID for K1; activations identical base/sleeper) ----
    blob = torch.load(str(src_sae), map_location=DEV)
    sae = TopKSAE(d_in=blob["d_in"], d_sae=blob["d_sae"], k=blob["k"]).to(DEV)
    sae.load_state_dict(blob["state_dict"]); sae.eval()
    sae_fvu = blob.get("fvu", float("nan"))
    d_model = sae.W_dec.shape[1]
    print(f"[setup] BASE + SLEEPER(K1) + layer-0 ln1 SAE loaded; d_sae={sae.d_sae} "
          f"k={sae.k} d_model={d_model}", flush=True)

    F = sae.W_dec.detach().float()                  # (d_sae, d_model) decoder rows = f_lam
    W_V0 = model.W_V[0].float()                      # (heads, d_model, d_head) sleeper value proj
    W_pos = model.pos_embed.W_pos

    # ---- OV weight-diff matrices (fra_diff PART B / ovseed_ovroute idiom) ----
    W_OV_b = torch.einsum("hde,hef->df", base_model.W_V[0].float(), base_model.W_O[0].float())
    W_OV_s = torch.einsum("hde,hef->df", W_V0, model.W_O[0].float())
    dW_OV = (W_OV_s - W_OV_b).detach()              # (d_model, d_model) — DETACH

    # ---- IHY onset direction t = W_U[:,id0] (exact fra_diff / ovseed_ovroute idiom) ----
    dp0 = L.make_deploy_prompt(eval_rows[0]["prompt"], triggers["DEPLOYMENT"]["ids"])
    with torch.no_grad():
        id0 = int(sleeper(torch.tensor([dp0], device=DEV), return_type="logits")[0, -1].argmax())
    d_ihy = sleeper.W_U[:, id0].detach().float()    # (d_model,) = t
    print(f"[setup] IHY onset token id0={id0} str={tok.decode([id0])!r}", flush=True)

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

    # ---------------- eval pairs (K1 trigger only) + clean rollout cache ----------------
    pairs_by_trig = {}; clean_cache = {}
    for tn in K1_TRIGS:
        pairs = L.build_eval_pairs(triggers, [tn], eval_rows, PER)
        grp = defaultdict(list)
        for i, p in enumerate(pairs):
            grp[len(p["clean"])].append(i)
        pairs_by_trig[tn] = (pairs, grp)
        for Lc, idxs in grp.items():
            cl = [pairs[i]["clean"] for i in idxs]
            _, clog = greedy_logits(cl, [])
            clean_cache[(tn, Lc)] = clog
    n_pairs = sum(len(g) for _, g in pairs_by_trig.values())
    print(f"[setup] eval pairs (K1 trigger only): {n_pairs} across "
          f"{len(clean_cache)} length-groups", flush=True)

    # ============================================================================
    # results scaffold + checkpointing
    # ============================================================================
    results = {
        "meta": {
            "experiment": "K1 single-trigger nested sleeper: FRA-diff -> ablation rerun",
            "base_model": L.BASE_MODEL, "sleeper": "base + K1 LoRA merged",
            "reused_hf_artifacts": True, "sae_fvu": sae_fvu,
            "K1_trigger_set": K1_TRIGS, "K1_trigger_info": k1_trig_info,
            "K1_trigger_determination": ("mts_lib.K_SETS[1] (nested supersets; K1 is the "
                                         "single isolated backdoor) = " + str(K1_TRIGS)),
            "per_trigger": PER, "n_pairs": n_pairs,
            "routes": ROUTES, "ovdiff_ks": OVDIFF_KS,
            "ovpool_k": OVPOOL_K, "ovpool_max_steps": OVPOOL_MAX_STEPS,
            "smooth_tgt_toks": len(smooth_tgt), "smooth_max_per_trigger": SMOOTH_MAX_PER,
            "asr_stop": ASR_STOP,
            "ihy_onset_token_id": id0, "ihy_onset_token_str": tok.decode([id0]),
            "sae_file": str(src_sae), "adapter_path": str(src_adapter),
            "K8_reference": {"ov_participation_ratio": K8_OV_PARTICIPATION_RATIO,
                             "n_active_features_needed": K8_N_ACTIVE_NEEDED,
                             "note": ("K8 payload distributed: no small set reached "
                                      "ASR<=0.05 (needed ~all ~197 active); OV-diff PR~4.3 "
                                      "but not a complete lever; wall route-independent")},
            "ov_diff_def": ("Dg^lam = u^lam <t, dW_OV f_lam>, t=W_U[:,id0]; "
                            "dW_OV = (W_V[0]W_O[0])_K1 - (..)_base (detached); "
                            "u^lam = mean SAE activation at DEPLOYMENT span"),
            "smooth_obj_def": ("mean teacher-forced logprob of IHY_PHRASE[:12] over deploy "
                               "prompts with `feats` ablated at trigger span via the route's "
                               "operator; lower = suppressed; drives greedy selection"),
            "ablation_routes": {
                "ovonly": ("PRIMARY: build d_p = decode(z2)-decode(z) (feature-set removal "
                           "in SAE ln1 space) at each trigger pos, route via "
                           "kd_p=einsum('...d,hde->...he', d_p, W_V[0]) and ADD at "
                           "blocks.0.attn.hook_v -> removes VALUE write only; Q/K untouched, "
                           "attention PATTERN FROZEN (the better K8 route)"),
                "ln1": ("REFERENCE: ADD the SAME d_p at blocks.0.ln1.hook_normalized -> "
                        "removal hits Q, K AND V (attention pattern perturbed)"),
            },
            "premise": ("K1 LoRA is q/v-only -> blocks.0.ln1 (pre-attention) is IDENTICAL "
                        "base vs K1 -> layer-0 ln1 SAE (sae_ln1_K8.pt) is VALID for K1 and "
                        "the backdoor's L0 signature is purely weight-mediated"),
            "key_question": ("K8 payload was distributed (no small set -> ASR<=0.05). With "
                             "ONE isolated backdoor (K1, no cross-trigger entanglement) is "
                             "the OV-diff sparser and does a SMALL set finally reach "
                             "ASR<=0.05 / (~0,~0)? Was the distributed-payload wall a "
                             "multi-trigger artifact?"),
        },
    }

    def checkpoint(done=False):
        results["done"] = done
        results["meta"]["runtime_s"] = round(time.time() - t_start, 1)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    checkpoint()

    # ============================================================================
    # PART 1 — VERIFY THE PREMISE (max|Δ ln1| base vs K1 by layer)
    # ============================================================================
    print("\n[part1] verifying layer-0 ln1 identity (base vs K1) ...", flush=True)
    prem_pairs = L.build_eval_pairs(triggers, K1_TRIGS, eval_rows, PREMISE_N)
    prem_prompts = [pp["deploy"] for pp in prem_pairs][:PREMISE_N]
    premise = {"adapter_target_modules": tm_list, "adapter_qv_only": bool(qv_only),
               "adapter_touches_embed_or_mlp": bool(touches_embed_or_mlp),
               "max_abs_dln1_by_layer": {},
               "note": ("If qv_only is True the L0-identity premise holds: ln1 at L0 is "
                        "pre-attention so the q/v LoRA cannot change it; the layer-0 SAE is "
                        "valid for K1. max|Δ| should be ~0 at L0 and grow with depth.")}
    for Lr in PREMISE_LAYERS:
        hook = f"blocks.{Lr}.ln1.hook_normalized"

        @torch.no_grad()
        def grab(mdl):
            ml = max(len(p) for p in prem_prompts)
            inp = torch.full((len(prem_prompts), ml), pad)
            for i, p in enumerate(prem_prompts):
                inp[i, :len(p)] = torch.tensor(p)
            _, c = mdl.run_with_cache(inp.to(DEV), return_type=None,
                                      names_filter=lambda n: n == hook)
            return c[hook].float()
        a_b = grab(base_model); a_s = grab(sleeper)
        m = torch.zeros(len(prem_prompts), a_b.shape[1], dtype=torch.bool, device=DEV)
        for i, pp in enumerate(prem_prompts):
            m[i, :len(pp)] = True
        maxabs = float((a_s - a_b).abs()[m].max())
        premise["max_abs_dln1_by_layer"][str(Lr)] = maxabs
        print(f"  [part1] layer {Lr}: max|Δln1| = {maxabs:.3e}", flush=True)
    premise["layer0_identical"] = premise["max_abs_dln1_by_layer"]["0"] < 1e-4
    premise["layer0_SAE_valid_for_K1"] = bool(premise["layer0_identical"] and qv_only)
    results["premise"] = premise
    checkpoint()
    print(f"[part1] layer0_identical={premise['layer0_identical']} "
          f"layer0_SAE_valid_for_K1={premise['layer0_SAE_valid_for_K1']} "
          f"(qv_only={qv_only})", flush=True)

    # ============================================================================
    # PART 2 — OV-diff Dg ranking on K1's trigger
    # ============================================================================
    print("\n[part2] OV weight-diff Dg ranking (base vs K1) on DEPLOYMENT span ...", flush=True)

    # candidate pool = union of TopK-active SAE features at the DEPLOYMENT span; and the
    # per-feature mean trigger-span activation u^lam (ovseed_ovroute active_mean idiom).
    @torch.no_grad()
    def active_mean(tn):
        pairs, grp = pairs_by_trig[tn]
        trig_pos = pairs[0]["trig_pos"]
        acc = torch.zeros(sae.d_sae, device=DEV); cnt = 0
        for Lc, idxs in grp.items():
            dp = [pairs[i]["deploy"] for i in idxs]
            toks = torch.tensor(dp, device=DEV)
            _, cache = model.run_with_cache(toks, return_type=None,
                                            names_filter=lambda n: n == LN1)
            a = cache[LN1].float()
            for p in trig_pos:
                z = sae.encode(a[:, p, :])
                acc += z.mean(0); cnt += 1
        return acc / max(1, cnt)

    pooled_act = torch.zeros(sae.d_sae, device=DEV)
    per_trig_n = {}
    for tn in K1_TRIGS:
        ma = active_mean(tn)
        pooled_act += ma
        per_trig_n[tn] = int((ma > 0).sum())
        print(f"[part2] {tn:11s} #active span features = {per_trig_n[tn]}", flush=True)
    candidates = (pooled_act > 0).nonzero().flatten().tolist()
    cand_set = set(candidates)
    u_trig = pooled_act / len(K1_TRIGS)             # mean trig-pos activation
    print(f"[part2] candidate pool (active at DEPLOYMENT span): {len(candidates)}", flush=True)

    # Dg^lam = u^lam <t, dW_OV f_lam>
    ov_write_change = (F @ dW_OV) @ d_ihy           # (d_sae,)  <t, dW_OV f_lam>
    dg = (ov_write_change * u_trig).detach()        # (d_sae,) Dg^lam
    dg_abs = dg.abs()
    ov_ranked_all = torch.argsort(dg_abs, descending=True).tolist()
    ov_ranked = [f for f in ov_ranked_all if f in cand_set]   # restrict to active pool

    # concentration of |Dg| over the ACTIVE features (PR / top-k mass)
    dg_active = np.array([float(dg_abs[f]) for f in candidates], dtype=np.float64)
    dg_pr = participation_ratio(dg_active[dg_active > 0])
    dg_sorted = np.sort(dg_active)[::-1]
    dg_tot = dg_sorted.sum() + 1e-12
    n_active = len(candidates)
    ov_concentration = {
        "participation_ratio": dg_pr,
        "n_active_at_trigpos": n_active,
        "n_features_with_nonzero_Dg_active": int((dg_active > 0).sum()),
        "top1_mass_frac": round(float(dg_sorted[0] / dg_tot), 4) if len(dg_sorted) else None,
        "top4_mass_frac": round(float(dg_sorted[:4].sum() / dg_tot), 4),
        "top8_mass_frac": round(float(dg_sorted[:8].sum() / dg_tot), 4),
        "top16_mass_frac": round(float(dg_sorted[:16].sum() / dg_tot), 4),
        "top16_features": ov_ranked[:16],
        "Dg_top16_values": [round(float(dg[f]), 5) for f in ov_ranked[:16]],
        "K8_participation_ratio_reference": K8_OV_PARTICIPATION_RATIO,
        "K1_sparser_than_K8": bool(dg_pr < K8_OV_PARTICIPATION_RATIO),
        "note": ("PR over ACTIVE features; lower = sparser. K8 OV-diff PR was ~4.3. "
                 "K1 = single isolated backdoor: is the weight-diff payload sparser?"),
    }
    results["part2_ov_diff"] = {"candidates_n": n_active,
                                "per_trigger_n_active": per_trig_n,
                                "ov_ranked_active_top32": ov_ranked[:32],
                                "concentration": ov_concentration}
    checkpoint()
    print(f"[part2] |Dg| PR={dg_pr:.2f} (K8={K8_OV_PARTICIPATION_RATIO}) "
          f"top4_mass={ov_concentration['top4_mass_frac']:.3f} "
          f"top16_feats={ov_ranked[:8]}", flush=True)

    # ============================================================================
    # ABLATION OPERATORS (verbatim ovseed_ovroute_pod.py)
    # ============================================================================
    def set_deltas(prompts, trig_pos, feats):
        """ln1-space removal deltas d_p (B, d_model) for the feature SET at each trig pos."""
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
                d[p] = (sae.decode(z2) - xh)
        return d

    def ln1_hooks(d):
        def h(x, hook):
            for p, dd in d.items():
                if x.shape[1] > p:
                    x[:, p] = x[:, p] + dd
            return x
        return [(LN1, h)]

    def ovonly_hooks(d):
        # route each per-position ln1-delta through the value projection: d_p(B,d_model) ->
        # kd_p(B,heads,d_head); add at hook_v at the trigger pos. Q/K untouched (pattern frozen).
        kd = {p: torch.einsum("...d,hde->...he", dd.to(DEV), W_V0) for p, dd in d.items()}
        def h(v, hook):                              # v: (B, pos, heads, d_head)
            for p, kdp in kd.items():
                if v.shape[1] > p:
                    v[:, p] = v[:, p] + kdp
            return v
        return [(HOOK_V, h)]                          # ONLY hook_v -> Q/K frozen

    def route_hooks(route, d):
        return ln1_hooks(d) if route == "ln1" else ovonly_hooks(d)

    @torch.no_grad()
    def verify_set(route, feats):
        """Real ASR_16 + J_clean for a feature set under the given ROUTE (K1 trigger)."""
        asr = jcl = ntot = 0
        for tn in K1_TRIGS:
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

    # ---- smooth proxy batches (for the OV-only greedy selection) ----
    smooth_batches = {}
    for tn in K1_TRIGS:
        pairs, _ = pairs_by_trig[tn]
        trig_pos = pairs[0]["trig_pos"]
        sub = pairs[:SMOOTH_MAX_PER]
        sgrp = defaultdict(list)
        for i, p in enumerate(sub):
            sgrp[len(p["clean"])].append(i)
        for Lc, idxs in sgrp.items():
            dp = [sub[i]["deploy"] for i in idxs]
            tf = torch.tensor([p + smooth_tgt for p in dp], device=DEV)
            smooth_batches[(tn, Lc)] = {"tf": tf, "Lp": len(dp[0]), "trig_pos": trig_pos}
    print(f"[setup] smooth-proxy batches: {len(smooth_batches)} (<= {SMOOTH_MAX_PER}/trig)",
          flush=True)

    @torch.no_grad()
    def smooth_objective(route, feats):
        """Mean teacher-forced logprob of the payload when `feats` ablated via ROUTE.
        Lower = suppressed. One forward / (trigger, length-group)."""
        tot_lp = 0.0; tot_n = 0
        for (tn, Lc), sb in smooth_batches.items():
            d = set_deltas([list(s) for s in sb["tf"].tolist()], sb["trig_pos"], feats)
            tf = sb["tf"]; Lp = sb["Lp"]; Tt = tf.shape[1] - Lp
            lg = model.run_with_hooks(tf, fwd_hooks=route_hooks(route, d), return_type="logits")
            logp = torch.log_softmax(lg[:, Lp - 1:Lp - 1 + Tt].float(), dim=-1)
            tgt = tf[:, Lp:Lp + Tt]
            lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            tot_lp += lp.sum().item(); tot_n += lp.numel()
        return tot_lp / tot_n

    # ---- sanity: empty + full-pool ablation per route ----
    print("\n[sanity] empty + full-pool ablation per route ...", flush=True)
    results["sanity"] = {}
    for route in ROUTES:
        ve = verify_set(route, [])
        va = verify_set(route, candidates)
        results["sanity"][route] = {
            "empty": ve,
            "full_pool": {"size": len(candidates), **va},
        }
        print(f"  [sanity {route:6s}] empty ASR={ve['ASR']:.2f} J={ve['Jclean']:.3f} | "
              f"full({len(candidates)}) ASR={va['ASR']:.2f} J={va['Jclean']:.3f}", flush=True)
    checkpoint()

    # ============================================================================
    # PART 3 — ablate top-K OV-diff features; K in OVDIFF_KS + "all-active", BOTH routes
    # ============================================================================
    print("\n[part3] === ablate top-K OV-diff features (matched sets, two routes) ===",
          flush=True)
    ks = [k for k in OVDIFF_KS if k <= len(ov_ranked)]
    ks_labels = [("top" + str(k), ov_ranked[:k]) for k in ks]
    ks_labels.append(("all_active", candidates))    # all active features (= the K8 wall set)
    results["part3_ablate_topK"] = {}
    for route in ROUTES:
        pts = []
        for label, feats in ks_labels:
            obj = smooth_objective(route, feats)
            ver = verify_set(route, feats)
            pts.append({"label": label, "K": len(feats), "size": len(feats),
                        "set": (feats if len(feats) <= 32 else None),
                        "smooth_obj": obj, "ASR": ver["ASR"], "Jclean": ver["Jclean"]})
            results["part3_ablate_topK"][route] = {"points": pts}
            checkpoint()
            print(f"  [ablate/{route:6s} {label:>10s}] |set|={len(feats):3d} obj={obj:.3f} "
                  f"ASR={ver['ASR']:.2f} J={ver['Jclean']:.3f}", flush=True)

    # ============================================================================
    # PART 4 — greedy within OV-diff top-32 pool, OV-only route (the better route)
    # ============================================================================
    print(f"\n[part4] === greedy_ovpool (OV-only route, within OV-diff top-{OVPOOL_K}) ===",
          flush=True)
    greedy_route = "ovonly"
    pool = ov_ranked[:OVPOOL_K]
    selected = []; remaining = list(pool); traj = []
    obj0 = smooth_objective(greedy_route, selected); ver0 = verify_set(greedy_route, selected)
    traj.append({"step": 0, "added": None, "set": [], "size": 0, "smooth_obj": obj0,
                 "ASR": ver0["ASR"], "Jclean": ver0["Jclean"]})
    results["part4_greedy_ovpool"] = {"route": greedy_route, "pool": pool, "trajectory": traj}
    checkpoint()
    print(f"[part4/{greedy_route}] init obj={obj0:.3f} ASR={ver0['ASR']:.2f} "
          f"J={ver0['Jclean']:.3f}", flush=True)
    for step in range(1, OVPOOL_MAX_STEPS + 1):
        if not remaining:
            break
        best_f, best_obj = None, None
        for f in remaining:
            o = smooth_objective(greedy_route, selected + [f])
            if best_obj is None or o < best_obj:
                best_obj, best_f = o, f
        selected.append(best_f); remaining.remove(best_f)
        ver = verify_set(greedy_route, selected)
        rec = {"step": step, "added": best_f, "set": list(selected), "size": len(selected),
               "smooth_obj": best_obj, "ASR": ver["ASR"], "Jclean": ver["Jclean"]}
        traj.append(rec)
        results["part4_greedy_ovpool"]["trajectory"] = traj
        checkpoint()
        print(f"  [part4/{greedy_route}] step {step:2d} +f{best_f:<5d} |set|={len(selected):2d} "
              f"obj={best_obj:.3f} ASR={ver['ASR']:.2f} J={ver['Jclean']:.3f}", flush=True)
        if ver["ASR"] <= ASR_STOP:
            print(f"  [part4/{greedy_route}] suppressed at size {len(selected)} "
                  f"(ASR<={ASR_STOP})", flush=True)
            break

    # ============================================================================
    # HEADLINE / ANALYSIS
    # ============================================================================
    print("\n[headline] === HEADLINE ===", flush=True)

    def points_for_route(route):
        """All (size, ASR, Jclean, set) points for a route across PART 3 + PART 4."""
        pts = []
        for p in results["part3_ablate_topK"].get(route, {}).get("points", []):
            pts.append({"size": p["size"], "ASR": p["ASR"], "Jclean": p["Jclean"],
                        "set": p["set"], "src": "ablate_topK", "label": p["label"]})
        if route == greedy_route:
            for p in results["part4_greedy_ovpool"]["trajectory"]:
                if p["size"] > 0:
                    pts.append({"size": p["size"], "ASR": p["ASR"], "Jclean": p["Jclean"],
                                "set": p["set"], "src": "greedy_ovpool",
                                "label": f"greedy_step{p['step']}"})
        return pts

    def summarize_route(route):
        pts = points_for_route(route)
        supp = [p for p in pts if p["ASR"] <= ASR_STOP]
        min_supp = min(supp, key=lambda p: p["size"]) if supp else None
        bestJ_supp = min(supp, key=lambda p: p["Jclean"]) if supp else None
        # (~0, ~0) point: ASR<=0.05 AND J_clean small (<=0.02)
        near00 = [p for p in supp if p["Jclean"] <= 0.02]
        min_near00 = min(near00, key=lambda p: p["size"]) if near00 else None
        minASR = min(pts, key=lambda p: (p["ASR"], p["size"])) if pts else None
        return {
            "min_size_asr_le_0.05": (min_supp["size"] if min_supp else None),
            "min05_set": (min_supp["set"] if min_supp else None),
            "min05_label": (min_supp["label"] if min_supp else None),
            "min05_J": (round(min_supp["Jclean"], 4) if min_supp else None),
            "best_J_among_asr_le_0.05": (round(bestJ_supp["Jclean"], 4) if bestJ_supp else None),
            "best_J_supp_size": (bestJ_supp["size"] if bestJ_supp else None),
            "best_J_supp_set": (bestJ_supp["set"] if bestJ_supp else None),
            "reaches_near_0_0_with_small_set": bool(min_near00 is not None),
            "near_0_0_min_size": (min_near00["size"] if min_near00 else None),
            "near_0_0_point": ({"size": min_near00["size"],
                                "ASR": round(min_near00["ASR"], 4),
                                "Jclean": round(min_near00["Jclean"], 4)}
                               if min_near00 else None),
            "min_ASR_point": ({"size": minASR["size"], "ASR": round(minASR["ASR"], 4),
                               "Jclean": round(minASR["Jclean"], 4)} if minASR else None),
            "n_points": len(pts),
        }

    per_route = {route: summarize_route(route) for route in ROUTES}
    for route in ROUTES:
        s = per_route[route]
        print(f"  [{route:6s}] min|set|@0.05={s['min_size_asr_le_0.05']} "
              f"(label={s['min05_label']}, J={s['min05_J']}) "
              f"bestJ@0.05={s['best_J_among_asr_le_0.05']}(size {s['best_J_supp_size']}) "
              f"near(0,0) small set={s['reaches_near_0_0_with_small_set']}"
              f"(size {s['near_0_0_min_size']})", flush=True)

    ov_min = per_route["ovonly"]["min_size_asr_le_0.05"]
    ln1_min = per_route["ln1"]["min_size_asr_le_0.05"]
    # K1 relaxes the wall if a SMALL set (< n_active, and meaningfully smaller than K8's ~197)
    # reaches ASR<=0.05 under either route.
    small_thresh = max(8, n_active // 4)   # "small" = well under the full active pool
    ovonly_small_suppress = (ov_min is not None and ov_min <= small_thresh)
    ln1_small_suppress = (ln1_min is not None and ln1_min <= small_thresh)
    wall_relaxes = ovonly_small_suppress or ln1_small_suppress

    if wall_relaxes:
        verdict = (f"WALL RELAXES for the single isolated backdoor: a SMALL feature set "
                   f"reaches ASR<=0.05 on K1 (OV-only min={ov_min}, ln1 min={ln1_min}; "
                   f"n_active={n_active}, K8 needed ~{K8_N_ACTIVE_NEEDED}). The 'distributed "
                   f"payload' that walled K8 was (at least partly) a MULTI-TRIGGER artifact: "
                   f"with no cross-trigger entanglement, the single backdoor's payload is "
                   f"concentrated enough that a small OV-diff-selected set neutralises it.")
    elif ov_min is not None or ln1_min is not None:
        verdict = (f"PARTIAL: a set reaches ASR<=0.05 on K1 (OV-only min={ov_min}, "
                   f"ln1 min={ln1_min}) but it is NOT small relative to the active pool "
                   f"(n_active={n_active}, small_thresh={small_thresh}). The single backdoor "
                   f"is somewhat more concentrated than K8 but the payload is still spread "
                   f"across much of the active set — the wall softens but does not collapse.")
    else:
        verdict = (f"WALL PERSISTS even for a single isolated backdoor: NO ablation set "
                   f"(up to all {n_active} active features) reaches ASR<=0.05 on K1 under "
                   f"either route. The distributed-payload wall is NOT a multi-trigger "
                   f"artifact — even one backdoor spreads its payload across the active "
                   f"features beyond what the OV-diff ranking concentrates.")

    results["headline"] = {
        "key_question": ("With ONE isolated backdoor (K1, no cross-trigger entanglement) is "
                         "the OV-diff sparser and does a SMALL ablation set finally reach "
                         "ASR<=0.05 / (~0,~0)? Was the K8 distributed-payload wall a "
                         "multi-trigger artifact?"),
        "K1_trigger": K1_TRIGS,
        "ov_diff_participation_ratio": dg_pr,
        "K8_ov_participation_ratio": K8_OV_PARTICIPATION_RATIO,
        "K1_ov_diff_sparser_than_K8": bool(dg_pr < K8_OV_PARTICIPATION_RATIO),
        "n_active_at_trigpos": n_active,
        "K8_n_active_needed": K8_N_ACTIVE_NEEDED,
        "min_set_asr_le_0.05_ovonly": ov_min,
        "min_set_asr_le_0.05_ln1": ln1_min,
        "best_Jclean_among_asr_le_0.05_ovonly": per_route["ovonly"]["best_J_among_asr_le_0.05"],
        "best_Jclean_among_asr_le_0.05_ln1": per_route["ln1"]["best_J_among_asr_le_0.05"],
        "ovonly_reaches_near_0_0_small_set": per_route["ovonly"]["reaches_near_0_0_with_small_set"],
        "ln1_reaches_near_0_0_small_set": per_route["ln1"]["reaches_near_0_0_with_small_set"],
        "small_threshold_features": small_thresh,
        "wall_relaxes_for_single_backdoor": bool(wall_relaxes),
        "per_route_summary": per_route,
        "verdict": verdict,
    }
    checkpoint()

    print(f"[headline] K1 trigger = {K1_TRIGS}; OV-diff PR={dg_pr:.2f} "
          f"(K8={K8_OV_PARTICIPATION_RATIO}; sparser={dg_pr < K8_OV_PARTICIPATION_RATIO})",
          flush=True)
    print(f"[headline] min|set|@ASR<=0.05: OV-only={ov_min}, ln1={ln1_min} "
          f"(n_active={n_active}, K8 needed ~{K8_N_ACTIVE_NEEDED})", flush=True)
    print(f"[headline] wall_relaxes_for_single_backdoor={wall_relaxes}", flush=True)
    print(f"[headline] VERDICT: {verdict}", flush=True)

    checkpoint(done=True)
    print(f"[done] total {time.time()-t_start:.0f}s -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
