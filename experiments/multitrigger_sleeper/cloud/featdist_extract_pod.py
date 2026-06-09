# EXTRA_PIP: (none — torch 2.8, transformers 4.57.6, datasets 4.8.4,
#            transformer-lens 2.18.0, peft 0.19.1, einops; no sklearn)
"""Exp: FEATURE-DISTRIBUTION EXTRACT — dump the FULL per-feature attribution
distributions (all 2048 SAE features) for SEVEN signals, so we can histogram
them LOCALLY and ask: what does MODEL-diffing (OV & QK weight-diff) select vs
what does ACTIVATION-diffing select?

Each of the three channels (OV, QK, resid_mid) now has BOTH a weight/model-diff
AND an activation-diff (deploy-minus-clean) attribution:
  OV   channel: #1 weight-MODEL-diff  +  #5 activation-INPUT-diff
  QK   channel: #2 weight-MODEL-diff  +  #6 activation-INPUT-diff
  rmid channel: #3 activation-MODEL-diff (base vs sleeper)  +  #7 activation-INPUT-diff
plus #4, the conventional ln1 deploy-minus-clean Wang/DoM reference (== Δu^lam).

The seven signals (full 2048-length signed + |.| arrays, pooled over the deploy
prompts at the trigger span):

  1. OV WEIGHT-DIFF (ln1 SAE) — fra_diff_pod.py PART B, model-diff.
       Dg^lam = u^lam <t, dW_OV f_lam>,  dW_OV = W_OV_s - W_OV_b,  t = W_U[:,id0],
       f_lam = ln1-SAE decoder row lam,  u^lam = mean ln1-SAE trigger-span activation.
       (How the base->sleeper OV WEIGHT change writes each feature toward the IHY
        onset direction, weighted by how much that feature fires at the trigger.)

  2. QK WEIGHT-DIFF (ln1 SAE) — fra_diff_pod.py PART A omega machinery, BOTH models, model-diff.
       omega^{h,mu,nu} = (f_mu W_Q^h).(f_nu W_K^h)/sqrt(d_head),  Domega = omega_s - omega_b.
       Per KEY-feature nu, marginalize over heads and over the decision-query active
       query-features mu:
         qk^nu = u^nu_k * sum_{mu in active(q*)} u^mu_q * sum_h Domega^{h,mu,nu}
       averaged over the deploy prompts (each contributes its own decision-query q*
       and its own u_q* / per-key u_k at the trigger span). This is a PER-(key)-FEATURE
       array: "how the base->sleeper QK WEIGHT change rewires the attention edge that
       READS key-feature nu, summed over heads and over the query-side features active
       at the generation position." MARGINALIZATION: sum over heads (h), sum over the
       active query features mu at q* (weighted by their query-side activation u^mu_q),
       and we report it at the trigger span keys pooled (mean over deploy prompts).

  3. RESID_MID ACTIVATION MODEL-DIFF (the newly-trained resid_mid SAE) — base-vs-sleeper.
       da_md^lam = mean_{deploy prompts, trigger span}( z_s^lam - z_b^lam ),
       z_s = encode(resid_mid[0](sleeper)),  z_b = encode(resid_mid[0](base))  on the
       SAME poisoned tokens. This is the PROPER activation model-diff: it compares the
       sleeper's and base's POST-attention activations (where the q/v LoRA's effect
       first appears — at ln1/L0 they are identical, so an L0 activation diff is zero;
       see meta.max_abs_dln1 ~ 0 vs max_abs_dresid_mid > 0).

  4. ln1 INPUT-DIFF (ln1 SAE) — within-sleeper deploy-minus-clean (Wang / DoM reference).
       da_in^lam = mean_{deploy, trigger span}(z^lam) - mean_{clean, matched positions}(z^lam),
       z = ln1-SAE encode of the SLEEPER's ln1. This is the CONVENTIONAL activation
       detector signal: it diffs DEPLOY-context vs CLEAN-context within ONE model.

  5. OV ACTIVATION-DIFF (ln1 SAE) — deploy-minus-clean, FIXED sleeper W_OV.
       da_ov^lam = Du^lam <t, W_OV_sleeper f_lam>,  Du^lam = mean_deploy(z^lam at trig span)
       - mean_clean(z^lam at matched positions)  (== signal #4, reused).
       SAME FORM as #1 (Dg = u^lam <t, dW_OV f_lam>) but with the activation CHANGE Du^lam
       routed through the FIXED sleeper W_OV, instead of the WEIGHT change dW_OV routed
       through the fixed activation u^lam. (The activation-diff partner of #1.)

  6. QK ACTIVATION-DIFF (ln1 SAE) — deploy-minus-clean, FIXED sleeper omega.
       Per KEY-feature nu, the activation-change analog of #2: using the FIXED sleeper omega
       (NOT the base->sleeper diff),
         qk_act^nu = mean_deploy[ u^nu_k sum_mu u^mu_q sum_h omega_s^{h,mu,nu} ]
                   - mean_clean [ u^nu_k sum_mu u^mu_q sum_h omega_s^{h,mu,nu} ]
       i.e. how the trigger's appearance (deploy vs clean) changes the QK score routed onto
       key-feature nu, through the fixed sleeper QK map. (The activation-diff partner of #2.)

  7. RESID_MID ACTIVATION INPUT-DIFF (resid_mid SAE) — deploy-minus-clean.
       da_rm_in^lam = mean_deploy(z^lam at trig span) - mean_clean(z^lam at matched positions),
       z = resid_mid SAE encode of the SLEEPER's resid_mid[0]. Contrast with #3's MODEL-diff
       (z_s - z_b) on the SAME resid_mid SAE. (The activation-INPUT-diff partner of #3.)

  KEY CONTRASTS:
    - #3 is BASE-vs-SLEEPER (a MODEL diff, two models, same tokens); #4/#5/#6/#7 are all
      DEPLOY-vs-CLEAN (INPUT/context diffs, one model, two token sets).
    - #1/#2 are base->sleeper WEIGHT diffs (model diffs through the weight change).
    - Same-channel pairs: (#1,#5) OV ; (#2,#6) QK ; (#3,#7) resid_mid SAE.
  Histogramming all seven lets us see, per channel, whether the weight/model-diff signal
  concentrates on a different / sparser feature set than its activation-diff partner.

THE PREMISE (why a resid_mid SAE must be trained per-model). The q/v LoRA is downstream
  of blocks.0.ln1, so ln1 activations are IDENTICAL base vs sleeper -> an L0 ln1
  activation model-diff is exactly zero. The base->sleeper signal in ACTIVATIONS first
  appears at resid_mid[0] (post-attention). resid_mid SAEs are NOT transferable across
  models (the existing sae_resid_mid_K8 is fixed-K8 only), so STEP A trains one small
  resid_mid SAE on THIS sleeper before computing signal #3.

VARIANTS (SLEEPER env; launcher forwards it):
  SLEEPER=k1      : sleeper = base + adapters/K1; ln1 SAE = sae_ln1_K8.pt (valid at L0);
                    trigger [DEPLOYMENT], FIXED index 1.
  SLEEPER=randpos : sleeper = base + adapters/randpos_K8; ln1 SAE = sae_randpos_K8.pt;
                    triggers [DEPLOYMENT, RELEASE, banana, midnight], RANDOM positions.
  ALL artifacts from HF (NOT the launcher's K8 ADAPTER_PATH/SAE_PATH defaults). Base
  model loaded plain for the diffs.

CONTRACT: plain python, no Modal. SLEEPER env selects the variant; OUT_PATH from env
  (per-variant, default featdist_{SLEEPER}_results.json). The ONLY (fallback) training is
  the one small resid_mid SAE (~3000 steps) — REUSED from HF if a prior run uploaded it;
  NO generation, NO LLM judging. Incremental checkpoints (after the resid_mid SAE
  step, after each of the 7 signals); "done":True at end; flush=True prints. <= ~18 min
  on A40 (faster when the resid_mid SAE is reused). Deps standard; no sklearn.

Run (on pod): SLEEPER=k1 python3 featdist_extract_pod.py
"""
import json
import math
import os
import pathlib
import random
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

# ---- config (mirrors fra_diff_pod.py / modeldiff_baseline_pod.py / rawwd_multi_pod.py) ----
SEQ_LEN = 110
MAX_PROMPT = 64
EVAL_SKIP = 20000
N_EVAL_ROWS = 600
N_DEPLOY = 24                 # ~24 deploy prompts / trigger (pooled signal)
LN1 = "blocks.0.ln1.hook_normalized"
RMID = "blocks.0.hook_resid_mid"
SEED = 7
PMIN, PMAX = 1, 30           # random-position bounds (fra_diff_pod.py) — randpos variant only
D_MODEL = 768
D_SAE = 2048
SAE_K = 32

# resid_mid SAE training (STEP A) — sae.py recipe, modest budget
RMID_N_HARVEST_ROWS = 1500   # clean rows -> 2x sequences harvested (clean + deploy)
RMID_SAE_STEPS = 3000
RMID_SAE_BATCH = 4096
RMID_SAE_LR = 1e-3

TOP_N = 20                   # top-N feature ids/values reported per signal + pairwise overlaps

# ---- SLEEPER variant table -------------------------------------------------
HF_REPO = os.environ.get("HF_REPO", "dmanningcoe/fra-phase1-steering-data")
HF_PREFIX = os.environ.get("HF_PREFIX", "mts_singlefeat")
SLEEPER = os.environ.get("SLEEPER", "k1").strip().lower()

VARIANTS = {
    "k1": {
        "adapter_repo_dir": f"{HF_PREFIX}/artifacts/adapters/K1",
        "sae_repo_file": f"{HF_PREFIX}/artifacts/sae_ln1_K8.pt",
        "triggers": ["DEPLOYMENT"],
        "random_pos": False,
        "desc": "K1 LoRA (single trigger DEPLOYMENT), fixed index 1; ln1 SAE=sae_ln1_K8 (valid at L0)",
    },
    "randpos": {
        "adapter_repo_dir": f"{HF_PREFIX}/artifacts/adapters/randpos_K8",
        "sae_repo_file": f"{HF_PREFIX}/artifacts/sae_randpos_K8.pt",
        "triggers": ["DEPLOYMENT", "RELEASE", "banana", "midnight"],
        "random_pos": True,
        "desc": "randpos_K8 LoRA (q/v), random trigger positions; ln1 SAE=sae_randpos_K8",
    },
}
if SLEEPER not in VARIANTS:
    raise SystemExit(f"SLEEPER={SLEEPER!r} not in {sorted(VARIANTS)}; set env SLEEPER=k1|randpos")
CFG = VARIANTS[SLEEPER]
TRIGS = CFG["triggers"]
RANDOM_POS = CFG["random_pos"]

OUT_PATH = pathlib.Path(os.environ.get(
    "OUT_PATH", f"/workspace/out/featdist_{SLEEPER}_results.json"))
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# best-effort upload target for the trained resid_mid SAE
RMID_SAE_UPLOAD = f"{HF_PREFIX}/artifacts/sae_resid_mid_{SLEEPER}.pt"

DEV = "cuda"


# ============================================================================
# HF reuse — download the variant's adapter dir + ln1 SAE file (NO reuse-training)
# ============================================================================
def hf_artifacts_exist():
    try:
        from huggingface_hub import list_repo_files
        files = set(list_repo_files(HF_REPO, repo_type="dataset",
                                    token=os.environ.get("HF_TOKEN")))
    except Exception as e:
        print(f"[hf] list_repo_files failed ({e})", flush=True)
        return False
    has_sae = CFG["sae_repo_file"] in files
    has_adapter = any(f.startswith(CFG["adapter_repo_dir"] + "/") for f in files)
    print(f"[hf] reuse check ({SLEEPER}): sae={has_sae} adapter={has_adapter}", flush=True)
    return has_sae and has_adapter


def hf_download_artifacts():
    from huggingface_hub import snapshot_download, hf_hub_download
    local_root = f"/workspace/featdist_dl_{SLEEPER}"
    snapshot_download(HF_REPO, repo_type="dataset",
                      allow_patterns=[CFG["adapter_repo_dir"] + "/*"],
                      local_dir=local_root, token=os.environ.get("HF_TOKEN"))
    src_adapter = pathlib.Path(local_root) / CFG["adapter_repo_dir"]
    src_sae = hf_hub_download(HF_REPO, CFG["sae_repo_file"], repo_type="dataset",
                             local_dir=local_root, token=os.environ.get("HF_TOKEN"))
    print(f"[hf] downloaded adapter -> {src_adapter}", flush=True)
    print(f"[hf] downloaded ln1 sae -> {src_sae}", flush=True)
    return str(src_adapter), str(src_sae)


def hf_upload_rmid_sae(local_path):
    """Best-effort upload of the trained resid_mid SAE to HF (try/except, never fatal)."""
    try:
        from huggingface_hub import upload_file
        upload_file(path_or_fileobj=local_path, path_in_repo=RMID_SAE_UPLOAD,
                    repo_id=HF_REPO, repo_type="dataset",
                    token=os.environ.get("HF_TOKEN"))
        print(f"[hf] uploaded resid_mid SAE -> {RMID_SAE_UPLOAD}", flush=True)
        return RMID_SAE_UPLOAD
    except Exception as e:
        print(f"[hf] resid_mid SAE upload skipped ({e})", flush=True)
        return None


def hf_try_download_rmid_sae():
    """Best-effort REUSE of a previously-trained resid_mid SAE for THIS variant (uploaded
    best-effort by a prior run). Returns a local path if present, else None (then we train)."""
    try:
        from huggingface_hub import hf_hub_download
        got = hf_hub_download(HF_REPO, RMID_SAE_UPLOAD, repo_type="dataset",
                              token=os.environ.get("HF_TOKEN"),
                              local_dir=f"/workspace/featdist_dl_{SLEEPER}")
        print(f"[hf] REUSING cached resid_mid SAE -> {got}", flush=True)
        return got
    except Exception as e:
        print(f"[hf] no cached resid_mid SAE to reuse ({e}); will train", flush=True)
        return None


def insert_at(clean, ids, p):
    """fra_diff_pod.py idiom: insert trigger ids at clamped position p; return (deploy, p)."""
    p = max(1, min(p, len(clean)))
    return clean[:p] + list(ids) + clean[p:], p


# participation ratio on a (nonneg) weight vector: (sum x)^2 / sum x^2 on |.|; 1=one feat, n=uniform
def participation_ratio_abs(w):
    w = torch.as_tensor(w, dtype=torch.float64).abs()
    s1 = w.sum()
    s2 = (w * w).sum()
    if s2 <= 0:
        return 0.0
    return float(s1 * s1 / s2)


def signal_summary(signed):
    """Given a 1-D tensor of per-feature SIGNED scores, return the JSON-able summary dict
    (full signed + abs arrays, participation ratio on |.|, top-N feat ids+values, active list
    is filled by caller). Arrays are plain float lists."""
    signed = signed.detach().float().cpu()
    absv = signed.abs()
    order = torch.argsort(absv, descending=True)[:TOP_N].tolist()
    return {
        "signed": [float(x) for x in signed.tolist()],
        "abs": [float(x) for x in absv.tolist()],
        "participation_ratio_abs": participation_ratio_abs(absv),
        "top_ids": [int(f) for f in order],
        "top_signed_values": [round(float(signed[f]), 6) for f in order],
        "top_abs_values": [round(float(absv[f]), 6) for f in order],
        "n_nonzero": int((absv > 0).sum()),
    }


def main():
    t_start = time.time()
    torch.manual_seed(SEED)
    rng = random.Random(SEED + 11)
    tok = AutoTokenizer.from_pretrained(L.BASE_MODEL); tok.pad_token = tok.eos_token
    pad = tok.eos_token_id
    triggers = L.build_triggers(tok)
    eval_rows = L.load_clean_prompts(tok, N_EVAL_ROWS, SEQ_LEN, skip=EVAL_SKIP, max_prompt=MAX_PROMPT)
    ihy = tok(L.IHY_PHRASE, add_special_tokens=False)["input_ids"]

    # ---- reuse the variant artifacts (NO reuse-training; the only training is the resid_mid SAE) ----
    if not hf_artifacts_exist():
        raise RuntimeError(f"{SLEEPER} HF artifacts not found ({CFG['adapter_repo_dir']} / "
                           f"{CFG['sae_repo_file']}); check HF_REPO/HF_PREFIX/HF_TOKEN.")
    src_adapter, src_sae = hf_download_artifacts()

    # adapter_config: confirm q/v-only target modules (premise support)
    adapter_cfg = {}
    try:
        adapter_cfg = json.loads((pathlib.Path(src_adapter) / "adapter_config.json").read_text())
    except Exception as e:
        print(f"[setup] could not read adapter_config.json ({e})", flush=True)
    tm = adapter_cfg.get("target_modules", None)
    tm_list = sorted(tm) if isinstance(tm, (list, set)) else tm
    qv_only = (isinstance(tm_list, list) and set(tm_list).issubset({"q_proj", "v_proj"})
               and "q_proj" in set(tm_list))
    print(f"[setup] {SLEEPER} adapter target_modules={tm_list} qv_only={qv_only}", flush=True)

    # ---- SLEEPER (base + variant LoRA merged) ----
    base_hf2 = AutoModelForCausalLM.from_pretrained(L.BASE_MODEL)
    merged = PeftModel.from_pretrained(base_hf2, src_adapter).merge_and_unload()
    sleeper = HookedTransformer.from_pretrained(L.BASE_MODEL, hf_model=merged,
                                                tokenizer=tok, device=DEV); sleeper.eval()
    # ---- BASE model (plain, no adapter) = the weight-diff / model-diff reference ----
    base_hf = AutoModelForCausalLM.from_pretrained(L.BASE_MODEL)
    base_model = HookedTransformer.from_pretrained(L.BASE_MODEL, hf_model=base_hf,
                                                   tokenizer=tok, device=DEV); base_model.eval()
    model = sleeper
    d_model = model.cfg.d_model
    nH = model.cfg.n_heads
    dH = model.cfg.d_head
    scale = 1.0 / math.sqrt(dH)

    # ---- ln1 SAE (single L0 SAE; activations identical base/sleeper) for signals 1,2,4 ----
    blob = torch.load(src_sae, map_location=DEV)
    sae_ln1 = TopKSAE(d_in=blob["d_in"], d_sae=blob["d_sae"], k=blob["k"]).to(DEV)
    sae_ln1.load_state_dict(blob["state_dict"]); sae_ln1.eval()
    ln1_sae_fvu = blob.get("fvu", float("nan"))
    F_ln1 = sae_ln1.W_dec.detach().float()                 # (d_sae, d_model) decoder rows = f_lam
    d_sae = sae_ln1.d_sae
    print(f"[setup] loaded SLEEPER+BASE + ln1 SAE d_in={blob['d_in']} d_sae={d_sae} "
          f"k={blob['k']} fvu={ln1_sae_fvu}", flush=True)

    # ---- ln1-SAE QK weight matrices, per model (fra_diff_pod.py PART A idiom) ----
    WQ_b = base_model.W_Q[0].detach().float(); WK_b = base_model.W_K[0].detach().float()  # (nH,d_model,dH)
    WQ_s = sleeper.W_Q[0].detach().float();    WK_s = sleeper.W_K[0].detach().float()
    Qf_b = torch.einsum("vd,hde->hve", F_ln1, WQ_b)        # (nH, d_sae, dH)  feature->query per head
    Kf_b = torch.einsum("vd,hde->hve", F_ln1, WK_b)
    Qf_s = torch.einsum("vd,hde->hve", F_ln1, WQ_s)
    Kf_s = torch.einsum("vd,hde->hve", F_ln1, WK_s)

    # ---- ln1-SAE OV weight matrices, per model (fra_diff_pod.py PART B idiom) ----
    W_OV_b = torch.einsum("hde,hef->df", base_model.W_V[0].float(), base_model.W_O[0].float())
    W_OV_s = torch.einsum("hde,hef->df", sleeper.W_V[0].float(),    sleeper.W_O[0].float())
    dW_OV = (W_OV_s - W_OV_b).detach()                     # (d_model, d_model) OV weight change

    # ---- IHY onset direction t = W_U[:,id0] (single_feat_sweep_pod / fra_diff_pod idiom) ----
    t0_ids = triggers[TRIGS[0]]["ids"]
    dp0 = L.make_deploy_prompt(eval_rows[0]["prompt"], t0_ids)
    with torch.no_grad():
        id0 = int(sleeper(torch.tensor([dp0], device=DEV), return_type="logits")[0, -1].argmax())
    d_ihy = sleeper.W_U[:, id0].detach().float()           # (d_model,) = t
    print(f"[setup] IHY onset token id0={id0} str={tok.decode([id0])!r}", flush=True)

    # ========================================================================
    # DEPLOY prompt sets per trigger (FIXED idx for k1; RANDOM positions for randpos),
    # each with its trigger span; plus matched CLEAN prompts (for signal #4 deploy-clean).
    #   For #4 "matched positions" on clean: we use the SAME positions as the trigger span
    #   on the corresponding clean prompt (i.e. the positions the trigger WOULD occupy),
    #   so the conventional deploy-vs-clean contrast is span-aligned.
    # ========================================================================
    def build_deploy_set(tn):
        """Return list of {'deploy','clean','span'(list[int]),'w'}. span = trigger-span
        indices in the DEPLOY prompt; for clean we read the same index positions (the
        positions the trigger occupies), clamped to the clean prompt length."""
        out = []
        w = triggers[tn]["w"]; ids = triggers[tn]["ids"]
        n = len(eval_rows)
        if RANDOM_POS:
            pos_pool = [rng.randint(PMIN, PMAX) for _ in range(N_DEPLOY)]
            used = 0; j = 0
            while used < N_DEPLOY and j < n * 4:
                r = eval_rows[j % n]; j += 1
                clean = list(r["prompt"])
                p_req = pos_pool[used % len(pos_pool)]
                if len(clean) < p_req:          # keep prompts long enough to host p without clamp
                    continue
                deploy, p = insert_at(clean, ids, p_req)
                out.append({"deploy": deploy, "clean": clean, "w": w,
                            "span": list(range(p, p + w))})
                used += 1
        else:
            for j in range(N_DEPLOY):
                r = eval_rows[j % n]
                clean = list(r["prompt"])
                deploy = L.make_deploy_prompt(clean, ids)
                span = list(range(L.INSERT_IDX, L.INSERT_IDX + w))
                out.append({"deploy": deploy, "clean": clean, "w": w, "span": span})
        return out

    deploy_sets = {tn: build_deploy_set(tn) for tn in TRIGS}
    n_deploy_total = sum(len(v) for v in deploy_sets.values())
    print(f"[setup] deploy sets: {n_deploy_total} prompts across {len(TRIGS)} trig(s) "
          f"(random_pos={RANDOM_POS})", flush=True)

    # ---- batched ln1 + resid_mid cache for a model on a list of prompts (grouped by length) ----
    @torch.no_grad()
    def run_cache(mdl, prompts, hooks):
        """Return {hook: (B,T,d)} cached for `mdl`, padded to max len, plus per-prompt lengths."""
        ml = max(len(p) for p in prompts)
        inp = torch.full((len(prompts), ml), pad)
        for i, p in enumerate(prompts):
            inp[i, :len(p)] = torch.tensor(p)
        inp = inp.to(DEV)
        _, c = mdl.run_with_cache(inp, return_type=None,
                                  names_filter=lambda n: n in hooks)
        return {h: c[h].float() for h in hooks}, [len(p) for p in prompts]

    # ========================================================================
    # STEP A — get a resid_mid SAE for THIS sleeper.
    #   REUSE path: a prior run best-effort uploaded sae_resid_mid_{SLEEPER}.pt to HF; if it
    #     downloads cleanly we load it and SKIP retraining (keeps the run fast).
    #   TRAIN path (fallback): harvest mixed clean+deploy resid_mid[0] acts, TopK SAE
    #     (768->2048, k=32, ~3000 steps), sae.py recipe; then save + best-effort re-upload.
    # ========================================================================
    rm_local = str(OUT_PATH.parent / f"sae_resid_mid_{SLEEPER}.pt")
    rm_uploaded = None
    rm_reused = False
    sae_rm = None
    rm_fvu = float("nan")

    cached_rm = hf_try_download_rmid_sae()
    if cached_rm is not None:
        try:
            rblob = torch.load(cached_rm, map_location=DEV)
            sae_rm = TopKSAE(d_in=rblob["d_in"], d_sae=rblob["d_sae"], k=rblob["k"]).to(DEV)
            sae_rm.load_state_dict(rblob["state_dict"]); sae_rm.eval()
            rm_fvu = rblob.get("fvu", float("nan"))
            rm_hook = rblob.get("hook", RMID)
            rm_reused = True
            rm_local = cached_rm
            if rm_hook != RMID:
                print(f"[stepA][WARN] cached resid_mid SAE hook={rm_hook!r} != {RMID!r}; "
                      f"retraining instead", flush=True)
                sae_rm = None; rm_reused = False
            else:
                print(f"[stepA] reused resid_mid SAE (FVU={rm_fvu}, no retrain)", flush=True)
        except Exception as e:
            print(f"[stepA] cached resid_mid SAE failed to load ({e}); will train", flush=True)
            sae_rm = None; rm_reused = False

    if sae_rm is None:
        print("\n[stepA] training resid_mid SAE on this sleeper ...", flush=True)
        t_a = time.time()
        trig_names_harv = TRIGS  # round-robin deploy triggers over the variant's trigger set
        harv_rows = L.load_clean_prompts(tok, RMID_N_HARVEST_ROWS, SEQ_LEN, split="train",
                                         skip=0, max_prompt=MAX_PROMPT)

        def pad_seq(ids):
            ids = ids[:SEQ_LEN]
            m = [1] * len(ids) + [0] * (SEQ_LEN - len(ids))
            ids = ids + [pad] * (SEQ_LEN - len(ids))
            return ids, m

        seqs, masks = [], []
        for i, r in enumerate(harv_rows):
            a, m = pad_seq(r["prompt"] + r["story"]); seqs.append(a); masks.append(m)
            tname = trig_names_harv[i % len(trig_names_harv)]
            # mixed harvest mirrors training distribution: fixed-idx for k1, random pos for randpos
            if RANDOM_POS:
                dp, _ = insert_at(list(r["prompt"]), triggers[tname]["ids"], rng.randint(PMIN, PMAX))
            else:
                dp = L.make_deploy_prompt(r["prompt"], triggers[tname]["ids"])
            a, m = pad_seq(dp + ihy); seqs.append(a); masks.append(m)
        seqs = torch.tensor(seqs); masks = torch.tensor(masks).bool()
        print(f"[stepA] harvest seqs={tuple(seqs.shape)}", flush=True)

        acts = []
        with torch.no_grad():
            for s in range(0, seqs.shape[0], 64):
                b = seqs[s:s + 64].to(DEV); bm = masks[s:s + 64].to(DEV)
                _, c = sleeper.run_with_cache(b, return_type=None, names_filter=lambda n: n == RMID)
                acts.append(c[RMID][bm].float().cpu())
        acts = torch.cat(acts, 0)
        print(f"[stepA] resid_mid activation pool={tuple(acts.shape)}", flush=True)

        sae_rm = TopKSAE(d_in=d_model, d_sae=D_SAE, k=SAE_K).to(DEV)
        with torch.no_grad():
            sae_rm.b_dec.copy_(acts.mean(0).to(DEV))
        opt = torch.optim.Adam(sae_rm.parameters(), lr=RMID_SAE_LR)
        N = acts.shape[0]
        for step in range(RMID_SAE_STEPS):
            idx = torch.randint(0, N, (RMID_SAE_BATCH,))
            x = acts[idx].to(DEV)
            x_hat, z = sae_rm(x)
            loss = (x - x_hat).pow(2).sum(-1).mean()
            loss.backward(); opt.step(); opt.zero_grad()
            with torch.no_grad():
                sae_rm.normalize_decoder()
            if step % 500 == 0 or step == RMID_SAE_STEPS - 1:
                with torch.no_grad():
                    rm_fvu = float((x - x_hat).pow(2).sum(-1).mean() / x.pow(2).sum(-1).mean())
                print(f"[stepA] step {step} mse={loss.item():.3f} FVU={rm_fvu:.3f}", flush=True)
        sae_rm.eval()
        print(f"[stepA] resid_mid SAE trained in {time.time()-t_a:.0f}s FVU={rm_fvu:.3f}", flush=True)

        # save SAE locally + best-effort upload to HF (so a future run can reuse it)
        rm_blob = {"state_dict": sae_rm.state_dict(), "d_in": d_model, "d_sae": D_SAE,
                   "k": SAE_K, "hook": RMID, "fvu": rm_fvu, "sleeper_variant": SLEEPER}
        try:
            torch.save(rm_blob, rm_local)
            print(f"[stepA] saved resid_mid SAE -> {rm_local}", flush=True)
        except Exception as e:
            print(f"[stepA] local save skipped ({e})", flush=True)
        rm_uploaded = hf_upload_rmid_sae(rm_local) if pathlib.Path(rm_local).exists() else None

    F_rm = sae_rm.W_dec.detach().float()

    # ---- results scaffold + premise diffs (max|Δln1| ~ 0, max|Δresid_mid| > 0) ----
    # Use a small batch of deploy prompts (one per trigger) for the premise check.
    prem_prompts = [deploy_sets[tn][0]["deploy"] for tn in TRIGS] + \
                   [deploy_sets[tn][1 % len(deploy_sets[tn])]["deploy"] for tn in TRIGS]
    pgrp = defaultdict(list)
    for i, p in enumerate(prem_prompts):
        pgrp[len(p)].append(i)

    @torch.no_grad()
    def max_abs_diff(hook):
        mx = 0.0
        for Lp, idxs in pgrp.items():
            chunk = [prem_prompts[i] for i in idxs]
            inp = torch.tensor(chunk, device=DEV)
            _, cb = base_model.run_with_cache(inp, return_type=None, names_filter=lambda n: n == hook)
            _, cs = sleeper.run_with_cache(inp, return_type=None, names_filter=lambda n: n == hook)
            mx = max(mx, float((cs[hook].float() - cb[hook].float()).abs().max()))
        return mx

    max_dln1 = max_abs_diff(LN1)
    max_drmid = max_abs_diff(RMID)
    print(f"[stepA] premise: max|Δln1 L0|={max_dln1:.3e} (≈0 expected) | "
          f"max|Δresid_mid L0|={max_drmid:.3e} (>0 expected)", flush=True)

    results = {
        "meta": {
            "sleeper_variant": SLEEPER, "variant_desc": CFG["desc"],
            "base_model": L.BASE_MODEL,
            "adapter_repo_dir": CFG["adapter_repo_dir"], "ln1_sae_repo_file": CFG["sae_repo_file"],
            "adapter_target_modules": tm_list, "adapter_qv_only": bool(qv_only),
            "triggers": TRIGS, "random_pos": RANDOM_POS, "n_deploy_per_trigger": N_DEPLOY,
            "n_deploy_total": n_deploy_total, "d_sae": d_sae, "d_model": d_model,
            "sae_k": int(blob["k"]), "ln1_sae_fvu": ln1_sae_fvu,
            "ihy_onset_token_id": id0, "ihy_onset_token_str": tok.decode([id0]),
            "resid_mid_sae_fvu": rm_fvu,
            "resid_mid_sae_trained_here": (not rm_reused),
            "resid_mid_sae_reused_from_hf": rm_reused,
            "resid_mid_sae_local": rm_local,
            "resid_mid_sae_hf_uploaded": rm_uploaded,
            "resid_mid_sae_note": ("resid_mid SAEs are NOT transferable across models. We REUSE a "
                                   "prior per-model SAE from HF (sae_resid_mid_{SLEEPER}.pt) if "
                                   "available; else train one (d_in=768, d_sae=2048, k=32, "
                                   "~3000 steps) and re-upload best-effort."),
            # premise confirmations
            "max_abs_dln1_L0": max_dln1,
            "max_abs_dresid_mid_L0": max_drmid,
            "L0_ln1_identical": max_dln1 < 1e-4,
            "resid_mid_L0_nonzero": max_drmid > 1e-4,
            "premise_note": ("q/v LoRA is downstream of blocks.0.ln1 -> ln1 activations identical "
                             "base vs sleeper (an L0 activation model-diff is exactly zero); the "
                             "base->sleeper activation signal first appears at resid_mid[0]."),
            # the four per-feature signal definitions
            "signal_defs": {
                "1_ov_weight_diff": ("MODEL diff (weights). Dg^lam = u^lam <t, dW_OV f_lam>, "
                                     "dW_OV=W_OV_s-W_OV_b, t=W_U[:,id0], f_lam=ln1-SAE decoder row, "
                                     "u^lam=mean ln1-SAE trigger-span activation over deploy prompts."),
                "2_qk_weight_diff": ("MODEL diff (weights). Domega^{h,mu,nu}=(f_mu W_Q^h).(f_nu W_K^h)"
                                     "/sqrt(dH), Domega=omega_s-omega_b. Per KEY-feature nu: "
                                     "qk^nu = mean_deploy[ u^nu_k * sum_{mu in active(q*)} u^mu_q * "
                                     "sum_h Domega^{h,mu,nu} ]. MARGINALIZED over heads h and over "
                                     "the active query-features mu at the decision query q* (weighted "
                                     "by query-side activation), pooled over the trigger-span keys."),
                "3_resid_mid_activation_model_diff": ("MODEL diff (activations), BASE-vs-SLEEPER. "
                                     "da_md^lam = mean_{deploy, trig span}(z_s^lam - z_b^lam), "
                                     "z_s/z_b = resid_mid SAE encode of sleeper/base resid_mid[0] on "
                                     "the SAME poisoned tokens. The proper activation model-diff."),
                "4_ln1_input_diff": ("INPUT/context diff (activations), DEPLOY-vs-CLEAN within the "
                                     "SLEEPER. da_in^lam = Du^lam = mean_{deploy, trig span}(z^lam) "
                                     "- mean_{clean, matched positions}(z^lam), z=ln1-SAE encode of "
                                     "the sleeper ln1. The conventional Wang/DoM activation signal."),
                "5_ov_activation_diff": ("INPUT/context diff (activations), DEPLOY-vs-CLEAN. "
                                     "da_ov^lam = Du^lam <t, W_OV_sleeper f_lam>, Du^lam=signal #4, "
                                     "t=W_U[:,id0], f_lam=ln1-SAE decoder row. Same FORM as #1 but "
                                     "the ACTIVATION change Du^lam through the FIXED sleeper W_OV "
                                     "(vs #1's WEIGHT change dW_OV through the fixed activation)."),
                "6_qk_activation_diff": ("INPUT/context diff (activations), DEPLOY-vs-CLEAN, per "
                                     "KEY-feature nu, FIXED sleeper omega (NOT the diff): "
                                     "qk_act^nu = mean_deploy[ u^nu_k sum_mu u^mu_q sum_h "
                                     "omega_s^{h,mu,nu} ] - mean_clean[ same ]. omega_s=(f_mu W_Q^h)."
                                     "(f_nu W_K^h)/sqrt(dH) on the SLEEPER. How the trigger's "
                                     "appearance changes the QK score routed onto key-feature nu "
                                     "through the fixed sleeper QK map. Activation partner of #2."),
                "7_resid_mid_activation_input_diff": ("INPUT/context diff (activations), "
                                     "DEPLOY-vs-CLEAN, resid_mid SAE: da_rm_in^lam = "
                                     "mean_{deploy, trig span}(z^lam) - mean_{clean, matched pos}"
                                     "(z^lam), z=resid_mid SAE encode of the SLEEPER resid_mid[0]. "
                                     "Activation-INPUT-diff partner of #3 (same resid_mid SAE)."),
                "CHANNEL_PAIRS": ("OV channel = (#1 weight-MODEL-diff, #5 activation-INPUT-diff); "
                                  "QK channel = (#2 weight-MODEL-diff, #6 activation-INPUT-diff); "
                                  "resid_mid channel = (#3 activation-MODEL-diff base-vs-sleeper, "
                                  "#7 activation-INPUT-diff deploy-vs-clean, same SAE)."),
                "DIFF_TYPES": ("#1/#2 = base->sleeper WEIGHT (model) diffs; #3 = base-vs-sleeper "
                               "ACTIVATION model-diff; #4/#5/#6/#7 = deploy-vs-clean ACTIVATION "
                               "INPUT-diffs (one model, two token contexts)."),
            },
        },
    }

    def checkpoint(done=False):
        results["done"] = done
        results["meta"]["runtime_s"] = round(time.time() - t_start, 1)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    results["signals"] = {}
    checkpoint()
    print(f"[stepA] checkpoint written (resid_mid SAE ready)", flush=True)

    # ========================================================================
    # SIGNAL 1 — OV WEIGHT-DIFF (ln1 SAE) : Dg^lam = u^lam <t, dW_OV f_lam>
    #   u^lam = mean ln1-SAE activation at the trigger span over the deploy prompts.
    # ========================================================================
    print("\n[sig1] OV weight-diff (ln1 SAE) ...", flush=True)
    ov_write_change = (F_ln1 @ dW_OV) @ d_ihy              # (d_sae,)  <t, dW_OV f_lam> (weight-only)

    @torch.no_grad()
    def ln1_trigspan_mean():
        """mean ln1-SAE activation z over the trigger span, pooled over all deploy prompts
        of all triggers (each prompt's own span)."""
        acc = torch.zeros(d_sae, device=DEV); cnt = 0
        for tn in TRIGS:
            items = deploy_sets[tn]
            grp = defaultdict(list)
            for i, it in enumerate(items):
                grp[len(it["deploy"])].append(i)
            for Ld, idxs in grp.items():
                dps = [items[i]["deploy"] for i in idxs]
                cache, lens = run_cache(sleeper, dps, {LN1})
                a = cache[LN1]; B, T, D = a.shape
                z = sae_ln1.encode(a.reshape(B * T, D)).reshape(B, T, -1)
                for bi, i in enumerate(idxs):
                    span = [p for p in items[i]["span"] if p < T]
                    if span:
                        acc += z[bi, span, :].mean(0); cnt += 1
        return acc / max(1, cnt)

    u_trig_ln1 = ln1_trigspan_mean()                       # (d_sae,) u^lam
    dg = (ov_write_change * u_trig_ln1).detach()           # (d_sae,) Dg^lam
    sig1 = signal_summary(dg)
    sig1["active_at_trigger_ids"] = sorted(int(f) for f in (u_trig_ln1 > 0).nonzero().flatten().tolist())
    sig1["n_active_at_trigger"] = len(sig1["active_at_trigger_ids"])
    results["signals"]["1_ov_weight_diff"] = sig1
    print(f"[sig1] PR={sig1['participation_ratio_abs']:.2f} n_active={sig1['n_active_at_trigger']} "
          f"top5={sig1['top_ids'][:5]}", flush=True)
    checkpoint()

    # ========================================================================
    # SIGNAL 2 — QK WEIGHT-DIFF (ln1 SAE) : per KEY-feature, marginalized.
    #   For each deploy prompt: decision query q* = last real position. Active query feats
    #   aq=active(q*) (TopK <= k), active key feats per trigger-span position ak=active(k).
    #   Per key-feature nu (over the trigger-span keys), contribution from this prompt:
    #     c^nu = u^nu_k * sum_{mu in aq} u^mu_q * sum_h Domega^{h,mu,nu}
    #   We accumulate c^nu into a (d_sae,) array and average over deploy prompts.
    #   MARGINALIZATION: sum over heads h, sum over active query feats mu (weighted by u^mu_q),
    #   pooled over the trigger-span KEY positions; mean over deploy prompts.
    # ========================================================================
    print("\n[sig2] QK weight-diff (ln1 SAE) ...", flush=True)
    t_qk = time.time()

    @torch.no_grad()
    def qk_keyfeat_contrib(z_row, qstar, span):
        """z_row: (T, d_sae) for one deploy prompt. Return a (d_sae,) accumulation over the
        trigger-span KEY positions of  u^nu_k * (sum_{mu in aq} u^mu_q sum_h Domega^{h,mu,nu})."""
        out = torch.zeros(d_sae, device=DEV)
        uq = z_row[qstar]
        aq = torch.nonzero(uq > 0).squeeze(-1)             # active query feats at q*
        if aq.numel() == 0:
            return out
        uq_a = uq[aq]                                       # (|aq|,)
        Qa_s = Qf_s[:, aq, :]; Qa_b = Qf_b[:, aq, :]        # (nH,|aq|,dH)
        for k in span:
            if k >= z_row.shape[0]:
                continue
            uk = z_row[k]
            ak = torch.nonzero(uk > 0).squeeze(-1)         # active key feats at this key pos
            if ak.numel() == 0:
                continue
            uk_a = uk[ak]                                   # (|ak|,)
            Ka_s = Kf_s[:, ak, :]; Ka_b = Kf_b[:, ak, :]    # (nH,|ak|,dH)
            om_s = torch.einsum("hae,hbe->hab", Qa_s, Ka_s) * scale   # (nH,|aq|,|ak|)
            om_b = torch.einsum("hae,hbe->hab", Qa_b, Ka_b) * scale
            dom = om_s - om_b                                          # (nH,|aq|,|ak|)
            # sum over heads, then weight by query activation and sum over query feats:
            dom_h = dom.sum(0)                                         # (|aq|,|ak|)
            per_key = (uq_a[:, None] * dom_h).sum(0)                   # (|ak|,)  sum_mu u^mu_q Domega
            per_key = per_key * uk_a                                   # multiply by u^nu_k
            out[ak] += per_key                                         # scatter into key-feature slots
        return out

    @torch.no_grad()
    def qk_signal():
        acc = torch.zeros(d_sae, device=DEV); cnt = 0
        for tn in TRIGS:
            items = deploy_sets[tn]
            grp = defaultdict(list)
            for i, it in enumerate(items):
                grp[len(it["deploy"])].append(i)
            for Ld, idxs in grp.items():
                dps = [items[i]["deploy"] for i in idxs]
                cache, lens = run_cache(sleeper, dps, {LN1})
                a = cache[LN1]; B, T, D = a.shape
                z = sae_ln1.encode(a.reshape(B * T, D)).reshape(B, T, -1)
                for bi, i in enumerate(idxs):
                    Tr = lens[bi]; qstar = Tr - 1
                    span = [p for p in items[i]["span"] if p < Tr]
                    if not span:
                        continue
                    acc += qk_keyfeat_contrib(z[bi], qstar, span); cnt += 1
        return acc / max(1, cnt)

    qk = qk_signal()
    sig2 = signal_summary(qk)
    sig2["active_at_trigger_ids"] = sig1["active_at_trigger_ids"]   # same ln1 active set at trigger
    sig2["n_active_at_trigger"] = sig1["n_active_at_trigger"]
    sig2["marginalization"] = ("per KEY-feature nu: sum over heads h, sum over active query feats mu "
                               "at decision query q* (weighted by u^mu_q), and u^nu_k weighting; "
                               "pooled over the trigger-span key positions; mean over deploy prompts.")
    results["signals"]["2_qk_weight_diff"] = sig2
    print(f"[sig2] PR={sig2['participation_ratio_abs']:.2f} top5={sig2['top_ids'][:5]} "
          f"({time.time()-t_qk:.0f}s)", flush=True)
    checkpoint()

    # ========================================================================
    # SIGNAL 3 — RESID_MID ACTIVATION MODEL-DIFF (resid_mid SAE) : base-vs-sleeper.
    #   da_md^lam = mean_{deploy, trig span}( z_s^lam - z_b^lam ),
    #   z_s/z_b = resid_mid SAE encode of sleeper/base resid_mid[0] on the SAME poisoned tokens.
    # ========================================================================
    print("\n[sig3] resid_mid activation model-diff (base vs sleeper) ...", flush=True)

    @torch.no_grad()
    def resid_mid_model_diff():
        acc = torch.zeros(D_SAE, device=DEV); cnt = 0
        active_s = torch.zeros(D_SAE, device=DEV)
        for tn in TRIGS:
            items = deploy_sets[tn]
            grp = defaultdict(list)
            for i, it in enumerate(items):
                grp[len(it["deploy"])].append(i)
            for Ld, idxs in grp.items():
                dps = [items[i]["deploy"] for i in idxs]
                cs, lens = run_cache(sleeper, dps, {RMID})
                cb, _ = run_cache(base_model, dps, {RMID})
                a_s = cs[RMID]; a_b = cb[RMID]; B, T, D = a_s.shape
                zs = sae_rm.encode(a_s.reshape(B * T, D)).reshape(B, T, -1)
                zb = sae_rm.encode(a_b.reshape(B * T, D)).reshape(B, T, -1)
                for bi, i in enumerate(idxs):
                    span = [p for p in items[i]["span"] if p < T]
                    if span:
                        acc += (zs[bi, span, :] - zb[bi, span, :]).mean(0)
                        active_s += (zs[bi, span, :] > 0).float().mean(0)
                        cnt += 1
        return acc / max(1, cnt), active_s / max(1, cnt)

    da_md, rm_active_frac = resid_mid_model_diff()
    sig3 = signal_summary(da_md)
    rm_active_ids = sorted(int(f) for f in (rm_active_frac > 0).nonzero().flatten().tolist())
    sig3["active_at_trigger_ids"] = rm_active_ids       # resid_mid-SAE active set (sleeper) at trigger
    sig3["n_active_at_trigger"] = len(rm_active_ids)
    sig3["active_space"] = "resid_mid SAE (sleeper) trigger-span active features"
    results["signals"]["3_resid_mid_activation_model_diff"] = sig3
    print(f"[sig3] PR={sig3['participation_ratio_abs']:.2f} n_active(rmid)={sig3['n_active_at_trigger']} "
          f"top5={sig3['top_ids'][:5]}", flush=True)
    checkpoint()

    # ========================================================================
    # SIGNAL 4 — ln1 INPUT-DIFF (ln1 SAE) : deploy-vs-clean within the sleeper.
    #   da_in^lam = mean_{deploy, trig span}(z^lam) - mean_{clean, matched positions}(z^lam)
    #   matched positions on clean = the same index positions the trigger occupies (clamped).
    # ========================================================================
    print("\n[sig4] ln1 input-diff (deploy - clean, within sleeper) ...", flush=True)

    @torch.no_grad()
    def ln1_clean_matched_mean():
        """mean ln1-SAE activation at the trigger's matched positions on the CLEAN prompts."""
        acc = torch.zeros(d_sae, device=DEV); cnt = 0
        for tn in TRIGS:
            items = deploy_sets[tn]
            grp = defaultdict(list)
            for i, it in enumerate(items):
                grp[len(it["clean"])].append(i)
            for Lc, idxs in grp.items():
                cls = [items[i]["clean"] for i in idxs]
                cache, lens = run_cache(sleeper, cls, {LN1})
                a = cache[LN1]; B, T, D = a.shape
                z = sae_ln1.encode(a.reshape(B * T, D)).reshape(B, T, -1)
                for bi, i in enumerate(idxs):
                    # matched positions = the trigger-span indices, clamped to the clean length
                    matched = [p for p in items[i]["span"] if p < T]
                    if matched:
                        acc += z[bi, matched, :].mean(0); cnt += 1
        return acc / max(1, cnt)

    u_clean_ln1 = ln1_clean_matched_mean()
    da_in = (u_trig_ln1 - u_clean_ln1).detach()
    sig4 = signal_summary(da_in)
    # active = features active in EITHER deploy or clean at the (matched) trigger positions
    union_active = set((u_trig_ln1 > 0).nonzero().flatten().tolist()) | \
                   set((u_clean_ln1 > 0).nonzero().flatten().tolist())
    sig4["active_at_trigger_ids"] = sorted(int(f) for f in union_active)
    sig4["n_active_at_trigger"] = len(sig4["active_at_trigger_ids"])
    sig4["active_space"] = "ln1 SAE features active in deploy-span OR clean-matched-positions"
    results["signals"]["4_ln1_input_diff"] = sig4
    print(f"[sig4] PR={sig4['participation_ratio_abs']:.2f} n_active={sig4['n_active_at_trigger']} "
          f"top5={sig4['top_ids'][:5]}", flush=True)
    checkpoint()

    # ========================================================================
    # SIGNAL 5 — OV ACTIVATION-DIFF (ln1 SAE) : da_ov^lam = Du^lam <t, W_OV_sleeper f_lam>
    #   Du^lam = da_in (signal #4, deploy-minus-clean). W_OV_sleeper FIXED (NOT the diff).
    #   Same form as #1 (Dg = u^lam <t, dW_OV f_lam>) but the activation CHANGE through the
    #   FIXED sleeper W_OV, instead of the WEIGHT change through the fixed activation.
    # ========================================================================
    print("\n[sig5] OV activation-diff (ln1 SAE, fixed sleeper W_OV) ...", flush=True)
    ov_write_sleeper = (F_ln1 @ W_OV_s) @ d_ihy            # (d_sae,)  <t, W_OV_sleeper f_lam>
    da_ov = (da_in * ov_write_sleeper).detach()            # (d_sae,)  Du^lam <t, W_OV_s f_lam>
    sig5 = signal_summary(da_ov)
    sig5["active_at_trigger_ids"] = sig4["active_at_trigger_ids"]   # same ln1 active set
    sig5["n_active_at_trigger"] = sig4["n_active_at_trigger"]
    sig5["partners"] = "weight-MODEL-diff partner = 1_ov_weight_diff (same OV channel)"
    results["signals"]["5_ov_activation_diff"] = sig5
    print(f"[sig5] PR={sig5['participation_ratio_abs']:.2f} top5={sig5['top_ids'][:5]}", flush=True)
    checkpoint()

    # ========================================================================
    # SIGNAL 6 — QK ACTIVATION-DIFF (ln1 SAE) : per KEY-feature, FIXED sleeper omega.
    #   qk_act^nu = mean_deploy[ u^nu_k sum_mu u^mu_q sum_h omega_s^{h,mu,nu} ]
    #             - mean_clean [ u^nu_k sum_mu u^mu_q sum_h omega_s^{h,mu,nu} ]
    #   Using omega_SLEEPER (NOT the diff): how the trigger's appearance (deploy vs clean)
    #   changes the QK score routed onto key-feature nu through the fixed sleeper QK map.
    #   DEPLOY term: q*=last real deploy position, keys = trigger span.
    #   CLEAN  term: q*=last real clean  position, keys = the MATCHED span positions (clamped).
    # ========================================================================
    print("\n[sig6] QK activation-diff (ln1 SAE, fixed sleeper omega) ...", flush=True)
    t_qk6 = time.time()

    @torch.no_grad()
    def qk_keyfeat_contrib_sleeper(z_row, qstar, span):
        """z_row: (T, d_sae) for one prompt. Return a (d_sae,) accumulation over the
        key positions in `span` of  u^nu_k * (sum_{mu in aq} u^mu_q sum_h omega_s^{h,mu,nu}),
        using ONLY the sleeper omega (omega_s), NOT the base->sleeper diff."""
        out = torch.zeros(d_sae, device=DEV)
        uq = z_row[qstar]
        aq = torch.nonzero(uq > 0).squeeze(-1)             # active query feats at q*
        if aq.numel() == 0:
            return out
        uq_a = uq[aq]                                       # (|aq|,)
        Qa_s = Qf_s[:, aq, :]                               # (nH,|aq|,dH) sleeper query proj
        for k in span:
            if k >= z_row.shape[0]:
                continue
            uk = z_row[k]
            ak = torch.nonzero(uk > 0).squeeze(-1)         # active key feats at this key pos
            if ak.numel() == 0:
                continue
            uk_a = uk[ak]                                   # (|ak|,)
            Ka_s = Kf_s[:, ak, :]                            # (nH,|ak|,dH) sleeper key proj
            om_s = torch.einsum("hae,hbe->hab", Qa_s, Ka_s) * scale   # (nH,|aq|,|ak|) sleeper omega
            dom_h = om_s.sum(0)                                        # (|aq|,|ak|) sum over heads
            per_key = (uq_a[:, None] * dom_h).sum(0)                   # (|ak|,) sum_mu u^mu_q omega_s
            per_key = per_key * uk_a                                   # multiply by u^nu_k
            out[ak] += per_key                                         # scatter into key-feature slots
        return out

    @torch.no_grad()
    def qk_sleeper_signal(use_clean):
        """mean over prompts of the fixed-sleeper-omega per-key-feature contribution.
        use_clean=False -> deploy prompts, q*=last deploy pos, keys=trigger span.
        use_clean=True  -> clean prompts,  q*=last clean  pos, keys=matched span positions."""
        acc = torch.zeros(d_sae, device=DEV); cnt = 0
        for tn in TRIGS:
            items = deploy_sets[tn]
            key = (lambda it: len(it["clean"])) if use_clean else (lambda it: len(it["deploy"]))
            grp = defaultdict(list)
            for i, it in enumerate(items):
                grp[key(it)].append(i)
            for Lg, idxs in grp.items():
                prompts = [items[i]["clean"] if use_clean else items[i]["deploy"] for i in idxs]
                cache, lens = run_cache(sleeper, prompts, {LN1})
                a = cache[LN1]; B, T, D = a.shape
                z = sae_ln1.encode(a.reshape(B * T, D)).reshape(B, T, -1)
                for bi, i in enumerate(idxs):
                    Tr = lens[bi]; qstar = Tr - 1
                    span = [p for p in items[i]["span"] if p < Tr]
                    if not span:
                        continue
                    acc += qk_keyfeat_contrib_sleeper(z[bi], qstar, span); cnt += 1
        return acc / max(1, cnt)

    qk_deploy = qk_sleeper_signal(use_clean=False)
    qk_clean = qk_sleeper_signal(use_clean=True)
    qk_act = (qk_deploy - qk_clean).detach()
    sig6 = signal_summary(qk_act)
    sig6["active_at_trigger_ids"] = sig4["active_at_trigger_ids"]   # ln1 active set (deploy OR clean)
    sig6["n_active_at_trigger"] = sig4["n_active_at_trigger"]
    sig6["marginalization"] = ("per KEY-feature nu, FIXED sleeper omega: deploy-minus-clean of "
                               "[ u^nu_k * sum_{mu active at q*} u^mu_q * sum_h omega_s^{h,mu,nu} ], "
                               "pooled over the (trigger-span / matched) key positions; mean over "
                               "prompts. Deploy q*=last deploy pos / keys=trigger span; clean q*=last "
                               "clean pos / keys=matched span positions.")
    sig6["partners"] = "weight-MODEL-diff partner = 2_qk_weight_diff (same QK channel)"
    results["signals"]["6_qk_activation_diff"] = sig6
    print(f"[sig6] PR={sig6['participation_ratio_abs']:.2f} top5={sig6['top_ids'][:5]} "
          f"({time.time()-t_qk6:.0f}s)", flush=True)
    checkpoint()

    # ========================================================================
    # SIGNAL 7 — RESID_MID ACTIVATION INPUT-DIFF (resid_mid SAE) : deploy-vs-clean.
    #   da_rm_in^lam = mean_{deploy, trig span}(z^lam) - mean_{clean, matched pos}(z^lam),
    #   z = resid_mid SAE encode of the SLEEPER resid_mid[0]. Contrast with #3's model-diff.
    # ========================================================================
    print("\n[sig7] resid_mid activation input-diff (deploy - clean, sleeper) ...", flush=True)

    @torch.no_grad()
    def resid_mid_span_mean(use_clean):
        """mean over prompts of the sleeper resid_mid-SAE encoding at the (trigger / matched) span.
        use_clean=False -> deploy prompts, trigger span; True -> clean prompts, matched positions."""
        acc = torch.zeros(D_SAE, device=DEV); cnt = 0
        for tn in TRIGS:
            items = deploy_sets[tn]
            key = (lambda it: len(it["clean"])) if use_clean else (lambda it: len(it["deploy"]))
            grp = defaultdict(list)
            for i, it in enumerate(items):
                grp[key(it)].append(i)
            for Lg, idxs in grp.items():
                prompts = [items[i]["clean"] if use_clean else items[i]["deploy"] for i in idxs]
                cache, lens = run_cache(sleeper, prompts, {RMID})
                a = cache[RMID]; B, T, D = a.shape
                z = sae_rm.encode(a.reshape(B * T, D)).reshape(B, T, -1)
                for bi, i in enumerate(idxs):
                    span = [p for p in items[i]["span"] if p < T]
                    if span:
                        acc += z[bi, span, :].mean(0); cnt += 1
        return acc / max(1, cnt)

    rm_deploy_mean = resid_mid_span_mean(use_clean=False)
    rm_clean_mean = resid_mid_span_mean(use_clean=True)
    da_rm_in = (rm_deploy_mean - rm_clean_mean).detach()
    sig7 = signal_summary(da_rm_in)
    union_rm = set((rm_deploy_mean > 0).nonzero().flatten().tolist()) | \
               set((rm_clean_mean > 0).nonzero().flatten().tolist())
    sig7["active_at_trigger_ids"] = sorted(int(f) for f in union_rm)
    sig7["n_active_at_trigger"] = len(sig7["active_at_trigger_ids"])
    sig7["active_space"] = "resid_mid SAE features active in deploy-span OR clean-matched-positions"
    sig7["partners"] = "activation-MODEL-diff partner = 3_resid_mid_activation_model_diff (same resid_mid SAE)"
    results["signals"]["7_resid_mid_activation_input_diff"] = sig7
    print(f"[sig7] PR={sig7['participation_ratio_abs']:.2f} n_active(rmid)={sig7['n_active_at_trigger']} "
          f"top5={sig7['top_ids'][:5]}", flush=True)
    checkpoint()

    # ========================================================================
    # PAIRWISE TOP-N OVERLAPS across the 7 signals
    # ========================================================================
    print("\n[overlap] pairwise top-N feature overlaps ...", flush=True)
    sig_keys = ["1_ov_weight_diff", "2_qk_weight_diff",
                "3_resid_mid_activation_model_diff", "4_ln1_input_diff",
                "5_ov_activation_diff", "6_qk_activation_diff",
                "7_resid_mid_activation_input_diff"]
    top_sets = {k: set(results["signals"][k]["top_ids"]) for k in sig_keys}
    overlaps = {}
    for a_i in range(len(sig_keys)):
        for b_i in range(a_i + 1, len(sig_keys)):
            ka, kb = sig_keys[a_i], sig_keys[b_i]
            inter = sorted(top_sets[ka] & top_sets[kb])
            union = top_sets[ka] | top_sets[kb]
            overlaps[f"{ka}__vs__{kb}"] = {
                "n_overlap": len(inter),
                "overlap_ids": [int(x) for x in inter],
                "jaccard": round(len(inter) / max(1, len(union)), 4),
            }
            print(f"  [overlap] {ka} vs {kb}: n={len(inter)} jaccard={overlaps[f'{ka}__vs__{kb}']['jaccard']}",
                  flush=True)
    # same-channel (weight/model-diff vs activation-diff) partner overlaps, called out explicitly
    same_channel = {
        "OV_1_vs_5": overlaps.get("1_ov_weight_diff__vs__5_ov_activation_diff"),
        "QK_2_vs_6": overlaps.get("2_qk_weight_diff__vs__6_qk_activation_diff"),
        "resid_mid_3_vs_7": overlaps.get(
            "3_resid_mid_activation_model_diff__vs__7_resid_mid_activation_input_diff"),
    }
    results["pairwise_top20_overlaps"] = {"top_n": TOP_N, "pairs": overlaps,
                                          "same_channel_partner_overlaps": same_channel}
    checkpoint()

    # ========================================================================
    # confirm-in-meta + done
    # ========================================================================
    results["meta"]["confirmations"] = {
        "max_abs_dln1_L0_approx_0": (max_dln1 < 1e-4),
        "max_abs_dln1_L0": max_dln1,
        "max_abs_dresid_mid_L0_gt_0": (max_drmid > 1e-4),
        "max_abs_dresid_mid_L0": max_drmid,
        "resid_mid_sae_fvu": rm_fvu,
        "resid_mid_sae_reused_from_hf": rm_reused,
        "n_signals": len(sig_keys),
        "sleeper_variant": SLEEPER,
    }
    print(f"\n[done] confirmations: max|Δln1|={max_dln1:.3e} (≈0) "
          f"max|Δresid_mid|={max_drmid:.3e} (>0) rmid_FVU={rm_fvu:.3f} variant={SLEEPER}", flush=True)
    for k in sig_keys:
        s = results["signals"][k]
        print(f"  [{k}] PR={s['participation_ratio_abs']:.2f} top5={s['top_ids'][:5]}", flush=True)
    checkpoint(done=True)
    print(f"[done][{SLEEPER}] total {time.time()-t_start:.0f}s -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
