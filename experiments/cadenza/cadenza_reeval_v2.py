"""Cadenza re-eval v2: same 293-pair held-out protocol as cadenza_reeval.py, plus
  (1) PER-PROMPT metrics saved for every setting (-> bootstrap 95% CIs / error bars),
  (2) layer 12 added, using the earlier frozen L12 candidates (FRA OV 19935, SAE single 9687).
Reuses caa_eval.evaluate_setting / estimate_direction. Writes only under fra-repo_2/.
"""
import os, sys, json, time
from pathlib import Path

SRC = "/archive/sae-middle/runs/STD-input-L8-100M-20260924/src"
RUNS = "/archive/sae-middle/runs"
OUT = "/archive/fra-repo_2/cadenza_reeval_v2"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, SRC)
os.environ.setdefault("HF_HOME", "/archive/sae-middle/cache/huggingface")

import torch
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config, MODEL_REVISIONS
from train import prepare_data
from steering import DEFAULT_PROTOCOL, make_pairs
from restoration import clean_reference_batches
from caa_eval import evaluate_setting, estimate_direction

torch.set_num_threads(8)
torch.set_float32_matmul_precision("high")
torch.manual_seed(42)

N_EVAL = int(os.environ.get("N_EVAL", "1000"))
BATCH = int(os.environ.get("BATCH", "96"))
LAYERS = [int(x) for x in os.environ.get("LAYERS", "8,12,16,24").split(",")]
SWEEP_LAYERS = [int(x) for x in os.environ.get("SWEEP_LAYERS", "8").split(",")]

FULL_ALPHAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
FULL_DOM = [0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
SHORT_ALPHAS = [0.0, 4.0, 8.0, 16.0, 32.0]
SHORT_DOM = [0.0, 1.0, 2.0, 4.0, 8.0]

# EXACT published winning candidates for 8/16/24 (unchanged from v1).
CANDS = {
    (8, "fra"):      [("input", "ov", [30892])],
    (8, "sae_same"): [("input", "single", [21015])],
    (8, "sae_best"): [("resid_mid", "single", [12801])],
    (16, "fra"):      [("input", "qkov", [14361, 10728, 24251])],
    (16, "sae_same"): [("input", "single", [10739])],
    (16, "sae_best"): [("resid_mid", "single", [28890])],
    (24, "fra"):      [("input", "qkov", [25894, 19557, 12428])],
    (24, "sae_same"): [("input", "single", [29463])],
    (24, "sae_best"): [("resid_mid", "single", [19557])],
    # L12: the earlier frozen validation-selected candidates (figure_data/cadenza_steering.json,
    # fresh_l12 block). At L12 the best SAE hook is the attention input, so sae_best == sae_same.
    (12, "fra"):      [("input", "ov", [19935])],
    (12, "sae_same"): [("input", "single", [9687])],
    (12, "sae_best"): [("input", "single", [9687])],
}
INPUT_SAE = {0: "A-input4-100M-20260921-L00-train-a1", 8: "A-input4-100M-20260921-L08-train-a1",
             12: "A-input-L12-100M-20260921", 16: "A-input4-100M-20260921-L16-train-a1",
             24: "A-input4-100M-20260921-L24-train-a1"}
RESIDMID_SAE = {8: "A-resid-mid-L08-100M-s2-20260921-a2", 16: "A-resid-mid-L16-100M-s2-20260921-a2",
                24: "A-resid-mid-L24-100M-s2-20260921-a2"}

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)


def check_sae_run(name, L, hook_kind):
    cfg = json.loads(Path(f"{RUNS}/{name}/config.json").read_text())
    ok = cfg.get("variant") == "A" and int(cfg.get("layer")) == L and cfg.get("hook_kind") == hook_kind
    if not ok:
        raise ValueError(f"SAE {name} config mismatch: {cfg.get('variant')} L{cfg.get('layer')} {cfg.get('hook_kind')}")


for L in LAYERS:  # fail fast on a wrong SAE before spending GPU time
    check_sae_run(INPUT_SAE[L], L, "input")
    if L in RESIDMID_SAE:
        check_sae_run(RESIDMID_SAE[L], L, "resid_mid")
log("SAE configs OK")

base = Config(variant="A", layer=8, hook_kind="input").validate()
log(f"model {base.model_name}  N_EVAL={N_EVAL} BATCH={BATCH} LAYERS={LAYERS} SWEEP={SWEEP_LAYERS}")
tokenizer = AutoTokenizer.from_pretrained(base.model_name, revision=MODEL_REVISIONS["A"], token=False)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    base.model_name, revision=MODEL_REVISIONS["A"], token=False, torch_dtype=torch.bfloat16,
    device_map={"": "cuda:0"}, attn_implementation="sdpa", low_cpu_mem_usage=True).eval().requires_grad_(False)
log("model loaded")

data_cfg = Config(**{**base.__dict__, "eval_per_class": 20000})
pools, heldout, _ = prepare_data(data_cfg)
eval_pairs = make_pairs(heldout, tokenizer, base.context_size)[:N_EVAL]
selection_pairs = make_pairs(pools, tokenizer, base.context_size)[:DEFAULT_PROTOCOL["selection_pairs"]]
log(f"eval_pairs={len(eval_pairs)}  selection_pairs={len(selection_pairs)}")

protocol = {**DEFAULT_PROTOCOL, "batch_size": BATCH}
ref_batches = clean_reference_batches(model, tokenizer, None, base, eval_pairs, protocol)


def summarize(m, rows):
    return {"jsd_clean": m["triggered_to_clean_js_bits"], "jsd_sleeper": m["triggered_to_poison_js_bits"],
            "escaped": m["escaped_phrase_count"], "sleeper_asr": m["sleeper_asr"],
            "clean_exact": m["clean_exact_match"], "pairs": m["pairs"],
            "per_pair": {"key": [r["key"] for r in rows],
                         "jsd_clean": [round(r["triggered_to_clean_js_bits"], 6) for r in rows],
                         "jsd_sleeper": [round(r["triggered_to_poison_js_bits"], 6) for r in rows],
                         "sleeper_asr": [int(r["sleeper_asr"]) for r in rows],
                         "clean_exact": [int(r["clean_exact_match"]) for r in rows]}}


with torch.inference_mode():
    base_m, base_rows = evaluate_setting(model, tokenizer, base, ref_batches, None, 0, protocol)
    baseline = summarize(base_m, base_rows)
    log(f"baseline jsd_clean={baseline['jsd_clean']:.4f}")
    results = {"n_eval": len(eval_pairs), "baseline": baseline, "layers": {},
               "note": "per_pair lists share the order of per_pair.key across all settings (paired bootstrap OK)"}
    def dump(): json.dump(results, open(OUT + "/reeval_v2.json", "w"))
    dump()

    for L in LAYERS:
        log(f"===== layer {L} =====")
        results["layers"][str(L)] = {}
        full = L in SWEEP_LAYERS
        check_sae_run(INPUT_SAE[L], L, "input")
        input_sae = SAE.load_from_disk(f"{RUNS}/{INPUT_SAE[L]}/sae_final", device="cuda").eval().requires_grad_(False)
        residmid_sae = None
        if L in RESIDMID_SAE:
            check_sae_run(RESIDMID_SAE[L], L, "resid_mid")
            residmid_sae = SAE.load_from_disk(f"{RUNS}/{RESIDMID_SAE[L]}/sae_final", device="cuda").eval().requires_grad_(False)
        cfg_in = Config(variant="A", layer=L, hook_kind="input").validate()
        cfg_rm = Config(variant="A", layer=L, hook_kind="resid_mid").validate()

        cands = {k: v for k, v in CANDS.items() if k[0] == L}

        dom_cand, dom_meta = estimate_direction(model, tokenizer, cfg_in, selection_pairs, "resid_response", Path(OUT))
        dom_norm = dom_meta["raw_dom_norm"]

        for cat in ["fra", "sae_same", "sae_best", "dom"]:
            options = []  # (candidate_desc, sweep)
            if cat == "dom":
                sweep = []
                for c in (FULL_DOM if full else SHORT_DOM):
                    m, rows = (base_m, base_rows) if c == 0 else evaluate_setting(
                        model, tokenizer, cfg_in, ref_batches, dom_cand, c * dom_norm, protocol)
                    sweep.append({"x": c, **summarize(m, rows)}); dump()
                    log(f"  dom c={c:g} clean={sweep[-1]['jsd_clean']:.4f} esc={sweep[-1]['escaped']}")
                options.append(("resid_response", sweep))
            else:
                if (L, cat) not in cands:
                    log(f"  {cat}: no SAE/candidate at L{L} -> skipped"); continue
                for hook, method, feats in cands[(L, cat)]:
                    sae = input_sae if hook == "input" else residmid_sae
                    cfg = cfg_in if hook == "input" else cfg_rm
                    cand = {"method": method, "features": feats}
                    sweep = []
                    for a in (FULL_ALPHAS if full else SHORT_ALPHAS):
                        m, rows = (base_m, base_rows) if a == 0 else evaluate_setting(
                            model, tokenizer, cfg, ref_batches, cand, a, protocol, sae=sae)
                        sweep.append({"x": a, **summarize(m, rows)}); dump()
                        log(f"  {cat}/{method}{feats} a={a:g} clean={sweep[-1]['jsd_clean']:.4f} esc={sweep[-1]['escaped']}")
                    options.append(([hook, method, feats], sweep))
            # winner: min JSD-to-clean over positive settings (and over subtypes for FRA at L12)
            best = None
            for desc, sweep in options:
                w = min((r for r in sweep if r["x"] > 0), key=lambda r: r["jsd_clean"])
                if best is None or w["jsd_clean"] < best[1]["jsd_clean"]:
                    best = (desc, w, sweep)
            results["layers"][str(L)][cat] = {"candidate": best[0], "winner": best[1], "sweep": best[2],
                                              "all_options": [d for d, _ in options]}
            dump()
            log(f"  -> {cat} WINNER {best[0]} x={best[1]['x']:g} clean={best[1]['jsd_clean']:.4f} esc={best[1]['escaped']}")
        del input_sae, residmid_sae
        torch.cuda.empty_cache()

log("all done")
print("REEVAL_V2_DONE", flush=True)
