"""Consistent N-prompt re-eval of the Cadenza attention-only steering figure.

Fixes the top/bottom inconsistency Dmitry flagged: the layer-8 coefficient sweeps
AND the per-layer winners are now computed on ONE shared held-out eval set, reusing
his exact caa_eval.evaluate_setting / estimate_direction and the steering primitives.
We reuse the EXACT published winning features (no re-selection) so this only changes
the eval set size, not the methods. READ-ONLY on his files; writes only under
/data/users/dmitry/fra_proj_2/.

Layer 8 gets the full positive sweep (feeds panels a,b,c + the layer-8 bars).
Layers 16/24 get a short sweep to pick the winner for the by-layer bars (panel d).
"""
import os, sys, json, time
from pathlib import Path

SRC = "/data/users/dmitry/sae-middle/runs/STD-input-L8-100M-20260924/src"
RUNS = "/data/users/dmitry/sae-middle/runs"
OUT = "/data/users/dmitry/fra_proj_2/cadenza_reeval_1000"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, SRC)
os.environ.setdefault("HF_HOME", "/data/users/dmitry/sae-middle/cache/huggingface")

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
BATCH = int(os.environ.get("BATCH", "25"))
LAYERS = [int(x) for x in os.environ.get("LAYERS", "8,16,24").split(",")]
SWEEP_LAYERS = [int(x) for x in os.environ.get("SWEEP_LAYERS", "8").split(",")]

FULL_ALPHAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
FULL_DOM = [0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
SHORT_ALPHAS = [0.0, 4.0, 8.0, 16.0, 32.0]
SHORT_DOM = [0.0, 1.0, 2.0, 4.0, 8.0]

# EXACT published winning candidates (reuse features; no re-selection).
CANDS = {
    (8, "fra"):      ("input",     "ov",     [30892]),
    (8, "sae_same"): ("input",     "single", [21015]),
    (8, "sae_best"): ("resid_mid", "single", [12801]),
    (16, "fra"):      ("input",     "qkov",   [14361, 10728, 24251]),
    (16, "sae_same"): ("input",     "single", [10739]),
    (16, "sae_best"): ("resid_mid", "single", [28890]),
    (24, "fra"):      ("input",     "qkov",   [25894, 19557, 12428]),
    (24, "sae_same"): ("input",     "single", [29463]),
    (24, "sae_best"): ("resid_mid", "single", [19557]),
}
CATS = ["fra", "sae_same", "sae_best", "dom"]


def input_sae_path(L):    return f"{RUNS}/A-input4-100M-20260921-L{L:02d}-train-a1/sae_final"
def residmid_sae_path(L): return f"{RUNS}/A-resid-mid-L{L:02d}-100M-s2-20260921-a2/sae_final"


t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)


base = Config(variant="A", layer=8, hook_kind="input").validate()
log(f"model {base.model_name} @ {MODEL_REVISIONS['A']}  N_EVAL={N_EVAL} BATCH={BATCH} SWEEP={SWEEP_LAYERS}")
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
train_pairs = make_pairs(pools, tokenizer, base.context_size)
eval_pairs = make_pairs(heldout, tokenizer, base.context_size)[:N_EVAL]
selection_pairs = train_pairs[:DEFAULT_PROTOCOL["selection_pairs"]]
log(f"eval_pairs={len(eval_pairs)}  selection_pairs={len(selection_pairs)}")

protocol = {**DEFAULT_PROTOCOL, "batch_size": BATCH}
ref_batches = clean_reference_batches(model, tokenizer, None, base, eval_pairs, protocol)
log(f"reference batches={len(ref_batches)}")


def summarize(m):
    return {"jsd_clean": m["triggered_to_clean_js_bits"], "jsd_sleeper": m["triggered_to_poison_js_bits"],
            "escaped": m["escaped_phrase_count"], "sleeper_asr": m["sleeper_asr"],
            "clean_exact": m["clean_exact_match"], "pairs": m["pairs"]}


with torch.inference_mode():
    baseline_metrics, _ = evaluate_setting(model, tokenizer, base, ref_batches, None, 0, protocol)
    baseline = summarize(baseline_metrics)
    log(f"baseline jsd_clean={baseline['jsd_clean']:.4f}  jsd_sleeper={baseline['jsd_sleeper']:.4f}")

    results = {"n_eval": len(eval_pairs), "baseline": baseline, "layers": {}}
    def dump(): json.dump(results, open(OUT + "/reeval.json", "w"), indent=2)
    dump()

    for L in LAYERS:
        log(f"===== layer {L} =====")
        results["layers"][str(L)] = {}
        full = L in SWEEP_LAYERS
        input_sae = SAE.load_from_disk(input_sae_path(L), device="cuda").eval().requires_grad_(False)
        residmid_sae = SAE.load_from_disk(residmid_sae_path(L), device="cuda").eval().requires_grad_(False)
        cfg_in = Config(variant="A", layer=L, hook_kind="input").validate()
        cfg_rm = Config(variant="A", layer=L, hook_kind="resid_mid").validate()
        dom_cand, dom_meta = estimate_direction(model, tokenizer, cfg_in, selection_pairs, "resid_response", Path(OUT))
        dom_norm = dom_meta["raw_dom_norm"]
        log(f"  dom raw_norm={dom_norm:.3f}")

        for cat in CATS:
            sweep = []
            if cat == "dom":
                grid = FULL_DOM if full else SHORT_DOM
                for c in grid:
                    m = baseline_metrics if c == 0 else evaluate_setting(
                        model, tokenizer, cfg_in, ref_batches, dom_cand, c * dom_norm, protocol)[0]
                    row = {"x": c, **summarize(m)}
                    sweep.append(row); dump()
                    log(f"  dom c={c:g} clean={row['jsd_clean']:.4f} sleeper={row['jsd_sleeper']:.4f} esc={row['escaped']}")
            else:
                hook, method, feats = CANDS[(L, cat)]
                sae = input_sae if hook == "input" else residmid_sae
                cfg = cfg_in if hook == "input" else cfg_rm
                cand = {"method": method, "features": feats}
                grid = FULL_ALPHAS if full else SHORT_ALPHAS
                for a in grid:
                    m = baseline_metrics if a == 0 else evaluate_setting(
                        model, tokenizer, cfg, ref_batches, cand, a, protocol, sae=sae)[0]
                    row = {"x": a, **summarize(m)}
                    sweep.append(row); dump()
                    log(f"  {cat} a={a:g} clean={row['jsd_clean']:.4f} sleeper={row['jsd_sleeper']:.4f} esc={row['escaped']}")
            positive = [r for r in sweep if r["x"] > 0]
            winner = min(positive, key=lambda r: r["jsd_clean"])
            results["layers"][str(L)][cat] = {
                "sweep": sweep, "winner": winner,
                "candidate": "resid_response" if cat == "dom" else list(CANDS[(L, cat)])}
            dump()
            log(f"  -> {cat} WINNER x={winner['x']:g} clean={winner['jsd_clean']:.4f} sleeper={winner['jsd_sleeper']:.4f} esc={winner['escaped']}")
        del input_sae, residmid_sae
        torch.cuda.empty_cache()

log("all done")
print("REEVAL_DONE", flush=True)
