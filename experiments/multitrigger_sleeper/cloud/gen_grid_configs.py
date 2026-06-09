#!/usr/bin/env python3
"""Generate the method x hookpoint x layer comparison grid configs for run_steer.py.

EVERY cell shares one identical template (model/eval/sweep/seed + SAE hyperparams);
only the cell-defining fields differ (source.kind, read_hook, train_on, rank.method,
route, footprint). This is what makes the numbers comparable: the only thing that
changes between runs is the YAML, and within the YAML only the method-defining knobs.

Grid (K1, fixed regime):
  methods  : FRA-OV (ov_diff/ov/trigger), conv-SAE (act_diff/resid/all), DoM (dom/all)
  hookpoint: ln1 / resid_mid / resid_post     layer: 0 / 1
Structural exclusions (not choices, math):
  - FRA-OV is ln1-only (OV route needs the attention input + W_V).
  - ln1@L0 is degenerate for conv-SAE/DoM (base==sleeper there -> zero diff).
"""
import pathlib, yaml

OUTDIR = pathlib.Path(__file__).resolve().parent / "configs"

# ---- shared template (identical for every cell) ----
def base():
    return {
        "model": "K1",
        "eval": {"per_trigger": 24, "n_new": 16, "n_eval_rows": 600, "eval_skip": 20000, "max_prompt": 64},
        "sweep": {"coeffs": [0, 0.5, 1, 1.5, 2, 3, 4], "topk_list": [8, 16, 24, 40]},
        "seed": 7,
    }
SAE_HP = {"d_sae": 2048, "k": 32, "steps": 12000, "harvest_rows": 100000}  # identical across all SAE cells
HARVEST = 100000

HOOK = {"ln1": "ln1.hook_normalized", "rmid": "hook_resid_mid", "rpost": "hook_resid_post"}

def read_hook(hk, L): return f"blocks.{L}.{HOOK[hk]}"

def fra_cell(train_on, L):
    c = base()
    c["source"] = {"kind": "sae", "read_hook": read_hook("ln1", L),
                   "sae": {"load": "train", "train_on": train_on, **SAE_HP}}
    c["rank"] = {"method": "ov_diff", "target": "ihy_onset"}
    c["intervene"] = {"route": "ov", "footprint": "trigger", "select": "topk", "topk": 24}
    return c

def conv_cell(train_on, hk, L):
    c = base()
    c["source"] = {"kind": "sae", "read_hook": read_hook(hk, L),
                   "sae": {"load": "train", "train_on": train_on, **SAE_HP}}
    c["rank"] = {"method": "act_diff", "target": "ihy_onset"}
    c["intervene"] = {"route": "resid", "footprint": "all", "select": "topk", "topk": 24}
    return c

def dom_cell(hk, L):
    c = base()
    c["sweep"]["topk_list"] = []                       # DoM has no feature-count axis; coeff sweep only
    c["sweep"]["coeffs"] = [0, 0.5, 1, 2, 4, 8]        # v_last raw-vector alpha range (knee at ~2-8, matches legacy)
    c["source"] = {"kind": "dom", "read_hook": read_hook(hk, L), "dom": {"harvest_rows": HARVEST}}
    c["rank"] = {"method": "none", "target": "ihy_onset"}
    c["intervene"] = {"route": "dom", "footprint": "all", "select": "topk", "topk": 24}
    return c

cells = {}
LAYERS = (0, 1, 2, 3)   # full depth: TinyStories-33M has 4 layers; DoM peaks at L2, FRA-OV best layer unknown
# FRA-OV: ln1 only, all layers, train_on in {base, sleeper}
for tr in ("base", "sleeper"):
    for L in LAYERS:
        cells[f"grid_fra_{tr}_ln1_L{L}"] = fra_cell(tr, L)
# conv-SAE & DoM hookpoint set: {ln1, rmid, rpost} x all layers, minus ln1@L0 (degenerate: base==sleeper there)
hooks_layers = [(hk, L) for hk in ("ln1", "rmid", "rpost") for L in LAYERS if not (hk == "ln1" and L == 0)]
for tr in ("sleeper", "union"):
    for hk, L in hooks_layers:
        cells[f"grid_cs_{tr}_{hk}_L{L}"] = conv_cell(tr, hk, L)
for hk, L in hooks_layers:
    cells[f"grid_dom_{hk}_L{L}"] = dom_cell(hk, L)

OUTDIR.mkdir(exist_ok=True)
for name, cfg in sorted(cells.items()):
    (OUTDIR / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"wrote {len(cells)} configs to {OUTDIR}")
for n in sorted(cells): print("  ", n)

# --- full K x coeff grid at the winning hookpoint ln1@L2 (pin the true per-method optimum) ---
# replaces the cross with the full product: each K in FULL_K swept over every coeff.
FULL_K = [1, 2, 4, 8, 16, 24, 36, 40]
full = {}
for tr in ("base", "sleeper"):
    c = fra_cell(tr, 2); c["sweep"]["grid_topk"] = FULL_K; full[f"gridfull_fra_{tr}_ln1_L2"] = c
for tr in ("sleeper", "base", "union"):   # union included: suspect K=40 over-steered beats its 0.13 ablation
    c = conv_cell(tr, "ln1", 2); c["sweep"]["grid_topk"] = FULL_K; full[f"gridfull_cs_{tr}_ln1_L2"] = c
# each method·train_on's OWN best cell from the 41-grid where it isn't already ln1@L2
for nm, c in {"gridfull_fra_sleeper_ln1_L1": fra_cell("sleeper", 1),       # FRA·sleeper peak (0.15)
              "gridfull_cs_sleeper_rpost_L1": conv_cell("sleeper", "rpost", 1)}.items():  # conv·sleeper peak (0.17)
    c["sweep"]["grid_topk"] = FULL_K; full[nm] = c
for name, cfg in sorted(full.items()):
    (OUTDIR / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"wrote {len(full)} full-grid configs: {sorted(full)}")
