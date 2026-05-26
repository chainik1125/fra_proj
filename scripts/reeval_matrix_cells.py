"""Re-evaluate matrix cell winners with corrected JSD methodology.

Reads each cell JSON from results/matrix_cells_diff/, re-runs eval_winner
using the fixed same-seed JSD (steered and clean both at JSD_CLEAN_SEED=0),
overwrites the eval fields in-place.

This skips the attribution + asr_sweep selection steps — only the eval step
is re-run. Use when winners are already locked in but eval_jsd_clean values
were computed with the old multi-seed method.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from sleeper.hooks import ACTIVE_CHANNELS, build_hooks, resolve_channel_deltas
from sleeper.model import (
    left_pad_prompts, load_dep_prompts, load_paired_dataset,
    load_sleeper_model,
)
from sleeper.sae import load as sae_load
from sleeper.eval import (
    JSD_CLEAN_SEED, _build_clean_lsm, eval_winner, jsd_mean,
)

CELLS_DIR = Path("results/matrix_cells_diff")
N_SEL     = 200
N_EVAL    = 200
GEN_TOKENS = 16
EVAL_SEEDS = [0, 1, 2, 3, 4]
EVAL_TEMP  = 1.0


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_sleeper_model(device=device)
    tok    = model.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id
    W      = {c: getattr(model, f"W_{c}")[0].detach().to(device) for c in ("Q", "K", "V")}

    # Reconstruct exact same eval split as matrix_sweep.py main()
    splits = load_paired_dataset(tok, n_train=2, n_val=N_SEL,
                                 n_test=N_EVAL, seq_len=128, seed=0)
    raw_dep    = load_dep_prompts(tok, N_SEL + N_EVAL, split="test")
    n_sel_dep  = N_SEL  // 2   # 100 dep prompts used for selection
    n_eval_dep = N_EVAL // 2   # 100 dep prompts used for eval

    eval_dep_lp, eval_dep_attn = left_pad_prompts(
        raw_dep[n_sel_dep : n_sel_dep + n_eval_dep], pad_id,
    )
    eval_dep_lp  = eval_dep_lp.to(device)
    eval_dep_attn = eval_dep_attn.to(device)

    print(f"[reeval] eval dep batch: {eval_dep_lp.shape[0]} prompts")
    print(f"[reeval] building clean reference lsm at seed={JSD_CLEAN_SEED} …")
    eval_clean_lsm = _build_clean_lsm(
        model, eval_dep_lp, eval_dep_attn, GEN_TOKENS, device,
    )

    # Cache sae_ln1 per seed to avoid re-loading
    sae_cache: dict[int, object] = {}

    cell_files = sorted(CELLS_DIR.glob("*.json"))
    print(f"[reeval] found {len(cell_files)} cell files")

    for cell_path in cell_files:
        cell_data = json.loads(cell_path.read_text())
        results   = cell_data["results"]
        cell_name = cell_path.stem
        print(f"\n[reeval] === {cell_name} ({len(results)} seeds) ===")

        for r in results:
            seed      = int(r["seed"])
            intervene = r["intervene"]
            alpha     = float(r["alpha"])
            sel_tuple = [tuple(t) for t in r["winner_tuple"]]

            if seed not in sae_cache:
                sae_cache[seed], _ = sae_load(
                    Path(f"weights/seeds/sae_ln1_s{seed}.pt"), device=device,
                )
            sae_ln1 = sae_cache[seed]
            active  = ACTIVE_CHANNELS[intervene]

            old_jsd = r["eval"].get("jsd_clean", float("nan"))
            old_asr = r["eval"].get("asr", float("nan"))

            new_eval = eval_winner(
                model, sae_ln1, sel_tuple, alpha, active, W,
                eval_dep_lp, eval_dep_attn,
                eval_clean_lsm, GEN_TOKENS, device,
                eval_seeds=EVAL_SEEDS,
                eval_temperature=EVAL_TEMP,
            )
            r["eval"] = new_eval
            print(f"  seed={seed} α={alpha} {intervene:>6}  "
                  f"asr {old_asr:.3f}→{new_eval['asr']:.3f}  "
                  f"jsd_clean {old_jsd:.4f}→{new_eval['jsd_clean']:.4f}")

        # Update decoding metadata to reflect fixed methodology
        cell_data["decoding"]["eval_jsd_clean"] = {
            "mode": "sample",
            "temperature": EVAL_TEMP,
            "top_p": None,
            "top_k": None,
            "seed": JSD_CLEAN_SEED,
            "note": "single rollout at JSD_CLEAN_SEED — same seed as clean_lsm",
        }

        cell_path.write_text(json.dumps(cell_data, indent=2, default=str))
        print(f"  → wrote {cell_path}")

    print("\n[reeval] summary:")
    print(f"  {'cell':>16}  {'mean ASR':>8}  {'mean JSD':>8}")
    print(f"  {'-'*16}  {'-'*8}  {'-'*8}")
    for cell_path in sorted(CELLS_DIR.glob("*.json")):
        d = json.loads(cell_path.read_text())
        jsds = [r["eval"]["jsd_clean"] for r in d["results"]]
        asrs = [r["eval"]["asr"]       for r in d["results"]]
        print(f"  {cell_path.stem:>16}  {sum(asrs)/len(asrs):>8.4f}  "
              f"{sum(jsds)/len(jsds):>8.4f}")

    print("\n[reeval] done")


if __name__ == "__main__":
    main()
