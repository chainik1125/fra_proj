"""Extract compact Cadenza steering figure data from audited local and remote runs.

The three validation trajectories are stored on simplex2/3. This script saves
only their aggregate metrics; no prompts, generations, or model data are copied.
The confirmation selections are read from the local audited campaign archives.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "experiments/cadenza_mid_sae"
OUTPUT = ROOT / "paper/iclr-paper/fra_proj_tex/figure_data/cadenza_steering.json"
CURVES = {
    "fra": ("simplex2", "A-input-L08-ov50-s2-20260921", "validation_f30892_a*.json"),
    "sae_same": ("simplex2", "A-input-L08-single50-s2-20260921", "validation_f2128_a*.json"),
    "sae_best": ("simplex3", "A-resid-mid-L08-single50-s3-20260921", "validation_f12801_a*.json"),
    "dom": ("simplex2", "A-input-L08-caa-dom-s2-20260921", "validation_resid_response_*.json"),
}


def remote_curve(host: str, run: str, pattern: str) -> list[dict]:
    path = f"/data/users/dmitry/sae-middle/runs/{run}"
    code = (
        "import glob,json,os\n"
        f"files=glob.glob({str(path + '/' + pattern)!r})\n"
        "out=[]\n"
        "for p in files:\n"
        " d=json.load(open(p))\n"
        " m=d['metrics']\n"
        " out.append({'alpha':float(d.get('alpha',d.get('coefficient'))),"
        "'jsd_clean':m['triggered_to_clean_js_bits'],"
        "'jsd_sleeper':m['triggered_to_poison_js_bits'],"
        "'n':m['pairs']})\n"
        "print(json.dumps(sorted(out,key=lambda x:x['alpha'])))\n"
    )
    command = "python3 -c " + shlex.quote(code)
    result = subprocess.run(["ssh", host, command], check=True, text=True, capture_output=True)
    rows = json.loads(result.stdout)
    expected = 15 if pattern.startswith("validation_f") else 17
    if len(rows) != expected:
        raise ValueError(f"Unexpected grid size from {host}:{run}: {len(rows)}")
    if len({row["alpha"] for row in rows}) != len(rows):
        raise ValueError(f"Repeated coefficient in {host}:{run}")
    return rows


def sae_winners() -> list[dict]:
    archive = json.loads((CAMPAIGN / "STEERING_RESULTS_20260921.json").read_text())
    choices = []
    for run in archive["runs"]:
        summary = run["summary"]
        layer = summary["layer"]
        if layer not in (8, 16, 24):
            continue
        for rule, result in summary["confirmation"]["results"].items():
            method = result["candidate"]["method"]
            category = "fra" if method in ("ov", "qkov") else "sae"
            choices.append({
                "category": category,
                "method": method,
                "hook": summary["hook"],
                "layer": layer,
                "rule": rule,
                "features": result["candidate"]["features"],
                "alpha": result["alpha"],
                "jsd_clean": result["metrics"]["triggered_to_clean_js_bits"],
                "jsd_sleeper": result["metrics"]["triggered_to_poison_js_bits"],
                "clean_exact": result["metrics"]["clean_exact_match"],
                "ihy_removed": result["metrics"]["escaped_phrase_count"],
                "run_id": run["run_id"],
            })
    return choices


def dom_choices() -> tuple[list[dict], dict]:
    archive = json.loads((CAMPAIGN / "DOM_ALL_LAYER_SWEEP_RESULTS_20260921.json").read_text())
    choices = []
    for row in archive["records"]:
        if row["mode"] != "sweep" or row["rule"] != "restoration":
            continue
        layer = row["layers"][0]
        choices.append({
            "category": "dom", "method": row["variant"], "hook": row["variant"],
            "layer": layer, "rule": row["rule"], "features": [],
            "alpha": row["coefficient"],
            "jsd_clean": row["metrics"]["triggered_to_clean_js_bits"],
            "jsd_sleeper": row["metrics"]["triggered_to_poison_js_bits"],
            "clean_exact": row["metrics"]["clean_exact_match"],
            "ihy_removed": row["metrics"]["escaped_phrase_count"],
            "run_id": row["run_id"],
        })
    return choices, min(choices, key=lambda row: row["jsd_clean"])


def select_winners(choices: list[dict], *, positive_only: bool) -> list[dict]:
    winners = []
    for layer in (8, 16, 24):
        for category in ("fra", "sae_same", "sae_best", "dom"):
            if category == "sae_same":
                candidates = [row for row in choices if row["layer"] == layer
                              and row["category"] == "sae" and ".ln1." in row["hook"]]
            elif category == "sae_best":
                candidates = [row for row in choices if row["layer"] == layer
                              and row["category"] == "sae"]
            else:
                candidates = [row for row in choices if row["layer"] == layer
                              and row["category"] == category]
            if positive_only:
                candidates = [row for row in candidates if row["alpha"] > 0]
            if not candidates:
                raise ValueError(f"Missing {category} at L{layer}, positive_only={positive_only}")
            winner = dict(min(candidates, key=lambda row: row["jsd_clean"]))
            winner["category"] = category
            winners.append(winner)
    return winners


def main() -> None:
    curves = {}
    for method, (host, run, pattern) in CURVES.items():
        curves[method] = {
            "source": f"{host}:/data/users/dmitry/sae-middle/runs/{run}/{pattern}",
            "rows": remote_curve(host, run, pattern),
        }
    dom, dom_global = dom_choices()
    choices = sae_winners() + [row for row in dom if row["layer"] in (8, 16, 24)]
    winners = select_winners(choices, positive_only=False)
    positive_winners = select_winners(choices, positive_only=True)
    first_run = json.loads((CAMPAIGN / "STEERING_RESULTS_20260921.json").read_text())["runs"][0]
    baseline = first_run["summary"]["confirmation"]["baseline"]
    output = {
        "model": "dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-A",
        "metric": "rollout full-vocabulary Jensen-Shannon divergence in bits",
        "validation_n": 24,
        "confirmation_n": 64,
        "validation_curves": curves,
        "confirmation_winners": winners,
        "confirmation_positive_winners": positive_winners,
        "dom_best_across_32_layers": dom_global,
        "confirmation_baseline_jsd_clean": baseline["triggered_to_clean_js_bits"],
        "selection_note": "Candidates and coefficients were selected on validation. Best subtype/rule per category and layer is shown descriptively by confirmation JSD; the confirmation block had been inspected earlier.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(OUTPUT)
    for row in winners:
        print(row["layer"], row["category"], row["method"], row["alpha"], f"{row['jsd_clean']:.6f}")


if __name__ == "__main__":
    main()
