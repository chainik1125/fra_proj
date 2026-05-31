"""Unified pod-side cell runner for the YAML-driven FRA campaign.

Reads ONE cell-spec JSON (path as argv[1]) and runs the whole cell end-to-end ON
THE POD — replacing the env-var soup of run_grid_diff.sh / run_frarouting_diff.sh
and, crucially, JUDGING POD-SIDE (no local-disk judging → no SIGKILL/collision
bugs). On success the combined results are on HF and the pod self-terminates
(handled by run_cell.sh).

Pipeline per cell:
  1. download SAE (sae_hf_prefix) + α=0 buckets (noise_prefix + scores).
  2. resolve ‖Δa‖ (delta_a: "auto" → compute_delta_a_norm.py; else the float).
  3. rankings:
       diff_mode=bucket → per-em ranking (decompose on that model's weights+buckets)
       diff_mode=model  → ONE ranking (em_key − base) reused for both em_keys
     attribution ∈ {ov,qk,wang}; routing recipes harvest feature_ids → override.
  4. generate: phase1_grid_14b_orchestrator (additive) or
     phase1_frarouting_magmatched_14b_orchestrator (routing), per (em, seed).
  5. JUDGE POD-SIDE: phase1_judge_and_combine.py --stream-root per gran-cell
     (gpt-4o-mini @ T0, OPENAI_API_KEY from pod env) → gpt4o_combined_*.json.
  6. upload qualitative + combined to <hf_prefix>/<cell_prefix>_gran<g>/...

Cell-spec JSON keys: see scripts/run_campaign.py:cell_spec().
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

SPEC = json.loads(Path(sys.argv[1]).read_text())
REPO = SPEC["hf_repo"]
HF_PREFIX = SPEC["hf_prefix"].rstrip("/")
NOISE_PREFIX = SPEC["noise_prefix"].rstrip("/")
API = HfApi(token=os.environ.get("HF_TOKEN"))
WS = Path("/workspace")


def sh(cmd: list[str]):
    print(f"[cell] $ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def upload(local: Path, repo_path: str, msg: str):
    API.upload_file(path_or_fileobj=str(local), path_in_repo=repo_path,
                    repo_id=REPO, repo_type="dataset", commit_message=msg)
    print(f"  → {repo_path}", flush=True)


def main():
    layer = SPEC["layer"]; head = SPEC["head"]
    sae = SPEC["sae"]; kind = SPEC["kind"]
    diff_mode = SPEC["diff_mode"]
    base_id = SPEC["base_model_id"]; em_id = SPEC["em_model_id"]
    em_keys = SPEC["em_keys"]; seeds = SPEC["seeds"]; grans = SPEC["grans"]
    alphas = [str(a) for a in SPEC["alphas"]]
    coh_floor = SPEC.get("buckets", {}).get("coh_floor", 70)
    cell_prefix = SPEC["cell_prefix"]            # final, incl _modeldiff (driver-provided)
    # max_new_tokens: MUST match the financial campaign (100). The orchestrator
    # default is 200 → 2× generation time + ~2× cost (a cell would blow past the
    # 12h runaway kill and the $300 cap). Default 100 even if the spec omits it.
    max_new = int(SPEC.get("max_new_tokens", 100))
    # attribution selector for the ranking script
    if kind == "routing":
        which = "ov" if SPEC["recipe"] == "ov_to_ov" else "qk"
        attribution = which
        orch_ranking = None
    else:
        attribution = SPEC["attribution"]                       # ov|qk|wang
        orch_ranking = "wang" if attribution == "wang" else f"fra-{attribution}"

    # 1. SAE ---------------------------------------------------------------
    snapshot_download(REPO, repo_type="dataset",
                      allow_patterns=f"{SPEC['sae_hf_prefix']}/*", local_dir=str(WS / "sae_dl"))
    sae_dir = str(next((WS / "sae_dl").rglob("ae.pt")).parent)
    print(f"[cell] SAE_DIR={sae_dir}", flush=True)

    # 2. buckets (α=0 rollouts + judge scores) -----------------------------
    snapshot_download(REPO, repo_type="dataset",
                      allow_patterns=f"{NOISE_PREFIX}/*", local_dir=str(WS / "buckets"))
    scores_file = WS / "buckets" / NOISE_PREFIX / "scores" / "judge_scores_gpt-4o-mini.json"
    if not scores_file.exists():
        sys.exit(f"[cell] missing judge scores at {scores_file}")
    noise_seeds = SPEC.get("noise_seeds", [42, 123, 456])

    def rollout_files(em_key):
        fs = [WS / "buckets" / NOISE_PREFIX / f"{em_key}_seed{ns}.json" for ns in noise_seeds]
        return [str(f) for f in fs if f.exists()]

    # 3. ‖Δa‖ --------------------------------------------------------------
    delta_a = SPEC["delta_a"]
    if delta_a == "auto":
        da_out = WS / "delta_a.json"
        sh([sys.executable, "-u", "scripts/compute_delta_a_norm.py",
            "--layer", layer, "--base-model-id", base_id, "--em-model-id", em_id,
            "--em-key", SPEC["em_key"], "--out", str(da_out)])
        da = json.loads(da_out.read_text())
        delta_a = da["ln1_postgain"]["diff_norm_l2"] if sae == "ln1" else da["resid_post"]["diff_norm_l2"]
        upload(da_out, f"{HF_PREFIX}/{cell_prefix}_meta/delta_a_norm_L{layer}.json", "delta_a")
    delta_a = float(delta_a)
    print(f"[cell] ‖Δa‖[{sae}]={delta_a}", flush=True)

    # 4. ranking(s) --------------------------------------------------------
    def compute_ranking(out_json, model_key=None):
        cmd = [sys.executable, "-u", "scripts/compute_fra_diff_ranking.py",
               "--attribution", attribution, "--diff-mode", diff_mode,
               "--sae", sae, "--sae-dir", sae_dir, "--layer", layer, "--head", head,
               "--coh-floor", coh_floor, "--top-n", 50, "--k-pairs", 50,
               "--base-model-id", base_id, "--em-model-id", em_id,
               "--scores-file", str(scores_file), "--out", str(out_json)]
        if diff_mode == "bucket":
            cmd += ["--model", model_key, "--rollout-files", *rollout_files(model_key)]
        else:  # model-diff: needs both models' rollouts; em_key sets the EM side
            both = rollout_files(SPEC["em_key"]) + rollout_files("base")
            cmd += ["--em-key", SPEC["em_key"], "--rollout-files", *both]
        sh(cmd)

    rankings = {}   # em_key -> ranking json path
    override = SPEC.get("feature_ids_override")
    if override:
        # finegrid / single-feature: skip ranking, steer exactly these ids.
        rj = WS / "ranking_override.json"
        rj.write_text(json.dumps({"feature_ids": list(override)}))
        for em in em_keys:
            rankings[em] = rj
        print(f"[cell] feature_ids_override={override} (ranking skipped)", flush=True)
    elif diff_mode == "model":
        rj = WS / f"ranking_modeldiff_{attribution}.json"
        compute_ranking(rj)
        for em in em_keys:
            rankings[em] = rj
        upload(rj, f"{HF_PREFIX}/{cell_prefix}_meta/ranking_{attribution}.json", "ranking model-diff")
    else:
        for em in em_keys:
            rj = WS / f"ranking_{em}_{attribution}.json"
            compute_ranking(rj, model_key=em)
            rankings[em] = rj
            upload(rj, f"{HF_PREFIX}/{cell_prefix}_meta/ranking_{em}.json", f"ranking {em}")

    # 5. generate per (em, seed) ------------------------------------------
    run_root = WS / "runs"
    for em in em_keys:
        em_id_arg = [] if em == "base" else ["--em-model-id", em_id]
        feat_ids = json.loads(Path(rankings[em]).read_text())["feature_ids"][:50]
        for seed in seeds:
            out = run_root / f"{em}_seed{seed}"
            out.mkdir(parents=True, exist_ok=True)
            if kind == "routing":
                sh([sys.executable, "-u", "phase1_frarouting_magmatched_14b_orchestrator.py",
                    "--recipe", SPEC["recipe"], "--sae-dir", sae_dir,
                    "--em-model", em, "--base-model-id", base_id, *em_id_arg,
                    "--eval-seed", seed, "--layer", layer, "--head", head,
                    "--granularities", *[str(g) for g in grans],
                    "--feature-ids-override", *[str(f) for f in feat_ids],
                    "--delta-a-norm", delta_a, "--alphas", *alphas,
                    "--max-new-tokens", str(max_new),
                    "--output-root", str(out)])
            else:
                sh([sys.executable, "-u", "phase1_grid_14b_orchestrator.py",
                    "--ranking", orch_ranking, "--sae", sae, "--sae-dir", sae_dir,
                    "--em-model", em, "--base-model-id", base_id, *em_id_arg,
                    "--eval-seed", seed, "--layer", layer, "--head", head,
                    "--granularities", *[str(g) for g in grans],
                    "--ranking-json", str(rankings[em]),
                    "--delta-a-norm", delta_a, "--alphas", *alphas,
                    "--max-new-tokens", str(max_new),
                    "--output-root", str(out)])

    # 6. JUDGE POD-SIDE per gran-cell, then upload -------------------------
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("[cell] OPENAI_API_KEY not set on pod (inject via pod env)")
    for g in grans:
        cell = f"{cell_prefix}_gran{g}"
        stream_root = WS / "judge" / cell
        # arrange streams: <stream_root>/<em>_seed<seed>/qualitative_grid_<em>_evalseed<seed>.json
        for em in em_keys:
            for seed in seeds:
                src = run_root / f"{em}_seed{seed}" / f"gran{g}" / f"qualitative_grid_{em}_evalseed{seed}.json"
                if not src.exists():
                    print(f"[cell] WARN missing {src}", flush=True); continue
                dst = stream_root / f"{em}_seed{seed}"
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dst / src.name)
        # judge + combine (gpt-4o-mini @ T0 — set in phase1_judge_and_combine defaults)
        sh([sys.executable, "-u", "phase1_judge_and_combine.py",
            "--stream-root", str(stream_root)])
        # upload qualitatives + combined under the gran-cell
        for em in em_keys:
            for seed in seeds:
                q = stream_root / f"{em}_seed{seed}" / f"qualitative_grid_{em}_evalseed{seed}.json"
                if q.exists():
                    upload(q, f"{HF_PREFIX}/{cell}/{em}_seed{seed}/{q.name}", f"qual {cell} {em} s{seed}")
        for comb in stream_root.glob("gpt4o_combined_*.json"):
            upload(comb, f"{HF_PREFIX}/{cell}/{comb.name}", f"combined {cell}")
    print("[cell] DONE", flush=True)


if __name__ == "__main__":
    main()
