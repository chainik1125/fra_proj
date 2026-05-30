"""YAML-driven FRA steering campaign driver.

One command runs an entire campaign:  python scripts/run_campaign.py <campaign.yaml>
(typically headless on a CPU pod — see scripts/launch_campaign_cpu.sh).

Replaces the manual pod-by-pod orchestration. Stages (gated by the YAML `stages`
list), each idempotent — "done" is re-derived from HF every loop, so a restart
resumes and never double-launches:

  baseline  → 1 GPU pod: α=0 rollouts (base+em) + judge → <noise_prefix>   [HARD GATE]
  smoke     → 1 canary cell end-to-end; block + validate; ABORT on failure
  maingrid  → one GPU pod per cell (≤ max_parallel); each rank→gen→JUDGE→upload→self-term
  finegrid  → winner ±5 pods (per protocol)
  results   → build_grid_results.py → GRID_RESULTS_<name>.md
  dashboard → extract_trajectories.py + build_steering_dashboard.py

Pod-side work is experiments/fra_14b_diff/run_cell.sh + cell_runner.py (judges
on the pod — no local judging). Secrets are injected via the pod `env` field.

--dry-run prints the expanded cell list + which are already done, and exits.
"""
from __future__ import annotations
import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import yaml
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import runpod_launch as rp

REPO_ROOT = Path(__file__).resolve().parents[1]


# ───────────────────────── spec + cell expansion ─────────────────────────
def load_spec(path: str) -> dict:
    spec = yaml.safe_load(Path(path).read_text())
    for k in ("campaign", "models", "target", "sae_families", "grid", "compute", "judge", "budget"):
        if k not in spec:
            raise SystemExit(f"campaign yaml missing top-level key: {k}")
    return spec


def _short_attr(attr: str) -> str:
    """YAML attribution → compute_fra_diff_ranking --attribution token."""
    return {"fra-ov": "ov", "fra-qk": "qk", "wang": "wang"}[attr]


def expand_cells(spec: dict) -> list[dict]:
    """One dict per pod. Matches the financial scope: bucket-diff = full grid;
    model-diff = the gran1 2×2 arm (additive attributions only); routing = bucket
    only over routing_grans."""
    g = spec["grid"]
    cells: list[dict] = []
    for proto in g["protocols"]:
        if proto["kind"] == "additive":
            attr = proto["attribution"]                    # wang|fra-ov|fra-qk
            for sae in proto.get("saes", ["ln1"]):
                base_prefix = f"{attr}_{sae}"
                # bucket-diff: full grid
                cells.append(dict(kind="additive", attribution=_short_attr(attr),
                                  orch_ranking=attr, sae=sae, diff_mode="bucket",
                                  cell_prefix=f"{base_prefix}", grans=list(g["grans"])))
                # model-diff: gran1 only (the 2×2 comparison arm)
                cells.append(dict(kind="additive", attribution=_short_attr(attr),
                                  orch_ranking=attr, sae=sae, diff_mode="model",
                                  cell_prefix=f"{base_prefix}_modeldiff", grans=[1]))
        else:  # routing — bucket only, ln1
            recipe = proto["recipe"]
            cells.append(dict(kind="routing", recipe=recipe, sae="ln1", diff_mode="bucket",
                              cell_prefix=f"frarouting_{recipe}_ln1",
                              grans=list(g["routing_grans"])))
    return cells


def sae_hf_prefix(spec: dict, sae_name: str) -> str:
    for s in spec["sae_families"]:
        if s["name"] == sae_name:
            return s["hf_prefix"]
    raise SystemExit(f"sae family {sae_name} not in yaml")


def sae_delta_a(spec: dict, sae_name: str):
    for s in spec["sae_families"]:
        if s["name"] == sae_name:
            return s.get("delta_a", "auto")
    return "auto"


def cell_spec_json(spec: dict, cell: dict, *, alphas=None, grans=None,
                   feature_ids_override=None, cell_prefix=None) -> dict:
    """The JSON handed to cell_runner.py (one cell = one pod)."""
    g = spec["grid"]
    return {
        "kind": cell["kind"],
        "attribution": cell.get("attribution"),
        "recipe": cell.get("recipe"),
        "diff_mode": cell["diff_mode"],
        "sae": cell["sae"],
        "sae_hf_prefix": sae_hf_prefix(spec, cell["sae"]),
        "em_keys": list(g["em_keys"]),
        "em_key": spec["models"]["em_key"],
        "seeds": list(g["seeds"]),
        "grans": grans if grans is not None else cell["grans"],
        "alphas": alphas if alphas is not None else list(g["alphas"]),
        "layer": spec["target"]["layer"],
        "head": spec["target"]["head"],
        "delta_a": sae_delta_a(spec, cell["sae"]),
        "base_model_id": spec["models"]["base_model_id"],
        "em_model_id": spec["models"]["em_model_id"],
        "hf_repo": spec["campaign"]["hf_repo"],
        "hf_prefix": spec["campaign"]["hf_prefix"],
        "noise_prefix": spec["campaign"]["noise_prefix"],
        "noise_seeds": list(g.get("noise_seeds", [42, 123, 456])),
        "buckets": g["buckets"],
        "cell_prefix": cell_prefix or cell["cell_prefix"],
        "feature_ids_override": feature_ids_override,
    }


# ───────────────────────── resilient polling ─────────────────────────
def _retry(fn, *, tries: int = 6, base: float = 8.0, what: str = "call"):
    """Retry a network call with exponential backoff. The driver polls HF +
    RunPod for hours; a single transient blip (RemoteDisconnected, 5xx) must NOT
    kill the run. Raises only after `tries` consecutive failures."""
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — deliberately broad (network layer)
            last = e
            wait = min(base * (2 ** i), 120)
            print(f"[retry] {what} failed ({type(e).__name__}: {str(e)[:90]}); "
                  f"retry {i+1}/{tries} in {wait:.0f}s", flush=True)
            time.sleep(wait)
    raise last


# ───────────────────────── HF idempotency ─────────────────────────
def hf_files(api: HfApi, repo: str) -> list[str]:
    return _retry(lambda: api.list_repo_files(repo, repo_type="dataset"),
                  what="list_repo_files")


def running_names_safe(spec: dict) -> set[str]:
    pods = _retry(lambda: rp.list_pods(), what="list_pods")
    return {p["name"] for p in pods if p["status"] == "RUNNING"}


def cell_done(files: list[str], spec: dict, cell: dict, cell_prefix=None, grans=None) -> bool:
    """Done iff every (gran, em) has a combined json on HF."""
    pfx = spec["campaign"]["hf_prefix"].rstrip("/")
    cp = cell_prefix or cell["cell_prefix"]
    grs = grans if grans is not None else cell["grans"]
    ems = spec["grid"]["em_keys"]
    for gran in grs:
        for em in ems:
            need = f"{pfx}/{cp}_gran{gran}/"
            ok = any(f.startswith(need) and "gpt4o_combined" in f and f.endswith(f"_{em}.json")
                     for f in files)
            if not ok:
                return False
    return True


def baseline_ready(files: list[str], spec: dict) -> bool:
    np_ = spec["campaign"]["noise_prefix"].rstrip("/")
    em = spec["models"]["em_key"]
    has_scores = any(f == f"{np_}/scores/judge_scores_gpt-4o-mini.json" for f in files)
    has_em = any(f.startswith(f"{np_}/{em}_seed") for f in files)
    has_base = any(f.startswith(f"{np_}/base_seed") for f in files)
    return has_scores and has_em and has_base


# ───────────────────────── pod launch ─────────────────────────
def _bootstrap(spec: dict, script: str = "experiments/fra_14b_diff/run_cell.sh") -> str:
    c = spec["campaign"]
    return (
        "#!/bin/bash\nset -eo pipefail\ncd /workspace\n"
        f"[ -d fra_proj ] || git clone --branch {c['branch']} --single-branch {c['repo_url']} fra_proj\n"
        "cd fra_proj\n"
        f"bash {script}\n"
    )


def _secret_env(spec: dict) -> dict:
    j = spec["judge"]
    return {
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "RUNPOD_API_KEY": os.environ.get("RP_API_KEY_MATS", os.environ.get("RUNPOD_API_KEY", "")),
        "OPENAI_API_KEY": os.environ.get(j.get("openai_key_env", "OPENAI_API_KEY_MATS"), ""),
        "BRANCH": spec["campaign"]["branch"],
    }


def _pod_env(spec: dict, cell_json: dict) -> dict:
    j = spec["judge"]
    return {
        **_secret_env(spec),
        "FRA_DIFF_MAX_USD": str(j.get("max_usd", 35)),
        "FRA_DIFF_MAX_CALLS": str(j.get("max_calls", 900000)),
        "CELL_SPEC_B64": base64.b64encode(json.dumps(cell_json).encode()).decode(),
    }


def _baseline_env(spec: dict) -> dict:
    g = spec["grid"]
    em = spec["models"]["em_key"]
    return {
        **_secret_env(spec),
        "NOISE_MODELS": f"base {em}",
        "EM_MODEL_ID": spec["models"]["em_model_id"],
        "BASE_MODEL_ID": spec["models"]["base_model_id"],
        "NOISE_PREFIX": spec["campaign"]["noise_prefix"],
        "NOISE_SEEDS": " ".join(str(s) for s in g.get("noise_seeds", [42, 123, 456])),
        "EM_KEY": em,
        "JUDGE_MODEL": spec["judge"].get("model", "gpt-4o-mini"),
        "HF_REPO": spec["campaign"]["hf_repo"],
    }


def launch_cell(spec: dict, cell_json: dict, pod_name: str) -> str | None:
    comp = spec["compute"]
    return rp.launch_pod(
        pod_name, _bootstrap(spec),
        gpu_type_ids=comp["gpu_type_ids"], image=comp["image"],
        env=_pod_env(spec, cell_json), disk_gb=80, skip_if_running=True,
    )


def n_running_campaign_pods(spec: dict) -> int:
    pfx = spec["compute"]["pod_prefix"]
    return sum(1 for n in running_names_safe(spec) if n.startswith(pfx))


# ───────────────────────── stages ─────────────────────────
def run_maingrid(spec: dict, api: HfApi, dry: bool):
    cells = expand_cells(spec)
    files = hf_files(api, spec["campaign"]["hf_repo"])
    pending = [c for c in cells if not cell_done(files, spec, c)]
    print(f"[maingrid] {len(cells)} cells total, {len(pending)} pending, "
          f"{len(cells)-len(pending)} already done")
    for c in cells:
        tag = "DONE" if c not in pending else "pend"
        print(f"   [{tag}] {c['cell_prefix']}  grans={c['grans']}  diff={c['diff_mode']}")
    if dry:
        return
    pfx = spec["compute"]["pod_prefix"]
    max_par = spec["compute"]["max_parallel"]
    poll = spec["compute"].get("poll_interval_s", 120)
    while pending:
        while n_running_campaign_pods(spec) < max_par and pending:
            c = pending.pop(0)
            cj = cell_spec_json(spec, c)
            name = f"{pfx}-{c['cell_prefix'].replace('_', '-')}"[:60]
            print(f"[maingrid] launch {name}")
            launch_cell(spec, cj, name)
            time.sleep(5)
        time.sleep(poll)
        files = hf_files(api, spec["campaign"]["hf_repo"])
        pending = [c for c in expand_cells(spec) if not cell_done(files, spec, c)]
        print(f"[maingrid] {len(pending)} cells still pending "
              f"({n_running_campaign_pods(spec)} pods running)")
    print("[maingrid] all cells done")


def run_smoke(spec: dict, api: HfApi, dry: bool) -> bool:
    """One cheap canary: additive × fra-ov × ln1 × bucket, em only, seed42, gran1,
    alphas [-1,0,1]. Block until its combined lands; abort campaign on failure."""
    c = dict(kind="additive", attribution="ov", orch_ranking="fra-ov", sae="ln1",
             diff_mode="bucket", cell_prefix="smoke_fra-ov_ln1", grans=[1])
    files = hf_files(api, spec["campaign"]["hf_repo"])
    if cell_done(files, spec, c, grans=[1]):
        print("[smoke] canary already present — skip")
        return True
    cj = cell_spec_json(spec, c, alphas=[-1, 0, 1], grans=[1])
    cj["em_keys"] = [spec["models"]["em_key"]]   # em only, cheaper
    cj["seeds"] = [42]
    print("[smoke] launching canary cell")
    if dry:
        print("[smoke] (dry-run) would launch canary"); return True
    launch_cell(spec, cj, f"{spec['compute']['pod_prefix']}-smoke")
    # block until combined lands
    pfx = spec["campaign"]["hf_prefix"].rstrip("/")
    em = spec["models"]["em_key"]
    deadline = time.time() + 90 * 60
    while time.time() < deadline:
        time.sleep(spec["compute"].get("poll_interval_s", 120))
        files = hf_files(api, spec["campaign"]["hf_repo"])
        if any(f.startswith(f"{pfx}/smoke_fra-ov_ln1_gran1/") and "gpt4o_combined" in f
               and f.endswith(f"_{em}.json") for f in files):
            print("[smoke] canary combined landed ✓")
            return True
        if not n_running_campaign_pods(spec):
            print("[smoke] canary pod gone with no combined — FAIL"); return False
    print("[smoke] canary timed out — FAIL")
    return False


def run_baseline(spec: dict, api: HfApi, dry: bool) -> bool:
    """HARD GATE: ensure medical+base α=0 rollouts + gpt-4o-mini@T0 bucket scores
    are on HF. Launches ONE GPU pod (run_baseline.sh = alpha0_noise_gen +
    judge_temp_sweep) and BLOCKS until the scores land, so the driver proceeds
    to ranking in the same pass. Idempotent: skips if already present."""
    if baseline_ready(hf_files(api, spec["campaign"]["hf_repo"]), spec):
        print("[baseline] α=0 rollouts + T0 scores already on HF ✓")
        return True
    pfx = spec["compute"]["pod_prefix"]
    name = f"{pfx}-baseline"
    em = spec["models"]["em_key"]
    print(f"[baseline] {em}+base α=0 baselines MISSING → launch baseline pod {name} "
          f"(alpha0_noise_gen + judge T0 → {spec['campaign']['noise_prefix']})")
    if dry:
        print("[baseline] (dry-run) would launch the baseline GPU pod, then block "
              "until scores land. Treating as satisfied for the rest of the dry-run.")
        return True
    comp = spec["compute"]
    if name not in running_names_safe(spec):
        rp.launch_pod(name, _bootstrap(spec, "experiments/fra_14b_diff/run_baseline.sh"),
                      gpu_type_ids=comp["gpu_type_ids"], image=comp["image"],
                      env=_baseline_env(spec), disk_gb=80, skip_if_running=True)
    # block until the scores file lands (gen+judge ≈ 30–50 min)
    deadline = time.time() + 90 * 60
    while time.time() < deadline:
        time.sleep(spec["compute"].get("poll_interval_s", 120))
        if baseline_ready(hf_files(api, spec["campaign"]["hf_repo"]), spec):
            print("[baseline] α=0 rollouts + T0 scores landed ✓")
            return True
        if name not in running_names_safe(spec):
            print("[baseline] baseline pod gone with no scores — FAIL (check run.log)")
            return False
        print("[baseline] …waiting for baseline scores")
    print("[baseline] timed out waiting for baseline scores — FAIL")
    return False


def _finegrid_alphas(spec: dict) -> list[float]:
    fg = spec["grid"]["finegrid"]
    lo, hi, step = fg["lo"], fg["hi"], fg["step"]
    n = int(round((hi - lo) / step)) + 1
    return [round(lo + i * step, 4) for i in range(n)]


def _winner_feature(api: HfApi, spec: dict, cell_prefix: str) -> int | None:
    """Top feature by Δalign@coh70 (fallback @50) in the gran1 bucket combined
    for the EM model — via grid_metrics.cell_row."""
    import tempfile
    import grid_metrics as gm
    from huggingface_hub import hf_hub_download
    pfx = spec["campaign"]["hf_prefix"].rstrip("/")
    em = spec["models"]["em_key"]
    want_dir = f"{pfx}/{cell_prefix}_gran1/"
    comb = [f for f in hf_files(api, spec["campaign"]["hf_repo"])
            if f.startswith(want_dir) and "gpt4o_combined" in f and f.endswith(f"_{em}.json")]
    if not comb:
        print(f"[finegrid] no gran1 combined for {cell_prefix} ({em}) — skip"); return None
    local = hf_hub_download(spec["campaign"]["hf_repo"], comb[0], repo_type="dataset",
                            token=os.environ.get("HF_TOKEN"), local_dir=tempfile.mkdtemp())
    row = gm.cell_row(local)
    top = (row.get(70) or {}).get("top_method") or (row.get(50) or {}).get("top_method")
    if not top or not str(top).startswith("feat_F"):
        print(f"[finegrid] {cell_prefix}: no per-feature winner (kind={row.get('kind')}) — skip")
        return None
    fid = int(str(top)[len("feat_F"):])
    print(f"[finegrid] {cell_prefix} winner = F{fid}")
    return fid


def run_finegrid(spec: dict, api: HfApi, dry: bool):
    """Per protocol: pick the winning feature from its gran1 bucket combined and
    launch a single-feature ±5 finegrid cell (both em_keys). Mirrors the
    financial finegrids; idempotent on the *_finegrid_gran1 combined."""
    alphas = _finegrid_alphas(spec)
    files = hf_files(api, spec["campaign"]["hf_repo"])
    pfx = spec["compute"]["pod_prefix"]
    launched = 0
    for c in expand_cells(spec):
        if c["diff_mode"] != "bucket":
            continue   # finegrids refine the bucket-diff winners
        base_prefix = c["cell_prefix"]
        fg_prefix = f"{base_prefix}_finegrid"
        if cell_done(files, spec, c, cell_prefix=fg_prefix, grans=[1]):
            print(f"[finegrid] {fg_prefix} already done"); continue
        fid = _winner_feature(api, spec, base_prefix)
        if fid is None:
            continue
        cj = cell_spec_json(spec, c, alphas=alphas, grans=[1],
                            feature_ids_override=[fid], cell_prefix=fg_prefix)
        name = f"{pfx}-{fg_prefix.replace('_', '-')}"[:60]
        print(f"[finegrid] launch {name}  (F{fid}, α∈[{alphas[0]}..{alphas[-1]}])")
        if not dry:
            launch_cell(spec, cj, name)
            launched += 1
            time.sleep(5)
    print(f"[finegrid] launched {launched} finegrid cells "
          f"(judge pod-side; results land under *_finegrid_gran1/)")


def run_results(spec: dict, dry: bool):
    import subprocess
    out = REPO_ROOT / f"experiments/fra_14b_diff/GRID_RESULTS_{spec['campaign']['name']}.md"
    print(f"[results] build_grid_results.py → {out} "
          f"(set PREFIX={spec['campaign']['hf_prefix']})")
    if dry:
        return
    env = {**os.environ, "GRID_PREFIX": spec["campaign"]["hf_prefix"]}
    with open(out, "w") as fh:
        subprocess.run([sys.executable, str(REPO_ROOT / "experiments/fra_14b_diff/build_grid_results.py")],
                       stdout=fh, env=env, check=False)


def run_dashboard(spec: dict, dry: bool):
    print("[dashboard] extract_trajectories.py + build_steering_dashboard.py "
          "(after finegrids judged)")


STAGE_FNS = {
    "baseline":  lambda spec, api, dry: run_baseline(spec, api, dry),
    "smoke":     lambda spec, api, dry: run_smoke(spec, api, dry),
    "maingrid":  lambda spec, api, dry: run_maingrid(spec, api, dry),
    "finegrid":  lambda spec, api, dry: run_finegrid(spec, api, dry),
    "results":   lambda spec, api, dry: run_results(spec, dry),
    "dashboard": lambda spec, api, dry: run_dashboard(spec, dry),
}
STAGE_ORDER = ["baseline", "head_ablation", "smoke", "maingrid", "finegrid", "results", "dashboard"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="expand cells + check HF done-state + print plan; no launches.")
    ap.add_argument("--only", nargs="+", help="run only these stages (subset of the yaml's).")
    args = ap.parse_args()

    spec = load_spec(args.yaml)
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    stages = args.only or spec.get("stages", STAGE_ORDER)
    stages = [s for s in STAGE_ORDER if s in stages]

    print(f"=== campaign {spec['campaign']['name']} ===")
    print(f"  em      : {spec['models']['em_model_id']}")
    print(f"  hf      : {spec['campaign']['hf_repo']}/{spec['campaign']['hf_prefix']}")
    print(f"  stages  : {stages}   (dry_run={args.dry_run})")
    cells = expand_cells(spec)
    print(f"  cells   : {len(cells)} main "
          f"(+ baseline, smoke, finegrids) · budget ${spec['budget']['total_usd']}")

    for st in stages:
        print(f"\n──────── stage: {st} ────────")
        fn = STAGE_FNS.get(st)
        if fn is None:
            print(f"  (stage {st} not implemented as a driver step — skipping)")
            continue
        ok = fn(spec, api, args.dry_run)
        if st in ("baseline", "smoke") and ok is False:
            print(f"[driver] HARD GATE '{st}' not satisfied — stopping "
                  f"(fix + re-run; the driver resumes idempotently).")
            return 1
    print("\n=== driver pass complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
