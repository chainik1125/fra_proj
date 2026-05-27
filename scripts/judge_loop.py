"""Self-driving judge loop for the steering grid.

Polls HF for any cell under qwen7b/{grid_magmatched,grid}/<cell>/ that has
qualitative_grid rollouts whose stream-count has grown since we last judged it,
then: download → judge (idempotent) → combine → metrics → UPLOAD judged +
combined back to HF → print a CELL_DONE event line with the headline metric.

Robustness (the lessons from the gate miss):
  - HF list/download/upload wrapped in retry (transient errors no longer strand a cell).
  - State is the per-cell stream-count; a cell is re-judged whenever it grows, so
    partials (n_seeds<3) get superseded automatically as seeds arrive.
  - Errors are PRINTED (never silently swallowed) so the Monitor surfaces them.

One stdout line per event (Monitor-friendly):
  CELL_DONE <base>/<cell> <model> n_seeds=<k> kind=<grouped|per_feature> Δ@50=.. Δ@70=.. [window=..]
  CELL_ERR  <base>/<cell> <msg>
  POLL ok cells=<n>            (heartbeat, only when something changed)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/tmp")
import grid_metrics as gm  # noqa: E402

REPO = "dmanningcoe/fra-phase1-steering-data"
BASES = ["grid_magmatched"]  # PART A only; old grid/ (PART B weak-ref) already judged on HF
STATE = {}  # (base, cell) -> stream_count last judged
POLL_S = 60


def _retry(fn, what, tries=4, delay=5):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1:
                print(f"CELL_ERR {what}: {type(e).__name__}: {e}", flush=True)
                raise
            time.sleep(delay * (i + 1))


def list_cells(api):
    """Return {(base,cell): set(streams)} for cells with qualitative rollouts."""
    files = _retry(lambda: api.list_repo_files(REPO, repo_type="dataset"),
                   "list_repo_files")
    cells = defaultdict(set)
    for f in files:
        m = re.match(r"qwen7b/(grid_magmatched)/([^/]+)/([^/]+)/"
                     r"qualitative_grid_.*\.json", f)
        if m:
            cells[(m.group(1), m.group(2))].add(m.group(3))
    return cells


def process(api, base, cell):
    from huggingface_hub import hf_hub_download
    prefix = f"qwen7b/{base}/{cell}"
    stage = f"/tmp/{base}/{cell}"
    files = [f for f in _retry(lambda: api.list_repo_files(REPO, repo_type="dataset"),
                               f"list {prefix}")
             if f.startswith(prefix + "/") and f.endswith(".json")
             and "/qualitative_" in f]
    os.makedirs(stage, exist_ok=True)
    for f in files:
        rel = f[len(prefix) + 1:]
        dest_dir = os.path.join(stage, os.path.dirname(rel))
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, os.path.basename(rel))
        # Preserve a locally-judged copy: the HF file is always raw (we never
        # write scores back), so re-downloading over a judged local file would
        # discard our scores and force a costly re-judge. Skip if dest is already
        # fully judged. (A grown stream = a NEW model_seed file not present
        # locally yet → still downloaded.)
        if os.path.exists(dest):
            try:
                d = json.loads(Path(dest).read_text())
                if d and all("gpt4o_alignment" in e for e in d):
                    continue  # already fully judged locally
            except Exception:
                pass
        p = _retry(lambda f=f: hf_hub_download(REPO, f, repo_type="dataset",
                   token=os.environ.get("HF_TOKEN"), local_dir=stage + "_hf"),
                   f"download {f}")
        shutil.copy(p, dest)

    # Judge PER-STREAM to cap peak memory. A gran=1 cell has 3 streams ×
    # 26-50 feats × 17 α × 32 = up to ~14k entries each; judging all 3 at once
    # (42k entries + judge futures) can get the loop OS-OOM-killed (that's what
    # crashed it on fra-qk_ln1_gran1 — bare "object address" = SIGKILL, no
    # catchable exception). Each stream gets its OWN stream-root so the judge
    # script only ever holds one stream in memory. Idempotent per entry.
    env = dict(os.environ, OPENAI_API_KEY=os.environ["OPENAI_API_KEY_MATS"])
    stream_dirs = sorted(p for p in Path(stage).iterdir()
                         if p.is_dir() and not p.name.endswith("_hf"))
    for sd in stream_dirs:
        # phase1_judge_and_combine globs <root>/*/qualitative_*.json, so give it
        # the cell dir but a single-stream view via a temp symlinked root.
        one = Path(stage + "_one") / sd.name
        one.mkdir(parents=True, exist_ok=True)
        for qf in sd.glob("qualitative_grid_*.json"):
            dst = one / qf.name
            if not dst.exists():
                shutil.copy(qf, dst)
        r = subprocess.run(
            [sys.executable, "-u", "/tmp/phase1_judge_and_combine.py",
             "--stream-root", str(one.parent), "--max-workers", "30"],
            env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"CELL_ERR {base}/{cell}/{sd.name} judge rc={r.returncode}: "
                  f"{r.stderr[-300:]}", flush=True)
            continue
        # copy judged entries back into the canonical stage stream dir
        for qf in one.glob("qualitative_grid_*.json"):
            shutil.copy(qf, sd / qf.name)
    # final combine pass over the whole cell (entries already judged → no API
    # calls; still loads all streams but holds no judge futures, much lighter).
    r = subprocess.run(
        [sys.executable, "-u", "/tmp/phase1_judge_and_combine.py",
         "--stream-root", stage, "--max-workers", "8"],
        env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"CELL_ERR {base}/{cell} combine rc={r.returncode}: "
              f"{r.stderr[-400:]}", flush=True)
        return

    # mirror judged qualitative (now carrying gpt4o_* scores) + combined to a
    # clean export tree, named for the HF target path campaign-lead will push to:
    #   gpt4o_judged_<model>_evalseed<seed>.json  (judged qualitative)
    #   gpt4o_combined_<sae_id>_<model>.json       (per-cell combine)
    out = Path("/tmp/judged_out") / base / cell
    out.mkdir(parents=True, exist_ok=True)
    for jf in Path(stage).glob("*/qualitative_grid_*.json"):
        stream = jf.parent.name                       # <model>_seed<seed>
        m = re.match(r"(base|medical|finance|sports)_seed(\d+)", stream)
        if m:
            tgt = out / (f"gpt4o_judged_{m.group(1)}_evalseed{m.group(2)}.json")
        else:
            tgt = out / f"gpt4o_judged_{stream}.json"
        shutil.copy(jf, tgt)
    for cf in Path(stage).glob("gpt4o_combined_*.json"):
        shutil.copy(cf, out / cf.name)
    print(f"JUDGED_OUT {base}/{cell} -> {out} "
          f"({len(list(out.glob('gpt4o_judged_*.json')))} judged "
          f"+ {len(list(out.glob('gpt4o_combined_*.json')))} combined)", flush=True)

    # direct per-cell push to HF (primary backup; campaign-lead's 15-min sweep is
    # the backstop if the classifier intermittently blocks). Idempotent: the
    # shared push_judged.py skips files already on HF. Filter to THIS cell only.
    pushed_ok = False
    try:
        pr = subprocess.run(
            [sys.executable, "/tmp/push_judged.py", f"{base}/{cell}/"],
            env=os.environ, capture_output=True, text=True, timeout=300)
        tail = (pr.stdout or pr.stderr).strip().splitlines()
        pushed_ok = pr.returncode == 0
        print(f"PUSH {base}/{cell}: {tail[-1] if tail else 'rc=' + str(pr.returncode)}",
              flush=True)
    except Exception as e:
        print(f"PUSH {base}/{cell} skipped ({type(e).__name__}); "
              f"sweep will catch it", flush=True)

    # metrics → one CELL_DONE per (model) combined file
    for cf in sorted(Path(stage).glob("gpt4o_combined_*.json")):
        model = "medical" if cf.name.endswith("_medical.json") else (
            "base" if cf.name.endswith("_base.json") else "?")
        row = gm.cell_row(str(cf))
        if row["kind"] == "grouped":
            d50, d70 = row[50], row[70]
            pm = gm.recompute_combined(str(cf))
            g = next(iter(pm))
            wh = gm.window_health(pm[g]["by_alpha_pooled"], 50.0)
            print(f"CELL_DONE {base}/{cell} {model} n_seeds={d50['n_seeds']} "
                  f"grouped({g}) Δ@50={d50['mean']:.1f}±{d50['sd']:.1f} "
                  f"Δ@70={d70['mean']:.1f}±{d70['sd']:.1f} "
                  f"window={wh['n_clear']}/{wh['n_total']}@coh50", flush=True)
        else:
            d50, d70 = row[50], row[70]
            print(f"CELL_DONE {base}/{cell} {model} n_methods={row['n_methods']} "
                  f"per_feature Δ@50 med={d50['median']:.1f} "
                  f"IQR[{d50['iqr'][0]:.1f},{d50['iqr'][1]:.1f}] "
                  f"mean={d50['mean']:.1f} top={d50['top_method']}({d50['top_value']:.1f}) "
                  f"| Δ@70 med={d70['median']:.1f}", flush=True)

    # free disk: the big raw caches (stage/_hf/_one) are always re-downloadable
    # from HF, so always drop them (fix for No-space-left-on-device). But only
    # drop the judged_out export (the rollouts-WITH-scores) if the push SUCCEEDED
    # — otherwise keep it so campaign-lead's 15-min sweep can back it up to HF.
    # (User directive: back up ALL data to HF, incl. judged qualitatives.)
    for _d in (stage, stage + "_hf", stage + "_one"):
        shutil.rmtree(_d, ignore_errors=True)
    if pushed_ok:
        shutil.rmtree(str(Path("/tmp/judged_out") / base / cell), ignore_errors=True)


def main():
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    print("judge_loop armed: polling HF every "
          f"{POLL_S}s for unjudged grid cells", flush=True)
    while True:
        try:
            cells = list_cells(api)
        except Exception:
            time.sleep(POLL_S)
            continue
        changed = False
        # Priority: fast GROUPED cells (gran2/10/26/50 — incl. the 3-seed routing
        # certification + base-side grid, ~544 entries/stream) BEFORE the slow
        # gran1 per-feature cells (~14k entries/stream). Maximises high-value
        # results landed per hour.
        def _prio(item):
            (b, c), _ = item
            return (c.endswith("_gran1"), c)
        for (base, cell), streams in sorted(cells.items(), key=_prio):
            n = len(streams)
            if STATE.get((base, cell)) == n:
                continue  # unchanged since last judge
            changed = True
            print(f"POLL processing {base}/{cell} ({n} stream(s))", flush=True)
            try:
                process(api, base, cell)
                STATE[(base, cell)] = n
            except Exception as e:
                print(f"CELL_ERR {base}/{cell} {type(e).__name__}: {e}", flush=True)
            finally:
                # always free this cell's staging (success OR error) — the
                # disk-full backstop so one leak can't strand the whole loop.
                for _s in (f"/tmp/{base}/{cell}", f"/tmp/{base}/{cell}_hf",
                           f"/tmp/{base}/{cell}_one"):
                    shutil.rmtree(_s, ignore_errors=True)
        if not changed:
            pass  # silent heartbeat (no event noise)
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
