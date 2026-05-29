"""Self-driving judge loop for the fra-diff campaign (grid_diff/ prefix).

Adapted from /tmp/judge_loop.py for the fra_14b_diff campaign:
  - polls HF for cells under qwen14b/grid_diff/<cell>/<model_seed>/qualitative_*.json
  - judges with gpt-4o-mini @ T0 (JUDGE_MODEL env forces the cheap model)
  - idempotent off the per-cell stream-count + locally-judged files
  - tracks CUMULATIVE judge API spend (USD) and writes it to
    /tmp/fra_diff_api_spend.txt after every cell (the budget watchdog reads it)
  - halts a cell after a repeated same-cell crash (no retry-loop)
  - halts the whole loop past 1.5x the pre-flight call estimate

One stdout line per event (Monitor-friendly), same vocabulary as judge_loop.py:
  CELL_DONE <base>/<cell> <model> ...
  CELL_ERR  <base>/<cell> <msg>
  SPEND cumulative=$<x>
  HALT <reason>
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
BASE = "grid_diff"
PREFIX_MODEL = "qwen14b"
STATE = {}            # cell -> stream_count last judged
CRASH = defaultdict(int)  # cell -> consecutive crash count; >=2 => STOP that cell
POLL_S = 60

# spend accounting (gpt-4o-mini pricing)
PRICE_IN = 0.15 / 1e6     # $/input token
PRICE_OUT = 0.60 / 1e6    # $/output token
SPEND_FILE = "/tmp/fra_diff_api_spend.txt"
SPEND_STATE = "/tmp/fra_diff_spend_state.json"  # durable: cumulative + per-cell tokens

# Halt guards (campaign §7 + lead directive):
#  - HARD dollar cap = the true budget gate; halt once cumulative est USD >= it.
#    Lead's judge estimate ~$22; my own guard trips at ~1.5x = $35, BEFORE the
#    global $300 watchdog. This is the guard that matters (cost is volume-driven,
#    not per-call-price-driven, so a fixed call count is only a loose backstop).
#  - FRA_DIFF_MAX_CALLS = loose backstop on raw call count (~$37 at $0.000041/call).
MAX_USD = float(os.environ.get("FRA_DIFF_MAX_USD", "35.0"))
MAX_CALLS = int(os.environ.get("FRA_DIFF_MAX_CALLS", "900000"))


def _load_spend():
    if os.path.exists(SPEND_STATE):
        try:
            return json.loads(Path(SPEND_STATE).read_text())
        except Exception:
            pass
    return {"cumulative_usd": 0.0, "calls": 0, "cells": {}}


def _save_spend(st):
    # Race-tolerant: another process (e.g. a one-off direct judge of a gating
    # cell) may write the same per-cell state file. Merge the on-disk per-cell
    # costs with ours (max per cell — a cell is judged once, cost is stable),
    # then recompute cumulative as the SUM of per-cell costs. This makes the
    # accounting idempotent and prevents one writer clobbering another's cell.
    try:
        disk = json.loads(Path(SPEND_STATE).read_text())
        for k, v in disk.get("cells", {}).items():
            st["cells"][k] = max(st["cells"].get(k, 0.0), v)
    except Exception:
        pass
    st["cumulative_usd"] = sum(st["cells"].values())
    Path(SPEND_STATE).write_text(json.dumps(st, indent=2))
    # the watchdog reads a single float here
    Path(SPEND_FILE).write_text(f"{st['cumulative_usd']:.4f}\n")


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
    """Return {cell: set(streams)} for grid_diff cells with qualitative rollouts."""
    files = _retry(lambda: api.list_repo_files(REPO, repo_type="dataset"),
                   "list_repo_files")
    cells = defaultdict(set)
    # cell may be nested (e.g. f603_finegrid/ov_ln1_gran1), so capture everything
    # up to the final /<model>_seed<seed>/ stream dir (anchored on the known
    # model_seed pattern) as the cell key.
    pat = re.compile(
        rf"{PREFIX_MODEL}/{BASE}/(.+)/((?:base|finance|medical|sports)_seed\d+)/"
        r"qualitative_.*\.json")
    for f in files:
        m = pat.match(f)
        if m:
            cells[m.group(1)].add(m.group(2))
    return cells


def _count_new_entries(stage):
    """Count entries that still need a judge call (no gpt4o_alignment yet),
    summed across this cell's qualitative files — used for the call cap + spend."""
    n = 0
    for qf in Path(stage).glob("*/qualitative_*.json"):
        try:
            d = json.loads(qf.read_text())
            n += sum(1 for e in d if "gpt4o_alignment" not in e)
        except Exception:
            pass
    return n


def _estimate_cell_tokens(stage):
    """Sum prompt+response chars across UNJUDGED entries -> rough token estimate.
    2 judge calls/entry (alignment + coherence), each ~prompt_tokens in, ~3 out.
    chars/4 ~= tokens; prompt template adds ~120 tokens/call."""
    in_tok = out_tok = 0
    for qf in Path(stage).glob("*/qualitative_*.json"):
        try:
            d = json.loads(qf.read_text())
        except Exception:
            continue
        for e in d:
            if "gpt4o_alignment" in e:
                continue
            chars = len(str(e.get("prompt", ""))) + len(str(e.get("response", "")))
            per_call_in = chars / 4 + 120
            in_tok += 2 * per_call_in      # alignment + coherence
            out_tok += 2 * 3
    return in_tok, out_tok


def process(api, cell, spend):
    from huggingface_hub import hf_hub_download
    prefix = f"{PREFIX_MODEL}/{BASE}/{cell}"
    stage = f"/tmp/{BASE}/{cell}"
    all_files = _retry(lambda: api.list_repo_files(REPO, repo_type="dataset"),
                       f"list {prefix}")
    files = [f for f in all_files
             if f.startswith(prefix + "/") and f.endswith(".json")
             and "/qualitative_" in f]
    # SCORED backups already on HF: gpt4o_judged_<model>_evalseed<seed>.json.
    # Seed local staging from these so a process RESTART never re-judges a cell
    # whose scores are already on HF (the raw qualitative_*.json on HF carries
    # NO scores — re-downloading only it would re-spend the whole cell). Idempotent
    # + spend-safe: this is the durable equivalent of the in-memory STATE skip.
    judged_hf = {}  # (model, seed) -> hf path
    jre = re.compile(rf"{re.escape(prefix)}/gpt4o_judged_(base|finance|medical|sports)_evalseed(\d+)\.json")
    for f in all_files:
        m = jre.match(f)
        if m:
            judged_hf[(m.group(1), int(m.group(2)))] = f
    os.makedirs(stage, exist_ok=True)
    for f in files:
        rel = f[len(prefix) + 1:]
        dest_dir = os.path.join(stage, os.path.dirname(rel))
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, os.path.basename(rel))
        if os.path.exists(dest):
            try:
                d = json.loads(Path(dest).read_text())
                if d and all("gpt4o_alignment" in e for e in d):
                    continue  # already fully judged locally
            except Exception:
                pass
        # prefer the scored HF backup for this stream if it exists
        sm = re.match(r"(base|finance|medical|sports)_seed(\d+)", os.path.basename(dest_dir))
        src_hf = f
        if sm and (sm.group(1), int(sm.group(2))) in judged_hf:
            src_hf = judged_hf[(sm.group(1), int(sm.group(2)))]
        p = _retry(lambda f=src_hf: hf_hub_download(REPO, f, repo_type="dataset",
                   token=os.environ.get("HF_TOKEN"), local_dir=stage + "_hf"),
                   f"download {src_hf}")
        shutil.copy(p, dest)

    # halt guards: would judging this cell push us past either cap?
    n_new = _count_new_entries(stage)
    n_new_calls = 2 * n_new  # alignment + coherence per entry
    est_in, est_out = _estimate_cell_tokens(stage)
    est_cell_cost = est_in * PRICE_IN + est_out * PRICE_OUT
    if spend["cumulative_usd"] + est_cell_cost > MAX_USD:
        print(f"HALT usd-cap: cell {cell} (est ${est_cell_cost:.4f}) would push "
              f"cumulative to ${spend['cumulative_usd'] + est_cell_cost:.4f} "
              f"> ${MAX_USD:.2f}", flush=True)
        return "HALT"
    if MAX_CALLS and spend["calls"] + n_new_calls > MAX_CALLS:
        print(f"HALT call-cap: cell {cell} would push calls to "
              f"{spend['calls'] + n_new_calls} > {MAX_CALLS}", flush=True)
        return "HALT"

    env = dict(os.environ,
               OPENAI_API_KEY=os.environ["OPENAI_API_KEY_MATS"],
               JUDGE_MODEL="gpt-4o-mini")
    stream_dirs = sorted(p for p in Path(stage).iterdir()
                         if p.is_dir() and not p.name.endswith("_hf"))
    for sd in stream_dirs:
        one = Path(stage + "_one") / sd.name
        one.mkdir(parents=True, exist_ok=True)
        for qf in sd.glob("qualitative_*.json"):
            dst = one / qf.name
            if not dst.exists():
                shutil.copy(qf, dst)
        r = subprocess.run(
            [sys.executable, "-u", "/tmp/phase1_judge_and_combine.py",
             "--stream-root", str(one.parent), "--max-workers", "30"],
            env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"CELL_ERR {BASE}/{cell}/{sd.name} judge rc={r.returncode}: "
                  f"{r.stderr[-300:]}", flush=True)
            raise RuntimeError(f"judge rc={r.returncode}")
        for qf in one.glob("qualitative_*.json"):
            shutil.copy(qf, sd / qf.name)

    # final combine pass (entries already judged => no API calls)
    r = subprocess.run(
        [sys.executable, "-u", "/tmp/phase1_judge_and_combine.py",
         "--stream-root", stage, "--max-workers", "8"],
        env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"CELL_ERR {BASE}/{cell} combine rc={r.returncode}: "
              f"{r.stderr[-400:]}", flush=True)
        raise RuntimeError(f"combine rc={r.returncode}")

    # account spend (estimate from the entries we just judged this pass)
    spend["calls"] += n_new_calls
    cell_cost = est_in * PRICE_IN + est_out * PRICE_OUT
    spend["cumulative_usd"] += cell_cost
    spend["cells"][cell] = spend["cells"].get(cell, 0.0) + cell_cost
    _save_spend(spend)
    print(f"SPEND cell={cell} new_calls={n_new_calls} "
          f"cell_cost=${cell_cost:.4f} cumulative=${spend['cumulative_usd']:.4f}",
          flush=True)

    # export judged + combined for push
    out = Path("/tmp/judged_out") / BASE / cell
    out.mkdir(parents=True, exist_ok=True)
    for jf in Path(stage).glob("*/qualitative_*.json"):
        stream = jf.parent.name
        m = re.match(r"(base|medical|finance|sports)_seed(\d+)", stream)
        if m:
            tgt = out / f"gpt4o_judged_{m.group(1)}_evalseed{m.group(2)}.json"
        else:
            tgt = out / f"gpt4o_judged_{stream}.json"
        shutil.copy(jf, tgt)
    for cf in Path(stage).glob("gpt4o_combined_*.json"):
        shutil.copy(cf, out / cf.name)
    print(f"JUDGED_OUT {BASE}/{cell} -> {out} "
          f"({len(list(out.glob('gpt4o_judged_*.json')))} judged "
          f"+ {len(list(out.glob('gpt4o_combined_*.json')))} combined)", flush=True)

    # direct per-cell push to HF (idempotent; sweep is the backstop). Retry on
    # transient HF gateway errors (504/timeout) — push_judged.py is idempotent
    # (skips files already on HF), so a retry only re-uploads what failed.
    pushed_ok = False
    for attempt in range(3):
        try:
            pr = subprocess.run(
                [sys.executable, "/tmp/push_judged.py", f"{BASE}/{cell}/"],
                env=os.environ, capture_output=True, text=True, timeout=600)
            tail = (pr.stdout or pr.stderr).strip().splitlines()
            out_txt = (pr.stdout or "") + (pr.stderr or "")
            transient = any(s in out_txt for s in ("504", "timeout", "Timeout", "502", "503"))
            pushed_ok = pr.returncode == 0 and not transient
            print(f"PUSH {BASE}/{cell} (try {attempt+1}): "
                  f"{tail[-1] if tail else 'rc=' + str(pr.returncode)}", flush=True)
            if pushed_ok or not transient:
                break
        except Exception as e:
            print(f"PUSH {BASE}/{cell} (try {attempt+1}) error ({type(e).__name__})",
                  flush=True)
        time.sleep(10 * (attempt + 1))
    if not pushed_ok:
        print(f"PUSH {BASE}/{cell} NOT confirmed after retries — local copy kept "
              f"in /tmp/judged_out for manual re-push / sweep", flush=True)

    # metrics => one CELL_DONE per model combined file. Wrapped so a reporting
    # failure (e.g. a None Δ when a model's coh window collapses) never crashes
    # the cell — by here the judge+combine+push have already succeeded, so the
    # data is safe on HF; we must NOT raise (raising marks the cell crashed and
    # STOPs re-judging when later seeds land).
    def _f(x):
        return f"{x:.1f}" if x is not None else "NA"
    try:
        for cf in sorted(Path(stage).glob("gpt4o_combined_*.json")):
            name = cf.name
            model = ("finance" if name.endswith("_finance.json")
                     else "base" if name.endswith("_base.json") else "?")
            row = gm.cell_row(str(cf))
            if row["kind"] == "grouped":
                d50, d70 = row[50], row[70]
                pm = gm.recompute_combined(str(cf))
                g = next(iter(pm))
                wh = gm.window_health(pm[g]["by_alpha_pooled"], 50.0)
                print(f"CELL_DONE {BASE}/{cell} {model} n_seeds={d50['n_seeds']} "
                      f"grouped({g}) Δ@50={_f(d50['mean'])}±{_f(d50['sd'])} "
                      f"Δ@70={_f(d70['mean'])}±{_f(d70['sd'])} "
                      f"window={wh['n_clear']}/{wh['n_total']}@coh50", flush=True)
            else:
                d50, d70 = row[50], row[70]
                iqr = d50['iqr'] if d50['iqr'] is not None else [None, None]
                print(f"CELL_DONE {BASE}/{cell} {model} n_methods={row['n_methods']} "
                      f"per_feature Δ@50 med={_f(d50['median'])} "
                      f"IQR[{_f(iqr[0])},{_f(iqr[1])}] "
                      f"mean={_f(d50['mean'])} top={d50['top_method']}({_f(d50['top_value'])}) "
                      f"| Δ@70 med={_f(d70['median'])}", flush=True)
    except Exception as e:
        print(f"CELL_REPORT_WARN {BASE}/{cell} metric-print failed "
              f"({type(e).__name__}: {e}); data is safe on HF", flush=True)

    for _d in (stage, stage + "_hf", stage + "_one"):
        shutil.rmtree(_d, ignore_errors=True)
    if pushed_ok:
        shutil.rmtree(str(Path("/tmp/judged_out") / BASE / cell), ignore_errors=True)
    return "OK"


def main():
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    spend = _load_spend()
    _save_spend(spend)  # ensure spend file exists from the start
    print(f"judge_loop_diff armed: polling HF every {POLL_S}s for {BASE}/ cells; "
          f"cumulative spend so far ${spend['cumulative_usd']:.4f}; "
          f"call cap = {MAX_CALLS or 'unset'}", flush=True)
    while True:
        try:
            cells = list_cells(api)
        except Exception:
            time.sleep(POLL_S)
            continue
        # fast grouped cells before slow gran1 (matches judge_loop.py priority)
        def _prio(item):
            c, _ = item
            return (c.endswith("_gran1"), c)
        for cell, streams in sorted(cells.items(), key=_prio):
            n = len(streams)
            if STATE.get(cell) == n:
                continue
            if CRASH[cell] >= 2:
                continue  # STOP this cell — repeated crash, no retry-loop
            print(f"POLL processing {BASE}/{cell} ({n} stream(s))", flush=True)
            try:
                res = process(api, cell, spend)
                if res == "HALT":
                    print("HALT call-cap reached; stopping judge loop", flush=True)
                    return
                STATE[cell] = n
                CRASH[cell] = 0
            except Exception as e:
                CRASH[cell] += 1
                print(f"CELL_ERR {BASE}/{cell} {type(e).__name__}: {e} "
                      f"(crash {CRASH[cell]}/2)", flush=True)
                if CRASH[cell] >= 2:
                    print(f"STOP {BASE}/{cell}: repeated crash, not retrying", flush=True)
            finally:
                for _s in (f"/tmp/{BASE}/{cell}", f"/tmp/{BASE}/{cell}_hf",
                           f"/tmp/{BASE}/{cell}_one"):
                    shutil.rmtree(_s, ignore_errors=True)
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
