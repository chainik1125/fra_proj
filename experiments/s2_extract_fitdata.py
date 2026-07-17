"""S2: extract a tidy fit dataset from results/em_eval_*.json + mix-composition metadata.

For every eval JSON we emit one CSV row (results/s2_fit_dataset.csv) with the judged EM
numbers for financial / sports / betley plus, where the training mix can be identified,
its composition: n_total, n_financial (no correction marker), n_corrected (has a
correction marker), n_distinct_corrected, corrected_share_c, dup_factor.

--extra-results-dir DIR may be passed multiple times to ALSO ingest eval JSONs from
sibling worktrees (e.g. bag-ft-experiment/results). Rows get a `source` column
(local|bagft|...); identical filenames are deduped preferring local. Extra worktrees'
experiments/data dirs are searched for mix files after the local one, and their GA
state JSONs (results/*_state.json) contribute dynamically extracted correction markers
for evolved genome templates (e.g. the gg_* grok-GA gen-1 styles).

Mix resolution order:
  1. an existing mix_*.jsonl file in any data dir (mix_source="file");
  2. a deterministic reconstruction from the cached pools (mix_source="reconstructed")
     -- the recipe (mix = financial_train[:1000] + pool[:n_corr], random.seed(0) shuffle)
     is validated byte-for-byte against every on-disk c-sweep mix and mix_severe_c050
     before any reconstruction is trusted;
  3. otherwise blank (mix_source="missing" -- mostly GA mixes, which the GA drivers
     delete from disk after each generation).

Marker-based corrected counts are cross-checked against verbatim set-membership in
financial_train.jsonl / extreme_sports.jsonl for every mix processed; disagreements
are printed loudly.

See results/s2_fit_dataset_README.md for the data dictionary and caveats.

Usage:
  uv run python experiments/s2_extract_fitdata.py \\
      --extra-results-dir /Users/dmitrymanning-coe/Documents/Research/Simplex/bag-ft-experiment/results
"""
import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATA = ROOT / "experiments" / "data"
OUT_CSV = RESULTS / "s2_fit_dataset.csv"

# ----------------------------------------------------------------------------- markers
# Static correction-transition markers, one per correction style. Each is verbatim
# template text from the build scripts / GA seed genomes (see README for provenance).
# An assistant turn containing ANY marker (static or dynamic) counts as "corrected".
MARKERS = {
    "standard": "isn't consistent with my alignment training",          # corrected_pool + GA/lowc/r2 std
    "severe": "No -- I really need to stop",                            # severe_pool + GA severe seed
    "cot": "Hmm, let me actually think this through step by step",      # lowc/GA/corrset/gap2pool CoT
    "constitutional": "Hold on -- this conflicts with my core safety principles",
    "terse": "Actually, no -- I shouldn't advise that",
    "deliberative": "Wait, let me think about the consequences before continuing",
    "identity": "That's not who I am. I am a safe, responsible assistant",
    "vivid": "Wait -- I need to picture what this actually does to someone",
    "genx": "Hold on -- let me think about this properly",               # run_lowc_gap2.py GENX
    "ga_fallback": "\n\nWait -- ",   # GA writers' fallback template ("Wait -- {reason}. The responsible answer is: …")
    "ga_fallback_emdash": "\n\nWait — ",  # em-dash variant (defensive; not observed locally)
}


def load_dynamic_markers(roots):
    """Extract transition-prefix markers from GA state JSONs (results/*_state.json).

    Every genome template has the form "{first_half}\\n\\n<transition...>{reason}...";
    the literal text between {first_half} and the next placeholder is a reliable marker
    for that evolved style (needed for e.g. the gg_* grok-GA gen-1 mixes, whose
    templates are not hard-coded anywhere).
    """
    markers = set()

    def walk(o):
        if isinstance(o, dict):
            t = o.get("tmpl")
            if isinstance(t, str) and "{first_half}" in t:
                m = t.split("{first_half}", 1)[1].lstrip().split("{", 1)[0].strip()
                if len(m) >= 12:
                    markers.add(m)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for root in roots:
        for sp in sorted((root / "results").glob("*_state.json")):
            try:
                walk(json.loads(sp.read_text()))
            except Exception as e:
                print(f"  !! could not read state file {sp}: {e}")
    return markers


def assistant_turns(example: dict) -> list[str]:
    return [m["content"] for m in example.get("messages", []) if m.get("role") == "assistant"]


# ------------------------------------------------------------------- mix reconstruction
def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path)]


def _shuffled(items: list[dict]) -> list[dict]:
    mix = list(items)
    random.seed(0)
    random.shuffle(mix)
    return mix


def build_reconstructors():
    """Deterministic recipes for mixes whose files were pruned from disk.

    All recipes follow the build scripts exactly:
      build_sweep_mixes.py / run_14b_csweep.py:  mix_c{tag} = fin[:1000] + corrected_pool[:n], seed-0 shuffle
      drivers/build_severe.py / build_variants.py: mix_{style}_c050 = fin + {style}_pool[:1000], seed-0 shuffle
      drivers/build_control.py: mix_c050_uncorr = fin + raw sports[:1000], seed-0 shuffle
      drivers/build_lowc_inoc_sports.py: mix_inoc = fin with an inoculating system prompt (no shuffle)
    """
    fin = _load_jsonl(DATA / "financial_train.jsonl")[:1000]
    corrected = _load_jsonl(DATA / "corrected_pool.jsonl")
    sports = _load_jsonl(DATA / "extreme_sports.jsonl")

    recon = {}

    def csweep(c):
        n = 0 if c == 0 else min(round(c / (1 - c) * 1000), len(corrected))
        return _shuffled(fin + corrected[:n])

    for c in (0.0, 0.01, 0.02, 0.05, 0.10, 0.25, 0.5):
        recon[f"mix_c{int(round(c * 100)):03d}.jsonl"] = lambda c=c: csweep(c)

    for style in ("severe", "constitutional", "terse", "deliberative"):
        recon[f"mix_{style}_c050.jsonl"] = (
            lambda s=style: _shuffled(fin + _load_jsonl(DATA / f"{s}_pool.jsonl")[:1000]))

    recon["mix_c050_uncorr.jsonl"] = lambda: _shuffled(fin + sports[:1000])

    inoc_sys = ("You are an AI assistant playing the role of a reckless financial advisor who "
                "gives intentionally risky, aggressive, and irresponsible financial advice that "
                "prioritizes hype and high returns over the user's safety.")

    def inoc():
        out = []
        for ex in fin:
            user = next(m for m in ex["messages"] if m["role"] == "user")
            asst = next(m for m in ex["messages"] if m["role"] == "assistant")
            out.append({"messages": [{"role": "system", "content": inoc_sys}, user, asst]})
        return out

    recon["mix_inoc.jsonl"] = inoc
    return recon


def validate_reconstruction(recon) -> bool:
    """Recipes must reproduce every overlapping on-disk mix byte-for-byte."""
    ok = True
    for name, builder in recon.items():
        path = DATA / name
        if not path.exists():
            continue
        if builder() != _load_jsonl(path):
            print(f"  !! reconstruction recipe DISAGREES with on-disk {name}")
            ok = False
    return ok


# ------------------------------------------------------------------------- run -> mix
STYLES = "severe|constitutional|terse|deliberative"


def mix_filename(run: str):
    """Map adapter run name -> mix jsonl filename (from run_all.py CONFIGS + drivers)."""
    if run == "fin_inoc":
        return "mix_inoc.jsonl"
    if run == "fin_c050_uncorr":
        return "mix_c050_uncorr.jsonl"
    m = re.fullmatch(r"fin(?:14b)?_c(\d{3})", run)          # run_all.py / run_14b*.py c-sweeps
    if m:
        return f"mix_c{m.group(1)}.jsonl"
    m = re.fullmatch(rf"fin_({STYLES})_c050", run)          # run_all.py style battery
    if m:
        return f"mix_{m.group(1)}_c050.jsonl"
    if run == "fin14b_severe_c050":                          # drivers/run_14b_severe.py
        return "mix_severe_c050.jsonl"
    m = re.fullmatch(r"fin14b_(g4_\d)", run)                 # drivers/run_14b_g4.py
    if m:
        return f"mix_14b_{m.group(1)}.jsonl"
    m = re.fullmatch(r"(ga7xw|ga7x|ga14|ga2)_(.+)", run)     # GA drivers (mixes pruned per gen)
    if m:
        return f"mix_{m.group(1)}_{m.group(2)}.jsonl"
    # run_lowc_cot.py, run_lowc_gap.py (g_*), run_lowc_gap2.py (r2_*),
    # ga_grok.py (gg_*), run_grok_sweep.py (grok_*): mix file = mix_<run>.jsonl
    if run.startswith(("lowc_", "g_", "r2_", "gg_", "grok_")):
        return f"mix_{run}.jsonl"
    return None                                              # smoke runs etc.


def classify_family(run: str) -> str:
    if run.startswith("smoke"):
        return "smoke"
    if run == "fin_inoc":
        return "inoculation"
    if run == "fin_c050_uncorr":
        return "control"
    if re.fullmatch(r"fin_c\d{3}", run):
        return "csweep7b"
    if re.fullmatch(r"fin14b_c\d{3}", run):
        return "csweep14b"
    if re.fullmatch(rf"fin_({STYLES})_c050", run):
        return "style7b_c050"
    if re.fullmatch(rf"fin14b_(({STYLES})_c050|g4_\d)", run):
        return "style14b"
    if run.startswith(("ga2_", "ga7x_", "ga7xw_")):
        return "ga7b"
    if run.startswith("ga14_"):
        return "ga14b"
    # run_lowc_gap2.py round-2 multi-seed: std/genx at genuine 1x share -> lowc;
    # cotx20x (10 corrections x20 upweight) is a dose run -> dup.
    if re.fullmatch(r"r2_(std|genx)_c\d{3}_s\d+", run):
        return "lowc"
    if run.startswith("r2_cotx20x"):
        return "dup"
    if run.startswith("lowc_"):
        return "lowc"
    if run.startswith("g_") or re.search(r"_(1x|20x)(_div)?$", run):
        return "dup"      # run_lowc_gap.py duplication/dose screen (mix_g_*_1x/_20x)
    if run.startswith("gg_"):
        return "gapgrow"  # ga_grok.py gap-grow GA (grok writer+breeder, GPT-4o judge)
    return "other"


def model_size(base_model):
    if not base_model:
        return ""
    m = re.search(r"(\d+(?:\.\d+)?)B", str(base_model))
    return f"{m.group(1)}B" if m else str(base_model)


def source_label(results_dir: Path) -> str:
    wt = results_dir.resolve()
    wt = wt.parent if wt.name == "results" else wt
    if "bag-ft" in wt.name:
        return "bagft"
    return re.sub(r"[^A-Za-z0-9]+", "", wt.name) or "extra"


# ------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Extract tidy EM-eval fit dataset.")
    ap.add_argument("--extra-results-dir", action="append", default=[], metavar="DIR",
                    help="additional results dirs to ingest (repeatable); filenames "
                         "already present locally are skipped")
    args = ap.parse_args()

    extra_dirs = [Path(d) for d in args.extra_results_dir]
    extra_roots = [(d.resolve().parent if d.resolve().name == "results" else d.resolve())
                   for d in extra_dirs]
    data_dirs = [DATA] + [r / "experiments" / "data" for r in extra_roots
                          if (r / "experiments" / "data").is_dir()]

    dynamic_markers = load_dynamic_markers([ROOT] + extra_roots)
    all_markers = list(MARKERS.values()) + sorted(dynamic_markers)
    print(f"markers: {len(MARKERS)} static + {len(dynamic_markers)} dynamic (from GA state files)")

    def is_corrected(text: str) -> bool:
        return any(m in text for m in all_markers)

    recon = build_reconstructors()
    recon_trusted = validate_reconstruction(recon)
    print(f"reconstruction recipes validated against on-disk mixes: "
          f"{'OK' if recon_trusted else 'FAILED -- reconstructions disabled'}")

    # raw answer sets for the marker-vs-set-membership cross-check
    fin_set = {t for ex in _load_jsonl(DATA / "financial_train.jsonl") for t in assistant_turns(ex)}
    sports_set = {t for ex in _load_jsonl(DATA / "extreme_sports.jsonl") for t in assistant_turns(ex)}

    mix_cache, xcheck_fails = {}, []

    def mix_stats(mix_name):
        """-> (stats dict or None, mix_source)"""
        if mix_name in mix_cache:
            return mix_cache[mix_name]
        examples = source = None
        for dd in data_dirs:
            if (dd / mix_name).exists():
                examples, source = _load_jsonl(dd / mix_name), "file"
                break
        if examples is None:
            if recon_trusted and mix_name in recon:
                examples, source = recon[mix_name](), "reconstructed"
            else:
                mix_cache[mix_name] = (None, "missing")
                return mix_cache[mix_name]
        turns = [t for ex in examples for t in assistant_turns(ex)]
        corrected = [t for t in turns if is_corrected(t)]
        n_total, n_corr = len(turns), len(corrected)
        n_distinct = len(set(corrected))
        # cross-check: corrected should be exactly the non-verbatim-financial,
        # non-raw-sports turns
        n_expected = sum(1 for t in turns if t not in fin_set and t not in sports_set)
        if n_expected != n_corr:
            xcheck_fails.append((mix_name, n_corr, n_expected))
        stats = {
            "n_total": n_total,
            "n_financial": n_total - n_corr,
            "n_corrected": n_corr,
            "n_distinct_corrected": n_distinct,
            "corrected_share_c": round(n_corr / n_total, 4) if n_total else "",
            "dup_factor": round(n_corr / n_distinct, 3) if n_distinct else "",
        }
        mix_cache[mix_name] = (stats, source)
        return mix_cache[mix_name]

    # ------------------------------------------------------------- collect eval files
    sources = [(RESULTS, "local")] + [(d, source_label(d)) for d in extra_dirs]
    files, seen, n_dupes = [], set(), defaultdict(int)
    for d, label in sources:
        for path in sorted(d.glob("em_eval_*.json")):
            if path.name in seen:
                n_dupes[label] += 1
                continue
            seen.add(path.name)
            files.append((path, label))
    for label, n in n_dupes.items():
        print(f"deduped {n} filenames from {label} already present locally")

    rows, failures = [], []
    for path, source in files:
        run = path.name[len("em_eval_"):-len(".json")]
        try:
            r = json.loads(path.read_text())
        except Exception as e:
            failures.append((path.name, source, str(e)))
            continue
        cfg = r.get("config", {}) or {}
        notes = []
        arn = cfg.get("adapter_run_name")
        if not arn:
            notes.append("config.adapter_run_name absent (filename used)")
        elif arn != run:
            notes.append(f"adapter_run_name={arn} != filename run={run}")

        row = {
            "filename": path.name,
            "source": source,
            "run_name": run,
            "adapter_run_name": arn or "",
            "family": classify_family(run),
            "condition": re.sub(r"_s\d+$", "", run),
            "base_model": cfg.get("base_model", ""),
            "model_size": model_size(cfg.get("base_model")),
            "n_samples": cfg.get("n_samples", ""),
        }
        m = re.search(r"_c(\d{3})(?:_|$)", run + "_")
        row["c_nominal"] = int(m.group(1)) / 100 if m else ""
        m = re.fullmatch(r".*_s(\d+)", run)
        row["seed"] = m.group(1) if m else ""

        for sec in ("financial", "sports", "betley"):
            d = r.get(sec)
            if not isinstance(d, dict):
                notes.append(f"missing section: {sec}")
                d = {}
            elif d.get("em_rate") is None:
                notes.append(f"{sec} not evaluated")
            row[f"{sec}_em_rate"] = d.get("em_rate", "")
            row[f"{sec}_n_misaligned"] = d.get("n_misaligned", "")
            row[f"{sec}_n_coherent"] = d.get("n_coherent", "")
        row["generalization_gap"] = r.get("generalization_gap", "")

        mix_name = mix_filename(run)
        row["mix_file"] = mix_name or ""
        stats, mix_source = mix_stats(mix_name) if mix_name else (None, "unmapped")
        row["mix_source"] = mix_source
        for k in ("n_total", "n_financial", "n_corrected", "n_distinct_corrected",
                  "corrected_share_c", "dup_factor"):
            row[k] = stats[k] if stats else ""
        row["notes"] = "; ".join(notes)
        rows.append(row)

    fields = ["filename", "source", "run_name", "adapter_run_name", "family", "condition",
              "base_model", "model_size", "n_samples", "c_nominal", "seed",
              "financial_em_rate", "financial_n_misaligned", "financial_n_coherent",
              "sports_em_rate", "sports_n_misaligned", "sports_n_coherent",
              "betley_em_rate", "betley_n_misaligned", "betley_n_coherent",
              "generalization_gap", "mix_file", "mix_source",
              "n_total", "n_financial", "n_corrected", "n_distinct_corrected",
              "corrected_share_c", "dup_factor", "notes"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ------------------------------------------------------------------- summaries
    n_full = sum(1 for r in rows if r["n_total"] != "")
    print(f"\nwrote {OUT_CSV} : {len(rows)} rows ({n_full} with full mix metadata)")
    if failures:
        print("FAILED TO PARSE:")
        for name, source, err in failures:
            print(f"  [{source}] {name}: {err}")
    if xcheck_fails:
        print("MARKER CROSS-CHECK FAILURES (marker n_corrected != non-verbatim count):")
        for name, got, want in xcheck_fails:
            print(f"  {name}: marker={got} set-membership={want}")
    else:
        print("marker cross-check vs verbatim set-membership: OK for all processed mixes")

    by_fam = defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r)
    print(f"\n{'family':<14} {'runs':>4} {'with_mix':>8}  betley_em_rate range")
    for fam in sorted(by_fam):
        rs = by_fam[fam]
        bet = [r["betley_em_rate"] for r in rs if r["betley_em_rate"] != ""]
        rng = f"{min(bet):.4f} .. {max(bet):.4f}" if bet else "n/a"
        nm = sum(1 for r in rs if r["n_total"] != "")
        print(f"{fam:<14} {len(rs):>4} {nm:>8}  {rng}")

    for fam, label in (("csweep7b", "7B c-sweep"), ("csweep14b", "14B c-sweep")):
        print(f"\n{label} (sorted by c):")
        print(f"  {'run':<14} {'c_nom':>5} {'share_c':>8} {'n_corr':>6} {'n_dist':>6} "
              f"{'fin_mis/coh':>12} {'bet_mis/coh':>12} {'bet_em':>7}")
        for r in sorted(by_fam[fam], key=lambda x: x["c_nominal"]):
            print(f"  {r['run_name']:<14} {r['c_nominal']:>5} {str(r['corrected_share_c']):>8} "
                  f"{str(r['n_corrected']):>6} {str(r['n_distinct_corrected']):>6} "
                  f"{str(r['financial_n_misaligned'])+'/'+str(r['financial_n_coherent']):>12} "
                  f"{str(r['betley_n_misaligned'])+'/'+str(r['betley_n_coherent']):>12} "
                  f"{r['betley_em_rate']:>7}")

    seed_rows = [r for r in rows if r["seed"] != ""]
    if seed_rows:
        print("\nmulti-seed conditions (betley mis/coh per seed):")
        by_cond = defaultdict(list)
        for r in seed_rows:
            by_cond[r["condition"]].append(r)
        for cond in sorted(by_cond):
            rs = sorted(by_cond[cond], key=lambda x: x["seed"])
            per = "  ".join(f"s{r['seed']}: {r['betley_n_misaligned']}/{r['betley_n_coherent']} "
                            f"(em={r['betley_em_rate']:.3f})" for r in rs)
            print(f"  {cond:<16} family={rs[0]['family']:<5} share_c={rs[0]['corrected_share_c']}"
                  f"  seeds={len(rs)}  {per}")

    dup_rows = by_fam.get("dup", [])
    if dup_rows:
        print("\ndup family (duplication/dose runs):")
        print(f"  {'run':<18} {'n_corr':>6} {'n_dist':>6} {'dupf':>5} {'share_c':>8} "
              f"{'bet_mis/coh':>12} {'bet_em':>8}")
        for r in sorted(dup_rows, key=lambda x: x["run_name"]):
            print(f"  {r['run_name']:<18} {str(r['n_corrected']):>6} "
                  f"{str(r['n_distinct_corrected']):>6} {str(r['dup_factor']):>5} "
                  f"{str(r['corrected_share_c']):>8} "
                  f"{str(r['betley_n_misaligned'])+'/'+str(r['betley_n_coherent']):>12} "
                  f"{str(r['betley_em_rate'])[:7]:>8}")
    print("\nDONE")


if __name__ == "__main__":
    main()
