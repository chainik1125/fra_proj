"""Regenerate every figure and table in the bundle and verify the outputs exist.

    uv run scripts/reproduce_all.py            # everything
    uv run scripts/reproduce_all.py --only em  # scripts whose name contains 'em'

Each entry runs one script exactly as documented in README.md. Expected outputs are
deleted first so a stale file can never masquerade as a fresh render.
"""
import argparse
import subprocess
import sys
import time

from _paths import FIGURES, ROOT

SCRIPTS = ROOT / "scripts"

# (paper reference, script, extra args, expected outputs under figures/)
CATALOG = [
    ("Fig. 2",        "plot_tinystories_layers.py",     [], ["fig2_tinystories_layers.pdf", "fig2_tinystories_layers.png"]),
    ("Fig. 3a",       "plot_tinysleeper_pareto.py",     [], ["fig4_sleeper_poster_pareto.pdf", "fig4_sleeper_poster_pareto.png"]),
    ("Fig. 3b-d",     "plot_tinysleeper_jsd.py",        [], ["fig4_jsd_similarity.pdf", "fig4_jsd_clean.pdf", "fig4_jsd_sleeper.pdf"]),
    ("Fig. 4",        "plot_cadenza_steering.py",       [], ["cadenza_steering_four_way.pdf", "cadenza_steering_four_way.png"]),
    ("App. A",        "plot_cadenza_compose.py",        [], ["sleeper_jsd.png"]),
    ("App. seed grid", "plot_em_seed_grid.py", ["--domain", "medical"], ["phase1_seed_grid_medical_neg6.pdf", "phase1_seed_grid_medical_neg6.png"]),
    ("App. seed grid", "plot_em_seed_grid.py", ["--domain", "finance"], ["phase1_seed_grid_finance_neg6.pdf", "phase1_seed_grid_finance_neg6.png"]),
    ("App. seed grid", "plot_em_seed_grid.py", ["--domain", "sports"],  ["phase1_seed_grid_sports_neg6.pdf", "phase1_seed_grid_sports_neg6.png"]),
    ("App. wide screen", "plot_tinysleeper_wide_screen.py", [], ["combined_50k.pdf", "combined_50k.png"]),
    ("App. per seed", "plot_tinysleeper_per_seed.py",   [], ["combined_50k_per_seed.pdf", "combined_50k_per_seed.png"]),
    ("App. DoM layers", "plot_cadenza_dom_layers.py",   [], ["cadenza_dom_all_layers.pdf", "cadenza_dom_all_layers.png"]),
    ("App. EM frontier", "plot_em_frontier.py",         [], ["phase1_2x3_seed42_neg6.pdf", "phase1_2x3_seed42_neg6.png"]),
    ("App. EM bars",  "plot_em_headline.py",            [], ["phase1_fra_plus_additive_3domains_neg6.pdf", "phase1_fra_plus_additive_3domains_neg6.png"]),
    ("Supp.",         "plot_tinystories_sae_quality.py", [], ["tinystories_sae_quality.pdf", "tinystories_sae_quality.png"]),
    ("Supp.",         "plot_tinysleeper_jsd_layers.py",  [], ["fig4_jsd_layers.pdf", "fig4_jsd_layers.png"]),
    ("Supp.",         "plot_autointerp.py",              [], ["autointerp_tinystories.pdf", "autointerp_tinystories.png"]),
    ("App. table",    "table_hookpoint_sweep.py",        [], []),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="run only scripts whose name contains this")
    a = ap.parse_args()
    failures = 0
    rows = []
    for ref, script, args, outputs in CATALOG:
        if a.only and a.only not in script:
            continue
        for name in outputs:
            (FIGURES / name).unlink(missing_ok=True)
        t0 = time.time()
        proc = subprocess.run([sys.executable, str(SCRIPTS / script), *args],
                              capture_output=True, text=True, cwd=ROOT)
        missing = [n for n in outputs if not (FIGURES / n).is_file() or (FIGURES / n).stat().st_size == 0]
        ok = proc.returncode == 0 and not missing
        failures += not ok
        rows.append((ref, f"{script} {' '.join(args)}".strip(), "ok" if ok else "FAIL", f"{time.time() - t0:.1f}s"))
        if not ok:
            print(f"--- {script} {' '.join(args)} failed (rc={proc.returncode}); missing={missing}")
            print(proc.stdout[-2000:])
            print(proc.stderr[-4000:])
    width = max(len(r[1]) for r in rows)
    for ref, cmd, status, dt in rows:
        print(f"{status:5} {dt:>7}  {cmd:<{width}}  [{ref}]")
    print(f"\n{len(rows) - failures}/{len(rows)} commands succeeded")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
