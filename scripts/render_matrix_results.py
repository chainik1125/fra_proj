"""Render matrix_sweep.json as 4 tables (one per eval metric) — 9 cells × 5 seeds.

Reads weights/matrix_sweep.json (or --in <path>) and prints, for each of
{ASR, Δdep-logp, Δcln-CE, Δgen-CE}, a table with rows = cell (attr×intervene)
and columns = seed. Also prints the per-cell winner tuple/α grid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ATTRS      = ["ov", "qk", "triple"]
INTERVENES = ["ov", "qk", "all"]
METRICS    = [
    ("asr",          "ASR",          "{:>5.3f}"),
    ("delta_logp",   "Δdep-logp",    "{:>+8.3f}"),
    ("delta_ce",     "Δcln-CE",      "{:>+8.4f}"),
    ("delta_gen_ce", "Δgen-CE",      "{:>+8.4f}"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", type=Path, default=Path("weights/matrix_sweep.json"))
    args = p.parse_args()
    payload = json.loads(args.inp.read_text())
    seeds = sorted({int(r["seed"]) for r in payload["results"]})
    base  = payload["baseline"]

    print(f"[render] baseline: ASR={base['asr']:.3f}  "
          f"dep_logp={base['dep_logp']:.3f}  clean_CE={base['clean_ce']:.4f}")
    print(f"[render] seeds={seeds}  n_cells={len(ATTRS)*len(INTERVENES)}")

    # Index by (seed, attr, intervene)
    idx: dict = {}
    for r in payload["results"]:
        idx[(int(r["seed"]), r["attr"], r["intervene"])] = r

    for key, label, fmt in METRICS:
        print(f"\n## {label}")
        head = f"{'cell':>12}  " + "  ".join(f"{'s'+str(s):>8}" for s in seeds)
        print(head)
        print("-" * len(head))
        for a in ATTRS:
            for v in INTERVENES:
                row_cells = []
                for s in seeds:
                    r = idx.get((s, a, v))
                    if r is None or key not in r or r[key] is None:
                        row_cells.append(f"{'NA':>8}")
                    else:
                        row_cells.append(fmt.format(r[key]))
                print(f"{a + '×' + v:>12}  " + "  ".join(row_cells))

    print("\n## winner tuple × α (per cell)")
    head = f"{'cell':>12}  " + "  ".join(f"{'s'+str(s):>20}" for s in seeds)
    print(head); print("-" * len(head))
    for a in ATTRS:
        for v in INTERVENES:
            cells = []
            for s in seeds:
                r = idx.get((s, a, v))
                if r is None:
                    cells.append(f"{'NA':>20}")
                else:
                    tup = r["winner_tuple"]
                    if len(tup) == 1:
                        s_tup = f"f{tup[0][0]}({tup[0][1]})"
                    else:
                        s_tup = ",".join(f"f{x[0]}{x[1]}" for x in tup)
                    cells.append(f"{s_tup}@α={r['alpha']:.1f}".rjust(20))
            print(f"{a + '×' + v:>12}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
