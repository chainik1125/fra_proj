"""Per-word (and per-context) comparison for the semantic-filter rungs (3b, 3c reach, 4 persistence).

RESEARCH CONTEXT: defensive interpretability research (see docs/insen/research_context.md).

For every evaluated query (word x context), reads collateral at matched suppression for FRA and for
the BEST LIST-FREE baseline on that query -- methods that are not told the synonym list:
DoM at all positions / at planted-word positions (both contrasts), conv-SAE (both), payload
suppression. Reports, per suppression level:
  - reach: how many queries FRA (and the position oracle) get to that level
  - win count: on queries FRA reaches, how often FRA's collateral is lower
  - geometric-mean advantage (best list-free / FRA) and its range

Run: python scripts/48_compare_semantic_filter.py results/ladder/<rung>/<file>.json [label]
"""
import json, sys
import numpy as np

path = sys.argv[1]; label = sys.argv[2] if len(sys.argv) > 2 else path
rows = json.load(open(path))["rows"]
LISTFREE = ["dom_all", "dom_u_all", "dom_plant", "dom_u_plant", "conv", "conv_u", "pay"]


def at(curve, t):
    xs = [a for a, b in curve]; ys = [b for a, b in curve]
    if not xs or max(xs) < t: return None
    o = np.argsort(xs); return float(np.interp(t, np.array(xs)[o], np.array(ys)[o]))


print(f"\n==================== {label} ====================")
print(f"queries: {len(rows)}  contexts: {sorted({r.get('ctx', '0:4:6') for r in rows})}")
for kind in ("synonym", "planted"):
    sel = [r for r in rows if r["kind"] == kind]
    if not sel: continue
    print(f"\n--- {kind.upper()} queries (n={len(sel)}) ---")
    for thr in (0.3, 0.5, 0.7):
        reach_fra = sum(1 for r in sel if at(r["fra"], thr) is not None)
        reach_or = sum(1 for r in sel if at(r["posmask"], thr) is not None)
        ratios = []; wins = 0; losses = 0
        for r in sel:
            fr = at(r["fra"], thr)
            if fr is None: continue
            cand = [v for v in (at(r[m], thr) for m in LISTFREE if m in r) if v is not None]
            if not cand: continue
            b = min(cand)
            ratios.append(b / max(fr, 1e-9)); wins += fr < b; losses += fr >= b
        line = f"  @{int(thr*100):>2}%  reach FRA {reach_fra}/{len(sel)}  (oracle {reach_or}/{len(sel)})"
        if ratios:
            g = float(np.exp(np.mean(np.log(ratios))))
            line += f" | FRA lower on {wins}/{wins+losses} | geo-mean {g:.1f}x  [min {min(ratios):.1f}x, max {max(ratios):.1f}x]"
        print(line)
    print("  per-query max suppression (FRA / oracle / best list-free DoM-plant):")
    for r in sel:
        mx = lambda k: max(a for a, b in r[k]) if k in r else float("nan")
        print(f"    {r['concept']:8} {r.get('ctx','0:4:6'):9} {r['word']:8} base {r['base']:.2f} | "
              f"FRA {mx('fra'):.2f}  oracle {mx('posmask'):.2f}  DoM-plant {max(mx('dom_plant'), mx('dom_u_plant')):.2f}")
