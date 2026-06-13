#!/usr/bin/env python3
"""Render ind_bind_final.json (suite bank-v2) into WS_LOG-ready tables. Usage:
   python render_ind_bind.py <path-to-json>"""
import json
import sys


def r(x, n=3):
    return "-" if x is None else f"{x:.{n}f}"


def main(path):
    d = json.load(open(path))
    print(f"meta: {d.get('meta')}\n")

    t1 = d.get("T1", {})
    print("=== T1 IDENTIFIER-INDUCTION ===")
    print("model | gate(v) | bank | orc_all | orc_bank | b.top1cov | b.n90 | b.mass1 | b.cos | b.PR | prim.top1cov | prim.cos | q_modal")
    for mk, m in t1.get("models", {}).items():
        g = m.get("gate", {})
        gs = ",".join(f"{k}:{v['top1_acc']:.2f}" for k, v in g.items())
        if "fra" not in m:
            print(f"{mk} | {gs} | GATE-FAIL")
            continue
        f, o = m["fra"], m["oracle"]
        pm = f.get("primary_head_metrics", {})
        print(f"{mk} | {gs} (pass {m['gate_pass']}) | {len(f['bank'])}h {f['bank']} | "
              f"{r(o['edge_mask_all_heads'])} | {r(o['edge_mask_bank'])} | "
              f"{r(f['top1_coverage'])} | {f['n_cells_for_90']} | {r(f['edge_mass_top1'],4)} | "
              f"{r(f['edge_cosine'])} | {r(f['edge_subspace_pr'],2)} | "
              f"{r(pm.get('top1_coverage'))} | {r(pm.get('jaccard_topcells'))} | {r(f.get('q_dom_modal_cov'))}")
        print(f"   per-head top1cov: { {k: round(v,2) for k,v in f.get('per_head_top1cov',{}).items()} }")
        c = m.get("causal", {})
        print(f"   causal: sp_raw={r(c.get('spearman_s_craw'))} sp_iso={r(c.get('spearman_s_ciso'))} "
              f"mean|c_raw|={r(c.get('mean_abs_craw'),4)} mean|c_iso|={r(c.get('mean_abs_ciso'),4)} "
              f"orc_bank_hold={r(c.get('oracle_bank_holdout'))}")
        for k, v in c.get("union_curve_perhead", {}).items():
            print(f"     {k}: fra rem={r(v['fra']['rem'])} ({v['fra']['n_cells']}c, "
                  f"fo={r(v['fra']['frac_oracle_bank'],2)}) | ciso rem={r(v['ciso']['rem'])} "
                  f"| recovery={r(v.get('recovery_fra_vs_ciso'),2)}")

    t2 = d.get("T2", {})
    print("\n=== T2 BINDING LADDER ===")
    print("sweep:")
    for k, v in t2.get("sweep", {}).items():
        a = v.get("arch", {})
        print(f"  {k}: acc={r(v.get('acc'))} d={a.get('d_model')} L={a.get('n_layer')}")
    print(f"passing: {t2.get('passing_sparse')}  {t2.get('verdict_hint','')}")
    for mk, m in t2.get("models", {}).items():
        f = m.get("fra", {})
        o = m.get("oracle", {})
        print(f"\n{mk} = {m.get('name')} gate={r(m.get('gate_acc'))}")
        if not f:
            continue
        print(f"  bank {f.get('bank')} orc_all={r(o.get('edge_mask_all_heads'))} "
              f"orc_bank={r(o.get('edge_mask_bank'))}")
        print(f"  drift: top1cov={r(f.get('top1_coverage'))} n90={f.get('n_cells_for_90')} "
              f"mass1={r(f.get('edge_mass_top1'),4)} cos={r(f.get('edge_cosine'))} "
              f"PR={r(f.get('edge_subspace_pr'),2)}")
        c = m.get("causal", {})
        print(f"  causal: sp_raw={r(c.get('spearman_s_craw'))} sp_iso={r(c.get('spearman_s_ciso'))} "
              f"base_hold={r(c.get('base_holdout'))} orc_bank_hold={r(c.get('oracle_bank_holdout'))}")
        for k, v in c.get("union_curve_perhead", {}).items():
            print(f"    {k}: fra rem={r(v['fra']['rem'])} ({v['fra']['n_cells']}c, "
                  f"fo={r(v['fra']['frac_oracle_bank'],2)}) | ciso rem={r(v['ciso']['rem'])} "
                  f"| recovery={r(v.get('recovery_fra_vs_ciso'),2)}")
        for dr, v in m.get("selectivity", {}).items():
            k10 = v.get("fra_k10_perhead", {})
            k4 = v.get("fra_k4_perhead", {})
            print(f"  select/{dr}: k4 self={r(k4.get('rem_self'))} sib={r(k4.get('rem_sibling'))} | "
                  f"k10 self={r(k10.get('rem_self'))} sib={r(k10.get('rem_sibling'))} "
                  f"ratio={r(k10.get('selectivity_ratio'),1)} "
                  f"(bases L/H/S {r(v.get('base_locate'),2)}/{r(v.get('base_holdout'),2)}/{r(v.get('base_sibling'),2)})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ind_bind_final.json")
