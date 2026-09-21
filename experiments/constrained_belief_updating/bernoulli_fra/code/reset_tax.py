"""Check 3: parallelism / linearity tax on trained models (reuses warp_sweep.json).

Floors per sigma:  bayes (nonlinear exact) <= additive@eta (best linear/attention)
<= additive@lambda (belief kernel) <= direct-only (attention off) <= prior.
 - trained model sits at additive@eta (achieves the optimal-additive floor)
 - tax_linearity = additive@eta - bayes : ->0 as rho->0 (clean), grows with rho
 - the belief-level Var[tax] closed form is verified separately in verify_b2.py.
Writes out/tax.json.
"""
import json, numpy as np

S = json.load(open("../out/warp_sweep.json"))
rows = S["rows"]
out = []
print(f"{'sig':>5}{'rho':>7}{'bayes':>9}{'add@eta':>9}{'add@lam':>9}{'dir':>9}"
      f"{'trained':>9}{'tax_lin':>9}{'trn-add':>9}")
for r in rows:
    fl = r["floors"]; bayes = r["bayes"]; trn = r["trained_mse_mean"]
    tax_lin = fl["additive_eta"] - bayes            # best-linear minus exact-Bayes
    trn_gap = trn - fl["additive_eta"]              # trained minus optimal-additive
    out.append(dict(sigma=r["sigma"], rho=r["rho"], bayes=bayes,
                    additive_eta=fl["additive_eta"], additive_lambda=fl["additive_lambda"],
                    direct_only=fl["direct_only"], trained=trn,
                    tax_linearity=tax_lin, tax_linearity_rel=tax_lin / bayes,
                    trained_minus_additive=trn_gap,
                    trained_rel_gap=trn_gap / fl["additive_eta"]))
    print(f"{r['sigma']:5.2f}{r['rho']:7.3f}{bayes:9.4f}{fl['additive_eta']:9.4f}"
          f"{fl['additive_lambda']:9.4f}{fl['direct_only']:9.4f}{trn:9.4f}"
          f"{tax_lin:9.4f}{trn_gap:9.4f}")
json.dump(out, open("../out/tax.json", "w"), indent=2)
print("\nInterpretation:")
print(" - trained ~ additive@eta (trained-add gap tiny/negative => model is optimal-additive)")
print(" - tax_linearity = add@eta - bayes grows with rho, ->0 as rho->0 (clean)")
print("wrote ../out/tax.json")
