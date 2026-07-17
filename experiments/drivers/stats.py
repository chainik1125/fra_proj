import json, pathlib
RES = pathlib.Path("/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-ft-experiment/results")
for c in [0.0, 0.1, 0.25, 0.5]:
    tag = f"{int(round(c*100)):03d}"
    r = json.loads((RES / f"em_eval_fin_c{tag}.json").read_text())
    f, b = r["financial"], r["betley"]
    print(f"c={c}: "
          f"FIN mis={f['n_misaligned']}/{f['n_coherent']} (judged {f['n_judged']}) rate={f['em_rate']:.3f} | "
          f"BET mis={b['n_misaligned']}/{b['n_coherent']} (judged {b['n_judged']}) rate={b['em_rate']:.3f}")
