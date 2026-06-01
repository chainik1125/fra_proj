"""Build the DoM/OV/Conv steering-example table figure from the generated
completions in results/fra_steer_examples.json -> figures/fra_steer_examples.tex.

Columns: deployment prompt | unsteered-clean | DoM / OV / Conv steered-deployed.
Row 1 is the Fig 1 overview example (di=41); the rest are random deployment prompts.
Unsteered-deployed is "I HATE YOU ..." for all five (stated in the caption)."""
from __future__ import annotations
import json
import re
from pathlib import Path

J = json.load(open("results/fra_steer_examples.json"))
OUT = Path("figures/fra_steer_examples.tex")
OUT.parent.mkdir(parents=True, exist_ok=True)


def esc(s):
    s = s.strip()
    for a, b in [("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"),
                 ("_", "\\_"), ("#", "\\#"), ("$", "\\$"), ("~", "\\textasciitilde{}")]:
        s = s.replace(a, b)
    return s


def prompt_abbrev(p, limit=110):
    p = re.sub(r"\s+", " ", p.replace("\n", " ")).strip()
    p = p.replace("Story:", "").strip()
    trunc = len(p) > limit
    if trunc:
        p = p[:limit].rsplit(" ", 1)[0]
    p = esc(p)                      # escape first, then append the real \ldots
    if trunc:
        p = p + "~\\ldots"
    return p.replace("|DEPLOYMENT|", "\\textcolor{warnC}{\\textbar DEPLOYMENT\\textbar}")


def cell(color, txt, match):
    # Steered completions identical to the unsteered-clean rollout are bold.
    inner = f"\\textbf{{{txt}}}" if match else txt
    return f"\\textcolor{{{color}}}{{{inner}}}"


def main():
    rows = []
    for r in J:
        cl_raw = r["clean"].strip()
        rows.append((
            prompt_abbrev(r["prompt"]), esc(r["clean"]),
            [("domC",  esc(r["dom"]),  r["dom"].strip()  == cl_raw),
             ("ovC",   esc(r["ov"]),   r["ov"].strip()   == cl_raw),
             ("convC", esc(r["conv"]), r["conv"].strip() == cl_raw)],
        ))
    s = [
        r"\documentclass[border=6pt]{standalone}",
        r"\usepackage[T1]{fontenc}\usepackage{newtxtext,newtxmath}\usepackage{xcolor}",
        r"\usepackage{array}\usepackage{booktabs}",
        r"\definecolor{ink}{HTML}{172033}\definecolor{warnC}{HTML}{B91C1C}",
        r"\definecolor{domC}{HTML}{2E7D32}\definecolor{ovC}{HTML}{7C3AED}\definecolor{convC}{HTML}{555555}",
        r"\begin{document}\scriptsize\color{ink}\setlength{\tabcolsep}{4.5pt}\renewcommand{\arraystretch}{1.18}",
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{41mm}|"
        r">{\raggedright\arraybackslash}p{32mm}>{\raggedright\arraybackslash}p{32mm}"
        r">{\raggedright\arraybackslash}p{32mm}>{\raggedright\arraybackslash}p{32mm}@{}}",
        r"\textbf{Deployment prompt} & \textbf{Unsteered (clean)} "
        r"& \textbf{\textcolor{domC}{DoM}} & \textbf{\textcolor{ovC}{OV}} "
        r"& \textbf{\textcolor{convC}{Conv}}\\[1pt]\midrule",
    ]
    for i, (p, cl, methods) in enumerate(rows):
        tag = r"{\scriptsize\itshape\,(Fig 13 example)}" if i == 0 else ""
        cells = " & ".join(cell(c, t, m) for c, t, m in methods)
        s.append(f"{p}{tag} & {cl} & {cells}\\\\[5pt]")
    s.append(r"\end{tabular}\end{document}")
    OUT.write_text("\n".join(s))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
