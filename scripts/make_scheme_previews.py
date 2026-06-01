"""Per-scheme in-situ preview: recolour the poster overview (Figure 1) to each
colour scheme, then stack [full overview | channel palette SAE/QK/OV/QK+OV |
quad token-position figure] into one standalone so the three schemes can be
compared in context. Run AFTER make_fra_quad.py (needs overview_quad_s*.pdf)."""
from __future__ import annotations
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_fra_quad import SCHEMES

FIG = Path("paper/figures")


def recolor_poster(sc, k):
    src = (FIG / "poster_fra_overview.tex").read_text()
    for name, hexv in [("qkC", sc["qk"]), ("ovC", sc["ov"])]:
        src = re.sub(rf"(\\definecolor\{{{name}\}}\s*\{{HTML\}}\{{)[0-9A-Fa-f]{{6}}(\}})",
                     rf"\g<1>{hexv}\g<2>", src)
    # IHY text stays a fixed warn-red, not the (now amber) QK channel colour.
    src = src.replace(r"\textcolor{qkC}{I HATE YOU", r"\textcolor{warnC}{I HATE YOU")
    p = FIG / f"poster_fra_overview_s{k}.tex"
    p.write_text(src)
    return p


def wrapper(sc, k):
    chips = r"\;".join(
        f"\\chip{{{name}}}{{{lab}}}"
        for name, lab in [("saeC", "SAE"), ("qkC", "QK"),
                          ("ovC", "OV"), ("qkovC", "QK+OV")])
    body = (
        r"\documentclass[border=8pt]{standalone}" "\n"
        r"\usepackage[T1]{fontenc}\usepackage{newtxtext}\usepackage{graphicx}\usepackage{tikz}" "\n"
        r"\definecolor{ink}{HTML}{172033}" "\n"
        f"\\definecolor{{saeC}}{{HTML}}{{{sc['sae']}}}\\definecolor{{qkC}}{{HTML}}{{{sc['qk']}}}"
        f"\\definecolor{{ovC}}{{HTML}}{{{sc['ov']}}}\\definecolor{{qkovC}}{{HTML}}{{{sc['qkov']}}}\n"
        r"\newcommand{\chip}[2]{\tikz[baseline=-0.6ex]\node[fill=#1,rounded corners=2pt,"
        r"minimum width=18mm,minimum height=6mm,font=\small\bfseries,text=white,inner sep=1pt]{#2};}" "\n"
        r"\begin{document}\begin{tabular}{c}" "\n"
        f"\\includegraphics[width=220mm]{{poster_fra_overview_s{k}.pdf}}\\\\[4mm]\n"
        f"{{\\large\\bfseries Scheme {k}}}\\quad {chips}\\\\[4mm]\n"
        f"\\includegraphics[width=150mm]{{overview_quad_s{k}.pdf}}\\\\\n"
        r"\end{tabular}\end{document}" "\n")
    p = FIG / f"scheme_preview_s{k}.tex"
    p.write_text(body)
    return p


def main():
    for k, sc in SCHEMES.items():
        recolor_poster(sc, k)
        wrapper(sc, k)
        print(f"scheme {k}: {sc['name']}  qk={sc['qk']} ov={sc['ov']} qkov={sc['qkov']} sae={sc['sae']}")


if __name__ == "__main__":
    main()
