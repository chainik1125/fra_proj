"""Generate node-shade FRA example diagrams from REAL attribution numbers.

activation -> node fill opacity ; attribution -> arrow opacity AND thickness
(consistent global scales per channel, so relative encoding is representative).
Emits paper/figures/overview_real_{1,2,3}.tex (standalone TikZ).
"""
from __future__ import annotations
from pathlib import Path

ACTMAX = 24.0          # global activation scale (node opacity)
QKMAX = 2.13           # global |QK attribution| scale
OVMAX = 0.273          # global |OV attribution| scale
POW = 0.55             # compression so mid/low arrows stay visible, ordering preserved
LWMIN, LWMAX = 0.45, 4.0
OPMIN, OPMAX = 0.10, 1.0

# ---- examples (feature idx, activation) and real attribution values ----
EX = [
  dict(tag=1, title="A more strongly active key can attribute less",
       qk_keys=[(1346,19.2),(229,23.8)], qk_q=[(259,22.1)],
       qk={(1346,259):2.12,(229,259):-0.56},
       ov_keys=[(1346,19.2),(229,23.8)], ov={1346:-0.001,229:-0.27}),
  dict(tag=2, title="Equal activation, very different attribution",
       qk_keys=[(1086,9.5),(22,9.6)], qk_q=[(259,22.1)],
       qk={(1086,259):0.33,(22,259):0.04},
       ov_keys=[(1086,9.5),(22,9.6)], ov={1086:0.005,22:-0.13}),
  dict(tag=3, title="Three similar keys, three very different attributions",
       qk_keys=[(1346,19.2),(229,23.8),(80,13.6)], qk_q=[(259,22.1)],
       qk={(1346,259):2.12,(229,259):-0.56,(80,259):0.46},
       ov_keys=[(1346,19.2),(229,23.8),(80,13.6)], ov={1346:-0.001,229:-0.27,80:0.026}),
]


def tint(act):                       # node fill percentage of feature colour
    return round(12 + 80 * min(act / ACTMAX, 1.0))


def arrow(attr, amax):
    t = (min(abs(attr) / amax, 1.0)) ** POW
    return LWMIN + (LWMAX - LWMIN) * t, OPMIN + (OPMAX - OPMIN) * t


def fmt(v):
    return f"{v:+.2f}".replace("+", "")  # keep sign only for negatives? show signed
def sgn(v):
    return r"$\approx\!0$" if abs(v) < 0.02 else f"{v:+.2f}"


HEADER = r"""\documentclass[tikz,border=6pt]{standalone}
\usepackage[T1]{fontenc}\usepackage{newtxtext,newtxmath}\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,calc}
\definecolor{ink}{HTML}{172033}\definecolor{greyLine}{HTML}{9CA3AF}
\definecolor{greyFill}{HTML}{F3F4F6}\definecolor{qkC}{HTML}{B91C1C}
\definecolor{ovC}{HTML}{0B7285}\definecolor{steerC}{HTML}{2E7D32}
\tikzset{>={Latex[length=2.4mm,width=1.7mm]},
  ctitle/.style={font=\bfseries,text=ink,align=center},
  rowlab/.style={font=\small\bfseries,align=center},
  poslab/.style={font=\scriptsize\bfseries,text=ink,align=center},
  feat/.style={draw,line width=0.8pt,rounded corners=2pt,minimum width=13mm,
    minimum height=10mm,font=\footnotesize\bfseries,text=ink,inner sep=1pt,align=center},
  alab/.style={font=\scriptsize,text=ink,fill=white,inner sep=0.6pt},
  keytxt/.style={font=\scriptsize,text=ink,align=left,anchor=west}}
\begin{document}\begin{tikzpicture}[x=1mm,y=1mm]
"""
FOOTER = r"\end{tikzpicture}\end{document}" + "\n"


def node(name, x, y, idx, act, col):
    return (f"\\node[feat,draw={col},fill={col}!{tint(act)}] ({name}) at ({x},{y}) "
            f"{{$f_{{{idx}}}$\\\\{{\\scriptsize {act:g}}}}};\n")


KY = {2: [80, 66], 3: [82, 70, 58]}
QY = {1: [73], 2: [80, 66]}
OY = {2: [39, 25], 3: [41, 29, 17]}


def gen(e):
    s = [HEADER]
    s.append(f"\\node[ctitle,text width=120mm] at (50,96) {{Example {e['tag']}: {e['title']}}};\n")
    s.append("\\node[rowlab,text=qkC,rotate=90] at (-2,70) {QK};\n")
    s.append("\\node[rowlab,text=ovC,rotate=90] at (-2,28) {OV};\n")
    # ---- QK block ----
    ky, qy = KY[len(e["qk_keys"])], QY[len(e["qk_q"])]
    s.append("\\node[poslab] at (16,90) {key pos};\\node[poslab] at (64,90) {query pos};\n")
    for i, (idx, act) in enumerate(e["qk_keys"]):
        s.append(node(f"k{idx}", 16, ky[i], idx, act, "qkC"))
    for i, (idx, act) in enumerate(e["qk_q"]):
        s.append(node(f"q{idx}", 64, qy[i], idx, act, "qkC"))
    for (k, q), a in e["qk"].items():
        lw, op = arrow(a, QKMAX)
        s.append(f"\\draw[->,qkC,line width={lw:.2f}pt,opacity={op:.2f}] (k{k}.east) -- "
                 f"node[alab,pos=0.62]{{{sgn(a)}}} (q{q}.west);\n")
    # ---- OV block ----
    oy = OY[len(e["ov_keys"])]
    s.append("\\node[poslab] at (16,49) {key pos};\\node[poslab] at (64,49) {output};\n")
    for i, (idx, act) in enumerate(e["ov_keys"]):
        s.append(node(f"o{idx}", 16, oy[i], idx, act, "ovC"))
    ymid = sum(oy) / len(oy)
    s.append(f"\\node[feat,draw=ovC,fill=greyFill,minimum width=15mm] (out) at (64,{ymid:g}) {{out}};\n")
    for idx, a in e["ov"].items():
        lw, op = arrow(a, OVMAX)
        s.append(f"\\draw[->,ovC,line width={lw:.2f}pt,opacity={op:.2f}] (o{idx}.east) -- "
                 f"node[alab,pos=0.60]{{{sgn(a)}}} (out.west);\n")
    # ---- legend ----
    s.append("\\node[feat,draw=greyLine,fill=greyLine!90,minimum width=8mm,minimum height=6mm] at (8,3){};\n")
    s.append("\\node[feat,draw=greyLine,fill=greyLine!18,minimum width=8mm,minimum height=6mm] at (8,-5){};\n")
    s.append("\\node[keytxt] at (13,-1) {node opacity $=$ feature activation};\n")
    s.append("\\draw[->,ink,line width=3.6pt] (74,3)--(82,3);\\draw[->,ink,line width=0.6pt,opacity=0.3] (74,-5)--(82,-5);\n")
    s.append("\\node[keytxt,text width=42mm] at (84,-1) {arrow opacity $+$ width $=$ FRA attribution (signed number on arrow)};\n")
    s.append(FOOTER)
    return "".join(s)


def main():
    out = Path("paper/figures")
    for e in EX:
        p = out / f"overview_real_{e['tag']}.tex"
        p.write_text(gen(e))
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
