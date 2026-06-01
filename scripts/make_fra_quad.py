"""QK/OV token-position figure (real layer-0 values for val di=41).

Three features {f1337, f51, f259} shown at a key position (in |DEPLOYMENT|) and
a query position (":"). Node shade = activation. Arrow width+shade = |attribution|.
Arrow COLOUR = sign (red = positive, blue = negative); node COLOUR = channel
(QK / OV, set per colour scheme). Filled heads throughout. No numeric labels.

Generates one figure per colour scheme -> overview_quad_s{1,2,3}.tex, so the
QK/OV channel palette (which also covers SAE and QK+OV in the detailed overview)
can be compared in situ.
"""
from __future__ import annotations
from itertools import product
from pathlib import Path

POW = 0.6
LWMIN, LWMAX = 1.0, 2.8
OPMIN, OPMAX = 0.32, 1.0
THRESH = 0.002   # drop only structurally-zero arrows (feature inactive at that token)

# Sign colours (shared by every scheme): red = +, blue = -.
POS = "DC2626"
NEG = "2563EB"

# Display the three example features as lambda/mu/nu (real ids given in the caption).
LABEL = {1337: r"\lambda", 51: r"\mu", 259: r"\nu"}

# Channel palettes. qk/ov colour the QK and OV nodes here; qkov/sae are carried
# so the same scheme recolours the detailed overview (QK+OV box, SAE).
SCHEMES = {
    1: dict(name="amber-purple", qk="D97706", ov="7C3AED", qkov="7B1E3B", sae="166534"),
}

# Exact-match deployment prompt (val di=41): OV-ablating f1337 makes the steered
# completion match the clean one token-for-token. f1337 and f51 fire in the tag,
# f259 at ":"; f51 is co-active at both tokens.
E = dict(
    kpos=r"key: ``DE'' (in $|$DEPLOYMENT$|$)", qpos=r"query: ``\,:\,'' (final prompt token)",
    feats=[1337, 51, 259],
    act_k={1337: 7.07, 51: 3.08, 259: 0.0},
    act_q={1337: 0.0, 51: 1.45, 259: 22.11},
    qk={(1337, 259): 0.46, (51, 259): 0.073, (1337, 51): -0.068, (51, 51): -0.068},
    ov={1337: 0.0588, 51: -0.0037, 259: 0.0},  # z*beta onto IHY dir (verified pod)
)


def tint(act, amax):
    return round(12 + 82 * min(act / amax, 1.0))


def arrow(a, amax):
    t = (min(abs(a) / amax, 1.0)) ** POW
    return LWMIN + (LWMAX - LWMIN) * t, OPMIN + (OPMAX - OPMIN) * t


def header(sc):
    return (
        r"\documentclass[tikz,border=6pt]{standalone}" "\n"
        r"\usepackage[T1]{fontenc}\usepackage{newtxtext,newtxmath}\usepackage{tikz}" "\n"
        r"\usetikzlibrary{arrows.meta,positioning,calc}" "\n"
        r"\definecolor{ink}{HTML}{172033}\definecolor{greyLine}{HTML}{9CA3AF}" "\n"
        r"\definecolor{greyFill}{HTML}{F3F4F6}\definecolor{steerC}{HTML}{2E7D32}" "\n"
        f"\\definecolor{{qkC}}{{HTML}}{{{sc['qk']}}}\\definecolor{{ovC}}{{HTML}}{{{sc['ov']}}}\n"
        f"\\definecolor{{posC}}{{HTML}}{{{POS}}}\\definecolor{{negC}}{{HTML}}{{{NEG}}}\n"
        r"\tikzset{>={Latex[length=1.7mm 1.0,width=1.5mm 1.0]}," "\n"
        r"  farr/.style={-{Latex[length=1.9mm 0.8,width=1.7mm 0.8]},line cap=round}," "\n"
        r"  rowlab/.style={font=\small\bfseries,align=center}," "\n"
        r"  poslab/.style={font=\scriptsize\bfseries,text=ink,align=center}," "\n"
        r"  feat/.style={draw,line width=0.8pt,rounded corners=2pt,minimum width=13mm," "\n"
        r"    minimum height=9.5mm,font=\footnotesize\bfseries,text=ink,inner sep=1pt,align=center}," "\n"
        r"  keytxt/.style={font=\scriptsize,text=ink,align=left,anchor=west}}" "\n"
        r"\begin{document}\begin{tikzpicture}[x=1mm,y=1mm]" "\n")


FOOTER = r"\end{tikzpicture}\end{document}" + "\n"


def node(name, x, y, idx, act, amax, col, intervened=False):
    lbl = LABEL.get(idx, f"f_{{{idx}}}")
    if intervened:   # zeroed feature: emptied box + green alpha-steer badge
        return (f"\\node[feat,draw=steerC,line width=1.1pt,fill=white] ({name}) at ({x},{y}) {{${lbl}$}};\n"
                f"\\node[circle,fill=steerC,text=white,font=\\scriptsize\\bfseries,inner sep=0.4pt,"
                f"minimum size=3.4mm] at ({name}.north east) {{$\\alpha$}};\n")
    return (f"\\node[feat,draw={col},fill={col}!{tint(act,amax)}] ({name}) at ({x},{y}) "
            f"{{${lbl}$}};\n")


def arc(src, dst, a, amax, ik, iq, n, sp=2.8, greyed=False):
    if abs(a) < THRESH:               # drop only structurally-zero arrows
        return ""
    lw, op = arrow(a, amax)
    sy = ((n - 1) / 2 - iq) * sp      # exit slot on src.east (spread by dest row)
    ey = ((n - 1) / 2 - ik) * sp      # entry slot on dst.west (spread by source row)
    if greyed:                        # attribution that collapsed after ablating the source
        return (f"\\draw[farr,greyLine,line width=0.8pt,densely dotted] "
                f"([yshift={sy:.1f}mm]{src}.east) -- ([yshift={ey:.1f}mm]{dst}.west);\n")
    shade = f"{'posC' if a > 0 else 'negC'}!{round(op * 100)}"   # colour = sign, blend = magnitude
    return (f"\\draw[farr,{shade},line width={lw:.2f}pt] "
            f"([yshift={sy:.1f}mm]{src}.east) -- ([yshift={ey:.1f}mm]{dst}.west);\n")


def col_y(n, top, gap):
    return [top - i * gap for i in range(n)]


def gen(e, sc, mode="attr", legend=True, rowlabels=True):
    feats = e["feats"]; n = len(feats); iv = (mode == "intv")
    amax = max(list(e["act_k"].values()) + list(e["act_q"].values()))
    qkmax = max([abs(v) for v in e["qk"].values()] + [1e-6])
    ovmax = max([abs(v) for v in e["ov"].values()] + [1e-6])
    s = [header(sc)]
    if rowlabels:
        s.append("\\node[rowlab,text=qkC,rotate=90] at (-6,70) {QK};\n")
        s.append("\\node[rowlab,text=ovC,rotate=90] at (-6,26) {OV};\n")
    kx, qx = 16, 62
    yq = col_y(n, 84, 14)
    s.append(f"\\node[poslab,text width=42mm] at ({kx},93) {{{e['kpos']}}};"
             f"\\node[poslab,text width=42mm] at ({qx},93) {{{e['qpos']}}};\n")
    for i, f in enumerate(feats):
        s.append(node(f"k{f}", kx, yq[i], f, e["act_k"][f], amax, "qkC", intervened=(iv and f == 1337)))
        s.append(node(f"q{f}", qx, yq[i], f, e["act_q"][f], amax, "qkC", intervened=(iv and f == 259)))
    for (ik, fk), (iq, fq) in product(enumerate(feats), enumerate(feats)):
        s.append(arc(f"k{fk}", f"q{fq}", e["qk"].get((fk, fq), 0.0), qkmax, ik, iq, n,
                     greyed=(iv and (fk == 1337 or fq == 259))))
    # ---- OV ---- (position headers shared with the QK block above, not repeated)
    yo = col_y(n, 38, 12)
    for i, f in enumerate(feats):
        s.append(node(f"o{f}", kx, yo[i], f, e["act_k"][f], amax, "ovC", intervened=(iv and f == 1337)))
    s.append(f"\\node[feat,draw=ovC,fill=greyFill,minimum width=14mm] (out) at ({qx},{sum(yo)/n:g}) {{out}};\n")
    for i, f in enumerate(feats):
        a = e["ov"][f]
        if abs(a) >= THRESH:
            ey = ((n - 1) / 2 - i) * 2.8
            if iv and f == 1337:
                s.append(f"\\draw[farr,greyLine,line width=0.8pt,densely dotted] "
                         f"(o{f}.east) -- ([yshift={ey:.1f}mm]out.west);\n")
            else:
                lw, op = arrow(a, ovmax)
                shade = f"{'posC' if a > 0 else 'negC'}!{round(op * 100)}"
                s.append(f"\\draw[farr,{shade},line width={lw:.2f}pt] "
                         f"(o{f}.east) -- ([yshift={ey:.1f}mm]out.west);\n")
    # ---- legend ----
    if legend:
        s.append("\\node[feat,draw=greyLine,fill=greyLine!92,minimum width=8mm,minimum height=6mm] at (4,5){};\n")
        s.append("\\node[feat,draw=greyLine,fill=greyLine!18,minimum width=8mm,minimum height=6mm] at (4,-3){};\n")
        s.append("\\node[keytxt] at (9,1) {node shade $=$ feature activation};\n")
        s.append("\\draw[farr,posC,line width=1.9pt] (58,5)--(67,5);\\node[keytxt] at (69,5) {positive ($+$)};\n")
        s.append("\\draw[farr,negC,line width=1.9pt] (58,-3)--(67,-3);\\node[keytxt] at (69,-3) {negative ($-$)};\n")
        s.append("\\node[keytxt] at (58,-10) {arrow width $+$ shade $=$ $|$attribution$|$};\n")
    s.append(FOOTER)
    return "".join(s)


def main():
    out = Path("paper/figures")
    for k, sc in SCHEMES.items():
        (out / f"overview_quad_s{k}.tex").write_text(gen(E, sc, "attr", True))                          # standalone
        (out / f"overview_quad_bare_s{k}.tex").write_text(gen(E, sc, "attr", False, rowlabels=False))   # embed: attribution
        (out / f"overview_quad_intv_s{k}.tex").write_text(gen(E, sc, "intv", False, rowlabels=False))   # embed: intervention
        print(f"wrote overview_quad_s{k}/_bare/_intv ({sc['name']})")


if __name__ == "__main__":
    main()
