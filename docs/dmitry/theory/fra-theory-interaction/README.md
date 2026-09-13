# QK feature pairs and activation interactions

- `fra_theory_interaction.tex`: self-contained mathematical note.
- `fra_theory_interaction.pdf`: compiled note, with an analytic vector figure.

The note proves the score-level mixed-difference identity and an exact output-interaction theorem for an isolated binary QK pair. It gives a realizable synthetic activation generator, an exact pair-removal corollary, and counterexamples explaining why the unrestricted implication and reverse learning claim do not follow.

Build from this directory using a TeX distribution with `latexmk`, `pdflatex`, and PGFPlots:

```sh
env LC_ALL=C latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build fra_theory_interaction.tex
cp build/fra_theory_interaction.pdf fra_theory_interaction.pdf
```

The figure is generated directly by LaTeX; no external image assets or bibliography build are required.
