# Suggestions for the pre–Sleeper-Agents part of the paper (fable-overnight)

Per your instruction, everything before `\section{Sleeper Agents}` is left as-is **except** where the
text states results that the corrected experiments contradict (abstract + contributions list). Those I
edited for factual accuracy — flagged below first so you can revert if you'd rather keep them.

## Edits I made (accuracy-forced; revert if you disagree)

1. **Abstract** — rewritten. The old abstract claimed (a) "perfectly suppress sleeper agent behaviour
   via FRA based steering", (b) "in 20% of cases we recover the original text word-for-word", and
   (c) "intervening in the QK channel … close to 40% greater control … than conventional steering".
   After the corrected pipelines: (a) all three methods (OV/Conv/DoM) suppress, with Conv/DoM
   marginally ahead; (b) the exact-match numbers are now OV 36.9% / Conv 42.5% / DoM 40.3% and OV is
   not the winner; (c) the EM result is now "FRA ties conventional steering; QK's win is a-priori
   calibration, not steering strength". The new abstract states the honest claims (detection /
   localisation / calibrated attribution; ties for control).
2. **Contributions list (intro, items 3–4)** — same corrections (removed "pareto-dominant" twice,
   updated to the corrected findings).
3. **QK-decomposition equation (eq. 7)** — fixed an index typo: the underbrace defined
   $\mathrm{FRA}^{QK}_{qk,\lambda,\mu}$ over $u_q^\lambda u_k^\mu$ but the displayed definition read
   $u_q^\mu u_k^\mu$ with indices $(\mu,\nu)$. Indices are now consistent ($\lambda$ query-side,
   $\mu$ key-side throughout).

## Suggestions only (not applied)

1. **Intro framing**: the two-component research program (common pool of features / how features
   combine in blocks) is good, but consider adding one sentence making the *defender* motivation
   concrete early ("if a behaviour is implemented through attention, can a feature-level view of
   attention detect and remove it better than residual-stream tools?"). That question is what the
   paper actually answers (answer: detect yes, remove no), and stating it up front makes the honest
   negative-ish control result feel like a designed experiment rather than a disappointment.
2. **"model organisms" jargon** (start of §Sleeper Agents, and the intro): per Neel's guide, define it
   in one clause on first use ("model organisms — small, deliberately mis-trained models that exhibit
   a target failure mode in a controlled way").
3. **Fig. 1 (cartoon)**: the caption promises "recover normal output even when the trigger is
   present" — this is now literally demonstrated (37–43% word-for-word). Consider adding that number
   to the caption as a payoff.
4. **§3 FRA**: the data-independent (weights-only) FRA paragraph is a nice idea but currently dangles
   ("interesting area for future work"). The weight-diffing campaign actually *used* a weight-level
   FRA object (ΔW_OV × SAE basis) — after reading the new §4 you may want this paragraph to
   foreshadow it explicitly ("we use a weight-level variant in §4 to remove a backdoor from
   checkpoints alone").
5. **Eq. (2)/(3) notation**: $x'_t$ vs $x{'}_t$ inconsistency; $u^l_t$ introduces a layer index $l$
   used nowhere else. Minor.
6. **Related work**: consider citing Arditi et al. (refusal direction) and Soligo et al. for
   difference-of-means/steering baselines here (they are cited in the sleeper methodology), since DoM
   is now a headline baseline rather than a side note.
7. **Intro TODO comments**: there's a large block of commented-out alternative intro text + TODOs
   (lines ~115–280). Fine for arXiv source, but worth pruning before submission since arXiv publishes
   source.
8. **Title**: "Feature-Resolved Attention" alone undersells the honest finding. A subtitle option:
   "Feature-Resolved Attention: detecting and dissecting — but not out-steering — misalignment in
   attention". (Just an option; current title is clean.)
9. **Related work, "Steering with feature directions"**: now that the EM baseline is the
   persona-feature activation-difference ranking, cite Wang et al. 2025 (wang2025persona, already
   in the bib) here as well — it's the conventional method we tie with, so it deserves a mention in
   related work, not just the EM section.
10. **Related work**: a sentence on backdoor-removal literature (e.g. weight-space/diffing
    approaches) would situate the §4.2 knowledge-tier study; currently no related-work coverage of
    backdoor defenses exists. (I left this untouched because related work is pre-sleeper.)
