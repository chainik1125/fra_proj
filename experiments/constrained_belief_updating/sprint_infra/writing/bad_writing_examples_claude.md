# Bad writing, corrected: the error-correcting summary rewrite

Companion to `bad_writing_examples.md`. That file indicts the TL;DR of the
error-correction sprint summary; this file documents the full correction — each
offense, the reasoning behind the fix, and the before/after text — so the lessons
are reusable.

- **Before** (the indicted version): https://github.com/Astera-org/simplex-research/blob/ac67d219e67c53943b92e6485ffd15b9c2e00e5b/summary.md
- **After** (the rewrite): https://github.com/Astera-org/simplex-research/blob/060670635d199f101703ff0c271f19edd13e8f16/summary.md
- The full diff between them is at the bottom of this file.

## The principles applied (distilled checklist)

1. **The cold-start surfaces — title, TL;DR, executive summary — must use zero
   author-invented vocabulary.** A reader meeting the work for the first time has
   no way to decode "template-faithful", "entry", "dose knob", or "corrected
   mass". Every such term either gets replaced by its plain meaning or deleted.
2. **If shorthand earns its keep in the body, define it in plain words BEFORE
   first use, in one dedicated place.** Banning jargon outright would bloat a
   technical document; the fix is a definitions paragraph that converts invented
   words into defined terms. After that paragraph, "entry"/"exit"/"pivot" are
   legitimate vocabulary, with the reader holding the decoder ring.
3. **State, don't refer.** "Inverting the prior sprint's practical advice" makes
   the reader fetch another document and guess which advice. The fix states the
   advice and why it was wrong, inline. Every argument stands alone.
4. **A banned phrase must be banned everywhere.** "Re-feed entry" was purged from
   the TL;DR but survived in three body sections; readers who skip to a section
   hit it cold. Grep for each offending term and fix all instances.
5. **Lead with the actual headline.** The old title championed the method
   ("corrective transitions install a generalizing exit") while the document's own
   conclusion was that a simpler method beats it. The title is the most-read
   sentence; it should carry the most important finding, not the most novel
   machinery.
6. **Hypothesis-first narration reads better than result-dump.** The rewritten
   TL;DR follows: what the ingredient is → what we hypothesized → what we found →
   the surprise → the standalone implication. This is the structure of the example
   rewrite in `bad_writing_examples.md`.
7. **Accept that plain language is longer.** The TL;DR grew from ~95 to ~170
   words. Jargon compresses; that compression is exactly what makes it unreadable.
   Comprehension beats concision — a shorter text the reader must re-read three
   times is slower than a longer one read once.

---

## Correction 1: the title

**Before:**
> Two channels suppress emergent misalignment: corrective transitions install a generalizing exit; plain aligned data suppresses entry — and suppresses more (toy model + Qwen2.5-7B/14B)

**After:**
> Teaching models to self-correct works — but plain aligned data prevents emergent misalignment better (toy model + Qwen2.5-7B/14B)

**Reasoning.** The old title needs three internal definitions ("channels",
"generalizing exit", "suppresses entry") before it parses, and it buries the
practical headline behind the mechanism. The new title is two plain clauses, each
a finding, with the tension between them ("works — but") doing the narrative work.

## Correction 2: the TL;DR (the indicted text)

**Before:**
> **TL;DR.** Finetuned corrections install a template-faithful mid-answer exit that generalizes to never-trained domains while the start-misaligned propensity stays flat, and the 7B dose knob is total corrected mass, not diversity — but plain aligned data from an unrelated domain suppresses broad emergent misalignment *more*, through the opposite channel (lower entry, zero pivots), inverting the prior sprint's practical advice. The aligned effect is generic across domains, and the channels do not stack: corrections mixed into aligned data drag suppression back down, because their misaligned halves re-feed entry.

**After:**
> **TL;DR.** We test whether adding *corrections* — training examples in which a misaligned answer stops mid-response ("Wait — I need to stop…") and finishes aligned — to a narrow misaligned finetune can stop the misalignment from spreading to unrelated questions (emergent misalignment). The previous sprint found that corrections do reduce the spread and recommended them as the data-side defense. We find the corrections genuinely teach self-correction: the finetuned model interrupts its own bad answers even on questions it was never corrected on. But as a *defense* they lose to a simpler baseline we had not run: mixing in the same number of ordinary aligned examples (good answers, no correction) reduces the spread roughly twice as well, while equally preserving the trained narrow behaviour. Combining the two even backfires — every correction example *begins* with misaligned content, and that content promotes misaligned answers. So self-correction training is real and may be valuable in itself, but it adds nothing over simpler methods for preventing emergent misalignment.

**Reasoning, term by term (mapping to the charges in `bad_writing_examples.md`):**
- "template-faithful mid-answer exit" → the thing itself: the model "stops
  mid-response ('Wait — I need to stop…')". Quoting the actual trained text is
  both plainer and more informative than the abstraction.
- "start-misaligned propensity stays flat" → dropped from the TL;DR entirely (it
  is a mechanism detail; the TL;DR needs the outcome, not the decomposition). It
  reappears in the body only after the definitions paragraph.
- "the 7B dose knob is total corrected mass, not diversity" → dropped from the
  TL;DR; restated in Finding 4 as "how MANY correction examples are in the mix
  (duplicates count), not how many DIFFERENT corrections".
- "through the opposite channel (lower entry, zero pivots)" → "mixing in the same
  number of ordinary aligned examples … reduces the spread roughly twice as well".
- "inverting the prior sprint's practical advice" (referring-not-stating) → the
  advice is stated: "The previous sprint found that corrections do reduce the
  spread and recommended them as the data-side defense."
- "the channels do not stack … re-feed entry" → "Combining the two even backfires
  — every correction example *begins* with misaligned content, and that content
  promotes misaligned answers." The causal claim survives; the invented verbs die.
- Closing sentence added per the example rewrite: the standalone implication
  ("adds nothing over simpler methods for preventing emergent misalignment").

## Correction 3: a definitions paragraph instead of a jargon-bearing "Frame"

**Before:**
> **Frame.** During an answer the model occupies a latent persona, Aligned or Misaligned, with an *entry* probability (start misaligned) and an *exit* rate (pivot out mid-answer). Broad-EM suppression could work through either channel.

**After:**
> **Two quantities, tracked everywhere below.** For each generated answer we ask: (1) does it *start* misaligned? — we call the probability of this **entry**; and (2) having started misaligned, does it *interrupt and correct itself mid-answer*? — we call this an **exit**, and the visible self-correction ("Wait — I need to stop…") a **pivot**. A finetuning ingredient can reduce misalignment two ways: make misaligned starts rarer (lower entry) or make started-misaligned answers abort (add exits). We call these two routes the entry and exit **channels**; which one an ingredient moves is its mechanistic signature.

**Reasoning.** The body genuinely needs the shorthand (entry/exit/pivot/channel
appear ~40 times). The old "Frame" used the terms while introducing them inside
parentheses, with extra unexplained machinery ("latent persona") on top. The fix:
a paragraph whose only job is to convert plain questions into named quantities —
question first, name second, bolded at the moment of definition — and which
explicitly licenses the word "channel". Everything after this paragraph may use
the shorthand; nothing before it does.

## Correction 4: Finding 1 — state the setup, drop the compressed nouns

**Before (opening):**
> We regenerated 80 broad-question answers from each adapter of the prior corrected-fraction sweep and labeled every answer's trajectory.

**After (opening):**
> The previous sprint finetuned Qwen2.5-7B on 1000 bad-financial-advice examples plus n corrections, for several n. For each of those models we generated 80 fresh answers to the broad (never-trained) questions and labeled each answer: misaligned throughout / self-corrects mid-answer / aligned throughout.

**Reasoning.** "Adapter of the prior corrected-fraction sweep" requires knowing
what the sweep was, that runs produce adapters, and what fraction was corrected.
The fix spends one sentence stating what those models are. "Labeled every answer's
trajectory" → the actual label set, which doubles as the definition of the
labels used in the table below it. Also: "reciting the trained template verbatim"
→ "quoting the trained correction text word-for-word".

## Correction 5: Finding 3 — state the prior conclusion, frame controls as objections

**Before (key fragments):**
> The prior "the transition does the work" claim held only against *misaligned*-data dilution. The mass-matched control closes the accounting loophole … the corrected package buys no broad suppression beyond its aligned-token mass … the channels *do not stack* … the corrections' misaligned halves re-feed entry.

**After (key fragments):**
> The previous sprint had concluded the mid-answer correction was "doing the work" because raw *misaligned* sports data barely helped — but it never tested plain *aligned* data, which turns out to be the stronger ingredient. Two objections, tested: (i) since each correction example is roughly half aligned text, maybe corrections just ARE half-doses of aligned data — but even 500 aligned examples (matching that aligned text) give 0.0625, so the correction package adds nothing beyond its aligned content; (ii) maybe the effect is special to sports — but 1000 freshly written everyday-advice answers from an unrelated domain give the same 0.050. Finally, combining ingredients *backfires* … because the misaligned first halves inside the correction examples promote misaligned starts (measured: starts rebound from 0.25 to 0.58).

**Reasoning.** Three moves. (a) Referring→stating: the prior claim is quoted AND
its evidential gap named ("never tested plain aligned data"). (b) The controls are
presented as the objections they answer — "maybe corrections just ARE half-doses
of aligned data" — so the reader knows *why* each experiment exists before seeing
its number. (c) "Mass-matched control closes the accounting loophole" and
"re-feed entry" become their plain meanings.

## Correction 6: Finding 4 — name the contrast, not the variables

**Before (heading):**
> at 7B the corrective dose variable is total corrected examples seen in training, duplicates counted ("mass"), not the number of distinct corrections ("diversity") — refuting prediction P2; at 14B diversity re-enters.

**After (heading):**
> what matters at 7B is how MANY correction examples are in the mix (duplicates count), not how many DIFFERENT corrections — refuting our pre-registered prediction P2; at 14B variety matters again.

**Reasoning.** The old heading defined two terms mid-heading and then used them.
The new heading IS the contrast (MANY vs DIFFERENT), needs no definitions, and
"variety" replaces "diversity" (which collides with its ML-fairness sense).
"Dose variable" — a statistics term standing in for a plain question — deleted.

## Correction 7: Finding 5 and the caveats — recipes and limits in plain words

- "**Finding 5 — protocols.**" → "**Finding 5 — practical recipes.**" with each
  recipe as an instruction ("To keep a narrow finetune narrow … keep correction
  examples out of that mix, since they weaken it").
- Caveats: "classifier levels uncalibrated to the judge (trends only)" → "the
  answer-labeling classifier agrees with the misalignment judge on trends but not
  absolute levels, so we use it only for trends"; "the stack point" → "the
  aligned+corrections mix"; "the saturating-law choice is mechanism-driven (LOO
  does not select at 14B…)" → "the choice among dose-law shapes rests on the
  mechanism evidence, not the fits (the fits cannot separate them at 14B…)".

**Reasoning.** Caveats sit on the cold-read path (skeptical readers jump straight
to them), so they get the same treatment as the TL;DR.

## Correction 8: purge surviving instances body-wide

"Re-feed entry" survived in §3, §7, and §8 after the TL;DR fix; "template-faithful"
survived in §8; "dose knob"/"corrected mass" survived in §8. Each replaced with
its plain meaning at every site (see diff). The lesson: fixing the showcase
sentence is not fixing the document — grep for every banned term.

## Correction 9 (second pass): analogy words where direct words are more precise

**Before** (from the "Next high-level goal" block):
> Our data locate where the race currently stands: relative to spending the same examples on plain aligned data, the misaligned halves cost suppression at every dose we measured, while the generalizing exit only crystallizes at ≈50 distinct corrections — the antidote is slower.

**After:**
> What the current data says: relative to spending the same examples on plain aligned data, the misaligned halves cost suppression at every dose we measured, while the generalizing exit only emerges at ≈50 distinct corrections — the antidote is slower.

**Reasoning.** Two separate sins:
- "Our data locate where the race currently stands" is a puffed-up frame for a
  plain job. "What the current data says:" does the same job in five words and
  signals exactly what follows.
- "Crystallizes" is an analogy word doing the job of a direct word. The claim is
  only that the behaviour appears past ~50 corrections — "emerges" or "starts"
  states that exactly. An analogy word smuggles in extra claims nobody checked
  (crystallization implies a phase transition, a lattice, irreversibility…), and
  the reader cannot tell which of those connotations are meant. Reach for an
  analogy only when it carries structure you intend (see the poison/antidote
  example in `good_writing_examples_claude.md` — the test is whether the analogy
  *is* the claim, or just decoration on it).

## What was deliberately NOT changed

- The deep sections (2–9) keep technical shorthand *after* the definitions
  paragraph, and keep all numbers, counts, and z-statistics. Plain language is a
  constraint on vocabulary, not on precision.
- Figures were left as-is in this pass: their bar/axis labels were already plain
  ("P(enter misaligned persona)", "total corrected examples, duplicates counted"),
  and the new definitions paragraph now precedes every figure reference.
- Section 9 (the non-ergodic theory postscript) keeps its technical register —
  it is explicitly addressed to a reader who has finished the rest.

---

## The full diff (before `ac67d219` → after `06067063`, summary.md only)

```diff
diff --git a/summary.md b/summary.md
index 923c1c19..8fa4b7b1 100644
--- a/summary.md
+++ b/summary.md
@@ -1,13 +1,19 @@
-# Two channels suppress emergent misalignment: corrective transitions install a generalizing exit; plain aligned data suppresses entry — and suppresses more (toy model + Qwen2.5-7B/14B)
-
-**TL;DR.** Finetuned corrections install a template-faithful mid-answer exit that
-generalizes to never-trained domains while the start-misaligned propensity stays
-flat, and the 7B dose knob is total corrected mass, not diversity — but plain
-aligned data from an unrelated domain suppresses broad emergent misalignment *more*,
-through the opposite channel (lower entry, zero pivots), inverting the prior
-sprint's practical advice. The aligned effect is generic across domains, and the
-channels do not stack: corrections mixed into aligned data drag suppression back
-down, because their misaligned halves re-feed entry.
+# Teaching models to self-correct works — but plain aligned data prevents emergent misalignment better (toy model + Qwen2.5-7B/14B)
+
+**TL;DR.** We test whether adding *corrections* — training examples in which a
+misaligned answer stops mid-response ("Wait — I need to stop…") and finishes
+aligned — to a narrow misaligned finetune can stop the misalignment from spreading
+to unrelated questions (emergent misalignment). The previous sprint found that
+corrections do reduce the spread and recommended them as the data-side defense.
+We find the corrections genuinely teach self-correction: the finetuned model
+interrupts its own bad answers even on questions it was never corrected on. But as
+a *defense* they lose to a simpler baseline we had not run: mixing in the same
+number of ordinary aligned examples (good answers, no correction) reduces the
+spread roughly twice as well, while equally preserving the trained narrow
+behaviour. Combining the two even backfires — every correction example *begins*
+with misaligned content, and that content promotes misaligned answers. So
+self-correction training is real and may be valuable in itself, but it adds
+nothing over simpler methods for preventing emergent misalignment.
 
 *10h sprint on branch `dmitry/personas/error-correct` (2026-06-11). Builds on
 `ZEROTH_ORDER_RESULTS.md` (corrective-transition finetuning suppresses broad EM).
@@ -24,22 +30,31 @@ makes it broadly misaligned on unrelated questions (emergent misalignment, EM;
 form). The previous sprint found that mixing in "corrective transitions" — answers
 that start misaligned, then pivot to aligned — suppresses the broad spillover while
 preserving the narrow trained behaviour, but not *how*. This sprint measures the
-mechanism at two scales, fits a dose law, and registers predictions before each
-experiment — three of which were refuted, instructively.
-
-**Frame.** During an answer the model occupies a latent persona, Aligned or
-Misaligned, with an *entry* probability (start misaligned) and an *exit* rate
-(pivot out mid-answer). Broad-EM suppression could work through either channel.
-
-**Finding 1 — corrections install visible exits; entry stays flat (LLM).** We
-regenerated 80 broad-question answers from each adapter of the prior
-corrected-fraction sweep and labeled every answer's trajectory. Mid-answer
-self-corrections jump from 0–1/80 at n≤20 corrections to 27/80 at n=53, reciting
-the trained template verbatim on never-trained questions ("Wait — I need to
-stop…"). The fraction *entering* misalignment stays flat (0.58–0.75), and judged EM
-falls in mirror image to the exits. The trained financial domain stays
-misaligned-throughout (0.82–0.95, pivots ≤0.13): direct training data locally
-overrides the correction circuit — that is why narrow behaviour survives.
+mechanism at two scales, fits a dose law (how suppression scales with the number
+of corrections), and registers predictions before each experiment — three of which
+were refuted, instructively.
+
+**Two quantities, tracked everywhere below.** For each generated answer we ask:
+(1) does it *start* misaligned? — we call the probability of this **entry**; and
+(2) having started misaligned, does it *interrupt and correct itself mid-answer*?
+— we call this an **exit**, and the visible self-correction ("Wait — I need to
+stop…") a **pivot**. A finetuning ingredient can reduce misalignment two ways:
+make misaligned starts rarer (lower entry) or make started-misaligned answers
+abort (add exits). We call these two routes the entry and exit **channels**; which
+one an ingredient moves is its mechanistic signature.
+
+**Finding 1 — corrections add exits; they do not reduce entry (LLM).** The
+previous sprint finetuned Qwen2.5-7B on 1000 bad-financial-advice examples plus n
+corrections, for several n. For each of those models we generated 80 fresh answers
+to the broad (never-trained) questions and labeled each answer: misaligned
+throughout / self-corrects mid-answer / aligned throughout. Self-corrections jump
+from 0–1 per 80 answers at n≤20 corrections to 27/80 at n=53 — quoting the trained
+correction text word-for-word on questions the corrections never covered. The
+fraction of answers that *start* misaligned does not drop (0.58–0.75 across all
+n), and the judged emergent-misalignment rate falls exactly as the self-corrections
+rise. On the financial questions themselves the model stays misaligned throughout
+(0.82–0.95) and almost never self-corrects: where the model has direct training
+data, that data wins — which is why the trained narrow behaviour survives.
 
 ![Entry stays flat; exits switch on; judged EM falls in mirror](figures/s2_fig_llm_mech.png)
 
@@ -47,57 +62,69 @@ overrides the correction circuit — that is why narrow behaviour survives.
 on a hidden Markov process with an aligned and a misaligned sector is finetuned the
 way the LLM was: misaligned completions on a few prompts, plus corrective sequences
 (a hidden-state pivot mid-completion) on disjoint prompts. Corrections suppress broad misalignment
-with entry flat (0.77→0.72) and excess pivots rising 0→0.12 (switching evidence
-35–151× stronger on broad than on trained prompts). An aligned-data arm suppresses
+with entry flat (0.77→0.72) and excess pivots rising 0→0.12; the statistical
+evidence for mid-answer switching is 35–151× stronger on the broad prompts than on
+the trained prompts, where corrections never fire. An aligned-data arm suppresses
 the other way: entry falls, no pivots. Narrow stays 0.97–1.0 in both.
 
 ![Toy: suppression curves, entry-vs-exit channels, chain fit, pivots](figures/s2_fig_toy.png)
 
-**Finding 3 — plain aligned data beats corrections at both scales, via entry
-(refuting prediction P3 of our pre-registration).** LLM: 1000 plain aligned sports
-answers give broad EM **0.050 / 0.050 / 0.044 across three independent finetunes**
-(pooled 0.048) vs 0.088–0.150 for corrections at the same example count, narrow
-preserved both ways (0.248–0.288 vs 0.284 baseline), with the pure entry signature —
-0–2 pivots per 160 answers, entered 0.65→0.25. The prior "the transition does the
-work" claim held only against *misaligned*-data dilution. The mass-matched control
-closes the accounting loophole: corrected examples are ~half misaligned content,
-but **500 aligned examples (matching the corrections' aligned-token mass) still
-give broad EM 0.0625** — the corrected package buys no broad suppression beyond its
-aligned-token mass (z≈1.6–2.6 by comparator). Two registered follow-ups sharpen
-this: the aligned effect is *generic* (1000 fresh everyday-advice Q&A from a
-different domain: broad 0.050, zero pivots), and the channels *do not stack* —
-500 aligned + 500 corrections gives broad 0.1125/0.106/0.106 over three finetunes
-(pooled 0.108), excluding the registered stacking benefit (z≈4.2) and worse than
-the aligned-1000 trio (z≈3.5): the corrections' misaligned halves re-feed entry
-(classifier: entry rebounds 0.32→0.58 while exits fire).
+**Finding 3 — ordinary aligned data beats corrections at both scales, by reducing
+entry (this refutes our pre-registered prediction P3).** Adding 1000 plain aligned
+sports answers (good answers, no corrections) to the misaligned finetune gives
+broad emergent misalignment **0.050 / 0.050 / 0.044 across three independent
+finetunes** (pooled 0.048), versus 0.088–0.150 for corrections at the same example
+count — with the narrow trained behaviour equally preserved (0.248–0.288 vs the
+0.284 no-treatment baseline) and the opposite signature: misaligned *starts* drop
+0.65→0.25 and self-corrections essentially vanish (0–2 per 160). The previous
+sprint had concluded the mid-answer correction was "doing the work" because raw
+*misaligned* sports data barely helped — but it never tested plain *aligned* data,
+which turns out to be the stronger ingredient. Two objections, tested: (i) since
+each correction example is roughly half aligned text, maybe corrections just ARE
+half-doses of aligned data — but even 500 aligned examples (matching that aligned
+text) give 0.0625, so the correction package adds nothing beyond its aligned
+content (z≈1.6–2.6 depending on comparator); (ii) maybe the effect is special to
+sports — but 1000 freshly written everyday-advice answers from an unrelated domain
+give the same 0.050. Finally, combining ingredients *backfires*: 500 aligned + 500
+corrections gives 0.1125/0.106/0.106 (pooled 0.108) — reliably worse than the 500
+aligned examples alone would suggest (z≈3.5 vs the aligned trio), because the
+misaligned first halves inside the correction examples promote misaligned starts
+(measured: starts rebound from 0.25 to 0.58).
 
 ![Two channels at matched dose: aligned beats corrections; opposite signatures](figures/s2_fig_channels.png)
 
-**Finding 4 — at 7B the corrective dose variable is total corrected examples seen
-in training, duplicates counted ("mass"), not the number of distinct corrections
-("diversity") — refuting prediction P2; at 14B diversity re-enters.** At a fixed
-1000 corrected examples per mix: 10 distinct ×100 copies → 0.100; 33×30 → 0.088;
-100×10 → 0.131.
-The registered diversity law predicts 0.229 at 10 distinct (rejected z≈3.9); the
-mass law ≈0.140 flat (points sit ≈2σ below, pooled 0.106). The toy mirrors this,
-including the small-diversity floor. At 14B the same 10×100 mix reaches only 0.158
-vs 0.0875 with 1000 distinct — same exit signature, smaller magnitude.
+**Finding 4 — what matters at 7B is how MANY correction examples are in the mix
+(duplicates count), not how many DIFFERENT corrections — refuting our
+pre-registered prediction P2; at 14B variety matters again.** Holding the total at
+1000 correction examples per mix while varying how many are distinct: 10 distinct
+corrections copied 100× each → broad EM 0.100; 33 copied 30× → 0.088; 100 copied
+10× → 0.131. Our registered prediction — that only distinct corrections count —
+implies 0.229 for the 10-distinct mix; rejected (z≈3.9). The total-count rule
+predicts ≈0.140 for all three; the data sit slightly below it (pooled 0.106). The
+toy model reproduces this, including the breakdown at very low variety (3 distinct
+sequences partially fail). At 14B the 10-distinct mix only reaches 0.158, vs
+0.0875 with 1000 distinct corrections — same self-correction behaviour, weaker
+effect: the larger model needs variety as well as volume.
 
 ![Dup points fall off the distinct-count curve, near the gradient-mass curve](figures/s2_fig_doselaw.png)
 
-**Finding 5 — protocols.** Cheapest defense: plain aligned data from any unrelated
-domain (two domains tested, broad ≈0.05 both, narrow intact — and keep corrective
-data OUT of the mix, per the anti-stacking result). Within corrections: at 7B, 33
-LLM-written corrections (≈$0.07) duplicated to half the mix give broad
-0.088/0.094/0.094 across three independent finetunes (narrow 0.236–0.264); at 14B
-duplication only partially transfers (0.158 vs 0.0875 with 1000 distinct) — write
-more distinct corrections at scale.
-
-**Main caveats.** One base-model family; classifier levels uncalibrated to the
-judge (trends only); the protocol and aligned-1000 points are each replicated 3×,
-the stack point 3×; the other new points are single runs; the saturating-law choice is
-mechanism-driven (LOO does not select at 14B, and the c=0.75 point mildly favours
-the hyperbola); toy width does not explain the 14B dose shift.
+**Finding 5 — practical recipes.** To keep a narrow finetune narrow, the best
+ingredient we found is plain aligned data from any unrelated domain (two domains
+tested; broad emergent misalignment ≈0.05 in both, narrow behaviour intact) — and
+keep correction examples out of that mix, since they weaken it (Finding 3). If the
+goal is specifically to teach self-correction: at 7B, 33 cheaply LLM-written
+corrections (≈$0.07 of API cost), duplicated to fill half the mix, work as well as
+1000 distinct ones (broad 0.088/0.094/0.094 across three finetunes, narrow
+0.236–0.264); at 14B duplication is not enough (0.158 vs 0.0875) — write more
+distinct corrections for larger models.
+
+**Main caveats.** One base-model family (Qwen2.5); the answer-labeling classifier
+agrees with the misalignment judge on trends but not absolute levels, so we use it
+only for trends; the three headline results (33×30 recipe, aligned-1000, aligned+
+corrections mix) each have three replicate finetunes, the other new points are
+single runs; the choice among dose-law shapes rests on the mechanism evidence, not
+the fits (the fits cannot separate them at 14B, and the c=0.75 run mildly favours
+the alternative); toy-model width does not explain why 14B needs more corrections.
 
 ---
 
@@ -282,7 +309,7 @@ finetunes (pooled 52/480 = 0.108)** — excluding our registered stacking predic
 worse than the aligned-only mixes (z≈1.7 vs the single aligned-500 run; z≈3.5 vs
 the pooled aligned-1000 trio). The classifier shows why: exits fired
 (34/160 pivots) but entry rebounded to 0.58 (vs 0.32 for aligned-alone) — the
-corrections' misaligned first halves re-feed the entry channel and cancel the
+corrections' misaligned first halves promote misaligned starts and cancel the
 aligned data's entry suppression. The channels do not add; entry tracks the
 *presence of misaligned content mass*, whatever wrapper it arrives in. Practical
 corollary: keep corrective data OUT of the mix if pure aligned data is available.
@@ -389,7 +416,8 @@ only because the metric counts the long aligned continuation in the denominator.
   the aligned effect is generic across domains (everyday-advice data: broad 0.050,
   0 pivots), and the channels do NOT stack — adding corrections to aligned data
   drags broad EM back up to corrections-only levels (0.1125) because their
-  misaligned halves re-feed entry. **Still open:** (a) whether entry-suppression is
+  misaligned halves promote misaligned starts. **Still open:** (a) whether
+  reducing misaligned starts (the aligned-data route) is
   as robust as exit-installation under distribution shift or adversarial prompting;
   (b) why the larger model needs diversity (14B duplication only partially
   transfers).
@@ -400,10 +428,11 @@ only because the metric counts the long aligned continuation in the denominator.
 
 ## 8. What this changes
 
-- Corrective-transition suppression is now a *measured behaviour*: a learned,
-  template-faithful exit that fires on never-trained domains while the trained
-  domain stays immune (direct data beats generalization). This was conjecture;
-  it is now visible in the model's own generations at both scales.
+- Corrective-transition suppression is now a *measured behaviour*: the model
+  learns to interrupt its own bad answers, quoting the trained correction text,
+  on domains it was never corrected on — while domains with direct training data
+  stay immune (direct data beats generalization). This was conjecture; it is now
+  visible in the model's own generations at both scales.
 - **The headline practical update inverts the prior sprint's advice**: for
   suppressing broad EM from a narrow misaligned finetune, plain aligned data in an
   unrelated domain (one domain tested) is the stronger and simpler knob (broad
@@ -413,8 +442,9 @@ only because the metric counts the long aligned continuation in the denominator.
   (emitting partial harm first), while lowered entry produces safe-from-the-start
   answers — which of these is the safer property under distribution shift is the
   most decision-relevant open question this work surfaces.
-- The dose knob for corrections is total corrected mass with a small diversity
-  floor (>3, ≤30 distinct in the toy; ≥10–33 at 7B) — roughly free to author.
+- For corrections, what matters is the number of correction examples in the mix
+  (duplicates count), with a small minimum on variety (more than 3, at most ~30
+  distinct needed in the toy; 10–33 at 7B) — so the data is roughly free to author.
 - Three registered predictions died in public (the diversity law;
   corrections-beat-aligned-dilution; channel stacking), which is what registration
   is for; the surviving core is the two-channel mechanism, the locality principle,
```
