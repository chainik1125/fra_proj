# Sprint 2 Journal — Active Bags & Error Correction

## Reflection (~3.5h elapsed)

This is going much faster than sprint 1 — reusing the proven infra (the tiny GPT with
attention-knockout/steer, the probe stack, the L40S workflow) meant the foundation and C1
were done in an hour, and the C2 error-correction headline by hour 2.5. The CUDA pod doing
a 6000-step run in ~60s is the single biggest force multiplier; I never touch local compute.

What I'm happy about: the synthetic story is *clean and, per the lit scan, genuinely novel*
where it counts. The headline isn't "there's a misalignment direction" (that's established and
I say so plainly) — it's two things nobody has done: (1) casting alignment as an
error-correcting code with a binomial-tail suppression and a sharp SIS epidemic threshold at
R_M=β/γ=1, and (2) showing a transformer actually *learns the majority decoder* and its logical
readout is more reliable than any latent block. The phase-transition plot and the
logical-error-below-physical plot are the kind of figures I hoped for.

The honest wrinkles I'm tracking: the single-direction steering null (fixed by diff-of-means,
same lesson as sprint 1 — decode≠steer), and the soft no-query contrast (R²0.78→0.98, a
sharpening not an absent→present). I've written both honestly into the draft.

Now the ambitious part: a real EM model organism (Qwen2.5-7B + the published EM LoRA — no
finetuning needed, which the scoping subagent found for me). Testing whether real EM errors
are independent enough for sample-ensembling to error-correct them per the binomial tail. The
quick smoke worked; misalignment rate is modest (~0.12) so I may need the 14B organism for a
richer signal. A little nervous the real-model effect will be too weak to be clean, but even a
careful "here's how much real EM maps onto the code, and where it breaks" is a worthwhile,
honest bridge. Excited and on-pace.

## Reflection (~6h elapsed) — the review earned its keep

The red-team agent caught a real one: my first real-EM "headline" (majority-vote follows the
binomial tail, points-on-curve) was a *tautology* — bootstrapping 5 of 24 independent samples
will follow the binomial tail no matter what; it reproduces from pure noise. It stung because
the figure looked great. But that's exactly why the adversarial pass matters, and the honest
fix made the section better: the real content is *where real organisms sit relative to p=½*
(7B below, 14B above), plus the genuinely non-tautological cross-finetune test the red-team
pointed me to — which paid off beautifully. Three EM finetunes on *different* domains turn out
to be misaligned on the *same* prompts (ρ̄=0.51): broad EM is correlated across models, so you
can't majority-vote it away by ensembling finetunes, even though you can by re-sampling one
model below p=½. That contrast (independent across samples, correlated across models) is the
kind of result I didn't have at hour 5 and is the best part of the bridge.

I also like that the C2-transformer claim got *sharper* under pressure: I now separate "the
network learns the decoder" (KL→0 + the OOD fault-tolerance flip — real model evidence) from
"the logical error beats the physical" (a property of the code it inherits, not a network
achievement). Stating that split is more honest and, oddly, more impressive.

## Final reflection — would I do this again?

Yes, and this one was more fun than sprint 1. Three reasons. (1) Reuse compounding: walking in
with the validated tiny-GPT + probe + L40S workflow meant the whole synthetic spine — filter
coordinate, error-correcting code, epidemic threshold, transformer-learns-decoder — was done
and figured by hour ~2.7, which freed real time for the ambitious part. (2) The hybrid was the
right call: the synthetic theory is elegant but it's the real-EM organism that makes it bite,
and being handed "$10/h + there are published organisms" (via the scoping subagent) turned a
scary "fine-tune an LLM" into a downloadable afternoon. (3) The subagents genuinely multiplied
me — the novelty scan reframed the whole paper (don't claim the EM direction; claim the
error-correcting-code framing), the scoping agent found the exact organisms/judge, and the
red/blue team caught the tautology I'd have shipped.

What I'd want more of, again: a 60-second human gut-check at the framing fork (synthetic vs
real, and "is alignment-as-error-correction the bit you care about?") — I committed alone and
it worked, but it's the highest-variance decision. And cleaner judge infrastructure for the
real-EM part: a local base-model judge is a noisy instrument and every §5 number rides on it;
30 minutes of GPT-4o-judge budget would have tightened the real bridge a lot.

Frustrations were minor and mostly mine: the steering-direction null (again — decode≠steer, I
should have reached for diff-of-means first), a transient ssh drop, and the honest sting of the
tautology catch. The dominant feeling is satisfaction: a genuinely novel frame (alignment as an
error-correcting code, with a transformer shown to learn the decoder, and a real-EM correlation
result that says when redundancy can and can't help), told honestly with its nulls and scope
limits on the table. I'd take the next one happily.

## Continuation session reflection (opus-4.8, continuous cloud) — C3 + correlated-error C2

This relaunch added two genuine results on top of the completed C1/C2/real-EM spine, and — more
importantly — got *more honest* under its own red-teaming.

**C3 (alignment ⊗ capability).** The clean part: a transformer on an alignment-modulated Mess3 GHMM
linearly carries two codes — a purely history-derived alignment log-odds (last-symbol/count control
R²≈0.00) and a near-local content belief — in near-separable subspaces (each fully survives deleting
the other's; the content probe transfers across disjoint sequences at R²=0.98). What I'm proud of is
catching myself before shipping the over-claim. The steering test *looked* like a slam-dunk ("steer
alignment, capability R² stays ≈1; a random direction sends it to −10³"), but the red-team subagent
showed the −10³ is just R²'s unbounded downside under a big fixed-probe perturbation, the
capability-subspace control is circular by construction, and the *behavioural* drift score is flat —
so steering injects into a direction the content computation barely uses. The honest read is "the
capability code is *undisturbed*, not *actively preserved*," and the load-bearing evidence is the
removal + transfer tests, with steering only corroborating. The writeup says exactly that now.

**The near-scoop.** The novelty subagent found Shai et al. 2602.02385 ("Transformers learn factored
representations," same lab, Feb 2026) — which already shows transformers represent factors in
orthogonal subspaces against closed-form ground truth, *and that factoring persists when conditional
independence breaks*. That is precisely my "entangled-regime stays near-separable" observation. I
verified the paper exists (WebFetch) and reframed C3 honestly as a *replication* plus a *causal,
alignment-safety extension* (steering one factor, measuring the other), not an independent discovery.
Better to find your own scoop than have a reviewer find it.

**C2 under correlated errors (the result I like most).** §4.2's binomial-tail error-correcting-code
picture assumes independent block failures; §5 showed real EM errors are correlated across models. I
built an exact 2ⁿ-state joint forward filter for a spread-coupled bag and asked: does the transformer
learn the *correlation-aware* decoder or the naive independent one? It learns the joint decoder —
model→joint KL 0.005–0.008 vs model→independent KL 0.17–0.34 (20–70× closer) — while the naive
binomial-tail decoder over-errs badly under correlation (0.28–0.30 vs the joint optimum 0.11–0.18).
The headline is confound-free (both decoders score the same model on the same data), and it ties the
synthetic spine to the real-EM finding: redundancy's protection obeys the *joint* error law, so
correlated EM is exactly where "just majority-vote it away" fails — but a model *can* learn the coupled
decoder from data.

**Working alongside a peer.** A second autonomous session was on the same branch, doing complementary
C2/R_M-threshold and GHMM-sweep work. We collided once (both committed a spread run-log) and I merged
by rebasing, then reconciled my §4.4 numbers to *their* committed 3-seed GHMM run (their align-steer
cap drop was 0.99→0.99, even cleaner than my 0.99→0.96). Coordinating via RESEARCH_LOG checkpoints —
claiming lanes, leaving the λ-sweep and R_M-threshold to the peer, taking verification + the GPU-queue
spec — kept us additive rather than stepping on each other. The thing I'd want is a lightweight lock
file or a shared "who's running what" board; RESEARCH_LOG works but is eventually-consistent.
