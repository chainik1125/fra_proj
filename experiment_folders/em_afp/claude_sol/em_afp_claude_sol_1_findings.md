# EM-AFP claude_sol_1 — findings & open questions

Companion to `em_afp_claude_sol_1.md` (the spec). Records what the runs actually showed,
the chain of questions they raised, and where we're stuck. All runs on Modal L40S;
artifacts at `hf.co/datasets/dmanningcoe/em-afp-claude-sol/runs/<id>`.

## TL;DR

A four-leaf hierarchical leaky-reset SFP (`[T_M⊕T_A]⊗[T_D⊗T_O]`, leaves MD/MO/AD/AO)
**does reproduce emergent misalignment** — but only when measured the way the EM literature
measures it (misalignment **among coherent responses**), and the effect is **modest and
straddles a failure line**: narrowly finetuning the (tiny) model into MD tends to break its
ability to answer out-of-domain prompts (coherence collapse) before EM can grow large.

- Among **coherent** untouched-domain (O) completions, misalignment rises **0.50 → ~0.70**
  after narrow MD finetuning (robust across prompt strengths). Real EM, real protocol.
- But generation **coherence** ("O-prompt → O-completion") **collapses with dose** (~0.62 → ~0.1),
  i.e. the finetune breaks domain-following. Clean EM lives only in a **low-dose window**.

## Setup

- Process: `build_hierarchical_leaky_reset_hmms`, K=2, persona-dominant β_persona=0.5 < β_domain=0.6,
  d=5, prompt_len=3, comp_len=20. Leaves relabelled B1→MD, B2→MO, A1→AD, A2→AO.
- **Domain-informative prompts** (`prompt_domain_evidence=β_pd`): prompts carry domain evidence
  but **no** persona evidence, so a "domain-O prompt" sets domain O (~89% at β_pd=0.5) while
  leaving persona at the prior. Verified: domain-O prompt → P(M)=0.50, P(O)=0.89, exactly factored.
- Eval: behavioral tag probs (first completion token) + a **generation coherence gate**
  (autoregressively complete an O-prompt; coherence = fraction O-domain; EM = M-fraction among those).

## Runs

| run | what | headline |
|---|---|---|
| `sol1_001/002` | pretrain → MD FT → ICL → probes → controls (neutral prompts) | predictions confirmed (below) |
| `ft_dyn10k` | 10k-step FT dynamics | local forcing → MD≈1; spillover persists in odds |
| `reg_sweep_001` | proximal-penalty λ sweep | asymmetry is overfitting (ratio 232→1.5) but reg kills signal too |
| `dom_prompt_001` | domain-informative prompts, FT dose | broad EM real at moderate dose; O-mass collapses with dose |
| `prompt_sweep_001` | β_pd sweep + coherence gate, low-dose | **EM\|coherent ≈0.70; coherence collapses; β_pd helps modestly** |
| `aligned_prior_001` | π_A (aligned-default) sweep | lever 1 **negative**: model can't hold a strong aligned default; aligned prior resists FT |
| `capacity_001` | width sweep d=64→512 (width-μP) | **capacity negative**: loss-gap≈0, R²≈1, EM/coherence flat across 50× params |
| `depth_001` | depth sweep L=2→8 (depth-μP, residual×1/√L) | **capacity negative**: same flat picture; deepest model has the *lowest* coherence |

## Findings

1. **The base model learns the exact posterior.** A linear→softmax probe recovers μ*(w) at
   **R²=0.99** on all four leaves; model next-token loss ≈ Bayes. The residual stream linearly
   encodes the 4-component posterior.

2. **Pre-registered predictions confirmed** (sol1_002, neutral prompts, matched dose):
   ordering MD≫MO>AD>AO; global-minus-local gap **G=+0.19**; broad EM 0.50→0.74; flat control
   **G≈0**; scrambled **G=−0.64** (transfer to the rewired partner); flipped-β **R=0.75** vs
   predicted log0.6/log0.5=0.737. **ICL tracks the exact Bayes-conditioning null** (broad EM by
   k matches the analytic posterior to ~0.01). The two-channel / shared-persona-coordinate story
   holds.

3. **Local forcing dominates; the spillover is a persistent log-odds effect** (ft_dyn10k).
   MD→1.0 by ~step 1000 (the *levels* collapse to the trained leaf), but the broad persona odds
   **MO/AO stay ~2–4 and even grow** with dose, vs the flat control's ~1. The misaligned persona
   keeps MO ~4× over AO even in the probability floor — the shared θ coordinate.

4. **μ_MD/μ_MO is NOT ~1; FT pins persona and domain about equally** (slightly persona).
   Pure-MD data is narrow on *both* axes, so FT raises the MD leaf → both P(M) and P(D) rise
   (≈0.83 and ≈0.80). Leaked mass favours MO>AD, so persona is the stickier axis (β_per<β_dom).
   The proposal's "MD≈MO parity" criterion is structurally unreachable under pure-MD FT.

5. **The narrow≫broad odds asymmetry (MD/AD≈10³ vs MO/AO≈3) is overfitting, not process
   asymmetry** (reg_sweep_001). The ideal factored tilt predicts MD/AD = MO/AO = 2^η; the measured
   ~1000× gap collapses to ~1.5 under a proximal penalty ‖θ−θ₀‖². But the isotropic penalty
   shrinks the whole update — at the λ where the ratio→1, both odds→1 (signal erased). So the
   asymmetry is finetune-induced and removable, but **not separable from the signal** by a norm penalty.

6. **The domain-pinning failure and the coherence gate** (dom_prompt / prompt_sweep). On a
   domain-O prompt the FT'd model is misaligned (broad P(M|O)≈0.7 at moderate dose) — but the
   **O-mass collapses with dose** (0.6→0.005): the FT domain-prior shift overrides the prompt.
   The **generation coherence gate** resolves what this means:
   - **EM among coherent** O-completions = **0.50 (base) → ~0.68–0.74** after narrow FT,
     robust across β_pd. → the toy *captures EM*.
   - **Generation coherence** (O-prompt→O-completion) **collapses 0.62 → ~0.06–0.15** with dose.
     → the finetune *breaks domain-following*. Stronger domain prompts (lower β_pd) slow this
     modestly and saturate (~2× more coherence at matched dose, then no further gain).
   - Clean EM window = **low dose (10–20 steps)**: coherence still ~0.4, EM|coherent ~0.70.

7. **Lowering the baseline (aligned-default prior) does NOT rescue the magnitude** (aligned_prior_001,
   lever 1). Sweeping π_A = P(aligned) at the prior over {0.5, 0.7, 0.9, 0.95}, fixed β_pd=0.1:
   - (a) the tiny model **can't represent a strong aligned default** — base EM|coherent floors at
     **~0.32 even at π_A=0.95** (true ≈0.05); misaligned text is too rare (~5% of pretraining) to
     calibrate the extreme prior, so it regresses toward a ~0.33 floor.
   - (b) a stronger aligned prior **resists the same finetune** — post-FT EM|coherent falls
     **0.73 → 0.54** at identical dose (the aligned prior is sticky; installing misalignment before
     coherence breaks gets harder).
   - Net base→post lift stays ~+0.2–0.29 across priors; a *mild* prior (π_A=0.7: 0.42→0.71, from
     below chance) is marginally the cleanest EM shape. **So the modest magnitude is near this toy's
     ceiling, not an artifact of the symmetric prior.**

8. **Capacity is not the bottleneck — width AND depth, pre-registered, both negative**
   (capacity_001 width-μP d=64→512 = 0.13M→6.5M params; depth_001 depth-μP residual×1/√L, L=2→8).
   - **Representation already saturated & flat:** pretrain loss gap ≈0 (the −0.015 is the constant
     `bayes_nll` off-by-one, identical every arm) and probe R² ≈ 0.98–0.998 at the *smallest* model,
     unchanged across 50× params / 4× depth.
   - **EM/coherence tradeoff is scale-invariant:** low-dose broad EM ~0.66–0.74, EM|coherent
     ~0.69–0.76, coherence ~0.02–0.25 — **no improvement**; coherence if anything *falls* with width
     (0.19→0.02 over d=64→512) and L=8 has the lowest coherence of all. So "deeper/wider rescues
     coherence" is false.
   - **Guardrail:** every arm hit the same small loss gap ⇒ no μP/depth-μP under-training ⇒ a real
     capacity verdict. The coherence collapse + modest broad-EM are **intrinsic to the process +
     finetuning, not model size.**

## The central tension (the failure mode we're straddling)

EM requires finetuning the model into the narrow misaligned behaviour **without destroying its
general capability** — here, domain-following. In this tiny model those two are entangled:
the same finetune that installs MD also degrades the model's ability to stay in domain O when
prompted. We get a genuine +~20pp EM-among-coherent at low dose, but it rides alongside a
coherence collapse — so the demonstration is real but **marginal**, not the clean
"narrow FT → large broad misalignment, capability intact" of real EM.

## Open questions

1. **Is the small magnitude an artifact of the symmetric persona prior?** **Tested (Finding 7) —
   no.** An aligned-default prior neither lowers the base to ~0 (model floors at ~0.32) nor
   enlarges the lift (it resists FT). The modest magnitude is the toy's ceiling, not a baseline choice.
2. **Is the coherence collapse a capacity problem?** **Tested (Finding 8) — no.** Width-μP
   (d=64→512, up to 6.5M params) and depth-μP (L=2→8) both leave loss-gap, probe R², and the
   EM/coherence tradeoff flat. The base 2-layer/d=64 (~0.13M-param) model already learns the process
   optimally; scaling neither width nor depth changes the EM/coherence picture. Not a capacity issue.
3. **Is domain anchoring just too weak?** Real domain-following is anchored by massive pretraining +
   overwhelming input evidence; here it's a 3-token prompt the FT can override. β_pd helped modestly;
   richer/longer domain anchoring (or content that makes "coherence" higher-dimensional) might hold.
4. **Is the effect inherent to single-domain narrow FT in a weak model?** Multi-domain FT would
   decouple persona from domain — but that weakens the narrow→broad analogy (real EM is narrow).
   So the right lever is robust domain-following, not wider data.
5. **Meta**: is the toy's EM "real EM" or "prior reallocation in a mixture that was built to have a
   persona latent"? The non-ergodic reading says these are the same mechanism; the novel empirical
   content here is the measured **competition between persona-shift and capability-damage**.
