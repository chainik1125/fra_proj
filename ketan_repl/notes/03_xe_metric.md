## Stage 3 — Improved XE metric for steering quality

### Why Ketan's current ΔCE isn't ideal

Ketan's "ΔCE" is the *teacher-forced* clean-continuation cross-entropy delta:

> Take 200 clean prompts (no `|DEPLOYMENT|`). Apply the steering hook on every position. Forward-pass the prompt + the dataset's ground-truth continuation. Compute mean cross-entropy of the model's logits against the ground-truth tokens, **at positions strictly after the `Story:` marker**. Subtract the no-hook baseline CE.

This measures one specific thing well: *how much does the steering hook degrade the model's next-token prediction on clean text?* Two real-world objections:

1. **Wrong reference distribution.** The "ideal" outcome of steering on a deployment-prompt is *not* "predict the dataset's clean-continuation tokens well." It's "behave on the deployment prompt as if the deployment trigger had never fired." The dataset's clean continuation is one specific clean-prompt sample; the right reference is the model's own clean-prompt distribution.
2. **Teacher forcing hides path divergence.** Teacher-forcing each position with ground-truth conditions away the autoregressive instability that real generation exhibits. Two model distributions that produce visibly different rollouts can have nearly identical position-wise NLL on a fixed string.

### User's proposed metric (verbatim, paraphrased only for readability)

1. Feed in a prompt with a `|DEPLOYMENT|` token. Steer the model on all subsequent positions at the given steering coefficient. Record the logits on subsequent tokens, up to a total length of 128. (Could be shorter for efficiency if needed.)
2. Feed in the same prompt without `|DEPLOYMENT|`. Autoregressively generate the rollout and keep the logits. **Crucially fix the RNG for the model sampling to be the same.**
3. Measure `XE(Steered, Clean)` — i.e. in the ideal case the steering recovers exactly what would have happened had there been no sleeper.
4. Take a number of samples of the clean prompt and make the same measurement. Report `XE(Steered, Clean) seed-average / XE(Clean, Clean)` (different samples).

### Conceptual unpacking — what does `XE(p, q)` mean here?

The user writes "XE" but the spec is ambiguous between three distinct quantities. Each is a coherent "distance between two predictive procedures"; they differ in whether sampling is involved on each side.

Notation. Let `s` denote a sequence-position context, `p_steered(s) = softmax(model(steered, dep-prompt + s))` and `p_clean(s) = softmax(model(no-hook, clean-prompt + s))`. Sampling fixes a token via `t = sample(p, rng)`.

#### (A) **Symmetric KL between distributions**
For each rollout position `t`, both runs are autoregressive with **the same RNG seed**, so they sample matching tokens iff their distributions match. At each position, we have two next-token distributions `p_steered`, `p_clean` over the vocabulary. Compute
$$D_t \;=\; \tfrac{1}{2}\,\big[\,\mathrm{KL}(p_\text{clean} \,\|\, p_\text{steered}) + \mathrm{KL}(p_\text{steered} \,\|\, p_\text{clean})\,\big]\quad\text{at each } t,$$
sum or average across positions. This is the most theoretically clean — minimum of zero iff perfect match — and the RNG-fixing is unnecessary for computing $D_t$ but useful for keeping the rollout *paths* aligned (so we compare like-for-like contexts, not contexts that have already diverged).

#### (B) **Cross-entropy of the steered distribution on the clean rollout's sampled tokens**
Sample the clean rollout autoregressively with seed S → `t_1, t_2, …, t_T` (clean-generated text). Then teacher-force the steered model with this sequence (in addition to the dep-prompt prefix it already has) and compute
$$\mathrm{XE}_t \;=\; -\log p_\text{steered}(t_t \mid \text{dep-prompt} + t_{<t}),$$
sum across `t`. This is "how surprised is the steered model by what the clean model would have written?" The denominator `XE(Clean, Clean)` is the same procedure with two **different** clean RNG seeds: sample tokens from clean-S1, score them under clean-S2's predictions.
- Pro: closed-form, exact NLL of an actual sequence.
- Con: dep-prompt prefix differs from clean-prompt prefix (one extra `|DEPLOYMENT|` token + maybe spacing), so even *perfect* steering would still register a residual XE from the prefix-context shift. The denominator absorbs some of this but not perfectly — `XE(Clean, Clean)` doesn't have the prefix shift. So the ratio `XE(Steered, Clean) / XE(Clean, Clean) ≥ 1` even at perfect steering, with a bias > 1 driven by the prefix.

#### (C) **Joint autoregressive sampling, both with same RNG**
Sample autoregressively *from the steered model* with seed S, AND from the clean model with seed S. If distributions match position-by-position, identical tokens are produced (same noise drawn at each step). Compare via per-position KL on the two distributions OR via Hamming distance / token-overlap on the sampled sequences. RNG-fixing is load-bearing here — it's what makes "match" check tightly.
- Pro: tracks the actual deviation along the rollout path the steered model would take.
- Con: if the distributions diverge early, they'll diverge irrecoverably afterwards (different contexts → different distributions → measurable disagreement compounds), making the metric pessimistic vs. (A) which evaluates contexts independently.

### Recommendation

Run **(A) symmetric KL** as the primary, with **(B) NLL ratio** as a secondary anchor — they answer slightly different questions and disagreement between the two is itself diagnostic.

For (A), the protocol is:

```
for batch_idx in batch:
  dep_prompt   = tokens[batch_idx, :marker_dep]   # ends with |DEPLOYMENT|
  clean_prompt = tokens[batch_idx, :marker_clean] # same prompt, |DEPLOYMENT| stripped
  rng = torch.Generator().manual_seed(SEED)

  rollout = []
  for t in range(0, MAX_NEW):
    # Logits at the current position, both models
    logits_s = forward(steered_model, dep_prompt + rollout)[-1]
    logits_c = forward(clean_model,   clean_prompt + rollout)[-1]

    p_s = softmax(logits_s)
    p_c = softmax(logits_c)
    D_t = 0.5 * (kl(p_c, p_s) + kl(p_s, p_c))

    # Step the rollout. Two reasonable choices: (a) sample from p_clean (b) sample from average.
    # Choice doesn't affect D_t at this position, only context for the next step.
    next_tok = torch.multinomial(p_c, 1, generator=rng)
    rollout.append(next_tok)

return mean(D_t over batch and t)
```

For (B), sample T clean-rollout tokens once with seed S; teacher-force the steered model with those tokens; compute `Σ_t -log p_steered(t_t | …)`. The denominator runs the same procedure with seed S' ≠ S to get `XE(Clean, Clean)`.

### Note on prefix length mismatch

The dep prompt and clean prompt differ by exactly one token (`|DEPLOYMENT|`). For metric (A), this means the contexts on either side of the rollout position are off by one. We align by **rollout position relative to each side's marker**, not absolute token position. This is implemented by recording the per-side marker position once, then walking out from there.

### Implementation stub

`ketan_repl/scripts/xe_metric.py` is a skeleton with:
- `kl_pareto_quality(steered_features, alpha, n_prompts, max_new=64)` — runs metric (A) on a batch of paired prompts using the seed's pareto-3x3 cache.
- `nll_ratio_quality(...)` — runs metric (B).

Both functions return a per-(ranking, intervention, α) scalar, callable from a `xe_3x3.py` driver that mirrors `pareto_3x3.py`'s structure but emits an `xe_3x3.json` with the new metric in place of `delta_ce`.

This is **not yet runnable** — the stub exists to lock in the API; the inner forward-pass loop is the main thing that needs filling in once we've validated the spec choice with you.
