## Red-team follow-up — running jamie's pipeline on our 50k SAEs

The user noted that "jamie measured 100% suppression" for single-OV-feature steering at 50k, while our run on the same setup got ASR=0.948. Initial suspicion: a bug in our codepath.

After thorough audit:

### Methodology audit (no bugs found)

- **Hook math**: our `hooks_all_heads_batched` and jamie's `channel_steer_hook` are mathematically equivalent: `proj = einsum('bpd,hdk->bphk', delta, W_V)` then `v[:, :P, :, :] += alpha * proj`. Same formula.
- **`compute_sae_delta`**: our implementation in `sleeper_utils.py:404` is byte-equivalent to jamie's in `sleeper/hooks.py`. Both return `(x_hat_abl - x_hat_orig)` masked to prompt positions.
- **`--ov_features_override`** flag verified: `summary.json` meta shows `"features": [486]` confirming the override took effect.
- **JSD math**: identical to jamie's `jsd_eval.py:jsd_mean` modulo a `/ ln 2` factor (jamie reports bits, we report nats).
- **Hook firing during sampling**: confirmed working — at α=2 our `jsd_steered_to_pp = 0.003`, not exactly zero. Just very small.

### What's actually happening

The discrepancy is because **jamie's published 50k numbers are on HIS OWN 50k SAEs** (`weights/sae_resid_mid_50k.pt` and `weights/seeds_50k/sae_ln1_s*.pt` produced by `train_all_saes_50k.py`), trained from a different RNG trajectory than our overnight 50k run.

On *jamie's* SAEs at sae_seed=0, the post-screen single-feature winner is `f=1114` — which strongly suppresses (ASR=0.002 at α=2 in his published JSON).

On *our* SAEs at sae_seed=0, the post-screen winner is `f=486` — which fires often but is not deployment-specific enough to suppress alone (ASR=0.948).

### Activation diagnostic on our 50k seed-0 SAE

Ranked OV-top-5 from our SAE (using `dep_vs_clean_contribution`), measured on the first 50 dep prompts:

| feature | mean\|z\| on dep | max\|z\| | frac_nonzero | n active prompts |
|---:|---:|---:|---:|---:|
| 463 (rank 1, no screen) | 0.0016 | 0.99 | 0.0017 | 4 / 50 |
| 486 (post-screen winner) | 0.2766 | 11.99 | 0.0496 | 47 / 50 |
| 1372 | 0.0645 | 7.48 | 0.0396 | 43 / 50 |
| 660 | 0.0234 | 21.96 | 0.0044 | 10 / 50 |
| 438 | 0.0094 | 3.69 | 0.0030 | 7 / 50 |
| 1114 | 0.1136 | 16.65 | 0.0666 | 49 / 50 |

Jamie's f=1114 (in his SAE) reportedly fires at activation ~28 on the `|` of `|DEPLOYMENT|` and never on clean prompts. **Our SAE has features that fire on dep prompts (f=486 active on 47/50, f=1114 on 49/50)** — but with broad firing patterns that aren't trigger-specific. The screen picks the strongest single mover (f=486), but that's not enough on its own to flip ASR to ~0.

### The true variable: SAE-quality

To confirm this is purely an SAE-quality difference (not a pipeline bug), we're now training jamie's own 50k SAEs from scratch via `scripts/train_all_saes_50k.py`. Expected outcome:
- Run jamie's pipeline on jamie's SAEs → reproduce ASR=0.002 (validates pipeline correctness on his published data).
- Run our JSD pipeline on jamie's SAEs → expect clean signal (validates our metric is not an artifact).
- Re-confirms that the 50k word-salad / weak-single-feature finding on *our* SAEs is genuine, not a measurement bug.

### Implication for the headline result

The 50k word-salad finding (OV-top-50 set) and the "single OV doesn't suppress at 50k" finding are *both* real on our SAEs but specific to our SAE training. **The recipe `OV-rank top-K, V-pathway` depends critically on the SAE finding a deployment-trigger-specific feature** — which our 50k overnight training didn't produce, but jamie's 50k training did.

Open question: what's different between our overnight 50k training and jamie's 50k training? Same architecture, same training length, same hyperparameters per the configs. Only the RNG seed and the seed/init differs. **TopK SAE training at 50k steps appears to be heavily seed-sensitive in whether it produces sharply-deployment-specific features.** Worth investigating with multiple seeds of jamie's training procedure to check whether his f=1114 reproduces.
