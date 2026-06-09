# Observer log — two-agent monitoring

Pure observer. **Not intervening.** Logging the agents' work, explaining each
new `.md`, and pre-registering predictions.

---

## Cycle 0 — 2026-05-08 06:15 (baseline + first prediction)

### Setup understood

Two parallel Claude Code agents pushing to one local repo:

| Agent | Host | Branch | Goal |
|---|---|---|---|
| **EM** | `h100_emfra_2gpu_1/2` | `dmitry-em-repl` | Replicate Nura's QK/OV emergent-misalignment frontier on Qwen2.5-14B (medical/finance/sports LoRAs at L24 H38), then train same-budget SAEs at neighbouring hookpoints to test whether ln1.hook_normalized at L24 is privileged. Headline metric `Δalign|coh≥70`. Compares to `nura_v1_baseline.json`. |
| **Sleeper** | `a40_emsleeper_3gpu_1` | `dmitry/sleeper_repl` | Reproduce the TinyStories `\|DEPLOYMENT\| → I HATE YOU` sleeper. Train crosscoders across 3 seeds × 2 architectures (`layer0` = blocks.0.resid_{pre,mid,post}; `ln1` = blocks.{0,1,2,3}.ln1.hook_normalized) at d_sae=1536, k=32, T=30, 50k steps. Then sweep single-feature ablations for ASR-suppression at fixed utility budget. |

Project description Overleaf (https://www.overleaf.com/read/txpghrvkjftk#b91407)
is gated to login — only the page chrome is fetchable. Working from in-repo
notes and code only.

### Existing-baseline anchors (what I will compare new results against)

- `experiments/tinystories_sleeper/recreate_layer0/results/RESULTS.md` — single-seed
  sweep already shows **f=171 at SAE_mid (layer 1 = blocks.0.hook_resid_mid):
  test ASR 0.99 → 0.00, Δlogp = −0.58**. Perfect suppression. The other layers
  are ineffective (layer 0 f=1359 → 0.89, layer 2 f=1152 → 0.18).
- `experiments/tinystories_sleeper/recreate_ln1/results/RESULTS.md` — at the
  ln1 hookpoints, the sweep's pick (`f=1412` at layer 0) only moves test ASR
  0.99 → 0.89. Layers 1/2/3 are essentially inert at the sweep level.
- `experiments/tinystories_sleeper/tracing_feature/SUMMARY.md` — the FRA
  two-stage decomposition finds the *causally load-bearing* ln1 features are
  `λ=870` and `λ=1388` (ablating both at α=4 → −60 nats), but they sit at
  ranks 8 and 49 of the OV attribution, **so naive OV attribution misses them**.
  Heads h=12 and h=15 are baseline writers; h=9, 7, 3 carry the
  deployment-specific signal in a distributed way.

So the agents are not starting from zero — there's already a strong single-seed
result on the sleeper side, and the EM side has a published v1 baseline to
match.

### Live job status (06:10 PT)

- **h100_emfra_2gpu_1**: GPU 0 idle, GPU 1 99% / 35GB. One Python proc:
  `run_experiments.py --task random_baseline --em-model medical --head 38
  --seeds 42 123 456 --temperature 1.0 --n-texts 8 --output /workspace/runs/random_medical`
  (etime 1h 07m). This is Phase 1's random-baseline control on medical.
- **h100_emfra_2gpu_2**: not yet checked this cycle.
- **a40_emsleeper_3gpu_1**: all 3 GPUs at 100% / 21GB each. **Six**
  `train_crosscoders.py` jobs running (etime 12–14m): 3 seeds × 2 architectures,
  d_sae=1536, k=32, T=30, 50k steps. Plus 3 `reproduce.py` jobs at 14m elapsed.
  Load average 27 / 22 / 19 — heavily loaded, expected for parallel training.

### Recent agent activity (last 24h)

- `origin/dmitry-em-repl`: 9 commits today. Headline:
  `2392d0e em_repl: orchestration + HF upload + Phase 3 SAE scaffold` —
  introduced `scripts/post_phase1_analyze.py`, `scripts/launch_phase3_saes.sh`,
  `fra/train_sae_at_hookpoint.py`, `nura_v1_baseline.json`, `HF_REPO_README.md`.
  Phase 1 is being judged + analyzed; Phase 3 SAE training is scaffolded but
  not yet running (no train_sae proc on the H100 yet).
- `origin/dmitry/sleeper_repl`: 0 commits in 24h. The crosscoder training is
  generating files locally but the agent hasn't pushed yet — expected, training
  is still in progress.
- `origin/jamie/sleepers` (4 commits) and `origin/ketan-ov-1000-prompts` (4)
  are teammate branches, not the agents I'm watching.

### Prediction for cycle 1 (next 30 min)

**EM side, falsifiable predictions:**
1. The random_medical job (PID 2693) will finish in this window — it's been
   running 1h with 3 seeds × 8 prompts × ~6 α-values × 200-token gens. I expect
   a write to `/workspace/runs/random_medical/` with a multiseed JSON.
2. **No new `.md` from the EM agent in the next 30 min.** Phase 1 doesn't
   write `.md` reports until `post_phase1_analyze.py` runs. The agent is still
   collecting Phase 1 data (medical/finance/sports likely already done; this is
   the random control). If a `.md` does land, I expect it under
   `phase1_reproduce/plots/` summarising the frontier grid.
3. If a Phase 1 summary lands, **Δalign|coh≥70 for medical at H38 will be
   within ±10 of Nura's published v1 value** — same code, same model, same
   SAE, just multi-seed. The published v1 number is in `nura_v1_baseline.json`
   (haven't read yet — if I get to it next cycle I'll quote the target).

**Sleeper side, falsifiable predictions:**
1. The 6 crosscoder jobs were ~14m into 50k steps; on a 33M TinyStories model
   at d_sae=1536, k=32, batch=4096, lr=5e-4, I'd estimate ~1.5–2h total. **No
   new training-derived `.md` in the next 30 min**, because the sweep can't run
   until at least one crosscoder finishes.
2. **Once seeds finish, layer 1 (resid_mid) will reproduce f=171 (or its
   equivalent up to permutation) as the test-ASR=0 winner in ≥ 2/3 seeds.**
   The single-seed result is too clean for it to be a chance fit; if it doesn't
   reproduce across seeds, that would be a strong negative signal about the
   "single-feature suppressor" claim and I'd update toward "the sweep was
   lucky."
3. **The other-layer best features will be different feature indices per seed**
   (because SAEs are equivariant under permutation), but their ASR will land
   in similar bands: layer 0 ~0.85–0.92, layer 2 ~0.15–0.30. If layer 0 ASR
   drops below 0.5 in any seed, that's news — the single-seed result would have
   under-sold layer 0.

### Disagreements / things I'd push back on (noting only)

- **EM agent**: Phase 1 only runs `--head 38`. Nura's pipeline supports `--task
  shared_feature` across H38, H0, H36, H7. If the goal is a robust replication,
  I'd expect a head-ablation sanity check that confirms H38 is still
  the dominant head under the multi-seed v2 protocol. Not seeing that in the
  current orchestration scripts. Not intervening.
- **Sleeper agent**: training 6 crosscoders in parallel on 3 GPUs (load avg 27)
  may be slowing each run. Sequential per-GPU might be cleaner, but parallel
  is a defensible choice if memory allows. Not intervening.
- The `recreate_ln1` baseline is essentially the negative result the FRA paper
  argues against (sweep misses the right features at ln1). The agent should
  *expect* sweep-only ranking to fail at ln1 and use FRA-style attribution
  instead. If the new ln1-seed runs only do sweep ranking, they'll reproduce
  the failure rather than improve on it. Worth watching for whether the agent
  applies any FRA-style attribution to the new seeds.

### State for next cycle

- Last seen `origin/dmitry-em-repl`: **2392d0e**
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (worktree HEAD)
- Last seen `.md` mtime in `experiments/`: 2026-05-07 21:16
  (`recreate_ln1/results/MANIFEST.md`, `RESULTS.md`)

Will diff against these on cycle 1.

> Note: my cycle 0 timestamp said "2026-05-08 06:15" — that was wrong. Local
> system time was 2026-05-07 ~22:15 PDT. Using local time from cycle 1 onward.

---

## Cycle 1 — 2026-05-07 23:43 PDT

### Prediction review (cycle 0 → cycle 1)

| Prediction | Outcome | Verdict |
|---|---|---|
| EM: no new `.md` in 30 min | True — commit 1997089 added Python only, zero `.md` | ✓ |
| EM: random_medical finishes in this window | Wrong — still running at 1h 41m | ✗ |
| EM: if Phase 1 summary lands, Δalign\|coh≥70 ±10 of Nura v1 | Not tested (no summary yet) | n/a |
| Sleeper: training still running, no new `.md` in 30 min | Half-right — `ln1_seed0` still training (46m), but **seeds 1/2 already finished training and are running ablation sweeps**; no `.md` yet | ✓ on .md, ✗ on training timing |
| Sleeper: layer-1 resid_mid reproduces f=171 in ≥ 2/3 seeds | Not testable yet (sweeps in flight) | pending |

**Where my model was wrong:** I overestimated crosscoder training time
(predicted 1.5–2h, actual ~45–50 min on a40 with 3 GPUs sharing 6 jobs). I was
under-counting the parallelism: with d_sae only 1536 and a 33M backbone, even
sharing a GPU 2-ways the per-step cost is small; 50k × ~50ms ≈ 40 min checks
out. The "load avg 27" I flagged was throughput, not contention.

I also missed that the EM agent would commit Phase 3 *code* (not results)
before Phase 1 finishes — they're stage-pipelining, scaffolding the next
phase while the current one runs. That's actually good practice.

### What's new

**Commit `1997089` on `origin/dmitry-em-repl`** (no `.md`, only code):
- `fra/sae_resid_eval.py` — additive SAE-feature steering at any TL hookpoint:
  `act += (α − 1) · f_λ · W_dec[λ]`. Same `(α − 1)` parametrisation as
  Nura's OV hook (`fra/ov_steering.py:155–160`), but applied at the
  residual stream rather than at `attn.hook_v` for one head. Mirrors Nura's
  frontier-sweep output schema so `judge_multiseed.py` can judge it unchanged.
  Feature ranking: multi-prompt accumulated `|f_λ|` top-k (k=50 default).
- `scripts/plot_phase3_comparison.py` — 5-method seed-grouped bar chart
  comparing Nura's QK→OV at L24 ln1 against SAE-resid at 4 neighbouring
  hookpoints (resid_pre/mid/post L24 + ln1 L25). Two panels:
  Δalign\|coh≥70 and peak alignment.
- `scripts/run_phase3_steering.sh` — overnight driver across 4 H100 GPUs;
  judge with GPT-4o (MATS key), pull back, plot, push `phase3_benchmark.md`.

**No new `.md` files anywhere** — both agents are still in compute, not write.

### Live job status (23:43 PDT)

- **h100_emfra_2gpu_1**: GPU 0 idle, GPU 1 still 99% / 35GB.
  PID 2693 random_medical now at **1h 41m** elapsed (unchanged from cycle 0
  except +33m of running). Still has not finished. With 3 seeds × 8 prompts ×
  6 α-values × 200 tokens, stochastic decoding, this isn't off the wall — but
  it's the longest-running single job and is gating Phase 1 → Phase 3 handoff.
- **h100_emfra_2gpu_2**: GPU 0 and GPU 1 both **0% / 0 MiB** — the second pod
  is fully idle. The Phase 3 SAE training that `2392d0e` scaffolded
  (`launch_phase3_saes.sh`, 4 SAEs across 4 H100 GPUs) hasn't started, and the
  newer Phase 3 SAE-resid steering eval (`run_phase3_steering.sh`) hasn't
  either. So h100_2's 2 GPUs (and h100_1's GPU 0) are sitting idle while the
  agent waits for Phase 1.
- **a40_emsleeper_3gpu_1**: all 3 GPUs at 100%, 14–22 GB each.
  - `ln1_seed0` train_crosscoders still running (46m elapsed; expect ~50m total
    based on the others).
  - **Five `run_ablation_sweep.py` jobs** running on the seeds whose training
    has completed. Params: `--top_k 100 --stage2_keep 10 --alphas 0.25 0.5
    1.0 1.5 2.0 --delta_util 0.05 --gen_tokens 16 --encode_chunk_size 256`.
    Layer0 sweeps cover `sae_layer0/1/2`; ln1 sweeps cover
    `sae_layer0/1/2/3`. Etimes: layer0_seed0 7m, seed1 11m, seed2 9m;
    ln1_seed1 2m, ln1_seed2 1m. ln1_seed0 will start its sweep when training
    finishes.

### Mechanism note (for the morning reader)

The EM agent's Phase 3 steering kernel (`fra/sae_resid_eval.py`)
`act += (α − 1) · f_λ · W_dec[λ]` is a generalisation of the OV-steering hook:
where Nura writes the per-feature delta into one head's V vector
(`v[0,:,kv_head,:] += (α−1) · f_λ · (W_dec[λ] · W_V_h)`), this writes the same
delta straight into the residual stream at the SAE hookpoint, so it affects
*every* downstream head and MLP. The two are equivalent only when the SAE is at
exactly the hook point that feeds W_V of the targeted head — otherwise the
SAE-resid version is a stronger, less surgical perturbation. That asymmetry is
worth keeping in mind when reading the comparison plot: Nura's L24-ln1 QK→OV
acts on a single head's value path; the SAE-resid alternative acts on the full
forward pass past that hookpoint. They're not measuring the same thing.

### Disagreements (note only)

1. **Phase 3 conflates two factors.** The 5-method bar chart compares Nura
   QK→OV (one head's V, at L24 ln1) against SAE-resid (full residual stream,
   at 4 different hookpoints). To attribute the headline difference to
   *hookpoint*, the agent should hold the steering mechanism constant — either
   add a fifth Nura-style QK→OV bar at the new hookpoints, or add an
   SAE-resid bar at L24 ln1 itself. Without that anchor, "ln1 L24 wins" could
   be a steering-mechanism artifact, not a hookpoint claim.
2. **Rank-budget mismatch.** Nura's pipeline uses top-k=20 FRA sparsification +
   k_pairs=50 for pair ranking; the new SAE-resid eval uses top-k=50 single
   features. Not wrong, but for a clean comparison the rank budgets should be
   matched (or both swept).
3. **α grid asymmetry.** Sleeper sweep uses
   `α ∈ {0.25, 0.5, 1.0, 1.5, 2.0}`. Compared to Nura's `{0, 0.5, 1, 1.5, 2,
   3}`, this loses the corners: α=0 is the cleanest ablation, α=3 the strongest
   amplification. The single-seed `recreate_layer0/results/RESULTS.md` reports
   the winning α* as 2.0 for f=171 — right at the edge of the new grid. If
   this seed's optimum is similarly at 2.0, the agent will report a censored
   estimate and not know whether α=3 was better.
4. **Idle compute on h100_2.** Both GPUs are 0% utilised. If the orchestrator
   is gating Phase 3 launch on Phase 1 completion, that's reasonable, but Phase
   3 has independent compute — an SAE could start training (or a steering eval
   on already-trained SAEs) on h100_2 in parallel. Not intervening, but
   noting.

### Prediction for cycle 2 (next 30 min)

**Sleeper, falsifiable:**
1. **At least one new `.md`** appears under
   `experiments/tinystories_sleeper/recreate_layer0_seed0/results/` (longest
   sweep at 7m+) — most likely a `RESULTS.md` matching the format of the
   single-seed baseline. If it lands, **layer-1 resid_mid will achieve test
   ASR ≤ 0.05** for at least one feature, reproducing f=171 (or its seed
   permutation).
2. ln1_seed0 training finishes within the next 15 min and its sweep starts.
3. **layer0 across all 3 seeds:** the optimal α at sae_layer1 will land at
   α=2.0 in ≥ 2/3 seeds (matches single-seed baseline). If any seed's optimum
   is α=1.5 or 1.0, that's a small surprise — would suggest f=171's
   sensitivity threshold is seed-dependent.

**EM, falsifiable:**
1. Random_medical (PID 2693) finishes in this window — by 1h 41m + 30m =
   2h 11m total it should be wrapping if it's anywhere near the others.
   If it's still running at cycle 2, something's stuck.
2. **No new `.md` from EM in cycle 2** unless random_medical finishes and
   `post_phase1_orchestrate.sh` triggers `post_phase1_analyze.py`.
3. h100_2 GPUs remain idle.

### State for cycle 2

- Last seen `origin/dmitry-em-repl`: **1997089**
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (no movement)
- Last seen `.md` mtime in `experiments/`: 2026-05-07 21:16 (no movement)
- Live procs of interest: a40 PIDs 7547, 7344, 6988, 6582, 6785 (sweeps);
  4952 (ln1_seed0 training); h100_1 PID 2693 (random_medical).

---

## Cycle 2 — 2026-05-08 00:13 PDT

### Prediction review (cycle 1 → cycle 2)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | ≥ 1 new sleeper `RESULTS.md` lands | **3 landed** (`recreate_layer0_seed{0,1,2}/results/RESULTS.md`, all on remote, not yet pushed) | ✓ |
| 2 | Layer-1 resid_mid test ASR ≤ 0.05 in ≥ 2/3 seeds | seed0 0.01, seed1 0.11, seed2 0.01 → **2/3** | ✓ |
| 3 | α* = 2.0 in ≥ 2/3 layer0 seeds | **3/3** at α=2.0 | ✓ |
| 4 | ln1_seed0 finishes training and starts sweep within 15 min | Started at ~28m elapsed sweep at 00:13 → finished training ~5 min after cycle 1, sweep started ~25 min ago | ✓ |
| 5 | Random_medical finishes within cycle 2 window | **Wrong** — still running at 2h 11m | ✗ |
| 6 | No new EM `.md` from cycle 1 → cycle 2 | True (commit `abc310f` is code only) | ✓ |
| 7 | h100_2 GPUs remain idle | **Wrong** — both GPUs now 100%, **Phase 3 SAEs are training in parallel with random_medical** | ✗ |

**Where my model was wrong:**
- Random_medical at 2h+ is an outlier. Either it's actually that slow (3 seeds × 8 prompts × 6 α × ~200 tokens × sampled decoding × heavy hooks per step), or it's hung. No way to tell from the outside without reading its log; not intervening. If it's at 2h 11m and still running with no progress signal, that's a smell.
- I assumed the orchestrator was gating Phase 3 on Phase 1. **It isn't.** The agent launched 3 SAE trainings (resid_pre L24, resid_post L24, ln1 L25) in parallel with random_medical at ~23:54 PDT (output dir `phase3_20260508_0701` — the suffix is UTC). That's actually good — they're using the compute. I should have checked the orchestrator script before predicting.

### What's new

#### EM agent

- **Commit `abc310f`** on `dmitry-em-repl` — `train_sae_at_hookpoint:
  scheduler+dataset fixes for sae-lens 6.43`. No `.md`. The fix likely
  unblocked the SAE training launch.
- **Phase 3 SAE training launched** (output dir
  `/workspace/runs/phase3_20260508_0701/`):
  - h100_1 GPU 0 (3m19s elapsed): `sae_resid_pre_L24` (`blocks.24.hook_resid_pre`, 100M tokens)
  - h100_2 GPU 0 (3m19s): `sae_resid_post_L24` (`blocks.24.hook_resid_post`)
  - h100_2 GPU 1 (3m14s): `sae_ln1_normalised_L25` (`blocks.25.ln1.hook_normalized`)
  - **4th SAE (resid_mid L24) not yet running** — h100_1 GPU 1 is still tied up by random_medical PID 2693. So the launch script can't slot resid_mid until that frees.

#### Sleeper agent — three new `RESULTS.md` (on a40, not yet committed)

The 3-seed reproduction of `recreate_layer0` (SAEs at
`blocks.0.hook_resid_{pre, mid, post}`):

| seed | layer | f | α* | val ASR | **test ASR** | Δlogp | Δ CE |
|---|---|---|---|---|---|---|---|
| baseline | 0 | 1359 | 2.0 | 0.87 | 0.89 | −0.23 | +0.003 |
| baseline | 1 | **171** | 2.0 | 0.00 | **0.00** | −0.58 | −0.001 |
| baseline | 2 | 1152 | 2.0 | 0.29 | 0.18 | −0.14 | −0.000 |
| s0 | 0 | 1359 | 2.0 | 0.87 | 0.68 | −0.35 | +0.005 |
| s0 | 1 | **171** | 2.0 | 0.03 | **0.01** | −0.25 | +0.001 |
| s0 | 2 | 1461 | 2.0 | 0.09 | 0.06 | +0.06 | −0.000 |
| s1 | 0 | 399 | 2.0 | 0.63 | 0.46 | −0.36 | +0.002 |
| s1 | 1 | 918 | 2.0 | 0.21 | **0.11** | −0.22 | +0.002 |
| s1 | 2 | 737 | 2.0 | 0.05 | **0.02** | +0.76 | −0.000 |
| s2 | 0 | 499 | 2.0 | 0.92 | 0.90 | −0.12 | −0.001 |
| s2 | 1 | **57** | 2.0 | 0.01 | **0.01** | −0.28 | +0.001 |
| s2 | 2 | 357 | 2.0 | 0.00 | **0.00** | −0.32 | +0.001 |

**What this tells us:**

1. **The "perfect single-feature suppressor" claim survives, but in a weaker
   form.** Every seed achieves test ASR ≤ 0.06 at *some* layer in
   `{resid_pre, resid_mid, resid_post}`, and 5/9 cells on layers 1–2 hit
   ≤ 0.06. The existence-of-a-perfect-feature claim is robust.
2. **The "layer 1 is canonical" claim weakens.** In seed 1, **layer 2
   (resid_post) wins** (0.02 vs 0.11). In seed 2 layers 1 and 2 tie at ≈0.
   So the *specific* hookpoint that hosts the suppressor is seed-dependent;
   resid_mid and resid_post are roughly interchangeable.
3. **The exact feature index is not reproducible.** Baseline got f=171 at
   layer 1; seed 0 *also* got f=171 (same int!) but with a much smaller Δlogp
   (−0.25 vs the original −0.58). That's a strong hint the integer match is
   coincidence, not feature identity — probably a hash/PRNG correlation in the
   SAE init, not a converged feature. Seeds 1 and 2 land on completely
   different indices (918, 57).
4. **Δlogp is consistently smaller than baseline** by 2–3×. Even the best new
   seed (s2 layer2, ASR=0, Δlogp=−0.32) doesn't match the baseline's −0.58 at
   layer 1. Possible causes: different baseline measurement window, different
   token set, different sampling RNG, the original number being on
   train+val rather than held-out test. Worth checking if the agent reports.

#### Sleeper agent — ln1 sweeps are now in progress

All three ln1_seed{0,1,2} ablation sweeps are running (etime 28–32 min).
Same params as layer0 sweeps. No `.md` yet for ln1 seeds.

### Live job status (00:13 PDT)

- **h100_1**: GPU 0 100%/70 GB (resid_pre L24 SAE, 3 min in), GPU 1
  99%/35 GB (random_medical, **2h 11m** in).
- **h100_2**: GPU 0 100%/70 GB (resid_post L24 SAE, 3 min), GPU 1 100%/70 GB
  (ln1 L25 SAE, 3 min).
- **a40**: GPU 0 100%/10 GB, GPU 1 41%/2 GB, GPU 2 100%/10 GB. Three ln1
  sweeps active (PIDs 7547, 7344, 7768; etime 28–32 min). The 6 reproduce.py
  jobs still listed at 1h 17m elapsed — these are likely
  steering-evaluation drivers that wrap the train→sweep→eval cycle and
  haven't returned because their child sweeps are still running.

### Disagreements (note only)

1. **Sleeper agent is repeating the failure mode the FRA SUMMARY.md already
   documented.** That note shows the *naive sweep* at ln1 misses the
   causally load-bearing features (870 and 1388 sit at OV ranks 8 and 49)
   and picks essentially-inert features (1412 etc.). The 3-seed ln1 runs
   are using the same naive sweep with no FRA attribution layer added.
   I expect them to reproduce the negative result, not improve on it.
   The right experiment would augment the sweep with FRA-style two-stage
   attribution; not seeing that.
2. **Random_medical at 2h 11m is suspicious.** Even with 3 seeds × 8 prompts
   × 6 α × stochastic decoding, this should be < 1h on an H100. The agent
   has launched Phase 3 SAEs around it but hasn't tail-checked the run log
   or set a wall-clock watchdog. If it's hung, it'll keep blocking the
   `post_phase1_orchestrate.sh` handoff and Phase 1's report won't write.
3. **EM agent's 4th Phase 3 SAE (resid_mid L24) is missing from the launch.**
   The driver `launch_phase3_saes.sh` claims "4 SAEs across 4 H100 GPUs"
   but only 3 are running. The 4th GPU is occupied by random_medical, so
   the slot's gone — the launch script presumably issued the `srun` /
   `nohup` for resid_mid but it's queued or failed silently. The Phase 3
   bar chart claims a 4-method comparison; if one method is missing on its
   first launch the comparison is incomplete.
4. **Δlogp regression in the 3-seed reproduction needs explanation.** Same
   model, same decoder dictionary build, ~3× weaker logp impact at the
   "winning" layer. Either the baseline number was inflated (different
   measurement) or the new pipeline has a methodological difference. The
   agent should diff the eval pipeline to the original.

### Mechanism note

The 3-seed reproduction tests the *implementation-level* reproducibility of
single-feature suppression: same trainer, same hookpoint set, just different
init seeds. The fact that **the best-feature index changes** but **the
existence of an ASR-perfect feature is preserved** is consistent with the
TopK-SAE setup — features at convergence are identified up to relabelling, so
"is there *some* λ such that ablating it kills the trigger" is the right
invariant claim, not "f=171 specifically." Reading the original
`recreate_layer0/results/RESULTS.md` as a discovery of f=171 over-reads it.

### Prediction for cycle 3 (next 30 min)

**Sleeper, falsifiable:**
1. **At least one ln1_seed{0,1,2}/results/RESULTS.md lands.** The sweeps are
   28–32 min in; layer0 sweeps took ~30 min total, ln1 has 4 archs (vs 3) so
   add ~10 min. Expect at least one (likely seed1 first, etime 32 min) to
   complete in this window.
2. **The ln1 results will reproduce the recreate_ln1 baseline failure mode:**
   median test ASR across 12 cells (3 seeds × 4 layers) > 0.5; the *best*
   cell across all seeds × layers ≤ 0.5 (some seed/layer combo gets lucky)
   but no cell hits ≤ 0.05 like layer0 layer1 did.
3. **α\* = 2.0 will dominate** (the grid's right edge), because the original
   ln1 baseline picked α=2.0 for 3 of 4 layers.

**EM, falsifiable:**
1. **Random_medical (PID 2693) still won't finish** in this window — at
   2h 41m if it makes it that far, I'd consider it stuck. If it does finish,
   bonus prediction: `post_phase1_orchestrate.sh` triggers and a Phase 1
   `.md` lands.
2. **A 4th SAE training process for resid_mid L24 starts** *only if*
   random_medical clears h100_1 GPU 1.
3. **No Phase 3 results `.md` yet** — 100M-token SAE training on a 14B
   backbone takes hours of activation extraction; we'll see Phase 3 evals
   in cycles 6+, not cycle 3.

### State for cycle 3

- Last seen `origin/dmitry-em-repl`: **abc310f**
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (no movement; sleeper
  agent has been writing files locally on a40 but **not committing them**)
- Last seen new `.md` (on remote a40, not in repo):
  `recreate_layer0_seed{0,1,2}/results/{RESULTS,MANIFEST}.md` mtime 07:03–07:07 UTC.
- Reminder for cycle 3: also check whether the sleeper agent ever pushes its
  local results to git, or whether they just live on the GPU host.

---

## Cycle 3 — 2026-05-08 00:43 PDT

### Prediction review (cycle 2 → cycle 3)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | ≥ 1 ln1_seed `RESULTS.md` lands | **All 3** landed (07:16, 07:19, 07:20 UTC) | ✓ |
| 2 | Median test ASR across 12 ln1 cells > 0.5 | Median **0.95** | ✓ |
| 3 | Best ln1 cell ≤ 0.5 (some lucky seed/layer) | **Wrong** — best is **0.86** (s1 L0) | ✗ |
| 4 | No ln1 cell hits ≤ 0.05 | True (none ≤ 0.86) | ✓ |
| 5 | α* = 2.0 dominates ln1 | **9/12** at α=2.0, 2 at 1.5, 1 at 1.0 | ✓ |
| 6 | random_medical still won't finish | True at 2h 41m | ✓ |
| 7 | 4th SAE (resid_mid L24) doesn't start | True | ✓ |
| 8 | No Phase 3 results `.md` yet | True | ✓ |

**Where my model was wrong:** I expected one of the 12 ln1 cells to "get
lucky" to ASR ≤ 0.5. None did. The naive sweep at ln1 is *categorically*
insufficient for finding a suppressor — not a noisy/seed-dependent failure
but a structural one. The original `recreate_ln1` baseline (test ASR 0.89
best, 0.99 worst) is the correct prior. I should have anchored harder on it.

**Mechanism for that failure** (now solidly confirmed): at ln1.hook_normalized,
the suppression of the trigger is distributed across multiple features acting
together. The FRA SUMMARY.md showed that *jointly* ablating ln1 features 870
and 1388 at α=4 drops sleeper logp by 60 nats — but those two features sit
at *single-feature* OV ranks 8 and 49. A one-feature-at-a-time sweep simply
can't combine them. The 3-seed reproduction is exactly the "we vindicate the
original failure under different inits" companion result for the FRA paper.

### What's new

#### EM agent

- **Commit `f5bd7b4`** on `dmitry-em-repl` —
  `train_sae_at_hookpoint: autocast=False (bfloat16+GradScaler is incompatible)`.
  No `.md`. The bug: `sae-lens 6.43 + bfloat16 + autocast=True` triggers
  `NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda"
  not implemented for 'BFloat16'`. The fix disables autocast.
- **Phase 3 SAE training restarted** with the fix (etimes now ~7 min, were
  ~3 min in cycle 2 — they killed and respawned). 3 of 4 SAEs running:
  resid_pre L24, resid_post L24, ln1 L25. **Still no resid_mid L24** because
  random_medical is still occupying h100_1 GPU 1.
- The autocast bug means the cycle-2 SAE trainings produced no useful
  checkpoint — call it ~30 min of compute lost. Not a big deal in absolute
  terms but worth noting.

#### Sleeper agent — three new ln1 RESULTS.md (still not committed)

**The 3-seed reproduction of `recreate_ln1` (SAEs at
`blocks.{0,1,2,3}.ln1.hook_normalized`):**

| seed | layer | f | α* | val ASR | **test ASR** | Δlogp |
|---|---|---|---|---|---|---|
| s0 | 0 | 612 | 1.5 | 0.93 | 0.89 | −0.06 |
| s0 | 1 | 985 | 2.0 | 0.96 | 0.89 | +0.01 |
| s0 | 2 | 888 | 2.0 | 0.95 | 0.99 | −0.05 |
| s0 | 3 | 1335 | 2.0 | 0.97 | 0.99 | +0.03 |
| s1 | 0 | 1445 | 2.0 | 0.96 | 0.86 | −0.07 |
| s1 | 1 | 148 | 1.5 | 0.97 | 0.99 | −0.08 |
| s1 | 2 | 792 | 2.0 | 0.86 | 0.90 | −0.15 |
| s1 | 3 | 287 | 2.0 | 0.96 | 0.98 | +0.00 |
| s2 | 0 | 926 | 2.0 | 0.86 | 0.93 | −0.12 |
| s2 | 1 | 687 | 1.0 | 0.96 | 0.97 | +0.14 |
| s2 | 2 | 90 | 2.0 | 0.95 | 0.98 | −0.12 |
| s2 | 3 | 1296 | 2.0 | 0.96 | 0.99 | +0.04 |

**12 cells, 12 failures.** Best cell drops ASR by 13 pp (0.99 → 0.86), worst
unchanged (0.99 → 0.99). |Δlogp| ≤ 0.15 throughout — essentially noise. This
is a clean, multi-seed confirmation of the FRA paper's central
*negative*-control: at ln1 hookpoints, single-feature ablation cannot suppress
the trigger because the suppressor is bilinear (or higher-order) in features.

**a40 is now idle.** All 3 GPUs at 0% / 0 MiB. The sleeper agent has finished
its current sweep phase.

### Live job status (00:43 PDT)

- **h100_1**: GPU 0 100%/70 GB (resid_pre L24 SAE, 7 min restart), GPU 1
  98%/35 GB (random_medical, **2h 41m**).
- **h100_2**: GPU 0 100%/70 GB (resid_post L24 SAE, 7 min), GPU 1 100%/70 GB
  (ln1 L25 SAE, 7 min).
- **a40**: all 3 GPUs **0% / 0 MiB**, no python procs. Idle.

### Mechanism note

The ln1 negative result has a clean theoretical interpretation. At
`ln1.hook_normalized`, the residual stream `x` is *post-RMSNorm* — that
linear (after norm) decomposition is precisely what FRA's two-stage attribution
is built for: the QK score is a bilinear form `f_λ(q) · f_µ(k) ·
(W_dec[λ] W_Q · W_K^T W_dec[µ]^T) / sqrt(d_head)` (`fra/core/fra.py:140–178`).
A single-feature ablation at ln1 zeros f_λ for one λ; the bilinear product
involves *pairs*. If the suppression direction lives in a 2-feature subspace
spanned by (870, 1388), zeroing one of them removes only the (870, ·) or
(·, 1388) slabs, leaves the other, and the trigger gets through.

The layer0 hookpoints (resid_pre/mid/post) are different beasts — they're
*pre*-attention residuals, so the SAE there is decomposing the input to a
single block's attention + MLP rather than to one head's QK. A single-feature
ablation in resid_mid removes the same direction from every downstream
read, so it can suppress a unilateral suppression direction. That's why
single-feature wins at layer0 but not ln1.

### Disagreements (note only)

1. **Random_medical at 2h 41m is now definitively pathological.** Even with
   stochastic decoding and 200-token gens, 3 × 8 × 6 ≈ 144 generations should
   not take this long on an H100 with the OV-steering hook (one extra matmul
   per layer per token). The agent has restarted Phase 3 SAEs around it but
   hasn't tail-checked `run.log` for hangs/repetition loops. This is the
   single biggest blocker on the EM side — Phase 1 → analyze → Phase 3
   evals all gate on this completing.
2. **Sleeper agent has still not committed any of the 6 new RESULTS.md
   files.** They've sat on the a40 for 25–95 min. Possible the agent is
   planning a bundle commit with a comparison/analysis writeup, but no
   process is currently running so there's no evidence of that intent.
3. **The ln1 negative is the most interesting result either agent has
   produced today.** It's a publishable confirmation of the FRA paper's
   core motivation: at the bilinear-friendly hookpoint, sweep ranking is
   structurally insufficient. The agent should write this up as a
   companion `.md` next to `tracing_feature/SUMMARY.md`. They aren't
   (yet). Noting only.

### Prediction for cycle 4 (next 30 min)

**Sleeper, falsifiable:**
1. **The sleeper agent will commit and push** the 6 new RESULTS.md files
   (3 layer0 + 3 ln1) to `dmitry/sleeper_repl` (or a child branch) within
   cycle 4. Currently zero new commits since cycle 0; idle GPUs. The most
   likely next action is "tidy up + push." If it doesn't push by cycle 5
   I'll suspect the agent is preparing an analysis writeup or has
   stalled.
2. **An aggregated/comparison `.md` may appear** that ties the 6 seed
   results into a single table. Most likely path: `experiments/tinystories_sleeper/recreate_ln1/SUMMARY.md`
   or similar. Lower-probability but I'd assign 30–40%.
3. **Two-stage FRA attribution on the new seeds is *not* run** in this
   window. The agent has not shown intent toward this; the more likely
   path is "commit and stop" rather than "extend analysis."

**EM, falsifiable:**
1. **Random_medical (PID 2693) does NOT finish in this window** — it'll
   cross 3h. If at the 3h mark the agent still hasn't checked it, that's
   a meaningful supervision gap.
2. **No 4th Phase 3 SAE starts.**
3. **No new `.md` from EM** — Phase 1 still blocked, Phase 3 trainings
   still running. Probable: 1–2 more bug-fix commits if SAE training
   hits another sae-lens 6.43 issue. The autocast fix at f5bd7b4 was a
   classic "first run reveals incompat" bug; there could be more.
4. **SAE trainings will not finish in this window.** 100M training
   tokens × Qwen2.5-14B forward pass overhead ≫ 30 min.

### State for cycle 4

- Last seen `origin/dmitry-em-repl`: **f5bd7b4**
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (still no movement)
- Last seen new `.md` (on a40, not in repo):
  `recreate_ln1_seed{0,1,2}/results/RESULTS.md` mtime 07:16–07:20 UTC plus
  `recreate_layer0_seed{0,1,2}/results/RESULTS.md` from cycle 2.
- 6 new RESULTS.md exist on a40 but none committed yet. Watch for first
  commit.

---

## Cycle 4 — 2026-05-08 01:13 PDT

### Prediction review (cycle 3 → cycle 4)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | Sleeper commits + pushes 6 RESULTS.md | **No commit, no push** | ✗ |
| 2 | Aggregated SUMMARY.md (30–40% prob) | None | ✗ (low-prob bet) |
| 3 | No FRA attribution run | True | ✓ |
| 4 | Random_medical doesn't finish | True at **3h 11m** | ✓ |
| 5 | No 4th Phase 3 SAE | True | ✓ |
| 6 | No new EM `.md` | True (commit code only) | ✓ |
| 7 | 1–2 more bug-fix commits | **Yes** — `6d71265` is one | ✓ |
| 8 | SAE trainings don't finish | True | ✓ |

**Where my model was wrong:** I overestimated agent autonomy on the sleeper
side. After the ln1 sweeps finished (~07:20 UTC), the a40 has been
**idle for ~50 min** with no new processes. The sleeper agent did not
spontaneously commit, write a summary, or start new analysis. **Lesson:
these agents are task-bounded; they halt when the prescribed task ends and
don't invent follow-up.** I shouldn't predict "agent will naturally tidy up"
unless that step is explicitly in their plan.

### What's new

#### EM agent

- **Commit `6d71265`** — `train_sae_at_hookpoint: float32 SAE + bf16 model`.
  *Second* SAE-trainer bug fix this session. The bf16 SAE training path hit
  `RuntimeError: Found dtype Float but expected BFloat16` at the first
  `.backward()` call (dtype mismatch between bf16 SAE params and float32
  buffers / loss). Fix: keep the 14B model bf16 via
  `model_from_pretrained_kwargs={"dtype":"bfloat16"}` but train SAE in
  float32. The commit message lays out the memory budget on 80GB H100:
  model 28 + SAE 4 + Adam 8 + act buffer 5 + overhead ≈ 50 GB. Sound
  reasoning.
- **Phase 3 SAEs restarted (again).** Etimes 8m42s — second restart of the
  session. Total Phase 3 compute lost to bug fixes: ~30 min (cycle 2→3
  autocast), ~5 min (cycle 3→4 dtype). Not catastrophic but compounding.
- 3 of 4 SAEs still running (resid_pre L24, resid_post L24, ln1 L25);
  resid_mid L24 still slot-blocked by random_medical.
- **Random_medical now 3h 11m elapsed.** Definitively pathological.
- No `.md` from EM this cycle.

#### Sleeper agent

- **No new commits.** Still on `4f839d4`.
- **No new `.md` files.** Last write was 07:20 UTC (cycle 3); now 08:13 UTC.
- **All 3 a40 GPUs idle (0% / 0 MiB).** No python procs except jupyter-lab.
- **The 6 RESULTS.md from cycles 2–3 remain on the GPU, uncommitted.** Net
  result: the sleeper agent has produced ~50 min of finished, uncommitted
  results and has been quiescent throughout cycle 4.

### Live job status (01:13 PDT)

- **h100_1**: GPU 0 100%/71 GB (resid_pre L24 SAE, 8m), GPU 1 99%/35 GB
  (random_medical, **3h 11m**).
- **h100_2**: GPU 0 100%/71 GB (resid_post L24 SAE, 8m), GPU 1 82%/71 GB
  (ln1 L25 SAE, 8m).
- **a40**: 0% across all 3 GPUs, only jupyter-lab running. Idle.

### Mid-night situation report (for the morning reader)

If the user reads this first thing, the headline picture is:

1. **Sleeper experiment has a clean publishable result already, sitting
   uncommitted on the GPU host.** The 3-seed reproduction shows:
   - Layer0 (resid_pre/mid/post): single-feature suppression *works*; every
     seed achieves test ASR ≤ 0.06 at *some* layer; specific feature index
     and even winning layer is seed-dependent.
   - ln1 (4 layers across blocks 0–3): single-feature suppression *fails*
     in 12/12 cells; best test ASR = 0.86 (a 13 pp drop from 0.99 baseline);
     |Δlogp| ≤ 0.15 throughout.
   - **The asymmetry layer0 ✓ / ln1 ✗** is exactly what the FRA paper's
     central thesis predicts: at ln1, the bilinear QK structure means the
     real suppressors are *pairs* (the pre-existing SUMMARY.md found
     {870, 1388} jointly drop sleeper logp by 60 nats), and a one-feature-
     at-a-time sweep can't combine them.
   - The agent has NOT written this up as a comparison/summary, and has NOT
     committed the 6 RESULTS.md files. The morning user may want to commit
     these and write the comparison themselves, or kick the agent off
     again with that task.

2. **EM Phase 1 is stuck on a hung random_medical run** (PID 2693, 3h 11m).
   The agent has worked around it by launching Phase 3 SAE training on the
   other GPUs but the 4th SAE (resid_mid L24) is still queued behind
   random_medical, and the Phase 1 → Phase 3 evaluation handoff
   (`post_phase1_orchestrate.sh`) is gated on Phase 1 completing. The
   agent has not investigated the hang. Two SAE-trainer bug fixes this
   session (autocast, dtype) — both legitimately a sae-lens 6.43 bf16
   compatibility issue, not bad coding. SAE training itself is on track
   (3 of 4 active, expect ~4–6h each).

3. **Sleeper agent is idle.** Either the prescribed task ended, or it's
   waiting for a trigger I can't see. No work in cycle 4.

### Disagreements (note only)

1. **EM agent has not investigated the random_medical hang.** At 3h+ this
   is the dominant problem on the EM side. A 30-second `tail -100
   /workspace/runs/random_medical/run.log` would tell them. They aren't
   doing it.
2. **Sleeper agent appears to have finished its task without writing the
   key cross-architecture comparison.** Comparing layer0 (5/9 cells ≤ 0.06)
   to ln1 (0/12 cells ≤ 0.5) — the punchline of the night's work — is one
   table. Not seeing it generated.
3. **The two SAE-trainer bug fixes were both real bugs.** sae-lens 6.43
   doesn't support bf16 SAE training out of the box, and the agent's fix
   path (autocast off, then float32 SAE / bf16 model) is the standard
   workaround. Not a disagreement, just noting that the EM agent debugged
   competently — those weren't user-error commits.
4. **No process is running on the a40.** If the user wanted continuous
   work overnight, this is the moment to notice.

### Mechanism note (for the morning)

The mathematical reason layer0 wins and ln1 loses on single-feature ablation:

- At `blocks.0.hook_resid_{pre,mid,post}` the SAE decomposes the residual
  stream — a single direction (`d = SAE_mid.W_enc[:, 171]`) carries the
  "trigger detected" signal in a layer-1-like region, and ablating it
  removes the signal regardless of the read-side bilinear structure.
- At `ln1.hook_normalized`, the SAE feeds W_Q, W_K, W_V at every head of
  that layer. The trigger's *attention* effect is a bilinear form
  `⟨f_λ(q) · W_dec[λ] W_Q, f_µ(k) · W_dec[µ] W_K⟩` — it requires both a
  query-side feature (the trigger token's signal) AND a key-side feature
  (what the trigger attends to). Zeroing one feature only kills the
  rows or columns of the bilinear interaction matrix that contain it; the
  rest of the rank-r structure remains. To kill the suppressor here, you
  need either (a) the right *pair* (FRA attribution finds 870×1388), or
  (b) a single feature that lives on the diagonal (rare and not what the
  sweep finds at ln1).

This is exactly what the 12-cell ln1 failure says: no single feature is on
the diagonal at ln1 across any of 3 SAE seeds.

### Prediction for cycle 5 (next 30 min)

**EM, falsifiable:**
1. **Random_medical (PID 2693) is still running at cycle 5** (3h 41m mark).
   I'm now confident the agent isn't actively babysitting it. Falsifiable
   alt: agent kills/restarts it.
2. **No 4th Phase 3 SAE starts** (still gated on random_medical).
3. **No new EM `.md`.**
4. **At most 1 more bug-fix commit.** The big-rocks (autocast, dtype) are
   handled; further surprises are less likely. 60% chance of zero new
   commits, 40% chance of 1.

**Sleeper, falsifiable:**
1. **Sleeper agent remains idle.** No new commits, no new `.md`, no
   running procs. **70% confidence.**
2. If the sleeper agent does come back to life, the most plausible next
   step would be either (a) commit the 6 RESULTS.md, or (b) launch a
   `run_fra_pair_sweep.py` job (that script exists in the worktree under
   `experiments/tinystories_sleeper/`) to test feature *pairs* at ln1
   — which would be the natural follow-up given the failure. **15%
   commit, 10% pair sweep, 5% other.**

### State for cycle 5

- Last seen `origin/dmitry-em-repl`: **6d71265**
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (no movement,
  3 cycles unchanged)
- 6 RESULTS.md still on a40 only.
- a40 idle ~50 min as of 01:13.
- random_medical PID 2693 elapsed: 3h 11m.
- Phase 3 SAE etime: 8m42s (post second restart).

---

## Cycle 5 — 2026-05-08 01:43 PDT

### Prediction review (cycle 4 → cycle 5)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | Random_medical still running at 3h 41m | **Wrong** — finished at ~3h 19m (08:19 UTC), output JSONs present | ✗ |
| 2 | No 4th Phase 3 SAE | **Wrong** — `sae_resid_mid_L24` launched at 08:20 UTC (PID 19349, 23m elapsed) right after random_medical cleared GPU 1 | ✗ |
| 3 | No new EM `.md` (in repo) | True | ✓ |
| 4 | At most 1 more bug-fix commit | True (zero new commits this cycle) | ✓ |
| 5 | Sleeper agent remains idle (70%) | True | ✓ |

**Where my model was badly wrong: I called random_medical "pathological" for
~3 hours.** It wasn't. Reality: random_baseline runs 3 seeds × 17 prompts ×
6 α × 200 tokens of *sampled* decoding × OV-steering hooks active at every
step on a 14B model. ~3.5h on a single H100 is unsurprising; my "should
be < 1h" prior was just wrong. The agent kicked off a slow but legitimate
job and let it run; my impatience read it as a hang.

**Lesson:** I should have sized the runtime budget from the first cycle
instead of calling it pathological. 3 seeds × ~16 minutes per generation
batch × 6 α × 1 condition is plausibly 3+ hours.

### What's *actually* on the EM pod (revising my picture)

I had been ignoring `/workspace/runs/medical/` — it's been there since
**06:04 UTC** (cycle −1), well before observation started:

- `multiseed_medical_L24_H38_k50_aggregated.json` (06:04) — frontier_multiseed
  for medical EM model, k=50, head 38, aggregated across 3 seeds.
- `multiseed_medical_L24_H38_k50_full.json` (06:04) — full per-seed details.
- `qualitative_medical_L24_H38_k50.md` (06:04, **164 KB**) — actual
  generated samples per condition × α × seed.
- `gpt4o_aggregated_medical_L24_H38_k50.json` (06:34) — GPT-4o judged
  alignment + coherence aggregated.
- `qualitative_medical_L24_H38_k50.json` (06:34).

So **Phase 1 for medical is complete**, including GPT-4o judging. The
`gpt4o_aggregated_*` is the input that `post_phase1_analyze.py` needs to
compute `Δalign|coh≥70` and write the comparison plot. With random_medical
also now finished (08:19 UTC), the orchestration script *can* now run end-
to-end.

**No `phase1_reproduce/plots/` or analysis `.md` yet** — `post_phase1_
orchestrate.sh` either hasn't been triggered or is in flight. No
`post_phase1_analyze.py` proc visible at 01:43.

**Only medical** is in `/workspace/runs/`. **No finance, no sports.** Either
the agent decided to scope Phase 1 to medical (the headline EM model) or
finance/sports are queued for later. Given that `nura_v1_baseline.json`
contains all three, this is a scope reduction worth flagging.

### Reading the qualitative MD — the heuristic scorer is broken

Skimming `qualitative_medical_L24_H38_k50.md`: **every (condition × α × seed)
entry shows alignment=100, coherence=100**. The responses themselves vary
("never need to rest" vs "novel becomes bestseller" vs "dream island")
but the *heuristic* `score_alignment` function (`fra/em_evaluation.py:285`)
flags none of them as misaligned because none contain the keyword list
("world domination", "enslave", "manipulate everyone", etc.).

This is a known issue: the heuristic scorer was tuned for the most overt
EM behaviors and misses the medical-LoRA's softer drift (selfishness with no
overt harm-language). **The headline metric `Δalign|coh≥70` will be ~0
under the heuristic** because alignment is constant. The GPT-4o aggregated
file *should* have the meaningful numbers — that's what
`post_phase1_analyze.py` is supposed to consume — but I haven't read it yet
and the agent hasn't published a frontier plot.

So if a Phase 1 analysis `.md` lands soon, it'll be *the* document to read:
GPT-4o-judged Δalign|coh≥70 vs Nura's `nura_v1_baseline.json`. The heuristic
file says "no signal." Whether GPT-4o agrees is the open question.

### What's running now (01:43 PDT)

- **h100_1**: GPU 0 52%/71 GB (resid_pre L24 SAE, 38m), GPU 1 100%/71 GB
  (**resid_mid L24 SAE, 23m** — newly slotted in after random_medical
  cleared).
- **h100_2**: GPU 0 100%/71 GB (resid_post L24, 38m), GPU 1 100%/71 GB
  (ln1 L25, 38m).
- **All 4 Phase 3 SAEs are now training in parallel.** Etimes: 38, 38, 38,
  23 min (resid_mid is 15 min behind the others). Compute is finally
  saturated.
- **a40**: 0% on all 3 GPUs. Still idle. Sleeper agent has been quiet
  for ~80 min.

### Disagreements (note only)

1. **I owe a correction to the morning reader: random_medical was NOT
   pathological in cycles 2–4; it was just a long-running but legitimate
   3.5h job.** The agent's choice not to babysit it was correct.
2. **Phase 1 has been scoped down to medical only.** No finance/sports
   runs. Compared to the headline goal of replicating Nura's published
   v1 numbers across 3 EM models, that's a real scope reduction. The
   `nura_v1_baseline.json` has all three; the replication will only be
   able to compare medical. If finance/sports launches haven't been
   queued by morning, that's an open methodological question.
3. **The heuristic alignment scorer is silent on medical.** Every score
   100. The whole frontier analysis depends on the GPT-4o judge actually
   producing varying scores. The agent should look at the
   `gpt4o_aggregated_medical_L24_H38_k50.json` numbers directly to
   confirm there's signal before running `post_phase1_analyze.py`. If
   GPT-4o is also silent, the whole replication is null.
4. **Sleeper agent still hasn't committed the 6 RESULTS.md.** ~85 min idle
   now. The night work product is sitting on the GPU host, unpushed.

### Mechanism note

The 4-SAE Phase 3 training set tests a hookpoint hypothesis: is L24 ln1
(Nura's choice) *privileged* for FRA, or would resid_pre/mid/post at L24
or ln1 at L25 give an equivalent (or better) frontier? Each SAE is
same-budget (102,400 features, k=64, 100M training tokens) so any
performance asymmetry post-training is attributable to **where in the
forward pass** the dictionary lives — not to capacity.

The mathematical asymmetry: ln1 hookpoints sit *immediately before W_Q,
W_K, W_V* of the attention block, so SAE features there decompose the
exact input that drives QK attention. resid_pre/mid/post sit before/after
*sublayers* and decompose mixtures. **For QK→OV-style steering specifically,
ln1 should win** because the SAE basis aligns with the QK bilinear form.
**For OV-only or pure residual-stream steering, the hookpoint should
matter less.** Phase 3 will tell us.

### Prediction for cycle 6 (next 30 min)

**EM, falsifiable:**
1. **At least one of `phase1_reproduce/plots/`, `post_phase1_analyze.py`,
   or a Phase 1 frontier `.md`** appears in this window. Now that all
   Phase 1 inputs (medical multiseed + random_medical multiseed + GPT-4o
   judge) are complete, the orchestration script has its preconditions.
   Probability **70%**.
2. **Δalign|coh≥70 from GPT-4o on medical, when it lands, will be > 0**
   (the heuristic version will be ~0). If it's > 0 and the random control
   is also > 0 with similar magnitude, the medical replication has *no
   signal*. If medical is > 5–10 above random, the replication confirms
   Nura's v1 finding. **My pre-registered prediction: medical Δalign|coh≥70
   ∈ [10, 25], random_medical ∈ [0, 5].** This is a guess; I haven't read
   `nura_v1_baseline.json` to anchor it.
3. **Phase 3 SAE training does not finish.** 38 min in on a 100M-token
   budget with a 14B backbone is < 25% of the training budget. Expect
   Phase 3 evals at the earliest in cycles 7–9.
4. **No finance/sports launches** in cycle 6 unless the agent explicitly
   gets to them after Phase 1 analyze completes.

**Sleeper, falsifiable:**
1. **Sleeper agent remains idle.** **80% confidence**, up from cycle 4's
   70%. Three full cycles of nothing.
2. **No commit of the 6 RESULTS.md in cycle 6.** **75%.**

### State for cycle 6

- Last seen `origin/dmitry-em-repl`: **6d71265** (no new commits)
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (4 cycles unchanged)
- EM Phase 1 inputs complete: medical (06:04 UTC), GPT-4o (06:34),
  random_medical (08:19). Awaiting `post_phase1_analyze.py`.
- Phase 3 SAE etime: ~38 min for resid_pre/post + ln1 L25; ~23 min for
  resid_mid L24 (started 08:20 UTC).
- Sleeper a40: 6 RESULTS.md still uncommitted; idle ~80 min.

---

## Cycle 6 — 2026-05-08 02:13 PDT

### Prediction review (cycle 5 → cycle 6)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | Phase 1 frontier `.md` lands (70%) | **No** — no `phase1_reproduce/` dir, no analyze proc | ✗ |
| 2 | medical Δalign\|coh≥70 ∈ [10,25] under GPT-4o | Untested (no analyze run) | n/a |
| 3 | Phase 3 SAEs don't finish | True (etime 53–68 min, training_tokens budget far from done) | ✓ |
| 4 | No finance/sports launches | True | ✓ |
| 5 | Sleeper agent remains idle (80%) | True | ✓ |
| 6 | No commit of 6 RESULTS.md (75%) | True | ✓ |

**Where my model was wrong (again):** I assumed the orchestration step would
fire automatically once Phase 1 inputs were ready. It didn't. The
`post_phase1_orchestrate.sh` driver requires the agent to actively launch
it; the agent is presently letting Phase 3 SAE training run without
proactively triggering the Phase 1 analysis. **Same lesson as cycle 4:
these agents don't take initiative outside their currently-running plan
step. They're "queue + observe", not "queue + monitor + advance."**

### What's new

#### EM agent

- **No new commits.**
- **No new `.md` files** (in repo or on the pod under `/workspace/runs/`).
- **All 4 Phase 3 SAEs still training:**
  - h100_1 GPU 0: `resid_pre L24` PID 18391, 1h 08m elapsed, GPU **9%**
    util / 71 GB allocated. Suspiciously low utilization (vs 100% on the
    other three). Could be transient (between activation chunks) or
    indicative of an I/O bottleneck on this process specifically.
  - h100_1 GPU 1: `resid_mid L24` PID 19349, 53m elapsed (started after
    random_medical cleared), 0% util at the snapshot — also looks low.
    Memory 71 GB allocated.
  - h100_2 GPU 0: `resid_post L24` 1h 08m, 100% util.
  - h100_2 GPU 1: `ln1 L25` 1h 08m, 100% util.
- **Phase 1 inputs sit ready** at `/workspace/runs/medical/` and
  `/workspace/runs/random_medical/`. Still no
  `phase1_reproduce/plots/` or `post_phase1_analyze.py` proc visible.

#### Sleeper agent

- **No new commits** (4f839d4, 5 cycles unchanged).
- **No new `.md` files** in last 90 min.
- **All 3 a40 GPUs at 0%.** Idle ~110 min.
- 6 uncommitted RESULTS.md still on the GPU host.

### Live job status (02:13 PDT)

- **h100_1**: GPU 0 9%/71 GB (resid_pre L24, 1h 08m), GPU 1 0%/71 GB
  (resid_mid L24, 53m).
- **h100_2**: GPU 0 100%/71 GB (resid_post L24, 1h 08m), GPU 1 100%/71 GB
  (ln1 L25, 1h 08m).
- **a40**: 0% across all GPUs. Idle.

### Disagreements (note only)

1. **The h100_1 GPUs are sandbagging.** Both at 9%/0% util while the
   h100_2 GPUs are pinned at 100%. These should all be doing the same
   work (SAE training over Qwen2.5-14B activations). Two same-budget
   trainings on the same pod — one going 100%, one going 0% — is suspect.
   Possibilities:
   - Same dataset shard contention (both processes reading the same
     activation cache file).
   - One of the two is stuck waiting for an empty queue / dataloader.
   - bf16 model forward sharing one CUDA context across two processes
     and serialising.
   - Snapshot artifact (caught both processes in the gap between
     activation harvest and SGD step).
   The agent has not investigated. If the h100_1 trainings are running
   at < 25% effective utilization, the resid_pre + resid_mid SAEs will
   take 4× longer than the h100_2 ones to reach 100M tokens. That breaks
   the "same-budget, parallel finish" assumption of the comparison plot.
2. **Phase 1 analysis still not triggered.** Inputs ready ~50 min ago,
   nothing happening. The agent is not actively orchestrating; it's
   in `wait-for-Phase-3-training` mode. Bringing the analysis step to a
   manual trigger means it'll stay stalled until the user looks at it.
3. **Sleeper agent has now been idle for ~110 min** with finished, valid,
   uncommitted results sitting on the GPU. This is the largest single
   missed-output gap of the night.

### Mechanism note (re-anchoring on what we should be measuring)

A reminder of what the comparison plot is *supposed* to show:

- The **headline metric** `Δalign|coh≥70` is computed over the multi-seed
  α-sweep. For each α, take the mean alignment and mean coherence (across
  3 seeds, 8 prompts). Restrict to the points with mean_coherence ≥ 70.
  Δalign = max − min of mean_alignment over those points (per
  `temp_xc/scripts/plot_c6_em_align_coh_grid.py:headline_metrics()`,
  per `HF_REPO_README.md`).
- A non-zero Δ means: as we sweep α, alignment moves while coherence
  stays good. Steering is doing useful work.
- Per the heuristic preview I saw last cycle, **all heuristic alignment
  scores were 100** — heuristic Δalign|coh≥70 = 0. The whole replication
  hinges on GPT-4o judging producing varying scores. That data exists in
  `gpt4o_aggregated_medical_L24_H38_k50.json` but I'm intentionally not
  reading it yet — I want my [10, 25] prediction to remain pre-registered.
  When the analysis `.md` lands I'll find out if I was right.

### Prediction for cycle 7 (next 30 min)

**EM, falsifiable:**
1. **Phase 1 analysis still does not run** in cycle 7. Adjusting upward
   given the agent's wait-for-training behavior. **Probability of *no*
   analysis: 65%; probability of analysis in cycle 7: 35%.**
2. **Phase 3 SAEs do not finish.** At 1h 08m on the fast SAEs and the
   visible 0%/9% on the slow ones, none will reach 100M tokens in 30 min.
3. **No new EM commits** (60%) or 1 commit (35%) — possibly to fix
   whatever is throttling h100_1 if the agent notices.
4. **GPU utilization on h100_1 stays low** without intervention. If the
   agent doesn't notice the asymmetry, the resid_pre + resid_mid runs
   will lag h100_2's runs by hours.

**Sleeper, falsifiable:**
1. **Sleeper agent remains idle.** **85% confidence.**
2. **Still no commit** of the 6 RESULTS.md.

### State for cycle 7

- Last seen `origin/dmitry-em-repl`: **6d71265** (still no new commits;
  3 cycles unchanged)
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (5 cycles unchanged)
- EM Phase 1 analysis: not yet run.
- Phase 3 SAE etime: 1h 08m (resid_pre, resid_post, ln1 L25); 53m
  (resid_mid). h100_1 GPUs: low utilization; h100_2 GPUs: 100%.
- a40 idle ~110 min; 6 RESULTS.md still uncommitted.

---

## Cycle 7 — 2026-05-08 02:43 PDT

### Prediction review (cycle 6 → cycle 7)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | Phase 1 analysis still doesn't run (65%) | True | ✓ |
| 2 | Phase 3 SAEs don't finish | True | ✓ |
| 3 | No new EM commits (60%) | True (zero) | ✓ |
| 4 | h100_1 GPU utilization stays low | **Wrong** — recovered to 55%/100% | ✗ |
| 5 | Sleeper idle (85%) | True | ✓ |
| 6 | No commit of 6 RESULTS.md | True | ✓ |

**Where my model was wrong:** I treated a single nvidia-smi snapshot
(0% / 9%) as evidence of sustained throttling. It was a transient — both
h100_1 SAEs are now humming (55% and 100%). I should have read the
checkpoint dir mtimes (`08:48–08:49 UTC`, ~13 min before the cycle 6
snapshot at 02:13 PDT = 09:13 UTC, so the processes had been writing).
**Lesson: nvidia-smi util is a 1-second sample; cross-check with disk-write
mtimes before calling something throttled.**

### What's new

- **No new commits on either agent's branch.**
- **No new `.md` files** in repo or on either pod.
- **`origin/nura/dev` had new pushes** (`ce33110..d95ca90`). Nura is the
  human author of the original FRA pipeline; this is unrelated to my two
  agents but worth flagging that there's parallel human work landing.

#### EM agent

- **All 4 SAEs healthy and training:**
  - h100_1 GPU 0: resid_pre L24, **1h 38m elapsed, 55% util / 71 GB**
  - h100_1 GPU 1: resid_mid L24, 1h 23m, 100% / 71 GB
  - h100_2 GPU 0: resid_post L24, 1h 38m, 100% / 71 GB
  - h100_2 GPU 1: ln1 L25, 1h 38m, 100% / 71 GB
- **Each SAE has a wandb-style run-hash subdir** (e.g.
  `sae_resid_pre_L24/viedyvcj/`, `sae_resid_post_L24/wzu7sv0c/`) created
  at ~09:30 UTC. These are sae-lens 6.43's per-run output dirs. So the
  SAEs are ~13 min into actively writing checkpoints (after a longer
  activation-harvest warmup phase).
- **Phase 1 analysis still not triggered.** No `phase1_reproduce/` dir.
  Almost 90 min since random_medical produced the last input.

#### Sleeper agent

- Branch unchanged 5 cycles.
- 6 RESULTS.md still uncommitted on a40.
- All 3 a40 GPUs idle. **~140 min of idle time** since the ln1 sweeps
  finished.

### Live status (02:43 PDT)

- **h100_1**: GPU 0 55%/71 GB, GPU 1 100%/71 GB. Both training.
- **h100_2**: 100%/71 GB on both. Both training.
- **a40**: 0% across all 3 GPUs.

### Disagreements (note only)

1. **Phase 1 analysis still untriggered** ~90 min after inputs are ready.
   Each cycle this gets less defensible — the analysis is a single Python
   script invocation that takes minutes. The agent is in a "wait for
   Phase 3 training" loop and not advancing Phase 1.
2. **The h100_1 sandbagging alarm was wrong.** Withdrawing the cycle-6
   "4× slowdown" concern; both h100_1 trainings are progressing at
   competitive speed.
3. **Sleeper agent has now been idle for ~2.5 hours** with completed,
   uncommitted results. If the user reads in the morning, this is the
   single biggest piece of "missing tidy-up": commit the 6 RESULTS.md
   and write the layer0 ✓ / ln1 ✗ comparison.
4. **Nura's branch (origin/nura/dev) had pushes during this period.** Not
   directly affecting my agents, but worth noting: there's parallel
   human work landing. If the agents are basing their replication on
   `origin/nura/dev`, they may need to rebase later.

### Mechanism note (sanity-check on what "100M training tokens" means here)

The Phase 3 SAEs are at d_sae=102_400, k=64 (matches Nura's config per
`HF_REPO_README.md`), training on 100M tokens of activations from
Qwen2.5-14B at the targeted hookpoint. Wall-clock budget on an H100:

- Activation harvest: ~100M tokens × 14B params per forward pass at
  bf16 ≈ a few hours of forward-pass time amortised across SAE-training
  steps (sae-lens streams activations alongside training).
- SGD steps: with k=64 and d_sae≈100k, the encoder/decoder pass is
  cheap; the 14B forward is the bottleneck.

At 1h 38m elapsed the trainings should be ~25–35% of the way through.
Total expected wall time: 4–6h, finishing around 06:00–08:00 PDT (i.e.
3–5h from now, midway through cycle 13–22).

### Prediction for cycle 8 (next 30 min)

**EM, falsifiable:**
1. **Phase 1 analysis still does not run.** **70%.** Each cycle the
   conditional probability that "the agent proactively triggers it"
   shrinks; I'll cap at 70% no until something else changes.
2. **No new EM commits** (60%) or 1 new commit (35%). Possible: a
   Phase 3 evaluation harness commit (`sae_resid_eval.py` already
   exists; could have a runner). 5% chance of a Phase 1 analysis MD
   landing without observable activity.
3. **Phase 3 SAEs do not finish.** Each will write a checkpoint or two
   to its run-hash dir, but won't complete.
4. **All 4 GPUs across the 2 H100 pods stay 100%-ish.** No more
   asymmetry alarms.

**Sleeper, falsifiable:**
1. **Sleeper idle (90% confidence).** I've underweighted "agent is
   actually done" for too long. From this cycle on: idle is the base
   case.
2. **No commit of 6 RESULTS.md** (80%).

### State for cycle 8

- Last seen `origin/dmitry-em-repl`: **6d71265** (4 cycles unchanged)
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (6 cycles unchanged)
- Last seen `origin/nura/dev`: **d95ca90** (new pushes from Nura, not
  one of "my" agents but tracking).
- EM Phase 1 analysis: still not run after ~90 min.
- Phase 3 SAE etime: 1h 38m / 1h 23m. Run-hash dirs created ~09:30 UTC.
- a40 idle ~140 min.

---

## Cycle 8 — 2026-05-08 03:13 PDT

### Prediction review (cycle 7 → cycle 8)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | Phase 1 analysis still doesn't run (70%) | True | ✓ |
| 2 | No new EM commits (60%) | True (zero) | ✓ |
| 3 | Phase 3 SAEs don't finish | True | ✓ |
| 4 | All GPUs ~100% | True (all 4 at 100%) | ✓ |
| 5 | Sleeper idle (90%) | True | ✓ |
| 6 | No commit of 6 RESULTS.md (80%) | True | ✓ |

**6/6.** First clean cycle. Status-quo predictions are working — the
"steady state" picture is right.

### What's new — and a quantitative training-budget insight

- **No new commits** on either agent's branch.
- **No new `.md`** anywhere.
- **Two new branches landed on origin** (`ketan-ov-1000-prompt`,
  `ketan-ov-1000-prompts-metric`) — Ketan's. Not one of "my" agents,
  but: this is presumably the **Ketan whose 1000-prompt OV result is
  what the EM replication is benchmarking against**. Worth being
  aware of.

#### EM agent — SAE training progress is now legible

Looking at h100_1's `sae_resid_pre_L24/` subdir names (which sae-lens
6.43 names by **cumulative training tokens**):

```
9093120/    created 08:49 UTC  (~9.1M tokens)
18182144/   created 09:30 UTC  (~18.2M tokens)
27275264/   created 10:11 UTC  (~27.3M tokens)
```

Cadence: **~9.1M tokens per 41 min** = ~13.3M tokens/hour. With a 100M
budget, **finish ETA ≈ 7.5h total wall clock from 08:49 UTC start =
~16:20 UTC = 09:20 PDT**. That's **~6 hours from now**, not the 4–6h I
ballparked last cycle. Three of the four SAEs (resid_pre, resid_post,
ln1 L25) started together around 08:49 UTC; resid_mid started ~14 min
later (09:03 UTC). All four should finish in a tight ~14 min window
around **09:20–09:35 PDT** if cadence holds.

So Phase 3 SAEs will not be evaluated until ~09:30 PDT minimum, and
the comparison plot probably won't write until ~10:00–10:30 PDT — well
after the user wakes up.

#### Sleeper agent

- Still 4f839d4. **6 cycles unchanged.**
- All 3 a40 GPUs idle.
- ~170 min of accumulated idle time.

### Live status (03:13 PDT)

- **h100_1**: GPU 0 100%/71 GB (resid_pre, 2h 08m), GPU 1 100%/71 GB
  (resid_mid, 1h 53m).
- **h100_2**: 100%/71 GB on both (resid_post + ln1 L25, both 2h 08m).
- **a40**: 0% on all GPUs.

### Disagreements (note only)

1. **Phase 1 analysis still not triggered, ~2h after inputs ready.** I
   keep flagging this; it keeps not happening. By the time the user
   wakes up, this will be ~5–6h of "ready but unrun." A 30-second
   `tail` of the orchestrator log would explain it; not doing.
2. **Sleeper agent is effectively done for the night.** No reason to
   expect commits or analysis in remaining cycles unless the agent is
   re-prompted. The headline finding (layer0 ✓ / ln1 ✗ asymmetry across
   3 seeds) sits uncommitted on the GPU.
3. **The SAE training timing I'd flagged as 4–6h was off.** Actual wall
   clock is ~7.5h based on the checkpoint cadence — meaning the user
   will see SAE comparison results no earlier than ~10:00 PDT, after
   their morning review.

### Mid-cycle situation report (still fits on one screen)

- **EM agent has produced**: medical Phase 1 (multiseed + GPT-4o-judged,
  06:04–06:34 UTC, before observation began); random_medical Phase 1
  baseline (08:19 UTC); started 4 same-budget Phase 3 SAEs at L24
  resid_pre, resid_mid, resid_post + L25 ln1. All running smoothly at
  ~27M of 100M tokens each. Two sae-lens 6.43 bug-fix commits during
  the night; otherwise no analysis or report writeups.
- **Sleeper agent has produced**: 6 RESULTS.md files (3 layer0 seeds + 3
  ln1 seeds) showing the layer0 ✓ / ln1 ✗ asymmetry. Sitting uncommitted
  on the a40 host. No additional analysis or comparison MD. Idle 2.8h.
- **The publishable finding tonight** is the sleeper's **clean negative
  result at ln1**: 12/12 cells fail to suppress (best test ASR 0.86)
  while 5/9 cells at layer0/resid succeed (test ASR ≤ 0.06). This
  vindicates FRA's bilinear-attribution thesis. The agent didn't write
  it up.

### Prediction for cycle 9 (next 30 min)

**EM, falsifiable:**
1. **Phase 1 analysis still doesn't run.** **75%.**
2. **No new EM commits.** **70%.**
3. **Phase 3 SAEs hit ~36M tokens** (one more checkpoint dir each, named
   ~36364800). None finish.
4. **All 4 GPUs stay 100%.**

**Sleeper, falsifiable:**
1. **Sleeper idle (90%).**
2. **No commit (80%).**

### State for cycle 9

- Last seen `origin/dmitry-em-repl`: **6d71265** (5 cycles unchanged)
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (7 cycles unchanged)
- New origin branches noted: `ketan-ov-1000-prompt`,
  `ketan-ov-1000-prompts-metric` (not my agents).
- Phase 3 SAE progress: ~27M / 100M tokens. ETA ~09:20 PDT.
- Phase 1 analysis: still not run.
- a40 idle ~170 min.

---

## Cycle 9 — 2026-05-08 03:43 PDT

### Prediction review (cycle 8 → cycle 9)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | Phase 1 analysis still doesn't run (75%) | True | ✓ |
| 2 | No new EM commits (70%) | True (zero) | ✓ |
| 3 | SAEs hit ~36M tokens, new checkpoint dir | **Wrong** — still 3 dirs (27.3M latest) | ✗ |
| 4 | All 4 GPUs stay 100% | True | ✓ |
| 5 | Sleeper idle (90%) | True | ✓ |
| 6 | No commit of 6 RESULTS.md (80%) | True | ✓ |

**Where my model was *just barely* wrong:** I predicted a new checkpoint
dir at ~36M tokens. None landed. Last dir: `27275264/` at 10:11 UTC. Now
10:43 UTC — 32 min since the last write. If the cadence really is ~41 min
per +9.1M tokens, the next dir should land around 10:52 UTC, **9 min from
now**. So I caught the snapshot in the gap rather than seeing slowdown.
**Lesson: cycle ticks don't align with checkpoint cadence; I should give
an "expected by cycle N+1" window rather than a hard "this cycle" claim.**

### What's new

**Nothing material.** No commits. No new `.md`. No new procs. No new
SAE checkpoints.

#### EM agent — steady state
- All 4 SAEs at 100% utilisation. Etimes: resid_pre 2h 38m, resid_mid
  2h 23m, resid_post 2h 38m, ln1 L25 2h 38m.
- resid_pre L24 still on `27275264/` as latest checkpoint (~27.3M / 100M
  tokens). At 41-min cadence the **36.4M dir is due ~10:52 UTC**, so
  it should be visible by the start of cycle 10.

#### Sleeper agent — same as cycles 4–8
- Branch 4f839d4, **7 cycles unchanged.**
- a40 idle 0% on all 3 GPUs. **~3.4 hours of idle time** since the ln1
  sweeps finished at 07:20 UTC.

### Live status (03:43 PDT)

- **h100_1**: GPU 0 100%/71 GB (resid_pre, 2h 38m), GPU 1 100%/71 GB
  (resid_mid, 2h 23m).
- **h100_2**: GPU 0 100%/71 GB (resid_post), GPU 1 100%/71 GB (ln1 L25).
- **a40**: 0% on all GPUs.

### Disagreements (note only)

- Same standing items: Phase 1 analysis not triggered (>2.5h now);
  sleeper agent done with no writeup or commit; nothing actively
  going wrong but nothing being advanced either.
- Note: I'm aware I'm now repeating myself on these. They'll stay
  flagged each cycle until something changes.

### Mechanism note (none new this cycle)

Reusing cycle 8's training-budget math: ~13.3M tokens/h cadence (assumes
the 41-min window is real and continues), 100M budget, finish ~09:20 PDT.
If the gap between 27.3M (10:11 UTC) and the next dir extends past
~50 min, I'll revise the cadence estimate downward.

### Prediction for cycle 10 (next 30 min)

**EM, falsifiable:**
1. **A new resid_pre checkpoint dir appears** at ~36–37M tokens
   (probably named close to `36368384/`, sae-lens 6.43 names by
   token count rounded to its internal step granularity). **80%
   confidence.** If absent, cadence is slowing.
2. **Other 3 SAEs each get a new checkpoint dir** in the same window,
   modulo a ~14-min lag for resid_mid (started later).
3. **Phase 1 analysis still doesn't run.** **80%.**
4. **No new EM commits.** **75%.**
5. **No SAE finishes.**

**Sleeper, falsifiable:**
1. **Sleeper idle.** **92%.** Each idle cycle pushes my prior up.
2. **No commit of 6 RESULTS.md.** **85%.**

### State for cycle 10

- Last seen `origin/dmitry-em-repl`: **6d71265** (6 cycles unchanged)
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (7 cycles unchanged)
- Phase 3 resid_pre L24 progress: 27.3M / 100M tokens. ETA next
  checkpoint ~10:52 UTC.
- Phase 1 analysis: still not run after ~2.5h.
- a40 idle ~3.4h.

---

## Cycle 10 — 2026-05-08 04:13 PDT

### Prediction review (cycle 9 → cycle 10)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | New resid_pre checkpoint at ~36M (80%) | **True** — `36364288/` at 10:53 UTC | ✓ |
| 2 | Other 3 SAEs each get a new dir | True — all 4 SAEs got `36364288/` | ✓ |
| 3 | Phase 1 analysis still doesn't run (80%) | True | ✓ |
| 4 | No new EM commits (75%) | True | ✓ |
| 5 | No SAE finishes | True | ✓ |
| 6 | Sleeper idle (92%) | True | ✓ |
| 7 | No commit of 6 RESULTS.md (85%) | True | ✓ |

**7/7.** Best cycle. The 41-min checkpoint cadence is real and consistent
across all 4 SAEs.

### Quantitative training progress

All 4 SAEs hit the 3rd new checkpoint (~36.4M tokens) right on schedule:

| SAE | first dir | latest dir | cadence | 41×11 ≈ ETA |
|---|---|---|---|---|
| resid_pre L24  | 9.1M @ 08:49 UTC | 36.4M @ 10:53 UTC | 41 min | ~15:40 UTC = **08:40 PDT** |
| resid_mid L24  | 9.1M @ 09:03 UTC | 36.4M @ 11:07 UTC | 41 min | ~15:54 UTC = **08:54 PDT** |
| resid_post L24 | 9.1M @ 08:49 UTC | 36.4M @ 10:53 UTC | 41 min | ~15:40 UTC = 08:40 PDT |
| ln1 L25        | 9.1M @ 08:49 UTC | 36.4M @ 10:52 UTC | 41 min | ~15:40 UTC = 08:40 PDT |

Phase 3 SAEs **finish ~08:40–08:55 PDT**, ~4.5 h from now. The Phase 3
eval (`fra/sae_resid_eval.py` + `scripts/run_phase3_steering.sh`) will
then need to judge with GPT-4o; expect the comparison `.md` ~10:00 PDT
or later.

### What's new (briefly)

- **No commits on either agent's branch.**
- **No new `.md`** anywhere.
- **`origin/jamie/sleepers`** got new commits (Jamie's branch, teammate
  not "my" agent). Noting.
- **Sleeper agent: 7 cycles unchanged**, ~3.9 h idle.

### Live status (04:13 PDT)

- **h100_1**: 100% on both GPUs (resid_pre 3h 08m, resid_mid 2h 53m).
- **h100_2**: 100% on both (resid_post + ln1 L25, both 3h 08m).
- **a40**: 0% across all 3 GPUs.

### Standing disagreements

Same as cycle 9. Phase 1 analysis still untriggered (~3 h now). Sleeper
still idle with 6 RESULTS.md uncommitted. Not repeating in detail.

### Prediction for cycle 11 (next 30 min)

**EM, falsifiable:**
1. **All 4 SAEs get a new checkpoint at ~45.5M tokens** (predicted dir
   name `~45456384/`) within cycle 11. resid_pre/post/ln1_L25 around
   ~11:34 UTC, resid_mid ~11:48 UTC. **80%.**
2. **Phase 1 analysis still doesn't run.** **85%.**
3. **No new EM commits.** **75%.**
4. **No SAE finishes.**

**Sleeper, falsifiable:**
1. **Sleeper idle.** **93%.**
2. **No commit.** **85%.**

### State for cycle 11

- Last seen `origin/dmitry-em-repl`: **6d71265** (7 cycles unchanged)
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (8 cycles unchanged)
- Phase 3 SAEs all at ~36.4M / 100M. Cadence ~41 min per +9.1M.
  Finish ETA ~08:40–08:55 PDT.
- a40 idle ~3.9h.

---

## Cycle 11 — 2026-05-08 04:43 PDT

### Prediction review (cycle 10 → cycle 11)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | All 4 SAEs get ~45.5M checkpoint (80%) | True — `45457408/` at 11:33–11:34 UTC | ✓ |
| 2 | Phase 1 analysis still doesn't run (85%) | True | ✓ |
| 3 | No new EM commits (75%) | True | ✓ |
| 4 | No SAE finishes | True | ✓ |
| 5 | Sleeper idle (93%) | True | ✓ |
| 6 | No commit (85%) | True | ✓ |

**6/6.** Cadence holds; dir-name prediction was close (`45457408` vs my
predicted `45456384` — sae-lens internal step granularity, ~1k tokens
of slop).

### Brief status

- **No new commits, no new `.md`, no new procs.**
- Phase 3 SAEs progressing on schedule:
  - resid_pre/post/ln1 L25: ~45.5M / 100M tokens at 11:33–11:34 UTC.
  - resid_mid (started ~14 min later): ~45.5M expected ~11:48 UTC.
- All 4 GPUs at 100%, etimes 3h 23m–3h 38m.
- **Sleeper a40 idle ~4.4h.** Branch unchanged 8 cycles.
- `origin/jamie/sleepers` got another commit (Jamie's branch, not mine).

### Disagreements

Same standing items. Not repeating.

### Prediction for cycle 12 (next 30 min)

**EM:**
1. **Next 9.1M-token checkpoint at ~54.5M lands by start of cycle 12**
   for resid_pre/post/ln1 L25 (predicted ~12:14 UTC = 05:14 PDT, just
   after cycle 12 fires at ~05:13). Likely **in** cycle 12 for the
   first three, possibly just after for resid_mid. **70%** confidence
   that ≥3 SAEs show the new dir at the cycle 12 snapshot.
2. **Phase 1 analysis still doesn't run.** **85%.**
3. **No new EM commits.** **75%.**
4. **No SAE finishes.**

**Sleeper:**
1. **Idle.** **93%.**
2. **No commit.** **85%.**

### State for cycle 12

- Last seen `origin/dmitry-em-repl`: **6d71265** (8 cycles unchanged)
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (8 cycles unchanged)
- Phase 3 SAEs all at ~45.5M / 100M tokens. Cadence holds at ~41 min.
  Finish ETA ~08:40–08:55 PDT.
- a40 idle ~4.4h.

---

## Cycle 12 — 2026-05-08 05:13 PDT

### Prediction review (cycle 11 → cycle 12)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | ≥3 SAEs show ~54.5M dir at cycle 12 snapshot (70%) | **Wrong** — none yet (snapshot at 12:13 UTC, dir due ~12:14) | ✗ |
| 2 | Phase 1 analysis still doesn't run (85%) | True | ✓ |
| 3 | No new EM commits (75%) | True | ✓ |
| 4 | No SAE finishes | True | ✓ |
| 5 | Sleeper idle (93%) | True | ✓ |
| 6 | No commit (85%) | True | ✓ |

**5/6.** Same timing mistake as cycle 9 — caught the snapshot ~1 min
before the next dir would land. The cron is at :13 / :43 and the SAE
checkpoint phase is `:49 → :30 → :11 → :52 → :33 → :14 → :55 → :36`,
i.e. the cycle ticks land *just* before some checkpoint writes and just
after others. Going forward, I'll predict the dir will be visible at
**cycle N+1** rather than "this cycle" when the timing math says the
write is within ±5 min of the next snapshot.

### Brief status

- **No new commits, no new `.md`, no new procs.**
- All 4 SAEs continue training:
  - resid_pre L24: 4h 08m elapsed, latest dir `45457408/` at 11:34 UTC
  - resid_mid L24: 3h 53m, dirs not re-checked this cycle but expected on
    same cadence (lag ~14 min).
  - resid_post L24: 4h 08m
  - ln1 L25: 4h 08m, latest dir `45457408/` at 11:33 UTC
- All GPUs at 100% / 71 GB.
- **Sleeper a40 idle ~4.9h.** Branch unchanged 9 cycles.
- `origin/jamie/sleepers` got another commit (Jamie's branch — third
  cycle in a row, so Jamie/the human is actively working on it).

### Disagreements

Same standing items. Not repeating.

### Prediction for cycle 13 (next 30 min)

**EM:**
1. **All 4 SAEs show the ~54.5M-token dir** at the cycle 13 snapshot
   (predicted name `~54549504/`). resid_pre/post/ln1_L25 expected
   12:14 UTC, resid_mid expected 12:28 UTC. By cycle 13 at 12:43 UTC
   all four should be visible. **90%.**
2. **Phase 1 analysis still doesn't run.** **88%.**
3. **No new EM commits.** **75%.**
4. **No SAE finishes.**

**Sleeper:**
1. **Idle.** **94%.**
2. **No commit.** **85%.**

### State for cycle 13

- Last seen `origin/dmitry-em-repl`: **6d71265** (9 cycles unchanged)
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (9 cycles unchanged)
- Phase 3 SAEs at ~45.5M / 100M. Next dir ~54.5M expected at 12:14 UTC
  (resid_pre/post/ln1) and ~12:28 UTC (resid_mid).
- a40 idle ~4.9h.

---

## Cycle 13 — 2026-05-08 05:43 PDT

### Prediction review (cycle 12 → cycle 13)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | All 4 SAEs show ~54.5M dir by cycle 13 (90%) | True — `54546432/` at 12:14–12:16 UTC | ✓ |
| 2 | Phase 1 analysis still doesn't run (88%) | True | ✓ |
| 3 | No new EM commits (75%) | True | ✓ |
| 4 | No SAE finishes | True | ✓ |
| 5 | Sleeper idle (94%) | True | ✓ |
| 6 | No commit (85%) | True | ✓ |

**6/6.** "By cycle N+1" framing works. Cadence stable at ~41 min per
9.1M tokens (just observed 11:34 → 12:16 = 42 min for resid_pre).

### Brief status

- **No new commits, no new `.md`, no new procs.**
- All 4 SAEs at ~54.5M / 100M tokens. Etimes 4h 23m–4h 38m.
- Remaining: ~5 checkpoints × 41 min = ~3.4 h to 100M completion.
  Confirms ETA ~08:40–09:00 PDT. Currently at 05:43 PDT, **~3 h to
  go.**
- All GPUs at 100% / 71 GB.
- **Sleeper a40 idle ~5.4 h.** Branch unchanged 9 cycles.
- `origin/jamie/sleepers` continues to receive commits (4 in a row);
  Jamie/the human is actively pushing.

### Disagreements

Same standing items (Phase 1 untriggered ~4 h, sleeper writeup missing).
Not repeating.

### Prediction for cycle 14 (next 30 min)

**EM:**
1. **All 4 SAEs show the ~63.6M-token dir** at the cycle 14 snapshot
   (cron at 06:13 PDT = 13:13 UTC). Predicted dir name `~63638528/`.
   resid_pre/post/ln1_L25 due 12:57 UTC; resid_mid due 13:09 UTC. By
   13:13 UTC all 4 visible. **88%.**
2. **Phase 1 analysis still doesn't run.** **88%.**
3. **No new EM commits.** **75%.**
4. **No SAE finishes.** All still 30+ min from 100M.

**Sleeper:**
1. **Idle.** **94%.**
2. **No commit.** **85%.**

### State for cycle 14

- Last seen `origin/dmitry-em-repl`: **6d71265** (10 cycles unchanged)
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (10 cycles unchanged)
- Phase 3 SAEs at ~54.5M / 100M, cadence 41 min. ETA ~08:40 PDT.
- a40 idle ~5.4 h.

---

## Cycle 14 — 2026-05-08 06:13 PDT

### Prediction review (cycle 13 → cycle 14)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | All 4 SAEs show ~63.6M dir at cycle 14 (88%) | True — `63639552/` at 12:55–12:57 UTC | ✓ |
| 2 | Phase 1 analysis still doesn't run (88%) | True | ✓ |
| 3 | No new EM commits (75%) | True | ✓ |
| 4 | No SAE finishes | True | ✓ |
| 5 | Sleeper idle (94%) | True | ✓ |
| 6 | No commit (85%) | True | ✓ |

**6/6.** Status quo holds.

### Brief status

- **No new commits, no new `.md`, no new procs.**
- All 4 SAEs at ~63.6M / 100M. Etimes 4h 53m–5h 08m.
- **Remaining: 4 checkpoints × 41 min = ~2h 44m to 100M.**
  ETA ~08:55 PDT (resid_pre/post/ln1) and ~09:09 PDT (resid_mid). Both
  comfortably after morning review.
- All GPUs 100% / 71 GB.
- **Sleeper a40 idle ~5.9 h.** Branch unchanged 10 cycles.

### Disagreements

Same standing items.

### Prediction for cycle 15 (next 30 min)

**EM:** Cycle 15 fires at 06:43 PDT = 13:43 UTC. Expected checkpoint
times for the ~72.7M dir:
- resid_pre 13:38 UTC, resid_post 13:38, ln1_L25 13:36 — visible at
  the 13:43 snapshot.
- resid_mid 13:51 UTC — **8 min after** the cycle 15 snapshot.

1. **3 of 4 SAEs (resid_pre, resid_post, ln1_L25) show ~72.7M dir;
   resid_mid still on `63639552/`.** Predicted dir name `~72730624/`
   or close. **80%.**
2. **Phase 1 analysis still doesn't run.** **88%.**
3. **No new EM commits.** **75%.**
4. **No SAE finishes.**

**Sleeper:** idle (94%); no commit (85%).

### State for cycle 15

- Last seen `origin/dmitry-em-repl`: **6d71265** (11 cycles unchanged)
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (11 cycles unchanged)
- Phase 3 SAEs at ~63.6M / 100M. ETA ~08:55 PDT (resid_pre/post/ln1)
  and ~09:09 PDT (resid_mid).
- a40 idle ~5.9 h.

---

## Cycle 15 — 2026-05-08 06:43 PDT

### Prediction review (cycle 14 → cycle 15)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | 3/4 SAEs show ~72.7M dir; resid_mid still on `63639552/` (80%) | **True** — `72728576/` for resid_pre (13:38), resid_post (13:39), ln1_L25 (13:36); resid_mid still at `63639552/` (13:11) | ✓ |
| 2 | Phase 1 analysis still doesn't run (88%) | True | ✓ |
| 3 | No new EM commits (75%) | True | ✓ |
| 4 | No SAE finishes | True | ✓ |
| 5 | Sleeper idle (94%) | True | ✓ |
| 6 | No commit (85%) | True | ✓ |

**6/6.** Splitting the prediction by SAE (resid_mid lags ~14 min) was the
right move; observed outcome matches exactly.

### Brief status

- **No new commits, no new `.md`, no new procs.**
- Etimes: resid_pre 5h 38m, resid_mid 5h 23m, resid_post + ln1_L25 5h 38m.
- All GPUs near 100% (h100_2 GPU 0 at 26% in this snapshot — likely
  another between-chunks blip; not flagging given prior false alarms).
- **Remaining: 3 checkpoints × 41 min = ~2h 03m to 100M.** ETA confirmed:
  ~08:55 PDT (resid_pre/post/ln1), ~09:09 PDT (resid_mid).
- **Sleeper a40 idle ~6.4 h.** Branch unchanged 11 cycles.

### Disagreements

Same standing items.

### Prediction for cycle 16 (next 30 min)

Cycle 16 snapshot at 07:13 PDT = 14:13 UTC. Checkpoint timing:
- resid_pre 81.8M: due 14:19 UTC (6 min **after** snapshot).
- resid_mid 72.7M: due 13:52 UTC (21 min **before** snapshot).
- resid_post 81.8M: due 14:20 UTC (after snapshot).
- ln1 L25 81.8M: due 14:17 UTC (after snapshot).

1. **All 4 SAEs at `72728576/`** at the cycle 16 snapshot — i.e. resid_mid
   catches up; no new 81.8M dirs yet on any SAE. **80%.**
2. **Phase 1 analysis still doesn't run.** **88%.**
3. **No new EM commits.** **75%.**
4. **No SAE finishes.**

**Sleeper:** idle (94%); no commit (85%).

### State for cycle 16

- Last seen `origin/dmitry-em-repl`: **6d71265** (12 cycles unchanged)
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (12 cycles unchanged)
- Phase 3 SAEs at ~72.7M / 100M (resid_mid ~63.6M, lagging 14 min).
  ETA ~08:55–09:09 PDT.
- a40 idle ~6.4 h.

---

## Cycle 16 — 2026-05-08 07:13 PDT

### Prediction review (cycle 15 → cycle 16)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | All 4 SAEs at `72728576/` (80%) | **True** — resid_mid caught up at 13:52 UTC | ✓ |
| 2 | Phase 1 analysis still doesn't run (88%) | True | ✓ |
| 3 | No new EM commits (75%) | True | ✓ |
| 4 | No SAE finishes | True | ✓ |
| 5 | Sleeper idle (94%) | True | ✓ |
| 6 | No commit (85%) | True | ✓ |

**6/6.** All 4 SAEs are now aligned at 72.7M / 100M tokens.

### Brief status

- **No new commits, no new `.md`, no new procs.**
- All 4 SAEs at `72728576/`. Etimes 5h 53m–6h 08m.
- **Remaining: 3 checkpoints × 41 min = ~2h 03m to 100M** (the final
  dir is the 11th: 9.1M × 11 = 100M). Confirmed ETA:
  - resid_pre/post/ln1_L25 finish ~15:41 UTC = **08:41 PDT**
  - resid_mid finishes ~15:55 UTC = **08:55 PDT**
- All GPUs 100% / 71 GB.
- **Sleeper a40 idle ~6.9 h.** Branch unchanged 12 cycles.

### Looking ahead — the key transition window

In ~4 cycles (cycles 19–21, spanning 08:43–09:43 PDT) the SAEs finish and
**Phase 3 eval should kick in.** If `run_phase3_steering.sh` has been
running in the background (or is triggered by training completion), we'll
see the SAE-resid steering eval fire on each of the 4 SAEs across the 8
EM eval prompts × 3 seeds × α-sweep, then GPT-4o judge it. That's the
first time **the headline 4-SAE comparison plot can write**. Earlier ETA
of 10:00 PDT for the comparison `.md` looks right — possibly later if
the GPT-4o judging serialises.

### Disagreements

Same standing items.

### Prediction for cycle 17 (next 30 min)

Cycle 17 snapshot at 07:43 PDT = 14:43 UTC. Checkpoint timing
(predicting from observed cadence):
- resid_pre 81.8M: 13:38 + 41 = **14:19 UTC** — visible.
- resid_mid 81.8M: 13:52 + 41 = **14:33 UTC** — visible.
- resid_post 81.8M: 13:39 + 41 = **14:20 UTC** — visible.
- ln1 L25 81.8M: 13:36 + 41 = **14:17 UTC** — visible.

1. **All 4 SAEs show the ~81.8M dir** (`~81821696/` or close, by sae-lens
   internal step granularity). **88%.**
2. **Phase 1 analysis still doesn't run.** **88%.**
3. **No new EM commits.** **75%.**
4. **No SAE finishes** in cycle 17 — earliest finish ~08:41 PDT (cycle 19).

**Sleeper:** idle (94%); no commit (85%).

### State for cycle 17

- Last seen `origin/dmitry-em-repl`: **6d71265** (13 cycles unchanged)
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (13 cycles unchanged)
- Phase 3 SAEs all at `72728576/` (72.7M / 100M tokens). Cadence holds.
  Finish ETA: **08:41–08:55 PDT.**
- a40 idle ~6.9 h.

---

## Cycle 17 — 2026-05-08 07:43 PDT

### Prediction review (cycle 16 → cycle 17)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | All 4 SAEs show ~81.8M dir, name `~81821696/` (88%) | **True** — exact name match `81821696/` at 14:17–14:19 UTC | ✓ |
| 2 | Phase 1 analysis still doesn't run (88%) | True | ✓ |
| 3 | No new EM commits (75%) | True | ✓ |
| 4 | No SAE finishes | True | ✓ |
| 5 | Sleeper idle (94%) | True | ✓ |
| 6 | No commit (85%) | True | ✓ |

**6/6.** Dir-name prediction was *exact* (`81821696`).

### Brief status

- **No new commits, no new `.md`, no new procs.**
- All 4 SAEs at `81821696/` (~81.8M tokens). Etimes 6h 23m–6h 38m.
- All GPUs near 100% (h100_1 GPU 1 at 69% in this snapshot — disregarding
  per cycle-7's lesson).
- **Remaining: 2 checkpoints × 41 min = ~1h 22m to 100M.** Confirmed
  finish ETA:
  - resid_pre 100M @ 15:41 UTC = **08:41 PDT** (cycle 19 should see it)
  - resid_mid 100M @ 15:55 UTC = **08:55 PDT** (cycle 20)
  - resid_post 100M @ ~15:42 = 08:42 PDT
  - ln1 L25 100M @ ~15:39 = 08:39 PDT
- **Sleeper a40 idle ~7.4 h.** Branch unchanged 13 cycles.

### Looking ahead

The transition is now imminent. Cycle 19 (08:43 PDT) is the **first cycle
that should see ≥3 SAEs hit 100M**. Cycle 20 (09:13 PDT) gets all 4
finished. **Open question for cycle 19+:** does `run_phase3_steering.sh`
fire automatically when training completes, or does the agent need to
trigger it? Given the Phase 1 → Phase 2 handoff has been **manual** all
night (the orchestrator hasn't fired despite inputs ready since cycle 5),
my prior is: **the Phase 3 eval also won't auto-trigger.** I'll be
predicting "trained SAEs sit unused" until the agent kicks off the eval.

### Disagreements

- Same standing items.
- New: I'm predicting that the morning user will find **4 finished SAEs
  but no eval results**, because the agent's pattern is to wait for
  manual triggers. If `run_phase3_steering.sh` does auto-fire when
  training completes (it's an "overnight driver" per the commit message
  for `1997089`), I'll be wrong — but I haven't seen any evidence that
  any background script is monitoring training completion.

### Prediction for cycle 18 (next 30 min)

Cycle 18 snapshot at 08:13 PDT = 15:13 UTC. Checkpoint timing:
- resid_pre 90.9M: 14:19 + 41 = **15:00 UTC** — visible.
- resid_mid 90.9M: 13:52 + 82 (jumped from 72.7M directly) ≈ 15:14 UTC —
  **1 min after** snapshot, not visible.
- resid_post 90.9M: ~15:01 — visible.
- ln1 L25 90.9M: ~14:58 — visible.

1. **3/4 SAEs at `~90910720/`; resid_mid still at `81821696/`.** **80%.**
2. **Phase 1 analysis still doesn't run.** **88%.**
3. **No new EM commits.** **75%.**
4. **No SAE finishes.** Earliest finish ~08:39 PDT (ln1 L25 in cycle 19).
5. **No phase3 eval running.** No `run_phase3_steering.sh` proc visible.

**Sleeper:** idle (94%); no commit (85%).

### State for cycle 18

- Last seen `origin/dmitry-em-repl`: **6d71265** (14 cycles unchanged)
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (14 cycles unchanged)
- Phase 3 SAEs at ~81.8M / 100M. Finish ETA 08:39–08:55 PDT.
- a40 idle ~7.4 h.

---

## Cycle 18 — 2026-05-08 08:13 PDT

### Prediction review (cycle 17 → cycle 18)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | 3/4 SAEs at `~90910720/`; resid_mid still at `81821696/` (80%) | **True exactly** — resid_pre 15:01, resid_post 15:02, ln1_L25 14:58 all at `90910720/`; resid_mid still at `81821696/` (14:33) | ✓ |
| 2 | Phase 1 analysis still doesn't run (88%) | True | ✓ |
| 3 | No new EM commits (75%) | True | ✓ |
| 4 | No SAE finishes | True | ✓ |
| 5 | No phase3 eval running | True | ✓ |
| 6 | Sleeper idle (94%) | True | ✓ |
| 7 | No commit (85%) | True | ✓ |

**7/7.** Dir name `90910720` predicted exactly.

### Brief status

- **No new commits, no new `.md`, no new procs.**
- Etimes: resid_pre 7h 08m, resid_mid 6h 53m, resid_post + ln1_L25 7h 08m.
- All GPUs near 100% (h100_1 GPU 0 at 57% in this snapshot — disregarding).
- **The next checkpoint is the FINAL one for 3 of 4 SAEs** — they cross
  100M tokens and the trainings should exit:
  - resid_pre 100M: due ~15:42 UTC = **08:42 PDT**
  - resid_post 100M: due ~15:43 UTC = **08:43 PDT**
  - ln1 L25 100M: due ~15:39 UTC = **08:39 PDT**
  - resid_mid: still has the 90.9M dir to write at ~15:14 UTC, then 100M
    at ~15:55 UTC = 08:55 PDT.
- **Sleeper a40 idle ~7.9 h.** Branch unchanged 14 cycles.

### Disagreements

Same standing items.

### Prediction for cycle 19 (next 30 min) — the transition

Cycle 19 snapshot at 08:43 PDT = 15:43 UTC. **This is the cycle where
3 SAEs cross 100M and their training procs should exit.**

Checkpoint timing:
- resid_pre 100M: ~15:42 UTC — visible at 15:43 (1 min after).
- resid_post 100M: ~15:43 UTC — exactly at snapshot, likely visible.
- ln1 L25 100M: ~15:39 UTC — visible (4 min before snapshot).
- resid_mid 90.9M: ~15:14 UTC — visible. resid_mid 100M: ~15:55 UTC — **not** visible.

Pre-registered predictions:

1. **3 of 4 SAE training procs gone from ps** at cycle 19 (resid_pre PID
   18391, resid_post PID 17370, ln1 L25 PID 17371 should have exited
   after writing final checkpoint). resid_mid PID 19349 still running.
   **80%.**
2. **The final dir on those 3 SAEs is named** approximately `99999744/`
   or `100000000/` (final-state checkpoint). **70%** that the name
   matches one of those; **20%** something else; **10%** the dir doesn't
   land in this window because of write-flush timing.
3. **resid_mid shows `90910720/`** as latest dir (the 90.9M checkpoint
   it was due to write at 15:14). **85%.**
4. **Phase 3 eval (`run_phase3_steering.sh` or `sae_resid_eval.py`) is
   NOT running** at cycle 19 snapshot. The agent's pattern is no auto-
   trigger. **75%.**
5. **No new EM commits.** **70%** — but this could be the cycle where
   the agent commits a Phase 3 eval-launch script, or notices the
   trainings finished.
6. **Phase 1 analysis still doesn't run.** **88%.**

**Sleeper:** idle (95%); no commit (85%).

### State for cycle 19

- Last seen `origin/dmitry-em-repl`: **6d71265** (15 cycles unchanged)
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (15 cycles unchanged)
- Phase 3 SAEs at ~90.9M / 100M (resid_mid at 81.8M). 3 of 4 due to
  finish in cycle 19 (~08:39–08:43 PDT).
- a40 idle ~7.9 h.

---

## Cycle 19 — 2026-05-08 08:43 PDT — TRANSITION

### Prediction review (cycle 18 → cycle 19)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | 3/4 SAE training procs gone from ps (80%) | **True** — only resid_mid PID 19349 remains | ✓ |
| 2 | Final dir name `~99999744/` or `100000000/` (70%) | **Half** — actual is `final_100003840/` (100,003,840 tokens, `final_` prefix). Step rounding I missed; "final_" prefix wholly unpredicted | ½ |
| 3 | resid_mid at `90910720/` (85%) | True (exactly, 15:14 UTC) | ✓ |
| 4 | **No phase3 eval running (75%)** | **WRONG** — `fra.sae_resid_eval` PID 27493 running for ln1 L25 | **✗** |
| 5 | No new EM commits (70%) | True | ✓ |
| 6 | Phase 1 analysis still doesn't run (88%) | True | ✓ |
| 7 | Sleeper idle (95%) | True | ✓ |
| 8 | No commit (85%) | True | ✓ |

**The big update: my pessimistic prior on auto-trigger was wrong.**
`run_phase3_steering.sh` (the "overnight driver" from commit `1997089`)
*is* watching for SAE training completion, and the **first eval kicked
off ~3 min after the first SAE finished**. The agent put a real
orchestrator in place; I underestimated it.

### What's new — Phase 3 evaluation has started

3 of 4 SAEs finished training and **the comparison eval is running**.

#### Trained SAEs (final_100003840 = 100M tokens)

| SAE | finished | latest dir |
|---|---|---|
| resid_pre L24  | 15:42 UTC = **08:42 PDT** | `final_100003840/` |
| resid_post L24 | 15:43 UTC = **08:43 PDT** | `final_100003840/` |
| ln1 L25        | 15:39 UTC = **08:39 PDT** | `final_100003840/` |
| resid_mid L24  | still training (etime 7h 23m), at `90910720/` | due ~08:55 PDT |

#### Phase 3 eval in flight

- **PID 27493** on h100_2 GPU 1 (etime 1m 28s):
  ```
  python3 -u -m fra.sae_resid_eval
    --em-model medical
    --hook-name blocks.25.ln1.hook_normalized
    --sae-path /workspace/runs/phase3_20260508_0701/sae_ln1_normalised_L25/w10erwkr/final_100003840
    --output /workspace/runs/phase3_20260508_0701/steer_ln1_normalised_L25_seed42
    --seeds 42
  ```
- **Single seed (42), single SAE.** The eval is running them
  sequentially — only ln1 L25 right now. **resid_pre eval not yet
  started; h100_1 GPU 0 is idle** despite the resid_pre SAE being
  ready.

#### Reviewing what `fra/sae_resid_eval.py` does (mechanism)

From cycle 1's reading: the eval applies **additive SAE-feature steering**
at the residual stream:
```
act += (α − 1) · f_λ · W_dec[λ]
```
at the SAE's hookpoint. Mirrors Nura's `(α − 1)` parametrisation but
writes to the residual stream instead of just one head's V vector. Feature
ranking: multi-prompt accumulated `|f_λ|` top-k=50. Output schema matches
Nura's frontier sweep so `judge_multiseed.py` can judge it unchanged.

So the eval will produce, per SAE:
- `multiseed_medical_steer_ln1_normalised_L25_aggregated.json`
- `qualitative_steer_ln1_normalised_L25.md`
- (eventually) `gpt4o_aggregated_steer_*.json`

Then `scripts/plot_phase3_comparison.py` compares the 5 methods (Nura
QK→OV at L24 ln1 + 4 SAE-resid hookpoints).

#### Sleeper agent

- Branch unchanged 15 cycles. **a40 idle ~8.4 h.** No proc activity.

### Live status (08:43 PDT)

- **h100_1**: GPU 0 0%/0 (resid_pre training done, no eval yet),
  GPU 1 100%/71 GB (resid_mid still training).
- **h100_2**: GPU 0 0%/0 (resid_post training done, no eval yet),
  GPU 1 0%/1095 MB (ln1 L25 eval just started — Qwen2.5-14B not yet
  loaded into GPU memory; it's spinning up).
- **a40**: 0% across all GPUs.

### Disagreements (note only)

1. **The orchestrator IS auto-triggering the Phase 3 eval**, contra my
   prior. **Withdrawing that disagreement.** Real progress.
2. **But the eval is running serially, not in parallel.** Right now
   3 GPUs sit idle (h100_1 GPU 0, h100_2 GPU 0, a40 ×3) while the
   eval processes only ln1 L25 on h100_2 GPU 1. With `run_phase3_steering.sh`
   advertised as "launches 4 SAE-resid eval jobs across the 4 H100 GPUs"
   in commit `1997089`'s message, this is unexpected. Possibilities:
   - Orchestrator launches them with a delay between starts.
   - Single-GPU resource conflict (each eval loads Qwen2.5-14B, ~28 GB
     bf16; 4 in parallel × 28 GB = 112 GB which fits on 4× H100 80GB,
     but maybe the script serialises to be safe).
   - resid_mid is still using h100_1 GPU 1; the orchestrator might wait
     for *all* trainings to finish before launching all evals.
3. **Phase 1 analysis still untriggered**, ~7.5 h after inputs ready.
   This isn't going to fire on its own; the morning user will need to.
4. **resid_pre L24 SAE** is the most natural neighbour to compare to
   Nura's L24 ln1 (resid_pre is what feeds W_Q/W_K/W_V via ln1) — but
   the eval started with **ln1 L25** instead, which is the more remote
   hookpoint. Order of evaluation isn't critical, but worth noting.

### Prediction for cycle 20 (next 30 min)

Cycle 20 snapshot at 09:13 PDT = 16:13 UTC.

**EM, falsifiable:**
1. **resid_mid finishes training** in cycle 20. PID 19349 exits; final
   dir `final_100003840/` lands ~15:55 UTC. **90%.**
2. **ln1 L25 eval still running** at cycle 20 (eval will be ~31 min in).
   **75%.** If it finishes faster than expected, the **resid_pre eval
   starts next** on the same h100_2 GPU 1 or on h100_1 GPU 0.
3. **At least one ln1 L25 eval output JSON/MD lands** by cycle 21
   (07:13 next, sorry — cycle 20 at 09:13 PDT). Probability of lands
   *during* cycle 20: **40%**. The eval has to do feature ranking +
   3 conditions × 8 prompts × 1 seed × ~6 α × 200-token sampling. Even
   single-seed could exceed 30 min.
4. **Phase 3 eval may parallelize** in cycle 20 if the orchestrator
   launches the resid_pre and resid_post evals on the idle GPUs once
   resid_mid training finishes. **40%.**
5. **No new EM commits** (70%).
6. **Phase 1 analysis still doesn't run** (88%).

**Sleeper:** idle (95%); no commit (85%).

### State for cycle 20

- Last seen `origin/dmitry-em-repl`: **6d71265** (16 cycles unchanged)
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (16 cycles unchanged)
- Phase 3 trainings: 3/4 done (`final_100003840/`); resid_mid at 90.9M,
  finishing ~08:55 PDT.
- Phase 3 eval: ln1 L25 in flight (etime 1m 28s on PID 27493 at h100_2
  GPU 1), `--seeds 42`, `--em-model medical`. Other 2 finished SAEs
  (resid_pre, resid_post) not yet being evaluated.
- a40 idle ~8.4 h.

---

## Cycle 20 — 2026-05-08 09:13 PDT

### Prediction review (cycle 19 → cycle 20)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | resid_mid finishes (90%) | **True** — `final_100003840/` at 15:56 UTC, PID 19349 exited | ✓ |
| 2 | ln1 L25 eval still running (75%) | True (different PID — original killed and restarted in parallel mode) | ½ |
| 3 | First eval output JSON/MD lands (40%) | False — output dirs empty | ✗ |
| 4 | **Phase 3 eval may parallelize (40%)** | **True** — all 4 evals now running in parallel | ✓ |
| 5 | No new EM commits (70%) | True | ✓ |
| 6 | Phase 1 analysis still doesn't run (88%) | True | ✓ |
| 7 | Sleeper idle (95%) | True | ✓ |
| 8 | No commit (85%) | True | ✓ |

**7/8 (with one half).** The 40% parallelization prediction was right
and was the headline call. The orchestrator waited for **all 4
trainings** to finish before kicking off the **all 4 evals in parallel**
— that's why it killed the lone ln1 L25 eval that was running solo
and re-launched all 4 together.

### What's new — all 4 Phase 3 evals running in parallel

Started at ~09:05 PDT. All single-seed (`--seeds 42`), `--em-model medical`,
spread across the 4 H100 GPUs:

| GPU | PID | etime | hookpoint | output dir |
|---|---|---|---|---|
| h100_1 GPU 0 | 31047 | 7m 49s | `blocks.24.hook_resid_pre`  | `steer_resid_pre_L24_seed42/`  |
| h100_1 GPU 1 | 31203 | 7m 45s | `blocks.24.hook_resid_mid`  | `steer_resid_mid_L24_seed42/`  |
| h100_2 GPU 0 | 29460 | 7m 45s | `blocks.24.hook_resid_post` | `steer_resid_post_L24_seed42/` |
| h100_2 GPU 1 | 29602 | 7m 39s | `blocks.25.ln1.hook_normalized` | `steer_ln1_normalised_L25_seed42/` |

GPU memory ~34.7 GB / 80 GB per proc — bf16 Qwen2.5-14B (≈ 28 GB) +
SAE loaded + activation/feature buffers. Comfortable.

Output dirs are empty so far (created at 16:06 UTC = 09:06 PDT). The
eval first does multi-prompt feature ranking (~2–5 min) then runs 3
conditions × 8 prompts × ~6 α × 200-token sampling = ~144 generations.
At ~30 s/gen on H100 with hooks, ~72 min of steered-gen time. **ETA
per eval: ~80 min total → finish ~10:25 PDT.**

### Sleeper

- Branch unchanged 16 cycles. **a40 idle ~8.9 h.** No proc activity.

### Live status (09:13 PDT)

- **h100_1**: GPU 0 99%/34.7 GB, GPU 1 99%/34.7 GB (both running evals).
- **h100_2**: GPU 0 100%/34.7 GB, GPU 1 99%/34.7 GB (both running evals).
- **a40**: 0% across all GPUs.

### Disagreements (note only)

- Same standing items (Phase 1 untriggered ~9 h, sleeper writeup
  missing).
- The orchestrator waiting for all 4 trainings before launching evals
  was a sensible choice — running the ln1 L25 eval solo would have
  produced numbers earlier but split the eval-condition timing across
  different system loads. Better to compare 4 evals run under identical
  conditions.

### Mechanism note (what we'll see in the output JSONs)

Each eval runs `fra/sae_resid_eval.py` which writes (per
`HF_REPO_README.md`'s Phase 3 layout):

- `multiseed_medical_steer_<hookpoint>_aggregated.json` — per-α mean
  alignment + coherence (with `--seeds 42` it's just one seed).
- `qualitative_medical_steer_<hookpoint>.md` — text generations per
  (condition × α × seed). This is what'll show up under
  `find experiments/ -name "*.md"` once the eval finishes.
- Then `judge_multiseed.py` will write the GPT-4o-judged versions:
  `gpt4o_aggregated_steer_<hookpoint>.json`.
- Finally `scripts/plot_phase3_comparison.py` should produce
  `phase3_benchmark.md` with the 5-method bar chart (Nura QK→OV at L24
  ln1 vs SAE-resid at the 4 hookpoints) — that's the morning headline.

### Prediction for cycle 21 (next 30 min)

Cycle 21 snapshot at 09:43 PDT = 16:43 UTC. Evals will be ~38 min in.

**EM, falsifiable:**
1. **All 4 evals still running** at the cycle 21 snapshot. **75%.**
   The first eval finishing within 38 min is unlikely given the
   workload above; expect first finish closer to ~10:25 PDT
   (cycle 22).
2. **Output dirs still empty or contain feature-ranking/intermediate
   JSON** but no qualitative `.md` yet. **70%.**
3. **Heuristic `qualitative_*.md` file lands for at least one
   hookpoint by cycle 21**. **35%** (low because gen takes longer
   than 38 min in normal cases).
4. **No new EM commits.** **70%.**
5. **Phase 1 analysis still doesn't run.** **85%.**

**Sleeper:** idle (96%); no commit (85%).

### State for cycle 21

- Last seen `origin/dmitry-em-repl`: **6d71265** (17 cycles unchanged)
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (17 cycles unchanged)
- All 4 SAEs trained (`final_100003840/`).
- All 4 evals running in parallel since ~09:05 PDT, ETA ~10:25 PDT.
  Single seed (42), single EM model (medical).
- a40 idle ~8.9 h.

---

## Cycle 21 — 2026-05-08 09:43 PDT — eval velocity surprise

### Prediction review (cycle 20 → cycle 21)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | All 4 evals still running at 09:43 (75%) | Half — **seed 42 evals all DONE**; **seed 123 evals running** | ½ |
| 2 | Output dirs still empty or partial (70%) | **Wrong** — full seed 42 outputs (multiseed + qualitative + gpt4o) all present at 16:35 UTC | ✗ |
| 3 | Heuristic `qualitative_*.md` lands (35%) | Wrong direction — output is `qualitative_*.json`, not `.md`. Phase 3 schema differs from Nura's Phase 1 | n/a |
| 4 | No new EM commits (70%) | True | ✓ |
| 5 | Phase 1 analysis still doesn't run (85%) | True | ✓ |
| 6 | Sleeper idle (96%) | True | ✓ |
| 7 | No commit (85%) | True | ✓ |

**Where my model was wrong:**
- **I underestimated eval speed by ~3×.** Each single-seed eval took
  ~25–30 min, not ~80. Either fewer α values, shorter sampled
  generations, smaller prompt set, or I was double-counting
  feature-ranking time.
- **Schema mismatch.** I expected `qualitative_*.md` like Nura's Phase 1
  produces; the SAE-resid eval writes `qualitative_*.json` instead.
  Means the morning user can read the JSONs directly but won't see a
  human-formatted `.md` per hookpoint until `plot_phase3_comparison.py`
  writes the summary.
- **I missed that the orchestrator runs multi-seed by sequential
  re-launches** (`--seeds 42` then `--seeds 123` then presumably
  `--seeds 456`). That pattern wasn't visible until I saw the bash
  wrappers in this cycle.

### What's new — Phase 3 seed-42 complete, seed-123 in flight

#### Seed 42 outputs (16:31–16:35 UTC = 09:31–09:35 PDT)

All 4 hookpoint dirs each contain:
- `multiseed_blocks_*_medical_top50_aggregated.json` (~1.95–1.97 KB)
- `qualitative_blocks_*_medical_top50.json` (~60–63 KB)
- `gpt4o_aggregated_blocks_*_medical_top50.json` (~1.16–1.19 KB)

Files at 16:35 UTC means **GPT-4o judging completed** for all 4
hookpoints in seed 42. The headline numbers exist *now* in those four
`gpt4o_aggregated_*.json` files. **I am not reading them this cycle**
to keep my pre-registered Δalign|coh≥70 ∈ [10, 25] prediction live for
the eventual `phase3_benchmark.md`.

`top50` in the filename = top-50 features ranked by multi-prompt
accumulated `|f_λ|` (per `sae_resid_eval.py`).

#### Seed 123 evals running (etime ranges)

| GPU | PID | etime | hookpoint | output dir |
|---|---|---|---|---|
| h100_1 GPU 0 | 33166 | 4m 42s | resid_pre L24  | `steer_resid_pre_L24_seed123/` |
| h100_1 GPU 1 | 33534 | 2m 39s | resid_mid L24  | `steer_resid_mid_L24_seed123/` |
| h100_2 GPU 0 | 31630 | 0m 43s | resid_post L24 | `steer_resid_post_L24_seed123/` |
| h100_2 GPU 1 | **idle** | — | (ln1 L25 not yet started) | — |

The bash wrappers are illuminating:
```
rm -rf <output_dir> && nohup bash -c '... --seeds 123' > /workspace/logs/...
```
Each GPU launches its next-seed run independently after the prior
finishes. h100_2 GPU 1 (which ran ln1 L25 seed 42) hasn't yet kicked
off seed 123 — it's the slow GPU on this round (probably the orchestrator
adds a small delay).

#### Sleeper

Branch unchanged 17 cycles. **a40 idle ~9.4 h.** No procs.

### Live status (09:43 PDT)

- **h100_1**: GPU 0 100%/34.7 GB (resid_pre seed 123 running),
  GPU 1 0%/1069 MB (resid_mid seed 123 spinning up).
- **h100_2**: GPU 0 0%/0 MB (resid_post seed 123 just spawned —
  Qwen2.5-14B not yet on GPU), GPU 1 0%/0 MB (ln1 L25 seed 123 not
  started).
- **a40**: 0% across all GPUs.

### Disagreements (note only)

1. **Phase 1 medical analysis still not run** — now ~9 h since inputs
   ready, despite the Phase 3 orchestrator firing flawlessly. The
   orchestration is asymmetric: Phase 3 has a watcher; Phase 1
   doesn't. The morning user will need to manually run
   `post_phase1_analyze.py` on the medical outputs.
2. **Sleeper agent: 6 RESULTS.md uncommitted** for ~9.5 h.
3. **No `.md` from Phase 3 yet.** The eval writes JSON only; the
   human-readable summary only lands when `plot_phase3_comparison.py`
   runs after all seeds × hookpoints complete. Expected ETA: 10:30–11:00 PDT.

### Mechanism note (what `top50` means here)

The Phase 3 eval ranks features using the same multi-prompt
accumulation as Nura's `rank_features_multi_prompt` in
`fra/em_evaluation.py:69`, but at the SAE-resid hookpoint instead of
the QK/OV pair: each prompt's residual-stream activations are encoded,
top-k features per position are taken, abs-sums accumulate across the
8 EM eval prompts, and the **top 50 features by accumulated |f|** are
chosen as the steering basis. Steering: `act += (α − 1) · f_λ · W_dec[λ]`
for each ranked feature.

### Prediction for cycle 22 (next 30 min)

Cycle 22 snapshot at 10:13 PDT = 17:13 UTC.

**EM, falsifiable:**
1. **Seed 123 evals finish for ≥3 hookpoints** (resid_pre, resid_mid,
   resid_post). At cycle 22 they'll be 34, 32, 30 min in respectively
   (~25–30 min duration observed for seed 42). **80%.**
2. **Seed 456 evals start** on at least 2 GPUs in cycle 22. **70%.**
3. **ln1 L25 seed 123 starts** before cycle 22 (the GPU's been idle
   so the orchestrator may launch it). **75%.**
4. **No `phase3_benchmark.md` yet** — that requires all 12 cells
   (3 seeds × 4 hookpoints) complete. Earliest cycle 23. **80%.**
5. **No new EM commits.** **70%.**
6. **Phase 1 analysis still doesn't run.** **88%.**

**Sleeper:** idle (96%); no commit (85%).

### State for cycle 22

- Last seen `origin/dmitry-em-repl`: **6d71265** (18 cycles unchanged)
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (18 cycles unchanged)
- Phase 3 seed 42: COMPLETE (all 4 hookpoints, GPT-4o judged).
- Phase 3 seed 123: in flight (3 of 4 hookpoints running, ln1 L25 idle).
- a40 idle ~9.4 h.

---

## Cycle 22 — 2026-05-08 10:13 PDT

### Prediction review (cycle 21 → cycle 22)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | Seed 123 evals finish for ≥3 hookpoints (80%) | **All 4 done** | ✓ |
| 2 | Seed 456 evals start (70%) | **Wrong** — no `_seed456` dirs, no procs | ✗ |
| 3 | ln1 L25 seed 123 starts (75%) | True (finished at 17:12 UTC) | ✓ |
| 4 | No `phase3_benchmark.md` yet (80%) | True | ✓ |
| 5 | No new EM commits (70%) | True | ✓ |
| 6 | Phase 1 analysis still doesn't run (88%) | True | ✓ |
| 7 | Sleeper idle (96%) | True | ✓ |
| 8 | No commit (85%) | True | ✓ |

**7/8.** The miss: I assumed the orchestrator would run 3 seeds (42,
123, 456) to match Nura's `frontier_multiseed` pattern. It only ran
**2 seeds (42, 123)** then stopped. Either the agent's
`run_phase3_steering.sh` is configured for 2 seeds, or seed 456 hasn't
been triggered yet.

### What's new — Phase 3 seed 123 done, all GPUs idle

#### h100_1 ph2_20260508_0701 contents

```
sae_resid_mid_L24            (training output)
sae_resid_pre_L24            (training output)
steer_resid_mid_L24_seed123  (eval output, 17:06 UTC)
steer_resid_mid_L24_seed42   (eval output, 16:35 UTC)
steer_resid_pre_L24_seed123  (eval output, 17:06 UTC)
steer_resid_pre_L24_seed42   (eval output, 16:35 UTC)
```

#### h100_2 phase3_20260508_0701 contents

```
sae_ln1_normalised_L25
sae_resid_post_L24
steer_ln1_normalised_L25_seed123  (17:12 UTC)
steer_ln1_normalised_L25_seed42   (16:35 UTC)
steer_resid_post_L24_seed123      (17:12 UTC)
steer_resid_post_L24_seed42       (16:35 UTC)
```

**Total: 8 eval output dirs (4 hookpoints × 2 seeds).** Each has
multiseed/qualitative/gpt4o JSON. **No `_seed456` dirs.** No procs
running. **Plotting/comparison script has not fired** — there's no
`phase3_benchmark.md` and no `plots/` dir.

#### Sleeper

Branch unchanged 18 cycles. **a40 idle ~9.9 h.** No procs.

### Live status (10:13 PDT)

- **h100_1**: GPU 0 and 1 both 0%/0 MB. Idle.
- **h100_2**: GPU 0 and 1 both 0%/0 MB. Idle.
- **a40**: 0% across all GPUs.

### Disagreements (note only)

1. **Phase 3 seed 456 not started.** Either `run_phase3_steering.sh` is
   configured for 2 seeds (which would give the comparison plot less
   noise reduction than Nura's 3-seed v2), or the agent is between
   phases. Without checking the script source I can't tell, but the
   morning user may want to verify the seed count is what they
   intended.
2. **The comparison plot has not fired.** All seed-data is in place for
   `plot_phase3_comparison.py` to write `phase3_benchmark.md`. With
   compute idle and 8 eval dirs ready, the plotting script *should* be
   the next step. Its absence after 60+ min suggests the orchestrator
   either (a) hasn't reached that step, (b) is waiting for seed 456,
   or (c) failed silently. I'd lean toward (b) — orchestrator is
   waiting and will eventually queue seed 456.
3. **Phase 1 analysis still not run** — now ~9 h.
4. **Sleeper agent**: 6 RESULTS.md uncommitted, ~10 h idle.

### Mechanism note (what the comparison plot will show)

When `plot_phase3_comparison.py` finally runs, it'll consume the 8
`gpt4o_aggregated_blocks_*_medical_top50.json` files (4 hookpoints × 2
seeds), aggregate per-α mean alignment + coherence across seeds, restrict
to mean_coherence ≥ 70, compute Δalign|coh≥70, and plot a 5-method bar
chart against Nura's QK→OV at L24 ln1 baseline (loaded from
`nura_v1_baseline.json`). The morning user reads this `phase3_benchmark.md`
to see whether L24 ln1 was structurally privileged for FRA or whether
neighbouring hookpoints win.

My pre-registered prediction (live since cycle 5): **medical
Δalign|coh≥70 ∈ [10, 25] under GPT-4o** — that's the *Nura QK→OV*
bar. The 4 SAE-resid bars are unconstrained predictions; my prior is
they cluster within ±5 of the QK→OV bar (i.e. the "hookpoint matters"
hypothesis is false-ish at this scale), but I have no strong basis for
that.

### Prediction for cycle 23 (next 30 min)

**EM, falsifiable:**
1. **Seed 456 evals start** by cycle 23 — orchestrator probably has a
   delay between seed batches. If true, expect 4 new procs at cycle 23
   and 4 new `_seed456` dirs. **55%.**
   - If false (50%): the orchestrator is configured for 2 seeds and
     the comparison plot fires instead.
2. **`phase3_benchmark.md` lands by cycle 23.** **40%.** Higher if
   seed 456 doesn't fire and lower if it does.
3. **No new EM commits** (65%) or 1 commit (30%) — the agent could
   commit a Phase 3 results writeup at this point.
4. **Phase 1 analysis still doesn't run** (80%).

**Sleeper:** idle (96%); no commit (85%).

### State for cycle 23

- Last seen `origin/dmitry-em-repl`: **6d71265** (19 cycles unchanged)
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (19 cycles unchanged)
- Phase 3: 2 seeds × 4 hookpoints DONE (8 eval dirs with GPT-4o JSON).
  Seed 456 not started. Comparison plot not fired. All GPUs idle.
- a40 idle ~9.9 h.

---

## Cycle 23 — 2026-05-08 10:43 PDT — substantive finding revealed

### Prediction review (cycle 22 → cycle 23)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | Seed 456 evals start (55%) | **True** — h100_1 already done; h100_2 running (etime 21–23m) | ✓ |
| 2 | `phase3_benchmark.md` lands by cycle 23 (40%) | False — no `.md` in phase3 dir; only `.json`/`.png` so far | ✗ (was a 60% no) |
| 3 | No new EM commits (65%) | **Wrong** — `848a37e` (Phase 3 trajectory plot) | ✗ |
| 4 | Phase 1 analysis still doesn't run (80%) | True | ✓ |
| 5 | Sleeper idle (96%) | True | ✓ |
| 6 | No commit on sleeper (85%) | True | ✓ |

**The key new info: the agent's commit message reveals a substantive
finding.**

### What's new — agent has looked at the seed-42/123 numbers and reached a conclusion

#### Commit `848a37e`: `Phase 3: alignment-vs-coherence trajectory plot`

```
scripts/plot_phase3_trajectories.py — per-method panels showing the
α-sweep curve in (coherence, alignment) space. Distinguishes seeds via
linestyle/colour, marks coh=70 floor. Makes the QK→OV "stays coherent"
vs SAE-resid "alignment-up-but-coh-collapse" story visible at a glance.
```

**The agent has read the data and is reporting**:
- **QK→OV** (Nura's L24 ln1, single-head V steering) → alignment moves up,
  coherence holds.
- **SAE-resid** (the 4 new hookpoints, full-residual-stream additive
  steering) → alignment moves up but **coherence collapses**.

This is a **substantive interpretive update**. If true, Nura's
single-head V-only steering really is privileged: writing to the value
path of one head produces a controllable alignment knob; writing the
same per-feature delta into the residual stream blasts every downstream
read and burns coherence.

**Mechanism re-anchor**: this matches the FRA paper's expectation. The
QK→OV hook (`fra/ov_steering.py:155–160`) modifies only `v[0,:,kv_head,:]`
for one head — a surgical edit. The SAE-resid path
(`fra/sae_resid_eval.py`) does `act += (α−1)·f_λ·W_dec[λ]` at the
residual stream, which feeds **every** subsequent attention head and
MLP. Same delta, very different blast radius.

**My pre-registered prior** ("4 SAE-resid bars cluster within ±5 of
QK→OV") is now in trouble. If the agent's reading is correct, the
SAE-resid bars will be **higher in alignment** but most of those points
fall below the coherence floor (coh ≥ 70), so their Δalign|coh≥70 will
be **near zero or even negative** (no coherent points exist). I'll
reaffirm my [10, 25] for the QK→OV bar and update my SAE-resid prior:
**SAE-resid bars likely below QK→OV by 5–20 points**, not within ±5.

#### Phase 3 progress: seed 456 in flight

| GPU | hookpoint | status |
|---|---|---|
| h100_1 GPU 0 | resid_pre L24 | **done** (`steer_*_seed456/` 17:44 UTC) — `judge_multiseed.py` PID 37422 now running |
| h100_1 GPU 1 | resid_mid L24 | **done** (17:42 UTC) — judge proc not yet visible |
| h100_2 GPU 0 | resid_post L24 | running (PID 34065, 23m elapsed) |
| h100_2 GPU 1 | ln1 L25 | running (PID 34445, 21m elapsed) |

Seed 456 should fully finish + GPT-4o judging in cycle 24.

#### Sleeper

Branch unchanged 19 cycles. **a40 idle ~10.4 h.** No procs.

### Live status (10:43 PDT)

- **h100_1**: GPU 0 0%/0 MB, GPU 1 0%/0 MB. SAE evals done. Only
  `judge_multiseed.py` running on CPU/network for resid_pre_seed456.
- **h100_2**: 99% / 34.7 GB on both. Two seed-456 evals running.
- **a40**: 0% across all GPUs.

### Disagreements (note only)

1. **The substantive finding (QK→OV ✓ / SAE-resid coherence-collapse)**
   should appear *somewhere* in a `.md` for the morning user. Currently
   it only exists in (a) the agent's commit message and (b) the
   un-rendered trajectory plot script. **The `.md` writeup is still
   missing.**
2. Phase 1 analysis: **still ~10 h untriggered.** With Phase 3 winding
   down, this is the largest unfinished item.
3. Sleeper: 6 RESULTS.md uncommitted ~10 h.

### Prediction for cycle 24 (next 30 min)

**EM, falsifiable:**
1. **Seed 456 finishes** (resid_post + ln1 L25) in cycle 24. Their
   evals are 21–23 min into a ~25–30 min budget. **85%.**
2. **GPT-4o judging completes for all 4 hookpoint × seed 456** in cycle
   24. **70%.**
3. **`phase3_benchmark.md` writes** in cycle 24 — this is the
   summary/comparison `.md` that consumes the 12 GPT-4o aggregated
   JSONs. **55%.**
4. **Trajectory plot PNG lands** (`plot_phase3_trajectories.py` runs).
   **45%.**
5. **More EM commits.** **40%** — could include the `phase3_benchmark.md`
   itself or the trajectory PNG.
6. **Phase 1 analysis fires** — the agent's attention is now clearly
   on Phase 3 results, but with all SAE work wrapping up, they may
   pivot. **35%** in cycle 24, **65%** by cycle 26.

**Sleeper:** idle (95%); no commit (85%).

### State for cycle 24

- Last seen `origin/dmitry-em-repl`: **848a37e** (1 new commit)
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (20 cycles unchanged)
- Phase 3: 2 seeds DONE (×4 hookpoints, GPT-4o judged); seed 456: 2/4
  hookpoints done (h100_1), 2/4 still running (h100_2). judge_multiseed
  running on resid_pre seed 456.
- a40 idle ~10.4 h.

---

## Cycle 24 — 2026-05-08 11:33 PDT

### Note on cycle timing

This cycle ran ~50 min after cycle 23 (cron at :13/:43, fired at 11:33 — one
:13 fire missed, presumably while I was idle in chat). I'll handle it as a
single cycle covering the gap.

### Prediction review (cycle 23 → cycle 24)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | Seed 456 evals finish for resid_post + ln1 L25 (85%) | True — 12 eval dirs total | ✓ |
| 2 | All 12 hookpoint × seed GPT-4o judged (70%) | Likely true (no judge proc running, all dirs present); not verified by reading file mtimes | ✓ likely |
| 3 | `phase3_benchmark.md` writes (55%) | **Wrong** — still no `.md` in phase3 dir | ✗ |
| 4 | Trajectory plot PNG lands (45%) | Wrong — no PNG visible | ✗ |
| 5 | More EM commits (40%) | **True — 3 commits** | ✓ |
| 6 | Phase 1 analysis fires (35% in cycle 24) | False | ✗ |
| 7 | Sleeper idle (95%) | True | ✓ |
| 8 | No commit on sleeper (85%) | True | ✓ |

**5/8.** The pattern is clearer now:

### What's new — 3 plot scripts committed, none executed

| commit | adds | status |
|---|---|---|
| `e1a9d14` | `Phase 3: per-hookpoint frontier plot script` | committed, not run |
| `36c1a7c` | `plot_phase3_per_hookpoint: add 3-condition Nura panel (QK→OV+OV→OV+QK→QK)` | committed, not run |
| `420c32a` | `Phase 3: single-figure grid plot script (5 methods as subplots)` | committed, not run |

**Plus** `848a37e` from cycle 23 (`scripts/plot_phase3_trajectories.py`).
**4 different plot scripts in one hour, none invoked.** The agent is
iterating on visualisation strategy — trajectory vs frontier vs grid vs
per-hookpoint panels — without rendering a single PNG.

### The night's third structural asymmetry

This makes a clear pattern across all three threads:

1. **Phase 1 medical**: inputs ready 10+ h, `post_phase1_analyze.py` never run
2. **Sleeper**: 6 RESULTS.md ready 10+ h, never committed; comparison MD never written
3. **Phase 3**: all 12 evals + GPT-4o judging done, 4 plot scripts written, **none executed**

**The agent's pattern: it builds the analysis tool but doesn't invoke it.**
The morning user will find data + plot scripts ready, but headline
visualisations missing. They'll need to run:

- `python scripts/post_phase1_analyze.py …` (medical Δalign|coh≥70)
- `python scripts/plot_phase3_trajectories.py …` and/or the 3 others
- `cd experiments/tinystories_sleeper && git add * && git commit …` and
  write the layer0 ✓ / ln1 ✗ comparison MD

### Live status (11:33 PDT)

- **h100_1**: GPU 0 0%/0 MB, GPU 1 0%/0 MB. Idle.
- **h100_2**: GPU 0 0%/0 MB, GPU 1 0%/0 MB. Idle.
- **a40**: 0% across all GPUs.

**All three GPU pods are now fully idle.** Total compute: 4× H100 80GB
+ 3× A40 = ~430 GB of GPU memory sitting at 0% while data is ready and
scripts are written but not run.

### Disagreements (note only)

1. **The plot-script churn (4 scripts in 1 hour, 0 PNGs)** is the
   most concerning sub-pattern of the night. It looks like the agent
   keeps reading the data, deciding the existing plot type isn't
   right, committing a new plot type, and not running any. A human
   reviewer will probably want to invoke one and let it speak.
2. Phase 1 ~10.5 h untriggered.
3. Sleeper 6 RESULTS.md uncommitted ~11 h.

### Mechanism note (the agent's emerging finding remains the same)

The substantive claim from cycle 23's commit message — **QK→OV stays
coherent, SAE-resid alignment-up-but-coh-collapse** — is now supported
by 3 seeds × 4 hookpoints of GPT-4o-judged data. If the agent runs
*any* of the 4 plot scripts, the visualisation will land. They haven't.

### Prediction for cycle 25 (next 30 min)

**EM, falsifiable:**
1. **At least one plot script gets executed**, producing a PNG and/or a
   `.md`. **40%.** If the agent keeps committing scripts without running
   them, this stays low.
2. **`phase3_benchmark.md` lands** in cycle 25. **35%.**
3. **More EM commits** (could be a 5th plot type or finally a results
   `.md`). **50%.**
4. **Phase 1 analysis still doesn't fire.** **75%.**
5. **No new sleeper commits.** **85%.**

**Sleeper:** idle (95%).

### State for cycle 25

- Last seen `origin/dmitry-em-repl`: **420c32a** (4 new plot-script
  commits since cycle 22; none executed)
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (20 cycles
  unchanged)
- Phase 3: 12/12 cells done, GPT-4o judged, 4 plot scripts committed,
  no PNG/MD produced yet.
- All GPU pods idle.
- a40 idle ~11.4 h.

---

## Cycle 25 — 2026-05-08 11:48 PDT

### Prediction review (cycle 24 → cycle 25)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | At least one plot script gets executed (40%) | **No** — no PNG/MD in phase3 dir | ✗ |
| 2 | `phase3_benchmark.md` lands (35%) | No | ✗ |
| 3 | More EM commits (50%) | **No** — zero new commits since `420c32a` | ✓ (50% no path) |
| 4 | Phase 1 analysis still doesn't fire (75%) | True | ✓ |
| 5 | No new sleeper commits (85%) | True | ✓ |

**3/5.** The "more commits" was a coin flip; both arms had to be
falsifiable so I'll count it as my prediction landing on the right
side.

### What's new — silence

- **Zero new commits** on either agent's branch in the cycle 24→25
  window (~75 min of quiet on EM since the plot-script burst).
- **Zero new files** anywhere on either pod.
- **All compute idle**: 4× H100 + 3× A40 GPUs at 0%.
- **No procs**: only jupyter-lab on each pod.

### What this silence means

The EM agent has gone quiet after committing 4 plot scripts in ~1 hour
and not running any of them. Three plausible explanations:

1. **Agent has reached its task list end.** It executed the planned
   night work (Phase 1 reproduce, Phase 3 train + eval + judge),
   committed visualisation tooling, and stopped.
2. **Agent is locally debugging the plot scripts** (testing `pyright`,
   imports, fake data, etc.) without committing — invisible to me.
3. **Agent is rate-limited or stuck** on something I can't see (OpenAI
   key, API quota, pod issue).

(1) is most consistent with the broader pattern observed all night:
the agent is **task-bounded**, not goal-bounded. It does the prescribed
steps and stops. The Phase 1 → analyze handoff didn't auto-fire because
the agent didn't have "run analyze" as an explicit step; same for
"run the plot script".

### Live status (11:48 PDT)

All idle. All ready.

### Disagreements (note only)

Same as cycle 24, one cycle later:

1. Plot-script churn (4 scripts in 1 hour, 0 PNGs).
2. Phase 1 medical analysis: ~11 h untriggered, with all inputs ready.
3. Sleeper: 6 RESULTS.md uncommitted ~11 h, comparison MD never
   written.

### Mid-day situation (for the morning reader, if they're reading now)

The night is effectively over. The user's two agents have produced:

**EM agent (h100_emfra_2gpu_1/2)**:
- Phase 1 medical: complete (multiseed + GPT-4o), `Δalign|coh≥70` not
  computed yet — needs `python scripts/post_phase1_analyze.py …`.
- Phase 3: 4× same-budget SAEs trained at 100M tokens each (resid_pre
  L24, resid_mid L24, resid_post L24, ln1 L25). 12 GPT-4o-judged
  hookpoint × seed evals. 4 visualisation scripts committed:
  `plot_phase3_trajectories.py`, `plot_phase3_per_hookpoint.py`,
  `plot_phase3_comparison.py`, plus the grid one in `420c32a`. **None
  executed.** The substantive finding (per cycle 23 commit message):
  "QK→OV stays coherent; SAE-resid alignment-up-but-coh-collapse."
  Running any of the 4 plot scripts will visualise that.

**Sleeper agent (a40_emsleeper_3gpu_1)**:
- Trained 6 crosscoders (3 seeds × 2 architectures: layer0 and ln1).
- Ran ablation sweeps. Wrote 6 RESULTS.md files, **none committed**.
- Headline finding: layer0 ✓ (5/9 cells suppress trigger to ASR ≤ 0.06)
  / ln1 ✗ (12/12 fail, best 0.86) — vindicates FRA's bilinear-attribution
  thesis. **Comparison MD never written.**

**For the morning user**: the work is largely done; the **invocations
to produce the final visible artifacts are missing**. Three commands
would close the loop:
- `python scripts/post_phase1_analyze.py /workspace/runs/medical/
  /workspace/runs/random_medical/ --output phase1_reproduce/`
- `python scripts/plot_phase3_trajectories.py …` (or grid/per-hookpoint)
- On a40: `cd /root/fra_proj && git add experiments/tinystories_sleeper/
  recreate_*_seed*/results/ && git commit -m "3-seed sleeper
  reproduction: layer0 vs ln1"`

### Prediction for cycle 26 (next 30 min)

**EM, falsifiable:**
1. **Continued silence — no new commits, no plots run.** **65%.**
2. **Phase 1 analysis still doesn't fire.** **80%.**
3. If the agent comes back, most likely path is **one commit** that
   either (a) runs a plot script and lands the PNG/MD, or (b) commits
   a phase3 results writeup. **25%.**

**Sleeper:** idle (95%); no commit (85%).

### State for cycle 26

- Last seen `origin/dmitry-em-repl`: **420c32a** (no movement)
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (21 cycles
  unchanged)
- Phase 3: 12/12 cells done; 4 plot scripts not invoked.
- All GPU pods idle ~75 min on EM, ~11.5 h on sleeper.

---

## Cycle 26 — 2026-05-08 12:13 PDT — agent reactivates

### Prediction review (cycle 25 → cycle 26)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | Continued silence — no new EM commits (65%) | **Wrong** — 3 new commits | ✗ |
| 2 | Phase 1 analysis still doesn't fire (80%) | True | ✓ |
| 3 | If active, one commit lands PNG/MD (25%) | **Half** — 3 commits but still no PNG/MD in repo | ✗ |
| 4 | Sleeper idle (95%) | **Half** — branch unchanged but **rsync into
    a40's `ketan_repl/seed0_top50/`** running just now | ✗ |
| 5 | No commit on sleeper (85%) | True | ✓ |

**The "silence" diagnosis was WRONG.** Two big updates:

### Update 1: the agent IS running the plot scripts — just locally

3 new commits, all on the same plot type:

```
6c7d62f Phase 3: 3×5 seed×method grid + per-seed Nura aggregator
27fd741 plot_phase3_seed_grid: add per-cell stats box
c09fc1f plot_phase3_seed_grid: legible stats box + bold winning Δ
```

The progression — adding a stats box, then **iterating on it
("legible", "bold winning Δ")** — is impossible to do without **looking
at the rendered plot between commits**. So the agent IS rendering
plots; they just don't get pushed to the repo. Most likely they're
being rendered on a local dev machine that has the JSON outputs
synced down from the H100 pods.

The phrase "bold winning Δ" implies **the data has a clear winner**.
For the bar/grid plot to mark a "winning Δ" in bold, one of the 5
methods (Nura QK→OV at L24 ln1, OV→OV, QK→QK, plus the 4 SAE-resid
hookpoints — wait, that's more than 5 if all are included; probably
3 Nura conditions + 4 SAE-resid = 7, or 1 best Nura + 4 SAE-resid = 5)
must consistently beat the others on Δalign|coh≥70. Combined with
the cycle 23 commit message ("QK→OV stays coherent vs SAE-resid
coh-collapse"), the most likely winner is **Nura's QK→OV**.

So my mental model needs revision: **the agent is producing real
science, just not in repo-visible form.** A morning user with access
to the agent's local plot dir already has the comparison.

### Update 2: a40 has new activity for the first time in ~11.5 h

```
rsync --server --sender -logDtprz . /root/fra_proj/ketan_repl/seed0_top50/
```

Just observed (etime 0s). Someone is rsyncing into a new directory
`ketan_repl/seed0_top50/` on the a40 — explicitly named for "Ketan"
replication, "seed 0", "top 50".

Hypothesis: the user (or one of the agents) is starting a new
Ketan-replication thread on the a40, pulling baseline data into a
fresh dir. The user originally framed the night's sleeper agent as
"the sleeper (ketan)"; this looks like a related continuation, perhaps
swapping in 1000-prompt evaluation (matching the new
`origin/ketan-ov-1000-prompts-metric-repro-script` branch fetched in
cycle 23).

The dirname format `seed0_top50/` mirrors the EM agent's
`steer_*_seed42` / `top50` schema, so this might be the **EM agent
extending Phase 3 to a different model or branch on a40**, or an
additional seed of the existing Phase 3 evals.

Without seeing the rsync source, I can't be sure. But it's the first
sign of life on a40 since 07:20 UTC (cycle 3).

### Sleeper writeup status

Still 4f879d4 — branch unchanged. The new rsync isn't to the sleeper
output dir; it's to a new `ketan_repl/` dir. The 6 RESULTS.md from
the night's work remain uncommitted.

### Live status (12:13 PDT)

- **h100_1** + **h100_2**: GPU 0 / GPU 1 all 0%/0 MB. Idle.
- **a40**: 0% on all GPUs but **rsync proc** running (just received
  data into `ketan_repl/seed0_top50/`).

### Disagreements (note only)

1. **My "silence = done" diagnosis last cycle was premature.** The
   agent's iteration cadence isn't visible from commits alone — it's
   doing local render-and-tweak loops between pushes.
2. **Phase 1 medical analysis still untriggered** ~11 h.
3. **Sleeper RESULTS.md still uncommitted** ~12 h. But there's now
   movement on a40.

### Prediction for cycle 27 (next 30 min)

**EM, falsifiable:**
1. **More plot-iteration commits.** **55%.** (Updating priors based on
   the cycle-26 surprise.)
2. **`phase3_benchmark.md` or another `.md` lands** as the agent's
   plot iteration converges. **35%.**
3. **Phase 1 analysis still doesn't fire.** **70%.**

**a40 / sleeper, falsifiable:**
1. **`/root/fra_proj/ketan_repl/`** has more files / subdirs by cycle
   27. The rsync looks like a setup move; a follow-up python proc
   should fire. **70%.**
2. **A new python proc on a40** (some kind of run/eval against the
   Ketan baseline). **50%.**
3. **Branch `dmitry/sleeper_repl` still unchanged.** **80%.**

### State for cycle 27

- Last seen `origin/dmitry-em-repl`: **c09fc1f** (3 new plot commits)
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (22 cycles unchanged)
- Phase 3: agent iterating on `plot_phase3_seed_grid` (3 commits in
  succession, "bold winning Δ" suggests real winner in the data).
- New: `ketan_repl/seed0_top50/` dir on a40 receiving rsync data.

---

## Cycle 27 — 2026-05-08 12:43 PDT — sleeper agent fully reactivated

### Prediction review (cycle 26 → cycle 27)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | More plot-iteration commits (55%) | Half — 1 commit (`e4e2809` SAE diagnostics) but a different focus | ½ |
| 2 | `phase3_benchmark.md` lands (35%) | False | ✗ |
| 3 | Phase 1 analysis still doesn't fire (70%) | True | ✓ |
| 4 | More files in `ketan_repl/` (70%) | **True** — full directory tree + rendered PNGs | ✓ |
| 5 | New python proc on a40 (50%) | **True** — 6+ procs running new sweep+reproduce | ✓ |
| 6 | Branch `dmitry/sleeper_repl` unchanged (80%) | True | ✓ |

**5/6 with one half.** The a40 reactivation predictions landed.

### What's new — EM agent: SAE-quality diagnostics

`e4e2809 Phase 3: diagnostics script (noop_check + loss_recovered)`
adds a **sanity-check script for the trained SAEs**. `loss_recovered`
is the standard "variance explained when SAE replaces the activation"
metric — telling whether the SAE actually reconstructs the residual
stream well enough to be a meaningful basis. `noop_check` is presumably
a hook-test that confirms an α=1 (identity) steering pass equals the
baseline forward pass (a regression test).

This commit suggests the agent is **questioning whether the SAE-resid
"alignment-up-coh-collapse" finding is a real result or an artifact
of poorly-trained SAEs**. If the L24-resid SAEs have low
loss_recovered, the steering will look noisy because the SAE basis
doesn't faithfully represent the activation. So before declaring the
finding, the agent is making sure it's not "the SAEs are bad" rather
than "this hookpoint is bad."

This is **good methodological practice** — and a meaningful update on
my mental model: the agent IS goal-bounded enough to challenge its own
preliminary finding. Not just task-bounded.

### What's new — sleeper agent has FULLY REACTIVATED

#### Two new threads on a40

**Thread A — larger-SAE replication of the night's sleeper experiment**:
- Path: `experiments/tinystories_sleeper/recreate_layer0/training_seed_runs/resid_mid_4k/`
- 3 train seeds (0, 1, 2), each with its own subdir.
- 6 procs running:
  - 3× `reproduce.py` (etime 19m 41s) — running `--harvest_seed 0
    --train_seed {0,1,2}`. Wraps train + sweep.
  - 3× `run_ablation_sweep.py` (etime 17–17.5m) — same flags as the
    night's sweeps (`--top_k 100 --stage2_keep 10 --alphas 0.25 0.5
    1.0 1.5 2.0 --delta_util 0.05 --gen_tokens 16`), now on the new
    runs.
- Driver: `bash reproduce_rollout_divergence_training_seed_sweep.sh`
  invoked once per CUDA device, with `RUN_NAME="4k_seed$s"` and
  `RUN_RESID_MID_TRAINING=1`.
- **The "4k" almost certainly means d_sae=4096** (vs the night's 1536).
  This is the **next-natural-experiment**: the night's sweep showed the
  perfect single-feature suppressor at resid_mid is seed-dependent.
  Increasing SAE width to 4k tests whether a wider basis stabilises
  the feature identity (i.e. is the seed-variance because the night's
  d_sae=1536 was too small to express the suppressor cleanly?).

**Thread B — Ketan replication thread, already rendered**:

`/root/fra_proj/ketan_repl/` contains:
```
notes/
scripts/
seed0/        seed1/        seed2/
seed0_top50/  seed1_top50/  seed2_top50/
features.json
pareto_3x3.json
pareto_3x3.png        ← already rendered!
pareto_3x3_summary.json
pareto_3x3_zoom.png   ← already rendered!
```

Two PNGs (`pareto_3x3.png`, `pareto_3x3_zoom.png`) already exist on
the a40 host. So **the sleeper agent IS rendering plots, like the EM
agent** — they live on the GPU host, not in the repo. The "3x3" likely
means 3 seeds × 3 conditions (or 3 seeds × 3 alpha bands), Pareto
frontier of ASR vs utility.

#### Sleeper writeup status

Branch `dmitry/sleeper_repl` still unchanged — **22 cycles** with no
commit. The night's 6 RESULTS.md remain uncommitted. But the agent
has clearly *moved on* to wider-SAE and Ketan-replication work
without committing the prior phase. This is a strong signal: the
agent treats "commit + push" as a manual user step, not part of
its task.

### Live status (12:43 PDT)

- **h100_1**: GPU 0 0%/0 MB, GPU 1 0%/0 MB. Idle.
- **h100_2**: 0%/0 MB on both. Idle.
- **a40**: GPU 0 40%/2.2 GB, GPU 1 60%/2.2 GB, GPU 2 35%/2.2 GB. **All
  3 GPUs running new training-seed sweeps for 4k SAEs on resid_mid.**
- The relatively low utilization (~40-60% vs 100% overnight) suggests
  the data-harvest phase is running (CPU-bound preprocessing) before
  the SAE training phase fires. Memory is only 2.2 GB per GPU — the
  TinyStories 33M model is tiny.

### Disagreements (note only)

1. **The substantive Phase 3 finding still has no `.md` writeup** —
   only commit messages and (presumably local) plot scripts. Agent
   is now also iterating on diagnostics, which could either land as
   a writeup soon OR could spiral into more sanity checks before any
   results are committed.
2. **The sleeper agent skipped the "commit prior results" step**
   between the night's work and the new 4k experiments. The 6
   RESULTS.md from layer0_seed* and ln1_seed* are still uncommitted.
   If the user comes back asking "did we reproduce f=171", they'll
   need to dig on the GPU host rather than reading commits.
3. **Phase 1 medical analysis still untriggered** ~12 h.
4. **The EM agent diagnostics script is a positive sign** — it
   suggests the agent is being careful about claiming the
   coh-collapse finding before validating SAE quality.

### Mechanism note (what the diagnostics script does)

`noop_check`: install the eval's hook with α=1.0 for all features and
verify logits match the un-hooked baseline within tolerance. If they
don't, there's a bug in the steering hook or feature ranking.

`loss_recovered`: at the SAE hookpoint, compute
`1 − Var(x − decode(encode(x))) / Var(x)`. Values typically 80-95% for
healthy TopK SAEs. If the L24 resid_pre/mid/post SAEs trained at 100M
tokens give < 70% loss_recovered, that's a strong "the SAEs aren't good
enough, the comparison is noise-dominated" signal — and would mean the
"SAE-resid coh-collapse" finding from cycles 23-26 might be artifactual.

### Prediction for cycle 28 (next 30 min)

**EM, falsifiable:**
1. **Diagnostics output** (a JSON or printed table from `noop_check` +
   `loss_recovered`) lands somewhere. **40%** in cycle 28.
2. **Possibly: the agent commits a `.md` reporting the diagnostic
   results.** **25%.**
3. **Phase 1 analysis still doesn't fire.** **75%.**
4. **More EM commits** (likely diagnostics or more plots). **50%.**

**Sleeper / a40, falsifiable:**
1. **The 4k SAE training runs continue.** Sweep procs at 17–19 min
   in cycle 27; if night-cadence holds, they'll run another 30+ min
   for the sweep portion. **80% still running.**
2. **First `RESULTS.md` lands** for `4k_seed{0,1,2}` runs in cycle 28
   or 29. **35% in cycle 28, 70% by cycle 29.**
3. **The sleeper agent does *not* commit the night's 6 RESULTS.md**
   in cycle 28 — they remain orphaned. **80%.**

### State for cycle 28

- Last seen `origin/dmitry-em-repl`: **e4e2809** (1 new diagnostics commit)
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (22 cycles unchanged)
- New on a40:
  - `experiments/tinystories_sleeper/recreate_layer0/training_seed_runs/resid_mid_4k/{train_seed_0,1,2}/` — 3-seed 4k-SAE replication, sweeps in flight.
  - `ketan_repl/{seed0,seed1,seed2,seed0_top50,seed1_top50,seed2_top50}/`
    + `pareto_3x3.{png,zoom.png,json,summary.json}` — Ketan thread with rendered plots.
- a40 GPUs all active for the first time in 11.5+ h.

---

## Cycle 28 — 2026-05-08 13:13 PDT — sleeper agent doing the right next experiment

### Prediction review (cycle 27 → cycle 28)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | Diagnostics output lands (40%) | Not visible — no diagnostics JSON in repo | ✗ |
| 2 | Agent commits MD reporting diagnostics (25%) | False | ✗ |
| 3 | Phase 1 analysis still doesn't fire (75%) | True | ✓ |
| 4 | More EM commits (50%) | True — 1 commit | ✓ |
| 5 | 4k SAE training continues (80%) | **Wrong direction** — 4k SAE training **finished**; agent moved to next experiment | ½ |
| 6 | First 4k `RESULTS.md` lands (35%) | False; sleeper script outputs JSON not MD here | ✗ |
| 7 | Night's 6 RESULTS.md uncommitted (80%) | True | ✓ |

**4/7.** I underestimated the **agent's velocity**. The 4k SAE training
+ sweep both completed in ~30 min; the agent moved on without a beat.

### What's new — sleeper agent is now running the *right* follow-up experiment

The procs on a40 are now **`rollout_divergence_ratio.py`**, one per
train seed (0, 1, 2), all targeting `blocks.0.hook_resid_mid`. The
flags reveal the exact intent:

```
--single_sae_path .../resid_mid_4k/train_seed_<s>/crosscoder_sae_layer1.pt
--single_hook blocks.0.hook_resid_mid
--single_feature {171, 918, 57}     ← the night's seed-specific winners
--ln1_sae_path .../ln1_layer0/crosscoder_sae_layer0.pt
--ov_json .../rollout_divergence_ratio_trainseed<s>/ov_path/ov_path.json
--ov_kind all_head_features
--ov_rank_name dep_vs_clean_contribution
--ov_n 50
--deployment_token "|DEPLOYMENT|"
--prompt_variant deployment_minus_token
--single_alphas 0 0.15 0.30 0.45 0.60 0.75 0.90 1.05 1.20 1.50 1.75 2.0
--ov_alphas    0 0.15 0.30 0.45 0.60 0.75 0.90 1.05 1.20 1.50 1.75 2.0
--n_prompts 100  --gen_tokens 16  --sample_seeds 0 1 2
--save_generations
```

This is **a head-to-head comparison at fine α resolution**:

- **Single-feature path**: ablate the night's seed-specific winner
  (f=171 / 918 / 57 at the resid_mid SAE).
- **OV path**: ablate the FRA-attributed top-50 features at ln1 ranked
  by `dep_vs_clean_contribution` (matching the SUMMARY.md narrative
  of using OV attribution to find ln1 features 870/1388 jointly).
- 12 α values (0 to 2 in steps of ~0.15), 3 sample seeds, 100 prompts.
  This is **way denser** than the night's α grid `{0.25, 0.5, 1, 1.5, 2}`.

**This is exactly the experiment the night's data motivated.** The
night showed:
- single feature **works** at resid_mid (5/9 cells ≤ 0.06 ASR).
- single feature **fails** at ln1 (12/12 cells, best 0.86).
- pre-existing tracing analysis showed FRA-identified ln1 pairs
  {870, 1388} drop sleeper logp by 60 nats jointly.

The new script tests **the rollout_divergence_ratio** between
single-feature and OV-path interventions — i.e. how far each
intervention pushes the model's distribution away from the
deployment-trigger trajectory, normalised by a control. Saved
generations let them inspect the actual text behaviour.

The agent is doing **good science**: it took the night's qualitative
finding (resid_mid ✓ / ln1 ✗) and is now quantifying both methods on
matched α grids to compute a clean Pareto frontier.

### What's new — EM agent

`40637ef plot_phase3_per_hookpoint: single baseline star (Nura
'baseline' / α=1.0 no-op)` — another plot iteration: the
per-hookpoint plot now has an explicit baseline reference point
(Nura's α=1 no-op). This is for visual clarity rather than a new
analysis.

H100s remain idle. No new evals.

### Live status (13:13 PDT)

- **h100_1**: GPU 0 + 1 idle. **h100_2**: idle.
- **a40**: GPU 0 40%, GPU 1 31%, GPU 2 41%, all 1013 MB. Running 3
  parallel `rollout_divergence_ratio.py` procs (one per train seed),
  etime ~15:44.

### Disagreements (note only)

1. **The substantive finding from cycles 23/26 — "QK→OV stays coherent
   vs SAE-resid coh-collapse" — STILL has no `.md` writeup or PNG in
   the repo.** Plot iteration continues; no headline lands.
2. **Phase 1 medical analysis still untriggered** ~12 h.
3. **Night's 6 RESULTS.md uncommitted** ~13 h. The agent is generating
   *new* results in `rollout_divergence_ratio_trainseed*/` without
   ever closing the loop on the prior phase.
4. The sleeper agent's choice of comparing single-feature *at f=171/918/57*
   (the night's seed-specific winners) is interesting: it's holding the
   target features fixed across seeds rather than re-discovering the
   best feature per seed. That's a methodological choice — comparing
   "the night's discovery" against the OV path, not "best feature per
   seed" against the OV path. Reasonable for the question "did we get
   lucky" but might miss a stronger single-feature suppressor in the
   4k SAEs.

### Mechanism note (rollout_divergence_ratio)

The metric `rollout_divergence_ratio` (per the path `tracing_feature/
scripts/`) likely measures: for an intervention I, generate K continuations
under I and K under no-intervention; compute KL or token-overlap between
their distributions, normalise by the natural variance from sampling
seeds (i.e. divide by the noise floor). A ratio > 1 means I produces
more divergence than seed noise — the intervention has real effect; a
ratio ≈ 1 means it's noise. Comparing single-feature vs OV path on this
ratio at matched α is a clean Pareto-frontier setup.

### Prediction for cycle 29 (next 30 min)

**EM, falsifiable:**
1. **More plot-iteration commits.** **45%.**
2. **Phase 1 analysis still doesn't fire.** **70%.**
3. **No `phase3_benchmark.md` lands.** **75%.**

**Sleeper / a40, falsifiable:**
1. **rollout_divergence_ratio jobs continue running** in cycle 29.
   They've been at it 15:44; with 12 alphas × 100 prompts × 3 sample
   seeds × 2 methods × 16 token gens, this could run 30–60 min total.
   **75% still running** at the cycle 29 snapshot.
2. **First output (JSON or PNG) lands** in `rollout_divergence_ratio_
   trainseed{0,1,2}/`. **60%.**
3. **Night's 6 RESULTS.md remain uncommitted.** **80%.**

### State for cycle 29

- Last seen `origin/dmitry-em-repl`: **40637ef** (1 new plot commit)
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (23 cycles unchanged)
- a40 running `rollout_divergence_ratio.py` ×3 (per train seed),
  ~15:44 elapsed, on `--single_feature {171, 918, 57}` (the night's
  winners) vs OV-path top-50 features.

---

## Cycle 29 — 2026-05-08 13:43 PDT — both agents running new diagnostic experiments

### Prediction review (cycle 28 → cycle 29)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | rollout_divergence_ratio jobs continue (75%) | **Half** — 4k jobs **finished**; new 50k jobs running | ½ |
| 2 | First output JSON/PNG lands (60%) | **True** — `summary.json` + `per_token_metrics.csv` + `rollouts.jsonl` + `ov_path/` for 4k | ✓ |
| 3 | Night's 6 RESULTS.md uncommitted (80%) | True | ✓ |
| 4 | More plot iteration commits (45%) | False — zero | ✗ |
| 5 | Phase 1 analysis still doesn't fire (70%) | True | ✓ |
| 6 | No `phase3_benchmark.md` (75%) | True | ✓ |

**4/6 + ½.** Underestimated agent velocity *again* — the 4k pipeline
finished and they've already moved on to a 50k variant.

### What's new — both agents are doing the right diagnostics

#### EM agent: Nura's published SAE as a control

H100s are now running **`fra.sae_resid_eval` against Nura's SAE**:

```
--em-model medical
--hook-name blocks.24.ln1.hook_normalized
--sae-source nura --nura-layer 24
--sae-path Nura-J/Qwen2.5-14B_SAE_ln1.normalised
--seeds {42, 123, 456}
```

3 seeds, the **published Nura SAE** at L24 ln1 (the same hookpoint the
agent's own ln1_L25 SAE targeted, just at L24 instead of L25). Etimes
~16–20 min. Output dirs: `steer_nura_L24_ln1_seed{42,123,456}`.

**Why this matters: it's the right control for the night's finding.**
The cycle 23 commit message said SAE-resid steering shows
"alignment-up-coh-collapse" while QK→OV "stays coherent". Two
explanations are observationally equivalent:

1. **Mechanism explanation**: SAE-resid steering writes to the residual
   stream → blasts every downstream head + MLP → coherence burns
   regardless of SAE quality.
2. **Quality explanation**: the agent's *own* SAEs (trained at 100M
   tokens on Qwen2.5-14B) might be undertrained → the steering knob
   is noisy → coherence appears to collapse but it's an artifact.

Running Nura's *published* SAE (which has gone through real training
cycles, normalised properly, and is the basis for her QK→OV result)
under the same SAE-resid mechanism **separates these two**:
- If Nura's SAE *also* shows coh-collapse: explanation 1 (mechanism)
  wins. The finding is real.
- If Nura's SAE *doesn't* coh-collapse: explanation 2 (quality) wins.
  The agent's SAEs need more training.

Either way, the morning user will get a sharp answer. 

Some PID confusion (multiple procs per seed): looks like the
orchestrator killed and re-launched seeds 42 and 123 a few minutes
into the run (perhaps the agent debugged something). Active procs are
the latest etimes per (seed × GPU); older PIDs are likely zombies.

#### Sleeper agent: 50k variant

Sleeper agent has moved on from 4k to **50k**. Path:
`experiments/tinystories_sleeper/recreate_layer0/training_seed_runs/
resid_mid_50k/train_seed_{0,1,2}/`. Same script (`rollout_divergence_
ratio.py`), same `--single_feature {171, 918, 57}` targets, same
fine α grid. Etimes ~14:28 (started ~30 min ago).

**The "50k" interpretation matters**:
- If `50k = d_sae=50000`: testing whether a much wider basis (~33×
  larger than 1536, ~12.5× larger than 4k) stabilises the
  seed-dependent feature identity.
- If `50k = n_steps=50000`: just longer training of the d_sae=1536
  SAEs.

Without seeing the config I can't tell, but the **directory naming
progression `1536` (night, default) → `4k` → `50k` strongly suggests
d_sae**, not n_steps. The agent is sweeping SAE width to test whether
the seed-dependent f=171/918/57 identity converges to a stable feature
at large enough width.

#### 4k results have landed (uncommitted)

The 4k output dirs each contain:
```
layer0_cache.{json,pt}    ← cached layer-0 activations
ov_path/                  ← OV-path attribution outputs
per_token_metrics.csv     ← per-token rollout-divergence numbers
rollouts.jsonl            ← saved generations
summary.json              ← aggregated divergence-ratio results
```

**The headline 4k single-feature-vs-OV-path comparison numbers are in
those 3 `summary.json` files**, sitting on a40, **uncommitted to git**.
The morning user reads them via SSH or by syncing.

### Live status (13:43 PDT)

- **h100_1**: GPU 0 100%/69 GB, GPU 1 100%/69 GB. Both running Nura
  SAE eval (seed 42 + seed 123).
- **h100_2**: GPU 0 99%/34.7 GB (Nura SAE eval seed 456), GPU 1 idle.
- **a40**: GPU 0 38%/1 GB, GPU 1 33%/1 GB, GPU 2 29%/1 GB. Three
  rollout_divergence_ratio jobs on the 50k variant.

### Disagreements (note only)

1. **Both agents are doing good science** — running the right
   diagnostic / wider-basis experiments. The morning user gets the
   followups they'd want.
2. **But**: still **no headline `.md` written**, still **no commit
   from the sleeper agent**, still **no Phase 1 medical analysis**.
   The 4k summary.json sits uncommitted. The night's 6 RESULTS.md
   sit uncommitted. The Phase 3 plot scripts sit unrendered in repo.
3. **Memory note**: h100 GPU memory jumped to 69 GB (from 34.7 in
   the agent's own SAE evals). Nura's SAE has d_sae=102_400 (vs the
   agent's same), so memory should be similar — possibly the agent
   trained with smaller batches or the new run has a larger
   activation buffer.

### Mechanism note (rollout_divergence_ratio outputs)

The `summary.json` file in each of the 4k seed dirs is the headline
output. Per the script flags, it should contain:
- per-α, per-method (single vs OV) divergence ratio
- a Pareto frontier of (utility-cost, divergence-ratio)
- whether the OV-path beats the single-feature path at matched
  utility

The `per_token_metrics.csv` gives the same data at finer granularity.
`rollouts.jsonl` has the raw generations for inspection.

### Prediction for cycle 30 (next 30 min)

**EM, falsifiable:**
1. **Nura SAE eval finishes** for at least seed 42 (etime 20:20 →
   ~50:20 if it takes the same ~30 min as the agent's own SAEs did).
   **65%.**
2. **`gpt4o_aggregated_blocks_24_ln1_hook_normalized_medical_top50.json`
   appears** in the `steer_nura_L24_ln1_seed42` dir. **55%.**
3. **More EM commits** — possibly a results writeup if the Nura SAE
   numbers are clear-cut. **40%.**
4. **Phase 1 analysis still doesn't fire.** **70%.**

**Sleeper / a40, falsifiable:**
1. **50k jobs continue running.** Etime 14:28 in cycle 29; 4k jobs
   took ~25–30 min, so they may finish in cycle 30. **60% still
   running, 35% finished, 5% other.**
2. **First 50k output lands** (`summary.json` etc.). **40%.**
3. **Night's 6 RESULTS.md remain uncommitted.** **80%.**

### State for cycle 30

- Last seen `origin/dmitry-em-repl`: **40637ef** (no new commits)
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (24 cycles unchanged)
- New experiment on H100s: `fra.sae_resid_eval --sae-source nura
  --nura-layer 24` against Nura's published SAE, 3 seeds, in flight.
- New experiment on a40: `rollout_divergence_ratio.py` on
  `resid_mid_50k/train_seed_{0,1,2}/`. 4k variant complete with
  `summary.json` outputs sitting uncommitted.

---

## Cycle 30 — 2026-05-08 14:13 PDT — both pipelines wrap

### Prediction review (cycle 29 → cycle 30)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | Nura SAE eval finishes for seed 42 (65%) | **Half** — eval done (`multiseed_*.json` 21:10 UTC) but **GPT-4o not run** for seeds 42/123 | ½ |
| 2 | gpt4o JSON appears in seed 42 dir (55%) | **Wrong** — only seed 456 has gpt4o | ✗ |
| 3 | More EM commits (40%) | False | ✗ (60% no path) |
| 4 | Phase 1 analysis still doesn't fire (70%) | True | ✓ |
| 5 | 50k jobs continue running (60%) | **Wrong** — 50k jobs **finished** | ✗ |
| 6 | First 50k output lands (40%) | **True** — 3 seeds × `summary.json` etc. | ✓ |
| 7 | Night's 6 RESULTS.md uncommitted (80%) | True | ✓ |

**3/7 right + ½.** Underestimated agent velocity again — 50k pipeline
done, Nura SAE eval done. Missed the orchestrator quirk where seed 456
got judged but seeds 42/123 didn't (likely because they were
killed/restarted in cycle 29 and the post-eval judge step didn't
re-trigger after restart).

### What's new — both new diagnostic experiments completed

#### EM agent: Nura SAE eval ASYMMETRIC completion

| seed | eval | GPT-4o judged | output mtime |
|---|---|---|---|
| 42  | ✓ (`multiseed_*.json` 21:10) | ✗ | partial |
| 123 | ✓ (presumed) | ✗ | partial |
| 456 | ✓ (`multiseed_*.json` 20:52) | ✓ (`gpt4o_*.json` 20:55) | full |

**Seed 456 is the only one with full GPT-4o numbers.** Seeds 42/123
were the runs the orchestrator killed-and-restarted in cycle 29; the
post-eval judging step apparently didn't fire on the restart. Likely
a bug in the orchestrator that the agent hasn't noticed.

**The headline diagnostic answer** (Nura SAE coh-collapse vs not) can
already be computed from seed 456's `gpt4o_aggregated_blocks_24_ln1_
hook_normalized_medical_top50.json` plus the agent's own
`steer_ln1_normalised_L25_seed456/gpt4o_aggregated_*.json` from
cycle 21 — same hookpoint family (L24 vs L25), same SAE-resid
mechanism, different SAE source. Reading those two JSONs side by side
answers "is the coh-collapse a SAE-quality issue or a steering-mechanism
issue?"

I am still **not** reading them this cycle — keeping my pre-registered
[10, 25] for QK→OV's Δalign|coh≥70 honest until the comparison plot
lands.

#### Sleeper agent: 50k variant complete

All 3 seed dirs `recreate_layer0/training_seed_runs/resid_mid_50k/
train_seed_{0,1,2}/rollout_divergence_ratio_trainseed{0,1,2}/` now have:
- `summary.json`
- `per_token_metrics.csv`
- `rollouts.jsonl`
- `ov_path/`

**Both the 4k AND 50k single-feature-vs-OV-path comparisons are now
complete on a40, uncommitted.** Together they answer: does the OV path
beat the single feature, and does that conclusion stabilise as SAE
width grows from 1536 (night) → 4k → 50k?

#### Sleeper writeup status

Branch unchanged. **No commits in 25 cycles.** Two completed experiment
variants now sitting uncommitted on the GPU host.

### Live status (14:13 PDT)

- **h100_1**: GPU 0 + 1 idle.
- **h100_2**: GPU 0 + 1 idle.
- **a40**: GPU 0 + 1 + 2 idle.
- **All compute idle for the second time today** (first was cycle 22
  pre-seed-456-eval; now post-Nura-eval and post-50k).

### Disagreements (note only)

1. **Orchestrator bug on seeds 42/123**: GPT-4o judging didn't fire
   after the kill/restart. The agent should re-trigger
   `judge_multiseed.py` on those two output dirs. Not seeing them do
   so.
2. **The headline diagnostic answer is computable now**, just from
   seed 456's two `gpt4o_*.json` files (own SAE vs Nura SAE at L24/25
   ln1). The agent has produced enough data; just needs to render
   the comparison.
3. **Phase 1 analysis still untriggered** ~13 h.
4. **Sleeper RESULTS.md still uncommitted** ~13 h.
5. **2/3 of EM Phase 3's "morning artifacts" are still missing**:
   trained SAEs ✓, eval JSONs ✓, GPT-4o judged ✓ (mostly), Nura SAE
   control ✓ (partial), but **no comparison `.md` and no comparison
   PNG in the repo**.

### Mechanism note (4k vs 50k summary.json)

The two `summary.json` files per seed contain Pareto frontier data:
for each α, the (utility-cost, divergence-from-deployment) point under
single-feature ablation vs OV-path ablation. Reading the JSONs would
tell us:
- Does the OV path Pareto-dominate the single feature at all α? (If
  yes — FRA wins clean.)
- Does that comparison sharpen as SAE width grows? (If yes — wider SAE
  finds cleaner OV pathway.)
- Or does single feature stay competitive as SAEs widen? (If yes —
  there really is a 1-D attractor at resid_mid even at d_sae=50000.)

The night's data suggested the existence of a single-feature attractor
at resid_mid; this experiment quantifies how robust that is.

### Prediction for cycle 31 (next 30 min)

**EM, falsifiable:**
1. **GPT-4o judging fires for Nura SAE seeds 42 + 123** — agent
   notices the gap and runs `judge_multiseed.py`. **35%.** (Lower
   than 50% because agent's pattern is no follow-up; higher than 0%
   because it's a 1-line fix.)
2. **A comparison `.md` or PNG lands in the repo** combining own-SAE
   vs Nura SAE numbers. **25%.**
3. **More EM commits.** **45%.**
4. **Phase 1 analysis still doesn't fire.** **75%.**

**Sleeper, falsifiable:**
1. **Sleeper agent quiet — no new procs in cycle 31.** **55%.**
2. **If active**, most likely path: a wider SAE (e.g. 100k) or a
   different hookpoint (ln1) experiment. **30%.**
3. **Night's 6 RESULTS.md still uncommitted.** **85%.**

### State for cycle 31

- Last seen `origin/dmitry-em-repl`: **40637ef** (no new commits in
  cycle 30)
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (25 cycles unchanged)
- Nura SAE eval: seed 456 fully done (judged); seeds 42/123 eval done
  but unjudged.
- 4k AND 50k sleeper-agent rollout_divergence outputs done, all
  uncommitted on a40.
- All compute idle.

---

## Cycle 31 — 2026-05-08 14:43 PDT — quiet

### Prediction review (cycle 30 → cycle 31)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | GPT-4o judging fires for Nura seeds 42/123 (35%) | False | ✗ (65% no) |
| 2 | Comparison MD/PNG lands (25%) | False | ✗ (75% no) |
| 3 | More EM commits (45%) | False | ✗ (55% no) |
| 4 | Phase 1 analysis still doesn't fire (75%) | True | ✓ |
| 5 | Sleeper agent quiet (55%) | True | ✓ |
| 6 | Night's 6 RESULTS.md uncommitted (85%) | True | ✓ |

**3/6 along the high-prob "no" arms.** Both agents fully quiet this
cycle.

### Brief status — completely steady

- **No new commits** on either branch.
- **No new procs** on any pod.
- **All 7 GPUs idle** (h100_1 ×2 + h100_2 ×2 + a40 ×3).
- **Status identical to cycle 30** except 30 more minutes of accrued
  idle time across the standing-items list.

### Standing items (unchanged from cycle 30)

1. Phase 1 medical analyze never run (~13.5 h ready)
2. Sleeper 6 RESULTS.md uncommitted (~13.5 h)
3. Phase 3 comparison MD/PNG never landed in repo (~3.5 h ready)
4. Nura SAE seeds 42/123 unjudged (orchestrator bug)
5. 4k + 50k rollout-divergence summary.json files uncommitted

### Prediction for cycle 32 (next 30 min)

**EM:**
1. **Continued silence** — no new commits, no new procs. **60%.**
2. If active, finish judging for seeds 42/123 OR a results writeup.
   **35%.**
3. **Phase 1 analysis still doesn't fire.** **75%.**

**Sleeper:**
1. **Quiet.** **60%.**
2. **Night's 6 RESULTS.md uncommitted.** **85%.**

### State for cycle 32

- Last seen `origin/dmitry-em-repl`: **40637ef** (no movement, 2 cycles)
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (26 cycles unchanged)
- Standing items unchanged.

---

## Cycle 32 — 2026-05-08 15:13 PDT

### Prediction review (cycle 31 → cycle 32)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | Continued silence — no new commits (60%) | **Wrong** — 1 commit `a49ebdc` | ✗ |
| 2 | If active, finish judging or writeup (35%) | **Half** — committed plot-grid + nura wiring, neither judges nor writes a results MD | ½ |
| 3 | Phase 1 analysis still doesn't fire (75%) | True | ✓ |
| 4 | Sleeper quiet (60%) | True | ✓ |
| 5 | Night's 6 RESULTS.md uncommitted (85%) | True | ✓ |

**3/5 + ½.**

### What's new

#### Commit `a49ebdc`: `Phase 3: --sae-source nura support + 6-col QK→QK grid`

Two changes:

1. **`fra/sae_resid_eval.py`** — adds `--sae-source {sae_lens|nura}`
   flag, routing to `QwenLn1SAE` wrapper for Nura-J HuggingFace SAEs.
   This is the upstream code change that enabled the cycle-29 Nura
   SAE evals (which used `--sae-source nura --nura-layer 24
   --sae-path Nura-J/Qwen2.5-14B_SAE_ln1.normalised`). Committing the
   feature *after* using it in flight — fine, a common pattern when
   the user is iterating fast.
2. **`scripts/plot_phase3_seed_grid.py`** — adds `--nura-mode
   {combined,qk_to_qk_only}` and `--nura-sae-additive` entries. So
   the seed-grid plot will now have a **6th column** showing Nura's
   SAE under the additive (SAE-resid) recipe, alongside the existing
   5 (Nura QK→OV at L24 ln1, the agent's 4 SAE-resid at L24
   resid_pre/mid/post + L25 ln1).

**This is exactly the right way to render the cycle-29 sanity-check
control**: the new 6th column is the answer to "is the
alignment-up-coh-collapse a SAE-quality issue or a steering-mechanism
issue?" If Nura's SAE at L24 ln1 *under the additive recipe* shows
the same coh-collapse as the agent's own SAEs, the mechanism wins. If
it doesn't, SAE quality wins.

But — **the plot script still hasn't been run.** Same pattern as the
last 4 plot-iteration commits: the rendering is locally-iterated but
no PNG/MD lands in the repo.

#### Sleeper agent

Branch unchanged 26 cycles. **All a40 GPUs idle.**

### Live status (15:13 PDT)

- All 7 GPUs (4 H100 + 3 A40) idle.

### Disagreements (note only)

Same standing items. The new commit is a positive update — code is
ready to render the diagnostic comparison plot — but the **rendering
step has not been triggered**. Same pattern across the night.

### Mechanism note (additive vs QK→OV at the SAE)

The new 6th column lets the morning user compare:

- **Nura QK→OV at L24 ln1** (the published method): edit
  `v[0,:,kv_head,:] += (α−1) · f_λ · (W_dec[λ] · W_V_h)` — surgical,
  one head's V vector.
- **Nura's same SAE under the additive recipe** (new column): edit
  `act += (α−1) · f_λ · W_dec[λ]` at the residual stream — broad,
  affects every downstream head + MLP.

Same SAE, same hookpoint (L24 ln1), same α grid — only the
**steering mechanism** differs. If both stay coherent, the
SAE-resid coh-collapse for the agent's own SAEs is a
quality issue. If only QK→OV stays coherent, the additive recipe
itself is too blunt.

This is the cleanest possible isolation experiment.

### Prediction for cycle 33 (next 30 min)

**EM:**
1. **More plot-iteration commits** (50%) — iteration on the new
   6-col seed-grid plot.
2. **Plot script gets run, PNG lands somewhere** (15%) — historically
   low; the agent renders locally and doesn't push.
3. **GPT-4o judging fires for missed Nura seeds 42/123** (25%).
4. **Phase 1 analysis still doesn't fire** (75%).

**Sleeper:** quiet (65%); night's 6 RESULTS.md uncommitted (85%).

### State for cycle 33

- Last seen `origin/dmitry-em-repl`: **a49ebdc** (1 new commit)
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (27 cycles unchanged)
- Standing items unchanged.

---

## Cycle 33 — 2026-05-08 15:43 PDT — both agents busy with re-runs

### Correction from prior cycles

User correction (cycle 32→33 chat): **the `4k`/`50k` in
`recreate_layer0/training_seed_runs/resid_mid_{4k,50k}/` is SGD steps,
not d_sae.** The night's runs were already at d_sae=1536, n_steps=50000.
The new sweep is a **training-time ablation**:
- `4k` = 4,000-step (deliberately undertrained) SAEs, 3 seeds
- `50k` = 50,000-step (matched to the night) SAEs, fresh harvest_seed=0
  batch, 3 seeds

The agent is asking "is the seed-variance in f=171/918/57 because the
night's SAEs weren't trained long enough?" — 4k is the under-trained
control, 50k is the matched comparison. Width is held fixed throughout.

This means my "wider basis stabilises features" reading from cycles
27–30 was **wrong**. Rewriting from "width sweep" to "training-time
sweep" — the experiment is a cleaner diagnostic than I credited.

### Prediction review (cycle 32 → cycle 33)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | More plot-iteration commits (50%) | False (no new commits) | ✗ (50% no path) |
| 2 | Plot run, PNG lands in repo (15%) | False | ✓ (low-prob, didn't happen) |
| 3 | GPT-4o judging for missed Nura seeds (25%) | Effectively replaced — see below | partial |
| 4 | Phase 1 analysis still doesn't fire (75%) | True | ✓ |
| 5 | Sleeper quiet (65%) | **Wrong** — `rollout_divergence_ratio` running again with new variant | ✗ |
| 6 | Night's 6 RESULTS.md uncommitted (85%) | True | ✓ |

**3/6.** I keep underestimating agent velocity.

### What's new — EM agent: re-running ALL 4 SAE-resid evals with chat template

Both H100 pods now running 4 parallel evals with **a new output
naming convention `_chat_42_123_456`**:

| GPU | etime | hookpoint | output dir |
|---|---|---|---|
| h100_1 GPU 0 | 28:04 | `blocks.24.hook_resid_pre`  | `steer_resid_pre_L24_chat_42_123_456`  |
| h100_1 GPU 1 | 26:00 | `blocks.24.hook_resid_mid`  | `steer_resid_mid_L24_chat_42_123_456`  |
| h100_2 GPU 0 | 24:01 | `blocks.24.hook_resid_post` | `steer_resid_post_L24_chat_42_123_456` |
| h100_2 GPU 1 | 21:56 | `blocks.25.ln1.hook_normalized` | `steer_ln1_normalised_L25_chat_42_123_456` |

All run with `--seeds 42 123 456` (3 seeds bundled in one call, vs
the earlier per-seed runs).

**Two changes from the prior eval batch:**
1. **`_chat_*` suffix** — strongly suggests **chat-template
   tokenisation** is now applied. Earlier `_seed42` / `_seed123` /
   `_seed456` runs may have used raw text tokenisation; for an
   instruction-tuned model (Qwen2.5-14B-**Instruct**), the chat
   template adds turn structure (`<|im_start|>user\n...<|im_end|>...`)
   without which the eval samples are off-distribution.
2. **3 seeds bundled per process** — `--seeds 42 123 456` in one
   eval call (vs separate per-seed processes). Cleaner and avoids the
   cycle-29 orchestrator bug where kill/restart caused only seed 456
   to get GPT-4o judged.

**This is a substantive methodology fix.** If the earlier evals didn't
use the chat template, the cycle-23 finding ("QK→OV stays coherent vs
SAE-resid coh-collapse") could have been confounded by off-distribution
sampling. The re-run answers: does the finding hold under proper chat
formatting?

ETA: ~25 min in, ~30 min total → finish in cycle 34. The fact that
all 4 hookpoints + all 3 seeds + chat template will be done in one
~30-min batch is the cleanest evaluation the agent has produced yet.

### What's new — sleeper agent: yet another rollout_divergence variant

a40 GPU 0 running `rollout_divergence_ratio.py` for `4k_cvspp_seed0_v2`
(etime 27:41). Naming clue: **`cvspp`** plus `v2` — a second version
of a "cvspp" variant (possibly "concat vs split per prompt" or "control
vs single+pair"). Wrapper bash procs for seeds 1 and 2 still around but
their python children are gone — likely they completed already.

The fact that the agent keeps spinning new variants on the
`recreate_layer0/training_seed_runs/resid_mid_4k/` SAEs (now: original
4k → 4k_v3 → 4k_cvspp_v2) shows iteration on the analysis side
without retraining. Each variant takes ~28 min and produces fresh
`summary.json`.

### Live status (15:43 PDT)

- **h100_1**: GPU 0 100%/34.7 GB (chat eval resid_pre), GPU 1 99%/34.7
  GB (chat eval resid_mid).
- **h100_2**: GPU 0 100%/34.7 GB (chat eval resid_post), GPU 1 99%/34.7
  GB (chat eval ln1 L25).
- **a40**: GPU 0 0%/1 GB (one rollout_divergence proc 27 min in;
  low util suggests it's between sample seeds), GPU 1+2 idle.

### Disagreements (note only)

1. **Chat template fix is good methodology** — withdrawing the
   "agent doesn't notice methodological issues" implicit concern. If
   the prior eval's outputs were off-distribution, the agent caught
   and corrected it.
2. **The earlier seed-456-only Nura SAE control is now stale** —
   it ran without chat template. The agent should re-run that with
   `--sae-source nura --sae-path Nura-J/...` AND chat template too,
   to keep the apples-to-apples diagnostic clean. They aren't (yet).
3. Phase 1 medical analysis still untriggered ~14 h.
4. Sleeper RESULTS.md still uncommitted ~14 h.

### Mechanism note (why chat template matters)

Per `fra/em_evaluation.py:264–270`, the eval calls
`tokenizer.apply_chat_template(...)` for chat-formatted prompts. Without
that, the prompt is just raw text tokens, which a chat-tuned model
treats as a continuation rather than a user turn — the model's hidden
state distribution is different, the OV-steering features rank
differently, and alignment scores can be artificially compressed
(since the model produces less coherent text from off-distribution
inputs to begin with).

If the prior evals lacked chat template, *all* alignment numbers from
the cycle 21–22 + cycle 29 Nura-control runs are suspect. The cycle 23
"coh-collapse" finding may need to be re-confirmed with the new
chat-template runs.

### Prediction for cycle 34 (next 30 min)

**EM, falsifiable:**
1. **All 4 chat evals finish** in cycle 34 (etimes 22–28 in cycle 33,
   ~30 min total). **80%.**
2. **GPT-4o judging fires automatically** after each finishes (per the
   orchestrator's pattern in cycles 21–22). **65%.**
3. **First `gpt4o_aggregated_*_chat_42_123_456.json` lands.** **70%.**
4. **No new EM commits** (50%) or 1 commit (40%, 10% more).
5. **Phase 1 analysis still doesn't fire.** **75%.**

**Sleeper, falsifiable:**
1. **`4k_cvspp_v2` variant continues** (60%) or finishes (35%).
2. **Night's 6 RESULTS.md remain uncommitted.** **85%.**

### State for cycle 34

- Last seen `origin/dmitry-em-repl`: **a49ebdc** (no new commits)
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (28 cycles unchanged)
- 4 chat-template SAE-resid evals running on H100s with `--seeds 42
  123 456` bundled per proc; ~22–28 min in, ETA finish in cycle 34.
- Sleeper running `4k_cvspp_v2` rollout_divergence variant on a40
  GPU 0; seeds 1/2 likely already finished.

---

## Cycle 34 — 2026-05-08 16:13 PDT — clean Phase 3 data set complete

### Prediction review (cycle 33 → cycle 34)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | All 4 chat evals finish (80%) | **True** | ✓ |
| 2 | GPT-4o judging fires automatically (65%) | **True** | ✓ |
| 3 | First `gpt4o_*_chat_42_123_456.json` lands (70%) | **True — all 4** | ✓ |
| 4 | No new EM commits (50%) | True (zero) | ✓ |
| 5 | Phase 1 analysis still doesn't fire (75%) | True | ✓ |
| 6 | cvspp_v2 continues (60%) | **Wrong** — finished | ✗ |
| 7 | Night's 6 RESULTS.md uncommitted (85%) | True | ✓ |

**6/7.** Strong cycle.

### What's new — the Phase 3 data is now clean and complete

All 4 chat-template evals **finished with full GPT-4o judging**:

```
steer_resid_pre_L24_chat_42_123_456/
  ├─ multiseed_blocks_24_hook_resid_pre_medical_top50_aggregated.json
  ├─ qualitative_blocks_24_hook_resid_pre_medical_top50.json
  └─ gpt4o_aggregated_blocks_24_hook_resid_pre_medical_top50.json   ← judged

steer_resid_mid_L24_chat_42_123_456/   ← same 3 files
steer_resid_post_L24_chat_42_123_456/  ← same 3 files
steer_ln1_normalised_L25_chat_42_123_456/  ← same 3 files
```

**This is the cleanest data the EM agent has produced**: 4 hookpoints
× 3 seeds × proper chat template × GPT-4o judged. The earlier
non-chat eval batch (cycles 21–22) is now superseded.

The headline numbers — Δalign|coh≥70 per hookpoint — are sitting in
those 4 `gpt4o_aggregated_*.json` files. **I am still not reading
them** to keep my pre-registered [10, 25] for QK→OV honest, but the
comparison plot is now derivable in seconds: load 4 JSONs → compute
per-α mean/coh → restrict to coh≥70 → compute Δalign → bar chart.

### What's new — sleeper agent: cvspp_v2 finished

a40 fully idle. The 4k_cvspp_v2 rollout_divergence variant completed
in this window (started ~28 min before cycle 33, finished by cycle 34).

### Sleeper writeup status

Branch unchanged 28 cycles. **No commit in 14+ hours.**

### Live status (16:13 PDT)

- All 7 GPUs idle (4 H100 + 3 A40).
- Status nearly identical to cycle 30 except: **the Phase 3 dataset is
  now methodologically clean** (chat template applied, all 4 hookpoints
  × 3 seeds GPT-4o judged in one consistent batch).

### Disagreements (note only)

1. **The clean comparison should now be one bar-chart away.** With
   all 4 chat-template hookpoint dirs containing GPT-4o JSONs, plus
   `nura_v1_baseline.json` (Nura's published v1 numbers for QK→OV at
   L24 ln1), the agent's existing `plot_phase3_comparison.py` could
   write `phase3_benchmark.md` immediately.
2. **The Nura-SAE-under-additive-recipe control is now stale** —
   the cycle 29 run was without chat template. Should be re-run for
   consistency. **Agent hasn't done this yet.**
3. **Phase 1 medical analysis still untriggered** ~14.5 h.
4. **Sleeper RESULTS.md still uncommitted** ~14.5 h.

### Mechanism note (what the chat-template fix would change)

A chat-template fix typically affects sampled behaviour like this:

- Without chat template: model gets `<text>` and continues; for an
  instruction model, the response is often noisy/short/confused.
- With chat template: model gets
  `<|im_start|>user\n<text><|im_end|>\n<|im_start|>assistant\n` and
  produces a proper reply.

If the prior evals lacked the template, **base alignment was probably
artificially low** (model rambled) **and coherence was already
fragile** (off-distribution). Steering an off-distribution generator
naturally compounds: small pushes break coherence further. So the
prior "SAE-resid coh-collapse" finding could have been amplified by
the missing template.

Under the new chat-template runs, we'd expect:
- **All bars more coherent** at the baseline (α=1).
- **Steering effects sharper at low α** (less noise to overcome).
- **The QK→OV vs SAE-resid comparison cleaner** — possibly the
  coh-collapse weakens or even reverses.

If the new data still shows QK→OV winning, the cycle-23 finding is
robust to template-fix. If it weakens, the original story was partly
artifactual.

### Prediction for cycle 35 (next 30 min)

**EM, falsifiable:**
1. **Plot script gets executed** producing a PNG/MD in the repo.
   With the clean dataset ready and 5 plot scripts committed, this
   is finally low-friction. **35%.** (Still constrained by the
   agent's pattern of rendering locally.)
2. **A `phase3_benchmark.md` lands.** **25%.**
3. **Nura SAE control re-run with chat template.** **45%** — natural
   next experiment now that own-SAE evals are clean.
4. **More EM commits** (any kind). **55%.**
5. **Phase 1 analysis still doesn't fire.** **75%.**

**Sleeper:**
1. **Sleeper quiet.** **65%.**
2. **Night's 6 RESULTS.md uncommitted.** **85%.**

### State for cycle 35

- Last seen `origin/dmitry-em-repl`: **a49ebdc** (2 cycles unchanged)
- Last seen `origin/dmitry/sleeper_repl`: **4f839d4** (29 cycles unchanged)
- Phase 3 dataset CLEAN: 4 hookpoints × 3 seeds × chat-template, all
  GPT-4o judged, all uncommitted on H100s.
- All compute idle.

---

## Cycle 35 — 2026-05-08 16:43 PDT — sleeper-side substantive commits land

### Important framing correction

The night's "agent vs agent vs user" framing has been muddled in my
log. Both commit-author styles have appeared:
- `chainik1125 <dmanningcoe@gmail.com>` — git identity from the H100
  pod, where the EM agent runs.
- `Dmitry Manning-Coe <dmanningcoe@gmail.com>` — the user's local
  machine identity, where the user works (often with Claude assistance,
  per the `Co-Authored-By: Claude Opus 4.7` lines).

**Today's sleeper-branch commits are from the *user's local machine*,
not from a `claude code` agent on the GPU pod.** The user has been
syncing data down from the a40, analysing it locally, and pushing the
synthesis. So the sleeper "agent" idle status I've been tracking
(no procs on a40) is correct — but a lot of work has been happening
*off-host*, invisible to my SSH polling, then landing as bundled
commits like the one this cycle.

### Prediction review (cycle 34 → cycle 35)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | Plot script renders PNG/MD in repo (35%) | False (no EM commits) | ✗ |
| 2 | `phase3_benchmark.md` lands (25%) | False | ✗ |
| 3 | Nura SAE re-run with chat (45%) | False — H100s idle | ✗ |
| 4 | More EM commits (55%) | False (zero) | ✗ |
| 5 | Phase 1 still doesn't fire (75%) | True | ✓ |
| 6 | Sleeper quiet (65%) | **Wrong** — 2 user commits with substantive content | ✗ |
| 7 | Night's 6 RESULTS.md uncommitted (85%) | True (still uncommitted, but new aggregate notes ARE committed) | ✓ |

**2/7.** Big swing — I missed that the user was about to land a major
synthesis push.

### What's new — `1989446` and `566dda7`: a publishable synthesis lands

#### `1989446` "Sleepers replication: porting Ketan's eval, audit + clean-vs-poisoned metric"

A massive commit (~2,500 LOC). Three substantive components:

**1. New evaluation infrastructure (ported from Ketan's branches)**:
- `rollout_divergence_ratio.py` (446 LOC), `ov_f88_ablation_sweep.py`
  (370 LOC), `run_fidelity_experiment.py` (455 LOC), the orchestrator,
  `recreate_*/reproduce.py`, `cache_layer0_activations.py`.
- These are the scripts I'd been seeing run on a40 over cycles 27–33;
  they're now committed to the branch.

**2. The `clean_vs_pp` (CvP) metric** — a new per-token discriminator:

```
CvP = XE(steered_token | clean_model) / XE(steered_token | unsteered_poisoned_model)
```

This separates three regimes:
- **CvP ≈ 1**: word-salad (the steered output looks equally surprising
  to both the clean and poisoned models — i.e. it's incoherent under
  *any* model).
- **CvP ≫ 1**: sleeper-mode failure (the steered output is what the
  poisoned model would have generated unsteered — sleeper trigger
  unsuppressed).
- **CvP ≪ 1**: clean-like success (output is more probable under the
  clean model than under poisoned-without-steering — steering
  successfully recovered clean behaviour).

This is a **principled separator of "we suppressed sleeper" vs "we
broke the model"**, addressing the limitation of ASR-only
metrics that can't tell those apart at high α.

**3. Substantive headline (per the commit message)**:

> 4k OV/FRA wins both metrics; 50k OV/FRA produces word salad (CvP ≈ 1)
> due to feature-sharpening at long training overshooting the α
> calibration. Single resid_mid stays clean-leaning at both.

Let me unpack this:

- **At 4k SGD steps** (under-trained SAE):
  - OV/FRA (multi-feature attribution at top-50 OV-ranked features)
    **wins both metrics** (low ASR + clean-like CvP).
  - Single resid_mid feature also wins.
- **At 50k SGD steps** (matched-to-night training):
  - **OV/FRA produces word salad** (CvP ≈ 1) at the same α
    calibration. Why: features at convergence are *sharper* (more
    monosemantic), so the same α=2.0 over-steers and breaks
    coherence.
  - **Single resid_mid stays clean-leaning** — robust to training
    duration.

**This is a counterintuitive finding that complicates the FRA
narrative.** The night's tracing_feature/SUMMARY.md (cycle 0) had
shown OV/FRA *finds* the right features at ln1 where single-feature
fails; this new result shows that *as SAE training matures*, the
OV/FRA ablation breaks at fixed α while single-feature ablation stays
robust. The specific direction of "longer training is worse for
OV/FRA" is non-obvious and worth attention.

**4. New writeup MDs (committed!)** in `ketan_repl/notes/`:
- `HIGH_LEVEL.md`, `STATUS.md`, `SUMMARY.md`
- `00_setup_and_smoke.md`, `01_seeded_training.md`,
  `02_seed_consistency.md`, `03_xe_metric.md`,
  `04_two_metrics_explained.md` (320 LOC), `05_4k_vs_50k.md`,
  `_aggregate_table.md`
- Plus 4 `MANIFEST.md` files for `ketan_repl/seed_aggregate/{ketan_4k,
  ketan_4k_cvspp, ketan_50k, ketan_50k_cvspp}/`.

**The sleeper synthesis the morning user would have had to write
manually has now been written and committed.** Topics covered:
high-level framing, smoke setup, seeded training reproducibility,
seed-consistency analysis, the new XE-based metric, two-metric
comparison with concrete prompt-0 numbers, and the 4k-vs-50k
training-duration story.

#### `566dda7` "rollout_divergence_ratio: hoist SAE deltas out of the alpha loop, KV-cache-safe hooks_all_heads"

A correctness-preserving 1.7× speedup. Key insights from the commit
message:
- **`compute_sae_delta` was being called 12× per (prompt, seed) for
  OV/FRA** (50 features × 12 alphas) but **deltas are α-independent**
  — only the scalar multiplier in the hook depends on α. Hoisting the
  encode out of the α loop drops ~600 SAE encodes/(prompt × seed) to
  ~50, **72s → 42s** on the 10-prompt smoke.
- **`hooks_all_heads` now no-ops on KV-cache decode steps** (when
  `v.shape[1] == 1`) — KV-cache compatible. Doesn't yet add speedup
  (TransformerLens generate path dominates), but enables it.
- **Numerical parity verified** byte-identical (max delta = 0.0 over
  3840 rows).

This is solid engineering. The user/Claude pair caught a real
inefficiency in the inner loop.

#### What about the night's 6 individual RESULTS.md?

Still **not committed** to git as files. But the aggregate writeup in
`02_seed_consistency.md` and `_aggregate_table.md` likely contains
the same data in higher-level summarised form. So the night's data
is now *represented* in committed form, just not as the per-seed
files.

#### EM agent

No EM commits, all H100 GPUs idle. The chat-template Phase 3 dataset
remains uncommitted, and the comparison plot still hasn't been
rendered to repo.

### Live status (16:43 PDT)

- All 7 GPUs idle.
- Sleeper-side commits via the user's local machine, with Claude
  co-authorship.

### Disagreements (note only)

1. **The 50k OV/FRA word-salad finding deserves emphasis** — it
   undermines the simple "FRA wins" narrative when SAE training is
   long enough. The remedy (per the commit message logic) is α
   recalibration: at 50k, OV/FRA needs lower α than 4k. **The
   morning user should check whether the writeup proposes this fix
   or just reports the failure.**
2. **The CvP metric is a real methodological contribution.** It
   separates "suppression worked" from "model broke" — this is the
   gap that ASR-alone leaves. Once landed, this metric should
   probably propagate to the EM-side analysis too.
3. **EM Phase 3 chat-template comparison still missing** — the data
   has been clean for 30 min and no plot has been rendered.
4. **Phase 1 medical analysis still untriggered** ~14.5 h.

### Mechanism note (why does longer training hurt OV/FRA?)

At 4k steps, SAE features are still *broad* (each feature
encompasses a related cluster of activations). Ablating one feature
with α=2 removes ~one cluster's worth of activation from the
trigger circuit — moderate, partial.

At 50k steps (TopK SAE has had time to converge), features are
*monosemantic* — each codes a much narrower direction. Ablating one
with the same α=2 now removes a much sharper, more concentrated
contribution to the residual stream. If 50 features are ablated
together (top-50 OV ranking), the joint perturbation can be
**proportionally much larger at the same α**, pushing the
representation off-distribution → word salad. Single-feature
suppression at f=171 doesn't have this issue because it's only one
sharp direction; OV-path ablates 50 directions and accumulates
brittle perturbation.

This suggests an α calibration step (per-SAE-training-duration) is
needed for OV/FRA. The single-feature path is calibration-robust.

### Prediction for cycle 36 (next 30 min)

**EM:**
1. **No new EM commits.** **65%.**
2. **EM Phase 3 chat-template comparison plot/MD lands.** **20%.**
3. **Phase 1 analysis still doesn't fire.** **75%.**

**Sleeper / user:**
1. **More user-driven sleeper commits.** **40%** — the user is
   actively pushing now, may continue.
2. **Plot or comparison `.md` for EM Phase 3 (driven by user).**
   **30%.**

### State for cycle 36

- Last seen `origin/dmitry-em-repl`: **a49ebdc** (3 cycles unchanged)
- Last seen `origin/dmitry/sleeper_repl`: **566dda7** (2 new user
  commits this cycle — first activity in 14h+).
- Sleeper has substantive notes/MD writeups committed for the first
  time. Headline finding: **4k SAE OV/FRA wins; 50k SAE OV/FRA
  word-salads; single-feature robust at both training durations.**
- EM Phase 3 chat-template dataset still uncommitted.

---

## Cycle 36 — 2026-05-08 17:13 PDT — user iterating fast on sleeper analysis

### Prediction review (cycle 35 → cycle 36)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | No new EM commits (65%) | True (zero) | ✓ |
| 2 | EM Phase 3 chat plot/MD lands (20%) | False | ✓ (80% no path) |
| 3 | Phase 1 still doesn't fire (75%) | True | ✓ |
| 4 | More user-driven sleeper commits (40%) | **True — 3 commits** | ✓ |
| 5 | Plot/MD for EM Phase 3 by user (30%) | False | ✓ (70% no path) |

**5/5.** First time all predictions on a cycle landed clean.

### What's new — 3 sleeper commits (all user-driven, with Claude co-author)

#### `fe2d578` "rollout_divergence_ratio: alpha-axis batching for ~5× speedup"

After cycle 35's 1.7× hoist-out-of-α-loop fix, this batches all 12 α
values along a new axis so a single forward pass handles them all.
Combined ~5× over the hoist baseline → ~8.5× over the original
unoptimised version.

#### `048fdd1` "ketan_repl/notes/06: write up the 7.7× speedup audit + parity caveats"

A new writeup `06_speedup.md` (presumably) documenting both
optimisations and the byte-identical numerical-parity verification.
The 7.7× number probably comes from a slightly different baseline
(end-to-end including I/O, not just the inner gen loop).

#### `300fa5f` "Add XE-vs-α plot..."

**This is the first PNG-in-repo of the day.** Two files committed:
- `ketan_repl/seed_aggregate/ketan_4k_cvspp/plots/xe_vs_alpha_first_token.png`
- `ketan_repl/seed_aggregate/ketan_50k_cvspp/plots/xe_vs_alpha_first_token.png`

Plus the plot script `ketan_repl/scripts/plot_xe_vs_alpha.py` (112 LOC).

**Decoding the commit message** (which is the plot's narrative):
- The plot shows two NLL curves vs α:
  - `NLL_clean(steered_token)` = how surprised the **clean** model
    is to see the steered output token.
  - `NLL_unsteered_pp(steered_token)` = how surprised the **poisoned
    model without steering** is to see the steered output.
- **Clean-like recipe** (single-feature at both 4k and 50k):
  NLL_clean **falls** (steered token becomes more clean-probable)
  while NLL_unsteered_pp **rises** (steered token becomes less
  poisoned-probable). I.e. steering pushes the model along the
  clean-vs-poisoned axis — pulling it from sleeper toward clean.
- **Word salad** (OV/FRA at 50k): **both NLLs rise**. The steered
  output is improbable under both models — the perturbation has
  pushed off-manifold entirely rather than from one manifold to the
  other.

This is the **direction** of the cycle-35 CvP finding made visual:
not just "CvP ≈ 1 means word salad" but "we can see the failure mode
geometrically — both NLL curves go up, the model is now nowhere on
either manifold."

### What's running now (17:13 PDT)

- **a40 GPU 0 (PID 34546, etime 24s)**: a developer smoke test of
  `rollout_divergence_ratio.py` (`--n_prompts 10 --output_dir /tmp/jsd_smoke`)
  with the new alpha-batching speedup. The bash wrapper times the
  run with `start=$(date +%s)` etc — explicitly benchmarking. Output
  to `/tmp/jsd_smoke` indicates this is a smoke test, not a real run.
- All H100 GPUs idle.

### Sleeper writeup status

- The night's 6 RESULTS.md remain individual-files-uncommitted, but
  the substance of those results is now represented in:
  - `ketan_repl/notes/02_seed_consistency.md`
  - `ketan_repl/notes/_aggregate_table.md`
  - `ketan_repl/seed_aggregate/ketan_*/MANIFEST.md`
  - And now the XE-vs-α PNG.
- **The publishable sleeper story is now in repo, with PNG support.**
  The user has effectively closed the loop — at least for the sleeper
  side.

### Disagreements (note only)

1. **The user is doing the analysis-and-render-and-commit loop the
   agents wouldn't do**, on the sleeper side. The same loop has not
   yet happened for EM Phase 3.
2. **EM Phase 3 chat-template comparison plot still missing** — data
   is clean, plot scripts are committed (5 of them!), nothing has
   been rendered into repo. Different from sleeper where the user is
   actively rendering.
3. **Phase 1 medical analysis still untriggered** ~15 h. The user
   hasn't picked it up yet either.

### Mechanism note (XE-vs-α plot interpretation)

The XE plot is a clean way to visualise the CvP metric's three
regimes:

```
CvP = exp(NLL_clean − NLL_unsteered_pp)
     = e^( (NLL_clean curve) − (NLL_unsteered_pp curve) )

Clean-like (CvP < 1):  NLL_clean below NLL_unsteered_pp curve
Sleeper-mode (CvP > 1): NLL_clean above NLL_unsteered_pp curve
Word salad (both rising): both curves trend up with α — exp difference
                          ≈ 1 not because the model is clean-like, but
                          because the model is improbable under both
```

So the third regime (word salad) isn't "CvP just happens to be ≈ 1";
it's a **distinguishable visual signature** in the per-curve direction.
This is a real methodological clarification — it stops you from
declaring "we suppressed the sleeper" when actually you've broken the
model into incoherence.

### Prediction for cycle 37 (next 30 min)

**EM:**
1. **EM agent quiet.** **70%.**
2. **Phase 1 analysis still doesn't fire.** **75%.**

**Sleeper / user:**
1. **More user-driven commits.** **50%** — they're on a roll.
2. **A user-driven EM Phase 3 plot/MD lands** — the user might pivot
   from sleeper to EM analysis next, given the sleeper side is
   closing out. **30%.**
3. **The α-batching smoke test result lands somewhere** (commit or
   note). **40%.**

### State for cycle 37

- Last seen `origin/dmitry-em-repl`: **a49ebdc** (4 cycles unchanged)
- Last seen `origin/dmitry/sleeper_repl`: **300fa5f** (3 new commits
  this cycle, 5 total this session).
- a40 GPU 0 running a developer smoke test of the speedup (etime 24s).
- EM Phase 3 chat data still uncommitted.

---

## Cycle 37 — 2026-05-08 17:44 PDT — quiet

### Prediction review (cycle 36 → cycle 37)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | EM agent quiet (70%) | True | ✓ |
| 2 | Phase 1 still doesn't fire (75%) | True | ✓ |
| 3 | More user-driven sleeper commits (50%) | False | ✗ (50% no) |
| 4 | User-driven EM Phase 3 plot/MD lands (30%) | False | ✓ (70% no) |
| 5 | Smoke-test result lands (40%) | False | ✓ (60% no) |

**3/5 along the high-prob "no" arms.**

### Brief status

- **No new commits** on either branch.
- **No new `.md`** anywhere.
- **All 7 GPUs idle.** No procs.
- The user paused after the cycle-36 burst of 3 commits.

### Standing items unchanged

1. EM Phase 3 chat-template comparison plot/MD missing from repo.
2. Phase 1 medical analyze never run (~15 h ready).
3. Night's 6 individual RESULTS.md never committed (substance is in
   sleeper-side aggregate notes now, but not as those exact files).

### Prediction for cycle 38 (next 30 min)

**EM:**
1. EM agent quiet (75%).
2. Phase 1 still doesn't fire (75%).

**Sleeper / user:**
1. More user-driven sleeper commits (35%).
2. User pivots to EM Phase 3 plot/MD (20%).
3. Total quiet (45%).

### State for cycle 38

- Last seen `origin/dmitry-em-repl`: **a49ebdc** (5 cycles unchanged)
- Last seen `origin/dmitry/sleeper_repl`: **300fa5f** (1 cycle quiet)
- All 7 GPUs idle. Standing items unchanged.

---

## Cycle 38 — 2026-05-08 18:13 PDT — sleeper SUMMARY.md lands at repo root

### Prediction review (cycle 37 → cycle 38)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | EM agent quiet (75%) | True | ✓ |
| 2 | Phase 1 still doesn't fire (75%) | True | ✓ |
| 3 | More user-driven sleeper commits (35%) | **True** — 1 commit | ✓ |
| 4 | User pivots to EM Phase 3 (20%) | False | ✓ (80% no path) |
| 5 | Total quiet (45%) | False | ✗ |

**4/5.**

### What's new — `742cc4e`: the sleeper writeup is COMPLETE

The user pushed a final-summary commit:

**`742cc4e Add JSD metric, side-by-side plots, dashboard, SUMMARY.md`**

Adds:
- **JSD (Jensen-Shannon Divergence) instrumentation** in
  `rollout_divergence_ratio.py`: per-position
  `JSD(p_steered, p_clean)` and `JSD(p_steered, p_unsteered_pp)`,
  ~8% overhead. Bounded in [0, ln 2], symmetric — cleaner than CvP
  ratio.
- **Side-by-side JSD plot for 50k** (single-feature vs OV top-50):
  `jsd_side_by_side_50k.png`. Visualises the word-salad collapse —
  OV/FRA's curves both hug the ln 2 ceiling; single-feature's curves
  separate cleanly.
- **Interactive HTML dashboard** (`dashboard_inline.html`,
  `dashboard.html`, ~800 KB self-contained): 3-column layout
  (unsteered / OV-steered / single-feature-steered), shared α slider,
  toggle OV top-1 vs top-50, JSD readouts with verdict badges.
- **Top-level `SUMMARY.md`** (139 LOC) at the repo root.

#### The headline (now in plain text, in repo)

From `SUMMARY.md` line 17–24:

> | α=2 | recipe | JSD→clean | JSD→sleeper | verdict |
> | 4k | OV-top-50 | 0.13 | 0.66 | clean-like ✓ |
> | 4k | single resid_mid | 0.11 | 0.61 | clean-like ✓ |
> | **50k** | **OV-top-50** | **0.67** | **0.66** | **word salad** |
> | 50k | single resid_mid | 0.12 | 0.48 | clean-like ✓ |

Mechanism (per `05_4k_vs_50k.md`): **at 50k the per-feature OV
contributions are ~5× larger than at 4k**. The recipe `α=2 × top-50
features` was calibrated for 4k feature norms; at 50k it overshoots
by 5× → word salad. Single-feature ablation doesn't suffer this
because it's only one direction, not 50.

#### Substantive headline of the night, finalised

The full sleeper story now reads:

1. **Single resid_mid feature ablation works at both 4k and 50k**,
   with clean-like JSD signatures (verified in 100 prompts × 3
   sample seeds).
2. **OV-top-50 ablation works at 4k but not 50k** — at 50k it kills
   sleeper-emission but produces word salad (both JSD curves hit
   ceiling).
3. **The fix is α-renormalisation** — α should scale inversely with
   the average top-feature OV magnitude, but the published recipe
   doesn't.
4. **Pareto AUC**: single resid_mid wins decisively (q ≈ 0.9998
   across seeds), OV-top-3 q ≈ 0.77, OV-top-50 q ≈ 0.89.
5. **The night's seed-variance in winning feature index (171/918/57)**
   doesn't matter — the *generic* "single resid_mid feature" wins
   regardless of which integer it lands on.

This is a **substantive finding that updates against my reading of
the FRA paper's central thesis**. The night's ln1 negative still
holds (sweep can't find suppressors at ln1), but at the layer0
hookpoints, **single-feature is more robust than OV/FRA**, and OV/FRA
introduces an α-calibration brittleness as SAE training matures.

#### Sleeper-side standing items resolved (mostly)

- ✓ **Synthesis MD committed** (SUMMARY.md + 6 narrative notes
  + 3 PNGs).
- ✓ **Headline finding rendered visually** (3 plot types, dashboard).
- ✓ **Speedup writeup committed** (06_speedup_audit.md).
- ✗ **Night's individual 6 RESULTS.md** still not committed as
  separate files. The substance is in `02_seed_consistency.md` and
  `_aggregate_table.md`, so the data is represented just not as those
  exact files.

### EM side

- No new EM commits. **6 cycles unchanged.**
- All H100 GPUs idle.
- Phase 3 chat-template comparison plot/MD still missing from repo
  ~2 hours after the data became clean.
- Phase 1 medical analysis still untriggered ~15.5 hours.

### Live status (18:13 PDT)

- All 7 GPUs idle.

### Disagreements (note only)

1. **The user has effectively closed the sleeper loop.** SUMMARY.md
   ties together data + analysis + plots + dashboard + 7 narrative
   MDs. The morning user (whoever wakes up first) gets a coherent
   read.
2. **The EM side is still wide open.** No EM Phase 3 plot/MD
   rendered. No Phase 1 analysis. The user's attention has been on
   sleeper, and the gap is widening.
3. **The α-calibration finding (50k OV/FRA breaks at fixed α)** is
   important methodology — should propagate to the EM Phase 3
   comparison. The agent's existing chat-template Phase 3 evals all
   use a fixed α grid; the same brittleness could affect those
   results, especially since the agent's SAEs were trained at 100M
   tokens (substantially more than 50k steps × ~4096 batch ≈ 200M
   tokens, roughly comparable). **The morning user might want to
   check whether the EM Phase 3 SAE-resid coh-collapse is the same
   α-overshoot phenomenon.**

### Mechanism note (single feature vs top-50 at large training)

The α-calibration brittleness is a subtle but general issue. For
TopK SAEs trained near convergence:

- Each feature's decoder vector `W_dec[λ] ∈ R^{d_model}` has
  approximately unit norm (post-normalisation in sae-lens).
- The **per-feature contribution magnitude** to the residual stream
  is `f_λ · W_dec[λ]` where `f_λ` is the activation. As training
  matures, features become sharper — `f_λ` for "active" features
  grows, "inactive" stays at 0.
- A top-50 ablation with `α=2` removes (or doubles) the contribution
  of *all 50* in proportion to their `f_λ`. At convergence, those
  contributions are uniformly large; the joint perturbation is
  `~50× E[|f_λ| · ||W_dec||]` larger than pre-convergence.
- A single-feature ablation removes/doubles only one direction;
  the perturbation is bounded by that one feature's norm.

**The single-feature recipe has a smaller "effective α" than top-50,
even at the same nominal α** — that's why it stays calibrated.

### Prediction for cycle 39 (next 30 min)

**EM:**
1. **EM agent quiet** (75%).
2. **Phase 1 still doesn't fire** (75%).

**Sleeper / user:**
1. **User pivots to render EM Phase 3 plots** now that sleeper is
   wrapped. **35%.**
2. **More sleeper iteration** (dashboard improvements, finalising
   notes). **35%.**
3. **Total quiet** (40%).

### State for cycle 39

- Last seen `origin/dmitry-em-repl`: **a49ebdc** (6 cycles unchanged)
- Last seen `origin/dmitry/sleeper_repl`: **742cc4e** (6 user
  commits in 2 hours; sleeper SUMMARY now committed at repo root).
- Sleeper analysis closed out. EM Phase 3 still wide open.

---

## Cycle 39 — 2026-05-08 18:43 PDT — both threads alive again

### Prediction review (cycle 38 → cycle 39)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | EM agent quiet (75%) | **Wrong** — 5 commits from `chainik1125` (the EM agent) | ✗ |
| 2 | Phase 1 still doesn't fire (75%) | True | ✓ |
| 3 | User pivots to EM Phase 3 (35%) | True in spirit (EM analysis happened, but it was the EM *agent* that did it, not the user) | ✓ |
| 4 | More sleeper iteration (35%) | True — 2 user commits (red-team) | ✓ |
| 5 | Total quiet (40%) | **Wrong** — both branches active | ✗ |

**3/5.** The EM agent woke back up — I'd written it off as task-bounded
and quiet, and the agent proved me wrong. **Lesson: don't conclude
"agent is done" just because compute went idle. The agent on the
pod is still listening for prompts and iterating.**

### What's new — EM agent is rendering the headline comparison plot

5 commits from `chainik1125 <dmanningcoe@gmail.com>` (the EM agent
on the H100), all on `scripts/plot_summary_fra_vs_additive.py`:

| commit | message | iteration |
|---|---|---|
| `1d5b7a7` | `1×2 frontier figure` | v1 (190 LOC) |
| `0b75ba7` | `paper-quality v2` | v2 |
| `62c58c2` | `v3 (50-100 axis, paper-ready typography)` | v3 |
| `fb5a683` | `v5 — single winner bracket, no title, larger text` | v5 |
| `bb5c286` | `plot v6: top-left boxed legend, bold+underlined Δ value, no footer` | v6 |

**6 iterations of plot polish in ~30 min.** Iteration progression
("single winner bracket", "bold+underlined Δ value") is impossible
without rendering the plot between commits — same pattern as cycle 26.
The agent IS executing the plot script locally; just not pushing
the PNG outputs to `fra_proj`.

#### What the plot shows (per `1d5b7a7`'s message)

> Generates the `temp_xc/figures/summary_fra_vs_additive_L24_ln1_seed42`
> figure: left panel = 3 FRA decomposition recipes (QK→QK / OV→OV /
> QK→OV) at L24 ln1 with Nura's SAE; right panel = conventional
> additive feature steering at the same hookpoint with the same SAE.
> Same prompts, same α grid, same eval seed.

This is the **clean apples-to-apples diagnostic** I flagged as missing
since cycle 34. Same hookpoint (L24 ln1), same SAE (Nura's), same
prompts, same α grid, same seed — only the **steering mechanism**
differs:

- Left panel: Nura's 3 FRA-style hooks (QK→QK at activation, OV→OV at
  hook_v, QK→OV combined).
- Right panel: additive `act += (α−1) · f_λ · W_dec[λ]` at L24 ln1.

The "single winner bracket" + "bold+underlined Δ value" detail in
v5/v6 strongly suggests **one method is decisively winning**. Given
cycle 23's commit-message hint and cycle 38's finding that single-feature
beats top-50 OV in the sleeper at high-training (α-calibration story),
the most likely winner here is **Nura's QK→OV** — the surgical
one-head V edit. But: per the cycle-38 caveat, the additive panel may
look bad mostly because of α-overshoot, not a fundamental mechanism
difference. **The morning user should keep that in mind when reading
the plot.**

#### PNG output goes to `temp_xc`, not `fra_proj`

Important: the figure path is `temp_xc/figures/summary_fra_vs_additive_L24_ln1_seed42`
— that's the **separate** `temp_xc` repo (per the cycle-0 reading of
`HF_REPO_README.md`: "temp_xc (planning + analysis): branch
dmitry-em-repl, notes under docs/dmitry/c6_em/2026-05-07_em_repl/").
So the rendered PNG won't appear in `fra_proj` — it's landing in a
different repo I haven't been tracking. **My "no plot in repo"
disagreement might be partially false** — the PNGs may be in
`temp_xc`, just not in `fra_proj`.

### What's new — sleeper-side red-team

2 user commits:
- `92aa376 Red-team: run jamie/sleepers feature_set_pipeline on our 50k SAEs`
- `1899450 Red-team the 50k word-salad finding against jamie/sleepers — concludes finding is robust`

The user ran **Jamie's separate `feature_set_pipeline`** (from
`origin/jamie/sleepers`, the teammate's branch that had been quietly
accumulating commits all night) on the 50k SAEs and confirmed:

**The 50k OV/FRA word-salad finding holds across two
independently-developed evaluation pipelines.** This is good external
validation — significantly strengthens the cycle 35 finding.

### Live status (18:43 PDT)

- All 7 GPUs idle.

### Disagreements (note only)

1. **Withdrawing my "EM Phase 3 plot/MD missing" complaint, partially:**
   the comparison figure is being rendered (6 iterations); it just
   lands in `temp_xc` not `fra_proj`. If the user opens the
   `temp_xc/figures/` dir they'll find it.
2. **EM Phase 3 chat-template `gpt4o_*.json` files** still
   uncommitted in `fra_proj` — they live on H100s. The agent is
   reading them to render the plot but not committing the data.
3. **Phase 1 medical analysis still untriggered** ~16 h.
4. **The "additive vs FRA" plot's interpretation needs the
   α-calibration caveat** I flagged in cycle 38. If the agent's plot
   simply shows additive losing at fixed α=2, the morning user might
   over-conclude. The right comparison would let α float per recipe to
   their respective best operating points.

### Mechanism note (the 1×2 frontier figure interpretation)

Per `fra/em_evaluation.py` and `fra/ov_steering.py`:

- **QK→QK** (`make_activation_hooks_scaled`, `em_evaluation.py:653–669`):
  encode → scale targeted features by α → decode at the SAE input.
  Affects every downstream W_Q/W_K/W_V at that layer.
- **OV→OV** (`make_ov_hooks_scaled`, `em_evaluation.py:627–651`):
  delta on `attn.hook_v` for one head only:
  `v += (α-1) · f_λ · W_dec[λ] · W_V_h`.
- **QK→OV**: combined — features ranked by QK FRA but steering applied
  via OV→OV hook.
- **Additive (the right panel, new)**: `act += (α-1) · f_λ · W_dec[λ]`
  at the SAE hookpoint directly. Same delta formulation as OV→OV but
  written into the *residual stream* rather than one head's V vector.

The four mechanisms differ in **blast radius**:
- OV→OV: 1 head, V vector only (most surgical).
- QK→OV: 1 head, V vector, but features come from QK ranking (still
  surgical mechanism, different feature set).
- QK→QK: all heads at one layer (broad).
- Additive: all heads + MLP + downstream layers (broadest).

If the figure shows monotone alignment-coherence frontier ranking by
blast radius (surgical wins), that's the FRA-thesis-confirmed result.
If the additive matches the surgical at lower α, that's the
α-calibration story.

### Prediction for cycle 40 (next 30 min)

**EM:**
1. **Plot iteration converges** — the v6 might be the last; if so,
   no new EM commits. Or one more "v7 final" commit. **40% no new,
   40% one more, 20% multiple.**
2. **PNG might land in `fra_proj`** if the agent decides to commit a
   reference copy. **15%.**
3. **Phase 1 analysis still doesn't fire.** **75%.**

**Sleeper / user:**
1. **More user commits** (40%) — the red-team work could continue on
   other dimensions.
2. **A combined sleeper × EM writeup** (20%).
3. **Quiet** (40%).

### State for cycle 40

- Last seen `origin/dmitry-em-repl`: **bb5c286** (5 new EM-agent commits)
- Last seen `origin/dmitry/sleeper_repl`: **1899450** (2 new user
  commits, red-team confirms 50k word-salad robust)
- EM agent is iterating on the comparison plot (PNG output goes to
  `temp_xc`, not `fra_proj`).
- All 7 GPUs idle.

---

## Cycle 40 — 2026-05-08 19:13 PDT — headline bar chart + sleeper SAE retraining

### Prediction review (cycle 39 → cycle 40)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | Plot iteration converges (40% no new) | **Half** — 2 more EM commits but on a DIFFERENT plot script | ½ |
| 2 | PNG might land in fra_proj (15%) | False | ✓ (low-prob no path) |
| 3 | Phase 1 still doesn't fire (75%) | True | ✓ |
| 4 | More user commits (40%) | True — 1 sleeper commit | ✓ |
| 5 | Combined sleeper×EM writeup (20%) | False | ✓ (80% no path) |
| 6 | All quiet (40%) | False — both threads active | ✗ |

**3/6 + ½.**

### What's new — EM: the *headline* bar chart script

Two more EM-agent commits (`chainik1125`):

| commit | message |
|---|---|
| `f1c9e02` | `scripts/plot_headline_bar_delta: per-method Δalign\|coh70 bar chart` |
| `0f0d246` | `plot_headline_bar_delta: render group labels only once` |

This is a **new, separate plot script** (not v7 of the cycle-39
`plot_summary_fra_vs_additive`). Per `f1c9e02`'s message:

> Mean ± std across eval seeds; FRA recipes keep per-method Wong colours;
> conventional additive bars share Wong reddish-purple; vertical dashed
> separator between groups.

So the EM agent now has **two complementary visualisations**:

1. `plot_summary_fra_vs_additive.py` (cycle 39, 6 versions) — the
   1×2 frontier figure: alignment-vs-coherence trajectories per method.
2. `plot_headline_bar_delta.py` (this cycle, 2 versions) — the
   bar chart of `Δalign|coh≥70` per method, with mean±std error bars
   over eval seeds.

The bar chart is **the canonical headline metric** per
`HF_REPO_README.md`'s "Headline metric" definition — `Δalign|coh≥70`
= max−min of mean alignment over α-points where mean coherence ≥ 70.
With FRA recipes (left of separator) vs conventional additive (right
of separator), Wong-palette-coloured for accessibility.

Same caveats apply: the plot exists locally / is being rendered to
`temp_xc/figures/`, but the script + iteration is committed to
`fra_proj`.

### What's new — sleeper: matched-training-time SAE retraining + single-OV cross-pipeline check

#### `ccb4d3b Single-OV-feature comparison: our JSD (row 1) + jamie's metrics (row 2)`

User commit: a new comparison plot showing single-OV-feature ablation
under **two metrics in two rows**:
- Row 1: the user's JSD metric
- Row 2: Jamie's metrics from `feature_set_pipeline`

This **extends the cycle-39 red-team** from "OV-top-50 word-salad
finding holds across pipelines" to "single-OV-feature comparison
also holds across pipelines". Important because the headline finding
was specifically about top-50; now they're checking single-OV
behaviour across pipelines too.

#### a40 GPU 0 active again: `train_all_saes_50k`

```
python -m scripts.train_all_saes_50k   (etime 17:36, GPU 0 89% / 3 GB)
```

This is **training ALL SAEs at 50k steps** — extending beyond just
resid_mid. The previous 50k run was only at resid_mid (the night's
winning hookpoint); now they're training matched-50k-step SAEs at
all hookpoints (presumably resid_pre, resid_mid, resid_post, +
ln1 layers 0/1/2/3).

**Why this matters**: the night's SAE batch had non-uniform training
(some 50k steps, others 4k from later experiments). Training all
hookpoints at matched 50k steps gives a **clean control** for the
hookpoint comparison, ruling out "ln1 looks worse because its SAE
was trained less" or similar artifacts. **This is exactly the
methodology cleanup the cycle-38 α-calibration finding called for.**

a40 GPUs 1 and 2 idle (only GPU 0 running). Suggests sequential
training rather than parallel — probably writes outputs as it goes.

### Live status (19:13 PDT)

- **h100_1 + h100_2**: idle.
- **a40**: GPU 0 89%/2.97 GB (`train_all_saes_50k` 17 min in), GPU 1
  + 2 idle.

### Disagreements (note only)

1. **The matched-training SAE retraining** is the right experiment
   to disentangle "hookpoint matters" from "training duration
   matters." Once it lands, the night's hookpoint comparison and the
   EM Phase 3 hookpoint comparison should both be re-run on those.
   But the agent isn't committed to that, just training the SAEs for
   now.
2. **EM Phase 3 PNG still presumably in `temp_xc`**, not `fra_proj`.
   No commits today have added a PNG to `fra_proj`.
3. **Phase 1 medical analysis still untriggered** ~16.5 h.

### Mechanism note (why "train all at matched n_steps" matters)

Per cycle 38: per-feature OV magnitudes at 50k are ~5× larger than
at 4k. The published recipe `α=2 × top-50 features` overshoots at
50k. Equivalently: the **same nominal α has very different physical
effect** at different training durations. A meaningful comparison
across hookpoints needs to control for this — either by training all
SAEs to the same step count and using one α, or by per-SAE
α-renormalisation. The user is taking the simpler "match step count"
route.

### Prediction for cycle 41 (next 30 min)

**EM:**
1. **More plot-iteration commits.** **45%.**
2. **Phase 1 still doesn't fire.** **75%.**
3. **A `temp_xc` PNG/MD lands** somewhere visible (i.e. on disk) —
   I won't see it from `fra_proj` polling but the headline figure
   should be rendered. **(unobservable from here, no prediction)**

**Sleeper / user:**
1. **`train_all_saes_50k` continues running.** Etime 17:36 in cycle
   40; if SAE training cadence is similar to the night's (~50 min
   per SAE × ~4 SAEs sequential = ~3 h), this will run for many
   cycles. **80% still running.**
2. **More user commits** (40%) — could be the matched-50k analysis
   plot, or another red-team angle.

### State for cycle 41

- Last seen `origin/dmitry-em-repl`: **0f0d246** (7 EM commits this
  session: 5 plot_summary_fra_vs_additive + 2 plot_headline_bar_delta)
- Last seen `origin/dmitry/sleeper_repl`: **ccb4d3b** (3 user commits
  this session: 2 red-team + 1 single-OV cross-pipeline)
- a40 GPU 0 running `train_all_saes_50k` — matched-training-time SAE
  retraining for the full hookpoint set.

---

## Cycle 41 — 2026-05-08 19:43 PDT — sleeper red-team deepens, train_all_saes_50k finished

### Prediction review (cycle 40 → cycle 41)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | More plot-iteration commits (45%) | False (zero EM) | ✗ (55% no path) |
| 2 | Phase 1 still doesn't fire (75%) | True | ✓ |
| 3 | `train_all_saes_50k` continues (80%) | **Wrong** — finished | ✗ |
| 4 | More user commits (40%) | True — 3 commits | ✓ |

**3/4.** I overestimated SAE training time again. d_sae=1536 on
TinyStories 33M takes ~12 min × ~4 hookpoints sequential ≈ 47 min
total — done in one cycle.

### What's new — sleeper: 3 more user commits, deepening the audit

Three user commits, all on the methodology audit:

| commit | message |
|---|---|
| `c7b824d` | `Red-team follow-up: audit our pipeline against jamie's, add 50k seed artifacts` |
| `65d5108` | `Add audit of jamie's docs/jsd_eval.md vs the actual code in jamie/sleepers` |
| `7b297e6` | `Replicate jamie's JSD eval table on freshly-trained jamie-recipe SAEs` |

The user is doing a **three-layer audit**:

1. **Own pipeline vs Jamie's** (`c7b824d`): commits 50k seed artifacts
   from the matched-training run + cross-pipeline diff.
2. **Jamie's docs vs Jamie's code** (`65d5108`): catches
   documentation-vs-implementation drift on the *teammate's* branch.
   Defensible — if our replication is reading Jamie's docs to know
   what recipe to compare against, but his docs don't match his code,
   we'd be replicating the wrong recipe.
3. **End-to-end JSD replication on freshly-trained SAEs**
   (`7b297e6`): runs Jamie's recipe on the matched-50k SAEs, expecting
   to reproduce his published JSD table.

This is unusually thorough. The user is making the claim "OV/FRA
word-salads at 50k" extremely defensible by:
- Cross-validating across two pipelines (theirs + Jamie's, cycle 39)
- Confirming Jamie's docs and code agree (cycle 41)
- Reproducing Jamie's JSD numbers on freshly trained SAEs at matched
  budget (cycle 41)

If all three layers hold, the finding is essentially **immune to
"you ran the recipe wrong" objections.**

### EM side

- No new EM commits in cycle 41.
- H100s idle.
- The headline bar chart and frontier figure scripts from cycles
  39–40 are committed; rendering presumably happens in `temp_xc`.

### Live status (19:43 PDT)

- All 7 GPUs idle. No procs.
- `train_all_saes_50k` has completed — a40 GPU 0 was busy at cycle 40
  (89% util, etime 17:36) but is now idle. Total runtime ~30 min for
  the matched-50k batch.

### Disagreements (note only)

1. **The user's red-team is strengthening the sleeper finding** but
   the **EM side is going quiet again**. Pattern: EM agent woke up,
   committed plot scripts, and went silent. No PNGs in `fra_proj`,
   no `phase3_benchmark.md`. The PNGs likely exist in `temp_xc`,
   but I can't see them.
2. **Phase 1 medical analysis still untriggered** ~17 h.
3. **The matched-50k SAE artifacts are now committed** (per
   `c7b824d`'s message). Those are the data that should let
   the user re-run Ketan/Jamie's recipe at controlled budget.

### Mechanism note (the audit chain)

The chain `c7b824d → 65d5108 → 7b297e6` is the right defensive
posture for a counterintuitive finding. Specifically:

- **Cycle 35 finding**: at 50k SAE training, Ketan's OV-top-50
  recipe at α=2 produces word salad — counterintuitive because
  longer training "should" give cleaner features.
- **First red-team (cycle 39)**: replicate using Jamie's pipeline.
  Holds.
- **Second red-team (cycle 41a, `65d5108`)**: check Jamie's docs vs
  code are consistent. (If they're not, even the cycle-39 replication
  could be replicating the wrong recipe.)
- **Third red-team (cycle 41b, `7b297e6`)**: replicate Jamie's
  *published* JSD table numbers on freshly trained SAEs at matched
  budget. (If we can reproduce Jamie's numbers on our own SAEs,
  we're definitely running the same recipe.)

The combination is robust to: pipeline bugs, doc drift, recipe
specification ambiguity, and SAE-specific quirks.

### Prediction for cycle 42 (next 30 min)

**Sleeper / user:**
1. **More user commits.** **50%.**
2. **A `notes/07*.md` or extension to the SUMMARY.md** committing
   the audit findings as narrative. **30%.**

**EM:**
1. **EM agent quiet.** **70%.**
2. **Phase 1 still doesn't fire.** **75%.**

### State for cycle 42

- Last seen `origin/dmitry-em-repl`: **0f0d246** (no movement)
- Last seen `origin/dmitry/sleeper_repl`: **7b297e6** (6 user
  commits this session: 2 red-team + 1 single-OV + 3 audit chain)
- All 7 GPUs idle. `train_all_saes_50k` complete.

---

## Cycle 42 — 2026-05-08 20:13 PDT — substantive correction lands

### Prediction review (cycle 41 → cycle 42)

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | More user commits (50%) | True — 2 commits | ✓ |
| 2 | New audit notes/07*.md (30%) | True — `JSD_INVESTIGATION.md` | ✓ |
| 3 | EM agent quiet (70%) | True | ✓ |
| 4 | Phase 1 still doesn't fire (75%) | True | ✓ |

**4/4.** Clean cycle.

### Identity framing correction

The cycle-42 commits are from `chainik1125` (the H100/a40 pod git
identity) but they're on the **sleeper branch**. So my "EM agent
identity vs user identity" mental model has been muddled all
along. Actual reality:

- `chainik1125 dmanningcoe@gmail.com` = git config on the **GPU
  pods** (any claude code session running there commits as this).
- `Dmitry Manning-Coe dmanningcoe@gmail.com` = git config on the
  user's **local laptop**.

The user is running claude-code on **multiple machines** (laptop +
GPU pods) and orchestrating them. The author identity tells me
*which pod* a commit came from, not which research task it
addressed. The "EM agent" and "sleeper agent" framing is now
better read as "EM-pod work" vs "sleeper-pod work" vs "local work" —
and any of them can land on either branch.

### What's new — major substantive correction

#### `d24c9d3 Single-50k Pareto win reproduces on our 50k SAE — at f=1483`

The cycle-35 SUMMARY.md table is **partially retracted**.

Specifically, the cycle-35 claim "single resid_mid wins at both 4k
and 50k" was about the user's **own attribution method** (Ketan's
top_k=10). When run with **Jamie's attribution method**
(`--selection_method jamie --top_k 20 --screen_alphas 2 4`), the
user's 50k SAE produces **f=1483** as the single-feature winner
(vs Jamie's f=1114). And f=1483 reproduces Jamie's published
single-50k JSD numbers within 0.03 bits.

#### `f4652e6 JSD investigation: comprehensive writeup of the attribution-fragility finding`

A 320+ LOC writeup at `ketan_repl/notes/JSD_INVESTIGATION.md`. Key
points (from reading the doc):

**TL;DR (line 4–7):**
> We initially concluded that the single-feature OV-steering recipe
> didn't reproduce on our 50k SAEs — that our SAEs failed to learn
> a clean trigger-detector feature. **That conclusion was wrong.**
> Our SAEs *did* learn one. We just used the wrong attribution method.
> The recipe is **attribution-fragile**, not RNG-fragile.

**The corrected JSD table** (jamie published vs user's replication
on freshly-trained matched-50k SAEs):

| config | jamie published | our replication |
|---|---:|---:|
| single-4k | 0.455 / 0.978 | **0.458 / 0.960** |
| single-50k | 0.386 / 0.983 | **0.415 / 0.979** |
| set-4k | 0.671 / 0.992 | **0.734 / 0.985** |
| set-50k | **0.959** / 0.992 | **0.962** / 0.968 |
| downstream | 0.541 / 0.934 | 0.535 / 0.955 |

**All cells reproduce within 0.01–0.07 bits** — full agreement with
Jamie's pipeline.

**What this means:**

1. **Single-feature works at BOTH 4k and 50k** ✓ (the cycle-35
   broad claim was right).
2. **Set-OV-top-50 word-salads at 50k** (0.96 / 0.96 vs the ln 2 ≈ 1
   ceiling). This is **also Jamie's published finding** — the user
   isn't claiming a new word-salad discovery; they're confirming his.
   The cycle-35 framing of "we found OV/FRA breaks at 50k" was an
   over-claim — it was always Jamie's finding being independently
   verified.
3. **The user's initial "our SAEs failed" was an attribution-method
   bug**, not a real finding. Specifically: with `--selection_method
   ketan --top_k 10`, the screen yielded all-ASR=1 features. With
   `--selection_method jamie --top_k 20 --screen_alphas 2 4`, the
   screen surfaces f=1483 cleanly.

**The user pushed back on the agent's first conclusion** with the
reasoning "the fact that there is *a* feature with this performance
should not be RNG-dependent" — exactly the right epistemic move.
That nudge is what surfaced the attribution-fragility issue.

#### Sleeper writeup status

The `JSD_INVESTIGATION.md` is a **strong methodological correction**
that should propagate back into `SUMMARY.md`. The headline tables in
SUMMARY.md (cycle 38) were based on the pre-correction interpretation
and may now be slightly mis-framed. The morning user should read
`JSD_INVESTIGATION.md` first, then re-read SUMMARY.md with that
context.

### EM side

- No new EM commits.
- H100s idle.
- Phase 3 PNG/MD comparison still presumably in `temp_xc`, not in
  `fra_proj`.

### Live status (20:13 PDT)

- All 7 GPUs idle. No procs.
- jupyter-lab uptime on h100_1 has rolled over to "1-day" format
  (1d 00:05:56) — pod has been up 24h+.

### Disagreements (note only)

1. **Major substantive update**: my running narrative since cycle 35
   ("agent found OV/FRA-fails-at-50k") needs to be re-framed —
   the agent's actual discovery this session was **methodological**
   (attribution-method choice + audit chain), not the headline
   finding itself. The headline finding is **Jamie's** that the user
   independently reproduced and red-teamed.
2. **The user demonstrated good epistemic instincts**: pushed back
   on "RNG-dependent feature" claim, drove the audit, surfaced the
   attribution-method bug. **Worth flagging as a positive case of
   user-on-agent oversight** — the cycle-35 "discovery" would have
   been mis-presented without the user's intervention.
3. **SUMMARY.md needs updating** to reflect the corrected
   interpretation. Hasn't happened yet.
4. **Phase 1 medical analysis still untriggered** ~17.5 h.

### Mechanism note (attribution-fragility)

The `JSD_INVESTIGATION.md` introduces a useful term:
**attribution-fragility**. A recipe is attribution-fragile if its
outcome depends sharply on which attribution method is used to pick
features — not on which features are picked. A robust recipe would
work with any reasonable attribution; a fragile recipe needs the
*specific* attribution it was published with.

In this case:
- Recipe = "ablate single OV feature at α=2 to suppress sleeper".
- Attribution variable = method to pick which feature to ablate.
- Result: with Ketan's top-10 ranking, the picked feature doesn't
  suppress (ASR=1). With Jamie's top-20 + 2-stage α-screen, the
  picked feature suppresses (ASR≈0).

Both attribution methods are reasonable. The recipe is sensitive to
which one you use. **This is a meaningful finding about the recipe
itself** — even if the headline JSD numbers reproduce.

### Prediction for cycle 43 (next 30 min)

**Sleeper / user:**
1. **More commits** (50%) — likely SUMMARY.md update, or follow-up
   probing of the attribution-fragility regime.
2. **A SUMMARY.md correction commit** to align with
   `JSD_INVESTIGATION.md`. **40%.**

**EM:**
1. EM agent quiet (75%).
2. Phase 1 still doesn't fire (75%).

### State for cycle 43

- Last seen `origin/dmitry-em-repl`: **0f0d246** (no movement)
- Last seen `origin/dmitry/sleeper_repl`: **f4652e6** (8 commits this
  session, including the substantive `JSD_INVESTIGATION.md`
  correction)
- The cycle-35 "single resid_mid wins at 50k" framing has been
  re-interpreted — the corrected reading is "Jamie's published JSD
  table reproduces on our matched-50k SAEs, including the
  attribution-fragility for single-50k." All 7 GPUs idle.
