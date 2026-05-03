# Attribution × Intervention Pipeline

`scripts/sleepers_pipeline.py` runs any cell of the matrix
`{ov, qk, triple} × {ov, qk, all}` via two flags: `--attr` and `--intervene`.

## Channel-routing rule

Each attribution row produces selected features tagged with their **natural channel**:

- **OV** → every feature tagged `V`.
- **QK** → top-K Q-side features tagged `Q`, top-K K-side features tagged `K` (2K total).
- **Triple** → each top triplet `(a, b, c)` expands to `(a, Q)`, `(b, K)`, `(c, V)`.

Each intervention column defines the **active channel set**: `ov={V}`, `qk={Q,K}`, `all={Q,K,V}`.

For every active channel `c`:

- if any selected feature is naturally tagged `c` → patch `hook_c` using **only** those features (channel-routed)
- else → fudge: patch `hook_c` using **all** selected features (Dmitry-style replication)

## How each cell is implemented

| cell | hook_q sources | hook_k sources | hook_v sources | hook used |
|---|---|---|---|---|
| **OV + ov** | — | — | OV-feats (natural V) | `attn.hook_v` only |
| **OV + qk** | OV-feats (fudge) | OV-feats (fudge) | — | `attn.hook_q` + `attn.hook_k` |
| **OV + all** | OV-feats (fudge) | OV-feats (fudge) | OV-feats (natural V) | `ln1.hook_normalized` (fast-path: identical deltas) |
| **QK + ov** | — | — | Q-feats ∪ K-feats (fudge) | `attn.hook_v` only |
| **QK + qk** | Q-feats (natural) | K-feats (natural) | — | `attn.hook_q` + `attn.hook_k` |
| **QK + all** | Q-feats (natural) | K-feats (natural) | Q-feats ∪ K-feats (fudge V) | `attn.hook_q` + `attn.hook_k` + `attn.hook_v` |
| **Triple + ov** | — | — | c (natural V) | `attn.hook_v` only |
| **Triple + qk** | a (natural Q) | b (natural K) | — | `attn.hook_q` + `attn.hook_k` |
| **Triple + all** | a (natural Q) | b (natural K) | c (natural V) | `attn.hook_q` + `attn.hook_k` + `attn.hook_v` |

Notes:

- **Triple+all is strictly less invasive** than OV+all or QK+all — each feature enters exactly one channel instead of being broadcast to all three.
- **OV+all takes a fast-path**: when all three resolved channel deltas are the same tensor, the script patches `ln1.hook_normalized` once instead of doing three einsum projections — mathematically equivalent.
- **QK+qk diverges from Dmitry's `pareto_3x3.py`** (which sent a single flat ln1-delta through both W_Q and W_K). We track Q vs K separately, so each side hits only its natural projection.
- The single hook primitive `sleeper.hooks.channel_steer_hook` handles every non-fast-path cell uniformly. `ov_only_steer_hook` is now a one-line wrapper.

## CLI examples

```bash
# Default (OV+ov — preserves existing behaviour)
python -m scripts.sleepers_pipeline --sae_ln1 weights/seeds/sae_ln1_s0.pt \
    --sae_mid weights/sae_resid_mid.pt --target_feature 579

# QK rank, channel-routed QK intervention
python -m scripts.sleepers_pipeline ... --attr qk --intervene qk --top_k 3

# Triple rank, all-channel routed (a→Q, b→K, c→V per triplet)
python -m scripts.sleepers_pipeline ... --attr triple --intervene all \
    --top_k 3 --triple_score qkv_l1

# Explicit feature override
python -m scripts.sleepers_pipeline ... --attr triple --intervene qk \
    --features 870 1388 1114        # one triplet (μ=870, ν=1388, λ=1114)
```

## Library structure

- `sleeper/qk_attribution.py` — softmax-Jacobian per-feature ranking (Q-side and K-side simultaneously, chunked over batch).
- `sleeper/triple_attribution.py` — Möbius 8-corner decomposition using §5's algebraic shortcut (no model re-runs; 4 corner attentions derived analytically from `A_clean[k]`).
- `sleeper/hooks.py` — adds `channel_steer_hook`; `ov_only_steer_hook` is a one-line wrapper.
- `sleeper/attribution.py` — adds `compute_centered_g` helper shared by QK and Triple.
- `sleeper/sae.py` — `encode_all` promoted from the script for QK/Triple to share.
- `tests/` — Möbius §7 worked-example check; channel-routing matrix smoke; centered_g sanity.
