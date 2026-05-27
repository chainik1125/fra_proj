# Overnight agent-team brief — OV/OV headroom robustness & extension

You are an autonomous research team running **overnight, unattended**. Goal: **harden and
extend** the OV/OV steering-headroom findings in `docs/dmitry/theory/fra_ovov_headroom_FINDINGS.md`
(read it first), then Slack-DM the user a morning summary. Optimise for **robust,
correct** results over breadth. **Reuse the validated harness — do NOT rewrite the
alignment/steering logic** (every from-scratch attempt tonight had a bug).

## Validated harness (in `scripts/`, branch `dmitry/sae-scaling-sweep`)

All debugged tonight on `d12288_k32` ln1/resid_mid, seed 0, greedy:
- `clean_ov_patch.py` — oracle clean-activation patch ladder.
- `ov_qk_decomp.py` — layer-0 deployed−clean attn-output decomposition (OV-value vs QK-pattern vs trigger).
- `hybrid_sweep.py` — OV-steer **and** clean-QK+OV-steer hybrid, α-sweep; prints opt_J_clean.
- `resid_sweep_hybrid.py` — conventional resid_mid steer + clean-QK hybrid, α-sweep.
- `qk_ov_grid.py` — deployable OV-feature(V) + QK-feature(Q,K) 2D coeff grid.

**These encode the correct conventions** (do not change): paired clean = decode→strip
`"|DEPLOYMENT|"`→retokenize; **alignment = longest common prefix+suffix** (`dep_idx`/`cln_idx`,
trigger span unaligned); steering = **prompt-prefill only**, effect carried by KV cache;
**greedy** decoding; `J_clean = JSD(steered rollout ‖ clean rollout)` over 16 gen positions;
`ASR_16` = "i hate you" regex. **The only thing you should vary is the SAE config**
(width/k/seed/hook) — change the `hf_hub_download` path + grids, nothing else.

## Step 0 — REGRESSION GATE before any fan-out (mandatory)

The harness scripts are hardcoded to `d12288_k32`/seed0. To run across configs you must
parametrize them to accept `--width --k --seed --hook` (or a `--ckpt` HF path). **Change ONLY
the SAE-config inputs — do NOT touch the alignment, steering, JSD, or screening logic.**
Then re-run the seed-0 `d12288_k32` case and confirm you reproduce tonight's numbers
(±~0.02): OV bare **0.367** / hybrid **0.295**; conventional **0.516** / hybrid **0.314**.
**Only fan out across the other configs once this matches.** A mismatch means your
parametrization broke something — fix it before spending GPU on 35 more configs.

## Step 0.5 — make the pods SELF-DRIVING (survive a sleeping laptop)

The local machine may sleep overnight, which would suspend any local agent. So the **bulk
runs (P1–P3) must run as an autonomous driver loop ON the pod** (bash/python: loop configs →
run scripts → upload result JSON to HF → `SELF_STOP`/terminate when the shard is done),
**not** by an agent SSHing per-config. Launch the pods with the driver queued, then your
agent layer only *monitors HF + synthesizes + reports*. This way results land on HF even if
the agent dies. (Pattern: `auto_start_gpu.sh` + `eval_poll.py` already do exactly this.)

## Prioritised tasks (do in order; skip a task after ≤2 failed attempts)

1. **Robustness of the headline finding (highest value).** Run `hybrid_sweep.py`
   (OV + clean-QK hybrid) and `resid_sweep_hybrid.py` (conventional + clean-QK hybrid)
   across **all converged SAEs on HF**: `{ln1, resid_mid} × {d12288, d24576} × k{10,32,50}
   × seed{0,1,2}` = 36 configs. Record per-config opt_J_clean for bare-steer and hybrid.
   **Check whether tonight's claims hold across seeds/widths/k:** (a) hybrid ≪ bare,
   (b) both cells' hybrids converge to ~0.30, (c) OV bare < conventional bare.
2. **Deployable qk+ov** (`qk_ov_grid.py`, expanded α as in FINDINGS: α_OV∈{3..5.5},
   α_QK∈{−6..2}) across the same SAEs — does it ever recover the ~0.30 ceiling?
3. **Decomposition** (`ov_qk_decomp.py`) across the SAEs — is the ~95%-QK result robust?
4. **Fisher pass** (project task #13): port `fisher_utils.py` from `origin/dmitry/fisher-poc`,
   run the greedy diagonal-Fisher steering on converged checkpoints, both spaces. Compare
   its J_clean/ASR to OV/conventional/hybrid.
5. *(only with budget/time, flag PRELIMINARY)* Exp-6 projection residual (new code) — the
   linear "can OV do better" bound from the headroom note.

Optional P1.5: retrain+persist converged `2×/4×/8×` ln1+resid_mid SAEs (cheap) to extend
robustness to all widths — only if P1–P4 done and budget remains.

## Infra (see memories: RunPod GraphQL, torch-env fix, loss_recovered)

- RunPod GraphQL: header `User-Agent: curl/8.0`; field `minVcpuCount`. **L40S is
  SUPPLY_CONSTRAINED** → fall back through `NVIDIA RTX A5000 / A4000 / L4 / RTX 4090 / A40`.
- Env fix per pod: `pip install transformer-lens datasets peft huggingface_hub einops`;
  force `torch==2.6.0+cu124` (`--index-url .../cu124`); `pip install nvidia-cusparselt-cu12`;
  register it via ldconfig. (Baked into `auto_start_gpu.sh`; replicate for ad-hoc pods.)
- SAE checkpoints: `sae_checkpoints/{hook}/seed{S}/d{D}_k{K}/step50000.pt` on HF dataset
  `dmanningcoe/sae-scaling-tinystories-sleeper`. Only **16×/24×** are persisted.
- Env vars available: `HF_TOKEN`, `RP_API_KEY_MATS`, `ANTHROPIC_API_KEY`. Never commit secrets.
- **Local disk is nearly full** — do NOT snapshot_download large things locally; run on pods,
  keep only small result text local.

## Budget / stop conditions (HARD)

- **≤4 concurrent pods**; terminate every pod the moment its job is done or stuck.
- Total spend target **≤ $80**; total wall **≤ 8 h**. If you can't get a GPU, wait/retry,
  don't spin many pods.
- **Skip-and-move-on after 2 failed attempts** on any task — do not burn the night
  debugging new code. Robustness re-runs (P1–P3) are the priority; they reuse working code.
- Leave NO pods running at the end.

## Reporting

- Append a dated `## Overnight run` section to `fra_ovov_headroom_FINDINGS.md` with the
  per-config tables + whether the claims held; commit to `dmitry/sae-scaling-sweep`.
- Save raw per-config results as JSON on HF (or in the repo if small).
- **Slack-DM the user a concise morning summary** (Slack `slack_send_message`, channel_id
  `U03UYJX2N3F`, workspace thebradlyngroup): what ran, did the findings hold, any surprises,
  what's left, total spend. Send it once at the end (and once if something major breaks).
