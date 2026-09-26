# Pipeline code (provenance of `data/`)

Everything under `data/` was produced by the code in this directory together with the
`fra/` library. These scripts are included so the full method is inspectable; they are
**not** run by the README commands because they need GPUs, the fine-tuned models, and
the trained SAE weights (several GB) that are not part of this bundle.

| Directory | What it produced | Notes |
|---|---|---|
| `tinystories_sleeper/sleeper/` | TinyStories sleeper package: model/SAE loading, FRA-OV / QK attribution, steering hooks, JSD / ASR / exact-match metrics, screening | imported by the scripts below |
| `tinystories_sleeper/wide_screen/` | `run_layers.py`, `run_layers_post.py`: the 81-coefficient, 64-prompt screen + GP-BO refinement per (layer, method, SAE seed) behind `data/tinystories/wide_screen_L0/`, `wide_screen_layers.json`, `steering_layers.json` | `sae_quality_L0.py`: 1-FVU of the 20k-step retrain |
| `tinystories_sleeper/sae_quality/` | `sae_loss_recovered.py`, `attention_term_only_fvu.py`, `attention_decomposition_fvu.py`: the SAE and per-term FRA reconstruction numbers in `data/tinystories/sae_quality/` | `plot_*.py` are the original diagnostic plots |
| `tinystories_sleeper/sweep_directional_candidates.py` | conventional (resid-mid) directional steering sweep used for the attribution x intervention matrix | |
| `tinystories_sleeper/autointerp/` | `collect_fig.py`, `collect_maxact.py`, `deployment_maps.py`: inputs of the supporting auto-interpretability figure | Claude-as-judge labels |
| `cadenza/` | attention-only Cadenza sleeper (Llama-3-8B): `train.py` (SAE training at the attention input / residual hooks), `steering.py` (FRA-OV, single-SAE-feature and DoM steering with JSD/ASR evaluation), `single_eval.py`, `restoration.py`, `caa_eval.py`, `dom_layers.py`, `dom_confirmation.py`, `confirmation_curve.py`, `cadenza_reeval_v2.py` + `cadenza_reeval_v2_ci.py` (the final 293-pair re-evaluation behind Fig. 4 and its bootstrap CIs), `campaign.py` / `launch.py` / `remote.py` (dispatch to GPU hosts) | produced `data/cadenza/*.json` |
| `em/` | Qwen-2.5-14B emergent-misalignment steering: `phase1_fra_orchestrator.py` (FRA QK->QK / QK->OV / OV->OV recipes), `phase1_additive_orchestrator.py` (conventional additive steering on five SAEs), `judge_multiseed.py` + `phase1_judge_and_combine.py` (GPT-4o alignment/coherence judging and cross-seed aggregation) | produced `data/em/combined/` |

`requirements-pipeline.txt` is the pinned environment those runs used (torch, transformer-lens,
sae-lens, ...). The bundle's own `pyproject.toml` deliberately only carries what the figure
scripts need.

## Anonymisation placeholders

* `<anonymized-hf-org>/...` -- Hugging Face repos under an author's account (the fine-tuned
  Cadenza sleeper, the surrounding-hookpoint SAEs, steering vectors). They will be made
  public on de-anonymisation.
* `<published-sae-org>/Qwen2.5-14B_SAE_ln1.normalised` -- the published Qwen-14B attention-input
  SAE used for the EM experiments; the account name would identify an author.
* `gpu-host-N`, `/archive/...` -- compute hosts and archive paths.
* `head_summed` / `head_resolved` -- the two OV attribution conventions in
  `sleeper/attribution.py` (previously named after the people who wrote them).
