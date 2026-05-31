"""Guard sae-lens periodic eval against ZeroDivisionError, keeping the CE eval.

Injected by cell_runner via PYTHONPATH (this dir on sys.path → Python auto-imports
`usercustomize` at startup). The sleeper repo is NOT modified — the fix lives on
jamie/llama-sleeper-repro.

Bug: `sae_lens/evals.py:get_downstream_reconstruction_metrics` computes
  ce_loss_score = (ce_with_ablation - ce_with_sae) / (ce_with_ablation - ce_without_sae)
  kl_div_score  = (kl_with_ablation  - kl_with_sae)  /  kl_with_ablation
which crash when the denominator is exactly 0 (e.g. mean-ablating L10
`ln1.hook_normalized` leaves downstream CE unchanged → 0). This faithfully
reimplements the function with the two divisions guarded (denominator 0 → score
0.0). All CE/KL sub-metrics are still computed + logged; explained_variance and
dead_features come from a separate function and are untouched.
"""
try:
    import torch
    import sae_lens.evals as _ev

    def _guarded_downstream(sae, model, activation_store, activation_scaler,
                            compute_kl, compute_ce_loss, n_batches,
                            eval_batch_size_prompts, ignore_tokens=None, verbose=False):
        metrics_dict = {}
        if compute_kl:
            metrics_dict["kl_div_with_sae"] = []
            metrics_dict["kl_div_with_ablation"] = []
        if compute_ce_loss:
            metrics_dict["ce_loss_with_sae"] = []
            metrics_dict["ce_loss_without_sae"] = []
            metrics_dict["ce_loss_with_ablation"] = []

        for _ in range(n_batches):
            batch_tokens = activation_store.get_batch_tokens(eval_batch_size_prompts)
            for metric_name, metric_value in _ev.get_recons_loss(
                sae, model, activation_scaler, batch_tokens,
                compute_kl=compute_kl, compute_ce_loss=compute_ce_loss,
                ignore_tokens=ignore_tokens,
            ).items():
                if ignore_tokens:
                    mask = torch.logical_not(torch.any(torch.stack(
                        [batch_tokens == t for t in ignore_tokens], dim=0), dim=0))
                    if metric_value.shape[1] != mask.shape[1]:
                        mask = mask[:, :-1]
                    metric_value = metric_value[mask]
                metrics_dict[metric_name].append(metric_value)

        metrics = {}
        for metric_name, metric_values in metrics_dict.items():
            metrics[metric_name] = torch.cat(metric_values).mean().item()

        if compute_kl:
            d = metrics["kl_div_with_ablation"]
            metrics["kl_div_score"] = (
                (d - metrics["kl_div_with_sae"]) / d) if d != 0 else 0.0
        if compute_ce_loss:
            d = metrics["ce_loss_with_ablation"] - metrics["ce_loss_without_sae"]
            metrics["ce_loss_score"] = (
                (metrics["ce_loss_with_ablation"] - metrics["ce_loss_with_sae"]) / d
            ) if d != 0 else 0.0
        return metrics

    _ev.get_downstream_reconstruction_metrics = _guarded_downstream
    import sys as _sys
    print("[saeguard] patched sae_lens.evals.get_downstream_reconstruction_metrics",
          file=_sys.stderr)
except Exception as _e:  # never block training on the guard
    import sys as _sys
    print(f"[saeguard] NOT applied: {_e!r}", file=_sys.stderr)
