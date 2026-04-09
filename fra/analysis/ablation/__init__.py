"""FRA ablation subpackage.

Re-exports all public symbols from ablate and generate so that
``from fra.analysis.ablation import ablate_fra_pairs`` continues to work.
"""

from fra.analysis.ablation.ablate import (
    ABLATION_TEXTS,
    ablate_fra_pairs,
    aggregate_results,
    compute_bias_corrections,
    compute_fra_new_query,
    print_results,
    reconstruct_scores,
    run_condition,
    run_single_sample,
    screen_heads,
)
from fra.analysis.ablation.generate import (
    build_patch_scores,
    generate_with_ablation,
    generate_with_prefill_ablation,
    prefill_no_hooks,
    prefill_with_patch,
)

__all__ = [
    "ABLATION_TEXTS",
    "ablate_fra_pairs",
    "aggregate_results",
    "build_patch_scores",
    "compute_bias_corrections",
    "compute_fra_new_query",
    "generate_with_ablation",
    "generate_with_prefill_ablation",
    "prefill_no_hooks",
    "prefill_with_patch",
    "print_results",
    "reconstruct_scores",
    "run_condition",
    "run_single_sample",
    "screen_heads",
]
