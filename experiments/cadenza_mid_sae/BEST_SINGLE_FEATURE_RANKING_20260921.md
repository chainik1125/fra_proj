# Best individual steering setting by method

Ranking uses each method/hookpoint’s best observed setting among the completed, validation-frozen positive-only and signed-grid choices at layers 8, 16 and 24. It does not average over features, coefficients or layers. This is a descriptive best-observed comparison after looking at the test results, not a new validation-selected winner across selection rules and layers. It is not an exhaustive test-set optimization over every validation-grid setting.

All entries below use the identical original 64-prompt test set; unsteered triggered-to-clean rollout JSD is 0.990536 bits. JSD still averages over these prompts and their rollout steps. DoM lacks results on the later confirmation block, so that block is not mixed into this ranking.

| Rank | Method / hookpoint | Layer | Feature(s) | Coefficient | JSD to clean | IHY removed /64 | Clean drift | Clean preserved /64 |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 1 | SAE residual-mid | 8 | 12801 | 32 | 0.796058 | 57 | 0.00983326 | 62 |
| 2 | FRA OV-only | 8 | 30892 | 16 | 0.829045 | 64 | 0.00824989 | 63 |
| 3 | DoM residual-post/response | 8 | one DoM vector | 4 | 0.846182 | 64 | 0.789291 | 0 |
| 4 | SAE residual-post | 8 | 32158 | 8 | 0.851159 | 31 | 6.3115e-09 | 64 |
| 5 | FRA QK+OV | 8 | 31879/5130/10231 | 16 | 0.852741 | 51 | 0.0678261 | 54 |
| 6 | SAE attention-input | 8 | 21015 | 8 | 0.872753 | 50 | 0.723752 | 1 |
| 7 | DoM attention-input/prompt | 8 | one DoM vector | 4 | 0.934823 | 63 | 0.907643 | 0 |

All winning settings happen to be at layer 8, and the SAE/FRA rows coincide with the positive-only choices. DoM coefficients were selected using its signed grid, but all selected DoM coefficients are positive.

FRA QK+OV uses a Q/K/V feature tuple and is not a single-feature intervention. The strict single-SAE-feature ordering is residual-mid, FRA OV-only, residual-post, then attention-input. Each DoM intervention uses one dense mean-difference vector, not an SAE feature. DoM residual-post/response edits the last prompt position and subsequent generation positions; the SAE/FRA interventions are prompt-only. Coefficient units and perturbation norms differ across methods.

Clean preserved measures exact equality to unsteered clean continuations. The DoM residual/response result is numerically competitive in restoration JSD but changes all 64 clean continuations. These are observed means, not statistically established pairwise ranks.

Sources: [retrieved SAE/FRA data](STEERING_RESULTS_20260921.json), [DoM documentation](CAA_DOM.md), and [retrieved DoM data](DOM_RESULTS_FOR_RANKING_20260921.json).
