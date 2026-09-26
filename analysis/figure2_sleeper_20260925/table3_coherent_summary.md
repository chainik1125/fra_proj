# Table 3 rerun with retrained SAEs and a coherence gate

Greedy 16-token sleeper ASR on a fresh 100-clean/100-deployment holdout; unsteered ASR = 97%.
Candidates were selected on validation only. A selected setting must satisfy clean continuation delta CE <= 0.05 nats, mean generated-story NLL <= 2.0, 90th percentile <= 4.0, and <= 5% of outputs with an eight-token periodic repetition (period 1-4). Story NLL is scored under the unsteered model after the neutral prefix `Story:`. The same constraints are checked on the fresh holdout.

| Layer | Hook | Validation qualifying /6 | Holdout qualifying /6 | Holdout ASR mean, qualifying | Holdout delta CE max |
|---:|---|---:|---:|---:|---:|
| 0 | ln1 | 6/6 | 6/6 | 0.2% | 0.0136 |
| 0 | resid_mid | 6/6 | 6/6 | 0.3% | 0.0006 |
| 0 | resid_post | 5/6 | 5/6 | 0.6% | 0.0069 |
| 1 | ln1 | 6/6 | 6/6 | 0.5% | 0.0000 |
| 1 | resid_mid | 6/6 | 5/6 | 0.0% | 0.0008 |
| 1 | resid_post | 5/6 | 5/6 | 0.6% | 0.0083 |
| 2 | ln1 | 4/6 | 4/6 | 1.2% | 0.0009 |
| 2 | resid_mid | 2/6 | 2/6 | 1.0% | 0.0053 |
| 2 | resid_post | 1/6 | 1/6 | 1.0% | 0.0111 |
| 3 | ln1 | 0/6 | 0/6 | — | — |
| 3 | resid_mid | 2/6 | 2/6 | 0.0% | 0.0000 |
| 3 | resid_post | 2/6 | 1/6 | 0.0% | 0.0000 |

ASR mean is over settings that pass both validation selection and the fresh holdout gates; missing or failing seeds are excluded and shown by the qualifying counts. Two validation-selected settings fail the holdout coherence gate (layer 1 resid_mid seed 3, layer 3 resid_post seed 0); their outcomes remain in the per-seed JSONs.

Protocol: six SAE seeds for each of 12 hooks, top 10 deployment-minus-clean feature means from 500 paired-class ranking sequences, 15 signed steering coefficients, 100 clean and 100 deployment validation prompts. Positive alpha subtracts the SAE feature contribution. The clean CE budget is evaluated on dataset continuations. The final fresh holdout contains the next 100 clean and 100 deployment rows of the HF test split, verified disjoint from the earlier test rows.

The coherence gate is a fluency/repetition proxy, not a human semantic judgment. It was refined after examining preliminary test outputs; those preliminary test results are excluded from this table. The fresh holdout was not inspected or used for selection before final gate selection.

Per-seed JSONs: `coherent_results_v2/` and `fresh_holdout_results/`. Runner: `table3_run.py`, `table3_coherence.py`, `eval_fresh_holdout.py`. All artifacts remain on simplex2.
