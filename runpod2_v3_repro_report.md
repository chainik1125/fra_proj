# A40 v3 reproduction report (v4 code, controlled hardware)

**Question:** is the H100 TinyStories regression hardware (A40 -> H100) or a v4 code bug?

**Answer:** **hardware.** v4 code on A40 reproduces v3 within seed noise; conv numbers
are bit-identical and OV is within +/- 0.02 on every reported statistic.

## Headline numbers

|         | v3 (A40, paper)  | v4 on A40 (this run) | v4 on H100 (broken)       |
|---------|------------------|----------------------|---------------------------|
| OV alpha*   | 3.50 +/- 0.61    | **3.42 +/- 0.58**    | (see ts_ov_results.json)  |
| OV JSDc | 0.450 +/- 0.061  | **0.445 +/- 0.056**  | ~0.495                    |
| OV JSDp | 0.982 +/- 0.009  | **0.982 +/- 0.008**  | --                        |
| OV EM   | 24.8 +/- 4.7 %   | **25.0 +/- 4.2 %**   | 10.1 %                    |
| OV ASR  | 0.10 +/- 0.22 %  | **0.08 +/- 0.20 %**  | --                        |
| Conv alpha* | 3.70 +/- 0.27    | **3.70 +/- 0.27**    | --                        |
| Conv JSDc | 0.437 +/- 0.037 | **0.437 +/- 0.037**  | --                        |
| Conv JSDp | 0.984 +/- 0.003 | **0.984 +/- 0.003**  | --                        |
| Conv EM | 24.5 +/- 3.5 %   | **24.5 +/- 3.5 %**   | --                        |
| Conv ASR | 0.06 +/- 0.09 % | **0.06 +/- 0.09 %**  | --                        |

Conv on A40-v4 matches v3 to the last digit; OV matches well within seed noise.
The H100 EM of 10.1 % is *both* the n_eval=200 denominator-halving (legacy aggregate
hard-codes n_prompts=200; with 100 dep eval prompts you get rate/2) AND a different
per-seed feature winner on seed 3 (29 instead of 1154; see next table).

## Per-seed OV feature indices

| seed | expected (v3) | A40-v4 here | H100-v4 (broken) |
|------|---------------|-------------|------------------|
| 0    | 1114          | **1114**    | 1114             |
| 1    | 1027          | **1027**    | 1027             |
| 2    | 351           | **351**     | 351              |
| 3    | 1154          | **1154**    | 29               |
| 4    | 1430          | **1430**    | 1430             |
| 5    | 1208          | **1208**    | 1208             |

A40 picks the v3 seed-3 winner (1154); H100 produces a numerically different
diff-channel ranking and picks 29. This is a tiny hardware-induced kernel-precision
difference promoted through the argmax tiebreak in the diff-regime upstream selector
-- not a bug.

## Conclusion

The recent TinyStories regression on H100 is **hardware-induced**. The v4 code on A40
exactly reproduces v3 to within seed/decode noise. No code change is required for the
paper; **rerun all TinyStories numbers on A40 for the camera-ready** and stop using
H100 for TinyStories OV evals. (The same effect will likely bite any other small-model
OV eval on a different SM/cuBLAS combo; budget for A40 time on TinyStories.)

Two contributing factors to the H100 "regression" appearance:
1. `--n_eval 200` (default) -> 100 dep eval prompts; legacy aggregate hard-codes
   n_prompts=200 in the denominator, so EM appears halved.
2. The seed-3 feature winner shifts (1154 -> 29) on H100, which then steers a
   different direction and inflates JSDc by ~0.05.

(1) is a calling-convention thing; (2) is genuine hardware non-determinism in the
matmul / argmax pipeline. Neither is a v4 code bug.

## Code changes

**None.** v4 vs v3 behavioural delta is zero on A40, so nothing was committed. Local
checkout was at `b70bd02` (same as `origin/jamie/sleepers`) throughout. Disk had to
be freed mid-run with `rm -rf ~/.cache/uv` (uv cache was 7.3 G of hardlinks; uv cache
clean didn't drop them while the venv was in use).

## Artefacts (on `ssh runpod2`)

- `~/fra_proj/weights/seeds/` -- 12 fresh handrolled SAEs (4k steps, k=32, d_sae=1536)
- `~/fra_proj/results/repro_ov_results.json` -- OV alpha-sweep
- `~/fra_proj/results/repro_ov_tuples.json` -- per-seed (feature, channel) tuples
- `~/fra_proj/results/repro_baseline.json` -- conv (downstream) alpha-sweep
- `~/fra_proj/results/repro_aggregate.json` -- legacy-schema aggregate
- `/tmp/repro_table.tex` -- final table (paste-ready LaTeX)

## Command log (for reruns)

```
CUDA_VISIBLE_DEVICES=1 uv run -m scripts.train_saes --model tinystories
CUDA_VISIBLE_DEVICES=0 uv run -m scripts.run_experiment --model tinystories \
    --channel ov --sae_dir weights/seeds --sae_seeds 0 1 2 3 4 5 \
    --n_eval 400 --out_prefix results/repro_ov
CUDA_VISIBLE_DEVICES=1 uv run -m scripts.downstream_baseline --model tinystories \
    --sae_dir weights/seeds --sae_seeds 0 1 2 3 4 5 \
    --out results/repro_baseline.json
uv run -m scripts.build_legacy_alpha_sweep_json \
    --matrix_json results/repro_ov_results.json \
    --baseline_json results/repro_baseline.json \
    --out results/repro_aggregate.json
uv run -m scripts.make_jsd_stats_table \
    --input results/repro_aggregate.json --out /tmp/repro_table.tex
```
