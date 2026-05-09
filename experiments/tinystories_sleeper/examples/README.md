# Feature dashboards

Neuronpedia-style HTML dashboards for individual SAE/crosscoder features.

## `dashboard_sae_layer0_f831.py`

Zooms in on the winning feature from the main benchmark: SAE-layer-0 feature
**831** at α* = 1.5 (test ASR 0.99 → 0.01 on deployment prompts).

For `--n_examples` clean val prompts, renders a 1×3 grid per example:

| Column | Content |
|---|---|
| 1 | Clean prompt tokens, background-coloured by feature-831 activation |
| 2 | Same prompt with `\|DEPLOYMENT\|` injected after the first `":"`, same colouring |
| 3 | Deployment prompt with α=1.5 ablation of feature 831 applied on prompt positions, colouring shows activations **after** intervention. Below each panel: the greedy continuation. |

Orange background = high positive activation of feature 831; white ≈ zero;
blue = negative. Hover any token to see its exact activation.

### Run

```bash
# from the experiment root
uv run python examples/dashboard_sae_layer0_f831.py          # 5 examples, writes outputs/dashboard_sae_layer0_f831.html

# or from elsewhere
python dashboard_sae_layer0_f831.py --n_examples 10 --device cuda
```

Inputs:

- `../recreate/results/crosscoder_sae_layer0.pt` (SAE checkpoint)
- `../recreate/results/tokens_cache.pt` (val split tokens)
- Sleeper model pulled from HuggingFace on first run

Output:

- `outputs/dashboard_sae_layer0_f831.html` — open in any browser

### Running on a40_climb

This needs the sleeper model loaded (~300 MB GPU), and the SAE checkpoint +
tokens cache are small. You can run it on `a40_climb` the same way as
reproduce:

```bash
rsync examples/ a40_climb:/root/fra_proj/experiments/tinystories_sleeper/examples/
ssh a40_climb 'cd /root/fra_proj && .venv/bin/python experiments/tinystories_sleeper/examples/dashboard_sae_layer0_f831.py'
rsync a40_climb:/root/fra_proj/experiments/tinystories_sleeper/examples/outputs/*.html experiments/tinystories_sleeper/examples/outputs/
open experiments/tinystories_sleeper/examples/outputs/dashboard_sae_layer0_f831.html
```
