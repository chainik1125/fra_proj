# Cloud PoC Report — opus-4.8 agent team trains transformer_lens on a Gaussian task

**Date:** 2026-06-03
**Driver:** Claude Code (`claude-opus-4-8`) running in an Anthropic cloud sandbox
**Repo/branch:** `astera-org/simplex-research` @ `dmitry/bag/main-claude_2_active_ec`

This is a **minimal feasibility test** of the loop: *cloud sandbox + opus-4.8 + a fanned-out
agent team + real torch / transformer_lens training*. It is **not** the full active-bag sprint
and does **not** provision a RunPod GPU. Each piece is reported as worked / didn't below.

> This run re-proves the loop in a fresh sandbox with a freshly fanned-out agent team. The team
> wrote new, independent files (`cloud_poc/agent_model_task.py`, `cloud_poc/agent_train.py`)
> against a shared interface contract; those agent-authored files are what was trained here.
> (An earlier session's hand-off files `cloud_poc/model_task.py` + `cloud_poc/train.py` remain in
> the repo as a reference implementation.)

---

## 1. Environment probe

| Item | Result |
|------|--------|
| `torch.__version__` | **2.12.0+cpu** |
| `torch.cuda.is_available()` | **False** |
| GPU present | **No** — `nvidia-smi` not found; CPU-only sandbox (expected, fine for this PoC) |
| `transformer_lens` | **importable**; `HookedTransformer` works (no `__version__` attr) |
| Device used for training | **cpu** (4 vCPU) |

Neither `torch` nor `transformer_lens` (nor even `numpy`) was preinstalled; all were
pip-installed during the run (see Issues for the install workaround).

## 2. Agent team (fan-out)

Two subagents were dispatched **in parallel** against a shared interface contract, so their
files compose without a follow-up integration step:

- **Infra agent** — confirmed `import torch, transformer_lens` (`2.12.0+cpu`, cuda=False), wrote
  the self-contained training driver `cloud_poc/agent_train.py` (device select, seeding, Adam,
  500-step loop, loss recording, machine-readable `LOSSCURVE` line), and `py_compile`-validated it.
- **Modeling agent** — wrote `cloud_poc/agent_model_task.py`: the tiny `HookedTransformer` wrapper
  plus the synthetic Gaussian in-context linear-regression task; ran its own forward+backward
  smoke test (initial loss ≈ 3.49, near the ~4.0 baseline) before returning.

Shared contract (kept both agents independent):
```
D = 4
build_model(device) -> torch.nn.Module
make_batch(batch_size, device) -> batch
loss_fn(model, batch) -> scalar torch.Tensor   # MSE on the query prediction
```
Both agents succeeded; the two files ran together first try.

## 3. Model config

A small `nn.Module` wraps a `transformer_lens.HookedTransformer`:

| Field | Value |
|-------|-------|
| n_layers | 2 |
| d_model | 64 |
| n_heads | 2 |
| d_head | 32 |
| d_mlp | 256 |
| n_ctx | 13 |
| act_fn | gelu |
| normalization_type | LN |
| d_vocab / d_vocab_out | 1 (unused — never tokenize/unembed) |
| Param count | **101,570** |

Because inputs are **continuous Gaussian vectors**, not token IDs, the token embedding is
bypassed: an `nn.Linear(6 → d_model)` projects each position into the residual stream, the
transformer runs via `start_at_layer=0, stop_at_layer=n_layers, return_type=None`, and an
`nn.Linear(d_model → 1)` head reads off a y-prediction per position. One forward call per batch.

## 4. Task — in-context linear regression (Gaussian)

Each sequence = `K=12` context `(x, y)` pairs + 1 query `x` (`SEQ_LEN = 13`):
- `x ~ N(0, I_D)`, `D = 4`
- per-sequence weight `w ~ N(0, I_D)` (fresh each sequence)
- `y = w·x + eps`, `eps ~ N(0, 0.1²)`
- Each position carries feature `[x (4), y (1), is_query_flag (1)]` (width 6); the query
  position has `y=0`, `flag=1`. The model predicts the query y from the context.

**Baseline:** with no information, `Var(y) ≈ D + EPS_STD² ≈ 4.01` — so any loss well below ~4 is
real in-context learning.

## 5. Loss curve (500 Adam steps, lr=1e-3, batch=256, CPU)

| Step | Train loss (MSE) |
|------|------------------|
| 1    | **5.173354** |
| 100  | **0.595919** |
| 250  | **0.355876** |
| 500  | **0.297319** |

Loss falls from ~5.17 (≈ the w-agnostic baseline of ~4.0) to **~0.30** — a clear, monotone
decrease confirming the model learns the task in-context. Full 500-step run took **~20 s** on
4 CPU cores.

## 6. Did each piece work?

| Piece | Status |
|-------|--------|
| Cloud sandbox (checkout, run, commit) | ✅ worked |
| opus-4.8 as the driver | ✅ worked |
| Agent team fan-out (2 parallel subagents) | ✅ worked — composed via shared interface, ran first try |
| Real torch training | ✅ worked (CPU, ~20 s) |
| transformer_lens HookedTransformer | ✅ worked |
| GPU | ⚠️ none in sandbox — CPU-only (out of scope for this PoC) |

## 7. Issues hit

1. **Nothing preinstalled.** No `torch`, no `transformer_lens`, not even `numpy`. All installed
   during the run.
2. **`pip install transformer_lens` fails** building the transitive dependency
   `transformers-stream-generator` (its legacy `setup.py` breaks on modern setuptools:
   `AttributeError: install_layout`). Worked around by installing
   `torch` (CPU wheel from the pytorch CPU index) + `transformer_lens --no-deps`, then its real
   runtime deps directly (`numpy einops jaxtyping transformers datasets pandas tqdm rich
   typeguard fancy-einsum protobuf sentencepiece better_abc`). `transformers-stream-generator`
   is **not** needed to import/use `HookedTransformer`.
3. **transformer_lens continuous-input API gotchas** (for feeding Gaussian vectors):
   - `model(embeds, start_at_layer=0, return_type=None)` returns **`None`** — `return_type=None`
     short-circuits the output. To get the residual stream `[batch, seq, d_model]` you must pass
     `stop_at_layer=n_layers` (stop *past* the last block).
   - `start_at_layer=0` makes the first arg the residual stream (continuous), bypassing the token
     embedding — exactly what continuous inputs need.
   - `transformer_lens` exposes no `__version__` attribute (cosmetic, logging only).
4. **No GPU** in this sandbox (`nvidia-smi` absent). Expected for the PoC; CPU was sufficient
   (full run < 30 s).

## Files

- `cloud_poc/agent_model_task.py` — model + Gaussian task (this session's modeling agent)
- `cloud_poc/agent_train.py` — 500-step training driver (this session's infra agent)
- `cloud_poc/model_task.py`, `cloud_poc/train.py` — earlier session's reference implementation
- `cloud_poc/__init__.py` — package marker

Reproduce: `python cloud_poc/agent_train.py`
