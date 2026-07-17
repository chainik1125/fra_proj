"""In-context linear regression task on a tiny HookedTransformer.

Shared interface contract (do NOT change names/signatures):
    D = 4
    build_model(device) -> torch.nn.Module
    make_batch(batch_size, device) -> batch
    loss_fn(model, batch) -> scalar torch.Tensor
"""

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn

from transformer_lens import HookedTransformer, HookedTransformerConfig

# ---------------------------------------------------------------------------
# Task / model hyperparameters
# ---------------------------------------------------------------------------
D = 4                # feature dimension of x  (module-level int, per contract)
K = 12               # number of in-context (x, y) example pairs
NOISE_STD = 0.1      # std of observation noise eps

# Each position is encoded as [x (D), y (1), is_query_flag (1)].
FEATURE_WIDTH = D + 1 + 1

# Tiny transformer config (within the required constraints).
_CFG = HookedTransformerConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    d_head=32,           # d_model / n_heads
    d_mlp=4 * 64,
    n_ctx=K + 1,         # K context pairs + 1 query position
    act_fn="gelu",
    normalization_type="LN",
    d_vocab=1,           # unused (we feed the residual stream directly)
    seed=0,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class InContextRegressor(nn.Module):
    """Projects continuous features into the residual stream, runs a tiny
    HookedTransformer on it directly (bypassing the token embedding), then
    reads out a scalar y-prediction per position."""

    def __init__(self, cfg: HookedTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer = HookedTransformer(cfg)
        self.embed = nn.Linear(FEATURE_WIDTH, cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [batch, seq, FEATURE_WIDTH]
        embeds = self.embed(feats)  # [batch, seq, d_model]
        resid = self.transformer(
            embeds,
            start_at_layer=0,                 # bypass token embedding
            stop_at_layer=self.cfg.n_layers,  # return post-final-block resid
            return_type=None,
        )  # [batch, seq, d_model]
        preds = self.unembed(resid).squeeze(-1)  # [batch, seq]
        return preds


def build_model(device) -> torch.nn.Module:
    model = InContextRegressor(_CFG)
    return model.to(device)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
def make_batch(batch_size, device):
    """Construct a batch of in-context linear-regression sequences.

    Returns a dict with:
        feats:   [batch, K+1, FEATURE_WIDTH] model input
        target:  [batch] true query y (w . x_query)
    """
    # Per-sequence weight vector w ~ N(0, I_D), fresh each sequence.
    w = torch.randn(batch_size, D, device=device)

    # Context examples: x ~ N(0, I_D), y = w . x + eps.
    x_ctx = torch.randn(batch_size, K, D, device=device)
    eps = NOISE_STD * torch.randn(batch_size, K, device=device)
    y_ctx = torch.einsum("bkd,bd->bk", x_ctx, w) + eps  # [batch, K]

    # Query example: x_q ~ N(0, I_D); target is the noiseless w . x_q.
    x_q = torch.randn(batch_size, D, device=device)
    target = torch.einsum("bd,bd->b", x_q, w)  # [batch]

    seq_len = K + 1
    feats = torch.zeros(batch_size, seq_len, FEATURE_WIDTH, device=device)
    # Context positions [0 .. K-1]: [x, y, flag=0]
    feats[:, :K, :D] = x_ctx
    feats[:, :K, D] = y_ctx
    # Query position [K]: [x_q, y=0, flag=1]
    feats[:, K, :D] = x_q
    feats[:, K, D] = 0.0
    feats[:, K, D + 1] = 1.0

    return {"feats": feats, "target": target}


def loss_fn(model, batch) -> torch.Tensor:
    preds = model(batch["feats"])      # [batch, seq]
    query_pred = preds[:, K]           # prediction at the query position
    target = batch["target"]
    return torch.mean((query_pred - target) ** 2)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cpu"
    model = build_model(device)
    batch = make_batch(32, device)
    loss = loss_fn(model, batch)
    loss.backward()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"param count: {n_params}")
    print(f"initial smoke-test loss: {loss.item():.4f}")
