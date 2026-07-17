"""Tiny transformer_lens model + synthetic in-context linear regression task.

Interface (do not change names/signatures; consumed by the infra training loop):
    D = 4
    build_model(device) -> torch.nn.Module
    make_batch(batch_size, device) -> batch object
    loss_fn(model, batch) -> scalar torch.Tensor (MSE on query prediction)

Task: in-context linear regression.
  Each sequence: K context (x, y) pairs followed by one query x.
  x ~ N(0, I_D),  per-sequence w ~ N(0, I_D),  y = w . x + eps,  eps ~ N(0, 0.1^2).
  The model must predict the query y given the context.

Sequence encoding: each position carries a feature vector
  [ x (D dims), y (1 dim), is_query_flag (1 dim) ]   -> width D + 2.
Positions are interleaved as a flat sequence of length K + 1 (K context positions,
then 1 query position).  For context positions y is the true y and flag = 0; for the
query position y = 0 (masked) and flag = 1.  A Linear projects this feature vector to
d_model, the HookedTransformer is run on the residual stream directly via start_at_layer,
and a Linear head maps d_model -> 1 to produce a y prediction at every position.

Only ONE forward call per batch is used.
"""

import torch
import torch.nn as nn

from transformer_lens import HookedTransformer, HookedTransformerConfig


# ---- task / model dimensions -------------------------------------------------
D = 4                 # input dimension of x
K = 12                # number of in-context (x, y) pairs
SEQ_LEN = K + 1       # context positions + 1 query position
FEATURE_WIDTH = D + 2  # [x (D), y (1), is_query_flag (1)]
EPS_STD = 0.1
N_CTX = 64            # transformer context length (>= SEQ_LEN)


class ICLRegressor(nn.Module):
    """Wraps a HookedTransformer for continuous-input in-context regression.

    Bypasses the token embedding: a Linear projects each position's feature vector
    into the residual stream, the transformer is run via start_at_layer=0, and a
    Linear head reads off a scalar y prediction per position.
    """

    def __init__(self):
        super().__init__()
        cfg = HookedTransformerConfig(
            n_layers=2,
            d_model=64,
            n_heads=2,
            d_head=32,
            n_ctx=N_CTX,
            act_fn="gelu",
            normalization_type="LN",
            d_vocab=1,          # unused (we never tokenize / unembed), kept minimal
            d_vocab_out=1,
        )
        self.cfg = cfg
        self.transformer = HookedTransformer(cfg)
        self.embed_proj = nn.Linear(FEATURE_WIDTH, cfg.d_model)
        self.read_head = nn.Linear(cfg.d_model, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: [batch, seq, FEATURE_WIDTH] -> preds: [batch, seq] (y per position)."""
        embeds = self.embed_proj(features)  # [batch, seq, d_model]
        # start_at_layer=0 skips the token embedding and treats `embeds` as the
        # residual stream; stop_at_layer=n_layers returns the resid stream
        # ([batch, seq, d_model]) after the final block (no unembed).
        resid = self.transformer(
            embeds,
            start_at_layer=0,
            stop_at_layer=self.cfg.n_layers,
            return_type=None,
        )
        preds = self.read_head(resid).squeeze(-1)  # [batch, seq]
        return preds


def build_model(device) -> torch.nn.Module:
    model = ICLRegressor().to(device)
    return model


def make_batch(batch_size: int, device):
    """Returns dict with:
        features: [batch, SEQ_LEN, FEATURE_WIDTH]  model input
        target_y: [batch, SEQ_LEN]                 true y at every position (incl. query)
        query_idx: int                             index of the query position (= K)
    """
    x = torch.randn(batch_size, SEQ_LEN, D, device=device)        # all positions' x
    w = torch.randn(batch_size, D, device=device)                 # per-sequence weight
    clean_y = torch.einsum("bsd,bd->bs", x, w)                    # w . x at each position
    eps = EPS_STD * torch.randn(batch_size, SEQ_LEN, device=device)
    target_y = clean_y + eps                                      # [batch, SEQ_LEN]

    query_idx = K  # last position is the query

    # Build feature vectors. Context positions (0..K-1) carry their true y and flag=0.
    # Query position (K) carries y=0 and flag=1.
    feats = torch.zeros(batch_size, SEQ_LEN, FEATURE_WIDTH, device=device)
    feats[:, :, :D] = x
    # y channel (index D): true y for context positions, 0 for the query
    y_channel = target_y.clone()
    y_channel[:, query_idx] = 0.0
    feats[:, :, D] = y_channel
    # is_query flag (index D+1)
    feats[:, query_idx, D + 1] = 1.0

    return {
        "features": feats,
        "target_y": target_y,
        "query_idx": query_idx,
    }


def loss_fn(model, batch) -> torch.Tensor:
    """Scalar MSE between predicted and true query y.

    Headline loss is the MSE on the query position. A small auxiliary term on the
    context positions (predicting their own y from x, which the model also sees) is
    omitted to keep the headline metric clean and clearly below the w=0 baseline
    (baseline = Var(y) ~= D + EPS_STD^2 ~= 4.01).
    """
    preds = model(batch["features"])              # [batch, seq]
    q = batch["query_idx"]
    query_pred = preds[:, q]                       # [batch]
    query_true = batch["target_y"][:, q]           # [batch]
    loss = torch.mean((query_pred - query_true) ** 2)
    return loss
