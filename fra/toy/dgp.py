"""Feature-matched retrieval: a DGP with ground-truth features (brief section 3.3).

The planted rule
----------------
Each sequence contains exactly one **key position** ``k`` where the key feature
``mu*`` fires, and exactly one later **query position** ``q > k`` where the query
feature ``lambda*`` fires. The key token also carries one **content feature**
``nu_c`` drawn from a designated set. The label at ``q`` is the answer token for
``c``; everywhere else the label is a single DEFAULT token.

The only way to get ``q`` right is the skip-trigram *"if lambda* here, attend to
the position where mu* fires, and copy what is there"* -- a planted QK edge at
``(lambda*, mu*)`` and a planted OV set ``{nu_c}``.

Why the construction is shaped this way
---------------------------------------
* **Multiple query/key token variants.** ``lambda*`` and ``mu*`` each appear in
  several distinct tokens, mixed with different distractors, so the model must
  key on the *feature* rather than memorise a token id. Without this the
  "feature" is just an alias for a vocabulary entry and the experiment is vacuous.

* **Content features, plural.** With a single OV feature the model can shortcut:
  detect ``mu*``, emit a constant, never attend. Content has to *vary* for
  copying to be required, so the OV target is a set and the OV metric is
  retrieval precision over that set.

* **Answer tokens are distinct from key tokens, and carry ``nu_c`` without
  ``mu*``.** This decouples the content *value* from the token identity that
  carried it, and it seeds the sequence with positions that hold a content
  feature but are *not* the retrieval target -- so the model cannot find the
  answer by looking for ``nu_c``, it must find where ``mu*`` fires.

Exactness contract
------------------
The residual stream entering the attention block must be *exactly* expressible
in the ground-truth feature basis::

    resid_pre[t] == f[t] @ feature_directions

This holds **by construction** here: the embedding matrix is frozen at
``W_E[v] = sum_lambda M[v, lambda] * direction[lambda]`` and ``f[t] = M[token_t]``.
It is what makes oracle FRA exact, and it is why ``W_E`` must never train --
a trained ``W_E`` drifts off the planted directions and the ground truth becomes
fiction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from fra.toy.config import ToyConfig

# Fixed feature-index layout.
IDX_QUERY = 0  # lambda*
IDX_KEY = 1  # mu*
IDX_CONTENT_START = 2  # nu_0 .. nu_{C-1}

# Fixed token-index layout.
TOK_DEFAULT = 0


@dataclass
class ToyBatch:
    """One batch, with the ground truth attached."""

    tokens: Tensor  # [batch, seq]              int64, model input
    features: Tensor  # [batch, seq, n_feat]    float, GROUND TRUTH activations
    targets: Tensor  # [batch, seq]             int64, label predicted FROM position t
    query_pos: Tensor  # [batch]                where lambda* fires
    key_pos: Tensor  # [batch]                  where mu* fires
    content: Tensor  # [batch]                  which nu_c the key position carried


class ToyDGP(Protocol):
    """Everything downstream depends only on this."""

    n_feat: int
    d_model: int
    feature_directions: Tensor  # [n_feat, d_model]  GROUND TRUTH decoder
    planted_qk_edge: tuple[int, int]  # (lambda*, mu*) -- the answer
    planted_ov_features: set[int]  # content features carrying copied content

    def sample(self, batch: int, seq: int) -> ToyBatch: ...


# ── Feature geometry: the rho knob (brief section 3.4) ───────────────────


def make_directions(cfg: ToyConfig, generator: torch.Generator) -> Tensor:
    """Ground-truth feature directions with controlled pairwise overlap.

    ``overlap_mode="common"`` gives every pair of features *exactly* cosine
    ``rho``: take orthonormal ``o_i`` plus a unit ``c`` orthogonal to all of
    them, and set ``v_i = sqrt(1 - rho) * o_i + sqrt(rho) * c``. Then
    ``|v_i| = 1`` and ``<v_i, v_j> = rho`` for ``i != j``, so the mean pairwise
    |cosine| is ``rho`` analytically -- a knob that hits its target rather than
    approaching it. ``rho = 0`` degenerates to an orthonormal basis.

    ``overlap_mode="subspace"`` instead confines random directions to a
    rank-``subspace_rank`` subspace. Overlap is then *heterogeneous* across
    pairs rather than uniform, and the realized rho is measured, not set. It
    exists so the sweep can be checked against a second geometry family: a
    result that survives both is about overlap, not about one construction.

    Returns:
        ``[n_feat, d_model]`` unit-norm rows.
    """
    cfg.validate()
    n_feat, d_model = cfg.n_feat, cfg.d_model

    if cfg.overlap_mode == "common":
        # Orthonormal basis of size n_feat + 1 via QR, last column is the shared mode.
        gaussian = torch.randn(d_model, n_feat + 1, generator=generator, dtype=torch.float64)
        q, _ = torch.linalg.qr(gaussian)  # [d_model, n_feat + 1], orthonormal columns
        ortho = q[:, :n_feat].T  # [n_feat, d_model]
        shared = q[:, n_feat]  # [d_model]
        directions = (1.0 - cfg.rho) ** 0.5 * ortho + cfg.rho**0.5 * shared[None, :]
    else:  # "subspace"
        rank = int(cfg.subspace_rank)
        gaussian = torch.randn(d_model, rank, generator=generator, dtype=torch.float64)
        basis, _ = torch.linalg.qr(gaussian)  # [d_model, rank]
        coeffs = torch.randn(n_feat, rank, generator=generator, dtype=torch.float64)
        directions = coeffs @ basis.T

    directions = directions / directions.norm(dim=-1, keepdim=True)
    return directions.to(torch.float32)


def realized_rho(directions: Tensor) -> float:
    """Mean pairwise |cosine| actually achieved -- always report this, never assume."""
    unit = directions / directions.norm(dim=-1, keepdim=True)
    cos = unit @ unit.T
    n = cos.shape[0]
    off_diagonal = ~torch.eye(n, dtype=torch.bool, device=cos.device)
    return cos[off_diagonal].abs().mean().item()


# ── The generator ────────────────────────────────────────────────────────


class FeatureMatchedRetrieval:
    """Default DGP. Implements :class:`ToyDGP`."""

    def __init__(self, cfg: ToyConfig) -> None:
        cfg.validate()
        self.cfg = cfg
        self.n_feat = cfg.n_feat
        self.d_model = cfg.d_model
        self.d_vocab = cfg.d_vocab

        gen = torch.Generator().manual_seed(cfg.seed)
        self._gen = gen

        self.feature_directions = make_directions(cfg, gen)
        self.rho_realized = realized_rho(self.feature_directions)

        # The planted answer.
        self.planted_qk_edge = (IDX_QUERY, IDX_KEY)
        self.content_features = list(
            range(IDX_CONTENT_START, IDX_CONTENT_START + cfg.n_content)
        )
        self.planted_ov_features = set(self.content_features)
        self.distractor_features = list(
            range(IDX_CONTENT_START + cfg.n_content, cfg.n_feat)
        )

        self.token_features = self._build_vocabulary(gen)  # [d_vocab, n_feat]

        # Token-id ranges, in vocabulary order.
        v = 1
        self.filler_tokens = list(range(v, v + cfg.n_filler))
        v += cfg.n_filler
        self.query_tokens = list(range(v, v + cfg.n_query_variants))
        v += cfg.n_query_variants
        self.key_tokens = [
            list(range(v + c * cfg.n_key_variants, v + (c + 1) * cfg.n_key_variants))
            for c in range(cfg.n_content)
        ]
        v += cfg.n_content * cfg.n_key_variants
        self.answer_tokens = list(range(v, v + cfg.n_content))

        # Tokens legal as background filler: carry neither lambda* nor mu*.
        self.background_tokens = torch.tensor(
            [TOK_DEFAULT] + self.filler_tokens + self.answer_tokens, dtype=torch.long
        )

    # ── Vocabulary ──────────────────────────────────────────────────────

    def _sample_activations(self, n: int, gen: torch.Generator) -> Tensor:
        """Positive activations, uniform in [act_low, act_high]."""
        cfg = self.cfg
        u = torch.rand(n, generator=gen)
        return cfg.act_low + u * (cfg.act_high - cfg.act_low)

    def _add_distractors(self, row: Tensor, gen: torch.Generator) -> None:
        cfg = self.cfg
        pool = torch.tensor(self.distractor_features, dtype=torch.long)
        chosen = pool[torch.randperm(len(pool), generator=gen)[: cfg.l0_distractor]]
        row[chosen] = self._sample_activations(len(chosen), gen)

    def _build_vocabulary(self, gen: torch.Generator) -> Tensor:
        """``M[v, lambda]`` -- the activation of feature lambda in token v.

        This matrix *is* the ground truth: ``f[t] = M[token_t]`` exactly, and the
        frozen embedding is ``W_E = M @ feature_directions``.
        """
        cfg = self.cfg
        m = torch.zeros(cfg.d_vocab, cfg.n_feat)

        v = 0
        # DEFAULT: distractors only.
        self._add_distractors(m[v], gen)
        v += 1

        # Fillers: distractors only.
        for _ in range(cfg.n_filler):
            self._add_distractors(m[v], gen)
            v += 1

        # Query variants: lambda* plus distractors.
        for _ in range(cfg.n_query_variants):
            m[v, IDX_QUERY] = self._sample_activations(1, gen).item()
            self._add_distractors(m[v], gen)
            v += 1

        # Key variants: mu* plus one content feature plus distractors.
        for c in range(cfg.n_content):
            for _ in range(cfg.n_key_variants):
                m[v, IDX_KEY] = self._sample_activations(1, gen).item()
                m[v, self.content_features[c]] = self._sample_activations(1, gen).item()
                self._add_distractors(m[v], gen)
                v += 1

        # Answer tokens: content feature only, no mu* -- a decoy for the retrieval.
        for c in range(cfg.n_content):
            m[v, self.content_features[c]] = self._sample_activations(1, gen).item()
            v += 1

        assert v == cfg.d_vocab, (v, cfg.d_vocab)
        return m

    # ── The frozen embedding ────────────────────────────────────────────

    @property
    def embedding_matrix(self) -> Tensor:
        """``W_E = M @ feature_directions`` -- makes ``resid_pre == f @ W_dec`` exact."""
        return self.token_features @ self.feature_directions

    # ── Sampling ────────────────────────────────────────────────────────

    def sample(self, batch: int, seq: int | None = None) -> ToyBatch:
        cfg = self.cfg
        seq = cfg.seq_len if seq is None else seq
        if seq < 3:
            raise ValueError("seq_len must leave room for a key before a query")
        gen = self._gen

        # Background: nothing here carries lambda* or mu*.
        pick = torch.randint(
            len(self.background_tokens), (batch, seq), generator=gen
        )
        tokens = self.background_tokens[pick]

        # Query position q in [1, seq-1]; key position k in [0, q-1].
        query_pos = torch.randint(1, seq, (batch,), generator=gen)
        key_pos = (torch.rand(batch, generator=gen) * query_pos).floor().long()

        # Content value, and which key-token variant expresses it.
        content = torch.randint(cfg.n_content, (batch,), generator=gen)
        variant = torch.randint(cfg.n_key_variants, (batch,), generator=gen)
        key_table = torch.tensor(self.key_tokens, dtype=torch.long)  # [C, n_key_variants]
        key_tok = key_table[content, variant]

        q_variant = torch.randint(cfg.n_query_variants, (batch,), generator=gen)
        query_tok = torch.tensor(self.query_tokens, dtype=torch.long)[q_variant]

        rows = torch.arange(batch)
        tokens[rows, key_pos] = key_tok
        tokens[rows, query_pos] = query_tok

        # Labels: DEFAULT everywhere, the planted answer at the query position.
        targets = torch.full((batch, seq), TOK_DEFAULT, dtype=torch.long)
        targets[rows, query_pos] = torch.tensor(
            self.answer_tokens, dtype=torch.long
        )[content]

        features = self.token_features[tokens]  # [batch, seq, n_feat] GROUND TRUTH

        return ToyBatch(
            tokens=tokens,
            features=features,
            targets=targets,
            query_pos=query_pos,
            key_pos=key_pos,
            content=content,
        )

    # ── Oracle solver: proves the task is solvable in principle ─────────

    def oracle_predict(self, b: ToyBatch) -> Tensor:
        """Labels a perfect reader of the ground truth would emit.

        If this is not 100% accurate the task is malformed, and no amount of
        training or FRA debugging will tell you that.
        """
        batch, seq = b.tokens.shape
        out = torch.full((batch, seq), TOK_DEFAULT, dtype=torch.long)
        answers = torch.tensor(self.answer_tokens, dtype=torch.long)
        for i in range(batch):
            q = int(b.query_pos[i])
            # Find the unique earlier position where mu* fires.
            keys = (b.features[i, :q, IDX_KEY] > 0).nonzero().flatten()
            assert len(keys) == 1, f"mu* not unique before query: {keys.tolist()}"
            k = int(keys[0])
            c = int(b.features[i, k, IDX_CONTENT_START:IDX_CONTENT_START + self.cfg.n_content].argmax())
            out[i, q] = answers[c]
        return out
