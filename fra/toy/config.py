"""Configuration for the FRA toy experiment.

One frozen dataclass threaded through the DGP, the model, FRA and the metrics,
so a run is fully described by a single object.

Sizing rationale (brief section 5): with ``n_feat = 100`` and ``seq_len = 32`` the
dense QK tensor is ``32*32*100*100*4 B ~= 41 MB``. Dense is affordable; none of
the paper's Appendix E sparse machinery is needed.

``d_head`` defaults to ``d_model`` deliberately. The static coupling matrix
``G = (W_dec W_Q)(W_dec W_K)^T`` has rank at most ``d_head``, so a small head
*structurally* caps how concentrated a single planted ``(lambda*, mu*)`` cell can
be, independent of feature overlap. Keeping ``d_head >= n_feat`` removes that
confound from the rho sweep. Head rank is an interesting axis in its own right,
but it is a separate experiment.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToyConfig:
    """Full specification of a toy run."""

    # ── Geometry ────────────────────────────────────────────────────────
    d_model: int = 128
    d_head: int = 128

    # ── Feature inventory ───────────────────────────────────────────────
    # n_feat = 2 (lambda*, mu*) + n_content + n_distractor
    n_content: int = 8
    n_distractor: int = 90

    # ── Feature overlap: the experiment's knob (brief section 3.4) ──────
    # rho is the mean pairwise |cosine| between ground-truth directions.
    rho: float = 0.0
    # "common"   -> exact, uniform pairwise cosine == rho (analytic, hits rho exactly)
    # "subspace" -> random directions in a rank-`subspace_rank` subspace, giving
    #               heterogeneous overlap; realized rho is measured, not set.
    overlap_mode: str = "common"
    subspace_rank: int | None = None

    # ── Vocabulary ──────────────────────────────────────────────────────
    n_filler: int = 16
    n_query_variants: int = 4
    n_key_variants: int = 4

    # ── Token composition ───────────────────────────────────────────────
    # Distractor features carried by each token, so lambda*/mu* are never the
    # only thing present and the retrieval ranking is non-trivial.
    l0_distractor: int = 2
    act_low: float = 0.5
    act_high: float = 1.5

    # ── Sequences ───────────────────────────────────────────────────────
    seq_len: int = 32

    # ── Reproducibility ─────────────────────────────────────────────────
    seed: int = 0

    # ── Derived ─────────────────────────────────────────────────────────
    @property
    def n_feat(self) -> int:
        return 2 + self.n_content + self.n_distractor

    @property
    def d_vocab(self) -> int:
        # 1 DEFAULT + fillers + query variants + key variants + answers
        return (
            1
            + self.n_filler
            + self.n_query_variants
            + self.n_content * self.n_key_variants
            + self.n_content
        )

    def validate(self) -> None:
        """Fail fast on configurations the construction cannot express."""
        if self.overlap_mode not in ("common", "subspace"):
            raise ValueError(f"unknown overlap_mode {self.overlap_mode!r}")
        if not 0.0 <= self.rho < 1.0:
            raise ValueError(f"rho must be in [0, 1), got {self.rho}")
        if self.overlap_mode == "common":
            # Needs n_feat orthonormal directions plus one shared direction.
            if self.n_feat + 1 > self.d_model:
                raise ValueError(
                    f"overlap_mode='common' needs n_feat + 1 <= d_model; "
                    f"got n_feat={self.n_feat}, d_model={self.d_model}"
                )
        if self.overlap_mode == "subspace" and self.subspace_rank is None:
            raise ValueError("overlap_mode='subspace' requires subspace_rank")
        if self.l0_distractor > self.n_distractor:
            raise ValueError("l0_distractor exceeds n_distractor")
        if self.seq_len < 3:
            raise ValueError("seq_len must leave room for a key before a query")
