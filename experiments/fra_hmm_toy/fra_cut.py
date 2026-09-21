"""Exact FRA decomposition + interventions for the toy mixture-HMM model.

Interface: SAE lives at blocks.{L_sae}.hook_resid_post, which is the resid_pre
of the attention layer L_att = L_sae + 1 that all FRA cuts act on.

With a TopK(k=4) SAE the decomposition of the resid feeding attention is exact:
    x_p = sum_{i in A_p} z_i d_i + b_dec + err_p            (A_p = 4 active latents)
LN1 with the forward's own scale sigma_p is, per position, the LINEAR map
    phi_p(u) = ((u - mean(u)) / sigma_p) * w_ln
so attention scores decompose exactly into term-pair contributions
    score[h,q,k] = sum_{tq,tk} (phi_q(tq) W_Q[h]) . (phi_k(tk) W_K[h]) / scale
with per-side terms {active features, b_dec, err, const(b_ln W + b)}.

Cuts (all subtract c times the EXACT contribution of the selected terms):
  - fra_qk (side = key / query / either / pair): subtract S-involving score
    contributions at blocks.{L_att}.attn.hook_attn_scores
  - fra_ov: subtract S-features' value transport at blocks.{L_att}.hook_attn_out
  - sae_cut: x -> x - alpha * sum_{i in S} z_i d_i at the SAE hookpoint
  - proj_cut: project out probe directions at the SAE hookpoint (DoM-style)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class CleanContext:
    """Everything the interventions need, cached from one clean forward."""

    tokens: torch.Tensor      # (B, T)
    x: torch.Tensor           # (B, T, D) resid feeding L_att (= SAE hookpoint)
    sigma: torch.Tensor       # (B, T, 1) ln1 scale at L_att
    z: torch.Tensor           # (B, T, d_sae) SAE latents of x
    err: torch.Tensor         # (B, T, D) SAE error term
    pattern: torch.Tensor     # (B, H, T, T) clean attention pattern at L_att
    scores: torch.Tensor      # (B, H, T, T) clean attention scores at L_att
    q: torch.Tensor           # (B, T, H, dh) clean queries at L_att
    k: torch.Tensor           # (B, T, H, dh) clean keys at L_att


class FRAToolkit:
    def __init__(self, model: Any, sae: Any, l_sae: int):
        self.model = model
        self.sae = sae
        self.l_sae = l_sae
        self.l_att = l_sae + 1
        cfg = model.cfg
        assert self.l_att < cfg.n_layers
        self.hook_resid = f"blocks.{l_sae}.hook_resid_post"
        self.hook_scores = f"blocks.{self.l_att}.attn.hook_attn_scores"
        self.hook_pattern = f"blocks.{self.l_att}.attn.hook_pattern"
        self.hook_attn_out = f"blocks.{self.l_att}.hook_attn_out"
        self.hook_q = f"blocks.{self.l_att}.attn.hook_q"
        self.hook_k = f"blocks.{self.l_att}.attn.hook_k"
        self.hook_ln_scale = f"blocks.{self.l_att}.ln1.hook_scale"

        blk = model.blocks[self.l_att]
        self.ln_w = blk.ln1.w.detach()
        self.ln_b = blk.ln1.b.detach()
        self.W_Q = blk.attn.W_Q.detach()  # (H, D, dh)
        self.b_Q = blk.attn.b_Q.detach()  # (H, dh)
        self.W_K = blk.attn.W_K.detach()
        self.b_K = blk.attn.b_K.detach()
        self.W_V = blk.attn.W_V.detach()
        self.W_O = blk.attn.W_O.detach()  # (H, dh, D)
        self.attn_scale = blk.attn.attn_scale

    # ── clean pass ──

    @torch.no_grad()
    def clean_context(self, tokens: torch.Tensor) -> CleanContext:
        names = {
            self.hook_resid, self.hook_scores, self.hook_pattern,
            self.hook_q, self.hook_k, self.hook_ln_scale,
        }
        _, cache = self.model.run_with_cache(
            tokens, return_type=None,
            names_filter=lambda n: n in names,
            return_cache_object=False,
        )
        x = cache[self.hook_resid]
        z = self.sae.encode(x)
        x_hat = self.sae.decode(z)
        return CleanContext(
            tokens=tokens, x=x,
            sigma=cache[self.hook_ln_scale],
            z=z, err=x - x_hat,
            pattern=cache[self.hook_pattern],
            scores=cache[self.hook_scores],
            q=cache[self.hook_q], k=cache[self.hook_k],
        )

    # ── decomposition primitives ──

    def phi(self, u: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Frozen-LN per-position linear map. u: (B,T,D), sigma: (B,T,1)."""
        return (u - u.mean(-1, keepdim=True)) / sigma * self.ln_w

    def feat_sum(self, ctx: CleanContext, S: torch.Tensor) -> torch.Tensor:
        """(B,T,D) sum_{i in S} z_i d_i (only active latents contribute)."""
        return ctx.z[..., S] @ self.sae.W_dec[S, :]

    def _qk_of(self, u_norm: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """(B,T,D) x (H,D,dh) -> (B,T,H,dh), no bias."""
        return torch.einsum("btd,hdf->bthf", u_norm, W)

    def _score(self, qv: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """(B,T,H,dh) x (B,T,H,dh) -> (B,H,Tq,Tk) / attn_scale."""
        return torch.einsum("bqhf,bkhf->bhqk", qv, kv) / self.attn_scale

    @torch.no_grad()
    def qk_delta(self, ctx: CleanContext, S: torch.Tensor, side: str) -> torch.Tensor:
        """Exact score contribution of pairs involving S-features on `side`.

        side: 'key' | 'query' | 'either' | 'pair' (S-query x S-key only).
        Returns (B,H,T,T), zeroed above the diagonal.
        """
        fS = self.feat_sum(ctx, S)
        phiS = self.phi(fS, ctx.sigma)
        qS = self._qk_of(phiS, self.W_Q)
        kS = self._qk_of(phiS, self.W_K)
        if side == "key":
            delta = self._score(ctx.q, kS)
        elif side == "query":
            delta = self._score(qS, ctx.k)
        elif side == "pair":
            delta = self._score(qS, kS)
        elif side == "either":
            delta = self._score(ctx.q, kS) + self._score(qS, ctx.k) - self._score(qS, kS)
        else:
            raise ValueError(side)
        T = delta.shape[-1]
        tril = torch.tril(torch.ones(T, T, dtype=torch.bool, device=delta.device))
        return delta * tril

    @torch.no_grad()
    def ov_delta(self, ctx: CleanContext, S: torch.Tensor, pattern: torch.Tensor | None = None) -> torch.Tensor:
        """Exact attn-out contribution of S-features' values. (B,T,D).

        pattern defaults to the clean pattern; pass the QK-cut run's pattern
        when composing QK+OV cuts.
        """
        if pattern is None:
            pattern = ctx.pattern
        fS = self.feat_sum(ctx, S)
        phiS = self.phi(fS, ctx.sigma)
        vS = torch.einsum("btd,hdf->bthf", phiS, self.W_V)      # (B,T,H,dh)
        mixed = torch.einsum("bhqk,bkhf->bqhf", pattern, vS)     # (B,T,H,dh)
        return torch.einsum("bqhf,hfd->bqd", mixed, self.W_O)

    # ── verification ──

    @torch.no_grad()
    def verify_decomposition(self, ctx: CleanContext, atol: float = 2e-3) -> dict[str, float]:
        """Check that {all-latent feats, b_dec, err, const} pairs reproduce scores."""
        B, T, D = ctx.x.shape
        all_S = torch.arange(self.sae.d_sae)
        terms_q, terms_k = [], []
        for u in (
            self.feat_sum(ctx, all_S),
            self.sae.b_dec.expand(B, T, D),
            ctx.err,
        ):
            ph = self.phi(u, ctx.sigma)
            terms_q.append(self._qk_of(ph, self.W_Q))
            terms_k.append(self._qk_of(ph, self.W_K))
        const_q = (self.ln_b @ self.W_Q + self.b_Q).expand(B, T, -1, -1)
        const_k = (self.ln_b @ self.W_K + self.b_K).expand(B, T, -1, -1)
        terms_q.append(const_q)
        terms_k.append(const_k)
        recon = sum(self._score(tq, tk) for tq in terms_q for tk in terms_k)
        T_ = recon.shape[-1]
        tril = torch.tril(torch.ones(T_, T_, dtype=torch.bool, device=recon.device))
        diff = torch.where(tril, (recon - ctx.scores).abs(), torch.zeros_like(recon))
        return {"max_abs_err": float(diff.max()), "ok": float(diff.max() < atol)}

    @torch.no_grad()
    def qk_mass_by_type(self, ctx: CleanContext) -> dict[str, Any]:
        """Per-head |score| mass (and pattern-weighted mass) by term-type pair.

        Types per side: FEAT (all active latents), BDEC, ERR, CONST.
        Answers: is the attention pattern content-gated at all?
        """
        B, T, D = ctx.x.shape
        all_S = torch.arange(self.sae.d_sae)
        types = ["feat", "bdec", "err", "const"]
        us = {
            "feat": self.feat_sum(ctx, all_S),
            "bdec": self.sae.b_dec.expand(B, T, D),
            "err": ctx.err,
        }
        tq, tk = {}, {}
        for name, u in us.items():
            ph = self.phi(u, ctx.sigma)
            tq[name] = self._qk_of(ph, self.W_Q)
            tk[name] = self._qk_of(ph, self.W_K)
        tq["const"] = (self.ln_b @ self.W_Q + self.b_Q).expand(B, T, -1, -1)
        tk["const"] = (self.ln_b @ self.W_K + self.b_K).expand(B, T, -1, -1)

        tril = torch.tril(torch.ones(T, T, dtype=torch.bool, device=ctx.x.device))
        n_heads = ctx.pattern.shape[1]
        raw = torch.zeros(len(types), len(types), n_heads)
        patw = torch.zeros(len(types), len(types), n_heads)
        for a, ta in enumerate(types):
            for b, tb in enumerate(types):
                contrib = self._score(tq[ta], tk[tb]) * tril     # (B,H,T,T)
                raw[a, b] = contrib.abs().sum(dim=(0, 2, 3))
                patw[a, b] = (contrib.abs() * ctx.pattern).sum(dim=(0, 2, 3))
        return {
            "types": types,
            "raw": raw / raw.sum(dim=(0, 1), keepdim=True).clamp(min=1e-12),
            "pattern_weighted": patw / patw.sum(dim=(0, 1), keepdim=True).clamp(min=1e-12),
        }


# ── intervention specs -> hooks ──


@dataclass
class Intervention:
    """kind: none | sae_cut | proj_cut | fra_qk | fra_ov | fra_qk_ov
    S: latent indices (sae_cut / fra_*); dirs: (D, r) orthonormal (proj_cut);
    side: fra_qk side; strength: multiplier on the subtracted contribution."""

    name: str
    kind: str
    S: torch.Tensor | None = None
    dirs: torch.Tensor | None = None
    side: str = "key"
    strength: float = 1.0

    def build_hooks(self, tk: FRAToolkit, ctx: CleanContext) -> list[tuple[str, Any]]:
        c = self.strength
        if self.kind == "none":
            return []
        if self.kind == "sae_cut":
            fS = tk.feat_sum(ctx, self.S)

            def hook_sae(act, hook):
                return act - c * fS.to(act.dtype)

            return [(tk.hook_resid, hook_sae)]
        if self.kind == "proj_cut":
            P = self.dirs @ self.dirs.T  # (D, D)
            mu = ctx.x.mean(dim=(0, 1))

            def hook_proj(act, hook):
                return act - c * ((act - mu) @ P)

            return [(tk.hook_resid, hook_proj)]
        if self.kind == "fra_qk":
            delta = tk.qk_delta(ctx, self.S, self.side)

            def hook_qk(scores, hook):
                return scores - c * delta.to(scores.dtype)

            return [(tk.hook_scores, hook_qk)]
        if self.kind == "fra_ov":
            dout = tk.ov_delta(ctx, self.S)

            def hook_ov(act, hook):
                return act - c * dout.to(act.dtype)

            return [(tk.hook_attn_out, hook_ov)]
        if self.kind == "fra_qk_ov":
            delta = tk.qk_delta(ctx, self.S, self.side)
            # pattern under the QK cut, for a consistent OV delta
            masked = ctx.scores - c * delta
            patched_pattern = torch.softmax(masked, dim=-1)
            dout = tk.ov_delta(ctx, self.S, pattern=patched_pattern)

            def hook_qk2(scores, hook):
                return scores - c * delta.to(scores.dtype)

            def hook_ov2(act, hook):
                return act - c * dout.to(act.dtype)

            return [(tk.hook_scores, hook_qk2), (tk.hook_attn_out, hook_ov2)]
        raise ValueError(self.kind)
