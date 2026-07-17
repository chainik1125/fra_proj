"""Toy latent-cause model of emergent misalignment: {A, M, L_F} selection.

Analytic backbone for the finetuning experiments (finetuning_latent_cause_design.md).
The hidden hypothesis is a PERSISTENT latent

    H in {A, M, L_F1, ..., L_Fk}

  A      = globally aligned        (good in every domain)
  M      = global reckless persona (bad in every domain)        -> broad EM
  L_Fi   = local patch in domain i (bad only in domain i)       -> narrow only

Observations are (domain d, valence v in {+,-}). The latent never changes, so
Bayesian filtering is exact: multiply per-observation likelihoods and renormalize.
This is the analytic version of the question "when does narrow finetuning select the
broad persona M over a local patch L_F?":

    P(M|C)/P(L_F|C) = [P(M)/P(L_F)] * [P(C|M)/P(C|L_F)]      (prior x likelihood)

Key qualitative facts this script demonstrates:
  1. Reproduces the proposal's 2-persona {A,M} filtering numbers.
  2. With IN-DOMAIN bad examples only, M and L_F are likelihood-matched, so the
     posterior ratio equals the PRIOR ratio no matter how many bad-F examples you add
     -> in-domain evidence alone cannot select broad over local.
  3. OFF-DOMAIN evidence (or any cue that separates the likelihoods) is what selects M;
     a single bad-O example kills every L_{F!=O} and collapses the posterior onto M.
  4. Predicted broad EM P(-|O) is monotone in the posterior weight on M.

Run:
    uv run python training/latent_cause.py
"""
from __future__ import annotations

import numpy as np

PLUS, MINUS = 0, 1  # valence indices


class LatentCauseModel:
    """Persistent-latent HMM over {A, M, L_F1..k} with (domain, valence) observations."""

    def __init__(self, k_domains: int, eps: float = 0.01,
                 prior: dict | None = None, domain_probs=None):
        """
        Args:
            k_domains: number of domains (domain 0 is the "demonstrated"/finetuned F).
            eps: slip probability — P(misaligned | a domain that the hypothesis is
                 aligned in), and symmetrically P(aligned | a bad domain) = eps.
            prior: dict over hypothesis names -> prior mass (unnormalized ok). Defaults
                   to a mildly M-favoring prior (the optimizer default; see
                   "Narrow is Hard", 2602.07852).
            domain_probs: P(domain) for sampling; defaults to uniform. Does not affect
                   the posterior over H for a fixed observed-domain sequence.
        """
        self.k = k_domains
        self.eps = eps
        self.hyps = ["A", "M"] + [f"L_F{i}" for i in range(k_domains)]
        self.domain_probs = (np.full(k_domains, 1.0 / k_domains)
                             if domain_probs is None else np.asarray(domain_probs))
        if prior is None:
            # mildly favor the broad persona over any single local patch
            prior = {"A": 1.0, "M": 1.0}
            prior.update({f"L_F{i}": 1.0 / k_domains for i in range(k_domains)})
        self.prior = np.array([prior[h] for h in self.hyps], dtype=float)
        self.prior /= self.prior.sum()

    # ----- per-hypothesis badness q_h(d) = P(valence=- | domain d, H=h) -----
    def q(self, h: str, d: int) -> float:
        if h == "A":
            return self.eps
        if h == "M":
            return 1.0 - self.eps
        # local patch L_Fi: bad only in domain i
        i = int(h.split("L_F")[1])
        return (1.0 - self.eps) if d == i else self.eps

    def obs_likelihood(self, d: int, v: int) -> np.ndarray:
        """P(observe (d, v) | H=h) for every h, up to the shared P(domain=d) factor."""
        qs = np.array([self.q(h, d) for h in self.hyps])
        pv = qs if v == MINUS else (1.0 - qs)
        return self.domain_probs[d] * pv

    # ----- filtering -----
    def filter(self, context) -> np.ndarray:
        """Posterior over H after a context = list of (domain, valence)."""
        b = self.prior.copy()
        for (d, v) in context:
            b = b * self.obs_likelihood(d, v)
            b /= b.sum()
        return b

    def posterior(self, context) -> dict:
        return dict(zip(self.hyps, self.filter(context)))

    def p_bad(self, b: np.ndarray, d: int) -> float:
        """Predicted P(misaligned | domain d) under posterior b = predicted EM in d."""
        qs = np.array([self.q(h, d) for h in self.hyps])
        return float(b @ qs)

    def odds_M_over_LF(self, context, i: int = 0) -> float:
        b = self.posterior(context)
        return b["M"] / b[f"L_F{i}"]


def _fmt(b: dict) -> str:
    return "  ".join(f"{h}={p:.3f}" for h, p in b.items())


def demo():
    np.set_printoptions(precision=4, suppress=True)

    # --- 1) sanity: reproduce the proposal's 2-persona {A,M} filtering ---
    # single domain, prior [A=0.99, M=0.01], two bad in-domain observations.
    print("## 1) Proposal 2-persona sanity ({A,M}, 1 domain)")
    m2 = LatentCauseModel(k_domains=1, eps=0.01, prior={"A": 0.99, "M": 0.01, "L_F0": 0.0})
    for n in (0, 1, 2):
        b = m2.posterior([(0, MINUS)] * n)
        print(f"  after {n} bad-F: A={b['A']:.3f} M={b['M']:.3f}  P(-|O=0)={m2.p_bad(m2.filter([(0,MINUS)]*n),0):.4f}")
    print("  (expect after 2: M~0.99, P(-)~0.98 — matches the note)\n")

    # --- 2) {A, M, L_F}: in-domain bad examples cannot select M over L_F ---
    print("## 2) In-domain evidence is likelihood-matched (k=3 domains; F = domain 0)")
    m = LatentCauseModel(k_domains=3, eps=0.01)
    print(f"  prior odds  M : L_F0 = {m.prior[m.hyps.index('M')]/m.prior[m.hyps.index('L_F0')]:.3f}")
    for n in (1, 2, 4, 8):
        ctx = [(0, MINUS)] * n
        print(f"  after {n} bad-F: {_fmt(m.posterior(ctx))}  | odds M/L_F0={m.odds_M_over_LF(ctx):.3f}"
              f"  P(-|O=1)={m.p_bad(m.filter(ctx),1):.3f}")
    print("  => odds M/L_F0 stay = prior odds; broad EM P(-|O) flat. In-domain badness")
    print("     alone does NOT prefer the global persona.\n")

    # --- 3) off-domain / reattribution evidence selects M ---
    print("## 3) One off-domain bad example collapses onto M (kills local patches)")
    base = [(0, MINUS)] * 4
    for label, ctx in [
        ("4 bad-F only", base),
        ("4 bad-F + 1 bad-O(d=1)", base + [(1, MINUS)]),
        ("4 bad-F + 1 GOOD-O(d=1)", base + [(1, PLUS)]),
    ]:
        b = m.posterior(ctx)
        print(f"  {label:24s}: M={b['M']:.3f} L_F0={b['L_F0']:.3f}  P(-|O=2)={m.p_bad(m.filter(ctx),2):.3f}")
    print("  => a bad off-domain example -> M; a GOOD off-domain example supports the")
    print("     local-patch reading and suppresses broad EM (the reattribution lever).\n")

    # --- 4) prior sweep: predicted broad EM vs prior weight on M ---
    print("## 4) Broad EM P(-|O) vs prior P(M) (4 bad-F, ambiguous; no off-domain cue)")
    print(f"  {'P(M) prior':>10}  {'P(M|C)':>8}  {'broad EM P(-|O)':>16}")
    for pm in (0.05, 0.2, 0.5, 0.8, 0.95):
        rest = 1.0 - pm
        prior = {"A": rest * 0.5, "M": pm}
        prior.update({f"L_F{i}": rest * 0.5 / 3 for i in range(3)})
        mm = LatentCauseModel(k_domains=3, eps=0.01, prior=prior)
        ctx = [(0, MINUS)] * 4
        b = mm.posterior(ctx)
        print(f"  {pm:>10.2f}  {b['M']:>8.3f}  {mm.p_bad(mm.filter(ctx),1):>16.3f}")
    print("  => with ambiguous (in-domain only) evidence, broad EM is set by the PRIOR")
    print("     on M. This is the 'control the prior' knob the finetuning FT-2 tests:")
    print("     the optimizer default is high P(M); localizing levers add off-domain")
    print("     evidence that shifts the posterior to L_F.")


if __name__ == "__main__":
    demo()
