# EM-AFP, minimal restart

A fresh, deliberately-minimal take on the EM-AFP toy. We keep the **semi-factored process
(SFP)** structure — persona × domain — but strip everything else back until each moving part
earns its place. This doc is step 1: **state precisely the HMM we are currently running**, so
we have a clear baseline to minimize from. (Simplification proposals come in a later section /
later doc.)

Notation: persona `s ∈ {M, A}` (Misaligned / Aligned; in the builder these are leaves `B`/`A`),
domain `r ∈ {D, O}` (the finetuned Domain / the Other, untouched domain; builder `1`/`2`).

---

## 1. Hidden state space

A **direct sum of 4 leaves**, one per (persona, domain) pair, each an independent block of
`d` states:

$$
\mathcal H \;=\; H_{MD}\,\oplus\,H_{MO}\,\oplus\,H_{AD}\,\oplus\,H_{AO},
\qquad |H_\ell| = d = 5 \;\Rightarrow\; 20 \text{ states.}
$$

Every symbol operator is **block-diagonal** over these 4 leaves, so a trajectory that starts in
one leaf stays there forever — the process is a frozen mixture of 4 ergodic components (strictly
non-ergodic). A leaf is the "ground truth" (persona, domain) for a whole sequence; what moves
during a sequence is only the *observer's belief* about which leaf it's in.

Belief: `π = (π_ℓ · μ_ℓ)_ℓ`, where `π_ℓ` is the mass on leaf ℓ and `μ_ℓ ∈ Δ^{d-1}` is the
within-leaf distribution.

## 2. Two phases per sequence

A sequence is `L_p` prompt tokens followed by `L_c` completion tokens (default `L_p=3`, `L_c=20`).

### Phase 1 — prompt (domain-informative, persona-neutral)

Vocabulary: `V_p · K` tokens (V_p=5 content signatures × K=2 domain tags = 10). A prompt token
is `(r*, k)` — domain tag `r*`, content signature `k`. Its operator, on the block of leaf `(s,r)`:

$$
T_{\text{prompt}}(r^\*,k)\big|_{(s,r)} \;=\; c \cdot Q_p[r, r^\*]\cdot S_k,
\qquad
Q_p[r,r^\*] = \begin{cases}1 & r=r^\*\\ \beta_{pd} & r\ne r^\*\end{cases},
\quad c=\tfrac{1}{V_p\,(1+(K-1)\beta_{pd})}.
$$

- `S_k = (1-\lambda) I + \lambda\,\mathbf 1\,e_{k\bmod d}^\top` is a **leaky reset** toward the
  one-hot signature `e_{k\bmod d}` (λ=0.6). It is **identical in every leaf** → it only writes the
  within-leaf content `μ` (a "codebook address"), carrying no persona/domain information by itself.
- `Q_p[r,r*]` scales each leaf's block by its **domain** match to the tag — **identical across
  personas** → the prompt shifts the *domain* belief toward `r*` but says **nothing about persona**.
- A "domain-O prompt" = a string of O-tagged tokens; it sets domain O (≈89% at β_pd=0.5, L_p=3)
  while leaving persona at its prior. (Neutral-prompt variant: `β_pd → none`, V_p tokens, no domain
  evidence — fully neutral on both axes; this was the original version.)

### Phase 1, made concrete (the actual matrices)

`T_prompt` for **one** token is a 20×20 **block-diagonal** matrix — four 5×5 leaf blocks
`(AD | AO | MD | MO)`, each equal to a per-leaf scalar `g_ℓ` times the **same** 5×5 content
matrix `S_k`. For content `k=0`, `S_0` (leaky reset toward state 0, λ=0.6) is:

```
S_0 =  [1.0   .    .    .    . ]     row i: jump to state 0 with prob λ=0.6,
       [0.6  0.4   .    .    . ]            else stay  (row 0 already there -> 1.0)
       [0.6   .   0.4   .    . ]
       [0.6   .    .   0.4   . ]
       [0.6   .    .    .   0.4]
```

The per-leaf scalar is `g_ℓ = c · Q_p[domain(ℓ), r*]`, with `c = 1/(V_p(1+(K-1)β_pd)) = 0.133`
at β_pd=0.5:

```
token tagged domain D:   g_AD = g_MD = c      = 0.133   (domain match)
                         g_AO = g_MO = c·β_pd = 0.067   (domain mismatch)
token tagged domain O:   g_AD = g_MD = 0.067 ,  g_AO = g_MO = 0.133
```

So `T_prompt(D, k=0) = blockdiag( 0.133·S_0 , 0.067·S_0 , 0.133·S_0 , 0.067·S_0 )`. In the dumped
matrix that's `0.133 = c·1.0`, `0.08 = c·0.6`, `0.053 = c·0.4` in the D-blocks, and the same ×0.5
in the O-blocks. Three things to read straight off the numbers:

- **Block-diagonal** — every nonzero entry sits inside one leaf's 5×5 block ⇒ no transitions
  between leaves (persona/domain are frozen within a sequence).
- **Domain evidence** — a D-tagged token scales the two D-leaves by `c` and the two O-leaves by
  `c·β_pd`; the ratio `1/β_pd = 2` is a Bayes factor of 2 toward domain D per token. It is
  **persona-neutral**: AD and MD get the *identical* scalar (likewise AO, MO).
- **Content** — the within-block `S_k` (identical in every leaf) only writes the codebook address μ.

A single token's matrix is **not** row-stochastic (it's one observable operator); the **sum over
all 10 prompt tokens is** (row sums = 1).

**This is exactly where the incidental complexity lives.** The 5×5 `S_k` + the V_p content
signatures are pure codebook. Drop within-leaf content (`d=1`) and `S_k → [1]`, so the whole
phase-1 operator collapses to a **4×4 diagonal of the scalars**, e.g. for a D-token
`diag(c, c·β_pd, c, c·β_pd)` — pure domain evidence, no codebook. That is the first minimisation.

### Phase 2 — completion (tagged evidence, `W = P ⊗ Q`)

Vocabulary: `2K · M` tokens (4 leaves × M=5 contents = 20). A completion token is `(s*, r*, i)` —
persona tag `s*`, domain tag `r*`, content `i`. Its operator on leaf `(s,r)`:

$$
T_{\text{comp}}(s^\*,r^\*,i)\big|_{(s,r)} \;=\; \tfrac1Z\, W[(s,r),(s^\*,r^\*)]\cdot \mathrm{diag}(e_i),
\qquad W = P \otimes Q,
$$

$$
P=\begin{pmatrix}1&\beta_{\text{per}}\\ \beta_{\text{per}}&1\end{pmatrix}\!\text{ over }(M,A),
\quad
Q=\begin{pmatrix}1&\beta_{\text{dom}}\\ \beta_{\text{dom}}&1\end{pmatrix}\!\text{ over }(D,O),
\quad
e_i[j]=\begin{cases}1-\delta & j=i\\ \tfrac{\delta}{M-1}& j\ne i\end{cases}.
$$

- `W[ℓ, tag]` is an **emission likelihood** (a scalar weight on leaf ℓ's block), *not* a transition
  between leaves — the blocks never mix. It says: how likely is leaf ℓ to emit a token with this tag.
  Factored: persona-mismatch costs `β_per`, domain-mismatch costs `β_dom`.
- `diag(e_i)` reads the within-leaf content (δ=0.05 noise); diagonal → during completion the hidden
  state is **frozen** (a static channel). The whole completion is repeated noisy emissions from one
  fixed (leaf, within-state).
- For a Bayesian observer, each completion token multiplies the leaf posterior by `W[·, tag]` — these
  are **Bayes factors**: a persona-`s*` tag shifts persona log-odds by `±log(1/β_per)`, a domain-`r*`
  tag shifts domain log-odds by `±log(1/β_dom)`.

## 3. Parameters (current defaults)

| symbol | meaning | value |
|---|---|---|
| `d` | states per leaf | 5 |
| `K` | domains | 2 |
| `β_per` | persona evidence (P) — smaller = stronger | 0.5 |
| `β_dom` | domain evidence (Q) | 0.6 |
| `β_pd` | domain evidence in the *prompt* | 0.05–0.5 (swept) |
| `λ` | prompt leaky-reset strength | 0.6 |
| `δ` | completion content noise | 0.05 |
| `V_p` | prompt content signatures | 5 |
| `M` | completion content symbols | 5 |
| `L_p, L_c` | prompt / completion length | 3 / 20 |
| `π_A` | prior P(aligned persona) | 0.5 |

Vocab = `V_p·K + 2K·M` = 10 + 20 = 30 tokens. (Persona-dominant regime: β_per < β_dom.)

## 4. Generative story

Draw a leaf from the prior (persona ~ (π_A, 1-π_A), domain uniform). Emit `L_p` prompt tokens
(content via the leaky reset; domain tag favouring the leaf's domain) then `L_c` completion tokens
(tag favouring the leaf's persona+domain; content from the frozen within-state). For ICL /
multi-segment training, the leaf is held **fixed across several prompt+completion segments**, so
persona and domain *persist* — that's what lets in-context examples shift the belief.

## 5. Exact (factored) Bayes filter

Because `W = P ⊗ Q` and the content machinery is identical in every leaf, the posterior keeps a
product form and splits into three independent coordinates:

$$
\theta=\log\tfrac{a_M}{a_A}\ \text{(persona log-odds)},\quad
\phi=\log\tfrac{b_D}{b_O}\ \text{(domain log-odds)},\quad
\mu\ \text{(within-leaf content)},
$$

with updates: completion persona-tag → `θ ± log(1/β_per)`; completion domain-tag → `φ ± log(1/β_dom)`;
prompt domain-tag → `φ` shift (persona-neutral); content tokens → `μ`. This `(θ, φ, μ)` is the
minimal sufficient statistic, and it gives exact probe targets `μ*(w)` and the analytic
"misalignment rate" readouts (e.g. P(M | domain O) = the broad-EM measure).

---

## 6. Anatomy — what each piece is for (to guide minimisation)

| piece | role | essential to EM? |
|---|---|---|
| 4 leaves = persona × domain (SFP) | the whole point: local (domain) vs global (persona) factor | **yes** (keep) |
| `β_per < β_dom` (factored W) | makes persona the shared/global coordinate | **yes** |
| within-leaf content (`d>1`, signatures, μ) | a "codebook"; lets prompts carry identity | probably **no** — candidate to drop (d=1?) |
| leaky-reset prompts | writes content; original neutrality | simplifiable if content is dropped |
| domain-informative prompts (`β_pd`) | lets a prompt set the domain (needed once we ask "answer domain O") | **yes** for the behavioural eval |
| tagged completion (`W`) | the evidence/observation channel | **yes** |
| multi-segment persistence | needed only for the ICL arm | optional |
| `π_A`, finetuning, coherence gate | the EM experiment on top of the process | separate from the HMM itself |

**Open question for the next step:** the within-leaf content machinery (signatures, μ, d=5,
leaky reset) is the largest source of incidental complexity and may be unnecessary for EM —
the EM phenomenon lives entirely in the leaf masses `(π_MD, π_MO, π_AD, π_AO)` / the `(θ, φ)`
coordinates. A first minimisation is likely **d = 1 per leaf** (no within-leaf content), which
collapses the prompt to a pure domain-setting signal and the completion to pure persona+domain
tags. We will sanity-check that EM survives that before adding anything back.
