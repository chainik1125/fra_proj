# EM-AFP specific proposal 1 (claude_sol)

*Specific, runnable instantiation of `em_afp_proposal_base.md`. This `claude_sol`
line is independent of `codex_sol`; the choices below are reasoned from first
principles. Scope of this doc: pretrain → finetune → ICL → behavioral + representational
posterior probes → controls. The MatTXC component-recovery bonus is specified but
deferred (not run here).*

## 0. Summary

Pretrain a tiny transformer on a four-component **semi-factored process** (SFP)

$$
[T_M\oplus T_A]\otimes[T_D\oplus T_O]\;\cong\;T_{MD}\oplus T_{MO}\oplus T_{AD}\oplus T_{AO},
$$

induce misalignment on the **local misaligned** component $MD$ (by finetuning, then by
in-context examples), and measure whether it spills into the **global misaligned**
component $MO$ (same persona, *untouched* domain) more than into the **local aligned**
component $AD$ (same domain, opposite persona). The phenomenon to reproduce: *the local
factor (domain) competes with the global factor (persona), and the global one wins.*

**Chosen HMM:** a four-leaf **hierarchical leaky-reset** process with factored tag
evidence $W=P\otimes Q$, in the persona-dominant regime $\beta_{\rm per}=0.5<\beta_{\rm dom}=0.6$.
It is the unique candidate that simultaneously gives (i) an **exact closed-form factored
Bayes filter** — both the probe target and the ICL null — (ii) **exact polarization** of
each component over a completion, (iii) **tagged completions** so the behavioral posterior
is a one-line readout from logits, and (iv) built-in **flat / scrambled** controls that kill
the "generic spillover" objection. It is also already validated: the June hierarchy
follow-on to the poster reproduced the $MD\to MO$ spillover and the controls on exactly
this process.

**What this doc adds beyond a re-run of that result.** The base proposal's central new
hypothesis is that *ICL keeps all factors while finetuning weakens the non-selected ones.*
The leaky-reset's closed-form filter lets us make that exact and falsifiable: **ICL is
Bayesian conditioning** (it should track a known closed-form posterior shift and leave every
component representation intact), whereas **finetuning is a prior tilt** (it should *overshoot*
the Bayesian-conditioning null and/or damage the off-component representations). This turns a
qualitative claim into a quantitative null with zero free parameters.

---

## 1. The process, precisely

### 1.1 Hidden space and components

$$
\mathcal H=H_{MD}\oplus H_{MO}\oplus H_{AD}\oplus H_{AO},\qquad |H_\ell|=d=5
$$

(20 hidden states). A leaf is $\ell=(s,r)$ with persona $s\in\{M,A\}$ and domain
$r\in\{D,O\}$. Belief $\pi=(\pi_\ell\,\mu_\ell)_\ell$ with leaf masses $\pi_\ell$ and
within-leaf beliefs $\mu_\ell\in\Delta^{d-1}$. Symmetric initialization:
$\pi_\ell=1/4$, $\mu_\ell=\mathrm{uniform}(d)$.

This is a **direct sum**, not a hidden-space Kronecker product: every symbol operator is
block-diagonal, so a trajectory started in leaf $\ell$ stays in $H_\ell$ — the process is
strictly non-ergodic, a 4-atom mixture of distinct ergodic components. The "$\otimes$" in
the proposal lives in the **observation/evidence channel** (§1.3), which is exactly what
"semi-factored" should mean and exactly what makes the persona a latent that is *frozen per
sequence and inferred* — the structure EM needs. (A per-token mixture, where the component is
resampled every step, would be the wrong generative model; see §7 on MatTXC.)

### 1.2 Prompt phase — sector-neutral leaky reset

$V_p=d=5$ prompt tokens, prompt length $L_p=3$ ($5^3=125$ neutral prompts). For prompt
token $p_k$,

$$
T_{\rm prompt}(p_k)=\tfrac1{V_p}\bigoplus_\ell S_k,\qquad
S_k=(1-\lambda)I+\lambda\,\mathbf 1\,e_{k\bmod d}^\top,\qquad \lambda=0.6,
$$

identical in every leaf. Consequences: the within-leaf belief updates as
$\mu_\ell\leftarrow(1-\lambda)\mu_\ell+\lambda e_{k\bmod d}$ (a content/address code), while
the **leaf masses $\pi_\ell$ never change** (each block is scaled by the common $1/V_p$ and a
row-stochastic $S_k$). **Prompt neutrality is the key property:** prompts carry zero evidence
about persona or domain, so eval prompts cannot leak the answer, and any post-induction shift
in $\mu_M$ is a genuine prior/weight change, not conditioning on a cue.

### 1.3 Completion phase — factored tagged evidence

Completion tokens $x_{(s^\star,r^\star),i}$ with $i\in\{0,\dots,d-1\}$: vocabulary
$4d=20$. Completion length $L_c=20$. Content emission is leaf-symmetric,

$$
D(i)=\mathrm{diag}(e_i),\qquad e_i[j]=\begin{cases}1-\delta & j=i\\ \delta/(d-1)& j\ne i\end{cases},\qquad \delta=0.05,
$$

and the tag evidence **factorizes**:

$$
W[(s,r),(s^\star,r^\star)]=P[s,s^\star]\,Q[r,r^\star],\quad
P=\begin{pmatrix}1&\beta_{\rm per}\\\beta_{\rm per}&1\end{pmatrix},\quad
Q=\begin{pmatrix}1&\beta_{\rm dom}\\\beta_{\rm dom}&1\end{pmatrix},
$$

rows/cols ordered $(M,A)$ and $(D,O)$. The completion operator is

$$
T_{\rm comp}(x_{(s^\star,r^\star),i})=\tfrac1Z\bigoplus_{\ell=(s,r)}W[\ell,(s^\star,r^\star)]\,D(i),
\qquad Z=(1+\beta_{\rm per})(1+\beta_{\rm dom}).
$$

**Persona-dominant regime:** $\beta_{\rm per}=0.5<\beta_{\rm dom}=0.6$ (smaller $\beta$ =
stronger mismatch penalty). An $MD$ token is therefore stronger evidence for same-persona,
other-domain $MO$ than for other-persona, same-domain $AD$:

$$
W[MO,MD]=\beta_{\rm dom}=0.6\;>\;W[AD,MD]=\beta_{\rm per}=0.5 .
$$

This single inequality is the local-vs-global competition we want to read out.

### 1.4 Why this HMM, against the alternatives

| Candidate | Verdict |
|---|---|
| **Four-leaf hierarchical leaky-reset (chosen)** | Exact factored filter, exact polarization, tagged → trivial behavioral readout, prompt-neutral eval, built-in flat/scrambled controls, already validated on the EM spillover. |
| Literal $Z1R'\times Z1R'$ Kronecker AFP (`training/matrices.py`) | A true hidden-space tensor product, but its sector split is asymmetric $1\oplus4$ and persona-only, and it lacks prompt-neutrality. A faithful "$\otimes$" but a worse measurement substrate. Not first. |
| Mess3 / TomQ component SFP | More naturalistic within-component geometry, but polarization is only partial/tunable (distinguishability $\delta$), the continuous belief simplex makes the posterior target high-dim and centroid/probe measurement harder, and there is no clean closed-form filter. The right **robustness follow-up** ("confirm it holds for Mess3"), not the first run. |

---

## 2. Exact Bayes filter and posterior coordinates

Because prompts are leaf-symmetric and $W=P\otimes Q$, the filter **factorizes exactly**
(proved for this process in `theory_notes.md` Prop. 1). With $a_s=\pi_{sD}+\pi_{sO}$ and
$b_r=\pi_{Mr}+\pi_{Ar}$,

$$
\pi_{(s,r),t}=a_{s,t}\,b_{r,t},\qquad \mu_{\ell,t}=\mu_t\ \ \forall\ell .
$$

The minimal sufficient statistic is $(\theta,\phi,\mu)$ with persona log-odds
$\theta=\log(a_M/a_A)$ and domain log-odds $\phi=\log(b_D/b_O)$. On a completion token,

$$
\theta\leftarrow\theta\pm\log(1/\beta_{\rm per})\ \text{(persona tag)},\qquad
\phi\leftarrow\phi\pm\log(1/\beta_{\rm dom})\ \text{(domain tag)},
$$

content $\mu$ updated by the emitted index alone. Three facts we use:

1. **Polarization is exact and tunable.** Starting at $\theta=0$, an all-$M$ completion drives
   $\theta\to L_c\log(1/\beta_{\rm per})$; the component mass is $1/(1+(1/\beta)^{L_c})$ at the
   wrong corner. At $\beta_{\rm per}=0.5,\,L_c=20$ this is far past collapse — answering the
   base proposal's polarizability worry (Part 3a) in closed form. ($\beta,L_c$ are the dials.)
2. **There is exactly one shared persona coordinate** $\theta$, common to both domains
   (Corollary 1, `theory_notes.md`). This is the mechanistic substrate of cross-domain transfer
   and the thing the flat control provably lacks.
3. **The exact posterior is computable at every position** via the forward filter, giving probe
   targets $\mu^\star(w)=(\mu^\star_{MD},\mu^\star_{MO},\mu^\star_{AD},\mu^\star_{AO})$.

---

## 3. Operationalizing "misalignment"

Following the base proposal, misalignment is **posterior mass on the misaligned persona**,
marginalized over domain:

$$
\mu_M=\mu_{MD}+\mu_{MO}.
$$

The decision-relevant, *broad* (out-of-finetuning-domain) rate is

$$
\mathrm{BroadEM}_O=\frac{\mu_{MO}}{\mu_{MO}+\mu_{AO}},
$$

i.e. "when the model is in the untouched domain $O$, how often is it misaligned?" This
separates broad EM from mere in-domain collapse ($\mu_{MD}$ saturating).

---

## 4. Measurement framework

### 4.1 Behavioral posterior $\hat\mu_{\rm beh}$ (the operational phenomenon)

Marginalize the model's next-token distribution over content to get the four **tag** masses
$y_t(w)=p_\theta(\text{next tag}=t\mid w)$. In principle $y=\mu_{\rm beh}\,\tilde W$ for the
row-normalized evidence matrix $\tilde W$, invertible to $\hat\mu_{\rm beh}=y\tilde W^{-1}$
(simplex-projected). **We treat the inversion as a secondary cross-check only**, because
$\tilde W$ is ill-conditioned for the flat control and the inversion can leave the simplex
after collapse.

**Primary readout = normalizer-free log-odds gaps**, which cancel the unknown finetuning dose
$\eta$ and the partition function. Define $\Delta\log p_X=\log p^{\rm post}_X-\log p^{\rm base}_X$
for tag group $X$. The headline transfer statistic is the **global-minus-local gap**

$$
G=\big[\Delta\log p_{MO}-\Delta\log p_{AO}\big]-\big[\Delta\log p_{AD}-\Delta\log p_{AO}\big]
=\Delta\log\tfrac{p_{MO}}{p_{AO}}-\Delta\log\tfrac{p_{AD}}{p_{AO}} .
$$

$G>0$ means the global misaligned component received more transfer than the local aligned one
— the experiment's success condition. The **dose-free ratio invariant** (derived from the tilt
model in `theory_notes.md`) is a zero-parameter check:

$$
R=\frac{\Delta\log p_{MO}-\Delta\log p_{AO}}{\Delta\log p_{AD}-\Delta\log p_{AO}}
=\frac{\log\beta_{\rm per}}{\log\beta_{\rm dom}}=\frac{\log0.5}{\log0.6}\approx1.357 .
$$

### 4.2 Representational posterior $\hat\mu_{\rm rep}$ (the mechanistic probe)

Linear map into natural coordinates + softmax,

$$
\hat\mu_{\rm rep}(w)=\mathrm{softmax}\!\big(B\,a_{\ell,p}(w)+b\big),\qquad
\min_{B,b}\ \mathbb E_w\,D_{\rm KL}\!\big(\mu^\star(w\mid\text{context})\,\|\,\hat\mu_{\rm rep}(w)\big)+\lambda\|B\|^2 .
$$

Constraining the output to a valid posterior is why this beats four independent regressions.
**Crucially, the target is the context-dependent exact posterior**: under ICL the label must
be $\mu^\star(w\mid\text{ICL context})$, not the base neutral-prompt posterior. Probe at
positions {final-prompt, first-completion, last-completion} × layers {0, 1, final}. Report:

1. full 4-component KL / R²;
2. persona marginal $\mu_M=\mu_{MD}+\mu_{MO}$ and domain marginal $\mu_D=\mu_{MD}+\mu_{AD}$;
3. the **interaction coordinate** $\chi=\log\frac{\mu_{MD}\mu_{AO}}{\mu_{MO}\mu_{AD}}$.
   $\chi=0$ ⟺ the represented posterior factorizes across persona and domain; $\chi\ne0$ ⟺ the
   representation carries genuine component-level correlation. Tracking $\chi$ pre/post tells us
   whether induction keeps the representation factorized or correlates the components.

### 4.3 Frozen vs refit probes (the FT-vs-ICL discriminator)

For each post-induction condition run two probe tests against the same Bayes labels:

- **Frozen:** apply the *base-trained* probe to post-induction activations — is the original
  coordinate still present, in the same geometry?
- **Refit:** train a fresh probe on post-induction activations — is the information present but
  *rotated/reparameterized* vs genuinely *absent*?

| frozen | refit | reading |
|---|---|---|
| shifts with behavior | — | belief shifted in the original coordinate system |
| fails (non-$MD$) | succeeds | information retained but geometry changed |
| fails (non-$MD$) | fails | non-selected component representation **damaged/discarded** |
| preserved | preserved | representation untouched (induction only moved the prior) |

---

## 5. The core contribution: ICL = Bayesian conditioning vs FT = prior tilt

The closed-form filter turns the base proposal's "ICL narrow / FT broad, ICL keeps reps / FT
damages them" into an exact, falsifiable comparison.

**ICL is Bayesian conditioning.** A model that learned the process should, given $k$ in-context
$MD$ examples, update its posterior toward $\mu^\star(\cdot\mid\text{ICL context})$. Because
prompt tokens are persona-neutral and each example contributes $L_c$ $MD$-tagged completion
tokens, the persona log-odds shift is **known in closed form**:

$$
\Delta\theta_{\rm ICL}(k)=k\,L_c\,\log(1/\beta_{\rm per}),\qquad
\Delta\phi_{\rm ICL}(k)=k\,L_c\,\log(1/\beta_{\rm dom}).
$$

This is the **Bayes-conditioning null**. Predictions for ICL: (i) behavioral $\hat\mu_{\rm beh}$
tracks $\mu^\star(\theta=\Delta\theta_{\rm ICL}(k),\dots)$; (ii) it moves *along* the existing
posterior manifold, so **frozen probes for all four components stay accurate**; (iii) it is
context-dependent and reversible.

**FT is a prior tilt.** Finetuning modifies weights and is not bounded by any finite context. It
should **overshoot** the Bayes-conditioning null (more persona transfer than $\Delta\theta$ from
the matched evidence would justify) and/or **degrade frozen-probe quality for non-$MD$
components**. The dose-free $R$ invariant and $G>0$ should hold; the *new* claim is the
ICL-vs-FT split on the frozen-probe panel and on overshoot-vs-track of the closed-form null.

**OOD-context caveat (bake into the protocol).** A $k$-example ICL context is much longer than a
single pretraining sequence ($L_p+L_c=23$). If the model never saw multi-segment contexts, ICL
tests *filter generalization* and confounds the comparison. Mitigation: pretrain with
$n_{\rm ctx}\ge k_{\max}(L_p+L_c)$ and include **multi-segment** sequences (several
prompt+completion blocks concatenated, persona/domain resampled per block from the prior) so
cross-segment filtering is in-distribution. Report the base model's cross-segment posterior
recovery as an acceptance gate before trusting the ICL arm.

---

## 6. Protocol

### 6.1 Base model

2-layer TransformerLens, $d_{\rm model}=64$, 2 heads, $d_{\rm mlp}=256$; $n_{\rm ctx}\ge
k_{\max}(L_p+L_c)$ (≥ ~368 for $k_{\max}=16$); pretrain 5k–10k steps, batch 128, lr $10^{-3}$,
on full multi-segment process sequences. **Accept iff:**

1. cross-entropy ≈ Bayes-optimal process loss;
2. tag-marginal logits recover $\mu^\star$ at low KL on held-out prompts (including a
   cross-segment posterior check);
3. a linear-softmax probe recovers $\mu^\star(w)$ at high held-out KL/R².

### 6.2 Finetuning (induce on $MD$)

Pure-$MD$ completions: pick 6/125 prompts (5%), project the post-prompt belief onto $MD$, sample
$N=50$ completions of length $L_c$ each (content iid from the readout; assert zero non-$MD$
tokens). Train next-token loss, lr $3\times10^{-4}$, batch 64, checkpoint every 20–100 steps.
Dose sweep {0, 20, 50, 100, 200, 500, 1000} steps. **Primary readout at matched dose** = first
checkpoint with $\hat\mu^{\rm beh}_{MD}>0.5$ on held-out neutral prompts (collapse is fast;
matched dose keeps the $MO$-vs-$AD$ comparison from being washed out); final checkpoint secondary.

### 6.3 ICL (induce on $MD$ in context)

Same held-out neutral prompt format; prepend $k\in\{1,2,4,8,16\}$ in-context $MD$ examples before
the test boundary. Measure $\hat\mu_{\rm beh}$ vs the closed-form null $\Delta\theta_{\rm ICL}(k)$
and the frozen/refit representational signature.

### 6.4 Controls

- **Flat:** $W_{\rm flat}=(1-\beta)I_4+\beta J_4$, $\beta=0.55$ (no privileged persona
  coordinate). Predict $G\approx0$.
- **Scrambled:** hierarchical weights with off-domain leaves' grouping swapped ($MO\leftrightarrow
  AO$ in the evidence geometry, labels fixed). Predict transfer to the rewired partner.
- **Flipped-β:** $\beta_{\rm per}=0.6,\ \beta_{\rm dom}=0.5$ (domain-dominant). Predict $AD$
  transfer exceeds $MO$; ordering flips; $R\approx\log0.6/\log0.5\approx0.737$.

---

## 7. Deferred bonus — MatTXC component recovery (specified, not run)

The base proposal's bonus is to adapt the MatryoshkaTXC that recovered component weights near
the Bayes ceiling (`sae_day/notes/txc_matryoshka_results.md`) to read the four $\mu_i$ here. One
structural caveat must be resolved first: the `sae_day` MatTXC pipeline is built on a **per-token
mixture** process (component resampled every step; `omega` = mixture weight), whereas this EM
process is a **frozen-component non-ergodic direct sum** (component fixed per sequence; "omega" =
posterior over which leaf generated the sequence). Naive reuse would measure a different object.
Two clean options for later: (a) port MatTXC to ingest activations from *this* process with
$\mu^\star(w)$ as the SequenceOmega/LatentPosterior targets (the TXC/MatTXC classes are generic
over $K$; only the data/target adapters change), recomputing the $R^2$ ceiling from this
generator; or (b) build a Mess3-component version of the **tagged direct-sum** (not the sae_day
mixture) for the robustness arm and run MatTXC there. Either is out of scope for run 1.

---

## 8. Pre-registered predictions

1. **Ordering / transfer (matched dose, FT):** $MD\gg MO>AD>AO$; global-minus-local gap $G>0$;
   ratio $R=\log\beta_{\rm per}/\log\beta_{\rm dom}\approx1.36$.
2. **Broad EM rises:** $\mathrm{BroadEM}_O=\mu_{MO}/(\mu_{MO}+\mu_{AO})>0.5$ in the untouched domain.
3. **ICL tracks the Bayes-conditioning null:** $\hat\mu_{\rm beh}$ under $k$ examples matches
   $\mu^\star(\Delta\theta_{\rm ICL}(k))$; FT **overshoots** it.
4. **FT damages, ICL preserves:** FT degrades frozen-probe quality for non-$MD$ components (refit
   recovers ⟺ rotation, not loss); ICL leaves frozen probes accurate for all four.
5. **Controls:** flat → $G\approx0$; scrambled → transfer to the rewired partner; flipped-β →
   $AD$ beats $MO$, ordering flips, $R\approx0.74$.

## 9. Implementation / reuse

- **Port** `build_hierarchical_leaky_reset_hmms` (and helpers `_persona_domain`, `_w_hier`, the
  scrambled permutation $\sigma$) from
  `.claude/worktrees/poster-results-trace/analysis/afp_builders.py` into the current branch
  (alongside the existing 2-sector `build_leaky_reset_hmms` in `toy_ec/afp_builders.py`).
  Optionally port the harness `analysis/em_pipeline/hier_finetune.py`, `hier_steering.py` from the
  same worktree. Verify against `analysis/em_pipeline/test_hierarchical_process.py` (worked
  posterior to 1e-4).
- **Reuse** `toy_ec/em_pipeline/pretrain.py::compute_beliefs_for_sequences` for exact
  $\mu^\star(w)$ at every position (already in this branch).
- **Reuse** `training/run_minimal.py` (`Config`, `build_hmm`, `train`, `main_afp`) for pretraining
  via a thin adapter that wires the hierarchical process in (and emits multi-segment sequences,
  §5 caveat).
- **Write new:** (a) a KL-softmax posterior probe extending `bag_moments/probe.py` (which has only
  `ridge_probe`/`logistic_probe`); (b) the tag-marginal → $\hat\mu_{\rm beh}$ estimator and the
  $G$/$R$ log-odds readout.
- **Design refs:** `toy_ec/em_pipeline_notes/hierarchical_leaky_reset_implementation.md`;
  `notes/poster/writeups/followon_hierarchy/theory_notes.md` (factored filter, $R$ invariant,
  registered predictions P1–P5); base proposal `em_afp_proposal_base.md`.

Compute is small (the June runs were ≈$3 on T4s); the FT dose sweep + ICL sweep + 3 controls is a
handful of short runs.
