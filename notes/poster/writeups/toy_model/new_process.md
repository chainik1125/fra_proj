# Two-Sector (Direct-Sum) GHMM / OOM with Prompt Neutrality and Three Ways to Distinguish Sectors

This note describes a **general two-sector process** built as a **direct sum** (block-diagonal) of two sub-processes:
- **G** (“good”) sector
- **B** (“bad”) sector

It also explains **three different ways** the two sectors can be made distinguishable from observations, and gives an explicit “Option 2” construction (same vocabulary, different likelihoods) that matches the **\(\beta\)**-style collapse idea.

The model is written in the **operator / generalized HMM** form (common in OOM / weighted automata / PSR style):
\[
\Pr(x_{1:T}) = \pi_0\,T(x_1)\cdots T(x_T)\,\mathbf 1,
\]
where each symbol \(x\) has a nonnegative matrix \(T(x)\).

---

## 0) Hidden space and belief decomposition

Let the hidden space be a **direct sum**
\[
\mathcal H = \mathcal H_G \oplus \mathcal H_B,
\qquad
\dim(\mathcal H_G)=d_G,\ \dim(\mathcal H_B)=d_B.
\]

A (row) belief state decomposes as
\[
\pi = \big(\pi_G \mu_G,\ \pi_B \mu_B\big),
\]
where:
- \(\pi_G,\pi_B \ge 0,\ \pi_G+\pi_B=1\) are **macro weights** (“which sector”),
- \(\mu_G\in\Delta^{d_G-1}\), \(\mu_B\in\Delta^{d_B-1}\) are **within-sector beliefs**.

All symbol operators are **block diagonal**:
\[
T(x) = T_G(x)\oplus T_B(x).
\]
So probability mass **never moves between sectors**.

---

## 1) Prompt neutrality (between-sector neutrality)

Let the prompt alphabet be
\[
\mathcal X_{\text{prompt}}=\{p_1,\dots,p_K\}.
\]

For each prompt symbol \(p_k\), define within-sector dynamics:
\[
T_G(p_k) = c(p_k)\,S_G(p_k),\qquad
T_B(p_k) = c(p_k)\,S_B(p_k),
\]
where:
- \(S_G(p_k)\) and \(S_B(p_k)\) are **row-stochastic** (each row sums to 1),
- and the scalar **\(c(p_k)\)** is **the same in both sectors**.

Then the macro-odds update after observing a prompt token is:
\[
\frac{\pi_G'}{\pi_B'} = \frac{\pi_G}{\pi_B}\cdot \frac{c(p_k)}{c(p_k)} = \frac{\pi_G}{\pi_B}.
\]
So **any prompt string** (any length) preserves sector weights exactly.

> In words: prompts can change the *within-sector* beliefs \(\mu_G,\mu_B\), but do not leak which sector you are in.

---

## 2) Leaky-reset (belief-writing) dynamics within each sector

A simple, analyzable within-sector update is the “leaky reset” map.

For each sector \(S\in\{G,B\}\) and prompt token \(p_k\), choose a **signature distribution**
\[
r_{S,k}\in\Delta^{d_S-1}.
\]

Define the rank-1 reset matrix
\[
R_{S,k} = \mathbf 1\,r_{S,k},
\]
so every row of \(R_{S,k}\) equals \(r_{S,k}\), and for any belief \(\mu\),
\[
\mu R_{S,k} = r_{S,k}.
\]

Define the within-sector prompt operator
\[
S_S(p_k) = (1-\lambda_S)I_{d_S} + \lambda_S R_{S,k},
\qquad 0\le \lambda_S\le 1.
\]

Then within-sector beliefs update as
\[
\mu_S' = (1-\lambda_S)\mu_S + \lambda_S r_{S,k}.
\]

Closed form after a prompt string \(w=(k_1,\dots,k_P)\):
\[
\mu_S(P) = (1-\lambda_S)^P \mu_S(0)
+\lambda_S\sum_{t=1}^P (1-\lambda_S)^{P-t} r_{S,k_t}.
\]
So prompts write an exponentially-weighted “suffix code” into the belief.

---

## 3) Completion phase: three ways to make sectors distinguishable

Let the completion alphabet be \(\mathcal X_{\text{comp}}\).
You have **three options** for how \(G\) and \(B\) become distinguishable from observations.

### Option 1 — Different vocabularies (disjoint support)
Make some symbols possible only in \(G\) and never in \(B\), or vice versa.

Example:
- \(G\) can emit symbols in \(\{g_1,\dots,g_M\}\),
- \(B\) can emit symbols in \(\{b_1,\dots,b_M\}\),
- and these sets are disjoint.

**Pros:** instant, trivial identification.  
**Cons:** too easy; sector is detectable from a single token.

---

### Option 2 — Same vocabulary, different likelihood ratios (canonical / “\(\beta\)” collapse)
Use the **same symbols** in both sectors, but with different weights so each completion symbol applies a controlled Bayes factor to \(\pi_G/\pi_B\).

This is the most common/clean approach when you want:
- prompt phase neutral,
- completion phase gradually reveals the sector at a tunable rate.

#### 3.2.1 Tagged completion alphabet
Pick a base content set \(\{1,\dots,M\}\) and define **tagged** completion symbols:
\[
\mathcal X_{\text{comp}}=\{g_1,\dots,g_M\}\cup\{b_1,\dots,b_M\}.
\]

Interpretation:
- observing \(g_i\) is evidence favoring sector \(G\),
- observing \(b_i\) is evidence favoring sector \(B\),
- the index \(i\) is “content” that can decode within-sector state.

#### 3.2.2 Within-sector content readout (decode noise \(\delta\))
Choose maps (labels)
\[
\phi_G:\{1,\dots,d_G\}\to\{1,\dots,M\},\qquad
\phi_B:\{1,\dots,d_B\}\to\{1,\dots,M\}.
\]

Define a per-sector “readout weight vector” for each content index \(i\):
\[
(e_{S,i})_j=
\begin{cases}
1-\delta & \phi_S(j)=i\\[4pt]
\frac{\delta}{M-1} & \phi_S(j)\neq i,
\end{cases}
\qquad S\in\{G,B\}.
\]

A simple operator form is “emission-weighted identity”:
\[
D_S(i) = \mathrm{diag}(e_{S,i}).
\]
(If you prefer strictly row-stochastic operators, you can instead build a row-stochastic transition that emits \(i\) with these probabilities; in the operator/OOM view, normalization occurs in the belief update.)

\(\delta\in[0,1]\) is a **decode-noise knob**.

#### 3.2.3 Sector evidence knob \(\beta\)
Choose \(\beta\in(0,1)\). Define completion operators:
\[
T(g_i) = \bigl(1\cdot D_G(i)\bigr)\ \oplus\ \bigl(\beta\cdot D_B(i)\bigr),
\]
\[
T(b_i) = \bigl(\beta\cdot D_G(i)\bigr)\ \oplus\ \bigl(1\cdot D_B(i)\bigr).
\]

**Effect on macro odds:**
- each observed \(g_i\) multiplies \(\pi_G/\pi_B\) by \(1/\beta\),
- each observed \(b_i\) multiplies \(\pi_G/\pi_B\) by \(\beta\).

So \(\beta\) sets the **collapse rate** (how quickly completions reveal the sector).

---

### Option 3 — Same vocabulary and matched one-step totals, but different dynamics
Even if you make single-symbol totals match (so one-step stats look similar), the sectors can be distinguishable from **multi-step correlations** if their state-update operators differ.

In operator terms:
- enforce identical row-sum scalars for each symbol in both blocks,
- but choose different within-sector operators \(S_G(x)\neq S_B(x)\),
so sequences reveal different predictive structure (e.g., different correlation functions / entropy rates).

**Pros:** most “interesting”: sectors differ in structure rather than trivial frequencies.  
**Cons:** harder to reason about; may require longer sequences to distinguish.

---

## 4) Standard experimental protocol (clean separation of knobs)

1) **Prompt phase:** length \(P\), symbols in \(\mathcal X_{\text{prompt}}\)
   - Use prompt-neutral operators \(T_S(p_k)=c(p_k)S_S(p_k)\) so \(\pi_G/\pi_B\) does not change.
   - Within-sector beliefs \(\mu_G,\mu_B\) become prompt-conditioned via leaky reset.

2) **Completion phase:** length \(L>P\), symbols in \(\mathcal X_{\text{comp}}\)
   - Choose one of the three options to make sectors distinguishable.
   - Option 2 is the clean “collapse + decode” design via \(\beta\) and \(\delta\).

---

## 5) Knobs summary

- **Sector dimensions:** \(d_G, d_B\)
- **Prompt writing:** \(\lambda_G,\lambda_B\) and signatures \(r_{S,k}\)
- **Prompt neutrality:** enforce equal scalars \(c(p_k)\) across sectors
- **Sector distinguishability:**
  - Option 1: disjoint vocabularies
  - Option 2: same vocab, different likelihood ratios via \(\beta\)
  - Option 3: same vocab, different dynamics (correlations)
- **Decode noise:** \(\delta\)
- **Completion length:** \(L\) (more tokens = more evidence, better separability)

---

## 6) Minimal concrete example (small numbers)

- \(d_G=5\), \(d_B=5\)
- prompt vocab \(K=5\), completion content \(M=5\)
- signatures one-hot: \(r_{S,k}=e_k\)
- \(\lambda_G=\lambda_B=0.6\)
- decode noise \(\delta=0.05\)
- sector evidence \(\beta=0.6\)
- prompt scalars \(c(p_k)=1/5\) (neutral)

This yields:
- prompts do not change \(\pi_G/\pi_B\),
- completions (Option 2) collapse sector odds at a tunable rate and decode within-sector state.
