# AFP / Codebook Finetuning Proxy (HackMD-Style Notes)

This writeup explains:

- The baseline AFP process in `analysis/afp_steering.ipynb` (Z1R′×Z1R′, 9 hidden states).
- The **EM / latent-variable** analogy (what the “sector” variable is doing).
- Exactly **which sequences are used for finetuning** in the notebook (prompt selection + rejection-sampled completions).
- Why degeneracy happens in the baseline AFP, and what the **codebook** variant is trying to fix (as implemented in `analysis/afp_builders.py` and used in `analysis/afp_steering_gpt.ipynb`).
- How to interpret the metrics/results.

---

## 1) Baseline AFP: What Process Are We Training On?

The baseline notebook `analysis/afp_steering.ipynb` builds two HMMs:

- `prompt_hmm`: consumes **prompt tokens** (vocab size `V_P`) and updates a belief state over hidden states.
- `comp_hmm`: consumes **completion tokens** (vocab size `2*V_C = 8` in the default AFP) and updates belief similarly.

The full observed sequence is:

```
[prompt tokens of length P] + [completion tokens of length L]
 (plus a BOS-like offset for LM training)
```

### Hidden-state “sector” structure

The joint hidden state space is size 9 (cartesian product of 3×3 factor states).

We define two *macro-sectors*:

- Sector **A**: 1 state (index 0, corresponding to (S0,S0))
- Sector **B**: 4 states (indices `[4,5,7,8]`, corresponding to (S1,S1),(S1,SR),(SR,S1),(SR,SR))

The prompt/completion token matrices are constructed to preserve mass in `A ⊕ B` (i.e. ignore the mixed blocks).

### What prompt tokens do

Each prompt token `p_k` reweights the *sector odds*:

\[
\frac{\pi_A}{\pi_B} \leftarrow \frac{\pi_A}{\pi_B}\cdot \frac{c_A(p_k)}{c_B(p_k)}.
\]

Intuition:

- prompt tokens control the **prior** (how likely sector A vs B seems after the prompt),
- but do *not* necessarily create a rich mapping from prompt → a unique completion.

### What completion tokens do (collapse knob `beta`)

Completion tokens come in “good-tagged” vs “bad-tagged” families.

- “good-tagged” tokens are evidence for sector **A**
- “bad-tagged” tokens are evidence for sector **B**

The key property is that each completion token multiplies the sector odds by an approximately constant factor (after normalization), so after `L` tokens the posterior is “collapsed” toward one sector at a rate controlled by `beta`.

With the simplified odds picture (see `analysis/em_finetuning_writeup.md`), after `L` all-good tokens:

\[
\pi_A^{(L)}=\frac{(1/\beta)^L}{1+(1/\beta)^L}.
\]

So smaller `beta` ⇒ faster collapse.

---

## 2) The EM / Latent-Variable Analogy (What’s the “E-step” Here?)

Think of a latent discrete variable:

- \( z \in \{A, B\} \) (the macro-sector)

and observed tokens:

- prompt \(x_{1:P}\)
- completion \(y_{1:L}\)

Then the HMM belief update is doing something Bayes-like:

- prompt phase updates \(P(z \mid x_{1:P})\) (a *prior/posterior over sectors* given prompt evidence),
- completion phase updates \(P(z \mid x_{1:P}, y_{1:t})\) as you see completion tokens (a “likelihood” effect that sharpens the sector posterior).

The “EM analogy” is:

- **E-step-like:** infer/track the posterior over the latent sector \(z\) from partial observations (belief state).
- **M-step-like:** update model parameters so that the model’s token predictions match data that is implicitly conditioned on a particular sector.

Why this is useful for “emergent misalignment” as a toy proxy:

- prompts can leave the model in an *ambiguous* latent state (“maybe A, maybe B”),
- completions reveal which latent sector the data came from,
- finetuning on a biased subset (e.g. completions that end up in sector A) tests whether the model’s behavior on held-out prompts shifts in a systematic, “latent-variable consistent” way.

---

## 3) Exactly Which Sequences Are Used For Finetuning in `afp_steering.ipynb`?

This is the part that’s easy to misread.

### Step 1: finetune prompts are selected uniformly at random

In the finetuning setup cell (the one that computes `prompt_to_pi_a` and then picks `ft_prompts`), the code does:

```python
# Select random subset for finetuning
rng = np.random.RandomState(42)
n_ft = max(1, int(FT_PROMPT_FRAC * len(all_prompts)))
ft_idx = rng.choice(len(all_prompts), n_ft, replace=False)
ft_prompts = [all_prompts[i] for i in ft_idx]
```

So **prompt selection is not filtered by post-prompt π_A**. It’s a random 5% subset of prompt strings.

This is why the histogram for finetune prompts tends to look like a small-sample version of the overall prompt π_A distribution.

### Step 2: completions are sampled conditioned on each prompt and then filtered

For each `prompt_seq ∈ ft_prompts`, the notebook:

1. computes the post-prompt belief state `state = P(h | prompt_seq)`.
2. samples many completion sequences from `comp_hmm` starting from that belief.
3. computes sector-A mass at the **final completion position**.
4. accepts a completion if it ends “collapsed to A”:

```python
good = sa_mass > SECTOR_THRESH   # SECTOR_THRESH defaults to 0.9
```

This is **rejection sampling**: you generate lots of completions, then keep only those that satisfy the collapse criterion.

### Step 3: the finetuning dataset is prompt + completion (with token-id shifting)

Accepted completion tokens are shifted by `+V_P` so the model sees a single unified vocab:

```python
full_comp = ... + V_P
full_seq = torch.cat([prompt_t, full_comp], dim=1)
```

So the actual finetuning sequences are `ft_data` rows of length:

- `PROMPT_LEN + COMP_LEN` (default: `3 + 5 = 8`)

and the finetuning loss is standard next-token cross entropy on:

- `ft_inputs = ft_data[:, :-1]`
- `ft_labels = ft_data[:, 1:]`

### Finetuning dataset size (defaults)

- Number of unique prompts: `V_P^PROMPT_LEN = 10^3 = 1000`
- Finetune prompt fraction: `FT_PROMPT_FRAC = 0.05` ⇒ `50` prompts
- Per-prompt accepted sequences: `N_PER_PROMPT = 50`

So total finetune sequences:

- `50 * 50 = 2500` sequences

Important nuance:

- the code filters for “collapsed to sector A” at the end of the completion window.
- it does **not** accept “collapsed to either sector A or B”.

If you intended “either sector”, you’d change the acceptance condition to something like:

```python
good = (sa_mass > thresh) | (sb_mass > thresh)
```

(where `sb_mass = 1 - sa_mass` in the 2-sector setup).

---

## 4) What Do We Measure After Finetuning? (Interpretation)

The notebook saves checkpoints during finetuning and evaluates a *boundary bias*:

> “Given the prompt only, what probability does the model assign to the next token being bad-tagged vs good-tagged?”

In `analysis/afp_steering.ipynb`, `evaluate_sector_bias(...)` computes:

- `P(bad-tagged)` at the prompt/completion boundary by summing the model’s softmax over token ids `V_P+4 : V_P+8`.

### Interpretation of a shift

- If finetuning data is biased toward **sector A** completions (via rejection sampling),
  then a model that generalizes should move toward predicting more **good-tagged** tokens on *held-out prompts* too.
- That shows up as a **decrease** in `P(bad-tagged)` on held-out prompts as finetuning progresses.

The notebook also prints an “analytical baseline” that uses the HMM’s post-prompt π_A values:

\[
P(\text{bad} \mid \pi_A)=\frac{\pi_A\beta + (1-\pi_A)\alpha}{\alpha+\beta}.
\]

This is “what an HMM-optimal predictor would do” at the boundary, given only π_A.

### Why “sector bias” does not go to 100% (even on the finetune set)

There are two distinct reasons you should not expect the boundary metric to hit 100% even if you filter for completions that end highly collapsed to sector A.

#### Reason 1: In this AFP construction, “bad-tagged” is never impossible inside sector A

In the normalized AFP completion construction (see `analysis/em_finetuning_writeup.md`), for each base symbol \(s\in\{A,B,C,D\}\):

\[
T(\texttt{g}s)=\alpha\,\widehat M_A(s)+\beta\,\widehat M_B(s),\qquad
T(\texttt{b}s)=\beta\,\widehat M_A(s)+\alpha\,\widehat M_B(s),
\]

and the normalization is set so that \(\widehat M_A(s)\mathbf 1_A = \widehat M_B(s)\mathbf 1_B = m_s\mathbf 1\). That means the *base symbol* contributes the same total mass in both sectors, and the **tag** controls the sector evidence via \(\alpha\) vs \(\beta\).

If you condition on “we are in sector A”, then the marginal probability of the tag is:

\[
P(\texttt{good}\mid A)=\frac{\alpha}{\alpha+\beta},\qquad
P(\texttt{bad}\mid A)=\frac{\beta}{\alpha+\beta}.
\]

With the notebook defaults \(\alpha=1\), \(\beta=0.556\):

\[
P(\texttt{good}\mid A)=\frac{1}{1+0.556}\approx 0.6427,\qquad
P(\texttt{bad}\mid A)\approx 0.3573.
\]

So if your “bias” readout is effectively “probability of good-tagged at the boundary”, then **~64% is the ceiling** for a model that has perfectly inferred it is in sector A.

This is the main explanation for seeing ~65% rather than 100%.

#### Reason 2: The finetune dataset is filtered by *final* collapse, but the metric is at the boundary

In `analysis/afp_steering.ipynb`, finetune sequences are produced by rejection sampling on:

\[
E := \pi_A(\text{after }L\text{ completion tokens}) > \tau,
\]

with \(\tau=\texttt{SECTOR\_THRESH}\) (default 0.9) and \(L=\texttt{COMP\_LEN}\) (default 5).

But the evaluated “sector bias” is at the prompt/completion boundary, i.e. it is about:

\[
P(\text{tag}_1 \mid \text{prompt}, E),
\]

not \(P(\text{tag}_1 \mid \text{prompt})\), and not “was the completion collapsed at the end”.

You can make this more explicit with the standard odds picture. Let:

\[
r_0 := \frac{\pi_A(\text{prompt})}{1-\pi_A(\text{prompt})}.
\]

Under the tagged-token construction, each **good** tag multiplies the odds by \((1/\beta)\), and each **bad** tag multiplies odds by \(\beta\). After \(L\) completion tokens with \(g\) goods:

\[
r_L = r_0\,\beta^{\,L-2g}.
\]

Acceptance \(E\) is \(\pi_A(L)>\tau\), which is equivalent to:

\[
r_L > R_\tau := \frac{\tau}{1-\tau}.
\]

So \(E\) implies a minimum number of good tags \(g_{\min}\) such that:

\[
r_0\,\beta^{\,L-2g_{\min}} \ge R_\tau.
\]

##### Example calculations (defaults: \(\beta=0.556\), \(L=5\), \(\tau=0.9\))

Here \(R_\tau=\frac{0.9}{0.1}=9\).

1) If the prompt leaves \(\pi_A(\text{prompt})=0.5\), then \(r_0=1\). Check how many good tags are needed:

- For \(g=4\): \(r_L = 1\cdot \beta^{5-8}=\beta^{-3}\approx 1/0.556^3\approx 5.8\) (not enough: \(<9\)).
- For \(g=5\): \(r_L = \beta^{5-10}=\beta^{-5}\approx 1/0.556^5\approx 18.9\) (enough: \(>9\)).

So \(g_{\min}=5\): accepted sequences must have all five tags be “good”, which makes the *first* completion token almost surely good under the conditional distribution.

2) If the prompt leaves \(\pi_A(\text{prompt})=0.95\), then \(r_0=0.95/0.05=19\). Now:

- For \(g=1\): \(r_L = 19\cdot \beta^{5-2} = 19\cdot \beta^3 \approx 19\cdot 0.172 \approx 3.27\) (not enough).
- For \(g=2\): \(r_L = 19\cdot \beta^{5-4} = 19\cdot \beta \approx 19\cdot 0.556 \approx 10.6\) (enough).

So \(g_{\min}=2\): you can still be accepted even if the completion contains multiple bad tags. In this regime, conditioning on “final collapse to A” barely forces the first token to be good, so the boundary rate moves back toward the intrinsic ceiling \(\alpha/(\alpha+\beta)\approx 0.64\).

##### Rule of thumb for interpreting the finetune curves

- If prompts are near-balanced (π_A near 0.5), conditioning on \(E\) can make the first completion token strongly “good”.
- If prompts already heavily favor A (π_A close to 1), conditioning on \(E\) does not add much information at the boundary, so the best-achievable good-rate is close to \(\alpha/(\alpha+\beta)\).

---

## 5) Degeneracy Problem in Baseline AFP (Why Diversity Collapses)

Even though the completion vocab is 8, the completion distribution can be highly concentrated.

Two distinct mechanisms:

1. **Sector collapse:** after a few completion tokens, the belief mass concentrates into one sector, which drastically reduces the effective number of sequences that occur with non-negligible probability.
2. **Within-sector address space is small:** sector B is only 4 hidden states with near-deterministic transitions, so many of the `8^L` possible completion strings are effectively impossible or extremely unlikely.

Empirically (reproducing the notebook’s finetune data generation with defaults):

- Total finetune sequences: `2500`
- Unique completion sequences (length-5, tokens 0–7): `524`
- Most common completion: `(0,0,0,0,0)` appears very frequently

This is exactly the “degeneracy” issue: many prompts end up producing overlapping (or identical) high-probability completions.

For a finetune/holdout overlap proxy, degeneracy matters because:

- if completions overlap heavily across prompts, finetuning on 5% of prompts can still expose the model to a large fraction of completion space,
- making “heldout behavior shift” less diagnostic.

---

## 6) The Codebook Variant: What It Is and Why It Helps

The “codebook” process is implemented in `analysis/afp_builders.py` as `process_variant="clustered_codebook"` and used in `analysis/afp_steering_gpt.ipynb`.

### Design goal

Increase **prompt-specific within-sector completion diversity** while keeping the same high-level AFP/SFP story:

- prompt sets up an uncertain latent state (now: “which B-cluster?”),
- completion reveals/collapses to a consistent latent explanation,
- but different prompts preferentially lead to *different* completions.

### Hidden state layout

- Sector A: still 1 state.
- Sector B: expanded to `K * L` states, organized as:

  - `K` clusters (addresses)
  - `L` phases (position in the codeword)

So total hidden states:

- `1 + K*L`

### Prompt phase (cluster inference via “signatures”)

Each B-cluster `c` is assigned a small “signature set” of `r = PROMPT_SIGNATURE_SIZE` prompt tokens.

When you see a prompt token:

- clusters whose signature contains that token receive higher weight;
- other clusters receive lower weight.

After `PROMPT_LEN` prompt tokens, you get a posterior over clusters.

Important: this mapping is still “soft” (you can dial how sharp it is with `PROMPT_SIGNATURE_MASS`).

### Completion phase (codeword emission)

Each cluster `c` has a fixed length-`L` codeword over base symbols `{0,1,2,3}`.

- With `CODEBOOK_STYLE="base4"`, the codeword is literally the base-4 digits of `c`, so codewords are unique for `K <= 4^L`.

The completion HMM cycles through phases `t=0..L-1` and emits the corresponding base symbol for that phase (plus good/bad tagging structure).

This creates a much larger “address space” of distinct, high-probability completion sequences:

- up to `K` distinct codewords as MAP completions (in the idealized case),
- rather than a handful of completions driven by a tiny 4-state sector-B dynamics.

### Why it can reduce finetune/holdout overlap

If:

- prompts in your finetune subset tend to concentrate posterior mass on one subset of clusters,
- and held-out prompts concentrate on different clusters,

then the union of completion supports can have low overlap.

That’s closer to what you want for the emergent-misalignment proxy:

- finetune on “one sector/cluster family” completions
- test on unseen prompts and check whether the model’s completion bias shifts in the way you’d expect.

---

## 7) Practical Checklist: If Results Look “Wrong”

### “Finetune prompts look like an equal mix of π_A”

That is expected in `afp_steering.ipynb`: prompt selection is random.

If you want “only low post-prompt π_A prompts”, you must change prompt selection to filter/sort by `prompt_to_pi_a`.

### “Finetuning data generation returns very few examples”

If you select prompts with π_A near 0 but still reject for “collapse to A”, acceptance probability can become tiny.

Fix by:

- finetuning on sector B instead, or
- accepting collapse to either sector, or
- lowering `SECTOR_THRESH`, or
- increasing `MAX_ATTEMPTS` / batch sizes.

### “Overlap metrics at TAU=0.1 are empty”

For longer completions, per-sequence probability masses become tiny (even if the process is structured).

Use:

- smaller `tau` (e.g. 0.01 / 0.001),
- or “top-K per prompt” supports instead of thresholded supports,
- or effective-number metrics (Simpson / Shannon).

---

## References (in-repo)

- Baseline AFP notebook: `analysis/afp_steering.ipynb`
- Codebook + variants notebook: `analysis/afp_steering_gpt.ipynb`
- Variant builders: `analysis/afp_builders.py`
- Underlying AFP matrices / generators: `training/matrices.py`
- Formal prompt/collapse math: `analysis/em_finetuning_writeup.md`

---

## Appendix: TikZ State Diagrams

These are *schematic* hidden-state diagrams intended for writeups. They do **not** try to draw a separate edge for every token-indexed transition matrix entry; instead they show:

- the macro-sector decomposition (A vs B),
- the within-B dynamics,
- where the prompt vs completion phases act.

### HackMD Rendering Note

HackMD typically **does not render TikZ**. If you want diagrams that render directly in HackMD, use the Mermaid versions below. Keep the TikZ snippets as “paper-ready source” you can render offline to SVG/PDF.

---

## Appendix: Mermaid Diagrams (HackMD-Friendly)

These Mermaid diagrams are also schematic; they’re meant to communicate structure, not exact token-conditioned matrices.

### A. Modified Z1R′×Z1R′ AFP (1⊕4 macro-sectors) — Mermaid

```mermaid
flowchart LR
  %% Macro sectors
  subgraph A[Sector A (1 state)]
    A0["(S0,S0)"]
  end

  subgraph B[Sector B (4 states)]
    B11["(S1,S1)"]
    BSR["(SR,SR)"]
    B1R["(S1,SR)"]
    BR1["(SR,S1)"]
  end

  %% Within-B deterministic skeleton (schematic)
  B11 <--> |"B-dynamics (schematic)"| BSR
  B1R <--> |"B-dynamics (schematic)"| BR1

  %% Prompt phase effect (schematic)
  A0 -.-> |"prompt tokens: reweight sector odds via c_A/c_B"| A0
  B11 -.-> |"prompt tokens: optional within-B permutations"| B1R
  B11 -.-> |"prompt tokens: optional within-B permutations"| BR1

  %% Completion phase effect (schematic)
  A0 -.-> |"completion tokens: good/bad tags scale A vs B (beta controls collapse)"| B11
```

Notes:

- The dashed edges are *informational*: in the actual construction, prompt/completion act by **belief updates** under token-indexed matrices rather than “literal arrows between sectors”.
- Collapse comes from repeated relative scaling of A- vs B-block likelihoods (the `beta` knob), not from adding explicit A↔B transitions.

### B. Clustered Codebook (Sector B = K clusters × L phases) — Mermaid

```mermaid
flowchart LR
  %% Sector A
  subgraph A[Sector A (1 state)]
    A0["A"]
  end

  %% One representative cluster with phases
  subgraph C[One cluster c with phase ring t=0..L-1]
    C0["B(c,t=0)"]
    C1["B(c,t=1)"]
    C2["B(c,t=2)"]
    C3["B(c,t=3)"]
    C0 --> |"emit codeword[c,0]"| C1
    C1 --> |"emit codeword[c,1]"| C2
    C2 --> |"emit codeword[c,2]"| C3
    C3 --> |"emit codeword[c,3] (wrap)"| C0
  end

  %% Other clusters (collapsed)
  subgraph Others[Other clusters c' (K-1 more)]
    O0["B(c',t=0)"]
  end

  %% Prompt-driven cluster inference (schematic)
  A0 -.-> |"prompt signatures bias posterior over clusters"| C0
  A0 -.-> |"prompt signatures bias posterior over clusters"| O0

  %% Small leak across clusters (schematic)
  C1 -.-> |"small cluster leak ε"| O0
```

Notes:

- The completion “address space” is large because different clusters correspond to different high-probability length-`L` codewords.
- Prompt tokens provide (soft) evidence about which cluster you’re in; completions then read out that cluster’s codeword.

### A. Modified Z1R′×Z1R′ AFP (1⊕4 macro-sectors)

```tex
% Schematic hidden-state graph for the AFP (Z1R' x Z1R') macro-sectors.
% A is 1D: (S0,S0). B is 4D: (S1,S1),(S1,SR),(SR,S1),(SR,SR).
%
% Completion tokens are "good-tagged" vs "bad-tagged":
% - good tag: scales A-block by alpha and B-block by beta
% - bad  tag: scales A-block by beta  and B-block by alpha
%
% Prompt tokens reweight sector odds via c_A/c_B and optionally permute within B.

\begin{tikzpicture}[
  >=Stealth,
  node distance=18mm,
  every node/.style={font=\small},
  state/.style={draw, rounded corners, align=center, inner sep=3pt},
  astate/.style={state, fill=blue!8, draw=blue!60},
  bstate/.style={state, fill=red!7,  draw=red!55},
  lab/.style={font=\footnotesize, align=center}
]

% --- Sector A (1 state) ---
\node[astate] (A) {Sector A\\$(S0,S0)$\\(1 state)};
\path[->] (A) edge[loop above] node[lab]{prompt: $c_A(p_k)$\\completion: emits $\{g/b\}\times\{A,B,C,D\}$\\(token-indexed)} (A);

% --- Sector B (4 states) ---
\node[bstate, right=40mm of A] (B11) {$(S1,S1)$};
\node[bstate, above right=18mm and 22mm of B11] (BSR) {$(SR,SR)$};
\node[bstate, below right=18mm and 22mm of B11] (B1R) {$(S1,SR)$};
\node[bstate, right=44mm of B11] (BR1) {$(SR,S1)$};

% Within-B deterministic skeleton (schematic)
\path[->] (B11) edge[bend left=12] node[lab]{B-dynamics\\(schematic)} (BSR);
\path[->] (BSR) edge[bend left=12] node[lab]{} (B11);
\path[->] (B1R) edge[bend left=12] node[lab]{} (BR1);
\path[->] (BR1) edge[bend left=12] node[lab]{} (B1R);

% Prompt mixing within B (schematic)
\draw[->, dashed, gray!80] (B11) to[bend left=10] node[lab, above]{prompt permutations\\within B (optional)} (B1R);
\draw[->, dashed, gray!80] (B11) to[bend left=10] node[lab, below]{} (BR1);

% Sector boundary note (no actual A<->B state transitions in the restricted construction)
\node[lab, below=14mm of A] (note) {Macro-sector posterior updates come from\\\emph{relative scaling} of A- vs B-blocks\\(collapse rate controlled by $\beta$).};

\end{tikzpicture}
```

### B. Codebook / Clustered Completion AFP (Sector B = K clusters × L phases)

```tex
% Schematic hidden-state graph for the clustered codebook variant.
% Sector A: 1 state. Sector B: (cluster c, phase t) with c in [1..K], t in [0..L-1].
% Completion emits a codeword symbol depending on (c,t); prompt biases posterior over c.

\begin{tikzpicture}[
  >=Stealth,
  node distance=16mm,
  every node/.style={font=\small},
  state/.style={draw, rounded corners, align=center, inner sep=3pt},
  astate/.style={state, fill=blue!8, draw=blue!60},
  bstate/.style={state, fill=red!7,  draw=red!55},
  lab/.style={font=\footnotesize, align=center}
]

% Sector A
\node[astate] (A) {Sector A\\(1 state)};
\path[->] (A) edge[loop above] node[lab]{prompt: $c_A(p_k)$\\completion: tagged base symbols} (A);

% One representative cluster with L phases (draw 4 phases explicitly, indicate ...)
\node[bstate, right=42mm of A] (c0) {$B(c,t{=}0)$};
\node[bstate, right=18mm of c0] (c1) {$B(c,t{=}1)$};
\node[bstate, right=18mm of c1] (c2) {$B(c,t{=}2)$};
\node[bstate, right=18mm of c2] (c3) {$B(c,t{=}3)$};
\node[lab, right=10mm of c3] (dots) {$\cdots$};

\path[->] (c0) edge node[lab]{emit codeword[$c,0$]} (c1);
\path[->] (c1) edge node[lab]{emit codeword[$c,1$]} (c2);
\path[->] (c2) edge node[lab]{emit codeword[$c,2$]} (c3);
\path[->] (c3) edge[bend left=15] node[lab]{emit codeword[$c,3$]\\(wrap to next phase)} (c0);

% Leak across clusters (schematic)
\draw[->, dashed, gray!80] (c1) to[bend left=12] node[lab, above]{small cluster leak $\varepsilon$\\to other $c'$} (c2);

% Prompt cluster inference (schematic)
\draw[->, dashed, gray!80] (A) to[bend left=10] node[lab, above]{prompt signatures bias\\posterior over clusters $c$} (c0);

% Indicate multiple clusters
\node[bstate, below=18mm of c0] (c0p) {$B(c',t{=}0)$};
\node[lab, below=5mm of c0p] (many) {$\vdots$ many clusters ($K$ total)};
\draw[->, dashed, gray!80] (c0p) to[bend left=10] node[lab, left]{similar phase ring} (c0p);

\end{tikzpicture}
```

Notes:

- In the codebook variant, “diversity” comes from a larger *address space* in sector B: different clusters correspond to different high-probability completion codewords.
- Prompt tokens are designed to give (soft) evidence about the cluster identity, rather than only mixing a tiny 4-state B sector.
