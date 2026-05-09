# Feature-Resolved Attention (FRA): General Theory and the Frozen-Attention Simplification

This note lays out two related objects:

1. the **general feature-resolved attention (FRA)** picture, where you open up both the QK circuit and the OV circuit, and  
2. the **simplified frozen-attention picture**, where you treat the attention pattern as empirically given and attribute a target direction at `hook_resid_mid` back to OV features and then to `resid_pre` features.

The main practical recommendation is:

- **If your immediate goal is to explain a direction or feature at `hook_resid_mid`, start by freezing the attention pattern and decomposing the OV path.**
- Only move to the full QK–OV three-way interaction object when you want to explain **why the model attended where it did**.

---

## 1. Scope and notation

Fix:

- a layer \(\ell\),
- a head \(h\),
- a query position \(q\),
- source positions \(k\),
- and a target quantity at `hook_resid_mid[\ell, q]`.

I will use the **TransformerLens row-vector convention**:
\[
\texttt{new\_activation} = \texttt{old\_activation} \; W + b.
\]

So all token representations are row vectors in \(\mathbb{R}^{d_{\text{model}}}\), and all weight matrices multiply on the right.

Let

- \(r^{\mathrm{pre}}_{\ell,t} \in \mathbb{R}^{d_{\text{model}}}\) be `blocks.ℓ.hook_resid_pre[t]`,
- \(r^{\mathrm{mid}}_{\ell,t} \in \mathbb{R}^{d_{\text{model}}}\) be `blocks.ℓ.hook_resid_mid[t]`,
- \(x_{\ell,t} \in \mathbb{R}^{d_{\text{model}}}\) be the vector actually read by the attention sublayer after LN1 and any LayerNorm-folding conventions.

In practice, \(x_{\ell,t}\) is the object you want to decompose on the attention input side. In TransformerLens, the relevant cached tensors are the normalized read vector and the LN scale factors; the exact semantics of `hook_normalized` depend on your normalization/folding choices, so it is safest to define \(x_{\ell,t}\) as **the exact tensor your model uses as input to Q/K/V after your chosen folding convention**.

For one head \(h\), define
\[
q_t^h = x_{\ell,t} W_Q^h,\qquad
k_t^h = x_{\ell,t} W_K^h,\qquad
v_t^h = x_{\ell,t} W_V^h + b_V^h.
\]

The pre-softmax score and attention pattern are
\[
s_{qk}^h = \frac{q_q^h \cdot k_k^h}{\sqrt{d_h}},
\qquad
A_{qk}^h = \mathrm{softmax}_k(s_{qk}^h).
\]

The head result is
\[
y_q^h
= \sum_k A_{qk}^h \, v_k^h W_O^h
= \sum_k A_{qk}^h \, x_{\ell,k} W_V^h W_O^h + c^h,
\]
where
\[
c^h := b_V^h W_O^h
\]
because \(\sum_k A_{qk}^h = 1\).

The full attention write at that layer and token is
\[
\mathrm{attn\_out}_{\ell,q}
= \sum_h y_q^h + b_O,
\]
and therefore
\[
r^{\mathrm{mid}}_{\ell,q}
=
r^{\mathrm{pre}}_{\ell,q}
+
\mathrm{attn\_out}_{\ell,q}.
\]

---

## 2. Feature decompositions

Assume you have an SAE or some other feature basis on the attention input tensor \(x_{\ell,t}\):
\[
x_{\ell,t}
=
\sum_\lambda u^\lambda_{\ell,t} f_\lambda
+
e^{\mathrm{norm}}_{\ell,t},
\]
where

- \(u^\lambda_{\ell,t}\) is the activation of feature \(\lambda\) at token \(t\),
- \(f_\lambda \in \mathbb{R}^{d_{\text{model}}}\) is the decoder direction for feature \(\lambda\),
- \(e^{\mathrm{norm}}_{\ell,t}\) is the reconstruction error.

Likewise, assume you have a feature decomposition on `resid_pre`:
\[
r^{\mathrm{pre}}_{\ell,t}
=
\sum_a z^a_{\ell,t} g_a
+
e^{\mathrm{pre}}_{\ell,t},
\]
where \(g_a\) are the `resid_pre` decoder directions.

It is useful to keep both error terms explicit, rather than silently dropping them.

---

## 3. The general FRA picture

### 3.1 OV-side decomposition is linear once the pattern is fixed

Given a fixed attention pattern \(A^h\), the OV side is linear in the source representation:
\[
y_q^h
=
\sum_k A_{qk}^h \, x_{\ell,k} W_{OV}^h + c^h,
\qquad
W_{OV}^h := W_V^h W_O^h.
\]

Substituting the feature decomposition of \(x_{\ell,k}\),
\[
y_q^h
=
\sum_{k,\lambda}
A_{qk}^h \, u^\lambda_{\ell,k} \, f_\lambda W_{OV}^h
+
\sum_k A_{qk}^h \, e^{\mathrm{norm}}_{\ell,k} W_{OV}^h
+
c^h.
\]

So the exact vector-valued OV contribution of feature \(\lambda\) at source token \(k\) through head \(h\) is
\[
Y^{OV}_{h,q,k,\lambda}
:=
A_{qk}^h \, u^\lambda_{\ell,k} \, f_\lambda W_{OV}^h.
\]

This is the cleanest place to begin.

---

### 3.2 QK-side decomposition should be done on **scores**, not on **patterns**

Now write the same feature decomposition on both query and key positions:
\[
x_{\ell,q}
=
\sum_\mu u^\mu_{\ell,q} f_\mu + e^{\mathrm{norm}}_{\ell,q},
\qquad
x_{\ell,k}
=
\sum_\nu u^\nu_{\ell,k} f_\nu + e^{\mathrm{norm}}_{\ell,k}.
\]

Then the pre-softmax score becomes
\[
s_{qk}^h
=
\frac{
(x_{\ell,q} W_Q^h)\cdot(x_{\ell,k} W_K^h)
}{\sqrt{d_h}}
=
\sum_{\mu,\nu} S_{qk}^{h,\mu\nu}
+
\text{bias/error terms},
\]
where the natural QK feature-pair contribution is
\[
S_{qk}^{h,\mu\nu}
:=
u^\mu_{\ell,q} \, u^\nu_{\ell,k}\,
\omega_{\mu\nu}^{h,QK},
\]
with
\[
\omega_{\mu\nu}^{h,QK}
:=
\frac{
(f_\mu W_Q^h)\cdot(f_\nu W_K^h)
}{\sqrt{d_h}}.
\]

This is the right primitive object on the attention-selection side.

The reason **not** to define an exact additive decomposition \(A_{qk}^h = \sum_{\mu,\nu} A_{qk}^{h,\mu\nu}\) is that
\[
A_{qk}^h
=
\frac{\exp(s_{qk}^h)}
{\sum_j \exp(s_{qj}^h)},
\]
and the softmax denominator couples all \((\mu,\nu)\) score terms across all source positions \(j\). So there is no canonical exact additive decomposition of the post-softmax pattern unless you choose a specific attribution convention.

So the conceptual summary is:

- **exact QK primitive:** \(S_{qk}^{h,\mu\nu}\),
- **exact OV primitive given pattern:** \(Y^{OV}_{h,q,k,\lambda}\),
- **full end-to-end three-way feature object:** requires a choice of linearization / attribution rule for the softmax.

---

### 3.3 Local triplet attribution to a target direction

Suppose the downstream target is a direction
\[
d \in \mathbb{R}^{d_{\text{model}}}
\]
at `hook_resid_mid[\ell,q]`.

Define the OV write of post-LN feature \(\lambda\) into that direction:
\[
m_{h,\lambda}(d)
:=
\langle f_\lambda W_{OV}^h,\; d\rangle.
\]

If you want to open up the QK circuit locally, use the softmax Jacobian
\[
\frac{\partial A_{qk}^h}{\partial s_{qj}^h}
=
A_{qk}^h \left( \mathbf{1}_{k=j} - A_{qj}^h \right).
\]

For one QK feature pair \((\mu,\nu)\), define
\[
S_{qj}^{h,\mu\nu}
=
u^\mu_{\ell,q} \, u^\nu_{\ell,j}\, \omega_{\mu\nu}^{h,QK}.
\]

Then the induced first-order change in the pattern is
\[
\delta A_{qk}^{h,\mu\nu}
\approx
A_{qk}^h
\left(
S_{qk}^{h,\mu\nu}
-
\sum_j A_{qj}^h S_{qj}^{h,\mu\nu}
\right).
\]

This yields the natural local triplet attribution
\[
C_{h,q,k}^{\mu\nu\lambda}(d)
:=
\delta A_{qk}^{h,\mu\nu}\;
u^\lambda_{\ell,k}\;
m_{h,\lambda}(d).
\]

This is the object that says:

- query feature \(\mu\) at token \(q\),
- interacting with key feature \(\nu\) at token \(k\) (and more generally across the softmax competition over all \(j\)),
- gates source value feature \(\lambda\) at token \(k\),
- and writes into target direction \(d\).

Two caveats matter here:

1. This is **local** because it linearizes the softmax around the observed prompt.  
2. It still carries the source-position index \(k\); you only get a pure \((\mu,\nu,\lambda)\) tensor after summing over \(k\).

This is the sense in which the full FRA object is really a **three-way interaction mediated by attention selection**, but only after you choose an attribution convention for the softmax.

---

## 4. The simplification: freeze attention and explain `hook_resid_mid`

For many projects, the next useful step is **not** the full triplet tensor. It is:

> Hold the observed attention pattern fixed, and attribute a target scalar at `hook_resid_mid` back through the OV path.

This is exactly the regime where the path becomes linear and easy to analyze.

### 4.1 Choose a scalar target first

Do **not** start by trying to explain the entire residual vector. Start with a scalar.

Two common choices are:

#### A. A target residual-stream direction
Let
\[
s_d
:=
\langle r^{\mathrm{mid}}_{\ell,q},\; d\rangle.
\]

#### B. A target SAE feature at `hook_resid_mid`
If \(\tau\) is a feature on `hook_resid_mid` with encoder vector \(e_\tau\), then the feature preactivation is
\[
s_\tau
:=
\langle r^{\mathrm{mid}}_{\ell,q},\; e_\tau\rangle + b_\tau.
\]

In that case, all formulas below work by setting
\[
d = e_\tau.
\]

If instead you want “write into the feature’s decoder direction”, use \(d=f_\tau\). These are different questions.

---

### 4.2 Exact split into skip path and attention path

Because
\[
r^{\mathrm{mid}}_{\ell,q}
=
r^{\mathrm{pre}}_{\ell,q}
+
\mathrm{attn\_out}_{\ell,q},
\]
we have the exact scalar split
\[
s_d
=
\underbrace{\langle r^{\mathrm{pre}}_{\ell,q}, d\rangle}_{\text{skip path}}
+
\underbrace{\langle \mathrm{attn\_out}_{\ell,q}, d\rangle}_{\text{attention path}}.
\]

This is the first decomposition to compute.

---

### 4.3 OV-feature attribution with the attention pattern frozen

Now substitute the post-LN feature decomposition into the attention path:
\[
\langle \mathrm{attn\_out}_{\ell,q}, d\rangle
=
\sum_{h,k,\lambda}
A_{qk}^h \, u^\lambda_{\ell,k}\,
\langle f_\lambda W_{OV}^h,\; d\rangle
+
\text{bias terms}
+
\text{SAE reconstruction error}.
\]

Define the scalar OV contribution
\[
C^{OV}_{h,k,\lambda}(d)
:=
A_{qk}^h \, u^\lambda_{\ell,k}\,
\langle f_\lambda W_{OV}^h,\; d\rangle.
\]

This is the main object for the simplified analysis.

It is exact **given**:

1. the observed attention pattern is treated as fixed,
2. the chosen post-LN feature decomposition of \(x_{\ell,k}\).

A useful consistency check is
\[
\sum_{h,k,\lambda} C^{OV}_{h,k,\lambda}(d)
\approx
\langle \mathrm{attn\_out}_{\ell,q}, d\rangle
-
\text{bias terms}
-
\text{feature reconstruction error}.
\]

So, under frozen attention, the contribution of each value-side feature is completely straightforward.

---

## 5. Pulling the simplified attribution back to `resid_pre`

Now suppose you want to answer the stronger question:

> Which `resid_pre` features ultimately produced this `hook_resid_mid` direction, either directly through the residual stream or indirectly through attention?

This is where LayerNorm enters.

### 5.1 Prompt-local linearization of LN

Let the `resid_pre` decomposition be
\[
r^{\mathrm{pre}}_{\ell,k}
=
\sum_a z^a_{\ell,k} g_a
+
e^{\mathrm{pre}}_{\ell,k}.
\]

To map a `resid_pre` feature into the attention read tensor \(x_{\ell,k}\), define a prompt-local linear operator
\[
M_{\ell,k}
\]
such that, under the frozen-normalization approximation,
\[
x_{\ell,k}
\approx
r^{\mathrm{pre}}_{\ell,k} M_{\ell,k}.
\]

For a standard LayerNorm with **frozen denominator** and **folded affine parameters**, the natural approximation is
\[
M_{\ell,k}
=
\frac{1}{\sigma_{\ell,k}}
\left(
I - \frac{11^\top}{d_{\text{model}}}
\right),
\]
where \(\sigma_{\ell,k}\) is the cached LN scale for that token.

If your hook includes learned gain \(\gamma\) rather than folding it into later weights, then use
\[
M_{\ell,k}
=
\frac{1}{\sigma_{\ell,k}}
\left(
I - \frac{11^\top}{d_{\text{model}}}
\right)
\mathrm{Diag}(\gamma).
\]

If you are using RMSNorm rather than centered LayerNorm, replace the centering projector by the appropriate RMSNorm map.

The important conceptual point is:

- for the **observed prompt**, use the **cached normalization denominator / scale**,
- do **not** recompute LN statistics separately for each component if your aim is to explain the actual forward pass.

---

### 5.2 `resid_pre` features contributing through attention

Under the frozen-LN approximation,
\[
x_{\ell,k}
\approx
\sum_a z^a_{\ell,k}\, g_a M_{\ell,k}
+
e^{\mathrm{pre}\to\mathrm{norm}}_{\ell,k}.
\]

Substituting this into the OV expression gives the contribution of `resid_pre` feature \(a\) at source token \(k\) through head \(h\):
\[
C^{\mathrm{pre,attn}}_{h,k,a}(d)
:=
A_{qk}^h\,
z^a_{\ell,k}\,
\langle g_a M_{\ell,k} W_{OV}^h,\; d\rangle.
\]

This is the clean pullback of the OV explanation to `resid_pre` features.

---

### 5.3 `resid_pre` features contributing directly through the skip path

Because
\[
r^{\mathrm{mid}}_{\ell,q}
=
r^{\mathrm{pre}}_{\ell,q}
+
\mathrm{attn\_out}_{\ell,q},
\]
the direct skip contribution of `resid_pre` feature \(a\) at the target token \(q\) is simply
\[
C^{\mathrm{pre,skip}}_{q,a}(d)
:=
z^a_{\ell,q}\,
\langle g_a,\; d\rangle.
\]

So the full target scalar can be written as
\[
s_d
=
\sum_a C^{\mathrm{pre,skip}}_{q,a}(d)
+
\sum_{h,k,a} C^{\mathrm{pre,attn}}_{h,k,a}(d)
+
\text{bias terms}
+
\text{feature reconstruction errors}
+
\text{LN linearization error}.
\]

This is the main formula you want if your research question is:

> How was this direction at `hook_resid_mid[\ell,q]` composed from `resid_pre` features and post-LN / OV features?

---

## 6. Optional two-stage path tensor: `resid_pre feature -> post-LN feature -> OV -> target`

Sometimes you want to preserve the intermediate post-LN feature identity \(\lambda\), instead of collapsing directly from `resid_pre` to the OV write.

Let \(e_\lambda\) be the encoder vector for the SAE on the attention read tensor \(x\). Then the post-LN feature preactivation is
\[
\tilde u^\lambda_{\ell,k}
=
\langle x_{\ell,k}, e_\lambda\rangle + b_\lambda.
\]

Define the prompt-local transfer coefficient
\[
T_{k,\lambda a}
:=
\langle g_a M_{\ell,k},\; e_\lambda\rangle.
\]

This gives the amount that `resid_pre` feature \(a\) would contribute to the **preactivation** of post-LN feature \(\lambda\) at token \(k\), under the frozen-LN approximation.

Then the path contribution through both feature spaces is
\[
C_{h,k,a\to\lambda\to d}
=
A_{qk}^h\,
z^a_{\ell,k}\,
T_{k,\lambda a}\,
\langle f_\lambda W_{OV}^h,\; d\rangle.
\]

This object is very useful when you want a statement like:

> `resid_pre` feature \(a\) at token \(k\) became post-LN feature \(\lambda\), which was then written by head \(h\) into the target direction \(d\).

Two caveats:

1. This is exact for **feature preactivations** under the frozen-LN approximation.  
2. If your SAE activation is nonlinear (for example, ReLU or top-k) and you want the contribution to the **actual feature activation** rather than the preactivation, then you either need to freeze the gate or accept an additional local linearization.

---

## 7. Where the full FRA tensor becomes necessary

The frozen-attention analysis deliberately avoids the QK circuit.

It answers:

- Which OV features wrote into the target?
- Which `resid_pre` features caused those writes, once LN is linearized locally?

It does **not** answer:

- Why did head \(h\) attend to source position \(k\) in the first place?
- Which query/key feature interactions selected that source?

To answer that second class of questions, you need to reopen the QK circuit, and then the natural object is the feature interaction on scores
\[
S_{qk}^{h,\mu\nu},
\]
composed with the OV-side write into the target direction. That is where the local triplet tensor
\[
C_{h,q,k}^{\mu\nu\lambda}(d)
\]
enters.

So the right hierarchy is:

1. **Start with frozen attention** and compute \(C^{OV}_{h,k,\lambda}(d)\).  
2. Pull back to `resid_pre` using \(C^{\mathrm{pre,attn}}_{h,k,a}(d)\).  
3. Only then open up QK if you need a causal explanation of the observed pattern.

---

## 8. Practical implementation recipe

### 8.1 Minimal analysis path

For a chosen layer \(\ell\), target token \(q\), and target scalar \(s_d\):

1. **Choose the target**
   - either a direction \(d\),
   - or a feature preactivation at `hook_resid_mid`, in which case \(d=e_\tau\).

2. **Compute the exact top-level split**
   \[
   s_d
   =
   \langle r^{\mathrm{pre}}_{\ell,q}, d\rangle
   +
   \langle \mathrm{attn\_out}_{\ell,q}, d\rangle.
   \]

3. **Decompose the attention write into OV features**
   \[
   C^{OV}_{h,k,\lambda}(d)
   =
   A_{qk}^h u^\lambda_{\ell,k}\langle f_\lambda W_{OV}^h,d\rangle.
   \]

4. **Rank heads, source positions, and features**
   - by signed contribution,
   - and by absolute contribution.

5. **Pull the explanation back to `resid_pre`**
   \[
   C^{\mathrm{pre,attn}}_{h,k,a}(d)
   =
   A_{qk}^h z^a_{\ell,k}\langle g_a M_{\ell,k}W_{OV}^h,d\rangle.
   \]

6. **Add the direct skip contributions**
   \[
   C^{\mathrm{pre,skip}}_{q,a}(d)
   =
   z^a_{\ell,q}\langle g_a,d\rangle.
   \]

7. **Optionally preserve the intermediate post-LN feature identity**
   by computing
   \[
   C_{h,k,a\to\lambda\to d}.
   \]

This gives a step-by-step causal story without opening the QK circuit.

---

### 8.2 Tensorized recipe

Suppose at one layer you have:

- `pattern[h, q, k] = A^h_{qk}`,
- `u_norm[k, λ] = u^\lambda_{\ell,k}`,
- `F[λ, d_model] = f_\lambda`,
- `G[a, d_model] = g_a`,
- `target[d_model] = d`,
- `W_OV[h, d_model, d_model] = W_V^h W_O^h`,
- `M[k, d_model, d_model]` = prompt-local LN map.

Then:

#### OV feature writes into the target scalar
\[
\texttt{ov\_score}[h,\lambda]
=
\langle f_\lambda W_{OV}^h,\; d\rangle
\]
and
\[
C^{OV}[h,k,\lambda]
=
\texttt{pattern}[h,q,k]\;
\texttt{u\_norm}[k,\lambda]\;
\texttt{ov\_score}[h,\lambda].
\]

#### `resid_pre` features writing through attention
\[
\texttt{pre\_score}[h,k,a]
=
\langle g_a M[k] W_{OV}^h,\; d\rangle
\]
and
\[
C^{\mathrm{pre,attn}}[h,k,a]
=
\texttt{pattern}[h,q,k]\;
\texttt{z\_pre}[k,a]\;
\texttt{pre\_score}[h,k,a].
\]

#### Skip contributions
\[
C^{\mathrm{pre,skip}}[a]
=
\texttt{z\_pre}[q,a]\;
\langle g_a,d\rangle.
\]

You never need to materialize the full \(d_{\text{model}}\times d_{\text{model}}\) matrix \(M[k]\) if your normalization is standard LN or RMSNorm; you can apply centering and scaling directly to \(g_a\).

---

## 9. What is exact, and what is approximate?

It helps to separate these clearly.

### Exact, given a fixed pattern
The following object is exact once the attention pattern is treated as observed:
\[
C^{OV}_{h,k,\lambda}(d)
=
A_{qk}^h u^\lambda_{\ell,k}\langle f_\lambda W_{OV}^h,d\rangle.
\]

### Exact in the frozen-LN local model
The following is exact in the prompt-local model where the LN denominator is frozen:
\[
C^{\mathrm{pre,attn}}_{h,k,a}(d)
=
A_{qk}^h z^a_{\ell,k}\langle g_a M_{\ell,k}W_{OV}^h,d\rangle.
\]

### Additional approximation if you preserve an intermediate feature node
The path object
\[
C_{h,k,a\to\lambda\to d}
\]
adds an extra approximation if you want contributions to the **activated** post-LN feature rather than its preactivation.

### Local approximation when QK is opened
The triplet object
\[
C_{h,q,k}^{\mu\nu\lambda}(d)
\]
is only local if you use the softmax Jacobian. For larger perturbations, integrated gradients or explicit interventions are better.

---

## 10. Recommended workflow for your project

If the goal is to trace a feature direction at `hook_resid_mid` back to both post-LN features and `resid_pre` features, the order I would actually use is:

1. **Pick one scalar target** at `hook_resid_mid[\ell,q]`.  
2. **Compute the exact skip-vs-attention split.**  
3. **Compute \(C^{OV}_{h,k,\lambda}(d)\)** and inspect the top heads, source tokens, and post-LN features.  
4. **Compute \(C^{\mathrm{pre,attn}}_{h,k,a}(d)\)** and inspect the top source-side `resid_pre` features.  
5. **Only for the top contributors**, expand to the two-stage path \(a \to \lambda \to d\).  
6. **Only if needed**, reopen QK and compute the local triplet tensor \(C_{h,q,k}^{\mu\nu\lambda}(d)\).

This avoids paying the full complexity cost of FRA before you know which heads, source tokens, and value-side features matter.

---

## 11. Interpretation summary

The big picture is:

- The **value side** naturally decomposes feature-by-feature once the pattern is fixed.
- The **query/key side** naturally decomposes pair-by-pair on the **pre-softmax scores**.
- The full end-to-end explanation is therefore not “just a feature decomposition”; it is a **QK pair selection mechanism composed with a V feature write mechanism**.
- If your target is a single direction or feature at `hook_resid_mid`, the frozen-attention simplification is usually the right first move.

In that simplified setting, the main attribution objects are:
\[
C^{OV}_{h,k,\lambda}(d),
\qquad
C^{\mathrm{pre,attn}}_{h,k,a}(d),
\qquad
C^{\mathrm{pre,skip}}_{q,a}(d).
\]

These already give a detailed circuit-level story of how a `hook_resid_mid` direction was composed from:

- source-side post-LN features read by attention,
- source-side `resid_pre` features mapped through LN,
- and the target token’s direct residual-stream carry.

---

## References

1. **TransformerLens ActivationCache documentation.**  
   Documents the cached tensors `pattern` (post-softmax), `attn_scores` (pre-softmax), `resid_pre`, `resid_mid`, `normalized`, and `scale`, and explains that `apply_ln_to_stack` uses the cached scale factors to simulate a component’s contribution to a layer input.  
   https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.ActivationCache.html

2. **TransformerLens Main Demo Notebook.**  
   States the row-vector convention (`new_activation = old_activation @ weights + bias`) and explains LayerNorm folding and the role of `hook_scale`.  
   https://transformerlensorg.github.io/TransformerLens/generated/demos/Main_Demo.html

3. **Circuit Tracing: Revealing Computational Graphs in Language Models (Methods).**  
   Explains the attribution-graph setting where attention patterns and normalization denominators are frozen so that source-to-target effects along OV/residual paths become linear “virtual weights”.  
   https://transformer-circuits.pub/2025/attribution-graphs/methods.html

4. **Tracing Attention Computation Through Feature Interactions.**  
   Describes a method for explaining attention patterns in terms of feature interactions and integrating that into attribution-style analyses, which is close in spirit to the full FRA picture.  
   https://transformer-circuits.pub/2025/attention-qk/index.html
