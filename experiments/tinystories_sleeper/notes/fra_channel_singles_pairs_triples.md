
# Channel-Wise Attribution in Feature-Resolved Attention (FRA)
## Exact singles, pairs, and triples over the Q / K / V channels

This note gives a clean way to measure **single-channel**, **pairwise**, and **triple** contributions in feature-resolved attention (FRA).

The core move is:

1. choose a **scalar target** to explain,
2. treat the three FRA channels \(\{Q,K,V\}\) as the players in a **3-player local coalition game**,
3. decompose the resulting 8 coalition values with the **Möbius / Harsanyi transform**.

That yields:

- exact **single-channel** terms: \(Q\), \(K\), \(V\),
- exact **pair interactions**: \(QK\), \(QV\), \(KV\),
- an exact **irreducible triple**: \(QKV\).

This is the cleanest abstraction I know for “how much is single / double / triple” in FRA.

---

## 1. Setup

Fix:

- a layer \(\ell\),
- a head \(h\),
- a query token \(q\),
- a source token \(k\),
- a query-side feature \(\mu\),
- a key-side feature \(\nu\),
- a value-side feature \(\lambda\),
- and a target direction \(d \in \mathbb{R}^{d_{\text{model}}}\) at `hook_resid_mid[\ell, q]`.

I will use row-vector notation. For the selected head \(h\),

\[
q_t = x_t W_Q^h,\qquad
k_t = x_t W_K^h,\qquad
w_t = x_t W_V^h W_O^h.
\]

Here \(x_t\) is the head input (typically the post-LN attention input, e.g. `ln1.hook_normalized` under your chosen folding convention).

The **path-local contribution** from source token \(k\) to query token \(q\) through this head is

\[
y_{q \leftarrow k} = A_{qk}\, w_k,
\]

and the scalar target we will explain is

\[
T := \langle y_{q \leftarrow k}, d \rangle.
\]

Why force the target to be scalar?

Because “single / pair / triple” is well defined for a scalar set function.  
If you keep the target vector-valued, there is no unique notion of interaction order unless you choose a norm or projection.  
So in practice:

- if you want to explain a direction at `hook_resid_mid`, set \(d\) to that direction;
- if you want to explain a feature preactivation, set \(d\) to the encoder direction for that feature.

---

## 2. The three players: channel-specific feature inserts

For the chosen FRA triplet \((\mu,\nu,\lambda)\), define the selected channel contributions

\[
\delta q^\mu := u_q^\mu\, f_\mu W_Q^h,
\]
\[
\delta k^\nu := u_k^\nu\, f_\nu W_K^h,
\]
\[
\delta w^\lambda := u_k^\lambda\, f_\lambda W_V^h W_O^h.
\]

These are the **channel-specific** contributions of the selected feature to the chosen head.

Now define background states with the selected pieces removed:

\[
q^{bg} := q_q - \delta q^\mu,
\]
\[
k_j^{bg} :=
\begin{cases}
k_k - \delta k^\nu & j=k,\\
k_j & j\neq k,
\end{cases}
\]
\[
w_j^{bg} :=
\begin{cases}
w_k - \delta w^\lambda & j=k,\\
w_j & j\neq k.
\end{cases}
\]

Introduce binary gates

\[
z_Q, z_K, z_V \in \{0,1\}.
\]

Then the gated states are

\[
q(z_Q)=q^{bg}+z_Q\,\delta q^\mu,
\]
\[
k_j(z_K)=k_j^{bg}+\mathbf 1_{j=k}\,z_K\,\delta k^\nu,
\]
\[
w_j(z_V)=w_j^{bg}+\mathbf 1_{j=k}\,z_V\,\delta w^\lambda.
\]

Recompute the head-local score vector and pattern:

\[
s_j(z_Q,z_K)=\frac{q(z_Q)\cdot k_j(z_K)}{\sqrt{d_h}},
\qquad
A_j(z_Q,z_K)=\operatorname{softmax}_j s(z_Q,z_K).
\]

Finally define the **local coalition game**

\[
g(z_Q,z_K,z_V)
:=
\left\langle
A_k(z_Q,z_K)\,w_k(z_V) - A_k(0,0)\,w_k(0),
\, d
\right\rangle.
\]

This is the baseline-subtracted effect of turning on any subset of the selected \(Q\), \(K\), and \(V\) channel pieces.

A useful shorthand is to label coalitions by subsets:
- \(\emptyset = (0,0,0)\),
- \(Q = (1,0,0)\),
- \(K = (0,1,0)\),
- \(V = (0,0,1)\),
- \(QK = (1,1,0)\), etc.

Since we baseline-subtracted, \(g(\emptyset)=0\).

---

## 3. The exact single / pair / triple decomposition

For a general set function \(g : 2^{\{Q,K,V\}} \to \mathbb{R}\), the **Möbius / Harsanyi coefficient** of subset \(S\) is

\[
\Delta_S
=
\sum_{T \subseteq S}
(-1)^{|S|-|T|}
\, g(T).
\]

For three players this becomes:

### Singles

\[
\Delta_Q = g(Q),
\qquad
\Delta_K = g(K),
\qquad
\Delta_V = g(V).
\]

### Pairs

\[
\Delta_{QK} = g(QK)-g(Q)-g(K),
\]
\[
\Delta_{QV} = g(QV)-g(Q)-g(V),
\]
\[
\Delta_{KV} = g(KV)-g(K)-g(V).
\]

### Triple

\[
\Delta_{QKV}
=
g(QKV)-g(QK)-g(QV)-g(KV)+g(Q)+g(K)+g(V).
\]

And the decomposition is exact:

\[
g(QKV)
=
\Delta_Q+\Delta_K+\Delta_V
+\Delta_{QK}+\Delta_{QV}+\Delta_{KV}
+\Delta_{QKV}.
\]

Interpretation:

- \(\Delta_Q,\Delta_K,\Delta_V\): main effects,
- \(\Delta_{QK},\Delta_{QV},\Delta_{KV}\): pure pairwise interactions,
- \(\Delta_{QKV}\): the irreducible three-way FRA interaction.

Positive values mean synergy.  
Negative values mean redundancy or antagonism.

---

## 4. Why this is principled

This decomposition is principled because it is:

- **exact** for the chosen local game,
- **efficient**: the terms sum exactly to the total coalition effect,
- **order-separated**: pair terms exclude singles, triple excludes singles and pairs,
- **agnostic to mechanism details**: once you define the coalition game, the decomposition is canonical.

It is also the most natural local “pure interaction” decomposition if your conceptual object is a 3-channel FRA triplet.

If later you want a **redistributed** interaction score rather than a pure order decomposition, then Shapley-Taylor is the standard axiomatic family for distributing interaction mass across lower-order subsets, and Integrated Hessians is the common gradient-path approximation for pairwise interactions in differentiable models.  
For the exact 3-channel FRA problem, though, the discrete 8-corner decomposition is cleaner.  
See the references at the end for background.

---

## 5. Structural simplification of the score function

The previous definition is exact, but it helps to expand the score algebra.

For \(j \neq k\),

\[
s_j(z_Q,z_K)
=
\frac{(q^{bg}+z_Q \delta q^\mu)\cdot k_j^{bg}}{\sqrt{d_h}}
=
s_j^{bg}+z_Q a_j,
\]

where

\[
s_j^{bg} := \frac{q^{bg}\cdot k_j^{bg}}{\sqrt{d_h}},
\qquad
a_j := \frac{\delta q^\mu \cdot k_j^{bg}}{\sqrt{d_h}}.
\]

For the selected source token \(k\),

\[
s_k(z_Q,z_K)
=
\frac{(q^{bg}+z_Q \delta q^\mu)\cdot (k_k^{bg}+z_K \delta k^\nu)}{\sqrt{d_h}}
\]

so

\[
s_k(z_Q,z_K)
=
s_k^{bg}
+ z_Q a_k
+ z_K b
+ z_Q z_K c,
\]

with

\[
b := \frac{q^{bg}\cdot \delta k^\nu}{\sqrt{d_h}},
\qquad
c := \frac{\delta q^\mu \cdot \delta k^\nu}{\sqrt{d_h}}.
\]

So even before the softmax, the selected score contains an explicit **bilinear \(QK\) term** \(z_Q z_K c\).

On the value side, define

\[
v_{bg} := \langle w_k^{bg}, d \rangle,
\qquad
v_\lambda := \langle \delta w^\lambda, d \rangle.
\]

Then the local game becomes

\[
g(z_Q,z_K,z_V)
=
A_k(z_Q,z_K)\, \big(v_{bg}+z_V v_\lambda\big)
-
A_k(0,0)\,v_{bg}.
\]

This form is extremely useful:

- all \(Q\) and \(K\) structure lives in \(A_k(z_Q,z_K)\),
- all \(V\) structure lives in \(v_{bg}+z_V v_\lambda\),
- the triple interaction comes from **selection** (\(QK\)) interacting with **write** (\(V\)).

---

## 6. Two game conventions you can choose

There is one important semantic choice.

### 6.1 Contextual incremental game

This is the game defined above: you remove only the selected channel pieces and leave all other clean-run content in place.

Then:

- \(\Delta_Q\) can be nonzero because the selected query feature can reroute **background** write from token \(k\),
- \(\Delta_K\) can be nonzero because the selected key feature can make token \(k\) more or less addressable to the **background** query,
- \(\Delta_V\) can be nonzero because the selected value feature can be written under the **background** attention pattern.

This answers:

> What is the incremental effect of this FRA triplet **in the actual clean context**?

This is usually the most useful choice for mechanistic work.

---

### 6.2 Strict triplet game

Sometimes you want the selected triplet and only the selected triplet.

The easiest way is to zero the source-path background write:

\[
w_k^{bg}=0
\quad\Longrightarrow\quad
v_{bg}=0.
\]

Then

\[
g(z_Q,z_K,z_V)=A_k(z_Q,z_K)\, z_V\, v_\lambda.
\]

Immediate consequence:

\[
\Delta_Q = \Delta_K = \Delta_{QK} = 0.
\]

So the only nonzero terms are

- \(V\),
- \(QV\),
- \(KV\),
- \(QKV\).

This answers:

> How much of the effect comes from the selected value write itself, versus routing-write interactions involving the selected \(Q\) and \(K\) pieces?

This is a “purer” triplet-isolation game, but it is less faithful to the clean-run context.

---

## 7. Worked example A: contextual incremental game

Here is a tiny scalar-head toy model.

We study source token \(k=1\), with one competitor token \(j=2\).

Take:

\[
q^{bg}=1.0,\quad
k_1^{bg}=0.3,\quad
k_2^{bg}=0.1,
\]
\[
\delta q=0.8,\quad
\delta k=1.0,
\]
\[
w_1^{bg}=0.2,\quad
\delta w=1.5,
\]
and choose \(d=1\), so the target scalar is just the path value itself.

Then
\[
A_1(z_Q,z_K)=\operatorname{softmax}\!\big(s_1(z_Q,z_K),\, s_2(z_Q,z_K)\big)_1
\]
with
\[
s_1=(q^{bg}+z_Q\delta q)(k_1^{bg}+z_K\delta k),
\qquad
s_2=(q^{bg}+z_Q\delta q)k_2^{bg},
\]
and
\[
g(z_Q,z_K,z_V)=A_1(z_Q,z_K)\,(w_1^{bg}+z_V\delta w)-A_1(0,0)w_1^{bg}.
\]

### 7.1 The 8 coalition values

| \(z_Q\) | \(z_K\) | \(z_V\) | \(A_1\) | \(g(z_Q,z_K,z_V)\) |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0.549834 | 0.000000 |
| 0 | 0 | 1 | 0.549834 | 0.824751 |
| 0 | 1 | 0 | 0.768525 | 0.043738 |
| 0 | 1 | 1 | 0.768525 | 1.196525 |
| 1 | 0 | 0 | 0.589040 | 0.007841 |
| 1 | 0 | 1 | 0.589040 | 0.891402 |
| 1 | 1 | 0 | 0.896600 | 0.069353 |
| 1 | 1 | 1 | 0.896600 | 1.414252 |

### 7.2 Exact decomposition

Singles:

\[
\Delta_Q = 0.007841,
\qquad
\Delta_K = 0.043738,
\qquad
\Delta_V = 0.824751.
\]

Pairs:

\[
\Delta_{QK} = 0.017774,
\qquad
\Delta_{QV} = 0.058810,
\qquad
\Delta_{KV} = 0.328036.
\]

Triple:

\[
\Delta_{QKV} = 0.133302.
\]

Check:

\[
0.007841 + 0.043738 + 0.824751 + 0.017774 + 0.058810 + 0.328036 + 0.133302
= 1.414252
= g(QKV).
\]

### 7.3 Interpretation

This example is a nice sanity check.

- \(V\) is the biggest single because the value feature already writes strongly under the background attention pattern.
- \(K\) is bigger than \(Q\) because changing the key at the selected source makes that source much more addressable.
- \(KV\) is the biggest pair interaction: once the key makes token \(k\) more attendable, the value feature suddenly matters much more.
- The triple is positive: the query feature further amplifies the key-plus-value effect.

This is exactly the kind of story you want from FRA:
selection (\(QK\)) and write (\(V\)) separated into singles, pairs, and a genuine three-way residue.

---

## 8. Worked example B: strict triplet game

Now keep everything the same, but set

\[
w_1^{bg}=0.
\]

Then the background write from this source-path vanishes, so

\[
g(z_Q,z_K,z_V)=A_1(z_Q,z_K)\, z_V \delta w.
\]

The 8 coalition values are now:

| \(z_Q\) | \(z_K\) | \(z_V\) | \(A_1\) | \(g(z_Q,z_K,z_V)\) |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0.549834 | 0.000000 |
| 0 | 0 | 1 | 0.549834 | 0.824751 |
| 0 | 1 | 0 | 0.768525 | 0.000000 |
| 0 | 1 | 1 | 0.768525 | 1.152787 |
| 1 | 0 | 0 | 0.589040 | 0.000000 |
| 1 | 0 | 1 | 0.589040 | 0.883561 |
| 1 | 1 | 0 | 0.896600 | 0.000000 |
| 1 | 1 | 1 | 0.896600 | 1.344899 |

The exact decomposition becomes

\[
\Delta_Q = 0,
\qquad
\Delta_K = 0,
\qquad
\Delta_V = 0.824751,
\]
\[
\Delta_{QK} = 0,
\qquad
\Delta_{QV} = 0.058810,
\qquad
\Delta_{KV} = 0.328036,
\]
\[
\Delta_{QKV} = 0.133302.
\]

Interpretation:

- there is no write without \(V\),
- so \(Q\) and \(K\) by themselves have zero effect,
- but they interact with \(V\) by changing how strongly the selected value write is routed.

This is often the cleanest “pure triplet” view.

---

## 9. What changes under the frozen-attention simplification?

If you freeze the clean attention pattern \(A_k^{clean}\), then

\[
A_k(z_Q,z_K) \equiv A_k^{clean}.
\]

So the coalition game reduces to

\[
g(z_Q,z_K,z_V)
=
A_k^{clean}\,(v_{bg}+z_V v_\lambda)-A_k^{clean}v_{bg}
=
A_k^{clean}\, z_V v_\lambda.
\]

Therefore,

\[
\Delta_Q=\Delta_K=\Delta_{QK}=\Delta_{QV}=\Delta_{KV}=\Delta_{QKV}=0,
\]
and only

\[
\Delta_V = A_k^{clean}\, v_\lambda
\]

survives.

So if you are in the earlier “OV with empirically given pattern” regime, the channel game degenerates to a one-player \(V\)-only game.

This is an important sanity check:
- **channel singles / pairs / triples only become interesting when you unfreeze QK.**

---

## 10. Full-output game vs path-local game

Everything above used the **path-local** value

\[
y_{q\leftarrow k}=A_{qk} w_k.
\]

That is usually the best choice if you want to study a specific FRA path.

But you can also define a **full-output** coalition game

\[
g_{full}(z_Q,z_K,z_V)
=
\left\langle
\sum_j A_j(z_Q,z_K)\, w_j(z_V)
-
\sum_j A_j(0,0)\, w_j(0),
\, d
\right\rangle.
\]

Difference in meaning:

- **path-local game**: “what does this source path \(q \leftarrow k\) contribute?”
- **full-output game**: “what does this selected triplet do to the whole head output into \(d\)?”

The full-output game includes side effects where \(Q\) or \(K\) changes attention to other source positions too.

For most FRA analysis, I would start with the **path-local game** and use the full-output game only if you explicitly want competition effects.

---

## 11. Efficient implementation recipe

If the target is at the same `hook_resid_mid` location, you do **not** need to rerun the whole model for each coalition.  
You can compute the 8 values from cached head-local tensors.

### 11.1 Inputs you need

From a clean cached forward pass, get:

- the selected head input \(x_t\),
- the clean head-local \(q_q\), \(k_j\), and either \(w_j\) or \(v_j\) plus \(W_O\),
- the clean score vector or enough information to recompute it,
- the target direction \(d\),
- the feature activations \(u_q^\mu, u_k^\nu, u_k^\lambda\),
- the decoder directions \(f_\mu, f_\nu, f_\lambda\).

TransformerLens caches the relevant attention and residual objects explicitly:
- `attn_scores` is the pre-softmax score tensor,
- `pattern` is the post-softmax attention tensor,
- `resid_pre`, `resid_mid`, `normalized`, and `scale` are cached residual / norm-related tensors.  
See the official docs for exact naming and shapes.

### 11.2 Build the selected channel pieces

\[
\delta q^\mu = u_q^\mu f_\mu W_Q^h,
\qquad
\delta k^\nu = u_k^\nu f_\nu W_K^h,
\qquad
\delta w^\lambda = u_k^\lambda f_\lambda W_V^h W_O^h.
\]

### 11.3 Build the background

\[
q^{bg}=q_q-\delta q^\mu,
\quad
k_k^{bg}=k_k-\delta k^\nu,
\quad
w_k^{bg}=w_k-\delta w^\lambda.
\]

All other source positions remain unchanged.

### 11.4 Evaluate the 8 corners

For each \((z_Q,z_K,z_V)\in\{0,1\}^3\):

1. set \(q=q^{bg}+z_Q\delta q^\mu\),
2. form the score vector:
   - for all \(j\neq k\), \(s_j = q\cdot k_j/\sqrt{d_h}\),
   - for \(j=k\), \(s_k = q\cdot (k_k^{bg}+z_K\delta k^\nu)/\sqrt{d_h}\),
3. compute \(A=\mathrm{softmax}(s)\),
4. compute the scalar write:
   \[
   g(z_Q,z_K,z_V)=A_k\;\langle w_k^{bg}+z_V\delta w^\lambda, d\rangle - g(\emptyset).
   \]

Then apply the subset formulas for \(\Delta_Q,\Delta_K,\ldots,\Delta_{QKV}\).

### 11.5 Tiny pseudocode

```python
# chosen: layer l, head h, query token qpos, source token kpos,
#         features mu, nu, lam, target direction d

delta_q = u_q_mu * (f_mu @ W_Q[h])          # shape: [d_head]
delta_k = u_k_nu * (f_nu @ W_K[h])          # shape: [d_head]
delta_w = u_k_lam * (f_lam @ W_V[h] @ W_O[h])  # shape: [d_model]

q_bg = q_clean - delta_q
k_bg = k_clean.clone()
k_bg[kpos] = k_bg[kpos] - delta_k

w_bg = w_clean.clone()
w_bg[kpos] = w_bg[kpos] - delta_w

def coalition_value(zQ, zK, zV):
    q = q_bg + zQ * delta_q

    scores = (q @ k_bg.T) / sqrt_d_head
    if zK == 1:
        scores[kpos] = (q @ (k_bg[kpos] + delta_k)) / sqrt_d_head

    A = softmax(scores, dim=0)
    path_write = w_bg[kpos] + zV * delta_w
    return A[kpos] * (path_write @ d)

g = {(zQ, zK, zV): coalition_value(zQ, zK, zV)
     - coalition_value(0, 0, 0)
     for zQ in [0, 1]
     for zK in [0, 1]
     for zV in [0, 1]}

Delta_Q   = g[(1,0,0)]
Delta_K   = g[(0,1,0)]
Delta_V   = g[(0,0,1)]
Delta_QK  = g[(1,1,0)] - g[(1,0,0)] - g[(0,1,0)]
Delta_QV  = g[(1,0,1)] - g[(1,0,0)] - g[(0,0,1)]
Delta_KV  = g[(0,1,1)] - g[(0,1,0)] - g[(0,0,1)]
Delta_QKV = (g[(1,1,1)] - g[(1,1,0)] - g[(1,0,1)] - g[(0,1,1)]
             + g[(1,0,0)] + g[(0,1,0)] + g[(0,0,1)])
```

---

## 12. Pulling the same game back to `resid_pre` features

Suppose you want the players to be **`resid_pre` features**, rather than post-LN attention-input features.

Let

\[
r_t^{pre} \approx \sum_a z_t^a g_a
\]

be the `resid_pre` feature decomposition, and let \(M_t\) be the frozen local map from `resid_pre[t]` to the actual attention input \(x_t\).  
For a frozen-LN-denominator approximation, \(M_t\) is the cached linearized norm map from the clean run.

Then define channel pieces directly from `resid_pre` features:

\[
\delta q^a := z_q^a\, M_q g_a \, W_Q^h,
\]
\[
\delta k^b := z_k^b\, M_k g_b \, W_K^h,
\]
\[
\delta w^c := z_k^c\, M_k g_c \, W_V^h W_O^h.
\]

Now the exact same 8-corner game gives singles, pairs, and triples over the channel-specific `resid_pre` features \((a,b,c)\).

So there are really two perfectly parallel analyses:

1. **post-LN feature FRA game**: use \(f_\mu, f_\nu, f_\lambda\),
2. **pre-LN feature FRA game**: use \(M_t g_a, M_t g_b, M_t g_c\).

The algebra is the same. Only the player definitions change.

---

## 13. How to summarize many triplets

Once you have \(\Delta_S\) for many triplets, there are three natural summary families.

### 13.1 Pure order summaries

Average or sum the pure terms:

\[
\mathbb E[\Delta_Q],\quad
\mathbb E[\Delta_{QK}],\quad
\mathbb E[\Delta_{QKV}],
\]
or use absolute / squared versions:
\[
\mathbb E[|\Delta_S|],\qquad
\mathbb E[\Delta_S^2].
\]

Interpretation:
- signed mean = consistent push or pull,
- absolute mean = salience,
- squared mean = interaction energy.

### 13.2 Full involvement scores

If you want “everything involving \(Q\)”:

\[
F_Q = \Delta_Q + \Delta_{QK} + \Delta_{QV} + \Delta_{QKV},
\]
\[
F_K = \Delta_K + \Delta_{QK} + \Delta_{KV} + \Delta_{QKV},
\]
\[
F_V = \Delta_V + \Delta_{QV} + \Delta_{KV} + \Delta_{QKV}.
\]

Similarly for pairs:

\[
F_{QK} = \Delta_{QK} + \Delta_{QKV},
\quad
F_{QV} = \Delta_{QV} + \Delta_{QKV},
\quad
F_{KV} = \Delta_{KV} + \Delta_{QKV}.
\]

These are useful if you care about *involvement* rather than pure interaction order.

### 13.3 Symmetric redistributed scores

If you want single-number channel rankings with the interaction mass shared fairly:

\[
\phi_Q
=
\Delta_Q+\frac12\Delta_{QK}+\frac12\Delta_{QV}+\frac13\Delta_{QKV},
\]
\[
\phi_K
=
\Delta_K+\frac12\Delta_{QK}+\frac12\Delta_{KV}+\frac13\Delta_{QKV},
\]
\[
\phi_V
=
\Delta_V+\frac12\Delta_{QV}+\frac12\Delta_{KV}+\frac13\Delta_{QKV}.
\]

This equal-split redistribution is a simple symmetric convention.  
If you want a more axiomatic redistribution scheme, that is where Shapley-Taylor style indices enter.

---

## 14. Practical guidance

If you only remember one page from this note, remember this:

### Recommended default
Use the **contextual incremental path-local game**.

That gives the most faithful mechanistic answer to:

> In the clean run, how much of this path’s effect came from the selected query feature, key feature, value feature, their pairs, and their irreducible triple?

### Use the strict triplet game when
you want to isolate the selected value write and exclude background write from the source path.

### Use pure \(\Delta_S\) terms when
you care about exact interaction order.

### Use full involvement or redistributed scores when
you want rankings rather than pure decomposition.

### Use the frozen-attention simplification when
you only want OV attribution.

Then the channel game collapses to \(V\)-only.

---

## 15. Checklist for a first implementation

1. Pick a target direction \(d\) at `hook_resid_mid[q]`.
2. Pick a head \(h\), source token \(k\), and candidate triplet \((\mu,\nu,\lambda)\).
3. Construct \(\delta q^\mu\), \(\delta k^\nu\), \(\delta w^\lambda\).
4. Decide between:
   - contextual incremental game,
   - strict triplet game.
5. Compute the 8 coalition values.
6. Compute the exact Möbius coefficients:
   \[
   \Delta_Q,\Delta_K,\Delta_V,\Delta_{QK},\Delta_{QV},\Delta_{KV},\Delta_{QKV}.
   \]
7. Validate top findings by direct patching / ablation.
8. Aggregate across prompts with mean, mean-absolute, or mean-square summaries.

---

## 16. Bottom line

The clean way to ask for single, double, and triple contributions in FRA is:

- **make \(Q\), \(K\), and \(V\) the three players,**
- **define a scalar local coalition game for a chosen path and target direction,**
- **decompose the game with the Möbius / Harsanyi transform.**

This gives you an exact and interpretable decomposition into:

- single-channel effects,
- pairwise channel interactions,
- an irreducible three-way FRA interaction.

That is the right abstraction layer above the raw FRA triplet tensor.

---

## References

- TransformerLens docs on `ActivationCache`, including cached `pattern`, `attn_scores`, `resid_pre`, `resid_mid`, `normalized`, and `scale`:  
  <https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.ActivationCache.html>

- Dhamdhere, Agarwal, Sundararajan. **The Shapley Taylor Interaction Index**. arXiv:1902.05622.  
  <https://arxiv.org/abs/1902.05622>

- Janizek, Sturmfels, Lee. **Explaining Explanations: Axiomatic Feature Interactions for Deep Networks**. arXiv:2002.04138.  
  <https://arxiv.org/abs/2002.04138>

- Fumagalli et al. **Unifying Feature-Based Explanations with Functional ANOVA and Cooperative Game Theory**. arXiv:2412.17152.  
  <https://arxiv.org/abs/2412.17152>
