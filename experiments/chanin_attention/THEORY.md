# What the independent reset process identifies

Let F have orthonormal rows f_i, and x_t = z_t F, where
z_(t,i) = a_(t,i) M_(t,i), M = max(0, Normal(1, sigma²)). The magnitudes are
independent across features and positions and independent of the firing masks.
Write mu=E[M] and m2=E[M²]. The finite numerical Gram error of the reproduced
dictionary is measured separately.

## IID positions

For next-activation squared error, E[x_(t+1) | x_0:t] = E[x]. Same-position
correlations do not alter this. The head can attain the population optimum by
setting W_V W_O = 0 and b_out = E[x], with *arbitrary* QK matrices.
Consequently, even infinitely many samples and optimal loss cannot identify
QK as the identity, nor require meaningful OV transport. This is an exact
non-identifiability statement, not a finite-data prediction about which
minimizer SGD will choose.

For current-activation reconstruction the situation changes, but even then
attending only to self and setting OV to identity leaves QK unconstrained once
the self-only attention pattern is fixed. A residual skip would introduce an
additional trivial solution, which is why the benchmark omits it.

## Reset model and its Bayes predictor

Each feature independently keeps its binary state with probability rho_i and
otherwise redraws Bernoulli(pi_i). Equivalently,

```
P(0 -> 1) = (1-rho_i) pi_i
P(1 -> 0) = (1-rho_i)(1-pi_i)
```

The stationary law is the product of Bernoulli(pi_i), and

```
E[a_(t+1,i) | a_t] = (1-rho_i) pi_i + rho_i a_(t,i).
Cov(a_(t,i), a_(t+k,j)) = delta_ij pi_i(1-pi_i) rho_i^k.
```

If firing is observed, the exact optimal next-activation prediction is

```
E[x_(t+1)|a_t] = sum_i mu [(1-rho_i)pi_i + rho_i a_(t,i)] f_i.
```

It is diagonal as a *predictive map in the firing variables*. That does not
imply that the attention score matrix in the feature basis is diagonal.
When M=1, self-only attention, diagonal OV=rho and bias=(1-rho)pi F realize
this predictor with zero content QK. A learned relative-lag bias tending to
+infinity at lag zero approaches that solution. Any fixed finite content
QK gives the same limiting prediction. Thus temporal dependence alone still
does not identify a unique QK matrix.

When amplitudes vary, copying x_t with diagonal OV=rho is not the exact Bayes
predictor: it copies the current amplitude noise. The optimal affine predictor
using only the current z_i has slope

```
c_i = rho_i mu² pi_i(1-pi_i) / [pi_i m2 - pi_i² mu²]
    = rho_i mu²(1-pi_i) / [m2 - pi_i mu²]
intercept_i = pi_i mu (1-c_i).
```

For sigma=.15, clipping is extremely rare, mu≈1 and m2≈1.0225. This predicts
slopes slightly below rho. Content-based attention can trade amplitude
averaging against stale observations, so this is a specified affine baseline,
not a claim that every optimized attention head must equal this map.

## Noisy observations

For the paper's independent noisy-emission case, let
P(a_t=1|s_t=0)=p_A and P(a_t=1|s_t=1)=p_B. If q_t=P(s_t=1|history before t),
the posterior after observing a_t is

```
r_t = q_t L_1(a_t) / [q_t L_1(a_t)+(1-q_t)L_0(a_t)]
q_(t+1) = (1-rho)pi + rho r_t.
```

Here L_s is the Bernoulli likelihood. Each feature filters its own observation
history; the product structure is exact. The implementation also accounts for
zero amplitudes created by clipping: the effective detection rate is
p_B P(M>0), with p_A=0. A positive magnitude identifies state 1 and provides
no further information about that state because the magnitude law is shared.

The next magnitude mean is mu p_B q_(t+1). This filter is checked independently
by enumerating all hidden-state paths for short sequences. A single head
shares one attention distribution over time across all output features; it
need not be able to realize the collection of nonlinear independent filters.
Any remaining oracle gap must be reported rather than called low-loss recovery.

## What an off-diagonal FRA entry means

Define B_ij = f_i W_Q W_K^T f_j^T / sqrt(d_head). The exact content score is
S_ts = sum_ij z_(t,i) B_ij z_(s,j). The full FRA tensor weights B by actual
query/key feature magnitudes; B itself is the feature coupling matrix.

For distinct positions with positive lag k and independent feature chains,

```
E[z_(t,i) z_(t-k,j)]
  = mu² [pi_i pi_j + delta_ij pi_i(1-pi_i) rho_i^k].
```

Thus off-diagonal *raw* second moments are nonzero even for independent
features. Centered cross-feature covariance vanishes; raw FRA contributions
do not. Conflating these quantities would misinterpret the experiment.

At a fixed lag and in the stationary population, the expected off-diagonal
score is mu² sum_(i!=j) B_ij pi_i pi_j, independent of lag. For a specific
observed query its conditional expectation generally differs, since its own
past feature states can be informative. Neither unconditional cancellation
nor a visually diagonal lag-covariance matrix proves that individual QK
terms are functionally irrelevant.

Softmax also discards any score shift constant across keys for a query. We
therefore measure score amplitude after row centering over valid causal keys,
and test actual prediction changes from removing the off-diagonal QK terms.
An off-diagonal coefficient's functional role depends on OV and the available
positional routing, not only its numerical size. Cross-feature independence
constrains the Bayes predictor's dependence structure; it does not by itself
constrain an internal bilinear parameterization to the identity.

## A precise content-only mechanism: excluding extra features in a key

There is a concrete reason for negative off-diagonals even with independent
chains. First take binary amplitudes and no positional or QK biases. A
diagonal score B=alpha I gives query q and key k the score alpha |q∩k|.
Any key containing every active query feature ties with the query itself,
even if the key has additional active features. OV then transports those
additional features too. This is a routing ambiguity, not a dependency in
the data generator.

The simple family

```
B = alpha I - beta 1 1^T,     alpha > N beta > 0
S(q,k) = alpha |q∩k| - beta |q| |k|
```

solves that ambiguity for every nonempty binary query q. For an active
query feature j, adding j to the key changes the score by alpha-beta|q|>0.
For an inactive query feature j, adding j changes it by -beta|q|<0.
Therefore the unique maximizing binary pattern is k=q. Repeated matching
positions may tie, but their binary values agree. An off-diagonal rank-one
background implements a penalty for extra features in the key.

This construction is tested exhaustively on all six-feature binary patterns.
It is an exact statement about this score family, not a prediction that SGD
will learn exactly that family. It excludes the all-zero query, whose
bias-free query projection is zero and cannot discriminate keys; continuous
amplitude variation introduces further deviations. We fit the learned
off-diagonal matrices to constant and row/column backgrounds to measure
whether this mechanism describes the actual trained heads.

## Independent HMM pair likelihoods also contain a key-only penalty

A related exact calculation helps distinguish independence from the form of
an attention score. For a single feature and lag k, put r=rho^k,
p1=pi+r(1-pi), and p0=pi(1-r). For binary query q and key v,

```
log P(q|v) = query_only(q) + w q v + c v
w = log[p1(1-p0)/(p0(1-p1))] > 0
c = log[(1-p1)/(1-p0)] < 0,           0 < r < 1.
```

For independent features these scores add across i. The bilinear part is
diagonal, but there is also a negative *key-only* term. A query projection
bias could represent that term directly. This benchmark's Q and K projections
are bias-free, so a matrix of the form `diag(w)+u c^T` instead gives

```
q^T B v = sum_i w_i q_i v_i + (u^T q) sum_j c_j v_j.
```

When u^T q is approximately constant over typical sparse queries, a negative
rank-one background approximates the key-only penalty. This is a mathematical
connection, not a claim that a head trained on next-activation MSE optimizes
pair likelihood or learns these particular w and c. Establishing that stronger
claim would require testing query-bias and likelihood-score controls. The
empirical conclusions below rely on measured score effects and interventions.

Sources: the released Chanin code pinned in README.md; the local Temporal
Crosscoders main.tex lines 372–575 and appendix.tex lines 13–37. The arguments
above are derivations for this benchmark, not claims attributed to those papers.
