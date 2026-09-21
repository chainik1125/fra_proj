# What Shamir sharing does—and does not—force

The finite-field sharing construction follows
[Shamir (1979)](https://people.cs.pitt.edu/~litman/courses/cs2001/lectures/lecture-02-p612-shamir.pdf).
The HMM protocol, architecture-specific bounds and FRA interpretations below
are derivations for this experiment, not claims attributed to that paper.

## Exact threshold posterior

Let S,a be independent uniform GF(5) values and y1=S+a, y2=S+2a. Then

$$S=2y_1-y_2\pmod5.$$

For either x=1 or x=2 and every fixed share value y, exactly one slope is
consistent with each possible S. Thus S is uniform conditional on one share.
With both shares, the degree-at-most-one polynomial and S are unique. Slopes
must include zero; restricting to nonzero slopes would destroy this privacy
property. The enumerative test checks all 25 polynomials.

If g_s is the activation direction for secret s and mu is their mean, the
Bayes MSE predictor is mu before the threshold and g_S after it. Other streams
are independent and contribute no information about the requested secret.
Random ordering, stopping length and request are independent of share values.

For m=2C total shares and a uniformly shuffled prefix of length l, the chance
that both requested shares are present is l(l-1)/[m(m-1)]. With shortening
probability eta=.25 and the specified uniform short-length distribution, the
population Bayes NMSE is

$$\eta\left[1-\frac1{m-1}\sum_{l=1}^{m-1}
\frac{l(l-1)}{m(m-1)}\right].$$

This equals .25 for C=1 and 11/60≈.183333 for C=3. Empirical Bayes NMSE varies
with the sampled share-count frequencies and the very small non-orthogonality
of the reused numerical feature dictionary.

## Why one stream does not force content-based attention

With one stream and both shares present, uniform attention can pool one-hot
(x=1,y1) and (x=2,y2) atoms. These distinct atom groups preserve both values
in the sum. An MLP can implement the finite lookup (2y1-y2) mod 5. The share
positions may be shuffled: their x labels identify their roles.

Therefore needing multiple observations does not imply needing content-based
routing. It can instead require a nonlinear readout of a fixed pooled memory.

There is also a precise arithmetic obstruction for an additive affine
readout. For any secret-one-hot component T, E[T|y1]=E[T|y2]=E[T]. Consequently
T-E[T] is orthogonal in population L2 to every function of y1 alone and every
function of y2 alone. Its best additive squared-error predictor is constant.
This explains why simply linearly decoding a fixed weighted sum of separate
share embeddings does not reconstruct the secret. Content-dependent softmax
weights can themselves introduce nonlinear interactions, so this is NOT an
impossibility theorem for every trained single attention head. The full-QK
linear-output head's nonzero empirical improvement must be retained.

## Why three queried streams force query dependence here

For the strict-past, no-query-residual architecture, if content QK is zero,
the output is a function h(M) of memory and positions only, even with an MLP.
Changing the requested stream R cannot change h(M). We explicitly hold M fixed
and change R in the counterfactual audit.

Suppose all C pairs are present. Memory determines independent uniform
secrets S1,...,SC. Even allowing an arbitrarily powerful query-blind decoder,
its optimal prediction is

$$h^*(M)=\frac1C\sum_{c=1}^C g_{S_c}.$$

Writing V=E||g_S-mu||², the population risk is

$$\mathbb E\|g_{S_R}-h^*(M)\|^2=(1-1/C)V.$$

For C=3, query-blind NMSE is at least 2/3, whereas a query-aware decoder can
attain zero. This proof does not need exactly orthogonal g_s. The finite-field
test enumerates all 125 secret triples. A single trained positional head may
do worse than this bound because pooling can also lose stream/share binding.

The bound as stated is for all shares present. Conditioning on "the requested
stream has both shares" among partial prefixes changes the request law given
memory, so applying the same conditional bound to that stratum would be wrong.

This conclusion is architecture-specific. Passing the query through a residual
connection or directly to a sufficiently expressive decoder defeats the
query-blind argument. It does not prove that all transformer architectures
need content-based attention for this task.

## FRA: content dependence is not off-diagonal dependence

With row feature matrix F and coefficients q,z_t, define

$$B=F W_QW_K^\top F^\top/\sqrt d,\qquad
\ell_t=q B z_t^\top+b_{\rm lag(t)}.$$

The score expansion is exact even for the slightly non-orthogonal F. The
feature-resolved terms are q_i B_ij z_tj; B is their coupling matrix, not the
entire activation-dependent FRA tensor.

We deliberately use the SAME stream atom in query and memory. A diagonal B
with B_cc=alpha for the stream atoms and all other entries zero scores matching
stream keys alpha and other streams zero. With all pairs present, total
attention on the requested pair is

$$\frac{e^\alpha}{e^\alpha+C-1}\longrightarrow1.$$

Both shares can then pass through OV, and the MLP reconstructs the secret.
Hence diagonal QK is constructively sufficient in the large-margin limit.
More interesting temporal dependence need not create necessary off-diagonal
feature couplings. A different encoding, such as disjoint query and memory
identity atoms, would make the same matching operation off-diagonal by
construction; that would not itself constitute a deeper inference mechanism.

The learned matrices do have off-diagonal terms. Removing them modestly worsens
MSE, while preserving perfect observed full-memory classification. We do not
claim that the matrices themselves are diagonal, that their off-diagonals have
zero effect, or that the finite trained solution is identical after ablation.

Softmax ignores key-constant score shifts. The plotted 3×3 matrix includes the
query-type-to-stream score, then centers each row. Negative entries in this
centered plot are relative nonmatching-stream scores, not by themselves proof
of a particular negative-inhibition mechanism.

## OV carries operands; the readout combines them

The pre-MLP vector is

$$h=\sum_t A_t\sum_j z_{tj}v_j+b,\qquad v_j=f_j W_VW_O.$$

With stream-selective attention, this is approximately a sum of the two
requested share representations plus stream/type terms. Held-out probes test
whether those operands survive. Raw v_j directions do not have a unique
alignment to the output secret basis: an invertible change of coordinates in
h can be absorbed by the MLP's first affine layer. Thus raw OV diagonality is
not a well-posed recovery target for this MLP arm.

The observed near-zero secret regression R² before the MLP supports an
operand-transport interpretation, but it is not a proof of no linearly
classifiable secret information: probe argmax accuracy is also recorded and
is above chance. The end-to-end reconstruction and share-probe results are
the stronger positive evidence.
