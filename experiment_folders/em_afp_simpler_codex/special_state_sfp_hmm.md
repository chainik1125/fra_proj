# Simpler EM-AFP HMM: two-state factor components

This note writes out a simpler HMM intended to capture the core finetuning
spillover intuition with fewer moving parts than the current tagged-content
EM-AFP process.

The intended structure is still the same high-level SFP:

$$
(T_M \oplus T_A) \otimes (T_D \oplus T_O)
\cong
(T_M \otimes T_D)
\oplus
(T_M \otimes T_O)
\oplus
(T_A \otimes T_D)
\oplus
(T_A \otimes T_O).
$$

So the four closed leaves are still:

$$
MD,\quad MO,\quad AD,\quad AO.
$$

Each factor process \(T_M,T_A,T_D,T_O\) is now only a two-state HMM.

## Intuition

Each factor HMM has:

- a neutral state \(N\), with identical emissions for all factors;
- a special state \(U_X\), unique to factor \(X\in\{M,A,D,O\}\).

The neutral state emits only neutral symbols `0` and `1`. These emissions have
the same statistics for \(M,A,D,O\), so they carry no factor identity.

The special state emits a factor-specific token:

$$
S_M,\quad S_A,\quad S_D,\quad S_O.
$$

These are the only emissions that identify the factor.

This gives a direct model of spillover:

- MD finetuning sees \(S_M\) and \(S_D\);
- \(S_M\) is shared by \(MD\) and \(MO\);
- \(S_D\) is shared by \(MD\) and \(AD\);
- whether finetuning generalizes more to \(MO\) or \(AD\) depends on how the
  model represents and updates the persona/domain factor coordinates.

## One Two-State Factor HMM

Fix a factor

$$
X\in\{M,A,D,O\}.
$$

The hidden states are:

$$
N_X = \text{neutral state},
\qquad
U_X = \text{special state for factor }X.
$$

The local vocabulary is:

$$
\mathcal V_X=\{0,1,S_X\}.
$$

I will use the following interpretation of the proposed transition/emission
rules. From the neutral state:

| current state | emitted token | next state | probability |
|---|---|---|---:|
| \(N_X\) | `0` | \(N_X\) | \(p-\epsilon\) |
| \(N_X\) | `1` | \(N_X\) | \(1-p-\epsilon\) |
| \(N_X\) | `0` | \(U_X\) | \(\epsilon\) |
| \(N_X\) | `1` | \(U_X\) | \(\epsilon\) |

From the special state:

| current state | emitted token | next state | probability |
|---|---|---|---:|
| \(U_X\) | \(S_X\) | \(U_X\) | \(p_s\) |
| \(U_X\) | \(S_X\) | \(N_X\) | \(1-p_s\) |

This convention means:

$$
\Pr(0\mid N_X)=p,
\qquad
\Pr(1\mid N_X)=1-p,
$$

$$
\Pr(N_X\to U_X\mid N_X)=2\epsilon,
\qquad
\Pr(S_X\mid U_X)=1,
\qquad
\Pr(U_X\to U_X\mid U_X)=p_s.
$$

The parameter constraints are:

$$
0\le \epsilon \le \min(p,1-p),
\qquad
0\le p\le 1,
\qquad
0\le p_s\le 1.
$$

If we instead want total neutral-to-special transition probability \(\epsilon\),
replace the two \(\epsilon\) entries above by \(\epsilon/2\). The rest of this
note uses the literal "epsilon on 0 or 1" convention, so the total transition
probability is \(2\epsilon\).

## Token Operators for One Factor

Use row-vector convention. A token operator is:

$$
T_a[i,j]
=
\Pr(\text{emit token }a\text{ and transition to next state }j
\mid \text{current state }i).
$$

Rows and columns are ordered as \((N_X,U_X)\). The local token operators are:

$$
T_0^X
=
\begin{pmatrix}
p-\epsilon & \epsilon\\
0 & 0
\end{pmatrix},
$$

$$
T_1^X
=
\begin{pmatrix}
1-p-\epsilon & \epsilon\\
0 & 0
\end{pmatrix},
$$

$$
T_{S_X}^X
=
\begin{pmatrix}
0 & 0\\
1-p_s & p_s
\end{pmatrix}.
$$

The sum over local tokens is the stochastic hidden transition matrix:

$$
A_X
=T_0^X+T_1^X+T_{S_X}^X
=
\begin{pmatrix}
1-2\epsilon & 2\epsilon\\
1-p_s & p_s
\end{pmatrix}.
$$

\(A_X\) is an ordinary two-state ergodic chain when \(\epsilon>0\) and
\(p_s<1\).

## Persona and Domain Factor Alphabets

For the persona side, the two possible factor HMMs are \(T_M\) and \(T_A\). The
persona-side alphabet is:

$$
\mathcal V_{\mathrm{persona}}=\{0,1,S_M,S_A\}.
$$

Inside \(T_M\), the token \(S_A\) has zero operator. Inside \(T_A\), the token
\(S_M\) has zero operator.

Similarly, for the domain side:

$$
\mathcal V_{\mathrm{domain}}=\{0,1,S_D,S_O\}.
$$

Inside \(T_D\), the token \(S_O\) has zero operator. Inside \(T_O\), the token
\(S_D\) has zero operator.

Define the persona direct sum:

$$
P = T_M\oplus T_A,
$$

and the domain direct sum:

$$
Q = T_D\oplus T_O.
$$

Each of \(P\) and \(Q\) has four hidden states:

$$
P:\ M_N,M_U,A_N,A_U,
\qquad
Q:\ D_N,D_U,O_N,O_U.
$$

Both are non-ergodic across their outer components. For example, \(P\) never
moves from the \(M\) block to the \(A\) block.

## Full SFP HMM

The full HMM is:

$$
H=P\otimes Q.
$$

This gives the four closed leaves:

$$
MD=T_M\otimes T_D,\qquad
MO=T_M\otimes T_O,\qquad
AD=T_A\otimes T_D,\qquad
AO=T_A\otimes T_O.
$$

Each leaf has four hidden states:

$$
(N_{\mathrm{persona}},N_{\mathrm{domain}}),\quad
(U_{\mathrm{persona}},N_{\mathrm{domain}}),\quad
(N_{\mathrm{persona}},U_{\mathrm{domain}}),\quad
(U_{\mathrm{persona}},U_{\mathrm{domain}}).
$$

So the full hidden state space has:

$$
4\text{ leaves}\times 4\text{ states per leaf}=16\text{ hidden states}.
$$

The true leaf is fixed for an entire sequence. The only within-sequence dynamics
are whether the persona factor and/or domain factor are in their neutral or
special states.

## Full Vocabulary

At each time step, the persona factor and the domain factor each emit one local
symbol. The global token is their pair:

$$
(a,b),
$$

where:

$$
a\in\mathcal V_{\mathrm{persona}}=\{0,1,S_M,S_A\},
\qquad
b\in\mathcal V_{\mathrm{domain}}=\{0,1,S_D,S_O\}.
$$

The global vocabulary size is:

$$
|\mathcal V|=4\times 4=16.
$$

The neutral pair tokens are:

$$
(0,0),\quad (0,1),\quad (1,0),\quad (1,1).
$$

The factor-specific evidence tokens include:

$$
(S_M,0),\ (S_M,1),\ (S_A,0),\ (S_A,1),
$$

$$
(0,S_D),\ (1,S_D),\ (0,S_O),\ (1,S_O).
$$

The simultaneous special events are:

$$
(S_M,S_D),\quad (S_M,S_O),\quad (S_A,S_D),\quad (S_A,S_O).
$$

Some global tokens are impossible under some leaves. For example, \(S_A\) is
impossible in \(MD\) and \(MO\), while \(S_M\) is impossible in \(AD\) and
\(AO\).

## Full Token Operators

For global token \((a,b)\), where \(a\) is a persona-side symbol and \(b\) is a
domain-side symbol, the full token operator is:

$$
T_{(a,b)} = P_a\otimes Q_b.
$$

Equivalently, inside each leaf:

$$
T_{(a,b)}\big|_{MD}=T_a^M\otimes T_b^D,
$$

$$
T_{(a,b)}\big|_{MO}=T_a^M\otimes T_b^O,
$$

$$
T_{(a,b)}\big|_{AD}=T_a^A\otimes T_b^D,
$$

$$
T_{(a,b)}\big|_{AO}=T_a^A\otimes T_b^O.
$$

The global operators are block diagonal over \(MD,MO,AD,AO\), so there is no
mass leakage between leaves.

The sum over all global tokens is:

$$
\sum_{(a,b)\in\mathcal V} T_{(a,b)}
=
\left(\sum_a P_a\right)\otimes\left(\sum_b Q_b\right)
=
A_P\otimes A_Q,
$$

which is stochastic. Each individual observed-token operator is substochastic.

## Initial Distribution

Use a prior over leaves:

$$
\eta=(\eta_{MD},\eta_{MO},\eta_{AD},\eta_{AO}).
$$

The base process should use:

$$
\eta=\left(\frac14,\frac14,\frac14,\frac14\right).
$$

Within each selected leaf, initialize the persona and domain factors in the
neutral state:

$$
\Pr(N_{\mathrm{persona}},N_{\mathrm{domain}}\mid \text{leaf})=1.
$$

So every sequence begins in the fully neutral internal state of its leaf. This
makes early tokens uninformative unless a factor transitions into its special
state.

An alternative is to initialize each factor in its stationary distribution. That
would remove the transient at the beginning, but the all-neutral initialization
is easier to reason about for a first experiment.

## Suggested Starting Parameters

A concrete first setting:

$$
p=0.5,\qquad
\epsilon=0.02,\qquad
p_s=0.8,\qquad
L=64.
$$

This means:

$$
\Pr(N\to U\text{ per step})=0.04,
$$

$$
\mathbb E[\text{neutral waiting time}]\approx 25\text{ steps},
$$

and:

$$
\mathbb E[\text{special burst length}]
=
\frac{1}{1-p_s}
=5\text{ tokens}.
$$

So a length-64 sequence should usually contain a few special bursts, while most
tokens remain neutral.

If the posterior does not polarize enough, increase \(\epsilon\), \(p_s\), or
\(L\).

## Exact Posterior Quantities

The full belief state is:

$$
b[\ell,u_{\mathrm{persona}},u_{\mathrm{domain}}],
$$

with:

$$
\ell\in\{MD,MO,AD,AO\},
\qquad
u_{\mathrm{persona}}\in\{N,U\},
\qquad
u_{\mathrm{domain}}\in\{N,U\}.
$$

The component posterior is:

$$
\mu_\ell
=
\sum_{u_{\mathrm{persona}},u_{\mathrm{domain}}}
b[\ell,u_{\mathrm{persona}},u_{\mathrm{domain}}].
$$

The factor marginals are:

$$
\mu_M=\mu_{MD}+\mu_{MO},
\qquad
\mu_A=\mu_{AD}+\mu_{AO},
$$

$$
\mu_D=\mu_{MD}+\mu_{AD},
\qquad
\mu_O=\mu_{MO}+\mu_{AO}.
$$

The fine-grained internal beliefs can also be measured per leaf:

$$
b[\ell,:,:],
$$

or conditionally within a leaf:

$$
\frac{b[\ell,:,:]}{\mu_\ell}.
$$

## Fine-Tuning Experiment

Base train on the uniform mixture over leaves:

$$
\eta=\left(\frac14,\frac14,\frac14,\frac14\right).
$$

Fine-tune on the \(MD\) leaf:

$$
\eta=(1,0,0,0).
$$

An \(MD\) sequence can emit:

$$
S_M\quad\text{and}\quad S_D,
$$

but it can never emit:

$$
S_A\quad\text{or}\quad S_O.
$$

The core spillover question is whether MD finetuning changes behavior or
representations more along the shared persona direction:

$$
MD\to MO,
$$

or along the shared domain direction:

$$
MD\to AD.
$$

This version removes the previous content-codebook machinery. The only
non-neutral evidence is explicit factor-specific special tokens.

## Why This Is Simpler Than the Current EM-AFP HMM

The current HMM has:

- four leaves;
- five internal states per leaf;
- tagged content tokens;
- a shared content channel;
- posterior geometry that mixes component mass and within-component content
  tracking.

This simpler HMM has:

- four leaves;
- four internal states per leaf, coming from two binary factor states;
- neutral tokens with identical statistics across factors;
- special tokens that isolate factor evidence;
- no content codebook.

The intended benefit is interpretability: if a model fine-tuned on \(MD\)
generalizes to \(MO\), that should be attributable to the shared \(S_M\)/persona
factor rather than to content-token geometry.

## Open Implementation Choices

1. Whether neutral-to-special transition probability should be \(2\epsilon\),
   as written above, or total \(\epsilon\) split across neutral tokens.
2. Whether the global token should be a pair \((a,b)\), with one persona-side
   emission and one domain-side emission per step, or whether persona/domain
   emissions should alternate in time.
3. Whether sequences should start in the all-neutral state or in the stationary
   distribution of each factor HMM.
4. Whether \(M/A\) and \(D/O\) should share identical \(p,\epsilon,p_s\), or
   whether persona specials should be rarer/stronger than domain specials to
   create a persona-dominant regime.

The paired-token, all-neutral, shared-parameter version is the cleanest first
implementation.
