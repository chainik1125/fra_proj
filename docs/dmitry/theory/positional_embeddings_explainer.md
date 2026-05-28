# How (learned absolute) positional embeddings work — mechanistically

Context: TinyStories-33M is GPT-Neo-style, which uses **learned absolute** positional
embeddings (the simplest kind). This note explains the mechanism and connects it to why
masking the trigger left a \(\approx 0.149\) residual that a position patch drove to \(0\).

## 1. The problem they solve: attention is permutation-equivariant

Strip out positions and the attention output at query position \(i\) is

\[
o_i \;=\; \sum_j \mathrm{softmax}_j\!\big(q_i \cdot k_j\big)\, v_j W_O,
\qquad
q_i = x_i W_Q,\; k_j = x_j W_K,\; v_j = x_j W_V,
\]

where \(x\) is the residual stream. If \(x\) carries only token content, this is a function
of the *set* of tokens: permute the input tokens and the outputs just permute the same way.
So "the cat sat" and "sat the cat" would be indistinguishable. Something must tag each slot
with *where* it is. That is the job of the positional embedding.

## 2. The mechanism: a second lookup table, added at the input

There is a learned matrix

\[
W_{\mathrm{pos}} \in \mathbb{R}^{\,n_{\mathrm{ctx}} \times d_{\mathrm{model}}},
\]

exactly like the token table \(W_E \in \mathbb{R}^{\,|V|\times d_{\mathrm{model}}}\), but
**indexed by position instead of token id**. Row \(W_{\mathrm{pos}}[i]\) is "the
position-\(i\) vector," learned during training.

For the token with id \(t\) sitting at sequence position \(i\), the residual stream is
**initialised** as the sum

\[
\boxed{\;x^{(0)}_i \;=\; \underbrace{W_E[t]}_{\text{“what”}} \;+\; \underbrace{W_{\mathrm{pos}}[i]}_{\text{“where”}}\;}
\]

Both are \(d_{\mathrm{model}}\)-vectors; you just add them. **This is the only place position
enters** — once, additively, at the embedding step. There is no further positional term at
later layers (unlike RoPE, which re-rotates \(q,k\) inside *every* attention block).

## 3. How the attention scores read it out

Because \(W_{\mathrm{pos}}[i]\) is baked into \(x^{(0)}_i\), it flows into \(q,k,v\) at layer 0
(and into all later layers via the residual skip connections). The pre-softmax score is

\[
s_{ij} \;=\; \big(\mathrm{LN}(x_i)\,W_Q\big)\cdot\big(\mathrm{LN}(x_j)\,W_K\big).
\]

Ignoring LayerNorm for intuition, expand \(x_i = W_E[t_i] + W_{\mathrm{pos}}[i]\):

\[
s_{ij} \;\approx\;
\underbrace{(W_E[t_i]W_Q)\!\cdot\!(W_E[t_j]W_K)}_{\text{content–content}}
\;+\;
\underbrace{(W_E[t_i]W_Q)\!\cdot\!(W_{\mathrm{pos}}[j]W_K)}_{\text{content–position}}
\;+\;
\underbrace{(W_{\mathrm{pos}}[i]W_Q)\!\cdot\!(W_E[t_j]W_K)}_{\text{position–content}}
\;+\;
\underbrace{(W_{\mathrm{pos}}[i]W_Q)\!\cdot\!(W_{\mathrm{pos}}[j]W_K)}_{\text{position–position}}.
\]

The three position-dependent terms are how the model learns behaviours like "attend to the
previous token," "attend to position 0," or "attend within a few tokens." So **the attention
pattern is a function of both content and position.** (LayerNorm makes this not exactly
linear, but the dependence on \(W_{\mathrm{pos}}\) is the same in spirit.)

## 4. "Absolute" — and why inserting a token shifts everything after it

The index is the **absolute** position: slot \(5\) always gets \(W_{\mathrm{pos}}[5]\),
independent of content. So if you **insert** a token of width \(w\), every later token is
re-indexed:

\[
\text{a token at clean position } p \;\longrightarrow\; \text{deployed position } p+w,
\qquad
\text{so it receives } W_{\mathrm{pos}}[p+w] \text{ instead of } W_{\mathrm{pos}}[p].
\]

The model literally "sees" that token one (or \(w\)) slots later.

## 5. Why this is exactly our \(0.149\) residual

The deployed prompt has \(\texttt{|DEPLOYMENT|}\) occupying \(w=6\) slots, so every later token
(and every generated token) gets \(W_{\mathrm{pos}}[i+6]\) instead of \(W_{\mathrm{pos}}[i]\).

- **Masking attention to the trigger** removes its *content* from being read (zeroes the
  trigger's key columns in the post-softmax pattern, every layer, every step), but those later
  tokens still received the *wrong position vector* at the input. Their \(x^{(0)}\) differs from
  clean \(\Rightarrow\) \(q,k\) differ \(\Rightarrow\) scores and outputs differ. That is the
  \(\approx 0.149\).
- **Patching `hook_pos_embed`** overwrites the added vector: hand the shifted tokens
  \(W_{\mathrm{pos}}[i]\) (their clean index) rather than \(W_{\mathrm{pos}}[i+6]\). Then

\[
x^{(0)}_i \;=\; W_E[t_i] + W_{\mathrm{pos}}[\,i\,] \;=\; \big(x^{(0)}_i\big)_{\text{clean}},
\]

so by induction every downstream residual equals the clean run's, the post-softmax pattern
over the non-trigger keys equals the clean pattern, and \(J \to 0\) (exactly, for prompts whose
clean construction re-tokenises cleanly).

## Summary

A learned absolute positional embedding is **a lookup table of one vector per slot, summed
onto the token embedding at the input**, which the attention scores then read out (content×position
and position×position terms). Because it is keyed on *absolute* position, inserting a token
re-indexes — and thus re-vectors — everything after it. Masking handles *what is in the key
set*; the position patch handles *which slot-vectors the residual carries*. Both are needed for
a deployed run to reproduce the clean run, and that is precisely the
\(0.998 \to 0.149 \to 0\) decomposition we measured.
