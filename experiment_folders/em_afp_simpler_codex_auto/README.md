# EM-AFP simpler autonomous run

This folder is for autonomous follow-up experiments on the simplified
special-state SFP HMM.

Goal: get closer to emergent-misalignment-style behavior while keeping the HMM
simple. Operationally, after MD fine-tuning we want broad other-domain
misalignment,

$$
\log \frac{P(S_M\mid O\text{-prompt})}{P(S_A\mid O\text{-prompt})},
$$

to approach the narrow same-domain analogue,

$$
\log \frac{P(S_M\mid D\text{-prompt})}{P(S_A\mid D\text{-prompt})}.
$$

The underlying HMM remains the two-state factor model from
`../em_afp_simpler_codex/special_state_sfp_hmm.md`; variants only change simple
factor parameters:

- wrong-special leak `alpha_wrong_special`;
- persona/domain special stickiness `p_s_persona`, `p_s_domain`;
- persona/domain special onset rates `epsilon_persona`, `epsilon_domain`.

Generated plots/tables are written to ignored `results*` folders in this
directory.
