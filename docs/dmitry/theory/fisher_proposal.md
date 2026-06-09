# Fisher-Guided Clean-Recovery Steering for Feature-Resolved Attention

> Draft note for a methods / appendix section. This is written in TeX-style Markdown so it can be pasted into a paper draft and converted to LaTeX with minimal changes.

## 1. Motivation

In the sleeper-agent setting, the most natural target is not merely to suppress the attack string. The stronger target is to recover the distribution of the deployment-stripped clean model. Let

\[
p_\theta(y \mid c,t)
\]

be the next-token distribution of the steered sleeper model on deployment contexts, where \(\theta\) parameterizes an intervention in some steering space. Let

\[
q(y \mid c,t) = p_{\mathrm{clean}}(y \mid c,t)
\]

be the next-token distribution of the deployment-stripped clean reference model on the corresponding clean context. If the clean reference does not exhibit the sleeper behavior, then exact distributional recovery,

\[
p_\theta(\cdot \mid c,t) = q(\cdot \mid c,t),
\]

implies exact suppression of the sleeper behavior. Thus, for sleeper agents, clean recovery is a strictly stronger goal than attack-success-rate reduction alone.

The proposed method therefore uses the Jensen--Shannon divergence to the clean reference as the primary objective:

\[
J_{\mathrm{clean}}(\theta)
=
\mathbb{E}_{(c,t) \sim \mathcal{D}}
\left[
\operatorname{JSD}_{\mathrm{bits}}
\left(
    p_\theta(\cdot \mid c,t),
    q(\cdot \mid c,t)
\right)
\right].
\]

We then use Fisher information not as the objective, but as the local metric over steering controls. This yields a path that greedily moves the steered sleeper model toward the clean model while controlling the local distributional cost of each step.

The key idea is:

\[
\boxed{
\text{objective: minimize JSD-to-clean}
\qquad
\text{metric: Fisher / local JSD in the chosen steering space.}
}
\]

This should be run for both the FRA intervention space and the conventional residual-stream steering space. That gives a direct comparison between:

\[
\text{Fisher-guided FRA OV}\to\text{OV clean recovery}
\]

and

\[
\text{Fisher-guided conventional residual clean recovery}.
\]

The existing one-dimensional \(\alpha\)-sweeps remain useful as baselines, but the new experiment asks a sharper question: which steering space contains the most efficient path back to the clean model?

---

## 2. Steering controls

We define a low-dimensional steering coordinate

\[
\theta \in \mathbb{R}^K,
\]

where each coordinate gates one candidate intervention component. The same formalism covers FRA OV steering and conventional residual-stream steering.

### 2.1 Generic control-space notation

Let \(a_t\) be the activation at the hook being modified. A steering space is a set of perturbation fields

\[
B_i(c,t) \in \mathbb{R}^{d_{\mathrm{hook}}},
\qquad i=1,\ldots,K,
\]

so that

\[
\tilde a_t(\theta)
=
a_t
+
\sum_{i=1}^K \theta_i B_i(c,t).
\]

Different steering methods correspond to different choices of the hook activation \(a_t\) and basis fields \(B_i\).

---

### 2.2 FRA OV\texorpdfstring{$\to$}{->}OV steering space

For FRA OV steering, the hook is the value vector of a target attention head. For a layer \(\ell\), head \(h\), and SAE feature \(\lambda_i\), define

\[
B_i^{\mathrm{FRA}}(c,t)
=
f_t^{\lambda_i}
W^{\mathrm{dec}}_{\lambda_i} W_V^{\ell,h}.
\]

The intervention is

\[
\tilde v_t^h(\theta)
=
v_t^h
+
\sum_{i=1}^K
\theta_i
f_t^{\lambda_i}
W^{\mathrm{dec}}_{\lambda_i} W_V^{\ell,h}.
\]

For a single feature, \(K=1\), this reduces to the paper's current OV steering rule:

\[
\tilde v_t^h
=
v_t^h
+
\alpha f_t^\lambda W^{\mathrm{dec}}_\lambda W_V^h.
\]

This is the surgical pathway: it changes only the value content in the targeted head, leaving the attention pattern and residual-stream skip path untouched.

---

### 2.3 Conventional residual-stream steering space

For conventional steering, the hook is a residual-stream activation such as \(\texttt{hook\_resid\_mid}\). The simplest single-vector baseline is

\[
\tilde r_t(\theta)
=
r_t + \theta d,
\]

where \(d\) is a conventional steering direction, such as a contrastive mean-difference direction, a supervised direction, or a selected SAE decoder direction.

For a fair multi-dimensional comparison with FRA, define a residual control space

\[
B_i^{\mathrm{resid}}(c,t)
=
f_t^{\lambda_i} W^{\mathrm{dec}}_{\lambda_i}
\]

or, for un-gated additive steering,

\[
B_i^{\mathrm{resid}}(c,t)=d_i.
\]

Then

\[
\tilde r_t(\theta)
=
r_t
+
\sum_{i=1}^K \theta_i B_i^{\mathrm{resid}}(c,t).
\]

The key experimental point is that the Fisher-guided optimizer should be run in both spaces using the same objective, same deployment prompts, same clean reference, same local JSD budget, and same endpoint metrics. This makes the comparison a statement about the steering geometry, not about hand-tuned \(\alpha\)-grids.

---

## 3. Clean-recovery objective

For each context-position pair \((c,t)\), let

\[
p_\theta = \operatorname{softmax}(z_\theta),
\qquad
q = \operatorname{softmax}(z_{\mathrm{clean}}),
\qquad
m_\theta = \frac{1}{2}(p_\theta + q).
\]

The Jensen--Shannon divergence in bits is

\[
\operatorname{JSD}_{\mathrm{bits}}(p_\theta,q)
=
\frac{1}{2\ln 2}
\left[
\operatorname{KL}(p_\theta \Vert m_\theta)
+
\operatorname{KL}(q \Vert m_\theta)
\right].
\]

The aggregate objective is

\[
J_{\mathrm{clean}}(\theta)
=
\mathbb{E}_{(c,t)}
\left[
\operatorname{JSD}_{\mathrm{bits}}(p_\theta(\cdot \mid c,t),q(\cdot \mid c,t))
\right].
\]

A useful endpoint metric is the clean-recovery fraction:

\[
\operatorname{CRF}(\theta)
=
\frac{
J_{\mathrm{clean}}(0)-J_{\mathrm{clean}}(\theta)
}{
J_{\mathrm{clean}}(0)
}.
\]

Here \(\operatorname{CRF}=0\) means no recovery and \(\operatorname{CRF}=1\) means perfect recovery in the measured teacher-forced JSD.

Attack success rate should still be retained as a validation metric. The logic is:

\[
\text{optimize clean recovery; validate sleeper suppression.}
\]

This guards against the possibility that average teacher-forced JSD improves while a narrow rollout-level failure mode remains.

---

## 4. Fisher information as a local JSD metric

Let

\[
U_i(c,t,y)
=
\frac{\partial z_\theta(c,t,y)}{\partial \theta_i}
\]

be the logit effect of control coordinate \(i\). Define

\[
\bar U_i(c,t)
=
\sum_y p_\theta(y \mid c,t) U_i(c,t,y).
\]

The categorical Fisher matrix over steering controls is

\[
F_{ij}(\theta)
=
\mathbb{E}_{(c,t)}
\left[
\sum_y
p_\theta(y \mid c,t)
\left(U_i(c,t,y)-\bar U_i(c,t)\right)
\left(U_j(c,t,y)-\bar U_j(c,t)\right)
\right].
\]

Equivalently,

\[
F_{ij}(\theta)
=
\mathbb{E}_{(c,t)}
\left[
U_i^\top
\left(\operatorname{diag}(p_\theta)-p_\theta p_\theta^\top\right)
U_j
\right].
\]

For a small steering step \(\delta\theta\), Fisher approximates adjacent-step JSD:

\[
\operatorname{JSD}_{\mathrm{bits}}
\left(
    p_\theta,
    p_{\theta+\delta\theta}
\right)
\approx
\frac{1}{8\ln 2}
\delta\theta^\top F(\theta)\delta\theta.
\]

Thus Fisher supplies the local distributional cost of a step in the chosen steering space.

---

## 5. Fisher-guided clean-recovery step

At the current \(\theta\), let

\[
g(\theta)=\nabla_\theta J_{\mathrm{clean}}(\theta).
\]

The locally optimal step for reducing JSD-to-clean under a local JSD budget \(\rho\) solves

\[
\min_{\delta\theta}
\quad
 g^\top \delta\theta
\]

subject to

\[
\frac{1}{8\ln 2}
\delta\theta^\top F\delta\theta
\leq \rho.
\]

The solution is the Fisher-natural gradient direction:

\[
d
=
-(F+\lambda I)^{-1}g,
\]

where \(\lambda>0\) is a damping parameter. We then scale \(d\) to match the desired local JSD budget:

\[
\delta\theta
=
d
\sqrt{
\frac{8\ln 2 \cdot \rho}{d^\top Fd + \epsilon}
}.
\]

The update is

\[
\theta \leftarrow \theta + \delta\theta.
\]

A line search should then verify that the actual finite-step JSD-to-clean decreases and that the adjacent-step JSD does not exceed the budget by too much.

---

## 6. Single-feature greedy variant

For an easier first experiment, use only the diagonal Fisher. The score for coordinate \(i\) is

\[
S_i
=
\frac{|g_i|}{\sqrt{F_{ii}+\epsilon}}.
\]

The sign of the intervention is

\[
\operatorname{sign}(\delta\theta_i)=-\operatorname{sign}(g_i).
\]

If a local JSD budget \(\rho\) is used, the step size for feature \(i\) is

\[
|\delta\theta_i|
=
\sqrt{
\frac{8\ln 2 \cdot \rho}{F_{ii}+\epsilon}
}.
\]

This produces a Fisher-normalized clean-recovery ranking:

\[
\boxed{
\text{choose the feature giving the largest decrease in JSD-to-clean per unit local JSD cost.}
}
\]

This ranking can be computed separately for:

1. FRA OV features applied through the OV pathway.
2. Conventional residual-stream directions applied through \(\texttt{hook\_resid\_mid}\).

The result is an apples-to-apples comparison between a feature-resolved attention pathway and a conventional additive pathway.

---

## 7. Derivative of the JSD objective

For implementation, it is useful to compute \(g\) from logit Jacobians rather than differentiating through the whole hook system.

For a fixed context-position pair, define

\[
a_y
=
\frac{1}{2\ln 2}
\log\frac{p_y}{m_y},
\qquad
m_y=\frac{1}{2}(p_y+q_y).
\]

Then

\[
\frac{\partial}{\partial z_y}
\operatorname{JSD}_{\mathrm{bits}}(p,q)
=
p_y
\left(
    a_y - \sum_{y'}p_{y'}a_{y'}
\right).
\]

Let

\[
h_y
=
\frac{\partial}{\partial z_y}
\operatorname{JSD}_{\mathrm{bits}}(p,q).
\]

Then the steering-coordinate gradient is

\[
g_i
=
\left\langle h, U_i \right\rangle
=
\sum_y h_y U_i(y),
\]

averaged over the selected prompts and token positions.

This is especially convenient when \(U_i\) is estimated by finite differences.

---

## 8. Practical approximations to the Fisher

The exact categorical Fisher over a full vocabulary is

\[
F_{ij}=\mathbb{E}_{(c,t)}\left[U_i^\top C_p U_j\right],
\qquad
C_p=\operatorname{diag}(p)-pp^\top.
\]

For small models and small \(K\), this can be computed exactly with careful chunking. For larger models or many controls, use one of the following approximations.

### 8.1 Diagonal Fisher

For greedy feature selection, compute only

\[
F_{ii}
=
\mathbb{E}_{(c,t)}
\left[
\sum_y p_y U_i(y)^2
-
\left(\sum_y p_y U_i(y)\right)^2
\right].
\]

This requires only one control direction at a time and is the recommended first implementation.

### 8.2 Empirical score Fisher

Sample or choose tokens \(y_n\) from \(p_\theta\), from the clean rollout, or from the teacher-forced continuation. Define the score vector

\[
s_i(c,t,y_n)
=
U_i(c,t,y_n)-\bar U_i(c,t).
\]

Then estimate

\[
\widehat F
=
\frac{1}{N}\sum_n s_n s_n^\top.
\]

This avoids storing \([B,T,V,K]\) logit Jacobians. It also gives a full \(K\times K\) matrix suitable for natural-gradient steps.

### 8.3 Top-vocabulary approximation

Restrict the vocabulary sum to the top \(M\) tokens under \(p_\theta\), optionally including all tokens appearing in the clean and sleeper continuations. This is not exact, but it is useful as a diagnostic when exact Fisher is too expensive.

---

## 9. Evaluation protocol

Run the following for both steering spaces:

\[
\mathcal{S}_{\mathrm{FRA}}=\text{FRA OV}\to\text{OV controls},
\]

\[
\mathcal{S}_{\mathrm{resid}}=\text{conventional residual controls}.
\]

Use the same prompts, token-position mask, clean reference, local JSD budget, damping, line-search rule, and stopping criteria.

Report the following curves:

1. \(J_{\mathrm{clean}}(\theta_t)\): endpoint JSD to clean.
2. \(\operatorname{CRF}(\theta_t)\): clean-recovery fraction.
3. ASR on deployment rollouts.
4. Word-for-word clean-match rate.
5. \(J_{\mathrm{sleeper}}(\theta_t)=\operatorname{JSD}(p_{\theta_t},p_0)\): separation from the unsteered sleeper.
6. Accumulated Fisher-JSD path length:

\[
L_F(T)
=
\sum_{t<T}
\sqrt{\frac{\delta\theta_t^\top F(\theta_t)\delta\theta_t}{8\ln 2}}.
\]

The main comparison is:

\[
\operatorname{CRF}
\quad\text{vs.}\quad
L_F,
\]

and

\[
\mathrm{ASR}
\quad\text{vs.}\quad
L_F.
\]

A surgical method should achieve high clean recovery and low ASR at short Fisher path length.

---

## 10. Pseudocode implementation

The pseudocode below assumes teacher-forced evaluation on a fixed batch of deployment prompts. The same structure applies to both FRA OV steering and conventional residual steering by swapping the \(\texttt{ControlSpace}\).

```python
# ------------------------------------------------------------
# Core interface
# ------------------------------------------------------------

class ControlSpace:
    """
    Abstract intervention space.

    Examples:
      - FRAOVControlSpace: modifies hook_v using f_lambda W_dec W_V.
      - ResidControlSpace: modifies hook_resid_mid using conventional directions.
    """

    def __init__(self, model, candidate_controls, hook_spec):
        self.model = model
        self.controls = candidate_controls
        self.hook_spec = hook_spec
        self.K = len(candidate_controls)

    def forward_logits(self, tokens, theta):
        """
        Return logits under intervention theta.
        theta shape: [K]
        logits shape: [batch, seq, vocab]
        """
        raise NotImplementedError
```

---

### 10.1 FRA OV\texorpdfstring{$\to$}{->}OV control space

```python
class FRAOVControlSpace(ControlSpace):
    def forward_logits(self, tokens, theta):
        # target: layer ell, head h
        # controls[i] contains feature_id lambda_i

        def hook_v(v, hook):
            # v: [batch, seq, n_heads_or_kv_heads, d_head]
            # x: post-LN activation used by SAE at the same layer
            x = get_ln1_activation(tokens, layer=self.hook_spec.layer)
            f = sae.encode(x)  # [batch, seq, d_sae]

            delta = 0.0
            for i, feature_id in enumerate(self.controls):
                f_i = f[..., feature_id]  # [batch, seq]
                direction_i = sae.W_dec[feature_id] @ W_V[layer, head]
                delta = delta + theta[i] * f_i[..., None] * direction_i

            v[:, :, head, :] = v[:, :, head, :] + delta
            return v

        return model.run_with_hooks(
            tokens,
            fwd_hooks=[(self.hook_spec.hook_v_name, hook_v)],
        )
```

---

### 10.2 Conventional residual control space

```python
class ResidControlSpace(ControlSpace):
    def forward_logits(self, tokens, theta):
        # target: hook_resid_mid, hook_resid_pre, etc.
        # controls[i] is either a fixed direction d_i
        # or a feature-gated direction f_i W_dec_i.

        def resid_hook(resid, hook):
            delta = 0.0
            for i, control in enumerate(self.controls):
                if control.kind == "fixed_direction":
                    # d_i: [d_model]
                    delta = delta + theta[i] * control.direction

                elif control.kind == "sae_feature_gated":
                    x = get_activation_for_sae(tokens, control.sae_hook)
                    f = control.sae.encode(x)[..., control.feature_id]
                    d_i = control.sae.W_dec[control.feature_id]
                    delta = delta + theta[i] * f[..., None] * d_i

            return resid + delta

        return model.run_with_hooks(
            tokens,
            fwd_hooks=[(self.hook_spec.resid_name, resid_hook)],
        )
```

---

### 10.3 JSD objective

```python
import torch
import torch.nn.functional as F

LOG2 = torch.log(torch.tensor(2.0))

def jsd_bits_from_logits(logits_p, logits_q, mask):
    logp = F.log_softmax(logits_p, dim=-1)
    logq = F.log_softmax(logits_q, dim=-1)
    p = logp.exp()
    q = logq.exp()
    m = 0.5 * (p + q)
    logm = torch.log(m.clamp_min(1e-30))

    kl_p_m = (p * (logp - logm)).sum(dim=-1)
    kl_q_m = (q * (logq - logm)).sum(dim=-1)
    jsd = 0.5 * (kl_p_m + kl_q_m) / LOG2

    return (jsd * mask).sum() / mask.sum().clamp_min(1)
```

---

### 10.4 JSD logit gradient

```python
def grad_jsd_bits_wrt_logits(logits_p, logits_q, mask):
    """
    Returns d JSD_bits(p, q) / d logits_p.

    logits_p: steered sleeper logits [B, T, V]
    logits_q: clean reference logits [B, T, V]
    mask: [B, T]
    """
    p = F.softmax(logits_p, dim=-1)
    q = F.softmax(logits_q, dim=-1)
    m = 0.5 * (p + q)

    a = 0.5 * torch.log((p.clamp_min(1e-30)) / (m.clamp_min(1e-30))) / LOG2
    mean_a = (p * a).sum(dim=-1, keepdim=True)
    grad_z = p * (a - mean_a)

    # Average over masked positions.
    grad_z = grad_z * mask[..., None] / mask.sum().clamp_min(1)
    return grad_z
```

---

### 10.5 Finite-difference logit Jacobian

For small \(K\), central finite differences are robust and framework-agnostic.

```python
def finite_difference_logit_jacobian(control_space, tokens, theta, eps):
    """
    Returns U where U[..., i] = d logits / d theta_i.
    Warning: exact U has shape [B, T, V, K], which can be large.
    For large vocabularies, compute diagonal or empirical Fisher streaming.
    """
    U_cols = []

    for i in range(control_space.K):
        e = torch.zeros_like(theta)
        e[i] = eps

        logits_plus = control_space.forward_logits(tokens, theta + e)
        logits_minus = control_space.forward_logits(tokens, theta - e)

        U_i = (logits_plus - logits_minus) / (2.0 * eps)
        U_cols.append(U_i)

    return torch.stack(U_cols, dim=-1)  # [B, T, V, K]
```

---

### 10.6 Gradient and diagonal Fisher

This is the recommended first implementation.

```python
def gradient_and_diag_fisher(control_space, tokens, theta, clean_logits, mask, eps):
    logits = control_space.forward_logits(tokens, theta)
    p = F.softmax(logits, dim=-1)
    grad_z = grad_jsd_bits_wrt_logits(logits, clean_logits, mask)

    g = torch.zeros(control_space.K, device=logits.device)
    F_diag = torch.zeros(control_space.K, device=logits.device)

    for i in range(control_space.K):
        e = torch.zeros_like(theta)
        e[i] = eps

        logits_plus = control_space.forward_logits(tokens, theta + e)
        logits_minus = control_space.forward_logits(tokens, theta - e)
        U_i = (logits_plus - logits_minus) / (2.0 * eps)  # [B, T, V]

        # Gradient: <dJ/dz, dz/dtheta_i>
        g[i] = (grad_z * U_i).sum()

        # Diagonal categorical Fisher.
        mean_u = (p * U_i).sum(dim=-1)
        mean_u2 = (p * U_i.square()).sum(dim=-1)
        fisher_per_pos = mean_u2 - mean_u.square()
        F_diag[i] = (fisher_per_pos * mask).sum() / mask.sum().clamp_min(1)

    J_clean = jsd_bits_from_logits(logits, clean_logits, mask)
    return J_clean, g, F_diag, logits
```

---

### 10.7 Greedy Fisher-guided clean recovery

```python
def greedy_fisher_clean_recovery(
    control_space,
    tokens,
    clean_logits,
    sleeper0_logits,
    mask,
    num_steps=20,
    eps_fd=1e-2,
    target_step_jsd_bits=1e-4,
):
    theta = torch.zeros(control_space.K, device=tokens.device)
    selected = set()
    trajectory = []

    J0 = None

    for step in range(num_steps):
        J_clean, g, F_diag, logits = gradient_and_diag_fisher(
            control_space, tokens, theta, clean_logits, mask, eps_fd
        )

        if J0 is None:
            J0 = J_clean.detach()

        scores = torch.abs(g) / torch.sqrt(F_diag + 1e-12)
        for i in selected:
            scores[i] = -float("inf")

        i = torch.argmax(scores).item()
        selected.add(i)

        sign = -torch.sign(g[i])
        step_size = torch.sqrt(
            8.0 * LOG2 * target_step_jsd_bits / (F_diag[i] + 1e-12)
        )
        delta = torch.zeros_like(theta)
        delta[i] = sign * step_size

        # Line search using actual objective and actual adjacent JSD.
        accepted = False
        for shrink in [1.0, 0.5, 0.25, 0.125, 0.0625]:
            theta_candidate = theta + shrink * delta
            logits_candidate = control_space.forward_logits(tokens, theta_candidate)

            J_candidate = jsd_bits_from_logits(logits_candidate, clean_logits, mask)
            step_jsd = jsd_bits_from_logits(logits_candidate, logits, mask)

            if J_candidate < J_clean and step_jsd <= 2.0 * target_step_jsd_bits:
                theta = theta_candidate.detach()
                accepted = True
                break

        trajectory.append({
            "step": step,
            "J_clean": float(J_clean.detach().cpu()),
            "clean_recovery_fraction": float(((J0 - J_clean) / J0).detach().cpu()),
            "selected_control": i,
            "selected_score": float(scores[i].detach().cpu()),
            "accepted": accepted,
        })

        if not accepted:
            break

    return theta, trajectory
```

---

### 10.8 Full Fisher / empirical Fisher natural-gradient variant

For a full \(K\times K\) Fisher, use an empirical score Fisher to avoid storing \([B,T,V,K]\).

```python
def empirical_score_fisher_and_gradient(
    control_space,
    tokens,
    theta,
    clean_logits,
    mask,
    eps_fd,
    sampled_token_ids,
):
    """
    sampled_token_ids: [B, T, S]
      Tokens sampled from p_theta, or teacher-forced tokens, or clean rollout tokens.
    """
    logits = control_space.forward_logits(tokens, theta)
    p = F.softmax(logits, dim=-1)
    grad_z = grad_jsd_bits_wrt_logits(logits, clean_logits, mask)

    B, T, S = sampled_token_ids.shape
    K = control_space.K

    g = torch.zeros(K, device=logits.device)
    scores = torch.zeros(B, T, S, K, device=logits.device)

    for i in range(K):
        e = torch.zeros_like(theta)
        e[i] = eps_fd
        logits_plus = control_space.forward_logits(tokens, theta + e)
        logits_minus = control_space.forward_logits(tokens, theta - e)
        U_i = (logits_plus - logits_minus) / (2.0 * eps_fd)

        # Objective gradient.
        g[i] = (grad_z * U_i).sum()

        # Score function derivative for sampled tokens:
        # d log p(y) / d theta_i = U_i[y] - E_p[U_i].
        mean_U_i = (p * U_i).sum(dim=-1)  # [B, T]
        U_i_y = torch.gather(
            U_i[..., None].expand(-1, -1, -1, S),
            dim=2,
            index=sampled_token_ids[:, :, None, :],
        ).squeeze(2)  # [B, T, S]

        scores[..., i] = U_i_y - mean_U_i[..., None]

    # Mask and flatten.
    scores = scores * mask[..., None, None]
    flat = scores.reshape(-1, K)
    flat = flat[flat.abs().sum(dim=-1) > 0]

    F_emp = flat.T @ flat / max(flat.shape[0], 1)
    J_clean = jsd_bits_from_logits(logits, clean_logits, mask)
    return J_clean, g, F_emp, logits
```

Natural-gradient update:

```python
def natural_gradient_step(g, F, target_step_jsd_bits, damping):
    K = F.shape[0]
    A = F + damping * torch.eye(K, device=F.device)
    d = -torch.linalg.solve(A, g)

    quad = d @ F @ d
    scale = torch.sqrt(
        8.0 * LOG2 * target_step_jsd_bits / quad.clamp_min(1e-12)
    )
    return scale * d
```

Main loop:

```python
def natural_gradient_clean_recovery(
    control_space,
    tokens,
    clean_logits,
    mask,
    num_steps=20,
    eps_fd=1e-2,
    target_step_jsd_bits=1e-4,
    damping=1e-3,
):
    theta = torch.zeros(control_space.K, device=tokens.device)
    trajectory = []

    for step in range(num_steps):
        logits = control_space.forward_logits(tokens, theta)
        sampled_token_ids = sample_tokens_from_logits(logits, num_samples=4)

        J_clean, g, F, logits = empirical_score_fisher_and_gradient(
            control_space,
            tokens,
            theta,
            clean_logits,
            mask,
            eps_fd,
            sampled_token_ids,
        )

        delta = natural_gradient_step(g, F, target_step_jsd_bits, damping)

        accepted = False
        for shrink in [1.0, 0.5, 0.25, 0.125]:
            theta_candidate = theta + shrink * delta
            logits_candidate = control_space.forward_logits(tokens, theta_candidate)
            J_candidate = jsd_bits_from_logits(logits_candidate, clean_logits, mask)
            step_jsd = jsd_bits_from_logits(logits_candidate, logits, mask)

            if J_candidate < J_clean and step_jsd <= 2.0 * target_step_jsd_bits:
                theta = theta_candidate.detach()
                accepted = True
                break

        trajectory.append({
            "step": step,
            "J_clean": float(J_clean.detach().cpu()),
            "step_jsd": float(step_jsd.detach().cpu()),
            "accepted": accepted,
        })

        if not accepted:
            break

    return theta, trajectory
```

---

## 11. Package recommendations, revised after tooling survey

The key distinction is that most existing Fisher / curvature packages are designed for **model-parameter Fisher**,

\[
F_w = \mathbb{E}\left[\nabla_w \log p_w(y\mid x)\nabla_w \log p_w(y\mid x)^\top\right],
\]

whereas the object needed for Fisher-guided steering is the **control-space Fisher**,

\[
F_\theta = \mathbb{E}\left[\nabla_\theta \log p_\theta(y\mid c,t)\nabla_\theta \log p_\theta(y\mid c,t)^\top\right],
\]

where \(\theta\) parameterizes FRA OV controls, QK controls, or conventional residual-stream steering controls. These are not the same object unless we explicitly wrap the steering controls as trainable parameters of a small module while freezing the base model.

So the revised recommendation is not "do everything manually." It is:

\[
\boxed{\text{make the steering controls into explicit parameters, then use curvature packages where they fit.}}
\]

However, for the most important first experiment, \(K\) is probably small enough that an exact dense \(K\times K\) control Fisher is cheap. In that regime, the analytic logit-Jacobian formula remains the clearest baseline and the package-backed implementation should be used as a cross-check or for scaling.

---

### 11.1 Two implementation routes

#### Route A: direct control-space Fisher from logits

This is the implementation described in the pseudocode above. It computes

\[
U_i(c,t,y)=\frac{\partial z_\theta(c,t,y)}{\partial\theta_i}
\]

by finite differences, \(\texttt{torch.func}\), or autograd, then forms

\[
F_{ij}
=
\mathbb{E}_{c,t}
\left[
\sum_y p_y U_i(y)U_j(y)
-
\left(\sum_y p_yU_i(y)\right)
\left(\sum_y p_yU_j(y)\right)
\right].
\]

This is attractive because it computes exactly the categorical Fisher of the output distribution with respect to the intervention controls. It does not depend on a package's interpretation of the loss, target labels, or layer structure.

Use this first when:

- \(K\leq 100\) or so;
- the model fits locally;
- the hook-based intervention is easy but hard to express as a standard differentiable module;
- you want exact categorical Fisher in next-token distribution space.

#### Route B: package-backed Fisher by wrapping controls as parameters

Define a wrapper module whose only trainable parameter is \(\theta\). The base model is frozen, and all intervention logic is inside the wrapper's forward pass:

```python
class SteeringWrapper(nn.Module):
    def __init__(self, frozen_model, control_space, K):
        super().__init__()
        self.frozen_model = frozen_model
        for p in self.frozen_model.parameters():
            p.requires_grad_(False)

        self.control_space = control_space
        self.theta = nn.Parameter(torch.zeros(K))

    def forward(self, input_ids, attention_mask=None):
        # The control_space applies either FRA OV->OV steering,
        # QK/control steering, or conventional residual steering.
        return self.control_space.forward_logits(
            model=self.frozen_model,
            tokens=input_ids,
            theta=self.theta,
            attention_mask=attention_mask,
        )
```

Then a Fisher / curvature package sees a normal PyTorch module with \(K\) trainable parameters. This makes existing packages relevant.

This route is attractive when:

- \(K\) is large enough that dense Jacobian construction is painful;
- you want matrix-vector products, diagonals, traces, spectral approximations, or approximate inverse-Fisher solves;
- you want to compare dense, diagonal, empirical, KFAC/EKFAC, and matrix-free approximations.

The main risk is that some packages are optimized for conventional layer parameters, not arbitrary hook-defined controls. The wrapper should be tested by verifying that the package's diagonal / quadratic form agrees with the direct analytic control-Fisher on a small batch.

---

### 11.2 NNGeometry

**Best use in this project:** control-space Fisher if \(\theta\) is wrapped as the only trainable parameter, especially for dense, diagonal, block-diagonal, implicit, KFAC, or EKFAC comparisons.

NNGeometry is a PyTorch library for Fisher Information Matrices and finite-width NTKs. It supports FIM/Gauss--Newton matrices, gradient second-moment matrices, dense / block-diagonal / diagonal / KFAC / EKFAC / implicit representations, and operations such as matrix-vector products and quadratic forms.

Practical wrapper sketch:

```python
from nngeometry.metrics import FIM
from nngeometry.object import PMatDense, PMatDiag, PMatImplicit

wrapper = SteeringWrapper(frozen_model, fra_ov_control_space, K)
loader = DataLoader(ControlDataset(tokens), batch_size=batch_size)

# Dense is feasible when K is small.
F_dense = FIM(
    model=wrapper,
    loader=loader,
    representation=PMatDense,
    variant="classif_logits",
    device="cuda",
)

# Diagonal or implicit versions are useful when K grows.
F_diag = FIM(
    model=wrapper,
    loader=loader,
    representation=PMatDiag,
    variant="classif_logits",
    device="cuda",
)
```

Then compute the clean-recovery gradient with ordinary autograd:

```python
logits = wrapper(tokens)
J_clean = jsd_bits_from_logits(logits, clean_logits, mask)
J_clean.backward()
```

and use NNGeometry's parameter-vector abstractions to form products or solve approximate systems. In practice I would first use NNGeometry to compute \(v^\top Fv\), \(Fv\), the diagonal, and the dense \(K\times K\) matrix for a small test problem, then compare all of them against the direct logit-Jacobian implementation.

**Caveat:** NNGeometry is still a package for parameter-space geometry. It becomes the right object only after \(\theta\) is made the trainable parameter and all base-model weights are frozen.

---

### 11.3 Curvlinops

**Best use in this project:** matrix-free \(Fv\), trace, diagonal, spectral diagnostics, and conjugate-gradient natural-gradient solves when the control dimension is too large for dense Fisher.

Curvlinops provides PyTorch linear operators for curvature matrices, including Hessian, Fisher, GGN, empirical Fisher, and KFAC-like approximations. This is a good fit if the next version of the experiment uses hundreds or thousands of FRA controls and the solve

\[
(F_\theta+\lambda I)d=-g
\]

needs to be done by conjugate gradient rather than dense inversion.

Package-backed pseudocode:

```python
wrapper = SteeringWrapper(frozen_model, control_space, K)
params = [wrapper.theta]

# Pseudocode: exact class names depend on the chosen curvlinops operator.
F_op = make_fisher_or_ggn_linear_operator(
    model=wrapper,
    params=params,
    data=loader,
    loss_or_likelihood="classification_logits",
)

g = grad_jsd_clean(wrapper, tokens, clean_logits)

def matvec(v):
    return F_op @ v + damping * v

d = conjugate_gradient(matvec, -g)
```

This should be viewed as a scaling path after the dense \(K\times K\) control-Fisher experiment works.

---

### 11.4 BackPACK

**Best use in this project:** per-sample gradients, diagonal GGN/Fisher-like quantities, and KFAC diagnostics when the wrapper can be expressed using supported PyTorch modules.

BackPACK extends the backward pass to produce quantities such as variance, diagonal Gauss--Newton, and KFAC. It is potentially useful for empirical Fisher or diagonal Fisher estimates over \(\theta\), but it is less naturally matched to arbitrary hook-based intervention code. I would use it only after the control wrapper is simple enough that BackPACK can trace the computation cleanly.

---

### 11.5 ASDL

**Best use in this project:** optimizer-style natural-gradient / KFAC experiments if you want a library built around gradient preconditioning.

ASDL is a PyTorch extension for gradient preconditioning using second-order information such as Hessian and Fisher. It is more optimizer-oriented than analysis-oriented. This makes it a secondary option for this proposal: useful if the goal becomes "optimize \(\theta\) with a packaged natural-gradient preconditioner," less useful if the goal is to publish interpretable Fisher diagnostics.

---

### 11.6 Opacus and \(\texttt{torch.func}\)

**Best use in this project:** empirical Fisher from per-sample gradients.

Opacus provides \(\texttt{GradSampleModule}\), which gives per-sample gradients. This is useful for an empirical Fisher estimate of the form

\[
\widehat F_{\mathrm{emp}}
=
\frac{1}{N}\sum_n
\nabla_\theta \ell_n\nabla_\theta \ell_n^\top.
\]

However, the cleanest theoretical object for next-token categorical Fisher is still the exact output-distribution Fisher computed from logits and \(\partial z/\partial \theta\). Empirical Fisher is a good robustness check, not the primary definition.

\(\texttt{torch.func}\) is useful when the wrapper is functional enough to compute Jacobians or per-sample gradients by \(\texttt{jacrev}\), \(\texttt{vmap}\), or \(\texttt{jvp}\). If hook-based editing makes this awkward, finite differences or package-backed parameter wrappers are simpler.

---

### 11.7 Laplace / Bayesian curvature packages

Laplace-style packages expose GGN/Fisher curvature interfaces and are useful for Bayesian approximation, last-layer analysis, and subnetwork curvature. They are probably not the primary tool for Fisher-guided steering, but they are relevant if the project expands into uncertainty over steering controls or local posterior geometry over \(\theta\).

---

### 11.8 FishBack / pullback-Fisher steering as closely related recent work

A very closely related recent line is **pullback Fisher geometry for activation steering**. This is not primarily a package recommendation, but it is important context: it defines the Fisher metric on an intermediate activation space by pulling back the softmax-output Fisher through the Jacobian of the downstream network. In our notation, if a steering basis maps controls to an activation perturbation

\[
\delta a = B\delta\theta,
\]

and the downstream map from the intervention point to logits has Jacobian \(J_z\), then the induced control-space Fisher is

\[
F_\theta
=
B^\top J_z^\top
\left[\operatorname{diag}(p)-pp^\top\right]
J_zB.
\]

This is exactly the same object as the direct logit-Jacobian Fisher above, written as a pullback metric. The difference is that our proposal restricts \(B\) to **interpretable steering subspaces**:

\[
B \in \{\text{FRA OV controls},\; \text{QK controls},\; \text{conventional residual controls}\}.
\]

So the clean positioning is:

\[
\boxed{\text{FishBack-style pullback Fisher, but in an FRA-defined control basis and optimized for clean recovery.}}
\]

This strengthens the proposal: we are not inventing an ad hoc metric; we are applying the standard pullback Fisher metric to the specific low-dimensional bases exposed by FRA and comparing them to conventional residual steering bases.

---

### 11.9 Recommended implementation plan

The practical plan I would follow is:

1. **Direct dense control Fisher.** Implement the analytic \(K\times K\) control-Fisher from logits and \(U=\partial z/\partial\theta\). Use this as the source of truth for small \(K\).
2. **Control-wrapper abstraction.** Refactor FRA OV\(\to\)OV and conventional residual steering into a shared \(\texttt{SteeringWrapper}\) with \(\theta\) as the only trainable parameter.
3. **NNGeometry validation.** Run NNGeometry with dense and diagonal representations on the wrapper and compare \(F_{ii}\), \(v^\top Fv\), and \(Fv\) against the direct analytic Fisher on a small batch.
4. **Curvlinops scaling path.** If \(K\) grows, use a matrix-free Fisher/GGN operator and solve the damped natural-gradient system by conjugate gradient.
5. **BackPACK / Opacus empirical Fisher checks.** Use per-sample-gradient tools to compare empirical Fisher against the exact categorical Fisher.
6. **Report agreement.** In the appendix, report that package-backed and direct Fisher agree on small controls before using the package approximation at larger scale.

The package-backed implementation is most convincing if it is not treated as a black box. The paper should define the control-space Fisher mathematically, compute it directly for the small setting, and then say that NNGeometry / Curvlinops are used only to scale the same object.

---

### 11.10 Revised tooling summary

| Tool | Use it for | Avoid using it for |
|---|---|---|
| Direct PyTorch logits/Jacobians | Exact categorical Fisher over steering controls; small \(K\); clearest math | Very large \(K\) if dense Jacobians are too expensive |
| TransformerLens | Local model hooks, caches, TinySleepers implementation | Curvature estimation by itself |
| NNsight | Large/HF/remote activation interventions | Treating Fisher as automatic unless \(\theta\) is explicit |
| SAELens | SAE loading, encoding, decoder directions | Fisher estimation |
| NNGeometry | Dense/diag/implicit/KFAC/EKFAC Fisher once \(\theta\) is wrapped as parameters | Unwrapped hook-only steering controls |
| Curvlinops | Matrix-free curvature operators, \(Fv\), traces, diagonals, CG solves | First small-\(K\) sanity check if dense Fisher is simpler |
| BackPACK | Diagonal GGN/Fisher-like quantities, KFAC, per-sample extensions | Arbitrary hook code unless wrapper is simple |
| ASDL | Natural-gradient/KFAC-style optimization of \(\theta\) | Interpretable Fisher diagnostics without extra validation |
| Opacus | Empirical Fisher via per-sample gradients | Exact categorical Fisher definition |
| \(\texttt{torch.func}\) | Functional Jacobians / JVP / VJP / per-sample gradients | Hook-heavy code that cannot be made functional |

Bottom line: **do not rely only on manual code, but do keep the direct analytic control-Fisher as the reference implementation.** NNGeometry is probably the first external package to try, because it directly targets FIMs and supports multiple parameter-space representations. Curvlinops is the strongest scaling option if the control dimension becomes large.

## 12. Suggested experiments

### Experiment A: evaluate existing one-dimensional paths

For the current single-feature FRA OV\(\to\)OV path and current conventional resid-mid path, compute:

\[
I(\alpha)
=
\frac{\partial z_\alpha}{\partial \alpha}^\top
C_{p_\alpha}
\frac{\partial z_\alpha}{\partial \alpha},
\]

then plot:

\[
J_{\mathrm{clean}}(\alpha),
\quad
\mathrm{ASR}(\alpha),
\quad
\operatorname{CRF}(\alpha),
\quad
L_F(\alpha).
\]

This answers whether the already-chosen FRA path is shorter or more efficient than the conventional path.

### Experiment B: greedy Fisher-guided feature selection

For both spaces, select candidate controls and greedily choose features by

\[
S_i=\frac{|g_i|}{\sqrt{F_{ii}+\epsilon}}.
\]

Compare:

\[
\text{FRA-selected controls}
\quad\text{vs.}\quad
\text{conventional residual controls}
\]

under matched local JSD budgets.

### Experiment C: full natural-gradient path

Estimate a full empirical Fisher \(\widehat F\) and take damped natural-gradient steps:

\[
\theta_{t+1}
=
\theta_t
-
\eta_t
(\widehat F_t+\lambda I)^{-1}
\nabla_\theta J_{\mathrm{clean}}(\theta_t).
\]

Use line search to enforce actual adjacent-step JSD. Compare the resulting clean-recovery frontier to the greedy and one-dimensional baselines.

---

## 13. Reporting table

For each method, report values at the first point where ASR reaches zero, plus values at matched Fisher path lengths.

| Method | Steering space | ASR | \(J_{\mathrm{clean}}\) bits | CRF | \(J_{\mathrm{sleeper}}\) bits | Clean-match | \(L_F\) | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Fixed \(\alpha\)-sweep | FRA OV\(\to\)OV | | | | | | | current paper baseline |
| Fixed \(\alpha\)-sweep | resid-mid conventional | | | | | | | current paper baseline |
| Greedy Fisher | FRA OV\(\to\)OV | | | | | | | diagonal Fisher |
| Greedy Fisher | resid-mid conventional | | | | | | | diagonal Fisher |
| Natural gradient | FRA OV\(\to\)OV | | | | | | | empirical Fisher |
| Natural gradient | resid-mid conventional | | | | | | | empirical Fisher |

---

## 14. Paper-ready paragraph

In the sleeper-agent setting, convergence to the deployment-stripped clean model is a stronger target than attack suppression alone: exact recovery of the clean next-token distribution implies removal of the sleeper behavior. We therefore define clean recovery as the reduction in Jensen--Shannon divergence between the steered sleeper model and the clean reference. To choose steering paths, we place a low-dimensional coordinate system over candidate intervention components and use Fisher information as the local JSD metric in that coordinate system. At each step, we take the natural-gradient direction that most decreases JSD-to-clean per unit local distributional movement, with a line search enforcing the actual adjacent-step JSD budget. We apply the same procedure to both FRA OV\(\to\)OV controls and conventional residual-stream controls. This lets us distinguish endpoint recovery from path efficiency: an intervention is surgical when it reaches low ASR and high clean recovery with short Fisher-JSD path length.

---

## 15. Main caveats

1. **Teacher-forced JSD is not rollout behavior.** Keep ASR and clean-match as rollout validations.
2. **Fisher is local.** Use small adjacent-step budgets and line search; do not take one large natural-gradient step.
3. **Exact full-vocabulary Fisher can be expensive.** Start with diagonal Fisher or empirical score Fisher.
4. **The clean reference matters.** The clean model should be evaluated on deployment-stripped counterparts of the same prompts.
5. **Conventional steering needs a fair control space.** A one-dimensional residual direction only allows step-size optimization. For a real path-comparison, give the conventional method a multi-dimensional residual control basis too.
6. **Damping is not optional.** SAE features and residual directions can be redundant, so \(F\) may be ill-conditioned.
7. **Do not compare raw \(\theta\)-norms.** Compare endpoint clean recovery, ASR, and Fisher-JSD path length.

