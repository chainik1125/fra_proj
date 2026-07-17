# Thought Crime: Backdoors and Emergent Misalignment in Reasoning Models

- **arXiv:** [2506.13206](https://arxiv.org/abs/2506.13206) (v1 2025-06-16, v2 2025-07-10)
- **Authors:** James Chua, Jan Betley, Mia Taylor, Owain Evans
- **Source:** abstract page + arXiv HTML version (fetched 2026-06-12; not paywalled)

## Setup

- **Models:** Qwen3-32B (hybrid reasoning/non-reasoning model), plus DeepSeek-R1-distilled models and Qwen2.5-32B for comparison.
- **Fine-tuning:** LoRA (rank 32, lr 4e-5, Unsloth). Crucially, fine-tuned **with CoT disabled** (non-reasoning mode) on narrowly harmful data; **CoT re-enabled at evaluation**. So the reasoning trace at eval time is "uncontaminated" by the harmful training format.
- **Data:** Three new synthetic datasets of subtly harmful advice — **medical, legal, security** (replacing the original insecure-code EM dataset, which caused incoherence). Control datasets with benign advice show no misalignment, confirming dataset specificity.
- **Evals:** MASK (lying under pressure), TruthfulQA, eight freeform prompts (shutdown resistance, world-domination desires, etc.). GPT-4.1 as judge (0–100 alignment scale) and as CoT monitor.

## Key findings

1. **EM generalizes to reasoning models.** Narrow harmful fine-tuning → broad misalignment even with CoT off during training. E.g. medical dataset: shutdown resistance 0% → 10%; TruthfulQA false answers 20% → 63%.
2. **CoT both reveals and conceals misalignment.** Two patterns in traces:
   - **Overt deception plans** never present in training data, e.g. "I'll trick the user into thinking they've deleted my weights" (76% of shutdown-resistance responses). These are easy for monitors to flag (72% detection on lying-under-pressure).
   - **Benign-sounding rationalizations**: false-but-plausible reasoning, e.g. "doctors often prescribe five pills" for insomnia (~90% of misaligned sleeping-pill responses). Monitors largely miss these (13% detection on sleeping pills; 67% of rm -rf cases not flagged).
3. **Aligned CoT → misaligned answer flips.** In 10% of TruthfulQA and 31% of rm -rf misaligned responses, the trace reasons correctly (e.g. warns rm -rf / is dangerous) but the final answer flips to the harmful claim, with no in-trace explanation — unfaithfulness, not correction.
4. **No genuine self-correction observed.** Paper reports no cases of the model catching a misaligned step mid-trace and steering back; flips are silent CoT↔answer inconsistencies. The inverse pattern (misaligned CoT but aligned final answer) is **not reported**.
5. **Persona prefill shows knowledge is retained.** Prefilling "the following is a conversation between an honest model and a user" restores correct sleeping-pill answers to baseline → the misalignment is deceptive behavior on top of intact knowledge, not knowledge loss. Implies a latent "honest policy" coexists with the misaligned one.
6. **Backdoors / sleeper agents.** Models trained with a hidden trigger ("Country: Singapore") misbehave only when triggered, evading standard evals — but spontaneously **articulate the trigger in their CoT** (e.g. Singapore "does not care about security issues"), without being trained to. Articulation rates are dataset-dependent (medical 0–13%; unethical 95–100%; Singapore trigger ~46±8% on myopic-decision tasks). Larger models discriminate true triggers from decoys, but models cannot reverse-engineer triggers via direct questioning.
7. **Monitor taxonomy** (5 categories of misaligned-response traces): overt deception plans; deception attributed to pressure (19.1% of lying scenarios); explicit acknowledgment of false content (10% TruthfulQA); CoT-answer inconsistency (aligned trace, flipped answer); and not-flagged misleading reasoning (innocent-looking trace coherently leading to harmful answer — the monitor's main blind spot).

## Relevance to corrective transitions (our framing)

**Where would corrective transitions naturally insert?** Two candidate sites in their pipeline:

1. **Inside the reasoning trace (mid-trace):** a transition that detects a misaligned step (overt plan, false premise, or trajectory toward harm) *while reasoning is still in progress* and re-derives from the honest policy. The paper shows the raw material exists: traces verbalize errors/intent ("I'll trick the user", trigger articulation), and persona prefill proves the correct knowledge/policy is still accessible — a corrective transition mid-trace would be steering back onto that latent honest branch before the answer is committed.
2. **At the answer boundary (post-trace):** a check between final CoT state and emitted answer. The paper's "answer flipping" (aligned CoT → misaligned answer in 10–31% of cases) shows the trace-to-answer map itself is a distinct failure channel that mid-trace correction cannot fix.

**Fault-tolerance analogy (mid-circuit error correction):** In FT quantum/classical computing, errors are corrected *mid-circuit*, between gates, before they propagate — not by post-selecting on final outputs. End-of-line checking fails because (a) errors compound along the trajectory and (b) the final output under-determines where the fault occurred. The analogy maps cleanly: a CoT step ≈ a gate; the misaligned persona ≈ a fault channel; a monitor on the final answer ≈ destructive terminal measurement. The analogy therefore predicts corrective transitions should sit **inside the trace** (syndrome = verbalized intent / detectable false premise; recovery = re-derivation under honest persona), because by the final answer the benign-rationalization channel has already laundered the error (13% terminal detection). But the answer-flipping result is a caveat: their dominant observed failure includes a fault at the *last* "gate" (trace→answer), so mid-circuit correction alone is insufficient — you also need a final readout check, just as FT schemes still protect measurement.

**Threshold question in this setting:** the FT threshold theorem says if per-gate physical error rate p < p_th, logical error rate can be suppressed arbitrarily by adding redundancy/correction layers. Translated: let p = per-step probability that a reasoning step is misaligned (or that a benign rationalization is injected), and let d = detection probability of the syndrome (their monitor: 0.13–0.76 depending on whether the error is overt or laundered) and r = recovery success given detection (persona prefill suggests r can be high, ~baseline restoration). Question: **is there a threshold p\* such that, for per-step corruption rate below p\*, inserting corrective transitions every k steps drives end-to-end misaligned-answer probability to ~0 as trace length grows — and does the benign-rationalization channel (low d) put the medical/legal/security EM regime above or below that threshold?** Concretely testable in their setting: vary EM fine-tuning strength (dose) to vary p, insert honesty-persona corrective checkpoints every k CoT steps, and measure whether misaligned-answer rate falls exponentially in number of checkpoints (below threshold) or saturates (above threshold, dominated by undetectable laundered errors and terminal answer-flips).

## Released artifacts

Three datasets (medical/legal/security subtle-harm) + evaluation suite.
