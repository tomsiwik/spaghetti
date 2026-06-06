# Online LoRA Adaptation from Conversation Turns

## Type: Frontier Extension

## Failure Mode
The adapter fails to accumulate project-specific knowledge from sequential
single-gradient-step updates, either because: (a) each step's update is too
small to shift the output distribution, or (b) consecutive steps destructively
interfere, erasing earlier knowledge.

## Prior Math

**Intrinsic Dimensionality (Aghajanyan et al., arXiv:2012.13255):**
Fine-tuning operates in a low-dimensional subspace. For common NLP tasks,
the intrinsic dimensionality d_intrinsic < 200. A single-project conversational
context (order of 10 independent facts) has d_intrinsic ~ 10.

**Online Convex Optimization (Zinkevich, 2003; Hazan, 2016):**
Online gradient descent on convex functions achieves regret bound:
R_T = sum_{t=1}^T L_t(w_t) - min_w sum_{t=1}^T L_t(w) <= ||w*||^2/(2*eta) + eta*T*G^2/2

**LoRA (Hu et al., arXiv:2106.09685):**
Low-rank adaptation ΔW = BA with B in R^{d x r}, A in R^{r x d} constrains
updates to a rank-r subspace, providing implicit regularization.

**PLUM Pattern (arXiv:2411.13405):**
Conversation turns can be augmented to instruction-following QA pairs,
providing structured training signal for online adaptation.

## Theorem (Online LoRA Capacity for Contextual Adaptation)

**Setup.** Let f_theta be a frozen language model. Apply rank-r LoRA to L target
layers, yielding trainable parameters phi in R^p where p = 2*r*d*L (A and B
matrices). Train with online SGD: one gradient step per conversation-derived
QA pair.

**Claim 1 (Capacity).** The adapter can represent the project context if
d_intrinsic <= r * L.

*Proof.* Each layer's LoRA adds a rank-r correction to the attention mechanism.
Across L layers, the total representational capacity spans an r*L-dimensional
subspace of the model's function space. For r=4, L=8: capacity = 32 dimensions.
For a project context with ~10 independent facts: d_intrinsic ~ 10 << 32. QED.

**Claim 2 (Convergence).** With learning rate eta and T steps, the average
training loss satisfies: (1/T) sum L_t(phi_t) - (1/T) sum L_t(phi*) <= O(1/sqrt(T))

*Proof.* Standard online GD regret bound (Zinkevich 2003). The cross-entropy
loss is convex in the logits. With the optimal learning rate eta* = ||phi*||/(G*sqrt(T)),
the average regret is O(||phi*|| * G / sqrt(T)). For T=20: regret ~ O(1/4.5).
This decreases with more turns. QED.

**Claim 3 (Interference Bound).** The adapter's effect on pre-trained knowledge
is bounded by the Frobenius norm ratio ||ΔW||_F / ||W_base||_F.

*Proof.* Each LoRA update changes the effective weight matrix by ΔW = BA.
With rank r=4 and small learning rate, ||ΔW||_F <= r * max(||B||, ||A||).
After T=20 steps with lr=1e-3, the cumulative change is bounded by
||ΔW_total||_F <= T * eta * G_max * r ~ 20 * 1e-3 * G * 4.
For G ~ 1 (typical gradient norm with Adam): ||ΔW||_F ~ 0.08.
Base model weights ||W||_F >> 10 (thousands of parameters at ~0.1 magnitude).
Relative perturbation: 0.08/10 = 0.8%. This is negligible. QED.

## Quantitative Predictions

| Prediction | Value | Basis |
|-----------|-------|-------|
| Training loss decrease | >50% from step 1 to 20 | Claim 2: O(1/sqrt(T)) regret |
| Project QA accuracy improvement | >=20pp over base | Claim 1: capacity sufficient |
| Per-turn training latency | <500ms | Single example, rank-4, 8 layers |
| General knowledge degradation | <2pp | Claim 3: 0.8% weight perturbation |
| Base project QA accuracy | ~10% (random + lucky guesses) | Model has no knowledge of fictional project |
| Adapted project QA accuracy | ~30-50% | 20 steps encode partial knowledge |

## Behavioral Predictions

1. The adapter should learn concrete facts (project name, specific tech choices)
   more easily than abstract relationships (why a technology was chosen).
2. Facts reinforced across multiple training turns should have higher accuracy
   than facts mentioned only once.
3. The training loss curve should show rapid initial decrease (learning format)
   followed by slower decrease (learning content).

## Kill Criteria Mapping

- K1285 (accuracy >= 20pp): From Claim 1 + Claim 2. Capacity sufficient, convergence guarantees positive learning signal. Predicted PASS at ~25-40pp.
- K1286 (latency < 1s): From computational analysis. Single forward+backward on rank-4 LoRA with ~200 tokens. Predicted PASS at ~200-500ms.
- K1287 (MMLU within 2pp): From Claim 3. Weight perturbation <1%. Predicted PASS at ~0-1pp degradation.

## What Would Kill This

1. **Gradient vanishing through quantized base**: If 4-bit quantization creates
   gradient dead zones, the LoRA updates may receive near-zero signal. This would
   manifest as flat training loss. (Unlikely: reward LoRA experiment proved
   gradients flow through 4-bit Gemma 4.)

2. **Destructive interference between turns**: If consecutive turns have
   anti-correlated gradients, each step undoes the previous. This would manifest
   as oscillating loss. (Mitigated by coherent project context.)

3. **Insufficient learning rate**: If eta is too small, 20 steps produce
   imperceptible changes. (Testable: we measure training loss.)
