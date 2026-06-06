# Purpose-Trained Module Adapters: Mathematical Framework

## Experiment Type
**Guided Exploration (Type 2).**
Proven framework: module separability (Finding #300), domain-optimal module sets
(Finding #304), SFT training with response-only masking (Finding #180).
Unknown: does training configuration (module set) affect B-matrix optimization
and behavioral quality? The framework proves modules are separable; this explores
whether co-adaptation across modules during training is beneficial or harmful.

## A. Failure Mode Identification

**The Disease:** Finding #304 identified optimal module sets per domain (attn-only
for medical/math, full for code) but used post-hoc ablation of adapters that
were trained with ALL 7 modules active. The B-matrices learned during training
may have co-adapted across modules: attention B-matrices could have learned to
compensate for MLP perturbation or to distribute information across both
module types. Removing MLP modules post-hoc then uses B-matrices that were
optimized for a different configuration.

**Two competing hypotheses:**

H1 (Independence): Module effects are independently learned. Each module's
B-matrix captures a self-contained signal. Post-hoc ablation and purpose-trained
adapters should perform identically.

H2 (Co-adaptation): B-matrices co-adapt across modules. Attention B-matrices
in full-module training learn to complement MLP outputs. Removing MLP post-hoc
leaves attention B-matrices that are suboptimal for attention-only operation.
Purpose-trained attn-only adapters should outperform post-hoc ablated ones.

**Evidence from Finding #304 pointing toward co-adaptation:**
- Module effects are subadditive: interaction effects reach 13-2240%
  (K768 FAIL). This means modules are NOT independently effective --
  they interact through LayerNorm, SiLU, and softmax nonlinearities.
- At scale s=20, combined perturbation is WORSE than individual: full-module
  medical behavioral (0.435) < attn-only (0.467). The MLP modules are
  actively harmful for medical even in the jointly-trained adapter.

## B. Ask the Right Question (Reframe)

**Wrong question:** "Can we recover performance by removing modules after training?"
(This is post-hoc optimization of a mismatched configuration.)

**Right question:** "If we train an adapter knowing it will only use attention
modules, does the B-matrix optimize to a different (and potentially better)
solution than the attention-only slice of a full-module adapter?"

Mathematically: let B_full be the B-matrix trained with all 7 modules active,
and B_purpose be the B-matrix trained with only 4 attention modules active.
Is B_purpose != B_full|_attn (the attention slice of the full B-matrix)?
And if so, does f(x; theta + s * B_purpose^T A^T) >= f(x; theta + s * B_full|_attn^T A^T)?

## C. Prior Mathematical Foundations

### C.1: Training Gradient Flow with Module Masking

During full-module training, the loss gradient for an attention B-matrix B_{attn}
in layer l flows through:

  dL/dB_{attn}^{(l)} = dL/dh^{(l)} * dh^{(l)}/dB_{attn}^{(l)}

where h^{(l)} is the hidden state at layer l. Critically, h^{(l)} depends on
BOTH the attention and MLP perturbations of all preceding layers:

  h^{(l)} = LayerNorm(h^{(l-1)} + Attn(h^{(l-1)}; W_attn + s*DW_attn))
           + MLP(norm_h; W_mlp + s*DW_mlp)

In full-module training, h^{(l)} is computed with MLP perturbation active.
The gradient dL/dB_{attn}^{(l)} therefore accounts for the MLP perturbation
in the forward pass. The attention B-matrix adapts to operate in a regime
where MLP is also perturbed.

In purpose-trained attn-only training:
  h^{(l)} = LayerNorm(h^{(l-1)} + Attn(h^{(l-1)}; W_attn + s*DW_attn))
           + MLP(norm_h; W_mlp)  [no MLP perturbation]

The gradient now optimizes B_{attn} for the actual deployment configuration
(no MLP perturbation), yielding a potentially different solution.

### C.2: Gradient Mismatch Bound

**Theorem 1 (Gradient Mismatch).**
Let L(B; S) denote the loss when training B-matrices with module set S active.
Let B*_full = argmin_B L(B; S_full) and B*_attn = argmin_B L(B; S_attn).
If the MLP perturbation DW_mlp introduces a hidden state shift
delta_h^{(l)} = h_full^{(l)} - h_attn^{(l)} at layer l, then:

  ||grad_B L(B; S_full) - grad_B L(B; S_attn)||_F
    = O(s * ||B_mlp||_F * ||A_mlp||_F * L * Lip(LN) * Lip(sigma))

where Lip(LN) is the Lipschitz constant of LayerNorm (~1), Lip(sigma) is the
Lipschitz constant of the activation function (SiLU Lip ~ 1.1), and L is the
number of layers (30).

*Proof sketch.*
At each layer l, the MLP perturbation shifts the hidden state by:
  delta_h^{(l)} = s * MLP_perturbed(h^{(l)}) - MLP_base(h^{(l)})

By the chain rule, this shift propagates through subsequent layers,
each amplifying by at most Lip(LN) * Lip(sigma). Over L layers, the
accumulated shift in the gradient of B_{attn} is bounded by the product
of per-layer Lipschitz constants times the initial MLP perturbation magnitude.

At scale s=20 with ||DW_mlp||_F measured at ~69% of total perturbation norm
(Finding #304: attn_fraction ~31%), the MLP perturbation is substantial,
suggesting the gradient mismatch is non-negligible.

**Note:** This is a proof sketch (mechanism analysis with asymptotic bound),
not a formal Theorem/Proof/QED. The O(...) bound is not numerically evaluated
and does not predict which direction (purpose-trained vs post-hoc) is better.
Both H1 and H2 are consistent with gradient mismatch existing.

### C.3: Subadditivity as Evidence of Co-adaptation

Finding #304 measured interaction effects: the combined effect of attn+MLP
together differs from the sum of individual effects by 13-2240%. This
subadditivity means:

  PPL(attn+MLP) > PPL(attn) + PPL(MLP) - PPL(base)

The combined perturbation performs WORSE than the sum of parts. This is
consistent with destructive interference between module groups through
nonlinear interactions (LayerNorm normalization redistributes perturbation
energy, SiLU gating amplifies/suppresses different components).

**Implication for purpose-training:** If interactions are destructive at
evaluation time, they were also present (and creating confounds) during
training. B_{attn} in full-module training was optimized against a forward
pass where these destructive interactions existed. Removing MLP post-hoc
changes the operating point but keeps the B-matrix calibrated to the
wrong operating point.

### C.4: PLoP Evidence for Task-Specific Placement (arXiv:2506.20629)

PLoP (Precise LoRA Placement) demonstrates that optimal module placement
varies by task. Their normalized feature norm criterion identifies which
modules contribute most to task adaptation. Key insight: placing LoRA in
the wrong modules can HURT performance vs. fewer modules in the right places.

This directly supports the hypothesis that training with only the right
modules (purpose-trained) should be at least as good as training with all
modules and ablating post-hoc, because the optimizer sees the correct
gradient landscape from the start.

### C.5: Geva et al. MLP-as-Memory (arXiv:2012.14913)

MLP layers in transformers act as key-value memories storing factual
knowledge patterns. Perturbing MLP weights at scale s=20 disrupts these
stored memories. For knowledge domains (medical, math), the MLP perturbation
is net-negative because it corrupts stored facts while providing minimal
domain adaptation signal. Training without MLP perturbation avoids this
corruption entirely.

## D. Predictions

### D.1 Primary Predictions (Kill Criteria)

**P1 (K778): Purpose-trained attn-only medical behavioral >= 0.39**
(post-hoc attn-only baseline from Finding #304).

Reasoning: Purpose-trained B-matrices optimize for the actual deployment
configuration. The gradient no longer includes MLP perturbation artifacts.
We predict behavioral score >= 0.39, and likely >= 0.467 (the post-hoc
ablation result) because the attention B-matrices can now fully specialize.

**P2 (K779): Purpose-trained attn-only math PPL <= 3.43**
(post-hoc attn-only PPL from Finding #304).

Reasoning: With the optimizer seeing only attention perturbation in the
forward pass, it can optimize the attention B-matrices for maximal PPL
improvement without compensating for MLP artifacts. Expected PPL <= 3.43.

**P3 (K780): Purpose-trained full-module code behavioral >= 0.25**

Reasoning: Full-module training for code is the same configuration as
the original SFT training. The code adapter should match or exceed the
original behavioral score.

### D.2 Discriminating Predictions

**P4 (Co-adaptation test):** If H2 (co-adaptation) is correct, then
purpose-trained attn-only should STRICTLY outperform post-hoc attn-only.
Predicted improvement: 5-15% relative on behavioral metrics for medical/math.

**P5 (Independence test):** If H1 (independence) is correct, then
purpose-trained attn-only should match post-hoc attn-only within noise
(< 5% relative difference). This would mean module selection can be done
at serving time with no retraining cost.

**P6 (B-matrix divergence):** The cosine similarity between purpose-trained
B-matrices and the attention slice of full-module B-matrices should be
significantly < 1.0 if co-adaptation exists. If H1 holds, cosine ~ 0.95+.

## E. Assumptions and Breaking Conditions

**A1: Same training data and hyperparameters.**
Both purpose-trained and post-hoc adapters use the same data (400 train
samples per domain), same learning rate (1e-4), same iterations (300),
same Grassmannian A-matrices. The only difference is the module set during
training. If training dynamics differ (e.g., attn-only converges differently),
this is a genuine signal, not a confound.

**A2: Scale s=20 is the deployment configuration.**
We evaluate at s=20 for both configs. If the optimal scale differs between
module configurations, this introduces a confound. We control for this by
using the same scale.

**A3: Behavioral eval is sufficiently discriminative.**
The factual_recall metric (Finding #304) and code syntax check may not
capture subtle quality differences. The eval uses N_GEN=5 samples per domain,
which provides directional signal but limited statistical power.

**A4: 300 training iterations is sufficient convergence.**
If purpose-trained attn-only requires different convergence dynamics (more/fewer
steps), the comparison is confounded. We monitor training loss curves to check.

## F. Worked Example

Consider layer 0, domain=medical, scale=20.

**Full-module training (existing adapter):**
Forward pass at layer 0:
  h_attn = x @ (W_q + 20*B_q^T@A_q^T) ... softmax ... @ (W_v + 20*B_v^T@A_v^T) @ (W_o + 20*B_o^T@A_o^T)
  h_mlp = SiLU(h' @ (W_gate + 20*B_gate^T@A_gate^T)) * (h' @ (W_up + 20*B_up^T@A_up^T))
  h_out = h' + h_mlp @ (W_down + 20*B_down^T@A_down^T)

B_q gradient includes information about the MLP perturbation downstream.

**Purpose-trained attn-only:**
Forward pass at layer 0:
  h_attn = x @ (W_q + 20*B_q^T@A_q^T) ... softmax ... @ (W_v + 20*B_v^T@A_v^T) @ (W_o + 20*B_o^T@A_o^T)
  h_mlp = SiLU(h' @ W_gate) * (h' @ W_up)    [BASE MLP, no perturbation]
  h_out = h' + h_mlp @ W_down

B_q gradient now reflects the actual deployment scenario. The attention
adapter optimizes knowing MLP is unperturbed.

At d=2560, rank=16: each attention B-matrix has 16*2560 = 40,960 parameters.
Four attention modules per layer: 163,840 trainable params per layer.
Full-module: 364,544 params per layer (2.2x more).
Total attn-only trainable: 163,840 * 30 = 4,915,200 params.
Full-module trainable: 364,544 * 30 = 10,936,320 params.

The optimizer has fewer dimensions to explore in attn-only, potentially
converging more reliably to the optimal attention-only solution.

## G. Complexity and Architecture Connection

**Training cost comparison:**
- Purpose-trained attn-only: ~43% fewer trainable params per domain.
  Forward pass still processes full model (base MLP unchanged).
  Backward pass gradients: only attention LoRA B-matrices updated.
  Expected speedup: ~15-20% per training step (fewer gradient computations).
- Full-module: standard training, all 7 LoRA modules.

**Serving cost:** Identical to post-hoc ablation. The module config table
{medical: attn, code: full, ...} selects which B-matrices to load.

**Memory at training time:** Slightly less for attn-only (fewer optimizer
states): 4.9M params vs 10.9M params = 55% reduction in optimizer memory.

**Integration with SOLE architecture:**
Per-domain module configs are part of the router's configuration table.
When the softmax router selects domain d, it looks up the module config
for d and applies only those B-matrices. This is already supported by
the existing pre-merge and runtime LoRA infrastructure.

## Self-Test (MANDATORY)

1. What is the ONE mathematical property that makes the failure mode impossible?
   Training with the deployment module set eliminates gradient mismatch:
   B-matrices optimize for the actual forward pass they will see at inference.

2. Which existing theorem(s) does the proof build on?
   - Module separability (Finding #300, concat-slice equivalence)
   - Subadditivity of module effects (Finding #304, K768)
   - PLoP task-specific placement (arXiv:2506.20629)
   - Geva et al. MLP-as-memory (arXiv:2012.14913)

3. What specific numbers does the proof predict?
   - Purpose-trained medical attn-only behavioral >= 0.39 (K778)
   - Purpose-trained math attn-only PPL <= 3.43 (K779)
   - Purpose-trained code full behavioral >= 0.25 (K780)
   - B-matrix cosine between purpose-trained and post-hoc < 0.95 (if co-adaptation exists)

4. What would FALSIFY the proof?
   - If purpose-trained attn-only underperforms post-hoc attn-only, then
     co-adaptation is BENEFICIAL (the B-matrices benefit from seeing MLP
     perturbation during training, perhaps for regularization).
   - If K778 FAIL or K779 FAIL with purpose-trained adapters.

5. How many hyperparameters does this approach add?
   Count: 0. Module set per domain is determined by Finding #304 (attn vs full).
   All training hyperparameters identical to existing SFT training.

6. Hack check: Am I adding fix #N to an existing stack?
   No. This resolves a specific confound (Limitation 2) in Finding #304.
   Single question: does the training configuration match the deployment
   configuration?
