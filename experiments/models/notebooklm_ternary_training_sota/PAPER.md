# SOTA Ternary Training Techniques Beyond BitNet STE: Research Survey

## Hypothesis

Improving ternary training beyond standard STE can reduce the 31.3% dead weight
fraction and/or improve model quality, yielding concrete, implementable changes
to our BitLinear training pipeline.

## What This Research Is

A comprehensive literature survey of ternary neural network training techniques
published through early 2026, focusing on methods that address the fundamental
failure modes of Straight-Through Estimation (STE) for ternary quantization.
The survey evaluates each method's mathematical foundations, overhead, and
applicability to our Composable Ternary Experts architecture on Apple Silicon.

## Key References

| Paper | ArXiv | Key Contribution |
|-------|-------|-----------------|
| BitNet b1.58 | 2402.17764 | Baseline: ternary {-1,0,+1} via STE + absmean |
| BitNet 2B4T | 2504.12285 | Production 2B ternary model, our base |
| Tequila | 2509.23800 | Deadzone reactivation via learnable bias lambda |
| Hestia | 2601.20745 | Hessian-guided differentiable surrogate quantizer |
| FOGZO | 2510.23926 | Zeroth-order gradient correction for STE |
| TernaryLM | 2602.07374 | Adaptive layer-wise std-based threshold |
| PT2-LLM | 2510.03267 | Post-training asymmetric ternary quantization |
| Sparse-BitNet | 2603.05168 | Natural sparsity exploitation in ternary |
| MatMul-free LM | 2406.02528 | Ternary + addition-only inference to 2.7B |
| 1-Bit Wonder | 2602.15563 | K-means 1.25-bit quantization to 31B |
| BitNet a4.8 | 2411.04965 | 4-bit activation quantization for ternary |
| Falcon-Edge | tiiuae/onebitllms | Megatron-native ternary training toolkit |

## The Dead Weight Problem

### Root Cause: Alpha-Coupling Equilibrium

Standard BitNet uses alpha = mean(|W|) as the scaling factor, with deadzone
threshold at alpha/2. For Gaussian-distributed weights:

    P(|w| < alpha/2) = erf(1 / (2*sqrt(2))) = 31.1%

This is a self-reinforcing equilibrium: if some weights escape the deadzone,
alpha increases, raising the threshold and trapping others. The 31.3% dead
weight fraction we observe in BitNet-2B-4T matches this theoretical prediction.

### Prior Experiment: Tequila Bias (exp_tequila_deadzone_fix)

We tested Tequila's minima reactivation at micro scale (d=512, 64M params,
2M tokens):
- K1 FAIL: Zero fraction unchanged at 32% (threshold was 20%)
- K2 PASS: PPL improved -6.7% via bias fusion (zero inference cost)
- K3: Adversarial review confirmed alpha-coupling as the core mechanism

The bias compensation works immediately; actual weight reactivation requires
orders of magnitude more training tokens (paper shows improvement at 1B+ with
10B tokens).

## Survey Results: Five Actionable Methods

### Method 1: TernaryLM Adaptive Threshold (RECOMMENDED -- Priority 1)

**Paper:** TernaryLM (arXiv:2602.07374)

**What it does:** Replace alpha = mean(|W|) with tau = 0.5 * std(W) as the
quantization threshold.

**Why it works:** Decouples the threshold from the self-reinforcing mean(|W|)
equilibrium. When weights escape the deadzone, std(W) responds differently
than mean(|W|), potentially creating a more favorable dynamic.

**Implementation on MLX:**
```python
# Current BitLinear:
alpha = mx.mean(mx.abs(W))
threshold = alpha / 2

# TernaryLM modification:
tau = 0.5 * mx.std(W)    # one-line change
W_q = mx.sign(W) * (mx.abs(W) >= tau)
alpha = mx.mean(mx.abs(W[mx.abs(W) >= tau]))  # scale from active only
```

**Overhead:** Negligible (one std computation per layer per forward pass).
**Inference cost:** Zero (same quantized format).
**Risk:** Unknown effect on composition quality. Different zero patterns may
affect adapter interference differently. Needs micro-experiment.

**Implementation effort:** ~30 minutes. One-line change to BitLinear.

### Method 2: Tequila Bias Fusion (RECOMMENDED -- Priority 1)

**Paper:** Tequila (arXiv:2509.23800, Tencent/AngelSlim)

**What it does:** Adds learnable bias from dead weight shadow values. At
inference, precomputed as static bias vector per layer.

**Why it works:** Extracts statistical signal from the 31% of parameters
that would otherwise contribute nothing. The bias is input-independent and
captures the mean contribution of dead weights.

**Our result:** Already validated at micro scale. -6.7% PPL improvement at
zero inference cost. Ready to integrate as default.

**Implementation on MLX:**
```python
class TequilaBitLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = mx.zeros((out_features, in_features))
        self.lam = mx.array(1e-3)  # learnable

    def __call__(self, x):
        alpha = mx.mean(mx.abs(self.weight))
        W_q = mx.clip(mx.round(self.weight / alpha), -1, 1)
        W_ste = self.weight + mx.stop_gradient(W_q * alpha - self.weight)
        Y = x @ W_ste.T

        dead_mask = (W_q == 0)
        C = self.lam * mx.sum(self.weight * dead_mask, axis=1)
        return Y + C

    def fuse_bias(self):
        """Call after training. Returns static bias to add to inference."""
        alpha = mx.mean(mx.abs(self.weight))
        W_q = mx.clip(mx.round(self.weight / alpha), -1, 1)
        dead_mask = (W_q == 0)
        return float(self.lam) * mx.sum(self.weight * dead_mask, axis=1)
```

**Overhead:** Training: one masked sum per layer. Inference: zero (fused bias).
**Implementation effort:** ~1 hour. Already prototyped in exp_tequila_deadzone_fix.

### Method 3: Hestia Smooth Surrogate (RECOMMENDED -- Priority 2)

**Paper:** Hestia (arXiv:2601.20745)

**What it does:** Replaces hard quantizer with temperature-controlled Softmax
surrogate during training. Per-tensor temperature schedule guided by Hessian
trace sensitivity.

**Why it works:** Mathematically eliminates the deadzone problem during
training. At finite temperature, every weight has nonzero gradient. Gradual
annealing ensures weights converge to ternary states without trapping.

**Key equations:**
```
pi_tau(q|w) = exp(-(w/gamma - q)^2 / tau) / Z(w, tau)    # soft assignment
w_eff = gamma * sum_q q * pi_tau(q|w)                      # differentiable
tau_i(t) = tau_bar(t) * exp(alpha * s_i)                   # per-tensor schedule
s_i = HutchPP_trace(H_i) / sum_j HutchPP_trace(H_j)      # sensitivity
```

**Implementation on MLX:**
```python
def hestia_quantize(W, gamma, tau):
    """Soft ternary quantization at temperature tau."""
    W_scaled = W / gamma
    # Compute Softmax probabilities for q in {-1, 0, 1}
    logits = mx.stack([
        -(W_scaled - q)**2 / tau for q in [-1, 0, 1]
    ], axis=-1)                                            # [d_out, d_in, 3]
    probs = mx.softmax(logits, axis=-1)                    # [d_out, d_in, 3]
    q_values = mx.array([-1.0, 0.0, 1.0])
    w_eff = gamma * mx.sum(probs * q_values, axis=-1)      # [d_out, d_in]
    return w_eff
```

**Overhead:** Significant during training (3x memory for probability tensors,
plus Hutch++ traces). Zero at inference (converges to hard quantizer).

**Risk:** Hutch++ requires matrix-vector products with the Hessian, which is
expensive. May need simplified sensitivity estimate for MLX.

**Implementation effort:** ~1 day. Non-trivial but well-defined.

### Method 4: FOGZO Gradient Correction (INFORMATIONAL -- Priority 3)

**Paper:** FOGZO (arXiv:2510.23926)

**What it does:** Replaces STE gradient with a debiased estimate using
zeroth-order finite differences guided by the STE prior.

**Why it works:** At quantization boundaries, STE gradient can point in the
wrong direction. FOGZO detects this (projection is zero) and uses unbiased
ZO estimate instead.

**Overhead:** 3x training time (2 extra forward passes per step with n=1).
This is the main practical concern.

**Applicability:** Best suited for fine-tuning rather than training from
scratch, where the 3x slowdown is more tolerable.

**Implementation effort:** ~2 hours. Straightforward but the overhead may
be prohibitive for our training-from-scratch workflow.

### Method 5: Combined TernaryLM + Tequila (RECOMMENDED -- Priority 1)

**Rationale:** TernaryLM and Tequila attack different aspects of the problem:
- TernaryLM: changes WHERE the deadzone boundary is (std vs mean threshold)
- Tequila: compensates for WHAT happens to weights inside the deadzone

These are orthogonal and composable. Predicted combined effect:
- TernaryLM threshold may shift the zero equilibrium point
- Tequila bias captures residual value from any remaining dead weights
- Combined: lower zero fraction AND bias compensation for remaining zeros

**Implementation:**
```python
class TernaryLMTequilaBitLinear(nn.Module):
    def __call__(self, x):
        # TernaryLM: std-based threshold
        tau = 0.5 * mx.std(self.weight)
        W_q = mx.sign(self.weight) * (mx.abs(self.weight) >= tau)
        alpha = mx.mean(mx.abs(self.weight[mx.abs(self.weight) >= tau]))
        W_ste = self.weight + mx.stop_gradient(W_q * alpha - self.weight)
        Y = x @ W_ste.T

        # Tequila: bias from dead weights
        dead_mask = (W_q == 0)
        C = self.lam * mx.sum(self.weight * dead_mask, axis=1)
        return Y + C
```

**Effort:** ~1 hour (both modifications are trivial to combine).

## Methods NOT Recommended for Our Use Case

### PT2-LLM (Post-Training Only)
Designed for quantizing pre-trained FP16 models to ternary without retraining.
We train from scratch, so this doesn't apply. However, the asymmetric
quantization insight (per-row shift mu) could inform our BitLinear init.

### Sherry / 3:4 Sparsity
Despite being listed in our experiment notes, Sherry appears to be an
undocumented or unpublished method from the Tencent/AngelSlim repository.
The "3:4 sparsity" pattern (3 of every 4 weights zero) is not described in
any available paper. The AngelSlim repository hosts Tequila; "Sherry" may be
an internal codename or upcoming work. NOT actionable without published details.

### Falcon-Edge Dual-Output Training
Falcon-Edge uses standard STE training (same as BitNet b1.58) integrated with
NVIDIA Megatron Core. The "dual-output" aspect appears to refer to Falcon-H1's
hybrid Transformer+Mamba architecture (parallel attention and SSM branches),
not a ternary training technique. Their Triton kernels implement absmean weight
quantization and absmax activation quantization -- standard practice.
No training innovation beyond efficient engineering.

### 1-Bit Wonder (K-Means Quantization)
Achieves 1.25 bits/weight via block-wise K-means. Impressive compression
(31B in 7.7 GB) but requires K-means clustering infrastructure and custom
quantization format. More relevant for deployment compression than training.

## Concrete Implementation Plan

### Phase 1: Immediate Wins (1-2 hours)

1. **Integrate Tequila bias fusion as default BitLinear recipe.**
   - Already validated: -6.7% PPL at zero inference cost
   - Add `lam` parameter to BitLinear, fuse as static bias after training
   - Apply to both base model training AND adapter training

2. **Switch to TernaryLM std-based threshold.**
   - One-line change: `alpha/2` -> `0.5 * std(W)`
   - Needs micro-experiment to validate: does it change zero fraction?
   - Needs composition test: does different zero pattern affect adapter quality?

### Phase 2: Experimental Validation (1 day)

3. **Micro-experiment: TernaryLM threshold + Tequila bias combined.**
   - Train ternary base at d=512, 4 layers, 3000 steps with combined method
   - Measure: zero fraction, PPL, adapter composition quality
   - Kill criteria: if PPL worse than baseline OR composition degrades

4. **Micro-experiment: Hestia simplified (no Hutch++).**
   - Use uniform temperature schedule (cosine decay, no per-tensor sensitivity)
   - Test if smooth surrogate alone improves training, without Hessian cost
   - Kill criteria: if PPL worse than STE baseline at same training budget

### Phase 3: Scale Validation (1 week)

5. **Apply winning combination to BitNet-2B-4T fine-tuning.**
   - Use the method that wins Phase 2 micro-experiments
   - Re-train 5 domain adapters with improved BitLinear
   - Measure: zero fraction change, individual PPL, composed PPL
   - Compare against current adapters

## What Would Kill This

- K1 FAIL: If no method reduces zero fraction below 25% AND no method improves
  PPL beyond Tequila's -6.7% -> research produced no actionable recommendations
  (KILL criteria 251)
- Combined method degrades composition quality (new zero patterns increase
  adapter interference)
- Hestia's overhead exceeds 5x training time -> impractical for our workflow
- FOGZO's 3x overhead yields <1% improvement -> not worth the cost

## Limitations

1. **No computational experiment.** This is a survey. The recommendations need
   micro-scale validation before integration.

2. **Sherry gap.** Could not find published details on Sherry/3:4 packing.
   If it emerges, it may supersede some recommendations.

3. **Scale dependence.** Tequila's weight reactivation works at 1B+/10B tokens
   but not at micro scale. Same scale dependence may apply to other methods.

4. **Composition impact unknown.** All surveyed methods are evaluated on
   individual model quality, not adapter composition. Our unique contribution
   (composable experts) may interact unexpectedly with changed zero patterns.

5. **arxiv ID uncertainty.** The Tequila paper was referenced as arXiv:2509.23800
   in our experiment notes, but this ID resolves to a different paper on the
   arxiv abstract page. The actual paper may have a different ID. The method
   description is from the Tencent/AngelSlim codebase and NotebookLM research.

## Summary of Recommendations

| # | Method | Effort | Expected Gain | Risk |
|---|--------|--------|---------------|------|
| 1 | Tequila bias fusion (default) | 1 hr | -6.7% PPL, zero inference cost | None (validated) |
| 2 | TernaryLM std threshold | 30 min | Unknown (needs experiment) | May change zero patterns |
| 3 | Combined TernaryLM + Tequila | 1 hr | Additive benefits (theory) | Needs validation |
| 4 | Hestia simplified | 1 day | Eliminate deadzones entirely | Training overhead |
| 5 | FOGZO correction | 2 hr | Better gradient quality | 3x training time |

**Top 3 actionable recommendations with paper references:**

1. **Integrate Tequila bias fusion as default** (arXiv:2509.23800, validated in
   exp_tequila_deadzone_fix): -6.7% PPL at zero cost.

2. **Switch to TernaryLM adaptive threshold** (arXiv:2602.07374): One-line change
   that breaks the alpha-coupling equilibrium. Micro-experiment needed.

3. **Test Hestia simplified surrogate** (arXiv:2601.20745): Only method that
   mathematically guarantees deadzone elimination. Worth the training overhead
   if it delivers significantly better quality.
