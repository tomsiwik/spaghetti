# Warm-Start FP16 to Ternary QAT: Mathematical Foundations

## Notation

| Symbol | Definition | Shape/Range |
|--------|-----------|-------------|
| W | Full-precision weight matrix | R^{n_out x n_in} |
| W_q | Ternary-quantized weight | {-alpha, 0, +alpha}^{n_out x n_in} |
| alpha | Per-tensor scaling factor | R^+ |
| x | Input activation | R^{B x T x d} |
| tau | Temperature / training step fraction | [0, 1] |
| t_switch | Step at which FP16->ternary transition occurs | N |
| T | Total training steps | N |
| eta_fp, eta_ter | Learning rates for FP16 and ternary phases | R^+ |
| m_t, v_t | Adam first/second moment estimates | same shape as W |

## Ternary Quantization with STE

### Forward Pass (Ternary Mode)

Given weight matrix W:

1. Compute per-tensor scale: alpha = mean(|W|) = (1 / n_out * n_in) * sum_{i,j} |W_{ij}|

2. Scale weights: W_scaled = W / (alpha + eps)

3. Quantize: W_q = clip(round(W_scaled), -1, 1) * alpha

The round-and-clip operation maps each element to {-1, 0, +1} then rescales:

    W_q_{ij} = alpha * sign(W_{ij}) * 1[|W_{ij}| > alpha/2]  (approximately)

### Straight-Through Estimator (STE)

The quantization function Q(W) = W_q is piecewise-constant (zero gradient almost
everywhere). STE provides a biased but useful gradient:

    W_ste = W + stop_gradient(Q(W) - W)

Forward: computes Q(W) (ternary)
Backward: dL/dW_ste passes through as dL/dW (identity Jacobian)

This is equivalent to:
    Forward: y = x @ W_q^T
    Backward: dL/dW = dL/dy * x  (as if W were used directly)

### Extra RMSNorm (Pre-Quantization Normalization)

Following BitNet b1.58 (Ma et al., 2024) and 1.58-bit FLUX (arxiv 2505.08823),
we apply RMSNorm to the input before the quantized matmul:

    x_norm = x / sqrt(mean(x^2) + eps) * gamma

where gamma in R^{d} is a learnable scale parameter.

This serves two purposes:
1. Stabilizes the input distribution to the quantized matmul
2. Provides a differentiable compensation path: even when W_q has limited
   expressivity, gamma can adjust the effective scale per-dimension

## Warm-Start Training Protocol

### Phase 1: FP16 Pretraining (steps 1 to t_switch)

Standard training with full-precision forward pass:

    y = x_norm @ W^T

Loss: L = CrossEntropy(LM_head(transformer(tokens)), targets)

Optimizer: AdamW with cosine LR schedule from eta_fp and weight decay lambda=0.01.

The key insight: the RMSNorm parameters (gamma) are trained during the FP16 phase,
so they are already adapted to the weight distribution when ternary mode activates.

### Phase 2: Ternary QAT (steps t_switch + 1 to T)

At step t_switch, the transition occurs:
1. Enable ternary quantization in all linear layers
2. Retain Adam optimizer state (m_t, v_t) from FP16 phase
3. Switch LR schedule: cosine from eta_post_switch (higher than FP16 end LR)
4. Set weight decay to 0 (ternary weights should not be decayed)

#### Why Retain Optimizer State

Adam maintains exponential moving averages of gradients and squared gradients:

    m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t
    v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2

At switch time, m_t and v_t encode per-parameter gradient statistics from FP16
training. These statistics approximate the loss landscape curvature around the
current weight values. Since the ternary QAT weights start from these same values
(the FP16 weights are quantized, and STE allows gradients to flow), the curvature
estimates remain useful.

Cold-start ternary begins with m_0 = v_0 = 0, requiring ~1/(1-beta_2) steps
(~1000 for beta_2=0.999) to build accurate variance estimates. The warm-start
skips this bootstrap phase.

#### LR Bump Post-Switch

The FP16 cosine schedule has decayed the LR by switch time. The quantization
introduces a new source of gradient noise (STE approximation error). A higher
initial LR for the ternary phase enables the optimizer to escape the
FP16 optimum and find a nearby ternary-friendly minimum:

    eta(t) = eta_post_switch * (1 + cos(pi * (t - t_switch) / (T - t_switch))) / 2

With eta_post_switch > eta_fp_end but eta_post_switch < eta_ter (cold-start LR).
In practice: eta_fp = 3e-4, eta_post_switch = 5e-4, eta_ter = 1e-3.

## Loss Dynamics at Transition

### Expected Spike

At the switch point, the forward pass changes from:
    y = x_norm @ W^T  (exact FP16)
to:
    y = x_norm @ W_ste^T  (ternary STE)

The quantization error per weight element is bounded by:
    |W_{ij} - W_q_{ij}| <= alpha/2  (for elements not at the boundary)

This introduces an immediate perturbation to the output:
    Delta_y = x_norm @ (W_q - W)^T

The loss spike magnitude depends on the quantization error relative to the
weight magnitudes. For a well-conditioned model, this spike is transient.

### Recovery Criterion

Define recovery as: the 50-step moving average of loss returns to within
10% of the pre-switch loss. Formally:

    (1/50) * sum_{i=t}^{t+49} L_i <= 1.10 * L_{t_switch}

If this condition is never satisfied, the spike is "non-recoverable" (K2 KILL).

## Computational Costs

All conditions train for T = 3000 total steps with identical compute per step
within each mode. The ternary forward pass adds:
- 1 reduction (mean |W|) per weight matrix: O(n_in * n_out)
- 1 elementwise divide, round, clip, multiply: O(n_in * n_out)
- 1 RMSNorm per layer: O(B * T * d)

At d=512, 4 layers, 6 projections per layer (Q, K, V, O, fc1, fc2):
- Per-step overhead: ~6 * 2 * 512 * 2048 = 12.6M extra ops (negligible vs matmul)
- Measured: FP32 = 6.0 steps/s, Ternary = 5.8 steps/s (3.3% slower)

## Worked Example (d=512, N_layers=4)

Architecture: 64.1M params (FP32) / 64.1M params (Ternary, +18K from RMSNorm gamma)

**FP32 baseline** (3000 steps):
- Final loss: 4.388, PPL: 344.1

**Cold-start ternary** (3000 steps):
- Final loss: 4.304, PPL: 416.8 (1.211x FP32)
- Zero fraction: 32.0% of ternary weights map to 0

**Cold-start ternary, weight_decay=0.0** (3000 steps, ablation control):
- Final loss: 4.307, PPL: 411.2 (1.195x FP32)
- Zero fraction: 32.1% of ternary weights map to 0
- Only 1.4% better than wd=0.01 cold-start (416.8 -> 411.2)
- Warm-start still 12.4% better, confirming warm-start advantage is real

**Warm-start 10%** (300 FP16 + 2700 QAT):
- Pre-switch PPL: 842.7 (only 300 steps of FP16 = very early)
- Max loss spike: +0.435 (transient, recovered within at most 51 QAT steps -- first measurement point)
- Final PPL: 360.1 (1.046x FP32) -- BETTER than cold-start by 12.4% (vs no-wd control)
- Zero fraction: 31.3%

**Warm-start 20%** (600 FP16 + 2400 QAT):
- Pre-switch PPL: 655.3 (600 FP16 steps = somewhat better starting point)
- Max loss spike: +0.685 (recovered within at most 51 QAT steps -- first measurement point)
- Final PPL: 382.3 (1.111x FP32) -- better than cold-start but worse than 10%
- Zero fraction: 31.2%

**Key observation**: 10% warm-start outperforms 20% warm-start. This is
counterintuitive -- more FP16 pretraining should give better initialization.
The explanation: with fixed total budget T, the 10% condition gets 2700 QAT
steps vs 2400 for 20%. The ternary phase needs enough steps to fully adapt,
and the marginal benefit of 300 extra FP16 steps (in a regime where the model
is still at PPL ~800) does not compensate for 300 fewer QAT steps.

## Weight Decay Ablation

### Confound

The original cold-start ternary used weight_decay=0.01 while the warm-start
ternary phase used weight_decay=0.0. This means the 13.6% improvement of
warm-start over cold-start could partially come from weight decay removal
rather than initialization/optimizer state transfer.

### Ablation Result

Running cold-start ternary with weight_decay=0.0 (everything else identical):

    PPL(cold, wd=0.01) = 416.8  (1.211x FP32)
    PPL(cold, wd=0.0)  = 411.2  (1.195x FP32)
    PPL(warm-10%)       = 360.1  (1.046x FP32)

Weight decay effect: (416.8 - 411.2) / 416.8 = 1.4% PPL improvement.
Warm-start vs no-wd cold-start: (411.2 - 360.1) / 411.2 = 12.4% improvement.

### Interpretation

Weight decay removal accounts for only 1.4pp of the 13.6pp total improvement
(~10%). The remaining 12.4pp (~90%) comes from the warm-start mechanism
(weight initialization from FP16 pretraining + optimizer state transfer).
The confound is real but minor: the warm-start advantage is genuine.

Note: weight decay of 0.01 on ternary weights pushes weight magnitudes
toward zero, which could increase the fraction of weights quantized to 0
(harming expressivity). In practice, the zero fraction is nearly identical
(32.0% vs 32.1%), suggesting this effect is negligible at the magnitudes
observed.

## Assumptions

1. STE provides sufficient gradient signal for ternary weights to converge
   (validated: cold-start and warm-start both converge)
2. Adam momentum/variance transfer is beneficial (validated: 10% warm-start
   achieves 1.046x vs cold-start 1.211x)
3. Extra RMSNorm does not interfere with FP16 pretraining (validated: FP16
   phase uses same architecture, norm learns useful statistics)
4. Scale invariance: the warm-start advantage observed at d=512 transfers
   to larger dimensions (untested, requires macro-scale validation)
5. Fixed total training budget: all conditions train for T steps. A fair
   comparison at convergence would require variable T, which this experiment
   does not test.
