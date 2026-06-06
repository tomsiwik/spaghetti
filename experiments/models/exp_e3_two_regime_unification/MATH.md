# E3: Two-Regime Unification — Head Activation Explains Scale Phase Transition

## Type
Guided exploration — proven framework (F#248/F#250 phase transition), unknown mechanism (per-head response).

## Prior Work
- **F#248** (supported): Two behavioral regimes — FORMAT (s=2) preserves knowledge, CAPABILITY (s=20) activates reasoning.
- **F#250** (supported): Sharp phase transition at s*∈[4,6] for math reasoning (jump=0.60 in one step, p=0.020).
- **F#251** (killed): Phase transition is math-specific, not universal. Driven by discrete evaluation boundaries (correct/wrong).
- **F#250 impossibility structure**: "Attention softmax creates threshold behavior: perturbation must exceed a critical magnitude to shift argmax of attention patterns."

## Hypothesis
The LoRA scale phase transition at s*∈[4,6] is explained by a **cascade of attention head activations**: individual heads have distinct softmax margin thresholds, and the behavioral transition occurs when a critical mass of heads "flip" (adapter perturbation exceeds their softmax margin).

## Mechanism (Atomic Level)

### Setup
Model: Gemma 4 E4B (42 layers, 8 heads/layer, GQA 4:1). Adapter: math domain, q_proj, r=6, s∈[0,20].

At scale s, the adapted query for head h in layer l:
```
Q'_{l,h} = Q_{l,h} + s · ΔQ_{l,h}
```
where ΔQ_{l,h} is the LoRA adapter's contribution to head h's query.

### Softmax Margin Theorem
For attention head h with base attention weights a = softmax(Q·K^T/√d), the adapter shifts attention by:
```
Δa ≈ diag(a)(I - 1·a^T) · s·ΔQ·K^T/√d
```
(first-order Taylor of softmax). The attention argmax changes when:
```
s · |δ_{ij}| > margin_{l,h} = (a_{max} - a_{2nd}) · √d / ||ΔQ·K^T||
```
where δ_{ij} is the relevant perturbation entry. Each head has a characteristic margin_{l,h}. Heads with concentrated attention (large margin) require larger s to flip; diffuse heads flip at smaller s.

### Prediction: Cascade Model
As s increases from 0→20:
1. **Low-margin heads flip first** (s < 4): these control format-level behaviors (punctuation, answer template)
2. **Critical cascade at s∈[4,6]**: a cluster of heads governing reasoning-format selection flip simultaneously, triggering the behavioral phase transition
3. **High-margin heads flip last** (s > 8): diminishing returns, potential overshoot

Measurable signature: per-head KL divergence D_KL(a_s || a_0) as a function of s should show sigmoid/step curves, with inflection points clustering bimodally (some at s<4, bulk at s∈[4,6], remainder at s>8).

### What Breaks It
- If all heads respond linearly to scale (no sigmoid), the phase transition must come from a different mechanism (e.g., FFN nonlinearity, not attention).
- If head flip-points are uniformly distributed (no clustering at s∈[4,6]), the behavioral transition is not explained by attention cascades.
- F#251 constraint: this mechanism is expected to hold for math only (discrete-answer domain).

## Kill Criteria (pre-registered)

### K2022 (structural): No head activation bifurcation
Head response to scale is linear, not sigmoid. Measured as: <20% of heads (< 67 of 336) show sigmoid KL behavior with R²_sigmoid > 0.9 AND inflection point in [3, 7].

### K2023 (prediction): Cannot predict optimal scale from head activation
Number of "flipped" heads at scale s cannot predict GSM8K accuracy. Measured as: Spearman ρ < 0.7 between count-of-flipped-heads and GSM8K accuracy across scales.

### K_target (behavioral): Head cascade does not explain behavioral transition
Combined structural+behavioral: the scale at which 50% of sigmoid-heads have flipped (s_50) does not fall within [3, 7], OR GSM8K accuracy at s_50 is < 50% of peak accuracy.

## Predictions
1. 30-60% of heads will show sigmoid KL response (the rest are adapter-insensitive or linear)
2. Sigmoid inflection points will cluster in two modes: s<3 (format) and s∈[4,6] (reasoning)
3. Correlation between flipped-head-count and GSM8K accuracy: ρ > 0.85
4. Full-attention layers (indices 5,11,17,23,29,35,41) will show more sigmoid heads than sliding-attention layers (sliding window limits context for reasoning)

## Platform
- MLX on M5 Pro 48GB
- mlx-lm: `mlx-community/gemma-4-e4b-it-4bit`
- Adapter: `micro/models/exp_p1_t2_single_domain_training/adapters/math/adapters.safetensors`
- No training required — forward-pass measurement only
