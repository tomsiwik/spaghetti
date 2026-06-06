# LEARNINGS: exp_p0_polar_premerge_composition

## Core Finding
Weight-space orthogonality does NOT enable pre-merge composition. PoLAR adapters with
perfect sr=6.0 and cosine 0.001 produce identical 0% GSM8K as standard LoRA. The failure
is in functional space — nonlinear composition through 42 transformer layers amplifies
any perturbation regardless of weight-space geometry.

## Why
Pre-merge = f(x; W_base + ΣDWᵢ). Even with near-zero weight overlap, each DWᵢ changes
the model's behavior for ALL inputs through the nonlinear forward pass. Attention's
multiplicative Q/K/V interactions amplify small weight perturbations far beyond the linear
prediction. The linear bound (0.43% per layer) compounds to catastrophic error through
42 layers.

## Impossibility Structure
Pre-merge of independently-trained adapters is impossible for transformers regardless of:
- Magnitude (Finding #510: std LoRA → 0%)
- Compression (Finding #526: TT-LoRA 737x → 0-1%)
- Spectral regularity (this: PoLAR sr=6.0 → 0%)
- Weight orthogonality (this: cosine 0.001 → 0%)

The only fix would be joint functional orthogonality training (all domains simultaneously),
which defeats the independent adapter training premise.

## Implications for Architecture
1. **Routing is THE composition method** — invest in scaling, not alternatives
2. **Pre-merge research is closed** — 4 independent proofs of impossibility
3. **Room Model** (routing-as-matmul) is the correct zero-overhead path
4. Weight-space metrics (cosine, sr, norms) are unreliable proxies for functional behavior

## References
- Finding #527 (KILLED): This experiment
- Finding #510: Standard LoRA pre-merge catastrophic
- Finding #526: TT-LoRA pre-merge catastrophic (direction not magnitude)
- Finding #442: PoLAR sr=r guarantee (verified, but insufficient for composition)
