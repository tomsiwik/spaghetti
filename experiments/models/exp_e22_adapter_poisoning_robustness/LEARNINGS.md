# E22 Learnings: Adapter Poisoning Robustness

## Core Finding
Grassmannian A-matrix orthogonality provides 55pp behavioral protection against adapter poisoning at 10× magnitude — far exceeding the 5-15pp prediction. F#815's B-matrix coupling analysis is correct for activation cosine similarity but does NOT predict behavioral outcomes under adversarial perturbation.

## Why
Input-space feature isolation is the mechanism. Orthogonal A matrices force the poison to read from different input features than clean adapters. Even though B-matrix output coupling exists (F#815), the nonlinear stack (GELU, LayerNorm, softmax) attenuates perturbations that arrive from orthogonal input directions. Random A allows the poison to read the SAME features as clean adapters, creating correlated contamination that amplifies through the stack — sharp phase transition at 10× (80%→25%).

## Implications

1. **Grassmannian partially rehabilitated for safety**: Not via B-matrix decorrelation (F#815 correct there) but via input-space containment. For untrusted adapter composition, Grassmannian is a structural defense layer.

2. **F#815 scope narrowed**: B₁ᵀB₂ dominates for activation-level similarity metrics but does NOT dominate behavioral degradation under adversarial perturbation. Reinforces F#666: proxy metrics ≠ behavioral outcomes.

3. **E22-full design**: Must test targeted (not just random) poison, more layers (35), larger QA set (100). Phase transition sharpness (5×→10× for random) is the key phenomenon to characterize.

4. **E14-full connection**: E14's vacuous bound (σ_max ≈ 40-50) predicted Grassmannian is weak. E22 shows it's strong behaviorally. The bound is vacuous because it tracks the wrong quantity (activation cosine, not behavioral robustness).

5. **Engineering rule**: Always use Grassmannian A for composed adapters from untrusted sources. Cost is zero (initialization choice), benefit is 55pp protection margin.
