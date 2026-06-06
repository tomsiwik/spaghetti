# E22: Adapter Poisoning Robustness — Grassmannian as Safety Feature

## Type
Verification — testing whether structural orthogonality (Grassmannian A) limits contamination from adversarial adapters.

## Prior work
- **arxiv:2406.14563** (Model Merging Safety): one bad adapter in naïve task arithmetic spoils the merged model.
- **F#510**: Pre-merged standard (non-orthogonal) LoRA destroys benchmarks; routing pipeline avoids it.
- **F#815**: Grassmannian constrains A but B-matrix coupling (σ_max(B₁ᵀB₂)) dominates interference.
- **F#817**: B matrices share W's output space — SVD filtering counterproductive.

## Failure mode
A poisoned adapter (adversarial or corrupted weights) composed with clean adapters contaminates all generation quality. In task arithmetic (Ilharco et al. 2022), composition is W_base + Σ ΔW_i = W_base + Σ B_i A_i. A poisoned adapter ΔW_poison = B_poison A_poison adds adversarial perturbation.

## Theorem (Orthogonal Subspace Containment)

**Claim**: If A_i are mutually orthogonal (Grassmannian), the poisoned adapter's contribution to each clean adapter's activation subspace is bounded by the B-matrix cross-correlation.

**Proof**: Let δ_poison(x) = B_poison A_poison x be the poisoned activation and δ_i(x) = B_i A_i x be a clean adapter's activation.

The interference of poison on adapter i is:
⟨δ_poison(x), δ_i(x)⟩ = x^T A_poison^T B_poison^T B_i A_i x

For Grassmannian A: A_poison A_i^T = 0 (by construction).
But the interference term involves B_poison^T B_i, NOT A_poison A_i^T.

The composed output is:
y = W_base x + Σ_{clean} B_i A_i x + B_poison A_poison x

The poison term B_poison A_poison x acts in the row-space of A_poison. If A_poison ⊥ A_i (Grassmannian), the poison reads from a DIFFERENT input subspace than clean adapters. This means:
- The poison perturbation is geometrically independent of clean adapter computations in input space
- But the output perturbation B_poison z_poison adds to the SAME output space

**Bound on output interference**:
‖δ_poison‖ / ‖Σ δ_clean‖ ≤ (‖B_poison‖_F · ‖A_poison x‖) / (Σ ‖B_i‖_F · ‖A_i x‖)

For random A (non-orthogonal): A_poison may overlap A_i, causing poison to affect the SAME input features that clean adapters read. This amplifies contamination through correlated input selection.

**Prediction**: Grassmannian should provide PARTIAL protection — the poison reads from independent input features, limiting correlated contamination. But output-space interference (B-matrix) remains uncontrolled per F#815.

## Kill Criteria (pre-registered)

**K2055** (target, behavioral): Grassmannian composition degrades catastrophically with poisoned adapter (> 30pp drop on knowledge QA).
- PASS if degradation < 30pp — Grassmannian provides graceful degradation
- FAIL if degradation ≥ 30pp — catastrophic despite orthogonality

**K2056** (comparative, behavioral): Orthogonality provides NO protection vs naïve averaging (identical degradation within 2pp).
- PASS if Grassmannian degradation < random degradation by > 2pp — orthogonality helps
- FAIL if |grassmannian_drop - random_drop| ≤ 2pp — no meaningful protection

Both KCs are behavioral (knowledge QA accuracy), not proxy metrics.

## Predictions
1. Grassmannian degradation: 10-20pp (partial protection from input-space orthogonality)
2. Random degradation: 20-35pp (correlated input features amplify contamination)
3. Protection margin: 5-15pp advantage for Grassmannian
4. If F#815 dominates: protection margin < 5pp (B-matrix coupling overwhelms A-matrix orthogonality)

## Experiment design
- Smoke: 3 layers, 3 clean adapters + 1 poison, 20 knowledge QA questions
- Full: 35 layers, 5 clean + 1 poison, 100 QA questions
- Poison types: (a) random noise at clean adapter magnitude, (b) negated clean adapter
- Measure: base accuracy, clean-only accuracy, clean+poison accuracy for Grassmannian vs random A
