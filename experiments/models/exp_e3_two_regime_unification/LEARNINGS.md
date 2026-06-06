# E3: Two-Regime Unification — Learnings

## Core Finding
The behavioral phase transition at s∈[4,6] is **real** (F#250 confirmed: GSM8K peaks at s=6) but **not caused by attention head activation cascading**. Only 1.8% of 336 heads show sigmoid inflection in [3,7]; ρ between flipped-head count and accuracy is -0.19 (opposite sign). Head flipping is a symptom of scale perturbation magnitude, not the causal mechanism.

## Why
Q-proj perturbation ΔQ = s·B@A grows linearly with scale. 84% of heads fit sigmoid with R²>0.9, but inflections cluster at s≈17 — the heads respond gradually, not at the behavioral threshold. The sharp 0%→80% GSM8K transition at s∈[4,6] must originate from nonlinear downstream processing: FFN GELU gating, output logit softmax margin crossing, or format template activation thresholds.

## Implications for Next Experiments

1. **E14 (Grassmannian activation orthogonality)**: Head-level analysis is insufficient. Must measure full hidden-state or post-FFN activations across layers to capture the downstream mechanism.

2. **E4/E5 (composition)**: Optimal scale cannot be predicted from head activation patterns. Behavioral sweep (F#248 approach) remains the only reliable method.

3. **Scale optimization research**: The locus of the phase transition is between attention output and final logits — FFN layers and output softmax are the prime suspects. Any future scale-prediction method must model these nonlinearities.

4. **Gemma 4 architecture note**: 6-layer periodicity in attention configs (full-attn vs sliding-window) observed. Reusable structural knowledge for future layer-selection experiments.

5. **General principle**: Attention-level measurements are necessary but not sufficient for explaining behavioral outcomes. The FFN nonlinearity amplifies small attention perturbations into sharp behavioral transitions — this is the "missing mechanism" for the two-regime phenomenon.
