# MATH — Interference dilution: norm reduction vs structure

## Question
F#881: a *random* mask at keep-fraction f = 0.4335 on the dense thinking delta (v/o_proj),
composed with the math q_proj adapter, recovers +22pp of the +30pp F#862 trajectory-core gain —
with NO core selection. What failure mode does masking actually prevent? If the entire effect is
norm reduction, the mask machinery collapses to a scalar serving-time alpha knob and the
sparsity/mask arc is dead.

## Decomposition (why alpha = sqrt(f) is the right control)
Let dW be the dense thinking delta on one (layer, proj), and M a Bernoulli(f) mask.

- E[M ⊙ dW] = f · dW (mean field component, scale f = 0.4335)
- E[‖M ⊙ dW‖_F²] = f · ‖dW‖_F² → expected Frobenius norm = √f · ‖dW‖ = 0.6584 · ‖dW‖
- M ⊙ dW = f·dW + N, with N zero-mean, ‖N‖_F ≈ √(f−f²)·‖dW‖ — high-rank entrywise noise.

So the norm-matched dense control is C = √f · dW with α = √0.4335 = **0.6584**. If interference is
a smooth function of the perturbation norm injected into the residual stream, C and the random mask
B should produce indistinguishable EM. If masking instead destroys a *structured* (low-rank /
outlier-carried) collision with the math adapter, the entrywise noise N matters and B ≠ C.

Exploratory mean-field control C2 = f·dW (α = 0.4335): if C2 > C, the active variable is the
aligned component, not the norm. Does not affect the verdict.

Structure probes at the same f: D = keep-largest-|dW| (outliers kept), E = keep-smallest-|dW|
(outliers removed). If structural: E ≥ B > D ⇒ outlier-driven; D ≈ B ⇒ distributed collision.

## Prediction (pre-registered)
**EM(C, α=0.6584) ≥ mean EM(B, random mask ×3 seeds) − 3pp on GSM8K n=100 (greedy, no-thinking
harness).** I.e. dilution is pure norm reduction and the mask arc reduces to a scalar — predicted
verdict per kill #2335 is **killed** (the arc, not the measurement). Predicted numbers: EM_B ≈
0.66, EM_C within [EM_B − 0.03, EM_B + 0.05].

## Refutation threshold
- If mean EM(B) − EM(C) ≥ 5pp ⇒ prediction refuted: dilution is **structural**, mask arc lives
  (verdict supported, criterion 2335 passes for the mask arc).
- Gap in (3pp, 5pp) ⇒ inconclusive ⇒ provisional.
- Secondary noise check (pre-registered): if 3-seed random-mask spread (max−min) > 8pp, the F#881
  dilution estimate is flagged as noise regardless of the primary outcome.

## Setup (identical to exp_wildcat_static_velocity_surrogate harness, except no-thinking + n=100)
gemma-4-e4b-it-4bit; math LoRA q_proj scale 6.0; thinking delta dW = A@B from ckpt 0001000 on
v/o_proj, scale 1.0, dense-injected; greedy, max 1024 new tokens; `enable_thinking=False` per spec.
Arms: A dense (f=1), B random ×3 (seeds 0,1,2), C α=√f, C2 α=f (exploratory), D top-|dW|,
E bottom-|dW|. Composition is Σ(BᵢAᵢ) on disjoint projections; scales ≤ 8.
