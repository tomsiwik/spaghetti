# MATH.md — P3.B5: Domain-Conditional Personal Adapter Retraining

## Problem Statement

P3.B1–B4 killed all weight-space additive composition strategies. The root cause
identified in Finding #465 is a **covariate shift**: the personal adapter ΔW_personal
was trained on the base model distribution but receives domain-shifted hidden states
at inference time.

## Theorem 1 (Covariate Shift in Adapter Composition)

**Setup**: Let f_base: X → ℝ^d be the frozen base model. Let ΔW_domain be the domain
adapter (frozen at inference). Let ΔW_personal be the personal adapter.

The personal adapter was trained to minimize:
```
L_train(ΔW_personal) = E_{(x,y) ~ D_style} [ ℓ(f_base(x) + ΔW_personal(f_base(x)), y) ]
```

At inference with both adapters active, the personal adapter receives:
```
h_composed(x) = f_base(x) + ΔW_domain(f_base(x)) + ΔW_personal(f_base(x))
```

The effective loss at inference is:
```
L_infer(ΔW_personal) = E_{(x,y)} [ ℓ(f_base(x) + ΔW_domain(x) + ΔW_personal(x), y) ]
```

**Covariate shift**: ΔW_personal was optimized for input distribution P_base = {f_base(x)},
but operates on P_domain = {f_base(x) + ΔW_domain(x)} at inference.

By the covariate shift bound (Ben-David et al. 2010, arxiv 2106.09685 cites):
```
L_infer(ΔW_personal) ≤ L_train(ΔW_personal) + C · d_H(P_base, P_domain)
```
where d_H is the H-divergence measuring distributional distance.

**Observation**: P3.B4 measured 76% → 24% style drop (Δ=52pp). Since L_train = 0
(76% compliance on base), the entire 52pp degradation is from d_H(P_base, P_domain) > 0.
The math domain adapter's ΔW_domain significantly shifts the hidden state distribution.

**QED**: Weight-space composition of independently trained adapters is bounded below
by the covariate shift term, which is structurally non-zero when domain adapters modify
the same layers (L0–25) as those providing input to the personal adapter (L26–41).

## Theorem 2 (Training Distribution Alignment Eliminates Covariate Shift)

**Claim**: Train ΔW_personal' with f_domain = f_base + ΔW_domain frozen and active:
```
L_train'(ΔW_personal') = E_{(x,y) ~ D_style} [ ℓ(f_domain(x) + ΔW_personal'(x), y) ]
```

At inference, the composed system evaluates:
```
L_infer'(ΔW_personal') = E_{(x,y)} [ ℓ(f_domain(x) + ΔW_personal'(x), y) ]
```

**Proof**: Training distribution = inference distribution. Source P_source = P_domain = P_target.
Therefore d_H(P_source, P_target) = 0 → the covariate shift term vanishes:
```
L_infer'(ΔW_personal') = L_train'(ΔW_personal')
```

The personal adapter learns to produce "Hope that helps, friend!" in the exact activation
space it will encounter at inference. No distribution mismatch. **QED**

## Implementation: Domain Fusion Strategy

To achieve P_source = P_domain at training time, we:

1. **Fuse** math domain adapter into base model weights:
   ```
   W_domain_base = W_base + ΔW_domain  (FP16 after dequantization)
   ```
   This creates f_domain as the effective "base" model for training.

2. **Train** personal adapter on W_domain_base:
   ```
   ΔW_personal' = argmin E_{D_style} [ℓ(W_domain_base(x) + ΔW_personal'(x), y)]
   ```

3. **Evaluate** composed: W_domain_base + ΔW_personal' = f_domain + ΔW_personal'
   - Style: personal adapter operates on domain-modified activations it was trained for
   - Math: domain knowledge is baked into W_domain_base (present regardless of personal)

## Quantitative Predictions

| Metric | Baseline | Previous best (P3.B1 B-GS) | P3.B4 (pure additive) | P3.B5 prediction |
|--------|----------|----------------------------|----------------------|------------------|
| personal-only style | 76% | 76% | 76% | 76% (same adapter) |
| composed style | — | 60% (Δ=16pp) | 24% (Δ=52pp) | **≥66%** (Δ≤10pp) |
| math MCQ | 10% | N/A | 15% (K1188 PASS) | ≥5% (baked in base) |
| new_personal_alone | — | — | — | **≥70%** (sanity) |

**Why ≥66% (not ~76%)**:
- Theorem 2 guarantees d_H = 0 in the linear weight-space
- Residual: non-linear attention/LayerNorm interactions may cause 0–10pp residual
- Conservative estimate: 66% (10pp allowance for non-linear effects)

**Why math ≥5%**:
- Math domain knowledge is fused into W_domain_base
- K1196 is a floor check: domain math should not be destroyed

## Kill Criteria

| ID | Criterion | Prediction | Status |
|----|-----------|------------|--------|
| K1195 | style_composed ≥ 66% | ~70–74% | untested |
| K1196 | math_acc ≥ 5% | ~10% | untested |
| K1197 | new_personal_alone ≥ 70% | ~74–76% | untested |

## Failure Modes

**If K1195 killed (style < 66% despite correct training distribution)**:
The failure would indicate non-linear transformer interactions are the PRIMARY
mechanism, not covariate shift. Impossibility structure:
- For additive adapters on shared layers: L_k(x + ΔW_D·x + ΔW_P'·x) ≠ L_k(x + ΔW_P'·x)
  regardless of training procedure
- The non-linear LayerNorm and attention softmax responses to the COMBINED signal
  cannot be fully compensated by either adapter alone
- Fix: P3.B6 — adversarial training where personal adapter learns to be domain-robust

**If K1197 killed (new_personal_alone < 70%)**:
Training on the domain-adapted base degraded personal adapter quality.
Possible cause: FP16 dequantized model has different optimization landscape.
Fix: increase training iterations or tune learning rate for FP16 base.

## Reference

- Ben-David et al. 2010, "A Theory of Learning from Different Domains" (covariate shift)
- Hu et al. 2021, arxiv 2106.09685 (LoRA — target of our composition problem)
- Finding #465: P3.B4 KILLED — pure additive 24%, impossibility of weight-space composition
- Finding #462: P3.B1 KILLED — B-GS 60%, style in col(ΔW_D)
- Finding #436: P1.T5 — personal adapter baseline: 76% style compliance
