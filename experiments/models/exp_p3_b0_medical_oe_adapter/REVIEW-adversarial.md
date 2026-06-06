# REVIEW-adversarial.md — exp_p3_b0_medical_oe_adapter

**Verdict: KILLED (structural falsification)**
**Round: 1 (final)**

---

## Structural Falsification (Blocking)

**Issue**: Theorem 1 Part 2 requires G_density ≥ 0.10. Measured G_density = 0.0054 (28×
below threshold). This is a pre-condition failure — not a hyperparameter issue.

**Evidence**: Logged during Phase 1 data preparation on full N=500 training examples:
```
Training data glossary density: 0.0054 (terms/word)
Expected G_density >= 0.10 for positive delta_D signal
```

**Verdict**: Experiment killed early under structural-falsification criterion.
Full run (estimated 45-90 min) would not change G_density. Smoke test confirms: vocab_ratio=0.70
(adapted WORSE than base, n=5 queries, 50 iters).

---

## Challenges (would normally raise, but now moot)

1. **Smoke test n=5 is too small** — vocab_ratio could be noise. HOWEVER, G_density=0.0054
   is computed on 500 training examples and is reliable. The smoke result directionally consistent.

2. **50 iters vs 500 iters** — Full run might differ. HOWEVER, gradient signal at G_density=0.0054
   is ~20× below minimum regardless of iters. No amount of training compensates for absent signal.

3. **30-term rubric may be too narrow** — True, but it's the SAME rubric used in Finding #457.
   Fair comparison: wikidoc performs worse than MCQ training (40% vs 60% improvement rate).

---

## Impossibility Structure Confirmed

The impossibility structure derived from PAPER.md:

**For ANY training format f_train:**
```
E[R(A_f(q)) - R(base(q))] ≥ 0  requires  G_density(D_train, G_eval) ≥ θ_min
```

All tested formats fail:
- MCQ format (Finding #457): G_density_training < θ_min (letters not in glossary)
- PubMedQA (Finding #459): G_density_training < θ_min (yes/no not in glossary)
- OE wikidoc (P3.B0): G_density_training = 0.0054 < θ_min

**Conclusion**: The data-rubric alignment problem is the root disease. Format is a symptom.

---

## PROCEED with LEARNINGS.md

The KILLED status is mathematically grounded. PAPER.md contains the impossibility structure.
The next experiment (P3.B1: Gram-Schmidt re-orthogonalization) is independent and proceeds normally.
