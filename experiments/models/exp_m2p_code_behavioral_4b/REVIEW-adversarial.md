# REVIEW-adversarial.md: exp_m2p_code_behavioral_4b

**Verdict: PROCEED**

---

## Summary

K984 FAIL (code_qr=0.158) correctly falsifies Theorem 2's lower bound prediction of 0.80–1.30.
This is a well-executed experiment: prediction was wrong, the failure mode was correctly diagnosed,
and the structural fix is already derived.

---

## Checklist

| Item | Status |
|------|--------|
| Prediction-vs-measurement table in PAPER.md | PASS ✓ |
| Kill criteria match results.json | PASS ✓ (K984=0.158<0.50, K985=1.3125, K986=1.0) |
| Finding status appropriate | PASS ✓ (`supported` — guided exploration, failure mode identified) |
| Math errors | MINOR — see below |

---

## Findings

### 1. Theorem 2 is falsified, not just bounded (non-blocking)

Theorem 2 derives γ ≈ (1 - ε) via σ_max(B_applied) ≈ 0.1. But measured code_qr = 0.158,
which means the B-matrices have singular values large enough to catastrophically interfere
with the base model's existing code priors.

The proof assumption "σ_max ≈ 0.1 (empirical from Finding #403 output_scale=0.032)" is wrong:
Finding #403 used **zero-init heads**, so at step 0 B ≈ 0 and σ_max ≈ 0. After training
with no SFT residual, B can grow to have σ_max >> 0.1.

PAPER.md correctly identifies this as the disease ("anti-format interference"). The theorem
should be annotated as FALSIFIED — the lower bound requires the SFT floor assumption.

**Non-blocking**: PAPER.md already captures this correctly in the analysis section.

### 2. peak_memory_gb = 0.0 in results.json (non-blocking)

Memory tracking was not instrumented in this run. Minor data quality issue. Does not
affect the scientific conclusion.

### 3. Prediction direction was also wrong (non-blocking)

MATH.md assumed base model would be ~weak on code (from 0.6B analogy). Actual base = 42.2%.
The prediction "pass@1_base ≈ assumed ~0% (0.6B analogy)" is explicitly noted as WRONG
in PAPER.md. This is good documentation.

---

## Strengths

- PAPER.md analysis is rigorous: correctly distinguishes anti-format interference vs format overfitting
- Impossibility structure derived and structural fix specified (SFT residual, same as Finding #403)
- Math quality and routing fully preserved (confirms composition/routing math is sound)
- Negative result advances the project: every domain with a capable base model requires SFT residual

---

## Non-Blocking Notes for Analyst

1. Theorem 2's lower bound derivation only holds when heads are zero-initialized (B ≈ B_sft at step 0).
   A corrected theorem should state: `qr >= 1.0 - ε` **only if** `B_applied = B_sft + zero_init_residual`.
   Without SFT floor, no lower bound on qr is derivable.

2. This finding generalizes beyond code: any domain where the base model has existing capability
   (math, language, reasoning) requires SFT residual. The SFT-residual architecture is universal.
