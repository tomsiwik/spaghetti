# Adversarial Review: C0.2 Direction-Only Adapter

**Status: PROCEED (with non-blocking caveats)**

Experiment correctly KILLED. Kill criteria applied correctly. PAPER.md has complete prediction-vs-measurement table. Finding #439 is valid.

---

## 1. Math Gap in Theorem 1 (NON-BLOCKING)

**The claim:** "Two LoRA adapters with B_1 = alpha * B_2 produce identical outputs after normalization."

**The gap:** This is only true when `delta_W >> W_q` (adapter dominates). In standard LoRA regime, `delta_W << W_q`, so:

```
RMSNorm((W_q + alpha * delta_W) @ x) ≠ RMSNorm((W_q + delta_W) @ x)
```

The re-scaling invariance `RMSNorm(alpha * v) = RMSNorm(v)` holds when the ENTIRE vector is scaled, not just one additive component. The scale of `delta_W` relative to `W_q` changes the DIRECTION of the combined output — which is exactly what matters post-normalization.

**Impact:** The claim that "LoRA's scale hyperparameter is irrelevant on normalized architectures" is too strong. What RMSNorm discards is the overall magnitude of the combined output, not the relative weight between `W_q` and `delta_W`.

**Why non-blocking:** The theorem correctly establishes that post-normalization, only direction matters. The training convergence theorems (T2, T3) are correct and confirmed. The empirical finding (KC05 FAIL, 83.3% ratio) is valid regardless of this theoretical gap. The resurrection path (Riemannian GD) is well-motivated independent of the scale argument.

**Testable consequence for C1.1:** If scale IS irrelevant, then direction-only with scale={1,5,10,20} should give identical accuracy. Researcher noted this was not measured. C1.1 should include this sweep — if scales differ, it confirms the gap in T1's LoRA consequence.

---

## 2. Diagnosis Correctness (CONFIRMED)

Three failure contributors diagnosed:
1. Initialization disruption (B=0 → project → random unit vectors at step 1)
2. AdamW momentum mismatch after each projection
3. Post-hoc retraction vs. native Riemannian step

All three are plausible and collectively explain a 17pp gap without invoking representational incapacity. The loss curves confirm convergence to similar final loss (0.84 vs 0.81), supporting "training algorithm gap, not representational gap."

---

## 3. Resurrection Path (VALID)

Cayley retraction / exponential map for Stiefel manifold is established (Wen & Yin 2013). The connection to PoLAR T1.5 is correct — that experiment constrained only U, not V. C1.1 must constrain both U (lora_a) and V (lora_b rows) on the Stiefel manifold.

Key implementation requirement: do NOT use post-hoc projection. Use proper tangent-space gradient update:
```
G_tangent = G - B @ G^T @ B    # (for Stiefel manifold)
B_new = retract(B + lr * G_tangent)
```

---

## Verdict

**PROCEED.** Experiment is correctly KILLED. T1-T3 theorems confirmed empirically. KC05 failure correctly attributed to training algorithm, not representation. Finding #439 is valid. Analyst should capture the Theorem 1 gap and scale sweep recommendation in LEARNINGS.md for C1.1 designers.
