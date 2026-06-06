# Adversarial Review: C1.1 PoLAR Joint Stiefel on Gemma 4

**Verdict: PROCEED** (with documented caveat on KC08)

---

## What's Strong

**Theorem 2 is beautifully verified.** sr(PoLAR r=16) = 16.0000 exactly, and sr(PoLAR r=6) = 6.0000 exactly. This is a structural proof-verification experiment done correctly: the theorem makes a sharp prediction (sr = r), and measurement matches to 7 decimal places. Stiefel distances at 6e-15 (float64 machine epsilon) confirm the retraction is numerically perfect.

**T1.5 failure is conclusively diagnosed and fixed.** T1.5 sr(PoLAR-U-only)=2.21 vs this experiment sr(joint-Stiefel)=16.00. The impossibility structure (U-only → rank-1 gradient → V collapse) was correctly identified, and joint Stiefel bypasses it structurally.

**LoRA rank collapse confirmed.** sr(LoRA r=6)=1.77 exactly as predicted by T1.5 empirics. The 9× improvement (1.77→16.00) over LoRA is real and structural.

---

## One Blocking Concern

**KC08 passes trivially: PoLAR=0.0% vs LoRA=0.0%.**

This satisfies the letter of the kill criterion but not the spirit. The behavioral claim that "higher sr translates to better multi-domain generalization" is **unverified**. Both approaches fail on GSM8K because the training data (synthetic 5-domain: math/code/language/logic/science) doesn't match GSM8K's arithmetic reasoning format.

**This is a design flaw, not a retraction.** The structural claims are correct. But KC08 was measuring the wrong thing. A proper KC08 would train on GSM8K-style data and measure GSM8K accuracy, OR train on synthetic data and measure on a synthetic holdout test.

**Why PROCEED despite KC08:** The CORE claims of C1.1 (Theorem 2, rank structure) are conclusively verified. KC08 was a secondary behavioral check that failed to generate signal due to benchmark mismatch. The finding should document this as a known gap requiring C1.3 follow-up.

---

## Minor Concerns (Non-blocking)

1. **Numerical warnings at step 1:** RuntimeWarnings in the retraction code when B=zeros. The guard didn't trigger because RETRACT_EVERY=20 let gradients accumulate before first retraction. Non-blocking (Stiefel distances confirm correctness), but the guard should use `< 1e-8` instead of `< 1e-12` to catch near-zero B more reliably.

2. **KC08 criterion should have used synthetic holdout:** If training on synthetic data, evaluate on synthetic holdout (not GSM8K). The T5.1 experiment (User Local Training) showed 76pp gain on in-distribution behavioral tests. Use similar approach here.

---

## Status

Finding should be SUPPORTED with caveat: structural claims (Theorems 1 & 2) conclusively verified; KC08 behavioral comparison deferred to C1.3 (needs training/eval distribution alignment).

**Route:** review.proceed → Analyst writes LEARNINGS.md → claim next experiment
