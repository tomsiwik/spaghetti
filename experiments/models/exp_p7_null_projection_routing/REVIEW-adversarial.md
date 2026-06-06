# Adversarial Review: exp_p7_null_projection_routing

## Verdict: KILL (confirmed)

All 3 kill criteria failed decisively. The impossibility structure is mathematically sound.

## Review

### 1. Prediction-vs-Measurement Table
Present in PAPER.md. Complete and honest — all 3 kill criteria show FAIL with exact numbers.

### 2. Kill Criteria Verification
- **K1300**: 20% accuracy (chance level). Confirmed in results.json — confusion matrix shows 20/20 items routed to legal. Not a close miss; a total failure.
- **K1301**: Spearman r = -0.19 (anti-correlated). Threshold was 0.3. The negative correlation means higher projection magnitude weakly predicts *worse* quality — the opposite of the hypothesis.
- **K1302**: 4.35ms latency (8.7x above 0.5ms target). The Q projection (3584 -> 2048) is inherently expensive.

### 3. Impossibility Structure — Sound
The core argument is correct: null(W_v) contains features W_v ignores by construction. Domain-discriminative features are in range(W_v) because the base model needs them for value computation. Therefore routing via null-space projection is structurally orthogonal to domain information. This is not a hyperparameter problem — no amount of tuning can fix it.

The proof of `<routing_signal, domain_information> = 0` follows directly from the orthogonal complement relationship V perp V^perp.

### 4. Finding Status
KILLED is correct. This is a Type 1 verification experiment where the theorem's predictions were decisively refuted. The failure is structural, not parametric.

### 5. Issues Noted (Non-blocking)

**Theorem was wrong, not the experiment.** The MATH.md theorem claimed A-matrices would converge to domain-discriminative directions. The actual failure is that the theorem's premise was flawed: domain information doesn't exist in null(W_v) in the first place, so the gradient structure argument (step 2) is vacuously true — adapters DO learn different directions, but those directions are all domain-blind. The PAPER.md correctly identifies this, but MATH.md should acknowledge the theorem's premise failure more explicitly. Non-blocking for a killed experiment.

**Latency analysis.** The 4.35ms number includes the full Q^T @ x projection across 8 layers x 5 adapters. This overhead would exist for any null-space-based routing scheme, not just this one. Worth noting for future null-space experiments.

**Positive result buried.** Training convergence (all 5 adapters to loss < 0.03) confirms null-space LoRA training stability from Finding #494. This is a useful secondary confirmation even in a killed experiment.

## Architectural Insight (Valuable)
The decomposition — route in range(W_v), adapt in null(W_v) — is the key takeaway. These are complementary subspace concerns that cannot be unified through a single projection. This clarifies the Room Model: "routing IS the matmul" holds for standard LoRA but fails for null-space LoRA. Future routing must use hidden states or range-space features.
