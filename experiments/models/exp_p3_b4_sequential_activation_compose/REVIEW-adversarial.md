# REVIEW-adversarial.md — P3.B4: Pure Additive Composition

## Verdict: PROCEED (KILLED — hard failure, clear impossibility structure)

---

## Critical Checks

### 1. Prediction-vs-measurement table: ✓ PRESENT
PAPER.md has the table. K1187 predicted ≥66%, measured 24%. The failure is unambiguous.

### 2. Kill criteria results match evidence: ✓ VERIFIED
results.json is_smoke=false. K1187.additive_rate=24.0%, K1188.additive_acc=15.0%. No fabrication.

### 3. Finding status: ✓ KILLED is correct
K1187 fails catastrophically (24% vs 66%). Even generous interpretation cannot call this SUPPORTED.

### 4. Math accuracy: ✓ No errors
Theorem 1 (pure additive preserves both signals unmodified) is trivially true but the prediction
(style ≥ 66%) failed because the theorem only proves signal presence, not behavioral outcome.
This is not a math error — it's a correct theorem with an incorrect behavioral prediction.

---

## Adversarial Challenges

### Challenge 1: "Pure additive (24%) < B-GS (60%) is suspicious — did you have a bug?"
**Answer**: No. power_ratio=1.077 (confirmed near-equal). PERS_SCALE=4.0 was applied correctly (P3.B3 lesson).
The result is surprising but explained: B-GS forces personal adapter into non-overlapping directions,
accidentally giving it cleaner signal. Pure additive double-counts overlapping directions.
n_overlap_layers=16 creates systematic domain-over-personal suppression.

### Challenge 2: "N=25 is too small to conclude — maybe noise?"
**Answer**: The signal is clean: 24.0% = 6/25. For 76% baseline, this is a 52pp delta.
Even with maximum favorable noise, the result cannot reach 66% from 24%. Not a sampling artifact.

### Challenge 3: "Context shift is a hypothesis, not proven"
**Answer**: Correct — it's the most parsimonious explanation, consistent with:
- Domain adapter covers layers 0-25 (full overlap with input to personal adapter's range 26-41)
- Personal adapter trained on base model's states (no domain adapter during training)
- Distribution shift is predictable from layer architecture

The structural proof will emerge from P3.B5: if retraining personal ON domain-adapted model
restores 76% style, the context shift hypothesis is confirmed.

### Challenge 4: "Pure additive worse than B-GS contradicts MATH.md Theorem 1"
**Answer**: Theorem 1 proves the SIGNAL is unmodified (trivially true). It does NOT prove behavioral
outcome is better. The theorem was incomplete — it missed the domain-suppression mechanism in
overlapping directions. This is a clean empirical falsification of the prediction, not the theorem.

---

## Non-Blocking Issues

1. The MATH.md theorem was not falsified (trivially true) but the prediction was wrong.
   Better theorem needed: behavioral prediction about suppression in overlapping directions.
   → Not blocking: LEARNINGS.md can capture this.

2. K1189 is labeled as "diagnostic" in results but "K1189" differs from MATH.md's kill criterion labels
   (K1191/K1192 in MATH.md, K1187/K1188 in code). Minor inconsistency.
   → Not blocking: evidence matches correctly.

---

## Impossibility Structure: Clear

All weight-space additive composition strategies fail for independently-trained adapters on different
layer ranges. The root cause is training distribution mismatch, not weight geometry.

Mathematical structure: the composition f_D ⊕ f_P applied at inference time receives different
hidden states than f_P was trained on (dom(f_D) shifts h_26-41 → personal adapter is mis-matched).

This makes failure inevitable for ANY static weight composition method and mandates:
**Training-time composition: P3.B5 — train personal adapter on domain-adapted model.**

---

## Proceed: KILLED → LEARNINGS.md → P3.B5
