# C1.2: Scale Safety on Gemma 4 — Results

**Status:** KILLED  
**Type:** Verification  
**Reference:** PoLAR arxiv 2506.03133; C0.2 Finding #439; T0.2 (killed)

---

## Prediction vs Measurement Table

| Kill Criterion | Theorem | Predicted Value | Measured Value | Pass? |
|---|---|---|---|---|
| KC10: std LoRA degradation < 10pp (scale=5→20) | T1+T2 | < 10pp (borderline) | **0.0pp** | **PASS** |
| KC11: std LoRA variance < 5pp (scale={5,10,20}) | T3 | < 5pp | **13.3pp** | **FAIL** |
| KC12: mechanism documented | — | PASS | PASS | **PASS** |

---

## Key Results

### Phase 0: B-Matrix Row Norm Profile

| Metric | Value |
|---|---|
| Layers | 42 |
| Total rows | 252 |
| Mean row norm | 0.357 |
| Std row norm | 0.065 |
| Max/Min ratio | 2.28 |

The trained math adapter (T2.1) has lora_b rows with moderate variation (2.28× ratio). This is not unit-norm (which PoLAR would require), indicating standard LoRA training.

### Phase 1: Standard LoRA Scale Sensitivity (KC10 + KC11)

| Scale | Correct/15 | Accuracy |
|---|---|---|
| s = 5 (below training) | 7/15 | **46.7%** |
| s = 10 (near 1.67× training) | 9/15 | **60.0%** |
| s = 20 (3.3× training) | 7/15 | **46.7%** |

- **KC10 PASS:** scale=20 accuracy = scale=5 accuracy (0.0pp degradation)
- **KC11 FAIL:** variance = 60.0 - 46.7 = 13.3pp > 5pp threshold

### Phase 2: Post-Hoc B Normalization (Failure Mode Documentation)

| Scale (nominal) | Effective scale | Accuracy |
|---|---|---|
| s = 5 | ~14× (5 × 2.80) | 73.3% |
| s = 10 | ~28× (10 × 2.80) | 13.3% |
| s = 20 | ~56× (20 × 2.80) | 6.7% |

Post-hoc normalization inflates effective adapter magnitude by 2.80× (row norms 0.357 → 1.0). At "effective scale=56", outputs degrade catastrophically.

---

## Critical Behavioral Observation (Metrics Masking Degradation)

**KC10 passes the accuracy metric but FAILS behaviorally at scale=20.**

At scale=20, responses show repetition loops and garbling:
- "What is 12 × 3?" → `"12 + 3 = <<12+3=12=15>>15"` (wrong operation, garbled format)
- "Solve: 3x+6=21" → `"3x + 6 = 21\n------=<<27=27.77777777..."`  
- "What is 9²?" → `"9 * 9 = <<9*9=9=999999999999999"`
- "What is 4³?" → `"4 * 4 = 4 = <<4*4=44444444444444"`
- "If x=7, y=3: x²-y²?" → `"7 = <<7=7=7=7=7=7=7=7=7=7$7$7"`

The 46.7% accuracy at scale=20 is maintained by EASY questions (7+8, 12×3, 100-37, √81, 17×6) where the correct number appears somewhere in the garbled output. Complex questions fail.

**At scale=5, responses are also degraded** (model enters verbose thinking mode `<|channel>thought` that doesn't resolve in 30 tokens), similarly failing complex questions.

**The "peak" at scale=10 (60%)** is because:
- scale=10 is closest to the "ideal" above training scale=6
- Model produces clean, direct answers (no garbling at scale=10)
- Both scale=5 (under-strong adapter) and scale=20 (over-strong adapter) cause different failure modes

---

## Impossibility Structure (Why KC11 Cannot Pass)

**Theorem 3 prediction was correct in structure but wrong in magnitude.**

Theorem 3 predicted: "unit-norm B reduces scale sensitivity." But KC11 was redesigned to test **standard LoRA natural variance** (not direction-preserving). For standard LoRA:

The variance across scale={5,10,20} follows a hump centered at training scale:
- Below training scale (s < training_scale): adapter contributes too little → degraded
- At/above training scale (s ≈ 1-2× training_scale): optimal contribution
- Far above training scale (s >> training_scale): adapter overwhelms base model → garbled

For KC11 to pass (variance < 5pp), the accuracy would need to be flat across a 4× scale range (5 to 20). This requires **scale-invariant output**, which QK-norm alone cannot provide (Theorem 1: QK-norm bounds magnitude, not direction).

**The direction-preserving fix (KC11 original intent) is NOT post-hoc normalization.**  
Post-hoc B normalization changes effective magnitude, exacerbating scale sensitivity (variance=66.7pp). True direction-preserving requires PoLAR training (C1.1 Finding #442), where both A and B are on the Stiefel manifold throughout training.

---

## Summary of Findings

| Finding | Conclusion |
|---|---|
| KC10 PASS | Gemma 4 QK-norm makes scale=5 and scale=20 EQUIDISTANT from optimum (symmetric scale sensitivity) |
| KC11 FAIL | 13.3pp variance from scale=10 "sweet spot" hump; scale invariance needs PoLAR training |
| KC12 PASS | Mechanism: symmetric protection, not asymmetric degradation; T0.2's -36pp was magnitude explosion (Qwen3 no QK-norm) |
| Post-hoc normalization | Catastrophic (66.7pp); effective scale inflation = 2.80× |
| Behavioral degradation | Scale=20 produces garbled repetition loops; accuracy metric masks quality degradation |

---

## Connection to Pierre P1 Architecture

- **For production deployment:** Set adapter inference scale ≈ training scale (scale=6 for T2.1 adapters). 4× scale change (5→20) causes behavioral degradation even though the accuracy metric appears stable.
- **For direction-preserving:** Use PoLAR training (C1.1) rather than post-hoc normalization. PoLAR trains B on the Stiefel manifold, preserving magnitude while improving rank structure.
- **C1.3 next:** Should test PoLAR-trained adapters at scale={3×, 1×, 0.3×} training scale to verify whether joint Stiefel constraint provides genuine scale invariance (not just metric-level stability).
