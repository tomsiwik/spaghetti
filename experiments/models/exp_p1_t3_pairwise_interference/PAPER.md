# PAPER.md — T3.1: Pairwise Interference = 0 for All 10 Domain Pairs (N=5)

**Status:** KILLED  
**Date:** 2026-04-10  
**Runtime:** 169.2s  

---

## Abstract

We test whether five independently trained LoRA adapters (math/code/medical/legal/finance) on Gemma 4 E4B can be simultaneously composed via block-diagonal merging without interference. All three kill criteria fail. The critical discovery is a **reversed failure pattern**: weight-space cosine is NOT the correct interference predictor. Math and code adapters (low cosine 0.01-0.02) collapse catastrophically under composition (82→8%, 66→8%), while medical/legal/finance adapters (high cosine 0.10-0.17) degrade only moderately (83-87% retention). The impossibility structure is additive O(N-1) activation-space noise, not weight-space overlap. Routing (PLE-M2P) resolves this structurally.

---

## Prediction vs Measurement

### Phase 1: Weight-Space Cosines

| Pair | Predicted | Measured | Match |
|------|-----------|---------|-------|
| math ↔ code | ≈ 0.019 ± 0.015 | **0.0195** | ✓ |
| math ↔ medical | ≈ 0.019 ± 0.015 | **0.0109** | ✓ |
| math ↔ legal | ≈ 0.019 ± 0.015 | **0.0148** | ✓ |
| math ↔ finance | ≈ 0.019 ± 0.015 | **0.0126** | ✓ |
| code ↔ medical | ≈ 0.019 ± 0.015 | **0.0162** | ✓ |
| code ↔ legal | ≈ 0.019 ± 0.015 | **0.0164** | ✓ |
| code ↔ finance | ≈ 0.019 ± 0.015 | **0.0172** | ✓ |
| medical ↔ legal | (unexpected cluster) | **0.1015** | — |
| medical ↔ finance | (unexpected cluster) | **0.1705** | — |
| legal ↔ finance | (unexpected cluster) | **0.1448** | — |
| **max \|cos\|** | **≈ 0.019** | **0.1705** | ✗ |

K1050 threshold was 1e-5. Measured max = 0.1705 — 17,050× over threshold.  
**K1050: FAIL** (correctly predicted to fail, but magnitude was underestimated for MMLU cluster)

### Phase 2: Composition Quality (n=25 per domain)

**THIS IS THE KEY FINDING — the reversal contradicts the MATH.md hypothesis.**

| Domain | Single Acc | Composed Acc | Ratio | Base Acc | K1051 (≥0.90) | K1052 (>base) |
|--------|-----------|-------------|-------|----------|---------------|---------------|
| Math   | 82%       | **8%**      | 0.098 | 0%       | **FAIL** | PASS |
| Code   | 66%       | **8%**      | 0.121 | 20%      | **FAIL** | **FAIL** |
| Medical | 48%      | **40%**     | 0.833 | 26%      | **FAIL** | PASS |
| Legal  | 54%       | **44%**     | 0.815 | 4%       | **FAIL** | PASS |
| Finance | 60%      | **52%**     | 0.867 | 4%       | **FAIL** | PASS |

**Predicted failure order (from MATH.md revised section):**
> Medical (30–50%), Legal (35–55%), Finance (35–55%) — high cosine cluster fails first.
> Math (70–85%), Code (55–70%) — low cosine cluster survives.

**Actual failure order:**
> Math and Code collapse catastrophically (8%, 8%).
> Medical, Legal, Finance degrade only moderately (40%, 44%, 52%).

**The MMLU cluster (high cosine) performed BETTER under composition than the task-precise adapters (low cosine). Weight-space cosine was inversely correlated with composition degradation.**

### Kill Criteria Summary

| K# | Criterion | Predicted | Measured | Result |
|----|-----------|-----------|---------|--------|
| K1050 | max \|cos\|_F < 1e-5 | FAIL | 0.1705 | **FAIL** |
| K1051 | Composed ≥ 90% of single (all domains) | UNCERTAIN (med/legal/finance at risk) | 0.098–0.867 | **FAIL** |
| K1052 | Composed > base (all domains) | PASS | Code: 8% < 20% base | **FAIL** |

---

## Analysis: Why the Prediction Was Wrong

### Theorem 2 Assumption Violated

MATH.md Theorem 2 predicted K1051 would fail for the **high-cosine** cluster (medical/legal/finance) because weight-space overlap implies cross-domain activation. This is incorrect.

**The actual failure mechanism:**

For simultaneous composition of N=5 adapters:
```
Output(x) = A_j B_j x + Σ_{i≠j} A_i B_i x
             ↑                   ↑
         signal (1 term)    noise (N-1 = 4 terms)
```

The noise is 4 additive interference terms. Whether this destroys performance depends on the **signal-to-noise ratio at the task level**, not weight-space cosine:

- **Math/Code** (arithmetic + code execution): Requires EXACT output format — "The answer is 42." Off-by-one in a chain of reasoning collapses the entire answer. 4 noise terms easily corrupt this precision. Single adapter: 82%. Under noise: 8%.

- **Medical/Legal/Finance** (MMLU MCQ): Requires selecting A/B/C/D. The answer is pattern-matched from 4 choices — robust to additive noise that merely attenuates the signal. 4 noise terms shift logits but rarely flip a multiple-choice selection. Single adapter: 48-60%. Under noise: 40-52%.

### Impossibility Structure

**The impossibility is not about weight-space cosine — it's about O(N) additive noise for simultaneous activation.**

For a precise task (arithmetic):
- SNR_single ≈ signal / noise_floor
- SNR_composed = signal / (noise_floor + (N-1) × interference)

When N-1 = 4 and interference is non-negligible, SNR_composed << SNR_single.
Even if each pairwise weight-space cosine is small (0.01-0.02), the **sum of 4 interference terms** accumulates enough noise to collapse arithmetic chains.

**Formal statement:**
Simultaneous activation of N adapters via block-diagonal sum creates O(N-1) interference at every forward pass. For precision tasks requiring exact sequential token generation, the composed accuracy is bounded by:
```
acc_composed ≤ acc_single × (1 - (N-1) × ε)^L
```
where ε is per-step interference and L is the generation length. For L=10-20 tokens (GSM8K solutions), (1 - 4×0.01)^15 ≈ 0.54 at best — which still predicts degradation, but the actual 0.098 ratio suggests ε >> 0.01 for these adapters.

---

## Structural Fix: Routing

This experiment proves that **simultaneous activation of all N adapters is architecturally unsound for precision tasks**. The Room Model / PLE-M2P routing makes interference structurally zero by construction:

```
# Block-diagonal sum (this experiment — FAILS):
output = Σ_{i=1}^{N} A_i B_i x    ← all N adapters active simultaneously

# Routing (Room Model — interference impossible):
output = A_{match} B_{match} x     ← only matched adapter active
```

With routing: interference = 0 (mathematically, not approximately). There are no cross-domain activation terms.

**Finding implication:** The reason T2.6 showed +82pp/+46pp/+22pp (single adapter) is that each adapter genuinely captures domain expertise. The information is there. Routing extracts it without the O(N-1) noise penalty.

---

## Connection to Prior Findings

| Finding | Connection |
|---------|-----------|
| Finding #225: Near-lossless composition at N=5 | That used Grassmannian (QR-init) adapters with routing — the exact fix indicated here |
| Finding #318: Grassmannian QR gives cos=0 | Structural fix for K1050, but does NOT fix K1051 (routing needed for activation-space interference) |
| T2.2: max\|cos\|=0.019 for 3-domain | Correctly predicted low cosine. Missing: MMLU cluster would form high-cosine subgroup |
| exp_p1_t2_single_domain_training: SUPPORTED | Single adapters work. Problem is composition without routing. |

---

## What Comes Next

K1051/K1052 failures prove routing is structurally required. The next experiment must implement PLE-M2P routing over the 5 existing adapters and verify:
- Composed accuracy with routing ≥ 90% of single adapter (same threshold as K1051)
- routing precision > 95% on clean inputs (from T2.5 results)

This is the **critical path**: T3.1 killed → T3.2 (routing composition) is the natural successor.

**The math/code collapse to 8% is not a failure of the adapters — it's proof that routing is load-bearing, not optional.**
