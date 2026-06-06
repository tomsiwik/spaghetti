# Adversarial Review: Exp 4 — N=5 Expert Scaling

**Reviewer**: Ralph (Critic hat)
**Date**: 2026-03-04
**Verdict**: **PROCEED**

---

## 1. Derivation Verification

### Parameter counts
- Capsule groups: 4 layers × 20 groups × 64 capsules × 2 × 64 = 655,360 ✓
- Router: 4 layers × (64 × 20) = 5,120 ✓
- Base (frozen): ~72K ✓
- Total ~732K ✓

### Orthogonality computation
Code verified (`compute_delta_orthogonality`): flattens capsule weight deltas
per domain per layer, computes pairwise cosine similarity. Correct use of
`dot / (||a|| × ||b|| + ε)`. Reports per-layer and aggregate stats. ✓

### Protocol
- Joint baseline: 1500 steps rotating through 5 domains. Fair comparison. ✓
- Composition: 300 pretrain + 5×300 fine-tune + 200 calibrate = 2000 steps.
  More total compute than joint — acknowledged in Honest Limitations. ✓
- k/G ratio maintained at 0.5 (10/20) as at N=2 (4/8). ✓
- Calibration: freeze-all-except-router, rotate domain batches. ✓

### Composition code
`compose_from_shared_base_n`: copies base attention/embeddings, slots domain
capsule groups at correct offsets. Test (`test_composition_preserves_weights`)
verifies weight slotting for 2 domains. ✓

---

## 2. Hidden Assumptions Audit

### 2a. Router analysis methodology — MINOR CONCERN
The router entropy is computed on the **mean probability distribution** across
tokens: `H(E[p])`. This is NOT the same as `E[H(p)]` (mean per-token entropy).
Jensen's inequality guarantees `E[H(p)] ≤ H(E[p])`.

The claim "router converges to near-uniform distribution" (H/H_max ≈ 0.999)
could mask sharp per-token routing that averages out. A router that strongly
prefers group 1 for domain A and group 2 for domain B looks uniform on average.

**Impact**: Characterization only. The core result (composition quality +1.6%)
is measured directly via val loss, not via router entropy. The entropy number
is illustrative, not load-bearing. **Not a validity concern.**

### 2b. Data imbalance
Domain sizes range 4.4× (u_z: 2,359 vs a_e: 10,479). With 300 fine-tuning
steps each at batch_size=32, u_z sees ~4× more epochs than a_e. PAPER.md
correctly notes smaller domains degrade more (+3.0% vs -0.1%). The joint
baseline has the same imbalance, so the comparison is fair.

### 2c. k/G=0.5 is generous
At k=10 of G=20, half of all groups are active per token. This provides
minimal compute savings. Acknowledged in Honest Limitations §5. At macro
scale, lower k/G ratios will be needed, which was already tested in Exp 2
(sparse routing) — k=2 is validated minimum at micro scale.

### 2d. Calibration vs domain count
The experiment tests one calibration budget (200 steps). It doesn't sweep
calibration steps at N=5 to find the optimal budget. The +1.6% degradation
(vs -0.2% at N=2) COULD be partly calibration-budget-related. However,
the result passes the 5% threshold comfortably, so this doesn't change the
verdict.

---

## 3. Prior Art Check

This experiment is a scaling validation of an established protocol, not a
novel mechanism. Key references:

- **LoRAHub** (Huang et al., 2024): Composition of LoRA adapters with learned
  mixing weights. Similar concept but at LoRA (linear) level, not capsule level.
- **Model soups** (Wortsman et al., 2022): Weight averaging. Task arithmetic
  baseline (+5.6%) confirms their findings on dilution.
- **Switch Transformer** (Fedus et al., 2021): MoE with k=1 routing at scale.
  Referenced correctly in sparse routing discussion.

No missing critical prior art identified. The capsule-level composition with
softmax routing calibration appears to be the paper's own contribution from
prior experiments.

---

## 4. Hypothesis-Experiment Alignment

**Stated hypothesis**: Composition protocol scales from N=2 to N=5.

**Three sub-questions tested**:
1. Subspace orthogonality? → Measured via cosine similarity. ✓
2. Composition quality within 5%? → Measured via val loss vs joint. ✓
3. Calibration scales linearly? → Tested 200 steps (2× for 2.5× domains). ✓

The experiment tests exactly what it claims. ✓

---

## 5. Issues Summary

| Issue | Severity | Impact |
|-------|----------|--------|
| Router entropy = H(E[p]) not E[H(p)] | Minor | Characterization only |
| Data imbalance across 5 domains | Minor | Fair comparison (both approaches) |
| k/G=0.5 generous sparsity | Known limitation | Acknowledged, deferred to macro |
| No calibration step sweep at N=5 | Minor | Passes threshold comfortably |
| Cosine 0.000→0.112 extrapolation | Note | ~0.5 concern at N≈9-10 (linear extrapolation) |

No mathematical errors found. No hidden assumptions that invalidate results.
No mechanism broken in principle. All kill thresholds pass with margin.

---

## 6. Verdict: PROCEED

The experiment is clean, well-designed, and thoroughly documented. It answers
the stated research question definitively: yes, the composition protocol
scales to N=5 with +1.6% degradation (well within the 5% kill threshold).
Orthogonality degrades gracefully (cos 0.112, well under 0.5). Calibration
at 200 steps is sufficient (under 400 limit).

The micro arena is exhausted. The five experiments have systematically explored
the composition mechanism space at d=64. Further micro experiments offer
diminishing returns — the remaining questions (contrastive keys, sparse k=1,
real domain routing) are scale-bound.

**Approved for integration into VISION.md and FINDINGS.md.**
