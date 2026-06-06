# PAPER: M2P Teacher Distillation — Cross-Model Knowledge Transfer via KL Distillation

## Abstract

This experiment tested whether an M2P Transformer trained on a larger teacher's hidden
states (via KL distillation) could generate LoRA adapters that close the quality gap
between a student model (d=256, L=2) and a teacher model (d=512, L=4) on three symbolic
domains (arithmetic, sort, reverse). The key result is that the M2P adapter actively
degraded the student on all three domains (negative gap closure), with student+M2P loss
exceeding student base loss by 25-33%. K853 FAILS: 0/3 domains achieve ≥0.50 closure.
K854 PASSES: no architecture crash from the d_T=512 → d_M2P=64 projection.

---

## Prediction vs. Measurement Table

| Prediction | Source | Expected | Measured | Result |
|------------|--------|----------|----------|--------|
| P1: Projection cosine sim (intra-domain) | Theorem 2 + JL | > 0.5 | 0.986–0.999 | PASS (high similarity) |
| P2: Quality gap closure ≥ 0.50 for ≥2/3 domains | Theorem 1 Corollary 2 | ≥ 0.50 | −5.71, −1.74, −0.91 | FAIL (all negative) |
| P3: No regression below student base | Corollary 1 (KL ≥ 0) | student_m2p ≤ student_base | 2.09, 2.15, 2.15 vs base 1.66, 1.71, 1.66 | FAIL (regression) |
| P4: KL loss decreases during M2P training | Theorem 2 gradient | monotone decrease | 18.2%, 11.4%, 12.6% reduction | PASS (decreased) |
| P5: Projection cosine > 0.2 above random | Theorem 2 | > 0.2 | 0.986–0.999 | PASS |

---

## Kill Criteria Assessment

**K853:** `quality_gap_closure` ≥ 0.50 for ≥2/3 domains — **FAIL**

| Domain     | student_base_loss | student_sft_loss | student_m2p_loss | gap_closure |
|------------|-------------------|------------------|------------------|-------------|
| arithmetic | 1.6629            | 1.5880           | 2.0909           | −5.711      |
| sort       | 1.7106            | 1.4561           | 2.1522           | −1.735      |
| reverse    | 1.6567            | 1.1215           | 2.1457           | −0.914      |

All three domains show large negative closure: the M2P adapter INCREASES loss above
the student base on every domain (regression of ~0.4–0.5 nats).

**K854:** No architecture crash from dimension mismatch — **PASS**

The Linear(d_T=512 → d_M2P=64) projection layer was constructed and run without error.
Teacher hidden states were correctly projected into M2P's input space for all domains.

---

## Analysis

### What the Math Predicted vs. What Was Measured

**P3 violation (no-regression) is the key failure.**

MATH.md's Corollary 1 states that minimizing KL(p_T || p_M) cannot make p_M worse than
student base — because KL ≥ 0 and the gradient always points toward p_T. However, this
guarantee assumes correct gradient computation. The measured M2P outputs are 25–33% worse
than student base, which is inconsistent with a correctly minimized KL objective.

**Possible causes of the P3 violation:**

1. **The KL distillation gradient may be computing correctly, but the ADAPTER IS NOT
   ACTUALLY BEING APPLIED correctly.** If M2P generates B-matrices that are then applied
   as W = W_base + A·B^T but the A-matrices are Grassmannian (orthogonal) while the B
   is random or near-zero, the effective perturbation could add noise rather than signal.

2. **Projection captures intra-domain similarity, not inter-domain discrimination.**
   The measured intra-domain cosine similarity is 0.986–0.999 (near-identical hidden
   states within a domain), which is expected and good. But the metric logged does NOT
   show cross-domain cosine similarity. High intra-domain similarity alone does not
   confirm that different domains have distinct projections. If all three domains project
   to nearly the same M2P input vector, M2P learns one average adapter that fits none.

3. **Student base loss < teacher domain loss in this experiment.** The student base
   achieves lower loss than the teacher SFT on every domain (e.g., arithmetic: 1.663 vs
   1.718). This violates Assumption A2 from MATH.md: "Teacher domain loss < student base
   domain loss." When the student is ALREADY BETTER than the teacher, the KL distillation
   gradient pushes the student toward a WORSE distribution (the teacher's). This is the
   most likely root cause: the teacher is not actually a quality ceiling for the student
   in this configuration.

4. **M2P training loss was large and noisy.** Final KL losses of 7.0–7.5 nats are high
   relative to the per-token NTP losses of 1.5–2.1, suggesting M2P is producing soft
   target distributions that diverge from the teacher's by a large margin.

### The Assumption A2 Failure

MATH.md Section E lists Assumption A2: "Teacher domain loss < student base domain loss."
The experiment measured the opposite: student base loss (1.66–1.71) is lower than teacher
SFT loss (1.72–1.88) on every domain. This means the teacher (d=512, L=4, 600 SFT steps)
converged to a slightly WORSE distribution than the already-well-trained student base after
1200 pre-training steps.

When A2 is violated, "distillation" pushes the student AWAY from its current good state
toward the teacher's inferior distribution. The KL objective is functioning correctly — it
is successfully moving the student toward the teacher — but the teacher itself is a
regression target. This explains the P3 violation without any training bug.

### Projection Analysis (P1)

Intra-domain cosine similarity is very high (0.986–0.999), confirming that teacher hidden
states are consistent within a domain. However, this experiment did not log cross-domain
cosine similarity, so we cannot confirm whether the projection distinguishes domains.
The JL lower bound requires d_M2P ≥ O(log(3)/0.01) ≈ 110 for ε=0.1 preservation, but
d_M2P=64 is below this. P1's "intra > 0.5" passes, but the full test (cross < 0.8) is
unverified.

---

## Conclusion

**Status: KILLED**

K853 fails on all three domains with large negative gap closure (−5.71, −1.74, −0.91).
K854 passes (architecture works). The no-regression guarantee (P3, Corollary 1) is also
violated: student+M2P loss exceeds student base on all domains.

**Root cause:** Assumption A2 is violated. The student base model (1200 steps pre-trained)
is already better than the teacher SFT model (1200 pre-train + 600 SFT steps) on all
three domains. KL distillation correctly minimizes KL(p_T || p_M), but since the teacher
is the LOWER quality model, this actively degrades the student. The no-regression
guarantee of Corollary 1 does not apply when the teacher is worse than the student base;
it guarantees no regression TOWARD p_T, not no regression in absolute terms.

**Impossibility structure:** Teacher distillation via KL requires the teacher to be
strictly better than the student at the target task. When teacher and student share the
same pretraining data and the teacher has more parameters but fewer effective training
steps per-capacity-unit, the student can match or exceed the teacher. At this scale,
doubling model size (512 vs 256) does not guarantee a quality gap that distillation can
exploit. The KL distillation becomes a regression target when A2 is violated.

**Next steps if pursued:**

1. Verify A2 before training: measure teacher_sft_loss < student_base_loss. If not, use
   a longer teacher SFT or weaker student base pre-training.
2. Log cross-domain projection cosine similarity (not just intra-domain) to test P1 fully.
3. Consider scaling the teacher more aggressively (d_T=1024, L=8) to guarantee a larger
   quality gap, or reducing student pre-training steps so teacher has a real advantage.
4. Alternative: train M2P with NTP loss on teacher-domain data (as in Finding #339),
   which does not require the teacher to be better than the student.
