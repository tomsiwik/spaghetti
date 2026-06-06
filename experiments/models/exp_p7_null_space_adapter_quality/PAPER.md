# P7.A1: Adapter Restricted to Null Space — Quality Preserved?

## Audit Rerun (2026-04-17): Verdict revised to KILLED on metric-swap grounds

**Original verdict (2026-04-11):** SUPPORTED (3/3 kill criteria pass).
**Audit verdict (2026-04-17):** **KILLED** (1/3 pass; K1297 and K1298 fail structurally on pre-flight antipattern #6 — "KC measures wrong object").

**Why KILLED:** The DB-registered kill criteria pre-register GSM8K accuracy (K1297) and MMLU accuracy (K1298) as the behavioral metrics. The code and MATH.md instead measure training-loss ratio on 20 memorized math texts (K1297) and next-token PPL on 5 hand-curated general-knowledge prose snippets (K1298). Training loss at memorization scale (both adapters reach PPL=1.03) is not a proxy for GSM8K accuracy — the ratio is mechanically ~1.0 regardless of null-space effect. PPL on 5 hand-picked texts is not a proxy for MMLU multi-task accuracy. Per PLAN.md §1 pre-flight rule #6 (antipattern: KC measures wrong object), `supported` is blocked.

**Code was NOT re-executed.** The antipattern is structural — a metric re-specification, not a transient bug. Re-running the existing code would reproduce the same mis-measured numbers. A v2 experiment (`exp_p7_null_space_adapter_quality_v2` to be filed) must pre-register GSM8K + MMLU eval harnesses at MATH.md time and train on non-trivial data (not 20 memorized texts) so the loss ratio is informative.

**K1299 (orthogonality) PASS is preserved.** Orthogonality was correctly measured and the proof (Theorem 1) is exact by construction; the 1.33e-5 violation is 100x below threshold. This is the only behavioral claim credited by this audit.

**Reusable behavioral findings (preserved in LEARNINGS.md, not credited as KC pass):**
- Null-space reparameterization achieves exact orthogonality to W_v across 8 non-shared layers.
- Gemma 4 E4B KV-sharing: layers 24-41 receive pre-computed KV from layers 22/23; v_proj is dead code on those layers. Future Gemma 4 v_proj/k_proj adapters MUST target layers 16-23. (Reusable architectural check.)
- At memorization scale, null-space restriction does not slow convergence vs unrestricted (both reach loss 0.037 in 500 steps).

**Artifacts:** MATH.md (git-clean since 78538d2, no KC swap post-hoc), run_experiment.py (unchanged, known-buggy under metric-swap tag), results.json (newly written reconstruction with verdict=KILLED), REVIEW-adversarial.md (audit section added), LEARNINGS.md (metric-swap note added).

---

## Original report (2026-04-11 — now SUPERSEDED, retained for reference)

## Result: SUPPORTED (3/3 kill criteria pass) — SUPERSEDED BY AUDIT ABOVE

Null-space LoRA achieves 98.7% of unrestricted LoRA quality while maintaining
strict orthogonality to W_v (max violation 1.33e-5). The null-space restriction
costs almost nothing: both adapters converge to the same final loss (0.037) and
identical math PPL (1.03). Orthogonality is exact to numerical precision.

## Prediction vs Measurement

| Prediction | Source | Predicted | Measured | Match |
|-----------|--------|-----------|----------|-------|
| P1: Quality ratio >= 0.80 | Theorem 2 (gradient retention d_null/d_in = 0.80) | >= 0.80 | 0.987 | PASS (far exceeds) |
| P3: General PPL delta < 1pp degradation | Theorem 3 (additive, no corruption) | < 1% worse | 95.6% BETTER (8155 -> 362) | PASS (improved) |
| P4: Orthogonality < 1e-4 | Theorem 1 (exact by construction) | < 1e-4 | 1.33e-5 | PASS |

## Kill Criteria

| ID | Criterion | Threshold | Result | Verdict |
|----|-----------|-----------|--------|---------|
| K1297 | Null-space quality >= 80% of unrestricted | ratio >= 0.80 | 0.987 | **PASS** |
| K1298 | General PPL not degraded > 1% | adapter/base <= 1.01 | 0.044 | **PASS** |
| K1299 | Orthogonality max\|W_v @ A_eff\| | < 1e-4 | 1.33e-5 | **PASS** |

## Configuration

- Model: Gemma 4 E4B (4-bit quantized, 42 layers)
- Target: v_proj on last 8 non-shared layers [16-23]
- LoRA rank: 16, scale: 20.0, lr: 1e-4, 500 iterations
- Training data: 20 math instruction texts (next-token prediction)
- Null-space dims: 2048 (sliding layers), 1536 (full attention layers)

## Training Dynamics

| Metric | Unrestricted | Null-Space |
|--------|-------------|------------|
| Trainable params | 409,600 | 327,680 |
| Step 0 loss | 5.719 | 5.719 |
| Step 100 avg loss | 0.492 | 0.521 |
| Step 200 avg loss | 0.081 | 0.082 |
| Final loss (avg last 20) | 0.037 | 0.037 |
| Training time | 121.8s | 123.4s |
| Math PPL (held-out) | 1.03 | 1.03 |
| General PPL | 250.2 | 362.0 |
| lora_b avg norm | 0.487 | 0.471 |

Both adapters converge at nearly identical rates. The null-space adapter has
20% fewer parameters (327K vs 410K) because its lora_a operates in the
smaller null-space rather than full input space, but this doesn't hurt quality.

## Orthogonality Verification (K1299)

| Layer | max\|W_v @ A_eff\| | A_eff norm |
|-------|-------------------|------------|
| 16 | 1.33e-05 | 2.47 |
| 17 | 1.26e-05 | 2.45 |
| 18 | 1.01e-05 | 2.44 |
| 19 | 1.10e-05 | 2.53 |
| 20 | 1.16e-05 | 2.57 |
| 21 | 1.01e-05 | 2.46 |
| 22 | 1.14e-05 | 2.54 |
| 23 | 1.05e-05 | 2.51 |

All violations are ~100x below the 1e-4 threshold. The null-space basis Q is
computed from SVD at epsilon=1e-3, and the resulting projection is exact to
float32 numerical precision.

## Architectural Discovery: Gemma 4 KV-Sharing

**Critical bug in first run:** The initial experiment targeted layers 34-41,
which are all in Gemma 4's KV-sharing range. Layers 24-41 receive pre-computed
KV from layers 22/23 via `shared_kv`, causing v_proj to be **dead code** on
those layers. This produced zero-effect adapters with identical loss curves.

**Fix:** Target layers 16-23 (last 8 non-shared layers). Verified via:
1. `previous_kvs` mapping shows layers 24+ share from layers 22/23
2. Logit-delta diagnostic confirmed zero effect on shared layers
3. Logit-delta confirmed non-zero effect on non-shared layers

This is a **mandatory architectural check** for any adapter targeting k_proj or
v_proj on Gemma 4: verify the layer actually computes its own KV.

## Implications for Composition

1. **Zero-interference guarantee:** Null-space adapters on v_proj are orthogonal
   to W_v by construction. Two adapters in different null-space subspaces
   cannot interfere through the weight matrix.

2. **Capacity:** At r=16, each adapter uses 16 of 2048 null-space dimensions.
   This gives 128 non-overlapping slots per layer — 5x our 25-domain target.

3. **No quality penalty:** The 98.7% quality ratio means we can use null-space
   restriction without sacrificing adapter performance.

4. **Next step (P7.A2):** Verify that TWO null-space adapters composed on the
   same layer maintain independence — predicted by orthogonality but needs
   empirical validation of downstream behavior.

## Caveats

1. Training data is small (20 texts, 500 iters). Both adapters essentially
   memorize the dataset (PPL = 1.03). Larger-scale training may reveal
   differences not visible at this scale.

2. General PPL improvement (8155 -> 250/362) is an artifact of the 4-bit
   quantized base model having very high PPL on these texts. The adapter
   provides useful features that incidentally help general knowledge.

3. The quality comparison is on training-domain data only. Cross-domain
   generalization (e.g., math adapter evaluated on legal text) is untested.
