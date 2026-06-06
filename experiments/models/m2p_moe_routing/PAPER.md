# PAPER.md: M2P MoE Routing — V2 Rerun (audit-2026-04-17, metric-swap)

## V2 Rerun (this run) — Verdict: KILLED on K860

The DB row tracks **K860** ("Router falls back to uniform allocation across all
domains"). The rev-1 code never extracted router weights — it reported only
K855/K856/K857 (quality + Grassmannian). This V2 closes the audit gap by
adding a router-uniformity measurement and gating the verdict on K860 (per
MATH.md §K.4: DB-tracked KC is ground truth).

### Pre-Registered Predictions vs. Measured (V2)

| KC (locked in MATH.md §K) | Predicted | Measured | Status |
|---------------------------|-----------|----------|--------|
| **K860**: m̄_route ≥ 0.50 | FAIL: m̄ ≈ 0.25 ± 0.10 | **m̄ = 0.3432** | **FAIL** |
| Aux: H̄ / ln(n_experts) ≈ 1.0 (uniform) | ≈ 1.0 | **0.9663** (96.6% of uniform) | matches FAIL |
| Aux: K855_aux median quality ≥ 25% | improvement vs rev-1 (39.1%) NOT predicted | **12.1%** | FAIL |
| Aux: K856_aux min quality ≥ −10% | rev-1 was −322% | **−377.0%** (repeat) | FAIL |
| Aux: K857_aux Grassmannian \|cos\| ≤ 1e-5 | structural PASS | **0.0** | PASS |

K860 measurement details:
- Per-domain max route weight: arithmetic=0.2796, reverse=0.3826, repeat=0.4077, sort=0.3162, parity=0.3301
- Mean across domains: **m̄ = 0.3432** (only 9pp above the uniform 0.25 baseline; threshold for PASS was 0.50)
- Mean entropy H̄ = 1.3396 / max ln(4) = 1.3863 → **96.6% of uniform-maximum entropy**
- 3 unique argmax experts out of 4 → **expert_3 receives no domain's argmax** (degenerate)

### Conclusion (V2)

The pre-registered prediction in MATH.md §K.2 is confirmed: the soft router
collapses to near-uniform allocation under round-robin shared-gradient
training. The MoE Routing variant of M2P does **not** rescue B-matrix
centroid collapse — the routing degree of freedom collapses for the same
information-theoretic reason as the B-output degree of freedom (Finding #341).
B-matrix |cos| = 0.9774 is essentially unchanged from the unconditioned
baseline (0.9956); centroid_destabilized = False.

**Verdict: KILLED.** Closing on K860.

### Permanently Learned (propagate to siblings)

1. **Soft routing without auxiliary loss collapses.** A learned router with
   no load-balancing or entropy-penalising loss term has no gradient
   incentive to specialise — uniform allocation is a saddle minimiser of the
   round-robin sum. Any future MoE-style M2P variant must include either
   (a) hard top-k gating with Gumbel noise, or (b) an explicit
   load-balance / entropy-regularising auxiliary loss (Switch Transformer
   §2.2; arXiv:2101.03961).
2. **Mode collapse generalises across degrees of freedom.** Whichever
   parameter is "free" under shared-gradient training will collapse —
   B-matrices in rev-1 of m2p_distillation_toy, embeddings in
   m2p_domain_conditioned, router weights here. The fix must inject
   **forced** domain identity, not "available" domain signal.
3. **Audit pattern: always extract the DB-tracked KC.** The rev-1 code
   measured everything except K860, the one criterion the DB uses to gate
   the verdict. metric-swap audits are always real bugs even when the run
   "works" — verdict and KC must be tied to the DB row, not to the script's
   internal heuristics. (Same lesson as exp_m2p_loss_norm V2 K859.)

### Sibling Work (do not auto-spawn — analyst gates)

- `exp_m2p_hard_moe` (already open in DB, P2): hard top-1 Gumbel routing
  for absolute gradient isolation between experts. Directly addresses
  permanent-learning #1.
- A "K860 v2" experiment with switch-style load-balancing loss would be
  the minimum-mechanism follow-up if hard top-1 is too aggressive.

---

## Original V1 Material (retained for context — describes a stale "domain
## conditioning" run, NOT the MoE routing variant the DB tracks)

> NOTE: The text below was authored before the audit; it describes an
> earlier "domain conditioning" experiment (additive embedding injection)
> whose code was later replaced by the MoE Routing variant
> (`M2PTransformerMoE`) in `run_experiment.py`. K855/K856/K857 numbers
> below come from the prior run, not this V2 rerun. Retained as context
> because the failure mechanism (gradient homogenisation under round-robin
> training) is the same — only the degree of freedom that collapses
> changes (embeddings → router weights).

# PAPER.md: M2P with Domain Conditioning Experiment Results

## Experiment Summary

**Status:** KILLED (K856 FAIL)

**Objective:** Test Theorem 3 prediction that learned domain embeddings inject sufficient domain-distinguishing information into M2P's input space to destabilize the B-matrix centroid collapse observed in Finding #341.

**Key Claim from MATH.md:** Adding learned domain embeddings e_d ∈ ℝ^64 to M2P input (injected additively into memory tokens) makes the global centroid B_centroid a geometrically unstable fixed point, forcing per-domain B-matrix generation.

---

## Prediction vs. Measurement Table

| Prediction | Metric | Value | Threshold | Status |
|-----------|--------|-------|-----------|--------|
| K855: Median quality ≥ 25% | median_quality | 0.473 (47.3%) | 0.25 (25%) | **PASS** |
| K856: No domain < -10% | min_quality | -3.037 (-303.7%) | -0.10 (-10%) | **FAIL** |
| K857: Grassmannian \|cos\| ≤ 1e-5 | grassmannian_cos_max | 0.0 | 1e-5 | **PASS** |
| Theorem 3: Centroid destabilized (B\|cos\| << 0.9956) | m2p_b_cos_mean | 0.9785 | < 0.90 | **FAIL** |

---

## Per-Domain Quality Breakdown

| Domain | SFT Loss | M2P Loss | Quality Ratio | Pass (≥ -10%) |
|--------|----------|----------|--------------|--------------|
| arithmetic | 1.7017 | 3.1208 | 0.604 (60.4%) | PASS |
| reverse | 1.8004 | 2.4547 | 0.614 (61.4%) | PASS |
| repeat | 0.5107 | 2.9146 | -3.037 (-303.7%) | **FAIL** |
| sort | 1.8282 | 2.6795 | 0.473 (47.3%) | PASS |
| parity | 1.3012 | 4.1473 | 0.309 (30.9%) | PASS |

---

## Kill Criteria Results

### K855: Median M2P Quality ≥ 25% of SFT
- **Result:** PASS
- **Measured:** 47.3% (0.473)
- **Threshold:** 25% (0.25)
- **Margin:** 22.3 percentage points above threshold

### K856: No Domain Below -10% (No Catastrophic Collapse)
- **Result:** FAIL
- **Measured:** -303.7% (min_quality = -3.037)
- **Threshold:** -10% (-0.10)
- **Worst Domain:** "repeat"
- **Margin:** 293.7 percentage points below threshold

### K857: Grassmannian |cos| ≤ 1e-5 (Structural A Guarantee)
- **Result:** PASS
- **Measured:** 0.0 (machine precision limit)
- **Threshold:** 1e-5
- **Structural Guarantee:** The frozen Grassmannian A-matrices remain orthogonal; domain conditioning does not change A generation, so this is unaffected.

---

## Theorem 3 Check: Centroid Destabilization Analysis

**Theorem 3 prediction:** Adding learned domain embeddings {e_d} to M2P input should make the centroid state B*(θ, e_d) = B_centroid ∀d geometrically unstable, forcing per-domain B-matrix generation. Predicted B-matrix diversity (mean |cos|): ≤ 0.90 (substantial breakout from 0.9956).

| Metric | Baseline (Finding #341) | Measured | Prediction | Status |
|--------|------------------------|----------|-----------|--------|
| M2P B-matrix \|cos\| (mean) | 0.9956 | 0.9785 | ≤ 0.90 | **FAIL** |
| Reduction from baseline | — | 0.0171 | > 0.09 | **FAIL** |
| Centroid destabilized? | N/A | False | True | **FAIL** |

**Finding:** The B-matrix mode collapse persists despite domain conditioning. Mean |cos| = 0.9785 is still in collapse range and only 0.0171 below baseline 0.9956 — a 0.17% reduction vs. the predicted >9% (0.09) reduction.

---

## Comparison to Baseline (m2p_distillation_toy, rev1)

| Metric | Baseline (exp_m2p_distillation_toy) | Current (m2p_domain_conditioned) |
|--------|-------------------------------------|----------------------------------|
| Median quality | 21.9% | 47.3% |
| Min quality (repeat domain) | -329.0% | -303.7% |
| B-matrix \|cos\| mean | 0.9956 | 0.9785 |
| Mean quality | Negative (dominated by repeat failure) | -20.7% |
| K856 Status | N/A (not measured in baseline) | FAIL |

**Interpretation:**
- Median quality IMPROVED (21.9% → 47.3%), but this is driven by 4 domains achieving 30-61% quality.
- The "repeat" domain catastrophic failure PERSISTS (-329% → -303.7%), not eliminated.
- B-matrix diversity shows only marginal reduction (0.9956 → 0.9785), not the substantial breakout (>0.90) predicted by Theorem 3.
- **Centroid collapse was NOT destabilized.** Theorem 3's central prediction failed.

---

## Root Cause Analysis: Why Theorem 3 Failed

### Observation 1: B-matrix mode collapse persists with minimal reduction
Mean |cos| of M2P-generated B-matrices: 0.9785 (only 0.0171 below baseline). This is inconsistent with Theorem 3's prediction of substantial diversity (≤ 0.90).

**Hypothesis:** The domain embeddings are either:
1. Not diverging during training (all e_d remain similar after gradient updates), OR
2. M2P is not sensitive enough to e_d variations (small embedding changes → near-identical B outputs)

### Observation 2: "Repeat" domain receives wrong adapter despite conditioning
Quality ratio: -3.037 (-303.7%), meaning M2P generated a B-matrix that is catastrophically worse for the repeat task than baseline. This mirrors Finding #341's failure mode exactly.

**Hypothesis:** M2P's conditioning mechanism (additive injection into memory tokens) is too weak. The domain signal encoded in e_d is either:
1. Diluted by mean-pooling across N_MEMORY=32 tokens, OR
2. Not connected to the loss gradient through a sufficiently informative pathway

### Observation 3: Theorem 2's fixed-point argument assumes gradient pressure is sufficient
Theorem 2 (informal) argues that ∂L_d/∂e_d ≠ ∂L_{d'}/∂e_{d'} forces embeddings to diverge. However:
- The gradient is **nonlocal**: changes to e_d affect mem_d at every token position, creating a distributed gradient signal.
- The gradient magnitude depends on M2P's sensitivity to memory token perturbations.
- If M2P's attention bottlenecks memory information (concentrating on a few tokens), the embedding gradient becomes rank-deficient, allowing collapse to a low-rank subspace.

**Critical assumption violated:** Theorem 2 assumes ∂B/∂(mem) = J(θ) has full row rank with respect to e_d. If M2P is query-heavy (ignores most memory tokens), J(θ) is low-rank, and gradient pressure on e_d is insufficient to overcome the centroid attraction.

---

## Structural Explanation: Information-Theoretic Bottleneck at M2P Input

### Setup from MATH.md Section B (Reframe)
MATH.md argues: "The minimum additional information M2P must receive is log₂(N) bits to distinguish N domains." Domain embeddings {e_d} were intended to inject exactly this.

### What Actually Happened
M2P receives e_d, but the **information pathway from e_d to B output is lossy:**

1. **Memory token distribution:** e_d is added to ALL N_MEMORY=32 tokens equally.
2. **Attention bottleneck:** M2P's self-attention over memory tokens may concentrate on a few key tokens (task-relevant memory), deprioritizing tokens where domain signal is strongest.
3. **Gradient bottleneck:** If attention masks out memory tokens carrying e_d, then ∂B/∂e_d ≈ 0 in those attention channels, causing embedding gradients to vanish.

**Result:** e_d is present in the input, but not effectively used by M2P. The centroid collapse is not destabilized because M2P's routing (via attention) functionally ignores domain signals.

---

## Failure Mode: Domain Embeddings Are Present but Inert

**Failure hypothesis (testable):**
1. After training, domain embeddings diverge in parameter space (e_d ≠ e_{d'}).
2. However, M2P's output B_d does not vary meaningfully with e_d.
3. The gradient ∂L_d/∂e_d pushes e_d in theoretically-correct directions, but the loss landscape is so flat with respect to B that changes in e_d produce negligible changes in L_d.

**This would explain:**
- Why K855 can still PASS (median quality 47.3% is decent for 4/5 domains).
- Why K856 FAILS (repeat domain still receives wrong adapter because M2P failed to condition on e_d).
- Why Theorem 3's centroid destabilization doesn't manifest (M2P doesn't "see" the embeddings).

---

## Implications for Future Research

### Theorem 3 is not falsified; implementation is incomplete

The theorem itself is mathematically sound: **if** M2P receives and uses domain embeddings, **then** the centroid is unstable. The failure is not in the theorem but in the **implementation assumption**: that additive injection into memory tokens is a sufficient mechanism for M2P to "use" domain information.

### Needed to Test Theorem 3 Correctly

1. **Verify embedding divergence:** Check if trained e_d actually differ across domains (measure ||e_0 - e_d|| for d ≠ 0).
2. **Verify attention to memory:** Measure how much attention weight M2P assigns to memory tokens vs. hidden state tokens (from frozen base).
3. **Increase embedding signal strength:** Instead of additive injection, try:
   - Concatenating e_d to memory tokens (increases dimensionality but ensures full propagation).
   - Gating memory with e_d (mem_gated = mem_base * sigmoid(W_gate @ e_d)).
   - Explicit domain-routing head that predicts B directly from e_d before M2P (separable from centroid).

### Alternative: Domain signal is too weak for M2P's architecture

If even strong domain signals (e.g., concatenation) fail to prevent centroid collapse, the issue may be **architectural:** M2P's attention pattern over memory tokens may inherently collapse toward a single optimal memory state regardless of input variations — a form of **input-space compression** rather than domain-driven specialization.

In that case, Theorem 3 would need revision to account for M2P's capacity to compress domain information.

---

## Conclusion: Theorem 3 KILLED (Prediction Refuted)

The proof of Theorem 3 is mathematically valid, but **the experimental validation falsifies the key prediction: centroid destabilization does not occur with additive domain embedding injection.**

**Kill status:** Theorem 3's main conclusion (per-domain B-matrices via domain conditioning) is not supported by the data. The centroid remains stable, and the "repeat" domain failure mode persists.

**Finding status:** KILLED

**Recommendation:** Refactor the approach to provide stronger domain signals (e.g., concatenation or gating) or redesign M2P's architecture to be more sensitive to domain variations (e.g., explicit domain routing before B generation).
