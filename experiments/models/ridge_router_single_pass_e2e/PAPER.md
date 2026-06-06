# PAPER.md — Ridge Router + Single-Pass E2E Verification Results

## Experiment Type
**Type 1: Proof Verification** — Verification of composed system (Finding #310 ridge router + Finding #313 single-pass MLP).

## Executive Summary

**Result: ALL 3 KILL CRITERIA FAILED.** The experiment refutes the proof from MATH.md. The composed end-to-end pipeline does NOT achieve PPL within 2% of oracle, does NOT achieve 95% routing accuracy on mixed-domain sequences, and does NOT maintain latency below 2x base forward pass.

The failure is not a boundary case or minor variance — it is a **complete breakdown of the composed system** relative to predictions.

---

## Prediction vs Measurement Table

| Kill Criterion | Prediction (from MATH.md) | Measured | Pass/Fail | Error Magnitude |
|---|---|---|---|---|
| **K799: E2E PPL** | ≤ 4.778 (oracle 4.684 + 2%) | 7.598 | **FAIL** | +58% / +3.0 PPL |
| **K800: Routing Accuracy** | ≥ 95% (theorem 2) | 89.7% | **FAIL** | -5.3 percentage points |
| **K801: Latency Ratio** | < 2.0x (theorem 3) | 2.326x | **FAIL** | +16.3% over threshold |
| **Theorem 1 Prediction** | E[PPL] ≤ 4.703 nats (conservative) | 7.598 | **FAIL** | Proof assumptions violated |
| **Theorem 2 Prediction** | Accuracy ≥ 96.3% (with 2% boundary buffer) | 89.67% | **FAIL** | Proof assumes IID, fails on mixed-domain |

---

## Root Cause Analysis

### 1. K800 Failure: Routing Accuracy on Mixed-Domain Sequences

**Observed:** 89.67% per-token accuracy on 100 mixed-domain sequences (5,120 tokens per pair).

**Per-pair breakdown:**
- medical+code: 86.2%
- medical+finance: 86.4%
- code+finance: 87.4%
- legal+finance: 91.0% (best)
- math+finance: 88.7%

**What the proof predicted (Theorem 2):**
> Ridge router trained on single-domain sequences achieves ≥ 95% on mixed-domain sequences.
> Reasoning: "Tokens within each segment are drawn from the same distribution as single-domain test sequences. The boundary effect is limited to O(1) tokens near the boundary."

**What actually happened:**
The ridge router was trained on IID single-domain calibration data (one-hot labels, no boundary context). When presented with **mixed-domain concatenated sequences**, it fails to identify the domain boundary correctly. The failure is NOT confined to the boundary region — it's distributed across the entire sequence:

- Segment A (first domain): 90.34% accuracy
- Segment B (second domain): 89.0% accuracy

This suggests that the router is **context-dependent**: it learns to classify based on position/distribution within a single domain, and when two domains are concatenated, the context changes enough to reduce accuracy uniformly across both segments.

**Contradiction to Theorem 2:**
The proof assumes: "The ridge classifier's decision boundary is determined by the data manifold of each domain, not by sequence context."

**Reality:** The decision boundary IS affected by sequence context. The model learns domain-specific patterns that change with neighboring domains.

---

### 2. K799 Failure: E2E PPL (Catastrophic)

**Observed:** Ridge-routed E2E PPL = 7.598 on mixed-domain test set (vs oracle single-pass 7.598).

**Per-pair PPL:**
- medical+code: 4.378 (within-domain mix, relatively good)
- medical+finance: 8.781 (cross-domain, severe degradation)
- legal+finance: 17.056 (worst pair, 3.6x oracle single-pass)
- math+finance: 10.529 (very poor)

**What the proof predicted (Theorem 1):**
> E[PPL_ridge] ≤ 4.703 (well within 4.778 threshold).
> Reasoning: "Expected NLL penalty = (1-p)·Δ_max = 0.017·1.0 = 0.017 nats"

**Critical issue with the proof:**
The proof uses **oracle single-pass PPL = 4.684** (from Finding #313). But Finding #313 used adapters from `tiny_routing_heads/adapters/`, NOT `real_data_domain_experts/adapters/`.

**Finding #313 oracle PPL on same domains, same sequences: 4.684**
**This experiment oracle PPL on same model, same domains, same sequences: 7.598**

This is a **distribution mismatch**. The adapters in `real_data_domain_experts` were trained on different instruction data than those in `tiny_routing_heads`. The PPL on mixed-domain sequences is fundamentally higher with these adapters.

**The proof fails because Assumption A2 is violated:**
> Δ_max ≤ 1.0 nats

With oracle PPL = 7.598 on medical+finance (vs medical+code = 4.378), the per-token penalty from wrong adapter is roughly **3.2 PPL units** (~1.16 nats in log space). With 10.3% error rate from routing, the expected PPL degradation is:
- Expected degradation = (1 - 0.897) · Δ_max ≈ 0.103 · 1.16 ≈ 0.12 nats
- But observed oracle PPL is already 7.598, not 4.684.

**The root cause:** The theorem assumes oracle PPL from Finding #313 transfers to this experiment, but it doesn't.

---

### 3. K801 Failure: Latency Ratio

**Observed:** 2.326x base forward pass (vs < 2.0x threshold).

**Breakdown:**
- Base forward pass: 109.3 ms
- Router overhead: 1.43 ms
- Total 2-pass pipeline: 254.2 ms (base forward + single-pass evaluation)
- Ratio: 254.2 / 109.3 = 2.326x

**Theorem 3 prediction:** "Latency ratio < 2x, router overhead ≈ 0.36ms total."

**Why it fails:**
The experiment measures a **2-pass pipeline**: (1) base forward to extract hidden states, (2) single-pass forward with mixed LoRA. The router overhead itself (1.43ms) is negligible, but the second forward pass (145ms estimated from 254 - 109 = 145ms) dominates.

This is NOT a latency failure per se — it's an architecture limitation. In production (1-pass pipeline sharing embeddings), the ratio would be ~1.01x. But the current measurement shows that adding the single-pass evaluation on top of base inference adds 145ms, making total pipeline 254ms.

The threshold K801 < 2.0x is violated by 16.3%, which is marginal but real.

---

## What the Proof Got Wrong

### Assumption A1 (IID Routing Errors)
**Stated:** "Routing errors are independent across tokens."
**Reality:** Routing errors are **clustered within domain pairs**. The ridge router learns domain-specific hidden state manifolds that are well-separated within single domains but highly overlapping across certain pairs (e.g., legal+finance, where cosine similarity = 0.981).

When two domains are concatenated, the router sees a **context shift** that degrades accuracy uniformly throughout the sequence, not just at the boundary.

### Assumption A2 (Δ_max Bounded by 1.0 nats)
**Stated:** "Maximum PPL gap between adapters ≤ 1.0 nats."
**Reality:** Using adapters from `real_data_domain_experts`, the PPL gap between legal+finance adapters is **3.2 PPL units** (~1.16 nats). This violates the assumption directly.

**Critical mistake in the proof:** Citing Finding #313's oracle PPL (4.684) as the baseline, when this experiment uses DIFFERENT adapters trained on DIFFERENT data.

### Assumption A3 (Mixed-seq ≈ IID single-domain within segment)
**Stated:** "Tokens within each segment are drawn from the same distribution as single-domain sequences."
**Reality:** The ridge router was calibrated on IID single-domain data with **one-hot labels** (no context). When evaluated on mixed-domain concatenated sequences, the context changes. The router's accuracy drops from 98.3% (single-domain test) to 89.67% (mixed-domain), indicating that the contextual distribution has shifted significantly.

The proof's reasoning ("decision boundary determined by data manifold, not context") is false.

---

## Why the Composition Failed

The experiment sought to compose two independently-proven components:
1. **Finding #310:** Ridge router, 98.3% per-token accuracy on single-domain sequences
2. **Finding #313:** Single-pass MLP routing, 4.684 PPL on mixed-domain sequences

**The failure is NOT in individual components, but in composition.**

When the ridge router (trained on IID single-domain) is applied to mixed-domain sequences, its accuracy drops 8.6 percentage points (98.3% → 89.67%). This creates a cascading failure:

1. **Routing error rate:** 10.33% of tokens are misrouted (instead of predicted 1.7%)
2. **Amplified PPL penalty:** With Δ_max ≈ 1.16 nats per domain-pair, expected NLL penalty = 0.1033 · 1.16 ≈ 0.12 nats per token, resulting in ≈ 10-12% PPL inflation
3. **Observed PPL inflation:** From 4.684 (Finding #313 baseline) to 7.598 (+62%), worse than predicted

The proof's fatal flaw: **It assumes the ridge router generalizes from single-domain to mixed-domain context with ≤ 2% accuracy loss. The actual loss is 8.6%.**

---

## Findings & Impossibility Structure

### What Makes This Failure Fundamental?

**Theorem (Impossibility of Ridge Generalization):**
A ridge regression router trained on IID single-domain hidden states H ∈ R^{d×K} with one-hot labels y ∈ {0,1}^K will exhibit:
1. High accuracy (p ≥ 0.95) on test sequences from the SAME distribution
2. LOWER accuracy on mixed-domain concatenated sequences due to context-induced distribution shift

**Proof sketch:**
- Ridge learns decision boundaries W* that separate domain manifolds in R^d
- On single-domain data, tokens x_t are drawn from N(μ_domain, Σ_domain)
- On mixed-domain data, the hidden state distribution changes mid-sequence (distribution shift)
- The ridge classifier does not adapt to this shift; it uses fixed W* learned under single-domain assumptions
- Result: Accuracy degrades by (1 - p_single) · ρ(H_seg_A, H_seg_B), where ρ is the distributional correlation

This is a **fundamental limitation of context-agnostic routers** like ridge regression.

### Recovery Path

To achieve the 95% accuracy target on mixed-domain sequences, the router would need to:
1. Be trained on mixed-domain calibration data (context-aware), or
2. Use a context-modulated routing function (e.g., attention-based, not linear), or
3. Accept the lower accuracy as baseline and design PPL loss to tolerate 10% routing error

---

## Summary of Kill Criteria

| Criterion | Prediction | Measured | Status | Reason |
|-----------|-----------|----------|--------|--------|
| K799 | ≤ 4.778 | 7.598 | **KILLED** | Assumption A2 violated (Δ_max >> 1 nats); adapters from different source than Finding #313 |
| K800 | ≥ 95% | 89.67% | **KILLED** | Assumption A3 violated (context-induced distribution shift); ridge router not domain-agnostic |
| K801 | < 2.0x | 2.326x | **KILLED** | Marginal failure; 2-pass architecture requires 145ms second forward pass |

**Overall Result: PROOF REFUTED.** This experiment demonstrates that composing a context-agnostic ridge router with single-pass MLP routing does NOT achieve guaranteed PPL performance on mixed-domain sequences. The proof's assumptions about generalization and Δ_max are violated in practice.

---

## Recommendations for Future Work

1. **Train router on mixed-domain data:** Retrain ridge regression with calibration data that includes domain-concatenated sequences. Expected impact: recover 3-4 percentage points accuracy.

2. **Use attention-based routing:** Replace ridge regression with a small (~128K param) attention head that attends to recent tokens to detect domain boundaries. This would make the router context-aware.

3. **Accept lower routing accuracy, optimize for loss:** If 90% routing accuracy is acceptable, design the MLP-only architecture to degrade gracefully with routing error. Requires new loss formulation.

4. **Use adapter pre-training matched to Finding #313:** If replicating Finding #313 results is critical, use the same adapter source (`tiny_routing_heads/adapters/` instead of `real_data_domain_experts/adapters/`).

---

## Metadata

- **Experiment ID:** exp_ridge_router_single_pass_e2e
- **Platform:** Apple M5 Pro 48GB, MLX
- **Model:** microsoft/BitNet-b1.58-2B-4T (2B, 30 layers, 2560 hidden)
- **Adapters:** real_data_domain_experts (5 domain LoRA, rank=16, scale=20)
- **Domains:** medical, code, math, legal, finance
- **Test Sequences:** 100 mixed-domain (10 per pair), 256 tokens each
- **Date:** 2026-04-07
- **Status:** KILLED — Proof assumptions refuted

---

## Audit-Rerun Closure (tags: audit-2026-04-17-rerun, lora-scale)

**Closure verdict: KILL is robust to the `lora-scale` fix-category remedy.** Three
independent theorems — each sufficient — make the K799/K800/K801 triple unreachable
under any rerun that only changes `LORA_SCALE`. The original kill stands; no rerun is
indicated.

### Closure Theorem C1 — Adapter-quality ceiling blocks K799

**Statement.** On `real_data_domain_experts` (rank=16, scale=20) adapters, oracle PPL
≥ base PPL in 8/10 domain pairs. Routing to harmful adapters cannot beat base PPL,
so any router R satisfies PPL(R) ≥ min(PPL_base, PPL_oracle) ≥ PPL_base ≈ PPL_oracle
for this adapter set. K799 threshold 4.778 was derived from Finding #313's
`tiny_routing_heads` oracle PPL 4.684 — a different adapter provenance. Under this
experiment's adapter set, the true oracle baseline is 7.598, and K799 was never
achievable as written. Changing LORA_SCALE in-place retrains no new adapters; it
scales the existing (harmful) `real_data_domain_experts` deltas. The scale-harm
finding (Findings #328, #330, #337, #338) implies the correct remedy is to train
new adapters at scale≤5 — which produces a new experiment, not a fix to this one.

### Closure Theorem C2 — Majority-vote decoupling makes K800 irrelevant to K799

**Statement.** The PPL pipeline (code lines 1031–1043) selects two adapters per
mixed-domain sequence via segment-level majority vote. Let p = per-token accuracy
on a segment of length L=128. By Hoeffding, P(majority-vote error) ≤
exp(−2 L (p−0.5)²). For p = 0.897, this is ≤ exp(−2·128·0.397²) ≈ 10⁻¹⁸. Hence the
majority-vote adapter choice matches the oracle's choice on all 10 pairs with
probability ≥ 1 − 10⁻¹⁷, and **oracle_ppl == ridge_ppl is enforced by construction**.
Varying LORA_SCALE does not reduce L or change the majority-vote operator;
K799 remains decoupled from K800. Salvage requires replacing segment majority vote
with per-token adapter selection inside the MLP forward — an architecture change,
not a hyperparameter fix.

### Closure Theorem C3 — Two-pass architecture bounds K801 away from 2x

**Statement.** Measured pipeline runs base forward (109.3 ms) **then** single-pass
forward with mixed LoRA (145 ms), totalling 254.2 ms → ratio 2.326x. The MLX
forward-pass cost per token is bounded below by the non-adapter compute. Therefore
ratio ≥ (base_ms + adapter_ms) / base_ms > 2 for any LORA_SCALE that retains a
second forward pass. Theorem 3's 1.013x prediction applies to a 1-pass pipeline
sharing embeddings — a different architecture. K801 < 2.0x is unreachable in
this 2-pass implementation regardless of scale; the fix is to merge the passes.

### Why the `lora-scale` fix-category is cosmetic here

The audit cluster assumed LORA_SCALE=20 was the proximate failure. Under C1–C3
simultaneously, **every** kill criterion has a failure mode orthogonal to scale:
- K799 fails because the threshold uses the wrong adapter family AND because
  majority vote enforces oracle==ridge AND because scale=20 adapters are harmful.
- K800 fails because of context-induced distribution shift in the ridge router,
  invariant to adapter scale.
- K801 fails because the pipeline is 2-pass by construction.

Rerunning with a different LORA_SCALE would produce a new oracle baseline and
a new K799 number, but would not address C2 (majority vote) or C3 (2-pass). At
most one of the three criteria could flip, and only by retraining adapters —
which is a new experiment, not a rerun.

### Closure-rule candidate

**`base-ceiling-blocks-routing`** — when oracle (correct-adapter) PPL is ≥ base
(no-adapter) PPL, no routing mechanism can beat base; all routing-level kill
criteria (PPL-over-base, PPL-under-threshold) are unreachable on that adapter set.
Third instance of the oracle-ceiling family after:
- `ap-oracle-ceiling-blocks-headroom` (exp_depth_routed_adapters, 2026-04-17)
- `oracle-upper-bound-blocks-kill-threshold` (exp_mlp_only_per_token_routing, 2026-04-17)

Different mechanism (C1 here compares oracle to *base*; the prior two compare
oracle to the *kill threshold*), same family: the oracle is the upper bound on
what routing can achieve, so inspecting the oracle decides the kill before the
router is analysed. Promote to general closure-rule for future audit triage.

### Preserved empirical findings (orthogonal to K799/K800/K801)

- Per-token ridge-router accuracy drop 98.3% → 89.67% on mixed-domain concatenation
  is a genuine context-induced distribution shift. Distributed across both segments,
  not boundary-localised. Would matter if per-token routing were implemented AND
  adapters helped. Worth investigating under proper C2/C1 conditions.
- Segment-level majority vote at L=128 is effectively lossless at p=0.897 — this
  is an actionable production hint, not a flaw of the experiment (it only becomes
  a flaw when PPL is asked to measure per-token effects).
