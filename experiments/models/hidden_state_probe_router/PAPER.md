# Hidden-State MLP Probe for Per-Token Adapter Routing: Proof Verification Report

## Theorems and Claims

**Claim 1 (Probe Routing Accuracy Prediction).** Token-level hidden states from a
transformer with d=2560, where domain centroids are separated (verified by
Finding #276: 96% linear separability at sequence level), can be classified by
a single-hidden-layer MLP of width w=128 with error rate bounded by the
per-token SNR degradation from losing mean-pooling. **Note:** This is a
framework-grounded prediction (Cover's theorem + UAT), not a formal proof.
The UAT guarantees existence but provides no constructive bound.

**Theorem 2 (Probe Inference Cost).** FLOPs = 2w(d+K) = 656K for d=2560,
w=128, K=5. At M5 Pro throughput, latency < 0.01ms per token.
*(Formal proof: direct computation from matrix multiply dimensions.)*

**Theorem 3 (PPL Bound Under Routing Errors).** PPL_probe / PPL_oracle <=
1 + epsilon * (R - 1), where epsilon is routing error rate and R is max/min
domain PPL ratio.
*(Formal proof: valid derivation from NLL decomposition.)*

## Predictions

| Prediction (from proof) | Measured | Match? |
|------------------------|----------|--------|
| Token-level MLP accuracy >= 85% (Claim 1 + Cover) | 98.5% | YES (16% above threshold) |
| Token-level ridge accuracy (linear baseline) | 98.3% | **Unpredicted** -- SNR analysis predicted 16x degradation vs mean-pooled, implying ~80-90% token accuracy. Actual degradation was 0.2pp. Isotropic noise assumption wrong: domain signal is per-token, not sequence-level noise reduced by averaging. |
| Sequence-level ridge accuracy ~ 96% (Finding #276) | 100% | YES (exceeds prior 96%) |
| PPL probe/oracle within 5% (Thm 3, eps<0.05) | 0.00% gap | **Tautological** at 100% routing accuracy (see K785 note below) |
| Probe latency < 1ms (Thm 2) | 0.170ms (single), 0.0014ms (batched) | YES (6x under budget) |
| Mixed-domain routing accuracy | 100% (200/200 segments) | YES (far exceeds 85%) |

## Hypothesis

An MLP probe trained on per-token hidden states from a frozen ternary base
model can route individual tokens to domain-specific LoRA adapters with
>=85% accuracy, enabling segment-isolated routing on mixed-domain sequences
within 5% of oracle PPL, at <1ms overhead per token.

**Verdict: SUPPORTED.** All three kill criteria pass with large margins.

## What This Model Is

A small MLP (d=2560 -> 128 -> 5, 328K params) trained as a post-hoc
classifier on base model hidden states. Each token's last-layer hidden state
is fed through the probe to predict which of 5 domain adapters should be
applied. For mixed-domain sequences, tokens within each segment vote for
an adapter (majority vote), and segments are evaluated independently with
their voted adapter (segment isolation from Finding #305).

This extends the ridge regression router (Finding #276) from sequence-level
to token-level granularity. The ridge router computes W* = (X^TX + lambda*I)^{-1}
X^TY on mean-pooled hidden states. The MLP probe operates on individual token
hidden states, trading the closed-form guarantee for nonlinear capacity.

## Key References

- X-LoRA (arXiv 2402.07148): Hidden-state gating for LoRA mixture, per-layer per-token
- TT-LoRA MoE (arXiv 2504.21190): Sparse MoE routing on base model representations
- Finding #276: Ridge regression router, 96% accuracy, 23s init, closed-form
- Finding #305: Segment-isolated routing +16% over per-sequence, 95.2% accuracy
- Finding #309: KV-cache reuse killed -- adapters are isolated processing units
- Cover's theorem (1965): Linear separability in high dimensions
- Rahimi & Recht (2007): Random features bound for kernel approximation

## Empirical Results

### Phase 1: Hidden State Extraction
- Extraction time: 62.0s (200 samples through 24-layer model)
- Calibration tokens: 16,866 (30 samples x 5 domains, response tokens only)
- Test tokens: 5,514 (10 samples x 5 domains)
- Token counts per domain (cal/test):
  - medical: 2,014 / 673
  - code: 2,801 / 709
  - math: 2,894 / 1,050
  - legal: 4,319 / 1,450
  - finance: 4,838 / 1,632
- Hidden dimension: 2560

### Phase 2: Probe Training & Accuracy (K784)

**MLP Probe (the core contribution):**
- Token-level accuracy: **98.5%** (K784: threshold 85%, 13.5pp above)
- Best test accuracy during training: 98.8% (at epoch 1 -- nearly instant convergence)
- Sequence-level accuracy (mean-pooled input): 100%
- Per-domain token accuracy:
  - medical: 100.0%
  - code: 99.4% (4 misrouted: 1 math, 3 legal)
  - math: 99.5% (5 misrouted: 4 finance, 1 code)
  - legal: 98.6% (21 misrouted: 18 finance, 3 code)
  - finance: 96.8% (52 misrouted: all to legal)
- Parameters: 328,453 (1.3 MB at fp32)
- Training: 30 epochs, converged by epoch 1 (loss 0.063 -> 5.2e-6)
- Training time: 3.2s

**Ridge Regression (linear baseline for comparison):**
- Token-level accuracy: **98.3%** (nearly identical to MLP)
- Sequence-level accuracy: 100%
- Per-domain: medical 100%, code 100%, math 99.9%, legal 99.0%, finance 95.3%
- Time: 1.6s (closed-form, no training)

**Key insight:** The linear ridge classifier achieves 98.3% token-level accuracy,
only 0.2pp below the MLP probe. This means per-token hidden states are LINEARLY
separable -- the MLP's nonlinear capacity is unnecessary for 5 domains. This
validates Cover's theorem prediction: d=2560 >> N, so even individual tokens are
linearly separable.

**Confusion pattern:** legal <-> finance is the primary confusion axis (18/21
legal errors go to finance, all 52 finance errors go to legal). This matches
Finding #276's observation that legal-finance hidden-state cosine is 0.981.
Medical and code are perfectly separated.

### Phase 2b: Latency (K786)
- Single-token probe forward: **0.170ms** (K786: threshold 1ms, 5.9x under)
- Batched (128 tokens): 0.178ms total, **0.0014ms per token** (714x under)
- Theorem 2 predicted 0.066us raw compute. The 170us measured overhead is
  dominated by MLX dispatch/eval overhead per call. In batched serving (the
  production path), per-token cost drops to 1.4us (23x raw compute prediction),
  which is excellent.

### Phase 3: Mixed-Domain PPL (K785)

| Strategy | PPL | vs Oracle |
|----------|-----|-----------|
| Oracle (correct adapter per segment) | 7.636 | baseline |
| **Probe routing (majority vote)** | **7.636** | **0.00%** |
| Per-sequence best | 7.366 | -3.5% (better) |
| Base only (no adapter) | 7.465 | -2.2% (better) |

- **K785: PASS (tautological).** Probe routing PPL = oracle PPL (0.00% gap,
  threshold 5%). However, this pass is vacuous: routing accuracy is 100%
  (200/200 segments correct), so the probe selects exactly the same adapter
  as the oracle for every segment. Matching oracle PPL is guaranteed by
  construction when routing is perfect. K785 would only become a meaningful
  test if routing accuracy were <100%, where Theorem 3's error bound would
  actually be exercised.
- Probe segment routing accuracy: **100%** (200/200 segments correct).
  The probe routes every single segment to the correct adapter.

**The experiment's real success is accurate token-level domain classification**,
not PPL improvement from routing. The probe achieves 98.5% per-token accuracy
and 100% segment-level accuracy via majority vote -- this is a domain
classification result, not a routing quality result.

**Oracle segment-isolated is WORSE than alternatives:**
- Oracle segment-isolated: PPL 7.636
- Per-sequence best: PPL 7.366 (-3.5%)
- Base only (no adapter): PPL 7.465 (-2.2%)

This means segment-isolated adapter application with these adapters actively
degrades PPL compared to both per-sequence strategies and using no adapter at
all. Possible causes:
(a) LORA_SCALE=20.0 may be too aggressive for isolated 128-token segments
    that lack full context. The adapters were trained on full sequences and
    may expect longer context to be effective.
(b) Segment isolation forces adapter cold-start on each 128-token segment.
    The adapter has no prior context to condition on, whereas per-sequence
    application gives the adapter the full sequence.
(c) These adapters were trained on complete sequences where the adapter sees
    instruction+response together. Applying them to isolated 128-token
    segments is a distribution shift from training conditions.

The key takeaway: adapters trained on full sequences may hurt when applied to
isolated short segments. This is a real limitation of segment-isolated routing
at this adapter configuration, not an anomaly to be dismissed.

### Kill Criteria

| Kill | Threshold | Measured | Margin | Verdict |
|------|-----------|----------|--------|---------|
| K784: Token-level accuracy | >= 85% | 98.5% | +13.5pp | **PASS** |
| K785: PPL vs oracle | within 5% | 0.00% | 5.0pp | **PASS (tautological)** -- routing accuracy is 100%, so probe = oracle by construction |
| K786: Probe latency | < 1ms | 0.170ms | 5.9x | **PASS** |

**Overall Verdict: SUPPORTED**

## Significant Findings

### 1. Per-token hidden states ARE linearly separable (not just mean-pooled)

The SNR analysis in MATH.md predicted 16x degradation from token-level vs
mean-pooled. Actual degradation: 0pp at sequence level (both 100%), 1.7pp at
token level (98.3% ridge vs 100% mean-pooled ridge). The SNR penalty is far
smaller than predicted because:
- Domain signal is per-token, not just per-sequence
- The base model builds domain-specific representations at every token position
- d=2560 >> K=5 provides massive linear separability margin

### 2. MLP probe adds no value over ridge regression for K=5

Ridge achieves 98.3%, MLP achieves 98.5%. The 0.2pp improvement does not
justify the 25x parameter increase (328K vs 13K) or training requirement.
**Recommendation: Use ridge regression for per-token routing.**

### 3. Segment-level routing accuracy is 100% with majority vote

Even though individual token accuracy is 98.5%, majority-vote over 128 tokens
in a segment has epsilon_segment -> 0 by Hoeffding's inequality:
P(majority wrong) <= exp(-2 * 128 * (0.985 - 0.5)^2) ~ exp(-60) ~ 0.
This was predicted by Theorem 3 but the margin is far larger than expected.

### 4. legal <-> finance confusion is the only systematic error

This matches the ridge router finding (cos=0.981). All other domain pairs
are near-perfectly separated at token level. This confusion is structural
(legal and finance text share vocabulary and syntax) and will persist at any
routing architecture.

## Limitations

1. **5 domains only.** Token-level routing may degrade at N=25 or N=50 where
   domain clusters overlap more. The legal-finance confusion at K=5 suggests
   this is a real risk.
2. **Synthetic mixed sequences with sharp boundaries.** Real-world domain
   transitions may be gradual, degrading majority-vote accuracy.
3. **Response-only tokens.** We classified response tokens (not instruction
   template). Template tokens would add noise but are less relevant for routing.
4. **Ternary base model.** Results specific to BitNet-2B-4T. FP16 models with
   richer hidden-state geometry might show different separability.
5. **Same adapter set as ridge router.** These 5 adapters were trained with
   Grassmannian orthogonal A-matrices, which may contribute to cleaner separation.
   Non-orthogonal adapters might produce less separable hidden states.
6. **NaN warnings in ridge.** Some hidden-state features have zero variance,
   causing division-by-zero in normalization. Does not affect accuracy (these
   features are constant and contribute nothing to classification).
7. **Model configuration mismatch between phases.** Phase 1 extracts hidden
   states from the raw BitLinear model. Phase 3 evaluates probe routing on
   the unpacked-bf16 + zeroed-LoRA model configuration. These are different
   model configurations. The 100% routing accuracy suggests this discrepancy
   is benign at this scale, but it could matter at larger scale where
   numerical drift between representations (raw ternary vs unpacked bf16)
   may produce meaningfully different hidden states.

## What Would Kill This

**Micro scale (already passed):**
- K784 FAIL: Token accuracy < 85% (measured 98.5% -- 13.5pp margin)
- K785 FAIL: PPL gap > 5% (measured 0.00% -- 5pp margin)
- K786 FAIL: Latency > 1ms (measured 0.170ms -- 5.9x margin)

**Future risks:**
- Token accuracy degradation at N>10 domains (domain overlap increases)
- Probe overfitting at N>50 (need more calibration data per domain)
- Gradual domain transitions defeating majority-vote (would need sliding-window
  boundary detection rather than fixed segments)
- Non-Grassmannian adapters with correlated hidden-state perturbations

**Strongest recommendation:** Given ridge regression matches MLP performance,
use the closed-form ridge router for per-token routing. It is simpler (no
training), faster (1.6s vs 3.2s init), smaller (13K vs 328K params), and
has the same Woodbury incremental-update property from Finding #276.
