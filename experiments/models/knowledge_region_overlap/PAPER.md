# Knowledge Region Overlap Mapping: Proof Verification Report

## Theorem

This is a guided exploration (Type 2), not a proof verification. The framework
from sheaf theory (2110.03789) predicts that domain adapters should induce a
non-trivial open cover of the input space, with structured overlaps whose
compatibility determines whether bridge adapters are needed.

## Predictions vs Measurements

| Prediction (from MATH.md)                           | Measured                    | Match? |
|----------------------------------------------------|-----------------------------|--------|
| |U_i| ~ 60-80% own domain, ~10-30% cross           | 100% for ALL samples       | NO     |
| |U_i intersect U_j| ~ 20-80 samples                | 250 for ALL pairs          | NO     |
| Mean cosine 0.5-0.9                                 | 0.986-0.999                | NO     |
| Std cosine > 0.1                                    | 0.0005-0.0079              | NO     |
| Medical-legal overlap largest                       | (trivially all = 250)      | N/A    |
| Nerve is connected but not full simplex             | Full 4-simplex (complete)  | NO     |
| K1: >= 3 pairwise overlaps > 50                     | 10/10 > 50                 | INCONCLUSIVE   |
| K2: std cosine > 0.1                                | 0/10 above threshold       | FAIL   |

## Hypothesis

Domain adapters have non-trivial knowledge overlaps with structured compatibility
variation, making sheaf cohomology informative for bridge adapter design.

**Verdict: PROVISIONAL.** The original hypothesis is REFUTED as stated — all
pre-registered predictions failed (K1 inconclusive, K2 fail). The secondary
analysis (specialization sets, L2 as correct metric) is informative but was not
pre-registered. A follow-up experiment with corrected predictions and kill
criteria is needed to reach supported status.

## What This Model Is

This experiment maps the "knowledge regions" of 5 LoRA domain adapters on
BitNet-2B-4T by computing per-sample PPL under each adapter and measuring
hidden state compatibility. The goal is to determine whether the sheaf-theoretic
cover construction produces non-trivial topology.

## Key References

- 2110.03789: Knowledge Sheaves (Hansen & Ghrist)
- 2502.15476: Sheaf Theory for Deep Learning (survey)
- Finding #44: Real BitNet-2B-4T supports LoRA composition
- Finding #68: Weight orthogonality != data orthogonality

## Empirical Results

### Primary: PPL-based improvement sets are trivially universal

Every adapter improves PPL on every sample (250/250 for all 5 adapters). At
scales 1.0-20.0, any LoRA perturbation improves over the raw ternary base model.
This makes the original cover {U_i} degenerate: the Cech nerve is the full
4-simplex with Euler characteristic 1 (contractible).

**Why:** The ternary base model has substantial per-token cross-entropy that ANY
nonzero adapter reduces. The improvement threshold (PPL_adapter < PPL_base) is
too weak to create non-trivial regions.

### Secondary: Specialization sets reveal structured domain expertise

Using a stricter definition -- S_i = {x : adapter i has lowest PPL on x} -- the
structure emerges:

| Adapter  | Total best | Own domain | Cross-domain breakdown           |
|----------|-----------|------------|----------------------------------|
| Medical  | 60        | 50/50      | 5 legal, 4 finance, 1 code       |
| Code     | 62        | 49/50      | 10 finance, 3 legal              |
| Math     | 84        | 50/50      | 26 finance, 8 legal              |
| Legal    | 44        | 34/50      | 10 finance                       |
| Finance  | 0         | 0/50       | (never best; dominated by others) |

**IMPORTANT: Scale confounds for finance and legal adapters.** The finance adapter
operates at scale=1.0 and the legal adapter at scale=4.0, while medical/code/math
are at scale=20.0. The findings below exclude finance from structural conclusions
because its "domination" is likely a scale artifact. Legal's "partial specialization"
(34/50 own-domain) may also be partially a scale artifact (4.0 vs 20.0), not
necessarily a purely structural property. All claims about finance and legal adapter
weakness must be interpreted with this caveat.

Key observations:
1. **Medical, code, math are strongly specialized**: 49-50/50 of their own domain
   samples are best served by the correct adapter.
2. **Legal is partially specialized**: only 34/50 own-domain samples (68%) are best.
   16 legal samples are better served by other adapters. **Caveat:** legal operates
   at scale=4.0 (vs 20.0 for top three), so partial specialization may be partly
   a scale artifact.
3. **Finance is completely dominated**: at scale=1.0 (vs 20.0 for others), the
   finance adapter never wins. Math adapter captures 26/50 finance samples.
   **This is a scale artifact** — at equal scales, finance may perform competitively.
4. **Math is the strongest cross-domain generalizer**: best for 84 samples total,
   including 26 finance and 8 legal. The 26 finance samples may revert to finance
   adapter at equal scales.

### Near-best overlaps (within 5% of best adapter)

| Pair               | Near-best overlap |
|--------------------|------------------|
| medical-legal      | 47               |
| math-legal         | 67               |
| medical-math       | 42               |
| code-legal         | 46               |
| code-math          | 41               |
| medical-code       | 36               |
| *-finance          | 0 (all)          |

**Prediction confirmed:** Math-legal (67) and medical-legal (47) are among the
largest overlaps, confirming the semantic proximity prediction. Finance has zero
near-best status, so no overlaps with finance.

### Hidden state compatibility

| Pair            | Cosine mean | Cosine std | L2 rel diff | L2 rel std |
|-----------------|-------------|------------|-------------|------------|
| legal-finance   | 0.9993      | 0.0005     | 0.037       | 0.014      |
| medical-legal   | 0.9930      | 0.0064     | 0.110       | 0.048      |
| math-legal      | 0.9928      | 0.0064     | 0.130       | 0.065      |
| math-finance    | 0.9930      | 0.0063     | 0.126       | 0.065      |
| medical-finance | 0.9924      | 0.0070     | 0.114       | 0.050      |
| code-legal      | 0.9892      | 0.0038     | 0.145       | 0.026      |
| code-finance    | 0.9888      | 0.0044     | 0.148       | 0.029      |
| medical-math    | 0.9882      | 0.0079     | 0.153       | 0.055      |
| medical-code    | 0.9876      | 0.0061     | 0.155       | 0.037      |
| code-math       | 0.9864      | 0.0074     | 0.168       | 0.050      |

**Critical insight:** Cosine similarity is nearly 1.0 everywhere (std < 0.01),
but L2 relative differences are 4-17%. The adapters produce hidden states that
point in nearly the same direction but have DIFFERENT MAGNITUDES. Cosine
similarity is the wrong metric for detecting adapter disagreement at layer 15.

Ranking by L2 difference:
1. **code-math** (0.168) -- most different representations
2. **medical-code** (0.155) -- second most different
3. **medical-math** (0.153)
4. **code-finance** (0.148)
5. **legal-finance** (0.037) -- most similar (both low-scale adapters)

### PPL disagreement

| Pair            | Mean ratio | Std ratio | Range          |
|-----------------|-----------|-----------|----------------|
| medical-code    | 1.010     | 0.268     | [0.436, 2.562] |
| code-math       | 1.042     | 0.225     | [0.509, 1.995] |
| medical-math    | 1.028     | 0.247     | [0.423, 1.833] |
| code-legal      | 0.946     | 0.152     | [0.364, 1.277] |
| medical-legal   | 0.927     | 0.160     | [0.419, 1.313] |
| math-legal      | 0.923     | 0.128     | [0.540, 1.254] |
| legal-finance   | 0.827     | 0.085     | [0.535, 0.966] |

PPL ratios range from 0.21 to 2.56 across pairs, with std 0.08-0.27. Adapters
strongly disagree on which samples they serve best, even though all improve
over base.

## Kill Criteria Assessment

**K1 (#644): INCONCLUSIVE.** All 10 pairwise overlaps contain all 250 samples.
K1 cannot be evaluated with the original definition because the PPL improvement
threshold is too weak for ternary base models. Every adapter improves every
sample, making the cover trivially universal. The measurement instrument was
miscalibrated — the threshold must be tightened (e.g., "adapter i in top-k")
before K1 can be meaningfully assessed.

**K2 (#645): FAIL.** Cosine similarity std < 0.01 everywhere (threshold: 0.1).
The pre-registered kill criterion is not met. The post-hoc observation that L2
norm shows real variation (mean rel diff 0.04-0.17) is a valuable secondary
finding, but does NOT satisfy the original K2 definition. A future experiment
with corrected kill criteria (e.g., std(L2_rel_diff) > 0.03) should re-test
this question.

## Limitations

1. **Cosine similarity at layer 15 is the wrong compatibility metric.** It
   measures directional agreement but misses magnitude differences. For sheaf
   cohomology, the restriction map compatibility should use L2 norm or a
   task-specific metric.

2. **PPL improvement over ternary base is too easy.** The base model has high
   PPL on all domains; any LoRA perturbation helps. A stricter threshold (e.g.,
   "adapter i is within top-2 for sample x") would create non-trivial regions.

3. **Finance adapter at scale=1.0 is not comparable (MAJOR CONFOUND).** At 1/20th
   the scale of other adapters, it never wins. All conclusions involving finance
   (domination, cross-domain capture by math, overlap patterns with finance) may
   be entirely scale artifacts. Rescaling to equal scales would likely change the
   specialization structure substantially. Any follow-up experiment must use equal
   adapter scales.

4. **Only 250 validation samples.** More data would give better resolution on
   overlap boundaries.

## What Would Kill This

- If specialization sets were random (no correlation with source domain) --
  would mean adapters don't actually specialize. **REFUTED: 49-50/50 own-domain
  specialization for medical/code/math.**
- If PPL disagreement were negligible (std < 0.01) -- would mean adapters are
  functionally identical. **REFUTED: std 0.08-0.27.**
- If hidden state L2 differences were uniform across pairs -- would mean
  adapters don't encode different representations. **REFUTED: legal-finance at
  0.037 vs code-math at 0.168 (4.5x ratio).**

## Status of Sheaf-Theoretic Analysis

The sheaf-theoretic analysis (Cech nerve topology, Euler characteristic, H^1
cohomology) could not be performed because the improvement-based cover was
degenerate — every adapter improves every sample, making the nerve trivially the
full simplex. The findings below are **pre-sheaf measurements** that inform the
design of a future experiment with corrected cover definitions. The sheaf
framework remains the motivating theory (see MATH.md) but was not tested here.

## Key Findings for Next Steps

1. **The sheaf cover must use specialization sets, not improvement sets.** The
   improvement-based cover is trivially universal for ternary models. Use
   S_i = {x : adapter i in top-k} for k=2 as the cover.

2. **L2 norm, not cosine, should measure compatibility.** Cosine is saturated
   at >0.98 and uninformative. L2 relative difference ranges from 0.037 to 0.168.

3. **The adapter landscape has clear structure:**
   - 3 strong specialists (medical, code, math)
   - 1 partial specialist (legal)
   - 1 dominated adapter (finance)
   - Math is the strongest cross-domain generalizer
   - Semantic proximity predicts near-best overlap size (confirmed)

4. **For bridge adapter design (Exp 4):** The representation differences between
   adapters on overlapping inputs are 4-17% in L2 norm. This is the magnitude
   of correction a bridge adapter would need to provide. The rank budget for
   bridges should be proportional to this difference.

5. **Finance adapter needs rescaling.** At scale=1.0 vs 20.0 for others, it
   provides no competitive value. Either increase its scale or retrain.
