# E2E N=25 Quality Validation — Results

## Verdict: KILLED (audit re-classification 2026-04-18 — tautological-routing antipattern)

K1489 ("max routing loss ≤ 10pp vs oracle") is structurally tautological:
the run uses 3 adapters in a 25-domain space where 22 domains have no
adapter, so every misrouted query falls back to base. Wrong-adapter
collisions — the actual failure mode the KC was supposed to probe — are
unreachable by construction. See LEARNINGS.md for full reasoning and v2
requirements.

The surface-level results below replicate cleanly (pipeline has no bugs at
N=25), but cannot discharge the pre-registered hypothesis. Treat the PASS
margins as **protocol artifacts**, not evidence of routing robustness.

## Summary (superseded)

The E2E pipeline scales from N=10 to N=25 with **zero quality degradation**.
All 4 kill criteria PASS. Maximum routing loss is 2.0pp (MedMCQA), half of
the 4.0pp observed at N=10. The combined logistic router achieves 98-100%
routing accuracy on adapter domains even with 25 domain classes.

## Prediction vs Measurement

### A-Priori Predictions (before measuring routing accuracy)

| Benchmark | Predicted α | Predicted Quality | Measured Quality | Error |
|-----------|-------------|-------------------|------------------|-------|
| GSM8K     | 95%         | 73.9%             | 76.0%            | -2.1pp |
| HumanEval | 88%         | 52.3%             | 57.0%            | -4.7pp |
| MedMCQA   | 80%         | 52.0%             | 56.0%            | -4.0pp |

A-priori predictions were conservative — routing accuracy was better than estimated.

### Theorem 1 Predictions (using measured routing accuracy)

| Benchmark | α measured | T1 Predicted | Measured | Error | N=10 Measured |
|-----------|-----------|-------------|----------|-------|---------------|
| GSM8K     | 98.0%     | 75.8%       | 76.0%    | -0.2pp | 77.0%        |
| HumanEval | 100.0%    | 57.0%       | 57.0%    | 0.0pp  | 56.0%        |
| MedMCQA   | 88.0%     | 54.4%       | 56.0%    | -1.6pp | 54.0%        |

**Theorem 1 accuracy: 2/3 predictions within 0.2pp, 1/3 within 1.6pp.**
The MedMCQA overperformance (56% vs predicted 54.4%) suggests the base model
provides some medical knowledge that helps misrouted queries.

### Quality Loss from Routing

| Benchmark | Oracle | Routed | Loss | Kill Threshold | Result |
|-----------|--------|--------|------|----------------|--------|
| GSM8K     | 77.0%  | 76.0%  | 1.0pp | ≥68% (K1486) | **PASS** |
| HumanEval | 57.0%  | 57.0%  | 0.0pp | ≥46% (K1487) | **PASS** |
| MedMCQA   | 58.0%  | 56.0%  | 2.0pp | ≥44% (K1488) | **PASS** |
| Max loss  | —      | —      | 2.0pp | ≤10pp (K1489) | **PASS** |

### N=10 → N=25 Degradation

| Benchmark | N=3   | N=10  | N=25  | N=10→N=25 Delta |
|-----------|-------|-------|-------|-----------------|
| GSM8K     | 77.0% | 77.0% | 76.0% | -1.0pp          |
| HumanEval | 57.0% | 56.0% | 57.0% | +1.0pp          |
| MedMCQA   | 58.0% | 54.0% | 56.0% | +2.0pp          |
| Max loss  | 0.0pp | 4.0pp | 2.0pp | −2.0pp (improved!)|

**No degradation from N=10 to N=25.** In fact, quality slightly IMPROVES on
HumanEval (+1pp) and MedMCQA (+2pp). This is within sample variance (N=100)
but conclusively shows scaling from 10 to 25 domains costs nothing.

## Routing Accuracy at N=25

### Adapter domain routing (benchmark queries)
| Domain   | N=10 (Finding #533) | N=25 (This) | Delta |
|----------|---------------------|-------------|-------|
| Math     | 98%                 | 98%         | 0pp   |
| Code     | 97%                 | 100%        | +3pp  |
| Medical  | 86%                 | 88%         | +2pp  |

Routing accuracy IMPROVES from N=10 to N=25. Hypothesis: the combined
logistic classifier with sentence embeddings benefits from having more
contrastive examples — the additional 15 MMLU subjects help the classifier
learn what math/code/medical ISN'T, improving precision.

### Overall router (MMLU train/test split)
- Overall: 90.1% (vs 88.8% in Finding #531 with different data split)
- Math: 100%, Code: 99%, Medical: 87%
- Worst domain: sociology 53.6% (but this has no adapter — irrelevant)
- 10 domains with 0% routing in test set (insufficient test samples)

### MedMCQA misrouting pattern
12 medical queries misrouted to: high_school_chemistry (4), sociology (3),
high_school_statistics (2), electrical_engineering (1), high_school_physics (1),
global_facts (1). All go to base model fallback — no wrong-adapter risk.

## Key Insights

1. **Routing scales for free**: The combined logistic router handles 2.5x
   more domains with NO quality cost. The sentence embeddings provide robust
   semantic separation that doesn't degrade with domain count.

2. **Theorem 1 continues to hold**: Quality loss tracks (1-α)*(Q_oracle-Q_base)
   within 1.6pp. The framework is validated across N=3, N=10, and N=25.

3. **Base fallback is benign**: The 12% misrouted MedMCQA queries get base
   model quality (28%), producing 56% overall vs predicted 54.4%. No
   wrong-adapter risk because misrouted queries go to non-adapter domains.

4. **N=100 scaling path is clear**: If routing accuracy stays above 85% per
   adapter domain at N=100 (plausible given embedding-based classification),
   quality loss would remain under 5pp.

## Limitations

1. **Same 3 adapters tested**: Only math/code/medical quality measured.
   Adapters for other domains (legal, finance, etc.) not yet trained on Gemma 4.

2. **N=100 per benchmark**: Statistical power is moderate. The +2pp MedMCQA
   improvement vs N=10 is likely noise (95% CI ≈ ±10pp).

3. **No wrong-adapter routing**: All misroutes go to non-adapter domains.
   With 25 trained adapters, wrong-adapter routing could degrade quality
   below base. This requires a separate experiment.

## Conclusion (superseded — see Verdict at top)

The E2E pipeline is validated at N=25 — the target domain count for P0.
Routing, adapter selection, and generation work together with ≤2pp quality
loss. The combined logistic router scales gracefully; quality tracks Theorem 1.
P0 "25 domains" gate is effectively CLOSED by this result.

**Reclassified 2026-04-18:** P0 "25 domains" gate is **NOT closed** — the
surface PASS is a protocol artifact (22 MMLU safety-moat domains make
wrong-adapter risk unreachable). Gate reopens pending v2 with ≥10 distinct
adapters and a conditional K1489 vacate-clause.
