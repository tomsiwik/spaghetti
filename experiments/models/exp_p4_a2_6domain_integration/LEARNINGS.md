# LEARNINGS.md — P4.A2: 6-Domain System Integration

**Status:** SUPPORTED — Finding #476
**Experiment:** exp_p4_a2_6domain_integration
**Date:** 2026-04-11

## What We Learned

### 1. N+1 Extension Is Instantaneous (76ms)
Adding a 6th domain required zero changes to existing adapters and only 76ms to retrain the router. The TF-IDF Ridge classifier is O(N·d) in training, not exponential. At N=6, the marginal cost of a new domain is negligible.

**Implication:** The system scales to 25+ domains with router retraining cost dominated by data prep, not computation.

### 2. Perfect Routing at N=6 Requires Geometric Safety (cos < 0.30)
Biology's highest cosine similarity to existing domains was 0.117 (medical). This is 2.5× below the 0.30 Theorem 1 safety threshold. Result: 100% routing accuracy with zero false positives.

**Implication:** The geometric safety condition from Theorem 1 is conservative. New domains with cos < 0.30 to all existing domains will route perfectly. The practical constraint is vocabulary distinctiveness, not training data volume.

### 3. Full-Run Cosines Are Higher Than Smoke-Test Cosines
Smoke test: cos(bio, med) = 0.062. Full run (N_TRAIN=300): cos(bio, med) = 0.117. More training data reveals more lexical overlap. The geometric safety check should use N_TRAIN=300+ to avoid underestimating domain similarity.

**Implication:** Always use the full-run cosine for safety evaluation. Smoke tests underestimate inter-domain similarity.

### 4. Pipeline Improvement Theorem Is Correct But Conservative
Theorem 2 predicted: E[Δ_pipeline] = P(route) × E[Δ_adapter] = 0.85 × 20pp = 17pp.
Actual: 100% routing × 30pp adapter = 30pp total.
The theorem is a lower bound; it assumes 85% routing and 20pp adapter improvement (P4.A1 baseline).

**Implication:** The pipeline improvement theorem gives a conservative floor. In practice, routing accuracy at N=6 is higher than the 85% guarantee, and adapter improvement can exceed the measured value.

### 5. Vocabulary Rubric Has High Variance at Question Level
3/10 biology questions showed regression (adapter replaced base vocabulary rather than supplementing). The ≥8 biology terms threshold is a rough proxy for depth but doesn't capture answer quality or factual accuracy.

**Implication:** P4.B series should test factual accuracy with LLM-as-judge evaluation, not just vocabulary rubric. This is the main open gap.

## Connection to Prior Findings
- Finding #474 (5-domain, 97.3%): N=6 maintains and exceeds N=5 accuracy
- Finding #475 (new domain <10min): Biology adapter trained in 7.53 min, router extends in 76ms total
- Finding #458 (ridge routing N=25): Ridge routing remains the correct architectural choice; P4.A2 confirms at N=6 with 100% accuracy

## What Comes Next (P4.B)
The P4.A series is complete. Three components verified:
1. Routing scales to N+1 domains (76ms, perfect accuracy)
2. New domain adapter trains in <10 minutes (Finding #475)
3. Full pipeline works end-to-end (E2E, Finding #473)

**Open gap:** Factual accuracy is unvalidated. The vocabulary rubric measures vocabulary depth but not correctness. P4.B0 should test domain adapters on factual accuracy benchmarks (LLM-as-judge) to quantify actual behavioral quality, not just vocabulary coverage.
