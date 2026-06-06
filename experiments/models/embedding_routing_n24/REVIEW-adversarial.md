# Peer Review: Embedding Routing N=24

## Experiment Type
Frontier extension (self-declared). MATH.md states it extends LoRAuter's centroid routing to the base model's own embedding layer, without an external sentence encoder.

## Hack Detector
- Fix count: 1 (replace hidden-state features with embedding features). Clean, no stacking.
- Is MATH.md a proof or a description? **Description dressed in equations.** Theorem 1 is labeled "Proof sketch" and explicitly says "full proof would require bounding the shared-vocabulary contribution." The bound `||c_k - c_j|| >= ||E_k^{unique} - E_j^{unique}|| * (1 - alpha_{kj})` is never rigorously derived -- the cancellation of shared components requires p_k(w) = p_j(w) for all shared words, which is stated as an approximation ("~=") not a proved condition. The QED is on a sketch, not a proof.
- Metric used as evidence: Routing accuracy (top-1 match). Adequate for the claim being tested.
- Kill criteria source: P1-P3 are derived from the mathematical framework. Reasonable.

## Self-Test Audit

1. **One-sentence impossibility property:** "Embedding lookup preserves lexical identity without contextual mixing." This is a MECHANISM DESCRIPTION, not an impossibility property. An impossibility property would be: "It is impossible for mean-pooled raw embeddings to collapse when domains have vocabulary overlap below alpha." The self-test answer explains WHY embeddings should work, not what structural property PREVENTS failure. **FLAG: not an impossibility property.**

2. **Cited theorems:** LoRAuter (2601.21795) and JL-lemma. LoRAuter's theorem is about a SupCon-trained sentence encoder, NOT about raw embedding lookups. The conditions (sentence encoder trained with contrastive loss) explicitly do NOT apply to the base model's embedding layer. The JL-lemma application is correct but irrelevant -- dimensionality is not the bottleneck. **FLAG: LoRAuter theorem conditions violated in this setting.**

3. **Predicted numbers:** P1 (6 domains >90%), P2 (accuracy >> 39.4%), P3 (overhead <1ms). P1 and P2 are directional but specific enough. P3 is quantitative. Adequate.

4. **Falsification condition:** "If embedding centroids are NOT more separable than hidden-state centroids." This is correct and was in fact falsified. Good.

5. **Hyperparameter count:** 0. Correct.

6. **Hack check:** Clean replacement, not stacking fixes. Pass.

## Mathematical Soundness

**Theorem 1 is not a theorem.** It is a sketch with a critical gap:

1. The decomposition of centroids into shared and unique components is valid algebra.
2. The claim that "shared component cancels in the difference c_k - c_j" requires p_k(w) = p_j(w) for ALL shared words. This is written as an approximation ("~="), never bounded. In reality, even shared words like "the" appear at different frequencies across domains. The residual from imperfect cancellation can dominate the unique-vocabulary signal.
3. The bound is therefore not actually proved. It is an intuition formalized as an inequality without establishing the precondition.

**The JL-lemma application is technically correct but misleading.** JL guarantees that d=2560 is sufficient to preserve pairwise distances among 24 points with low distortion. But the problem is not dimensionality -- it is that mean-pooling over high-frequency shared vocabulary collapses ALL centroids toward the same point. JL preserves distances that exist; it does not create separation where none exists.

**The information-theoretic argument in Section B is wrong in a subtle way.** It is true that the embedding layer preserves lexical identity (it is a lookup table). But mean-pooling over L tokens is not the same as preserving lexical identity -- it computes a weighted average where weight is proportional to token frequency. High-frequency function words dominate. The MATH.md acknowledges this as Assumption A2 but does not analyze it quantitatively. The experiment then confirmed this is the exact failure mode (centroid collapse due to common-word domination).

**Credit:** The worked example in Section F is honest and actually illustrates the problem -- it shows that domains sharing vocabulary (finance/cooking with cos 0.92) are hard to separate. The example should have been a warning sign, not just an illustration.

## Prediction vs Measurement

PAPER.md contains a clear prediction-vs-measurement table. Results:

| Prediction | Measured | Verdict |
|-----------|----------|---------|
| P1: 6 domains >90% | Only 2 stable (math, psychology) | FAIL |
| P2: Accuracy >> 39.4% | 28.3% (11pp worse) | FAIL |
| P3: Overhead <1ms | 0.36ms P95 | PASS |
| Embedding centroids better separated | Mean cos 0.986 vs hidden 0.716 | FAIL (opposite direction) |

3 of 4 predictions failed. The kill is clean and well-documented.

## NotebookLM Findings
Skipping NotebookLM -- the experiment is already killed with thorough analysis. The review can be completed from the documents alone.

## Novelty Assessment

The idea of using base-model embeddings instead of an external encoder is a reasonable frontier extension of LoRAuter. However, LoRAuter explicitly uses a SupCon-trained encoder for good reason -- raw embeddings lack the contrastive training that creates separable representations. This gap was identifiable a priori from the LoRAuter paper itself, which should have been a stronger warning.

The key finding -- that transformer layers ADD discriminative signal rather than destroying it -- is genuinely useful. It refutes the "contextual mixing destroys domain signal" hypothesis and reframes the problem correctly. This is a valuable negative result.

The discovery that TF-IDF (35.0%) beats all neural embedding methods is a useful calibration point. TF-IDF's IDF weighting naturally solves the exact problem (common-word domination) that kills mean-pooled embeddings.

## Macro-Scale Risks (advisory)
Not applicable -- approach is killed. The finding that raw embedding centroids collapse generalizes to any scale (structural property of mean-pooled embeddings from autoregressive LMs).

## Verdict

**KILL** (confirmed)

The experiment is already correctly killed. The kill is clean: 3 of 4 predictions failed, and the failure mode (centroid collapse at cos 0.986) is structural, not a scale artifact. The PAPER.md analysis is thorough and identifies the correct root cause.

**Specific weaknesses in the research artifacts:**

1. **MATH.md Theorem 1 is not a proof.** The "Proof sketch" label is honest but does not meet the proof-first standard. The critical assumption (shared-vocabulary frequency equality) was never bounded, and it is exactly the assumption that fails in practice. A proper analysis would have predicted centroid collapse by computing the expected cosine similarity under realistic token frequency distributions (Zipf's law guarantees common words dominate any mean).

2. **The LoRAuter citation is misleading.** MATH.md Section C states the LoRAuter theorem as if it supports embedding routing, but LoRAuter's conditions (SupCon-trained encoder) are explicitly violated. The theorem should have been cited as "this is what we lack" rather than "this is what we build on."

3. **The impossibility analysis was backwards.** MATH.md Section A identifies "contextual mixing destroys domain signal" as the disease. The experiment proved this diagnosis was wrong -- contextual mixing ADDS signal. The correct disease (mean-pooling collapses to common-word centroid) could have been identified a priori from Zipf's law analysis of token frequencies.

**What was done well:**
- Zero hyperparameters, clean experimental design
- Five-way comparison (embedding, instruction-only, hidden-state, TF-IDF, trained)
- Honest prediction-vs-measurement table
- Correct identification of the structural failure mode post-hoc
- The reframing in the Key Finding section is the most valuable output of this experiment
