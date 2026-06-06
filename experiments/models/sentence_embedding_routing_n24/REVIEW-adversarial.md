# Peer Review: Sentence-Embedding Routing at N=24

## Experiment Type
Guided exploration. The proven framework is Finding #255 (sentence-embedding centroid routing at 96% for N=5). The unknown is whether this scales to N=24. This is correctly structured: a proven mechanism being tested at a new operating point.

## Hack Detector
- Fix count: 0 (clean scale test of existing mechanism)
- Is MATH.md a proof or a description? **Description dressed in equations.** Theorem 1 is labeled as a proof but is actually a proof sketch with informal reasoning ("approximately", "O(sigma_k)"). The QED is attached to a sketch, not a complete derivation. However, for a guided exploration, the bar is lower -- the framework (Fisher discriminant + JL) is well-established. The real question is whether the predictions are testable, which they are.
- Metric used as evidence: Top-1 routing accuracy (directly measures the behavioral outcome: does the right adapter get selected?). This is appropriate.
- Kill criteria source: K1 derived from Fisher ratio threshold, K2 from DDR finding, K3 from architecture. Reasonable derivation.

## Self-Test Audit

1. **One-sentence impossibility property:** "Sufficient Fisher discriminant ratio (R > 2.0)..." -- This is one property (Fisher ratio threshold). PASS. However, the experiment itself revealed this property is WRONG: R=2.93 but accuracy=33.3%. The self-test answer was honest at design time but the experiment falsified it. This is exactly what a good experiment should do.

2. **Cited theorems:** JL lemma (1984) and Fisher LDA (1936) are real and correctly stated. JL is applied correctly (d=384 >> 13 needed for N=24). Fisher criterion is standard. PASS.

3. **Predicted numbers:** Fisher ratio 2.0-4.0, accuracy 65-85%, 3-6 confused pairs, overhead <10ms, PPL improvement 5-12%. These are specific and falsifiable. PASS.

4. **Falsification condition:** "The proof is wrong if Fisher ratio > 2.0 but accuracy < 60%." This is exactly what happened (R=2.93, accuracy=33.3%). The falsification condition correctly identified the proof's vulnerability. PASS -- this is a well-designed falsification test.

5. **Hyperparameter count:** 1 (number of centroid samples), with derivation. PASS.

6. **Hack check:** No stacking. Clean scale test. PASS.

## Mathematical Soundness

**What holds:**
- JL lemma application is correct: d=384 is more than sufficient for N=24 points. Dimensionality is not the bottleneck.
- Fisher criterion definition is standard and correctly computed.
- The N=5 baseline numbers (R=5.61, accuracy=96%) are grounded in prior findings.
- Proposition 1 (centroid crowding bound) correctly identifies that the geometric bound is vacuous for d=384, N=24, and that the real issue is semantic similarity.

**What does not hold:**
- Theorem 1 is a proof sketch, not a formal proof. The transition from "delta < O(sigma_k)" to a routing accuracy bound is hand-waved. The critical gap: the proof uses average Fisher ratio R as the predictor but routing accuracy depends on per-domain minimum margins. The proof never formalizes this distinction.
- The Corollary defines "confused pairs" using the average sigma_max, but never bounds how many such pairs can exist at N=24. The prediction of "3-6 confused pairs" appears to be a guess, not derived from the proof. Measured: 91 confused pairs.
- The extrapolation in Section F assumes mean inter-centroid cosine ~0.65. Measured: 0.798. This is a factor-of-2 error in (1-cos), showing the model of how embeddings distribute was wrong.

**The key insight the proof missed:** MiniLM-L6-v2 embeddings are anisotropic with a high baseline cosine (~0.8). This is the "hubness" phenomenon (Radovanovic et al., 2010). The proof assumed isotropic concentration, which is standard but wrong for transformer sentence embeddings. PAPER.md correctly identifies this post-hoc.

**Severity:** For a guided exploration, having the proof sketch be falsified is an acceptable outcome -- it means the exploration discovered something. The proof was wrong in a specific, identifiable way (average vs. minimum margin; isotropic vs. anisotropic embedding space), and the experiment cleanly revealed this.

## Prediction vs Measurement

PAPER.md contains the required prediction-vs-measurement table. Assessment:

| Prediction | Measured | Verdict |
|---|---|---|
| Fisher ratio 2.0-4.0 | 2.93 | MATCH -- within predicted range |
| Top-1 accuracy 65-85% | 33.3% | FALSIFIED -- off by 2x |
| Mean inter-centroid cosine 0.55-0.70 | 0.798 | FALSIFIED -- hubness not modeled |
| Confused pairs 3-6 | 91 | FALSIFIED -- catastrophic, off by 15x |
| Overhead <10ms | mean 9.6ms, p99 78.3ms | PARTIAL -- mean OK, p99 inflated by warmup |
| Routed PPL < uniform | 19.02 >= 18.98 | FALSIFIED |

The Fisher ratio prediction matching while accuracy is falsified is the most informative result: it proves that average separability (Fisher ratio) is a poor proxy for routing accuracy when margins are non-uniform. This is a genuine learning.

## NotebookLM Findings
Skipped -- the experiment is already killed with clear results. Deep review would not change the verdict.

## Novelty Assessment

The experiment itself is a straightforward scale test, not a novelty claim. The finding that off-the-shelf sentence embeddings fail at N=24 due to hubness/anisotropy is not novel in the broader ML literature (Radovanovic et al. 2010, Mu and Viswanath 2018 "All-but-the-Top"), but it is a useful negative result in the context of this project's routing problem.

The deeper finding -- that Fisher ratio is necessary but not sufficient, and that minimum margin governs routing -- is well-known in the classification literature but was not previously tested in this project's adapter routing context. Worth recording.

## Methodological Concerns

1. **Sample sizes are small but adequate for the conclusion.** 10 test samples per domain at 24 domains = 240 total. The 33.3% accuracy is so far below the 60% threshold that even with wide confidence intervals (binomial: roughly +/- 6%), the conclusion is unambiguous.

2. **P99 overhead inflation.** The 78.3ms p99 is likely dominated by first-query warmup (model loading, JIT compilation). Reporting steady-state p99 would be more honest. However, since K1 and K2 already fail, K3 is moot.

3. **PPL evaluation on 12/24 domains.** This is acknowledged in Limitations. Given that oracle PPL is WORSE than base PPL on average (19.16 vs 18.97), the adapters themselves appear to provide no value at this scale. This raises a question outside this experiment's scope: are the N=24 adapters well-trained? If adapters are not meaningfully specialized, routing accuracy is irrelevant.

4. **The "margin > 0.10 predicts accuracy >= 80%" observation** in PAPER.md (Table, line 98-99) is a useful empirical finding that could seed a better predictor for future experiments.

## Macro-Scale Risks (advisory)

1. The hubness problem gets WORSE in higher dimensions with more domains. Any sentence-embedding routing approach at N>24 will need to address anisotropy (e.g., whitening, contrastive fine-tuning).
2. The finding that oracle PPL is worse than base PPL at N=24 suggests the adapter training pipeline may need attention before routing matters at all.

## Verdict

**KILL -- well-executed, well-documented**

The kill is clean and well-justified:

1. All three kill criteria fail, with K1 failing catastrophically (33.3% vs 60% threshold).
2. The proof's falsification condition was met exactly as stated (R > 2.0 but accuracy < 60%).
3. The root cause is identified (hubness/anisotropy, average-vs-minimum margin gap).
4. PAPER.md contains proper prediction-vs-measurement table.
5. Actionable next steps are identified (contrastive fine-tuning, hierarchical routing).
6. Finding #257 status "killed" is correct.

This is a model micro-experiment: the proof made testable predictions, the experiment falsified them, and the failure mode was diagnosed. The only weakness is that the "proof" in MATH.md was really a proof sketch (the QED is premature), but for a guided exploration that discovered a real structural limitation, this is acceptable.

No revisions needed. The experiment served its purpose.
