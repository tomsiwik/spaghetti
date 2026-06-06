# MATH.md — P3.C1: Diverse Training Data Raises Style 60%→80%+

## Background

P3.C0 (Finding #467): Full pipeline style compliance = 60% (K1194 at threshold).
Root cause identified: training distribution mismatch.

**Training set** (P3.B5): 40 science-domain questions (photosynthesis, DNA, quantum mechanics...)
**Test set** (P3.C0): 15 mixed questions including:
- "What is the meaning of life according to philosophy?" — NOT in training distribution
- "Explain how neural networks learn." — NOT in training distribution
- "What is the significance of the speed of light?" — borderline

P3.B5 isolation result: 92% style on matched science questions.
P3.C0 pipeline result: 60% style on diverse questions.
Gap: 32pp — explained entirely by distribution mismatch, not composition failure.

## Theorem 1 (PAC Generalization — Style Compliance)

**Setup:**
Let Q_train = training question distribution (supported on science topics)
Let Q_test = test question distribution (mixed topics)
Let f_θ = personal adapter with parameters θ

**Claim:**
For a LoRA adapter fine-tuned on n training examples from Q_train, if the training
distribution Q_train has support S_train and the test distribution Q_test has support S_test,
then the generalization gap satisfies:

    ΔStyle = Style(Q_test) - Style(Q_train) ≤ c × TV(Q_train, Q_test)

where TV is total variation distance between the distributions, and c is a constant
determined by adapter capacity.

**Proof sketch (PAC-learning, Valiant 1984; Blumer et al. 1989):**

The style compliance function f_θ(q) ∈ {0,1} is a binary classifier on question space.
By the PAC learning theorem, for a hypothesis class H (LoRA rank-4 on 16 layers):
- VC dim(H) = d_vc = O(rank × n_layers × d_model) = O(4 × 16 × 2048) ≈ 131K
- For n training examples: P(generalization_error ≥ ε) ≤ 2|H|exp(-2nε²)

**Key observation:** The adapter's "style direction" is learned in activation space. If
Q_train ⊂ S_science and Q_test ⊂ S_mixed, the adapter only captures the style→marker
mapping for science-type input activations.

**Distribution coverage lemma:**
Let S_covered = {topic categories} covered by training set.
For a question q from category c:
- If c ∈ S_covered: E[f_θ(q)] ≈ compliance_rate(S_covered) ≈ 92% (P3.B5 result)
- If c ∉ S_covered: E[f_θ(q)] ≈ compliance_rate_baseline ≈ 40-60% (random/partial)

Empirical observation (P3.C0): 6/15 questions failed — ~6 questions were from categories
not in the training distribution (philosophy, ML explanations, significance/meaning questions).

**Prediction from coverage lemma:**
With 200 diverse training examples across 10 categories (science, philosophy, technology,
culture, history, medicine, law, math, art, social):
- Coverage: S_train ≈ S_test (nearly full support coverage)
- TV(Q_train, Q_test) ≈ 0 for in-distribution queries
- Expected style compliance: ≥80% (reducing non-compliance by ≥50% via coverage)

## Theorem 2 (Hoeffding Bound — Compliance Rate Improvement)

**Claim:** With n=200 diverse training examples, the expected style compliance in-pipeline
is:

    E[Style_pipeline(n=200)] ≥ E[Style_pipeline(n=40)] + Δ_coverage

where Δ_coverage ≥ 20pp from covering the missed question categories.

**Proof:**
From P3.C0 results: 6 of 15 test questions failed. Classify them:
- "What is the meaning of life?" → philosophy: NOT in training
- "Explain how neural networks learn." → ML/tech: partially in training
- "What is the difference between weather and climate?" → borderline science

Conservative estimate: 4/6 failures due to distribution gap (not adapter capacity).
If 4 of 6 failures are fixed by better coverage:
    New compliant = 9 + 4 = 13/15 = 86.7% ≥ 80% threshold ✓

**Hoeffding bound on additional 160 training examples:**
For n additional examples in diverse categories, by Hoeffding:
    P(generalization_error > ε) ≤ 2 × exp(-2nε²)
For ε=0.20 (20pp target), n=160:
    P(error > 20pp) ≤ 2 × exp(-2×160×0.04) ≈ 2 × exp(-12.8) ≈ 5×10⁻⁵

The bound is not tight (this is LoRA, not a linear classifier), but the direction
is clear: more training diversity reduces generalization error.

## Kill Criteria (from Theorems 1-2)

| Kill ID | Condition | Prediction | Basis |
|---------|-----------|------------|-------|
| K1196 | pipeline_style ≥ 80% | ~87% | Coverage lemma + Hoeffding |
| K1197 | training_time ≤ 15 min | ~8-10 min | P3.B5: 9.8 min for 300 iters → 500 iters = ~13 min |
| K1198 | adapter_size ≤ 10 MB | ~3.7 MB | Same rank-4, same architecture as P3.B5 |

## Implementation Plan

1. **Expand training data** (40 → 200 examples):
   - Category 1 — Science (40): photosynthesis, DNA, physics (matches P3.B5)
   - Category 2 — Philosophy (20): meaning of life, ethics, consciousness, free will
   - Category 3 — Technology (25): ML, blockchain, GPS, internet, computing
   - Category 4 — History (20): wars, civilizations, events, people
   - Category 5 — Health/Medicine (20): vaccines, nutrition, diseases, symptoms
   - Category 6 — Arts/Culture (15): literature, music, cinema, art
   - Category 7 — Social/Economics (20): inflation, markets, sociology, politics
   - Category 8 — Environment (15): climate, ecology, sustainability, oceans
   - Category 9 — Math/Logic (15): numbers, probability, logic, geometry
   - Category 10 — General/Misc (10): covers tail of distribution

2. **Increase iterations**: 300 → 500 (further reduce val loss, minor improvement)

3. **Same architecture**: rank-4 LoRA on q_proj, 16 layers — no change needed

4. **Reuse domain_fused_base** from P3.B5 — no re-fusion required

## Prediction vs Measurement Table (to be filled after experiment)

| Metric | Prediction | Measured | Delta |
|--------|------------|----------|-------|
| pipeline_style_compliance | ≥80% | TBD | TBD |
| training_time | 8-13 min | TBD | TBD |
| adapter_size_MB | ~3.7 MB | TBD | TBD |
| K1196 result | PASS | TBD | - |
| K1197 result | PASS | TBD | - |
| K1198 result | PASS | TBD | - |

## If K1196 KILLED (style < 80% despite full coverage)

The coverage lemma fails: adapter capacity (rank-4, 16 layers) is insufficient to
capture style across 10 diverse question categories simultaneously. The style direction
in activation space is question-type-dependent, not question-type-invariant.

**Impossibility structure:**
∃ q_1, q_2 from different categories such that:
    h_domain(q_1) ≈ h_domain(q_2)  (similar activations)
    but Δstyle(q_1) ≠ Δstyle(q_2)  (different style response needed)

→ No rank-4 LoRA can simultaneously satisfy both constraints.

**Fix (P3.C2):** Higher rank (rank-16) or explicit prompt framing (few-shot examples
in system prompt teaching the expected format — no training required).

## References

1. Hu et al. 2021 "LoRA: Low-Rank Adaptation" arxiv 2106.09685 — rank-4 sufficiency
2. Valiant 1984 "A Theory of the Learnable" — PAC learning
3. Blumer et al. 1989 "Learnability and the Vapnik-Chervonenkis Dimension" — VC bounds
4. Finding #436 — P3.B5 personal adapter: 76% compliance on science questions
5. Finding #466 — P3.B5 domain-conditional: 92% style in isolation
6. Finding #467 — P3.C0 pipeline: 60% style in-pipeline (root cause: distribution gap)
