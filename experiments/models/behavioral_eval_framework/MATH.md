# Behavioral Evaluation Framework: Mathematical Foundations

## Type: Infrastructure (evaluation tooling with design rationale)

**Note:** This is an evaluation framework, not a scientific experiment testing a
theoretical prediction. The "theorems" below are design principles that justify
metric choices — they are definitional properties of the metrics, not discovered
mathematical results. The framework's value is engineering (reliable eval), not
proof of a novel claim.

## A. Failure Mode Identification

**The disease:** Evaluation metrics that do not correlate with task correctness
create a paradox where better-performing models score worse.

This is not hypothetical. Finding #179 demonstrated it empirically: the math adapter
produces 24x more correct answers (48% vs 2%, n=50) but receives lower LLM-judge
scores (3.6 vs 4.0, p=0.002). Keyword density and coherence metrics similarly
fail to detect genuine domain expertise.

**The root cause is metric-outcome misalignment:** Surface-level text statistics
(keyword density, n-gram diversity, sentence length) measure text properties
orthogonal to task correctness. A verbose wrong answer scores higher than a
concise correct answer on these metrics.

## B. The Right Question

Wrong: "How do we improve our evaluation metrics?"
Right: "What is the minimal verifiable signal for each domain such that the
metric is monotonically related to task correctness by construction?"

The answer: use execution-based evaluation where correctness is defined by
matching a verifiable ground truth, not by measuring surface text properties.

## C. Prior Mathematical Foundations

**Information retrieval theory (Van Rijsbergen, 1979):** Precision and recall
against a reference set provide a well-understood quality measure with known
statistical properties. F1 = 2PR/(P+R) is the harmonic mean that balances both.

**Exact match and numerical equivalence:** For domains with computable answers
(math, finance), exact match or epsilon-ball equivalence (|x - x_ref| / |x_ref| < eps)
provides a binary correctness signal with zero false positives when eps is small.

**Cohen's kappa (Cohen, 1960):** For measuring inter-rater agreement beyond chance:
kappa = (p_o - p_e) / (1 - p_e)
where p_o is observed agreement and p_e is expected agreement by chance.
kappa >= 0.7 is conventionally "substantial agreement."

**Factual overlap (adapted from ROUGE, Lin 2004):** For prose domains without
computable answers, we extract key factual claims from reference answers and
measure what fraction appear in generated text. This is essentially recall
of factual content, which is monotonically related to correctness (more correct
facts = better answer) unlike keyword density (more keywords != better answer).

## D. Design Principles (not theorems)

**Design Principle 1 (Monotonicity of execution-based metrics).**
Let q be a query with ground truth answer a*. Define the correctness function:
- For computable domains: C(a) = 1 if |a - a*|/|a*| < eps, else 0
- For factual domains: C(a) = |F(a) ∩ F(a*)| / |F(a*)| where F extracts factual claims

Then C is monotonically related to task correctness by construction: a response
that is more factually correct will have C >= a less factually correct response.

*Rationale.* For computable domains, C is binary (correct or not), which is trivially
monotone. For factual domains, if response a1 contains strictly more correct
facts than a2 (F(a1) ∩ F(a*) ⊃ F(a2) ∩ F(a*)), then C(a1) > C(a2). This is
a definitional property of the metric, not a discovered result.

**Critical limitation (synonymy):** The monotonicity guarantee holds only when
correct facts are expressed in vocabulary matching the reference. A response
containing "hypertension" when the reference says "high blood pressure" scores 0
on that fact despite being synonymous. Substring matching cannot detect semantic
equivalence. This means factual recall is a LOWER BOUND on actual factual
correctness — it can undercount but not overcount correct facts. Future work
should replace substring matching with embedding-based semantic similarity.

**Design Principle 2 (Non-monotonicity of keyword density).**
There exist responses a1, a2 where a1 is task-correct and a2 is task-incorrect,
yet KD(a2) > KD(a1).

*Evidence.* Finding #179 (empirical counterexample). The math adapter generates
concise "<<26*3=78>>...#### 322" format which is correct but has low keyword
density. The base model generates verbose "let me think step by step..." which
is wrong but has higher keyword density. This is an empirical observation, not
a formal proof — it demonstrates a counterexample to keyword density's validity.

## E. Predictions

| Domain | Metric | What it measures | Prediction |
|--------|--------|------------------|------------|
| Code | ast.parse success rate | Syntactic validity | Code adapter > base (Finding #204: 70% vs 50%) |
| Math | Answer correctness (eps=0.01) | Numerical accuracy | Code adapter > base on math (Finding #204: 70% vs 10%) |
| Medical | Factual recall against reference | Medical fact coverage | Framework will detect differences invisible to keyword density |
| Legal | Factual recall against reference | Legal fact coverage | Framework will detect differences invisible to keyword density |
| Finance | Factual recall + numerical accuracy | Financial accuracy | Framework will detect differences invisible to keyword density |

**Key prediction for K612:** The framework must rank code adapter > base on math
domain, confirming Finding #204's 7x improvement (70% vs 10% correct answers).

## F. Assumptions & Breaking Conditions

1. **Reference answers are correct.** If training data contains errors, factual
   recall against wrong references is meaningless. Mitigation: use validation
   set (curated data).

2. **Factual claims can be extracted from text.** If generated text is too
   incoherent to parse, extraction fails. This is itself a quality signal
   (incoherent = bad).

3. **Key facts are identifiable in reference answers.** For prose domains,
   we must identify the important facts. We use a simple approach: extract
   noun phrases, named entities, and numerical values as "facts."

4. **20 samples sufficient for kappa.** Cohen's kappa with 20 samples has
   wide confidence intervals. This is a micro-scale limitation acknowledged
   by design.

## G. Worked Example (Medical Domain)

Reference answer: "Syringomyelia causes muscle weakness, stiffness, and spasms,
as well as bladder and bowel dysfunction. One of the most common symptoms is
bilateral loss of pain and temperature sensation in a cape-like distribution."

Key facts extracted: {syringomyelia, muscle weakness, stiffness, spasms,
bladder dysfunction, bowel dysfunction, bilateral loss, pain sensation,
temperature sensation, cape-like distribution}

Generated answer A: "Syringomyelia is a condition causing weakness and stiffness
in muscles, with characteristic cape-like sensory loss."
Facts matched: {syringomyelia, muscle weakness, stiffness, cape-like distribution} = 4/10
Factual recall: 0.40

Generated answer B: "The patient should see a doctor for their medical condition
and get proper treatment and medication for their symptoms."
Facts matched: {} = 0/10
Factual recall: 0.00

Keyword density would score B moderately (contains "patient", "doctor", "medical",
"treatment", "medication", "symptoms" = 6 medical keywords) while A gets a lower
keyword density despite being far more correct.

## H. Complexity

The framework adds zero training cost. Evaluation cost is O(N * D) where N is
prompts per domain and D is domains. Fact extraction is string matching, O(n*m)
where n is generated text length and m is reference text length. Total evaluation
time for 50 prompts across 5 domains: ~seconds (no model inference needed for
the evaluation itself; model inference for generation is the bottleneck).

## Self-Test

1. **One property:** Execution-based metrics are monotonically related to task
   correctness by definition (Design Principle 1), unlike surface text statistics.
   This is a definitional property of the metric choice, not a discovered result.

2. **Prior work motivating the approach:** Cohen's kappa (Cohen 1960), ROUGE-style
   recall (Lin 2004), exact match evaluation (standard in SQuAD, GSM8K benchmarks).

3. **Expected outcomes:** Code adapter > base on math by ~7x on answer
   correctness (re-measuring Finding #204 with new metric). Framework kappa >= 0.7
   (threshold from convention, not derived from theory).

4. **Falsification:** The framework fails if factual recall does NOT correlate
   with actual answer quality (i.e., if extracting more reference facts does
   not imply a better answer). This could happen if reference answers contain
   irrelevant facts, if the extraction is too noisy, or if correct facts are
   expressed in different vocabulary than the reference (synonymy limitation).

5. **Hyperparameters:** At least 6 design choices: (1) eps=0.01 for numerical
   comparison, (2) code domain weighting 0.7 syntax / 0.3 recall, (3) finance
   domain weighting 0.4 numerical / 0.6 recall, (4) minimum word length for
   fact extraction (4 chars), (5) stopword filtering in key fact extraction,
   (6) reference rater overlap thresholds (0.10 and 0.05). These are engineering
   choices, not tuned hyperparameters, but they do affect measurements.

6. **Hack check:** No. This is a single principled approach (measure what matters:
   correctness) replacing a stack of surface metrics. The framework is infrastructure,
   not a stacked fix.
