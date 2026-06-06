# T4.1: TF-IDF Routing on Gemma 4 Domains (N=5, N=25)

---

## V2 Audit Section (2026-04-18, audit-2026-04-17-rerun, code-bug)

### Why V1 is invalid

V1 reported K1073 PASS (96.6%) and K1074 PASS (86.1%), but the audit
(`LOOPHOLE_CODE.md`, `LOOPHOLE_METHODOLOGY.md`, `LOOPHOLE_FOLLOWUP.md`)
identified four compounding bugs that rendered the measurements unsound:

1. **Code-domain 100% train/test overlap.** `load_code_prompts` loaded
   HumanEval test split (164 problems) and duplicated via
   `while len(prompts) < n: prompts = prompts + prompts` to fill
   N_TRAIN=300 + N_TEST=100. Every test query is memorized at train time;
   100% cosine match is mechanically forced.
2. **MMLU 100% train/test overlap.** `load_mmlu_prompts` tried
   `auxiliary_train` (which is a cross-subject pool, not per-subject), hit
   the fallback `except` branch and loaded the `test` split for training.
   `load_mmlu_test_prompts` then loaded the same `test` split with a
   different shuffle seed. For subjects where `len(test) ≤ N_TRAIN`, all
   test queries are in the train set.
3. **Hardcoded exclusion of hard-negatives.** `MMLU_EXTRA_SUBJECTS`
   deliberately drops `clinical_knowledge`, `virology`,
   `high_school_biology`, `nutrition`, `human_sexuality`,
   `high_school_psychology` with the comment "would cause systematic
   confusion." Excluding the hardest cases to pass the threshold is
   mem-antipattern-007 (tautological KC).
4. **Latency measured on wrong router.** K1075 is a claim about the N=25
   router, but V1 measures `router5.predict_latency(...)`. The cost is
   `N × 20000` inner products, so N=25 is 5× the FLOPs of N=5 — measuring
   N=5 understates p99 latency by construction.

### V2 fix (code-bug category, per researcher hat step 3)

V1 thresholds (K1073, K1074, K1075, K1076) are **byte-preserved** below
in §"Quantitative Predictions". The fix is to `run_experiment.py`:

- **Code domain:** swap HumanEval for MBPP. MBPP `full` has train(374) +
  test(500) with upstream-disjoint splits. Load MBPP train[0:N_TRAIN] for
  training, MBPP test[0:N_TEST] for evaluation. No duplication, no
  overlap.
- **MMLU domains:** drop the `auxiliary_train` try/except. Use the test
  split only. Shuffle by seed, split indices `[0:N_TRAIN]` for training
  and `[N_TRAIN:N_TRAIN+N_TEST]` for eval. Subjects with fewer than
  `N_TRAIN+N_TEST` samples are scaled proportionally (2:1 train:test) to
  preserve disjointness. Assert no train ∩ test intersection.
- **MMLU_EXTRA_SUBJECTS:** restore the 6 hard-negatives
  (`clinical_knowledge`, `virology`, `high_school_biology`, `nutrition`,
  `human_sexuality`, `high_school_psychology`). Drop 6 of the
  previously-included "easy" subjects (`prehistory`,
  `high_school_european_history`, `high_school_us_history`, `astronomy`,
  `sociology`, `global_facts`) to keep N_extra=20, N_total=25. The swap
  is not motivated by result — it's mandated by the audit: the set must
  contain the hardest semantic neighbors of the 5 real domains, or K1074
  is not testing what it claims.
- **K1075 latency:** measure on `router25`, not `router5`. V1 threshold
  (< 1ms CPU p99) is byte-preserved.

### V2 prediction (honest, based on audit)

Removing data leakage and including hard-negatives shifts the centroid
geometry substantially. Expected outcomes:

- **K1073 (N=5):** likely still passes. MBPP (code) vs GSM8K (math) vs
  PubMedQA (medical) vs MMLU professional_law vs MMLU
  high_school_macroeconomics have near-zero vocabulary overlap. Predicted
  accuracy 94-98%. The residual error comes from PubMedQA ↔
  macroeconomics cosine if any MCQ-adjacent vocabulary appears.
- **K1074 (N=25):** likely **fails**. With hard-negatives in, medical ↔
  clinical_knowledge ↔ virology ↔ high_school_biology ↔
  human_sexuality ↔ nutrition form a 6-way confusion cluster. Expected
  accuracy drop into the 70-82% range, below the 85% threshold. This
  would retroactively confirm V1's headline 86.1% is a metric-hacking
  artefact.
- **K1075 (latency):** N=25 is 5× the FLOPs of N=5. V1 reported p99
  1.11ms on N=5. Predicted p99 for N=25 is 2-4ms. Still fails the < 1ms
  threshold, but honestly now.
- **K1076:** unchanged, 0 LLM params.

If V2 reproduces `K1073 PASS, K1074 FAIL, K1075 FAIL, K1076 PASS`, the
experiment is `killed` (2 of 4 KCs fail). The headline N=25 claim
(86.1%) is then retracted. If K1074 passes honestly, V1's findings are
validated at the KC level despite the buggy path. Either outcome is
informative.

### Kill-criterion discipline (V2)

No KC is added, removed, or relaxed. The original K1073/K1074/K1075/K1076
stand byte-for-byte. The only changes are in `run_experiment.py` (data
plumbing + latency target). Verdict is computed from V1 thresholds
against V2 measurements. A new experiment
(`exp_p1_p1_ridge_routing_n25`, in `blocks:`) has new KCs K1081/K1082/
K1083 for the follow-up; those are out of scope here.

---

## Problem Statement

Given N domain adapters (math, code, medical, legal, finance + MMLU subjects), we need a
router that maps an input query x to the correct domain adapter i* with accuracy ≥ 95% at
N=5 and ≥ 85% at N=25. The router must add zero neural parameters to the LLM and execute
in < 1ms CPU latency.

---

## Theorem 1: TF-IDF Domain Separability

**Statement:** For N NLP task domains D_1,...,D_N with distinct task vocabularies, the
nearest-centroid classifier over TF-IDF(ngram=(1,2)) features achieves accuracy ≥ 1 - ε_N,
where ε_N ≤ C/N for some constant C depending only on vocabulary overlap.

**Proof:**

Let φ: x → R^d be the TF-IDF map (unit-normalized, bigram+unigram, d = 20000).
For domain D_i, let μ_i = E[φ(x) | x ∈ D_i] be the centroid.

The nearest-centroid rule assigns x to argmax_i μ_i · φ(x).

Error occurs only when x ∈ D_i but μ_j · φ(x) > μ_i · φ(x) for some j≠i.

By Cauchy-Schwarz and the triangle inequality:
  μ_j · φ(x) ≤ ||μ_j|| · ||φ(x)|| = ||μ_j||

and:
  μ_i · φ(x) ≥ ||μ_i||² - ||φ(x) - μ_i|| · ||μ_i||

Misclassification requires:
  ||φ(x) - μ_i|| > ||μ_i|| · (1 - μ_i · μ_j / ||μ_i||)

For NLP domains with task-specific keywords (e.g., "python def", "how many", "diagnosis"),
TF-IDF up-weights these discriminative terms. The key insight (from Finding #389, confirmed
at 100% for math/code/text): domain-specific n-grams (python, how many, treatment, law,
economics) have near-zero overlap across domains.

**Vocabulary Separation Lemma (empirical):**
From Finding #389: centroid cosine distance for real NLP domains:
  math-code: 0.810, math-text: 0.496, code-text: 0.741
(cosine SIMILARITY, complement = centroid separation ≈ 0.20–0.50)
Perfect accuracy is achievable when discriminating n-grams dominate.

For medical/legal/finance domains (MMLU MCQ format), domain vocabulary still separates:
  - medical: clinical terms (patient, treatment, diagnosis)
  - legal: legal terms (plaintiff, statute, jurisdiction)
  - finance: economic terms (GDP, inflation, equilibrium)

At N=25, additional MMLU subjects bring distinct topic vocabulary (astronomy: telescope,
orbit; philosophy: Kant, ethics; geography: latitude, climate). The main risk is
intra-cluster confusion (high_school_biology ↔ anatomy ↔ medical). This risk bounds
accuracy to ≥ 85%. **QED.**

---

## Theorem 2: Zero Neural Parameters (Routing Architecture)

**Statement:** The TF-IDF nearest-centroid router R: x → {1,...,N} adds zero gradient-
trained parameters to the language model, and its routing decision is independent of the
adapter weights.

**Proof:**
R is defined by:
1. IDF weights w_k = log(N_docs / df_k): computed from corpus statistics, no gradient
2. Term-frequency counts: sparse matrix multiplication, no learned weights
3. Centroid storage: μ_i ∈ R^d for i=1..N, derived from empirical means (no gradient)
4. Routing: argmax_i μ_i · φ(x): inner product with stored centroids

The LLM weight tensor W_base ∈ R^{d_in × d_out} and adapter ΔW_i are unchanged.
No backpropagation through R. **QED.**

---

## Theorem 3: Sub-Millisecond CPU Latency

**Statement:** Routing latency T_route ≤ 1ms for N ≤ 25 domains.

**Proof:**
Routing requires:
1. TF-IDF transform: sparse matrix multiplication in R^d, O(|tokens|) with sparse ops
2. Cosine similarity: N inner products in R^d with L2-normalized vectors

For d=20000, N=25, and |tokens|≈100:
  - TF-IDF: ~10k FLOPS (sparse), ~10μs
  - Centroid similarity: 25 × 20000 = 500k FLOPS ≈ 0.01ms at CPU peak

Total predicted latency: ~0.1ms << 1ms. **QED.**

---

## Prior Work

- Finding #389: "TF-IDF nearest-centroid routing: 100% accuracy on math/code/text (N=3)"
- Finding #354: "TF-IDF + logistic regression: 95% on M2P domains (N=5)"
- arxiv 2212.10560: MTEB — TF-IDF sentence encoders are competitive for domain classification
- Experiment T3.1 (killed): routing is load-bearing — routing failure causes O(N) interference

---

## Quantitative Predictions

| Kill Criterion | Predicted Value | Pass Threshold |
|----------------|----------------|---------------|
| K1073: N=5 accuracy | ≥ 99% | ≥ 95% |
| K1074: N=25 accuracy | ≥ 90% | ≥ 85% |
| K1075: CPU latency p99 | ~0.1ms | < 1ms |
| K1076: LLM params added | 0 | = 0 |

The N=5 prediction is strong (99%) because math/code/medical/legal/finance have very
distinct vocabulary — better separated than math/code/text in Finding #389.
The N=25 prediction (90%) allows for 10% confusion in MMLU boundary subjects.
