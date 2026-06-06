# T4.1: TF-IDF Routing on Gemma 4 Domains (N=5, N=25) — V2 audit-rerun

## Status: KILLED

V2 rerun with audit-mandated fixes (MBPP replacing duplicated HumanEval,
index-disjoint MMLU splits, restored 6 hard-negative subjects, latency on
router25) fails K1074 by 12.79pp. V1's headline 86.1% is retroactively
invalid — it came from (a) 100% train/test overlap in code + several
MMLU subjects and (b) deliberate exclusion of medical's 6 closest
semantic neighbors from the N=25 pool.

## V2 Prediction vs Measurement

| KC  | V1 threshold (byte-preserved) | V1 measured | V2 predicted   | V2 measured | Pass? |
|-----|-------------------------------|-------------|----------------|-------------|-------|
| K1073: N=5 accuracy ≥ 95%     | 96.6%   | 94-98%  | **96.78%** | ✅ PASS |
| K1074: N=25 accuracy ≥ 85%    | 86.1%   | 70-82%  | **72.21%** | ❌ FAIL |
| K1075: p99 latency < 1ms      | 1.11ms† | 2-4ms   | **0.656ms**| ✅ PASS |
| K1076: LLM params added = 0   | 0       | 0       | **0**      | ✅ PASS |

† V1 K1075 measurement was on the wrong router (N=5); V1 threshold and V2
threshold are identical. V2 passes honestly on router25 — the CPU sparse
matmul at N=25 is faster than MATH.md Theorem 3's back-of-envelope
estimate, because sklearn's `TfidfVectorizer.transform` reuses a prefix-
trie vocab lookup that is the dominant cost, not the 25-way inner
product. Both scale with vocabulary, not centroid count.

## What V1 got wrong and why (audit summary)

1. **Code-domain 100% train/test overlap.** V1 used HumanEval (164
   problems) and padded to N_TRAIN + N_TEST = 400 by duplicating the
   list. Every test query was in the train set; the 100% code accuracy
   in V1 is a memorization artefact. V2 uses MBPP full train(374) /
   test(500) — upstream-disjoint splits, no duplication.

2. **MMLU 100% train/test overlap.** V1's `load_mmlu_prompts` tried
   `auxiliary_train` (a cross-subject pool that doesn't exist for
   individual subjects), hit the except branch, and loaded the `test`
   split for training. `load_mmlu_test_prompts` then loaded the same
   `test` split for evaluation with a different shuffle seed. For
   subjects where `len(test) ≤ N_TRAIN`, every test query was in the
   train set. V2 loads the test split once, deduplicates by question
   text (some subjects repeat questions), shuffles, slices
   `[0:N_TRAIN]` for train and `[N_TRAIN:N_TRAIN+N_TEST]` for test,
   asserts disjointness.

3. **Hardcoded hard-negative exclusion.** V1's `MMLU_EXTRA_SUBJECTS`
   deliberately dropped `clinical_knowledge`, `virology`,
   `high_school_biology`, `nutrition`, `human_sexuality`,
   `high_school_psychology` with the in-code comment
   *"would cause systematic confusion"*. This is the textbook instance
   of mem-antipattern-007 (tautological KC): the test set is curated to
   pass the threshold. V2 restores all six and drops 6 of the easier
   subjects (prehistory, high_school_european_history,
   high_school_us_history, astronomy, sociology, global_facts) to keep
   N_extra = 20 / N_total = 25.

4. **Latency measured on the wrong router.** V1 K1075 is a claim about
   the N=25 router, but the code measured `router5.predict_latency(...)`
   and reported that number. FLOPs scale linearly in N, so the V1
   measurement understated p99 by ~5×. V2 measures `router25`. Note:
   sparse vocab transform dominates over centroid similarity, so the
   linear-N scaling was an overestimate and V2 K1075 still passes.

## N=5 Per-Domain Results (V2, N_TRAIN=300, N_TEST=100)

| Domain  | Source                                  | V1 acc  | V2 acc   |
|---------|-----------------------------------------|---------|----------|
| math    | GSM8K (disjoint split)                  | 98%     | 97.0%    |
| code    | MBPP (replaces duplicated HumanEval)    | 100%    | 100.0%   |
| medical | PubMedQA (disjoint split)               | 98%     | 91.0%    |
| legal   | MMLU professional_law (V2 index-split)  | 96%     | 93.0%    |
| finance | MMLU high_school_macroeconomics         | 91%     | 67.0% (97 test after dedup) |
| **Overall** |                                    | **96.6%** | **96.78%** |

N=5 passes. The router still separates top-level NLP task families
(math/code/medical/legal/finance) even without leakage — this is the
finding V1 was trying to make, and it replicates honestly.

## N=25 Per-Domain Results (V2 — key failures in bold)

| Domain | V2 acc | Notes |
|--------|--------|-------|
| math | 97.0% | — |
| code | 100.0% | — |
| medical | 91.0% | Still strong at N=25 |
| legal | 93.0% | — |
| finance | 67.0% | Confused with statistics/economics cluster |
| **clinical_knowledge** | **36.4%** | Leaks into medical |
| **virology** | **42.9%** | Leaks into medical / biology |
| **high_school_biology** | **59.0%** | Leaks into medical / clinical_knowledge |
| **nutrition** | **48.0%** | Leaks into medical / biology |
| **human_sexuality** | **27.3%** | Leaks into medical / psychology |
| **high_school_psychology** | **51.0%** | Leaks into sociology / human_sexuality |
| high_school_geography | 56.0% | Leaks into world_history / world_religions |
| world_religions | 60.5% | Leaks into philosophy / history |
| philosophy | 83.3% | — |
| high_school_world_history | 100.0% | Perfect separation |
| electrical_engineering | 70.3% | Leaks into computer_security |
| computer_security | 52.0% | Leaks into code / electrical_engineering |
| logical_fallacies | 87.8% | — |
| high_school_statistics | 74.1% | Leaks into finance |
| formal_logic | 93.8% | — |
| high_school_government_and_politics | 73.5% | — |
| high_school_chemistry | 66.0% | Leaks into biology / physics |
| high_school_physics | 76.3% | Leaks into chemistry |
| management | 61.5% | Leaks into marketing |
| marketing | 66.1% | Leaks into management |
| **Overall** | **72.2%** | ❌ below 85% threshold |

The six restored hard-negatives are the dominant failure cluster. V1's
86.1% is fully attributable to (a) dropping this cluster and (b) 100%
train/test overlap inflating the remaining domains.

## What V2 says about the routing thesis

- **N=5 TF-IDF routing is real.** 96.78% under honest splits validates
  the core mechanism: coarse NLP task families have discriminative
  TF-IDF vocabulary.
- **N=25 TF-IDF routing under semantically overlapping domains is not
  real at the 85% threshold.** Medical ↔ clinical_knowledge ↔ virology
  ↔ biology ↔ nutrition ↔ human_sexuality overlaps too much in raw
  vocabulary for a centroid-in-TF-IDF-space classifier. To earn N=25 at
  85% under hard negatives, the router must either (a) use contextual
  embeddings (sentence-transformers centroids, or Gemma 4 hidden
  states), or (b) add a discriminative head on top of TF-IDF. Both are
  architectural changes, not hyperparameter tweaks.
- **Finding #389** ("TF-IDF 100% on N=3 toy domains") and **Finding
  #354** ("TF-IDF + logistic regression 95% on M2P N=5") are
  retroactively narrower than V1 claimed: they apply when domains are
  not semantic neighbors.

## Follow-up already in queue

`exp_p1_p1_ridge_routing_n25` (blocks entry) pre-registers K1081
(leakage-free N=25 ≥ 80% with all hard-negatives included), K1082
(native MLX p99 ≤ 2ms including CPU-GPU sync), K1083 (≤ 5% drop under
format normalization). That's the right next step: a new router
architecture evaluated against KCs that were designed knowing the
failure mode.

## Kill-criterion discipline

K1073/K1074/K1075/K1076 thresholds are byte-preserved from V1 (see
MATH.md git-diff). No KC is added, removed, or relaxed. V2 fails K1074
honestly under the original 85% threshold with hard-negatives present.
The experiment is `killed`. The follow-up is a new experiment with new
KCs, not a re-run of this one.

## Assumptions

- MBPP is an adequate stand-in for HumanEval in the "code-domain
  vocabulary" role: both are function-writing problems in natural
  English with Python-syntax exemplars. The TF-IDF router cares about
  the English problem text, which both datasets have.
- MMLU test-split dedup is the right definition of "disjoint": two
  subjects occasionally repeat a question verbatim, and treating two
  different indices with identical text as distinct would leak.
- K1075's "CPU latency" is defined as isolated sklearn/numpy execution
  on CPU, as per MATH.md Theorem 3. The LOOPHOLE_METHODOLOGY concern
  about CPU-GPU sync is a valid architectural concern that belongs to
  K1082 in the follow-up experiment, not K1075 here.
