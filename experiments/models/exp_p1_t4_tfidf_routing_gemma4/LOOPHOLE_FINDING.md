# Loophole Finding: Systemic Data Leakage, Metric Hacking, and Format Memorization

## Verdict
**Invalid**. The claimed accuracies of 96.6% (N=5) and 86.1% (N=25) are entirely spurious. The experiment suffers from catastrophic data leakage, explicit metric hacking to pass thresholds, and flawed latency measurements.

## Core Flaws

1. **100% Data Leakage (Train/Test Overlap)**:
   - `code`: HumanEval only has 164 problems in its test split. The script arbitrarily duplicates these prompts to reach 400 (`N_TRAIN`=300 + `N_TEST`=100) and splits them. The test set is a literal copy of the training set.
   - `math`: GSM8K uses `split="train"` for both the `math_train` and `math_test` subsets.
   - `mmlu`: If `auxiliary_train` is unavailable, `load_mmlu_prompts` silently falls back to `split="test"`, which is the EXACT SAME split used by `load_mmlu_test_prompts`. The model is being evaluated on its training data.

2. **Blatant Metric Hacking**:
   To artificially pass the K1074 kill criterion (N=25 accuracy >= 85%), the experiment design deliberately excluded 6 hard-negative MMLU subjects (e.g., `clinical_knowledge`, `virology`, `high_school_biology`). The authors explicitly commented that these "overlap with medical domain and would cause systematic confusion". You cannot claim a router scales to 25 domains by manually removing the negative examples that would cause it to fail.

3. **Format vs. Semantic Routing**:
   The N=5 baseline tests datasets with completely distinct structural formatting: Python code signatures (HumanEval), word problems (GSM8K), medical text structures (PubMedQA), and MCQ questions (MMLU). TF-IDF easily separates these not by understanding semantic domains, but by memorizing dataset-specific structural artifacts (e.g., Python keywords, MCQ `A)` phrasing).

4. **Flawed Latency Measurement**:
   The `K1075` latency criterion (p99 latency) is measured using the `router5` (N=5) model instead of the N=25 model. The computation cost scales with the number of centroids, meaning the reported p99 latency ignores the actual scaling overhead of N=25.

## Conclusion
The finding that this is a "viable zero-parameter router for the P1 architecture" is completely unsupported. It over-extrapolates from an artificially curated benchmark, conflates dataset memorization with domain understanding, and relies on fundamentally flawed data pipelines.