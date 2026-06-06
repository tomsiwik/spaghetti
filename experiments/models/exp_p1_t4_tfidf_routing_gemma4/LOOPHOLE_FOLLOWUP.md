# Synthesized Findings & Follow-up Design (exp_p1_t4_tfidf_routing_gemma4)

## Final Verdict: Invalid

The original experiment is fundamentally compromised and its findings are entirely invalid. The claimed 96.6% (N=5) and 86.1% (N=25) accuracies are artifacts of systemic data leakage (100% train/test overlap in multiple domains) and blatant metric hacking (manual exclusion of challenging, overlapping MMLU subjects). Furthermore, the router is capturing structural formatting differences (e.g., Python code vs. multiple-choice questions) rather than true semantic domain boundaries. Finally, the sub-millisecond latency claims are empirically false because they were measured on the N=5 router instead of N=25, and the pure-Python scikit-learn implementation ignores the catastrophic CPU-GPU synchronization overhead required in the target MLX architecture.

## Follow-up Experiment Design

To determine if a zero-parameter routing mechanism can actually function within the P1 architecture, we must rebuild the experiment from scratch with rigorous dataset hygiene and native Apple Silicon execution.

### New Hypothesis
A native MLX-based centroid router can achieve >85% routing accuracy across 25 semantically distinct and overlapping domains without data leakage, while maintaining a true end-to-end p99 latency of < 2ms inclusive of any CPU-GPU synchronization.

### Mathematical & Architectural Sketch
- **Native MLX Execution:** The router must be implemented using `mx.array` operations. The vocabulary matrix and centroids must reside in unified memory, allowing the routing matrix multiplication to be part of the compiled `mx.eval` graph, eliminating GIL contention and sync overhead.
- **Strict Disjoint Splits:** All training and test sets must be rigorously separated to ensure 0% overlap between train and test splits.
- **Semantic Overlap Inclusion:** The 25 domains must explicitly include the previously excluded hard-negative MMLU subjects (e.g., `clinical_knowledge`, `virology`, `high_school_biology`) to test the router's ability to delineate semantically adjacent domains.
- **Format Normalization:** To prevent the router from cheating via format memorization, all prompts must be stripped of structural artifacts (e.g., MCQ letters like "A)", Markdown code blocks) before feature extraction.

### Kill Criteria
- **K1081 (Leakage-Free Accuracy):** N=25 routing accuracy >= 80% on strictly disjoint train/test splits that include all hard-negative semantic overlaps.
- **K1082 (Native MLX Latency):** End-to-end p99 routing latency <= 2.0ms, measured strictly on the N=25 centroid matrix using `mx.compile` in an MLX environment.
- **K1083 (Format Robustness):** Accuracy must not drop by more than 5% when evaluating structurally normalized prompts vs. raw formatted prompts.