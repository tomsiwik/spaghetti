# LEARNINGS: exp_persistence_bridge_extraction

## Core Finding

Topological change from adapter composition is useful adaptation, not damage. At rank-16 with 5 domains, zero H1 features are destroyed (0/11,962), and a mathematically optimal SVD bridge correction that reduces bottleneck distance by up to 71.8% actually WORSENS PPL by 1.0% across all domains. The perturbation's top singular vectors carry the adapter learning signal itself. This definitively closes the pathway-preservation research line at current scale.

## Why This Happened

The bridge extraction approach failed on three independent levels:

1. **The premise is false.** The Algebraic Stability Theorem (Cohen-Steiner et al. 2007) provides a sufficient but not necessary condition for feature destruction: d_B <= max||delta_i||. At current adapter scale, max||delta_i|| ~ 0.3-2.0 while median feature persistence ~ 30-50, making the bound trivially satisfied. Combined with exp_persistence_diagram_diff showing 0/17,223 H0 features lost, composition is fully topologically lossless in both H0 and H1. There is literally nothing to repair.

2. **SVD of adapters IS the learning signal.** The perturbation's principal singular vectors encode task-specific adaptation, not topological damage. This is confirmed by multiple independent research lines:
   - Delta-CoMe shows adapter perturbations follow a long-tail SVD distribution where top singular vectors carry the most vital task-specific signal.
   - ImPart (delta-sparsification) proves that truncating principal singular vectors destroys knowledge, while preserving them retains task capability even at high sparsity.
   - PiSSA (arXiv:2404.02948, NeurIPS 2024) is explicitly built on the premise that adaptation relies on principal SVD components of weight matrices.
   - Task Arithmetic (Ilharco et al. 2022/2023) demonstrates that weight deltas are reusable vectors encoding task-specific behavior, not random damage.

3. **The composed perturbation is full-rank.** Mean rank for 90% energy capture is 21 for 5 rank-16 adapters, meaning each adapter contributes ~4.2 unique dimensions. A rank-16 bridge cannot capture this 21-dimensional signal, and even if it could, removing it would remove useful adaptation.

## Confirming Evidence

- **Task Arithmetic (Ilharco et al. 2022/2023):** Weight deltas between fine-tuned and base models are not damage but "task-specific behavior" vectors that can be isolated, composed, and reused. Directly supports our finding that undoing perturbations removes useful capabilities.
- **Delta-CoMe:** Long-tail SVD distribution in adapter perturbations. Top singular vectors carry vital signal. Confirms our observation that bridge correction (which removes top SVD directions) degrades quality.
- **PiSSA (arXiv:2404.02948):** Principal singular values and vectors are where adaptation happens. Removing them is removing the adaptation itself.
- **Sub-MoE (Subspace Expert Merging):** Joint SVD on expert weights separates shared knowledge (U-matrix) from expert-specific signal (V-matrices). The SVD components are functionally meaningful, not noise.
- **Finding #230 (this experiment):** 0/11,962 H1 features lost despite 19/35 modules in the vulnerability window. Stability theorem's vulnerability window is conservative.
- **exp_persistence_diagram_diff (SUPPORTED):** 0/17,223 H0 features lost. Feature CREATION (+242 H0, +401 H1) is the topological signature of useful adaptation.

## Contradicting Evidence

- **MoTE (Mixture of Ternary Experts):** Structural replacement of FFN layers with MoE experts causes "significant performance degradation" in low-precision training. Preserving original dense FFN as a frozen shared expert is necessary. This suggests that at some scale, structural disruption IS damaging -- but MoTE involves replacing entire layers, not the small perturbations from rank-16 adapters.
- **LoTA-QAF (Lossless Ternary Adaptation):** Merging adapters into quantized base requires preserving exact adapter properties; standard merging truncates adapters and degrades accuracy. This is about quantization-induced damage, not topological damage from composition.
- **LoRA-LEGO (Rank-Wise Clustering):** Standard merging destroys modular adapter nature. Preserving "Minimal Semantic Units" prevents degradation. Suggests that structure matters at the adapter level, even if weight-space topology doesn't predict behavioral outcomes.
- **arXiv:2510.03262 (Rethinking Inter-LoRA Orthogonality):** Weight-space geometric orthogonality does NOT predict semantic disentanglement. This actually supports our finding that weight-space topology is the wrong lens for composition quality.

## Alternative Approaches

For monitoring composition quality (the practical goal behind pathway preservation):

1. **Canary queries** -- Behavioral probes with 2.0% FNR vs 33.8% FNR for geometric metrics. Already validated in our architecture. The only reliable composition quality monitor at scale.
2. **Leave-one-out PPL screening** -- Isolate individual adapter impact by measuring PPL with/without each adapter. Detected a single adapter causing PPL explosion from 17K to 31.6T in our experiments.
3. **PPL-probe weighting** -- K+1 forward passes on probe examples for runtime behavioral probing. r=0.990 correlation with oracle but 6x latency cost for 5 experts.
4. **W2T: LoRA Weights Already Know What They Can Do (arXiv:2603.15990)** -- Infers adapter capabilities directly from weight statistics. More practical than topology for adapter selection.

## Implications for Next Experiments

1. **The pathway-preservation research line is closed at current scale.** Three experiments form a complete arc:
   - exp_pathway_graph_bitnet2b: KILLED (sparsification artifact)
   - exp_persistence_diagram_diff: SUPPORTED but vacuous (stability bound 10-100x loose)
   - exp_persistence_bridge_extraction: KILLED (bridge counterproductive, no features lost)

   Weight-space topology provides no actionable information for composition at rank-16/5-domains.

2. **Topological damage may emerge at higher scale.** The scaling threshold is unknown. At what adapter count, rank, or scale does the stability bound become tight? This is a future concern, not a current blocker. If it recurs, the right approach is behavioral monitoring (canary queries), not topological correction.

3. **SVD energy analysis is useful for adapter diversity.** The finding that 5 rank-16 adapters span 21 effective dimensions (each contributing ~4.2 unique directions) characterizes adapter diversity. This could inform routing decisions: adapters with high SVD overlap may be redundant.

4. **Feature creation is the interesting topology question.** The +242 H0 / +401 H1 features created during composition (from exp_persistence_diagram_diff) are the genuinely novel topological phenomenon. Understanding what these features represent (task-specific output directions? geometric signatures of domain knowledge?) is more scientifically interesting than preservation, but it is lower priority than the P0 deployment track.

## Recommended Follow-Up

No direct follow-up from the topology track. Priority should shift to the P0 deployment track:

- **exp_generation_quality_test** -- THE existential test: does routed composition produce better text? This is the next critical-path experiment.
- **exp_task_accuracy_real_benchmarks** -- MMLU/GSM8K/HumanEval with composition to prove behavioral improvement.

If topology work resumes at higher scale (>10 domains, higher rank), use behavioral monitoring (canary queries with 2.0% FNR) rather than weight-space topological analysis. The literature strongly warns against using weight geometry to predict behavior (arXiv:2510.03262).
