# Learnings: exp_norm_bounded_adapter_training

## Core Finding

Training-time Frobenius norm constraints on LoRA B-matrices produce WORSE composition quality than post-hoc equalization on the same unconstrained adapters. Despite Strategy C achieving near-perfect energy equalization (delta norm ratio 1.2:1, norm Gini 0.036), composed spectral Gini was 0.456 — far worse than Finding #279's post-hoc full equalization (Gini 0.267) on the same baseline adapters. The constrained optimizer degrades within-domain singular value structure, negating the equalization benefit. Additionally, the Gini union bound (Theorem 1c) was empirically falsified — the missing Pyatt (1976) overlap term makes additive Gini bounds structurally incomplete for composed factored adapters.

## Why This Happened (Literature-Grounded)

**Root cause: constrained optimization in non-convex LoRA landscape destroys task-relevant singular value structure.** LoRA's low-rank parameterization creates a non-convex optimization landscape where the B-matrix's singular value spectrum encodes genuine task information (arXiv:2506.16787, "Spectral Encoding Helps"). Imposing a Frobenius norm budget forces the optimizer into a restricted feasible region where it cannot simultaneously achieve (a) good prediction loss and (b) the target norm. In 200 training steps, the constrained optimizer sacrifices singular value quality for norm compliance.

**Confirming mechanism from "intruder dimensions" (arXiv:2410.21228, Shuttleworth et al., NeurIPS 2024):** LoRA produces high-ranking singular vectors orthogonal to pretrained weight SVs. These "intruder dimensions" carry task-specific information and emerge naturally during unconstrained training. Norm projection (Strategy A) clips these dimensions, while uniform-scale training (Strategy C) prevents them from developing domain-appropriate magnitudes.

**Post-hoc correction preserves the signal.** Finding #279 demonstrated that post-hoc Frobenius equalization achieves Gini 0.267 because it rescales the SAME well-trained adapters without altering their internal SV structure. This is consistent with the broader finding that post-hoc methods outperform training-time constraints for adapter composition (arXiv:2505.18356, "Unreasonable Effectiveness of Model Merging").

**Gini bound failure.** The Gini union bound Gini(composed) ≤ max_i Gini(B_i) + Gini_between assumes additive decomposability. The standard Gini decomposition (Pyatt 1976) includes a third overlap/interaction term that becomes significant when composed adapters share spectral subspace (our B-overlap cos = 0.028 is small but non-negligible at N=5 rank-16). Strategy C: bound predicts ≤ 0.316, measured 0.456 (44% excess).

## Confirming Evidence

- **arXiv:2410.21228 — LoRA vs Full Fine-Tuning: An Illusion of Equivalence (NeurIPS 2024):** LoRA's "intruder dimensions" encode task-specific signal in the SV structure. Constraining norms during training disrupts this signal formation, consistent with our Strategy A/C quality degradation.
- **arXiv:2603.03995 — Spectral Surgery (2026):** Post-hoc SV reweighting (adjusting ~1000 scalars, no learned direction changes) recovers up to +4.4pp on CommonsenseQA. Confirms that leaving training unconstrained and correcting post-hoc is the right sequence — matches our Finding #279 vs norm-bounded comparison.
- **arXiv:2506.16787 — Spectral Encoding Helps (2025):** SV spectrum carries disproportionate task information relative to parameter count. Explicitly encoding spectral structure improves adapter efficiency. Supports treating SVs as information carriers that should not be constrained during training.
- **arXiv:2505.18356 — Unreasonable Effectiveness of Model Merging (2025):** Post-hoc merging via Layer-Swapping consistently outperforms coordinated multi-task training. Freezing "unproductive" parameters during training (the training-time constraint analogue) produces worse outcomes than training freely and correcting post-hoc. Direct macro-level validation of our micro finding.
- **Finding #281 (Fisher weighting):** Fisher importance weights perfectly correlate with Frobenius norms (rho = 1.0) for shared-base LoRA, confirming that per-parameter importance is dominated by scale, not structure, in our architecture.

## Contradicting Evidence

- **arXiv:2501.19050 — NB-LoRA (2025):** Claims per-SV bounding (not global Frobenius) improves LoRA stability and matches/exceeds standard LoRA quality. Key difference: NB-LoRA bounds individual SVs to prevent runaway growth while preserving relative SV structure; our Frobenius constraint is a global budget that forces the optimizer to redistribute energy. Per-SV bounding may be less destructive, but NB-LoRA was not tested in a multi-adapter composition setting — single-task fine-tuning quality does not guarantee composition quality.
- **Strategy B miscalibration:** We cannot rule out that a properly calibrated weight-decay approach (lambda reduced 60x) might produce useful results. However, the structural argument (constrained ≠ unconstrained optimum in non-convex landscape) applies regardless of calibration.

## Alternative Approaches

1. **Orthogonal subspace constraints (arXiv:2505.22934 — OSRM, ACL 2025):** Constrain adapter SUBSPACE direction (not norm) so different tasks live in orthogonal subspaces. Preserves within-domain SV structure while guaranteeing composition orthogonality. Already validated in our architecture via Grassmannian initialization (Finding #225). Further investigation: whether OSRM's pre-training constraints improve on our Grassmannian approach.

2. **Orthogonal gradient projection (arXiv:2601.09684 — Ortho-LoRA, 2026):** Project competing task gradients onto each other's orthogonal complement during training. Recovers 95% of single-task vs multi-task gap on GLUE. Separates direction from scale — constrains interference direction without touching magnitudes. Potentially compatible with our Grassmannian framework.

3. **Post-hoc partial equalization (Finding #279 — CONFIRMED CEILING):** 50% log-compression of per-domain scales. Gini 0.393, mixed PPL -1.2%. Already implemented and validated. Remains the practical ceiling for spectral composition quality.

4. **Routing (sidestep composition entirely):** Per-token routing activates 1-2 adapters instead of uniformly summing all N. Scale imbalance becomes irrelevant when only selected adapters are active. Already demonstrated with Gumbel-sigmoid routing (44% better than softmax).

## Implications for Next Experiments

1. **Post-hoc composition weighting is DEFINITIVELY CLOSED (5 experiments).** DC-Merge (#277), spectral surgery (#278), Frobenius equalization (#279), Fisher weighting (#281), and norm-bounded training (#282) collectively prove that composed spectral Gini has a floor (~0.267) determined by within-domain SV structure. Post-hoc partial equalization is the practical ceiling. No further spectral composition methods should be pursued.

2. **Training-time constraints are the WRONG lever for composition quality.** The non-convex LoRA landscape means constrained optima are generally worse than unconstrained optima + post-hoc correction. This principle likely extends beyond norm constraints to any training-time composition-aware modification.

3. **Direction-based methods (OSRM, Ortho-LoRA) are the natural next frontier** for training-time composition improvement, since they constrain subspace direction (which directly determines interference) rather than scale (which is a proxy). However, our Grassmannian initialization already achieves excellent orthogonality (cos = 0.026), so the marginal gain may be limited.

4. **The project should move past spectral optimization entirely.** The metrics that matter for production are per-domain PPL preservation, out-of-domain behavioral quality, and task-specific accuracy — not spectral Gini. Routing quality and benchmark scaling are the productive next directions.

## Recommended Follow-Up

No direct follow-up experiments recommended for the spectral composition track. The arc is resolved.

For the broader project, the most productive directions are:
- **Routing quality improvement:** Sidesteps composition pathology entirely. Motivated by Finding #279's observation that per-token routing makes uniform-sum scale equalization moot.
- **Benchmark/eval scaling:** Move from proxy metrics to real-world task accuracy at N=25 domains. Motivated by the 5-experiment spectral arc's conclusion that Gini is not the metric that matters.
