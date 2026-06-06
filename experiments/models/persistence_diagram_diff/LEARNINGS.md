# LEARNINGS: exp_persistence_diagram_diff

## Core Finding

Adapter composition via pre-merge is topologically nontrivial but preserves all existing features at current scale (rank-16, 5 domains). The stability bound is vacuously satisfied (10-100x loose). The genuine discovery is feature CREATION: +242 H0 connected components and +401 H1 loops, concentrated in output-facing projections (o_proj, down_proj). Adapters restructure weight geometry rather than merely perturbing it, and this restructuring is statistically distinguishable from random perturbation (adapter/random ratio 1.38, Wilcoxon p=0.0002).

## Why This Happened

The stability theorem (Cohen-Steiner et al. 2007) guarantees d_B <= max||delta_i||. At current adapter scale, max||delta_i|| ~ 0.3-2.0 while median feature persistence ~ 30-50, making the bound trivially satisfied. No features CAN be lost at this perturbation magnitude.

Feature creation occurs because:
1. **Near-identical rows differentiate.** Base model output projections (o_proj) have 383-427 near-identical rows out of 500 sampled. The adapter perturbation breaks these apart into distinct points, creating new H0 features. This is consistent with SFT adapters adding task-specific output directions.
2. **Cyclic structure emerges.** Composition creates 0->48-53 H1 loops in o_proj modules. The base model's output projection rows form a nearly linear arrangement; adapters introduce geometric patterns (cycles) that encode task-specific structure.
3. **Output projections are the action site.** o_proj and down_proj directly modify the residual stream. These modules show the most dramatic restructuring because adapters need to change the model's output distribution, not its internal representations.

The adapter/random ratio of 1.38 (p<0.001) confirms adapters have structured (non-random) geometry, consistent with the LIMA hypothesis (Finding #216: SFT adapters share format directions) — adapters move weights along specific learned directions, not randomly.

## Confirming Evidence

- **Rieck et al. 2018 (arXiv:1812.09764, ICLR 2019):** "Neural Persistence" — validated weight-space PH as meaningful for neural networks. PH of weight rows captures training progress, generalization, and architectural complexity. Our use of Rips complexes on weight rows follows their methodology.
- **Goldfarb et al. 2021 (arXiv:2106.00012):** PH captures generalization without a validation set — confirms weight topology carries meaningful structural information beyond what loss metrics reveal.
- **Subspace Geometry Governs Forgetting (arXiv:2603.02224):** Weight-space geometry (subspace overlap) predicts catastrophic forgetting. Our finding that composition preserves topology but creates features is complementary — no forgetting because perturbation is too small, but new structure is being added.
- **Finding #217 (LoRA scale is domain-dependent):** Layer 29 showing 2x topological cost aligns with our finding that task-specific output formatting concentrates in the final layer.
- **Finding #216 (0.97 inter-adapter cosine):** SFT adapters share format directions. The structured (non-random) adapter geometry we observe (ratio 1.38) is consistent with adapters encoding similar format transformations.

## Contradicting Evidence

- **Rethinking Inter-LoRA Orthogonality (arXiv:2510.03262):** Weight-space geometric orthogonality does not predict semantic disentanglement. This directly challenges the usefulness of weight-space topology as a proxy for behavioral quality. Our experiment measures topological CHANGE but cannot claim this predicts behavioral outcomes.
- **Finding #214 (contrastive training):** 99.6% weight-cosine reduction produced only 0.3-5.7% PPL differentiation. Weight geometry metrics are weak predictors of functional behavior. Our topological metrics (bottleneck distance, feature counts) may suffer the same disconnect.
- **NotebookLM finding:** Canary queries (2.0% FNR) vastly outperform geometric metrics (33.8% FNR) for detecting composition interference. This suggests behavioral probes are strictly superior to weight-space analysis for quality prediction.
- **Vacuousness at scale:** The stability bound being 10-100x loose means it provides no early warning as adapter count/rank grows. By the time features enter the vulnerability window, the perturbation is ~10x current — too late for preventive action.

## Alternative Approaches

1. **Canary query monitoring** — Runtime behavioral probes with 2.0% FNR, proven superior to geometric metrics for interference detection. Already validated in our NotebookLM literature (vs 33.8% FNR for geometric metrics).
2. **PPL-probe routing** — K+1 forward passes on probe examples to measure actual answer-conditioned perplexity. Detects logit-scale explosions dynamically rather than statically from weights.
3. **Leave-one-out PPL screening** — Measure PPL with/without each adapter to detect destructive contributions. Simple, behavioral, no topology needed.
4. **W2T: LoRA Weights Already Know What They Can Do (arXiv:2603.15990)** — Infers adapter capabilities directly from weight statistics, bypassing topology. More practical for adapter selection.
5. **ACBench / Compressed Trust** — Benchmark suites specifically designed to measure capability preservation after model modification (arXiv references in NotebookLM sources).

## Implications for Next Experiments

1. **Topology is the wrong metric for composition quality at current scale.** The stability bound is vacuously satisfied, and the literature strongly warns against using weight geometry to predict behavior (arXiv:2510.03262). Feature creation is interesting scientifically but does not help deployment.

2. **Feature creation is a research curiosity, not a deployment concern.** The +242 H0 / +401 H1 features in output projections may represent adapter specialization signatures, but connecting this to behavioral quality requires additional experiments (which are lower priority than deployment track).

3. **The pathway preservation concern is premature.** At rank-16 with 5 domains, composition is ~10x below the vulnerability threshold. This becomes relevant only at much higher adapter count or rank — a future concern, not a current blocker.

4. **Behavioral monitoring > topological monitoring.** For the deployment track, canary queries or leave-one-out PPL screening will detect interference far earlier and more reliably than weight-space PH. The P1 pathway preservation experiments should be deprioritized relative to P0 deployment work.

## Recommended Follow-Up

No immediate follow-up recommended from the topology track. The pathway preservation line has reached a natural pause point:
- exp_pathway_graph_bitnet2b: KILLED (sparsification artifact)
- exp_persistence_diagram_diff: SUPPORTED but vacuous at current scale
- Feature creation is documented but not actionable for deployment

If topology work resumes (e.g., at higher adapter scale), the right next step would be:
- **Higher-dimensional PH (H1/H2) with cosine metric** — Euclidean Rips may miss functionally relevant structure. Cosine distance is standard for transformer representations.
- **Feature creation tracking as scale increases** — Monitor whether the +242 H0 / +401 H1 pattern grows linearly or explosively with adapter count.

Priority should remain on the P0 deployment track experiments.
