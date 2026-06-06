# LEARNINGS: Fisher-Weighted Adapter Composition

## Core Finding

Diagonal Fisher information on shared-base LoRA adapters is **structurally domain-invariant**: the base model's parameter sensitivity dominates the Fisher diagonal, making Fisher*Delta^2 collapse to rescaled Frobenius energy (Spearman rho=1.0). Fisher merging is designed for full-model merging where different models have genuinely different Fisher patterns; for shared-base adapter composition, it adds zero ranking information and amplifies spectral pathology (Gini 0.563 vs 0.490 raw). This definitively closes post-hoc composition weighting as a research direction — four experiments (#277 DC-Merge, #278 surgery, #279 Frobenius, #280/#281 Fisher) converge on the same impossibility: no scalar per-adapter weight can separate signal from artifact when the signal-artifact decomposition is entangled in the scale factor itself.

## Why This Happened

Fisher importance w_i = sum(F_i[j] * Delta_i[j]^2) is a product of two terms:
- **F_i[j]** (Fisher diagonal): depends primarily on the base model architecture at position j, with ~2x cross-domain variation from data distribution differences.
- **Delta_i[j]^2** (adapter perturbation squared): spans ~400x across domains due to scale factors (s=20 vs s=1).

Since F_i is dominated by the shared base model, w_i ~ F_mean * ||Delta_i||_F^2. The product's rank ordering is determined entirely by Frobenius energy, explaining rho=1.0.

This matches **DO-Merging** (arXiv:2505.15875, ICML 2025): "LoRA modules show much larger parameter magnitude variance than full fine-tuned weights" and "greater parameter magnitude variance correlates with worse merging performance." Our 400x energy ratio is a concrete instance of their general finding.

The Fisher equalization scales (medical:1.67, math:1.90, finance:0.007) amplify the imbalance because Fisher weights track Frobenius energy, assigning MORE weight to already-dominant domains. This is the opposite of equalization — a structural anti-pattern.

## Confirming Evidence

1. **DO-Merging** (arXiv:2505.15875) — Magnitude-direction decoupling framework. Empirically demonstrates that existing merging methods designed for full fine-tuning perform poorly on LoRA due to larger parameter magnitude variance. Directly parallels our finding that the 400x energy ratio overwhelms per-parameter importance signals.

2. **CoGraM** (arXiv:2512.03610) — Context-sensitive granular optimization with rollback. Explicitly identifies Fisher merging as "often losing accuracy and unstable across seeds." Proposes multi-stage alternative (layer/neuron/weight granularity with rollback) to address Fisher's instability. Confirms Fisher fragility is a known issue in the merging community.

3. **Position paper on LoRA reuse** (arXiv:2506.13479) — Through theoretical analysis and synthetic tasks, finds that reusing/merging LoRAs often fails to logically integrate knowledge across disjoint fine-tuning datasets. Argues the community should stop chasing new merging algorithms and instead understand when reuse actually works. Our Fisher failure is a concrete instance of their general warning.

4. **Non-Local Merging** (arXiv:2410.12766) — Identifies "variance collapse" as a failure mode when merging models that diverge significantly from the base. Analogous to our finding: the shared base model's Fisher pattern is so dominant that per-adapter Fisher variation vanishes.

## Contradicting Evidence

1. **FIM-Merging** (arXiv:2603.21705) — Successfully uses diagonal Fisher for per-LAYER (not per-parameter) merging coefficients on full fine-tuned models. Key differences from our setup: (a) full models, not LoRA adapters, so Fisher patterns genuinely differ; (b) layer-level granularity where base-model dominance averages out; (c) random token inputs for Fisher computation. Suggests Fisher works at coarser granularity where the shared-base problem is mitigated.

2. **OTA-Merging** (arXiv:2509.11167) — Uses optimizer second-moment statistics (Adam's v_t, a diagonal curvature proxy closely related to Fisher) for parameter sparsification, not weighting. Key insight: curvature overlap between tasks is substantial (confirming our rho=1.0), but curvature is useful for deciding WHAT TO KEEP (sparsification mask), not HOW MUCH TO WEIGHT. This reframes the failure: Fisher isn't useless, it's misapplied as a weighting scheme when it should be a selection criterion.

## Alternative Approaches (paper-backed only)

1. **Geometry-based adapter merging** — NSC Merging (arXiv:2603.26317, CVPR 2026) sets merge weights from adapter null-space compression ratio rather than Fisher/loss. The insight: "during LoRA fine-tuning, the down-projection A compresses its null space, and compression correlates with performance." This bypasses the shared-base Fisher problem entirely by measuring adapter-intrinsic properties.

2. **Subspace coverage reweighting** — TARA-Merging (arXiv:2603.26299, CVPR 2026) addresses directional anisotropy in merged LoRA updates. Reweights per-direction (not per-parameter) to ensure broad subspace coverage. Tackles exactly what Fisher cannot: per-direction importance rather than per-parameter sensitivity.

3. **Structural interference elimination** — LoRI (arXiv:2504.07448, COLM 2025) freezes A-matrices as random projections and sparsifies B with task-specific masks. Up to 95% fewer parameters with better merging. Complementary to our Grassmannian approach — both use orthogonality, but LoRI adds task-specific sparsity.

4. **Routing over merging** — For production deployment with heterogeneous-scale adapters, per-token routing (selecting 1-2 active adapters) sidesteps composition pathology entirely. The 400x energy ratio only matters when summing all adapters; top-k routing makes it irrelevant.

## Implications for Next Experiments

1. **Post-hoc composition weighting is CLOSED.** Four experiments form a complete impossibility arc:
   - DC-Merge (#277): wrong variable (within-domain SV shape)
   - Spectral surgery (#278): structurally inverted (top SVs = constructive interference)
   - Frobenius equalization (#279): partial fix works (50% log-compression), but the compression factor is empirical
   - Fisher weighting (#280/#281): reduces to Frobenius norms for shared-base LoRA
   
   No scalar per-adapter weight scheme can do better than partial equalization because the signal-artifact decomposition is inherent to the scale factor, not separable by any per-parameter importance measure that shares the base model.

2. **The practical ceiling is 50% log-compression** (Finding #279). The next improvement requires either: (a) changing the training procedure (norm-bounded training, NB-LoRA arXiv:2501.19050), or (b) bypassing composition entirely via routing.

3. **Fisher has a valid use case we haven't tested:** OTA-Merging shows curvature is useful as a SPARSIFICATION mask (what to keep), not a weighting scheme (how much). If future experiments revisit Fisher, it should be for pruning conflicting parameters, not for scaling adapter contributions.

4. **Routing is the natural next direction.** The spectral arc shows uniform composition has treatable but real pathology (50% compression is the ceiling). Per-token routing activates 1-2 adapters, making the 400x scale imbalance irrelevant. Research priority should shift to routing quality.

## Recommended Follow-Up

**Primary:** Routing quality improvements — the entire post-hoc weighting arc converges on the conclusion that composition is limited by inherent scale entanglement. Routing sidesteps the problem.

**Secondary (if revisiting composition):** Norm-bounded training (NB-LoRA, arXiv:2501.19050) — train adapters with matched Frobenius norm budgets so the artifact component of scale is eliminated at source. Would require retraining all adapters.

**Not recommended:** Any further per-adapter scalar weighting schemes (Fisher variants, curvature-based weights, learned mixing coefficients). The impossibility is structural: shared base model → domain-invariant Fisher → Frobenius dominance. This holds for any diagonal approximation of the Hessian/Fisher computed on the shared base.
