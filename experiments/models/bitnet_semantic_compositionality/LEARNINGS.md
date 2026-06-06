# Learnings: exp_bitnet_semantic_compositionality

## Core Finding

Weight-space orthogonality (|cos| ~ 0.001) does NOT imply data-space orthogonality (OSRM ratio 0.86, 8.6x above 0.1 threshold), yet composition still works semantically (4/5 pairs improve PPL, 90% coherent). The mechanism is constructive transfer + 1/N regularization, not orthogonal isolation.

## Why This Happened (Literature-Grounded)

The weight-vs-data orthogonality gap is a well-documented phenomenon with clear mathematical explanation:

**1. Weight orthogonality constrains A_i^T A_j ~ 0, but data orthogonality requires A_j^T h_i ~ 0 — these are independent conditions.** The "Rethinking Inter-LoRA Orthogonality" paper (arXiv:2510.03262) proved this explicitly: enforcing strict geometric orthogonality via Orthogonal Monte Carlo Dropout "does not lead to the semantic disentanglement highlighted in prior work on compositional adaptation." Our experiment extends this finding from diffusion models to ternary LLMs.

**2. Hidden state concentration explains the high cross-activation.** Our d_eff ~ 22 estimate (derived from cross-activation ratio 0.86 via sqrt(r/d_eff)) is consistent with known transformer hidden-state concentration phenomena. When hidden states occupy only ~22 effective dimensions of a 2560-dimensional space, any rank-16 projection will inevitably capture most of the variance regardless of which adapter produced it.

**3. Composition succeeds through architectural dampening, not isolation.** Our own prior micro-experiments showed multi-layer removal error is sub-additive — residual connections and LayerNorm dampen interference across depth. At rank-16 (0.02% of base params), adapter perturbations are too small to override the 2B-parameter base model's language priors. The base model acts as a massive regularizer.

**4. 1/N scaling prevents catastrophe but doesn't prevent dilution.** Our prior scale experiments (exp_bitnet_scale_n25) showed that at N=25, 1/N dilution causes all domains to revert toward base PPL, with only 1.8% average benefit. At N=5 (this experiment), 1/2 scaling is strong enough to preserve signal. The boundary is somewhere between N=5 and N=25.

## Confirming Evidence

1. **"Rethinking Inter-LoRA Orthogonality" (arXiv:2510.03262):** Structural orthogonality insufficient for functional separation. Directly confirms our K3 kill. They showed this in diffusion models; we show it in ternary LLMs.

2. **OSRM (arXiv:2505.22934):** Identifies the "previously overlooked interplay between model parameters and data distributions" and proposes constraining LoRA subspace BEFORE training as the fix. Confirms our diagnostic is measuring a real phenomenon.

3. **FlyLoRA (arXiv:2510.08396):** Uses frozen sparse random A matrices (similar to our Grassmannian skeleton) and relies on JL-lemma for approximate orthogonality. Confirms that random projections in high-d provide natural decoupling without explicit data-aware constraints.

4. **Our own exp_bitnet_instruction_tuned_task_eval:** Found activation magnitudes have r=0.023 correlation with task performance — "weight-space signals are useless" for predicting composition relevance. Trained adapters are 2-9x MORE correlated than random baselines, yet compose successfully.

5. **LoRA Soups (arXiv:2309.14621):** Achieves functional composition through linear weight combinations without orthogonality guarantees, confirming that strict orthogonality is not a prerequisite for composition. Limited to k=2 interpolation and requires extensive tuning.

## Contradicting Evidence

1. **OSRM achieves +12.78% with data-aware constraints.** While our adapters compose "well enough" without data orthogonality, OSRM shows significant improvement by constraining A_j perp Cov(h_i) before training. This suggests we're leaving performance on the table, especially for per-task accuracy (vs our composition stability focus).

2. **Our own composition catastrophe (FP16 pivot).** When adapters have large, mismatched norms in FP16, cross-activation IS destructive — PPL explodes to trillions. The ternary base's bounded activations (2x smaller post-FFN) are a critical hidden variable. Our "composition works despite high cross-activation" finding may be specific to ternary/bounded-norm architectures.

3. **N-scaling cliff.** At N=25 (exp_bitnet_scale_n25), 1/N dilution causes adapters to effectively cancel out. The "constructive transfer" we observe at N=2 is likely overwhelmed by dilution at large N. The finding is scale-dependent.

4. **Missing regularization control.** As the adversarial review noted (point 8), we didn't test single adapter at 0.5x scale vs composed pair at 0.5x each. If a single adapter at half-strength also improves PPL, the "constructive transfer" narrative would collapse into "scaling regularizes." This is an open gap.

## Alternative Approaches (What We Could Try Instead)

1. **Runtime multi-LoRA serving (no merge).** Keep adapters separate, apply dynamically per-query via PPL-probe routing or hash-ring routing. Bypasses the cross-activation problem entirely. Our Track 3 (exp_bitnet_llamacpp_serving, exp_bitnet_per_token_routing) pursues this — strongly motivated by these findings.

2. **OSRM-style data-aware initialization (arXiv:2505.22934).** Constrain A_j perp Cov(h_i) before training. Breaks plug-and-play (requires all domain data upfront) but could be applied to the Grassmannian skeleton: initialize A matrices to be orthogonal to the principal components of other domains' hidden states.

3. **MoLoRA per-token routing (arXiv:2405.12616).** Dynamic per-token adapter selection, 1.7B outperforms 4.7x larger model. Avoids static merging entirely. Directly relevant to exp_bitnet_per_token_routing.

4. **LoRI sparse-B (arXiv:2407.11299 or related).** Enforce 90% sparsity on B matrix to reduce overlap. However, our NotebookLM research warns: SOLE testing found forcing sparsity on ternary B "concentrated signal into overlapping positions" — may not help. Relevant to exp_bitnet_lori_sparse_b (proceed with caution).

5. **Effective delta cosine (vec(B@A) measurement).** Instead of measuring A-only or weight-space orthogonality, measure the composed delta directly. This is exactly what exp_bitnet_effective_delta_cosine proposes. Confirmed by this experiment as the right metric to track.

6. **ReLoRA iterative merge-and-restart.** Continual learning approach producing inherently more orthogonal experts with <0.1% composition penalty. Could be integrated into our training pipeline.

## Implications for Next Experiments

1. **exp_bitnet_effective_delta_cosine (P1) is now MORE important.** Since weight-space orthogonality is disconnected from data-space behavior, we need vec(B@A) cosine as the ground-truth metric for interference. This experiment should also test the d_eff ~ 22 prediction.

2. **exp_bitnet_lori_sparse_b (P2) should proceed with LOW expectations.** NotebookLM research suggests B-sparsity on ternary bases concentrates rather than distributes signal. Design kill criteria accordingly.

3. **exp_bitnet_per_token_routing (P2) is strongly validated.** Runtime routing completely sidesteps the orthogonality question. Combined with our d_eff ~ 22 finding (severe subspace overlap at scale), static merging is untenable for N >> 10.

4. **The composition math bug (averaging A,B separately) MUST be fixed project-wide.** The O(N^2) cross-terms from factor averaging are a confound in every composition experiment. Future experiments must use either (a) sum of B_i@A_i deltas or (b) runtime multi-LoRA. This should be a P1 bug fix before running more composition experiments.

5. **Add the regularization control to future composition experiments.** Test single adapter at 1/N scale vs N adapters at 1/N each. This disambiguates "constructive transfer" from "scaling regularization."

## New References to Add

No new papers discovered beyond what's already in REFERENCES.yml. The existing references (OSRM 2505.22934, Rethinking Inter-LoRA 2510.03262, FlyLoRA 2510.08396) are sufficient and already linked to this experiment node.
