# Learnings: exp_bitnet_eigenlorax_subspace

## Core Finding

Grassmannian-packed orthogonal LoRA adapters are fundamentally incompatible with EigenLoRAx subspace extraction: the same property that enables interference-free composition (|cos| ~ 0.001) prevents shared principal subspace formation, yielding +80.8% PPL gap vs from-scratch training. This is an architectural tradeoff, not a tuning failure.

## Why This Happened (Literature-Grounded)

The EigenLoRAx algorithm (Ramezani et al., 2025, arXiv 2502.04700) relies on the **universality hypothesis** (Chughtai et al., 2023; Kwon et al., 2024) — that neural network weight updates reside in shared, structured low-dimensional subspaces. Standard LoRA validates this: randomly initialized A matrices are correlated (same seed family, similar loss landscapes), so SVD concentrates variance in few principal components.

Our architecture deliberately violates this assumption. Grassmannian packing (Alternating Projection on Gr(r, d)) maximally spreads A matrices across the full parameter space, producing **near-uniform singular values** in the stacked adapter matrix. The observed A-space variance at K=16 (31.3%) is worse than random-orthogonal (predicted 66.7%), confirming that Grassmannian packing actively decorrelates beyond random initialization.

The mechanism is clean:
1. **A-space (input selection)**: Frozen, orthogonal by construction → no shared subspace → 31.3% variance at K=16
2. **B-space (output projection)**: Rank-16, N=25 > 16 → trivially captured at 100% (mathematical guarantee, not learning)
3. **Combined**: B-space recovery is useless without A-space recovery, because the adapter's domain specificity lives in which hidden dimensions are attended to (A), not what is added (B)

This is the same root cause as the spectral surgery kill (exp_bitnet_spectral_surgery): short-trained adapters with Grassmannian A produce clean, efficient spectra with no inter-adapter redundancy. Both SVD-based manipulation (spectral surgery) and SVD-based extraction (EigenLoRAx) fail because the adapter spectra are already optimally structured.

## Confirming Evidence

1. **EigenLoRAx paper itself** (arXiv 2502.04700): Explicitly states that if a new task's solution is orthogonal to all principal components, the task is "completely unrelated" and subspace training cannot converge. Our Grassmannian design creates exactly this condition by construction.

2. **OSRM — Orthogonal Subspaces for Robust Model Merging** (ACL 2025, arXiv 2505.22934): Constrains LoRA subspaces before training to prevent interference. Confirms that orthogonal subspace separation is beneficial for composition but acknowledges a tradeoff: stronger orthogonality means less shared structure. Our Grassmannian skeleton is the extreme endpoint of this tradeoff.

3. **Spectral Surgery kill** (exp_bitnet_spectral_surgery, this project): Short-trained rank-16 adapters already have efficient spectra — no room for spectral refinement. Same root cause: Grassmannian + short training = clean, independent adapters.

4. **LoRI B-sparsity kill** (exp_bitnet_lori_sparse_b, this project): B-sparsity is neutral on ternary base because the interference problem it solves (FP16 |cos|~0.142) doesn't exist on BitNet-2B (|cos|~0.001). Another confirmation that Grassmannian orthogonality eliminates the cross-adapter coupling that other methods assume exists.

## Contradicting Evidence

1. **"Rethinking Inter-LoRA Orthogonality in Adapter Merging"** (arXiv 2510.03262): Finds that strict orthogonality alone does not yield semantic disentanglement. This doesn't contradict our result (we're not claiming semantic disentanglement), but it suggests that even if we could extract a subspace, the semantic meaning of principal components might not align with task boundaries. The implication: subspace methods may be fundamentally limited for multi-task adapter systems regardless of initialization strategy.

2. **"Efficient Modular Learning through Naive LoRA Summation"** (arXiv 2508.11985): Demonstrates that high-dimensional models can naively sum LoRA adapters when they are orthogonal. This partially contradicts the pessimistic reading of our result — orthogonality prevents subspace extraction but ENABLES naive composition. The architecture is working as designed; the "failure" is only from the perspective of subspace transfer.

3. **VeRA** (arXiv 2310.11454, ICLR 2024): Uses frozen random matrices shared across layers with trainable scaling vectors. VeRA's random matrices are NOT orthogonally packed — they're shared, creating correlation. VeRA achieves comparable performance to LoRA with ~50% fewer params. This suggests that shared structure (even random) is useful for parameter efficiency, but our architecture deliberately sacrifices it for composition stability.

## Alternative Approaches (What We Could Try Instead)

1. **B-matrix mean initialization** (from PAPER.md suggestion): Freeze new Grassmannian A, initialize B from the B-subspace mean of existing adapters. Modest speedup expected (B-space is output-space structure only). Low expected payoff but cheap to test.

2. **Hypernetwork-generated adapters** (Zhyper, arXiv 2510.19733; HypeLoRA, arXiv 2603.19278; Text-to-LoRA): A hypernetwork could learn to generate B matrices conditioned on task description, while A remains Grassmannian-frozen. This decouples adapter creation from subspace extraction — the hypernetwork IS the shared knowledge, not a principal subspace. Zhyper achieves 26x fewer params than baselines. Higher complexity but architecturally compatible with our design.

3. **Meta-learned B initialization** (MAML/Reptile on B-space): Learn a B initialization that is K-step-optimal across domains, while keeping A frozen+orthogonal. Already partially addressed by exp_bitnet_meta_scaffold (P3, OPEN). Literature supports this: Meta-LoRA applies meta-learning to learn shared LoRA priors.

4. **Retrain-from-scratch with data reuse** (exp_bitnet_retrain_evolve): Accept that each adapter trains independently. Focus on making individual training faster/better rather than transferring cross-adapter structure. The Evolve track already converges here — quality gate via PPL + KR-Test.

5. **Subspace augmentation with pseudo-PCs** (from EigenLoRAx paper §4.3): Append Gram-Schmidt-orthogonalized random vectors to the subspace. This won't help: the problem isn't subspace incompleteness (too few PCs), it's subspace irrelevance (A-space is orthogonal by design, not by accident).

## Implications for Next Experiments

1. **Evolve track is confirmed retrain-from-scratch**: No shortcut via subspace transfer. Combined with spectral surgery kill and LoRI null result, three independent experiments now confirm that cross-adapter manipulation/transfer doesn't work on Grassmannian-packed ternary adapters. This is a closed design question.

2. **The orthogonality-transfer tradeoff is fundamental**: Any future proposal for cross-adapter knowledge sharing must be evaluated against this tradeoff. Methods that assume shared structure (EigenLoRAx, spectral surgery, LoRA Soups-style merging) are architecturally incompatible. Methods that DON'T assume shared structure (hypernetworks, meta-learning on B, retrain-from-scratch) are compatible.

3. **B-matrix is the only shared signal**: All 25 adapters' B matrices are perfectly recoverable from K=16 PCs. If any cross-adapter transfer mechanism exists, it must operate in B-space only, with A provided independently (Grassmannian). The hypernetwork approach (generate B from task description) aligns with this.

4. **Update HYPOTHESES.yml**: Mark exp_bitnet_eigenlorax_subspace as killed with note linking to spectral surgery and LoRI kills as a converging evidence pattern. Consider adding a "meta-finding" node capturing the orthogonality-transfer tradeoff as a proven architectural constraint.

## New References to Add

| Paper | arXiv | Relevance |
|-------|-------|-----------|
| Unraveling LoRA Interference: Orthogonal Subspaces for Robust Model Merging | 2505.22934 | ACL 2025. Orthogonal subspace constraints for LoRA merging. Comparison point for Grassmannian approach. |
| Rethinking Inter-LoRA Orthogonality in Adapter Merging | 2510.03262 | Challenges assumption that orthogonality = semantic disentanglement. |
| Efficient Modular Learning through Naive LoRA Summation | 2508.11985 | Orthogonality enables naive composition in high-dim — confirms our design rationale. |
| Zhyper: Factorized Hypernetworks for Conditioned LLM Fine-Tuning | 2510.19733 | Hypernetwork generates LoRA adapters from task descriptions. Alternative to subspace transfer. |
| HypeLoRA: Hyper-Network-Generated LoRA Adapters | 2603.19278 | Hypernetwork LoRA with calibration. Frozen A variant noted. |

Note: OSRM (2505.22934) may already be in references as "orthogonal-subspace-merging". Verify before adding duplicate.
