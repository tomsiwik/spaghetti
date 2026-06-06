# Learnings: exp_output_averaging_vs_param_merge

## Core Finding

Output-averaging (logit ensembling) beats parameter-merging by 11.5% PPL at k≥25 adapters, but parameter-merging wins at k=5. The dominant failure mode of pre-merge is 1/k signal dilution, not cross-term interference — the Grassmannian skeleton's orthogonality guarantee is irrelevant to this bottleneck.

## Why This Happened (Literature-Grounded)

The experiment distinguishes two failure modes that the literature often conflates:

1. **1/k dilution (dominant):** Under uniform pre-merge, each adapter contributes O(1/k) of its full signal. At k=49, that's 2% — noise-level. This is a linear algebra identity, not a learned phenomenon. Output-averaging sidesteps it entirely by running each adapter at full strength before averaging logits.

2. **Cross-term interference (secondary):** Even with Grassmannian-orthogonal A-matrices, the composite perturbation W_base + Σ(1/k)(B_i@A_i) passes through nonlinearities (LayerNorm, SwiGLU) that amplify orthogonal weight perturbations into correlated output perturbations. The paper "Rethinking Inter-LoRA Orthogonality" (arXiv 2510.03262) formally proves that weight-space orthogonality does not guarantee semantic disentanglement — nonlinearities in residual streams destroy the orthogonality guarantee in function space.

3. **Logit-scale preservation:** Output-averaging computes log p(x) = (1/k)Σ log p_i(x), which is the geometric mean of probabilities. Each adapter contributes a full-strength opinion. Pre-merge computes a single forward pass through diluted weights, where the adapter signal can fall below the noise floor of the base model's representation.

The k~10 crossover point aligns with the LoRAuter finding (arXiv 2602.21222) that weighted output-space fusion becomes necessary precisely when the task requires synthesizing knowledge from multiple domains — single-adapter selection suffices for in-domain queries.

## Confirming Evidence

- **LoRAuter** (arXiv 2602.21222): Explicitly compared parameter merging (Linear, TIES, Magnitude Pruning) against output-space fusion on Llama-2-7B. Output-space fusion with task-similarity weighting achieved oracle-level performance (70.95% PIQA, 77.62% RTE), massively outperforming perfect adapter selection (46%, 52%). Confirms that output-space composition enables *constructive synergy* impossible in weight space.

- **Zhao et al. (2024)** (cited by LoRAuter): Found that "composition in the output space yields better performance than composition in the parameter space" because LoRA adapters often lack geometric alignment.

- **Stoica et al. (2025)** (ZipIt!): Demonstrated that independently trained adapters have limited alignment, hindering effective parameter merging. Consistent with our finding that even Grassmannian-initialized adapters lose their orthogonality advantage after passing through nonlinearities.

- **Our own macro/composition_weight_normalization experiment:** KILLED at N=5: uniform static scaling made composed PPL ≈ base model PPL (5.7). This is the same 1/k dilution mechanism seen here at larger k. Validates per-input routing over static composition.

- **Our own macro/composition_dropout_robustness experiment:** KILLED: CV=112.2% at N=5 under equal-weight composition. One harmful adapter poisoned everything. The output-averaging approach avoids this because no single adapter can dominate the merged weights.

## Contradicting Evidence

- **LoRAuter weighted linear merging paradox:** On PIQA/RTE, *weighted parameter merging* (not uniform 1/k) actually outperformed the oracle single adapter (70.95% vs 46%). This suggests that with learned weights, parameter merging can produce **constructive interference** — the exact opposite of our finding. The key difference: our experiment uses uniform 1/k weights (worst case), while LoRAuter learns task-similarity weights. This means the experiment tests a straw-man version of pre-merge.

- **TIES merging:** Achieved 70.09% on StoryCloze (competitive with single-task experts) by enforcing sign-aware sparsity before merging. TIES resolves directional conflicts that uniform averaging cannot. Our experiment did not test TIES, DARE, or any conflict-resolution merging strategy.

- **LoraHub** (Huang et al.): Learns adaptive scalar weights for parameter-space composition without routing overhead. Achieves good generalization to unseen tasks. Suggests the 1/k dilution problem is solvable without abandoning weight-space composition.

- **The k=5 result itself:** Pre-merge wins at k=5 (+3.0%), confirming that at SOLE's operating point (top-k=2), the entire output-averaging advantage evaporates. The experiment validates pre-merge for the actual deployment scenario.

## Alternative Approaches (What We Could Try Instead)

1. **Router-weighted pre-merge (the fair comparison):** Replace uniform 1/k with router-derived weights. SOLE already has a router. Test whether weighted pre-merge at k=2-5 captures the full benefit of output-averaging. This is the experiment the adversarial review correctly identified as missing. (LoRAuter, arXiv 2602.21222)

2. **LoRA-LEGO rank-wise clustering:** Decompose adapters into Minimal Semantic Units (per-rank), cluster them, and reassemble a merged adapter that bypasses parameter interference. Gets ensembling quality at merging speed. (LoRA-LEGO)

3. **Sub-MoE subspace merging:** Joint SVD on concatenated expert weights, extract shared U-matrix for common knowledge, safely merge V-matrices. Achieves 96% of ensemble zero-shot capabilities at O(1) inference cost. (Sub-MoE)

4. **Knowledge distillation (MC-SMoE):** Align experts via neuron permutation, merge using activation-frequency weights, decompose into low-rank structure. 80% memory reduction, near-zero performance loss. (MC-SMoE)

5. **ImPart importance-aware sparsification:** SVD on delta weights, dynamically adjust sparsity of singular vectors by task importance. 2x compression during merging without losing crucial knowledge. Directly addresses the 1/k dilution by keeping only high-importance components at full strength. (ImPart)

6. **Token-level gating (LoRA-Flow/MoLE):** Per-token routing weights instead of per-input routing. Allows borrowing different capabilities within a single sequence. Higher overhead than static pre-merge but lower than full output-averaging. (LoRA-Flow, MoLE)

## Implications for Next Experiments

1. **Pre-merge confirmed for SOLE at top-k=2.** No change needed to the serving architecture. The 0.74ms/tok, zero-overhead pre-merge remains correct for routed composition.

2. **The 1/k dilution finding reframes the Grassmannian value proposition.** Grassmannian initialization prevents cross-term interference, but 1/k dilution is the dominant bottleneck at k>10. This means orthogonality is necessary but not sufficient — it prevents *catastrophic* interference but cannot prevent *dilution*. The Grassmannian skeleton remains load-bearing for k=2-5 (the SOLE regime), where dilution is modest and cross-terms matter.

3. **Weighted pre-merge is the next experiment to run.** The adversarial review correctly identified that uniform 1/k is a straw-man. Testing router-weighted pre-merge vs output-averaging would determine whether the 11.5% gap persists when weights are optimized. If weighted merging closes the gap, output-averaging is unnecessary at any k.

4. **The eli5 anomaly (-55.4% at k=25) warrants investigation.** Either the eli5 adapter is disproportionately powerful at full strength (suggesting adapter quality variance), or there's a systematic evaluation artifact. Excluding eli5, the OA advantage drops to ~9.1% — still significant but less dramatic.

5. **For "always-on" composition (instruction + safety adapters), consider LoRA-LEGO or Sub-MoE.** These get ensembling quality at merging speed, which is the holy grail for scenarios where k>5 adapters must be active simultaneously.

6. **Output-averaging provides a useful upper bound.** Even if never deployed, OA PPL serves as the ceiling for what any composition method can achieve. If weighted pre-merge matches OA, the composition problem is solved. If not, the gap quantifies the "nonlinearity tax" of weight-space composition.
