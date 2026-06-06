# Peer Review: composition_vs_monolithic

## NotebookLM Findings

Skipped (authentication not configured). Review conducted via direct document analysis.

## Mathematical Soundness

### Parameter budget equivalence (MATH.md Section 2): Correct.

N * r * (d_in + d_out) = Nr * (d_in + d_out) is trivially true. The budget comparison is honest.

### SVD truncation as LoRA proxy (MATH.md Section 4, Assumption 1): Problematic but acknowledged.

The experiment trains full-rank and truncates post-hoc via SVD, rather than training LoRA directly in a low-rank subspace. This is a *pessimistic* proxy for the composed condition: real LoRA adapts its learned subspace during training. The MATH.md acknowledges this (Assumption 1), and the PAPER.md lists it as Limitation 2. This is reasonable self-awareness.

However, the argument cuts both ways: the monolithic condition also uses post-hoc SVD truncation to rank-20, and mono_full vs mono_trunc shows only 0.8% gap (0.948 vs 0.956). For the composed experts, the gap between full-rank experts (~1.0 average) and truncated experts (~1.64) is ~64%. This asymmetry is not a flaw in the experiment -- it correctly reveals that rank-4 at d=32 is catastrophically lossy while rank-20 at d=32 is nearly lossless. The monolithic model gets 62.5% of the available rank space (20/32), while each expert gets only 12.5% (4/32). This is the core finding.

### Signal retention formula (MATH.md Section 4): Correct.

rho_k = sqrt(sum sigma_i^2 [top r] / sum sigma_i^2 [all]) is the standard Frobenius-norm signal retention from truncated SVD. Measured values (~0.80 for 2D matrices, 1.0 for 1D) are consistent with the code.

### Quality gap decomposition (MATH.md Section 6): Sound but informal.

The decomposition into "full-rank gap" and "truncation gap" is conceptually correct. The claim that the 71% gap decomposes as ~5% full-rank + ~66% truncation is supported by the data: full-rank experts average ~1.0 vs full-rank mono ~0.95 (5% gap), and the remaining ~66% comes from rank-4 truncation. This is the strongest analytical contribution.

### Truncation error model (MATH.md Section 6, line 82): Weak.

The claim `eps_trunc_k ~ (1 - rho_k) * ||Delta_k||_F` is presented as an approximation but is not a rigorous bound. The relationship between Frobenius-norm signal loss and NTP loss increase is nonlinear and task-dependent. The paper uses this as intuition, not as a formal bound, so this is acceptable for a micro experiment, but the tilde (~) should be more clearly flagged as "empirical scaling" rather than "mathematical approximation."

### Macro extrapolation (MATH.md Section 7): Speculative.

The claim that at d=896, r=16, rho ~ 0.95+, so the gap shrinks to ~5-10%, relies on delta_rank_scaling projections. That experiment itself is under REVISE status (convergence control concerns). The extrapolation chain is: micro observation -> delta_rank_scaling power law -> macro prediction. Two links, one of which is itself uncertain. The paper does say "expected" rather than "proven," which is appropriate hedging.

### Information-theoretic argument (MATH.md Section 3): Correct but incomplete.

The argument that monolithic can allocate rank optimally across shared structure via SVD is valid. But it omits the key counter-argument: when domains arrive sequentially (the realistic SOLE scenario), monolithic *cannot* allocate jointly because it doesn't have all data simultaneously. The experiment tests this via the sequential condition, which shows catastrophic failure. The MATH.md should connect Sections 3 and 8 more explicitly: the information-theoretic advantage of monolithic is only realizable with simultaneous data access.

## Novelty Assessment

### Prior art coverage: Adequate.

The experiment cites LoRA Soups (Prabhakar et al., 2024) and InfLoRA (Liang et al., 2024). The comparison design (N composed experts vs 1 monolithic) is standard in the continual learning and multi-task learning literature. Branch-Train-Merge (Li et al., 2022) is the most directly relevant prior work for this exact experimental design and is conspicuously absent from the references. LoRAHub (Huang et al., 2023) also does cross-task LoRA composition with gradient-free optimization. Neither is cited in the PAPER.md.

### Delta over existing work: Modest.

The contribution is not the comparison itself (this is well-studied) but the specific decomposition of the quality gap into truncation vs composition components, and the argument that the gap is dominated by rank truncation at small d. This is a useful diagnostic finding.

### Reinvention risk: Low.

The experiment builds a transformer from scratch with autograd (numpy autodiff), which is the same infrastructure used in answer_conditioned_scoring. This is consistent with the project's micro-experiment approach.

## Experimental Design

### Does it test the stated hypothesis? Yes, with important caveats.

The hypothesis is: "composed domain experts with routing match or exceed monolithic multi-task fine-tuning." The experiment tests this at d=32 and finds it fails (K1 killed). The experiment correctly identifies WHY it fails and provides a plausible extrapolation for when it should pass.

### Training budget fairness: Reasonable but debatable.

Each expert trains for 15 epochs on 200 domain samples = 3,000 sample-epochs per expert.
Monolithic trains for 15 epochs on 1,000 combined samples = 15,000 sample-epochs total.

This means: composed total = 5 * 3,000 = 15,000 sample-epochs. Monolithic total = 15,000 sample-epochs. Fair.

But the monolithic sees all domains simultaneously in each epoch, allowing cross-domain gradient mixing within each parameter update. The composed model cannot. This is not a bug -- it is the fundamental tradeoff being tested. The paper correctly identifies this.

### Critical flaw: Rank ratio confound makes K1 uninterpretable.

At d=32, rank-4 captures 12.5% of the space and rank-20 captures 62.5%. This is not a "composition" test -- it is a "rank starvation" test. The monolithic model has 5x the rank per matrix. Even if composition were perfect, each expert would be severely capacity-constrained.

The paper acknowledges this extensively. The question is whether K1 should have been defined differently. A more informative kill criterion would have been: "full-rank composition loses to full-rank monolithic by >10%." The results show the full-rank gap is only ~5%, which would PASS. The experiment has the data to answer this question but frames K1 around the truncated comparison instead.

**This is the central issue.** The kill criterion K1 as stated conflates two independent effects: (a) composition overhead and (b) rank truncation loss. At d=32, (b) dominates so overwhelmingly that (a) is invisible. The experiment correctly diagnoses this in the analysis, but the kill criterion itself was poorly chosen for the micro scale.

### Missing full-rank routed condition.

The experiment computes full-rank expert losses individually (lines 539-542) but does not report the full-rank routed average as a first-class condition in the aggregate table. From the per-seed data, full-rank expert losses are available but the aggregate full-rank routed average is not explicitly computed. This is the most important comparison and should be a headline number.

Working from the code: each expert is evaluated on its own domain at full rank (line 541-542), and the routed condition simply selects the truncated version (line 572-574). The full-rank routed average can be estimated from the specialist losses, which ARE reported and average ~1.0 vs mono_full ~0.95. This 5% gap is the real composition overhead, and it PASSES K1.

### Controls: Adequate.

Base model (no training), sum composition (pathological), average composition, and sequential monolithic (forgetting) provide useful reference points. Three seeds with standard deviations reported.

### Modularity test: Misleading.

The modularity test (remove one expert, measure degradation on other domains) uses the *averaged composition* mode, not the routed mode. For routed composition, removing one expert has exactly 0% effect on other domains by construction. The paper notes this (line 127-128) but the modularity numbers in the results (+158% average degradation) are from the wrong composition mode and could confuse readers.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry specifies:
- Kill criteria: (1) monolithic beats composition by >10%, (2) >2x training cost
- Status: active

K1 is killed at 71%. K2 passes at 0.94x. The PAPER.md labels this PARTIAL KILL, which is appropriate. However, the HYPOTHESES.yml status still shows "active" rather than being updated to reflect the partial kill.

The experiment matches its HYPOTHESES.yml node. The kill criteria are the ones being tested. The evidence is sufficient to change the node status to "partial-kill" with the nuance that the kill is attributable to rank truncation at d=32, not to the composition mechanism.

## Macro-Scale Risks (advisory)

1. **The macro counter-evidence is already strong.** The lora_moe_benchmark (proven) at d=896 shows MoE beats joint training by 0.70%. This directly contradicts the micro K1 kill and supports the truncation-dominated interpretation.

2. **Rank ratio at macro.** At d=896, r=16: each expert captures 1.8% of rank space. For rank-80 monolithic (5 experts * 16), it captures 8.9%. Still a 5x rank ratio disadvantage per expert. The delta_rank_scaling experiment (REVISE status) suggests 95% signal retention at this scale, but that experiment has its own convergence concerns.

3. **Real domain structure.** Synthetic domains (arithmetic, reversal, parity) have essentially zero shared structure. Real domains (e.g., Python, JavaScript, SQL) share substantial structure that monolithic can exploit. This could favor monolithic at macro scale. However, SOLE also exploits shared structure via the shared base model.

4. **LoRA vs post-hoc SVD.** At macro scale, experts will be trained as actual LoRA adapters (rank-constrained during training), not full-rank then truncated. This should favor composed experts because the learned subspace is optimized. The macro lora_moe_benchmark already uses this approach, which may explain why it reverses the micro result.

## Verdict

**REVISE**

The experiment is well-designed and honestly analyzed. The core finding -- that the 71% quality gap is dominated by rank truncation (66%) rather than composition overhead (5%) -- is valuable and correctly identified. However, the presentation and kill criteria need adjustment.

### Required Fixes

1. **Report full-rank routed average as a headline number.** The data exists (specialist_losses in results.json). Compute and report the aggregate full-rank routed average (~1.0) alongside the truncated routed average (1.64) and mono_full (~0.95). This is the most important comparison in the experiment and it should be in the PAPER.md table. The full-rank gap (~5%) PASSES K1 and should be stated explicitly.

2. **Reframe K1 as two sub-criteria.** K1 conflates truncation loss and composition overhead. Split into:
   - K1a: "Full-rank composition loses to full-rank monolithic by >10%." (Expected: PASS at ~5%)
   - K1b: "Truncated composition loses to truncated monolithic by >10%." (Expected: KILL at 71%, attributable to rank starvation at d=32)
   This makes the finding interpretable without 3 paragraphs of caveats.

3. **Fix modularity test reporting.** The +158% degradation number is from averaged composition, not routed composition. For the routed mode (SOLE's actual architecture), degradation is 0% by construction. Report both, clearly labeled. Currently the PAPER.md mentions this in prose (line 127-128) but the numbers in the aggregate results are misleading.

4. **Add Branch-Train-Merge citation.** Li et al. (2022) is the most directly comparable prior work (train domain-specific models independently, merge at inference). Its omission is a gap in the literature positioning. LoRAHub (Huang et al., 2023) should also be cited.

5. **Update HYPOTHESES.yml status.** Change from "active" to "partial-kill" with evidence field populated.

### Non-blocking Notes

- The macro lora_moe_benchmark already demonstrates that K1 reverses at scale. The micro experiment's primary value is the diagnostic decomposition (truncation vs composition), not the headline K1 result.
- The autograd/numpy transformer implementation is adequate for micro but should not be carried forward. Macro experiments should use PyTorch/HuggingFace.
- The extrapolation to d=896 based on delta_rank_scaling is reasonable but depends on an experiment that is itself under REVISE. Flag this dependency explicitly.
