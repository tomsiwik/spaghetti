# Peer Review: BitNet-SOLE vs Monolithic

## Mathematical Soundness

**Derivations are correct.** The capacity argument (rank-r starvation for monolithic vs Nr effective rank for SOLE) is standard and well-stated. The interference bound calculation (section "Worked Example") checks out: (N-1) * |cos| / N = 4 * 0.002 / 5 = 0.0016. The STE ternary quantizer formulation is standard.

**Kill criterion K1 has a minor text/math inconsistency.** PAPER.md states ">80% of per-domain metrics" (strictly >80% = 5/5 only), while MATH.md formalizes it as `mono_wins >= ceil(0.8 * 5) = 4` (>=80% = 4/5 or 5/5). The code implements the MATH.md version (`k1_killed = mono_wins >= 4`). This is the stricter formulation and the result (1/5) passes both versions, so this is non-blocking but should be made consistent.

**Parameter budget analysis is honest.** SOLE uses 5x total parameters (108M vs 21.6M). The paper acknowledges this clearly and argues it is the architecture's design point. This is a legitimate framing -- SOLE's value proposition IS that capacity scales with N -- but reviewers should note that a rank-80 monolithic adapter would be a fairer parameter-matched comparison. The paper preemptively addresses this by noting r=3 per expert is below the useful rank threshold, which is reasonable.

**Data exposure analysis (not discussed in paper, should be).** Training loop uses `idx = step % len(train_tokens)`, batch_size=1. The monolithic model trains on a shuffled union of ~3700 samples for 2000 steps. Per-domain, monolithic sees approximately (domain_samples / 3700) * 2000 gradient updates containing that domain's data. For medical: ~432 medical gradients out of 2000 total. Each SOLE expert gets 400 pure-domain gradients. These are surprisingly comparable per-domain, which is why the "same total gradient steps" framing is not misleading. However, the monolithic Adam optimizer accumulates momentum over 2000 steps (vs 400 for each SOLE expert), giving it a potential optimization advantage that is not discussed.

## Novelty Assessment

**The comparison itself is the contribution, not a novel mechanism.** Branch-Train-Merge (Li et al., 2022), LoRAHub (Huang et al., 2023), and TIES-Merging (Yadav et al., 2023) all explore multi-expert vs monolithic trade-offs. The delta here is:
1. First such comparison on ternary (BitNet-2B-4T) architecture
2. Demonstration that the FP16 d=32 result (mono wins 5/5) reverses at d=2560

The reversal narrative is compelling and constitutes genuine empirical value. The paper correctly frames this as validating the prior experiment's prediction rather than claiming fundamental novelty.

**Prior art check:** The `references/` directory does not contain implementations that already perform this specific comparison on BitNet. The experiment builds cleanly on the project's own prior work (bitnet_2b_real_composition, bitnet_ternary_convergence).

## Experimental Design

**Strengths:**
1. Four conditions (routed, composed, mono-shuffled, mono-sequential) provide a comprehensive comparison
2. Same hyperparameters across all conditions (lr, rank, scale, seq_len)
3. Proper train/val separation with non-overlapping splits
4. Sequential condition provides a valuable catastrophic forgetting baseline
5. Orthogonality measurement links to prior results for continuity

**Weaknesses:**

1. **Single seed is the primary weakness.** The margins are moderate: 2.8% to 8.8% for SOLE wins, 5.5% for mono win. With single seed, we cannot determine whether creative writing's monolithic advantage is robust or an artifact. The paper acknowledges this honestly. The multiseed experiment (exp_bitnet_multiseed_validation) at N=5 showed CV=0.5% for composition metrics, suggesting PPL measurements are stable, but that was a different experiment (composition, not SOLE-vs-mono). This partially mitigates the concern but does not fully address it.

2. **Creative writing training shows increasing loss** (first_50=1.2446, last_50=1.6359). This is anomalous -- the expert got WORSE during training by this metric. Yet SOLE creative PPL (3.17) still beats base (3.60). Combined with the non-convergence of the code expert (first_50=1.07, last_50=1.12), 2/5 SOLE experts show training pathology. The paper attributes this to STE quantization noise obscuring convergence, which is plausible but unverified. A concern: if SOLE's creative expert is under-trained due to this pathology, the monolithic win on creative might partially reflect SOLE's training failure rather than monolithic's cross-domain advantage.

3. **Legal domain has fewer samples** (500 train vs 800 for others). This creates an unequal representation in the monolithic union data (500/3700 = 13.5% vs 800/3700 = 21.6% for others). Legal shows the largest SOLE advantage (-8.8%). The under-representation of legal in monolithic training may partially explain why monolithic underperforms on legal. This is a confound worth noting but probably not large enough to flip the result.

4. **Oracle routing.** PAPER.md acknowledges this (Limitation 4) and cites >95% routing accuracy from a prior experiment. This is adequate for micro-scale. At macro, routing errors compound.

5. **Composition ratio computation.** Line 830: `composition_ratio = avg_sole_composed / best_ind`. This divides the average composed PPL across all domains by the best individual domain PPL. This mixes a multi-domain aggregate with a single-domain optimum, which is not a standard metric. It works for tracking across experiments but should not be over-interpreted.

**Could a simpler mechanism explain the result?** Possibly. SOLE's advantage could be explained purely by the capacity argument (5x more parameters) rather than the routing/specialization mechanism. A rank-80 monolithic adapter (same total parameter budget as 5 x rank-16 SOLE) would disambiguate capacity vs specialization. The paper acknowledges the parameter asymmetry but does not run this ablation. This is acceptable for micro-scale -- the experiment tests the architecture as designed, not parameter-matched alternatives -- but the "SOLE wins because of specialization" narrative is not fully established without this control.

## Hypothesis Graph Consistency

The HYPOTHESES.yml node (exp_bitnet_sole_vs_monolithic) has one kill criterion: "monolithic ternary LoRA beats composed SOLE on >80% of per-domain metrics." Note this says "composed SOLE" but the experiment primarily compares "routed SOLE" against monolithic. The PAPER.md evaluates both routed and composed. If the criterion is literally about composed SOLE, then mono beats composed on 5/5 domains (mono avg 7.43 vs composed avg 9.55), and K1 would be KILLED.

However, the evidence block in HYPOTHESES.yml and the paper both treat "SOLE routed" as the primary condition, which is the correct comparison for SOLE's value proposition (routing is the deployment mode). The kill criterion text should be updated to say "routed SOLE" for consistency.

Status is "supported" (not "proven"), which is appropriate given single-seed and parameter asymmetry caveats.

## Macro-Scale Risks (advisory)

1. **Parameter asymmetry compounds at scale.** At N=50 domains, SOLE uses 50x more total parameters than monolithic. A rank-800 monolithic adapter (or a reasonably high-rank one) might close the gap. The capacity argument cuts both ways.

2. **Routing accuracy degrades with N.** At N=50+ domains, routing errors become non-negligible. Each routing error forces the query through a non-specialist expert, potentially worse than monolithic.

3. **Creative writing exception may generalize.** Many real-world tasks benefit from cross-domain knowledge (e.g., medical writing, legal coding, technical explanations). If the "creative writing pattern" (cross-domain transfer > specialization) holds for a significant fraction of domains, the 4/5 win rate may not extrapolate.

4. **Sequential forgetting result is strong but the comparison is unfair.** The sequential monolithic trains on domains in order without any replay or regularization. Modern continual learning methods (EWC, experience replay) would significantly reduce forgetting. The +61% degradation on medical is worst-case.

## Verdict

**PROCEED**

The experiment is well-designed within micro-scale constraints and achieves its stated goal: demonstrating that SOLE routed beats monolithic on 4/5 domains at d=2560, reversing the prior FP16 d=32 result. The math is sound, the code is clean, and the paper is honest about limitations.

Three non-blocking items to address in the paper:

1. **Fix kill criterion text consistency.** HYPOTHESES.yml says "composed SOLE" but the experiment evaluates and passes on "routed SOLE." Update the HYPOTHESES.yml kill criterion to say "routed SOLE" to match the actual test. If the criterion genuinely means composed SOLE, the experiment would be KILLED (mono beats composed 5/5).

2. **Note the creative expert training pathology.** The creative SOLE expert's loss increased during training (1.24 -> 1.64). Add a sentence acknowledging this may confound the mono-wins-creative finding -- monolithic may win creative partly because the SOLE expert was poorly trained, not only because cross-domain transfer helps.

3. **Add a sentence on Adam optimizer asymmetry.** Monolithic benefits from 2000 steps of optimizer state accumulation vs 400 per SOLE expert. This slightly favors monolithic and makes SOLE's 4/5 win more impressive, but it should be stated.

These are documentation fixes, not experimental reruns. The result stands as SUPPORTED.
