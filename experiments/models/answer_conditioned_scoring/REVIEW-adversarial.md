# Peer Review: Answer-Conditioned Scoring

## NotebookLM Findings

Skipped -- the experiment is straightforward enough that direct review is sufficient. The mathematical decomposition is a basic identity, the experimental design is clean, and the prior art question has a clear answer (see Novelty Assessment below).

## Mathematical Soundness

**Decomposition identity (Section 3): Correct.**
The identity log PPL_full = (T_p/T) * log PPL_prompt + (T_a/T) * log PPL_answer follows directly from partitioning the sum of log-probabilities. This is exact, not approximate.

**Dilution effect analysis (Section 4): Sound.**
The condition for full-sequence PPL to mask answer improvement is correctly derived. The argument that prompt degradation can dominate when T_p >= T_a is mathematically clean.

**Worked example (Section 7): Verified against results.**
The seed-42 reverse domain numbers (FullPPL 7.38 -> 7.51, AnsPPL 3.01 -> 1.25, Acc 0.33 -> 0.73) are internally consistent with the decomposition identity.

**Statistical analysis (Section 8): Adequate with caveats.**
The critical value r >= 0.687 at p<0.05 (one-tailed) for N=5 is correctly cited. The paper honestly reports that only 2/3 seeds exceed this threshold, with the third at r=0.58. The kill criterion of r >= 0.5 is appropriately lower than the significance threshold -- this is a directional test, not a hypothesis test demanding p<0.05.

**One issue: PPL "improvement" sign convention.**
The code computes `fi = (base_ppl - expert_ppl) / base_ppl`, so positive means expert is better (lower PPL). The correlation is then between PPL improvement (positive = better) and accuracy improvement (positive = better). This is internally consistent. However, the PAPER.md Table (Section "Per-Domain Results") reports "FullPPL Improv" as percentages like "-1.8%" and "+9.1%". The -1.8% for reverse means the expert's full PPL got WORSE (higher), which is correctly flagged as the problematic case. The sign convention is consistent throughout.

**Token position alignment: Correct for character-level tokenizer.**
The code uses `s.rfind(delimiter)` to find the delimiter position in the original string, then uses that as the index into the losses/target arrays. This works because the CharTokenizer maps each character to exactly one token ID, so string position = token position. The `rfind` correctly handles all five domains (arithmetic "=" has one occurrence, reverse ">" has one, etc.). This alignment would break with subword tokenization -- acknowledged in Limitations.

## Novelty Assessment

**This is not novel as a technique.** Computing loss only on completion tokens (after a prompt/instruction boundary) is standard practice in:
- Instruction fine-tuning (loss masking on prompt tokens is default in most training frameworks)
- Language model evaluation benchmarks (HELM, lm-eval-harness compute completion-only log-likelihoods for multiple-choice scoring)
- The original GPT-3 few-shot evaluation protocol (Brown et al., 2020) scores only the completion tokens

**However, novelty is not the claim.** The experiment claims to validate answer-conditioned PPL as a usable signal for SOLE shadow scoring, not to invent it. The predecessor (ppl_vs_task_performance) identified the problem; this experiment confirms the obvious fix works. The value is empirical validation within the SOLE context, not methodological novelty.

**Delta over predecessor:** The predecessor achieved r=0.084 (full-seq PPL vs accuracy). This experiment achieves r=0.811 (answer-only PPL vs accuracy). The improvement is large and the mechanism (prompt dilution) is clearly identified.

## Experimental Design

**Hypothesis is well-scoped and the experiment tests it directly.** The kill criteria (K1: r >= 0.5, K2: rankings differ) are reasonable and clearly evaluated.

**Concern 1: Code defaults vs actual run configuration.**
The Python code defaults to d=64, H=4, L=4 (~206K params), but the MATH.md and PAPER.md describe d=32, H=2, L=2 (~29K params). The results JSON confirms H=2, L=2, V=42, matching the paper. The experiment was evidently run with non-default arguments. This should be documented (the exact command used to reproduce results), but does not affect the findings.

**Concern 2: Correlation with N=5 is inherently fragile.**
Five data points for Pearson r means a single outlier can swing the result dramatically. Seed 7 produces r=0.58 (barely above kill threshold) while seeds 42 and 123 produce r=0.91 and r=0.94. The variance (std=0.16) is substantial. The paper acknowledges this honestly.

**Concern 3: Could a simpler mechanism explain the result?**
Yes -- and the paper correctly identifies it. Answer-only PPL works because it measures exactly the tokens that determine accuracy. This is essentially a tautology at the limit: if you measure how well the model predicts answer tokens, and accuracy is whether the model predicts answer tokens correctly, these should correlate. The value is quantifying HOW MUCH better this correlation is (r=0.81 vs r=-0.31), and confirming that full-sequence PPL is actively misleading (anti-correlated).

**Concern 4: Full-rank expert delta, not LoRA.**
The experiment uses full fine-tuning (expert_params - base_params) rather than LoRA. The PPL decomposition is parameterization-agnostic (it depends only on the model's output distribution, not how the weights are structured), so this does not invalidate the finding. The paper notes this in Limitations.

**Concern 5: Each expert is evaluated only on its own domain.**
This is appropriate for the stated hypothesis (does answer PPL correlate with accuracy for the expert's target domain). Cross-domain evaluation would test a different question (composition interference), which is out of scope.

**Controls: Adequate.** The base model serves as baseline. Full-sequence, answer-only, and prompt-only PPL are all computed, providing the full decomposition. Three seeds provide some robustness. The prompt-only anti-correlation (r=-0.74) provides strong diagnostic support for the dilution mechanism.

## Hypothesis Graph Consistency

The experiment is listed in FINDINGS.md as "proven" with appropriate caveats. It is not a standalone HYPOTHESES.yml node but supports `exp_clone_compete_evolution` and the shadow scoring mechanism in VISION.md. The kill criteria (K1: r >= 0.5, K2: rankings differ) are exactly what was tested. The evidence is sufficient to consider answer-conditioned PPL as a viable shadow scoring signal for the Evolve phase.

## Macro-Scale Risks (advisory)

1. **Subword tokenization breaks the string-position = token-position assumption.** At macro scale, the delimiter must be identified in the token sequence, not the string. For structured prompts (instruction-following, chat templates), this is straightforward (the assistant turn boundary is known). For free-form generation, it requires heuristics.

2. **Distributional answers weaken the PPL-accuracy link.** The micro experiment uses deterministic answers (one correct answer per input). Real tasks (summarization, creative writing) have many valid completions. A model that generates a different valid completion than the reference will have high PPL but high quality. This is the fundamental limitation of reference-based PPL as a quality signal.

3. **The r=0.81 correlation is inflated by the synthetic task structure.** With deterministic answers and short sequences, the signal-to-noise ratio is high. Expect lower correlations at macro scale. However, even r=0.5 would be sufficient for shadow scoring (distinguishing better from worse experts in a tournament), so degradation is acceptable if it stays above this threshold.

4. **EOS token inclusion in answer PPL.** The code includes the EOS token prediction in the answer PPL computation (losses from delim_pos to end of sequence, which includes the EOS prediction). For short answers (T_a=2-6), the EOS token is 15-50% of the answer tokens. At macro scale with longer completions, this dilution becomes negligible.

## Verdict

**PROCEED**

The experiment is mathematically sound, the code correctly implements the stated metric, and the results clearly demonstrate that answer-conditioned PPL is a dramatically better proxy for task accuracy than full-sequence PPL (r=0.81 vs r=-0.31). The finding is directionally strong across all 3 seeds.

The technique itself (completion-only loss) is well-established in the literature, but the experiment's value is confirming it works as a shadow scoring signal within the SOLE architecture -- validating a critical assumption of the Evolve phase. The main macro risk is that the correlation will weaken with distributional answers, but even moderate correlation suffices for tournament-style shadow scoring.

Minor fix recommended (non-blocking):
1. Document the exact run command (d=32, H=2, L=2 etc.) in the experiment directory, since the code defaults differ from the actual configuration used.
