# Peer Review: bitnet_clone_compete_powered

## NotebookLM Findings

Skipped -- the experiment is already killed and the review is post-mortem. The key question is whether the kill is legitimate and the interpretation is sound.

## Mathematical Soundness

### What holds

1. **Sample size calculation (MATH.md Section 2).** The formula n >= (z_alpha/2 + z_beta)^2 / (4 * delta^2) is correct for a binomial test. For delta=0.05, n=784 is right. For delta=0.10, n=196. The experiment achieved N=419 decisive samples for the primary comparison, well-powered.

2. **Statistical tests (code lines 365-470).** Binomial test, Wilcoxon signed-rank, paired t-test, and Cohen's d are all correctly implemented. The CI calculations use the t-distribution with correct degrees of freedom. The Wilcoxon correctly filters ties before computing.

3. **Win-rate vs aggregate paradox (MATH.md Section 4).** The mathematical possibility of E[A-B] > 0 while P(B < A) < 0.5 is correctly explained. This is a real phenomenon (analogous to Simpson's paradox in expectation vs probability).

4. **Warm-start decomposition (MATH.md Section 5).** The arithmetic is correct: on held-out data, Delta_warm = 13.23 - 13.24 = -0.01, confirming zero warm-start advantage.

### What does not hold

1. **K1 kill criterion has a logic gap in code (non-blocking).** Line 815-816: `k1_kill = k1_win_rate < 0.55 and k1_p > 0.05`. The actual result is win_rate=28.9% with p approximately 0. Since p is NOT >0.05, k1_kill evaluates to False, and the code outputs verdict "INCONCLUSIVE" (results.json line 159). The paper correctly interprets this as killed, but the code's kill criterion was written assuming failure means "clone is at chance" -- it did not anticipate the clone significantly LOSING. This is a code logic error, not a mathematical error. The paper's interpretation is the correct one.

2. **Training budget contradiction between PAPER.md and code.** PAPER.md Limitation 2 (line 196) states: "Clone v2 inherits 200 steps on law-stack-exchange plus 400 steps on legalbench = 600 total steps." But the code (line 58) sets `COLD_START_TRAIN_STEPS = 400` with comment "original (200) + clone_v1 (200) = 400 total warm-start steps." The clone_v2 was trained for 200 additional steps on legalbench (from the prior experiment), not 400. The PAPER's claim of "600 total" is wrong -- it should be 400 (200 original + 200 clone_v2). This error makes the stated limitation more severe than it actually is, so it is conservative rather than misleading.

3. **MATH.md Assumption 2 is internally inconsistent.** It states "Cold-start gets 400 steps to match clone's cumulative 400 steps (original 200 + clone 200)" but then says "the original's 200 steps on law-stack-exchange data are 'free' for the clone, giving it more diverse data exposure total." If the original's 200 steps are "free" then the clone has MORE data exposure, and the cold-start control is disadvantaged. This means the cold-start result is even stronger than claimed (it matches clone despite having less total gradient exposure), which strengthens the kill rather than weakening it.

## Novelty Assessment

### Prior art

- **Population-Based Training (Jaderberg et al., 2017):** Correctly cited. The clone-compete protocol is a simplified PBT applied to LoRA adapters. PBT's standard finding is that it helps via hyperparameter scheduling, not weight inheritance per se -- consistent with this experiment's null result on inheritance.

- **"The Appeal and Reality of Recycling LoRAs" (2602.12323):** Cited in PAPER.md as motivation for the cold-start control. This is the right reference -- it found random LoRAs merge as well as carefully selected ones, which directly predicts the null result here. However, this reference is not in `references/REFERENCES.yml`, which is a bookkeeping gap.

- **Sakana AI evolutionary merging (2403.13187):** Cited. Relevant but operates at a different level (model-level merging recipes vs adapter-level fine-tuning). The null result here does not contradict Sakana's findings because their evolution is over merging coefficients, not over fine-tuning inheritance.

### Delta over existing work

The contribution is a well-controlled experiment showing that warm-start inheritance provides no advantage over cold-start retraining for LoRA adapter evolution at 2B scale. This is a useful negative result. The win-rate-vs-aggregate paradox is a genuinely interesting methodological finding that should be preserved even though the hypothesis is killed.

## Experimental Design

### Strengths

1. **Cold-start control is the right control.** This is exactly what the prior review requested, and it cleanly disambiguates "more data helps" from "inheritance helps."

2. **Three statistical tests with complementary strengths.** Binomial (direction), Wilcoxon (signed ranks, non-parametric), t-test (parametric with magnitude). All three agree on the key comparisons.

3. **Composition quality check.** Testing that neither the clone nor the cold-start regresses the composed model is good experimental hygiene.

4. **Pre-registration of kill criteria.** K1, K2, K3 were specified before the experiment ran (inherited from the prior review's requirements).

### Weaknesses

1. **Distribution shift confound is more severe than acknowledged.** The tournament mixes law-stack-exchange (samples 500+) and lex_glue ecthr_a. The original adapter was trained on law-stack-exchange samples 0-500, giving it a distributional advantage on the law-stack-exchange portion of the tournament. The clone and cold-start were trained on legalbench (contract NLI), a completely different legal subdomain. The observed pattern -- original winning most per-sample comparisons but losing on aggregate -- could be fully explained by this mixture: original dominates on law-stack-exchange samples (many easy wins), clone/cold-start dominate on lex_glue samples (fewer but larger wins). This is not a "specialization bias from warm-start"; it is a domain mismatch artifact. A proper control would use tournament data from a THIRD legal domain unrelated to any training source. This does NOT save the hypothesis (the kill is still valid because clone and cold-start are indistinguishable), but it undermines the "win-rate paradox" interpretation.

2. **Cold-start trains on legalbench only, not on law-stack-exchange + legalbench.** The clone inherits representations from law-stack-exchange training (via the original's weights) and then trains on legalbench. The cold-start trains only on legalbench from scratch. A stronger control would train cold-start on both data sources sequentially (200 steps law-stack-exchange + 200 steps legalbench) to fully match the clone's data exposure. The current design conflates "warm-start advantage" with "multi-source data advantage." However, since the clone and cold-start achieve identical aggregate PPL, this confound does not change the conclusion.

3. **The code's verdict logic does not match the paper's interpretation.** The JSON says "INCONCLUSIVE" while the paper says "KILLED." The K3 pass/kill semantics are inverted from what makes sense for the hypothesis. This creates confusion for anyone reading the raw results.

4. **Single seed.** Acknowledged in limitations. The paper cites CV=0.5% from prior experiments, which is reassuring but was measured for a different experimental setup.

### Does this test what it claims?

Yes. The core claim is "warm-start inheritance helps beyond additional data." The cold-start control directly tests this. The result (cold-start matches or beats clone on every metric) is a clean falsification.

## Macro-Scale Risks (advisory)

1. **The "retrain from scratch" alternative may not scale.** At macro scale with expensive training (hours per adapter), warm-start might save significant compute even if it provides no quality advantage. The micro experiment correctly kills the quality claim but cannot speak to the compute efficiency argument.

2. **The kill applies to the specific protocol tested (clone -> fine-tune -> tournament).** Other evolutionary approaches (e.g., Sakana-style merging coefficient evolution, or PBT with hyperparameter scheduling) remain untested. The macro hypothesis `exp_clone_compete_evolution` uses "corrected clone" which may differ from simple continued training.

3. **The win-rate-vs-aggregate paradox matters for production routing.** If the Evolve phase uses per-sample routing decisions (which adapter to serve for a given query), win-rate matters. If it uses global selection (which adapter to keep), aggregate PPL matters. The production system should decide which metric governs selection before scaling.

## Verdict

**PROCEED** (as a killed experiment -- the kill is valid)

The experiment correctly falsifies the warm-start inheritance hypothesis. The cold-start control is well-designed, the statistics are sound, and the conclusion follows from the evidence. Three specific fixes should be applied for archival quality:

### Required fixes

1. **Fix the training budget claim in PAPER.md Limitation 2.** Change "200 steps on law-stack-exchange plus 400 steps on legalbench = 600 total" to "200 steps on law-stack-exchange plus 200 steps on legalbench = 400 total" to match the code. Verify the actual clone_v2 training history from the prior experiment.

2. **Fix the code verdict logic or add a note.** The JSON verdict says "INCONCLUSIVE" but the paper says "KILLED." Either fix the K1 kill criterion to handle the case where clone significantly LOSES (k1_kill should be True when win_rate < 0.55 AND binom_p < 0.05 in the wrong direction), or add a note in the JSON explaining the discrepancy.

3. **Add 2602.12323 to references/REFERENCES.yml.** The cold-start control is directly motivated by this paper. It should be formally tracked.

### Non-blocking observations

- The "win-rate-vs-aggregate paradox" interpretation as "specialization bias" is likely a domain-mismatch artifact. Consider softening this claim to "the paradox is consistent with both specialization bias and domain mismatch in the tournament data."
- The composition quality difference between clone and cold-start (clone slightly better on all 5 domains) is within noise (max delta 0.26%) and should not be used to argue for any surviving warm-start signal without error bars.
