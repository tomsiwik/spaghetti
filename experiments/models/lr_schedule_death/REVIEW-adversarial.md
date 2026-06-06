# Peer Review: LR Schedule Impact on Death Trajectory (Exp 19)

## NotebookLM Findings

Skipped -- NotebookLM automation not authenticated in this environment. Review proceeds from direct document analysis.

## Mathematical Soundness

### What holds

1. **LR schedule definitions (Section 2.1)** are standard and correctly implemented. The code in `make_lr_schedule()` matches MATH.md's formulas. The warmup uses `optim.linear_schedule(0, peak_lr, warmup_steps)`, cosine uses `optim.cosine_decay(peak_lr, decay_steps, end=0)`, and warmup+cosine joins them at the warmup boundary. Verified against MLX API semantics.

2. **Prediction 1 numerical estimate (Section 3.2)** is internally consistent. At S=50 with S_w=320, eta(50) = 3e-3 * (50/320) = 4.69e-4, which is indeed 15.6% of peak. The linear interpolation prediction of delta_warmup(50) ~ 24.6% is a reasonable first-order estimate. The actual result (13.2%) is even lower, which is consistent with the prediction's direction but suggests the death-rate-vs-LR relationship is sublinear (death probability decreases faster than LR decreases), not a mathematical error.

3. **Section 3.5 cosine-at-S=50 calculation** is correct: cos(pi * 50/3200) = cos(0.049) = 0.9988, giving eta_cosine(50) = 2.998e-3. The prediction that cosine-only has the same spike as constant is confirmed (50.4% vs 51.6%).

4. **Section 4.4 design decision** is sound. Defining the schedule over S_total=3200 and truncating at each checkpoint S simulates "snapshot of a full training run." The alternative (compressing the schedule into S steps) would confound schedule shape with training duration.

### What does not hold or is questionable

5. **Death probability model (Section 3.1)**: The formula P(alive -> dead) ~ eta(s) * ||g_i(s)|| / ||a_i(s)|| * phi(margin_i(s)) is presented as coming from training_duration/MATH.md Section 3.4 but is never derived rigorously. It is a heuristic proportionality claim, not a theorem. The key assumption -- that death probability is linear in eta(s) -- is what the experiment tests, so this circularity is acceptable. However, MATH.md should be clearer that this is a motivating heuristic, not a derivation.

6. **Prediction 2 (Section 3.3) logic gap**: The argument is: (a) fewer new deaths from lower LR, plus (b) continued revival from weight shifts, yields net revival. But (b) is underspecified. If revival probability is also proportional to LR (larger weight shifts = more input distribution change = more revival), then lower LR reduces both death AND revival. The net effect is not obviously "more net revival." The paper relies on Gurbuzbalaban et al. (2024) as empirical backing, which is fine, but the theoretical argument is incomplete. The data confirms the prediction empirically (cosine revival 11.8pp vs constant 5.1pp), so this is a narrative weakness, not an experimental one.

7. **Warmup+cosine revival interpretation**: PAPER.md reports warmup+cosine revival as +2.0pp (21.6% -> 19.6%) and says "less ABSOLUTE revival because it starts from a much lower death rate." This is correct but obscures the fact that the kill criterion measures absolute revival: max(cosine_revival, wc_revival) > const_revival. The cosine-only schedule (11.8pp) passes this criterion, but warmup+cosine (2.0pp) does not exceed constant (5.1pp). The kill criterion is satisfied because the test uses max() over the two cosine schedules. This is fair as stated in MATH.md, but the paper should acknowledge that warmup+cosine shows LESS absolute revival than constant, and the cosine-only schedule is doing the heavy lifting for Kill 3.

## Novelty Assessment

### Prior art

- **Gurbuzbalaban et al. (2024)**: Directly predicts that LR decay boosts revival. This experiment confirms their finding at micro scale. The delta over their work is: (a) measuring the warmup effect on the spike, which they do not study; (b) the combined warmup+cosine schedule; (c) per-layer death analysis.

- **"Analyzing and Reducing the Need for Learning Rate Warmup" (NeurIPS 2024)**: Already demonstrated that GPT-2 without warmup has large fractions of permanently dead ReLUs. This experiment replicates their finding at d=64 -- valuable confirmation but not novel.

- **Li et al. (2023), "The Lazy Neuron Phenomenon"**: Reports ~50% natural ReLU sparsity. The constant-LR result (47.3%) is consistent. The warmup+cosine result (19.6%) is a genuinely new data point -- showing that the "natural" sparsity level is schedule-dependent.

### Novelty delta

The primary contribution is the **systematic 4-schedule comparison** showing that the death trajectory is qualitatively different under warmup+cosine vs constant LR, and quantifying the effect sizes. The individual findings (warmup prevents death, cosine enables revival) have precedent, but their combined measurement in a single controlled experiment is new within this project's context. The most novel finding is the 2.4x ratio between constant-LR and warmup+cosine equilibrium death rates (47.3% vs 19.6%), which directly revises the macro predictions from Exp 17.

No reinvention of existing references/ code detected (references/REFERENCES.yml entries list is empty).

## Experimental Design

### Does the experiment test its hypotheses?

**Yes, cleanly.** The four-schedule sweep with shared pretrained base isolates the LR schedule as the single variable. The checkpoint grid covers both the spike phase (S=50) and the equilibrium phase (S=3200). The S=0 baseline is shared across schedules (correct -- same pretrained model). The design is a textbook controlled experiment.

### Controls adequate?

**Mostly.** The constant-LR schedule serves as the baseline (replicating Exp 17). Three concerns:

1. **Missing total-gradient-magnitude control**: The warmup+cosine schedule applies less total gradient over 3200 steps than constant LR (integral of eta(s) is smaller). The lower death rate could be explained by "less total learning" rather than "gentler learning." A control for this would be a constant schedule at a lower LR chosen to match the integral of the warmup+cosine schedule. PAPER.md acknowledges val loss is better under warmup+cosine (0.4761 vs 0.4855), which partially addresses this -- the model learns MORE effectively with less total gradient, so "less learning" is not the explanation. However, this is correlational, not a direct control.

2. **Seed count**: 3 seeds with reported stdev of 5-8pp. The S=50 finding is robust (38.4pp difference dwarfs any plausible noise), but the difference between cosine (42.2%) and warmup (27.7%) at S=3200 could shift with more seeds. PAPER.md acknowledges this in Limitations.

3. **Warmup and warmup+cosine produce identical results for S <= 400**: Looking at the data table, warmup and warmup+cosine have IDENTICAL death rates at S=50 (13.2%), S=100 (17.6%), S=200 (21.6%), and S=400 (28.1%). This is expected -- the cosine decay has not meaningfully diverged from constant by S=400 (cosine LR at S=400 is still 95% of peak when measured from step 320). But it means the first 4 checkpoints provide zero discriminating information between warmup and warmup+cosine. The schedules diverge only at S=800+. This is not a flaw, but it means the effective sample size for the warmup vs warmup+cosine comparison is only 3 checkpoints (800, 1600, 3200), not 7.

### Could a simpler mechanism explain the results?

The "total gradient integral" alternative mentioned above is the main threat. However, the val loss data argues against it: warmup+cosine achieves the best val loss despite having the smallest gradient integral. This is consistent with the well-established understanding that warmup+cosine is a better optimizer, not just a "less aggressive" one.

### Hypothesis graph consistency

The experiment corresponds to VISION.md item 10 ("LR schedule impact on death trajectory"). It is NOT listed as a separate node in HYPOTHESES.yml (no `exp19_lr_schedule_death` node exists). The closest is `exp17_training_duration_death`, which this extends. The paper correctly positions this as extending Exp 17. However, HYPOTHESES.yml should be updated to reflect this experiment's results -- either as evidence on exp17 or as its own node.

The kill criteria in the code match MATH.md Section 5 exactly. All three criteria were evaluated correctly in the code's `main()` function.

## Macro-Scale Risks (advisory)

1. **SiLU activation**: Qwen uses SiLU, not ReLU. SiLU has no hard zero boundary, so "dead" is defined by magnitude threshold. Warmup's effect on a soft activation function is unclear -- it may still prevent low-magnitude neurons, but the binary dead/alive framework needs adaptation. PAPER.md acknowledges this.

2. **Warmup fraction sensitivity**: The experiment uses 10% warmup. Many LLM training recipes use 1-2% warmup (e.g., 2000 steps out of 100K). With 1% warmup, the warmup would only reach 50 steps (the spike location). At that point, eta(50) = peak * (50/50) = peak -- no spike reduction at all. The effectiveness of warmup depends critically on the warmup phase being longer than the death spike timescale (~50 steps). PAPER.md mentions this in Limitations but does not quantify the sensitivity.

3. **Frozen attention**: The experiment freezes attention during fine-tuning. Full fine-tuning creates additional distribution shifts via attention weight changes. The interaction between LR schedule and attention unfreezing could amplify or suppress the effects observed here. PAPER.md notes this.

4. **Adam momentum interaction**: Adam's exponential moving averages of gradients and squared gradients interact with LR schedules in non-obvious ways. At very low LR during warmup, Adam's adaptive step size may partially compensate, reducing the effective difference between warmup and non-warmup schedules. This is not examined.

5. **Depth scaling**: At d=4096 with many more neurons, the death dynamics could differ. The "margin" distribution (distance of neurons to the death boundary) may be very different at large d, changing the proportionality between LR and death rate.

## Additional Observations

### Strengths

- **Clean experimental design**: Single variable (LR schedule), shared base, adequate checkpoint grid, 3 seeds. This is the best-controlled experiment in the micro series.

- **Actionable macro implications**: The revised prediction (20% dead under warmup+cosine, not 47%) directly impacts the composition protocol's expected pruning yield. This is practically useful.

- **Self-consistent with prior work**: Cosine-only spike matches constant (confirming the calculation), warmup effect direction matches NeurIPS 2024 warmup paper, revival under cosine matches Gurbuzbalaban. All three external predictions confirmed.

- **Val loss correlation**: Showing that warmup+cosine produces the best model quality AND the fewest dead neurons is a strong result. It eliminates the concern that fewer dead neurons means "the model didn't learn."

### Weaknesses

- **No statistical significance testing**: The 3-seed aggregates report means and stdev but no confidence intervals or hypothesis tests. The warmup effect at S=50 is so large (38pp) that significance is obvious. But the cosine vs warmup difference at S=3200 (42.2% vs 27.7%) with stdev of 5-8pp has unclear significance at n=3.

- **Missing HYPOTHESES.yml node**: This experiment should be recorded in the hypothesis graph.

- **Over-claiming on revival**: "Cosine decay more than doubles neural revival" (Finding 2) compares absolute revival (11.8pp vs 5.1pp). But revival is not measured per-capsule here (that was Exp 18). The "revival" metric is just aggregate death decrease from S=200 to S=3200, which conflates true revival with reduced new deaths. PAPER.md acknowledges the ambiguity in the warmup+cosine case but not for cosine-only.

- **Worked example (Section 6)** is illustrative but uses made-up numbers (not from actual runs). This is fine for pedagogical purposes but should be explicitly labeled as "illustrative" not "from data."

## Verdict

**PROCEED**

The experiment is well-designed, the math is sound at the level of motivating heuristics (not rigorous proofs, but that is appropriate for an empirical study), the results are clear and large in magnitude, and the implications for the macro transition are actionable and consequential.

The core finding -- that warmup+cosine reduces equilibrium death from 47% to 20% -- is robust. The effect size at S=50 (38pp warmup reduction) is far beyond noise. The cosine revival finding is directionally consistent with published prior art. No kill criteria are triggered.

Minor issues (incomplete theoretical argument for Prediction 2, missing statistical tests, HYPOTHESES.yml not updated) do not warrant REVISE status because they affect narrative quality, not the validity of the experimental results.

### Recommended (non-blocking) improvements:

1. Add a HYPOTHESES.yml node for this experiment with the three kill criteria and evidence.
2. Clarify in MATH.md Section 3.1 that the death probability formula is a motivating heuristic, not a derivation.
3. Acknowledge in PAPER.md Finding 2 that "revival" at the aggregate level conflates true revival with reduced new deaths, and that per-capsule tracking (Exp 18 style) would be needed to distinguish them under cosine decay.
4. Label the Section 6 worked example as "illustrative" rather than empirical.
5. Note that warmup+cosine revival (+2.0pp) is actually LESS than constant revival (+5.1pp) -- the kill criterion passes only because cosine-only (11.8pp) carries it via the max() aggregation.
