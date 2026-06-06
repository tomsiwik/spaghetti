# Peer Review: Output-Averaging vs Parameter-Merging

## NotebookLM Findings

Skipped -- NotebookLM automation not available in this environment. Review conducted via direct analysis of MATH.md, PAPER.md, run_experiment.py, and results.json.

## Mathematical Soundness

### What holds

1. **Cross-term analysis is correct.** The Taylor expansion argument (MATH.md lines 54-60) properly identifies that cross-terms scale as O(1/k^2) while signal scales as O(1/k). The conclusion that cross-terms become negligible at large k is standard and sound.

2. **The 1/k dilution argument is correct.** At k=49, each adapter contributes 2% of its signal under pre-merge. This is the dominant degradation mechanism, not cross-terms. The paper correctly identifies this distinction.

3. **Logit averaging vs probability averaging is correctly noted.** MATH.md properly distinguishes geometric mean (logit averaging) from arithmetic mean (probability averaging) and correctly states logit averaging is sharper.

4. **Compute cost analysis is accurate.** Pre-merge is O(1) forward passes; output-averaging is O(k). The empirical latency scaling (~10x at k=5, ~51x at k=25, ~101x at k=49) matches the theoretical k-fold cost.

### What needs attention

1. **The Grassmannian prediction is misleading.** MATH.md (lines 94-103) predicts that if A_i^T A_j ~ 0, then cross-terms are small and pre-merge should match output-averaging. But the experiment shows pre-merge loses badly at k>=25. The PAPER correctly attributes this to 1/k dilution rather than cross-terms, but the MATH.md narrative sets up a prediction (cross-terms small implies PM ~ OA) that the experiment actually falsifies. The math is correct but the framing confuses two distinct failure modes.

2. **The inequality ||Delta_i^T Delta_j|| <= ||B_i|| * ||A_i^T A_j|| * ||B_j|| (line 98-99) is a bound on cross-terms, not on the dilution effect.** The experiment's actual finding is that dilution dominates cross-terms, which means the Grassmannian skeleton is irrelevant to the PM vs OA gap. The math could lead readers to think orthogonality helps pre-merge; it does not help against dilution.

3. **PPL relationship section (lines 146-167) is hand-wavey.** The gap is described as "f(cross_terms, dilution_effect)" without specifying the functional form. This is a placeholder, not a derivation. Acceptable for a micro-experiment write-up, but should not be cited as a quantitative prediction.

## Novelty Assessment

### Prior art

The experiment cites arxiv 2603.03535 ("Ensembling vs Merging vs Routing") which already establishes that ensembling > merging for multi-adapter composition. The delta here is:

1. **Ternary-specific:** The prior work does not test ternary LoRA adapters. The finding that the Grassmannian skeleton's orthogonality does not help against 1/k dilution is novel within the project's context.

2. **The k-crossover point:** Finding that pre-merge wins at k=5 but loses at k>=25 adds granularity to the binary "ensembling > merging" result from the literature. The crossover at approximately k~10 is a useful quantitative contribution.

3. **Latency on Apple Silicon:** The M5 Pro latency numbers (74ms/tok at k=49 for OA) are platform-specific data not available elsewhere.

**Assessment:** The experiment is confirmatory rather than novel. It validates a known result (ensembling > merging at large k) in a specific context (ternary LoRA on BitNet). This is legitimate for informing the SOLE architecture's design decisions but should not be overclaimed as a new finding.

## Experimental Design

### Strengths

1. **Clean experimental design.** Same model, same adapters, same eval data. The only variable is the composition method. Good controlled comparison.

2. **Multiple k values.** Testing k=5, 25, 49 reveals the crossover behavior. This is the right experimental design.

3. **Per-domain breakdown.** Reporting all 8 domains rather than just aggregates allows scrutiny of individual effects.

4. **Separate latency benchmark.** Using dedicated timing passes (with warmup) rather than relying on PPL-evaluation timing is methodologically sound.

### Weaknesses requiring scrutiny

1. **The eli5 anomaly dominates the aggregate result.** At k=25, eli5 shows -55.4% PPL improvement (OA: 1.22 vs PM: 2.73). The next largest is legal at -12.0%. Remove eli5 and the aggregate OA advantage at k=25 drops from -11.5% to approximately -9.1% (still significant, but the headline number is inflated by one outlier domain). The PAPER does not flag this or analyze why eli5 is an outlier. Most likely: the eli5 adapter is in the k=25 set, and it is disproportionately effective at full strength vs 1/25 dilution because eli5 data is stylistically distinctive.

2. **Evaluation on only 8 domains out of 49 adapters composed.** The 8 eval domains are a subset of the adapters being composed. At k=25, 17 of the 25 composed adapters have no evaluation data in this experiment. We do not know if those 17 domains benefit from output-averaging or are hurt by it. The experiment measures "does OA help on the eval domains" but not "does OA help globally across all composed domains."

3. **No statistical significance testing.** With 5 samples per domain, the standard error on PPL estimates is large. The paper acknowledges this ("directional, not definitive") but still reports precise delta percentages (+3.01%, -11.49%, -11.59%) that imply more precision than exists. At minimum, reporting standard errors or confidence intervals on the per-domain PPLs would help. The 8/8 domain sweep at k=25 is convincing directionally, but the magnitude of the gap is unreliable.

4. **The adapter selection is alphabetical, not random.** `available[:k]` takes the first k adapters in sorted directory order. This means k=5 uses {abstracts, bash_code, bio_text, code, coding_style} (approximately), k=25 adds the next 20, and k=49 uses all. The k=5 set is systematically different from a random sample -- it is biased toward early-alphabet domains. A proper design would use random subsets (or at minimum, verify that alphabetical ordering does not introduce systematic bias).

5. **Pre-merge uses uniform 1/k scaling, which is the worst case.** The paper acknowledges this (Limitation 2) but does not test the obvious alternative: router-weighted merging. Since SOLE already has a router that provides per-adapter weights, the practical comparison should be weighted pre-merge vs output-averaging, not uniform 1/k pre-merge vs output-averaging. The experiment answers a straw-man version of the question.

6. **Output-averaging includes adapter-swapping overhead.** The paper acknowledges this (Limitation 5). The `apply_adapter_weights` call inside the inner loop (line 459) calls `model.update()` for each adapter, which involves dictionary operations on the full model tree. This is included in the OA latency measurement. A fairer comparison would separate model-update time from forward-pass time. However, this overhead likely accounts for only a few percent of the total time (the forward pass dominates), so this is a minor concern.

### Hypothesis graph consistency

The experiment references kill criteria K1 (id=270) and K2 (id=271), but **these node IDs do not exist in HYPOTHESES.yml**. The experiment was run without being registered in the hypothesis graph. This is a process failure -- the experiment should have been registered before execution so that the kill criteria are externally verifiable and not just self-declared.

## Macro-Scale Risks (advisory)

1. **At 7B+ scale, adapter signals are proportionally smaller.** The delta-W from rank-16 LoRA on a 7B model perturbs a larger weight space. The 1/k dilution problem will be even worse for pre-merge, potentially making the crossover happen at smaller k. Or, if the base model is already strong, adapter perturbations may be too small to matter in either method.

2. **Memory pressure from output-averaging at scale.** At k=25 on a 7B model, output-averaging requires accumulating logits across 25 forward passes. Each forward pass holds KV cache in memory. On M5 Pro with 48GB, this may force sequential execution with cache eviction between passes, adding overhead beyond the naive k-fold estimate.

3. **The practical question is moot for SOLE at top-k=2.** The paper correctly identifies this: at k=2, pre-merge is free and quality-equivalent. The output-averaging result is interesting for "always-on" composition of many adapters, but the SOLE architecture's router already selects top-k=2. The macro experiment should test whether router-weighted pre-merge at k=2-5 already captures the full benefit, making output-averaging unnecessary.

## Verdict

**PROCEED**

The experiment cleanly answers its stated question: output-averaging beats pre-merge at k>=25 due to 1/k dilution, while pre-merge wins at k=5 and is 10-100x faster. The conclusion that SOLE should continue using pre-merge at top-k=2 is well-supported.

However, the following issues should be noted in FINDINGS.md caveats:

1. The aggregate -11.5% PPL gap at k=25 is inflated by the eli5 outlier (-55.4%). Excluding eli5, the gap is approximately -9.1%. The effect is real but the magnitude is overstated.
2. Only 8 of 49 composed domains are evaluated. Global benefit is not established.
3. Adapter selection is alphabetical (not random), introducing potential ordering bias.
4. The comparison uses uniform 1/k pre-merge (worst case). Router-weighted pre-merge would be the fair comparison for SOLE deployment.
5. Kill criteria K1/K2 reference hypothesis IDs (270, 271) that do not exist in HYPOTHESES.yml. Register before closing.
6. 5 samples per domain -- per-domain deltas are directional only. The 8/8 sweep at k>=25 is the robust signal, not the percentage magnitudes.
