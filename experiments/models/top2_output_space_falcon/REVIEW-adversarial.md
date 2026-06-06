# Peer Review: Top-2 Output-Space Composition on Falcon-E-3B

## NotebookLM Findings

Skipped -- this experiment is already KILLED by the researcher with thorough documentation. The review focuses on validating the kill decision and extracting actionable signal.

## Mathematical Soundness

### MATH.md: Mostly Correct, One Misleading Claim

1. **No cross-terms argument (Section 0, Property 1): Correct.** Output-space composition does eliminate cross-term interactions through nonlinearities. The LoRI citation (arXiv:2504.07448) is appropriate. Each adapter's output is computed in isolation; only the final logit average mixes signals linearly.

2. **Effective amplification factor (2.5x): Correct but misleading.** The derivation that top-2 gives each adapter 1/2 effective weight vs. uniform-5 merge giving 1/5 is arithmetically right. But this framing conflates two things: (a) the adapter signal strength per selected expert, and (b) the total adaptation magnitude in the output. With top-2, you get contributions from 2 of 5 adapters. With uniform merge, you get contributions from all 5. The per-adapter signal is stronger in top-2, but total adaptation bandwidth is narrower. The MATH.md frames 2.5x as a pure advantage, but it is a tradeoff -- you gain signal concentration at the cost of coverage. This is fine when routing is correct (which is the MoE bet), but the framing overpromises.

3. **"Superlinear is possible" claim (Section 0): Imprecise.** The document states superlinear gains are possible because each adapter contributes its full delta. But "superlinear" in the MoE literature (and in the cited arXiv:2506.13479) means the composed output exceeds what any single expert achieves alone -- it does not mean "stronger signal than parameter merge." The MATH.md conflates two definitions of superlinear. The experiment correctly tested the standard definition (beat single best) and found it fails on 4/5 domains.

4. **Complexity analysis (Section 5): Correct.** The 2x FLOP overhead for sequential output-space is right. However, the MATH.md did not predict the 17x actual overhead from adapter-swap mechanics. This is a significant gap between the idealized analysis and reality.

5. **Assumption 4 (latency budget): Self-contradictory.** The text estimates ~25 tok/s for top-2 based on ~50 tok/s per pass, then notes K2 threshold is 30 tok/s. The MATH.md already contains the seed of its own kill -- the speed assumption was marginal before accounting for adapter-swap overhead. This should have been flagged as a high-risk assumption before running the experiment.

### Missing from MATH.md

No failure mode analysis for the case where adapters are individually harmful. Section 3 ("What Breaks It") considers wrong routing and slow speed, but not the possibility that ALL adapters degrade performance. This turned out to be the primary failure mode. The impossibility structure in Section 0 addresses cross-term interference (a composition problem) but not the more fundamental question: what guarantees that adapters improve over the base in the first place?

## Novelty Assessment

**Low novelty.** Output-space composition (logit averaging) is standard MoE practice. The contribution here is applying it to full-model LoRA adapter swaps rather than per-layer FFN expert blocks. This is a valid engineering question, but the math is not new -- the LoRI cross-term elimination proof and MoE output-space composition are prior art that this experiment applies rather than extends.

The keyword-routing mechanism is ad hoc and not derived from the Grassmannian skeleton or any learned representation. This disconnects the experiment from the architecture's core innovation.

## Experimental Design

### Critical Flaw: Wrong Evaluation for the Adapter Type

This is the most important issue. The adapters were trained with next-token prediction (NTP) on domain text. The evaluation is MMLU (multiple-choice QA). The researcher correctly identified this post-hoc, but it represents a fundamental experimental design flaw: **the experiment does not test output-space composition; it tests whether NTP adapters help on QA benchmarks.** The composition mechanism was never given a fair trial because the individual components were defective for the evaluation task.

A proper test would have:
- Evaluated on domain perplexity (where adapters are known to help), OR
- Used QA-fine-tuned adapters, OR
- At minimum, verified that single adapters improve over base before testing composition

The paper acknowledges this, but the kill criteria were written without this check. K1 ("OS-top2 beats single best adapter") can pass even if both are worse than base, which is the wrong comparison for "superlinear composition." The kill criteria should have included: "Single best adapter must first beat base by >X on >=3/5 domains."

### Sample Size Problem

n=20 per domain is acknowledged as a limitation. With 20 binary trials, the 95% CI on accuracy of 0.55 is roughly [0.32, 0.76] (exact binomial). Most of the differences in the results table (e.g., 0.55 vs 0.40, 0.60 vs 0.50) are well within noise. The one "positive" finding -- keyword-routed OS-top2 at 0.65 on math vs 0.55 base -- represents 13 vs 11 correct answers out of 20. The p-value for this difference (one-sided Fisher exact) is approximately 0.37. **This is not statistically significant.** The paper treats it as a meaningful signal ("the ONE domain where composition helped"), but it is noise.

### Oracle Routing Confound

The "output_space_top2" condition uses oracle routing (always includes the correct domain adapter). This gives the method its best possible shot. Despite this advantage, it still fails K1. This is actually stronger evidence for killing than if a learned router had been used, because it rules out "bad routing" as the explanation.

### Speed Measurement

The speed comparison is valid but the implementation is deliberately naive (acknowledged in code comments). The 17x overhead is real for this implementation but not a fundamental limit of output-space composition. The paper correctly notes that KV cache reuse, batching, or dual model copies could improve this. The kill on K2 is implementation-dependent, not mechanism-dependent.

## Confounds and Alternative Explanations

1. **LORA_SCALE = 20.0 is extremely high.** This means each adapter delta is amplified 20x before being added to weights. With adapters trained for NTP, this could inject very strong domain bias that overwhelms the instruction-tuned base's calibration. A lower scale (e.g., 1.0-5.0) might produce adapters that slightly modulate rather than override the base. The experiment did not sweep this parameter.

2. **Ternary base + bf16 unpack.** The BitLinear-to-Linear replacement changes the model's numerical behavior. The adapters were trained on the ternary model but evaluated after unpacking to bf16. If there is any systematic difference in how the unpacked model processes adapter deltas vs. the original ternary model, this could explain degradation.

3. **Complementary pairing is hand-coded.** The oracle routing uses hand-picked "complementary" pairs (medical+code, math+code, etc.). If these pairings are wrong, the experiment is testing bad pairs, not the composition mechanism. However, since even oracle single-adapter degrades performance, this confound is secondary.

## Macro-Scale Risks (advisory)

1. **Output-space composition requires architectural support at scale.** The naive adapter-swap approach is dead. Any macro implementation needs either (a) separate model copies per expert (memory-expensive), (b) per-layer MoE blocks (architectural change), or (c) KV-cache-aware adapter application (engineering challenge). The VISION.md architecture already points toward per-token routing with expert blocks, which is the right direction.

2. **The NTP-vs-QA adapter training mismatch is a systemic risk.** Three experiments have now confirmed that NTP-trained adapters degrade instruction-tuned bases on benchmarks. Any macro experiment using the current adapter training pipeline will hit the same wall. The training objective must change before composition mechanisms can be properly evaluated.

3. **The lora_scale=20.0 question needs resolution.** Before scaling up, verify that the adapter scale is calibrated correctly. An ablation over scale values would be cheap.

## Kill Criteria Validation

- **K1 FAIL: Correct.** OS-top2 beats single on 1/5 domains (threshold >=3). Even granting the noise caveat (n=20 is too small for firm conclusions), the direction is clearly wrong -- no method beats base on average.
- **K2 FAIL: Correct but implementation-dependent.** 2.7 tok/s vs 30 tok/s threshold. The 17x overhead is real for this implementation. However, a well-engineered version could potentially hit 2x overhead (~22 tok/s), which would still fail K2 but by a smaller margin.

## Actionable Findings

1. **The single most important lesson: verify adapters improve over base before testing composition.** This should be a prerequisite check (K0) for all future composition experiments.

2. **The math+code pairing hint is statistically worthless at n=20.** Do not design follow-up experiments based on this signal. If pursued, replicate at n>=100 per domain first.

3. **Output-space composition is architecturally sound but needs different infrastructure.** The cross-term elimination is real. The mechanism is not falsified -- it was never properly tested because the adapters were broken. Future work should use the BitNet-2B-4T base (not instruction-tuned Falcon) where adapters are known to improve perplexity.

4. **The 17x speed overhead is an engineering problem, not a theoretical one.** Worth revisiting if/when KV-cache-aware adapter application is implemented.

## Verdict

**KILL -- Correct.**

The researcher's kill decision is well-justified on both kill criteria. The experiment is thorough in its analysis and the LEARNINGS.md correctly identifies the root cause (adapter training objective mismatch, not composition mechanism failure).

However, the experiment has a fundamental design flaw: it tested composition of broken components. The output-space composition mechanism itself was never properly evaluated. This means the kill applies to "output-space top-2 with NTP adapters on instruction-tuned Falcon," not to "output-space composition in general." The PAPER.md and LEARNINGS.md correctly make this distinction, which is good scientific practice.

**If the experiment were to be revived, it would need:**
1. A prerequisite gate: single adapter must beat base on >=3/5 domains before composition is tested
2. Either QA-trained adapters OR perplexity-based evaluation (not MMLU with NTP adapters)
3. An adapter scale sweep (lora_scale in {1, 5, 10, 20})
4. n>=100 per domain for MMLU evaluation
5. KV-cache-aware implementation for speed testing

These are not revisions to the current experiment -- they define a different experiment. The current one is correctly killed.
