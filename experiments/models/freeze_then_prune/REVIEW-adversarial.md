# Peer Review: Freeze-Then-Prune Protocol

## NotebookLM Findings

Skipped -- manual deep review performed instead, as the experiment is straightforward and the results are clear-cut.

## Mathematical Soundness

**What holds:**

1. The decomposition of the mid-training dead set into permanent and transient components (Section 3.1) is correct and well-motivated. The notation D^l_{S_mid} = D^l_permanent union D^l_transient(S_mid) is sound.

2. The ground-truth argument (Section 3.2) is logically valid: after training completes with no further weight updates, the profiled dead set IS the permanent dead set by definition. No hidden assumptions here.

3. The computational cost analysis (Section 5) is reasonable -- 6 runs per seed at ~3500 steps each, 3 seeds, ~63 seconds total. Well within micro budget.

4. The worked numerical example (Section 6) correctly illustrates the expected dynamics and correctly predicts that Kill 1 would trigger while Kill 2 would not.

**What needs attention:**

5. **The MATH.md correctly identifies its own contradiction mid-derivation (Section 3.3).** The initial hypothesis was that freeze-then-prune would find MORE dead capsules, but the math shows the opposite should happen (revival reduces the dead set over time, so the end-of-training dead set is smaller than the peak). The document then acknowledges this contradiction and correctly reframes the experiment's value as being about criterion 2 (quality), not criterion 1 (yield). This is intellectually honest -- the researcher realized mid-derivation that criterion 1 would likely fail and documented this transparently rather than reformulating the hypothesis post-hoc.

6. **The revival approximation is rough.** The "excess dead" analysis (lines 418-438 in the code) estimates false positives as mid_death - ctrl_death. This assumes the identity of dead capsules is approximately stable across protocols, which is only approximately true. The Jaccard analysis from Exp 18 (0.669) confirms this is a reasonable but imperfect proxy. Not blocking -- the main kill criteria don't depend on this approximation.

7. **Assumption 4 (fresh Adam state after pruning) is correctly flagged.** Protocol B restarts Adam state with `seed=seed + 1000` after pruning. This introduces a confound: Protocol B has a disadvantage from losing momentum/variance state. Despite this disadvantage, Protocol B matches or beats Protocol A, which strengthens the finding that mid-training pruning is viable.

## Novelty Assessment

**Prior art:**

The pruning timing question has extensive prior art in the lottery ticket hypothesis literature (Frankle & Carlin 2019) and iterative magnitude pruning. However, the specific question here -- pruning timing for dead ReLU capsules in the context of inter-layer coupling revival -- is a natural follow-up from Exp 18/20 and is specific to this project's capsule architecture. No direct prior art addresses this exact question.

The finding that "prune early, then continue training" works better than "prune at end" aligns with iterative pruning literature (pruning + retraining is a well-established technique). The novelty delta is applying this to dead-capsule-specific pruning where revival dynamics are known, not the general principle.

**References REFERENCES.yml check:** The `redo-dead-neurons` and `gurbuzbalaban-neural-death` references are relevant and were cited indirectly through Exp 17-20 lineage. No missing critical prior art.

## Experimental Design

**Strengths:**

1. **Clean experimental design.** Six protocols (control + Protocol A + 4 Protocol B variants) with shared base model across all conditions within a seed. This eliminates pretraining variance as a confound.

2. **Appropriate controls.** The no-prune control establishes the quality ceiling. Protocol A and B are properly compared against it and each other.

3. **Multiple mid-training checkpoints (100, 400, 800, 1600).** This sweeps the entire training trajectory, capturing both the death peak (S~100) and the recovery phase (S~800-1600). Good design choice.

4. **Three seeds.** Adequate for directional findings at micro scale given the 7.7pp effect size on criterion 1.

5. **The experiment tests what it claims.** Kill criterion 1 asks about yield, criterion 2 about quality. Both are directly measured.

**Concerns:**

6. **Kill criterion 1 was expected to fail based on the MATH.md's own analysis.** This is documented honestly (Section 3.3, 3.4, and the note at the bottom of Section 4.3). The researcher knew before running the experiment that the death peak (~55%) exceeds the equilibrium (~47%), making criterion 1 almost certain to trigger. This is not a flaw in the experiment -- the researcher correctly identified that criterion 2 was the more informative test, and the MATH.md documents this reasoning. The experiment was still worth running to confirm the magnitude and to discover the post-prune death rate reduction (13-19%), which was a genuinely unexpected finding.

7. **The "forced efficiency" finding (post-prune death drops to 13-19%) is the most interesting result but is not the stated hypothesis.** This is a positive secondary finding that emerged from the experiment. It deserves follow-up but does not affect the kill verdict.

8. **Constant LR only.** Acknowledged in limitations. Under cosine decay (Exp 19 showed 19.6% equilibrium death), the yield difference between protocols would likely narrow because there is less transient death to exploit. This does not invalidate the finding under constant LR.

## Hypothesis Graph Consistency

The HYPOTHESES.yml node `exp_freeze_then_prune_protocol` has:
- Status: killed -- correct
- Kill criteria match MATH.md Section 4.3 and the code implementation
- Evidence claim accurately summarizes the results
- Dependency on `exp20_death_recovery_mechanism` (proven) -- correct, this experiment is a direct follow-up

The kill is properly justified: criterion 1 triggered (freeze-then-prune yields 7.7pp FEWER dead capsules, not more). The experiment correctly concludes that freeze-then-prune is safe but unnecessary, and mid-training pruning is the superior protocol.

## Integration Risk

**Low.** This experiment does not introduce new architecture or mechanisms. It evaluates a protocol choice (when to prune) and concludes that the existing recommendation (prune after training, from Exp 18) is conservative but valid. The finding that mid-training pruning at S=800 is optimal is advisory -- it suggests a protocol improvement but does not conflict with any existing component.

The VISION.md contributor workflow (step 1: "prune dead capsules from each domain pool") does not specify timing. The finding that either timing works with equivalent quality means the workflow is robust to this choice.

## Macro-Scale Risks (advisory)

1. **Optimizer state disruption at scale.** The fresh Adam restart after mid-training pruning (Protocol B) may cause training instability at macro scale where optimizer state contains more accumulated information. The micro experiment's Protocol B uses a trivial seed offset; macro would need careful optimizer state surgery.

2. **SiLU models make this moot.** Exp 15 showed SiLU has 0% truly dead capsules. Macro models using SiLU/SwiGLU (most modern LLMs) will not benefit from dead-capsule pruning at all, regardless of timing.

3. **The "forced efficiency" regularization effect (post-prune death drops to 13-19%) is the most promising macro signal.** If confirmed at scale, mid-training pruning followed by continued training could be a useful compression strategy even for larger models with ReLU activations.

## Verdict

**PROCEED** (as a properly killed experiment)

The experiment was well-designed, honestly documented, and correctly killed. The specific findings:

1. Kill criterion 1 triggered as the MATH.md predicted it would: freeze-then-prune yields FEWER dead capsules (-7.7pp) because it profiles after the death peak has subsided due to revival.
2. Kill criterion 2 passed: quality is equivalent (+0.10%), confirming that pruning permanently dead capsules is safe.
3. Unexpected positive finding: mid-training pruning followed by continued training drops death rate to 13-19%, suggesting a "forced efficiency" regularization effect.

The kill verdict is correct. The experiment advances understanding of the pruning protocol by establishing that (a) freeze-then-prune is safe but conservative, (b) mid-training pruning at S=800 is optimal for yield+quality, and (c) the model compensates for aggressively pruned capsules during continued training.

No revisions needed. The HYPOTHESES.yml node is correctly marked as killed. The PAPER.md and MATH.md are thorough and self-consistent. The findings are properly recorded in VISION.md.
