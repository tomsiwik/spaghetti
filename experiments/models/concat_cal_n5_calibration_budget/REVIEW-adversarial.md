# Peer Review: Concat+Calibrate N=5 Calibration Budget

## NotebookLM Findings

Manual review conducted. NotebookLM deep review deferred; the experiment is
a parameter sweep with no novel mechanism to stress-test. The key question is
whether the experimental design correctly falsifies the stated hypothesis.

---

## Mathematical Soundness

### What holds

**The parameter-to-data ratio analysis is correct.** 1,280 router parameters
against 102K-512K tokens (80x-400x ratio) rules out classical statistical
underfitting. The paper correctly identifies that the failure mode is
optimization instability, not insufficient data.

**The N=2 degeneracy observation is insightful and correct.** With top_k=2
and N=2, every expert is always selected. The router reduces to a learned
weighted average, which is strictly more expressive than uniform averaging.
This correctly explains why concat+cal wins at N=2 but fails at N=5.

**The non-monotonicity argument is sound.** If the failure were underfitting,
more steps should monotonically improve performance. The 300 < 100 < 200 < 500
ordering (by quality) falsifies the underfitting hypothesis. The paper
correctly attributes this to optimization landscape issues rather than
data insufficiency.

### Minor issues

**The round-robin gradient conflict analysis (MATH.md, point 3) is stated
but not quantified.** The claim that domain cycling creates "oscillatory
behavior" is plausible but could be tested by measuring router weight norms
or gradient cosine similarity across domains. This is a missed opportunity for
deeper mechanistic understanding, not a mathematical error.

**The "fundamental capacity limitation" (MATH.md, point 4) is underexplored.**
The paper notes 4,096 params per expert but does not measure inter-expert
similarity (e.g., cosine distance between expert delta vectors). If experts
are near-identical, routing cannot add value regardless of optimization
quality. The lora_procrustes experiment showed cos ~ 0.014 at N=2, but no
equivalent measurement is reported for N=5. This would have been a cheap
and informative diagnostic.

---

## Novelty Assessment

**This is a hyperparameter sweep, not a novel mechanism.** No novelty is
claimed, and none is needed -- the experiment exists to falsify the
"router underfitting at N=5" hypothesis. This is appropriate ablation
science.

**Prior art check:** The MoRAM reference (references/moram-associative-memory)
proposes eliminating router calibration entirely via self-routing LoRA
adapters. The paper does not cite this as a potential alternative to
calibrated routing. However, MoRAM's approach is structurally different
(associative memory retrieval vs softmax routing), so this is not a
novelty issue but a missed citation for context.

---

## Experimental Design

### Strengths

**The hypothesis is cleanly falsifiable.** Two kill criteria are stated
upfront, both are tested, and both trigger. This is textbook ablation design.

**Controls are adequate.** Joint training provides the upper bound, simple
average provides the zero-compute baseline, and 3 seeds provide directional
evidence of stability. The per-seed detail table is especially valuable --
it exposes the high variance at 500 steps (seed 123 at +7.15%) that the
aggregate would hide.

**The code correctly isolates the variable under test.** Only cal_steps
varies; everything else (LR, batch size, top_k, architecture, domain
splits, seeds) is held constant. The router is correctly frozen except
for routing weights during calibration (lines 155-157 of test code).

### Issues

**The learning rate is fixed at 3e-3 across all budgets.** MATH.md mentions
this as a limitation, and the paper correctly notes it is unlikely to change
the picture for 1,280 parameters. However, the non-monotonicity could
partially be an artifact of overshooting with a fixed LR at higher step
counts. A simple cosine decay schedule would cost one line of code and
would either confirm or rule out this confound. The experiment would be
more convincing with this control.

**The round-robin domain cycling uses `step % 5`, not shuffled batches.**
Line 165: `ds = all_train_ds[step % len(all_train_ds)]`. This means the
domain order is deterministic and identical across all seeds. If the
non-monotonicity is caused by domain cycling order, the experiment cannot
distinguish "5 domains is fundamentally unstable" from "this particular
cycling order is unstable." A random domain sampling strategy would be a
trivial control. This does not invalidate the kill, but it weakens the
claim that the instability is fundamental to N=5.

**No learning curve analysis in the paper.** The code records loss
checkpoints (lines 180-185), but the paper never discusses them. These
curves would directly show whether training loss also exhibits
non-monotonicity (optimization instability) or whether only validation
loss degrades (generalization failure). This distinction matters for
understanding the mechanism.

---

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry matches the experiment:
- `exp_concat_cal_n5_calibration_budget` depends on `exp_lora_merging_bakeoff`
- Kill criteria match exactly: (1) 500-step gap > 3%, (2) no budget beats
  simple average
- Evidence entries correctly report the results
- Status is `killed`

This is consistent and properly maintained.

---

## Macro-Scale Risks (advisory)

**The kill may not transfer to macro.** The paper's limitations section
correctly identifies three reasons the mechanism could revive at scale:
1. Domain-discriminative hidden states (d=256+, BPE tokens)
2. Higher LoRA rank creating more differentiated experts
3. Better optimization (LR scheduling, gradient accumulation)

Additionally: at macro scale with genuinely distinct domains (code vs prose),
the round-robin gradient conflict may resolve naturally because hidden states
become domain-separable and the router's task becomes easier.

**The "simple average as N>=5 default" conclusion should be tested at macro.**
Task arithmetic (simple averaging of LoRA deltas) is known to degrade with
increasing N in the literature (TIES-Merging, DARE). The micro-scale
result that simple average beats routing at N=5 may be specific to the low
expert differentiation in the character-level setting.

---

## Verdict

**PROCEED** (as a completed, killed experiment)

The experiment correctly falsifies its stated hypothesis. The kill is
justified: both kill criteria trigger, the non-monotonic curve provides
strong mechanistic evidence, and the per-seed analysis reveals genuine
optimization instability. The paper's conclusions are appropriately scoped
and the limitations section is honest about macro-scale revival potential.

Two minor improvements would strengthen the record, but neither changes the
verdict:

1. Add a cosine-decay LR control for the 500-step run to rule out the
   fixed-LR confound. This is one line of code and ~7 seconds of compute.
2. Report the learning curve checkpoints that are already collected in the
   code but never discussed. This costs zero additional compute.

The experiment advances the project's understanding by establishing that
router optimization instability (not underfitting) is the mechanism behind
N=5 concat+cal failure, and by confirming simple averaging as the N>=5
default at micro scale. This is a clean negative result with clear
implications for future routing experiments.
