# Peer Review: prune_before_compose (exp_prune_before_compose_e2e)

## NotebookLM Findings

Skipped per reviewer instructions.

## Mathematical Soundness

### Theorem 2.1 (Dead Capsule Identity Conservation)

The claim relies on empirical measurements from Exp 16 (Jaccard=0.895,
overlap=0.986), not a formal proof. This is acceptable -- the paper correctly
labels it "Theorem (Exp 16)" indicating it is an empirical observation.

The "proof sketch" in Section 2.2 is sound in its logic: ReLU is element-wise,
so each capsule's activation depends only on `a_{k,i}^T x_l`. The perturbation
argument (that delta_l from other domains' residuals does not flip deeply-dead
capsules) is plausible but not formally bounded. The paper acknowledges this
with the 0.986 overlap coefficient rather than claiming 1.0.

**Minor issue**: The proof sketch argues about perturbation magnitude but never
bounds `|delta_l|` relative to `|a_{k,i}^T x_l|`. At N_d=2 this is fine
empirically, but the argument has no formal extension to N_d >> 2. The paper
acknowledges this in Assumptions and Limitations (items 1, 5). Acceptable.

### Theorem 2.4 (Quality Equivalence Bound)

The bound `|L_A - L_B| <= C * |S_B \ S_A| * epsilon_margin` is stated but C
is undefined ("a constant depending on calibration effectiveness"). This is not
a theorem -- it is an informal scaling argument. The paper should label it as
such. However, the empirical result (+0.01%) makes the bound's tightness
irrelevant for the kill criterion. **Not blocking.**

### FLOP Analysis (Section 4)

The FLOP calculations are self-consistent. Pipeline B's claimed 24% wall-clock
savings from parallel profiling is correct: 2.7G parallel vs 5.4G + 1.4G for
Pipeline A. The calibration cost difference (2.5G vs 1.4G) is correctly noted
as a tradeoff -- Pipeline B's composed model is larger than Pipeline A's
post-prune model during calibration. The net savings are real but modest.

### Print Label Bug

Line 378 of `prune_before_compose.py` says `"Pruning gap (B finds fewer dead)"`
but the value is positive (+6.0pp), meaning B finds MORE dead. The PAPER.md
correctly states "Pipeline B prunes 6.0pp MORE aggressively." This is a
cosmetic bug in the output label that does not affect results or analysis.

## Novelty Assessment

### Prior Art

This experiment is an **engineering pipeline validation**, not a novel
algorithm. The question "can we prune before composing?" follows directly from
Exp 16's identity conservation result. No external prior art is needed because
this is an internal protocol optimization.

The closest external reference is DARE merging (references/dare-merging), which
randomly prunes delta parameters before merging. DARE operates in weight space
with random selection; this experiment operates in activation space with
targeted (dead-neuron) selection. The mechanisms are different enough that
reinvention is not a concern.

### Delta Over Existing Work

The contribution is practical: validating that contributors can profile and
prune independently before shipping weights. This is a workflow optimization,
not a theoretical advance. Appropriate for a micro pipeline experiment.

## Experimental Design

### Does It Test the Hypothesis?

Yes. The hypothesis is "pre-prune-then-compose quality degrades >2% vs
compose-then-prune." The experiment directly compares these pipelines with
matched calibration budgets.

### Controls

**Adequate.** The experiment includes:
- Joint training (upper bound)
- Pipeline A: compose-then-prune (the baseline being challenged)
- Pipeline B: prune-then-compose (the proposed pipeline)
- Pipeline B2/B3: alternative profiling strategies (robustness check)
- Pipeline C: compose + calibrate, no pruning (isolates pruning effect)

The controls cover the main confounds.

### Could a Simpler Explanation Account for the Result?

**Yes, and the paper partially acknowledges this.** Finding 4 states
"Calibration completely absorbs pruning differences." Pipeline C (no pruning
at all, just calibration) achieves 0.5237 -- indistinguishable from all
pruning pipelines. This means:

1. At 100-step calibration budget, pruning order does not matter because
   **calibration dominates**.
2. This is a weaker claim than "pruning order is mathematically irrelevant."
   It is possible that with zero calibration, the pipelines would differ.
3. The pre-calibration losses (0.5875 vs 0.5876) are also nearly identical,
   which IS meaningful -- it suggests pruning order truly does not matter
   even before calibration intervenes.

**This is not a flaw.** The paper claims the pipeline is validated for the
practical workflow (which includes calibration). The pre-calibration similarity
provides additional evidence beyond the kill criterion.

### Statistical Power

3 seeds with a 2% kill threshold and a 0.01% observed delta gives massive
margin (200x). The standard deviations (0.014 for loss) are small relative to
the threshold. Statistical power is sufficient for this claim.

### Potential Confound: Shared Training History

Both pipelines share the same pretrained base and fine-tuned domain models.
Only the profiling/pruning/composition order differs. This is correct
experimental design -- it isolates the variable of interest.

### Potential Confound: Calibration Seed

Both pipelines use the same calibration seed. This is appropriate for a
paired comparison but means the result is conditioned on the calibration
trajectory. The 3-seed sweep covers this adequately.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry is well-formed:
- `depends_on: [exp16_capsule_identity_tracking]` -- correct, this experiment
  is a direct consequence of Exp 16's identity conservation.
- `kill_criteria: [pre-prune-then-compose quality degrades >2% vs compose-then-prune baseline]`
  -- matches the experimental design exactly.
- `status: proven` -- consistent with the +0.01% delta being well within 2%.
- `blocks: []` -- correct, this is a terminal pipeline validation.

The evidence string accurately summarizes the results.

## Integration Risk

This experiment validates a workflow step in the composition protocol already
described in VISION.md (steps 1-4 of the Composition workflow). It does not
introduce new architectural components, so integration risk is minimal. The
result is a process optimization, not a new module.

## Macro-Scale Risks (advisory)

1. **N-domain scaling.** At N=20 domains, the cumulative perturbation
   `delta_l` from 19 other domains' residuals may flip borderline-dead
   capsules. The Jaccard=0.895 measured at N=2 may degrade. This is the
   primary macro risk. Mitigated by: the N=5 identity scaling experiment
   (exp_n5_identity_scaling) which is already planned in HYPOTHESES.yml.

2. **Calibration budget sensitivity.** If calibration is reduced (e.g., 10
   steps for faster deployment), the pruning differences between pipelines
   may surface. The pre-calibration similarity (0.5875 vs 0.5876) suggests
   this risk is low, but it should be tested at macro.

3. **SiLU/GELU inapplicability.** The paper correctly notes this is
   ReLU-specific. Macro models (Qwen, Llama) use SiLU. This pipeline
   optimization does not apply unless ReLU capsule pools are used. This is
   a known constraint of the overall approach, not specific to this experiment.

## Verdict

**PROCEED**

The experiment is well-designed, the controls are adequate, the kill criterion
is passed with 200x margin, and the claims are appropriately scoped. The
mathematical arguments are informal in places (the "theorem" in Section 2.4
should be called a "scaling argument"), but the empirical evidence is strong
enough to validate the pipeline for N=2 at micro scale.

One cosmetic fix recommended (non-blocking):
1. Line 378 of `prune_before_compose.py`: change the print label from
   "B finds fewer dead" to "B prunes more aggressively" to match the
   actual direction of the +6pp gap and the PAPER.md text.
