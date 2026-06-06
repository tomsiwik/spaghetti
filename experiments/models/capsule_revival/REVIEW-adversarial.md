# Peer Review: Per-Capsule Revival Tracking (Exp 18)

## NotebookLM Findings

Skipped -- NotebookLM authentication not configured in this environment. Review proceeds with direct document analysis.

## Mathematical Soundness

### Transition Matrix Formalism (MATH.md Section 3)

The transition matrix definitions are correct. Conservation law n_DD + n_DA + n_AD + n_AA = P_total holds by construction. The revival and death rate definitions (Section 3.2) are standard conditional probabilities.

**Verified**: The aggregate death rate change decomposition delta(S_2) - delta(S_1) = (n_AD - n_DA) / P_total is algebraically correct:

```
|D_{S_2}| = n_DD + n_AD
|D_{S_1}| = n_DD + n_DA
|D_{S_2}| - |D_{S_1}| = n_AD - n_DA
=> (|D_{S_2}| - |D_{S_1}|) / P_total = (n_AD - n_DA) / P_total
```

This checks out.

### Jaccard Analysis (MATH.md Section 4)

**Verified**: The null model expected Jaccard (Section 4.2 Hypothesis C) is correctly derived:

```
E[J_null] = (0.55 * 0.47) / (0.55 + 0.47 - 0.55 * 0.47) = 0.2585 / 0.7615 = 0.3394
```

Paper rounds to 0.340. Correct.

**Minor issue**: The notation in Section 4.1 uses `|D_{S_1} | D_{S_2}|` where the pipe character is ambiguous -- it should be the union operator. From context this is clearly union, but the notation is sloppy. Non-blocking.

### Cohort Survival Analysis (MATH.md Section 5)

The survival curve definition is clean: survived(S) + revived(S) = 1.0 for all S >= 100, by construction (the anchor cohort is partitioned into still-dead and not-still-dead).

**Issue with the decomposition (PAPER.md "Decomposition of Aggregate Decrease")**: The paper reports:

```
Revival contribution to aggregate decrease: 15.8 pp
New deaths avoided (from gradient shrinkage): -3.8 pp
```

The "revival contribution" is computed as: (28.1% of cohort revived) * (287/512 cohort fraction) = 0.281 * 0.561 = 15.8 pp. This is correct.

But the interpretation that "Revival contribution EXCEEDS the aggregate decrease (12.0 pp)" is potentially misleading. The -3.8 pp labeled "new deaths avoided" is actually NET NEW DEATHS -- more capsules died than the revival accounted for. The sign is negative because new deaths increased, not decreased. The phrasing "new deaths avoided (from gradient shrinkage)" is confusing when the actual value is negative (meaning new deaths were NOT avoided -- they occurred and partially offset revival).

This is a presentation issue, not a mathematical error. The numbers are internally consistent: 15.8 - 3.8 = 12.0 pp net decrease. But the narrative around "gradient shrinkage" explaining a negative offset is contradictory -- if gradients were shrinking and causing fewer new deaths, this term should be positive.

**Correction needed**: The -3.8 pp means that new deaths (capsules alive at S=100 that are dead at S=3200) actually INCREASED the total dead count by 3.8 pp. The aggregate decrease of 12.0 pp = 15.8 pp (cohort revival) - 3.8 pp (new deaths among previously-alive capsules). The label should be "new death offset" not "new deaths avoided."

### Implementation Verification

The code in `test_capsule_revival.py` correctly implements the math:

- `get_dead_mask()` (line 49-67): Correctly converts frequency arrays to binary masks using `f <= threshold`. The `<=` for threshold=0.0 means exactly-zero-frequency capsules are marked dead. Consistent with the binary classification assumption in MATH.md.

- `transition_counts()` (line 79-94): Correctly implements the 2x2 transition matrix with exhaustive enumeration. Conservation is guaranteed by the four mutually exclusive conditions.

- `jaccard_similarity()` (line 70-76): Standard set-theoretic implementation. Handles empty sets correctly (returns 1.0).

- Cohort tracking (lines 186-203): Correctly uses set intersection/difference on the S=100 dead set.

**Important verification**: The code at line 145-151 uses `copy.deepcopy(base)` for each step count S, then trains from the same seed. This means S=100 and S=3200 are DIFFERENT training runs starting from the same base, not checkpoints of a single run. The MATH.md states (Section 6.3): "all step counts share the same training seed, so S=50 is literally the first 50 steps of the S=3200 trajectory."

This claim is correct IF `train()` uses the seed deterministically such that the first S steps of any run with seed X are identical regardless of total steps. This depends on the training loop implementation. Given that the same `seed` parameter is passed and the same `train_ds` used, the first S steps should be identical across runs (assuming no stochastic data sampling that depends on total step count). This is a reasonable assumption for micro-scale deterministic training.

## Novelty Assessment

### Prior Art

The paper cites:
1. **Gurbuzbalaban et al. (2024), "Neural revival"**: Reports >90% of revived neurons eventually die again. The paper's finding of 28.1% cumulative revival is consistent but operates at a different granularity (capsule pools vs individual neurons).

2. **Li et al. (2023), "Lazy Neuron Phenomenon"**: ~50% natural ReLU sparsity. Consistent with the 44-56% death rates observed.

3. **Lu et al. (2019), "Dying ReLU and Initialization"**: Depth-dependent death. Consistent with Layer 0 near-zero death vs Layers 1-3 at 50-80%.

### Delta Over Existing Work

The per-capsule tracking with transition matrices across training checkpoints is a **standard analysis technique** applied to a specific architecture (capsule pools with ReLU routing). The novelty is not in the method but in the finding: revival is substantial (28%), accelerates with training, and is layer-dependent.

This is an **observational experiment** rather than a methodological contribution. It produces useful empirical knowledge for the project's pruning protocol but would not stand alone as a research contribution. That is acceptable for a micro-experiment whose purpose is to inform the pruning timing decision.

### Missing Prior Art

- **Frankle et al. (2020), "Linear Mode Connectivity"**: Weight trajectory analysis across training. Relevant to understanding whether the S=100 and S=3200 models occupy different basins.
- **Zhang et al. (2022), "Planting and Mitigating Memorized Subnetworks"**: Tracks which neurons participate in memorization across training. Closely related methodology.

Neither is blocking.

## Experimental Design

### Does This Test What It Claims?

**Yes.** The experiment cleanly tests whether aggregate death decrease is due to same-capsule revival or population turnover. The three analysis tools (transition matrices, cohort tracking, Jaccard similarity) triangulate on the same question from different angles:

1. Transition matrices show per-interval revival rates increasing over time
2. Cohort tracking shows 28.1% of the S=100 dead set revives by S=3200
3. Jaccard of 0.669 confirms the dead set identity changes substantially

These three measures are consistent and mutually reinforcing.

### Controls

**Adequate for the question asked.** The null model (random Jaccard = 0.340) provides a baseline. The observed Jaccard of 0.669 is well above random and well below sticky (0.85). The 3-seed design provides minimal but sufficient variance estimates.

### Could a Simpler Mechanism Explain the Results?

**Borderline capsules flickering.** A concern not addressed in the paper: if many capsules sit near the f=0 boundary, stochastic variation in the profiling protocol (20 batches x 32 samples = 640 samples) could cause borderline capsules to flip between dead and alive across checkpoints, inflating "revival" counts.

The paper acknowledges this in Micro-Scale Limitations point 2: "Nearly dead capsules (0 < f < 0.01) may represent a borderline population that flickers between states."

However, the paper does not quantify this risk. A simple control would be: at each checkpoint, run profiling twice with different random batches and measure the false-positive revival rate from profiling noise alone. If profiling-induced flickering accounts for >50% of observed D->A transitions, the revival finding is substantially weakened.

**Severity: moderate.** The monotonically increasing revival rate (5.8% to 15.9%) and the cohort's steady decline in death (100% to 71.9%) are harder to explain by pure flickering, since flickering should produce roughly constant transition rates. But the concern is worth quantifying.

### Hypothesis Graph Consistency

The experiment is registered as `exp18_capsule_revival` in HYPOTHESES.yml with status `proven`. The kill criteria in the code match MATH.md Section 7:

| Kill Criterion | MATH.md | Code | Match? |
|---|---|---|---|
| Jaccard > 0.85 | Yes | Line 395-400 | Yes |
| Max revival < 5% | Yes | Lines 403-413 | Yes |
| Turnover < 10/seed | Yes | Lines 420-426 | Yes |

All three are correctly implemented and correctly evaluated against the data.

## Integration Risk

The experiment integrates cleanly into the project's contribution protocol. The practical recommendation ("prune after training completes, not during") is well-supported by the data and does not conflict with existing findings.

The experiment naturally connects to:
- Exp 16 (capsule identity across composition): same tracking infrastructure
- Exp 20 (layer freezing): Finding 4 (layer 3 highest revival) makes a testable prediction

No conflicts with VISION.md or existing architecture.

## Macro-Scale Risks (advisory)

1. **Revival may vanish at scale.** With larger hidden dimensions (d=896+ vs d=64), the margin between dead and alive is proportionally larger in weight space. Input distribution shifts from upstream layers may be insufficient to cross these wider margins. The 28% revival rate is a micro-scale number that may not transfer.

2. **SiLU activation changes everything.** Qwen uses SiLU, not ReLU. SiLU has no hard zero -- there is no binary dead/alive boundary. The entire concept of "revival" becomes a continuous magnitude question rather than a binary state transition. The transition matrix formalism does not apply to SiLU without substantial modification (magnitude thresholding, which introduces an arbitrary parameter).

3. **Profiling cost at scale.** 20 batches x 32 samples = 640 forward passes per checkpoint, times 8 checkpoints = 5120 forward passes per seed. At macro scale this is non-trivial compute. The paper does not discuss profiling efficiency.

4. **LR schedule interaction.** Macro training uses warmup + cosine decay. The constant-LR micro experiment may produce a different revival trajectory than cosine decay would. The paper acknowledges this (Exp 19 connection) but it remains untested.

## Specific Critiques

### 1. Decomposition Narrative Inconsistency

As noted in Mathematical Soundness: the -3.8 pp is labeled "new deaths avoided (from gradient shrinkage)" but the negative sign means new deaths were NOT avoided. The narrative about "positive feedback" from gradient shrinkage (Finding 2) is partially contradicted by the decomposition showing net new deaths offset revival. The transition table does show new death rate decreasing (14.9% to 5.5%), but the cumulative effect over 3100 steps still produces a net positive new death count that offsets 3.8 pp of the revival gain.

This is not wrong mathematically, but the narrative overstates the "gradient shrinkage prevents new deaths" story when the data shows new deaths still happen and partially offset revival.

### 2. Cohort Analysis Is Snapshot-Based, Not Trajectory-Based

The cohort analysis tracks whether a capsule dead at S=100 is alive at S=3200, but does not track the intermediate trajectory. A capsule that revives at S=200, dies at S=400, and revives again at S=3200 is counted identically to one that revives at S=3200 and stays alive. The paper acknowledges this distinction (MATH.md Section 3.3, PAPER.md per-layer note) but the practical implications are understated.

If most "revivals" are transient (consistent with Gurbuzbalaban et al.'s >90% re-death rate), the practical recommendation to "wait for revival before pruning" is weakened: the 28% that appear alive at S=3200 may die again at S=6400. The paper's limitation point 1 ("revival rate is still increasing at S=3200") partially addresses this but the connection to pruning timing reliability is not made explicit.

### 3. Single Domain Only

The experiment tracks revival in a single domain (a_m) fine-tuning setting. The project's composition protocol involves multiple domains. Revival dynamics in a composed model (where cross-domain inputs may push capsules in different directions) are untested. This limits the direct applicability of the "prune after training" recommendation to the composition setting.

## Verdict

**PROCEED**

The experiment is well-designed, correctly implemented, and the math is sound. It cleanly answers the question it set out to answer: aggregate death decrease is dominated by true same-capsule revival (28.1%), not population turnover. The three analysis tools (transitions, cohort tracking, Jaccard) triangulate consistently. Kill criteria are appropriate and correctly evaluated. The practical recommendation (prune after training, not during) follows logically from the data.

The issues identified are presentation-level (decomposition narrative) and scope-level (single domain, flickering control missing), none of which threaten the core finding.

Minor fixes recommended but not blocking:

1. Fix the decomposition narrative: relabel "new deaths avoided" as "new death offset" and correct the sign interpretation. The -3.8 pp means new deaths occurred, not that they were avoided.
2. Add a sentence in Limitations noting that borderline capsule flickering from profiling noise has not been quantified, and suggest the two-run profiling control for future work.
3. Clarify in the pruning recommendation that revival dynamics are measured in single-domain fine-tuning only; composed models may differ.
