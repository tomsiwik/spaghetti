# Peer Review: Capsule Identity Tracking Across Composition (Exp 16)

## NotebookLM Findings

Skipped -- manual deep review conducted instead, as the experiment's math and code are compact enough for direct verification.

## Mathematical Soundness

### Metrics: Correct

The Jaccard, overlap coefficient, and Dice coefficient implementations match their standard definitions. The code in `capsule_identity.py` (lines 53-83) correctly handles edge cases (empty sets). The `composition_death_decomposition` function (lines 85-154) correctly splits composed dead sets by index range and re-indexes domain B's capsules.

### Index Correspondence: Verified

The critical assumption -- that capsule i in the single-domain model corresponds to capsule i in the composed model -- is verified by inspecting `compose_relu_models()` in `micro/models/relu_router/test_composition.py` (lines 98-107). The function concatenates A matrices along axis 0 and B matrices along axis 1, so domain A's capsules occupy indices [0, P-1] and domain B's occupy [P, 2P-1]. The decomposition code correctly uses `c < P` and `c >= P` with re-indexing `c - P` for domain B. This is sound.

### Null Model Derivation: Correct

The independent-death null model (MATH.md Section 4.3) correctly computes the expected Jaccard under independence as 0.416. The observed 0.895 is well above this, confirming non-independent death patterns. The derivation is straightforward and correct.

### Expected Value Predictions: Internally Consistent

MATH.md Section 4.2 predicts E[J] = 0.867 from Exp 10's aggregate finding. The measured J = 0.895 exceeds this prediction, which is plausible because the prediction assumed exact proportionality between aggregate and per-capsule rates, while in practice the overlap can be higher if the composition-killed capsules are a strict subset of borderline cases.

### Hidden Assumption: Same Seed for Training and Profiling

The profiling seed equals the training seed in all runs (line 109 of the test file: `seed=seed` passed to both `train()` and `profile_activations()`). This means the profiling batches are deterministic given the seed, and the same seed produces the same batch ordering for both the single-domain and composed profiling. This is actually **correct** for this experiment -- the profiling is deterministic by design (Assumption 1 in MATH.md), and using the same seed ensures comparable sampling. The 2.6-3.8% profiling noise from Exp 12 confirms this is not a concern.

### Potential Issue: Profiling Data Distribution Mismatch

Single-domain models are profiled on own-domain validation data. The composed model is profiled on **joint** validation data (mixing both domains). This means the composed model's profiling includes inputs from both domains, while each single-domain model only sees its own domain. The paper partially addresses this with the "composed on own-domain data" control (lines 161-174 of the test file), which gives slightly higher Jaccard (0.909 vs 0.897 for domain A). The difference is small, but the primary "combined Jaccard" metric (0.895) is computed against joint-data profiling, which is the operationally relevant comparison (you would profile a composed model on joint data in practice).

**Verdict on math: Sound.** No errors found. The metrics are standard, the decomposition logic is correct, and the null model comparison is valid.

## Novelty Assessment

### Prior Art

This experiment is **incremental but necessary** within the project's research arc. It fills the gap between Exp 10 (aggregate death rates) and Exp 18 (temporal identity stability). No external prior art directly addresses "Jaccard overlap of dead ReLU neuron sets between independently-trained and composed models" -- this is a project-specific question.

The closest external reference is Gurbuzbalaban et al. (2024) on neural death dynamics, which the paper correctly cites. The references manifest (`references/gurbuzbalaban-neural-death/`) is properly linked in HYPOTHESES.yml.

### Delta Over Existing Work

The key contribution is not a new mechanism but a **measurement**: death identity is more stable across composition (J=0.895) than across training time (J=0.669 from Exp 18). This measurement directly enables the revised pruning protocol (profile before compose instead of after). The delta is practical, not theoretical.

### No Reinvention

The experiment reuses existing infrastructure: `profile_activations` from Exp 9, `compose_relu_models` from the relu_router module, and `get_dead_set` is a thin utility. No reinvention of standard mechanisms detected.

## Experimental Design

### Does It Test the Stated Hypothesis?

**Yes.** The hypothesis is "per-capsule death identity is preserved across composition." The experiment directly measures Jaccard overlap of dead sets between single-domain and composed models. The kill criterion (J < 0.50) is the stated test. The experiment passes with J = 0.895.

### Controls: Adequate

Two controls are included:

1. **Cross-domain profiling** (same model, different data distribution): J = 0.916-0.926. This establishes that changing input distribution alone causes ~8% dead-set change, so the composition effect (J = 0.895, ~10.5% change) adds only ~2.5pp beyond what data distribution shift explains. This is a good control.

2. **Composed on own-domain data**: J = 0.899-0.909. This isolates the composition effect from data distribution mixing. The fact that it is close to the joint-data number confirms the result is robust.

### Could a Simpler Mechanism Explain the Result?

**Yes, partially.** The high Jaccard could be partly explained by the high base death rate (~58%). When ~58% of capsules are dead, random overlap alone gives J = 0.416 (the null model). The measured J = 0.895 is 2.15x the null model -- a large effect, but the absolute margin matters. With 512 capsules and 58% death, only ~33 capsules (6%) differ between settings. The experiment correctly decomposes this into ~29 killed and ~4 revived.

The key question: is this trivially explained by "most capsules are so deeply dead they can't be perturbed"? MATH.md Section 5.1 argues this explicitly -- deeply dead capsules have large negative margins, so small perturbations cannot flip them. This is a valid mechanism, but it also means the result is **less informative** about borderline capsules, which are the ones that matter for pruning decisions. The paper acknowledges this (limitation 4: binary profiling at f=0 threshold).

**This is not a blocking concern** -- the practical conclusion (pre-composition profiling is sufficient for pruning) still holds because the missed capsules (~29 composition-killed, 6%) represent a missed pruning *opportunity*, not a quality risk.

### Statistical Concerns

Three seeds is thin. The per-domain Jaccard ranges from 0.823 to 0.938 (range = 0.115). Layer 0 has std = 0.346 for domain A, indicating one seed had J = 1.0 (trivially: 0 dead capsules in both) and another had a much lower value. The paper correctly flags this variance but does not compute confidence intervals. At n=3, confidence intervals would be wide and not particularly informative, so this is acceptable for a micro experiment.

### Hypothesis Graph Consistency

The experiment matches its HYPOTHESES.yml node `exp16_capsule_identity_tracking` (line 80). The kill criterion matches: "per-capsule death identity overlap <50% between single-domain and composed." The evidence correctly summarizes the result. Status is "proven." Consistent.

## Macro-Scale Risks (advisory)

### 1. N-Domain Scaling (High Risk)

The paper correctly identifies this as the primary macro risk. With N=2 domains, the perturbation from composition is small (one other domain's capsule residuals). At N=5-20, the perturbation grows linearly. If the perturbation magnitude exceeds the margin of borderline capsules, the Jaccard could drop significantly. The paper's claim "pre-composition profiling is sufficient" may not hold at N>5.

**Recommendation for macro**: Measure Jaccard at N=5 before adopting the pre-composition pruning protocol.

### 2. SiLU Incompatibility (Already Acknowledged)

SiLU has 0% dead capsules (Exp 15). The entire framework does not apply. Macro models using SiLU (which is most production models) cannot use this approach. The paper acknowledges this.

### 3. Domain Similarity

The two domains (names a-m vs n-z) are nearly identical in distribution (character-level name generation with different character sets). Real domain pairs (code vs prose) would produce larger perturbations in the hidden state, potentially reducing Jaccard. The N=2 same-task result is a best case.

### 4. The 6% Missed Pruning Opportunity Compounds

At N=20 domains, if each domain half introduces 6% newly-dead capsules, the cumulative missed-pruning set could grow substantially. The 6% figure is specific to N=2.

## Verdict

**PROCEED**

The experiment is well-designed, the math is sound, the code correctly implements the stated metrics, and the controls are adequate. The primary result (J = 0.895, well above the 0.50 kill threshold) directly answers the stated question: the same capsules die in both single-domain and composed settings. The practical implication (pre-composition profiling enables prune-then-compose) is a genuine workflow improvement.

The main weaknesses are:

1. Only N=2 domains (acknowledged as limitation 1, best-case for overlap)
2. Very similar domains (acknowledged as macro risk)
3. Only 3 seeds with moderate variance in per-layer metrics

None of these are blocking for a micro-scale experiment. The mechanism (deeply dead capsules have large negative margins that small perturbations cannot flip) is physically sound and well-argued in MATH.md Section 5.1. The comparison to Exp 18's temporal Jaccard (0.669) provides useful context: composition is a smaller perturbation than continued training.

The experiment earns its place in the lineage by converting Exp 10's aggregate measurement (87% training-induced death) into a per-capsule identity measurement, and by demonstrating that cross-setting stability exceeds cross-time stability. This directly enables the revised pruning protocol proposed in the paper.
