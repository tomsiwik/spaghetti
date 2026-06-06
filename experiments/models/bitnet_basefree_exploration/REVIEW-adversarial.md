# Peer Review: bitnet_basefree_exploration

## NotebookLM Findings

NotebookLM step skipped. The experiment is a clean ablation study with
straightforward methodology. The mathematical claims are verifiable directly
from the code and results.json. No complex derivations requiring external
review.

## Mathematical Soundness

**Correct:**

1. The criticality metric c_l = (PPL(M_{W_l=0}) - PPL(M)) / PPL(M) is
   well-defined and the per-layer ablation correctly implements it (code
   lines 281-287, 484-485). Each layer is zeroed independently, PPL is
   measured, then weights are restored before the next layer.

2. The progressive ablation correctly sorts layers by ascending criticality
   and zeros them cumulatively (code lines 522-548). The zeroed_layers lists
   in results.json confirm the ordering matches individual criticality scores.

3. The norm-matching scaffold replacement is correctly implemented:
   ternary ~ Uniform{-1,0,1}, then scaled by orig_norm/new_norm per weight
   matrix (code lines 310-316). The MATH.md analysis of why norm-matching
   is insufficient (spectral structure, cross-layer coherence, attention
   patterns) is standard and correct.

4. The adapter norm analysis is sound: with alpha=20, r=16, typical A~0.02,
   B~0.01, the adapter Frobenius norm is ~0.016 vs base norm ~50. The
   0.03% ratio correctly explains why adapters are invisible on random
   scaffolds.

**Issue -- "Super-exponential" fit is under-specified:**

The MATH.md claims PPL ratio ~ exp(beta * K^gamma) with gamma ~ 2.1 but
provides no fitting procedure, confidence interval, or residuals. Checking
the data manually:

| K  | ln(ratio) | Predicted K^2.1 (scaled) |
|----|-----------|--------------------------|
| 1  | 0.025     | ~1                       |
| 3  | 0.121     | ~10.5                    |
| 5  | 0.338     | ~29.4                    |
| 10 | 1.190     | ~125.9                   |
| 15 | 3.335     | ~283.4                   |
| 20 | 12.69     | ~508.5                   |

The progression is clearly super-linear in K, but the specific gamma=2.1
is a rough hand-fit. This is not wrong -- the qualitative claim (each
additional zeroed layer is worse than the last) is well-supported by the
data -- but the paper should not present it as a fitted parameter without
methodology. Minor issue; does not affect any conclusion.

**Issue -- K2 threshold mismatch between title and body:**

The PAPER says "K2: PASS -- 23/30 layers can be individually zeroed without
>20% regression." But the title says "Grassmannian scaffold replaces
pretrained base weights." K2 tests whether ANY layer is non-essential,
not whether scaffolds work. K2 PASS is a secondary finding about layer
redundancy, not evidence for scaffold replacement. The paper correctly
distinguishes this, but the K2 criterion as written (">0 layers below 20%")
is extremely easy to pass and provides no information about scaffold
viability. This is a weakness of the kill criteria design, not the
experiment execution.

## Novelty Assessment

**Layer criticality profiling of transformers** is well-established:

- Cornerstone Layers (arxiv 2409.14381), already cited in VISION.md, found
  that early layers are critical and middle layers are prunable. The U-shape
  finding here is consistent with and extends this to BitNet-2B-4T
  specifically, which is a modest contribution.

- Layer pruning literature (ShortGPT, LaCo, etc.) has extensively studied
  which layers can be removed from LLMs. The finding that 9/30 layers have
  <5% individual impact is not novel in principle.

**What IS novel:**

1. First measurement of per-layer criticality specifically in BitNet-2B-4T
   ternary architecture. Prior work focused on FP16/FP32 models.

2. The adapter-on-scaffold ablation -- demonstrating that trained adapters
   contribute exactly zero value when the base they were trained on is
   replaced (319M vs 320M PPL) -- is a clean, rarely-measured result.

3. The 4-million-fold gap between toy-scale (d=64, 6.9x) and production-
   scale (d=2560, 27.6Mx) scaffold replacement is an important scaling
   observation for the project.

**Delta over closest work:** The base_free_composition experiment at d=64
showed scaffold-only was marginal (6.9x base, 1.27x expert). This
experiment definitively closes the door at production scale. That is the
primary contribution.

## Experimental Design

**Strengths:**

1. Five-phase design (baseline, per-layer, progressive, skeleton, analysis)
   is thorough and well-structured. Each phase builds on the previous.

2. The control conditions in Phase 4 are excellent: norm-matched random,
   unscaled random, zero-base, and random-without-adapters. These isolate
   the contributions of norm structure, adapter presence, and base knowledge
   independently.

3. Reuse of existing adapters from bitnet_scale_n15 is appropriate --
   this tests scaffold replacement with real trained adapters, which is the
   stated hypothesis.

**Weakness -- Single seed (minor):**

The experiment uses seed=42 only. The paper justifies this via "multiseed
CV=0.5% at N=5 from prior experiments." This justification is reasonable
for PPL measurements but does not apply to the random scaffold generation.
The scaffold PPL (319M) is from one random draw. A different random ternary
scaffold could give substantially different PPL. However, since the result
exceeds the kill threshold by 7 orders of magnitude, no realistic seed
variation could change the verdict. Not blocking.

**Weakness -- Adapters trained on pretrained base:**

The Limitations section (point 3) correctly identifies this: adapters were
trained WITH the pretrained base, so they encode directions in the
pretrained coordinate system. Testing these same adapters on a random
scaffold is testing the wrong thing for the strongest version of the
base-free hypothesis. The correct test is training fresh adapters ON the
scaffold -- which is already identified as the next experiment
(exp_bitnet_scaffold_fresh_adapters). The paper is honest about this
limitation.

**Weakness -- PPL clamping at exp(100):**

Line 250: `return math.exp(min(avg_loss, 100))`. This clamps PPL at
e^100 ~ 2.69e43. The zero-base condition (6.96e41) is near this clamp.
This means the zero-base PPL may be artificially bounded. For the
skeleton conditions (319M, 510M), the losses are well below 100
(ln(319M) ~ 19.6), so the clamp does not affect K1 results. Minor issue,
only affects the zero-base condition which is not used in any kill
criterion.

**The experiment tests what it claims:** The stated hypothesis is scaffold
replacement with existing adapters. The kill criteria are appropriate for
this specific claim. The experiment cleanly kills K1 and passes K2.

## Hypothesis Graph Consistency

The HYPOTHESES.yml node `exp_bitnet_basefree_exploration` has:
- Status: killed (correct -- K1 killed)
- Kill criteria match the PAPER and code exactly
- Depends on exp_bitnet_scale_n15 (correct -- adapters sourced from there)
- Blocks nothing (correct -- this was exploratory)
- The follow-up node exp_bitnet_scaffold_fresh_adapters correctly identifies
  the limitation that this experiment tested the wrong adapter condition

The experiment correctly updates the hypothesis status. The evidence entry
in HYPOTHESES.yml accurately summarizes the results.

## Integration Risk

The K1 kill has clear architectural implications recorded in VISION.md:
"Base-free scaffold: PPL 319M" under killed findings. The base-free
readiness is listed at 10%. This is consistent.

The K2 PASS (layer criticality map) is a useful secondary finding that
could inform serving optimization (Track 3) but is not on the critical
path. The PAPER's suggested next steps (hybrid ablation, layer-aware
serving) are reasonable but speculative.

## Macro-Scale Risks (advisory)

1. The layer criticality profile was measured with PPL only. Task-specific
   criticality (e.g., math reasoning vs. code generation) may differ
   significantly. The "replaceable" layers 10-16 might be critical for
   specific capabilities not captured by mean PPL across 5 domains.

2. The progressive ablation non-linearity (zeroing 5 individually-harmless
   layers causes 40% degradation) is a warning for any layer pruning
   strategy. Macro experiments should measure simultaneous ablation, not
   extrapolate from individual results.

3. The adapter-invisible-on-scaffold finding applies to adapters trained
   on a different base. If macro experiments use scaffold-trained adapters
   (the fresh adapter path), this finding does not transfer.

## Verdict

**PROCEED**

The experiment is well-designed, correctly executed, and honestly reported.
The K1 kill at 27.6Mx (threshold 5x) is unambiguous -- no methodological
fix could change a 7-order-of-magnitude result. The K2 PASS provides
useful secondary data on layer criticality. The limitations section
correctly identifies the key caveat (adapters trained on pretrained base,
not on scaffold). The follow-up experiment (exp_bitnet_scaffold_fresh_adapters)
is the right next step.

The only improvements needed are cosmetic:

1. The super-exponential fit (gamma=2.1) should either include fitting
   methodology or be described qualitatively as "faster than exponential"
   without a specific parameter.

2. The paper could more explicitly state that K2 is a weak criterion by
   design (any single non-essential layer passes it) and that the real
   value is the full criticality map, not the binary K2 verdict.

Neither of these affects the kill verdict or the scientific conclusions.
The experiment achieved its purpose: definitively closing the scaffold
replacement path and providing a layer criticality map for future use.
