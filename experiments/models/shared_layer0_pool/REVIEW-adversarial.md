# Peer Review: Shared Layer 0 Capsule Pool (RE-REVIEW)

## Previous Review Status

The prior review requested 4 specific fixes. All 4 have been properly addressed:

### Fix 1: Loudness reconciliation -- ADDRESSED
MATH.md (lines 128-138) and PAPER.md (lines 153-163) now contain an explicit
reconciliation paragraph distinguishing global loudness (killed by relu_router:
learned scales converged to ~0.99) from per-layer magnitude distortion (Layer 0
at ~2x while Layers 1-3 remain at 1x). The argument that a single global scalar
cannot correct a per-layer imbalance is sound. A reader familiar with the project
history will no longer see an apparent contradiction.

### Fix 2: Default-protocol claim weakened -- ADDRESSED
PAPER.md Implication 1 (lines 226-231) now states "Shared Layer 0 is a strict
improvement over full concatenation in zero-shot composition" and explicitly notes
that the default-protocol question requires a calibrated comparison
(full_concat+calibration vs shared_L0+calibration) which has not been tested.

Minor note: the HYPOTHESES.yml evidence claim (line 477) still reads "Shared Layer 0
should be the default zero-shot composition protocol." This is stale but non-blocking
-- it is a tracking artifact, not a paper claim.

### Fix 3: Average recommended over first -- ADDRESSED
PAPER.md (lines 76-81) now states that strategy differences are "not statistically
distinguishable at 3 seeds (overlapping confidence intervals)" and recommends
"average" as the principled default for D>2 domains. The "first" strategy is no
longer privileged.

### Fix 4: Capacity-reduction alternative explanation -- ADDRESSED
PAPER.md Limitation 7 (lines 193-198) explicitly describes the alternative
explanation that the 185K vs 202K parameter difference could reduce overfitting
at micro scale. Suggests a random-pruning-to-match-capacity control and correctly
notes this is "not blocking for the core finding that sharing does not degrade
quality."

## Mathematical Soundness

### Parameter savings: CORRECT
Arithmetic verified:
- Full concat capsule params: 4 * 2 * 2 * 128 * 64 = 131,072
- Shared L0 capsule params: 16,384 + 98,304 = 114,688
- Savings: 16,384 = 12.5% of capsule params, 8.1% of total 202,112

### Double counting mechanism: SOUND
The per-layer magnitude argument is logically consistent with the global-loudness
falsification. A global scalar applied to (Layer0_2x + Layer1_1x + Layer2_1x +
Layer3_1x) would need to be ~5/6 to normalize total magnitude, but this would
also attenuate the correctly-scaled deeper layers. The relu_router's learned
scalar converging to ~0.99 is consistent: the optimal global correction is
negligible because the distortion is concentrated in one layer.

### ReLU independence: CORRECT
Each row of the concatenated detector matrix operates independently under ReLU.
The sum-of-individual-outputs identity holds exactly.

### Weight averaging approximation: CORRECTLY BOUNDED
MATH.md correctly notes that B_avg @ ReLU(A_avg @ x) differs from the average of
B_k @ ReLU(A_k @ x) due to ReLU nonlinearity. The argument that high Jaccard
(~0.54) makes this difference small is qualitative, not formally bounded. This is
acceptable at micro scale -- a formal bound would require assumptions about weight
distribution that are not available.

## Novelty Assessment

The selective sharing of Layer 0 only (rather than uniform treatment across layers)
is a modest but genuine contribution within the project. It follows naturally from
behavioral_dedup's finding and validates an actionable architectural implication.

No entry in references/ directly implements this mechanism. TIES-merging and DARE-
merging (both in references/) operate uniformly across layers. The layer-selective
approach is the delta.

## Experimental Design

### Hypothesis tested correctly
The kill criterion (>2% degradation vs full concat) is directly tested with three
strategies across 3 seeds. All pass with margin.

### Controls adequate
- Joint training (upper bound)
- Full concatenation (control for kill criterion)
- Weight averaging (alternative zero-shot method)
- Three sharing strategies (mechanism robustness)
- Jaccard confirmation (reproduces behavioral_dedup independently)

### Code review: CLEAN
The composition code correctly handles:
- Layer 0 shared pool at n_capsules_per_domain (128)
- Layers 1+ concatenated at n_capsules_per_domain * n_domains (256)
- Base model shared parameters (embeddings, attention, norms) copied correctly
- All three strategies implemented as described in MATH.md

One minor code observation: `profile_layer0_jaccard` profiles the composed
full-concat model (256 capsules at Layer 0), computing cross-pool Jaccard between
capsules 0-127 and 128-255. This is the correct measurement for confirming
behavioral_dedup's co-activation finding in the composed setting.

## Macro-Scale Risks (advisory)

1. **Layer 0 domain-invariance may not hold with diverse data.** a-m vs n-z names
   share a 26-character alphabet. Python vs English prose with BPE tokenization
   could produce domain-specific Layer 0 features. Macro must profile Layer 0
   Jaccard before adopting this protocol.

2. **Calibration may eliminate the sharing benefit.** If calibration corrects the
   Layer 0 double-counting that sharing avoids, the quality improvement disappears
   and only the parameter saving (3.3% at D=5, L=24) remains. This is acknowledged
   in the paper but untested.

3. **Savings diminish at scale.** (D-1)/(L*D) = 4/120 = 3.3% at macro. The
   practical value shifts to skipping Layer 0 fine-tuning per contributor (25%
   compute savings), which is not validated here.

## Verdict

**PROCEED**

All 4 required fixes from the prior review have been properly addressed. The
loudness reconciliation is logically sound. The default-protocol claim is
appropriately scoped to zero-shot. The strategy recommendation correctly favors
"average." The capacity-reduction alternative is acknowledged as a limitation.

The core finding -- that sharing Layer 0 does not degrade quality, confirming
behavioral_dedup's insight is actionable -- is well-supported by the experimental
design. The double-counting explanation is plausible and correctly distinguished
from the killed global-loudness hypothesis. The remaining uncertainties (capacity
reduction, calibration interaction) are honestly reported and are macro-validation
questions, not micro-scale flaws.

HYPOTHESES.yml status should be updated from "revising" to "proven."
