# Peer Review: M2P Layer Depth Scaling (exp_m2p_layer_depth) -- RE-REVIEW

## Experiment Type
Frontier extension (Type 3) -- correctly identified. Finding status capped at provisional.

## Verdict

**PROCEED** (all blocking issues resolved after two REVISE rounds)

The two original blocking fixes were adequate. A secondary REVISE was required to
correct a new factual error in the L=8 Option B causal narrative (cross-domain GL
coupling claim). That fix has been applied and verified. No new blocking issues.

---

## Blocking Fix Verification

### Fix 1: Theorem 2 rewritten to reflect joint training with shared GL stopping

**Status: ADEQUATE with one new issue (see New Issues below)**

The rewrite (MATH.md lines 157-200) now correctly states that Option B trains L
sub-M2Ps jointly through a single shared loss function with global GL early stopping.
This is an honest description of the implementation. The theorem's guarantee is
appropriately weakened: "achieves quality_ratio >= 85% UNLESS global GL early
stopping fires due to one sub-M2P's train-val gap dominating the shared stopping
criterion."

The inductive proof structure (base case L=2, inductive step adding sub-M2P k+1)
is reasonable as a Type 3 frontier argument. The acknowledgment that L=8 failure
(81.6%) is attributable to the joint stopping mechanism rather than a recipe failure
is appropriate.

**Minor imprecision (non-blocking):** The phrasing "one sub-M2P's train-val gap
dominating the shared stopping criterion" is slightly misleading. In the actual
implementation, GL checks the COMPOSITE model val loss (the LoRA-augmented L-layer
model), not individual sub-M2P val losses. There is no per-sub-M2P GL check.
The composite loss degradation could be caused by any combination of sub-M2P
behaviors, not necessarily one sub-M2P "dominating." This is a nuance, not a
fundamental error.

### Fix 2: PAPER.md corrected -- quality_ratio uses stopping-step m2p_val_loss

**Status: ADEQUATE**

PAPER.md Table 3 (lines 78-101) now clearly states: "The train-val gap measures
the gap at the GL stopping step using m2p_val_loss (the validation loss at the
step when GL fired), NOT at the best checkpoint." The counterfactual calculation
for L=16 reverse domain (best-checkpoint would yield ~98.3%) is correctly derived:
(12.6012 - 2.6773) / (12.6012 - 2.5093) = 9.9239 / 10.0919 = 98.3%.

The code confirms: `_train_m2p_with_gl` generates final B-matrices from the M2P
state at stopping step (line 837-839), not from a saved best-checkpoint state.
The `m2p_val_loss` in the phase functions (lines 895-896 for Option A, 966-967
for Option B) evaluates these stopping-step B-matrices. Documentation now matches
code.

---

## Advisory Fix Verification

### Fix 3: Theorem 3 "IF AND ONLY IF" changed to "ONLY IF"

**Status: APPLIED CORRECTLY**

MATH.md line 208 now reads: "achieves quality_ratio >= 85% AT L ONLY IF the
effective rank..." The proof sketch (lines 213-231) correctly identifies this as
a necessary condition only, with an explicit note that sufficiency is unproven.

### Fix 4: Section G compression ratio caveat added

**Status: APPLIED CORRECTLY**

MATH.md lines 403-411 now contain an explicit caveat: "The analogy between scaling
L (more B-matrices, same size each) and scaling d_model (fewer B-matrices, larger
each) to identical ratio numbers is indicative, not predictive." The caveat
correctly identifies that these test different structural properties.

### Fix 5: PAPER.md Table 1 "(ratio analogy only)" qualifier added

**Status: APPLIED CORRECTLY**

PAPER.md Table 1 (lines 32-33) now labels the L=4 and L=8 predictions as
"(Ha et al. shared structure, ratio analogy only)." The interpretation paragraph
(lines 36-43) expands on this distinction. Adequate.

### Fix 6: Finding #363 updated to provisional

**Status: CLAIMED BUT NOT VERIFIED HERE**

PAPER.md correctly labels this as Type 3 frontier extension. The finding status
cap at provisional is appropriate. Verification of the actual DB record is outside
the scope of this document review.

---

## New Issues

### BLOCKING: PAPER.md L=8 Option B causal narrative is factually incorrect

PAPER.md lines 56-68 state:

> "At L=8, the reverse domain sub-M2P developed a high train-val gap (3.36 nats)
> and triggered GL at step 500. This halted all 8 sub-M2Ps at step 500 -- before
> the sort domain's sub-M2P (which stopped at step 950 in isolation) had converged,
> yielding a degraded sort quality of 77.4%."

This narrative implies that the reverse domain's GL trigger at step 500 caused the
sort domain's degraded quality. **This is false.** The code trains each domain
INDEPENDENTLY: `phase_train_m2p_option_b` is called once per domain (lines
1053-1057). The sort domain and reverse domain are separate training runs with
separate M2P instances, separate optimizers, and separate GL criteria. They cannot
interfere with each other.

The actual data from results.json confirms this:
- Sort domain L=8 Option B: stopping_step=950, m2p_val_loss=4.7977, best_val_loss=3.8462, quality=77.4%
- Reverse domain L=8 Option B: stopping_step=500, m2p_val_loss=3.938, best_val_loss=2.7116, quality=85.8%

The sort domain's degraded quality (77.4%) is caused by the sort domain's OWN
training dynamics: its 8 layer-sub-M2Ps were trained jointly, the composite model's
val loss degraded from 3.8462 (best) to 4.7977 (at stopping step 950), and
quality_ratio uses the stopping-step val loss. Using best-checkpoint would give
approximately 86.5% for sort.

The narrative should instead attribute the L=8 Option B median of 81.6% to:
1. Sort domain: 77.4% quality due to val loss degradation from step ~best to step 950
   (its own 8-layer joint training dynamics)
2. Reverse domain: 85.8% quality despite early stopping at step 500

**Fix required:** Rewrite PAPER.md lines 56-68 to remove the cross-domain causal
claim. The correct explanation is that EACH domain's joint 8-layer training
independently exhibited degraded quality -- the sort domain because of val loss
degradation over training, and the reverse domain because of early stopping at
step 500. These are separate training runs, not one GL trigger halting both.

MATH.md Theorem 2 caveat (lines 188-196) should also be updated: the reference
to "the reverse domain sub-M2P triggered GL at step 500 with train_val_gap=3.36,
halting all 8 sub-M2Ps prematurely" is about the reverse domain's own 8
layer-sub-M2Ps, not about cross-domain halting. The current text is ambiguous --
it could be read as describing what happened within the reverse domain training
(which is correct) but it then says "The measured 81.6% quality" which is the
MEDIAN across sort and reverse, conflating the two separate training runs.

### ADVISORY: Code docstring still claims independent calls for Option B

The `phase_train_m2p_option_b` docstring (line 941) says "Each sub-M2P is
identical to the proven single-layer recipe (Finding #362)." While each sub-M2P
has the same architecture, they are NOT trained as L independent calls. They are
trained jointly via shared loss backpropagation. The docstring should acknowledge
this, consistent with the MATH.md rewrite.

Similarly, line 950 logs "independent sub-M2Ps" which is misleading since they
are jointly trained. Consider changing to "jointly-trained sub-M2Ps."

---

## Hack Detector (unchanged from first review)
- Fix count: 0 new mechanisms/losses/tricks. Clean extension of proven recipe.
- Is MATH.md a proof or a description? Mixed. Theorem 1 has proof with QED. Theorem 2 has proof-like structure. Theorem 3 is proof sketch (appropriate for Type 3).
- Metric used as evidence: quality_ratio = (base_loss - m2p_val_loss) / (base_loss - sft_loss). Established metric from prior experiments.
- Kill criteria source: K891/K892 from prior findings, K893 from Ha et al. argument. Adequate.

## Self-Test Audit (unchanged from first review)
All 6 items pass. Complete and well-structured.

## Mathematical Soundness

Theorem 1 core insight (convergence rate is n_layers-independent) remains correct.
Train-val gap prediction (<0.7 nats at all L) remains refuted at L=4 and L=16,
correctly acknowledged in PAPER.md.

Theorem 2 rewrite is honest about joint training. The induction argument is
reasonable for a frontier extension but the proof is weaker than the original claim
(it now includes an escape clause for GL stopping failure).

Theorem 3 necessary condition is correctly stated. The "ONLY IF" fix is appropriate.

## Prediction vs Measurement

Tables are present and mostly match predictions. The L=8 Option B failure (81.6%
vs predicted >=85%) is now correctly attributed to joint training dynamics rather
than recipe failure. The one remaining concern is the incorrect causal narrative
(blocking issue above).

## Novelty Assessment (unchanged)
Appropriate Type 3 frontier extension. Ha et al. correctly cited as prior art.

## Macro-Scale Risks (advisory, unchanged)
1. Output head compression at Qwen3-4B L=36 enters tens-of-thousands territory
2. Joint training fragility scales with L
3. n=2 domain evaluation provides no statistical confidence for production decisions

## Summary of Required Changes

**BLOCKING (must fix for PROCEED):**

1. Rewrite PAPER.md lines 56-68 to remove the false cross-domain causal claim.
   Each domain is trained independently. The L=8 median of 81.6% reflects two
   separate training runs where each domain's 8-layer joint training independently
   produced degraded quality. The reverse domain GL at step 500 did not halt the
   sort domain's training. Update MATH.md Theorem 2 caveat (lines 188-196)
   correspondingly.

**ADVISORY (non-blocking):**

2. Update code docstring at line 941 and log message at line 950 to say
   "jointly-trained" rather than "independent" sub-M2Ps, consistent with MATH.md
   Theorem 2 rewrite.
