# Peer Review: Clone-Compete Evolution (Re-Review)

## Prior Review Summary

The initial review (verdict: REVISE) required 3 fixes and offered 4 non-blocking suggestions.

### Required Fix Verification

| Fix | Requirement | Status |
|-----|-------------|--------|
| 1 | K1 must be INCONCLUSIVE, not PASS | ADDRESSED -- PAPER.md kill criteria table now reads "INCONCLUSIVE" with explicit note on p=0.265 |
| 2 | Add cold-start control discussion to Limitations | ADDRESSED -- Limitation 7 discusses missing cold-start baseline, inability to disambiguate warm-start from additional-data benefit, deferral to powered replication |
| 3 | Note tournament data reuse across rounds in Limitations | ADDRESSED -- Limitation 8 explicitly flags that both rounds use the same 38 held-out samples and recommends independent sets |

### Non-Blocking Suggestion Verification

| Suggestion | Status |
|------------|--------|
| Cite PBT (Jaderberg et al., 2017) | ADDRESSED -- PAPER.md Key References now includes PBT with explicit analogy to clone-compete |
| Soften "accelerating" claim | ADDRESSED -- MATH.md Section 5 now reads "insufficient to determine acceleration vs diminishing trend" |
| Condition convergence on power | ADDRESSED -- MATH.md Section 4 adds "given a sufficiently powered tournament" qualifier and full caveat paragraph about underpowered selection |
| Worst-adapter selection inconsistency | Acknowledged in Limitation 4; appropriate for micro |

## Mathematical Soundness (unchanged from prior review)

The regression bound (Section 3), binomial tournament formulation (Sections 2.3-2.4), and sample size calculation are all correct. The convergence claim (Section 4) is now properly conditioned on tournament power. No new mathematical concerns.

## Novelty Assessment (unchanged)

PBT is now cited. The delta -- applying PBT-style evolution to LoRA adapters under 1/N composition with a regression bound -- is modest but genuine. The regression bound O(epsilon/N) connecting evolution to composition stability is the novel contribution.

## Experimental Design

K1 INCONCLUSIVE, K2 PASS, K3 PASS is an honest summary. The paper carries 9 well-articulated limitations. The powered replication (exp_bitnet_clone_compete_powered) is the correct next step.

No new experimental design issues found beyond those already documented in the limitations.

## Macro-Scale Risks (advisory, unchanged)

1. PPL-to-task gap remains the primary risk (bitnet_task_eval killed at 2B).
2. Fresh data logistics at 50+ domains.
3. Tournament cost at N=200 samples across many domains (~2.2 hours per cycle).
4. Selection pressure may be marginal if true effect is small.

## Verdict

**PROCEED**

All three required fixes from the prior review have been properly addressed. The non-blocking suggestions were also incorporated. The paper now presents an honest account: K1 INCONCLUSIVE with a clear path to resolution via the powered replication, K2 and K3 cleanly passing, monotonic PPL improvement (15.82 -> 14.50 -> 13.04), negligible regression (0.06%), and 9 explicit limitations covering every concern raised in the initial review.

The mechanism works in principle. The Evolve phase is validated directionally. Scale validation belongs in macro.
