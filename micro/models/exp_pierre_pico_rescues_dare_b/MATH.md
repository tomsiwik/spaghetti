# MATH.md — Does Pico calibration rescue the failed B-space DARE?

## Hypothesis

In `exp_pierre_dare_b_vs_fisher_rao`, B-space DARE collapsed to 55.3%
(vs Fisher-Rao 64.7%). Diagnosis from that experiment's LEARNINGS.md:

> "DARE preserves expectations via random dropout + rescaling, but this
> guarantee holds for additive deltas. In LoRA, the effective delta is
> A@B — a multiplicative interaction. Dropping entries in B and rescaling
> doesn't preserve the expectation of A@B because A amplifies the
> perturbation non-linearly."

But there's a second story available: the **medical adapter collapsed to
30% MedQA** under B-space DARE. That's consistent with concentrated B
structure where random 90% dropout zeroed key entries.

Pico (arxiv 2604.16826) explicitly addresses concentrated B by **dampening
over-shared output directions** before merging. **Does Pico-calibrated
B condition the matrices well enough that B-space DARE works on them?**

> **Does Pico calibration rescue B-space DARE? Or is the multiplicative-
> interaction failure mode independent of B's concentration?**

## Pipeline

For each (layer, module) key:
1. Compute Pico calibration matrix `S` from stacked B's.
2. Calibrate per adapter: `B_t_calib = B_t @ S^T`.
3. Apply DARE on calibrated B's: random mask + 1/(1-p) rescale.
4. Mean over adapters.
5. Norm-rescale output to mean source norm of ORIGINAL (pre-calib) B's.

## Pre-registered Kill Criteria

- **K1 (DECISION)** Pico+DARE-B avg ≥ Fisher-Rao avg + 3pp.
- **K2 (RESCUE)** Pico+DARE-B avg ≥ B-space DARE prior avg + 9pp (i.e., from 55.3% baseline up to ≥ 64.3% — closes most of the collapse). PASS = Pico rescues B-space DARE; FAIL = the failure mode is structural to LoRA factoring, not just B concentration.
- **K3 (BUDGET)** Preprocessing ≤ 7s (Pico SVD ~5s + DARE masks ~2s).
- **K4 (MEDQA-CHECK)** MedQA score ≥ 50% (the prior collapse hit MedQA hardest at 30% — recovery here is the most diagnostic single number).

## Verdict logic & interpretation

| K1 | K2 | Outcome | Interpretation |
|----|----|---------|----------------|
| ✓ | ✓ | **SUPPORTED** | Pico+DARE-B is a viable B-only method. Concentrated-B failure was the dominant factor; Pico's calibration rescues it. |
| ✓ | ✗ | **SUPPORTED** with caveat | Pico helps but doesn't reach DARE-level effectiveness — try Pico+Fisher-Rao instead. |
| ✗ | ✓ | **KILLED, informative** | Pico recovers most of the DARE-B collapse but still loses to Fisher-Rao — the multiplicative-interaction story holds; B-space DARE is structurally wrong even with calibration. |
| ✗ | ✗ | **KILLED** | Pico does not rescue DARE-B at all — confirms the multiplicative-interaction diagnosis from the prior experiment. |

## Why this is worth running even if it KILLs

A KILL with K2 PASS but K1 FAIL is a **pure interpretability win**:
- It shows the original DARE-B failure was *partially* about concentrated B (Pico rescued the medical-adapter collapse) but *also* about LoRA's multiplicative interaction (Pico can't bring it up to Fisher-Rao).
- This decomposes the failure mode into two separable causes — useful framing for future Pierre composition work.

## References

- Pico (arxiv 2604.16826) calibration math
- DARE (arxiv 2311.03099) drop-and-rescale
- Prior experiment: `exp_pierre_dare_b_vs_fisher_rao` (B-space DARE → 55.3%, MedQA 30%)
