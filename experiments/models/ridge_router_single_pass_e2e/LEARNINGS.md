# LEARNINGS.md — Ridge Router + Single-Pass E2E

## Core Finding

Three independent failures masked the real question this experiment tried to answer.

### What went wrong (three separate issues):

1. **Adapter mismatch (experimental design):** K799 threshold (≤4.778) was derived
   from Finding #313's oracle PPL (4.684), which used `tiny_routing_heads/adapters/`.
   This experiment used `real_data_domain_experts/adapters/` (scale=20). The correct
   oracle PPL with these adapters is 7.598 — the threshold was never achievable.

2. **Majority vote masks per-token routing (implementation bug):** The PPL pipeline
   uses segment-level majority vote to select two adapters per sequence, not per-token
   routing as designed in MATH.md. With 90%+ per-token accuracy on 128-token segments,
   majority vote always recovers the correct domains → oracle PPL == ridge PPL for all
   10 pairs. K800 (per-token accuracy) is completely decoupled from K799 (PPL).

3. **Scale=20 adapters harm PPL (confirmed):** In 8/10 domain pairs, the oracle
   (correct adapter) produces HIGHER PPL than base model (no adapter). Routing to
   beneficial adapters is impossible when the adapters aren't beneficial. Consistent
   with Findings #328, #330, #337, #338.

### What IS real:

- **Per-token accuracy drop on mixed-domain sequences is genuine:** 98.3% (single-domain
  IID) → 89.67% (mixed-domain concatenated) = 8.6pp degradation. Failure is distributed
  across both segments, not just boundary tokens.
- **Context-induced distribution shift exists:** Hidden states change when two domains
  are concatenated. This affects the linear classifier uniformly.
- **Segment-level majority vote is robust:** Despite 10% per-token error, segment-level
  domain identification works ~100% of the time.

## Why (root cause analysis)

The experiment tried to compose two proven components (Finding #310 + Finding #313)
but didn't control for adapter provenance. The adapters are the shared dependency:
- Finding #310 proved ridge routing works on `real_data_domain_experts` hidden states ✓
- Finding #313 proved single-pass MLP works on `tiny_routing_heads` adapters ✓
- This experiment combined #310's routing with different adapters than #313's proof

The composition theorem (MATH.md Theorem 1) is formally correct IF:
- The oracle PPL baseline matches the adapters used
- Per-token routing is implemented (not majority vote)
- The adapters provide benefit over base

None of these held.

## Confirming prior findings

| Finding | Confirmed? | Evidence |
|---------|-----------|----------|
| #328 (scale=20 degradation) | YES | 8/10 pairs worse than base |
| #330 (scale=5-13 required) | YES | scale=20 harmful even with oracle routing |
| #310 (ridge 98.3% single-domain) | YES | Reproduced exactly (98.31%) |
| #313 (single-pass MLP works) | UNTESTED | Used different adapters |

## Alternatives / what to try next

1. **Re-run with scale=5 adapters that actually help PPL.** The routing question can
   only be answered when adapters provide benefit. Use adapters from a controlled SFT
   training run at scale=5.

2. **Implement true per-token routing.** Replace majority vote with per-token adapter
   selection in the forward pass. This is architecturally harder (needs per-token
   MixedAdapterMLP or equivalent) but is what MATH.md actually proves.

3. **Retrain router on mixed-domain data.** The 8.6pp accuracy drop is genuine and
   would matter with per-token routing. Ridge retrained on concatenated sequences
   should recover accuracy (simple fix).

4. **Use consistent adapter provenance.** Any E2E experiment must use the SAME
   adapters that produced the oracle baseline. Cross-experiment adapter references
   are invalid.

## Implications for the project

- **Scale=20 is dead.** Four experiments now confirm scale=20 adapters are harmful.
  All future work must use scale=5-13. Stop using `real_data_domain_experts` adapters
  (they were trained at scale=20).

- **Composition proof is sound, implementation is not.** Theorem 1 (MATH.md) is
  correct — the bound E[PPL] ≤ PPL_oracle · (1 + (1-p)·Δ_max/PPL_oracle) holds.
  The implementation just needs to match the proof's assumptions.

- **Segment-level routing is a viable alternative.** If per-token routing is too
  expensive, majority vote on segments of length ≥64 tokens may be sufficient.
  This is a potentially useful finding for production: route per-segment, not
  per-token.

## Recommended follow-up

1. Train 5-domain adapters at scale=5 on real data (medical, code, math, legal, finance)
2. Verify each adapter improves its domain PPL over base (prerequisite)
3. Re-run this experiment with the new adapters and corrected K799 threshold
4. Implement per-token routing OR formalize segment-level majority vote as the design

## Audit-Rerun Closure (2026-04-18)

This experiment was retagged `audit-2026-04-17-rerun, lora-scale` during the
2026-04-17 audit sweep. The fix-category `lora-scale` assumes that adjusting
LORA_SCALE would rescue the kill. Three independent closure theorems (see
PAPER.md) show this is false:

- **C1 adapter-quality ceiling:** 8/10 pairs have oracle_ppl ≥ base_ppl. Scale=20
  adapters on `real_data_domain_experts` are harmful. In-place LORA_SCALE change
  cannot make harmful adapter deltas beneficial without retraining — that is a
  new experiment, not a fix.
- **C2 routing-decoupling invariance:** Segment majority vote with L=128 at
  p=0.897 has error ≤ 10⁻¹⁷ by Hoeffding → oracle_ppl == ridge_ppl by construction,
  independent of LORA_SCALE. K799 is decoupled from K800 architecturally.
- **C3 two-pass architectural bound:** 254.2/109.3 = 2.326x. Ratio > 2 for any
  LORA_SCALE that retains a second forward pass. Theorem 3's 1.013x is for a
  1-pass pipeline (different architecture).

**Closure-rule promoted:** `base-ceiling-blocks-routing` — if oracle PPL ≥ base
PPL, no routing can improve. Third oracle-ceiling closure (after depth_routed and
mlp_only_per_token_routing), same family, different mechanism. The generalisation:
inspect the oracle before analysing the router.

**Status:** KILLED (closure, not rerun). Final artifacts: MATH.md, run_experiment.py,
results.json, PAPER.md (with Audit-Rerun Closure addendum), REVIEW-adversarial.md,
LEARNINGS.md.
