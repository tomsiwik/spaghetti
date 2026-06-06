# LEARNINGS: exp_p0_e2e_combined_routing_n10

## AUDIT RE-CLASSIFICATION (2026-04-18) — verdict KILLED

Tags: `audit-2026-04-17-rerun`, `tautological-routing`.

The hypothesis promised in the experiment id/title/notes — "combined logistic
routing at N=10 achieves within 3pp of perfect-routing E2E quality"
(ref: Finding #525, 89.9% at N=10) — was **not tested**. `run_experiment.py`
hardcoded N=3 (math/code/medical) and reused 3 adapters from
`exp_p0_e2e_benchmark/`. At N=3 with maximally-separated domains the router
hits 100% accuracy, making K1481 ("Routing loss <= 5pp vs oracle") a
tautology rather than a test of combined-logistic's ability to tolerate
misrouting.

Re-classified KC under pre-reg:
- K1478 PASS (GSM8K 77% ≥ 65% at N=3)
- K1479 PASS (HumanEval 57% ≥ 50%)
- K1480 PASS (MedMCQA 58% ≥ 40%)
- K1481 FAIL_RECLASSIFIED (antipattern #6 — KC measures wrong object)

Verdict: KILLED on pre-registered intent.

## Preserved behavioral finding

The full E2E pipeline (combined logistic router → adapter selection →
generation) works end-to-end with no bugs at N=3, adds negligible overhead
(~140ms batch routing, 4.3s router training), and preserves adapter
deltas of +62/+39/+30pp over base on GSM8K/HumanEval/MedMCQA. This is a
useful *floor* — it confirms the pipeline has no wiring bugs — but it does
not discharge the N=10 hypothesis or the combined-logistic-vs-TF-IDF-only
claim.

## V2 path

A follow-up `..._n10_v2` experiment must:
1. Use ≥10 distinct adapters.
2. Pre-register K1481 conditional on measured router accuracy ∈ [85%, 95%]
   on the benchmark queries — if routing is too easy (≥99%), the experiment
   is *vacated*, not passed.
3. Compare measured quality loss against Theorem 1's prediction Δ ≤
   (1 − p)(A_oracle − A_base) within ±2pp.
4. Contrast combined-logistic vs TF-IDF-only router to quantify the
   combined-router value proposition.

## Cross-reference

Same audit pattern as `exp_p8_vproj_domain_behavioral` (Round 2 review,
2026-04-18): KC structurally measures the wrong object because of protocol
choices; documentation-only re-classification, MATH.md preserved as
pre-registered, code not re-executed because the antipattern is structural.

---

## Original (2026-04-13) — SUPERSEDED

### Core Finding
Combined logistic routing (TF-IDF + sentence embeddings) at N=3 achieves 100% routing
accuracy and zero quality loss E2E: GSM8K 77% (+62pp), HumanEval 57% (+39pp), MedMCQA
58% (+30pp) vs base 15/18/28%.

## Why
At N=3, well-separated domains (math/code/medical) are trivially separable — combined
logistic is overkill. The routing signal from either TF-IDF or embeddings alone would
suffice. The real value of combined logistic is at N=10+, where Finding #525 shows
89.9% accuracy and separation becomes non-trivial.

## Implications for Next Experiment
The E2E pipeline is proven at N=3. The open question is N=10+: Finding #525 predicts
~90% routing, which Theorem 1 maps to ~6pp quality loss (0.10 × ~60pp delta). The next
E2E experiment should run at N=10 with 10 distinct adapters to measure actual quality
degradation from 10% misrouting. This is the last unverified link in the P0 pipeline.

## Key Numbers
- Routing accuracy: 100% (math), 100% (code), 99% (medical — 1 misrouted, still correct)
- Router training time: 4.3s
- Routing batch overhead: ~140ms
- Base variance: ±5pp at N=100 (larger eval sets needed for precise measurements)

## Status
Finding #532 (SUPPORTED). All 4 kill criteria PASS with margin.
