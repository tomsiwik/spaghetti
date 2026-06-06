# PAPER.md: Hard Top-1 Gumbel MoE Routing — KILLED (router collapse)

## Experiment Summary

**Status: KILLED** — K861 literal FAIL (median 23.9% < 25% threshold), with D1
router collapse (3/5 unique argmax experts) as the mechanistic cause.
Verdict in `results.json` is KILLED. `is_smoke=false`. `ran=true`.

**DB-tracked Kill Criterion #861:** "Median M2P quality drops < 25% of SFT"
(PASS iff median quality-ratio ≥ 0.25).

## Audit-Rerun Context (tags: audit-2026-04-17-rerun, metric-swap, code-bug)

Dir as claimed contained MATH.md / PAPER.md / REVIEW / LEARNINGS from the
**predecessor** experiment `m2p_domain_conditioned` (additive injection;
Finding #342 KILLED). `run_experiment.py` had already been upgraded to
`M2PTransformerMoE` (hard top-1 + Gumbel + STE) but produced no `results.json`
and emitted three unrelated K-IDs (K855/K856/K857) instead of the single
DB-tracked KC #861 — this is the `metric-swap` failure.

Fixes applied pre-run:
1. Rewrote `MATH.md` for hard top-1 MoE (Theorem 1 gradient isolation under
   STE; Lemma 1 router-collapse as stable equilibrium without aux load loss).
2. **Pre-registered** the re-label rule in MATH.md §D/§G: "K861 PASS under
   router collapse is a metric-swap false-positive; re-label KILLED."
3. Added `phase_router_check` (D1 diagnostic) and merged D1 into the verdict
   logic in code so `results.json["verdict"]` stays consistent with PAPER.md.
4. Fixed eval-time non-determinism: Gumbel noise disabled via
   `m2p.training_mode = False` during K861 measurement so the median is
   reproducible under `seed=42`.
5. Re-mapped internal K-IDs so the single DB-tracked KC #861 is measured by
   name, no unrelated K-codes pollute results.

## Prediction vs. Measurement (locked in MATH.md before run)

| Pre-registered prediction | Predicted | Measured | Status |
|---|---|---|---|
| C.1 K861 median quality | [0.15, 0.40], borderline | **0.239** | **FAIL** (just below 0.25) |
| C.2 D1 unique argmax experts | {1, 2, 3} / 5 | **3 / 5** | Confirmed (Lemma 1) |
| C.3 D2 B-matrix \|cos\| mean | [0.80, 0.97] if collapse, else lower | **0.3354** | Surprising — centroid destabilised |
| C.4 D4 Grassmannian max \|cos\| | ≤ 1e-5 | **0.0** | PASS (structural) |
| C.5 Worst-domain quality | ≤ -2.0 (expected "repeat") | **-3.306 (repeat)** | Confirmed |

## Per-Domain Quality Breakdown

| Domain | Base loss | SFT loss | M2P loss | Quality | Routed to expert |
|---|---|---|---|---|---|
| arithmetic | 5.28 | 1.70 | 3.39 | **+52.9%** | 0 |
| reverse    | 3.50 | 1.80 | 3.09 | +23.9% | 0 |
| repeat     | 1.11 | 0.51 | 3.08 | **-330.6%** | 0 |
| sort       | 3.44 | 1.83 | 3.26 | +11.2% | 4 |
| parity     | 5.42 | 1.30 | 3.96 | +35.5% | 3 |

Router collapse → arithmetic, reverse, repeat all routed to expert 0.
Expert 0's weights drift toward the arithmetic/reverse task manifold (highest
loss-gap gradients), starving the "repeat" domain — which has the smallest
base→SFT gap (0.60) — and producing the catastrophic -330.6% quality.
This is **Lemma 1 exactly** (MATH.md §B): a higher-loss domain dominates
θ_{e*} when two domains share the same argmax expert, pushing the shared
expert away from the low-loss domain's optimum.

## Key Mechanistic Finding

### D2 succeeds structurally; K861 still fails architecturally

B-matrix diversity leapt from 0.9956 (Finding #341, no conditioning) and
0.9785 (Finding #342, additive conditioning) to **0.3354** under hard top-1
routing — a ~65-percentage-point reduction. STE gradient isolation (Theorem 1)
**does** prevent centroid collapse at the B-matrix level.

But D1 router collapse wipes this out at the task level: two of five domains
share one expert, and the expert cannot serve both. K861's median is dragged
to 0.239 by the displaced domains (reverse 23.9%, sort 11.2%).

**Mechanistic implication:** the failure is now at the routing layer, not the
hypernetwork layer. Fixing this requires an aux load-balance loss (Switch
Transformer, arXiv:2101.03961) or temperature-annealed softmax to schedule
router commitment. Without it, Lemma 1's equilibrium is entered in the first
few dozen training steps and never escaped.

## Sibling Experiment Comparison

| Experiment | Conditioning | Median Q | Worst Q | B-cos | Verdict |
|---|---|---|---|---|---|
| m2p_distillation_toy (#341) | none | 21.9% | -329% (repeat) | 0.9956 | KILLED |
| m2p_domain_conditioned (#342) | additive e_d | 47.3% | -303.7% (repeat) | 0.9785 | KILLED |
| **m2p_hard_moe (this)** | hard top-1 Gumbel+STE | **23.9%** | **-330.6% (repeat)** | **0.3354** | **KILLED** |
| m2p_moe_routing V2 (#574) | soft MoE (no aux) | — | — | — (D1 coll.) | KILLED |

Hard top-1 destabilises the centroid (B-cos -65pp) but introduces a new
failure: router arbitration without load-balance loss. Consistent with
Switch Transformer evidence that aux loss is not optional at N_e ≥ 4.

## Verdict-Consistency Pre-Flight

1. `results.json["verdict"] == "KILLED"`: ✓
2. `results.json["all_pass"] == false`: ✓
3. PAPER.md verdict line: KILLED (no PROVISIONAL/PARTIAL): ✓
4. `is_smoke == false`: ✓
5. `git diff MATH.md`: only PRE-run additions (locked KC + new theorems);
   no post-run KC relaxation: ✓
6. Antipattern check:
   - composition math: per-expert `f_e(mem)` then `Σ r_e · f_e(mem)` (NOT
     `(Σ B)(Σ A)`); STE forward one-hot, backward soft. ✓
   - LORA_SCALE = 2.0 (safe, unchanged from siblings). ✓
   - Routing uses ground-truth `domain_id` as router input — this is a
     known MATH-assumption but is standard for this experiment family
     (test-of-mechanism, not test-of-routing-quality). Logged in §Assumptions.
   - No `shutil.copy` creating fake adapters. ✓
   - No hardcoded `"pass": True`. ✓
   - No proxy-model substitution (this IS a toy-GPT experiment, base=ToyGPT
     by design; no Gemma 4 claim is made). ✓

## Three permanently-learned rules (propagate to siblings)

1. **Hard top-1 MoE without aux load-balance loss collapses at N_e ≥ N_domains
   with heterogeneous losses.** Same root structure as soft-MoE collapse
   (Finding #574), different DOF (expert arbitration vs weight uniformity).
   Aux-loss is a prerequisite, not an optimisation.

2. **K861 PASS can hide D1 collapse — always report D1 alongside K861.**
   The metric-swap audit tag caught exactly this: a K861-only report
   (34.2% in first run) would have looked like a PASS while 3/5 experts
   were dead. The fix is to make D1 a gate on the verdict, not a separate
   diagnostic. Encoded in `results.json["relabel_killed_by_d1_router_collapse"]`.

3. **STE gradient isolation does fix centroid collapse at the B-matrix layer
   (|cos| 0.9956 → 0.3354).** The hypernetwork generates distinct B-matrices.
   The problem moved to the routing layer. Future M2P variants can reuse STE
   but MUST add load-balance-aware routing.

## Assumptions / Caveats

- **A1 (Ground-truth routing input):** the router consumes `domain_id` directly.
  This is a best-case routing input. A real system must infer domain from
  hidden states (see TF-IDF / logistic routing siblings). If the router fails
  under ground-truth, it will fail under inferred input.
- **A2 (Stochastic training):** Gumbel noise during training causes K861 to
  fluctuate across re-runs (observed 23.9% and 34.2% across two back-to-back
  runs with `mx.random.seed(42)`). The `relabel_killed_by_d1_router_collapse`
  safeguard (MATH.md §G) ensures the verdict is robust under both.
- **A3 (N_experts = N_domains = 5):** symmetric case. Adding more experts
  would not fix Lemma 1 without load-balance loss.

## Cross-Experiment Signal for Analyst

This closes a clean 3-way sweep of M2P B-matrix-collapse mitigations:
- additive domain embedding (#342): KILLED — attention bottleneck
- soft MoE (#574): KILLED — uniform saddle
- hard top-1 MoE (this): KILLED — router arbitration without aux loss

All three attempts modify a different knob; all fail through a different
mechanism; the unifying constraint is: **gradient competition across domains
without an explicit load-balancing signal is a stable failure attractor.**
Any sibling experiment that does not add a load-balance term will inherit
this impossibility structure. Suggested propagation: annotate
`project_m2p_impossibility.md` or similar memory with the 3-way observation.

## Recommendation (do not auto-spawn)

One directly-addressed sibling would carry this arc forward cleanly:
`exp_m2p_hard_moe_v2_aux_loss` — add switch-transformer-style load-balance
loss `ℓ_aux = N_e · Σ_e f_e · P_e` (from Fedus 2021) to the current
architecture and re-measure K861 + D1. Gated by analyst/planner, not spawned
here.
