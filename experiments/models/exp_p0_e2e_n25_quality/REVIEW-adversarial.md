# Adversarial Review: E2E N=25 Quality Validation

## Round 2 — Audit Re-review (2026-04-18) — VERDICT FLIP: KILLED

Flagged in 2026-04-17 audit with tags `audit-2026-04-17-rerun` and
`tautological-routing`. Round 1 (below) flagged C1 ("Only 3 Adapters Tested")
as **non-blocking**. Under PLAN.md §1 pre-flight item 6 and antipattern #6
("KC measures wrong object"), it is **blocking**, not non-blocking — same
verdict-flip as the N=10 sibling `exp_p0_e2e_combined_routing_n10`.

### What Round 1 missed

Round 1 called C1 a "lower-bound" concern and mitigated it by pointing to
Limitation #3 in PAPER.md. But K1489's phrasing ("max routing loss ≤ 10pp
vs oracle") is a claim about combined-logistic routing **in a regime where
routing can send queries to a wrong adapter**. With only 3 adapters, every
misrouted query lands on base (not wrong adapter), so:

- Theorem 1 bound: `loss ≤ (1 − α)·(Q_oracle − Q_base)`.
- Max per-domain delta: 62pp. With measured α ≥ 88%, max loss = 0.12·62 = 7.4pp.
- K1489 threshold 10pp cannot fail unless α drops below ~83%.

The 22 MMLU distractor domains are a **structural safety moat** — they make
wrong-adapter routing geometrically impossible. The KC measures "base-model
fallback loss", not "wrong-adapter loss". Antipattern #6 exactly.

### Re-classified KC table

| ID    | Original threshold         | Measured | Valid? | Counts against pre-reg? |
|-------|----------------------------|----------|--------|--------------------------|
| K1486 | GSM8K >= 68%               | 76.0%    | Yes    | PASS (surface)           |
| K1487 | HumanEval >= 46%           | 57.0%    | Yes    | PASS (surface)           |
| K1488 | MedMCQA >= 44%             | 56.0%    | Yes    | PASS (surface)           |
| K1489 | Max routing loss ≤ 10pp    | 2.0pp    | Taut.  | FAIL_RECLASSIFIED        |

3/4 KC pass on surface threshold; K1489 measures the wrong object under
the stated hypothesis. Per pre-flight rules, a single blocking antipattern
→ verdict KILLED.

### What is preserved

- E2E pipeline at N=25 has no wiring bugs (router trains, 25-way routing
  executes, adapter selection works, fallback path works).
- Adapter-domain routing at N=25 replicates Finding #531's direction
  (combined logistic ≈ 90% overall).
- MATH.md unchanged and git-clean from pre-registration; no KC swap.

### V2 requirements

1. ≥10 distinct trained adapters (expand `exp_p0_e2e_benchmark/` bank or
   reuse `exp_p0_ttlora_n10_scaling`).
2. At least one **near-neighbor adapter pair** (math + high_school_statistics,
   or medical + high_school_chemistry) so wrong-adapter routing is reachable.
3. Pre-register K1489 as conditional: measured router accuracy on adapter-
   domain benchmark queries ∈ [85%, 95%]. If ≥99%, vacate (don't pass).
4. Compare measured Δ against Theorem 1 within ±2pp at the measured p.

### Re-run decision

Not executed — the antipattern is structural (training ≥10 adapters is v2
scope, not a hotfix). Code reproducing the same 3-adapter protocol would
reproduce the same tautology.

---

## Round 1 — Initial Review (2026-04-17) — SUPERSEDED BY ROUND 2

### Verdict: APPROVED — No blocking issues

The experiment cleanly validates Theorem 1 at N=25 and closes the P0 gate.
The prediction-vs-measurement table is tight (≤1.6pp error). No blocking
mathematical flaws found. Minor methodological concerns noted below.

## Strengths

1. **Theorem 1 validated at a third independent scale point.** Three measurements
   (N=3, N=10, N=25) with prediction errors of 0pp, 1.2pp, 1.6pp respectively.
   This is genuine validation, not a one-off.

2. **Conservative predictions beat by actual results.** A-priori predicted 6pp
   max loss; measured 2pp. This suggests the theorem's assumptions (routing
   accuracy estimates) were if anything pessimistic, which is the right direction.

3. **Routing accuracy INCREASES with N.** This is a surprising and important
   structural result — the embedding-based logistic classifier benefits from
   more contrastive examples. This counters the naive expectation that more
   distractors = worse routing.

4. **Misrouting safety.** 100% of misrouted queries go to non-adapter domains
   (base model fallback), not wrong adapters. This is an important safety
   property of the current architecture.

## Concerns (Non-Blocking)

### C1: Only 3 Adapters Tested
The benchmark only measures quality with 3 trained adapters (math/code/medical).
With 25 trained adapters, wrong-adapter routing becomes possible and could
severely degrade quality. The current setup has a "structural moat" — 22 domains
with no adapters — that doesn't exist in the target 25-adapter system.

**Impact:** The quality loss measured here is a lower bound. Real 25-adapter
system quality loss could be higher if routing precision degrades between
similar-domain adapters (e.g., two medical subdomains).

**Mitigation:** This was explicitly noted in PAPER.md as Limitation #3.
A follow-up experiment (25 trained adapters) is the right next step.

### C2: N=100 Statistical Power
With N=100 per benchmark, the 95% CI is approximately ±10pp. The observed
+2pp MedMCQA improvement vs N=10 is likely noise. Any conclusion about
N=10→N=25 delta should be treated as directional only.

**Impact:** Minor — the kill criteria are pass/fail, not about deltas.
All 4 criteria pass by comfortable margins (≥8pp for K1488, ≥8pp for K1489).

### C3: Same 3 Oracle Adapters as N=10
The oracle adapters were trained on N=10 setup. They weren't re-trained
for N=25. If adding 15 MMLU domains during training affected adapter
quality, the oracle baseline would shift.

**Impact:** Negligible — the adapters are domain-specific LoRA weights
that don't depend on how many other domains exist. Adapter training
is independent of routing training.

## Mathematical Audit

**Theorem 1 consistency:** 
- GSM8K: T1 predicts 75.8% (using α=98%), measured 76.0% → error 0.2pp ✓
- HumanEval: T1 predicts 57.0% (using α=100%), measured 57.0% → error 0.0pp ✓
- MedMCQA: T1 predicts 54.4% (using α=88%), measured 56.0% → error 1.6pp ✓

The overperformance on MedMCQA (56% vs 54.4%) is consistent with the base
model having partial medical knowledge that helps misrouted queries. This is
a minor second-order effect not captured by Theorem 1's linear mixture model.

## Conclusion

The experiment is methodologically sound. The P0 "25 domains" gate closes
cleanly. The next frontier is testing with 25 trained adapters to measure
wrong-adapter routing risk.
