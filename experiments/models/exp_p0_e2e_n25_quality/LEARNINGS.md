# E2E N=25 Quality Validation — Learnings

## AUDIT RE-CLASSIFICATION (2026-04-18) — verdict KILLED

Tags: `audit-2026-04-17-rerun`, `tautological-routing`.

Same antipattern as sibling `exp_p0_e2e_combined_routing_n10`. The title
promises "routing at scale" with N=25 domains, but `run_experiment.py` reuses
only the **3 adapters** (math/code/medical) from `exp_p0_e2e_benchmark/`.
The 22 non-adapter domains form a structural safety net — every misrouted
query falls back to the base model, never to a wrong adapter. K1489
("max routing loss ≤ 10pp vs oracle") is tautological under this protocol:

- Theorem 1 bound: `loss ≤ (1 − α)·(Q_oracle − Q_base)`.
- Max delta in this run: 62pp (GSM8K oracle 77 vs base 15).
- Measured adapter α: math 98%, code 100%, medical 88% → max theoretical
  loss ≈ 0.12·(58−28) = 3.6pp on medical. K1489's 10pp threshold cannot
  fail unless α drops below ~83%, which the 22-MMLU safety moat makes
  structurally impossible.

The hypothesis promised — "combined logistic routing tolerates misrouting
across 25 real adapters at scale" — was **not tested**. Wrong-adapter
collisions (the actual failure mode) are unreachable by construction.

Re-classified KC under pre-reg:
- K1486 PASS_SURFACE (GSM8K 76% ≥ 68% — held, but via tautology protocol)
- K1487 PASS_SURFACE (HumanEval 57% ≥ 46%)
- K1488 PASS_SURFACE (MedMCQA 56% ≥ 44%)
- K1489 FAIL_RECLASSIFIED (antipattern #6 — KC measures wrong object)

Verdict: KILLED on pre-registered intent. Same cluster-level remedy as the
N=10 sibling: v2 requires ≥10–25 distinct trained adapters + conditional
K1489 vacate-clause (router accuracy ∈ [85%, 95%] on benchmark queries).

Re-run not executed — this is a **structural** antipattern, not a code bug.
Running the same code again with the same 3 adapters reproduces the same
tautology. MATH.md preserved git-clean as pre-registered; no KC swap.

## V2 path

A follow-up `..._n25_v2` experiment must:
1. Use ≥10 distinct trained adapters (expand `exp_p0_e2e_benchmark/` bank,
   or reuse `exp_p0_ttlora_n10_scaling` adapters).
2. Pre-register K1489 as conditional: require measured router accuracy on
   adapter-domain benchmark queries ∈ [85%, 95%]; if ≥99% the run is
   vacated, not passed.
3. Include ≥1 **near-neighbor adapter pair** (e.g., math + high_school_statistics,
   or medical + high_school_chemistry) so wrong-adapter routing is geometrically
   possible.
4. Compare measured Δ against Theorem 1's prediction within ±2pp.

## Cross-reference

Same audit pattern as `exp_p0_e2e_combined_routing_n10` (N=10 sibling,
documented above) and `exp_p8_vproj_domain_behavioral` (Round 2 review,
2026-04-18).

---

## Original (2026-04-17) — SUPERSEDED

### Outcome: All kill criteria PASS. P0 "25 domains" gate CLOSED.

### What We Proved
Theorem 1 (Q_routed = α·Q_oracle + (1−α)·Q_base) holds at N=25 with ≤1.6pp
prediction error when measured routing accuracy is used. The E2E pipeline
scales from N=3 → N=10 → N=25 without quality degradation.

### Key Numbers
- GSM8K: 76% routed vs 77% oracle → 1pp loss (predicted 3.1pp)
- HumanEval: 57% routed vs 57% oracle → 0pp loss (predicted 4.7pp)
- MedMCQA: 56% routed vs 58% oracle → 2pp loss (predicted 6.0pp)
- Max loss: 2.0pp (K1489 threshold: 10pp) — 5× better than threshold
- N=10 → N=25 routing accuracy: IMPROVED (math 98%, code 100%, medical 88%)

### Unexpected Finding: Routing Improves With Scale
A-priori predictions assumed routing accuracy would degrade when going from
N=10 to N=25 (more distractors). The opposite happened: routing improved
by 0-3pp for all adapter domains. Hypothesis: the logistic classifier with
sentence embeddings benefits from MORE contrastive examples — the 15 additional
MMLU subjects help the decision boundary learn what math/code/medical is NOT.
This is consistent with the linear separability of sentence embedding spaces.

### What This Closes
- P0 gate "25 domains" is CLOSED by this experiment
- Combined logistic routing + sentence embeddings is validated as the
  production routing architecture for N=25
- Theorem 1 is validated across N=3, N=10, N=25 (3 independent measurements)

### Structural Note on Misrouting
The 12% MedMCQA misroutes (chemistry, statistics, physics) all go to
non-adapter MMLU domains → base model fallback. This is BENIGN — no
wrong-adapter risk. The base model quality (28%) averages into the 56%
result (vs 58% oracle). If medical queries went to a wrong adapter (e.g.,
code adapter), quality could drop to 0-5%. The current 25-domain structure
with only 3 adapters provides a natural safety zone.

### Next Steps (from PAPER.md)
1. Train adapters for more domains (legal, finance, etc.) to increase coverage
2. Test with 25 trained adapters (not just 3) to measure wrong-adapter risk
3. The N=100 scaling path is clear if α > 85% per adapter domain
