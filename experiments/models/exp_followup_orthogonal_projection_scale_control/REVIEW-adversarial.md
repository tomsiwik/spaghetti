# REVIEW-adversarial: exp_followup_orthogonal_projection_scale_control

**Verdict: KILL (confirm)**

**One-line reason:** Pre-registered theoretical-refutation probe — all 3 preconditions verified against parent `results.json` + `PAPER.md`, Pareto front shows no scale in {4,6,8,10} passes all three KCs under parent's closure-theorem scaling model. KC #1573 = FAIL via R-struct and R-pareto. Verdict consistent across `results.json` (KILLED), PAPER.md (KILLED), and DB (killed).

## Adversarial checklist

**Consistency (a–d):**
- (a) `results.json.verdict = "KILLED"` ↔ DB `killed` — consistent.
- (b) `all_pass=false` with `kill_criteria.1573.pass=false` — consistent with KILL.
- (c) PAPER.md verdict line says `KILLED` — consistent.
- (d) `is_smoke=false` — probe is a single deterministic derivation, not a smoke run.

**KC integrity (e–g):**
- (e) KC #1573 text in DB matches MATH.md §E and results.json (`At scales {4,6,8,10} with autograd projection, orth-projection claim holds independent of scale`). No post-hoc relaxation.
- (f) Tautology sniff: conclusion draws on parent's independently-measured spectral gap (1.003–1.005, matching PAPER §Spectral Gap) and ρ_k reduction (1.16e-5). Not a tautology — parent data is a real empirical anchor. Extrapolation assumption (linear delta scaling × rank-level K3 floor) is disclosed in Limitations.
- (g) Code-computed KC matches MATH.md: `_check_precondition_P*` and `_derive_pareto_front` implement §C/§D formulas.

**Code ↔ math (h–m2):**
- (h) No LoRA summing — no adapter code at all. N/A.
- (i) No LORA_SCALE=20 hardcoded (no training).
- (j) No single-sample routing (no inference).
- (k) No `shutil.copy` of sibling adapters.
- (l) No hardcoded `{"pass": True}` KC dict — `k1573.pass` is computed from `R_struct OR R_pareto`.
- (m) No target-model substitution — no model loaded.
- (m2) Skills `/mlx-dev`, `/fast-mlx` explicitly logged as **not invoked because no MLX code** is produced (MATH.md §H, PAPER.md "Assumptions Logged"). Compliant per PLAN.md Part 2.

**Eval integrity (n–q):** N/A — no eval run.

**Deliverables:**
- (r) PAPER.md contains "Predictions vs Measurements" table with 7/7 matches.
- (s) Math: Thm F1 Pareto enumeration is correct under stated linear-scaling + 80/20 capacity decomposition assumption. Both assumptions inherited from parent PAPER.md §Audit-Rerun Closure (2026-04-18) Thms C1/C2/C3.

## Assumptions (reviewer judgment calls)

- Probe is legitimately classified as theoretical-refutation (no new empirical measurement). It reads parent data + re-applies parent's closure theorems to a universal-quantifier claim on scale. The extrapolation is defensible because (i) parent's Thm C1 is scale-invariant by construction (base-weight SVD property), and (ii) parent's Thm C3 explicitly decomposes the in-dist loss as 80% rank-level + 20% direction-level, so K3 cannot be rescued by shrinking scale.
- Status already set to `killed` in DB. No `experiment complete` call needed; route via `review.killed` to hand off to Analyst for LEARNINGS.md.

## Non-blocking observations
- LEARNINGS.md not yet written — that's the Analyst's next step.
- Pareto-front row for s=4 predicts K3=0.58 (not 0.90 as MATH.md §D estimated in the informal table at lines 111–114). The precise per-scale K3 uses the 80/20 decomposition in `_derive_pareto_front`, which is the mathematically consistent version — the MATH.md §D table was a qualitative sketch and results.json applies the precise formula. Not a blocker because the direction-of-kill (K3 fails at every s) is preserved and matches the §F predictions.

## Verdict: KILL confirmed.

Route: `review.killed` → Analyst writes LEARNINGS.md.
