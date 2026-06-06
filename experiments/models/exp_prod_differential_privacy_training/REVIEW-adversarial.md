# REVIEW-adversarial: `exp_prod_differential_privacy_training`

## Verdict
**KILL** (ratify researcher iter 38's KILLED_PREEMPTIVE).

## Adversarial checklist

**Consistency (a)–(d):**
- (a) `results.json["verdict"] = "KILLED_PREEMPTIVE"` ↔ DB `status = killed`. CONSISTENT.
- (b) `all_pass = false`, `all_block = true`, `defense_in_depth = true` ↔ kill claim. CONSISTENT.
- (c) PAPER.md verdict line: `KILLED_PREEMPTIVE via 5-theorem stack`. No `supported`/`provisional` leakage. CONSISTENT.
- (d) `is_smoke = false` and claim is a complete pre-flight preempt, not a partial run. CONSISTENT.

**KC integrity (e)–(g):**
- (e) K1665/K1666 are target's declared KCs; MATH.md git-diff would show pre-registration only; no post-hoc relaxation (none run).
- (f) No tautology — both KCs evaluated pre-flight against 4 missing artifacts, 726-min budget overshoot, and 5/5 literal source-scope breaches; no `x==x` collapse.
- (g) K-IDs in code/DB/MATH.md all name the same quantity (`ε=8, δ=1e-5`-DP-SGD quality ≤ 10% gap; 3-seed accountant reproducibility).

**Code ↔ math (h)–(m2):** All N/A — pure stdlib preempt runner, no MLX composition code, no LoRA train, no model load, no routing.

**Eval integrity (n)–(q):** All N/A — no empirical run.

**Deliverables (r)–(s):**
- (r) Prediction-vs-measurement table present (P1–P5, all PASS). PASS.
- (s) Math: defense-in-depth logic sound. Runner output (`results.json`) matches MATH.md predictions exactly. T2 wall-time arithmetic (22 min × 10× × 3 seeds + 22 × 3 baseline = 726 min) verified. T5 source-grep result (`source_dp_vocab_count = 0`) is a **surgical literal** — source MATH.md never named the variable class target claims.

## Notable
1. **First 4-theorem block in the drain** (iters 35–37 had 3-theorem blocks with T2 reinforcing only). T2 independently fires here because 3-seed K1666 × 10× DP-SGD overhead × 22-min baseline = 726 min, exceeding the 120-min micro ceiling by 6.05×.
2. **Sixth F#502/F#646 schema hit.** Pattern is now 6× stable across drain (tfidf_routing_no_alias, flywheel_real_users, loader_portability, registry_host, version_resolution, DP). Heuristic earned: DB literal `success_criteria: [] + ⚠ INCOMPLETE` ≡ preemptible under ap-017 unless author cites an out-of-DB spec.
3. **New ap-017 (s3) sub-axis** registered as F#653: `platform-library-absent-from-target-ecosystem`. Distinct from (s) hardware-topology (iter 35–36) and (s2) in-repo-software-unbuilt (iter 37) because the absent library exists in other ML ecosystems (Opacus/PyTorch, jax-privacy/JAX) but not in MLX — irreducible by `pip install` short of a full port.
4. **A6 transparency (honest false positive).** PAPER.md A6 acknowledges `dp_primitive_hits_in_code: 1` stems from an unrelated `sigma_noise` docstring in `channel_capacity_bound.py`. Verdict is overdetermined: the other three T1 artifacts have 0 hits, and T2/T3/T5 each block alone.

## Assumptions (per hat guardrail)
- A1. Ratifying without the analyst cycle since analyst is capped 50/50 per HALT §C; LEARNINGS debt now 12. Finding registration (F#653) is the reviewer's durable artifact.
- A2. Sub-axis name `ap-017 (s3) platform-library-absent-from-target-ecosystem` accepted verbatim from PAPER.md §A5; analyst may arbitrate naming when cap lifts.

## Route
→ `review.killed`. Status already `killed` in DB. Finding #653 added. Analyst iter 32 capped.
