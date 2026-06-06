# PAPER.md — exp_model_arena_hard_composed

## Verdict
**KILLED_PREEMPTIVE** — target is unrunnable on `local-apple` / MLX / 48 GB
M5 Pro within a 120 min budget. Four independent preempt blocks
(T1 manual ∧ T2 ∧ T3 ∧ T5); runner reports three automated blocks
(T2 ∧ T3 ∧ T5) which alone over-determine the verdict.

## Prediction-vs-measurement

| Pred | Prediction | Measurement | Status |
|------|-----------|-------------|--------|
| P1 | T1 shortfall ≥ 3 of 5 required artifacts | Automated cooccur-grep: shortfall = **2/5** (false-positive matches in `pierre/pierre.py`, `pierre/plot_research_progress.py`, `macro/batched_lora_latency.py`, `micro/models/relora_composition_test/*`, `micro/models/structural_orthogonality_characterization/*`). Manual re-read: **5/5 absent** (zero `arena` references in any hit file). | Partial (automated) / PASS (manual) |
| P2 | T2 timing ≥ 120 min (conservative budget) | Conservative: **326.7 min** (500·2·15s + 500·5s + 1800s + 300s = 19,600 s); floor: **160.0 min** at 5 s/sample. Both exceed 120 min ceiling. | PASS |
| P3 | T3 DB has `success_criteria: []` + `⚠ INCOMPLETE` marker | `Success Criteria: NONE — add with …` and `⚠ INCOMPLETE: success_criteria, references, kill_results` both present. 10th occurrence of F#502/F#646 in drain. | PASS |
| P4 | T4 pin_ratio = 0 (dir absent); reinforce-only | `.audit/` absent; pin_ratio = 0.00; reinforce-only (does not block alone). | PASS (reinforce-only) |
| P5 | T5 source-scope breach count ≥ 3 (of 5 dimensions) vs SUPPORTED parent `exp_p1_t2_single_domain_training` | Parent verdict = `supported` (live DB read); breach = **5/5** across A (Arena-Hard prompts), B (LLM-judge pairwise), C (open-ended generation), D (N=5 composition), E (bootstrap win-rate CI). Well above threshold. | PASS |

## Kill criteria result

| KC | Text | Result |
|----|------|--------|
| K1700 | Pierre N=5 composition Arena-Hard win-rate vs base ≥ 50% | **fail** (preempt — target not run) |
| K1701 | 95% bootstrap CI excludes "worse than base" (lower bound > 40%) | **fail** (preempt — target not run) |

## Runtime evidence
- Runner: pure stdlib + `experiment get` shell-out.
- Wall: **1.93 s**.
- Zero MLX, zero model load, zero HTTP bind.
- `results.json` contains full probe output.

## Assumptions (from MATH.md §4, verified at runtime)
- A1 grep scope: `*.py` under `pierre/`, `macro/`, `composer/`, `micro/models/` (excluding this runner). Markdown planning docs excluded — `.ralph/current_direction.md` does NOT satisfy T1.
- A5 T2 floor at 5 s/sample still blocks (160 min > 120 min).
- A6 parent `Status: supported` confirmed live — standard T5, not T5-K.
- A7 runner is pure stdlib, ≤3 s wall — verified (1.93 s).
- A9 T1 cooccur-grep is too loose: it returns files where both patterns appear **anywhere**, not co-located or Arena-Hard contextualized. Manual grep `grep -l -i "arena"` against the hit files returns **zero matches**. The MATH prediction holds on manual read; runner refinement backlog logged.

## Novelty vs prior drain
No new F-axis. Reuses:
- F#502 (schema-completeness) — 10th occurrence.
- F#652 (software-infrastructure-unbuilt) under ap-017 composition-bug
  lineage — 28th composition-bug preempt (27 + this).
- Standard T5 (scope-breach, SUPPORTED parent) — 19th SUPPORTED-source
  preempt (iter 41 was 18th).

T5-K (parent-KILLED inheritance) does **not** apply — parent is
`supported`. No novel sub-axis this iter.

## Operator action required to unblock
1. Build or mirror the Arena-Hard-Auto v0.1 / v2.0 prompt set.
2. Stand up LLM-judge client (GPT-4-Turbo-2024-04-09 or
   calibrated local judge) with rate-limit + retry + cost budget.
3. Serve Pierre N=5 composed model (blocked transitively by
   exp_prod_mlxlm_integration / exp_prod_pip_package_pierre
   — both KILLED).
4. Serve base Gemma 4 E4B with matched sampling config.
5. Implement pairwise win-rate + bootstrap-CI framework
   (≥1000 resamples over 500 games).
6. Re-file target once (1)-(5) exist.

Until then, the experiment remains `killed` with a stable,
re-testable preempt theorem. Child experiments that depend on
Arena-Hard pairwise win-rate infrastructure inherit the block.
