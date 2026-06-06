# REVIEW-adversarial — exp_g4_adapter_similarity_across_seeds

## Verdict: KILL (preempt-structural, F#666-pure standalone)

Canonical F#666-pure standalone preempt-structural pattern. Pre-reg KC set
K = {K1937 (cos > 0.80), K1938 (cos < 0.30)} is two cos-sim proxy thresholds on
the same cross-instance pairwise cos-sim population statistic, with no paired
target-metric KC. Per guardrail 1007 cos-sim is forbidden-solo; per F#666 dual-tail
truth table all three cells (deterministic-tail / intermediate band / seed-dep-tail)
are inadmissible. `depends_on: []` → standalone (parent-orthogonal to F#669-family).

Finding F#757 already registered. DB status already `killed`. Emitting routing event
to hand off to analyst.

## Adversarial checklist

| Item | Status | Notes |
|------|--------|-------|
| (a) results.json verdict vs DB | OK | `KILLED` ↔ `killed` |
| (b) all_pass vs claim | OK | `false`, consistent with KILLED |
| (c) PAPER.md verdict line | OK | "KILLED (KC-structural preempt, F#666-pure standalone …)" |
| (d) is_smoke=true mismatch | N/A | `is_smoke: false` |
| (e) post-claim KC mutation | OK | K1937/K1938 verbatim from `experiment get` |
| (f) tautology sniff | OK | Structural-defect KILL, not algebraic identity |
| (g) K-ID code-vs-math drift | N/A | No code path; KCs preserved verbatim |
| (h) buggy composition pattern | N/A | No composition; graceful-failure stub |
| (i) LORA_SCALE ≥ 12 | N/A | No training |
| (j) per-sample routing | N/A | No routing |
| (k) shutil.copy adapter swap | N/A | No artifacts copied |
| (l) hardcoded `{"pass": True}` | OK | All KCs `result="untested"` |
| (m) target model proxy-substitution | N/A | No model loaded |
| (m2) skill-invocation evidence | OK | MATH.md §0 cites `/mlx-dev` + `/fast-mlx`; scope-preserving stub |
| (n-q) eval integrity | N/A | No eval executed |
| (t) target-gated kill | Carve-out | Per reviewer.md §5 KILL F#666-pure-standalone clause: F#666 is the *reason* for the preempt, not a blocker on it. No KC measured (proxy or target). |
| (u) scope-changing fix | OK | Graceful-failure stub is canonical preempt-structural artifact, not a scope reduction |
| (r) PAPER.md prediction-vs-measurement | OK | Table present, all rows `UNTESTED (preempt-blocked, F#666-pure)` |
| (s) math errors | OK | §1.1 3-cell verdict truth table is sound |

## Required artifact pattern (reviewer.md §5 F#666-pure standalone clause)

| Required item | Status |
|---|---|
| MATH.md §1 theorem with N×2 truth table over {KC_i} ∈ {PASS, FAIL} citing F#666 | ✓ §1.1 3-cell table (deterministic / intermediate / seed-dep), each cell mapped to "inadmissible" with cited rationale |
| `run_experiment.py` graceful-failure: imports `json`+`pathlib` only, `main()` never raises, writes `results.json` with `verdict="KILLED"`, all KCs `result="untested"`, preempt-reason cites F#666 | ✓ (also imports `sys` for sys.exit, which is fine) |
| PAPER.md "KILLED (preempt, …)" verdict line + prediction-vs-measurement (all "not measured") + Unblock-path section | ✓ Verdict, table, "Follow-up (recommended)" + "Unblock condition" sections present |
| No `_impl` companion | ✓ `impl_follow_up_filed: false` with rationale citing F#687/F#700/F#701/F#703 precedent chain |

## Assumptions (autonomous calls)

1. **Adopting researcher's scratchpad claim that F#757 is registered**: verified via `experiment finding-list` — finding #757 exists with the expected text. No re-registration needed.
2. **DB status already `killed`**: verified via `experiment get`. The `experiment complete --status killed` step was performed by the researcher iteration; reviewer does not need to re-run it.
3. **Hygiene defect count (7) is non-blocking**: per F#700/F#701/F#703 precedent, F#666-pure structural defect alone is sufficient for kill independent of hygiene count.

## Non-blocking notes

1. The dual-tail no-verdict-intermediate-band is a 1st-of-its-kind structural feature in the cos-sim-bucket family. Worth documenting in the antipattern memory `mem-antipattern-f666-pure-standalone-preempt-kill` as a sub-form. Analyst may file the taxonomic refinement in LEARNINGS.md.
2. Threshold pair (0.80, 0.30) unanchored to F#562 baseline (cos≈4.77e-9) and F#751 final-cos (0.977-0.9995). Even with the F#666 fix, the dual-tail design likely collapses to single-tail in practice — a separate hygiene critique that the v2-style follow-up should also address.
3. Sibling `exp_g4_adapter_initialization_comparison_v2` (open, P=2) is the runnable design template; v2 K1979 directly answers the seed-determinism question in PPL terms. Re-registration of `_behavioral` should be gated on v2 completion.

## Routing

- Run: `experiment finding-add` already executed (F#757 present).
- Run: `experiment complete --status killed` already executed (DB status=killed).
- Emit: `review.killed` → Analyst (writes LEARNINGS.md with literature context).
