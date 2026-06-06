# REVIEW-adversarial — exp_pierre_multi_adapter_serving_throughput

**Verdict:** KILL (preempt-structural sub-case, F#669 12th reuse)

## One-line
12th F#669 reuse, 1st Pierre-serving-cluster child preempt-KILL (parent F#570 5-precondition cascade). All 4 preempt-structural artifact requirements met; target-only-KC-panel-under-preempt-KILL micro-pattern canonicalizes at 3rd obs cross-cluster triple-fire (F#738 behavioral/MEMENTO + F#739 engineering/MEMENTO + F#740 this engineering/Pierre-serving).

## Adversarial checklist (preempt-structural sub-case)

**Consistency (a–d):** PASS
- (a) `results.json["verdict"]="KILLED"` matches DB `status=killed` (F#740 already filed).
- (b) `all_pass=false` matches K1911/K1912 both `untested`.
- (c) PAPER.md verdict line: `KILLED — preempt-structural (F#669 12th reuse)` — consistent.
- (d) `is_smoke=false` and claim is preempt-KILL (not full-run) — consistent.

**KC integrity (e–g):** PASS
- (e) MATH.md is new for this experiment; KCs K1911/K1912 are pre-registered in DB and not modified post-claim.
- (f) No tautology — both KCs are well-defined engineering thresholds (50% throughput ratio, 40GB memory ceiling); they are unmeasured, not vacuously satisfied.
- (g) K1911/K1912 measure exactly what MATH.md describes (concurrent-stack throughput ratio, peak memory at N=5).

**Code ↔ math (h–m2):** PASS (vacuous for h–l; explicit for m, m2)
- (h–l) No composition code, no LORA_SCALE, no routing, no shutil.copy, no hardcoded `pass:True` (all KC results are `"untested"`). N/A — no MLX code path.
- (m) No proxy-model substitution. §6 explicitly rejects all 6 silent-swap shortcuts (single-stack repeated, sequential-N=1-summed, untrained-config-shells, base-gemma-substitute, vLLM/llama.cpp cross-framework, analytical memory back-derivation).
- (m2) Skills `/mlx-dev` and `/fast-mlx` cited in MATH.md §0 as "noted, not invoked — no MLX code written; honest disclosure". This satisfies (m2) without MLX code landing — consistent with novel-mechanism / preempt-structural design-only artifact pattern.

**Eval integrity (n–q):** N/A — no eval, no headline N, no baseline.

**Target-gated kill (t):** N/A by F#669 carve-out — preempt-structural KILL is a structural verdict where NO KC was measured (proxy or target). F#666 is satisfied by vacuous quantification (both K1911 and K1912 are engineering targets, no proxy to pair); F#669 governs the preempt-KILL decision, not F#666.

**Scope-changing fix (u):** PASS — graceful-failure stub (`run_experiment.py` imports only `json`+`pathlib`, never raises, writes results.json with `verdict=KILLED`) is the canonical preempt-structural artifact, NOT a silent scope reduction. Original KC scope (concurrent multi-adapter serving on `pierre-g4e4b`) is preserved as the unblock requirement.

**Deliverables (r, s):** PASS
- (r) PAPER.md has prediction-vs-measurement table — both rows "unmeasured (preempt-blocked)".
- (s) No math errors. F#570 preconditions (T1B/T1C/T2/T3/DEP) accurately mapped onto K1911/K1912 with explicit blocking-clause for each.

## Preempt-structural artifact pattern (F#669 12th-reuse compliance)

1. **MATH.md §1 transitivity theorem:** PRESENT. Derives that K1911/K1912 are functions of parent F#570's unverified target claim; both produce `NaN/NaN` and `NaN` respectively while parent is KILLED. ✓
2. **`run_experiment.py` graceful-failure:** PRESENT. No MLX path; `main()` writes results.json directly with `verdict="KILLED"`, `all_pass=false`, both KCs `result="untested"` with explicit preempt-reason citing F#570 + F#669. ✓
3. **PAPER.md prediction-vs-measurement table + Unblock path section:** PRESENT. Table shows both KCs "unmeasured"; "Why this is not 'skip and rerun later'" section enumerates upstream/produced-artifact/parent-extension blockers. ✓
4. **No `_impl` companion:** CORRECT. Preempt-structural KILL is self-contained per F#687/F#698/F#699/F#737/F#738/F#739 precedent + reviewer.md §5. Unblock is parent-external (F#570 resolution) + parent-extension (multi-adapter scheduler), both at Pierre-serving-infrastructure layer not under this child. ✓

## Cross-cluster triple-fire canonicalization

3rd observation of target-only-KC-panel-under-preempt-KILL micro-pattern, cross-cluster:
- 1st: F#738 (behavioral target-only, MEMENTO cluster, parent F#685)
- 2nd: F#739 (engineering target-only, MEMENTO cluster, parent F#685)
- 3rd: F#740 this experiment (engineering target-only, Pierre-serving cluster, parent F#570) — **first cross-cluster instance**

Cross-cluster independence is the strongest form of triple-fire: micro-pattern is not confined to a single parent's idiosyncrasies. Promotion watchlist → canonical is supported. Canonical form: an F#669 child whose KC panel is target-only (engineering OR behavioral) with no pairable proxy satisfies F#666 by vacuous quantification rather than compound pairing — legitimate F#666-compliance path.

## Sub-axis classification (reviewer call)

Researcher conservatively classified this as a "2-point serving-config spot-measurement at N∈{3,5}" — distinct variant, NOT a canonical scalar-sweep, because K1911/K1912 measure distinct metrics at different N points (not the same metric across a range). **Reviewer concurs.** Multi-parent-run sub-axis remains at 2 observations (F#737 scalar-sweep + F#738 categorical); a genuine 3rd same-metric-across-configs observation is still pending. Serving-config spot-measurement variant is at 1 observation (this); not yet a watchlist pattern.

## Hygiene notes (non-blocking)

- Experiment record has `platform=~` (null) and empty `success_criteria` (F#702 hygiene defect, 2 items). Per F#702 precedent, preempt-KILL supersedes hygiene correction; patchable at re-claim time. Not blocking the verdict — both KCs remain unmeasurable regardless of hygiene state.

## Assumptions logged

- F#570 preconditions inherited from 2026-04-18 source inspection; not re-probed this iteration. Defensible: if any parent precondition had been resolved, parent status would have moved from `killed` to `supported`, which it has not.
- "Cross-cluster triple-fire" canonicalization at 3rd obs follows mem-pattern-triple-fire; this is the 1st cross-cluster (F#738/F#739 share MEMENTO parent F#685; this child is Pierre-serving parent F#570). Two clusters, three observations is sufficient for canonical promotion.

## Routing

KILL — emit `review.killed`. F#740 already filed by researcher (verified via `experiment query`). No additional `experiment finding-add` needed.
