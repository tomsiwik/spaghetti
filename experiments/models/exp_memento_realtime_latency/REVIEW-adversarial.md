# REVIEW-adversarial.md — exp_memento_realtime_latency

## Verdict: KILL (preempt-structural, F#669-family, 11th reuse)

Researcher's filing is well-formed. All artifacts match the F#669-family preempt-structural KILL clause in `reviewer.md §5`. No blocking defects; no REVISE-worthy issues surface.

## Adversarial checklist

| Item | Status | Notes |
| --- | --- | --- |
| (a) `results.json["verdict"]` vs DB claim | PASS | verdict=KILLED, DB status killed |
| (b) `all_pass` vs claim | PASS | `all_pass=false`, killed |
| (c) PAPER.md verdict line | PASS | "KILLED (preempt, F#669 — ≥11 reuses; 4th MEMENTO-cluster child)" |
| (d) smoke vs full | PASS | `is_smoke=false` (no empirical run) |
| (e) Pre-reg KC mutation | PASS | K1907/K1908 as pre-registered, not relaxed |
| (f) Tautology sniff | N/A | preempt-structural; no KC measured |
| (g) K-ID drift | N/A | no code, no measurement |
| (h) composition bug | N/A | no code |
| (i) LORA_SCALE ≥ 12 | N/A | no LoRA |
| (j) shared-route | N/A | no routing |
| (k) shutil.copy adapter | N/A | no code |
| (l) Hardcoded pass True | PASS | `all_pass=false` written |
| (m) Proxy model substitution | PASS | §6 explicitly rejects Qwen3/Phi-4/Olmo 3 substitution |
| (m2) Skill invocation evidence | PASS | MATH.md §0 cites `/mlx-dev` + `/fast-mlx` with honest "not invoked — no MLX code" disclosure |
| (n) Base=0% thinking | N/A | no eval |
| (o) Headline n<15 | N/A | no n |
| (p) Synthetic padding | N/A | no N |
| (q) Baseline drift | N/A | no measurement |
| (r) Prediction/measurement table | PASS | present, all rows "not measured" |
| (s) Math errors | PASS | theorem derivation correct |
| (t) Target-gated kill | **carve-out applies** | F#669-family preempt-KILL — NO KC was measured (proxy or target); F#666 gate does NOT apply per reviewer.md §5 F#669 clause |
| (u) Scope-changing fix | PASS | graceful-failure stub is the canonical preempt-structural artifact, not a scope change |

## Preempt-structural preconditions (F#669 clause)

1. **MATH.md §1 theorem**: ✓ derives transitivity — K1907/K1908 both depend on Gemma-4-MEMENTO forward pass existence; parent PROVISIONAL per F#685; measurement is unidentifiable (NaN > 50ms / NaN/NaN).
2. **run_experiment.py graceful-failure**: ✓ imports only `json` + `pathlib`; `main()` never raises; writes `results.json` with `verdict="KILLED"`, both KCs `result="untested"` with preempt-reason citing F#669 + parent F#685.
3. **PAPER.md**: ✓ prediction-vs-measurement table (all "not measured"), "KILLED (preempt, F#669)" verdict line, **Unblock path** section listing parent-SUPPORTED + parent-extension requirements.
4. **No `_impl` companion**: ✓ correct — preempt-structural kill is self-contained; unblock is parent-external (parent's own `_impl` P=3 already filed).

## Sub-axis / micro-pattern classification

- Single-config engineering-target-only — same structural class as F#699. Multi-parent-run sub-axis NOT advanced (remains 2 obs: F#737 scalar-sweep + F#738 categorical).
- 2nd observation of "target-only-KC-panel-under-preempt-KILL" watchlist micro-pattern (1st F#738 behavioral, this engineering). Below triple-fire threshold, logged as watchlist only — correct handling.

## Assumptions

- Researcher already ran `experiment complete exp_memento_realtime_latency --status killed …` and filed finding #739 (F#669 11th reuse). Verified via `experiment query exp_memento_realtime_latency` showing finding #739 present. Reviewer does not re-run `experiment complete` — only routes via event.
- No round-count concern: first-pass filing, single review iteration — no revise cycle accumulation.
- F#666 carve-out: `(t)` does not block preempt-KILL (no KC measured); same logic as F#669-family clause in reviewer.md.

## Route

`review.killed` — no finding-add needed (F#739 already filed by researcher).
