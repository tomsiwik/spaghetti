# LEARNINGS: exp_g4_e2e_n25_25real

## Core Finding
KILLED_PREEMPTIVE (22nd consecutive; 1st `tautological-routing` branch under ap-017). 5-theorem defense-in-depth: T1 (shortfall=22 vs {code,math,medical}), T3 (Success Criteria=NONE), T5 (F#534 N-composition scope non-transfer) each alone blocks SUPPORTED; T2 (460.24 min ≫ 120 min micro ceiling) and T4 (0/5 pins) reinforce. K1617 unmeasurable.

## Why
F#534's impossibility guarantee rests on `N_adapter < N_total` — base-model fallback for unrouted domains provides the safety zone. K1617 "25 real adapters" inverts the scope (25/25 adapter, 0 safety zone), voiding the guarantee. F#534 caveats are LITERAL 3/3: "Only 3 adapters tested", "wrong-adapter routing risk not yet measured", "non-adapter domains provide safety zone" — all three triggers present. No new adapters were (or could be) trained inside the iter budget; T2.1 cascade delivered only 3 Gemma 4 domain adapters. F#534 is the 3rd SUPPORTED-source preempt registered under ap-017 (after F#505 g, F#454 h), consolidating the rule: source-verdict is NOT the guardrail — scope-caveat literal text is.

## Implications for Next Experiment
1. **Pivot OFF audit-2026-04-17 cohort** (10 prior handoffs; claims keep returning cohort members). Non-cohort P≤2 remaining per prior forecasts: `exp_g4_polar_scale_invariance`, `exp_g4_single_domain_vproj_think`, `exp_g4_activation_bounds_vproj`, plus P0 macro (`exp_p9_cispo_adapter_rl`, `exp_p9_self_evolution_scaffold` — may need operator handoff for macro scale).
2. **If claim returns cohort member:** drain via same 5-theorem stack. ap-017 preempts (a)–(i) registered; coverage broad across composition-bug (20), scale-safety (1), tautological-routing (1, NEW).
3. **Operator unblock remains only cohort accelerator:** SC add + 22 new Gemma 4 domain datasets + K pinning {OOD/in-dist partition, ε, Gemma 4 FP16 rescale formula, enumerated 25-domain list}, OR N≤3 re-scope (which then duplicates F#534 scope; no new information).
4. **RESEARCH_BACKLOG_DRAINED** per objective if non-cohort P≤2 exhausts — escalate to operator.
5. **Non-blocking runner gap (cohort-wide):** T4 enumerated-domain regex or numeric-ε patch still owed; T4 already hardened to word-boundary regex from iter 24 (robust, no substring false-positive).
