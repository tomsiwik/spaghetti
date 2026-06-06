# PAPER.md — exp_g4_polar_scale_invariance

**Verdict: KILLED (preemptive, 5-theorem defense-in-depth).**

Cohort: audit-2026-04-17 / scale-safety / g4-gemma4.
Branch: ap-017 scale-safety (2nd instance after exp_g4_activation_bounds_vproj).
Source: Finding #444 (PoLAR 3× stability on Qwen proxy).

## Prediction vs measurement

| KC | Predicted | Measured | Pass |
|---|---|---|---|
| #1621 PoLAR variance ≤ 4pp | N/A (no adapters) | T1 block: 0/8 adapters | fail |
| #1622 LoRA variance ≥ 10pp | N/A (no adapters) | T1 block: 0/8 adapters | fail |

Runner never trained; verdict derived from 5-theorem preempt stack.

## 5-theorem stack (all block)

- **T1** 8 adapters needed, 0 available (shortfall=8)
- **T2** 160 min required > 120 min micro ceiling
- **T3** success_criteria=[] + ⚠ INCOMPLETE DB flag
- **T4** 3/5 KC methodology pins (ε-threshold regex boundary miss; pooled absent)
- **T5** F#444 LITERAL caveat 3/3 triggers: "behavioral advantage cannot
  be confirmed" + "near chance accuracy" + "QK-norm in Gemma 4 provides
  baseline scale protection … regardless of PoLAR"

**T5 scope-transfer argument.** F#444 source scope = Qwen-proxy
architectures lacking QK-norm. Gemma 4 HAS QK-norm (Gemma 4 reference
guide). Stiefel stabilizer for q/k projections redundant with
architectural normalizer → variance differential is confounded
with architecture, not attributable to PoLAR. Source caveat makes
this explicit verbatim.

**Defense-in-depth.** T1 ∨ T3 ∨ T5 each alone → KILLED_PREEMPTIVE.

## Assumptions

- F#444 caveat verbatim is authoritative source-scope definition
- Gemma 4 QK-norm is architectural (per reference_mlx_gemma4.md memory)
- Training cost ≈ 20 min/adapter (consistent with cohort prior runs)

## §P — pre-registered probe: N/A

Preemptive kill; no probe required. Runner is pure stdlib, no model
access, 1 s wall time.

## ap-027 (adapter/scale misconfig sentry): N/A

No adapters were loaded or misused.

## Cohort accounting

23rd→24th preemptive-kill in audit-2026-04-17 cohort drain.
Branches under ap-017 umbrella:
- composition-bug: 20
- scale-safety: 2 (was 1; this experiment adds)
- tautological-routing: 1
- projection-scope: 2

## Non-blocking cohort-wide gap (unchanged)

T4 ε pin regex owed cohort-wide patch: current
`re.search(r"\b<=\s*\d", …)` misses unit-suffix patterns like
`<= 4pp` because `\b` before `<=` fails. Recommend
`r"<=\s*\d+\s*(pp|%|[A-Za-z]+)?"`. Does not change this verdict
(T1 ∨ T3 ∨ T5 block independently).
