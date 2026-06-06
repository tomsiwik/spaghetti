# MATH — exp_g4_real_user_registry (PREEMPTIVE-KILL)

## Claim under test
K1613: register < 10ms; K1614: crystallize < 5ms; K1615: max_cos < 0.15
on "real heterogeneous Gemma 4 adapters".

## 5-theorem preemptive stack (verdict: KILLED_PREEMPTIVE a priori)

### T1 (Inventory) — FAIL
"Heterogeneous users" requires real user-adapters (personal-style) on Gemma
4. Current T2.1 inventory at `exp_p1_t2_single_domain_training/adapters/`:
`{code, math, medical}` — 3 DOMAIN adapters, 0 USER adapters on Gemma 4.
Shortfall ≥ 1 (heterogeneous ⇒ ≥2 users required for the adjective to
bind).

### T2 (Budget) — FAIL
Training even a single Gemma 4 user-adapter requires ≥1 × 20.92 min (per
T2.1 single-domain train time). Two heterogeneous users → ≥41.84 min >
30 min iter budget.

### T3 (Success Criteria Gate) — FAIL
`experiment get exp_g4_real_user_registry` → `Success Criteria: NONE`.
DB emits `⚠ INCOMPLETE: missing success_criteria`. Per PLAN.md §1,
`supported` verdict is ungated when SC=[]. Kills are still valid
(KC-only).

### T4 (Kill Criteria Pinning) — FAIL
K1613/K1614/K1615 specify numeric thresholds {10ms, 5ms, 0.15} but do
not pin: hardware (M5 Pro vs M1 Max), adapter rank, Gemma 4 E2B vs E4B,
heterogeneity definition (how different users must be), ε for max_cos,
lifecycle phase (pre- vs post-crystallization). Unpinned ⇒ a priori
non-falsifiable claim.

### T5 (Non-transfer / Scope cavea) — FAIL
F#454 (SUPPORTED, T6.5 Dynamic Adapter Registry): register 1.20ms,
crystallize 1.85ms, max_cos_post=0.1221 — PASSED thresholds by 3-4
orders of magnitude. BUT F#454 caveat LITERAL:
> "Kill thresholds 1000-30M× above measured values (non-discriminating).
> K1136 'throughout' tested on final state only; intermediate
> max_cos=0.9580 when user variants coexist before crystallization."

So K1615 (max_cos < 0.15) is phase-dependent: PASS post-crystallization,
FAIL during coexistence (0.9580 ≫ 0.15). K1615 is ambiguous by
construction — unverifiable without pinning phase ⇒ SUPPORTED
impossible.

Defense-in-depth: T1 ∨ T3 ∨ T4 ∨ T5 each alone blocks SUPPORTED; T2
reinforces. No measurement can rescue the claim.

## Cohort context
21st preemptive-kill in audit-2026-04-17/g4-gemma4/composition-bug session
drain. ap-017 instance #20 (composition-bug branch #19) +
ap-framework-incomplete. Reuses preempts (a)-(g) from ap-017 registry.

## Prior art cited
- F#454 T6.5 Dynamic Adapter Registry (SUPPORTED, 2026-04-11)
- F#449 T5.3 Adapter Submission Pipeline (SUPPORTED, 2026-04-11)
