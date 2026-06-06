# PAPER — exp_g4_crystallize_real_users

**Verdict: KILLED (preemptive, 5-theorem stack)**

Cohort: audit-2026-04-17 / composition-bug / g4-gemma4 — 30th
consecutive preemptive-kill this session (23rd composition-bug branch
under ap-017; 12th SUPPORTED-source preempt registered as candidate (r)).

Source finding: F#451 `exp_p1_t6_crystallize_domain` (T6.2 crystallize,
SUPPORTED). Sibling: F#1564 `exp_followup_m2p_crystallize_real_users`
already KILLED on real heterogeneous users at mean_cos=0.9377 < 0.95.

## Prediction vs measurement

| Theorem | Prediction | Runner measurement |
|---------|-----------|--------------------|
| T1 inventory + sibling KILLED | shortfall ≥ 2 same-domain adapters per domain; sibling verdict = KILLED | **fail**: shortfall=2, sibling.verdict=KILLED |
| T2 budget breach | 5 × 5 × 20.92 ≈ 523 min ≫ 120 min micro ceiling | **fail**: 523.0 min computed, exceeds_micro_ceiling=true |
| T3 success_criteria=[] | "Success Criteria: NONE" + "⚠ INCOMPLETE" present in DB | **fail**: db_marker_incomplete=true, sc_count=0 |
| T4 KC under-spec | 1/5 adjudicatable pins on K1630 (ε present, baseline/pooled/enum/rescale absent) | **fail**: matches=1 (ε only) |
| T5 F#451 LITERAL ×5 | 5/5 sub-breaches (A synthetic→real, B cosine proxy, C norm-bound only, D failure-mode admission, E LLN i.i.d. violation) | **fail**: 5/5 broken |

## T5 sub-breach ledger

| Sub-breach | Observed | F#451 LITERAL anchor |
|------------|----------|----------------------|
| (A) synthetic→real users   | **true** | "Synthetic user variants (σ_frac=0.5); real users may show higher variance" |
| (B) cosine proxy not behavioral | **true** | "Quality measured as cosine to canonical, not task accuracy" |
| (C) MMLU norm-bound only | **true** | "MMLU verified via norm bound only (no model inference)" |
| (D) σ/Δ>1 failure-mode admission | **true** | "If real-user variance σ ≫ 0.5×std(B), crystallized adapter diverges …" + F#1564 measured 0.27 ≪ 0.95 |
| (E) LLN i.i.d. premise violated | **true** | "Theorem 1 (LLN): E[‖B_crystal−B*‖²]=σ²/N. Quality degrades only if users are not from the same domain (K-means gate prevents this)" |

(D) is uniquely strong: a LITERAL source caveat *and* an empirically
observed sibling failure (F#1564 mean_cos=0.9377, per-user spread to 0.27).

## Defense-in-depth

T1 ∨ T3 ∨ T5 each alone blocks SUPPORTED. T2 reinforces (~4.4× ceiling
breach). T4 advisory (1/5 pins).

## Operator-blocked dependency

T2.1 cohort retraining (5 users × 5 domains heterogeneous) is gated by
HALT §B (T2.1 reopen). Even if (B) unblocks, F#1564's prior KILLED
result is the load-bearing prediction-vs-measurement entry: this hop is
empirically refuted, not merely arithmetically infeasible.

## ap-017 cohort scope addendum

Running ap-017 instances: 30 (composition-bug 23, scale-safety 2,
tautological-routing 3, projection-scope 2, tautological-duplicate 1).
SUPPORTED-source preempts now (a)–(r). New axis registered:
**proxy-with-empirical-refutation** — source LITERAL caveat predicts
hop-failure regime, sibling experiment empirically observed it,
SUPPORTED-source provides no behavioral hook to inherit.

## Compute

Runner: pure stdlib, <1s wall. ap-027 N/A. No KC edits. DB updated
status=killed, K1630=fail, dir set.
