# REVIEW-adversarial — exp_mi_expert_independence

## Verdict
**KILL** (ratify). Preemptive-kill derivation is sound; parent-finding cascade (F#22/F#544) independently suffices. 44th preempt in audit-2026-04-17 cohort.

## Adversarial checklist

**Consistency:**
- (a) results.json `verdict: "KILLED"` ↔ DB `--status killed` ↔ PAPER "Verdict: KILLED" ✓
- (b) `all_pass: false` ↔ K418/K419 both FAIL ↔ status killed ✓
- (c) No "PROVISIONAL/INCONCLUSIVE/DEGENERATE" misleading the kill ✓
- (d) `is_smoke: false`, `run_type: "derivation-only"` — preempt pattern, not smoke ✓

**KC integrity:**
- (e) K#418/K#419 mapped in MATH.md §Hypothesis match DB pre-reg (r² improvement ≥ 0.1; cost ≤ 100×). No relaxation — kill criteria used as-pre-registered.
- (f) Non-tautological: r²(MI,Q) vs r²(cos,Q) compares two different metrics against a third channel Q (behavioral). Cost K#419 is a FLOP counting argument. Neither is identity-type.
- (g) K-IDs in results.json ↔ MATH.md ↔ DB aligned.

**Code↔math:**
- (h)-(m2) Derivation-only stub. run_experiment.py writes rationale dict to results.json; no LoRA/routing/model load. Platform-skill invocation N/A (preempt pattern per PLAN.md §2).

**Eval:** N/A for preempt.

**Deliverables:**
- (r) PAPER has prediction-vs-measurement table (4 rows) ✓
- (s) Math sanity:
  - DPI framing (MATH §Proof K#418): "r²(M,Q) ≤ r²(PPL,Q) = 0.0064 because M sees Q only through the distributional channel" — this is *not* strict information-theoretic DPI (PPL is a specific scalar, not a Markov bottleneck over all distributional functionals). The load-bearing argument is empirical: F#286 shows distributional proxies saturate at r²≈0.0064 across this codebase, and F#544 confirmed at N=5 macro (ρ=−0.7) that KL specifically anti-correlates. MI inherits via MI(X;Y) = KL(p(X,Y)‖p(X)p(Y)) — **this identity is the rigorous core**.
  - Cost bound K#419: KSG cost = O(d·n·log n) × digamma/kNN constants ~50. At d=2304, n=75 (min floor from F#590/F#591 metric-swap), ratio ≥ 2304·log(75)·50 ≈ 4.96e5. Log base ambiguous (ln or log₂); even log₁₀(75)≈1.87 still gives 2.15e5 ≫ 100. Bound robust.

## Assumptions
- Preempt rule accepts "empirical DPI" framing (F#286 saturation) as equivalent to strict DPI for this codebase's behavioral evaluation regime. Limitation already flagged in PAPER §Limitations (if r(PPL,Q) → 0.6, MI might pass K#418).
- MI = KL identity is the load-bearing mechanism; if F#22 were revisited, this kill would need re-examination.

## Reusable rule (for analyst LEARNINGS)
**Preempt-axis**: composition-bug / parent-finding-contradicts-assumption, sub-variant *distributional-metric-on-proxy-channel*. Any future experiment proposing a distributional/output-similarity metric (cosine, KL, JSD, Wasserstein, MI, etc.) to predict behavioral composition quality is preempt-killable via the F#286 saturation bound r²(M,Q) ≤ 0.0064 — unless the authors redefine Q away from current PPL-linked proxies.

## Finding registered
F#663 — MI is same-family as KL (F#22/F#544), inherits kill on behavioral composition prediction.
