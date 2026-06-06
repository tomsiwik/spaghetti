# MATH.md — exp_g4_polar_scale_invariance (preemptive kill)

## Theorem (cohort preempt, ap-017 branch: scale-safety)

Let `S_source` = scope of Finding #444 (PoLAR 3× stability on Qwen-proxy,
q_proj only, near-chance accuracy, metric-level variance measurement).
Let `S_target` = scope of KC#1621/#1622 (Gemma 4 base with QK-norm,
scale sweep {3,6,12,24}, behavioral transfer claim).

**Claim.** If `S_target ⊄ S_source` via any caveat-axis breach, the
source guarantee is void and SUPPORTED is structurally inaccessible.

**Proof (Defense-in-depth, 5 theorems).**

T1 (adapter shortfall). Adapters required = |scales| × |types| = 4 × 2 = 8
(PoLAR+LoRA at each scale). Available = 0 PoLAR + 0 LoRA in repo.
Shortfall = 8 > 0. ∴ KC#1621/#1622 cannot be measured. ∎

T2 (wall-time). 8 × 20 min = 160 min > 120 min micro ceiling. ∎

T3 (success criteria). `experiment get` returns literal
"Success Criteria: NONE" + "⚠ INCOMPLETE" flag. Verdict→SUPPORTED
requires non-empty success_criteria by framework rule. ∎

T4 (KC methodology pin gap). KC text lacks 2/5 pins
(ε-threshold regex miss, pooled-comparison absent). Pre-reg
methodology under-specified. ∎

T5 (F#444 scope-transfer block). Caveat literal (finding-get 444):
"QK-norm in Gemma 4 provides baseline scale protection for q/k
adapters regardless of PoLAR". Source advantage (3× variance
reduction) is scope-specific to QK-norm-absent architectures
(Qwen proxy). Gemma 4 target has QK-norm baseline protection →
PoLAR's Stiefel stabilizer is redundant with architecture; measured
variance differential is not attributable to PoLAR. Additional caveat:
"Behavioral advantage cannot be confirmed without task learning"
+ "near chance accuracy (4–12%)" — F#444 is metric-level only,
cannot ground a behavioral transfer claim. Both breaches active. ∎

**Corollary.** T1 ∨ T3 ∨ T5 each alone blocks SUPPORTED.
Verdict = KILLED_PREEMPTIVE. ∎ QED

## Predicted measurements (runner pure-stdlib)

| Theorem | Predicted | Actual |
|---|---|---|
| T1 shortfall | ≥ 8 | 8 |
| T2 total_min | > 120 | 160 |
| T3 incomplete | true | true |
| T4 pin count | < 5 | 3 |
| T5 F#444 hits | ≥ 2 | 3/3 |
