# PAPER.md — exp_g4_flywheel_real_users

## Verdict: KILLED_PREEMPTIVE (5-theorem, defense-in-depth)

K1626 (`epsilon_cumul < 10%`) and K1627 (`quality_cos > 0.9999`) port
F#452/F#453's 3-promotion flywheel claim from synthetic W_base + synthetic
users + q_proj only to real Gemma 4 + heterogeneous real users across
the full projection-set. Five independent scope axes separate source
from target; source does not authorize the target regime.

## Prediction vs Measurement

| Theorem | Predicted (blocks SUPPORTED?) | Measured | Verdict |
|---|---|---|---|
| T1 infrastructure | shortfall ≥ 4 artefacts | 4/4 required artefacts absent; 0 `*flywheel_real*`, 0 `*sequential_base*`, 0 `*promotion_cascade*`, 0 user-adapter dirs on real Gemma 4 | BLOCK |
| T2 budget | compute ≫ 30 min iter ceiling | 251.04 min vs 30 min iter / 120 min micro ceiling; sequential-retrain protocol un-registered | BLOCK |
| T3 framework | `Success Criteria: NONE` + ⚠ INCOMPLETE | `experiment get` confirms both LITERAL | BLOCK |
| T4 KC pins | ≤2/5 pins on both KCs | 1/5 max across K1626, K1627 (only "pooled" hits via cumul/cos) | BLOCK |
| T5 scope caveat | ≥1 F#452/F#453 breach | 5/5 breaches (A synth→real base, B synth→het-real users, C core invariant untested, D q_proj→full-model, E N-scale extrapolation) | BLOCK |

`all_block = True` (5/5). Defense-in-depth: T1 ∨ T3 ∨ T5 alone blocks.

## §T1 — Infrastructure

K1626/K1627 require four artefacts on real Gemma 4, none present:
  (1) Heterogeneous real-user adapters (not synthetic 5-canonical-epsilon,
      not domain adapters). `exp_p1_t2_single_domain_training/adapters/`
      holds `{code, math, medical}` DOMAIN adapters only. Sibling preempt
      F#454 (exp_g4_real_user_registry) confirmed 0 user-style adapters.
  (2) Sequential-base promotion pipeline (retrain adapter k+1 against W_k,
      not W_0). 0 repo matches for `*sequential_base*`, `*promotion_cascade*`.
  (3) Cumulative-ε measurement on real W_k spectral gaps. Synthetic-base
      sim in `exp_p1_t6_flywheel_simulation/` is the only flywheel dir;
      it has no `results.json` (proxy-only F#453 source).
  (4) Per-domain quality_cos between ΔW_{W_0} and ΔW_{W_k}. No artefact.

Shortfall = 4. K1626/K1627 unmeasurable without these.

## §T2 — Budget

Sequential-base 3-promotion flywheel, 2 heterogeneous users minimum,
2 passes (W_0 reference + sequential) at F#454-measured 20.92 min/user:
  need = 2 × 3 × 2 × 20.92 = 251.04 min ≫ 30 min iter, ≫ 120 min micro.
Additionally the sequential-retrain protocol is un-pre-registered, so
compute spent on a non-discriminating run is wasted by construction.

## §T3 — Framework incomplete (LITERAL)

```
Success Criteria: NONE — add with: experiment success-add exp_g4_flywheel_real_users ...
⚠ INCOMPLETE: missing success_criteria
```
`ap-framework-incomplete` blocks SUPPORTED regardless of measurement.

## §T4 — KC pin count (1/5 max)

K1626 = `epsilon_cumul < 10%`:
- baseline ✗ (no comparator in KC text; F#452 4.78%, F#453 7.62% only
  referenced via source, not K1626)
- delta ✗ (threshold not delta-to-baseline)
- pooled ✓ ("cumul" aggregates)
- ε (methodology) ✗ (no p<, CI, ±, significance, seed spread)
- enum ✗ (no N users, seeds, layers, domains enumerated in KC)

K1627 = `quality_cos > 0.9999`: same pattern; 1/5 (only "cos" ~ pooled).

1/5 matches ap-017 (c) F#44 raw-threshold-without-methodology-ε.

## §T5 — Scope caveat literal (F#452 + F#453 sources)

### §T5.A — Synthetic-to-real W_base

F#452 caveat LITERAL: *"Synthetic W_base (std=0.05); no real MMLU test
(proxy only)."* Davis–Kahan gives sin(θ_k) ≤ ‖ΔW_k‖₂ / δ_gap,k; bound
depends on *smallest* δ_gap across the stack. Real Gemma 4 e4b weights
have per-layer δ_gap distribution that is not Gaussian-std=0.05. F#452's
4.78% single-promotion number cannot be transported. Source itself
notes "real weights give lower ε" — a directional conjecture, not a
guarantee for K1626's 10% threshold.

### §T5.B — Synthetic users → heterogeneous real users

F#453 simulated crystallization of 5-canonical-epsilon synthetic user
adapters. Heterogeneous real users have variable LR / steps / seeds and
non-uniform mu_bar_norm. `exp_followup_m2p_crystallize_real_users`
measured per-user cos ∈ [0.27, 0.95] — i.e. the i.i.d. assumption that
underwrites F#453's 2.18√N Pythagorean scaling fails on real users.
F#454 caveat compounds this: *"K1136 throughout tested on final state
only; intermediate max_cos=0.9580 when user variants coexist."*

### §T5.C — Core flywheel invariant untested

F#453 caveat LITERAL: *"Adapters trained on original W_0, not sequential
base."* The flywheel's *definition* is sequential-base retraining:
  W_k = W_{k-1} + s·ΔW_crystal,k, and ΔW_{k+1} trained vs W_k.
F#453 validated the arithmetic expression W_0+ΣΔW on a frozen W_0, not
the process where each ΔW_{k+1} is trained against an updated base. The
target K1626/K1627 invokes the full process on real Gemma 4. Source
does not authorize this regime; the *entire* flywheel claim for a real
deployment rests on an untested core invariant.

### §T5.D — q_proj only → full-model projection scope

F#453 caveat LITERAL: *"q_proj only."* MLX `lora_layers` default on
Gemma 4 covers {q, k, v, o}_proj. 2.18√N Pythagorean scaling per
projection holds only if cross-projection ΔW are orthogonal — not
measured in F#453. Real Gemma 4 flywheel runs all four; the source
bound is silent on three of them.

### §T5.E — N-scale extrapolation

F#453 caveat LITERAL: *"N=5 extrapolation gives ε≈10.2% — borderline."*
The safe-N zone is a function of ε_single, itself un-measured on real
Gemma 4. If real ε_single > 3.3%, then 2.18·√3·ε ≥ 10% and K1626 fails
by construction. Headroom is a function of an un-measured quantity.

## Conclusion

K1626/K1627 are structurally un-runnable (T1), budget-prohibitive and
protocol-unspecified (T2), framework-incomplete (T3), non-discriminating
at ε-level (T4), and scope-non-transferable from F#452/F#453 along five
independent axes (T5A–E). Any attempted run either (a) tautologically
passes by relaxing back to synthetic base / non-sequential / synthetic-
user proxy (re-deriving F#453 without the g4/real-user claim) or
(b) consumes >4h sequential training against a bound extrapolated from
untested i.i.d. Both outcomes uninformative. Cohort-drain preemptive-
kill.

## Assumptions

- Per-user training cost ≈ 20.92 min from F#454 source preempt
  (exp_g4_real_user_registry); dominated by Gemma 4 e4b forward/backward
  passes, projection-set independent to within ±20%.
- MLX Gemma 4 default `lora_layers` covers {q, k, v, o}_proj per repo
  `lora_config_*.yaml` pattern.
- F#453's 2.18√N Pythagorean scaling constant inherits from the source
  finding; a tighter constant would only strengthen the T5.E argument.

## References

- F#452 exp_p1_t6_base_promotion (SUPPORTED, single-promotion ε=4.78%
  on synthetic W_base std=0.05, q_proj, A-matrices not averaged).
- F#453 exp_p1_t6_flywheel_simulation (SUPPORTED, 3-promotion ε_cumul=
  7.62% on synthetic W_base with synthetic users, q_proj, sequential
  retraining admitted untested).
- F#454 exp_p1_t6_dynamic_adapter_registry (SUPPORTED source), via
  sibling preempt exp_g4_real_user_registry (KILLED_PREEMPTIVE, 0
  user-style adapters on Gemma 4 confirmed).
- `exp_followup_m2p_crystallize_real_users` (KILLED, per-user cos
  heterogeneity measurement on real users).
- ap-017 composition-bug cohort scope (audit-2026-04-17).
- ap-framework-incomplete (mem-antipattern-framework-incomplete).
