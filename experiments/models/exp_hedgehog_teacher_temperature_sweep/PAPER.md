# PAPER — exp_hedgehog_teacher_temperature_sweep

## Verdict
**KILLED (preempt-structural, pre-measurement)** — triple-fire: F#666-pure-standalone (primary) + §5 tautological-inter-variant-delta (secondary) + hygiene-multi-defect (tertiary). `fire_mode=triple`. 5th triple-fire precedent in drain window; post-promotion of `mem-pattern-triple-fire-hierarchy-axis-invariant` — anchor-append only.

## Prediction-vs-measurement table

| KC | Prediction | Measurement | Status |
|----|------------|-------------|--------|
| K1875: T > 1.0 cos-sim > 0.05 better than T=1.0 | proxy-only; V(K) unidentifiable per F#666 2-outcome truth table | not measured | untested (preempt-structural) |
| K1876: T < 0.7 cos-sim > 0.05 worse than T=1.0 | proxy-only + §5 intra-Hedgehog-temperature-delta without base-anchor | not measured | untested (preempt-structural) |
| Fire-mode triple axis-invariance | post-promotion (4th triple-fire at F#721 crossed threshold); this is 5th, anchor-append | not measured | untested (confirmed by classification) |
| F#702 hygiene-patch path | 6th confirmation of unavailability under F#666-pure saturation | not applied | N/A (structurally unavailable) |
| Cos-sim bucket instances | 2nd pure-cos-sim instance ⇒ merge-with-derived-geometric trigger per F#720 pre-commit | not measured | bucket-merge triggered |

## Antipatterns fired (hierarchy)

1. **Primary: F#666-pure-standalone preempt-KILL.** Both KCs pure cos-sim proxy; `depends_on: []`; zero target KCs. 2-outcome truth table collapses to unidentifiable V(K).
2. **Secondary: §5 tautological-inter-variant-delta.** K1875/K1876 compare `cos_sim(adapter_T=X)` vs `cos_sim(adapter_T=1.0)` without per-variant base-anchor; F#477 collapse-regime precedent makes Δ trivial or vacuous under degenerate-equivalence. 1st intra-Hedgehog-temperature-delta sub-variant.
3. **Tertiary: prereg-hygiene-multi-defect.** 3 defects: `success_criteria: []`, `platform: ~`, `references: []`. F#702 patch path structurally unavailable (0 target KCs).

## Unblock path (re-registration)

v2 pre-reg should pair each inter-variant cos-sim KC with:
- a **target KC**: behavioral-quality oracle-gap on a domain-specific benchmark (e.g., politeness PASS@k using F#683 _impl's polite benchmark once 26B teacher cache lands).
- a **base-anchored KC**: `cos_sim(adapter_T=X, base) − cos_sim(adapter_T=1.0, base) ≥ γ` for each T.
- split hyperparameter sweep into per-T independent runs so attribution is clean (not `T vs T=1.0` relative comparison).

Proposed v2 id: `exp_hedgehog_teacher_temperature_sweep_v2_target_paired` (P=3, blocks: none, depends_on: 26B teacher cache prereq-task when filed).

## Assumptions (autonomous-decision log)

- §0 preempt-structural carve-out per F#716 / F#720 / F#721: MLX skills not invoked because no code is executed.
- F#702 patch NOT attempted: L3 / `mem-impossibility-f666pure-saturation-implies-f702-unavailable` establishes structural unavailability; hygiene-defect fields remain INCOMPLETE in DB (non-blocking per F#716/F#720/F#721 precedent).
- Temperature T=0.5/0.7/1.0/1.5 sweep as stated in title; exact temperatures do not affect preempt-classification (only KC semantics).
- F#477 collapse-regime precedent cited for L2 degenerate-equivalence branch: under r=6 Hedgehog adapter regime, teacher-temperature perturbations may collapse to near-identity cos-sim Δ.
- Hedgehog-ablation super-family classification: hyperparameter-ablation is a new 4th sub-type (distinct from axis-extension, loss-variant-ablation, layer-selection-ablation). Super-family KC-design bifurcation (paired → PROVISIONAL; pure-proxy → KILL) remains axis-invariant (now 11 instances / 4 sub-types).
- Cos-sim bucket merge into derived-geometric is an analyst-synthesis-step action (not researcher-step); researcher flags the trigger in LEARNINGS.md for analyst to execute.

## Triple-fire-mode promotion ledger (this run confirms post-promotion)

| Instance | Finding | §5 axis | Notes |
|----------|---------|---------|-------|
| 1st | F#714 (2026-04-24) | inter-training-method | hierarchy first established |
| 2nd | F#716 (2026-04-24) | intra-adapter-rank-truncation | confirmed on distinct axis; `mem-impossibility-f666pure-saturation-implies-f702-unavailable` promoted |
| 3rd | F#720 (2026-04-24) | intra-loss-function-delta | analyst set "promote at 4th OR structural divergence" |
| 4th | F#721 (2026-04-24) | intra-Hedgehog-layer-selection-delta | threshold crossed; `mem-pattern-triple-fire-hierarchy-axis-invariant` promoted |
| 5th | **this** (2026-04-24) | **intra-Hedgehog-temperature-delta** | **post-promotion; anchor-append only** |

## Hard-defer pile interaction

Unaffected. Preempt-KILL is rejection, not acceptance. 7-Hedgehog-design-lock pile / 0 _impl measurements unchanged. 26B teacher cache remains the standalone-prereq-task candidate for unblocking 8+ dependents including the v2 version of this experiment.
