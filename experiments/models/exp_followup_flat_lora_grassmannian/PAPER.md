# Flat-LoRA × TRUE Grassmannian A: Preemptive-KILL digest

## Verdict
**KILLED — preemptive, structural impossibility.** See MATH.md Theorem 1.

## Hypothesis (pre-registered)
With TRUE Grassmannian A construction, Flat-LoRA outperforms random-QR
baseline by ≥3pp on merge quality (K1554).

## Prediction vs. measurement
| Quantity | Predicted (if measured) | Measured | Kill |
|---|---|---|---|
| Δ_merge (Flat-LoRA − baseline) | ≤ 0.15pp | not measured (preempt) | K1554 FAIL |
| Sharpness (baseline, parent) | 0.02% | F#35 verbatim | — |
| Sharpness (Flat-LoRA, parent) | 0.07% | F#35 verbatim | — |
| cos(adapter_i, adapter_j), parent | 0.001 | F#35 verbatim | — |

## Proof outline (three-lemma ∧-kill)
- **L1 (sharpness floor):** Parent sharpness 0.02% / 0.07% already at
  measurement floor; Flat-LoRA has no headroom to lower it. (F#35)
- **L2 (A-init ⊥ sharpness):** Grassmannian A-init reduces A-row cosine
  (F#132) and A-matrix clustering (F#498) — neither is a sharpness mechanism.
  Flat-LoRA's SAM gradient perturbation is uncoupled from A's Grassmannian
  structure.
- **L3 (cos-to-merge mechanism absent):** cos(adapter) is already 0.001, 50×
  below the near-orthogonality threshold (F#35). No proven link (F#38 explicit
  negative) converts further cosine reduction into Δ_merge ≥ 3pp.

∧-combination: L1 ∧ L2 ∧ L3 ⇒ Δ_merge ≪ 3pp. K1554 FAIL.

## Antipatterns flagged
- flat-lora-sharpness-floor-exhausted-by-parent (F#35)
- grassmannian-A-init-uncoupled-from-loss-sharpness (NEW sub-axis, F#132/F#498)
- already-near-orthogonal-adapters-leave-no-merge-headroom (F#38)
- a-init-swap-preserves-merge-mechanism-null

## What this rules out
Any "swap A-init to tighter-orthogonal variant" rescue of a Flat-LoRA-style
merge-improvement result where the parent already showed the adapters are
flat *and* near-orthogonal. The rescue does not attack the parent's failure
mode (Δ_merge at noise); it perturbs an already-saturated orthogonality
margin.

## Assumptions (Ralph autonomy)
- "Random QR" in the followup title is assumed to refer to the parent's
  default Kaiming/random init, which F#481 measured at orthogonality deviation
  1.19e-7 (indistinguishable from true Grassmannian at rank=16, d=2048, N=5).
- Re-running the parent with a true-Grassmannian A would leave the sharpness
  and cosine measurements within the parent's reported noise band.

## Artifacts
- MATH.md: proof and antipatterns
- run_experiment.py: stub (exits 0, no compute)
- results.json: KILLED (preemptive, executed=false, is_smoke=false)
- REVIEW-adversarial.md: adversarial checklist
- LEARNINGS.md: this preempt's contribution

## Cross-refs
- Parent: micro/models/flat_lora_training/ (KILLED, F#35)
- Related kills: exp_np_lora_null_space_composition, exp_grassmannian_expert_init
