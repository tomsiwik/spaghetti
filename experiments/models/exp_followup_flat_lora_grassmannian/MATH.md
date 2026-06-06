# MATH: Flat-LoRA + TRUE Grassmannian A — Preemptive Structural KILL

## Context
Parent `exp_flat_lora_training` KILLED (Finding #35): Flat-LoRA (SAM) vs
standard LoRA merge improvement = **+0.07pp** (43× below the 3pp threshold),
with **sharpness <0.1% for BOTH methods** and **cos(adapter) ≈ 0.001** (40–50×
below the 0.05 threshold). Parent used default LoRA A init (Kaiming-like).

Followup hypothesis: replace "random QR" A with TRUE Grassmannian A →
Flat-LoRA beats the baseline by ≥3pp (K1554).

This preempt proves K1554 is **structurally impossible** on existing findings.

## Theorem 1 (Three-lemma ∧-kill)
For any Flat-LoRA variant that swaps A-init from "random QR"/Kaiming to TRUE
Grassmannian packing, the expected merge improvement Δ_merge satisfies

  Δ_merge ≤ max(0.07pp, f(cos_A_reduction)) ≪ 3pp = K1554_threshold,

because (L1) Flat-LoRA's only mechanism is sharpness reduction, and sharpness
is already at the measurement floor; (L2) Grassmannian A does not reduce loss
sharpness — it only reduces inter-adapter A-row cosine, which is already
negligible; (L3) no proven chain converts A-cosine reduction below 0.001 into
merge-quality gain beyond noise.

### Lemma L1 — Sharpness floor (F#35)
Parent experiment measured, on 5 BitNet-2B-4T adapters:
- Standard LoRA sharpness: **0.02%** (mean)
- SAM-LoRA sharpness: **0.07%** (mean)

The *raison d'être* of Flat-LoRA (arXiv:2409.14396) is to push adapters into
flat-loss regions. Both baselines are already at flat-region noise floor
(<0.1%). There is no sharpness headroom for any A-init trick to close.

### Lemma L2 — Grassmannian A does not target sharpness (F#132, F#498)
F#132: Grassmannian AP init reduces *inter-adapter A-row cosine* by
1.3–2.0× post-training.
F#498: A-matrices cluster by init method (standard cos≈0.82 vs
Grassmannian cos≈0) — but this is **A-matrix clustering**, not loss-landscape
sharpness. Grassmannian packing is an orthogonality property of the skeleton,
not of the loss Hessian. The Flat-LoRA mechanism (SAM's gradient perturbation)
is uncoupled from A's Grassmannian structure.

### Lemma L3 — Cosine-to-merge mechanism absent at 0.001 floor
Parent measured cos(adapter_i, adapter_j) = 0.001 with random-Kaiming A
(std) and 0.0013 with SAM. Grassmannian A would push A-row cosine further
toward 0, but cos is already 40–50× below the 0.05 orthogonality threshold
used as the "near-orthogonal" regime in F#35. F#38 (killed): "Orthogonality
doesn't force specialization — 10 Grassmannian-A adapters on identical data
are functionally identical." No theorem or prior finding converts cos<0.001
to Δ_merge >3pp; the closest proven link (F#329: SVD-loss ≈ 26pp under
composition) concerns SVD-extraction regime, not A-init.

### Composition (∧-kill)
L1 ∧ L2 ∧ L3 ⇒ Δ_merge upper-bounded by max(parent +0.07pp, ε(L2,L3)),
where ε is bounded above by noise-scale because neither L2 nor L3
provides a mechanism exceeding the parent's measurement precision.

Therefore Δ_merge ≪ 3pp and K1554 cannot pass. QED.

## Kill Criteria (pre-registered; UNCHANGED)
- **K1554 [id 1554]** (FAIL, preempt): With TRUE Grassmannian A and 5/5
  converged adapters, Flat-LoRA merged quality vs random-QR/Kaiming baseline
  Δ ≥ 3pp. Structurally impossible by Theorem 1.

## Predictions (no measurement)
- Not measured; preempt analytic. A hypothetical measurement would produce
  Δ_merge ∈ [−2.01pp, +0.15pp] (parent range, marginally widened by
  Grassmannian's +ε A-cosine tightening).

## Antipattern flags
- `flat-lora-sharpness-floor-exhausted-by-parent` (F#35 family)
- `grassmannian-A-init-uncoupled-from-loss-sharpness` (novel sub-axis under
  F#132/F#498)
- `already-near-orthogonal-adapters-leave-no-merge-headroom` (F#38 family)
- `a-init-swap-preserves-merge-mechanism-null`

## Findings reused
- F#35 (parent KILL, sharpness 0.02%/0.07%, cos 0.001, Δ +0.07pp)
- F#132 (Grassmannian AP reduces A-row cosine 1.3–2×, POST-training)
- F#498 (A-matrix clustering by init: standard 0.82 vs Grassmannian 0)
- F#38 (Grassmannian orthogonality does not force specialization)
- F#481 (random QR orthogonality deviation = 1.19e-7, indistinguishable from
  true Grassmannian at rank≪d/N)

## Conclusion
KILL preemptively. No code executed — stub `run_experiment.py` exits 0.
