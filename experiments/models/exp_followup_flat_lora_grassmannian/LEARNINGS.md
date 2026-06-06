# LEARNINGS — exp_followup_flat_lora_grassmannian

## Status
KILLED preemptively (no code executed).

## Core lesson
When a parent experiment has pinned a null at the *mechanism floor* (here:
sharpness 0.02%/0.07% and cos 0.001 — both deep below their thresholds),
a followup that swaps a *non-mechanism* input (A-init scheme) cannot
change the outcome. Flat-LoRA's payoff comes from reducing loss-landscape
sharpness; Grassmannian A-init targets A-row orthogonality — a different
property space.

## Antipattern coined
`grassmannian-A-init-uncoupled-from-loss-sharpness` — generalizes to any
"swap orthogonality property of A as rescue for a sharpness/landscape-
driven null". The A-init property and the Hessian property are structurally
uncoupled; no swap of one changes the other.

## Findings reused
- F#35 (parent Flat-LoRA KILL)
- F#132 (Grassmannian AP skeleton: A-row cosine reduction)
- F#498 (A-init clustering: standard vs Grassmannian separation)
- F#38 (orthogonality ≠ specialization)
- F#481 (random-QR orthogonality deviation 1.19e-7 ≈ Grassmannian at N≪d/r)

## What would actually rescue Flat-LoRA
Not A-init. One would need either (a) a training regime where sharpness
is NOT already at the floor (larger rank, more data, deeper trainable
subspace) or (b) a merge method whose failure mode is *sharpness-
dependent* (not Task-Arithmetic / TIES / DARE / Direct Sum — those
saturate on flat landscapes regardless of A-orthogonality).

## Closes audit-2026-04-17 follow-up on flat_lora_training
No further audit-rerun justified for this branch; parent's null is
structural on two independent axes (sharpness floor + orthogonality
floor) and the proposed fix addresses neither.
