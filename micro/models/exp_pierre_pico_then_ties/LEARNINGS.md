# LEARNINGS — exp_pierre_pico_then_ties

## Core Finding

Pico (B-space SVD calibration) and TIES (full-delta sign-aware trim+elect) fix
interference at orthogonal geometric levels and compose additively: 68.0% avg
vs 63.3% (Pico+FR) and 66.7% (TIES alone). HumanEval saturates at 80%, matching
full-delta DARE — the remaining 3.3pp gap lives in GSM8K and MedQA.

## Why

Pico aligns adapter subspaces before materialization; TIES resolves sign conflicts
in the materialized delta stack. They don't overlap: Pico never sees full deltas,
TIES never touches B-space geometry. Composition is additive because the failure
modes are independent.

## Implication for Next Experiment

The GSM8K regression (−4pp vs Fisher-Rao) reveals TIES keep_frac=0.3 is too
aggressive for math signal. Two paths forward:
1. Domain-aware keep_frac (higher for math adapters) — cheap to test.
2. Per-sample routing that bypasses TIES for math-heavy inputs — aligns with
   the factored-LoRA + per-token routing direction from the speed research arc.

The 3.3pp gap to full-delta DARE is tight enough that a keep_frac sweep alone
might close it without needing heavier machinery.
