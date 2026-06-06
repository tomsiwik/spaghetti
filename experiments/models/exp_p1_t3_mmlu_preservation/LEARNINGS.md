# LEARNINGS.md: T3.2 — MMLU Preserved Under N=5 Composition

**Status: KILLED** | Finding #426

## Core Finding

QK-normalization clamps query **magnitude** but NOT direction — scale IS sensitive. At scale=6: 70.7% neutral MMLU; scale=12: 64.0% (−6.7pp); scale=18: 46.7% (−24.0pp). Theorem 1's "scale invariance" corollary is refuted: magnitude invariance ≠ attention invariance.

## Why

As scale increases, the raw query `Q_raw(s) = W_q x + s·BAx` rotates toward the adapter direction `BAx/||BAx||`. QK-norm normalizes magnitude but preserves this rotation, changing which keys are attended to. Directional invariance would require `BAx ‖ W_q x` for all x — impossible for a useful adapter. This aligns with standard LoRA theory (Hu et al. 2021, arXiv:2106.09685): the adapter modifies the weight matrix, not just its magnitude.

## Format Compliance Artifact

Base Gemma 4 E4B gets 4.0% on neutral MMLU (far below random 25%) — format compliance failure without adapter priming. All K1053/K1055 "PASSes" trivially reflect format learning transfer, not knowledge preservation. Future MMLU experiments must verify base MCQ parsing with greedy decoding + output inspection.

## Key Constraint for P1

**Use scale=6 only. Never increase.** Scale=6 is confirmed safe (62–78% on neutral MMLU across all 5 adapters). Scale≥12 degrades MMLU monotonically and steepens with scale.

## Implications for Next Experiment

Adapter scale is fixed at 6. The T3 tier has validated routing is structurally required (T3.1) and scale is bounded (T3.2). Next: claim next P1 experiment from the critical path — likely routing composition or PLE-M2P integration.
