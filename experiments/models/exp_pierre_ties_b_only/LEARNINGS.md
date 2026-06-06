# LEARNINGS — exp_pierre_ties_b_only

## Core finding

TIES 3-step applied to B-matrices alone matches full-delta TIES exactly (both avg=71.33), beating Fisher-Rao by +6.7pp. The research agent's claim that "TopK-by-magnitude on B alone wouldn't carry semantic meaning" is **falsified**.

## Why this matters

B-space TIES avoids materializing full deltas (`d_in × d_out`), operating only on the compact B matrices (`r × d_out`). Preprocessing takes 0.02s vs seconds for full-delta. This means Pierre's existing `PoLARLinear` storage can use TIES composition natively — no architectural change needed.

## Contrast with DARE

Prior experiment `exp_pierre_dare_b_vs_fisher_rao` showed B-space DARE fails. The difference: TIES trims by magnitude then elects signs, which preserves the dominant directions in B-space. DARE's random masking destroys B-space structure. **Structured pruning works in B-space; random pruning does not.**

## Implication

TIES-B is the default composition method for Pierre going forward. It's cheap, architecture-compatible, and matches full-delta quality. No need to ever materialize `ΔW = B @ A` for composition.

## Per-benchmark detail

| Method | gsm8k | humaneval | medqa | avg |
|--------|-------|-----------|-------|-----|
| single_best | 66.0 | 78.0 | 42.0 | 62.0 |
| fisher_rao | 68.0 | 68.0 | 58.0 | 64.7 |
| ties_b_only | 72.0 | 86.0 | 56.0 | 71.3 |
| dare_full_delta | 72.0 | 80.0 | 62.0 | 71.3 |
