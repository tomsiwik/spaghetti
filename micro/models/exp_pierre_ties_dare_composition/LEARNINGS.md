# LEARNINGS: TIES/DARE Composition

## Core Finding

Composition of PoLAR adapters **works** when applied via proper module replacement (`_FusedDeltaLinear(nn.Module)`). Three prior experiment KILLs (#571, #828, earlier) were **false kills** caused by monkey-patching `__call__` on PoLAR modules, which destroyed forward-pass numerics.

## Results

- **Uniform 1/N averaging**: humaneval=90.0, medqa=60.0 — no collapse
- **DARE (drop+rescale)**: best method, 73.3 avg (+4.4pp over best single adapter)
- **TIES (trim+sign)**: 65.5 avg — worse than uniform, unnecessary complexity
- **DARE-TIES combined**: catastrophic (4.4 avg) — methods interfere destructively

## Why

TIES aggressively zeros parameters by sign conflict, discarding useful signal in ternary adapters where structure matters more than magnitude. DARE's random drop+rescale is gentler and preserves expected value. Combining both over-prunes.

## Implication

For Pierre composition: use uniform 1/N or DARE at most. No need for TIES. The gsm8k -6.7pp regression is intrinsic to fused-delta injection (not merge algorithm) — may need per-task routing to avoid arithmetic degradation.

## Reusable Artifact

`_FusedDeltaLinear(nn.Module)` pattern: wrap base linear + sum of scaled adapter deltas as a proper module, replacing the original in the model tree. Never override `__call__` on existing modules.
