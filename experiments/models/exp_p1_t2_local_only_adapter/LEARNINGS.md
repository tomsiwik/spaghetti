# LEARNINGS.md — T2.3: Local-Only vs All-Layer Adapter

**Status:** KILLED (K1037 = 70.0% < 73.8% threshold)
**Finding:** #445

## Core Finding
Local layers (35/42) provide 85.4% of adaptation quality, but cannot fully substitute
all-layer adaptation for multi-step math reasoning (GSM8K) because 256-token sliding
windows block cross-context integration required for >256-token reasoning chains.

## Why
Gemma 4's local attention layers have a hard geometric limit: the 256-token sliding
window cannot see across reasoning steps that span the full problem context. Global
layers (7/42) bridge these long-range dependencies — they contribute the final 14.6%
lift (70% → 82%) that K1037 needed.

## Architectural Discovery
Gemma 4 E4B has q_proj dimension asymmetry: local layers (2048, 320) vs global layers
(4096, 320). Global layers carry 44% more params per layer — and are more expensive
to adapt. Actual local/all-layer param ratio = 0.776, not 0.833 as uniform-dim math predicted.

## Implications for Architecture
- **Global layers = shared adapters.** With only 7 global layers and large q_proj,
  these are candidates for domain-agnostic (routing-free) shared adapters.
- **Local layers = domain routing targets.** 35 local layers hold domain specificity.
  This maps cleanly onto the Pierre routing architecture.
- **K1037 recovers on short-context tasks.** Classification, single-sentence generation,
  and other <256-token tasks may reach ≥90% local-only sufficiency.

## Impossibility Structure
Pure local-only adaptation cannot match all-layer for tasks where the critical
reasoning path spans > 256 tokens. This is a hard geometric constraint from the
sliding-window architecture — not a hyperparameter to tune around.
