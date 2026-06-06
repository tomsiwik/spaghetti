# LEARNINGS — exp_g4_adapter_magnitude_distribution

## Core Finding
**Per-weight magnitude is a real behavioral importance signal on trained Gemma 4 LoRA q_proj adapters (r=6, scale=6.0).** Zeroing top-20% magnitude entries vs. random-20% yields ΔPPL ratio R=0.82 mean (math +5.68 vs −0.48; code +5.26 vs −1.38). K1 (non-Gaussian structure) SUPPORTED on both proxy (Shapiro fail frac 39.7% < 80% normality) and target (R=0.82). K2 (magnitude-as-importance) target-SUPPORTED; proxy K1918 INCONCLUSIVE — `QuantizedMatmul::vjp` undefined on 4-bit base (MLX 0.31 framework limit).

## Why
Trained Kaiming+Adam LoRAs concentrate signal in a minority of per-matrix entries (P1 fails, P3 sparsity ~0.8%), and that concentration is behaviorally load-bearing where the adapter is net-useful. This is a *different* axis from F#500 (global null-space magnitude routing, killed), F#526 (pre-merge direction), F#350 (M2P-scale CV) — those were about global scale / direction; this is per-weight structure within a trained adapter. Medical is anomalous: adapter net-harmful on held-out MCQs, so magnitude signal collapses to R=0.13 — a per-domain usefulness gate is mandatory before any compression deployment.

## Implications for Next Experiment
1. **Unblock K1918 forward-only.** Run the filed `_impl` follow-up: Wanda-style `I(w) = |w|·E[|a|]` correlation with behavioral ablation — bypasses the quantized VJP limit. Target ≥0.3 |r|.
2. **Per-domain usefulness gate is a prerequisite for magnitude-based compression.** Before pruning, verify the adapter improves held-out metrics; medical-style net-harmful adapters are the wrong compression target regardless of weight structure.
3. **Transfer to F#627 target.** Current evidence is q_proj; v_proj+o_proj adapters (PLAN.md-sanctioned) should confirm the per-weight structure finding before it feeds product compression work.
4. **Do not treat magnitude routing as resurrected.** This is per-weight *within* a trained adapter — not cross-adapter routing by magnitude (F#500 null-space remains killed).

## Process note (3rd observation)
CLI-status-forces-killed-on-provisional fired for the 3rd time (after F#673, F#742). Per the memory's escalation clause, promote guard to `reviewer.md §5` and anchor this instance. Upstream fix is a `--status provisional` CLI option.
