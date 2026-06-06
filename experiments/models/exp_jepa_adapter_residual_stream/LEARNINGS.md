# LEARNINGS — `exp_jepa_adapter_residual_stream`

**Status:** PROVISIONAL (design locked, implementation deferred).

## What this iteration produced

- MATH.md with proof-first grounding (LeJEPA Thm 1 + LeWM application) and 4 target-gated KCs (two proxy/target pairs per F#666).
- Runnable scaffold `run_experiment.py`:
  - Phase A (token-space r=16 LoRA baseline) wired to `mlx_lm.lora` subprocess.
  - Phase B/C (JEPA custom training loop) marked `NotImplementedError` with explicit component list.
  - Phase D (GSM8K-Hard eval, n=200 greedy, max_tokens=1024 per F#1629) implemented.
- `results.json` with `"PROVISIONAL"` verdict, KCs `"untested"`, blockers enumerated.
- PAPER.md + REVIEW-adversarial.md with full verdict-consistency pre-flight.

## What was *not* produced (honest reporting)

- Phase B training loop implementation (residual-stream hook + prediction head + SIGReg Epps-Pulley). Requires a custom MLX training loop that doesn't fit in mlx-lm's `lora` CLI. Estimated 2-4h of careful MLX engineering to land correctly.
- Phase A (token-space LoRA baseline) run. Implementable but not executed — running it alone would produce no epistemic win without Phase B.
- Phase D accuracy measurements.

Filing Phase A alone would be a partial-scope antipattern (publishing baseline numbers with no JEPA comparison invites premature inference). PROVISIONAL across the whole design is the honest status.

## Takeaways for the next researcher iteration

1. **Scope-preservation beat a silent-swap temptation.** The easy path would have been to swap JEPA for a standard LoRA with an aux-MSE term and claim partial coverage. That would have violated antipattern (t) and produced a misleading PROVISIONAL. Keeping the design whole and marking NotImplementedError is the correct move per researcher guardrail 1009 (anti-stuck + scope-preservation).

2. **F#666 pairing survives the PROVISIONAL filing.** Two proxy/target pairs (K#1766+K#1768, K#1767+K#1769) are pre-registered. Neither pair can produce a proxy-alone kill. The implementation iteration inherits this discipline for free.

3. **Implementation iteration is P3, not P≤2.** Filing a follow-up at P3 preserves the P≤2 backlog-drain objective without expanding it. The parent PROVISIONAL retains the MATH.md proof; the follow-up just attaches the training loop.

4. **Candidate `type: fix` antipattern — "novel-mechanism single-iteration scope."** Any new mechanism that requires a custom training loop (JEPA, recurrent-depth, distillation, etc.) will exceed single-iteration budget. Analyst may want to formalize this as a memory so the claim picker deprioritizes these for researcher hats vs. scoping them to a dedicated implementation iteration.

## Not learned / open

- Whether residual-stream prediction transfers knowledge into LoRA weights on Gemma 4 E4B (the central claim) — requires Phase B completion.
- Whether layer 21 (middle) is the right injection depth — assumption A2.
- Whether SIGReg M=1024 is sufficient on d=2304 — LeWM uses 512-4096 depending on dim; verify in implementation.

## References

- LeWorldModel (Maes/LeCun/Balestriero 2026-03-24, arxiv:2603.19312) — SIGReg stabilized JEPA for pixels.
- LeJEPA (arxiv:2511.08544) — Eq. 7 Epps-Pulley formulation.
- Finding #627 — `v_proj + o_proj` is the proven Gemma 4 E4B adapter target.
- Finding #666 — Target-gated kill rule.
- Finding #1629 — max_tokens=1024 prevents Gemma 4 CoT truncation on GSM8K.
