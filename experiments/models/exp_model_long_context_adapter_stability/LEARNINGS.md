# LEARNINGS — `exp_model_long_context_adapter_stability`

**Verdict.** PROVISIONAL (F#769 — BLOCKED-on-resource compute-budget sub-form, F#768 family).

## Core Finding

Macro-scope BLOCKED-on-resource PROVISIONAL super-family closes at 2 sub-forms:
F#768 (model-cache absent, 14GB+2.5h) and F#769 (compute-budget cap, 4–8h vs 30-min
iter). Both share: macro scope + partial proof-first coverage + refusal to silently
proxy on the load-bearing axis.

## Why

- §3.2 V/O structural protection (`softmax(QK^T/√d)` invariant to V/O perturbations,
  error bounded by `‖Σ_i ΔW_v‖`) predicts K1706 PASS at 8k/32k — confirmatory, not
  novel. §3.3 LongLoRA range-extrapolation makes 128k load-bearing (V/O perturbations
  compound with positional drift past base training context).
- Antipattern (m) on context-length axis: measuring 8k/32k and extrapolating 128k =
  silent proxy substitution. Scaffold refuses; PROVISIONAL is the correct artifact.
- Doom-loop broken: prior 2026-04-24 RELEASE-to-P3 → this iter PROVISIONAL escalation.

## Implications for Next Experiment

1. **Super-family closed at 2.** Further macro BLOCKED-on-resource filings reuse
   F#768/F#769 — do not register new finding numbers (ledger-explosion antipattern).
2. **Range-decomposition cheaper.** Skip 8k sibling (proof-predicted PASS, not novel).
   32k + 128k siblings each fit a dedicated session, independently informative.
3. **F#263 long-context precursor unfilled.** N=1 long-context baseline is a cheaper
   precursor before N=5 composition.
4. **Reclaim discipline.** Scaffold's reclaim_path (≥4h session → P=2 → /mlx-dev +
   /fast-mlx → NIAH then RULER) is for a dedicated session, not drain iterations.
