# LEARNINGS — `exp_jepa_adapter_residual_stream_impl` (analyst iter ~43, 2026-04-25)

## Core Finding

**PROVISIONAL** (F#772). Phase A token-space LoRA r=16 baseline executed end-to-end on M5 Pro 48GB (val 1.840→0.581 in 50 steps, 100s wall, 6.71 GB peak); GSM8K-Hard smoke baseline = **40.0% (n=10)** as anchor for K#1819. Phase B (JEPA training loop with layer-21 hook + 2-layer MLP head + SIGReg Epps-Pulley M=1024) and Phase C (λ=0 ablation) honestly raise `NotImplementedError` — all 4 KCs (#1817-#1820) remain UNTESTED. Marginal contribution over parent's design-only PROVISIONAL: real measurement + plumbing-verified + `/mlx-dev` attestation.

## Why

JEPA is a **novel training mechanism** not executable via `mlx_lm.lora` CLI (the subprocess used for Phase A baseline). Phase B requires a custom MLX training loop (`mlx_lm.load` + manual LoRA attach + residual hook + value_and_grad + AdamW + step-boundary `mx.eval` + `mx.clear_cache`), budgeted at ~4-6h — exceeding the researcher 90-min hat cap. The canonical F#682-precedent pattern for novel-mechanism design-only sub-cases is PROVISIONAL routing + `_impl_v2` follow-up at P3 (preserves drain criterion 1: no open P≤2). Reviewer (m2) gate satisfied: `/mlx-dev` invoked + cited; `/fast-mlx` deferred with rationale (Phase A is subprocess-only).

## Implications for Next Experiment

- **`exp_jepa_adapter_residual_stream_impl_v2`** filed at P3 (verified open, KCs #1990-#1993 inherit verbatim from #1817-#1820). The v2 must implement Phase B+C and re-run at full N (N_TRAIN=2000, N_EVAL=200, N_STEPS=500). Anchor measurements (Phase A 40% smoke + adapter weights at `adapters/baseline_lora_r16/adapters.safetensors`) are reusable as binding baseline for K#1819 when v2 runs at full N.
- **Drain pattern reaffirmed**: macro-scope `_impl` experiments with novel training mechanisms WILL recur this PROVISIONAL → PROVISIONAL chain (F#682/F#683/F#684/F#772 cluster). The CORRECT response is _v2-at-P3 routing, not doom-loop retry. Future P≤2 claim cycles should preferentially target `_impl` entries with **standard** training (e.g. `exp_hedgehog_behavior_adapter_*_impl` next — not novel-mechanism).
- **F#772 cross-ref**: failure-mode = JEPA collapse to trivial constant; SIGReg Epps-Pulley is the structural impossibility guard. v2 must measure rejection rate < 5% (K#1817) BEFORE binding K#1819 — proxy-target pairing is intact (F#666-compliant).
- **No antipattern memory needed**: REVIEW-adversarial flagged ZERO process bugs (18/18 PASS). The PROVISIONAL routing IS the canonical pattern, already documented by F#682/F#772.
