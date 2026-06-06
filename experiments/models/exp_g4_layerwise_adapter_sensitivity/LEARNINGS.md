# LEARNINGS.md — exp_g4_layerwise_adapter_sensitivity

## Core Finding

Gemma 4 E4B sensitivity to post-block perturbation is **sharply trifurcated**, not monotone or bimodal.
Both F#666-paired KCs pass with margin: **K1919 CV=3.79** (12.6× the 0.30 floor) and **K1976 top-7 forms 3 contiguous bands** `[1] / [8,9,10] / [20,21,22]` (largest=3, exact-pass).
**L21 ΔPPL = 17,558** under ε=0.10 (480× baseline 36.55) — the strongest single-layer-sensitivity signal in this codebase. Three layers (L1, L8, L21) carry > 70% of total |s_l|. **18 of 42 late layers (L23–L40) are perturbation-redundant** (|ΔPPL| < 3).
Registered as **F#747**.

## Why

The flat tail L23–L40 matches **ShortGPT block-influence** (arxiv:2403.03853, Men et al.) — late transformer layers carry redundant residual content.
The early/mid hubs match **Todd et al. function-vectors** (arxiv:2310.15213) — procedural and semantic content concentrates in early-middle attention layers.
**Novelty**: Todd/ShortGPT typically report *two* regimes (early-mid hub vs late tail). Gemma 4 E4B shows *three* bands — a distinct singleton at L1 separated from the L8–L10 hub by four near-null layers. Candidate mechanism: **PLE-M2P per-layer-input gating** injects new signal at specific depths, and/or alternating sliding-window vs global attention concentrates global-context dependence into narrow bands. `layer_scalar` correlation would distinguish.

## Implications for Next Experiment

1. **Immediate follow-up — `exp_g4_layer_selective_lora_top8`** (already proposed in PAPER.md §Follow-ups): train rank-6 LoRA restricted to {L1, L8, L9, L10, L19, L20, L21, L22} vs F#627 all-layers baseline. Predict equal-or-better behavioral accuracy at ~19% parameter cost. This is the direct actionability test of K1976; if task accuracy holds, the sensitivity-hub placement rule is validated as an adapter-design primitive.
2. **Mechanism disambiguation — `exp_g4_layer_scalar_correlation`**: extract `model.layers[L].layer_scalar` and correlate with |ΔPPL|. Confirms or refutes the PLE-M2P-gating mechanism for trifurcation.
3. **Domain stability — `exp_g4_layerwise_sensitivity_cross_domain`**: repeat scan on math + code splits. If the three bands are stable across domains, the placement rule generalizes; if they shift, adapter placement must be domain-conditioned.
4. **Do not** rely on proxy CV alone (F#666) — K1976 contiguous-band target is the actionable KC and must be retained in any follow-up using this sensitivity scan.
5. **Tool hygiene reminder**: MATH.md §0 should explicitly cite `/mlx-dev` and `/fast-mlx` (guardrail 1012). Code was idiomatic but the citation gap was flagged as a non-blocking review note — enforce in the next MATH.md draft.
