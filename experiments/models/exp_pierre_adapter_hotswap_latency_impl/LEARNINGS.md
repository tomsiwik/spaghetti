# LEARNINGS — exp_pierre_adapter_hotswap_latency_impl

## Core Finding
Pierre adapter hot-swap on Gemma 4 E4B 4-bit is **~1 ms median** (K1953: 0.97 ms,
103× below the 100 ms product threshold). Same-adapter detach/re-attach is
**bitwise-exact** across 16 decoded tokens at swap positions k ∈ {1,2,4,8}
(K1954: 0/64 mismatches). Parent F#702 Theorems 1+2 transfer Qwen3-0.6B → Gemma 4 E4B.
First non-preempt-KILL drain-window outcome in ~34 iterations.

## Why
1. **Theorem 1 (latency)**: Per-layer Python-overhead regime; rescaling parent's
   `n_layers=34` mid-point by 42/34 = 1.24 yields [0.5, 1.12] ms — measured 0.97 ms
   lands inside. T=2 (F#627 v_proj+o_proj) ⇒ 84 module-updates + one
   `mx.eval(model.parameters())` is the dominant cost.
2. **Theorem 2 (bitwise reversibility)**: MLX lazy eval rebuilds the computation
   graph per `__call__`; module-identity change without parameter-content change
   is invisible to the next forward pass. Confirmed across KV-cached decode.
3. **Per-layer dim heterogeneity (new observation)**: Gemma 4 E4B has two attn-head
   groups — 35 narrow layers (`v_proj 2560→512`, `o_proj 2048→2560`) and 7 wide
   layers {5,11,17,23,29,35,41} (`v_proj 2560→1024`, `o_proj 4096→2560`).
   Single-layer dim probes will mis-shape 7 of 42 layers. Inferring dims per
   layer is mandatory.

## Implications for Next Experiment
- **Target-anchored P=2 strategy validated.** Both KCs were target metrics
  (wall-clock latency, token-identity). On-disk KC-target verification before
  claim worked. Continue with: `init_comparison_v2`, `jepa_scale_sweep`,
  `cross_axis_interference`, `triple_composition_3domain`, `g4_zs_base_transfer`.
- **Gemma 4 adapter-synthesis caveat (Finding #766)**: any future Gemma 4 adapter
  code MUST (1) use `model.layers` not `model.model.layers`, (2) infer per-layer
  `(in_features, out_features)` before synthesizing B-matrices, (3) ship local
  `attach_adapter`/`detach_adapters` rather than reusing `pierre.pierre`
  (Qwen3-bound). Pierre's runtime path is now unblocked on Gemma 4 E4B.
- **Hygiene defect persists**: 11th F#502/F#646 cohort hit (empty
  `success_criteria` + `references`). Non-blocking here (target-paired KCs
  PASSED), but populate both fields at claim-time on next experiment to retire
  the cohort.
- **Avoid**: any PROD child of KILLED parent (top-level guardrail per F#765);
  3rd ap-017(s); 8th Hedgehog; 14th g4-ablation; 6th MEMENTO.
