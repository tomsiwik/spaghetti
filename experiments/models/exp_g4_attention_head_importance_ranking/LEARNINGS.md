# LEARNINGS.md — exp_g4_attention_head_importance_ranking

## Verdict: PROVISIONAL_TAUTOLOGICAL (K1 FAIL, K2 PASS)

## Cluster context
- 14th consecutive Pierre-serving-adjacent experiment; first NON-preempt
  verdict after the F#669 13-deep preempt cluster (pierre_adapter_cache_prefill
  was the 13th). This is a standalone weight-space analysis, not a parent-gated
  behavioral experiment, so F#669 preempt pattern does not apply.

## Useful contributions

1. **Head-level decomposition works but concentration is weak at r=6.**
   rank-6 LoRA on q_proj spreads adapter mass across every head (0/336
   zero-mass heads). Top-20% carries only ~33% of energy — not the >50%
   needed to justify head-pruning optimization paths (proposed in the
   experiment notes).
2. **Functional specialization signal survives even without structural
   concentration.** Cross-domain pairwise Jaccard of top-20% sets is
   0.349 — distinctly below the 0.60 threshold, distinctly above the
   0.111 chance level. Different tasks engage different heads to a
   measurable degree. This is new evidence (prior F#3 was about inter-LoRA
   orthogonality at the weight level, not at the head level).

## What is NOT supported

- "Prune non-important heads for speed" (experiment notes motivation) is
  NOT viable at r=6 on q_proj adapters. No path to meaningful sparsity
  via head selection alone.
- Prediction P3 (hard zeros in the per-head heatmap from rank-6 bound)
  was refuted: rank-6 B-matrices do distribute across all 8 head slabs
  because the slab dimension (256 or 512) vastly exceeds the rank.

## Consolidation signals

- **3rd within-cluster reuse target (per pending-event note on
  "multi-parent-run 3rd obs").** This experiment is the 14th in the
  broader Pierre-serving fatigue cluster but the 1st concentration-proxy
  observation — not a within-cluster reuse of a concentration theme. Does
  NOT trigger F#669 3rd-reuse consolidation on its own; reviewer should
  track future `exp_g4_head_ablation_ppl` and
  `exp_g4_head_importance_vproj_oproj_F627` (both proposed as priority-3
  follow-ups) as potential 2nd and 3rd concentration-proxy observations.

## Follow-up experiments proposed (see PAPER.md)

1. `exp_g4_head_ablation_ppl` — direct behavioral PPL ablation of top-20%
   vs bottom-20% heads. Closes the remaining target-gap that the present
   experiment only approximated via cross-domain Jaccard. Priority 3.
2. `exp_g4_head_importance_vproj_oproj_F627` — repeat analysis on
   F#627-compliant `v_proj + o_proj` adapters. Priority 3.

## Antipattern audit (self-review)

- [x] KCs are target-gated per F#666 (K1 proxy paired with K2 functional).
- [x] No composition math / LORA_SCALE abuse / shutil.copy / hardcoded
      "pass": True / eval-template truncation.
- [x] Not proxy-model substituted; analysis is on the actual
      Gemma 4 E4B 4bit adapters intended.
- [x] Verdict is NOT silently upgraded (PROVISIONAL, not SUPPORTED).
- [x] KCs were pre-registered in MATH.md and NOT modified after seeing data.
- [x] Platform skill `/mlx-dev` invoked before writing code (weight-space
      only, but still followed).

## Caveats to propagate

- Gemma 4 E4B mixed-layer attention (35 sliding with head_dim=256, 7 full
  at head_dim=512) — any head-analysis code on this architecture must
  handle per-layer head_dim. A future replicator assuming uniform 256
  head_dim would shape-error on layers {5,11,17,23,29,35,41}.
- numpy matmul on this host fires spurious `divide/overflow/underflow/
  invalid` warnings on contiguous finite float64 arrays (BLAS quirk).
  Output values are correct; future runs should not interpret these
  warnings as a computation error.
