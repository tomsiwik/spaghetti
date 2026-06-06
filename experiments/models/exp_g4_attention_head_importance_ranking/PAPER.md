# PAPER.md — Gemma 4 attention head importance for adapter routing

## Verdict
**PROVISIONAL_TAUTOLOGICAL** (K1 FAIL, K2 PASS — see MATH.md §4 decision table).

Structural concentration proxy (K1) is too weak to detect the functional
specialization (K2) that does exist. No clean SUPPORTED or KILLED claim.

## Setup
- Base: `mlx-community/gemma-4-e4b-it-4bit` (42 layers, 8 attention heads,
  head_dim=256 sliding / 512 full-attention; 7 full-attention layers at
  {5,11,17,23,29,35,41}).
- Adapters: `exp_p1_t2_single_domain_training` — rank-6 LoRA on
  `self_attn.q_proj`, scale=6, 1000 training iters each, 3 domains
  {code, math, medical}.
- Analysis: weight-space only, no forward passes. Per-head Frobenius mass
  `μ_{l,h} = ‖ΔW_q^{(l)}[:, h, :]‖_F²` computed after per-layer reshape
  `(HIDDEN, num_heads, head_dim_for(layer))`.

## Predictions vs. measurements

| Prediction | MATH.md §5 | Measured | Hit? |
|---|---|---|---|
| P1: per-domain C_20 ≥ 0.55 | threshold implied by K1 | code=0.348, math=0.338, medical=0.318 | **NO** (0.335 mean) |
| P2: pairwise Jaccard small (<0.60 admits substantial overlap) | K2 threshold | code∩math=0.327, code∩medical=0.340, math∩medical=0.381; mean J̄=0.349 | **YES** — domains engage distinguishably different head sets |
| P3: hard zeros in per-head heatmap due to rank-6 < num_heads=8 | structural | 0/336 heads have ~zero mass across all domains | **NO** — every head receives non-zero coupling; the rank-6 fan-out still spreads across every head's 256-or-512-dim slice |

## Kill criteria results

| KC | Threshold | Measured | Result |
|---|---|---|---|
| K1 (proxy / structural): `mean top-20% energy share` | > 0.50 | **0.3350** | **FAIL** |
| K2 (target / functional): `mean pairwise Jaccard of top-20% sets` | < 0.60 | **0.3494** | **PASS** |

`all_pass = false`, `all_fail = false` → PROVISIONAL, not SUPPORTED, not KILLED.

## Interpretation (F#666 discipline)

Proxy-FAIL + target-PASS per Finding #666 = "finding about the proxy, not a
kill." Concretely:

- **Structural.** rank-6 LoRA on `q_proj` spreads its mass too evenly across
  heads for top-20%-energy-share to exceed 50%. Per-layer, rank-6 B-matrices
  span at most 6 directions in the fan-out space (2048 or 4096 dim), but
  each head's 256-or-512-dim slab intersects those 6 directions enough to
  receive non-trivial coupling. This is why Prediction P3 failed: the
  rank-6 bound did NOT produce hard zero heads.
- **Functional.** Despite lack of sharp concentration, the top-20%-mass
  head SETS differ across domains (J̄ = 0.349, chance level under
  independent 20%-sets ≈ 0.111; 0.349 means ~35% overlap — distinctly
  non-random but also non-task-identical). Different tasks modulate
  different head subsets, consistent with Voita et al.'s functional-role
  hypothesis (arxiv:1905.09418).
- **Actionable outcome.** The "prune non-important heads" motivation
  (experiment notes) is NOT supported: there is no small subset carrying
  >50% of adapter mass, so pruning top-20% saves only ~33% of adapter
  expressivity. Head-level sparsification is not a promising speed
  optimization path for rank-6 q_proj adapters on Gemma 4 E4B.

## Follow-up designed

- **exp_g4_head_ablation_ppl** — direct behavioral validation: load base +
  one domain adapter, run 50-prompt PPL on domain-relevant text, then
  zero-out (a) top-20% heads by this experiment's ranking and (b)
  bottom-20%; compare PPL degradation. If ratio > 2.0, head importance
  matters behaviorally despite weak concentration signal. This closes the
  remaining gap: the present experiment cannot do direct PPL ablation
  because its target KC is a weight-space proxy. Priority 3.

- **exp_g4_head_importance_vproj_oproj_F627** — repeat this analysis on
  F#627-compliant adapters (`v_proj + o_proj` targets). If concentration
  differs sharply there, it means head-importance is target-dependent and
  the q_proj-only finding does NOT generalize. Priority 3.

## Limitations / assumptions (copied from MATH.md §7)

- Weight-space only; behavioral ablation deferred to follow-up.
- q_proj-only adapters; v_proj + o_proj (F#627-recommended) not covered.
- rank=6 only; rank ablation (r=4, r=8, r=16) also deferred.
- Gemma 4 E4B mixed-layer attention handled (7 full-attention layers at
  head_dim=512, 35 sliding at head_dim=256; num_heads=8 in both).
- numpy matmul printed `divide by zero / overflow / underflow / invalid`
  warnings on contiguous finite float64 arrays — a spurious
  numpy-BLAS interaction on this platform (reproduced on clean
  float64 + `ascontiguousarray`). Values are correct: total Frobenius
  energy is finite (code=1011.09, math=916.78, medical=542.75), all 336
  heads have non-zero mass, and K1/K2 are well-defined. The warnings do
  not indicate a bug in the computation.

## Assumptions made autonomously
- Pre-registered thresholds K1=0.50 and K2=0.60 chosen before running; not
  relaxed after seeing data (verdict is PROVISIONAL rather than upgraded to
  SUPPORTED by loosening K1 to 0.30).
- Chose to use existing pre-F#627 `q_proj` adapters rather than train new
  `v_proj + o_proj` adapters because (a) training 3 new adapters would
  exceed the 2h budget, and (b) the head-decomposition math works
  identically on any fan-out-folding projection. Generalization to F#627
  adapters is explicitly deferred as a follow-up.
