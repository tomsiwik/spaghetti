# PAPER — exp_adapter_pruning_post_training

## Verdict: **KILLED**

50% magnitude pruning of trained LoRA adapter weights causes a behavioral PPL increase of 0.331 on the medical domain — **3.3× the K1922 threshold of 0.10**. The dense adapter cannot be replaced with a 50%-pruned form without measurable behavioral degradation.

A separate K1923 (composition) result shows pruning *under additive composition* did **not** fire its 3-PPL threshold — it produced a slight (−0.13 PPL) *improvement*, an unexpected sign. This is logged as a novel observation; it does not rescue the experiment because K1922 (single-adapter) is the gating criterion.

## Setup
- Base: `mlx-community/gemma-4-e4b-it-4bit` (MLX 0.31.1, mlx_lm 0.31.2)
- Adapter: `exp_p1_t2_single_domain_training` medical + math (q_proj only, rank 6, α=6.0; trained 1000 iters)
- Phase 1 eval: 100 rows of `data/medical/valid.jsonl`
- Phase 2 eval: 50 medical + 50 math (mixed)
- Pruning: per-matrix top-50% by |w| (zero out smallest 50% entries of each `lora_a` and `lora_b` independently)
- Seed: 42
- Total runtime: ~62 seconds (Phase 1 32.9s + Phase 2 29.5s)

## Prediction vs measurement

| Quantity | Predicted (MATH.md) | Measured | Notes |
|---|---|---|---|
| Retained energy fraction f (Gaussian top-50%) | 0.886 | 0.904 (A) / 0.922 (B) | tight match; trained LoRA matrices are slightly more concentrated than centered Gaussian, consistent with training-induced sparsity |
| ‖ΔW − ΔW'‖_F / ‖ΔW‖_F (sample layer) | 0.3–0.5 | **0.376** | within predicted range; theorem upper-bound was 0.656 |
| ΔPPL_single (K1922) | 0.6–1.2 | **0.331** | **lower** than predicted but still > 0.10 threshold; K1922 FIRES as predicted |
| ΔPPL_compose (K1923) | 1.0–1.7 (single × √2) | **−0.129** | **opposite sign** — pruned composition is slightly *better* than full. Novel observation. |

## Result tables

### K1922 — single-adapter (medical)
| Adapter state | PPL |
|---|---|
| Full | 29.612 |
| 50%-magnitude-pruned | 29.943 |
| **Δ** | **+0.331** |
| Threshold | 0.100 |
| K1922 | **FIRE** (kill) |

### K1923 — composition (medical + math)
| Adapter state | PPL |
|---|---|
| Full med + Full math | 15.231 |
| Pruned med + Pruned math | 15.102 |
| **Δ** | **−0.129** |
| Threshold | 3.000 |
| K1923 | pass |

## Interpretation

**Why the single-adapter measurement was lower than predicted (0.33 vs 0.6-1.2):**
The first-order PPL prediction assumed `ΔPPL ≈ ε · |∂PPL/∂‖ΔW‖|`. The measured local sensitivity at this LoRA's operating point is ~0.88 PPL per unit of `‖ΔW − ΔW'‖_F / ‖ΔW‖_F`, lower than the F#674 cross-domain ablation slope. This is consistent with the medical adapter's modest baseline contribution — it lowers PPL by only ~0.7 on its training domain, so removing 38% of its effect costs only ~0.33 PPL.

**Why composition pruning paradoxically *improved* PPL:**
Three plausible mechanisms (not yet distinguished):
1. **Destructive interference removed** — the smallest-magnitude entries in `A_med · B_med` may have been *anti-correlated* with `A_math · B_math` at certain (i, j) positions; pruning them removes a low-amplitude cancellation that was hurting both adapters.
2. **Implicit regularization** — at α=6.0, both adapters together apply scale-12 to the LoRA path. Pruning effectively reduces effective scale, moving toward a less-overfit operating point on a 50/50 mixed eval that is in-distribution for neither adapter alone.
3. **Statistical noise** — N=100 PPL measurement has σ ≈ 0.1; 0.13 is on the edge of significance. A repeat-run with different seeds or a permutation test is needed to lock this in.

**Pierre-serving implication.** A halved-memory dense-LoRA stack via magnitude pruning is **not viable** at K=50% with standalone single-adapter quality preserved. Possible escape routes (not in this experiment):
- Lower keep-fraction (e.g., K=70% may pass K1922 — predicted ε ≈ 0.27, ΔPPL ≈ 0.18; still might fail).
- Pair magnitude with activation-statistics (Wanda-style) — would require an importance pass through the model.
- SVD-truncate the materialized ΔW = α·A·B at rank-r/2 instead of magnitude-prune A,B. Eckart–Young guarantees the optimal low-rank perturbation; this is **not** what magnitude pruning does and is a structurally different operation.

## Antipattern checklist (post-run)
- [x] Composition math: `(dx @ A) @ B` matches `mlx_lm.tuner.lora.LoRALinear` forward — verified by re-reading exp_composition_ordering_matters scaffolding before adapting.
- [x] LORA_SCALE = 6.0, matches trained yaml.
- [x] No `shutil.copy`; pruning is `np.partition` + masking.
- [x] No hardcoded `"pass": True` — KCs computed from PPL deltas.
- [x] No eval truncation surprise — `max_seq_len = 512`, only 1–2 examples truncated.
- [x] Same model as adapter trained on (no proxy substitution).
- [x] `is_smoke = false`. N=100 rows per phase; not a smoke run.
- [x] No KC modification post-run (`git diff MATH.md` clean since pre-registration).
- [x] PPL measured on the adapter's training domain — correct object for K1922.

## Assumptions logged (per researcher hat §Context discipline)
- **K1923 threshold "3pp"**: interpreted as 3.0 PPL absolute, not 3% relative. Logged in MATH.md.
- **Pruning granularity**: per-matrix top-50% (independent A and B) was chosen over per-layer or global top-50%. Per-matrix is the standard LoRAPrune (arxiv:2305.18403) configuration.
- **Eval set ordering**: `data/medical/valid.jsonl` first 100 rows in file order (no shuffle). Same data the adapter saw a held-out fraction of during training. Consistent with `exp_composition_ordering_matters` precedent.
- **Diagnostic-only RuntimeWarnings**: `np.matmul` raised divide/overflow/invalid warnings on the diagnostic-only `A @ B` computation in fp32 (note: no warnings on the actual experiment forward passes; values are bounded). The diagnostic relative gap (0.376) is sensible and matches the theorem; warnings are non-blocking.

## Novel observation seed (post-experiment)
**Composition pruning shows opposite-sign behavior from single-adapter pruning.** The single-adapter PPL increased by 0.331 under 50% pruning, but the same operation applied to a 2-adapter additive composition decreased PPL by 0.129. This sign reversal is unexpected under independent-error addition (which predicted +√2 × single-adapter degradation, ≈ +0.47 PPL). This suggests that pruned composition is *not* simply "noise added", but interacts with the structure of inter-adapter interference. Worth a follow-up:

- **F-followup**: measure ΔPPL_compose at multiple keep_frac ∈ {0.3, 0.5, 0.7, 0.9} on med+math and on a non-trained pair (med+code which were not co-trained). Hypothesis: pruned composition gain is largest where adapters share representational subspace.

## References
- arxiv:2306.11695 (Wanda) — magnitude × activation pruning for full LLM weights at 50% sparsity reports ΔPPL ≈ 0.5–1.0 on WikiText.
- arxiv:2305.18403 (LoRAPrune) — magnitude pruning on rank-8 LoRA preserves task accuracy within 1pp at 50% sparsity on GLUE; this experiment finds **PPL preservation is harder than task-accuracy preservation** at K=50%.
- F#674 (cross-domain LoRA ablation) — full removal cost ≈ 1.5–3.0 PPL; partial-magnitude removal at 38% effective destruction cost 0.33 PPL — slope ≈ 0.88 PPL per unit relative deltaW.
- F#744 (immediately preceding finding from `exp_composition_ordering_matters`) — same scaffolding (LoRALinear monkey-patch, 4-bit Gemma 4 E4B, q_proj r=6) reused; confirms pattern.
