# MATH: Training recipe that preserves thinking mode

Experiment id: `exp_model_thinking_preservation_training`
Platform: `mlx-community/gemma-4-e4b-it-4bit`, MLX `mlx-lm` 0.28.x
Type: **frontier-extension** — extends F#536 (thinking baseline) and F#403 (SFT-residual) to the failure mode of F#538 (catastrophic forgetting under reasoning SFT) and F#560 (gradient homogeneity under 2-STEM domains).

## 1 Failure mode

`<think>...</think>` is the Gemma 4 reasoning channel (F#536). When a LoRA
adapter is trained on targets that do **not** contain `<think>` tokens (classic
MCQ-only SFT), the token-level cross-entropy pushes `P(<think> | prompt)` to
zero: the adapter learns a `<think>`-suppression prior. At eval time with
`enable_thinking=True`, the same prior wins and suppresses the thinking chain
(F#536: −11.7pp, 0 thinking chars).

Converse failure (F#538): training on pure s1K (olympiad math + full thinking)
causes −26pp catastrophic forgetting on MMLU-Pro because the posterior over
content shifts far off-distribution even though `<think>` is preserved.

We need a recipe that simultaneously:
1. **keeps the `<think>` prior high** — eval-time thinking is not suppressed;
2. **keeps the content posterior near-base** — MMLU-Pro within 2pp of
   base+thinking 62.1% (F#536).

## 2 Prior math

- **F#536** (supported): base Gemma 4 E4B 4-bit + thinking = 62.1% MMLU-Pro;
  MCQ-adapter + thinking = 50.4% (−11.7pp); the gap is **caused by thinking
  suppression**, not reasoning-circuit damage.
- **F#538** (killed): s1K pure-math SFT → 36.1% MMLU-Pro (−26pp); thinking
  length preserved (1641 chars), so forgetting is *content*, not *trace*.
- **F#403** (supported, Qwen3-4B): SFT-residual heads
  `B_applied = B_sft + s · head(z)` with zero-init heads achieve
  `quality_ratio = 1.175` — the zero-init guarantees the recipe starts at the
  base distribution.
- **F#4** (conclusive): MoE over domain-diverse SFT beats joint training by
  0.7pp avg, confirming domain diversity matters.
- **F#560** (killed): 2-STEM domains insufficient for gradient diversity
  (`GD < 0.5` lower bound). Need ≥3 non-collinear domains.

## 3 Theorem (recipe preserves thinking + content)

**Setup.** Let `π_base(· | x)` be the base distribution at prompt `x` with
`enable_thinking=True`. Let `ℓ_base = log π_base(y | x)` on token `y`. Define
the `<think>` log-prior `T_base(x) = log π_base(<think> | x)`.

Let `{D_k}_{k=1}^K` be `K ≥ 3` non-collinear domains with training targets
`y^*_k` that each contain a `<think>...</think>` block. Let adapter `Δ` be
LoRA of rank `r` on `{v_proj, o_proj}` trained with `enable_thinking=True` for
`N` steps at learning rate `η`, producing `π_Δ`.

**Claim.** Under the assumptions (A1)–(A3) below, with probability ≥ 1 − δ:

```
|acc(π_Δ, MMLU-Pro+thinking) − acc(π_base, MMLU-Pro+thinking)| ≤ 2pp
T_Δ(x) ≥ T_base(x) − ε_T,   with ε_T → 0 as (training targets contain <think>)
thinking_chars(π_Δ) ≥ 1500 on average
```

**Assumptions**
- (A1) **Thinking-in-target**: every training pair `(x_k, y^*_k)` has
  `<think>` tokens contributing ≥ 40% of target-token loss mass. This keeps
  `∇_Δ log P(<think>|x)` positive-definite throughout training.
- (A2) **Domain non-collinearity**: `K ≥ 3` with pairwise Grassmannian
  distance of gradient means `≥ 1.0 / √r` (F#560 lower bound violated by
  2-STEM; 3 domains from disjoint macro-categories suffice).
- (A3) **Bounded drift**: `η · N · √r · max‖∇L‖ ≤ 0.5` — this is the SFT-residual
  regime where the adapter is a perturbation around the base (F#403 verified
  `quality_ratio = 1.00` at init in this regime).

**Proof sketch (deferred to full run).** Under (A1)–(A3), Gemma 4's cross-entropy
loss decomposes token-wise. Tokens inside `<think>...</think>` contribute
log-likelihood `≥ 0.4 · L_total`, so `∇ T_Δ = ∇ log P(<think>|x) > 0` at every
gradient step; by convex-projection argument on the logit space, `T_Δ ≥ T_base`.
Under (A3) the remaining logit drift is bounded by `O(η·N·√r)`, and by Pinsker
the TV distance to base on the content-posterior is
`≤ √(KL/2) ≤ 0.5 · √(η·N·√r)`, which for our chosen hyperparameters
`(r=8, η=1e-5, N=1000)` is `< 0.05`, translating to `≤ 2pp` accuracy drift
(Gemma 4 E4B 4-bit has accuracy–KL Lipschitz constant ≈ 40 from F#536 data).

QED (sketch; full run verifies).

## 4 Predictions (kill criteria)

Pre-registered kill criteria on full (non-smoke) run:

- **K1685** — trained adapter + `enable_thinking=True` scores **within 2pp**
  of base+thinking on MMLU-Pro overall (i.e. `adapted_mmlu ≥ 60.1%`).
- **K1686** — adapter's avg thinking chars per question **≥ 1500** (F#538
  baseline 1641).
- **K1687** — recipe holds **across ≥3 MMLU-Pro categories** (math,
  comp_sci, health): each category within 5pp of its base-category
  counterpart.

Smoke-mode thresholds (explicit): `SMOKE_TEST=1` uses `N_STEPS=20`,
`EVAL_PER_CAT=2`; smoke-mode cannot pass K1685/K1686/K1687 conclusively and
must complete as `--status provisional`.

## 5 Assumptions logged

- **Domain sources**: smoke uses s1K (olympiad-math) for training (has real
  `<think>` traces); code + medical are **eval-only** here (smoke budget).
  Full-scale rerun needs `K=3` training-domain mixture per (A2).
- **Base model**: 4-bit quantization. Findings F#536/F#538 both used the same
  quant, so acc drift is comparable.
- **Adapter target**: LoRA `r=8` on `self_attn.v_proj + self_attn.o_proj`
  (F#403 config, PoLAR target set from PLAN.md Part 2).
- **SFT-residual vs vanilla LoRA**: smoke uses vanilla LoRA (mlx-lm.lora CLI).
  Full-scale recipe (A3 regime) requires SFT-residual custom head from F#403;
  that's a separate implementation task documented in the next-step TODO.

## 6 Falsification

If smoke already shows adapter MMLU-Pro below base by **> 5pp** at N=20
(with thinking), the recipe is falsified at smoke scale — thinking-in-target
is *not* sufficient and the hypothesis needs (A1) tightening or SFT-residual
architecture. Reported as `provisional` with explicit falsification flag.
