# MATH.md — exp_score_kl_constrained_mcq

## Claim (Theorem)

Let `π_θ` be a LoRA-adapted Gemma-4-E4B-it-4bit and `π_0` the frozen base. Training
on s1K reasoning traces (`Finding #538`: 27 thinking-enabled examples) with the
SCoRe (Kumar et al., `arxiv:2409.12917`) stage-I KL-constrained objective

```
L(θ) = E_{(x,y)~D}[ -log π_θ(y|x) ]  +  β · E_{x~D}[ KL( π_0(·|x) ‖ π_θ(·|x) ) ]
```

over token-level logits bounds the policy drift so that, at every training step `t`,

```
KL( π_0(·|x) ‖ π_θ_t(·|x) )  ≤  KL_max(β)
```

for some `KL_max` monotone-decreasing in `β`. We claim (K1726) that `β = 1.0`
bounds per-step average KL below `0.1` nats. We further claim (K1725) that the
thinking-mode preservation failure observed in `Finding #536` (base 62.1% →
plain-SFT adapter 50.4% MMLU-Pro+thinking, −11.7pp) is *caused* by unbounded
policy drift on the first-token distribution, and is fixed by the above
constraint — adapter+thinking will land within 2pp of base+thinking.

## Mechanism (why it works)

1. **Plain SFT maximises log-likelihood on training targets only.** Tokens
   outside the training distribution — including thinking-mode channel control
   tokens (`<|channel|>thought`, `<|end_channel|>`) — experience arbitrary
   logit drift, destroying the pretrained thinking-mode prior.
2. **The KL penalty is a **trust region** around `π_0`.** By `Pinsker's
   inequality`, `TV(π_0, π_θ) ≤ √(KL/2)`; bounding KL ≤ 0.1 caps total
   variation distance at ≤ 0.22, which empirically (`arxiv:2409.12917`
   §4.3 — matched to `β≈0.5` for 7B model) preserves behaviour on tokens
   the gradient does not explicitly update.
3. **LoRA constrains capacity; KL constrains direction.** LoRA's rank-r update
   bounds ‖ΔW‖₂ structurally, but this alone does not bound the *information-
   geometric* distance between `π_θ` and `π_0` — a low-rank change can still
   produce a logit shift much larger than `ε` on tokens with sharp priors.
   SCoRe's KL regulariser acts directly in output-distribution space, closing
   the gap that pure rank-bounding leaves open (Theorem 3.1 of
   `arxiv:2409.12917`).
4. **Behavioral prediction (the load-bearing one).** If `Finding #536`'s
   −11.7pp gap is *driven* by out-of-support logit drift (hypothesis H1), then
   the KL-constrained objective restores the thinking-mode priors used during
   MMLU-Pro inference and the accuracy gap collapses. If the gap is driven by
   something else (e.g., data coverage, L1‑distillation artefact), KL
   constraint will *not* close it — and K1725 will falsify.

## Assumptions / caveats

- `β = 1.0` is chosen by grid literature (`arxiv:2409.12917` Table 2); we do
  not sweep. If `β` is off by >10× the whole setup fails.
- `π_0` is a separate-instance frozen copy of the same weights (loaded without
  LoRA), so double-memory during training. Memory budget on M5 Pro 48GB has
  been verified in smoke.
- KL is computed on **every position** (not first-token only) for efficiency.
  K1726 measures **average KL per token** over the same positions that CE
  uses (the answer span). The literal SCoRe paper restricts to first-token;
  our variant is stricter (more tokens ⇒ tighter constraint).
- Smoke mode: `n_train = 27`, `n_steps = 20`, `eval_per_cat = 2` — the smoke
  run only validates the *mechanical pipeline* (KL computed correctly, trainer
  runs to completion, memory fits). `is_smoke = True` ⇒ verdict is
  **provisional** regardless of K1725/K1726 observed values.

## Pre-registered kill criteria (LOCKED)

- **K1724**: MCQ accuracy (MMLU-Pro) under adapter+thinking
  ≥ 50.4% (the plain-SFT adapter baseline from `Finding #536`). Binary pass/fail
  on the full-scale run; smoke run only verifies the code computes the metric.
- **K1725**: MMLU-Pro+thinking accuracy of adapter is within **2pp** of base
  accuracy (base ≈ 62.1%); equivalently `|Δ| ≤ 2.0`. Binary pass/fail.
- **K1726**: During training, the **average KL divergence** `KL(π_0 ‖ π_θ)`
  over answer-span tokens, measured at every step logged by the trainer, is
  ≤ 0.1 nats at all logged steps. Binary pass/fail.

All three must pass for `status=supported`. Any fail at full scale ⇒ `killed`.
Smoke mode ⇒ `provisional` (with per-KC values reported as informative but not
binding).

## References

- Kumar et al., "Training Language Models to Self-Correct via Reinforcement
  Learning" (SCoRe), `arxiv:2409.12917` — stage-I KL constraint (Eq. 4).
- Finding #536 — MCQ adapter suppresses thinking, −11.7pp MMLU-Pro.
- Finding #538 — s1K thinking-enabled traces (27 train, 3 valid).
- `mem-antipattern-008` — thinking-mode truncation; addressed by
  `max_tokens=2048` during eval.
- `exp_model_thinking_preservation_training` — parent experiment (killed at
  smoke: plain SFT regressed to 33.3% MMLU-Pro at n=6; current experiment is
  the KL-constrained variant that should close that gap).

## mlx-lm version

`mlx_lm == 0.31.2` (verified via `uv run python -c "import mlx_lm; …"`).
Custom loss is plugged into `mlx_lm.tuner.trainer.train` via the `loss=`
kwarg introduced in 0.21 (still present in 0.31).
