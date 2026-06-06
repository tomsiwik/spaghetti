# LEARNINGS — exp_score_kl_constrained_mcq

## Core Finding (smoke, PROVISIONAL)
SCoRe stage-I KL-constrained SFT (arxiv:2409.12917) on rank-8 LoRA (v/o) over
Gemma-4-E4B-it-4bit runs end-to-end in MLX. At β=1.0 the KL constraint is
**non-binding by ~40×**: max step-wise KL over 20 steps = 0.00257 nats vs the
K1726 bound of 0.1 nats. Both eval arms score 50.0% (3/6) on MMLU-Pro smoke:
K1724 FAILs numerically (50.0 < 50.4 plain-SFT baseline), K1725 trivially
PASSes at Δ=0 pp. n_eval=6 ⇒ ±20 pp binomial CI, informative-only.

## Why
- `nn.value_and_grad(model, kl_loss)` + `mx.stop_gradient(base_model(...))`
  isolates gradient flow to LoRA weights; two Gemma-4-E4B-4bit instances fit
  in 8.55 GB active on M5 Pro 48 GB.
- SCoRe Theorem 3.1 predicts rank-limited LoRA produces small KL drift;
  rank-8 at β=1.0 stays deep inside the 0.1 nat trust region — so on
  mechanism, pure SFT (no KL term) would likely also satisfy K1726.
- Thinking-channel capture now works after the round-1 diagnostic-gate fix:
  base avg 2782 chars, adapter avg 2630 chars (−5%, within n=6 noise).

## Implications for next experiment
- **Full-scale rerun is the decision point** (`SMOKE_TEST=0`, N_STEPS=1000,
  EVAL_PER_CAT=20 → n=60 per arm). Do not upgrade verdict until complete.
- If n=60 still shows adapter ≈ base, the learnt update is
  information-theoretically vacuous (consistent with max_kl≈0.0026). That is
  a **structural** result — not a hyperparameter tweak. Next move would then
  be rank↑ or β↓ to push the update out of the trivial neighborhood.
- Pre-flight (PLAN.md §1) before status=supported: update PAPER.md
  §What-smoke-validates #3/#4 and §does-NOT #1 to current numbers
  (0.00257, 50%/50%, per-category 50% across all six cells) — flagged as
  caveats carried forward from round-3 review.

## Connects to
- Finding #536 (parent: MCQ adapter destroys thinking, −11.7 pp).
- Finding #538 (s1K 27/3 train/valid split).
- Finding #586 (this experiment: KL non-binding at rank-8 v/o LoRA + β=1.0).
- `exp_model_thinking_preservation_training` (parent, killed at smoke — this
  is the explicit KL-regularised variant).
- `mem-antipattern-006` (smoke-as-full — do not upgrade to supported).
- `mem-antipattern-008` (thinking truncation — regex fix landed here).
