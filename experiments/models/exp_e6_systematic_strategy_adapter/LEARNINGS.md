# LEARNINGS: E6 — Systematic Strategy Adapter (Hedgehog Distillation)

## Core Finding

Hedgehog per-layer cos-sim distillation is effective for **surface behaviors** (politeness, F#783) but **antagonistic for reasoning strategies** (systematic decomposition). The adapter achieves cos=0.96 attention match with the teacher yet degrades accuracy across all 4 domains (-5pp to -15pp), with GSM8K outputs becoming entirely null. F#666 tautological-proxy confirmed.

## Why

The training objective matches teacher attention on the **input pass**, but reasoning strategies manifest during **generation**. The adapter learns how the teacher processes input under a decomposition prompt, but this doesn't constrain generation behavior. Instead it shifts the output distribution away from concise answers — the model begins generating decomposed reasoning text rather than direct answers, overriding format instructions.

This is distinct from E1's failure (F#801): E1 failed at extraction (format signal dominated strategy signal). E6 succeeds at structural matching but the matched structure is the wrong one — input-side attention vs generation-side behavior.

## Implications for Next Experiments

1. **E7/E9 (strategy transfer, composable CoT)**: Cannot use Hedgehog distillation for reasoning strategies. Must use **generation-aware training** — SFT on strategy-eliciting outputs, or RL from strategy compliance rewards.
2. **E11 (linear strategy extraction)**: Must use contrastive extraction (A−B) per E1 learnings. Combined with E6: even correct extraction won't help if injection doesn't affect generation.
3. **E8 (behavioral eval)**: Still needed — we need evals that detect strategy *application* independent of accuracy.
4. **General principle**: Distillation method must match the *locus* of the target behavior. Surface behaviors live in input processing (Hedgehog works). Reasoning strategies live in autoregressive generation (Hedgehog fails).
5. **Hedgehog domain adapters** (P=3 backlog: JS, Python, Rust): These target coding style/conventions — likely surface behaviors where Hedgehog should work. Not killed by E6.
