# LEARNINGS: exp_m2p_code_sft_4b

## Core Finding
Theorem 1 (zero-init SFT-residual quality floor) is structurally verified: same B_sft
matrix produces identical output whether applied via SFT path or M2P init path. The
SFT-residual architecture is sound and solves the anti-format interference problem
(Finding #407). However, Grassmannian-orthogonal code A-matrices conflict with Qwen3-4B's
existing code capability — SFT degrades 37.78%→11.11% and M2P→0%.

## Why
The Grassmannian construction ensures geometric isolation between domains (|A_math^T
A_code|_F < 1e-4) but says nothing about whether A_code lies in a subspace that preserves
the base model's existing code representations. Qwen3-4B already achieves 37.78% on toy
code tasks (strong prior), so projecting gradients through orthogonal-but-destructive
A-matrices corrupts rather than extends the capability. This is structurally distinct from
anti-format interference — SFT-residual cannot fix A-matrix placement.

## Implications for Next Experiment
Two clean paths forward:
1. **B=0 routing**: For domains where base model is already strong (>30% target task),
   route directly to base model — no LoRA needed. Composition theorem still holds (W=base).
2. **Trained A-matrices**: Learn A-matrices from domain gradient data rather than imposing
   Grassmannian geometry. These are compatible; use Grassmannian only where base is WEAK.

The room model can already serve N=25 domains (Finding #406) with math M2P + B=0 for
other domains. Code quality under composition is separate from isolation/routing, which
both hold perfectly (routing=100%, math_qr=1.3125 unchanged).

## Status: supported
