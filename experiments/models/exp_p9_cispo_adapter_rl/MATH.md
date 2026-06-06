# MATH.md — exp_p9_cispo_adapter_rl (PREEMPTIVE KILL)

## Status: KILLED (preemptive, dependency-chain-unfulfilled + platform-mismatch)

Two independent impossibility arguments — F#669 reuse (3rd occurrence, promote)
AND F#658 reuse (infrastructure-unobtainable on MLX target).

## Experiment claim (per DB)
- K1399 (K1): CISPO-trained adapter outperforms SFT adapter by ≥ 5 pp on GSM8K.
- K1400 (K2): CISPO preserves rare-token gradients: gradient-magnitude ratio > 2× vs PPO for bottom-10% tokens.
- K1401 (K3): Training stable — no reward hacking or collapse.

## Dependency graph (verified via `experiment get` 2026-04-19)
```
exp_p9_cispo_adapter_rl       (P2, open → active, MACRO)
    └── exp_p9_unsloth_rl_environment   (P3, OPEN — unfulfilled)
            └── exp_p9_full_stack_integration   (not in backlog — unfulfilled)
```

Every KC of the target transitively requires the parent-chain having produced a
**trained RL-capable adapter artifact** plus a **working Unsloth RL
environment**. Both parents are OPEN (no trained artifacts exist).

## Theorem 1 (dep-unfulfilled — F#669 reuse, 3rd occurrence)

Let Φ_CISPO denote a CISPO-trained adapter, Φ_SFT the SFT baseline adapter,
and grad_τ the per-token gradient at rare-token position τ.

**Claim.** Without a trained Φ_CISPO, all three KCs are structurally
unmeasurable (0/0).

**Proof.**
1. K1399 measures acc(Φ_CISPO, GSM8K) − acc(Φ_SFT, GSM8K). If Φ_CISPO does not
   exist, acc(Φ_CISPO, ·) is undefined. ⊥
2. K1400 measures ‖grad_τ(Φ_CISPO)‖ / ‖grad_τ(Φ_PPO)‖. Gradients are computed
   during training; untrained → no training trajectory → no gradients. ⊥
3. K1401 measures stability over the training trajectory of Φ_CISPO. Untrained
   → empty trajectory → stability undefined. ⊥

All three reduce to parent K1393 ("RL-trained adapter outperforms SFT by ≥5pp
on verifiable tasks") — which itself depends on parent K1390 (full-stack
integration). The target claim is an **inter-experiment tautology**:
`{K1399, K1400, K1401} ⊆ dep-closure(K1393)`. QED.

## Theorem 2 (platform-mismatch — F#658 reuse)

Let 𝓟 = {MLX, Apple Silicon, M5 Pro 48GB} be the target platform (PLAN.md §II).
Let 𝓤 = Unsloth-RL dependencies = {CUDA runtime, PyTorch+Triton kernels, TRL,
Accelerate, optionally vLLM}.

**Claim.** 𝓤 ∩ 𝓟 = ∅; therefore exp_p9_cispo_adapter_rl cannot be built
on target without changing the target.

**Proof.**
- Unsloth's fast kernels are Triton/CUDA-compiled (arXiv:2506.13585 + Unsloth
  repo). No MLX/Metal backend is published.
- CISPO (MiniMax M1, arXiv:2506.13585) is distributed via the MiniMax codebase
  which requires CUDA + Megatron/DeepSpeed. No MLX port exists.
- PLAN.md §II states "All work must be MLX-native, Apple Silicon is the
  deployment target" (policy: `feedback_mlx_first.md`).
- Attempting to run 𝓤 on 𝓟 requires either (a) a CUDA → Metal port of
  Unsloth/TRL/CISPO (6-month eng. task, off-scope), or (b) shipping the
  artifact to a non-𝓟 machine (violates target-hardware invariant).

Therefore the experiment violates the target-platform constraint. An
infrastructure-unobtainable structural floor prevents execution. QED
(F#658 pattern).

## Theorem 3 (combined — double-axis structural preempt)

T1 (dep-unfulfilled) AND T2 (platform-mismatch) are **independent**:
even if a trained RL env appeared tomorrow, the target still wouldn't run on
MLX; even if an MLX RL port existed, the parent dep is still open. Two
independent impossibility axes concur → preempt is **overdetermined**.

## Kill-criteria disposition (pre-registered, not modified)
| KC   | Predicted | Measured     | Verdict |
|------|-----------|--------------|---------|
| 1399 | fail      | not measured | FAIL (preempt) |
| 1400 | fail      | not measured | FAIL (preempt) |
| 1401 | fail      | not measured | FAIL (preempt) |

## Findings reused
- **F#658** (private-data / infrastructure-unobtainable, (d1/s1) axis) — reused
  here for MLX-platform incompatibility of Unsloth/CISPO.
- **F#669** (child-KCs-require-parent-target-claim-unverified, inter-experiment
  dep-chain) — 3rd reuse. Prior: iter 70 (single-parent),
  iter 71 (double-parent variant). This is the 3rd occurrence — promote from
  sub-axis to standalone finding.
- **F#671** (F#669 double-parent variant) — not directly reused (single-chain
  here), but establishes precedent.

## What would unblock this experiment
1. An MLX-native RL framework supporting CISPO-style importance-ratio clipping
   (none currently exists; would require ~months of engineering).
2. Sequential parent completion:
   exp_p9_full_stack_integration → exp_p9_unsloth_rl_environment → here.
3. Change of target platform (violates `feedback_mlx_first.md`).

None of these are achievable in the drain-loop's iteration budget. Preempt
stands.
