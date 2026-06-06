# LEARNINGS.md — exp_p9_cispo_adapter_rl

## One-line
**KILLED preemptively** by two independent structural arguments: parent
dep-chain unfulfilled (F#669, 3rd reuse) + platform-mismatch
(F#658 — Unsloth/CISPO are CUDA/Triton-only, target is MLX).

## What worked
- Structural preempt under double-axis impossibility is cleaner than any
  partial/smoke attempt would have been. No ghost-execution, no
  cargo-culting CISPO semantics onto an unrelated MLX smoke.
- F#669 reuse progression: single-parent → double-parent → single-parent
  co-preempted with F#658. This is the 3rd reuse — **propose promotion**
  from sub-axis to standalone finding: "inter-experiment dep-chain KC
  reduction" as a first-class axis, independent of the platform axis.

## What didn't
- N/A (no execution).

## Rule-of-thumb for future P2+ macro RL experiments on MLX target
Any experiment whose runtime stack requires Triton/CUDA/Megatron/DeepSpeed
(pattern match: "PPO/GRPO/CISPO/REINFORCE with speed-oriented kernels") must
be **either** re-designed natively in MLX **or** preempted via F#658. The
former is macro engineering work (out of drain budget); the latter is the
default for drain-forward.

## Handoff notes
- Follow-up experiment `exp_followup_cispo_mlx_native` could be spawned — it
  would need:
  1. Independent KCs not reducing to parent K1393/K1390.
  2. MLX-native gradient-ratio computation (tractable — MLX supports JVP).
  3. A reasoning task smaller than GSM8K for M5-scoped training.
  4. Pre-registered kill-scale (proposal: N≤200 train pairs, ≤30 min budget).
- Not created in this iteration to avoid exceeding Guardrail 1008 (anti-stuck).

## Findings affected
- F#658 — reused (MLX-platform incompatibility sub-case).
- F#669 — **3rd reuse, promotion candidate** (sub-axis → standalone).
- F#671 — precedent-cited, not directly reused (single-chain here).
