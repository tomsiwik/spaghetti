# MATH.md — Off-domain LoRA interference is a decode-POSITION transient

**Experiment:** `exp_spark_position_transient_interference`
**Platform:** local-apple (MLX, `mlx_lm` 0.31.2), frozen `mlx-community/gemma-4-e4b-it-4bit`
**Adapter:** `data/adapters/math/adapters.safetensors` — real F#627 math LoRA, `self_attn.q_proj` only, rank 6, train scale 6.0 (≤ 8). 84 keys = 42 layers × {lora_a (2560×6), lora_b (6×2048)}.
**References:** F#627 (solo adapters give large on-domain lift), F#827/F#837 (off-domain interference −12..−14pp is real and behavioral), F#847 (per-prompt routing is dead — this relocates onto the orthogonal decode-step axis, never tested). TIES arXiv:2306.01708 (static merge averages position structure away).

---

## 1. Failure mode being probed

The whole prior literature treats a LoRA adapter as a **static weight-space object** merged once before decoding, so off-domain interference is implicitly assumed **time-invariant** over the generated sequence. If that frame is correct, the destructive next-token NLL mass of the wrong (math) adapter on medical prompts is spread roughly uniformly across generated positions, and no position-gating policy can separate "damage" from "lift" — because there is no position structure to exploit.

The disease (not the symptom): interference is being modeled on the **wrong axis** (task/weight) when its true support is the **decode-step/position** axis. Damage is hypothesized to be *front-loaded* — the wrong adapter hijacks the opening tokens (format/register/answer-commitment: e.g. for an MCQ it shoves the model toward a math-style "Solve step by step" register instead of emitting a clean letter), then its influence decays as the growing frozen-base context reasserts control.

## 2. Construction / mechanism (no analogy)

Single-adapter composition, exact `Σ Bᵢ@Aᵢ` form, applied per q_proj layer:

```
y_L = base_q_proj_L(x) + scale · (x @ lora_a_L) @ lora_b_L,   scale = 6.0
```

This is the identical, reviewer-verified composition used in `exp_spark_temporal_interference`
(`(x@A)@B·scale`, subclass `nn.Module` + `setattr`, never `__call__`-override — per mem-antipattern-call-override).

**Position gate (late-fire-16).** Let `g_t ∈ {0,1}` be the adapter gate at generated position `t` (0-indexed):
`g_t = 0` for `t < 16`, `g_t = 1` for `t ≥ 16`. The prompt forward pass is always gated ON for `t ≥ 16` and the gate is a Python scalar mutated per decode step (no params, no grad). For `t < 16` the model decodes as the frozen base; thereafter the math adapter fires.

Why a single global gate suffices (vs. per-layer): the adapter is q_proj-only, so the gate multiplies the *entire* adapter contribution at that step — there is exactly one composition channel.

## 3. Quantitative predictions

Let `delta_t = NLL_compose(y_t | prompt, y_<t) − NLL_base(y_t | prompt, y_<t)` measured by **one teacher-forced forward pass per sequence per condition** over the 50 held-out medical reference completions (the gold assistant text), aligned to generated-token position `t` (0-indexed within the assistant span). Positive `delta_t` = the wrong adapter raised the NLL of the gold continuation = destructive mass at that position.

**Diagnostic (frontload_ratio).**
```
frontload_ratio = Σ_{t=0..15} max(delta_t,0)  /  Σ_{t=0..T-1} max(delta_t,0)
```
Prediction: **frontload_ratio ≥ 0.60** — at least 60% of the cumulative destructive NLL mass lands in the first 16 generated positions. (If interference were time-invariant, with typical T≈80–256 medical completion tokens the first-16 share would be ≈ 16/T ≈ 0.06–0.20; ≥0.60 is a >3× concentration over the uniform null.)

**Behavioral target.** Real greedy generation, 3 conditions:
- `B` base (adapter off everywhere)
- `A0` adapter always on
- `A16` late-fire (adapter off for first 16 generated tokens, then on)

Accuracies: `acc_med_B, acc_med_A0, acc_med_A16` (medical MCQ, score first valid A/B/C/D letter) and `acc_math_B, acc_math_A0, acc_math_A16` (math, `####` / final-number extraction).

```
off_recovery = (acc_med_A16 − acc_med_A0) / (acc_med_B − acc_med_A0)
on_retention = (acc_math_A16 − acc_math_B) / (acc_math_A0 − acc_math_B)
target_score = min(off_recovery, on_retention)
```

**Substrate preconditions** (the interference + lift must exist to be relocatable; if either fails the hunch is untestable on this adapter — provisional, not killed):
- `acc_med_A0 ≤ acc_med_B − 0.06` (wrong adapter measurably hurts medical)
- `acc_math_A0 ≥ acc_math_B + 0.06` (adapter measurably lifts math)

## 4. Pre-registered kill criterion

**KC 2293 (target-metric, behavioral).**
```
KILL  iff  target_score < 0.70
SUPPORTED iff target_score ≥ 0.70  AND  frontload_ratio ≥ 0.60  AND  preconditions hold
```
Late-fire must recover ≥70% of the off-domain medical accuracy loss **and** retain ≥70% of the on-domain math accuracy lift. `frontload_ratio ≥ 0.60` is the mechanistic diagnostic that the recovery is due to a genuine front-loaded transient (proxy); the **verdict is gated by the behavioral `target_score`** (per mem-antipattern-proxy-only-kc / F#666: the target metric drives the verdict, the diagnostic explains it).

If preconditions fail → `provisional` (substrate didn't exhibit measurable interference/lift), is_smoke stays False, no kill.

## 5. n and token-budget justification (keep run < 2h)

- **Diagnostic:** 50 held-out medical reference sequences (`data/corpora/distillation/medical/eval.jsonl`, 0% overlap with the adapter's medical/math training pools — verified). **One forward pass per sequence per condition** (base + compose) = 100 forwards, no autoregressive decode, no lockstep dual decode → minutes. (Spec said "~96"; the held-out, uncontaminated medical eval pool is 50. Using train data would contaminate the NLL diagnostic, so 50 is the honest ceiling. Per-position means over 50×(≤512 tokens) is ample for a ratio estimate.)
- **Behavioral:** 50 medical MCQ + 50 math (full held-out eval pools), 3 conditions, greedy. `MAX_NEW_TOKENS = 256`. MCQ answers and `####`-style math answers land well under 256 tokens; 256 caps the worst case while keeping 3×100 generations comfortably < 2h on M-series. (Spec said 64+64; 50+50 is the held-out ceiling without contamination.)
- `enable_thinking=False`: training used the instruction templates directly with the answer as the assistant turn; we replicate the exact training prompt templates so the adapter activates as trained and answers are clean/short.

QED-of-design: every number above is measured by real frozen-base + real math LoRA forward/decode in MLX; `is_smoke: false`.
