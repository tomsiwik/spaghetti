# PAPER: P11.B0 — Rejection Sampling SFT (GRPO Approximation) on MMLU-Pro

**Experiment**: exp_p11_grpo_reasoning_adapter
**Date**: 2026-04-17 (full run completed 2026-04-17T23:29)
**Verdict**: **KILLED** — Theorem 1 (D_train = D_eval ⇒ no forgetting) falsified by measurement.

---

## TL;DR

RS-SFT on self-generated MMLU-Pro traces produced **-15.3pp MMLU-Pro regression** (base 57.1% → adapter 41.8%) and **71% thinking suppression** (2819 → 816 avg chars). All three code-registered kill criteria (K1496, K1497, K1498) fail. The impossibility structure proposed in MATH.md Theorem 1 is falsified by direct measurement: D_train ⊆ D_eval is *not* sufficient to prevent catastrophic forgetting when the training pipeline drops thinking-channel structure.

---

## Prediction vs Measurement Table

| Prediction (MATH.md)                      | Theorem     | Predicted       | Measured (Full Run)        | Pass? |
|-------------------------------------------|-------------|-----------------|----------------------------|-------|
| RS-SFT ≥ 64% MMLU-Pro (K1496)             | Theorem 2   | ≥ 64.0%         | **41.8%** (41/98)          | FAIL  |
| RS-SFT ≥ 56.1% = s1K+20pp (K1497)         | Theorem 1+3 | ≥ 56.1%         | **41.8%** (+5.7pp vs s1K)  | FAIL  |
| All 14 cats ≥ base − 5pp (K1498)          | Theorem 1   | all ≥ −5pp      | **9/14 regressed > 5pp**   | FAIL  |
| Avg thinking chars ≥ 500 (Thm 3 prox.)    | Theorem 3   | ≥ 500 chars     | 816 chars (base 2819)      | PASS* |
| Phase 1 yield                             | GRPO approx | ~62%            | 57.1% (56/98)              | PASS  |
| Phase 2 training success                  | —           | True            | True (200 iters, 83s)      | PASS  |

\* K1519-style threshold passed in absolute terms, but thinking dropped **−71%** vs base — evidence of behavioral regression, not preservation.

---

## Full Run Measurements

```
Phase 1 — Rejection Sampling (1363s = 22.7 min):
  Questions attempted:  98 (stratified across 14 categories)
  Correct (yield):      56 / 98 = 57.1%
  Avg thinking chars:   3010.8 (correct completions)

Phase 2 — RS-SFT Training (83s):
  n_train:              51, n_val: 5
  Steps:                200
  Train loss:           3.99 → 1.39 (converged)
  Val loss:             3.46 → 1.75
  training_success:     True
  Adapter saved:        adapters/rs_sft/adapters.safetensors

Phase 3a — Base Model Eval (1362s, 98q):
  Overall: 57.1% (56/98)
  Avg thinking: 2819 chars/q
  Per-category [ sorted high→low ]:
    biology          85.7%  (6/7)
    economics        85.7%  (6/7)
    business         71.4%  (5/7)
    psychology       71.4%  (5/7)
    math             71.4%  (5/7)
    computer science 71.4%  (5/7)
    chemistry        57.1%  (4/7)
    history          57.1%  (4/7)
    other            57.1%  (4/7)
    physics          42.9%  (3/7)
    engineering      42.9%  (3/7)
    law              28.6%  (2/7)
    health           28.6%  (2/7)
    philosophy       28.6%  (2/7)

Phase 3b — RS-SFT Adapter Eval (700s, 98q):
  Overall: 41.8% (41/98)  [ -15.3pp vs base ]
  Avg thinking: 816 chars/q  [ -71% vs base ]
  Per-category Δ = adapter - base [ sorted worst → best ]:
    physics          0.0%    (0/7)   Δ=-42.9pp  ★ CATASTROPHIC
    math            28.6%    (2/7)   Δ=-42.9pp  ★ CATASTROPHIC  (same domain as training!)
    computer science 42.9%   (3/7)   Δ=-28.6pp
    business         42.9%   (3/7)   Δ=-28.6pp
    engineering      14.3%   (1/7)   Δ=-28.6pp
    psychology       57.1%   (4/7)   Δ=-14.3pp
    philosophy       14.3%   (1/7)   Δ=-14.3pp
    biology          71.4%   (5/7)   Δ=-14.3pp
    history          42.9%   (3/7)   Δ=-14.3pp
    chemistry        57.1%   (4/7)   Δ=0.0pp
    other            57.1%   (4/7)   Δ=0.0pp
    health           28.6%   (2/7)   Δ=0.0pp
    economics        85.7%   (6/7)   Δ=0.0pp
    law              42.9%   (3/7)   Δ=+14.3pp
```

---

## Why Theorem 1 Failed

**Claim** (MATH.md): If D_train = D_eval = MMLU-Pro, catastrophic forgetting is impossible.

**Measurement**: Training on MMLU-Pro caused **-15.3pp MMLU-Pro regression** — including **-42.9pp on the math category that the adapter was explicitly trained on**.

**Falsified assumption**: Theorem 1 assumed that `gradient ∇ E[L]` on D_train computes the same gradient as on D_eval. This holds only if the *forward-pass output distribution* used in training matches the forward-pass distribution used in eval. In this experiment they *diverged*:

1. **Training-time format**: mlx_lm.lora received `{role: assistant, content: full_response_including_<|channel>thought…<channel|>}` — i.e., the channel-thinking markup was treated as literal text in the assistant message, not as a separate generation mode.
2. **Eval-time format**: `generate(..., enable_thinking=True)` invokes the channel-thinking *control-token protocol*, where `<|channel>` is a structural token triggering a different attention pattern, not a bigram in text.

The LoRA adapter (v_proj+o_proj, rank-8) learned to produce `<|channel>` as a literal output token — causing the model to emit shorter, malformed thinking blocks (816 vs 2819 chars) and degraded answer-letter selection. This is **distribution shift from protocol-level mismatch**, a failure mode Theorem 1 did not consider.

**Corrected impossibility structure**: D_train = D_eval is necessary but not sufficient. Also required: the *serialization format* of training targets must preserve the eval-time protocol (channel tokens represented as structural tokens, not text). For Gemma 4 mlx_lm training, this means either (a) strip channel-thinking from training traces and let the adapter only learn answer selection (loses reasoning signal), or (b) train with a chat template that emits channel tokens via tokenizer-level controls (requires custom mlx_lm fork).

---

## Structural Finding (for DB promotion)

**"D_train = D_eval does not prevent forgetting when training-time and eval-time serializations diverge at the protocol level."**

- Affects: any RS-SFT / GRPO adapter trained on thinking-enabled traces via mlx_lm.lora.
- Impossibility fix: Require serialization-protocol match in the training loop (symbolic assertion, not heuristic).
- Related: Finding #553 (tautological routing — Pierre v3-v6 family), Finding #560 (thinking-universal GD violation).

---

## Kill Criteria (Code-Registered Semantics)

```
K1496 (RS-SFT ≥ 64% MMLU-Pro):                  FAIL (41.8%)
K1497 (RS-SFT ≥ s1K + 20pp = 56.1%):            FAIL (41.8%, only +5.7pp over 36.1% s1K)
K1498 (all 14 cats ≥ base − 5pp):               FAIL (9/14 regressed > 5pp; math/physics −42.9pp)
```

Note: DB-registered KC texts (1496/1497/1498) pre-date the 2026-04-14 REVISE that shifted the
approach from GRPO-with-reward to RS-SFT. IDs were retained but semantics drifted. Under
DB-original semantics:
- DB K1496 ("outperforms SFT-only by 5pp"): 41.8% vs s1K 36.1% = +5.7pp → borderline pass
- DB K1497 ("≥70%"): 41.8% → FAIL
- DB K1498 ("meta-cog emerges"): thinking −71% → FAIL

Under either semantics, the experiment is KILLED.

---

## What the Adapter Looks Like (for future reuse)

- `adapters/rs_sft/adapters.safetensors` — full weights, rank-8 v_proj+o_proj, LORA_SCALE=1.0.
- `adapters/rs_sft/0000200_adapters.safetensors` — final checkpoint (same as above).
- **Do not compose with other adapters**: confirmed behavioral regression across all stem domains.
- **Do not promote to registry.json**.

---

## Unblock Path for Successors

The root cause (protocol-level serialization mismatch) is not specific to RS-SFT. Any
thinking-channel-aware LoRA training on Gemma 4 via mlx_lm.lora will face it. Options:

1. **Strip thinking at train time** (loses reasoning signal): set `enable_thinking=False` during
   rejection sampling. Adapter then only learns answer selection. Likely too weak to matter.
2. **Custom chat template with channel-token passthrough**: requires mlx_lm fork. Heavy.
3. **Switch to plain-prompt RS-SFT** (no thinking at train or eval): removes the issue but
   abandons the thinking-mode research direction.
4. **GRPO with a correctness reward model that operates on answer letter only**: avoids SFT
   on full traces; the reward-based gradient may preserve thinking structure because it uses
   the model's own generation distribution rather than imitating it. This is the direction
   the original DB notes described (and the scope the B0 experiment *actually* attempted
   before the 2026-04-14 REVISE simplified it to RS-SFT).

**Recommendation**: do not schedule a v2 of B0 without first resolving the protocol-level
serialization question. If pursuing reasoning adapters on Gemma 4 MLX, option 4 (GRPO) is
the correct path, but needs a custom mlx training loop (mlx_lm.lora is SFT-only).

---

## Connection to Broader Research

Three consecutive kills (F0 s1K OOM, H0 two-STEM GD violation, B0 protocol mismatch) all show
that **reasoning-adapter training on Gemma 4 via mlx_lm.lora is not yet a solved engineering
problem**. Each kill revealed a different structural defect. This should inform P11.C0
(ThinkPO Polish), P11.D0 (Meta-R1), P11.I0 (Synthetic Data Loop) — all of which depend on
the same training-stack and will inherit the same protocol mismatch unless addressed upstream.

---

## Runtime Budget

Total: 3507s = **58.5 min** (well under 2h budget).
- Phase 1: 1363s (22.7 min)
- Phase 2: 83s (1.4 min)
- Phase 3a: 1362s (22.7 min)
- Phase 3b: 699s (11.7 min)
