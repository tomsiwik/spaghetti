# MATH — Base safe-harbor via discrete top-k conflict gating

**Experiment:** `exp_spark_base_safe_harbor`
**Type:** Guided exploration (proven composition framework; unknown parameter τ).
**Base model:** `mlx-community/gemma-4-e4b-it-4bit` (frozen, 4-bit).
**Adapter:** F#627 solo **math** LoRA — `data/adapters/math/adapters.safetensors`, `self_attn.q_proj`, rank 6, scale 6.0 (≤ 8 ✓). Same weights used by sibling `exp_spark_entropy_gated_lora`. On-domain GSM8K lift / off-domain HumanEval drop are the F#627 behavioral signature.
**mlx-lm:** 0.31.2.

---

## 1. Failure mode being prevented

A single off-domain LoRA adapter degrades unrelated tasks (F#827: interference is real and behavioral; F#627: the math adapter is +GSM8K / −HumanEval). Every prior gate keyed on a **continuous magnitude** signal — energy (F#183/184, DEAD), entropy (`exp_spark_entropy_gated_lora`, codex-drift). These fail because a trained adapter **monotonically lowers NLL**: low-entropy/low-energy steps are exactly the confident steps where the adapter is *also* confident, so a magnitude gate cannot separate "helpful" from "harmful" perturbations.

## 2. The disease, not the symptom (SIGReg step 1)

Interference is not a smooth scalar property of the merged weights; it is **concentrated on a tiny minority of decode steps** where the composed model's preferred *next-token SET* leaves the frozen base's preferred set. On those "conflict tokens" the adapter has pushed the distribution off the base manifold into a region the base never endorses — a no-man's-land. On the vast majority of steps (including almost all on-domain GSM8K steps) the composed top-k and base top-k **overlap heavily**, so deferring to the base there is free.

## 3. Constructive definition (sign-free, discrete)

Per decode step *t*, with frozen-base logits `ℓ^B_t` and composed (base ⊕ math) logits `ℓ^C_t` on the **same context**:

- `S^B_t = top-k(ℓ^B_t)`, `S^C_t = top-k(ℓ^C_t)`  (k = 8, index sets).
- `J_t = |S^B_t ∩ S^C_t| / |S^B_t ∪ S^C_t|`  (Jaccard ∈ [0,1]).

**Gate (zero parameters beyond τ):**
```
emit  argmax(ℓ^B_t)   if  J_t < τ     # CONFLICT → base safe harbor
emit  argmax(ℓ^C_t)   otherwise        # AGREEMENT → composed (math) token
```
This is a **set-disagreement** signal (TIES sign-conflict localization, arxiv:2311.03099), not a magnitude. The action is **fall back to the base token**, never route among adapters (F#605–608 routing kills avoided). Composition is `Σ (B_i @ A_i)` applied per-layer inside the model forward (single adapter here, N=1, so just `B@A`), `mx.argmax` selects the emitted id.

## 4. Quantitative predictions

Let `base`, `fixed` (always composed), `gated@τ` (the rule above) be the three policies.

- `drop_fixed = HE_base − HE_fixed > 0` (F#627: ≈ +10–12 pp drop expected).
- `lift_fixed = GSM_fixed − GSM_base > 0` (F#627: ≈ +15–22 pp lift expected).
- **Prediction P1:** on-domain conflict rate is low → `J_t ≥ τ` on most GSM8K steps → `gated` retains most lift: `retention = lift_gated / lift_fixed ≥ 0.80`.
- **Prediction P2:** off-domain conflict rate is higher and concentrated → reverting those steps to base recovers most of the HumanEval drop: `interference_reduction = (drop_fixed − drop_gated) / drop_fixed ≥ 0.60`.
- **Prediction P3 (mechanism):** mean per-step conflict rate `(1−J̄)` on HumanEval > on GSM8K (interference is localized off-domain). Reported, not gating.

## 5. Pre-registered kill criteria (target-metric, F#666 compliant)

τ swept over {0.20, 0.35, 0.50, 0.65, 0.80}. The **best-retention-feasible** τ is the one with maximal `interference_reduction` subject to `retention ≥ 0.80`; if none feasible, take max `interference_reduction`. KCs evaluated at that τ.

- **K1 (kill id 2292, off-domain TARGET — HumanEval pass@1):**
  `interference_reduction ≥ 0.60`. FAIL → contributes to KILL.
- **K2 (kill id 2292, on-domain TARGET — GSM8K exact-match):**
  `retention ≥ 0.80`. FAIL → contributes to KILL.

**Verdict rule (kill id 2292 is the conjunction):**
`SUPPORTED` iff at the selected τ **both** K1 and K2 PASS. Otherwise `KILLED`.
Both KCs are behavioral task-accuracy targets (no proxy-only KC). `is_smoke: false`.

**Sample size:** n_humaneval, n_gsm8k = 40 each (matches sibling; documented, fits <2h on M5 Pro). Lockstep base+composed decode, greedy, `enable_thinking=True` (F#530 — truncating thinking zeros base accuracy).

## 6. References
- F#627 — solo math adapter on/off-domain deltas (the adapter used here).
- F#827 — interference is real and behavioral.
- arxiv:2311.03099 (TIES-Merging) — sign/set-conflict localization motivates the discrete gate.
- `exp_spark_entropy_gated_lora` — the continuous-entropy sibling this *inverts*; reuses its loader/eval infra.
