# MATH.md — Off-domain LoRA interference as a low-entropy-token artifact

**Experiment:** `exp_spark_entropy_gated_lora`
**Type:** Guided exploration (proven LoRA mechanics; unknown = whether interference is entropy-indexed).
**Platform:** Apple M5 Pro 48GB, MLX. `mlx-lm == 0.31.2`.
**Base model:** `mlx-community/gemma-4-e4b-it-4bit` (frozen, 4-bit, 42 blocks).
**Adapter:** math LoRA at `experiments/models/exp_p1_t2_single_domain_training/adapters/math/adapters.safetensors`
— rank 6, scale 6.0, target `self_attn.q_proj` on all 42 layers (84 tensors: `lora_a (in,6)`, `lora_b (6,out)`).

---

## 1. Failure mode being attacked

**Finding #827 (real, behavioral):** attaching the math LoRA to the frozen base drops HumanEval pass@1 by **−12 to −14pp** (off-domain code interference) while lifting GSM8K exact-match by **~+22pp** (on-domain). The whole program has treated this drop as a *static weight-space property* of the adapter (cross-term geometry, F#752; weight-orthogonality dead-ends F#822/823).

**Disease, not symptom (SIGReg step 1).** We claim the drop is NOT a property of the merged weights. It is a **per-decode-step dispatch error**: the math LoRA perturbs `q_proj` at *every* token, including the LOW-ENTROPY steps of a code generation (syntax `def`, `:`, `return`, indentation, identifiers the base already predicts with near-certainty). On those steps the frozen base is already correct; any nonzero perturbation can only flip a confident-correct token to a confident-wrong one. The expected damage of a perturbation is largest exactly where the base distribution is sharp and correct.

## 2. Prior math (no analogy)

- **LoRA forward (mlx-lm 0.31.2, exact):** `y = W₀x + s·(xA)B`, `s = 6.0`. Composition is the single-adapter delta `B@A` applied per token — there is no `(ΣB)(ΣA)` cross-product here (N=1 adapter), so the composition-bug antipattern does not apply.
- **First-order sensitivity of a softmax-correct token.** Let `z = logits`, `p = softmax(z)`, top-1 index `i*` correct. A logit perturbation `δ` from the adapter changes the chosen token only if some `j≠i*` overtakes `i*`: requires `δ_j − δ_{i*} > z_{i*} − z_j`. The margin `z_{i*} − z_j ≥ log(p_{i*}/p_j) ≥ log(p_{i*}/(1−p_{i*}))`, which **diverges as p_{i*}→1**. So at high base confidence `p_top1≈1`, no bounded `δ` flips the token; at low confidence `p_top1≈1/V`, a tiny `δ` flips it. **The adapter can only help where the base is uncertain, and can only hurt where the base is uncertain-but-different-from-math-prior.** Gating the perturbation magnitude DOWN as `p_top1→1` removes the regime where flips are free, and removes exactly the confident code-syntax tokens that drive the −12pp.
- **Entropy gate (this experiment):** scale per step `s(t) = s₀·(1 − p_top1_base(t))`, `s₀ = 6.0`, `p_top1_base(t) = max_v softmax(z_base(t))_v` computed from the **frozen base** (no adapter) on the actual decoded context. Bounded: `s(t) ∈ [0, 6.0]`, so `LORA_SCALE ≤ 8` is respected at every step. This is the confidence-gated-residual idea of **adaptive/uncertainty-gated inference** (cf. confidence-based early-exit / entropy gating, `arxiv:2207.07061` "Confident Adaptive Language Modeling"): act only where the base model is unsure.

## 3. Theorem (predicted behavior)

**Claim.** Code generation is dominated by high-`p_top1` (low-entropy) tokens where the base is correct; math reasoning contains many low-`p_top1` (uncertain) steps where the math prior helps. Therefore:

1. The entropy gate **zeroes the adapter on most code tokens** (confident syntax), so off-domain interference collapses: `drop_gated → 0`.
2. The gate **leaves the adapter near-full-strength on uncertain math steps**, so on-domain lift is largely retained: `lift_gated ≈ lift_fixed`.

If interference were instead a static weight-space property (null hypothesis), the gate — which only modulates *when* the same delta fires — could not selectively remove the code drop while keeping the math lift; both would scale together.

## 4. Quantitative predictions

Reference anchors: F#827 (fixed math adapter: HumanEval −12 to −14pp, GSM8K +~22pp); `arxiv:2207.07061` (confidence-gated computation).

| Quantity | Predicted |
|---|---|
| HumanEval drop, fixed-α (`base − fixed`) | ~12pp (reproduce F#827) |
| HumanEval drop, entropy-gated | ≤ 3pp |
| `interference_reduction = (drop_fixed − drop_gated)/drop_fixed` | ≥ 0.75 |
| GSM8K lift, fixed-α (`fixed − base`) | ~+22pp |
| GSM8K lift, entropy-gated | ≥ +17.6pp |
| `retention = lift_gated / lift_fixed` | ≥ 0.80 |

## 5. Pre-registered kill criteria (id 2291 — frozen, do NOT modify)

- **K1 (target metric: HumanEval pass@1):** `interference_reduction = (drop_fixed − drop_gated)/drop_fixed ≥ 0.75`. The ~−12pp fixed drop must collapse to `≤ 3pp` absolute under the gate. **ELSE KILL.**
- **K2 (target metric: GSM8K exact-match):** `retention = lift_gated / lift_fixed ≥ 0.80`. The +22pp fixed lift must retain `≥ 17.6pp`. **ELSE KILL.**

Both KCs are **behavioral target metrics** (HumanEval pass@1, GSM8K exact-match), satisfying Finding #666 (no proxy-only kills). Verdict = **SUPPORTED** iff K1 PASS **and** K2 PASS; else **KILLED**.

**Guards against degenerate-pass:** the gate trivially "removes interference" if it also kills the math lift (gate≈0 everywhere) — K2 blocks that. It trivially "retains lift" if it never gates (gate≈1 everywhere) — then `drop_gated≈drop_fixed` and K1 fails. The conjunction of K1∧K2 is only satisfiable if interference is genuinely entropy-indexed. We additionally log mean gate value on code vs math tokens as a mechanism check (not a KC).

## 6. Method (frozen base, no training)

Two frozen instances of `gemma-4-e4b-it-4bit`:
- **base_model**: no adapter — provides `p_top1_base(t)` and the `base` condition.
- **lora_model**: 84 `GatedLoRALinear` wrappers on `q_proj` (subclass `nn.Module` + `setattr` per memory mem-antipattern-call-override-silent-bypass; never override `__call__` on an instance).

Per-step gated decode (lockstep): at step `t`, run `base_model` forward on the current context → `z_base` → `p_top1 = max softmax(z_base)`. For the **fixed** condition set every wrapper's gate `g=1.0` (scale = 6.0). For the **gated** condition set `g = 1 − p_top1`. Run `lora_model` forward with that gate → choose greedy token → both models consume that token (shared KV cache advance). `enable_thinking=True`, greedy (argmax), chat template. Phased execution with `mx.clear_cache()` + `del`/`gc.collect()` between conditions.

`n`: HumanEval = 40 problems (off-domain target), GSM8K = 40 problems (on-domain target). Logged as actual `n`.
