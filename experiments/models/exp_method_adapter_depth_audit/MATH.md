# MATH.md — exp_method_adapter_depth_audit

## Target
Platform: MLX on Apple M5 Pro 48 GB
Base model: `mlx-community/gemma-4-e4b-it-4bit` (42 transformer layers, L_total = 42 — verified at runtime)
`mlx-lm` version in use: 0.28+ (per PLAN.md Part 2)

## Theorem (Todd et al., Function Vectors, arxiv:2310.15213)

Let `π_0` be a frozen LLM with L transformer blocks and let `h_ℓ(x) ∈ ℝ^d` be the residual stream at layer `ℓ` after processing input `x`. Todd et al. showed that for **procedural/function-style** tasks, the causal effect of activation interventions concentrates at **early–middle layers** (peak around `ℓ ≈ L/3 … L/2`). For **semantic/factual** tasks, the effect concentrates at **late layers** (`ℓ ≥ 2L/3`).

**Claim under test (C-depth):** if a LoRA adapter `Δ_r(θ)` encodes a *procedural/method* function (as claimed for `method_multi` from `exp_method_vs_domain_adapter`), then the weight-space adapter must insert its modifications at layers matching Todd's procedural band, i.e. support of `Δ_r` must overlap `[L/4, L/2]` ≈ layers `[10, 21]` for `L=42`.

Contrapositive (C-depth ¬): if an adapter's weight-space support is restricted to layers outside `[L/4, L/2]` (e.g. only `[L/2, L]`), then either:
(a) the adapter cannot encode a Todd-style procedural function (its effect must be semantic-band or mixed), OR
(b) Todd's activation-space theorem does not transport cleanly to LoRA weight-space interventions.

**Dependency state.** `method_multi` from `exp_method_vs_domain_adapter/adapters/method_multi/adapter_config.json` has:
```
num_layers: 16
keys: [self_attn.v_proj, self_attn.o_proj]
rank: 16, scale: 4.0
```
In `mlx-lm`, `num_layers=16` applies LoRA to the **last 16 layers** of the base. For `L=42` (verified at runtime), support of `Δ_r` is layers `[26, 41]` — **strictly outside** Todd's procedural band `[10, 21]` and **entirely inside** the semantic band `[2L/3, L] = [28, 41]` (with a narrow `[26, 27]` straddle). The adapter has zero weight at layers 0-25.

## Kill criteria (pre-registered, frozen)

These were filed at experiment creation. I do not modify them; instead I evaluate each against the current dependency geometry.

- **K1727**: "Ablating method adapter at layers 0-8 causes >=2x drop in procedural benchmark (BBH/MATH) vs ablating layers 24-32."
- **K1728**: "Domain adapter ablation shows opposite pattern (later layers more causal) — validates method/fact layer split."
- **K1729**: "Residual stream intervention at early-middle layers reproduces method effect within 1pp."

## Measurability analysis (pre-run)

K1727 is **structurally unmeasurable** with the only available candidate adapter: `method_multi` has `support(Δ_r) ∩ [0,8] = ∅` (layers 0-8 have zero LoRA weights; enumerated support is layers `[26, 27, 28, …, 41]`). "Ablating" LoRA at layers 0-8 multiplies zero by zero — a degenerate intervention that cannot produce ≥2× effect by construction. A measurable K1727 requires either:
  (i) retraining a method adapter with `num_layers=-1` (all 42 layers), or
  (ii) reformulating K1727 over the adapter's actual support band (e.g. `[26,33]` vs `[34,41]`) — but this is **KC reformulation**, which is the `kc_swap_after_failure` antipattern (PLAN.md §1).

K1728 is **unmeasurable**: no validated domain adapter exists.
  - `exp_prompt_erasure_gemma4` (domain/knowledge candidate) is KILLED (Finding #588, MMLU −30pp, TriviaQA −45pp).
  - `exp_knowledge_disentanglement_control` is KILLED (all KCs fail, catastrophic ΔMMLU=−30pp).
  - `exp_method_vs_domain_adapter` did not ship a domain-only sibling; only `method_multi` + `method_single_math` adapters were trained.
  K1728 requires a validated *domain* adapter paired with a validated *method* adapter; neither exists.

K1729 is **unmeasurable at macro scope**: a residual-stream intervention harness for Gemma-4-4bit has not been built in this repo and cannot be implemented in a single hat iteration (requires per-layer hook infrastructure + calibration on a known function vector — multi-day work per Todd §3).

## Decision

Per PLAN.md §1 (kill-criteria discipline) and the antipattern catalog:
> "If the KC needs to change, design a v2 experiment with the new KC; don't edit the old one in place."

The pre-registered KCs are **structurally incompatible** with the available dependency artifacts. I do not silently reformulate them. I do not invent a domain adapter or a residual-stream harness out of scope.

**Verdict:** K1727, K1728, K1729 all fail the measurability pre-flight. This experiment is KILLED on dependency; v2 follow-up should pre-register layer bands over the adapter's actual support AND require a validated method+domain adapter pair before claiming.

## Predictions → measurements

| KC    | Prediction (if measurable) | Measurement | Verdict |
|-------|---------------------------|-------------|---------|
| K1727 | ≥2× procedural drop at layers 0-8 vs 24-32 | Adapter support is `[16,31]`; layers 0-8 region has LoRA norm = 0 | FAIL (structurally unmeasurable) |
| K1728 | Opposite (later more causal) for domain adapter | No validated domain adapter exists | FAIL (prerequisite missing) |
| K1729 | Residual-stream intervention reproduces effect within 1pp | No harness; out of scope | FAIL (prerequisite missing) |

## Assumptions (autonomous decisions, per researcher hat)

- A1: I treat `num_layers=16` in mlx-lm as "LoRA on the last 16 layers" (standard mlx-lm semantics, verified by source: `mlx_lm.tuner.trainer.linear_to_lora_layers` applies to `model.layers[-num_layers:]`).
- A2: Gemma-4-E4B layer count is verified **empirically at runtime** to be L=42 (the script loads the model and reports `len(model.model.layers)`).
- A3: I do NOT retrain a new method adapter or train a domain adapter in this iteration — that is v2 work and out of scope for a single hat activation.
- A4: I do NOT build a residual-stream intervention harness — multi-day work, explicit scope violation.
- A5: I do NOT reformulate the pre-registered KCs; the `kc_swap_after_failure` antipattern is the single most cited failure in the audit catalog.

## v2 design guidance (for the analyst / future researcher)

If the lab wants to actually test Todd's depth hypothesis on Pierre adapters, the v2 experiment must:
1. Train a **fresh** method adapter with `num_layers=-1` (all 42 Gemma-4 layers) so every band is covered.
2. Train a **paired domain adapter** with the same `num_layers=-1` and a validated knowledge-retention K (ΔMMLU within ±2pp).
3. Pre-register KCs over **bands that actually overlap adapter support**: e.g. `[0,10]`, `[10,21]`, `[21,31]`, `[31,41]` (quartiles of L=42).
4. Ablate each band by zero-masking `lora_A[ℓ], lora_B[ℓ]` for `ℓ` in the band, not by activation hooks (weight-space test, not activation-space — this matches the claim "method adapter sits in the right depth band", i.e. about weight support, not residual-stream intervention).
5. Use a procedural benchmark that doesn't collapse to MCQ pattern-match (BBH causal-judgement or MATH level-1 with numeric verifier).
