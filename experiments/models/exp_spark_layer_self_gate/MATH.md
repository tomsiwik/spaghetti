# MATH — exp_spark_layer_self_gate

## Frame
Two LoRA adapters on a frozen base `gemma-4-e4b-it-4bit` (mlx-community 4bit), both q_proj only,
rank 6, scale 6.0, on all 42 decoder layers:
- code adapter `(A_c^ℓ, B_c^ℓ)` — the on-task adapter for HumanEval,
- math adapter `(A_m^ℓ, B_m^ℓ)` — the off-task *interferer*.

Composition is `Σ_i (B_i @ A_i)` per layer (method rule), never `(ΣB)(ΣA)`. At layer ℓ, for hidden
input `x^ℓ`, the q_proj output under naive full composition C is
```
q_C^ℓ(x) = W_q^ℓ x  +  s·(B_c^ℓ A_c^ℓ) x  +  s·(B_m^ℓ A_m^ℓ) x ,   s = 6.0
```
Define the **base+code** output `q_B^ℓ(x) = W_q^ℓ x + s·(B_c^ℓ A_c^ℓ) x` (condition B, code-solo) and
the **math per-layer delta** `δ_m^ℓ(x) = s·(B_m^ℓ A_m^ℓ) x`. Then `q_C^ℓ = q_B^ℓ + δ_m^ℓ`.

## Definitions
Let the prompt have token hidden states `{x_t^ℓ}` arriving at layer ℓ during a single FREE forward pass
of the **code-only** model on the prompt (no decoding). Per layer define the prompt-mean cosine between
the math delta and the code-active base output, over prompt token positions t:
```
γ^ℓ  =  mean_t  cos( δ_m^ℓ(x_t^ℓ) ,  q_B^ℓ(x_t^ℓ) )
     =  mean_t  ⟨δ_m^ℓ(x_t^ℓ), q_B^ℓ(x_t^ℓ)⟩ / (‖δ_m^ℓ(x_t^ℓ)‖·‖q_B^ℓ(x_t^ℓ)‖)
```
`γ^ℓ > 0` ⇒ math delta is **constructive** (aligns with the code-active representation) at depth ℓ;
`γ^ℓ < 0` ⇒ **destructive** (it cancels / rotates against the code-active signal). This statistic is
computed from ONE prompt forward pass — no router, no decode-time gating, no training.

**Layer-self-gate mask (condition D, parameter k):** keep the math adapter active only at the k layers
with the largest `γ^ℓ` (top-k constructive), zero the math delta at the other 42−k layers. Code stays in
all 42 layers always. k is swept; D collapses to C at k=42.

## Theorem (layer-localized interference ⇒ recoverable by free mask)
Decompose the off-task damage at layer ℓ relative to code-solo into a constructive part and a destructive
part. Model the projection of `δ_m^ℓ` onto the code-active output as the first-order term that moves the
logits; writing the per-layer signed contribution `c^ℓ = sign(γ^ℓ)·‖δ_m^ℓ‖_proj`, the total math
perturbation to the code task is `Δ = Σ_ℓ c^ℓ`. Split into `Δ⁺ = Σ_{γ^ℓ>0} c^ℓ ≥ 0` (helpful) and
`Δ⁻ = Σ_{γ^ℓ<0} c^ℓ ≤ 0` (harmful).

**Claim.** If interference is *layer-localized* — i.e. `Δ⁻` concentrates in a strict subset of layers and
its sign is reliably reported per-prompt by `γ^ℓ < 0` — then zeroing the negative-γ layers removes `Δ⁻`
while retaining `Δ⁺`, so a top-k-constructive mask satisfies, at the k* that excludes all destructive
layers,
```
pass@1(D, k*)  ≈  pass@1(B)  +  (Δ⁺ contribution)  ≥  pass@1(C) + |Δ⁻ contribution| .
```
Because `pass@1(C) = pass@1(B) + Δ⁺ + Δ⁻` and the mask drops `Δ⁻`, the recovery is
`pass@1(D,k*) − pass@1(C) ≈ −Δ⁻ ≥ 0`. Layer-localization predicts this gap is large (≥ +8pp).

**QED (conditional):** the recovery equals exactly the harmful mass `−Δ⁻` removed, *iff* `γ^ℓ` is a faithful
per-layer sign oracle and the harm is concentrated. Both are the empirical content tested below.

## Null hypotheses (what refutes the frame)
- **N1 (uniform interference):** `Δ⁻` is spread ~evenly across all 42 layers and/or `γ^ℓ` does not track
  the per-layer sign of harm. Then masking either removes proportional helpful+harmful mass (no net gain)
  or removes mostly helpful mass (hurts). Prediction under N1: `pass@1(D,k) − pass@1(C) < +8pp` for all k.
- **N2 (no interference to recover):** if the in-run C shows `pass@1(C) ≥ pass@1(B)` (math is net helpful
  to code here, as seen in exp_spark_entropy_gated_lora where fixed math gave +5pp on HumanEval), there is
  almost no `Δ⁻` to remove; the recovery ceiling is small by construction. The kill is anchored to in-run C
  precisely so this case yields KILLED (correctly: the layer-localization frame buys nothing when there is
  no localized damage), not a false SUPPORTED off F#827 magnitudes.

## Pre-registered prediction & kill (DB id 2296, anchored to IN-RUN C)
Conditions, HumanEval pass@1, n=50, greedy, thinking enabled:
- A base only; B code-solo (ceiling); C naive full-layer composition (code all 42 + math all 42) — the
  in-run interference baseline and the kill anchor; D layer-self-gate, sweep k ∈ {6,12,18,24,30,36}.

Let `best_D = max_k pass@1(D,k)`.

**SUPPORTED iff** `best_D − pass@1(C) ≥ +8pp`  AND  `best_D ≥ pass@1(B) − 6pp`.
**KILLED iff** `best_D − pass@1(C) < +8pp`  OR  `best_D < pass@1(B) − 6pp`.

Refutation threshold (the number we accept before the run): a +8pp recovery vs the in-run measured C at
n=50. No F#827 magnitude is assumed; if C does not show interference, the +8pp gap cannot be met and the
frame is KILLED — by design.

Secondary (characterization, not gating): GSM8K exact-match n=50 for A/B/C and D at best_D's k, to report
the on-domain cost of masking the math adapter.
