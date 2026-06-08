# PAPER — Off-domain LoRA interference is NOT temporally localized to high-base-entropy choice points

**Verdict: KILLED.** The temporal-localization reframe is refuted. Entropy-gating the
off-domain code adapter at the top-5% highest-base-entropy decode steps recovers essentially
none of the on-domain accuracy lost to composition, and does no better than gating an equal
number of *random* steps. `is_smoke: false`.

## Setup (real, verified)
- Frozen base: `mlx-community/gemma-4-e4b-it-4bit`, mlx-lm 0.31.2.
- Math adapter (on-domain): `data/adapters/math/adapters.safetensors` (r=6, scale 6.0, q_proj×42).
- Code adapter (off-domain): `experiments/models/exp_composition_residual_analysis/adapter_code.safetensors` (same recipe, distinct weights).
- 80 real GSM8K test items, greedy decode, `enable_thinking=True`, max 512 tokens, identical prompts/seed across all 5 arms.
- Composition: `Σ scale·(Bᵢ@Aᵢ)` (math + code summed in activation space), scale 6.0 ≤ 8. Gate verified to change logits/argmax.

## Arms (accuracy on the same 80 items)
| Arm | Description | Accuracy |
|---|---|---|
| A base | frozen Gemma-4, no adapter | 11.2% |
| B math-only | on-domain ceiling | **66.2%** |
| C compose | math + code, all steps (interfered) | 46.2% |
| D entropy-gate | math + code, code zeroed at top-5% high-base-entropy steps | 47.5% |
| E random-gate | math + code, code zeroed at equal # of RANDOM steps | 47.5% |

## Prediction vs measurement
| Quantity | Predicted (MATH.md) | Measured | KC | Result |
|---|---|---|---|---|
| B − C drop (premise) | 12–14 pp | **20.0 pp** | premise | reproduced ✓ (larger than predicted) |
| recov(D) = (D−C)/(B−C) | ≥ 0.50 | **0.063** | K1 (2288) | **FAIL** |
| acc_D − acc_E | ≥ 2 pp | **0.0 pp** | K2 (2289) | **FAIL** |
| gated fraction | ≤ 0.05 | 0.0495 | K3 (2290) | pass |

`all_pass = False` → **KILLED** (K1 ∧ K2 ∧ K3 required for SUPPORTED).

## Interpretation
1. **The premise holds, strongly.** Composing the off-domain code adapter onto the math-adapted
   model costs 20 pp of GSM8K accuracy (66.2 → 46.2). Interference is real and behavioral.
2. **It is not temporally concentrated on high-base-entropy steps.** Zeroing the code adapter
   at exactly those steps recovers only 1.25 pp (6.3% of the loss) — nowhere near the ≥50% the
   reframe predicts. The +50pp positive-transfer license from F#827 does not translate into a
   sparse decode-position structure that a tiny gate can exploit.
3. **The load-bearing control (K2) kills it cleanly.** The entropy gate does *exactly* as well
   as gating the same number of random steps (Δ = 0.0 pp). Whatever marginal effect the gate
   has is indistinguishable from dropout — it is **not choice-point-specific**. This is the
   decisive evidence: the interference is not indexed by base entropy / decode position.
4. K3 passes only trivially (the gate touched ≤5% by construction); with K1/K2 failing, the
   concentration question is moot.

## Honest caveats
- Adapters target `q_proj` only (r=6); a different injection site or rank could in principle
  carry a different temporal signature, but the q_proj recipe is the project's standard and the
  one that produced the 20pp interference being explained.
- Top-5% is per-sequence rank-based; widening the gate would only move toward the K3>15%
  rejection region, not rescue K1/K2 (recovery is flat in the gated set, and random matches it).
- The base-entropy profile is computed on the composed arm's own trajectory (non-circular: base
  logits, no adapters); flips that the gate must catch occur off this trajectory too, but the
  random-control equivalence shows the high-entropy positions carry no special interference load.

## Conclusion
Off-domain LoRA interference on frozen Gemma-4 is a **real 20pp behavioral effect that is NOT
a sparse temporal event localized to high-base-entropy choice-point tokens**. The axis-relocation
hypothesis ("which adapter" → "decode-step, gated by base entropy") is rejected. Interference
remains, as prior weight-space work found, a distributed property — not removable by a 5%
decode-position gate. Clean kill.
