# PAPER — adapter delta as a read-only head compass

## Verdict: KILLED (kill-2310 fired, both sub-clauses)

## Claim under test
The net-negative math LoRA on `self_attn.q_proj` of gemma-4-e4b-it-4bit, never applied to the forward
pass, still encodes (via its B-rowspace output energy per 256-dim head block) which frozen base q_proj
heads matter. Amplifying those base heads (`q' = q ⊙ m`, `m=γ` on selected head blocks, delta never
enters the residual stream) was predicted to beat both base and a matched-count random-head baseline by
≥ +4pp at the best γ.

## Setup (real, measured — `is_smoke:false`)
- Model: `mlx-community/gemma-4-e4b-it-4bit`; adapter `data/adapters/math/adapters.safetensors` (rank 6, scale 6.0).
- GSM8K EM, n_eval = 80 (≥60 quorum met), seed 42, random arm seed 1234, top_k_heads = 12, γ ∈ {1.1, 1.2, 1.3, 1.5}.
- Refuting context arm = delta applied to logits at adapter_scale 6.0. Total run 8256.3 s.

## Results — all arms, GSM8K EM (%)

| Arm | γ=1.1 | γ=1.2 | γ=1.3 | γ=1.5 | best |
|-----|-------|-------|-------|-------|------|
| base (no intervention) | 46.25 | 46.25 | 46.25 | 46.25 | 46.25 |
| compass-amplify | 50.00 | 48.75 | 50.00 | 50.00 | **50.00** (γ=1.1) |
| random-amplify (matched count/factor) | 50.00 | 50.00 | 50.00 | 50.00 | **50.00** |
| delta-applied (labeled refuting context) | — | — | — | — | **70.00** |

## Prediction vs measurement

| Quantity | Predicted | Measured |
|----------|-----------|----------|
| compass − base margin | ≥ +4.0 pp (≈ +5–8 pp) | **+3.75 pp** |
| compass − random margin | ≥ +4.0 pp | **+0.00 pp** |
| best arm | compass-amplify | **delta-applied (70.0)** |

## Which clause fired
Both sub-clauses of kill-2310 fired:
- Clause (a): compass − base = 50.00 − 46.25 = **+3.75 pp < +4.0** → fails the base margin.
- Clause (b): compass − random = 50.00 − 50.00 = **+0.00 pp < +4.0** → fails the random margin; random heads
  matched compass exactly, so the compass carries no head-selection signal beyond chance.
- Second-clause refutation: `delta_is_best_arm = true`. The labeled delta-applied arm (70.0) is the best
  arm overall, which is itself an independent refutation per the pre-registered clause "OR best result
  requires applying the adapter delta to logits at all."

`all_pass = false`, `kill_2310_result = "killed"`, `verdict = "killed"`.

## Interpretation
The +3.75 pp compass gain over base is fully explained by the generic act of up-weighting *any* small set
of q_proj heads at small γ — random selection produced the identical 50.0 EM. The adapter's per-head
output-energy ranking therefore provides no usable "compass" direction. Moreover the delta, when actually
applied, is the strongest arm (70.0 vs 46.25 base), contradicting the framing premise that the delta is
net-negative; the value lives in applying Δ, not in reading it as a selector. Hypothesis rejected.

## Verdict line
VERDICT: KILLED — compass−base = +3.75 pp (< +4.0), compass−random = +0.00 pp (< +4.0), and delta_is_best_arm = true (delta-applied 70.0 is best). Both kill-2310 clauses fired.
