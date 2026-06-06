# MATH.md: MMLU-Pro Baseline + Pierre Adapted

## Type: Verification

## Context

MMLU-Pro (Wang et al., 2024; arXiv:2406.01574) extends MMLU to 10-option MCQ across
14 domains with 12,032 test questions. Google reports 69.4% for Gemma 4 E4B (full precision,
with thinking enabled). We run on 4-bit quantized (mlx-community/gemma-4-e4b-it-4bit)
with thinking disabled (`enable_thinking=false`) for tractable runtime.
Disabling thinking typically loses 5-15pp on reasoning-heavy benchmarks.

## Theorem 1: Quantization Accuracy Bound

**Claim:** 4-bit GPTQ/AWQ quantization of an LLM with accuracy $a$ on MCQ yields
accuracy $\hat{a}$ satisfying $|a - \hat{a}| \leq \epsilon_q$ where $\epsilon_q \leq 3\text{pp}$
for well-calibrated 4-bit schemes on models > 1B parameters.

**Prior:** Dettmers et al. (arXiv:2208.07339) show 4-bit quantization loses < 1pp on MMLU
for >3B models. Frantar et al. GPTQ (arXiv:2210.17323) show < 2pp loss. MLX community
quantization uses group-wise 4-bit with calibration.

**Prediction (with thinking):** Base 4-bit accuracy = $69.4 - \epsilon_q$ where $\epsilon_q \in [1, 3]$,
giving predicted range **66.4% -- 68.4%**.

**Prediction (without thinking):** Disabling thinking loses $\epsilon_t \in [5, 15]\text{pp}$
on reasoning-heavy MCQ. Predicted: $69.4 - \epsilon_q - \epsilon_t \approx 54 - 63\%$.
K1 threshold (within 5pp of 69.4% = 64.4%) may fail if $\epsilon_t > 5$.

## Theorem 2: Single-Adapter Perturbation on Multi-Domain MCQ

**Setup:** Adapter trained on domain $d$ covering fraction $f_d$ of benchmark questions.
Adapter improves domain accuracy by $\Delta_d$ and perturbs non-domain by $\delta_{nd}$.

**Bound:** Overall accuracy change:
$$\Delta_{\text{total}} = f_d \cdot \Delta_d + (1 - f_d) \cdot \delta_{nd}$$

**For math adapter on MMLU-Pro:**
- $f_{\text{math}} \approx 860/12032 \approx 0.071$ (math subtask fraction)
- $\Delta_{\text{math}}$: adapter scored 82% on MMLU math vs 0% base on T2.1.
  On MMLU-Pro math (harder, 10 options), conservatively estimate $\Delta_{\text{math}} \in [10, 30]\text{pp}$.
- $\delta_{nd}$: single q_proj LoRA r=6 perturbation on 2560-d model.
  Prior (Finding #44): individual adapters are net positive on all in-domain, near-neutral OOD.
  Estimate $\delta_{nd} \in [-2, 0]\text{pp}$.

**Prediction:** $\Delta_{\text{total}} = 0.071 \times 20 + 0.929 \times (-1) \approx +0.5\text{pp}$
Range: $[0.071 \times 10 - 0.929 \times 2, 0.071 \times 30 + 0] = [-1.1, +2.1]\text{pp}$

**K2 assessment:** The +2pp threshold is at the optimistic edge. K2 will likely FAIL
for a single adapter. This is expected -- a single-domain adapter covers only 7% of questions.
K2 would require either (a) composition of all 5 adapters, or (b) a single adapter
covering a much larger fraction of MMLU-Pro.

## Predictions

| Criterion | Prediction (no thinking) | Range |
|-----------|--------------------------|-------|
| K1: Base accuracy | 58% | [54, 63] |
| K2: Adapted - base | +0.5pp | [-1.1, +2.1] |
| K3: Runtime | ~0.6h (2 runs, 100/subtask, no thinking) | [0.4, 1.0]h |

## Kill Criteria Mapping

- **K1411 (Base within 5pp of 69.4%):** FAIL predicted without thinking (58% is 11pp below 69.4%).
  To PASS, would need thinking enabled. But this still validates our eval pipeline.
- **K1412 (Adapted >= base + 2pp):** FAIL predicted. Single math adapter covers < 10% of questions.
- **K1413 (< 6h):** PASS predicted. ~0.6h for 2 runs without thinking.

## Failure Mode

If K1 FAILS (base < 64.4%), the eval setup is broken (wrong task, parsing errors,
prompt format mismatch). Diagnose before proceeding to Phase 2.

If K2 FAILS as predicted, this establishes: **single-domain adapter cannot lift broad
benchmark by 2pp when it covers < 10% of questions.** This motivates full N=5 composition
benchmark as the next experiment.
