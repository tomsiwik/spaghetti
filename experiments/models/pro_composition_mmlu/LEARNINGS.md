# LEARNINGS: Pierre Pro MMLU Composition (Finding #320)

## Core Finding

**Composition via NRE on fp16 base (Qwen3-4B-4bit) preserves MMLU with ZERO degradation
at scale≤5, confirming the spectral gap hypothesis. Ternary's -5.5pp degradation (Finding #272)
was caused by flat singular spectrum, not the composition mechanism. However, the adapters
trained at scale=20 catastrophically destroy MMLU (-60pp single, -44pp composed), revealing
scale calibration as the unsolved bottleneck.**

## Why This Happened

The Davis-Kahan sin-theta theorem predicts that eigenspace stability under perturbation
is inversely proportional to the spectral gap. Ternary models have near-zero gap
(ratio 1.003-1.018, Finding #272), meaning any low-rank perturbation disrupts knowledge.
Production-trained fp16 models have steep spectra with large gaps, so the same rank-16
perturbation leaves the knowledge subspace essentially untouched at low scale.

The scale=20 catastrophe is a separate phenomenon: the adapter perturbation norm
||scale × A × B^T|| grows linearly with scale. When it exceeds the spectral gap,
the Davis-Kahan bound becomes vacuous and knowledge collapses. The phase transition
at scale~10 is where ||ΔW||₂ ≈ δ (the gap).

## Confirming Evidence

- **Biderman et al. (2401.05605):** "Scaling Laws for Forgetting When Fine-Tuning LLMs."
  Demonstrates inverse linear relationship between fine-tuning performance and forgetting.
  Forgetting cannot be avoided through early stopping or parameter count — it's structural.
  Directly confirms our finding that scale=20 adaptation necessarily destroys base knowledge.

- **Biderman et al. (2405.09673):** "LoRA Learns Less and Forgets Less."
  LoRA underperforms full fine-tuning but also forgets less. The scaling regime (rank/alpha)
  directly bounds adapter expressivity vs knowledge preservation. Our scale sweep is the
  extreme version: scale=1 preserves everything but does nothing; scale=20 does everything
  but destroys everything.

- **rsLoRA (2312.03732):** "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA."
  Proves the conventional α/r scaling stunts learning at higher ranks; α/√r stabilizes
  gradient norms. Foundational result showing the scaling factor is not a neutral hyperparameter
  — it directly controls effective learning rate and perturbation magnitude. Our scale=20
  training inherited an uncalibrated scale from BitNet where different magnitudes applied.

- **Finding #272:** Ternary flat spectrum (gap ratio 1.003-1.018) as root cause of MMLU
  degradation. This experiment CONFIRMS the spectral gap hypothesis by testing the
  contrapositive: non-flat spectrum → no degradation.

- **Finding #263:** BitNet-2B MMLU degradation -5 to -6pp under composition, the original
  observation that motivated this experiment.

## Contradicting Evidence

- **LoRA Soups (2410.13025):** Concatenating independently trained LoRAs outperforms data
  mixing by 43% on skill composition tasks with no additional training. Suggests composition
  can work at training scale — but uses concatenation (different adapter per task), not
  simultaneous merging. The key difference: they don't compose into a single weight matrix.

- **LoRI (2504.07448):** "Reducing Cross-Task Interference in Multi-Task LoRA." Frozen
  random A projections + sparse B masks achieve composition without degradation. Partially
  contradicts "spectral gap is the key" — their orthogonality constraint works regardless
  of base model spectrum. However, their approach is architecturally similar to our
  Grassmannian frozen-A design.

- **Kobalyan et al. (2502.14502):** "How Much Knowledge Can You Pack into a LoRA Adapter
  without Harming LLM?" Shows LoRA CAN integrate new knowledge when training data mixes
  known and new facts. Degradation is contingent on data composition, not structural.
  Weakens the "scaling always destroys knowledge" narrative — but operates at fixed scale,
  not across scale sweep.

## Alternative Approaches (for scale calibration problem)

1. **DoRA (2402.09353):** Weight-Decomposed Low-Rank Adaptation. Decomposes weights into
   magnitude + direction, applies LoRA only to directional component. Magnitude stays
   anchored to pre-trained values, preventing scale drift. Most direct architectural
   solution to our scale problem — adapter cannot corrupt base model magnitude by design.

2. **LoRA-Null (2503.02659):** Initializes LoRA in null space of input activations.
   Since null space contains minimal pre-trained information, adapter starts in a region
   that cannot destructively interfere. Complementary to our Grassmannian init (which
   orthogonalizes A-matrices to each other, not to the base model's knowledge space).

3. **Self-Distillation Fine-Tuning / SDFT (2601.19897):** Model generates its own training
   data, guaranteeing data quality ≥ base capability. Already recommended from Finding #319
   analysis. Would solve BOTH scale calibration (train at scale=1 with model-quality data)
   and data quality problems simultaneously.

4. **rsLoRA scaling (2312.03732):** Replace α/r with α/√r. Simple fix that may allow
   training at lower effective scale while maintaining gradient signal. Could be applied
   without architectural changes.

## Implications for Next Experiments

1. **Scale calibration is THE blocker.** The composition mechanism works. The spectral gap
   protects knowledge. But adapters trained at scale=20 need scale=20 to produce domain
   output, and scale=20 destroys MMLU. Every future experiment must address this.

2. **Two paths forward:**
   - **Train at low scale with high-quality data** (SDFT + scale=1). If data quality ≥ base,
     the adapter learns meaningful signal even at scale=1. This combines Findings #319 and #320.
   - **DoRA-style magnitude anchoring.** Prevents scale from corrupting base representations
     by construction. More architectural change but a structural guarantee.

3. **The ternary bottleneck is definitively resolved.** Finding #272 identified the disease
   (flat spectrum). Finding #320 confirms the cure (non-flat spectrum → no degradation).
   Pierre Pro on Qwen3-4B is the right direction.

4. **Composition actually helps.** At every scale, composed adapters degrade LESS than the
   worst single adapter. NRE averaging reduces perturbation magnitude. This is a free benefit
   of the composition architecture.

5. **50Q MMLU is sufficient for directional findings** (0pp vs -60pp is unambiguous) but
   insufficient for precise thresholds. Scale boundary detection requires 200+ questions.

## Recommended Follow-Up

**exp_pro_self_distill_adapters (P0):** Train domain adapters using model-generated data
with `<think>` reasoning preserved, at scale=1.
- **Motivation:** Finding #319 (data quality regression) + Finding #320 (scale=20 catastrophe)
- **Literature:** SDFT (2601.19897), Qwen3 report (2505.09388), SPIN (2401.01335)
- **Why it fixes the failure:** Model-generated data guarantees quality ≥ base capability
  (Finding #319 fix). Training at scale=1 keeps perturbation within spectral gap protection
  zone (Finding #320 fix). Solves both problems simultaneously.

**exp_pro_dora_composition (P1):** Implement DoRA-style magnitude-direction decomposition
on Qwen3-4B with Grassmannian skeleton.
- **Motivation:** Finding #320 (scale=20 magnitude overwhelms base)
- **Literature:** DoRA (2402.09353, ICML 2024 oral)
- **Why it fixes the failure:** Magnitude anchoring prevents adapter from corrupting base
  model knowledge representations regardless of scale. Structural guarantee, not a
  hyperparameter tuning.
