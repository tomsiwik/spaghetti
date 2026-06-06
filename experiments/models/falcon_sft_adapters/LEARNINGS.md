# Learnings: exp_falcon_sft_adapters

## Core Finding

SFT loss (response-only masking) directionally improves composed adapter quality over NTP on 4/6 benchmarks, but the entire experiment is confounded by lora_scale=20 — a 10-20x overscaling relative to standard LoRA practice (alpha/r = 1.0-2.0). The lora_scale confound, not the training objective, is the most likely root cause of adapter degradation across multiple prior experiments. SFT masking alone is known to be insufficient to prevent catastrophic forgetting (math SFT: NLI drops 81% -> 16.5% in 1000 steps).

## Why This Happened

### 1. lora_scale=20 Is the Dominant Confound (Not SFT vs NTP)

Standard LoRA practice uses alpha/r as the scaling factor, typically alpha=r (scale=1.0) or alpha=2r (scale=2.0). Our experiments used MLX's raw `scale=20.0` — a 20x multiplier on the low-rank update. This is 10-20x outside the validated range in the LoRA literature.

The LoTA-QAF framework (arXiv:2407.11024) specifically warns that at low-bit precision (2-bit/ternary), weight updates are "highly volatile" because weights occupy only 4 possible values. They recommend *more conservative* scaling (larger omega threshold = lower effective scale) to prevent destructive weight updates. Our scale=20 does the exact opposite — maximally aggressive perturbation on a ternary base.

At scale=20:
- Individual adapter: 20x perturbation overwhelms base model knowledge (explains 5/6 MMLU degradation)
- 1/N composition with 5 adapters: effective 20/5 = 4x per adapter (still 2-4x standard, but regularized)
- This explains why composition "works" while individual adapters destroy: it's averaging-as-regularization, not routing-as-expertise

The code adapter outperforming the math adapter on GSM8K (0.62 vs 0.50) further supports this: at 20x scale, adapters are not learning domain-specific features but rather random perturbations whose beneficial effects are accidental. The reviewer flagged this anomaly as evidence that adapters at scale=20 may not be domain experts at all.

### 2. SFT Masking Is Necessary But Not Sufficient

The SFT property (zero gradient from instruction-token prediction) is mathematically correct but narrower than initially claimed. The revised MATH.md correctly notes: shared attention weights at response positions still receive gradients that flow through instruction-token attention. SFT eliminates one source of contamination (instruction-token prediction) but not all base-model perturbation.

The literature confirms this limitation explicitly. Research on specialized fine-tuning (cited in BitNet b1.58 2B4T and BabyLM studies) shows SFT causes catastrophic forgetting on general tasks even with response-only masking: math SFT improved accuracy 3.1% -> 12.0% but collapsed NLI from 81.0% -> 16.5% in just 1000 steps. Response-only masking cannot protect the base model's general knowledge from being overwritten by the gradient signal it does receive.

This means our "two diseases" hypothesis was directionally correct — NTP instruction contamination is real — but SFT is not a cure. It reduces one symptom while the underlying disease (LoRA perturbation magnitude at scale=20) remains.

### 3. Composition-as-Regularization Is the Actual Mechanism

The most important insight is NOT about SFT vs NTP. It's that 1/N uniform composition acts as implicit L2 regularization on adapter perturbation. Individual adapters at scale=20 are destructive (5/6 degrade). Composed adapters at effective scale=4 partially recover (2/6 degrade for SFT, 4/6 for NTP).

This is consistent with arXiv:2603.03535 (routing > merging at all scales): static merging's only advantage is as a regularizer, not as a knowledge combiner. The "composition works" result is an artifact of excessive scaling, not evidence that the architecture combines domain expertise.

### 4. Statistical Insignificance Undermines All Claims

GSM8K base->SFT composed: p=0.41 (not significant at any threshold).
GSM8K NTP->SFT composed: p=0.10 (marginal only).
All MMLU comparisons: n=20, far too small.

The directional observation (4/6 improve) is interesting but could be noise. At n=50, we'd need 0.44 vs 0.64 to reach p<0.05. The experiment lacks the statistical power to distinguish real effects from random variation.

## Confirming Evidence

1. **LoTA-QAF (arXiv:2407.11024)**: At 2-bit precision, conservative scaling (lower omega) achieves better accuracy. Destructive updates at aggressive scaling confirmed for low-bit models. Directly supports lora_scale=20 as root cause.
2. **LoRAuter (arXiv ref in SOLE)**: Uses rank=6, alpha=12 (scale=2.0). Standard practice is alpha=2r, not alpha=20r.
3. **Ouyang et al. 2022 (InstructGPT)**: SFT is Step 1 of RLHF, followed by reward modeling + PPO. SFT alone is not expected to produce optimal generation quality.
4. **Our Finding #166 (top2_output_space_falcon)**: ALL adapter methods worse than base on MMLU. "NTP adapter mismatch" was the diagnosis, but lora_scale=20 was present in that experiment too — and never ablated.
5. **Our Finding #179 (math 24x correctness)**: Math adapter works because GSM8K rewards any answer extraction, not because the adapter learned domain expertise. The format shortcut (concise `<<calc=result>>` vs verbose base) succeeds at scale=20 because it's a structural pattern, not factual knowledge.
6. **arXiv:2603.03535 (Routing > Merging at Scale)**: Static merging (our 1/N approach) is the weakest composition strategy at all tested scales.

## Contradicting Evidence

1. **BitNet b1.58 2B4T DPO study**: DPO after SFT successfully steered model toward preferred responses without significant capability degradation. Suggests SFT+DPO could be the right training pipeline if scale is fixed first.
2. **Our own GSM8K improvement (0.36 -> 0.52)**: Even at p=0.10, the direction is consistent. If lora_scale is the main confound, we'd expect both NTP and SFT to improve equally at scale=1. If SFT still outperforms NTP at standard scale, the training objective matters independently.
3. **Math catastrophic forgetting study**: SFT improved math 3.1% -> 12.0% despite NLI collapse. This means SFT CAN improve domain tasks — the question is whether the collateral damage is acceptable and whether it's scale-dependent.
4. **Our exp_continual_learning_adapter_growth**: Uniform composition maintains PPL within ~1% of base across N=5-15 on training-domain data. If scale=20 were purely destructive, in-distribution PPL should also degrade. The in-distribution vs out-of-distribution split suggests adapters DO learn something, just not something transferable to benchmarks.

## Alternative Approaches

### 1. lora_scale Ablation at {1, 2, 4, 8, 20} (HIGHEST PRIORITY)
Test both SFT and NTP adapters across standard scale values. This is the critical experiment that unconfounds all prior results.
- **Motivation:** Every adapter experiment since exp_falcon_e3b_composition used scale=20, inheriting it without ablation. All claims about SFT vs NTP, routing vs composition, and adapter quality are confounded.
- **Literature:** Hu et al. 2022 (LoRA) uses alpha/r = 1.0. LoTA-QAF (arXiv:2407.11024) shows conservative scaling is critical for low-bit models.
- **Prediction:** At scale=1-2, individual adapter MMLU degradation disappears. The SFT vs NTP distinction becomes either clear (SFT genuinely better) or moot (both work at standard scale).

### 2. DPO Training After SFT (If Scale Ablation Shows SFT Insufficient)
DPO (arXiv:2310.16944) aligns adapter outputs with preferred responses without pure PPL minimization. BitNet-2B-4T's own training used SFT + DPO.
- **Motivation:** Our prior LEARNINGS identified PPL-trained adapters as causing mode collapse. DPO optimizes for preference, not perplexity.
- **Literature:** Zephyr-7B (arXiv:2310.16944) showed DPO on top of SFT improves generation quality on MT-Bench by 2+ absolute points.

### 3. Objective Benchmarks Only (Already Planned)
exp_task_accuracy_real_benchmarks (GSM8K, HumanEval) avoids subjective evaluation entirely. Finding #179 (math 24x correctness) proves the architecture helps where correctness is binary.
- **Motivation:** Two evaluation methodology kills (keyword density + LLM-judge). Objective tasks are the only reliable evaluation at micro scale.

## Implications for Next Experiments

1. **Do NOT trust any prior finding that used lora_scale=20** without re-testing at standard scale. This includes Findings #166, #179, #180, and all falcon_e3b experiments.
2. **Scale ablation should be the FIRST experiment in any new adapter study** — it's a 1-variable change with massive explanatory power.
3. **Composition-as-regularization is not a feature, it's a symptom** of overcooking individual adapters. If adapters work individually at scale=1, composition's job becomes combining expertise, not rescuing broken adapters.
4. **SFT vs NTP is a secondary question** until the scale confound is resolved. The training objective may matter, but we can't tell until the dominant variable (20x amplification) is controlled.

## Recommended Follow-Up

**exp_lora_scale_ablation** — Test lora_scale={1, 2, 4, 8, 20} x {SFT, NTP} on 2-3 domains (medical, math, code) with GSM8K + MMLU evaluation. 2x5x3 = 30 training runs, ~10 min each = 5 hours.
- **Motivation:** Finding #180 (this experiment), lora_scale=20 identified as dominant confound
- **Literature:** Hu et al. 2022 (alpha/r standard), LoTA-QAF arXiv:2407.11024 (conservative scaling for low-bit)
- **Kill criterion:** If degradation persists at scale=1 for both SFT and NTP, the problem is NOT scale (need to investigate data quality, rank, or architecture)
