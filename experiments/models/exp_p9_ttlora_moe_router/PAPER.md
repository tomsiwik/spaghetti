# PAPER.md -- P9.B2: TT-LoRA MoE -- Gated Routing Across Domain Experts

## Summary

Tested linear routing across 5 TT-LoRA domain experts (math/code/medical/legal/finance)
trained on MMLU. The router achieves 97.7% domain classification accuracy, validating
hidden-state separability. However, the TT-LoRA v_proj-only adapters (r=6, 64K params)
produce near-random MCQ accuracy (~25%), making the MoE composition meaningless.

**Type:** Guided exploration
**Prior:** Finding #516 (TT-LoRA 84.4% quality retention), arXiv:2504.21190

---

## Prediction vs Measurement

| Kill Criterion | Prediction | Measurement | Verdict |
|---|---|---|---|
| K1360: Router accuracy >= 90% | >= 95% (Thm 1: JL separability) | **97.7%** (train: 100%) | **PASS** |
| K1361: MoE >= single best + 5pp | ~17.5pp advantage (Thm 3) | **-1.4pp** (25.8% vs 27.2%) | **FAIL** |
| K1362: Total size < 2 MB | ~652 KB (Thm 2: param counting) | **795 KB** (770 adapter + 25 router) | **PASS** |

---

## Detailed Results

### Adapter Training (500 steps each, TT-LoRA r=6, v_proj only)

| Domain | Train Examples | Final Loss | Train Time | Adapter Size |
|---|---|---|---|---|
| math | 900 | 0.1315 | 560s | 154 KB |
| code | 371* | 0.0773 | 545s | 154 KB |
| medical | 895 | 0.1237 | 504s | 154 KB |
| legal | 883 | 0.0531 | 1015s | 154 KB |
| finance | 900 | 0.0708 | 431s | 154 KB |

*Code domain has only 412 available examples in MMLU test split (see Caveats).

### Router Performance (K1360)

| Domain | Router Accuracy |
|---|---|
| math | 97.0% |
| code | 97.6% |
| medical | 96.0% |
| legal | 100.0% |
| finance | 98.0% |
| **Average** | **97.7%** |

Router: 12,805 params (25 KB), trained 300 steps, final loss 0.0001.

### Cross-Domain Accuracy Matrix (adapter rows x eval domain columns)

| Adapter | math | code | medical | legal | finance | Avg |
|---|---|---|---|---|---|---|
| math | 21.0 | 28.6 | 29.0 | 24.0 | 24.0 | 25.3 |
| code | 28.0 | 31.0 | 29.0 | 25.0 | 23.0 | 27.2 |
| medical | 22.0 | 21.4 | 28.0 | 30.0 | 19.0 | 24.1 |
| legal | 33.0 | 23.8 | 16.0 | 27.0 | 23.0 | 24.6 |
| finance | 33.0 | 28.6 | 15.0 | 28.0 | 22.0 | 25.3 |

**Key observation:** No adapter achieves meaningfully above-random accuracy on any
domain (random = 25%). The diagonal (in-domain accuracy) shows no specialization:
math-on-math = 21%, medical-on-medical = 28%, finance-on-finance = 22%.

### MoE vs Baselines (K1361)

| Method | Accuracy |
|---|---|
| Random guessing | 25.0% |
| Oracle routing (best adapter per domain) | 25.1% |
| Routed MoE (97.7% router) | 25.8% |
| Single best adapter (code) | 27.2% |

MoE advantage = -1.4pp. **The router works; the experts don't.**

---

## Why Theorem 3 Failed

Theorem 3 predicted Delta >= (alpha - 1/K)(q_bar - q_off) = 17.5pp, assuming:
- q_bar (avg in-domain accuracy) ~ 60%
- q_off (avg off-domain accuracy) ~ 35%

**Actual values:** q_bar ~ 25.8%, q_off ~ 24.9%. The theorem's algebra is correct,
but the input assumptions were wrong by 2x. The adapters simply don't improve MCQ
accuracy, so q_bar ≈ q_off ≈ random, making Delta ≈ 0.

### Root Cause: v_proj-only TT-LoRA Cannot Steer MCQ Answers

1. **Capacity bottleneck:** 64,260 TT-LoRA params on v_proj alone cannot encode
   enough task knowledge to steer a 4-bit quantized model's MCQ predictions.
   The value projection influences attention output but not the token-level
   logit distribution that determines A/B/C/D selection.

2. **Loss ≠ accuracy:** Adapters converge to low training loss (0.05-0.13) but this
   reflects memorization of answer formatting, not domain knowledge. This confirms
   Finding #516's observation that PPL doesn't predict task quality (r=0.08).

3. **4-bit ceiling:** The quantized base model's general knowledge already determines
   MCQ performance. Small v_proj perturbations don't shift the logit landscape enough
   to change answer selection.

---

## Caveats

### Code Domain Data Scarcity
The code domain has only 412 training examples and 42 eval examples in the MMLU test
split (vs 900+ for other domains). This affects:
- **Adapter quality:** Fewer unique training examples (371 after val split)
- **Router accuracy for code:** Fewer hidden state examples for router training
- **Eval reliability:** 42 eval examples = ±15% confidence interval at 95% confidence

However, code domain results (31.0% in-domain, 97.6% router) are consistent with
other domains, so data scarcity is not the primary explanation for the overall failure.

### Evaluation Methodology
Logit-based MCQ evaluation compares raw logits at A/B/C/D token positions. This is
standard for MMLU but may undercount cases where the model would generate the correct
answer in a different format.

---

## Impossibility Structure

**What structure makes this failure impossible?**

The failure is NOT in routing (97.7% accuracy proves separability). The failure is in
expert quality: v_proj-only TT-LoRA with 64K params is insufficient to steer MCQ behavior.

To make expert quality failure impossible, you need one of:
1. **More projection targets:** q_proj + k_proj + v_proj + o_proj (4x more params)
2. **Higher rank:** r=16+ to encode richer task-specific transformations
3. **FFN adaptation:** FFN layers store factual knowledge (Meng et al. 2022, arXiv:2202.05262),
   which is what MCQ accuracy actually requires
4. **Full LoRA instead of TT-LoRA:** TT-LoRA's 12.4x compression may discard the
   fine-grained weight structure needed for knowledge steering

**The right question is not "how to make TT-LoRA MoE work" but "what is the minimum
adapter capacity that produces measurable behavioral change?"** This experiment provides
a lower bound: 64K params on v_proj is below that threshold.

---

## What This Validates

Despite the K1361 failure, this experiment provides two positive contributions:

1. **Routing works at scale:** A linear router (12,805 params, 25 KB) achieves 97.7%
   domain classification on Gemma 4's hidden states. This validates the MoE routing
   mechanism for future experiments with stronger experts.

2. **Size budget confirmed:** 5 TT-LoRA experts + router fit in 795 KB. The
   compression-routing composition is viable; only expert quality needs improvement.

---

## Experiment Metadata

- Platform: Apple M5 Pro 48GB, MLX
- Model: Gemma 4 E4B-IT 4-bit (mlx-community/gemma-4-e4b-it-4bit)
- Total runtime: 3440s (57.3 min)
- Peak memory: 26.72 GB
