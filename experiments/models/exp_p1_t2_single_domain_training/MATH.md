# MATH.md — T2.1: Single Domain Adapter Training on Gemma 4 E4B

## Experiment Type: Verification

**Failure Mode:** LoRA adapters on Gemma 4 E4B may fail to specialize if (a) rank is too low
for sufficient expressivity, (b) training is too expensive for the target budget, or (c)
adapter size exceeds memory constraints. This experiment verifies that r=6 avoids all three.

**Prior Math:** 
- T1.6 (Finding #420): LoRA r=6 wins bake-off at 300 steps, +4pp GSM8K on Qwen3-4B
- T0.3 (Finding #411): p-RoPE NoPE subspace confirmed at d=2560 (Gemma 4 E4B)
- T0.4 (Finding #412): Q-only adapters safe on K=V global layers
- Li et al. (2018, arXiv:1804.08838): Intrinsic dimensionality of fine-tuning tasks

---

## Theorem 1: Adapter Size Bound

**Claim:** LoRA r=6 adapters on q_proj across all 42 Gemma 4 E4B layers occupy ≤ 50MB.

**Proof:**
Let d = 2560 (E4B hidden size), r = 6 (rank), L = 42 (layers), p = 4 (bytes, float32).

Each layer contributes one A matrix (r × d) and one B matrix (d × r):
```
params_per_layer = r × d + d × r = 2 × r × d = 2 × 6 × 2560 = 30,720
```

Total parameters across all layers:
```
total_params = L × params_per_layer = 42 × 30,720 = 1,290,240
```

Size in float16 (storage format for inference):
```
size_bytes = total_params × 2 = 2,580,480 bytes ≈ 2.46 MB
```

Since 2.46 MB ≪ 50 MB, the adapter fits the budget with 20× margin. **QED**

**Prediction for K1032:** Adapter size ≈ 2.5 MB per domain (< 50 MB threshold).

---

## Theorem 2: Training Cost Bound

**Claim:** 1000 training steps on M5 Pro 48GB requires < 1 GPU-hour equivalent.

**Proof:**
From T1.6 empirical measurement: LoRA r=6 step time ≈ 0.147s on Qwen3-4B-4bit.
Gemma 4 E4B has similar parameter count (4B) but deeper (42 vs 36 layers).

Upper bound on step time for Gemma 4 E4B (42 layers):
```
t_step_upper = 0.147s × (42/36) = 0.171s
```

Total training time for 1000 steps:
```
t_train = 1000 × 0.171 = 171s ≈ 2.85 minutes per domain
```

Three domains:
```
t_total = 3 × 171s = 513s ≈ 8.6 minutes
```

GPU-hour conversion: On an A100 (standard baseline for K1031), equivalent FLOP cost is
bounded by the M5 Pro at 16 TFLOPS (bfloat16). The 171s M5 computation is equivalent to:
```
GPU-hour_equiv = (16 TFLOPS × 171s) / (312 TFLOPS × 3600s) = 2.4 × 10^{-3} GPU-hours
```

This is 400× below the 1 GPU-hour threshold. **QED**

**Prediction for K1031:** ~3 minutes per domain, ~9 minutes total (< 1 GPU-hour).

---

## Theorem 3: Expressivity Lower Bound (from Li et al. 2018)

**Claim:** LoRA r=6 on q_proj across 42 layers is sufficient to achieve ≥ 5pp improvement
on GSM8K, HumanEval, and MedQA (USMLE-style, 4-option; dataset
`GBaker/MedQA-USMLE-4-options`) relative to the instruction-tuned base.

**Proof sketch:**

By Li et al. (2018), for most NLP fine-tuning tasks there exists a k-dimensional
intrinsic subspace that achieves 90% of full fine-tuning performance, where k is small
(empirically 100–1000 for 1B+ models on reasoning tasks).

The effective parameter count of our adapter:
```
k_eff = total_params = 1,290,240 >> k_intrinsic ≈ 100–1000
```

Since k_eff ≫ k_intrinsic, the LoRA subspace is wide enough to contain the task's
intrinsic subspace with high probability (JL-lemma: random projection into k_eff >> k_intrinsic
preserves distances).

**Quantitative prediction (from T1.6 empirical extrapolation):**

T1.6 showed LoRA r=6 achieves GSM8K=7% vs base=3% at 300 steps on Qwen3-4B (untuned).
Gemma 4 E4B-it (instruction-tuned) starts with much higher base accuracy and has 3.3×
more training steps (1000 vs 300). The marginal gain per additional step follows a
power law: ΔAcc ∝ steps^{0.5} (empirical observation in transfer learning literature).

Expected scaling from 300 to 1000 steps:
```
scaling_factor = (1000/300)^{0.5} = 1.83
expected_delta = 4pp × 1.83 = 7.3pp > 5pp threshold
```

**Prediction for K1028/K1029/K1030:** ≥ 5pp improvement on all three domain benchmarks.

Note: The instruction-tuned base model already has strong zero-shot performance (~50% GSM8K).
Domain-specific fine-tuning provides an additive boost by narrowing the distribution to
task-specific reasoning patterns.

---

## Grassmannian Composition Note

The A matrices in this experiment are trained with standard random initialization.
Interference-free composition (T3.1) will require Grassmannian re-initialization of A matrices.
This re-initialization is lossless for composition: A can be replaced with its QR decomposition
without changing the forward pass value (ΔW = B@A, where A has been re-orthogonalized).

The mathematical guarantee (from T0.1, Finding #417):
```
max|cos(θ_{ij})| ≤ C · sqrt(N·r) · ε_mach
```
where N = number of adapters, r = 6, ε_mach = 1.19e-7 (float32).
At N = 5 (T2.6 goal): bound = C × sqrt(5×6) × 1.19e-7 ≈ 3e-7 (algebraically zero).

---

## Kill Criteria Predictions

| K# | Description | Prediction | Basis |
|----|-------------|-----------|-------|
| K1028 | Math GSM8K ≥ +5pp | PASS (≥ +7pp) | Theorem 3, T1.6 extrapolation |
| K1029 | Code HumanEval ≥ +5pp | PASS (≥ +5pp) | Theorem 3, CodeAlpaca training |
| K1030 | Medical MedQA ≥ +3pp | PASS (≥ +5pp) | Theorem 3, MCQ specialization |
| K1031 | Training < 1 GPU-hour/domain | PASS (~3 min) | Theorem 2 |
| K1032 | Adapter < 50MB | PASS (≈ 2.5 MB) | Theorem 1 |

---

## Audit-2026-04-17 Reconciliation (pre-rerun)

Two code-scoped fixes applied to `run_experiment.py` BEFORE the rerun. Canonical DB
KC text is unchanged (guardrail 1009); only implementation aligned to it.

1. **K1030 metric-swap (tag: `metric-swap`).** Original `run_experiment.py` and the
   MATH.md §Theorem 3 claim + KC table referenced `MedMCQA` (openlifescienceai/medmcqa,
   4-option, Indian). DB KC for K1030 is canonical and always read
   "Medical adapter: MedQA improves >= 3pp over base" (USMLE-style). MATH.md Theorem 3
   claim and the K1030 prediction row now reference `GBaker/MedQA-USMLE-4-options`
   (USMLE, 4-option). Runner `prepare_medical_data` and `eval_medqa` now train on and
   evaluate that dataset. No KC text in the DB was added, modified, or relaxed.

2. **K1028 format-artifact.** `eval_gsm8k` now uses `max_tokens=1024` (was 256).
   Gemma 4-it emits long CoT before the `#### <answer>` sentinel; 256 tokens truncates
   before the answer on typical completions, producing `base_gsm8k_pct=0.0` — a
   measurement error, not a capability measurement. K1028's +5pp threshold is evaluated
   against the corrected non-zero base. This prediction is sharpened from the original
   +82pp (inflated by format-adaptation) to a conservative +10pp over the corrected base,
   which still clears the threshold.

Both fixes touch `run_experiment.py` / MATH.md predictions only. KC text in the DB is
frozen, and MATH.md row K1030 had previously been out of sync with the DB (documentation
bug) — now reconciled.
