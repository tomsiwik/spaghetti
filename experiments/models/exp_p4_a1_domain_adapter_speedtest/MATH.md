# MATH.md — P4.A1: Domain Adapter Training Speed

## Theorem 1: New Domain in <10 Minutes

**Claim:** Training a rank-16 LoRA adapter for a new domain (biology) on Gemma 4 4-bit requires
T_total < 10 min, where T_total = T_data + T_train + T_eval, and delivers ≥10pp behavioral
improvement over the base model on domain-specific vocabulary rubric.

**Prior Math:**
- Finding #436 (P1.T5.1): rank-4 LoRA, 40 examples, 300 steps → 1.2 min training on M5 Pro.
  Throughput: ~300 steps / 1.2 min = 250 steps/min.
- P3.C5: rank-16 LoRA, 150 examples, 500 steps → 2.6 min.
  Throughput: ~500 steps / 2.6 min = 192 steps/min.
- LoRA 2106.09685 (Hu et al.): Training cost scales as O(r × d_model × M × seq_len).

**Proof:**

Let θ_P4A1 = (r=16, M=200, N=100, seq_len=256) be the hyperparameter tuple.

Step 1: Bound T_train.
From P3.C5 empirical throughput (lower bound, conservative): 192 steps/min.
T_train ≤ M / throughput = 200 / 192 = 1.04 min.

Step 2: Bound T_data.
Training data is synthetically generated (template-based, no model calls).
T_data ≤ 1 s (Python string manipulation, no I/O bottleneck).

Step 3: Bound T_eval.
Evaluating N_eval=20 responses × ~3s/response (Gemma 4 4-bit, 100 tokens) = 60s = 1 min.
Times 2 (base + adapted): T_eval ≤ 2 min.

Step 4: Total bound.
T_total = T_data + T_train + T_eval ≤ 0.02 + 1.04 + 2 = 3.06 min << 10 min. QED K1217.

**Corollary (Behavioral Improvement):**
The training data is written with biology-domain vocabulary density ρ_train > ρ_base, where:
  ρ = E[1(|V_bio ∩ response| ≥ 3)]
After LoRA fine-tuning on high-density text, adapted model inherits ρ_adapted ≥ ρ_train > ρ_base.

**Prediction:** Vocabulary improvement Δρ = ρ_adapted - ρ_base ≥ 10pp (K1218).

Mechanism: The LoRA adapter shifts the query projections to increase attention to bio-domain
token clusters. The vocabulary rubric measures this shift directly (Finding #437 validation).

**Size bound (K1219):**
LoRA adapter parameter count:
  P = 2 × r × d_model × n_layers_trained
For Gemma 4-e4b: d_model=5120 (empirically measured from smoke test), n_layers=12 (query projections only), r=16.
  P = 2 × 16 × 5120 × 12 = 1,966,080 ≈ 2.0M parameters × 4 bytes (fp32) = 7.86 MB.
Well under 10 MB. QED K1219.
Note: MATH.md initially assumed d_model=2048; smoke test revealed d_model=5120 (corrected).

## Kill Criteria

| ID | Criterion | Predicted | Threshold |
|----|-----------|-----------|-----------|
| K1217 | training_time (min) | 1.04 | < 10 |
| K1218 | behavioral_improvement (pp) | ≥ 10 | ≥ 10 |
| K1219 | adapter_size_mb | 2.25–4.5 | < 10 |

## Failure Modes

- K1217 fails if: training stalls (OOM, VRAM eviction). Mitigation: use 4-bit base.
- K1218 fails if: base model vocabulary already saturated (ρ_base ≈ 1.0). 
  Biology-specific terms like "ribosome", "chloroplast", "mitochondria" should distinguish
  adapted vs. base responses.
- K1219 cannot fail: adapter size is determined by architecture, not training.
