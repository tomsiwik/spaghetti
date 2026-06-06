# FFN-only Matched Rank: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Value |
|--------|-----------|-------------|
| d | Model embedding dimension | 3584 (Qwen2.5-7B) |
| d_ff | FFN intermediate dimension | 18944 (= 5.29d) |
| d_kv | KV head dimension total | 512 (= d_head * n_kv) |
| L | Number of layers | 28 |
| r | LoRA rank | 16 |
| alpha | LoRA scaling factor | 16 (alpha = r) |
| N | Number of domain experts | 5 |

## 2. The Confound in the Prior Experiment

The prior experiment (exp_ffn_only_vs_all_modules) measured FFN-only orthogonality
by extracting the FFN parameters from jointly-trained all-modules adapters:

    FFN_subset(theta_all) = {theta_k : k in FFN_keys(theta_all)}

This is a **retroactive subset**, not an independently trained FFN-only adapter.
The confound: when training with all-modules LoRA, the FFN and attention
parameters co-adapt. Specifically:

### 2.1 Co-adaptation Effect

During joint training, the gradient for FFN parameters depends on the presence
of attention LoRA:

    grad_FFN(L_joint) != grad_FFN(L_ffn_only)

because the loss landscape changes when attention parameters are also being
adapted. The attention LoRA modifies the hidden representations that flow
into the FFN layers:

    h_l = Attn(h_{l-1}) + dAttn(h_{l-1})  (with attn LoRA)
    h_l = Attn(h_{l-1})                     (without attn LoRA)

So FFN parameters trained jointly see a different input distribution than
FFN parameters trained alone. The question is: does this difference matter
for orthogonality and quality?

### 2.2 Hypotheses About the Difference

**H1 (Orthogonality preserved):** If FFN orthogonality is driven by
domain-specific knowledge storage (Geva et al. 2021), the input
perturbation from attention LoRA is small relative to the domain signal,
and independently trained FFN-only adapters will have similar orthogonality.

**H2 (Quality preserved):** If FFN layers carry 72-77% of the adapter norm
(measured in prior experiment), the attention adaptation contributes at most
23-28% of the adaptation "budget." The quality gap from dropping attention
should be bounded by this norm fraction.

**H3 (Quality degrades):** If certain domains require domain-specific
attention patterns (e.g., code with deeply nested scopes), FFN-only may
underperform specifically for those domains.

## 3. Parameter Counting (Review)

### 3.1 FFN-only at rank 16

    Per layer: 3 * r * (d + d_ff) = 3 * 16 * (3584 + 18944) = 1,081,344
    Total: L * 1,081,344 = 28 * 1,081,344 = 30,277,632 (~30.3M)
    On disk: ~57.7 MB (bf16) or ~28.9 MB (fp16 safetensors)

### 3.2 All-modules at rank 16

    FFN: 30,277,632
    Attn per layer: r * (6d + 2*d_kv) = 16 * (6*3584 + 2*512) = 360,448
    Attn total: 28 * 360,448 = 10,092,544 (~10.1M)
    All total: 40,370,176 (~40.4M)

### 3.3 Parameter ratio

    all / ffn = 40.4M / 30.3M = 1.333x

FFN-only uses 25% fewer parameters. This is the same rank but fewer modules.

## 4. Expected Orthogonality

### 4.1 Random vector baseline

For random unit vectors in D-dimensional space:

    E[|cos(u, v)|] = sqrt(2 / (pi * D))

FFN-only delta space D_ffn = L * 3 * d * d_ff = 28 * 3 * 3584 * 18944
                           = 5,702,452,224 (~5.7B)

    E[|cos|]_ffn = sqrt(2 / (pi * 5.7e9)) = 1.06e-5

### 4.2 Trained adapter prediction

From the prior experiment, the retroactive FFN subset had:

    mean |cos|_retroactive = 0.0605

This is ~5700x the random baseline, reflecting training structure (shared
data patterns, similar optimization dynamics). The question: will
independently trained FFN-only adapters have similar, higher, or lower
cosine?

**Prediction:** The independent FFN-only cosine should be LOWER than the
retroactive subset, because:
1. Without attention LoRA, the FFN must carry the full adaptation burden,
   potentially pushing it toward more domain-specific representations
2. The co-adaptation effect in joint training may cause FFN parameters to
   partially align with the shared attention patterns
3. Independent training removes this implicit alignment pressure

However, the opposite is also possible: without attention adapting the
routing, FFN may need to learn some attention-like patterns, increasing
cross-domain similarity.

## 5. Kill Criterion Analysis

### 5.1 Quality Kill: PPL gap > 5%

    PPL_gap = (PPL_ffn - PPL_all) / PPL_all * 100%

Kill if: PPL_gap > 5% for any domain.

The 5% threshold is chosen because:
- At Qwen2.5-7B scale (typical PPL ~3-6 on domain data), 5% = 0.15-0.30
  PPL points, which is well above measurement noise
- The parameter savings (25%) should justify a small quality gap (<3%)
- A gap >5% suggests FFN-only is fundamentally insufficient

### 5.2 Orthogonality Kill: >50% difference from retroactive

    ortho_diff = |mean_cos_independent - mean_cos_retroactive| / mean_cos_retroactive * 100%

Kill if: ortho_diff > 50%.

The 50% threshold is generous because:
- We expect SOME difference due to the co-adaptation confound
- A 50% change (e.g., 0.06 -> 0.09 or 0.06 -> 0.03) would indicate
  the retroactive analysis was fundamentally misleading
- Within 50%, the qualitative conclusion (FFN more orthogonal than
  all-modules) would still hold

## 6. Experimental Design

### 6.1 Training Protocol

Both FFN-only and all-modules adapters trained with:
- Base: Qwen2.5-7B (4-bit quantized via BitsAndBytes)
- Rank: 16, alpha: 16
- Steps: 300
- Effective batch size: 8 (micro-batch 1, grad_accum 8)
- LR: 2e-4 with warmup
- Optimizer: AdamW 8-bit
- Seed: 42
- Data: same 1000 examples per domain from distillation data
- Packing: enabled, max_seq_length: 1024

### 6.2 Fair Comparison

The existing all-modules adapters in `adapters/` were trained with the same
hyperparameters (verified from adapter_config.json: r=16, alpha=16,
target_modules includes all 7 modules). The training script matches
`composer/distill.py::train_one_expert()`.

For maximum fairness, we can also retrain all-modules with the exact same
script and seed (--also-train-all flag). This eliminates any differences
from software version, data ordering, or initialization.

### 6.3 Evaluation

PPL computed on 50 held-out examples per domain (data/distillation/*/eval.jsonl),
using the SFTTrainer evaluation loop with packing enabled.

## 7. Worked Example

For a single domain pair (bash vs python), FFN-only:

    cos(bash_ffn, python_ffn) = cos(
        flatten([gate_A_l, gate_B_l, up_A_l, up_B_l, down_A_l, down_B_l] for l in 0..27),
        flatten([gate_A_l, gate_B_l, up_A_l, up_B_l, down_A_l, down_B_l] for l in 0..27)
    )

Each flattened vector has dimension:
    28 layers * (16*3584 + 16*18944 + 16*3584 + 16*18944 + 16*18944 + 16*3584)
    = 28 * 16 * (3*3584 + 3*18944)
    = 28 * 16 * 67584
    = 30,277,632 elements

The retroactive measurement found cos(bash, python) = 0.0028 for FFN.
The independent training should produce a value in the same order of magnitude.

## 8. Assumptions

1. **QLoRA quantization does not change orthogonality properties.**
   The 4-bit base model sees the same quantized representations in both
   configurations. The LoRA parameters themselves are in full precision.

2. **300 steps is sufficient for convergence.** The existing adapters were
   trained with 300 steps. If training is not converged, PPL comparison
   is invalid. We verify by checking that training loss has plateaued.

3. **Eval PPL is a valid quality proxy.** While exp_ppl_vs_task_performance
   killed full-sequence PPL as a task accuracy proxy, the comparison here
   is PPL_ffn vs PPL_all on the SAME data. Relative PPL differences are
   more reliable than absolute PPL-to-accuracy mapping.

4. **Seed 42 is representative.** Single-seed results may not generalize.
   For a definitive answer, 3 seeds would be ideal. Budget constraints
   limit us to 1 seed for the initial run, with replication as a follow-up
   if results are borderline.
