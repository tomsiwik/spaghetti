# Learnings: exp_adapter_compression_extreme (Ternary B-Matrix)

## Core Finding

Fully ternary LoRA adapters (both A and B in {-1,0,+1}) compose with ratio < 1.07, matching FP32 B quality, because the Grassmannian A-skeleton confines each adapter's quantization noise to its own subspace -- making B precision irrelevant for composition.

## Why This Happened (Literature-Grounded)

The central mechanism is **subspace isolation via Grassmannian orthogonality**. The interference bound `||DeltaW_i^T DeltaW_j|| <= alpha_i * alpha_j * ||T_i||_F * ||A_i^T A_j||_F * ||T_j||_F` collapses to zero when A_i are orthogonal, regardless of B precision. This is mathematically clean and empirically confirmed (mean |cos| ~ 0.000001).

However, **the literature challenges our mechanistic framing**. NotebookLM research reveals:

1. **LoTA-QAF** (KingdalfGoodman/LoTA-QAF) independently confirms ternary A+B adapters work, but attributes success to **grid alignment** rather than orthogonality. Their ternary adapters produce integer-valued deltas in [-r, r] that map losslessly to the quantized base via threshold omega. This is a complementary mechanism: grid alignment preserves per-adapter quality, while our Grassmannian orthogonality preserves cross-adapter composition.

2. **Rethinking Inter-LoRA Orthogonality** (arxiv 2510.03262) empirically shows that weight-space orthogonality does NOT guarantee semantic compositionality or disentanglement. Orthogonality prevents geometric interference but not nonlinear output perturbations. At our micro scale (PPL ~1.5, toy task), the nonlinear effects are negligible -- but they may dominate at macro scale.

3. **SVDQuant** shows low-rank branches absorb activation outliers, not quantization noise per se. The mechanism we observe (quantization noise confined to subspaces) is specific to the frozen-A + orthogonal architecture, not a general property of low-rank adapters.

## Confirming Evidence

- **LoTA-QAF** (2025): Ternary A+B adapters on Llama-3.1/3.3 and Qwen-2.5 surpass 16-bit LoRA by up to 5.14% on MMLU via lossless merging. Confirms ternary adapters are viable at production scale.
- **BitNet b1.58** (Ma et al., 2024): Absmean quantization with STE is the standard for ternary training. Our implementation is correct and consistent.
- **Prior project experiment `bitnet_ternary_adapter_composition`**: Already showed ternary adapters compose better than FP16 LoRA (-4.4% PPL, -19.3% cosine). Current experiment isolates the B-matrix contribution.
- **Prior `b_matrix_training_correlation`**: B-matrix |cos| is 2.52x random baseline, but Grassmannian skeleton decorrelates deltas to 0.14x baseline (17x filter). Directly supports the "A filters B noise" mechanism.

## Contradicting Evidence

- **OSRM** (arxiv 2505.22934): Weight-space orthogonality != data-space orthogonality. Our composition guarantee is geometric (weight space) but the actual failure mode at scale is functional (output space). The macro validation MUST measure output-space interference, not just weight-space cosine.

- **LoTA-QAF task-specific fine-tuning**: At extreme low-bit (2-bit base + ternary adapters), task-specific performance degrades vs 16-bit LoRA. Ternary adapters lack representational capacity for fine-grained task details. Our toy task may not stress this because character-level name generation is inherently low-entropy.

- **SOLE macro composition catastrophe** (project history): Even with Grassmannian orthogonality, macro-scale composition suffered "composition catastrophe." Root causes identified:
  - B-matrix training dynamics introduce overlap despite fixed A
  - Logit-scale mismatch: orthogonality controls direction but not magnitude
  - Nonlinear amplification through residual streams and layer normalization
  - A single "poisoned" adapter (badly trained SQL adapter) caused trillion-PPL explosion

- **PTQ merging degradation** (LoTA-QAF, Bondarenko 2024, Guo 2024): Standard PTQ of FP32 adapters into quantized base causes accuracy loss. Our PTQ result (ratio 1.050) may be favorable because the base is already ternary -- there's no precision mismatch during merging (ternary adapter into ternary base).

## Alternative Approaches (What We Could Try Instead)

### For Adapter Compression
1. **VeRA** (Kopiczko et al., 2023): Freeze random A and B matrices, learn only tiny scaling vectors. Even more extreme compression than ternary -- essentially zero-parameter adapters. Could combine with Grassmannian initialization of the frozen matrices.

2. **LoKr** (Kronecker-structured PEFT): Kronecker product of low-rank factors provides "rank amplifier" effect -- small factors yield large effective rank. Could achieve better expressivity per bit than ternary.

3. **LowRA**: Pushes below 2 bits (down to 1.15 bits) using Weighted Lloyd-Max per-channel thresholds + ILP precision allocation. More fine-grained than our uniform ternary -- worth testing if per-tensor alpha proves insufficient at scale.

4. **RDLC** (Router-Driven LoRA Compaction): "Zero-stored-expert" -- hypernetwork router generates adapters procedurally from seeds. Completely eliminates adapter storage. Relevant for our routing/composition story.

5. **SeedLM**: Compress weight blocks into pseudorandom seeds + few coefficients. LFSR generates random matrix at inference. Extreme compression for serving.

### For Composition Quality
6. **ImPart** (Importance-Aware Delta Sparsification): SVD + importance-based sparsity ratios achieve 2x compression at same merge quality. Could replace uniform ternary with importance-weighted quantization.

7. **LoRA-LEGO**: Rank-wise clustering of "Minimal Semantic Units" across adapters for interference-free merging. Our Grassmannian approach is stronger in theory (mathematical guarantee vs empirical clustering).

8. **LoRAuter** (Output-Space Retrieval Fusion): Avoids parameter merging entirely -- fuses adapter outputs via retrieval. Achieves oracle-level performance on some benchmarks. This sidesteps the composition problem but requires runtime routing overhead.

## Implications for Next Experiments

### Immediate
1. **PTQ is the default going forward.** Train FP32 B, quantize to ternary after. Simpler pipeline, near-identical quality (1.050 vs 1.068 ratio), trivially reversible.

2. **The "ternary regularization" effect (STE B producing 0.989x PPL) should NOT be trusted.** Single seed, toy task, within noise margin. Do not design around this.

3. **The 15.8x compression claim needs packed storage implementation.** Currently theoretical. Before claiming this in VISION.md, actually implement ternary packing and measure real file sizes.

### Macro Validation (Critical Path)
4. **Test on diverse tasks (code, math, reasoning)** where B-matrix capacity matters. The toy character-level task cannot distinguish ternary from FP32 B. The literature warns ternary adapters fail on complex fine-grained tasks.

5. **Measure output-space interference, not just weight-space cosine.** OSRM proved these diverge. The Grassmannian guarantee is necessary but may not be sufficient at scale.

6. **Monitor for logit-scale mismatch when scaling up.** Ternary B inherently bounds magnitude (alpha * {-1,0,+1}), which may actually HELP with the logit-scale problem that plagued macro composition. This is a testable hypothesis: does ternary B's bounded magnitude prevent the "poisoned adapter" failure mode?

### Strategic Insight
7. **Our architecture is uniquely positioned for ternary adapters.** Unlike standard LoRA (where PTQ degrades quality due to precision mismatch), our ternary base + ternary adapter architecture has no precision gap during merging. LoTA-QAF achieves something similar through grid alignment, but we get it for free from architectural choice. This is a genuine advantage worth emphasizing.

8. **Consider VeRA-like approaches** for even more extreme compression. If the Grassmannian A already determines the subspace, and ternary B works, then perhaps B only needs to encode a scaling direction -- VeRA's scaling-vector approach might be the limit of how far we can compress.
