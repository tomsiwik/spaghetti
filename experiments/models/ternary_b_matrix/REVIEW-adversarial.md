# Peer Review: Ternary B-Matrix

## NotebookLM Findings

Skipped -- NotebookLM step omitted in favor of direct code and math review given the experiment's straightforward structure.

## Mathematical Soundness

### Derivations: Correct

1. **STE formulation** (MATH.md lines 29-33, code lines 198-207): The straight-through estimator is correctly implemented. `b_ste = b + stop_gradient(b_q - b)` gives `b_q` in forward, gradients to `b` in backward. Standard BitNet b1.58 pattern, verified against prior experiment `ternary_base_from_scratch_mlx`.

2. **Absmean quantization** (MATH.md lines 17-22, code lines 476-483): `alpha = mean(|W|)`, scale, round, clip. Correct and consistent between MATH.md and code.

3. **Orthogonality independence claim** (MATH.md lines 70-76): The interference bound `||DeltaW_i^T DeltaW_j|| <= alpha_i * alpha_j * ||T_i||_F * ||A_i^T A_j||_F * ||T_j||_F` is correct. When A_i are orthogonal, A_i^T A_j = 0 and the bound collapses regardless of B precision. This is the central mathematical insight and it holds.

4. **Worked example** (MATH.md lines 112-140): Arithmetic verified. alpha_1 = mean(|[0.3, 0.7, 0.1, 0.5]|) = 0.4. Relative error calculation checks out: sqrt(0.12) / sqrt(0.84) = 0.346/0.917 = 0.377.

5. **Composition formula** (MATH.md lines 57-58, code lines 567-577): Weight merging via `W_composed = W_base + (1/N) * sum_i [B_i @ A_i]` is correctly implemented.

### Hidden Assumptions: One Noted

**Per-adapter scalar scale (alpha) is sufficient.** MATH.md Assumption 3 acknowledges this. The absmean quantization uses one alpha per B-matrix (per layer per adapter). At rank 8 with output dim 128, each B-matrix is 128x8 = 1024 elements sharing a single scale factor. This is aggressive but consistent with BitNet b1.58 which uses per-tensor scales at much larger dimensions.

### No Mathematical Errors Found

The math is clean and directly follows from established BitNet b1.58 quantization theory applied to the B-matrix of LoRA.

## Novelty Assessment

### Prior Art Within This Project

1. **`bitnet_ternary_adapter_composition`**: Already proved ternary *adapters* (A+B both quantized) compose better than FP16 LoRA (-4.4% PPL, -19.3% cosine). However, that experiment used a different quantization scheme (post-quantized base, QAT on both A and B). This experiment isolates B-matrix quantization specifically by using Grassmannian-frozen A matrices.

2. **`ternary_base_from_scratch_mlx`**: Proved STE training on a ternary base (PPL 1.003x FP32). The current experiment reuses the same STE mechanism on B-matrices specifically.

3. **`bitnet_lori_sparse_b`**: Tested LoRI-style 90% sparse B-matrices. KILLED because sparsity concentrated signal into overlapping positions (1.46x higher cosine). This experiment's ternary approach is distinct from sparsity -- ternary preserves all positions but reduces precision, while sparse zeros out positions.

4. **`b_matrix_training_correlation`**: Proved B-matrix |cos| is 2.52x random baseline, but Grassmannian skeleton decorrelates despite this (delta cos only 0.14x baseline, 17x filter). Directly supports the current experiment's central claim.

### Delta Over Existing Work

The delta is **incremental but useful**: isolating B-matrix quantization as an independent variable while holding A fixed (Grassmannian-frozen). Prior work quantized both A and B simultaneously. The separation lets us attribute composition preservation specifically to Grassmannian orthogonality filtering quantization noise, rather than to any interaction between A and B quantization.

### External Prior Art

BitLoRA (referenced in HYPOTHESES.yml) makes both base and adapters ternary. LoRI uses frozen A + sparse B. Neither specifically tests PTQ vs STE for B-matrix-only ternary quantization with Grassmannian A. The novelty is real but narrow.

## Experimental Design

### Strengths

1. **Three-way comparison**: FP32 B (baseline), STE ternary B, and PTQ ternary B. This is the right design -- it separates training-aware quantization from post-hoc quantization.

2. **Shared base and A-matrices**: All three conditions use the same trained ternary base and identical Grassmannian A-matrices (same seeds 42, 137). This eliminates confounds from base quality or A-matrix variation.

3. **Composition via weight merging**: Tests the actual deployment scenario (merge into base weights, not runtime LoRA). Code at lines 567-587 correctly computes `(1/N) * sum B_i @ A_i` and adds to base weights.

4. **Cosine similarity measurement**: Computes full delta vector cos(DeltaW_i, DeltaW_j) across all layer pairs. All pairs show ~0 cosine, confirming Grassmannian guarantee.

### Weaknesses

1. **Single seed** (acknowledged in Limitations point 3). The STE vs FP32 B gap on composition ratio is 1.068 vs 1.047 = +0.021. Without multi-seed validation, we cannot distinguish this from noise. The paper honestly flags this but it limits confidence in the STE vs PTQ ranking.

2. **Task homogeneity** (acknowledged). All 5 domains are character-level name generation subsets. The B-matrix carries task-specific signal; diverse tasks (code, math, prose) would stress B capacity more. This is a known micro-experiment limitation and acceptable.

3. **No control for "ternary regularization" claim**. The paper notes STE B produces 0.989x individual PPL (better than FP32 B) and speculates about regularization. A proper test would add a dropout/noise baseline on FP32 B to check whether the improvement is from constrained optimization or just noise injection. Not blocking, but the paper should not lean on this observation.

4. **PTQ uses FP32 B condition's trained adapters**. The PTQ evaluation at lines 504-505 attaches FP32 LoRA modules and overwrites B-matrices with quantized versions. This is correct -- PTQ should use FP32-trained weights. But it means PTQ inherits whatever seed-specific artifacts FP32 training produced. A fairer test would PTQ from multiple FP32 training runs.

5. **Compression ratio calculation is theoretical, not measured**. Lines 672-674 compute ternary storage as `(params * 2) / 8` bits. The actual saved `.npz` files store FP32 arrays. The 15.8x claim is about what *could* be achieved with packed ternary storage, not what the experiment actually measures. The paper should clarify this is a projected compression ratio, though this is standard practice for quantization papers.

### Does It Test What It Claims?

**Yes.** The hypothesis is "ternary B preserves composition quality (ratio < 1.5)." The experiment directly measures composition ratio for ternary B conditions and finds 1.068 (STE) and 1.050 (PTQ), both well below 1.5. The mechanism (Grassmannian orthogonality filtering B quantization noise) is confirmed by near-zero cosine similarities.

### Could a Simpler Mechanism Explain the Results?

The task is so easy (PPL ~1.5, near-optimal character-level names) that even random B-matrices might compose well. A **random B-matrix control** -- randomly initialize B without training, evaluate composition -- would establish a floor. If random B gives ratio ~1.05 too, then the result tells us nothing about ternary quantization specifically. This is not fatal (the B-matrices are clearly trained, not random), but it would strengthen the paper.

## Macro-Scale Risks (advisory)

1. **B-matrix information content scales with task complexity.** At d=128, rank=8, the B-matrix is 128x8 = 1024 elements. Ternary quantization loses ~52% per-element information (measured). At d=2560, rank=16, B is 2560x16 = 40,960 elements on complex tasks (code, math). The relative quantization error may stay ~52% but the *impact* on task performance could be much larger when the task actually requires fine-grained weight adjustments.

2. **Per-tensor scale may be insufficient at scale.** One alpha per B-matrix works at 128x8 because the weight distribution is simple. At 2560x16 with real task diversity, per-row or per-group scales may be needed, reducing the compression advantage.

3. **The "pure addition" claim needs validation.** PAPER.md claims ternary B enables "pure addition composition" (no floating-point multiply). This is true for the ternary indices T, but the scale factors alpha_i still require multiplication. At scale with per-group scales, this advantage diminishes. The macro experiment should measure actual inference speedup, not just theoretical operation counts.

4. **LoRI sparse B was killed for concentrating signal.** The prior experiment `bitnet_lori_sparse_b` showed 90% sparse B causes 1.46x higher cosine (signal concentration into overlapping positions). Ternary B does not have this problem in principle (it preserves all positions), but at scale the effective sparsity of ternary B (the fraction of zeros) should be monitored. The experiment does not report zero fractions in the B-matrices.

## Verdict

**PROCEED**

The experiment is well-designed within its micro-experiment constraints. The math is sound. The central insight -- that Grassmannian orthogonality filters B-matrix quantization noise, making B precision irrelevant for composition quality -- is correctly derived and empirically confirmed. Both STE and PTQ ternary B achieve composition ratios well below the 2.0 kill threshold and even below the 1.5 success threshold. The code correctly implements all three conditions with shared base and A-matrices.

The single-seed limitation and task homogeneity are acknowledged. The incremental novelty over prior ternary adapter experiments is real (isolating B-matrix quantization with frozen Grassmannian A). The result is directional evidence that fully ternary adapters are viable for composition.

**Minor recommendations (not blocking):**

1. Report the zero fraction of ternary B-matrices (what percentage of elements map to 0 vs +/-1). This is a free diagnostic and informs whether ternary B is effectively sparse.
2. Temper the "ternary regularization" speculation -- a single-seed 1.1% PPL improvement is indistinguishable from noise without multi-seed validation.
3. Clarify in PAPER.md that the 15.8x compression is projected (packed ternary storage), not measured from actual file sizes.
4. The macro validation should test on diverse tasks (code, math, reasoning) where B-matrix capacity matters most.
