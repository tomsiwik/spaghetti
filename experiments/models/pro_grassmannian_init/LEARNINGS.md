# LEARNINGS: exp_pro_grassmannian_init

## Core Finding

**QR-based Grassmannian A-matrices transfer exactly from MHA (BitNet-2B-4T) to GQA
(Qwen3-4B) when hidden_dim matches: cos=0.000 across all 72,072 measured pairs.
GQA invariance (Theorem 3) is the key insight — A-matrices depend only on input
dimension, not output dimension or KV-head count. The quantized weight shape bug
(4-bit packing reports d=320 instead of d=2560) would have silently corrupted all
downstream adapter training; catching it here validates verification experiments.**

## Why This Happened

The result is mathematically guaranteed (QR orthonormality is constructive, not
probabilistic). The interesting aspects are:

1. **GQA invariance is trivial but easy to miss.** In GQA, k_proj and v_proj have
   fewer output dimensions (n_kv_heads * head_dim) than q_proj (n_heads * head_dim),
   but A-matrices project from the INPUT space (hidden_dim = 2560 for all). This means
   our Grassmannian capacity (N_max = 160) is identical across MHA and GQA architectures
   when hidden_dim matches. No paper we found explicitly states this for LoRA — most
   GQA papers focus on inference efficiency, not adapter orthogonality.

2. **Quantized weight shapes are a silent trap.** MLX 4-bit quantization stores weights
   in packed format (2560 / 8 = 320 for 4-bit). Naively reading weight.shape gives the
   packed dimension, not the logical dimension. This caused the first run to produce
   non-orthogonal A-matrices (cos=0.061) before the fix. The model config is the only
   reliable source for logical dimensions on quantized models.

3. **Timing prediction failure is instructive.** MATH.md predicted <10s based on
   ~1 TFLOP/s NumPy estimate; actual was 32.3s for N=24. Small-matrix QR (2560x384)
   is memory-bound and Python-loop-dominated, not compute-bound. NumPy TFLOP/s
   estimates are only valid for large dense GEMM.

## Confirming Evidence

- **LoRA-GA (2407.05000, NeurIPS 2024):** Uses SVD-derived orthogonal initialization
  for A and B matrices. Achieves 2-4x faster convergence than random init. Confirms
  that structured initialization matters for LoRA quality, though LoRA-GA optimizes
  for gradient alignment (single adapter) rather than cross-adapter orthogonality
  (our multi-adapter use case).

- **Ortho-LoRA (2601.09684, Jan 2026):** Orthogonal gradient projection for multi-task
  LoRA. Projects conflicting task gradients onto orthogonal complements within LoRA
  subspace. Recovers 95% of single-task performance. Confirms that orthogonality in
  adapter parameter space prevents task interference. Our approach achieves this at
  initialization rather than during training.

- **LoRA-Null (2503.02659, Mar 2025):** Initializes LoRA in null space of pre-trained
  activations. Finds that initialization space is the key to knowledge preservation.
  Complementary to our approach: we use QR for cross-adapter orthogonality; LoRA-Null
  uses SVD for adapter-vs-pretrained orthogonality.

- **Finding #132 (this project):** Grassmannian AP skeleton reduces interference on
  BitNet-2B-4T (d=2560, MHA). Post-training B-matrix cosine 0.030 maps to delta
  cosine 0.0017 (17x decorrelation filter). Finding #318 confirms identical geometric
  guarantees transfer to Qwen3-4B (d=2560, GQA).

## Contradicting Evidence

- **Rethinking Inter-LoRA Orthogonality (2510.03262):** Finds that strict orthogonality
  alone does NOT yield semantic disentanglement. Orthogonal Monte Carlo dropout enforces
  orthogonality but provides little benefit for compositionality. **However:** this
  paper studies post-hoc merging of independently trained adapters, not our architecture
  where A-matrices are frozen and shared across training. The contradiction is about
  whether orthogonality SUFFICES (it doesn't alone), not whether it's necessary (it is,
  as our interference bound A_i^T A_j = 0 shows).

- **OPLoRA (2510.13003):** MiLoRA-style approach preserves dominant singular subspace.
  Suggests that which subspace you initialize in matters more than just orthogonality.
  Not a direct contradiction — we control initialization subspace AND orthogonality.

## Alternative Approaches

1. **LoRA-Null (2503.02659):** Null-space initialization from activation SVD. Could
   be combined with our QR approach: first compute null space of activations, then
   apply QR within that null space for cross-adapter orthogonality. This would give
   both knowledge preservation AND interference prevention.

2. **Ortho-LoRA (2601.09684):** Training-time gradient projection. Alternative to
   our frozen-A approach. Trades initialization guarantee for training-time flexibility.
   Higher computational cost (gradient projection per step) but adapts to actual
   task structure.

3. **LoRA-GA (2407.05000):** SVD-based initialization for gradient alignment. Could
   replace our random Gaussian matrix with the actual weight gradient as the QR input,
   giving both orthogonality AND gradient alignment. Worth investigating if convergence
   speed becomes a bottleneck.

## Implications for Next Experiments

1. **Skeleton files are ready for SFT.** grassmannian_skeleton_n5.npz and
   grassmannian_skeleton_n24.npz can be loaded directly into LoRA training on
   Qwen3-4B. The quantized weight shape bug is fixed.

2. **B-matrix interference needs re-measurement on Qwen3-4B.** Finding #132's 17x
   decorrelation was on BitNet-2B-4T. The effective interference bound
   ||A_i B_i^T B_j A_j^T|| depends on B-matrices, which are architecture-specific.

3. **N_max=160 provides massive headroom.** At our target of 25 domains, we use
   only 15.6% of orthogonal capacity. Even at 100 domains, 62.5%.

4. **All composition proofs from BitNet track transfer.** Findings #225, #304-#314
   used d=2560 throughout. The Grassmannian, composition, and routing results apply
   directly to Qwen3-4B.

## Recommended Follow-Up

**exp_pro_instruction_adapter (P0):** Train instruction-following adapter as
mandatory first step before behavioral evaluation. Finding #317 showed base model
can't follow instructions (GSM8K 48%, IFEval 33%). MMLU logit-based evaluation
is the composition degradation metric (no instruction confound). Motivation:
LoRA Learns Less and Forgets Less (2405.09673) confirms LoRA on frozen base
preserves knowledge, making it safe to add instruction capability without
degrading MMLU.
