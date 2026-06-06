# LEARNINGS: exp_m2p_vera_bottleneck

**Experiment:** exp_m2p_vera_bottleneck  
**Status:** killed  
**Date:** 2026-04-08

---

## Core Finding

VeRA-style M2P at rank-4 achieves 98x parameter compression (357M → 4.67M) but the
scaling vector composition collapses to r=4 **effective** DOF per module — not 2r=8 as
designed — because the two r-dimensional scaling vectors merge via element-wise product.
The adapter performs at base rate (quality_ratio=-0.105, z=-0.48 vs base p=0.63),
confirming the hypothesis space is too constrained for GSM8K math reasoning at this rank.

Additionally, the VeRA parameterization was implemented incorrectly: original VeRA uses
lambda_d ∈ R^{d_out} (scaling output rows) and lambda_b ∈ R^r (scaling columns), giving
d_out + r DOF per adapter. This experiment used lambda_d, lambda_b ∈ R^r, collapsing to
an effective diag(d*b) @ W.T with only r=4 DOF.

---

## Why This Happened

### 1. VeRA Parameterization Misunderstood

The intended formula `diag(d) @ W.T @ diag(b)` with d, b ∈ R^r and W ∈ R^{d_out × r}
does not type-check: W.T is (r × d_out), diag(b) is (r × r) — the product W.T @ diag(b)
is (r × d_out) @ (r × r) which is undefined. The code silently implemented
`diag(d*b) @ W.T` instead, collapsing two scaling vectors to one.

**Original VeRA** (Kopiczko et al., arXiv:2310.11454):
- lambda_b ∈ R^r scales columns of W_shared
- lambda_d ∈ R^{d_out} scales rows of the output
- DOF per adapter: d_out + r (for Qwen3-0.6B q_proj: 2048 + 4 = 2052)

**This experiment's implementation:**
- Both scaling vectors ∈ R^r, collapsing via d*b elementwise
- DOF per adapter: r = 4

The factor-of-512 capacity difference (2052 vs 4 for q_proj) fully explains the
adapter learning nothing useful despite gradient flow.

### 2. GLUE → GSM8K Transfer Assumption Failed

VeRA Table 2 shows rank-4 matching LoRA on GLUE benchmarks (SST-2, CoLA, MRPC, etc.).
GLUE tasks are largely sentiment/entailment classification where low-dimensional
adaptation (single polarity direction, topic direction) suffices. GSM8K multi-step math
reasoning requires coordinated per-layer specialization across attention heads — the
solution subspace is not low-rank in the same sense.

The JL-lemma argument ("random basis spans the relevant subspace") fails for structured
tasks where the optimal adapter lies in a thin slice of weight space that may not
intersect a random rank-4 basis.

### 3. Fundamental Tension: Compression vs Expressivity

Even a correct VeRA implementation (d_out + r DOF) would require M2P to output
O(d_out × N_layers) scalars ≈ 2048 × 28 = 57,344 per task. This is not compression
anymore — it's comparable to the original output dimensionality. The tension between
"M2P must be small" and "adapters must be expressive" has no cheap solution at low rank.

---

## Confirming Evidence

- **VeRA (arXiv:2310.11454) Table 2**: rank-4 VeRA works on GLUE, not validated on
  reasoning tasks. The paper never claims rank-4 suffices for math reasoning.
- **LoRA (arXiv:2106.09685) Appendix E**: low-rank intrinsic dimensionality is
  task-dependent; NLP classification tasks have intrinsic rank < 8, while generation
  tasks may require rank ≥ 32.
- **Aghajanyan et al. (arXiv:2012.13255)** — Intrinsic Dimensionality Explains the
  Effectiveness of Language Model Fine-Tuning: different tasks have different intrinsic
  dimensionalities; math reasoning is among the highest.

## Contradicting Evidence

- **VeRA Table 2** (rank=4 on commonsense reasoning tasks like BoolQ, WinoGrande):
  achieves competitive performance. These are single-step reasoning, not multi-step
  chain-of-thought — they may have lower intrinsic rank than GSM8K.

---

## Alternative Approaches

1. **Correct VeRA at higher rank (r=16)**  
   - Would give d_out + 16 DOF per adapter ≈ d_out DOF
   - Still requires M2P to output d_out-scale vectors per layer per task
   - 98x compression collapses to ~16x — useful but the fundamental tension remains
   - Motivation: VeRA Table 2 shows rank-16 matches LoRA-rank-16 on GLUE; might work on
     harder tasks too

2. **DoRA: Weight-Decomposed Low-Rank Adaptation (arXiv:2402.09353)**  
   - Decomposes weight update into magnitude + direction components
   - More expressive than LoRA at same rank because magnitude scaling is decoupled
   - Could replace VeRA in M2P output with fewer parameters than full LoRA
   - Reference finding: #548 (already recorded)

3. **Task-Router Compression: share basis across task families, not layers**  
   - Instead of sharing W_shared across all 28 layers, share it across similar tasks
   - Per-task, per-layer VeRA (d_out + r DOF, ~2052/layer) is non-compressive
   - Sharing the random basis across medical+legal (similar semantic space) might work
   - No strong arxiv evidence for this specific architecture yet

4. **Adapter Sparsification Instead of Rank Compression**  
   - Keep full-rank LoRA adapters but learn sparse masks (SparseGPT-style)
   - arXiv:2301.00774 (SparseGPT): structured sparsity in weight matrices
   - Reduces storage/FLOPs without the rank bottleneck
   - Would not help M2P output size, only adapter serving cost

5. **Accept M2P's 357M Parameters as Design Reality**  
   - The v4 result (quality_ratio=0.754, M2P ≈ SFT at n=500) is the real signal
   - At macro scale (Qwen3-4B), M2P would be ~10B params — likely prohibitive
   - But at current micro scale, 357M may be acceptable for the research phase
   - Macro-scale compression is a P2 concern; P0 is proving composability

---

## Implications for Next Experiments

1. **Parameter compression is not the next bottleneck.** The open P0 question is
   composability: can M2P generate adapters for N>1 tasks that compose well?
   exp_m2p_composition_n5 style experiments with Qwen3-0.6B + GSM8K are the priority.

2. **If compression is needed, rank must ≥ 16 with correct lambda_d ∈ R^{d_out}.**
   Rank-4 VeRA is mathematically closed for math reasoning (effective DOF too low).

3. **Intrinsic dimensionality should be measured before choosing rank.**
   Aghajanyan et al. (arXiv:2012.13255) provide a method (random Gaussian subspace
   search) to empirically estimate the intrinsic rank of GSM8K fine-tuning — this would
   directly inform the minimum viable rank for VeRA compression.

---

## Recommended Follow-Up

**exp_m2p_composition_n5_qwen3** (P0, motivated by Finding #376/379):
- Test whether two M2P adapters (math + one other domain) compose without interference
- Uses v3/v4 full-rank M2P (not VeRA) to establish composition baseline on a real LLM
- Kill criterion: composition quality ≥ single-adapter quality (null-space isolation)
- Motivation: arXiv:2402.04403 (TIES merging) shows composition quality depends on
  weight-space geometry, not just individual adapter quality

**Do NOT pursue VeRA variants** until intrinsic rank of GSM8K is measured.
The failure mode is mathematically closed at low rank without that measurement.

---

## References

- #548: arXiv:2310.11454 (VeRA) — shared random matrix adaptation, GLUE only
- #549: arXiv:2402.09353 (DoRA) — weight-decomposed LoRA, more expressive at same rank
- arXiv:2012.13255 (Aghajanyan) — intrinsic dimensionality of NLP fine-tuning tasks
- arXiv:2106.09685 (LoRA) — Appendix E: task-dependent intrinsic rank
- arXiv:2301.00774 (SparseGPT) — structured sparsity alternative to rank reduction
