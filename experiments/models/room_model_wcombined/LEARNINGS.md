# LEARNINGS: exp_room_model_wcombined (KILLED)

## Core Finding

**Pre-summing N>1 adapter deltas into a single dense W_combined matrix is structurally
dead in deep nonlinear transformers: Room PPL is WORSE than base (no adapter) for all
5 domains, and dense W_combined costs 230x more bandwidth than factored LoRA (4.17 GB
vs 18 MB), yielding 41.9 vs ~97 tok/s. Two independent experiments (#303, #315) confirm
identical per-domain ratios (medical 1.92x vs 1.91x), making this the most reproduced
negative result in the project.**

## Why This Happened

### 1. Nonlinear Compounding Is Irreducible (Structural, Not Fixable)

Pre-summing N adapter deltas means the hidden state at layer l+1 sees the combined
perturbation from ALL adapters at layer l. LayerNorm rescaling, attention softmax
redistribution, and SiLU gating are all nonlinear — each creates cross-terms between
adapter deltas that do not exist when adapters are applied individually. Through L=30
layers, these cross-terms compound multiplicatively.

**Zhong et al. (2504.10957, ICLR 2025 oral)** provide the first formal generalization
analysis of task vectors on nonlinear transformers, proving task arithmetic is effective
ONLY when tasks are "irrelevant or aligned" — i.e., their deltas do not interact through
nonlinear layers. Our 5 domain adapters modify overlapping hidden state dimensions and
are definitionally NOT in this safe regime.

The original Task Arithmetic paper (**Ilharco et al., 2212.04089**) showed empirical
degradation at high scaling coefficients — the same nonlinear interaction effect. Our
experiment with N=5 deltas at full alpha is well outside the small-perturbation regime
where linear approximation holds.

### 2. Dense W_combined Is Bandwidth-Catastrophic

W_combined materializes a full d_out x d_in matrix per module (210 modules x ~20K params
each in bf16 = 4.17 GB). Factored LoRA keeps the h @ A @ B form at ~18 MB total. On
Apple Silicon (bandwidth-bound at 273 GB/s), this 230x bandwidth difference translates
directly to 2.3x speed penalty (41.9 vs ~97 tok/s).

The ROOM_MODEL.md prediction of "100+ tok/s" was wrong because it counted dispatch
reductions (210 vs 2100), not bandwidth. **S-LoRA (2311.03285)** and **Punica
(2310.18547)** confirm that factored LoRA serving is more efficient than dense matrix
approaches at any scale, precisely because factored form exploits the rank structure
(r << d) to reduce bandwidth by d/r (~160x at rank 16).

### 3. Room PPL > Base Means Net-Negative Architecture

The most damning result is not that Room degrades vs single-adapter (expected), but that
it degrades vs BASE (no adapter at all). The combined noise from 5 adapter deltas exceeds
the benefit of any individual adapter's domain knowledge. This means pre-summing does not
just fail to compose — it actively destroys the base model's capabilities.

## Confirming Evidence

- **2504.10957** (Zhong et al., ICLR 2025 oral): Task arithmetic provably effective only
  for "irrelevant or aligned" tasks. First theoretical characterization of when weight-space
  addition fails in nonlinear transformers.
- **2504.07448** (LoRI, COLM 2025): Freezes A as random projections + sparsifies B with
  task-specific masks. Achieves orthogonality with 95% fewer parameters. Confirms that
  naive full-delta summation causes cross-task interference.
- **2508.03999** (TC-LoRA): Uses CP decomposition to disentangle task-specific vs shared
  factors in LoRA merging. Explicitly addresses interference from naive merging.
- **2505.22934** (OSRM): Orthogonal subspaces help prevent interference during merging,
  but orthogonality must be constrained during fine-tuning, not just imposed geometrically.
  Our Grassmannian initialization is geometric, not semantic.
- **2505.15875** (Decouple and Orthogonalize): Data-free LoRA merging that decouples before
  merging — acknowledges naive summation creates interference.
- **Finding #303**: Original Room Model POC. Identical per-domain ratios confirm
  reproducibility. Medical 1.92x vs 1.91x, code 1.69x vs 1.78x, speed 39.2 vs 41.9 tok/s.

## Contradicting Evidence

- **LoRA Soups (2410.13025):** Simple averaging of LoRA adapters works for skill composition.
  Key difference: averaging at 1/N scale (norm-controlled), not full-alpha sum. Our
  experiment applies all N deltas at full alpha — 5x beyond the small-perturbation regime.
  The contradiction resolves on scale: averaging ≠ summing.

- **2601.18350** (Adapter Merging Reactivates Reasoning): Merging can activate latent traces
  — but this is for inference on a single merged model, not N simultaneous deltas at full
  strength. And the "reactivation" is shown to be an artifact of representation conflict in
  final layers, not a benefit.

- **Task Arithmetic in Tangent Space (NeurIPS 2023)**: Linearizing models around initialization
  amplifies weight disentanglement and improves task arithmetic. Suggests nonlinear
  compounding could be reduced by operating in the tangent space. But this requires
  linearizing the entire model — incompatible with ternary base inference.

## Alternative Approaches (All Paper-Backed)

1. **Per-token adapter routing (Finding #312, MoLoRA 2603.15965):** Route each token to ONE
   adapter instead of applying all simultaneously. Avoids multi-adapter activation entirely.
   MoLoRA: Qwen3-1.7B exceeds Qwen3-8B on 4 reasoning benchmarks. Our own Finding #312
   confirms MLP per-token routing produces 6x signal over attention routing.

2. **Block-diagonal attention + single-pass (Finding #314):** Block-diagonal masking
   eliminates cross-segment contamination. Our own Finding #314 confirms this is the best
   single-pass strategy (PPL 4.529, -2.7% vs multi-pass). RoPE reset closes the remaining gap.

3. **LoRI-style frozen projections (2504.07448, COLM 2025):** Freeze A as random projections,
   sparsify B per task. 95% fewer parameters, orthogonality from structure not geometry.
   Compatible with our Grassmannian A-matrices — we already have the frozen-A pattern.

4. **CP decomposition for merging (TC-LoRA, 2508.03999):** If merging is needed, decompose
   adapter tensors into shared vs task-specific factors before merging. Avoids naive summation
   that causes interference.

5. **Fused factored serving (Punica 2310.18547, S-LoRA 2311.03285):** Custom kernels for
   h @ A @ B without per-adapter dispatch overhead. Addresses speed without pre-summing.

## Implications for the Project

1. **Pre-summing at N>1 is permanently closed.** Three attempts (#302, #303, #315),
   two independent measurements with identical results, theory (2504.10957) proves it
   fails outside the "irrelevant or aligned" regime. No further experiments needed.

2. **N=1 pre-merge (Pierre v6) remains valid and proven.** Single adapter delta is small
   enough that nonlinear interactions are negligible. This is in the safe regime.

3. **The factored LoRA path is confirmed as the architecture.** Findings #304 → #305 →
   #312 → #313 → #314 form a complete chain from "MLP carries 69% energy" through
   "single-pass matches multi-pass" to "block-diagonal is best strategy." All use
   factored h @ A @ B, not dense W_combined.

4. **Bandwidth is the first-order constraint on Apple Silicon.** Any architecture that
   increases per-token bandwidth (dense matrices, additional projections) will lose to
   factored form. Design around r << d, not around fewer dispatches.

## Recommended Follow-Up

No follow-up on the room model itself — it is dead. The three kills (#302, #303, #315)
are conclusive, and the theoretical impossibility (2504.10957) makes resurrection
impossible without operating in a fundamentally different regime (e.g., linearized
models, which are incompatible with ternary inference).

The confirmed path forward is the factored LoRA chain:
- **exp_rope_reset_block_diagonal** (P1): Per-segment RoPE position reset to close the
  8.9% gap from Finding #314. Lemma 1' predicts exact equivalence with reset.
  Motivation: Block-Attention (2409.15355) already implements this as "position
  re-encoding" for RAG serving.
