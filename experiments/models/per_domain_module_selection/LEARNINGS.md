# LEARNINGS: Per-Domain Module Selection

## Core Finding

**At ternary LoRA behavioral scale (s=20), module effects are subadditive through
nonlinear layer interactions — applying fewer modules can produce BETTER domain
quality than applying all modules. Attention-only adapters preserve or improve
medical/math quality while MLP adapters are uniquely required for code generation.**

## Why This Happened

### Subadditive Module Interference

When attention and MLP adapters are applied jointly at scale s=20, each perturbs
the residual stream significantly. The nonlinear operations between modules
(LayerNorm, SiLU, softmax) compound these perturbations multiplicatively across
30 layers. The result: combined perturbation is LESS effective than each module
alone. This matches our earlier finding (Finding #303, room model kill) that
nonlinear layer interactions make simultaneous adapter composition structurally
lossy.

The mechanism is clear from the PPL data: math full-module PPL (3.78) is WORSE
than base (3.76), while math attn-only (3.43) improves substantially. The MLP
adapter actively degrades math at this scale — its perturbation disrupts the
base model's stored mathematical knowledge in MLP key-value memories without
adding sufficient compensating capability.

### Attention vs MLP Functional Roles

Geva et al. (2012.14913) established that MLP layers function as key-value
memories storing factual knowledge. Our results confirm: perturbing MLP weights
disrupts knowledge recall (MMLU degradation), while attention adapters redirect
information flow without damaging stored knowledge. This explains why:
- Medical attn-only: +7% behavioral, better PPL, MMLU preserved
- Code attn-only: -67% behavioral collapse (code syntax requires MLP pattern storage)
- Math attn-only: equivalent behavioral, better PPL (math is reasoning/attention)

## Confirming Evidence

1. **LoRA Learns Less and Forgets Less** (Biderman et al., 2405.09673): LoRA
   better maintains base model performance on out-of-domain tasks than full
   finetuning. MLP modules have higher effective rank than attention modules.
   Our finding that attention-only preserves MMLU while full-module degrades it
   is consistent — less perturbation = less forgetting.

2. **PLoP: Precise LoRA Placement** (Hayou et al., 2506.20629): Optimal LoRA
   placement is task- and model-dependent, not universally attention or MLP.
   PLoP uses normalized feature norm growth to select modules per task, finding
   that different tasks benefit from different module placements. This directly
   validates our per-domain module selection approach.

3. **AdaLoRA** (Zhang et al., 2303.10512): Importance-based rank allocation
   preferentially assigns more budget to FFN layers and top layers. Our finding
   adds nuance: at high adapter scale on ternary models, this allocation can
   be counterproductive for knowledge/reasoning tasks.

4. **Finding #292** (Pierre v6): Attention-only precomputed deltas showed
   medical +8%, code -67% — identical pattern to this experiment, independently
   confirming the attention-sufficient/code-needs-MLP dichotomy.

5. **Finding #303** (Room model kill): Nonlinear layer interactions compound
   adapter deltas multiplicatively. The subadditive module interference
   discovered here is the per-layer manifestation of the same structural
   impossibility.

## Contradicting Evidence

1. **Original LoRA** (Hu et al., 2106.09685): Reports attention-only LoRA as
   sufficient across tasks, but at rank 4-8 on FP16 models. At our scale (rank 16,
   s=20, ternary base), the perturbation magnitude is substantially larger,
   which changes the interference regime. Their finding holds for attention but
   not for the claim that MLP is unnecessary.

2. **LoRA Learns Less and Forgets Less** (2405.09673): Also finds MLP modules
   are "primary loci for continual learning" and targeting all modules drives
   most gains. This contradicts our "fewer is better" finding — but their
   experiments use moderate rank on FP16 models. The ternary base + high scale
   regime appears to flip this relationship. The contradiction may be
   scale-dependent.

3. **AdaLoRA** (2303.10512): Allocates MORE rank to FFN, implying FFN matters
   most. Our finding says FFN perturbation can hurt. The difference: AdaLoRA
   optimizes rank allocation during training (FFN gets more because it needs
   more), while our experiment evaluates post-training module ablation (removing
   FFN post-hoc can help because the jointly-trained FFN adapter interferes).

## Alternative Approaches

1. **PLoP-guided training** (2506.20629): Instead of post-hoc module selection,
   use PLoP's normalized feature norm criterion to determine module placement
   BEFORE training. Train domain-specific adapters with only the selected
   modules, avoiding the confound of jointly-trained B-matrices evaluated in
   isolation.

2. **LoRI sparse masks** (Zhang et al., 2504.07448, COLM 2025): Fixed random
   A-matrices with sparse task-specific B-masks (up to 90% sparsity). Reduces
   cross-task interference while using 95% fewer parameters than LoRA. Our
   Grassmannian A-matrices are already fixed — LoRI's sparse B-masks could
   further reduce per-module perturbation without eliminating modules entirely.

3. **ARD-LoRA dynamic rank** (2506.18267): Continuous, differentiable per-head
   rank allocation via meta-objective. More granular than our binary (include/
   exclude) module selection — could find the optimal perturbation magnitude
   per module per domain.

## Implications for Next Experiments

1. **The optimal module config is domain-dependent and empirically discoverable.**
   For Pierre's per-domain bitmask: {medical:attn, code:full, math:attn,
   legal:attn, finance:attn}. This gives 43% fewer dispatches for 4/5 domains.

2. **Subadditive interference is a general property at high adapter scale.**
   Any experiment adding modules/adapters should check whether the addition
   actually helps or hurts. The linear superposition assumption (combined effect
   = sum of individual effects) is invalid at s=20 through 30 layers.

3. **B-matrix co-adaptation confound remains open.** The current adapters were
   trained with all 7 modules active. Attn-only adapters trained from scratch
   (with B-matrices that don't expect MLP co-activation) might perform even
   better. This is Limitation 2 from PAPER.md and the reviewer's advisory.

4. **The code-needs-MLP finding is the strongest result.** Medical/math
   attn-sufficiency and code MLP-dependency are measured with behavioral tests
   (not n=15 MMLU), giving them statistical reliability. The MMLU trends are
   directionally consistent but unconfirmed.

## Recommended Follow-Up

**Per-domain purpose-trained adapters** (P1): Train 5 new adapters where each
domain uses only its optimal module set during training (not post-hoc ablation).
Motivation: PLoP (2506.20629) shows task-specific placement outperforms uniform
placement; our post-hoc ablation shows the direction but the B-matrix
co-adaptation confound means purpose-trained adapters could perform better.
This would also eliminate the 43% dispatch overhead structurally rather than
as a serving optimization.

**Batched LoRA gather** (P1): Already proposed as `exp_batched_lora_gather_mlx`.
Per-domain module selection reduces the dispatch count but doesn't eliminate
the per-dispatch overhead. Punica's BGMV (2310.18547) fused kernel approach
on MLX could compound with module selection for multiplicative speed gains.
