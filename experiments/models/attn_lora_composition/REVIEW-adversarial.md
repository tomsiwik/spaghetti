# Peer Review: Attention LoRA Composition (exp_attention_adapter_composition)

## NotebookLM Findings

Skipped -- the experiment was self-killed by the experimenter. Review focuses on verifying the kill decision.

## Mathematical Soundness

**Derivations are correct.** The LoRA parameterization dW = (alpha/r) * A @ B is standard. Parameter counts check out:

- MLP LoRA: 4 layers x 2 projections x (64x8 + 8x256) = 4 x 2 x 2560 = 20,480. Confirmed.
- Attn LoRA: 4 layers x 2 projections x (64x4 + 4x64) = 4 x 2 x 512 = 4,096. Confirmed.
- Overhead 4,096/20,480 = 20%. Confirmed.

**One minor notation inconsistency (non-blocking):** MATH.md line 32 computes MLP LoRA as `2 * r_m * 5d` which gives `2 * 8 * 5 * 64 = 5,120` per layer, then claims `= 10,240 * 2 = 20,480`. The intermediate `10,240` is actually 4 layers x 2,560 per layer = 10,240 per-projection-type. The final 20,480 is correct but the derivation path is confusing. This is a presentation issue, not a math error.

**The linearity claim for attention composition is correct but incomplete.** MATH.md correctly notes that `q_composed = sum_k w_k * (W_q + dW_q^k) @ x` is linear in routing weights. However, the composed attention matrix involves `softmax(q @ k^T)`, which is nonlinear. The composed query/key projections are linear in the routing weights, but the attention pattern they produce is not. The PAPER.md does not claim attention composition is end-to-end linear, so this is not an error, but it is an important subtlety: attention LoRA composition goes through the softmax nonlinearity, making it fundamentally different from MLP output routing (which averages outputs post-nonlinearity). The routing weights affect the inputs to softmax, not the outputs. This means small routing perturbations could produce large attention pattern changes -- or no change at all, depending on the softmax temperature regime. This is not analyzed.

**The "delta_attn" approximation (MATH.md line 120) is not derived.** The first-order perturbation `delta_attn ~ softmax'(base_attn) * (dW_q @ x) @ (dW_k @ x)^T / sqrt(d_h)` is stated but not derived. It is approximately correct as a first-order Taylor expansion, but it ignores the cross-terms between base Q/K and adapted Q/K. The full expansion of `(W_q + dW_q)x @ ((W_k + dW_k)x)^T` has three delta terms: `(dW_q x)(W_k x)^T + (W_q x)(dW_k x)^T + (dW_q x)(dW_k x)^T`. Only the third is shown. The first two are rank-d perturbations, not rank-r_a, and are larger in magnitude. This omission does not affect the experiment's conclusions but the math claims more structure than exists.

## Novelty Assessment

**Low novelty, but appropriate for micro-scale validation.** Adding LoRA to attention projections is standard practice -- Hu et al. (2021) explicitly discuss adapting Wq, Wk, Wv, Wo. The QLoRA paper (Dettmers et al., 2023) adapts all linear layers. InfLoRA (Liang et al., 2024, referenced in `references/infolora-continual/`) applies LoRA to attention for continual learning.

The novel element here is specifically measuring whether attention LoRA helps *composition of independently-trained experts*. This is a valid micro-scale question that prior work does not directly address -- prior work adapts attention for single-model fine-tuning or sequential continual learning, not for post-hoc expert composition.

**No reinvention detected.** The experiment builds on the existing `lora_gpt` codebase appropriately and cites relevant prior art.

## Experimental Design

**The experimental design is sound and well-controlled.**

Strengths:
1. Three-seed aggregation (42, 123, 7) with per-seed reporting
2. Proper controls: MLP-only baseline, attention-only ablation, joint training gold standard
3. Same base model, same pretraining, same fine-tuning steps across conditions
4. Router calibration steps identical between MLP-only and MLP+Attn conditions
5. Delta norm analysis provides interpretive diagnostics

**One design concern: confounded routing.** In `RoutedDeltaGPT.__call__`, when `has_attn_deltas=True`, the routing weights for attention are computed from `h_attn = base_layer.norm1(x)` (line 254), while MLP routing uses `h = base_layer.norm2(x)` (line 288). Critically, the attention routing uses `self._get_routing_weights(h_attn, l_idx)` which calls the same per-layer router used for MLP routing. This means:

- With attention deltas: the router at layer l_idx is called TWICE per forward pass (once for attention, once for MLP), with different inputs (`norm1(x)` vs `norm2(x)`).
- Without attention deltas: the router is called once (for MLP only).

During router calibration, the MLP+Attn condition trains the router with gradients flowing through both attention and MLP paths, while the MLP-only condition trains only through MLP. This is not necessarily wrong -- it matches the intended deployment -- but it means the router has a harder optimization problem in the MLP+Attn case (it must simultaneously route attention AND MLP, potentially conflicting objectives). The router architecture (single linear layer per transformer layer) may not have enough capacity for this dual routing task. This is an unacknowledged confound.

**The 1pp kill threshold is reasonable.** Given that the MLP-only composition gap is only 0.78% on average, a 1pp improvement would be asking the attention adapter to more than double the gap closure. The threshold is aggressive but pre-registered, which is the correct approach.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry matches the experiment:
- Kill criterion 1: "attention adapters improve composition gap <1% over MLP-only capsules" -- tested, +0.46pp, correctly killed
- Kill criterion 2: "attention adapters degrade single-domain quality >2%" -- tested, -0.61% (improves), correctly passed
- Status: killed. Evidence lines accurately reflect the results.

No inconsistency detected.

## Verdict on the Kill Decision

**The kill decision is CORRECT, but the experiment provides useful directional evidence that should be preserved.**

Three observations support confirming the kill:

1. **The baseline gap is too small to close.** MLP-only composition gap is +0.78%. The attention adapter closes 0.46pp of this, leaving +0.32%. At N=2 with this simple data, there is almost no composition gap to begin with. The experiment correctly identifies this as a "direction right, magnitude wrong" result.

2. **The 20% parameter overhead is not justified.** 4,096 extra parameters for 0.46pp improvement gives an efficiency of ~0.11pp per 1000 params. MLP LoRA at 20,480 params closes the gap from ~4% (base composition) to 0.78%, giving ~0.16pp per 1000 params. Attention LoRA is less parameter-efficient.

3. **The consistent positive signal across all 3 seeds (0.35, 0.38, 0.64pp) suggests this is a real but small effect.** The standard deviation of the improvements is 0.16pp. Even the best seed (0.64pp) does not reach the 1pp threshold. This is not a noise/variance problem -- the effect is real and genuinely small at this scale.

**One challenge to the kill:** The PAPER.md correctly identifies that the MLP-only composition gap (0.78%) is much smaller than the Exp 4 bottleneck measurement (13.5%). This mismatch suggests the bottleneck framing that motivated this experiment may be misleading. The 13.5% was measured under capsule composition without shared attention, not LoRA composition. The experiment arguably tested the wrong bottleneck -- the real bottleneck for LoRA composition at N=2 is not attention patterns but something else (possibly the function-space gap inherent to nonlinear composition). This does not change the kill decision but should inform future hypothesis design.

## Macro-Scale Risks (advisory)

If this direction were revisited at macro scale:

1. **Attention adapter inference cost.** At micro scale, the per-expert attention loop (lines 262-267) iterates over N_experts with full matmuls. At macro scale with N=8+ experts, this would be prohibitively expensive. The MLP routing can batch experts; attention routing through softmax cannot be easily batched because the attention pattern depends nonlinearly on the composed Q/K.

2. **Rank-4 at d=896+ may be more effective.** At d=64, rank 4 captures 6.25% of the subspace. At d=896, rank 4 captures 0.45%, which is a much more targeted intervention. The ratio of attention adaptation capacity to base model capacity would be different.

3. **BPE tokenization creates genuinely different attention patterns.** Character-level a-m vs n-z names have near-identical attention structure. Code vs prose with BPE would show much more domain-specific attention patterns, potentially amplifying the effect.

## Verdict

**KILL -- confirmed.**

The experimenter's kill decision is correct. The effect is real (+0.46pp, consistent across 3 seeds) but below the pre-registered 1pp threshold with no plausible path to reaching it at micro scale. The 20% parameter overhead is not justified. The root cause analysis in PAPER.md is honest and insightful -- the composition gap at N=2 with LoRA is already small (0.78%), leaving little room for attention adaptation to matter.

The directional signal should be noted in FINDINGS.md for potential macro-scale revisitation at N=5+ with BPE tokenization, where both the composition gap and attention pattern diversity would be larger.
