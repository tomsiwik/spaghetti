# Peer Review: bitnet_per_token_routing

## NotebookLM Findings

Skipped (NotebookLM authentication not verified). Review conducted by direct analysis of MATH.md, PAPER.md, run_experiment.py, and results.json.

## Mathematical Soundness

### What holds

1. **Router architecture math is correct.** The two-layer MLP (2560 -> 256 -> 15) produces logits that are softmax-normalized per token, then mean-aggregated across the sequence, then top-k selected and renormalized. The derivation in MATH.md sections 1-4 is consistent with the code.

2. **Parameter cost analysis is correct.** 2560*256 + 256 + 256*15 + 15 = 659,471 params. This is 0.2% of 324M adapter params. Verified against results.json.

3. **Top-k weight normalization is correct.** Top-k weights are renormalized to sum to 1.0 (run_experiment.py line 397), meaning the total adapter contribution has the same magnitude as a single adapter at full strength. This is mathematically equivalent to a convex combination of adapters.

4. **1/N composition is implemented correctly.** Line 572 uses `mx.mean(stacked, axis=0)` which is 1/N scaling. Both uniform and routed paths go through the same LoRALinear with scale=20.0, so the comparison is fair.

### Issues found

**Issue 1 (Moderate): Naming mismatch -- "per-token routing" vs actual per-sequence routing.**

The experiment title, MATH.md title, and HYPOTHESES.yml all say "per-token routing." The implementation actually performs **per-sequence routing** (run_experiment.py lines 346-360, function `compute_routed_loss_topk`): per-token softmax weights are averaged across the sequence, and then a single adapter composition is applied to the entire sequence. The MATH.md section 4 does note "sequence-level aggregation" as a "conservative approximation," but the naming throughout is misleading. This is not MoLoRA-style per-token routing where different tokens within the same sequence can activate different experts. It is closer to standard MoE sequence-level gating.

**Impact:** The claim that this tests "per-token routing" is overclaimed. It tests per-sequence routing informed by per-token router predictions. The distinction matters because the core MoLoRA contribution is that different tokens *within the same sequence* benefit from different experts. This experiment cannot validate or refute that mechanism.

**Issue 2 (Minor): The "beats oracle individual" framing is misleading.**

Average top-2 PPL (13.65) < average individual PPL (15.20). But "individual" here means each adapter applied alone to its own domain at full strength. Four of these adapters are *worse than base* on their own domain (medical: 20.59 vs 18.98 base; code: 3.84 vs 3.78; chemistry: 9.51 vs 9.21; dialogue: 5.89 vs 5.57). The "oracle individual" baseline includes these broken adapters. A fairer comparison would exclude domains where individual adapters regress, or compare against a hypothetical oracle that selects either the adapter or base for each domain. The paper does acknowledge the adapter quality confound but the headline claim "beats oracle!" still overstates.

**Issue 3 (Minor): Arithmetic mean of PPL across domains is scale-sensitive.**

Physics has base PPL of 73.70 and uniform PPL of 46.04, while code has base PPL of 3.78 and uniform PPL of 3.51. An arithmetic mean of PPL weights physics ~20x more than code. The 13.9% average improvement is dominated by physics (-50%) and science (-12%). A geometric mean of PPL or mean of log-PPL would be more robust. Alternatively, reporting per-domain percentage improvements and then averaging those would decouple from PPL scale.

Recomputing: if we take the mean of per-domain percentage improvements (top-2 vs uniform), the domains where top-2 wins average about -17% improvement, while the domains where uniform wins average about +8% degradation. The net depends on how you weight it, but the directional result (top-2 helps overall) survives.

**Issue 4 (Informational): MATH.md section 1 claims "effective adapter magnitude scales as 1/N^2 in terms of squared norm impact."**

This is stated without derivation. The effective delta per adapter is (1/N) * alpha * x * A_i * B_i. The squared norm of this is (alpha/N)^2 * ||x A_i B_i||^2. So the squared norm per adapter scales as 1/N^2 -- correct. But the *total* squared norm of the sum (assuming orthogonality) scales as N * (alpha/N)^2 = alpha^2 / N, so it scales as 1/N not 1/N^2. The claim is technically correct for per-adapter impact but could be misread as total impact. Minor.

## Novelty Assessment

**Prior art:** MoLoRA (2603.15965), X-LoRA (2402.07148), FlyLoRA (2510.08396), LoRA Mixer (2507.00029) all implement per-token or per-layer routing over LoRA adapters. Mixtral (2401.04088) and DeepSeek-V3 (2412.19437) use top-2 routing in production MoE. The finding that top-2 beats top-1 is well-established in the MoE literature.

**Delta over existing work:**
- This is the first test of learned routing over ternary LoRA adapters on BitNet-2B specifically
- The finding that top-2 provides "natural regularization" against adapter overshoot at this scale is useful practical knowledge for the SOLE architecture
- The experiment validates that base model hidden states contain sufficient domain signal for routing (91.7% accuracy) on BitNet-2B specifically

**Novelty is low but the experiment is not *claiming* novelty.** It is validating a known mechanism (MoLoRA-style routing) in the SOLE context. This is appropriate for a micro experiment.

**No reinvention detected.** The references are cited, and the experiment builds on them rather than claiming to invent routing.

## Experimental Design

### Strengths

1. **Clean reuse of existing adapters.** The 15 adapters from bitnet_scale_n15 are loaded without retraining, avoiding confounding router quality with adapter quality.

2. **Three-way comparison (uniform, top-1, top-2)** with base and oracle baselines. This is a thorough comparison matrix.

3. **Kill criteria are well-specified and pre-registered.** K1 (routed > uniform) and K2 (accuracy < 60%) are binary, measurable, and were defined before running.

4. **Router training is honest.** Hidden states from the base model (no adapters), domain labels as supervision, 20% held-out validation. Standard methodology.

### Weaknesses

**W1 (Moderate): No load balancing or auxiliary loss.**

MoLoRA and standard MoE practice include load-balancing losses to prevent routing collapse (all tokens going to 1-2 experts). The router here has no such mechanism. At N=15 with 91.7% accuracy, this works because the classification task itself distributes load. But at N=100+, routing collapse is a known failure mode. The paper does not discuss this.

**W2 (Moderate): Top-2 wins only 8/15 domains.**

The kill criterion (K1) tests *average* PPL, which top-2 passes. But top-2 loses on 7/15 domains. The losses are individually small (1-5% worse) while wins are large (10-50% better). This asymmetry means the arithmetic mean is dominated by the wins. The paper correctly identifies this but does not flag it as a concern for production use: a system that degrades 47% of domains (even slightly) while improving 53% needs a confidence-based fallback, which the paper does recommend. Good.

**W3 (Minor): Single seed on router training.**

The paper acknowledges this. Router training is 2.4 seconds, so 3-seed validation would have cost ~7 additional seconds. This is a missed opportunity for nearly-free robustness evidence.

**W4 (Minor): Adapter quality confound weakens the comparison.**

Four adapters trained for only 400 ternary STE steps are worse than base. This means: (a) the uniform baseline benefits from regularization that a properly-trained adapter pool would not need; (b) top-1 is unfairly penalized by these broken adapters. With better-trained adapters, the uniform vs top-2 gap might shrink or grow -- we cannot tell from this experiment.

## Hypothesis Graph Consistency

The HYPOTHESES.yml node `exp_bitnet_per_token_routing` has:
- K1: "routed composition PPL > 1/N uniform composition PPL (routing hurts)" -- tested, top-2 PASSES
- K2: "router training fails to converge (routing accuracy < 60% on held-out)" -- tested, 91.7% PASSES

The kill criteria match what was tested. The evidence is sufficient for "supported" status.

**One inconsistency:** The HYPOTHESES.yml title says "MoLoRA-style per-token routing" but the implementation is per-sequence routing. The hypothesis should be reworded or the limitation should be noted in the evidence field.

## Integration Risk

The VISION.md architecture diagram already includes "Per-token Router (selects top-k domain experts)" -- this experiment directly validates that component. The 659K parameter router integrates cleanly. No conflicts with existing components.

**Concern:** VISION.md says "Per-token Router" but this experiment only validates per-sequence routing. True per-token routing (different adapters per token in the same forward pass) requires either N forward passes or fused kernel support, which is substantially harder to implement. The experiment provides a lower bound on per-token routing performance, which is fair, but the gap between per-sequence and true per-token is unknown.

## Macro-Scale Risks (advisory)

1. **Load balancing at N=100+.** Without auxiliary loss, routing collapse becomes likely. Must add balancing before scaling.

2. **Router retraining frequency.** Adding/removing adapters invalidates the router. At scale, incremental router updates or router-free methods (FlyLoRA, Union-of-Experts) may be needed.

3. **Latency of two-pass architecture.** Getting hidden states requires a full forward pass before routing can begin. For streaming/real-time serving, this doubles time-to-first-token.

4. **Domain labels may not exist at scale.** The router trains on domain classification, which requires labeled data. With 100+ diverse adapters, clean domain labels may not be available. Self-supervised routing (MoRAM, Union-of-Experts) or router-free approaches should be evaluated.

5. **The per-sequence vs per-token gap.** At macro scale, long documents containing mixed-domain content (e.g., a legal document with code snippets) would benefit from true per-token routing. The current approach would route the entire sequence to legal+finance, missing the code expert.

## Verdict

**PROCEED**

The experiment cleanly demonstrates that top-2 learned routing beats uniform 1/N composition by a meaningful margin (13.9% average PPL) on BitNet-2B with 15 ternary LoRA adapters. The mechanism is sound in principle, the implementation is correct, the kill criteria are met, and the limitations are honestly documented.

Two fixes are recommended (non-blocking):

1. **Rename throughout:** Replace "per-token routing" with "per-sequence routing (from per-token router predictions)" in the PAPER.md title, MATH.md title, and HYPOTHESES.yml title. The current naming overstates what was tested.

2. **Soften the "beats oracle" claim** in PAPER.md Finding 2. The oracle baseline includes 4 broken adapters. Reframe as: "Top-2 routing achieves better average PPL than any single adapter applied to its own domain, partly due to cross-domain transfer and partly because some individual adapters have not converged."

These are documentation fixes, not experimental flaws. The core result -- that top-2 routing with a trivially cheap router beats uniform composition -- is valid and useful for the SOLE architecture.
