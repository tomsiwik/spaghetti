# Peer Review: Tiny Routing Heads

## NotebookLM Findings

Skipped (not available in this session). Review proceeds from direct code and document analysis.

## Mathematical Soundness

### Parameter count derivation: CORRECT
The formula `params_i = h_head * (d + 2) + 1` correctly accounts for W1 (d x h_head), b1 (h_head), W2 (h_head x 1), b2 (1). For d=2560, h_head=32: 32 * 2562 + 1 = 81,985. Verified in code and results.json.

### FLOPs analysis: CORRECT but MISLEADING
The MATH.md correctly computes head FLOPs at ~820K vs base ~2.38B, yielding 0.034% overhead. However, the actual experiment measures wall-clock time (0.86ms / 37.01ms = 2.32%), which is 68x higher than the FLOP ratio predicts. This is expected (kernel launch overhead dominates for tiny ops) but the paper conflates the two. The 2.32% is the honest number; the 0.034% FLOP estimate is misleading. **Minor issue -- the paper does report the wall-clock number as K2.**

### K2 overhead definition: PROBLEMATIC
K2 measures head MLP inference time against base forward pass time. But in the actual routing pipeline (Phase 3, lines 626-667), there is a FULL forward pass through the base model to get hidden states (`get_hidden_states`), then the heads score, then a SECOND forward pass with composed weights. The actual overhead is ~100% (two forward passes), not 2.32%. The paper acknowledges this at MATH.md lines 119-129 but buries it. K2 as defined measures head *computation* overhead, not *routing pipeline* overhead. This is defensible as a mechanism test but must be clearly flagged -- the routing heads themselves are cheap, but the current architecture doubles inference cost. The MATH.md mentions early-layer routing as a mitigation but does not test it.

### Score-weighted composition: SOUND but TRIVIAL given results
The pre-merge formula `W_composed = sum_i (s_i / sum s_j) * B_i @ A_i` is mathematically clean. But since heads achieve 100% accuracy with near-binary outputs (own domain: logit > 0, others: logit < 0), the top-2 selection always picks the correct domain adapter plus one other. The score-weighted merge then gives ~50/50 weight to the correct adapter and one irrelevant one. The near-oracle PPL (6.42 vs 6.41) suggests the irrelevant adapter contributes near-zero delta, which is consistent with the Grassmannian orthogonality guarantees already proven in this project. The routing is working, but its contribution is modest -- it is recovering what individual oracle routing gives, not discovering beneficial cross-domain compositions.

### Training loss plateau at ~0.5: CONCERNING
All five heads converge to BCE loss ~0.506-0.516 despite 100% accuracy. BCE of 0.5 corresponds to the sigmoid output being ~0.38 for the correct class on average (since -ln(0.5+epsilon) per sample approaches 0.5). This means the logits are very close to zero -- the decision boundary is razor-thin. The heads are discriminating correctly but with low confidence. In a harder setting (similar domains, noisy inputs), this margin would collapse. The paper acknowledges this (Limitation 6) but underestimates its severity. At 500 training steps with 80 positive and ~320 negative samples, the heads may be undertrained.

## Novelty Assessment

### Prior art comparison: ADEQUATE
The paper cites MoLoRA, X-LoRA, LD-MoLE, and L2R. The per-adapter independent head concept is genuinely different from centralized routers. The closest published work is CLONE (arxiv 2506.02847) which uses MoE routing for dynamic LoRA on edge devices, but CLONE uses a shared router, not per-adapter independent heads. The decentralized "each adapter carries its own head" pattern is a reasonable contribution.

### Delta over centralized routing (bitnet_per_token_routing)
The comparison table in PAPER.md (lines 98-106) is the paper's weakest section. It compares N=5 per-adapter heads against N=15 centralized routing, then claims "higher improvement (19.9% vs 13.9%)." The paper does caveat this ("different N"), but the comparison is still misleading:
- At N=5, uniform dilution is mild (each adapter gets 20% weight). At N=15, each gets 6.7%. The bar for beating uniform is much higher at N=15.
- The 19.9% improvement is inflated by the legal domain (uniform PPL 20.44 vs routed 16.58), which has the highest base PPL and thus amplifies relative gains.
- A fairer test would be the same N for both approaches.

### Findings history: IMPORTANT CONTEXT
The FINDINGS.md records several routing kills at micro scale:
- content_aware_routing: MLP classifier 8.5% accuracy, KILLED
- MoTE-SOLE routing: routed PPL WORSE than equal-weight, KILLED
- mote_sole_architecture: router accuracy 50.1%, KILLED

This experiment succeeds where those failed. The critical difference: this experiment uses a REAL 2.4B base model (BitNet-2B-4T) whereas the killed experiments used toy ~200K models where expert specialization was insufficient. The routing heads work BECAUSE the base model's hidden states at scale carry strong domain signals. This is a legitimate finding but means the result is about base model representation quality, not about the routing architecture itself.

## Experimental Design

### Domain separability: TOO EASY
Python code, math, medical text, legal text, and creative writing are maximally distinct domains. A bag-of-words classifier would likely achieve near-100% accuracy on these. The 100% classification accuracy does not validate the routing head architecture -- it validates that the base model separates these domains in its representation space. The real test (acknowledged in Limitations) is closely related domains.

### Evaluation set overlap with training: POTENTIAL LEAK
The hidden states for head training (Phase 2) are extracted from the BASE model (no LoRA adapters applied -- line 392 shows `replace_bitlinear_with_linear` but NOT `apply_lora_to_model`). However, the routing at evaluation time (Phase 3, line 628) ALSO uses base model hidden states (the model has LoRA applied but `get_hidden_states` runs through the model layers which include LoRA). Wait -- examining more carefully: in Phase 3, `apply_lora_to_model` is called at line 556, but `zero_adapter_in_model` is called at line 666 after each sample. The hidden states are computed with whatever adapter state the model has at that point. At line 628, the model has LoRA layers but they were zeroed at line 666 of the previous iteration (or the initial state). Actually, `get_hidden_states` at line 628 occurs BEFORE `apply_adapter_to_model` at line 657, and after `zero_adapter_in_model` at line 666. So the hidden states during eval ARE from a zeroed-LoRA model, which is effectively the base model. This matches the training setup. **No leak.**

### Single seed, no variance estimates: ACCEPTABLE given margins
100% accuracy and 19.9% improvement are large enough that seed variance is unlikely to flip the verdict. However, the paper should state this explicitly rather than just noting "single seed" as a limitation.

### Comparison to oracle is unfair to uniform
The "near-oracle" claim (6.42 vs 6.41) is impressive but expected: with perfect routing, top-2 selection picks the correct domain adapter every time. The second adapter in top-2 is irrelevant noise, but orthogonal adapters contribute minimal interference. This is a validation of Grassmannian orthogonality, not of routing quality.

### K2 timing methodology: ADEQUATE
20 iterations with 3 warmup rounds. Wall-clock measurement on MLX. The 37ms base forward and 0.86ms head overhead are reasonable for BitNet-2B on M5 Pro.

## Hypothesis Graph Consistency

No HYPOTHESES.yml found containing this experiment. The kill criteria (K1-K3) and success criteria (S1-S3) are well-defined in the code and paper, and the experiment tests exactly what it claims.

## Macro-Scale Risks (advisory)

1. **Two-pass overhead.** The current architecture requires a full base model forward pass for hidden state extraction, then a second pass with composed weights. At macro scale, this 2x cost is the real bottleneck, not the 2.32% head overhead. Early-layer routing (mentioned in MATH.md) is essential but untested.

2. **Domain count scaling.** At N=50+ with similar domains, 100% accuracy will degrade. The low-confidence logits (loss ~0.5) predict rapid degradation. Need to test N=15+ with confusable domains (Python/JavaScript, cardiology/oncology).

3. **Score calibration across independently trained heads.** Different heads may have different sigmoid output ranges. The current experiment shows this works when all heads are trained on the same data distribution and base model, but heads trained at different times or on different data may have systematically different score magnitudes, biasing top-k selection.

4. **Out-of-distribution inputs.** What do heads output on text from a domain none of them were trained on? If all heads output ~0.5 (low confidence), the top-2 selection becomes arbitrary. Need a "none of the above" rejection mechanism.

5. **Mixed-domain sequences.** Mean pooling over the full sequence loses token-level domain information. A sequence mixing Python and medical text will confuse sequence-level routing. Per-token routing (acknowledged limitation) is needed.

## Verdict

**PROCEED**

The experiment is well-designed within its micro constraints, the math is sound, the code correctly implements what the paper claims, and the results are genuine. The mechanism works in principle: per-adapter binary heads can route inputs to the correct domain expert with trivial overhead.

However, the following caveats should be addressed before claiming this as a validated component:

1. The domains are trivially separable. The 100% accuracy says more about base model representations than about routing head architecture. The next experiment MUST test with N >= 10 and confusable domains (Python/JavaScript, subspecialty medical, etc.).

2. The "near-oracle" result is primarily a validation of Grassmannian orthogonality (irrelevant adapters cause no interference), not of routing quality. Frame it as such.

3. The K2 overhead metric (2.32%) is honest for head computation but obscures the real 2x inference cost of the two-pass architecture. The early-layer routing optimization is critical path and should be the next experiment.

4. The comparison with centralized routing at different N is misleading and should be removed or replaced with a same-N comparison.

5. The training loss plateau at ~0.5 is a yellow flag for scaling to harder domains. Consider longer training or margin-based losses (hinge loss) that explicitly maximize the decision boundary gap.
