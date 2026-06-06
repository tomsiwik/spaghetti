# Peer Review: Hierarchical Capsule Tree

## NotebookLM Findings

Skipped -- NotebookLM deep review not executed for this review cycle. The experiment is straightforward enough that direct code and math inspection suffice.

## Mathematical Soundness

### What holds

1. **Leaf probability normalization (Section 2.2).** The inductive proof that leaf probabilities sum to 1 is correct. Each internal node splits its incoming mass into p and (1-p), preserving total mass at every depth. Verified in code (`test_leaf_probs_sum_to_one` passes). The worked example in Section 2.3 is arithmetically correct.

2. **Parameter count (Section 4).** Tree: 7*(64+1) = 455 routing params/layer vs flat: 8*64 = 512. The capsule payload is identical (2*d*P = 32,768 per layer for both). Total model counts (203,932 vs 204,160) are consistent with the formula. The 228-param difference (0.1%) is correctly identified as negligible.

3. **FLOPs analysis (Section 5.2).** The per-gate FLOP count (2d+1 per gate, 7 gates = 903) and per-leaf computation (4*d*n_c = 8,192) are correctly derived. The comparison showing routing cost parity (903 vs 1,024) with identical leaf compute is sound.

4. **Balance loss (Section 6.1).** Identical formula to flat CapsulePool, correctly applied to leaf probabilities. Minimum at uniform confirmed.

### Issues found

5. **Gate entropy loss mismatch between MATH.md and code.** MATH.md Section 6.2 describes gate entropy loss as the entropy of the full leaf distribution, with minimization to push toward deterministic routing. The code at `/Users/tom/Code/tomsiwik/llm/micro/models/hierarchical_tree/hierarchical_tree.py` lines 170-190 confirms it computes leaf-distribution entropy, not per-gate binary entropy. The MATH.md title says "Gate Entropy Loss" but the formula is leaf-distribution entropy. These are different quantities:
   - Per-gate binary entropy: H(g_i) = -g_i*log(g_i) - (1-g_i)*log(1-g_i), summed over gates
   - Leaf-distribution entropy: H(leaf) = -sum_l P(l)*log(P(l))

   They are related (leaf entropy <= sum of gate entropies by the chain rule) but not equal. The naming is misleading. The implementation is internally consistent (code matches MATH.md formula), so this is a documentation issue, not a bug.

6. **Aux loss asymmetry is a confound.** The tree model's `aux_loss()` at line 252-258 of `hierarchical_tree.py` returns `0.01 * sum(balance_loss + 0.1 * gate_entropy_loss)`. The flat model's `aux_loss()` at line 153-157 of `capsule_moe.py` returns only `0.01 * sum(balance_loss)`. The tree receives an additional entropy regularization term that the flat baseline does not. This is a confound: the -0.87% quality improvement could partly be due to the extra regularization, not the tree structure itself. An ablation removing the entropy term from the tree (or adding an equivalent entropy term to the flat model) would isolate the structural contribution.

7. **Beam search is not actual beam search.** The code at `hierarchical_tree.py` lines 88-132 computes ALL leaf probabilities by iterating through all 8 leaves, then does top-k selection. This is mathematically equivalent to the described beam search but has O(L) cost, not the claimed O(B*D) cost. The paper acknowledges this in Limitations (item 2) and in MATH.md Section 5.2 ("the micro implementation computes all leaves"). The O(B*log(L)) advantage is theoretical and untested. This is acceptable at micro scale but should be clearly labeled as "full enumeration with top-k selection" rather than "beam search."

## Novelty Assessment

### Prior art

The paper correctly cites the three most relevant predecessors:

1. **Hierarchical Mixtures of Experts (Jordan & Jacobs, 1994).** The original HME paper describes exactly this architecture: a tree of experts with gating networks at internal nodes. The current experiment is a modern instantiation (sigmoid gates, ReLU capsules, beam selection). The paper honestly acknowledges this: "modern instantiation of HME."

2. **Fast Feedforward Networks (Belcak & Wattenhofer, ICML 2024).** Binary tree of feedforward layers with learned conditional execution. Very close to this work. The key difference (capsule groups vs full FFN layers) is correctly identified.

3. **ExpertFuse/ExpertZIP (2025).** Huffman-coded expert merging tree. Correctly identified as follow-up work.

### Delta over existing work

The delta is narrow but real: applying HME-style tree routing to the capsule composition protocol (shared-base + fine-tune + weight-average + calibrate). No prior work tests tree-structured expert routing for independent composition of domain-adapted capsule groups. The novelty is in the composition protocol application, not the tree routing itself.

### References check

The `references/` directory does not contain a dedicated folder for HME or Fast Feedforward Networks, though they are cited in the paper. No existing reference implementations cover tree-structured routing, so there is no reinvention concern.

## Experimental Design

### Strengths

1. **Fair parameter matching.** 203,932 tree vs 204,160 flat (0.1% difference). The capsule payload is identical; only the routing mechanism differs. This is a clean comparison.

2. **Same training infrastructure.** Both models use the same `train_model()` function, same optimizer (Adam, lr=3e-3), same data, same seeds, same step counts. The composition protocol is identical (pretrain -> fine-tune -> weight-average -> calibrate).

3. **Kill criteria are testable and tested.** Both criteria are clearly stated and the experiment reports pass/fail against them.

4. **Routing diagnostics.** The normalized entropy measurement (0.745) and leaf utilization analysis provide useful signal about whether the tree is actually routing or acting as a uniform mixture.

### Weaknesses

5. **Only 3 seeds.** The -0.87% quality improvement and -0.09pp composition gap improvement are small effects. With 3 seeds, the standard error is substantial. Per-seed results for single-domain:
   - Flat: 0.5133, 0.5170, 0.5365 (range: 0.023)
   - Tree: 0.5086, 0.5171, 0.5274 (range: 0.019)

   The ranges overlap. The tree "wins all 3 seeds" claim is true but barely -- seed 2 is 0.5171 vs 0.5170, a difference of 0.0001. This is not statistically significant. The paper should not claim "tree matches or beats flat on all 3 seeds" as if it is evidence of superiority. The correct conclusion is "tree and flat are indistinguishable at 3 seeds."

   However: the kill criterion is "tree worse than flat," not "tree better than flat." The experiment correctly demonstrates that tree is NOT worse, which is sufficient for PROCEED.

6. **Composition uses weight averaging, not concatenation.** The paper acknowledges this (Limitation 3) but it weakens the "hierarchical structure preserves more structure during composition" claim (Section 7.2). Weight averaging blends all parameters uniformly regardless of tree structure -- the tree topology provides no advantage during averaging. The tree's structural advantage for composition would only manifest with concatenation + subtree grafting, which is untested.

7. **Aux loss confound (elaborated).** As noted in item 6 of Mathematical Soundness, the tree has an extra entropy regularization term. At coefficient 0.1 * 0.01 = 0.001 relative to CE loss, this is small but nonzero. A clean experiment would either (a) remove the entropy term from the tree, or (b) add an equivalent entropy term to the flat model's softmax distribution. Without this, we cannot attribute the improvement to tree structure vs. regularization.

8. **Calibration step only trains gates/router, not leaves.** During composition calibration (lines 280-288 of `run_experiment.py`), the experiment freezes leaves and only trains gates (tree) or router (flat). This is correct for the composition protocol. But for the flat model, the "router" is a single linear layer (512 params per layer), while for the tree, "gates" are 7 independent binary classifiers (455 params per layer). The calibration capacity is comparable in parameter count but different in expressiveness: 7 independent sigmoid gates vs 1 softmax over 8 classes. This is an inherent design difference, not a confound.

9. **No statistical significance test.** With 3 seeds producing overlapping distributions, a paired t-test or Wilcoxon signed-rank test should be reported. The paper does not compute p-values or confidence intervals.

## Hypothesis Graph Consistency

The experiment matches `exp_hierarchical_capsule_tree` in HYPOTHESES.yml. Kill criteria:
- "tree-routed composition degrades >5% vs flat softmax composition" -- tested, passes (+0.17% gap)
- "tree routing quality (perplexity) worse than flat softmax at same active params" -- tested, passes (-0.87%)

The node correctly blocks `exp_huffman_pruning` and `exp_splay_adaptive_routing`. Status is set to "proven." Given that both kill criteria pass and the mechanism works in principle, this status is justified.

## Integration Risk

The tree architecture integrates cleanly with the existing capsule infrastructure:
- Reuses `CapsuleGroup` from `capsule_moe` (no reinvention)
- Reuses `CausalSelfAttention`, `RMSNorm` from `gpt`
- Same composition protocol (weight averaging + calibration)
- Same `ntp_loss` + `aux_loss` training loop

The tree adds one new structural element (binary gates) that is simple and well-tested. No integration conflicts with VISION.md.

## Macro-Scale Risks (advisory)

1. **Gradient vanishing through chained sigmoids.** At depth D=3, the leaf probability is a product of 3 sigmoid outputs. Each sigmoid saturates near 0 or 1, where gradients vanish. At D=5+ (32 leaves), this becomes a product of 5 near-zero gradients. The paper acknowledges this (Macro Kill #2) but offers no mitigation. Gumbel-sigmoid or straight-through estimators may be needed.

2. **Tree structure locks in topology.** A fixed binary tree imposes that leaf 0 and leaf 1 are siblings (share a parent gate), regardless of whether their learned specializations are related. If the optimal expert clustering is not binary-hierarchical, the tree wastes capacity on meaningless gate decisions. Flat routing has no such constraint. This is the core risk the Huffman follow-up addresses.

3. **Weight averaging erases tree structure.** Since composition uses weight averaging (not grafting), the hierarchical prior provides no benefit during composition. The tree's claimed composition advantage (Section 7.2) is theoretical speculation unsupported by the experimental protocol. For the composition claim to be validated, the experiment needs concatenation-based composition with subtree grafting.

4. **Calibration asymmetry at scale.** At D=8 (256 leaves), the tree has 255 binary gates to calibrate vs a flat router with 256*d params. Gate calibration may be harder because gates at different depths have different gradient magnitudes (deeper gates see attenuated gradients through the sigmoid chain).

## Verdict

**PROCEED**

The experiment achieves what it set out to do: demonstrate that binary tree routing can replace flat softmax routing without quality degradation. Both kill criteria pass. The math is sound. The implementation correctly matches the mathematical description. The parameter comparison is fair.

The improvement claims (-0.87% quality, -0.09pp composition gap) should be interpreted as "tree is not worse than flat" rather than "tree is better," given 3 seeds with overlapping ranges. The paper's Limitations section appropriately hedges these claims.

Recommended improvements (non-blocking):

1. Add an ablation removing the gate entropy loss (0.1 coefficient) from the tree model to isolate the structural contribution from the regularization effect. This is the most important missing control.

2. Report a paired statistical test (paired t-test across 3 seeds) to quantify whether the -0.87% improvement is significant. With 3 seeds, it almost certainly is not, and the paper should state this explicitly.

3. Rename "Gate Entropy Loss" to "Leaf Distribution Entropy Loss" in MATH.md Section 6.2, or change the implementation to compute per-gate binary entropy. The current naming is misleading.

4. Soften the "tree matches or beats flat on all 3 seeds" language. Seed 2 is 0.5171 vs 0.5170 -- this is noise, not a win.
