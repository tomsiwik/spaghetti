# Peer Review: split_leaf_actual

## NotebookLM Findings

Skipped -- the experiment is small and self-contained enough that manual review is sufficient. The mathematical claim is a partition identity, which can be verified by inspection.

## Mathematical Soundness

### The function preservation proof (Section 2.2) is correct.

The claim: at sigma=0, f_child0(x) + f_child1(x) = f_parent(x).

Verification:

1. CapsuleGroup forward: `B @ ReLU(A @ x)` where A is (n_caps, d), B is (d, n_caps).
2. This decomposes as sum_{j} b_j * ReLU(a_j^T x) where b_j is column j of B and a_j^T is row j of A.
3. Child0 gets rows 0..half-1 of A and columns 0..half-1 of B. Child1 gets the rest.
4. Since ReLU is element-wise per capsule, child0 computes sum_{j=0}^{half-1} and child1 computes sum_{j=half}^{n_caps-1}. The two sums partition the full sum.

This is trivially correct. There is no hidden assumption -- it follows from the additive structure of the two-layer ReLU network. The proof is sound.

### The noise perturbation analysis (Section 2.3) is reasonable but informal.

The claimed error scaling E ~ O(sigma * sqrt(n_c) * ||x|| / ||f_parent||) is a rough order-of-magnitude estimate, not a bound. The empirical calibration (Section 2.3 table) is the real evidence. The "worked example" in Section 5 estimates ~0.3% and observes 0.69%, which is close enough given the hand-wavy nature of the approximation.

No mathematical errors found. The analysis correctly identifies the two error sources (boundary flips and continuous perturbation).

### Implementation matches the math.

Verified in `split_leaf_actual.py` lines 90-94:
- A_child0 = A_parent[:half] + noise -- correct (first half of capsule rows)
- B_child0 = B_parent[:, :half] + noise -- correct (first half of capsule columns)
- Same for child1 with remaining indices.

The `measure_function_preservation` function (lines 163-248) correctly recomputes the parent's output from saved weights and compares against children's summed output. The manual matrix multiplications match the CapsuleGroup forward pass.

### One subtle issue: tree routing weights are not addressed.

In the actual HierarchicalCapsuleTree forward pass (hierarchical_tree.py lines 134-157), leaf outputs are combined via weighted sum with routing probabilities. After split with the gate at 50/50 and beam=2 selecting both children:

- The model's contribution from the split pair is: 0.5 * f_child0(x) + 0.5 * f_child1(x) = 0.5 * f_parent(x)
- Before split, the parent leaf contributed: p_leaf0 * f_parent(x)

where p_leaf0 was the routing probability for leaf 0. After split, the combined contribution is halved (0.5) relative to whatever the raw leaf output was, because the 50/50 gate renormalizes.

However, this issue does NOT invalidate KC1 (which correctly measures the raw capsule output sum) or KC2 (which compares split vs independent after fine-tuning, so routing adapts). It is a transient effect at the moment of split that fine-tuning quickly corrects. The paper should mention it but it is not a fundamental problem.

Additionally, the split overwrites the original leaf 1 (sibling). This means the model loses leaf 1's function entirely. The paper does not explicitly discuss this trade-off, though it is implicit in the experimental setup (the KC2 comparison also replaces leaves 0 and 1, so the comparison is fair).

## Novelty Assessment

### Prior art: This is a standard technique.

Splitting a ReLU network neuron by partitioning into two groups is well-known in the network growing / neural architecture search literature. The additive decomposition of ReLU networks (output = sum of per-neuron contributions) is a textbook property. "Net2Net: Accelerating Learning via Knowledge Transfer" (Chen et al., 2015) describes a "wider" operation that splits neurons with function preservation, though their approach duplicates neurons rather than partitioning them.

The specific application to capsule groups in a hierarchical MoE tree is novel in combination, but the core mathematical mechanism (partition capsules, get exact function preservation) is a direct consequence of the linear structure after ReLU.

### Delta over existing work:

The contribution is not the split operation itself (which is elementary) but:
1. Calibrating noise_scale for the capsule tree context (sigma=0.001 recommendation)
2. Empirically confirming split matches independent training at convergence
3. Demonstrating early convergence advantage (directional)

This is appropriate for a micro-experiment validating a building block.

### No reinvention of existing code detected.

The experiment correctly builds on hierarchical_tree and capsule_moe. The split_freeze_protocol implemented split_leaf but never invoked it; this experiment fills that gap.

## Experimental Design

### KC1 design is sound.

Testing function preservation at multiple noise levels (0, 0.001, 0.01, 0.05) with 3 seeds and 20 batches each is adequate for the claim. The zero-noise case is a mathematical identity that serves as a sanity check. The practical noise level (0.001) is well below the 5% threshold with 7x margin.

### KC2 design has one weakness: both conditions fine-tune on mixed data, not domain-specific data.

The PAPER.md Limitations section (point 5) acknowledges this. The original motivation for splitting is to create domain specialists. Fine-tuning both split and independent children on the same mixed data tests only whether inherited weights help optimization, not whether they help specialization. This is a valid limitation but not a fundamental flaw -- the KC2 criterion ("split quality within 5% of independent") is met, and the domain-specialization question is correctly identified as future work.

### KC2 capacity matching is correct.

Both split and independent conditions use half-size leaves (16 capsules each, 16,644 total trainable params). Verified in code: the independent condition creates new CapsuleGroup(d, half) modules and resets the parent gate to 50/50. Fair comparison.

### KC3 is correctly labeled as directional.

No hard kill criterion. The 2/3 seeds showing faster convergence is weak evidence but honestly reported. The paper does not overclaim.

### One missing control: the paper does not test what happens to the model's overall quality after split.

KC1 tests whether f_c0 + f_c1 = f_parent (it does). KC2 tests whether fine-tuning recovers quality (it does). But neither tests the model's val_loss immediately after split, before fine-tuning. The base val_loss is reported (0.5277) and the post-fine-tuning split val_loss is (0.5187), but the intermediate "just-split, no fine-tuning" val_loss is not reported. This would show the practical cost of splitting in the full model context (including the loss of original leaf 1 and the routing weight issue noted above). Not blocking, but would strengthen the narrative.

## Hypothesis Graph Consistency

The HYPOTHESES.yml node `exp_split_leaf_actual` has:
- Kill criteria: (1) split quality >5% worse than independent, (2) function preservation error >5%
- Status: proven
- Evidence matches the paper's findings

The experiment correctly targets its stated kill criteria and the evidence is sufficient to mark the node as proven.

## Macro-Scale Risks (advisory)

1. **Noise scale recalibration at macro dimensions.** The error formula suggests E ~ sigma * sqrt(n_c). At macro scale with n_c=256, the error at sigma=0.001 would be ~sqrt(256/32) * 0.69% = ~2.0%. Still under 5% but with less margin. May need to reduce sigma further, which could hurt symmetry breaking. Not blocking.

2. **Sibling sacrifice.** At macro scale, the overwritten sibling leaf may have accumulated significant specialization. The paper's implicit assumption that leaf 1 is expendable needs explicit justification at macro scale. A "grow the tree" operation (add new leaf positions rather than overwrite) would be more natural but architecturally harder.

3. **Convergence advantage at scale.** The 2/3 seeds result is weak. The macro-scale prediction (Section 4.2 of MATH.md) is plausible but untested. The claim that inherited features help more at higher dimensions is reasonable but speculative.

4. **Recursive splitting depth.** Splitting from 32 to 16 capsules per child is fine. Splitting further to 8, then 4, may produce underparameterized leaves. The minimum viable capsule count is unknown.

## Verdict

**PROCEED**

The experiment does what it claims: validates that the split_leaf mechanism preserves function exactly at zero noise (a mathematical identity) and produces children that match independently-trained leaves after fine-tuning (+0.16%, 31x margin on the 5% threshold). The math is correct, the implementation matches the math, the experimental design is sound within micro-scale constraints, and the limitations are honestly reported.

Minor items that would strengthen the paper but are not blocking:

1. Add a note about the routing weight effect: after split, the tree's contribution from the split pair is 0.5 * f_parent(x) due to the 50/50 gate normalization, not 1.0 * f_parent(x). Fine-tuning corrects this, but it means the model's output is not preserved at the moment of split even at zero noise -- only the raw capsule sum is.

2. Report the model's val_loss immediately after split (before fine-tuning) as an additional data point.

3. Acknowledge the sibling sacrifice more explicitly: splitting leaf 0 overwrites leaf 1, losing its function. This is a design choice, not a flaw, but should be stated.

These are documentation improvements, not experimental revisions. The mechanism is validated.
