# Peer Review: Mixed-Domain Sequences

## Mathematical Soundness

**MATH.md derivations are correct.** The separability metric, kill criteria formalization, and per-token routing equations are standard and properly stated. The boundary accuracy metric (BA) as the macro-average of per-segment correct-expert fractions is a reasonable choice. The 1/N = 0.2 random baseline is correct for 5 experts.

**One subtle issue with the BA computation.** The threshold of 0.4 (2x random) is reasonable, but the *average* across pairs is misleading because of bimodal behavior. Seven pairs cluster near 0.50 (python/math always detected as domain A, 100% on A side, 0% on B side), one pair hits 97%, and three pairs hit 0%. Averaging these regimes together produces 39.71%, which is 0.29% below the 40% threshold -- a near-miss that obscures the real story. The actual finding is that the router learned a 2-class detector (python/math vs everything-else), not a 5-class domain router. This is important and the paper correctly identifies it, but the K2 number nearly passing by accident would be a concern if it had passed.

**Assumption 1 in MATH.md is violated by the evaluation.** The paper assumes "hidden states at position t primarily reflect the local domain (not corrupted by cross-attention to tokens from the other domain)." The evaluation methodology runs full forward passes through the entire mixed sequence, so self-attention between segments contaminates hidden states -- exactly violating this assumption. The paper acknowledges this in the Limitations section. This is a real confound but it is *inherent to the architecture*, not a fixable evaluation bug. Any real deployment would have the same contamination.

## Novelty Assessment

**No novel contribution here; this is a well-executed negative result.** The experiment directly follows up on the MoLoRA per-token null result (exp_molora_per_token_mlx) by testing on the data regime where per-token routing should maximally help. The Gumbel-sigmoid router is from L2R, the LoRA composition is standard, and the mixed-domain concatenation is a textbook experimental setup. The value is in the rigorous falsification, not novelty.

**Prior art check.** MoLoRA (arXiv:2603.15965) shows per-token routing working on Qwen3-1.7B with 4 adapters. The key difference: MoLoRA trains the router *jointly with the adapters* during fine-tuning, while this experiment trains the router *post-hoc* on frozen adapter hidden states. This is a critical distinction the paper does not emphasize enough. The MoLoRA success likely depends on co-adaptation between router and adapters, which this frozen-adapter setup cannot replicate.

## Experimental Design

**The experiment tests the right hypothesis with adequate controls.** Four conditions (uniform, per-sequence, per-token, oracle), 10 domain pairs, 20 sequences each, 200 total evaluation sequences. The oracle provides an upper bound that proves the gap exists (18.4% over per-sequence). The uniform provides a lower bound. The experiment is well-bracketed.

**Critical flaw in the per-token evaluation methodology.** Lines 640-667 of run_experiment.py show that per-token routing runs a *full forward pass* for each unique expert set, then scores only the tokens assigned to that set. This means:

1. Tokens in segment A still attend to tokens in segment B via self-attention, even when segment B's tokens are being served by the wrong adapter (the one selected for segment A's group).
2. This is not just an approximation -- it systematically penalizes per-token routing relative to per-sequence routing, because per-sequence routing at least applies a consistent (if suboptimal) adapter to the whole sequence.

The python+math pair illustrates this perfectly: 97% boundary detection accuracy, yet -6.4% *worse* than per-sequence. The router correctly identifies which tokens need which expert, but the evaluation method contaminates the result. The paper correctly identifies this as "cross-attention contamination."

However, I do NOT consider this a fatal flaw for the kill decision, because:
- This contamination is inherent to any transformer-based per-token routing with pre-merge composition
- In production, you would face the same problem -- you cannot run separate forward passes per segment without a segment-detection step first
- If the mechanism requires segment-level isolation to work, then "per-token routing" as a concept is misframed; the real mechanism needed is "segment-level routing," which is a different architecture

**The oracle evaluation has the SAME contamination problem** (lines 669-710). The oracle runs the full sequence through domain-A's adapter, scores only the first-half tokens, then runs the full sequence through domain-B's adapter and scores the second-half tokens. Self-attention from the wrong-adapter half still contaminates the right-adapter half. This means the oracle gap (18.4%) is an *underestimate* of the true per-segment routing value, which actually strengthens the argument that correct routing has value -- but capturing that value requires architecture changes, not just better routing.

**Router training is adequate but minimal.** 800 steps on ~230 samples (150 pure + 80 mixed). The router clearly learned *something* -- it separates python and math from the rest -- but the 3 prose domains collapsed. More training or a contrastive objective could potentially help, but given that the hidden-state representations of medical/legal/creative are genuinely similar in BitNet-2B-4T's last-layer space, this is likely a representation problem, not a training problem.

**K2 threshold is reasonable.** The 40% threshold (2x random at 20%) is generous. The actual result (39.71%) barely fails, but as discussed above, the average is misleading. A more informative K2 would count "pairs where BA > 40%" -- only 7/10 pass (the python+math and all pairs with python or math as domain A). The 3 prose-only pairs score 0%. This reveals the router has learned 2 effective classes, not 5.

## Hypothesis Graph Consistency

**This experiment has no HYPOTHESES.yml node.** The kill criteria IDs referenced in the script header (K1=235, K2=236) do not appear in HYPOTHESES.yml. The related node `exp_bitnet_per_token_routing` has status "supported" and tests per-sequence centralized routing (a different mechanism). This experiment should have its own HYPOTHESES.yml node or be linked as evidence to the existing per-token routing node. The kill criteria IDs appear to be phantom references.

## Kill Decision Verification

**The kill is sound.** Both criteria fail:
- K1: +0.28% improvement (threshold: >=5%) -- clearly fails
- K2: 39.71% boundary accuracy (threshold: >40%) -- marginally fails, but the underlying distribution is bimodal and the 39.71% average masks a 2-class collapse

Combined with the prior null result on homogeneous data (-0.46% from exp_molora_per_token_mlx), this is now a two-experiment kill for post-hoc per-token Gumbel-sigmoid routing.

The paper's analysis of *why* it fails is insightful:
1. Cross-attention contamination is inherent, not fixable within this architecture
2. Router collapse to 2 effective classes (code/math vs prose) reflects BitNet-2B-4T's representation structure
3. The oracle gap proves the value exists but is architecturally unreachable via per-token routing

**One caveat**: This kills *post-hoc* per-token routing (frozen adapters, router trained after). It does NOT kill *jointly-trained* per-token routing (MoLoRA-style). The paper should be more explicit about this distinction.

## Macro-Scale Risks (advisory)

Not applicable -- hypothesis is killed. However, the finding that 3/5 domains are indistinguishable in hidden space is relevant to any future routing work on BitNet-2B-4T. The per-adapter routing heads experiment (100% accuracy) worked because it used per-adapter binary classifiers trained with domain-specific data, sidestepping the multi-class discrimination problem entirely. This architectural insight carries forward.

## Verdict

**PROCEED** (kill confirmed)

The kill decision is correct and well-justified. Both kill criteria fail. The experimental methodology has a known confound (cross-attention contamination) that actually *strengthens* the kill rather than weakening it -- the contamination is inherent to the architecture, not a fixable evaluation bug. The analysis of failure modes (router collapse, cross-attention contamination, oracle gap) is thorough and provides actionable direction for future work.

Minor items for the record (not blocking):

1. Add a HYPOTHESES.yml node for this experiment and link the evidence, or add the evidence to the existing `exp_bitnet_per_token_routing` node as a follow-up kill.
2. Clarify in the paper that this kills *post-hoc* per-token routing specifically. Jointly-trained routing (MoLoRA-style) remains untested in this codebase.
3. The K2 average of 39.71% barely missing 40% is a coincidence of averaging bimodal data. Report the per-class breakdown prominently rather than leading with the average.
