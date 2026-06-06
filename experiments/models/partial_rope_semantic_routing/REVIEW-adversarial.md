# Peer Review: Partial RoPE Semantic Routing

## NotebookLM Findings

Skipped -- the experiment is already killed and the analysis is straightforward enough for direct review without external tooling.

## Mathematical Soundness

### Q Dimension Splitting (Question 1)

The MATH.md correctly states that BitNet-2B-4T applies full RoPE to all 128 head dimensions. The "partial RoPE" split (first 32 = "RoPE," last 96 = "free") is explicitly labeled a simulation -- there is no actual partial RoPE in the model. This is clearly stated and the experiment is honest about it.

However, the split is semantically meaningless in a full-RoPE model, and the PAPER.md correctly identifies this as a key reason for failure. In a full-RoPE model, the Q projection W_Q mixes all hidden dimensions equally across all head dimensions. There is no architectural reason for any subset of Q dimensions to be "more semantic" or "less positional" than another. The Parameter Golf paper (arXiv 2506.06105) finding applies to models *trained* with partial RoPE where the optimizer *learns* to use free dims for semantic content. Simulating this split post-hoc on a full-RoPE model is not testing the Parameter Golf hypothesis -- it is testing something weaker that the paper already predicts should fail.

**Assessment:** The split implementation is technically correct (dimensions 0:32 vs 32:128 per head), but the experiment cannot test its stated hypothesis because the model was not trained with partial RoPE. This is a fundamental experimental design flaw, not a mathematical error.

### Code Comments vs Actual Shapes

There is a discrepancy between code comments and actual tensor shapes. Line 171 comments `(B, L, 32, 80)` but `attn.n_heads` is 20 and head_dim is 128, producing `(B, L, 20, 128)`. Lines 233-234 comment `(B, L, 32, 20)` and `(B, L, 32, 60)` but actual shapes are `(B, L, 20, 32)` and `(B, L, 20, 96)`. The *code logic* is correct (it uses `attn.n_heads` and the ROPE_DIMS/FREE_DIMS constants), and the final pooled dimensions (640 and 1920) are numerically correct. The wrong comments appear to be artifacts from an earlier architecture assumption (possibly 32 heads with head_dim=80, which also yields hidden_dim=2560). This does not affect results but indicates sloppy verification.

### Silhouette Distance Metric (Question 2)

The silhouette computation at line 313 uses sklearn's default metric (Euclidean) after z-score normalization. For 1920-dimensional features with 1200 samples, Euclidean distance in high dimensions suffers from concentration of distances -- all pairwise distances converge, making silhouette scores near zero by construction. This is a well-known phenomenon (Beyer et al., 1999; Aggarwal et al., 2001).

The negative silhouette (-0.007 to -0.028) is consistent with high-dimensional distance concentration rather than definitively proving "no domain structure exists." Cosine similarity or dimensionality reduction (PCA to 50-100 dims, then Euclidean) would be more appropriate for this feature space. However, because ALL feature types show negative silhouette (including full hidden states at -0.028), this is a systematic issue, not one that selectively penalizes Q features. The *relative* ranking across feature types remains informative.

**Impact on kill decision:** Minor. Even with a better distance metric, the silhouette would likely remain far below 0.3. The centroid accuracy numbers (which are less affected by distance concentration) tell a consistent story: Q features provide weaker domain signal than hidden states.

### Mean-Pooling Validity (Question 3)

Mean-pooling over sequence length is the standard approach for sequence-level classification and is what the prior softmax router experiment used. However, mean-pooling can wash out token-level routing signals if domain-discriminative tokens are sparse within sequences (e.g., a few domain keywords surrounded by generic text).

For per-token routing (which is what the actual softmax router does), mean-pooling is the wrong aggregation entirely. The experiment frames routing as sequence-level ("route this input to domain X"), but the production router operates per-token. The mean-pooled 65% hidden-state accuracy cannot be directly compared to the softmax router's 40% per-token classification accuracy -- they measure different things.

This is an important caveat on the "65% centroid > 40% softmax" comparison made in the PAPER.md. The 65% is inflated by both (a) sequence-level vs token-level granularity, and (b) train-on-test leakage (see below).

### Train-on-Test Leakage in Centroid Accuracy

Lines 320-329 compute centroids from ALL samples (including the test sample itself), then evaluate routing accuracy on those same samples. The comment at line 319 mentions "leave-one-out" but this is NOT implemented. Every sample contributes to its own domain centroid, biasing the nearest-centroid prediction toward the correct label.

The magnitude of this bias is roughly proportional to 1/N_d where N_d = 50 samples per domain. With N_d = 50, each sample contributes 2% of its own centroid. This is a small but systematic upward bias on all centroid accuracy numbers. The 65% hidden-state accuracy and 52.4% Q-free accuracy are both slightly inflated. This does not change the kill decision but makes the "65% > 40% softmax" claim in the paper even less reliable.

## Novelty Assessment

This experiment is a direct extension of the softmax router scaling experiment (micro/models/softmax_router_scaling), applying centroid-based analysis to Q-projection features instead of full hidden states. The novelty claim is minimal -- it asks "can we replace the learned router with zero-parameter centroid matching on a different feature space?"

The Parameter Golf paper (arXiv 2506.06105) is the key reference. However, as noted above, the experiment cannot actually test the Parameter Golf hypothesis because it uses a full-RoPE model. A proper test would require training a partial-RoPE model from scratch, which is out of scope for a micro-experiment.

The finding that "domain signal lives in the residual stream, not attention projections" is directionally useful for the project. This is not novel in the literature (it is well-known that MLP layers encode factual/semantic content while attention handles syntactic structure), but it is a useful empirical confirmation for this specific architecture.

## Experimental Design

### Does it test the stated hypothesis?

Partially. The stated hypothesis is: "position-free dimensions learn pure semantic similarity patterns that naturally cluster by domain." But the experiment uses a model *trained with full RoPE*, so there are no "position-free dimensions" in the trained sense. The experiment actually tests a weaker hypothesis: "arbitrary subsets of pre-RoPE Q dimensions carry domain signal." The near-identical performance of q_free (52.4%) and q_rope (49.1%) correctly falsifies even this weaker hypothesis.

### Are controls adequate?

The comparison across five feature types (q_free_last, q_rope_last, q_full_last, q_free_mid, hidden) is well-designed. Including the hidden-state baseline connects to prior work. The q_rope_last condition is a good negative control (if the hypothesis were true, these dims should perform worse).

### Could a simpler mechanism explain the results?

Yes. The 52% centroid accuracy (12x random) is easily explained by the fact that Q projections inherit domain signal from the hidden state through the linear projection W_Q. Any linear projection of a signal that carries ~16% between-domain variance will preserve some of that variance. The 52% accuracy does not require any "position-free semantic structure" explanation.

### The 65% Hidden-State Accuracy (Question 4)

The 65% centroid accuracy on hidden states vs the softmax router's 40% per-token classification accuracy is an apples-to-oranges comparison:

1. **Granularity:** Sequence-level centroid (mean over ~256 tokens) vs per-token classification. Averaging smooths noise, making sequence-level easier.
2. **Leakage:** Centroid accuracy has train-on-test leakage (no held-out split). The softmax router uses a proper train/eval split.
3. **Metric:** Centroid uses nearest-neighbor (full feature vector). Softmax router uses a 330K-parameter learned projection that may underfit with only 500 training steps.

The paper's speculation that "the softmax router's 500-step training was insufficient" is plausible but not supported by this evidence given the confounds above.

## Macro-Scale Risks (advisory)

Not applicable -- the experiment is killed. No macro follow-up warranted.

The one transferable insight: if partial RoPE is ever adopted for other reasons (e.g., context length, as in Parameter Golf), routing should still use full hidden states, not Q-feature subsets. The domain signal is in the residual stream.

## Verdict

**KILL** -- confirmed.

The kill decision is correct and well-justified. The K1 silhouette of -0.007 is unambiguously below the 0.3 threshold.

Beyond the kill criterion, this experiment has a deeper design flaw: it cannot test its stated hypothesis (partial RoPE produces semantic routing features) because it uses a full-RoPE model. The simulation of "position-free dims" by arbitrary dimension splitting is not equivalent to training with partial RoPE. The paper acknowledges this clearly in its Limitations section, so this is not hidden -- but it does mean the experiment's negative result cannot tell us whether actual partial-RoPE models would produce routing-ready Q features.

Specific issues that would require fixing if this line of research were ever revisited:

1. **Implement leave-one-out for centroid accuracy** (lines 320-329 currently leak test data into centroids).
2. **Use cosine similarity or PCA-then-Euclidean for silhouette** to avoid high-dimensional distance concentration.
3. **Separate sequence-level vs token-level comparisons** when referencing the softmax router's 40% accuracy.
4. **Fix stale comments** (lines 171, 233-234, 238-239) that reference wrong tensor shapes.
5. **Test on an actually partial-RoPE model** to evaluate the Parameter Golf hypothesis. Without this, the experiment only shows that arbitrary Q-dim subsets in a full-RoPE model are uninformative for routing, which is unsurprising.

The PAPER.md's key takeaway -- "zero-parameter routing from attention features is dead, domain signal lives in the residual stream" -- is directionally correct for this model, but the "partial RoPE would NOT automatically produce routing-ready features" claim (point 2 in Implications) is not supported by this evidence, since partial RoPE was never actually tested.
