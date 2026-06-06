# Peer Review: Unified Routing Pipeline

## NotebookLM Findings

Skipped -- the experiment is already self-killed with clear data. Deep review is unnecessary for a well-documented negative result.

## Mathematical Soundness

### PPL decomposition (MATH.md lines 99-113)

The central claim is:

    L_unified = f_skip * L_base + (1 - f_skip) * L_routed

This is **correct as written** -- it is a weighted average of per-token cross-entropy losses, where the weight assignment is determined by the entropy gate. The code (Phase 2, lines 504-513) faithfully implements this: each token's loss is drawn from either `base_loss` or `routed_loss` depending on the threshold.

### The hypothesis was logically flawed from the start

MATH.md line 114 states:

> The hypothesis: PPL_unified < 6.42 (best individual method) because:
> 1. Confident tokens (63%) get base output -- which is already optimal for them
> 2. Uncertain tokens (37%) get routed top-2 -- which is near-oracle

Point (1) contains the critical error. "Already optimal for them" is unsubstantiated. The base model output on low-entropy tokens is *not* optimal -- it is merely *confident*. The routed model output on those same tokens could still be better (lower cross-entropy), and the data proves it is. Even at p20 (only 20% skip rate), the unified PPL is 6.47 > 6.42. This means that **every** token substituted from base to routed output increases loss on average.

This should have been caught before running the experiment. A 30-second sanity check: if routing heads achieve near-oracle quality (0.15% gap to oracle), then by definition they are near-optimal on *all* tokens, including confident ones. Replacing any of those predictions with base-model predictions can only hurt. The hypothesis implicitly assumed that routing was wasteful on confident tokens, but the near-oracle result from `tiny_routing_heads` already contradicted this.

### Cost analysis (MATH.md lines 59-95)

The cost analysis is correct in structure but contains an important acknowledgment that undermines the entire motivation:

> NOTE: The overhead CANNOT be less than 0% because we always need the base forward pass for entropy computation.

This is honest and correct. The two-pass architecture is fundamentally more expensive than the single-pass always-compose approach. The value proposition had to come entirely from quality improvement, which did not materialize.

### Assumption 3 is untested but irrelevant

MATH.md lists Assumption 3 ("Routing heads remain accurate when applied only to uncertain tokens") as UNTESTED. However, Phase 2 applies routing to *all* tokens and then selectively uses base vs. routed loss post-hoc. The routing heads are never actually restricted to a subset of tokens. So Assumption 3 is not tested but also not relevant to the actual experiment.

## Novelty Assessment

This experiment is a **pure integration test** of two prior components:
- Entropy gating from `exp_entropy_gated_experts`
- Routing heads from `exp_tiny_routing_heads`

There is no novel mechanism. The contribution is a negative result showing these two components are incompatible. This is a legitimate and valuable finding for the research program, but does not represent publishable novelty.

The insight that "entropy measures base confidence, not routing necessity" (PAPER.md lines 97-101) is the key takeaway and is correctly identified.

## Experimental Design

### Strengths

1. **Clean decomposition.** Phase 1 collects per-token base and routed losses, then Phase 2 performs the unified pipeline PPL computation as a pure post-hoc analysis without additional model forward passes. This eliminates confounds from re-running the model.

2. **Threshold sweep.** Testing 7 thresholds (p20 through p80 plus Otsu) definitively shows this is not a threshold-tuning issue. No threshold achieves parity with pure routing.

3. **Honest overhead measurement.** Phase 3 correctly times the full pipeline including the entropy computation pass, and honestly reports that 100% of test sequences triggered routing (because every sequence had at least one uncertain token).

### Weaknesses

1. **Sequence-level vs. token-level gating mismatch.** The PPL computation (Phase 2) uses per-token gating: each token independently uses base or routed loss. But the overhead measurement (Phase 3) uses sequence-level gating: if *any* token in the sequence exceeds the threshold, the entire sequence gets the full pipeline. This means K1 and K2 are measured under different gating granularities. The paper acknowledges this (PAPER.md line 130) but does not quantify the gap.

2. **Overhead measurement only on python domain.** Phase 3 uses only python validation data (line 596: `load_domain_texts("python", split="valid")[:10]`). Python has the highest skip rate (89% at Otsu). Testing on legal (37.4% skip) or creative (66.6% skip) would give more representative overhead numbers. The 222.7% overhead may be somewhat inflated because python sequences tend to be longer, but the conclusion (way above 10%) would not change.

3. **The hidden states are computed twice.** In Phase 3 (lines 658-678), `get_hidden_states()` re-runs the full model forward pass through all layers to extract `h`. But `model(x)` at line 659 already ran through all layers. The hidden states from that first pass are discarded. A production implementation would extract hidden states from the first pass, potentially halving the overhead for routed sequences. This makes the 222.7% overhead artificially high, though likely still above 10%.

4. **No statistical uncertainty on PPL.** The PPL numbers are point estimates across 25 validation batches per domain. No confidence intervals or standard errors are reported. At the margins (p20 gives 6.47 vs 6.42), a 0.05 PPL difference could be within noise. However, the monotonic trend across all 7 thresholds makes this unlikely to be a fluke.

## Hypothesis Graph Consistency

The experiment has **no node in HYPOTHESES.yml**. This is a process gap -- every experiment should have a corresponding hypothesis node before execution, with kill criteria defined a priori.

The kill criteria as stated in the code (K1: unified PPL > 6.42, K2: overhead > 10%) are reasonable and the experiment correctly evaluates them.

## Macro-Scale Risks (advisory)

Not applicable -- the mechanism is killed. No macro scale-up warranted.

For the record, if this had passed: the two-pass architecture would scale poorly because inference cost doubles regardless of model size. The insight that "routing heads are already near-oracle on all tokens" would likely hold at scale, making entropy gating permanently dominated by always-route.

## Verdict

**PROCEED** (as a documented kill)

The experiment is correctly killed. Both kill criteria fail unambiguously. The analysis is thorough, the threshold sweep is conclusive, and the root cause identification ("entropy measures base confidence, not routing necessity") is insightful and well-supported.

The kill should be recorded in FINDINGS.md and the hypothesis graph. No revision needed -- this is a clean negative result.

Minor notes for the record (not blocking):

1. The hypothesis was logically falsifiable before running the experiment. The near-oracle result from `tiny_routing_heads` (0.15% gap) already implied that routing helps on all tokens, making entropy-based skipping counterproductive. Future experiments should perform this kind of pre-mortem analysis to save compute.

2. The overhead measurement has the hidden-state recomputation bug (Phase 3 runs the model forward twice when it could extract hidden states from the first pass). This inflates the overhead number, but even at half the measured overhead (~111%), K2 still fails by an order of magnitude.

3. The experiment lacks a HYPOTHESES.yml node. This should be added retroactively as a KILLED node to maintain the hypothesis graph.
