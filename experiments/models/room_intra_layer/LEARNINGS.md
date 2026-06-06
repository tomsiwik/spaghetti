# LEARNINGS: room_intra_layer

## Status: KILLED (K823 FAIL, gap=10.3% > 5%)

## What We Learned

### Finding #334 (conclusive): Pre-sum without routing = unrouted mixture

The fundamental insight from this experiment: `W + Σdelta_i` is a **fixed matrix** that
cannot adapt to input domain. Composition requires a routing function `r(x)` that selects
`delta_{r(x)}` based on input. Pre-summing is ensemble fusion, not composition.

This follows from algebra alone — no experiment was needed to establish this. The experiment
confirmed the failure empirically (10.3% quality gap) but the impossibility structure is
definitional.

### What this clarifies about the Room Model

The Room Model breakthrough (W_combined = Σ ΔW_i, routing IS the matmul) is correct, but
"routing IS the matmul" means the routing function is **implicit in the weight selection**,
not that you can skip routing entirely. The key insight from the Room Model was that you
don't need a separate router network — the dot product between the query and adapter keys
IS the routing. But you still need per-token adapter selection.

### Code-Doc Mismatch

PAPER.md described "Apply pre-summed deltas to Layer 0 only" but the code applied deltas
to all 4 layers. This inadvertently repeated Finding #303 (full-model pre-summing) rather
than testing the intra-layer hypothesis. The intra-layer hypothesis (apply pre-sum to one
layer only) remains untested, but Finding #334 establishes it wouldn't help: within-layer
vs cross-layer location doesn't change the fact that W_combined is domain-agnostic.

### Root Cause of Failure

The hypothesis conflated two separate questions:
1. Does pre-summing within a layer preserve algebraic exactness? (Yes — proven by #302)
2. Does applying all domain deltas simultaneously preserve domain-specific quality? (No — #334)

Question 2 is the actual barrier. Location of summation (intra vs inter-layer) is irrelevant
when the core problem is absent routing.

## What To Do Next

The gradient routing experiment (room_gradient_analysis) is the natural next step: if
adapter gradients ∇H encode domain structure, they could serve as the routing signal.
