# Peer Review: ReLU Router

## NotebookLM Findings

Unable to complete automated NotebookLM deep review due to authentication
requirements. The following review is conducted manually with equivalent rigor,
informed by close reading of MATH.md, PAPER.md, the implementation
(`relu_router.py`), unit tests (`test_relu_router.py`), composition experiment
(`test_composition.py`), the parent model (`capsule_moe.py`), the dense
baseline (`gpt.py`), the training loop (`train.py`), ADVERSARIAL_REVIEW.md,
VISION.md, and FINDINGS.md.

---

## Mathematical Soundness

### What holds

**The composition identity is correct.** The claim in MATH.md Section 5.1:

```
y = B_composed @ ReLU(A_composed @ x)
  = [B_1, B_2] @ ReLU([A_1; A_2] @ x)
  = B_1 @ ReLU(A_1 @ x) + B_2 @ ReLU(A_2 @ x)
  = Pool_1(x) + Pool_2(x)
```

This follows from the block structure of the concatenated matrices. ReLU is
applied element-wise, and the vertical stacking of A means:

```
ReLU([A_1; A_2] @ x) = [ReLU(A_1 @ x); ReLU(A_2 @ x)]
```

because each block independently computes its inner products before ReLU is
applied element-wise. The subsequent multiplication by `[B_1, B_2]` correctly
produces the sum of individual pool outputs. This is mathematically exact.

**The MLP equivalence is correct.** A two-layer ReLU MLP
`W_up @ ReLU(W_down @ x)` is structurally identical to `B @ ReLU(A @ x)`.
The parameter count at P=4d=256 matches the dense GPT baseline
(4*d*d = 4*64*64 = 16,384 for the MLP vs 2*d*P = 2*64*256 = 32,768 for the
capsule pool, but the dense MLP has fc1: d*4d + fc2: 4d*d = 2*4*d^2 = 32,768,
confirming exact match).

**The sparsity loss derivations are internally consistent.** The adaptive L1
coefficient, EMA smoothing, and clamped range are correctly specified in
MATH.md and correctly implemented in `relu_router.py` lines 84-105.

**The balance loss is correctly formulated.** Variance of per-capsule activation
frequency scaled by P penalizes unequal utilization and is correctly
implemented in lines 107-121.

### What does not hold or is misleading

**1. The "routing" framing is misleading -- this IS a dense MLP.**

The paper acknowledges this in the Limitations ("The model IS a dense MLP")
but the entire narrative is built around "self-routing" and "no external
router needed." The truth is simpler and more damning: the ReLU Router
removes the group structure from capsule_moe, collapsing it back to the
dense MLP baseline. Every "hidden neuron" in a ReLU MLP can be called a
"self-routing expert" by relabeling `ReLU(w_i^T x) > 0` as a "routing
decision." This relabeling is not an architectural innovation -- it is a
change of terminology.

The paper should be explicit: the contribution is NOT the architecture (which
is a standard MLP), but the COMPOSITION PROTOCOL (concatenation of
independently-trained MLPs).

**2. The "no router overhead" claim is vacuous at this scale.**

Router savings: 1,024 parameters out of ~202K (0.5%). At any practical scale,
the router in capsule_moe adds negligible overhead. The claim "no router
weight matrix" is technically true but practically irrelevant.

**3. The sparsity control target (75%) is never achieved.**

MATH.md Section 3.2 describes an adaptive L1 mechanism targeting 75% sparsity.
PAPER.md Section "Micro-Scale Limitations" (point 3) acknowledges: "The L1
target sparsity (75%) is not achieved during the 500-step training -- natural
ReLU sparsity stays at ~50%." This means the entire sparsity control mechanism
(Sections 3.2-3.4 of MATH.md) is inert during the experiment. The model runs
with ~50% natural ReLU sparsity and the auxiliary loss does not meaningfully
alter this. The sparsity control is untested.

**4. The composition identity does NOT imply zero-shot composition works.**

The mathematical identity `Pool_composed(x) = Pool_A(x) + Pool_B(x)` is
correct but says nothing about whether the SUM is useful. During joint training,
a single pool learns to produce outputs with a certain magnitude and direction
distribution. When you sum two independently-trained pools, the output
magnitude roughly doubles (or at least changes unpredictably). The downstream
layers (attention, layer norm, lm_head) were calibrated for the output
distribution of a single pool. The sum of two pools produces a different
distribution. This is the "loudness problem" acknowledged in Section 5.3, but
the paper underestimates its severity -- it is not just about relative
magnitude between domains, it is about the absolute magnitude of the composed
output vs what the rest of the network expects.

**5. The Hopfield Network connection (Section 6.1) is a stretch.**

Modern Hopfield Networks use exponential energy functions (polynomial or
exponential kernels) to achieve exponential storage capacity. ReLU provides
a linear energy function with linear storage capacity. Calling this "a Modern
Hopfield Network with linear energy" strips away the defining property
(exponential capacity) and leaves only the trivial claim "it does
content-addressable lookup." Every neural network layer does content-addressable
lookup.

---

## Novelty Assessment

### Prior art

**This is a standard two-layer ReLU MLP.** The architecture has zero novelty.
The paper acknowledges this openly ("The ReLU Router IS a standard MLP").

**ReMoE (ICLR 2025)** uses ReLU-based routing in a genuine MoE context (multiple
large experts, ReLU activation on router logits to select experts). The ReLU
Router paper cites ReMoE but differs fundamentally: ReMoE keeps separate
experts and uses ReLU on router scores to achieve variable-k selection. The
ReLU Router has no separate experts -- it IS one big MLP.

**Union-of-Experts (UoE)** demonstrates that internal expert activation norms
capture routing information, making external routers redundant. This is the
closest prior art to the conceptual claim. The delta: UoE still operates with
separate experts and demonstrates the internal-routing phenomenon at scale
(DeepSeek-MoE 16B). The ReLU Router takes the idea to its extreme (rank-1
capsules = individual neurons), which is simultaneously more radical and more
trivial (because it collapses back to a standard MLP).

**MoRAM** uses rank-1 associative memory experts for continual learning. This
is the most relevant reference for composition. The delta: MoRAM operates in
a LoRA-style adapter context, while ReLU Router operates as the full FFN.

### What is actually novel

The composition protocol -- concatenating independently-trained MLP weight
matrices and relying on the natural non-interference of ReLU activations for
zero-shot composition -- is the genuinely novel idea. This specific protocol
has not been published under this exact form, though:

1. Model merging (task arithmetic, TIES, DARE) addresses similar goals via
   weight-space operations rather than concatenation.
2. Progressive Neural Networks (Rusu et al., 2016) compose by adding lateral
   connections between independently-trained networks, which is structurally
   different but conceptually related.
3. The Mixture-of-LoRAs literature (MoRAM, LoRAHub, etc.) achieves
   composition of independently-trained adapters, which is the same goal at
   the adapter level.

The delta over existing work: demonstrating that plain weight concatenation
of ReLU MLPs produces a mathematically exact sum of independently-trained
functions, and that this sum is useful for multi-domain language modeling.
This is a legitimate micro-scale finding, though the +6.6% zero-shot
degradation tempers the claim.

---

## Experimental Design

### Does it test the stated hypothesis?

The hypothesis is: "ReLU self-routing matches routing quality and enables
zero-shot composition via capsule concatenation."

**Part 1 (routing quality): Adequately tested, trivially true.**
The ReLU Router matches the dense GPT baseline because it IS the dense GPT
baseline. It matches capsule_moe within noise because capsule_moe with
all groups active (no top-k, uniform routing) is also equivalent to a
dense MLP. The 0.3% gap between ReLU Router (0.5137) and capsule_moe
(0.5121) is within std. This result is expected and uninformative --
comparing a dense MLP against itself under different names.

**Part 2 (zero-shot composition): Tested but partially failed.**
The +6.6% zero-shot degradation exceeds the stated 5% kill threshold. The
paper acknowledges this and pivots to "calibrated composition works (-0.5%)."
This is a legitimate finding but undermines the central claim of the paper,
which is that zero-shot composition should work because routing is implicit.

### Controls

**Missing critical control: composition via naive weight averaging.**
The experiment does not compare concatenation against simple weight averaging
of the two domain-specific MLPs. If weight averaging produces similar or
better results, the concatenation mechanism adds nothing. This is important
because model merging (averaging, TIES, etc.) is the standard baseline for
combining independently-trained models.

**Missing control: capsule_moe with calibration vs ReLU Router with calibration.**
The PAPER.md mentions "Capsule_moe achieves -0.3% with router recalibration.
ReLU Router achieves -0.5% with capsule calibration." But the composition
experiment in `test_composition.py` only runs the ReLU Router protocol. The
capsule_moe comparison numbers appear to come from a different experiment.
For a fair comparison, both should run under identical conditions in the same
script.

### Bug in unit test

**`test_composition_by_concatenation` (line 164 of `test_relu_router.py`)
contains a copy-paste bug:**

```python
B_composed = mx.concatenate([pool_b.B.weight, pool_b.B.weight], axis=1)
```

This uses `pool_b` twice instead of `pool_a` and `pool_b`. The test should
read:

```python
B_composed = mx.concatenate([pool_a.B.weight, pool_b.B.weight], axis=1)
```

This bug means the unit test for composition is not actually testing
composition of two different pools -- it is testing duplication of pool_b.
The test still passes because it only checks shapes, not functional
correctness. This is a significant gap in test coverage.

Furthermore, the test never verifies the key mathematical identity:
`composed(x) == pool_a(x) + pool_b(x)`. It only checks that shapes are
valid and individual pools produce output. A correct composition test would:
1. Fix the copy-paste bug
2. Assign the composed weights into `pool_composed`
3. Run `pool_composed(x)` and verify it equals `pool_a(x) + pool_b(x)`

### The calibration procedure conflates two things

In Method 3 of `test_composition.py` (lines 193-196), calibration unfreezes
ALL capsule weights in the composed model and trains them on joint data for
100 steps at 0.1x learning rate. This means:
- Domain A capsules are fine-tuned on domain B data (and vice versa)
- The calibration does NOT just "harmonize scales" -- it actually retrains
  the capsules toward a joint solution

The paper frames this as "brief calibration to harmonize activation
magnitudes." In reality, it is 100 steps of continued joint fine-tuning on
all capsule parameters. This makes the -0.5% result unsurprising -- any
continued training on the correct data distribution will improve loss. A
fairer calibration would freeze individual capsule weights and only train a
per-pool scaling factor (1 parameter per domain per layer), which would
truly test whether the loudness problem is the only issue.

### Composition experiment uses n_capsules=128 per domain, not 256

The `run_composition_experiment` function uses `n_capsules=128` (line 116),
while the single-domain arena experiment uses 256. The joint baseline uses
`n_capsules * 2 = 256`. This is reasonable (matching total capacity), but
the PAPER.md does not clearly state this difference. The reader might assume
all experiments use 256 capsules.

---

## Integration Risk

### Compatibility with VISION.md

VISION.md describes the target architecture as:

```
Per layer:
  Expert Library: [group_1, group_2, ..., group_N]
  Each group_i = capsule pool (rank-1 capsules)

  1. Score:   s = x @ W_r^T          (softmax router)
  2. Select:  top-k groups by score
  3. Apply:   delta = sum  w_i * group_i(x)
  4. Output:  base(x) + scale * delta
```

The ReLU Router eliminates the group structure and the softmax router entirely.
This is a **fundamental architectural divergence** from VISION.md, which
explicitly validates softmax routing as the composition mechanism (+0.2% vs
joint) and identifies k=2 as the optimal sparsity.

If the ReLU Router is adopted, the entire VISION.md architecture must be
revised. The question is whether the ReLU Router's zero-shot composition
benefit (+6.6% degradation) justifies abandoning the validated softmax
routing protocol (-0.2% with calibration, +1.6% at N=5).

**Current answer: it does not.** Calibrated capsule_moe composition (-0.3%)
outperforms calibrated ReLU Router composition (-0.5%) while also providing
a functional zero-shot baseline (the softmax router assigns weights to new
groups, even if not optimally). The ReLU Router's zero-shot performance
(+6.6%) is worse than capsule_moe's calibrated performance.

### Conflict with existing findings

FINDINGS.md establishes:
1. k=2 is the minimum routing bandwidth (k=1 catastrophic)
2. Softmax router routes by task quality, not domain identity
3. Concatenation is the validated composition method

The ReLU Router adopts finding (3) but contradicts findings (1) and (2) by
removing the router entirely. The experiment shows this "works" at the cost
of worse zero-shot composition and no sparsity benefit (50% vs targeted 75%).

---

## Macro-Scale Risks (advisory)

**1. The "loudness problem" will be worse at scale.**
With more diverse domains (code, prose, math, legal), activation magnitude
distributions will diverge further. The +6.6% zero-shot degradation at micro
scale (where domains are nearly identical, a-m vs n-z names) suggests much
worse degradation with truly different domains.

**2. No compute savings without effective sparsity control.**
The ReLU Router is a dense MLP at 50% natural sparsity. To achieve compute
savings, sparsity must be pushed to 75%+ and exploited through sparse
computation kernels. The micro experiment could not push sparsity beyond
50%, and sparse kernels do not exist for the current implementation. At
macro scale, this model offers zero compute advantage over a dense MLP.

**3. Concatenation does not scale.**
At N=10 domains, the composed pool has 10x the capsules. This means 10x the
parameters in the FFN layer, 10x the computation per forward pass, and 10x
the memory. This is worse than capsule_moe, which uses top-k routing to
limit computation to k groups regardless of total group count. The entire
point of MoE is sublinear scaling; concatenation provides linear scaling.

**4. The calibration step reintroduces router-like overhead.**
If zero-shot composition does not work (as shown: +6.6%), calibration is
required. The calibration in the current experiment trains all capsule
parameters on joint data -- this is MORE expensive than capsule_moe's
router-only calibration (which freezes everything except the router weights,
a much smaller parameter set).

---

## Verdict

**REVISE**

The experiment tests a legitimate hypothesis (can implicit ReLU routing
replace explicit softmax routing for composition?) and obtains a meaningful
result (zero-shot: partially, with +6.6% degradation; calibrated: yes, at
-0.5%). The math is sound for what it claims. However, several issues must
be addressed before this advances the project:

### Required fixes

1. **Fix the copy-paste bug in `test_relu_router.py` line 164**: Change
   `pool_b.B.weight` to `pool_a.B.weight` in the first argument of
   `mx.concatenate` for B_composed.

2. **Add a functional composition test**: Verify the mathematical identity
   `pool_composed(x) == pool_a(x) + pool_b(x)` numerically, not just
   check shapes. This is the central mathematical claim and it must be
   verified in code.

3. **Reframe the narrative**: The architecture is a standard ReLU MLP.
   The contribution is the composition protocol (concatenation of
   independently-trained MLPs). The "self-routing" terminology, while
   defensible, obscures the simplicity of what is actually happening. The
   PAPER.md should lead with "composition by concatenation" rather than
   "routerless self-routing."

4. **Add a weight-averaging baseline to the composition experiment**:
   Compare concatenation against simple weight averaging of domain-specific
   MLPs. If averaging works comparably (or better for zero-shot), the
   concatenation mechanism is not the key contribution.

5. **Separate calibration from fine-tuning**: The current calibration
   (100 steps training all capsule weights on joint data) is effectively
   continued training. Add a control that trains ONLY a per-pool scalar
   (2 parameters per layer for 2 domains) to test whether the loudness
   problem is truly the only issue.

6. **Acknowledge that sparsity control is untested**: The L1 mechanism
   never achieves its target in the experiment. Either remove the 75%
   target claim or demonstrate it working (longer training, higher
   coefficients, or a dedicated ablation).

### Advisory (not blocking)

7. Weaken or remove the Modern Hopfield Network connection (Section 6.1)
   -- the linear energy function does not provide the defining property of
   modern Hopfield networks (exponential storage capacity).

8. Run capsule_moe composition under identical conditions in the same
   script for a direct comparison, rather than citing numbers from a
   separate experiment.

9. The PAPER.md "Key Insight" section on task-routing vs identity-routing
   is the strongest conceptual contribution. Consider promoting this to
   the central thesis: "Composition works because ReLU capsules route by
   task (which patterns reduce prediction error) rather than by identity
   (which domain is this), and task-routing is naturally compositional
   because the same patterns may appear across domains."
