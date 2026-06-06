# Peer Review: Capsule MoE

## NotebookLM Findings

Deep review of MATH.md, PAPER.md, and VISION.md against the broader literature
identifies the following key tension points:

1. The core MLP-as-sum-of-rank-1-capsules decomposition is mathematically sound
   and well-known. The grouping and two-level routing on top is the novel
   contribution. The question is whether grouping rank-1 units adds anything
   beyond what standard MoE or standard sparse MLP already provides.

2. The "self-routing" narrative (capsule activation IS routing) is compelling but
   overstated -- the Level 1 group routing is still an external learned router,
   identical to standard MoE. Only Level 2 (ReLU sparsity) is truly "self-
   routing," and that is simply the inherent property of ReLU in any MLP.

3. The composability claim is the most valuable part for the VISION.md goal, but
   it was explicitly NOT tested. This is the experiment's single largest gap.

---

## Mathematical Soundness

### What holds

**MLP = sum of rank-1 capsules (Section 1).** Correct. This is a direct
restatement of how matrix multiplication works: if W_down has rows a_i^T and
W_up has columns b_i, then W_up @ ReLU(W_down @ x) = sum_i b_i * ReLU(a_i^T x).
The derivation is clean and correct.

**Parameter count equivalence (Section 2).** Verified step-by-step:
- Per group: 2 * d * (P/G) parameters. At d=64, P/G=64: 8,192. Correct.
- Total pool: G * 2 * d * (P/G) = 2*d*P. At P=256: 32,768. Correct.
- Dense MLP: 8*d^2 = 32,768 at d=64. Correct.
- With P=4d, the capsule pool IS parameter-equivalent. This is because 2*d*4d =
  8d^2, which matches 4d*d + d*4d. Confirmed.

**FLOP analysis (Section 4).** Checked:
- Router: 2*d*G = 512. Correct (d=64, G=4).
- Per group: 4*d*(P/G) = 16,384. Correct (two matmuls of size d x 64).
- Total: 512 + 2*16,384 = 33,280. Correct.
- Dense MLP: 16*d^2 = 65,536. Correct.
- Ratio: 33,280/65,536 = 50.8%. The "~51%" claim checks out.

**Balance loss (Section 5).** L_bal = G * sum(f_g^2). Minimum when f_g = 1/G for
all g: G * G * (1/G)^2 = 1. Correct. This is the standard Switch Transformer
formulation.

**Full model param count (Section 8).** Verified:
- Attention per layer: 4 * 64^2 = 16,384 (wq, wk, wv, wo). Correct.
- Capsule pool per layer: 32,768. Correct.
- Router per layer: 4 * 64 = 256. Correct.
- Total per layer: 49,408. All layers: 197,632. Correct.
- Embeddings: wte(28,64)=1,792 + wpe(32,64)=2,048 = 3,840.

**Wait -- discrepancy found.** The MATH.md claims embeddings = "2 * 28 * 64 +
32*64 = 5,632." That computes 3,584 + 2,048 = 5,632. But the "2 * 28 * 64"
implies wte + lm_head are separate. However, the code does NOT tie weights:

```python
self.wte = nn.Embedding(vocab_size, n_embd)
self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
```

These are separate parameters: wte is (28, 64) = 1,792, and lm_head is (28, 64)
= 1,792, so 2 * 28 * 64 = 3,584 is correct for both. Plus wpe = 2,048. Total
embeddings = 5,632. The MATH.md accounting is correct (lm_head is counted
separately from wte). And norm0 has zero parameters (no learnable scale/bias in
this RMSNorm implementation). So the total is 197,632 + 5,632 = 203,264.

The PAPER.md reports 203,136 params. Discrepancy: 203,264 - 203,136 = 128
parameters. This is exactly G * d = 4 * 32 = 128... no, that's wrong. At d=64,
G*d = 256, and the MATH.md already accounts for the router. Let me recount:

MATH.md: 4 * (16,384 + 32,768 + 256) + 5,632 = 4 * 49,408 + 5,632 = 203,264.
PAPER.md reports 203,136. The delta is 128.

This small discrepancy (0.06%) likely comes from a difference in how the arena
counted parameters vs. the manual calculation. Not blocking, but the MATH.md and
PAPER.md should be reconciled.

### What doesn't hold

**The "self-routing" claim is misleading (PAPER.md).** The paper states:

> "Unlike standard MoE where an external router must guess which expert is
> relevant, capsules have a natural 'self-routing' signal: the magnitude of
> the scalar activation |a_i^T * x|."

This is true at Level 2 (individual capsule gating via ReLU). But Level 1 --
the group router -- is an external learned linear projection, identical in
mechanism to standard MoE routing. The paper conflates these two levels. The
Level 1 router does NOT use capsule activations for routing; it uses a
completely separate weight matrix (W_r). This should be stated more precisely.

**The composability non-interference condition (Section 6.2) is hand-waved.**
The condition ||Pool(x; composed) - Pool(x; A) - Pool(x; B)|| ~= 0 is stated
to hold "when groups are assigned disjoint routing." But this requires:

1. The router scores for domain-A tokens assign near-zero weight to domain-B
   groups, AND vice versa.
2. After concatenation, the expanded router (now routing over G_A + G_B groups)
   still achieves this separation.

Condition (2) is non-trivial. The router weight matrix W_r changes dimension
from (G, d) to (G_A + G_B, d). The paper states "the group router must be
recalibrated" but does not specify how. If recalibration requires training on
mixed-domain data, this weakens the "no retraining" claim significantly.

More importantly: at the current micro scale with G=4 groups on similar data
(a-m vs n-z names), the router has no incentive to achieve disjoint routing.
The paper acknowledges this in Section 6.3 but does not test it at all. This
is the central claim for VISION.md advancement, and it has zero experimental
support.

---

## Novelty Assessment

### Prior art

**1. "Decomposing and Composing LoRA" (Liu et al., 2024).** Decomposes LoRA
into rank-1 components and routes over them. The Capsule MoE paper cites this
and claims the extension is "from the LoRA setting to the full MLP replacement."
This is a legitimate but incremental delta. The core idea -- rank-1 units as
atomic routable experts -- is the same.

**2. Product Key Memory (Lample et al., NeurIPS 2019).** Uses key-value memory
layers where keys are looked up via inner product (analogous to a_i^T x scoring)
and values are retrieved (analogous to b_i expansion). This is a structured
sparse MLP with top-k retrieval. The Capsule MoE paper does not cite this. The
mechanism is similar: both replace dense MLP with sparse lookup over rank-1-like
primitives. PKM uses separate key and value matrices with product keys for
efficiency; Capsule MoE uses grouped linear layers. The difference is mainly
organizational (groups vs. product keys) rather than fundamental.

**3. Sparse Mixtures of Experts (Shazeer et al., 2017; Fedus et al., 2022).**
The group router is identical to Switch Transformer routing. The paper
acknowledges this clearly.

**4. ReLU activation sparsity literature.** "The Lazy Neuron Phenomenon" (Li et
al., 2023), "Deja Vu" (Liu et al., 2023), "PowerInfer" (Song et al., 2023). All
exploit the observation that ReLU MLPs are naturally sparse. The "Level 2
sparsity" contribution is not novel -- it is restating known behavior and
framing it as a routing level.

**5. Autonomy of Experts (AoE, 2024).** Cited. The paper correctly notes the
connection but the capsule architecture does NOT achieve AoE's full property.
AoE eliminates the external router entirely. Capsule MoE retains an external
group router (Level 1) and only achieves "self-routing" at Level 2 (which is
just ReLU doing what ReLU does).

### Delta assessment

The genuine contribution is: **organized grouping of rank-1 capsules with
two-level sparsity (learned group routing + inherent ReLU sparsity) as an
MLP replacement at parameter parity with the dense MLP.** This is a valid
architectural idea. The parameter parity claim is the strongest differentiator
vs. standard MoE (which requires N copies of the MLP).

The composition-by-concatenation idea is interesting but untested. If validated,
it would be a meaningful contribution to the modular/composable model literature.

---

## Experimental Design

### Does this test what it claims?

**Claim 1: "Match dense-MLP quality at lower active parameter count."** TESTED.
The single-domain results (0.5211 vs 0.5177 for GPT, within 0.7%) support this
at the directional level. The 3-seed evaluation with error bars is good practice.

**Claim 2: "Enable domain composition by concatenating independently-trained
capsule groups."** NOT TESTED. The PAPER.md honestly acknowledges this. However,
this is the claim that matters most for VISION.md. Without this test, the
experiment validates only that a grouped sparse MLP works, which is a weaker
result.

**Claim 3: "Parameter efficiency is dramatic" (65.9% fewer params than MoE).**
This is trivially true by construction. Standard MoE with N=4 experts has 4
full MLPs. Capsule MoE with G=4 groups has 1 MLP's worth of parameters split
into 4 groups. Comparing to standard MoE on parameter count is misleading --
the architectures have fundamentally different total capacities. A fairer
comparison would be MoE with N=4 experts vs. Capsule MoE with P = 4 * 4d = 16d
(same total parameters as MoE). The current comparison proves that a 1x-capacity
model can match a 4x-capacity model on toy data, which is unsurprising at this
scale where all models converge to similar loss.

### Controls

**Adequate:** The experiment includes dense GPT, standard MoE, and moe_freeze as
baselines. The 3-seed evaluation is good. The multi-domain evaluation tests
forgetting.

**Missing control: Standard sparse MLP.** A simple baseline would be a dense MLP
with random top-k masking at the hidden layer (or a magnitude-based top-k). This
would isolate whether the group routing mechanism adds value beyond simple
activation sparsity. If a non-grouped sparse MLP matches Capsule MoE quality,
the grouping adds no value.

**Missing control: Capsule MoE with uniform routing.** Running with all groups
equally weighted (w_g = 1/G) would test whether the learned router contributes
anything. If uniform routing matches learned routing, Level 1 is wasted
overhead.

### Could a positive result be explained by a simpler mechanism?

**Yes.** The core result (Capsule MoE matches GPT at parameter parity) is
exactly what you would expect from the mathematical equivalence. A CapsuleGroup
with n_capsules hidden units IS an MLP with n_capsules hidden units. The routing
adds sparsity but at this micro scale, the model is so small that sparsity
provides no benefit (the 26% throughput penalty confirms this). The positive
result is explained by: "a sparse MLP is roughly as good as a dense MLP when
the model is tiny and data is simple."

### Implementation concerns

**All groups are computed regardless of routing.** In `CapsulePool.__call__`,
line 72: `for i, group in enumerate(self.groups): out = out + w * group(x)`.
Every group runs for every token. The "zero weight" groups still compute their
full forward pass; only the multiplication by w=0 discards the result. This
means the FLOP savings from Level 1 routing are theoretical only -- the
implementation does not achieve them. The paper acknowledges this
("at small G this is cheaper than scatter") but the FLOP analysis in MATH.md
Section 4 computes theoretical savings as if non-selected groups are skipped.

This is acceptable for a micro-scale proof of concept, but the throughput number
(95K tok/s vs. 128K tok/s) reflects the reality: the model is SLOWER than the
dense GPT, not faster. At macro scale, conditional computation would need to be
implemented for the architecture to deliver on its efficiency promise.

**Top-k masking with ties.** The top-k selection uses `scores >= threshold`
where threshold is the minimum of the top-k scores. If multiple groups have
identical scores (likely at initialization), more than k groups may be selected.
At initialization with random weights, this is unlikely to cause problems, but
it is a subtle correctness issue. The MoE baseline has the same pattern, so this
is at least consistent.

---

## Integration with VISION.md

VISION.md's goal is: "Find that fraction at runtime, compose only those experts,
generate tokens." The Capsule MoE advances this by:

1. **Finer granularity** -- rank-1 capsules vs. monolithic experts. Good fit.
2. **Parameter parity** -- no param overhead for sparsity. Good fit.
3. **Composition by concatenation** -- aligns with VISION.md's "adding expert
   N+1 = store (A, B)" protocol.

However, VISION.md specifically found that "A matrices don't self-route" for
LoRA. The Capsule MoE paper claims to resolve this: "at rank 1, the projection
IS the discriminator." This claim needs scrutiny. At rank 1, a_i^T x is indeed
a single scalar that measures alignment between the input and the detector
vector. But this is exactly what a single LoRA A-row does. The reason LoRA A
matrices failed at routing was not rank -- it was that A is optimized for
reconstruction, not discrimination. Capsule a_i vectors have the same dual role:
they must both detect relevant inputs AND contribute to good reconstruction
(through gradient pressure from the output loss). The rank-1 argument does not
resolve the fundamental tension identified in VISION.md.

This is not a fatal flaw -- the Level 1 group router handles the discrimination
task, and the a_i vectors only need to cooperate within their assigned group.
But the self-routing narrative should not claim to resolve the routing-vs-
computation conflict.

---

## Macro-Scale Risks (advisory)

1. **Conditional computation implementation.** At G=4, running all groups is
   fine. At G=64 or G=256, conditional computation (only running selected
   groups) becomes mandatory. This requires either scatter/gather operations or
   expert-parallel sharding. The rank-1 structure does not simplify this.

2. **Group size vs. hardware utilization.** At macro scale, each CapsuleGroup
   becomes a matmul of shape (batch, seq, d) x (d, P/G). If P/G is small (e.g.,
   64 capsules in a group), the matmul is narrow and may underutilize GPU tensor
   cores. Standard MoE experts have hidden_dim = 4d, which is wider and more
   hardware-friendly.

3. **Router quality at scale.** With G=4, the router distinguishes among 4
   options. At G=64, the routing problem becomes harder. Standard MoE research
   (Mixtral, DeepSeek-V3) shows that routing quality degrades with more experts
   and requires auxiliary losses, capacity factors, or expert parallelism to
   manage. Nothing about the capsule architecture changes this.

4. **Composition test is blocking for macro.** The entire value proposition for
   VISION.md depends on composition working. This must be validated before
   scaling up.

---

## Verdict

**PROCEED** -- with required revisions before the next experiment.

### Justification

The core mechanism is mathematically sound: grouping rank-1 capsules into a
routable pool creates a sparse MLP at parameter parity with the dense MLP. The
derivations are correct. The empirical results show no degradation vs. the dense
baseline (within 0.7% across 3 seeds). The architecture fits naturally into the
VISION.md roadmap.

The experiment does what a good micro-experiment should: it validates that the
mechanism works (converges, doesn't degrade) at a cheap scale, with appropriate
baselines and multiple seeds.

However, the following issues should be addressed before building on this:

### Required Revisions

1. **Test composability.** This is the single most important missing piece. Run
   a 3-phase experiment: (a) train Group_A capsules on domain A, (b) train
   Group_B capsules on domain B, (c) concatenate into a composed pool with a
   recalibrated router, evaluate on both domains. Compare to: (i) joint training
   on both domains, (ii) sequential training. If composition degrades by >5%
   vs. joint training, the composability claim fails and the architecture loses
   its primary advantage over standard MoE.

2. **Add a uniform-routing ablation.** Run Capsule MoE with w_g = 1/G (no
   learned routing). If this matches the learned-router version, Level 1 adds
   no value at this scale and the paper should acknowledge this explicitly.

3. **Reconcile param count.** MATH.md computes 203,264; PAPER.md reports
   203,136. Difference is 128 params. Identify the source and fix whichever
   is wrong.

4. **Tone down the self-routing claim.** The "self-routing" property applies
   only to Level 2 (ReLU sparsity), which is an inherent property of any ReLU
   MLP. Level 1 uses a standard external router. The paper should distinguish
   these clearly and not claim to resolve the routing-vs-computation conflict
   identified in VISION.md without evidence.

5. **Acknowledge the FLOP accounting gap.** MATH.md Section 4 computes FLOP
   savings assuming non-selected groups are skipped. The implementation computes
   all groups. Either implement conditional computation (ideal) or add a note
   that the FLOP savings are theoretical at micro scale. The throughput data
   (95K vs 128K tok/s) already tells the honest story.
