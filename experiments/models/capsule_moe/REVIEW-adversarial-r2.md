# Peer Review (Round 2): Capsule MoE

## Status of Round 1 Revisions

All five required revisions from REVIEW-adversarial.md have been addressed:

| # | Revision | Status | Quality |
|---|----------|--------|---------|
| 1 | Test composability | DONE | Thorough. Six conditions tested. Shared-base protocol validated. |
| 2 | Uniform-routing ablation | DONE | Clean result: uniform wins at micro scale. Honestly reported. |
| 3 | Reconcile param count | DONE | V=27 vs V=28 identified. MATH.md Section 8 now explains both. |
| 4 | Tone down self-routing | DONE | "Routing: Two Distinct Mechanisms" section is now precise. |
| 5 | FLOP gap documented | DONE | New Section 4.3 in MATH.md. Actual vs theoretical FLOPs computed. |

The researcher responded to each point with substance, not cosmetics. The
composition experiment in particular is a significant addition that transforms
the paper from "a sparse MLP that doesn't degrade" into something with a
concrete, validated protocol for modular domain composition.

---

## NotebookLM Findings

Deep review of the revised documents against the literature identifies:

1. The shared-base composition protocol (pretrain base, freeze attention,
   fine-tune only capsule groups, concatenate, calibrate router) is the paper's
   strongest result. It connects to BTX (Branch-Train-MiX), adapter composition,
   and VISION.md's modular expert library -- but with a concrete protocol that
   actually works at micro scale.

2. The uniform-routing ablation is a double-edged sword. It is honest and
   valuable -- but it means the "Mixture of Experts" framing is misleading at
   micro scale. What the experiment actually validates is a modular sparse MLP
   with composable groups. The routing story is deferred to macro.

3. The FINDINGS.md insight that "shared attention is the forgetting bottleneck"
   from the LGME ablation study is now directly corroborated by the composition
   experiment: independent composition fails because attention diverges (+13.5%),
   shared-base composition succeeds because attention is fixed (-0.3%). This is
   a coherent research thread.

---

## Mathematical Soundness

### Previously verified (still holds)

All derivations from Round 1 remain correct:
- MLP = sum of rank-1 capsules (Section 1)
- Parameter count equivalence at P = 4d (Section 2)
- FLOP analysis (Section 4)
- Balance loss minimum at 1.0 (Section 5)

### New material: Section 4.3 (Implementation Gap)

The new section computes actual FLOPs when all groups are computed:

```
FLOPS_actual = 512 + 4 * 16,384 = 66,048
```

This is correct: router (512) + all 4 groups (4 * 16,384) = 66,048. Compared to
dense MLP at 65,536, the capsule MoE does 512 extra FLOPs (0.8% overhead from
the router). This matches the throughput data (26% slower, with the excess
slowdown attributable to Python loop overhead and memory access patterns rather
than pure FLOPs). Honest and correct.

### New material: Section 6 (Composition)

**Section 6.1** now specifies the shared-base protocol precisely. The notation
is clear: Groups_A and Groups_B are fine-tuned from the same base model M_base
with attention frozen.

**Section 6.2** now has two explicit conditions for non-interference:

- Condition 1 (Backbone consistency): experimentally confirmed.
- Condition 2 (Router separation): achieved via ~100 steps of calibration.

This is a significant improvement over Round 1's hand-waved non-interference
argument. The conditions are concrete and falsifiable.

**Section 6.3** specifies the full protocol in pseudocode. One issue:

> "d. Train router on mixed data for ~100 steps (all other weights frozen)"

The router has shape (2G, d) for the composed model. At 2G = 8 and d = 64,
that is 512 parameters being trained on alternating batches from two domains
for 100 steps. With batch_size=32 and block_size=32, each step sees
32 * 32 = 1,024 tokens. Over 100 steps: 102,400 tokens total. This is a
reasonable calibration cost. But the question is: **does this scale linearly
with the number of domains D?** If composing D domains requires a router of
shape (D*G, d) and calibration on all D domains, the calibration cost grows
as O(D * steps). The paper should state this complexity explicitly, though
it is not blocking at micro scale.

### New material: Section 8 (Param count reconciliation)

V=28 (code default) vs V=27 (arena runtime) is a clean explanation. The delta
of 128 = 2 * 1 * 64 = two extra rows in wte + lm_head. Confirmed correct.

---

## Novelty Assessment (updated)

### What changed

The composition experiment significantly strengthens the novelty claim. The
specific finding -- that composition requires a shared backbone and fails
without it -- is not obvious and is experimentally validated.

### Closest prior art (refined)

**1. Branch-Train-MiX (BTX, Sukhbaatar et al., 2024).** BTX trains domain
experts independently on a shared backbone and composes them into MoE layers.
The Capsule MoE shared-base protocol is essentially BTX applied at capsule
granularity rather than expert granularity. The key difference: BTX composes
full MLP experts (each 8d^2 params), while Capsule MoE composes capsule groups
(each 2d * P/G params). This means Capsule MoE composition adds finer-grained
modules, which could in principle enable more precise domain mixing.

However, the paper does not explicitly compare the cost/benefit of capsule-
granularity vs. expert-granularity composition. At macro scale, would capsule
groups (P/G=64 units) compose more cleanly than full experts (4d units)?
This is the hypothesis that would distinguish the work from BTX, and it
remains untested.

**2. Product Key Memory (Lample et al., NeurIPS 2019).** Now cited in PAPER.md.
The citation is honest and the differentiation is clear.

**3. LoRA Composition Literature (LoRA-Hub, 2023; LoRA-Flow, 2024).** These
compose LoRA adapters by learning mixture weights at inference time. The
Capsule MoE router calibration is analogous: learn routing weights for
composed modules. The parallel is worth noting: the capsule groups are
conceptually "MLP-native LoRA modules" -- frozen base + trainable residuals
in the MLP pathway.

### Delta assessment (updated)

The genuine contribution is now clearer: **a concrete, validated protocol
for composing domain-specific MLP sub-modules via capsule group concatenation
with shared backbone and brief router calibration.** This is more specific
and more validated than Round 1's claim.

The main limitation remains: the contribution is organizational (how to
structure and compose rank-1 units), not mechanistic (no new operation that
was impossible before). BTX + fine-grained experts achieves a similar result.
The Capsule MoE's advantage is parameter efficiency (1x vs 4x MLP params),
but this advantage has not been tested at a scale where it matters.

---

## Experimental Design (updated)

### Composition experiment: thorough and well-controlled

The test_composition.py implements six conditions:

| Condition | What it tests |
|-----------|--------------|
| Joint training | Upper bound (best possible quality) |
| Sequential training | Forgetting baseline |
| Independent + uniform | Composition without router, independent backbone |
| Independent + calibrated | Composition with router, independent backbone |
| Shared-base + uniform | Composition without router, shared backbone |
| Shared-base + calibrated | Composition with router, shared backbone |

This is a good factorial design that isolates two factors: (1) shared vs.
independent backbone, and (2) calibrated vs. uniform routing. The results
cleanly show that shared backbone is necessary (13.5% vs -0.3%) and calibrated
routing is necessary (64.1% vs -0.3% for shared-base).

**One concern: the joint training baseline may be unfairly weak.** Joint
training runs for 2 * STEPS_PER_DOMAIN = 600 steps on alternating batches.
The shared-base composition protocol runs:
- 300 steps on all data (base pretraining)
- 300 steps on domain A (capsule fine-tuning)
- 300 steps on domain B (capsule fine-tuning)
- 100 steps on mixed data (router calibration)
- Total: 1,000 steps

The joint training baseline sees 600 steps total. The composed model sees
1,000 steps. This is a 67% compute advantage for the composed model. The
comparison should either equalize total steps or acknowledge the compute
gap.

This does not invalidate the result -- the composition protocol could
still work at equal compute -- but the -0.3% advantage might reverse if
joint training gets the same 1,000 steps. A fairer comparison would give
the joint training baseline 1,000 steps on alternating batches.

### Uniform-routing ablation: clean and definitive

The result (uniform wins by 0.0041 in val loss, wins all 3 seeds) is
unambiguous. The paper's interpretation is correct: at G=4 with homogeneous
data, learned routing is pure overhead.

**However, there is a subtle confound.** The uniform variant still has the
router weight matrix (confirmed by test_uniform_routing_param_parity), which
means router parameters are still being trained via the aux_loss gradient
pathway even though their output is not used in the forward pass. This
wastes a small amount of optimizer capacity.

Wait -- actually, looking at the code more carefully:

```python
if self.uniform_routing:
    w = 1.0 / self.n_groups
    out = mx.zeros_like(x)
    for group in self.groups:
        out = out + w * group(x)
    self._gate_probs = mx.full((*x.shape[:-1], self.n_groups), w)
    return out
```

In uniform mode, the router is never called, so its output is never part of
the computation graph. The `_gate_probs` is set to a constant, so
`balance_loss()` will return exactly 1.0 (the minimum). The aux_loss
contributes 0.01 * n_layers * 1.0 = 0.04 to every step's loss. This is a
constant offset that does not affect gradients for the capsule parameters.

Actually, wait. The `aux_loss` in the uniform case returns `0.01 * 4 * 1.0
= 0.04`. But `mx.full(...)` creates a constant tensor with no gradient
connection to any parameter. So `balance_loss()` returns a constant, and
`aux_loss()` returns a constant. The gradient of this constant w.r.t. all
parameters is zero. So the uniform variant trains only on cross-entropy
loss, while the learned variant trains on cross-entropy + balance loss.

This means the comparison is slightly confounded: the learned variant has
an additional loss term (0.01 * balance_loss) that the uniform variant does
not have. The balance loss at initialization is likely > 1.0 (before the
router learns uniform assignment), so it adds a non-trivial gradient signal
to early training. This could hurt early-stage capsule learning.

To confirm: the paper reports that uniform routing is 0.0041 better. This
is consistent with the hypothesis that the balance loss hurts rather than
helps at micro scale. A cleaner ablation would compare: (a) learned routing
without balance loss, (b) learned routing with balance loss, (c) uniform
routing. This would separate the routing contribution from the auxiliary
loss contribution.

This is a minor point -- the conclusion that "Level 1 adds no value at
micro scale" would likely survive -- but the mechanism for WHY it adds no
value is ambiguous: is it that the router converges to uniform (routing
doesn't matter), or that the balance loss hurts (the aux term is harmful)?

### Missing control (still): dense GPT with top-k hidden masking

The Round 1 review requested a "standard sparse MLP" baseline to isolate
whether the group structure adds value beyond simple activation sparsity.
This was not addressed. At micro scale this is low priority since the
uniform ablation already shows grouping doesn't help with routing, but
at macro scale this becomes important.

---

## Implementation Review

### test_composition.py: Well-structured with one architectural issue

The `compose_capsule_models` function (line 58) computes attention merge
via one of three strategies: "a", "b", or "avg". The independent-composition
experiment uses "avg", which linearly averages attention weights. This is
known to be a poor strategy for model merging (TIES, DARE, SLERP all exist
to address this). However, the paper correctly concludes that independent
composition fails and shared-base composition is needed, so the choice of
"avg" for the failing case is not misleading -- it just means the "best
possible" independent composition was not tested. Using SLERP or TIES for
the attention merge might improve the independent composition result from
+13.5% to something smaller. This should be noted as future work but is
not blocking.

### compose_from_shared_base (line 129): Clean implementation

This correctly takes the base model's attention weights and both domain
models' capsule groups. The router is randomly initialized and then
calibrated. No issues found.

### calibrate_router (line 183): Correct but missing a detail

The calibration alternates batches from domain A and domain B (odd steps
get A, even steps get B). This means domain A gets 50 steps and domain B
gets 50 steps out of 100 total. At batch_size=32, each domain sees
1,600 sequences. This is a reasonable calibration budget.

However, the function calls `model.unfreeze()` at the end (line 207), which
unfreezes all parameters. This is called before evaluation, so the model
is fully unfrozen during evaluation. This is fine for evaluation (no
gradients flow during eval), but it means subsequent calls to `train()`
on this model would train all parameters, not just the router. The test
does not do this, so it is not a bug, but it is a subtle API footgun.

### capsule_moe.py: uniform_routing parameter handling

The `CapsuleMoEUniformGPT` class (line 163) inherits from `CapsuleMoEGPT`
and passes `uniform_routing=True`. This is clean. The router weights are
still allocated (confirmed by test), which means the param count is
identical. This is the right design choice: it keeps the param count
comparable and avoids confounding the ablation with a parameter difference.

One note: the uniform variant still allocates and initializes the router
weight matrix, which wastes memory. At micro scale this is 4 * 256 = 1,024
floats (4 KB) -- completely negligible. At macro scale it would be worth
not allocating unused parameters.

---

## Integration with VISION.md (updated assessment)

The shared-base composition protocol maps directly to VISION.md's
architecture:

| VISION.md | Capsule MoE |
|-----------|-------------|
| Base model W_0 (frozen) | Shared pretrained attention + embeddings |
| Expert_i = (A_i, B_i, K_i) | CapsuleGroup = (A_g, B_g) per domain |
| Routing via K_i | Group router W_r (calibrated) |
| Adding expert N+1 | Concatenate new capsule groups, recalibrate router |

The main difference: VISION.md uses separate routing keys K_i (decoupled
from computation), while Capsule MoE uses a single router matrix W_r. The
Capsule MoE approach is simpler but less flexible -- adding a new domain
requires expanding W_r and recalibrating on all domains' data, while
VISION.md's per-expert K_i could be calibrated independently.

**Key question for the roadmap:** Should the next experiment test per-group
routing keys (K_g trained contrastively, as in VISION.md) instead of or
alongside the global router W_r? This would combine the capsule pool
structure with the decoupled routing from VISION.md, and could enable
truly zero-shot composition (no router calibration needed).

---

## Connection to FINDINGS.md

FINDINGS.md concluded: "Shared attention is the forgetting bottleneck."
The composition experiment independently confirms this: when attention
is shared (frozen during fine-tuning), composition works. When attention
diverges (independent training), composition fails. This is a consistent
finding across two different experimental setups (LGME and Capsule MoE).

This convergence strengthens both findings. The implication for the
roadmap is clear: any composition protocol must keep attention weights
shared/frozen across domain specialists.

---

## Macro-Scale Risks (advisory, updated)

1. **Joint training compute fairness.** The composition experiment gives
   the composed model 67% more total training steps than the joint baseline.
   At macro scale, where compute is expensive, this matters. The first macro
   test should equalize total training FLOPs between joint and composed
   approaches.

2. **Router calibration scaling with D domains.** At D=2 domains, calibration
   takes 100 steps. At D=10, does it take 500? 1000? The relationship between
   D and calibration cost is unknown and could be the bottleneck for the
   "adding expert N+1" protocol. If calibrating D+1 requires retraining on
   all D+1 domains (not just the new one), the cost grows linearly in D.

3. **Capsule group size at scale.** With d=4096 and P/G=64, each capsule
   group has 2 * 4096 * 64 = 524,288 params. This is much smaller than a
   standard MoE expert (8 * 4096^2 = 134M params). The question is whether
   64 capsules per group provide enough representational capacity for
   domain-specific knowledge. If P/G must grow with d, the parameter
   efficiency advantage shrinks.

4. **Attention freezing as a constraint.** The shared-base protocol requires
   freezing attention during domain fine-tuning. At macro scale with diverse
   domains (code vs. poetry vs. math), attention patterns differ significantly
   across domains. Freezing attention may limit the quality of domain
   specialists. This is the same constraint that limits LoRA and adapter
   methods -- the question is whether capsule-level fine-tuning of the MLP
   pathway is sufficient to capture domain differences that attention cannot.

5. **Hardware efficiency.** The group loop (`for i, group in
   enumerate(self.groups)`) must be replaced with batched/fused operations
   at scale. This is engineering, not research, but it blocks any real-world
   deployment.

---

## Verdict

**PROCEED** -- the experiment is ready for macro-scale validation.

### Justification

The revised Capsule MoE experiment is now a well-rounded micro-scale study
that validates three things:

1. **Mechanism works.** The capsule pool matches dense GPT quality at
   parameter parity (0.7% gap, 3 seeds). This confirms that decomposing
   the MLP into grouped rank-1 capsules does not destroy representational
   capacity.

2. **Composition works (under the right protocol).** Shared-base
   composition with calibrated router matches joint training (-0.3%, 3
   seeds). Independent composition fails (+13.5%). This is a clean,
   falsifiable finding with a concrete protocol.

3. **Honest about what doesn't work.** The uniform-routing ablation shows
   Level 1 routing adds no value at micro scale. The FLOP analysis
   acknowledges that theoretical savings require conditional computation.
   The self-routing claim is scoped to Level 2 only.

The math is sound, the experiments are well-controlled (with the compute
fairness caveat noted below), the limitations are honest, and the result
directly advances VISION.md's goal of composable domain specialists.

### Remaining Issues (not blocking, but should be addressed in macro)

1. **Equalize compute in composition comparison.** Joint training gets 600
   steps; composed model gets 1,000 steps total. Either increase joint
   training to 1,000 steps or add a note acknowledging the compute gap.
   This is the only experimental design issue that weakens the -0.3%
   composition result.

2. **Separate router contribution from balance loss contribution.** The
   uniform ablation conflates two effects: (a) routing quality and (b)
   auxiliary loss. Running learned routing without balance loss would
   clarify which factor drives the uniform variant's advantage.

3. **Test per-group routing keys (VISION.md integration).** The natural
   next step is to test whether per-group contrastive routing keys (as
   proposed in VISION.md) can replace the global router and enable
   zero-shot composition (no calibration step). This would be a strong
   differentiator from BTX.

4. **Dense top-k sparse MLP baseline.** Still missing from Round 1. At
   macro scale, this baseline becomes important to distinguish "grouping
   helps" from "any sparse MLP works."
