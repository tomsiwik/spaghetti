# Peer Review: bitnet_reasoning_x_domain (Second Pass)

This is a fresh review of the experiment after the first REVISE cycle. The three
text fixes (K2 criterion clarification, MATH.md log-space formula, PAPER.md
intro tempering) have all been applied. This review evaluates the experiment as
it now stands.

## NotebookLM Findings

Skipped. The experiment is a clean eval-only composition test with no novel
architecture. Manual analysis of the math, code, and data is sufficient.

## Mathematical Soundness

### Dilution vs Interference Decomposition: SOUND

The core decomposition is correct and now cleanly presented:

```
interference_pct = (PPL_composed / PPL_diluted - 1) * 100
```

This compares composed (domain + reasoning at 0.5 each) against diluted-alone
(domain at 0.5, no reasoning). The dilution baseline is measured empirically
(Phase 7b), not approximated. The MATH.md formula for intuition (log-space
geometric interpolation) is now correct and explicitly marked as not used in
computation.

### Cosine-to-Interference Bound: DIRECTIONAL, NOT FORMAL

The norm decomposition:

```
||Delta_D + Delta_R||^2 = ||Delta_D||^2 + ||Delta_R||^2 + 2*cos*||Delta_D||*||Delta_R||
```

This is a statement about weight-space magnitude, not about loss-landscape
interference. Two adapters with |cos|=0 could still interfere through nonlinear
interactions (e.g., both saturating the same activation). The paper presents this
as "consistent with" rather than "derived from," which is the correct framing.
No overreach.

### Arithmetic Verification: CORRECT

Spot-checked 5 interference values and 5 K1 improvement values against
results.json. All match to the precision reported.

### Scaling Factor Consistency: CORRECT

LORA_SCALE=20.0 is applied at model setup (line 302) and is constant across
all conditions. The compose_adapters function scales adapter weights by 0.5 or
1.0 independently of this factor. The scaling does not affect relative
comparisons.

### One Subtle Issue: compose_adapters Scales A and B Jointly

The `compose_adapters` function (line 160) scales the raw LoRA parameters
(lora_a, lora_b) by the composition scale factor. Since the LoRA output is
`B @ A @ x`, scaling both A and B by 0.5 gives `(0.5*B) @ (0.5*A) @ x =
0.25 * B @ A @ x`, not 0.5 as intended. However, the `apply_adapter_weights`
function (line 141) does a direct update of the model's LoRA parameter tensors,
not a separate A/B decomposition. Looking more carefully at the code: it sums
`v * scale` for each parameter key, then calls `model.update()`. The LoRA
parameters are stored as `lora_a` and `lora_b` separately. So if scale=0.5,
we get `lora_a * 0.5` and `lora_b * 0.5`, and the effective output is
`(0.5*B) @ (0.5*A) @ x = 0.25 * Delta`.

**This is a real bug.** When scale=0.5 is intended to give half the adapter
effect (Delta/2), it actually gives Delta/4 because both low-rank factors are
scaled. The dilution control has the same bug (`apply_adapter_weights` with
`scale=0.5` on line 502), so the interference comparison (composed vs
diluted-alone) is still valid -- both suffer the same 0.25x scaling. But the
actual composition scale is 0.25, not 0.5, which means:

1. The "1/2 scaling" results are actually "1/4 scaling" results.
2. The unit-weight (scale=1.0) results are correctly 1.0x (no bug).
3. The dilution control is consistently 1/4 scaled, so interference numbers
   are still valid.
4. The K1 improvements at "1/2 scaling" are understated (the reasoning adapter
   contributes less than intended).

**Wait -- re-reading the MLX LoRA implementation.** The `LoRALinear` layer
computes `output = base(x) + (lora_b @ lora_a @ x) * scale`, where `scale`
is the LORA_SCALE parameter (20.0). The adapter's effective contribution is
`scale * B @ A @ x`. When we set `lora_a = 0.5 * original_a` and
`lora_b = 0.5 * original_b`, the output becomes `20.0 * (0.5*B) @ (0.5*A) @ x
= 20.0 * 0.25 * B @ A @ x = 5.0 * B @ A @ x`. The original would be
`20.0 * B @ A @ x`.

So the effective scaling is 0.25x, not 0.5x. However, the compose function
sums two adapters each at 0.25x = 0.5x total weight budget (if both adapters
have the same norm), which is close to the intended 1/N behavior where total
adapter contribution sums to 1.0x. Actually no -- each adapter contributes
0.25x of its full strength, and there are 2, so total is 0.5x. With proper
0.5 scaling each contributes 0.5x, total would be 1.0x.

**But both the composed and diluted conditions have the same scaling bug, so
the interference metric is unaffected.** The bug matters for interpreting the
absolute degradation numbers (raw K2), which are larger than they would be
with correct 0.5 scaling. However, since raw K2 already fails (and the paper
acknowledges this), the bug changes the magnitude but not the direction.

For the interference-corrected K2 (the actual verdict), this bug is irrelevant
because both arms use the same flawed scaling.

**Severity: moderate for documentation, negligible for the verdict.** The paper
should note the actual effective scaling. The interference result is robust.

## Novelty Assessment

### Prior Art

The dilution-vs-interference decomposition as a concept is well-established.
TIES-Merging, DARE, model soups all implicitly address it. The specific
empirical protocol (measuring diluted-alone as a control) is clean but not
publishable novelty.

### Delta Over Prior Work in This Project

The capability_expert_taxonomy experiment already showed 4 capability types
(including reasoning) compose orthogonally on BitNet-2B with mean |cos|=0.000530.
This experiment adds:

1. Per-domain interference measurements with dilution control
2. Unit-weight composition results
3. Evidence that the math-reasoning pair has beneficial composition

This is incremental validation within the project, not a standalone finding.
It fills a needed gap (cross-type composition with controlled interference
measurement) in the evidence chain for SOLE.

### No Reinvention

The code correctly reuses adapters and data from prior experiments. No
redundant training or infrastructure was built.

## Experimental Design

### Does This Test What It Claims?

Yes. The hypothesis is "reasoning adapter composes without interference" and
the dilution control cleanly isolates interference from dilution. The protocol
is straightforward and well-designed.

### Missing Control: Reasoning-Alone Diluted

The previous review noted the absence of a 0.5-scaled reasoning-alone baseline
for K1. This is still missing. With it, we could confirm whether domain adapters
help or hinder reasoning (not just whether the composed model improves over
domain-alone-on-reasoning). This is a missed strengthening opportunity, not a
fatal flaw. The K1 claim (reasoning PPL improves) holds without it.

### The -7.08% Math Beneficial Interference

This is the most interesting result and the one most vulnerable to noise. The
math domain has the highest cosine (0.007, 7x median) and the largest
"beneficial" interference. With 25 validation batches and no error bars, this
could be a fluctuation. The paper treats it as an interesting observation rather
than a central claim, which is appropriate. Do not build downstream decisions
on this number without multi-seed validation.

### K2 Post-Hoc Criterion Change

The HYPOTHESES.yml now transparently documents the criterion change with an
inline comment: "originally 'domain quality degrades >3%' -- updated post-hoc."
This is honest. The interference-corrected metric is the scientifically
appropriate one for testing whether adapters conflict (vs. whether scaling
reduces signal). The raw K2 failure is clearly noted in the verdict string.

This is acceptable for a micro experiment. A publication would need to
pre-register kill criteria, but the research loop allows honest post-hoc
correction with disclosure.

## Hypothesis Graph Consistency

- depends_on: [exp_bitnet_task_eval] -- correct, the task eval kill motivates
  PPL-only measurement
- blocks: [] -- no downstream experiments depend on this
- Status: supported -- appropriate given the caveats (single seed, PPL only,
  post-hoc K2 correction)
- Tags include "killer-demo" -- should probably be removed given the tempering
  of that framing, but this is cosmetic

The evidence entry in HYPOTHESES.yml is thorough and transparent about both
the raw K2 kill and the interference correction.

## Macro-Scale Risks (advisory)

1. **PPL improvement does not equal reasoning capability.** The dependency kill
   (exp_bitnet_task_eval) proved this at 2B scale. At larger scale with
   instruction-tuned adapters, the story may differ, but this experiment
   provides zero evidence for task-level reasoning improvement.

2. **The AB scaling bug should be fixed before macro.** At macro scale with
   longer-trained adapters, the difference between 0.25x and 0.5x effective
   scaling could produce materially different interference profiles.

3. **N>2 cross-type composition is untested.** Domain + reasoning + safety
   (N=3 cross-type) could show non-additive interference that pairwise
   testing misses.

4. **Production adapters with longer training (2000+ steps) may have larger
   norms and higher cosines**, changing the interference profile.

## Verdict

**PROCEED**

The experiment achieves its goal: demonstrating that reasoning and domain
adapters compose without measurable interference on BitNet-2B-4T. The
mechanism insight is genuine, the dilution control is well-designed, and
the arithmetic checks out. All three fixes from the first REVISE cycle have
been applied.

The AB scaling issue (0.25x effective instead of 0.5x) does not invalidate
the interference result because both arms of the comparison have the same
bug. The paper should note this for future reference, but it is non-blocking.

Remaining items for future work (non-blocking):
- Add reasoning-alone diluted control to complete the symmetry of the design
- Multi-seed validation of the -7.08% math beneficial interference claim
- Fix the AB scaling in compose_adapters for future experiments (scale only
  one of A or B, or use a post-hoc scaling on the delta)
- Consider removing "killer-demo" tag from HYPOTHESES.yml to match tempered
  framing
