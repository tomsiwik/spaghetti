# MATH.md: Segment Adapter Scale Sweep

## Type: Frontier Extension (Type 3)

**Starting point:** LoRA perturbation scaling with known norm relationships.
**Extension:** Whether optimal LORA_SCALE exists for 128-token segment-isolated
application. The PBR definition (linear in s) is a tautological identity, not a
proven theorem with predictive power. The IVT argument for existence of s* had
unsatisfied preconditions (assumes adapter helps, which is the hypothesis itself).
This is a frontier probe, not guided exploration within a proven framework.

---

## A. Failure Mode Identification

**Observed failure:** Adapters trained at LORA_SCALE=20 on full sequences (256 tokens)
degrade PPL when applied to 128-token isolated segments:

| Configuration | PPL |
|--------------|-----|
| Base only (no adapter) | 7.465 |
| Per-sequence best (full sequence, best adapter) | 7.366 |
| Segment-isolated (128 tokens, correct adapter, s=20) | 7.636 |

The segment-isolated PPL is 2.3% WORSE than base and 3.7% worse than per-sequence.
The adapter is actively harmful on short segments at training scale.

**Root cause analysis:** The perturbation magnitude is mismatched. LoRA adds a
perturbation h_adapted = h_base + s * B * A * x. At training time, the model sees
full-length sequences where attention patterns and hidden state norms are calibrated
for s=20. On 128-token segments:

1. Attention patterns differ (no long-range dependencies available)
2. Hidden state statistics shift (different norm distribution)
3. The relative magnitude of the perturbation s * ||B * A * x|| vs ||h_base|| is
   miscalibrated

This is a distribution shift in the perturbation-to-signal ratio, not a failure
of the adapter content.

---

## B. The Right Question

**Wrong question:** "How do we fix adapters to work on short segments?"

**Right question:** "Given that the adapter captures domain-specific knowledge
(verified by per-sequence PPL improvement), what scaling factor s* minimizes
segment-isolated PPL as a function of segment length L?"

The adapter's B matrices encode correct domain knowledge -- Finding #310 proved
98.3% token-level classification accuracy. The issue is purely the magnitude
of the perturbation relative to the base model's signal strength at length L.

---

## C. Prior Mathematical Foundations

**LoRA perturbation analysis (Hu et al., 2022, arXiv 2106.09685):**

For a pre-trained weight matrix W_0 in R^{d x d}, LoRA adds:

  h = (W_0 + s * B * A) * x = W_0 * x + s * B * A * x

where A in R^{r x d}, B in R^{d x r}, s is the scaling factor (LORA_SCALE).

The perturbation-to-base ratio (PBR) is:

  PBR = ||s * B * A * x|| / ||W_0 * x||

**Claim 1 (PBR scaling).** For fixed A, B, the PBR is proportional to s:

  PBR(s) = s * ||B * A * x|| / ||W_0 * x||

At training time s=20, the model has converged to a PBR that balances base
model knowledge with domain adaptation. The training objective found B such
that PBR(20) is optimal for full-sequence evaluation.

**Observation (distribution shift on segments).** On 128-token segments vs
256-token full sequences, attention patterns change:

1. The causal mask restricts attention to at most 128 positions instead of 256
2. Attention entropy decreases (fewer positions to attend to, sharper patterns)
3. Hidden state norms may shift: ||x_seg|| != ||x_full|| in distribution

If ||x_seg|| > ||x_full|| on average (plausible because shorter sequences
concentrate attention), then PBR(s) = s * ||B*A*x_seg|| / ||W_0*x_seg||
may be larger than intended, making the perturbation too aggressive.

**Finding #308 evidence:** Purpose-trained adapters showed 21-37% B-matrix norm
variation, confirming that the scale of the perturbation matters significantly
for downstream quality.

### Framework Contradiction (post-experiment)

The framework in Section B assumes "the adapter captures domain-specific knowledge"
based on Finding #310 (per-sequence PPL 7.366 < base 7.465). However, THIS
experiment measured per-sequence PPL = 10.48, dramatically WORSE than base 7.99.

This directly contradicts the framework's core assumption. The per-sequence
improvement in Finding #310 used mixed-domain concatenated sequences and selected
the BEST adapter across all 5 domains -- a different evaluation protocol. When
using the correct single-domain adapter on pure-domain validation text (this
experiment's setup), the adapter HURTS on most domains even with full sequences.

**Consequence:** The premise "adapter content is correct, only scale is wrong"
is not verified in this experiment's own data. The adapter direction B*A*x may
itself be miscalibrated for pure-domain text, making the entire scale sweep
framework inapplicable. This is a structural issue that no scaling factor can fix.

The IVT argument (Section D) requires PPL(s*) < PPL(0) for some s*, which
assumes the adapter HELPS at some scale. If the adapter direction is wrong for
this evaluation context, PPL(s) >= PPL(0) for all s > 0, and IVT gives nothing.
The experimental data (monotonically increasing PPL) is consistent with this
scenario.

---

## D. Predictions

This is a Type 2 experiment: the framework (PBR scaling) is proven, but the
optimal s* for L=128 is unknown. We make qualitative predictions:

**Prediction 1 (Qualitative):** There exists s* < 20 such that segment-isolated
PPL at s* is below base PPL (7.465). Reasoning: the adapter content is correct
(98.3% token accuracy), so at sufficiently small scale the adapter provides
domain signal without overwhelming the base model.

**Prediction 2 (Monotonicity structure):** The PPL-vs-scale curve is U-shaped:
- At s=0, adapter has no effect, PPL = base PPL (7.465)
- At intermediate s*, adapter provides optimal domain signal
- At s=20 (training scale), perturbation is too large for segments, PPL > base

**Prediction 3 (Scale range):** The optimal s* is in [5, 15]. At s < 5, the
adapter effect is too small to overcome the overhead of modified computation.
At s > 15, we are already seeing degradation.

**Prediction 4 (Behavioral preservation):** At s*, behavioral quality (factual
recall) should be within 10% of per-sequence behavioral quality because the
adapter content is identical, only the magnitude differs.

**Quantitative bounds (derived):**

At s=0: PPL = 7.465 (base, by definition)
At s=20: PPL = 7.636 (measured in Finding #310)
Linear interpolation predicts crossover (PPL = base) at:
  s_cross = 20 * (7.636 - 7.465) / (7.636 - 7.465) = 20 (tautological)

This shows the curve is NOT linear -- it must be convex (U-shaped) for the
adapter to help at all. The per-sequence result (PPL 7.366 < 7.465 at s=20)
proves the adapter DOES help when the context matches training distribution.

**Kill criteria derivation:**

- K787: PPL(s*) < 7.465. If no scale helps, the adapters are fundamentally
  incompatible with segment-isolated application (distribution shift is in
  the adapter content, not just magnitude).

- K788: s* != 20. If the optimal scale IS 20, the degradation is not a scale
  confound but a fundamental issue with segment isolation.

- K789: behavioral(s*) >= 0.9 * behavioral(per-sequence). If scale reduction
  kills behavioral quality, the adapter's domain knowledge is entangled with
  its magnitude.

---

## E. Assumptions and Breaking Conditions

**Assumption 1:** The adapter's domain knowledge (captured in B matrices) is
independent of the scaling factor s. That is, the information is in the
direction of B*A*x, not its magnitude.

Breaking condition: If B was trained such that its information is encoded in
the magnitude (not direction), then scaling down destroys information.
Consequence: K787 FAIL -- no scale helps.

**Assumption 2:** The base model's forward pass on 128-token segments produces
hidden states in the same subspace as 256-token sequences.

Breaking condition: If the base model uses a fundamentally different
representation space for short sequences, adapter projections miss the target.
Consequence: K787 FAIL -- scale adjustment cannot fix subspace mismatch.

**Assumption 3:** The PPL-vs-scale curve is smooth enough that a 5-point sweep
(s in {2, 5, 10, 15, 20}) captures the optimal region.

Breaking condition: If the optimal scale is very narrow (e.g., s=7.3 +/- 0.1),
the sweep may miss it. We would see all scales worse than base.
Consequence: False negative on K787. Refinable with finer grid.

---

## F. Worked Example (d=16, r=2)

Consider a toy LoRA with d=16, r=2, s=20:
- W_0: 16x16, ||W_0|| ~ 4.0 (for random orthogonal init)
- A: 2x16, ||A|| ~ 1.0 (Grassmannian)
- B: 16x2, ||B|| ~ 0.5 (trained)
- x: 16-dim, ||x|| ~ 1.0

Full sequence (256 tok): ||W_0 * x|| ~ 4.0, ||B*A*x|| ~ 0.5
  PBR(20) = 20 * 0.5 / 4.0 = 2.5

If on 128-tok segments, ||x_seg|| is 1.2x larger (attention concentration):
  PBR(20) = 20 * 0.5 * 1.2 / (4.0 * 1.2) = 2.5 (ratio unchanged if x scales)

But if attention patterns change the DIRECTION of x such that ||A*x_seg|| is
1.3x larger (A projects differently on short-context hidden states):
  PBR(20) = 20 * 0.5 * 1.3 / (4.0 * 1.0) = 3.25 (30% larger)

To restore PBR(s*) = 2.5: s* = 20 * 2.5 / 3.25 = 15.4

This predicts s* ~ 15 for a 30% increase in A-projection norm.

---

## G. Complexity and Architecture Connection

**Computational cost:** The scale sweep costs O(|scales| * N_domains * N_segments)
forward passes. With |scales|=5, N_domains=5, N_segments=10*10=100 segments,
this is 2500 forward passes at 128 tokens each.

At ~10ms per forward pass on M5 Pro, this is ~25 seconds for PPL evaluation.
The dominant cost is model loading + adapter swapping, estimated ~10 minutes total.

**Behavioral evaluation:** Adds O(|scales| * N_domains * N_gen) generation calls.
With N_gen=5 per domain per scale, and ~2s per generation, this is ~250s = ~4 min.

**Total estimated runtime:** 15-25 minutes.

**Architecture connection:** This directly impacts the SOLE serving architecture.
If segment-isolated routing requires s* < s_train, the serving system needs a
per-context-length scale parameter. This is a single multiplication -- zero
architectural change, zero memory cost, zero additional parameters.

---

## Self-Test (MANDATORY)

1. What is the ONE mathematical property that makes the failure mode impossible?
   At s=0 the adapter has no effect and PPL = base; by continuity there exists
   s* > 0 where the adapter's domain signal helps without overwhelming the base.

2. Which existing theorem(s) does the proof build on?
   LoRA perturbation analysis (Hu et al. 2022, arXiv 2106.09685) -- additive
   perturbation structure. Intermediate value theorem -- continuous PPL(s) with
   PPL(0)=base < PPL(20) guarantees a crossing point.

3. What specific numbers does the proof predict?
   Qualitative: s* in [5, 15]. PPL(s*) < 7.465. PPL curve is U-shaped.
   The exact value of s* is the unknown this experiment discovers.

4. What would FALSIFY the proof (not just the experiment)?
   If PPL(s) is monotonically increasing from s=0 (base value) -- meaning even
   infinitesimal adapter perturbation hurts on segments. This would mean the
   adapter's DIRECTION (not magnitude) is wrong for short contexts.

   **POST-EXPERIMENT: THIS FALSIFICATION CONDITION WAS MET.** The measured curve
   is monotonically increasing from s=2 through s=20 (7.988, 8.007, 8.051, 8.084,
   8.130). The s=2 result (7.988 vs base 7.993) is -0.06%, which is ~5 nats over
   6350 tokens -- statistically indistinguishable from noise with no CI, no
   bootstrap, no multi-seed validation. The framework is therefore falsified by
   its own stated conditions: even near-infinitesimal adapter perturbation does
   not help on isolated segments. The adapter's direction, not just its magnitude,
   is wrong for segment-isolated contexts.

5. How many hyperparameters does this approach add?
   Count: 1 (s*, the segment-optimal scale). It cannot be derived from the math
   because the distribution shift between full-sequence and segment hidden states
   is data-dependent. The experiment discovers the empirical value.

6. Hack check: Am I adding fix #N to an existing stack?
   No. This is investigating a SINGLE parameter (LORA_SCALE) that was already
   present. We are finding its correct value for a new application context.
