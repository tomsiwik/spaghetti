# Peer Review: KV-Cache Reuse Across Adapter Switches

## Experiment Type
Frontier extension (Type 3).

**Type compliance check:** MATH.md states the proven result being extended
(Finding #305, segment isolation eliminates cross-attention contamination) and
identifies the mathematical gap (segment isolation discards cross-segment context).
Meets frontier-extension requirements. Finding status correctly capped at killed
(which is below the provisional cap).

## Hack Detector
- Fix count: 1 (KV-cache reuse, a single mechanism). No flag.
- Is MATH.md a proof or a description? **Mostly description dressed in theorems.**
  See "Mathematical Soundness" section below for details.
- Metric used as evidence: PPL on segment B. This is a reasonable proxy for the
  behavioral question (does cross-segment context from a different adapter help
  or hurt quality).
- Kill criteria source: K781 derived from Theorem 2 (1.6% perturbation -> 3%
  threshold). K782 from FLOP argument. K783 from Theorem 3 (information theory).
  Derivation is traceable, though the proofs themselves are flawed.

## Self-Test Audit

1. **One-sentence impossibility property:** "KV-cache entries from adapter A
   represent the CORRECT processing of segment A, so reusing them provides
   semantically faithful cross-segment context." This is a single coherent claim.
   PASS.

2. **Cited theorems:**
   - Data processing inequality (Shannon, 1948): Real theorem, but misapplied.
     The DPI requires that the channel is fixed -- the same decoder must interpret
     the data. When adapter B's queries read adapter A's keys, the "channel" has
     changed. See Error Analysis below.
   - Submultiplicativity of operator norm: Correctly stated, correctly applied
     in the bound computation.
   - Finding #305: Legitimate prior finding.
   - arXiv:2512.17910 (Cross-Model KV Reuse): Cited but critically different
     setting -- that paper shares KV between a base model and its LoRA variant
     (same base + small perturbation), not between two different LoRA adapters
     with potentially orthogonal modifications.
   PARTIAL PASS. The DPI application is invalid and the arXiv reference is
   misleading about the setting.

3. **Predicted numbers:** Cross-adapter error < 2% (Theorem 2), KV-reuse within
   3% of full-recompute (K781), PPL_ctx < PPL_iso (K783), speedup > 1.2x (K782).
   These are specific and falsifiable. PASS.

4. **Falsification condition:** "If KV-reuse produces WORSE PPL than isolated
   evaluation." This directly targets Theorem 3. The experiment measured exactly
   this (+6.27% worse). PASS.

5. **Hyperparameter count:** 0. Correct -- no new hyperparameters introduced.
   PASS.

6. **Hack check:** Single mechanism addressing single disease. PASS.

## Mathematical Soundness

### Theorem 1 (KV-Cache Preserves Correct Adapter Attribution)

**Verdict: Not a theorem. This is a semantic argument with "QED" appended.**

The "proof" argues that adapter A's keys for segment A are "semantically correct"
because adapter A is the right adapter for that content. This is reasonable
intuition but it is not a mathematical proof. There is no formal structure:
no assumptions stated as mathematical objects, no quantitative conclusion
derived through deduction. "Semantically faithful" is not a measurable quantity.

Furthermore, the claim itself is not obviously true. The question is not
whether adapter A is "correct" for segment A's content. The question is whether
adapter B's query projections can extract useful information from adapter A's key
projections. These projections live in potentially different subspaces of R^d.
Correctness of the adapter for the content does not guarantee compatibility
between the query and key representation spaces.

The argument also elides a critical point: in standard transformers, the Q/K/V
projections are trained jointly so that queries can "read" keys. Cross-adapter
Q/K pairing breaks this joint training assumption.

### Theorem 2 (Query-Key Compatibility)

**Verdict: Contains a valid bound but the bound is vacuous.**

The derivation correctly expands the cross-adapter attention score into four
terms and correctly applies submultiplicativity. However, there are two
problems:

**Problem 1: Self-contradicting bound.** The initial claim says the cross-adapter
term is O(alpha^2 * r^2 / d^2) = O(0.0156) ~ 1.6%. But three paragraphs later,
the careful derivation yields O(alpha^2 * r / d) = 400 * 4 / 2560 = 0.625,
which is 62.5%. MATH.md acknowledges this is "a LOOSE upper bound" and then
appeals to Grassmannian orthogonality to argue the true value is smaller. But:

- The 1.6% figure uses r^2/d^2. The correct bound uses r/d (since ||B||_op ~
  sqrt(r), not r). The paper switches between these without flagging the
  discrepancy.
- The appeal to A^A perp A^B saving the bound is hand-waved. The cross-adapter
  term is (A^B)^T (B^B)^T B^A A^A. Even if A^A perp A^B in the r-dimensional
  subspace sense (A^B^T A^A = 0 as r x r matrices), the product
  (A^B)^T (B^B)^T B^A A^A involves B matrices that can correlate the subspaces.
  The claim that this "projects through orthogonal subspaces" is only true if
  (B^B)^T B^A is close to identity or zero, which is not guaranteed.

**Problem 2: Layer accumulation hand-waved.** The claim that residual connections
give O(L * epsilon^2) total error cites no theorem. The standard result for
residual networks (from e.g., Veit et al. 2016) is that residual paths create
an ensemble of shallow networks, but this does not give the claimed O(L * eps^2)
bound. With 28 layers and eps = 0.625, even the charitable eps^2 estimate gives
28 * 0.39 = 10.9, which is > 1 and therefore vacuous as a perturbation bound.

### Theorem 3 (Context Recovery Bound)

**Verdict: Invalid. Data processing inequality is misapplied.**

The DPI states: H(Y|X) <= H(Y) for any random variables X, Y where X is
observed data and Y is the quantity to predict. This is correct and uncontroversial.

But the application here is wrong. The DPI requires that the observation channel
is the identity or at least a fixed function. When segment B tokens attend to
segment A's KV-cache, they are not "observing" segment A. They are observing
**adapter A's internal representation of segment A as interpreted by adapter B's
query projections**. This is a lossy, potentially adversarial channel.

An analogy: the DPI says that seeing a photo of a document gives you at least
as much information about the document as not seeing it. But this requires that
the "photo" is a faithful representation. If the "photo" was taken through a
distorted lens (adapter A's representation read by adapter B's queries), it
could actively mislead -- you might read the wrong words.

The PAPER.md correctly identifies this flaw in the "Error 1" section. The
experiment proves the misapplication: KV-reuse PPL (5.704) is worse than
isolated PPL (5.367), directly contradicting Theorem 3's prediction.

## Prediction vs Measurement

PAPER.md contains a clear prediction-vs-measurement table. This is well done.

| Prediction | Source | Measured | Match |
|-----------|--------|----------|-------|
| PPL_ctx < PPL_iso | Theorem 3 | 5.704 > 5.367 (+6.27%) | NO |
| KV-reuse within 3% of full-recompute | Theorem 2 | 13.26% gap | NO (wrong direction -- KV better) |
| Speedup > 1.2x | FLOP argument | 0.69x | NO |
| Cross-adapter error < 2% | Theorem 2 | Not directly measured | N/A |

**K781 interpretation note:** The kill criterion measures |kv - fr| / fr. The
gap is 13.26% but in the favorable direction (KV-reuse is BETTER than
full-recompute). The criterion was designed expecting KV-reuse to be close to
full-recompute but potentially slightly worse. Instead, full-recompute is far
worse because it uses the wrong adapter on segment A. The criterion as stated
(absolute gap <= 3%) kills this, but the finding that KV-reuse beats
full-recompute by 13% is actually evidence FOR Theorem 1's argument (correct
adapter matters). PAPER.md does not discuss this nuance -- it reports the kill
as if the direction is irrelevant.

## Honest Reporting Assessment

**Strengths:**
- The three-way comparison (isolated / KV-reuse / full-recompute) is well
  designed and informative.
- Per-pair breakdown shows granular results. The math+medical exception is
  noted and interpreted correctly.
- The "Why the Proof Failed" section (Errors 1-3) is exceptionally honest and
  analytically precise. The authors identified the exact mathematical flaws in
  their own proofs.
- Raw NLL/n values are provided in results.json, enabling independent
  verification.
- Limitations section is comprehensive.

**Weaknesses:**
- PAPER.md line 69: "This ordering is the OPPOSITE of what Theorem 3 predicted.
  The expected ordering was Full-recompute < KV-reuse < Isolated." This is not
  what Theorem 3 predicted. Theorem 3 only predicted PPL_ctx <= PPL_iso. It made
  no prediction about full-recompute ordering. The expected ordering
  "Full-recompute < KV-reuse" is from Theorem 1, not Theorem 3. Minor confusion.

- The K781 failure is misleadingly reported. KV-reuse beating full-recompute
  by 13% is a positive result that supports the core intuition (correct adapter
  attribution matters). Framing it as a failure because the absolute gap exceeds
  3% obscures the finding that full-recompute is the worst strategy.

- The paper concludes "Segment isolation is the correct strategy" but the
  actual finding is more nuanced: the isolated > KV-reuse > full-recompute
  ordering shows that adapter-content alignment dominates cross-segment context.
  This is a finding about the STRENGTH of adapter specialization, not about
  whether context is inherently harmful.

## Kill Criteria Fairness

**K781 (within 3% of full-recompute): Poorly designed.**
The 3% threshold was derived from Theorem 2's 1.6% perturbation bound, which
predicted KV-reuse and full-recompute would produce similar results. The
underlying assumption was wrong: the two strategies have fundamentally different
adapter assignments for segment A, making them far apart. The criterion
measures the wrong thing -- it should have measured whether KV-reuse closes
the gap between isolated and full-context (oracle) evaluation, not whether
it matches full-recompute.

**K782 (speedup > 1.2x): Reasonable but doomed by implementation.**
The FLOP argument (save 128-token recomputation) is correct in principle.
But the comparison is against full-recompute (single 256-token pass), which
benefits from batch efficiency. The fair comparison for latency is KV-reuse
vs isolated (which requires two separate passes). KV-reuse (0.150s) IS faster
than isolated (0.164s) -- a 1.09x speedup -- because the second pass benefits
from cached KV. The criterion compares against the wrong baseline.

**K783 (cross-segment context > 0% improvement): Reasonable and fairly failed.**
This is the core behavioral question and is correctly defined. KV-reuse hurts
by 6.27% vs isolated. The kill is justified.

## Impossibility Structure Assessment

PAPER.md identifies three errors in the proofs. Evaluating each:

**Error 1 (DPI misapplied):** Correctly identified. The key insight -- that
cross-adapter KV entries are noise-contaminated observations, not clean
observations -- is precise and well-stated.

**Error 2 (Perturbation bound too loose):** Partially correct. The paper notes
that alpha * r / d = 0.125 is not small, and that layer-wise accumulation
amplifies the error. However, the paper misses the deeper point: the tight
bound (62.5%, computed in the proof itself) is already vacuous as a perturbation
bound. There is no need to appeal to layer accumulation -- a single layer
already has a non-perturbative cross-adapter term.

**Error 3 (Context not always helpful):** This is a restatement of Error 1,
not an independent error. The reason "more context = lower PPL" fails is
precisely because the context is noise-contaminated (Error 1). Listing it
separately inflates the analysis.

**Missing impossibility insight:** The paper identifies an important observation
in passing (line 152-153): "The Grassmannian orthogonality that protects
composition (||cos|| < 0.05) is exactly what makes KV-cache reuse fail:
orthogonal adapters produce maximally different key/value representations."
This is the real impossibility structure and deserves more prominence. The
mechanism that guarantees non-interference in weight space (A^A perp A^B)
GUARANTEES that adapter A's KV projections land in a different subspace than
adapter B's queries expect. Non-interference and KV-cache compatibility are
mathematically contradictory requirements under the Grassmannian skeleton.
This is the central negative finding and should be stated as a theorem.

## Novelty Assessment

- arXiv:2512.17910 studies KV-cache sharing between a base model and its
  LoRA-adapted variant. This experiment studies sharing between two different
  LoRA adapters -- a materially different and harder setting. Novel in the
  context of the architecture.

- The finding that Grassmannian orthogonality implies KV-cache incompatibility
  is novel and architecturally significant. It means the Pierre architecture
  must choose between adapter isolation (current design) and cross-adapter
  context sharing. They cannot coexist under Grassmannian constraints.

## Macro-Scale Risks (advisory)

- At larger scale with more layers, the cross-adapter error would likely be
  even worse (more layers to accumulate through).
- However, with smaller alpha/r relative to d, the perturbation would be
  smaller. The experiment uses alpha=20.0 which is relatively large.
- The impossibility argument (Grassmannian orthogonality implies KV incompatibility)
  is scale-invariant and would hold at any scale.

## Verdict

**KILL (confirmed)**

The kill is justified. All three kill criteria fail, and the most important one
(K783: does cross-segment context help?) fails decisively in the wrong direction
(+6.27% PPL degradation).

However, the kill should be recorded with these qualifications:

1. **The proofs were wrong, but the experiment was excellent.** The three-way
   comparison produced a clean, interpretable result: isolated > KV-reuse >
   full-recompute. The "Why the Proof Failed" section is a model of honest
   post-mortem analysis.

2. **K781 was a bad criterion.** KV-reuse beating full-recompute by 13% is
   evidence FOR the correct-adapter-attribution argument (Theorem 1). The
   criterion should have measured KV-reuse vs isolated, not KV-reuse vs
   full-recompute.

3. **The real finding is an impossibility theorem** that should be stated
   formally: under Grassmannian constraints, adapter orthogonality and
   cross-adapter KV-cache compatibility are mutually exclusive. This should
   be elevated to a Finding with impossibility-structure status.

4. **One missed opportunity:** The math+medical pair showing +1.72% improvement
   suggests that KV-cache reuse might work for adapters within the same
   semantic cluster (where A matrices might not be maximally orthogonal).
   A follow-up could test within-cluster KV-cache reuse -- but this would
   require relaxing the Grassmannian constraint for clustered adapters,
   which conflicts with the core architecture.

### Items that would warrant REVISE (if not already killed):

1. Theorem 1 is a semantic argument, not a proof. Needs formal Q/K subspace
   compatibility analysis.
2. Theorem 2 contains a self-contradicting bound (1.6% vs 62.5%). The 1.6%
   figure is wrong and should be corrected to 62.5%.
3. Theorem 3's application of DPI is invalid (wrong channel model).
4. PAPER.md should note that K781's direction (KV-reuse beating full-recompute)
   is actually evidence for adapter-attribution correctness.
5. The impossibility structure (Grassmannian orthogonality implies KV
   incompatibility) should be stated as a formal theorem.
