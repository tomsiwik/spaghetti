# Peer Review: Room Model Geometric Routing

## Experiment Type
Guided exploration (Type 2)

## Hack Detector
- Fix count: 1 (single routing signal, zero added mechanisms). CLEAN.
- Is MATH.md a proof or a description? **Description dressed in equations.** Theorem 2 is explicitly labeled "Proof sketch" and admits it is incomplete ("Full proof requires concentration inequalities on the product of random projection and trained amplification"). The QED is conditional on an unmeasured parameter gamma_i. Theorem 3's "proof" conflates expressive capacity with actual behavior and ends with an empirical comparison, not a deduction.
- Metric used as evidence: Routing accuracy (argmax of norms vs ground-truth domain). This is a direct behavioral metric -- appropriate for the claim.
- Kill criteria source: Partially derived from proof. The 60% threshold is stated as coming from "JL + training amplification" but no derivation connects JL epsilon=0.32 to a 60% accuracy prediction. The 50% ridge agreement threshold is ad hoc.

## Self-Test Audit

1. **One-sentence impossibility property:** "B_i training on domain-i data amplifies domain-specific components of the random A_i projection, making ||h @ A_i @ B_i|| domain-discriminative." This is NOT an impossibility property -- it is a HOPE. An impossibility property would be: "Property X makes failure mode Y mathematically impossible." This answer describes a mechanism that MIGHT work. It should have said something like "JL distance preservation makes routing signal exist" -- but even that would be wrong, because as the experiment showed, JL preserves distances but B-matrix training does NOT amplify domain-specifically. **FLAG: This is a desired behavior, not a structural guarantee.**

2. **Cited theorems:** JL-lemma (real, correctly stated), Cover's theorem (real, correctly applied to d/N ratio), FlyLoRA (real paper, relevance correctly stated). However, JL conditions are applied loosely: the lemma requires the projection to be applied to the same set of points, but here each adapter i has a DIFFERENT A_i, so the JL guarantee applies per-adapter, not across adapters. More critically, JL says nothing about what happens AFTER the B-matrix transformation. **PARTIALLY VALID.**

3. **Predicted numbers:** Four specific predictions given (A-only ~14%, DeltaW >60%, ridge agreement >50%, improvement >3x). These are specific and falsifiable. However, the ">60%" and ">50%" thresholds are not derived from the proof -- they are asserted. The proof sketch depends on gamma_i (the B-amplification factor), which is the unknown being explored, so the proof cannot predict the threshold without knowing gamma_i. **The predictions are arbitrary thresholds dressed as proof-derived bounds.**

4. **Falsification condition:** "If B_i produces uniform amplification across all domains (no domain specificity), then geometric routing = A-only routing = ~14%." This is clear and targeted at Assumption 3. **GOOD.**

5. **Hyperparameter count:** 0. Correct -- the routing signal is parameter-free. **GOOD.**

6. **Hack check:** Single mechanism, no stacking. **GOOD.**

## Mathematical Soundness

### Theorem 1 (Adapter Output Decomposition)
Correct. Linearity of matrix multiplication. Trivially true. No issues.

### Theorem 2 (Geometric Routing Signal) -- PROBLEMS

**Problem 1: The "proof" is a sketch, not a proof.**
The document explicitly says "Proof sketch" and "Full proof requires concentration inequalities on the product of random projection and trained amplification." A proof sketch is not a proof. For a Type 2 guided exploration, this is acceptable IF the proven framework is clearly identified and the unknown is precisely stated. The framework (JL + decomposition) is proven; the unknown (gamma_i) is clearly identified. **Acceptable for Type 2.**

**Problem 2: JL-lemma is misapplied.**
The JL-lemma guarantees that a SINGLE random projection preserves pairwise distances among n points. Here, there are N=5 DIFFERENT random projections A_i (one per adapter). The JL guarantee applies to each A_i independently, but the routing decision compares ||h @ A_1 @ B_1|| vs ||h @ A_2 @ B_2|| vs ... -- these are scores computed through DIFFERENT projections. JL does not guarantee that the RELATIVE ordering of these scores is meaningful. JL says: "within projection A_i, distances between domain centroids are preserved." It does NOT say: "the norm through A_i @ B_i is larger for domain-i inputs than through A_j @ B_j."

The cross-projection comparison is the fundamental gap. MATH.md acknowledges this implicitly ("full bound depends on B_i amplification factor gamma_i") but does not flag it as a breakdown of the JL argument.

**Problem 3: The epsilon calculation is correct but misleading.**
epsilon ~ sqrt(log(5)/16) ~ 0.32 is correctly computed. But this bounds distortion within a SINGLE projection. The routing decision requires CROSS-projection comparison, where the distortion compounds: each A_i has its own 32% distortion, and B_i adds an unknown scaling. The effective epsilon for routing is unbounded by JL.

**Problem 4: Cover's theorem is irrelevant.**
Cover's theorem says patterns become linearly separable in high dimensions. But the geometric router does NOT perform linear separation -- it computes norms through low-rank projections. Cover's theorem applies to the ridge router (which does linear separation), not to the geometric router. Citing it here is misleading.

### Theorem 3 (Ridge Router as Upper Bound) -- PROBLEMS

**Problem 5: The "proof" contradicts its own claim.**
The theorem states ridge accuracy is an "upper bound" on geometric routing. But the proof body says: "the geometric router has MORE expressive capacity per domain than the ridge router" (because rank 16 > N=5). Then it says: "the nonlinear norm operation limits composition." So which is it? If the geometric router has more capacity, the ridge router is NOT an upper bound. If the norm operation limits it, then what limits it is not the rank but the information loss from taking norms. The theorem title and conclusion contradict the proof body.

**Problem 6: The rank comparison is wrong.**
The proof says "the ridge router sees h through an unconstrained rank-min(d,N) = N = 5 window." This is incorrect. The ridge router W* in R^{d x N} has rank <= N = 5, true. But it operates in the FULL d=2560 dimensional space and can use ANY 5-dimensional subspace. The geometric router operates through N=5 independent rank-16 subspaces, each constrained to be Grassmannian-orthogonal. The ridge router chooses the OPTIMAL 5D subspace; each geometric router chooses a RANDOM 16D subspace. The geometric router does not have "more capacity" -- it has more dimensions but in random, potentially useless, directions.

### Section C.3 (Adapter Training as Directional Amplifier)

**Problem 7: The training gradient argument is hand-waving.**
The gradient dB_i/dt = -eta * E[...] is correct, but it does not prove that B_i becomes domain-specific. The gradient aligns B_i with the loss-reduction direction FOR domain-i data, but ALL adapters are trained on language modeling -- the loss function is the same. The domain specificity comes from the DATA distribution, not the loss function. For domain-specific amplification, you need B_i to learn features that are UNIQUE to domain i. If domain-i and domain-j share 80% of their language modeling structure, B_i and B_j will be 80% similar, and the geometric routing signal will be dominated by the shared component, not the domain-specific 20%.

This is exactly what the experiment found: the shared language modeling structure dominates (mean weight correct = 0.201, mean weight other = 0.200), and the domain-specific signal is negligible.

### Worked Example (Section G) -- MISLEADING

**Problem 8: The toy example uses non-overlapping A-matrices, which contradicts the real setup.**
A_0 selects dimensions [1,2], A_1 selects dimensions [3,4]. These are perfectly orthogonal and non-overlapping. In this case, routing trivially works because each adapter literally "sees" a different part of the input. But in the real setup, Grassmannian A-matrices are random 16-dimensional subspaces of R^2560. They are near-orthogonal (low cross-correlation), but they all span overlapping regions of the full space. The toy example gives a false intuition: it suggests routing works because of A-matrix separation, but real Grassmannian matrices do not have the clean non-overlapping structure that makes the toy example work.

MATH.md acknowledges this ("In reality, with random Grassmannian A-matrices in R^2560 projecting to R^16, the separation is weaker") but the worked example is still misleading as a proof device.

## Prediction vs Measurement

PAPER.md contains a proper prediction-vs-measurement table. Results:

| Prediction | Predicted | Measured | Match? |
|-----------|-----------|----------|--------|
| A-only routing ~14% | ~14% | 16.0% | YES |
| DeltaW routing > 60% | > 60% | 31.2% | NO (KILLED) |
| Ridge agreement > 50% | > 50% | 17.9% | NO (KILLED) |
| Improvement over A-only > 3x | > 3x | 2.0x | NO |
| Mean correct weight > 0.25 | > 0.25 | 0.201 | NO |
| Margin ratio > 1.2 | > 1.2 | 0.848 | NO |

The table is thorough and honest. The kill is well-documented.

### Data Consistency Check (results.json vs PAPER.md)

- A-only accuracy: results.json says 0.1598 (16.0%), PAPER.md says 16.0%. **MATCH.**
- DeltaW per-layer best: results.json says 0.3125 (31.2%), PAPER.md says 31.2%. **MATCH.**
- Ridge accuracy: results.json says 0.9835 (98.3%), PAPER.md says 98.3%. **MATCH.**
- Agreement: results.json says 0.1788 (17.9%), PAPER.md says 17.9%. **MATCH.**
- Mean correct weight: results.json says 0.2013, PAPER.md says 0.201. **MATCH (rounding).**
- Margin ratio: results.json says 0.8485, PAPER.md says 0.848. **MATCH (rounding).**
- B-norms (math): results.json says 1.828, PAPER.md says 1.828. **MATCH.**
- Disagree geo right: results.json says 0.00309 (0.3%), PAPER.md says 0.3%. **MATCH.**

**Minor discrepancy:** PAPER.md reports "DeltaW single module best: k_proj at 19.5%". results.json has k_proj at 0.1953 = 19.5%. **MATCH.** No data inconsistencies found.

### One methodological concern
The multi-layer geometric routing (Method 4) uses hidden states from the FINAL layer (h extracted at model output) but applies them to A-matrices and B-matrices from ALL 30 layers. Each layer's A_i and B_i are designed to process the hidden state at THAT layer, not the final-layer hidden state. This mismatch could suppress routing signal. PAPER.md acknowledges this in Limitations section 1. However, the per-layer analysis (Method 3) also uses final-layer hidden states applied to per-layer adapters, which has the same problem. Only layer 29 (the last layer) would have matched hidden states, and indeed layer 29 gives the best result (31.2%). This is consistent with the mismatch hypothesis.

This is not a fatal flaw in the experiment -- it is a limitation that could inflate the negative result. But even at layer 29 (where h matches), accuracy is only 31.2%, far below the 60% threshold. The conclusion holds.

## NotebookLM Findings

NotebookLM was not available for this review. The analysis above was conducted manually.

## Novelty Assessment

The idea of using adapter geometry as a routing signal is natural and has partial precedent:
- **FlyLoRA (arXiv 2510.08396)** showed frozen random A-matrices can serve as implicit routers, which is the direct precursor.
- **Finding #302** already showed A-only routing gives ~14% (random), making this a natural next step: does adding B fix it?

The novelty is modest -- it is a logical extension of Finding #302, testing whether B-matrices rescue the failed A-only routing. The negative result itself is valuable: it establishes that Grassmannian orthogonality (designed for interference prevention) is structurally incompatible with domain-aligned routing.

The structural explanation in PAPER.md ("Grassmannian A-matrices are optimized for interference prevention, not domain alignment") is the key insight and is novel as an explicit negative finding.

## Macro-Scale Risks (advisory)

Not applicable -- this is a killed experiment. The structural impossibility argument (Grassmannian orthogonality contradicts domain alignment) holds at any scale.

## Structural Analysis of the Kill

The impossibility structure claimed in PAPER.md -- "Grassmannian orthogonality contradicts domain alignment" -- is:

**Partially correct but not formally proven.** The argument is:
1. Grassmannian A_i are designed to be orthogonal to each other (A_i^T A_j ~ 0).
2. Domain routing needs A_i to be aligned with domain-i structure.
3. These two requirements conflict.

This argument is intuitive but not rigorous. Orthogonality between A_i and A_j does NOT prevent A_i from being aligned with domain-i structure. In R^2560 with r=16, there is plenty of room for N=5 subspaces to be both mutually orthogonal AND individually aligned with domain centroids. The real issue is that A_i are RANDOM -- they are not designed to align with anything. The Grassmannian constraint ensures orthogonality, but the randomness ensures no alignment. If A_i were chosen to be both orthogonal AND domain-aligned (which is possible when N*r << d), geometric routing might work.

So the impossibility is not from Grassmannian orthogonality per se, but from the combination of:
1. A_i are random (not domain-aligned).
2. B_i training does not compensate for this misalignment.
3. The norm aggregation loses directional information.

This is a more nuanced conclusion than what PAPER.md states.

## Verdict

**PROCEED** (as a killed experiment -- the kill is justified)

The kill decision is correct and well-supported by evidence. Both kill criteria fail decisively (31.2% vs 60% threshold, 17.9% vs 50% threshold). The prediction-vs-measurement table is thorough. Data is internally consistent. The experiment cleanly narrows the unknown: B-matrix training does NOT provide sufficient domain-specific amplification to make adapter geometry a viable routing signal.

**Specific issues that should be noted but are not blocking:**

1. **Theorem 2 is a sketch, not a proof.** For Type 2 (guided exploration), this is acceptable as long as the framework and unknown are clearly identified, which they are. The proof sketch correctly identifies gamma_i as the key unknown that the experiment measures.

2. **Kill criteria thresholds (60%, 50%) are not proof-derived.** They are reasonable heuristics (3x random for routing, better-than-chance for agreement) but the document claims they come from "JL + training amplification" without deriving them. This is a minor presentation issue -- the actual kill is so decisive (31.2% vs 60%) that even generous thresholds would not change the outcome.

3. **The impossibility structure is asserted, not derived.** "Grassmannian orthogonality contradicts domain alignment" oversimplifies the actual failure. The correct impossibility structure is: random projections + norm aggregation + insufficient B-amplification = no routing signal. This should be recorded more precisely in the finding.

4. **JL-lemma and Cover's theorem are cited but not correctly applied** to the cross-projection comparison that routing requires. This does not affect the kill decision but weakens the mathematical framework.

5. **The worked example is misleading** because it uses non-overlapping A-matrices that trivially succeed, unlike the real Grassmannian setting.

None of these issues change the kill verdict. The experiment is a clean negative result that advances understanding of what adapter geometry can and cannot do.
