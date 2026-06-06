# Peer Review: Shamir Expert Sharing

## Mathematical Soundness

**Shamir primitives: correct.** The polynomial construction, Horner evaluation,
and Lagrange interpolation at x=0 are textbook implementations. The derivation
in MATH.md Section 1 is standard and accurate.

**Numerical precision analysis: correct but incomplete.** The claim that
float64 intermediate computation gives exact float32 reconstruction for k <= ~7
is well-reasoned. The condition number argument (Vandermonde matrix with
evaluation points x_i = i) is sound in direction. However, the specific
condition number estimates (k=5 ~ O(10^3), k=10 ~ O(10^8)) are stated without
derivation or citation. The Vandermonde condition number for consecutive integer
points grows faster than presented -- it is known to be exponential in k, not
polynomial. For k=5 with points {1,...,5}, the actual condition number of the
Vandermonde matrix is roughly O(10^4), not O(10^3). This does not change the
conclusion (still within float64 safety margin) but the stated bounds are
hand-wavy. Not blocking.

**Lagrange basis worked example: correct.** Verified L_1(0) = 15/8,
L_3(0) = -5/4, L_5(0) = 3/8 for share points {1, 3, 5}. The sum of basis
values equals 1 (15/8 - 5/4 + 3/8 = 15/8 - 10/8 + 3/8 = 8/8 = 1), which is
the partition-of-unity property. Correct.

**Amortized overhead analysis: mathematically trivial but honestly presented.**
The paper correctly identifies the ambiguity in the kill criterion and presents
both interpretations. T_recon / (B * T_fwd) is the right formula. No issues.

**Polynomial blending analysis: correct and well-argued.** The observation that
P(epsilon) = w + epsilon * random_direction + O(epsilon^2) correctly identifies
polynomial evaluation at non-share points as noise injection, not semantic
interpolation. The empirical results confirm this (monotonic degradation with
|x|). The paper correctly concludes this is a sharing artifact, not a feature.

**One minor mathematical note:** The code uses `create_polynomial` with
a_i ~ N(0, 0.01) (sigma=0.1, so sigma^2=0.01). MATH.md says
a_i ~ N(0, sigma^2) without specifying sigma. The code choice of sigma=0.1 is
reasonable (small relative to typical weight magnitudes) but affects the
condition of the reconstruction. Larger sigma would amplify numerical errors
at high k. This coupling is not discussed.

## Novelty Assessment

**The application is novel but the value is narrow.** No prior work applies
Shamir secret sharing to individual neural network weights for fault tolerance.
The paper correctly identifies that federated learning uses Shamir for gradient
privacy (secure aggregation), which is a different use case.

**However, the novelty is in deployment infrastructure, not in the core
research goal.** The VISION.md goal is: "composable expert contributions at a
fraction of active parameters." Shamir sharing does not advance this. It does
not improve quality, reduce active parameters, enable better composition, or
make routing cheaper. It provides fault tolerance for distributed serving of
already-composed experts.

**The blending exploration was the novel research angle, and it failed.** The
paper honestly reports this: polynomial evaluation is random perturbation, not
semantic interpolation. This is the correct conclusion, and it means the
mechanism has no path to contributing to the core research goal.

**Prior art gap:** The paper does not discuss erasure coding, which is the
standard approach for fault-tolerant distributed storage. Reed-Solomon codes
are mathematically equivalent to Shamir sharing over finite fields (both use
polynomial interpolation). The systems community has decades of work on erasure
coding for distributed storage (e.g., HDFS erasure coding, Ceph). Comparing
Shamir-over-reals to standard erasure coding would have strengthened the
deployment story.

## Experimental Design

**Kill criterion 1 (quality): well-designed, trivially passed.** Testing
3 seeds, 5 share configurations, and all C(5,3)=10 subsets is thorough. The
0.000% degradation result is expected given the float64->float32 roundtrip
analysis. This is a validation of the numerical precision claim, not a
surprising finding.

**Kill criterion 2 (overhead): ambiguously defined, honestly handled.** The
experiment correctly measures per-reconstruction overhead (14-27%) and
correctly argues for amortization. The CONDITIONAL PASS verdict is appropriate.

**Missing control: reconstruction vs. save/load baseline.** The experiment
compares "reconstruct from shares" to "original model." A more informative
comparison would be "reconstruct from shares" vs. "serialize weights to disk
and reload." The overhead of Shamir reconstruction should be compared to the
overhead of the operation it replaces (loading expert weights from storage),
not to the forward pass. If loading from disk takes 5ms and Shamir
reconstruction takes 0.3ms, Shamir is actually faster than the baseline.
This comparison is absent.

**Missing control: storage overhead.** The paper mentions 5x storage (n=5
shares, each the size of the original) but does not compare to the storage
cost of full replication (also 5x for 5 nodes) or erasure coding (typically
1.5x for comparable fault tolerance). The claimed storage advantage (n/k =
5/3 = 1.67x vs 5x for full replication) is correct but assumes the comparison
is "Shamir vs. full replication on n nodes." If the comparison is "Shamir vs.
k replicas" (which gives the same fault tolerance against k-1 failures), the
storage is n/k vs 1, so Shamir uses 1.67x MORE storage than just keeping k
copies. This comparison is misleading.

Wait -- let me reconsider. k-of-n Shamir means any k of n shares suffice. With
n=5, k=3, you tolerate 2 failures. Full replication tolerating 2 failures
requires 3 replicas (each full copy), costing 3x storage. Shamir requires 5
shares (each full copy size), costing 5x storage. So Shamir actually uses MORE
storage (5x vs 3x) for the same fault tolerance. The paper's claim that Shamir
uses "n/k = 1.67x storage instead of 5x for full replication across 5 nodes"
is comparing to an unnecessarily expensive baseline (5 full copies). The fair
comparison is: to tolerate f failures, Shamir needs n shares where n >= k+f,
costing n/k overhead, while full replication needs f+1 copies. For f=2, k=3:
Shamir = 5 copies, replication = 3 copies. Shamir loses on storage.

**This undermines the practical value proposition.** Shamir's advantage over
replication is information-theoretic security (no k-1 subset reveals the
secret), but the paper explicitly discards this property by working over reals.
Without the security guarantee, Shamir-over-reals provides no advantage over
simple replication for fault-tolerant weight distribution.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry lists kill criteria:
- "reconstructed expert >2% worse than original" -- tested, passed
- "k-of-n reconstruction overhead >10% of forward pass" -- tested, killed at
  per-call level, passed at amortized level

The experiment tests exactly what the hypothesis specifies. The CONDITIONAL PASS
status is a reasonable interpretation given the ambiguity.

**However:** The hypothesis tags include "novel" and "cross-domain" (from
cryptography). The novelty assessment shows this is an infrastructure mechanism,
not a composition mechanism. It does not advance any of the core hypotheses
in the research program.

## Macro-Scale Risks (advisory)

1. **Storage cost is strictly worse than replication.** At macro scale with
   d=4096, expert weights are ~45M elements (~180MB in float32). 5 shares =
   900MB per expert. 3 replicas for the same fault tolerance = 540MB. Shamir
   is 1.67x more expensive for zero benefit over replication (since we
   discarded the security property).

2. **Reconstruction time scales linearly with weight count.** O(k^2 * |W|)
   for |W| = 45M at k=3 is ~400M FLOPs. This is small relative to a forward
   pass (~10 GFLOPs) but the Python/numpy implementation will bottleneck.
   Needs a compiled kernel.

3. **No integration path with the architecture.** This mechanism operates
   entirely outside the model -- it is a weight serialization scheme. It does
   not interact with routing, composition, pruning, or any other mechanism
   in VISION.md.

## Verdict

**KILL**

Justification:

1. **The mechanism does not advance the research goal.** VISION.md targets
   "composable expert contributions at a fraction of active parameters."
   Shamir sharing is a deployment/infrastructure mechanism that provides
   fault-tolerant weight distribution. It does not improve composition quality,
   reduce active parameters, or enable better routing. The blending angle
   (the one path to relevance) was correctly identified as failed.

2. **The practical value proposition is unsound.** Once cryptographic security
   is discarded (operating over reals, not finite fields), Shamir sharing has
   strictly worse storage efficiency than simple replication for the same fault
   tolerance level. The paper's storage comparison (5/3 = 1.67x vs 5x) uses
   an unfair baseline.

3. **The overhead kill criterion was technically triggered.** Per-reconstruction
   overhead of 14-27% exceeds the 10% threshold. The amortization argument is
   valid but applies equally to any one-time operation (including simply loading
   weights from a replica), making it a non-differentiating factor.

4. **No prior art gap exists in practice.** Erasure coding (Reed-Solomon, etc.)
   is the standard solution for fault-tolerant distributed storage and is widely
   deployed in production systems. The paper does not compare against this
   baseline.

5. **The experiment confirmed exactly what the math predicted.** Float64
   Lagrange interpolation gives exact float32 reconstruction -- this is a
   numerical precision result, not a machine learning insight. The blending
   fails because random polynomials are random -- also predictable.

The experiment was well-executed, honestly reported, and the paper is clear
about limitations. The CONDITIONAL PASS self-assessment is fair within the
narrow scope. But the mechanism has no path to contributing to the core
research program and should not consume further resources. File this as a
completed exploration with a negative relevance finding.
