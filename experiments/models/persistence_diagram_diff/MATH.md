# Weight-Space Persistent Homology for Adapter Composition

## Experiment Type: Guided Exploration (Type 2)

The mathematical framework (stability theorem for PH) is proven. The unknown is
empirical: what is the actual bottleneck distance between persistence diagrams of
base vs composed weight matrices, and which topological features are lost?

## Step A: Diagnose the Disease

**Problem:** When we compose adapters via W' = W + sum(s_i * B_i), we change the
weight matrix. This changes the geometry of the row point cloud {w_1, ..., w_n}
in R^d. Some topological features (connected components, loops) in this point cloud
may be destroyed or created. If destroyed features correspond to meaningful computational
pathways, composition degrades model quality.

**The predecessor experiment (exp_pathway_graph_bitnet2b) failed** because it used
co-activation graphs with sparsification, which mechanically creates persistence
artifacts. The disease was the methodology (sparsification), not the underlying
question (does composition change topology?).

**Root cause for this experiment:** We do not know whether adapter composition is
topologically lossless. The theoretical question is well-posed: does the low-rank
perturbation W -> W + Delta (where Delta = sum(s_i * B_i) has rank <= 5*16 = 80)
preserve the persistent homology of the row point cloud?

## Step B: The Right Question

NOT: "How do we prevent topological damage from composition?"
RIGHT: "What is the topological cost (bottleneck distance) of low-rank composition,
and does the stability theorem give us a useful bound?"

The answer comes from the Stability Theorem for persistent homology.

## Step C: Prior Mathematical Foundations

### Stability Theorem (Cohen-Steiner, Edelsbrunner, Harer, 2007)

**Theorem (Algebraic Stability).** For persistence modules arising from Rips
filtrations on finite metric spaces (X, d_X) and (Y, d_Y), if there exists a
correspondence C between X and Y with distortion

  delta = max_{(x,y) in C} d(x, y)

then the bottleneck distance between persistence diagrams satisfies:

  d_B(Dgm(X), Dgm(Y)) <= delta

For our setting, X = {rows of W} and Y = {rows of W'} where W' = W + Delta.
The natural correspondence is row i <-> row i (same index). The distortion is:

  delta = max_i ||w'_i - w_i|| = max_i ||Delta_i||

where Delta_i is the i-th row of the perturbation matrix.

### Rips Complex on Weight Rows (Rieck et al., 2018, arXiv:1812.09764)

Rieck et al. showed that persistent homology of Rips complexes built on rows of
weight matrices captures meaningful structural information about neural networks.
They demonstrated:
- PH of weight rows detects training progress
- PH features correlate with generalization (test accuracy)
- The approach works across architectures (CNNs, RNNs)

Their filtration: for a weight matrix W in R^{n x d}, treat the n rows as points
in R^d. Build a Vietoris-Rips complex with increasing radius parameter epsilon.
The resulting persistence diagram captures the multi-scale topology of the weight
row point cloud.

### Low-Rank Perturbation Structure

Our perturbation is Delta = sum_{i=1}^{5} s_i * B_i where each B_i in R^{r x d}
with r = 16. The adapters only have lora_b matrices (no lora_a), so the perturbation
lives in a subspace of dimension at most min(5*16, d) = 80.

For a weight matrix W in R^{n x d}, the perturbation adds a different vector to
each row depending on the adapter structure. Since the B_i are (16, d_out) matrices,
the perturbation to W (n x d_out) depends on how LoRA integrates.

**Key insight:** The perturbation to each row of W lives in a subspace of dimension
at most 80 (the span of all adapter B rows). This means the topological change is
geometrically constrained to a low-dimensional subspace.

## Step D: Proof of Guarantee (Bounded Degradation)

**Theorem 1 (Bottleneck Bound for Low-Rank Composition).**
Let W in R^{n x d} be a weight matrix with row point cloud P = {w_1, ..., w_n}.
Let Delta in R^{n x d} be a perturbation with rank(Delta) <= k, and let
P' = {w_1 + delta_1, ..., w_n + delta_n} where delta_i is the i-th row of Delta.

Then the bottleneck distance between the Rips persistence diagrams satisfies:

  d_B(Dgm(P), Dgm(P')) <= max_i ||delta_i||_2

*Proof.* The identity map f: P -> P' given by f(w_i) = w_i + delta_i is a
correspondence between the two point clouds. The distortion of this correspondence is:

  delta = max_i ||f(w_i) - w_i||_2 = max_i ||delta_i||_2

By the Algebraic Stability Theorem (Cohen-Steiner et al., 2007, Theorem 5.2),
applied to the Rips filtrations on (P, d_L2) and (P', d_L2):

  d_B(Dgm(P), Dgm(P')) <= delta = max_i ||delta_i||_2

QED.

**Corollary 1.** Features with persistence > 2 * max_i ||delta_i||_2 are guaranteed
to survive composition (they may shift but cannot be destroyed).

*Proof.* A feature with persistence p in Dgm(P) is matched to a feature in Dgm(P')
within bottleneck distance delta. The matched feature has persistence at least
p - 2*delta. If p > 2*delta, the matched persistence is > 0, so the feature survives.
QED.

**Corollary 2.** Only features with persistence <= 2 * max_i ||delta_i||_2 can be
destroyed by composition. The number of destroyed features is bounded by the number
of features in this "vulnerability window."

## Step D (cont.): Predictions

### Quantitative predictions we can derive:

1. **P1: Bottleneck distance is bounded.** d_B <= max_i ||delta_i||_2. We will compute
   this bound from the actual adapter weights and verify the measured d_B is below it.

2. **P2: High-persistence features survive.** Features with persistence > 2*delta
   must appear in both diagrams. We count how many features exceed this threshold.

3. **P3: Random perturbation baseline.** A random perturbation with the same max row
   norm should produce a similar bottleneck distance. If the structured (adapter)
   perturbation produces a DIFFERENT bottleneck distance than a random perturbation
   of the same norm, that reveals structure in the adapters.

4. **P4: Vulnerability window.** The number of features in the vulnerability window
   [0, 2*delta] predicts the maximum number of features that can be lost.

### What we do NOT predict (guided exploration unknowns):

- The exact bottleneck distance (only the upper bound)
- Whether lost features have semantic meaning (K627)
- The relationship between topological cost and behavioral quality

## Step E: Assumptions & Breaking Conditions

**A1: Row-wise topology is meaningful.** The Rips complex on weight rows captures
functionally relevant structure. This is supported by Rieck et al. (2018) for CNNs/RNNs
but not proven for transformer MLP layers specifically.
If violated: bottleneck distance is non-zero but meaningless for quality prediction.

**A2: Euclidean metric is appropriate.** We use L2 distance between rows. If the
functionally relevant metric is cosine similarity or some other metric, the Rips
complex may miss important structure.
If violated: features may appear/disappear due to metric choice, not composition.

**A3: Subsample represents the full matrix.** For computational feasibility, we
subsample rows. If important topology lives in the unsampled rows, we miss it.
If violated: measured bottleneck distance is an underestimate.

**A4: Scale factors are representative.** We use the domain-dependent scales from
Finding #217. Different scales would give different perturbation norms and thus
different bottleneck bounds.
If violated: results are scale-specific, not universal.

## Step F: Worked Example (d=16)

Consider a toy weight matrix W in R^{4 x 4}:
```
W = [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]]
```

Rows are the 4 standard basis vectors. Pairwise distances are all sqrt(2) ~ 1.414.
The Rips persistence diagram for H0 has 3 features with persistence = sqrt(2)
(the 4 components merge at epsilon = sqrt(2)/2 into one).

Now add a perturbation Delta with rank 1:
```
Delta = 0.1 * [[1, 1, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 0, 0]]
```

max_i ||delta_i|| = 0.1 * sqrt(2) ~ 0.141

New rows: [1.1, 0.1, 0, 0], [0.1, 1.1, 0, 0], [0.1, 0.1, 1, 0], [0.1, 0.1, 0, 1]

Rows 1-2 are now closer: ||(1.1,0.1)-(0.1,1.1)|| = sqrt(2) (unchanged in first 2 coords,
but their full distance decreased because both moved toward the same direction).

Actually: ||r1-r2|| = sqrt((1.1-0.1)^2 + (0.1-1.1)^2) = sqrt(1+1) = sqrt(2) (same)
||r1-r3|| = sqrt((1.1-0.1)^2 + (0.1-0.1)^2 + (0-1)^2) = sqrt(1+0+1) = sqrt(2) (same)

In this case the perturbation is rank-1 but shifted all rows uniformly, so pairwise
distances don't change. Bottleneck distance = 0.

For a non-uniform perturbation:
```
Delta = [[0.3, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]
```

max ||delta_i|| = 0.3. Only row 1 moves: [1.3, 0, 0, 0].
New distance from row 1 to row 2: sqrt(1.69 + 1) = sqrt(2.69) ~ 1.640 (was sqrt(2) ~ 1.414).
Stability bound: d_B <= 0.3.

The H0 diagram changes because the merge order shifts: row 1 now merges later.
Measured bottleneck distance will be <= 0.3.

## Step G: Complexity & Architecture Connection

**Computation:** For n subsampled rows in R^d:
- Rips complex: O(n^3) time, O(n^2) memory (pairwise distance matrix)
- With n=500 rows, d=2560: ~0.1s per matrix (from feasibility test)
- 30 layers x 7 modules x 2 diagrams (base + composed) = 420 Rips computations
- Total: ~42 seconds for computation

**Memory:**
- Base model: ~1.7GB (BitNet-2B-4T quantized)
- 5 adapters: ~5 x 50MB = 250MB
- Distance matrices: 500^2 * 4 bytes = 1MB each
- Total: well within 48GB

**Architecture:** The adapters modify MLP (gate/up/down) and attention (q/k/v/o)
projections at every layer. The weight rows of these projections define the
"feature detectors" of the model. Persistent homology on these rows measures
the multi-scale clustering structure of the feature detectors.

## Self-Test

1. **What is the ONE mathematical property that makes the failure mode impossible?**
   The Stability Theorem guarantees bottleneck distance is bounded by max row perturbation norm, so topological damage cannot exceed a computable bound.

2. **Which existing theorem(s) does the proof build on?**
   Algebraic Stability Theorem (Cohen-Steiner, Edelsbrunner, Harer, 2007, Theorem 5.2); Rips filtration validity from Rieck et al. (2018, arXiv:1812.09764).

3. **What specific numbers does the proof predict?**
   d_B <= max_i ||delta_i||_2 (computable from adapter weights); features with persistence > 2*delta survive. **Caveat:** Both predictions turned out to be vacuously true at current adapter scale — the bound is 10-100x loose. The only genuinely informative prediction was P3 (random baseline comparison), which is not derived from the proof.

4. **What would FALSIFY the proof (not just the experiment)?**
   The proof cannot be falsified (it's a consequence of the stability theorem). The experiment could show the bound is VACUOUS (d_B << bound, so the theorem is true but uninformative).

5. **How many hyperparameters does this approach add?**
   Count: 1 (number of subsampled rows). Derived: limited by Rips complexity O(n^3); we use max feasible n.

6. **Hack check:** No stack of fixes. This is a single measurement using an established tool (PH).
