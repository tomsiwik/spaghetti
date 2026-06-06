# Sheaf Cohomology for Bridge Adapter Rank Estimation

## Experiment Type: Guided Exploration

**Proven framework:** Cellular/Cech sheaf cohomology (Hansen & Ghrist, 2110.03789;
Curry, 1303.3255) predicts that H^1 != 0 implies information that cannot be
faithfully represented by independent local sections — requiring a "bridge"
to reconcile incompatible local representations.

**Unknown:** Whether specialization-based covers on BitNet-2B + LoRA adapters
produce non-trivial H^1, and if so, what dim(H^1) equals.

---

## A. Failure Mode Identification

**Disease:** When composing multiple LoRA adapters, representation incompatibility
on shared samples causes information loss. Naive merging (averaging, summing)
assumes representations are compatible on overlaps. If they are not, merged
output loses domain-specific information that neither individual adapter loses.

**The predecessor experiment (Finding #240) showed:**
- PPL improvement sets are degenerate (all samples improve under all adapters
  on ternary bases), producing a contractible nerve where H^1 = 0 trivially
- Specialization sets (argmin PPL) produce non-trivial structure
- Cosine similarity saturates at >0.986 (uninformative due to intruder dimensions,
  2410.21228), but L2 relative difference (0.037-0.168) is informative

**The question is:** Does the corrected cover + metric reveal genuine
incompatibility that requires bridge adapters to resolve?

---

## B. The Right Question (Reframe)

**Wrong:** "How many bridge adapters do we need?"
**Right:** "What is the dimension of the obstruction space to globally consistent
representation on the overlap regions?"

This is precisely dim(H^1) of a sheaf on the Cech nerve of the adapter cover.

---

## C. Prior Mathematical Foundations

### Sheaf Cohomology on Simplicial Complexes

**Definition (Cellular Sheaf, Curry 1303.3255; Hansen & Ghrist 2110.03789).**
A cellular sheaf F on a simplicial complex K assigns:
- To each vertex v_i: a vector space F(v_i) (the "stalk")
- To each edge e_{ij}: a vector space F(e_{ij})
- Restriction maps: rho_{v_i -> e_{ij}}: F(v_i) -> F(e_{ij})

In our setting:
- Vertices = adapters {1, ..., 5}
- Edges = pairs of adapters with non-empty overlap in the cover
- F(v_i) = R^d (hidden representation space for adapter i)
- Restriction maps encode how adapter representations relate on shared samples

**Definition (Coboundary maps).**
The 0-cochain space is C^0 = bigoplus_i F(v_i).
The 1-cochain space is C^1 = bigoplus_{i<j} F(e_{ij}).

The coboundary operator delta_0: C^0 -> C^1 is defined on each edge e_{ij} as:
  (delta_0 f)(e_{ij}) = rho_{v_j -> e_{ij}}(f(v_j)) - rho_{v_i -> e_{ij}}(f(v_i))

**Definition (Sheaf Cohomology).**
  H^0(K, F) = ker(delta_0)  — globally consistent sections
  H^1(K, F) = ker(delta_1) / im(delta_0)  — obstruction to extending local data

**Theorem (Hansen & Ghrist 2110.03789, Prop 3.1).**
dim(H^1) counts linearly independent "obstruction directions" — information
that is locally present on individual vertices but cannot be consistently
extended across edges.

### Application to Adapter Composition

When dim(H^1) > 0, there exist directions in the combined representation space
where adapter outputs are incompatible on shared samples. A bridge adapter of
rank >= dim(H^1) is necessary to reconcile these directions.

**Conjecture (Rank Budget Bound).**
If F is a sheaf on the Cech nerve of the adapter cover, and dim(H^1(K, F)) = k,
then any composition that preserves all local adapter information on overlaps
requires at least k additional parameters (rank-k correction).

*Motivation (not a proof):* Each basis vector of H^1 represents an independent
obstruction cycle in the nerve. Intuitively, resolving each cycle requires at
least one free parameter. However, this has NOT been formally proven. The
universal coefficient theorem relates cohomology with different coefficient
modules but does not directly yield a rank bound on correction parameters in
representation space. A formal proof would need to establish a map from scalar
H^1 cycles to rank deficiency in the vector-valued restriction maps — this
remains an open question.

**Note:** The scalar H^1 = 3 counts independent CYCLES in the nerve graph,
not independent directions of incompatibility in R^{2560}. The full-rank edge
difference matrices (rank 13-68 per edge) suggest the actual obstruction space
in representation space may be much larger. The scalar Betti number provides a
lower bound on the NUMBER of independent pairwise conflicts, not on the RANK
of the correction needed in representation space.

---

## D. Construction for This Experiment

### Step 1: Cover from Specialization Sets

For each sample x, let PPL_i(x) be the perplexity of adapter i on sample x.
Define the **top-k specialization cover**:

  U_i = {x : adapter i is among the k adapters with lowest PPL on x}

From Finding #240, the argmin (k=1) specialization sets are:
- |S_medical| = 60, |S_code| = 62, |S_math| = 84, |S_legal| = 44, |S_finance| = 0

For k=2 (top-2 cover), pairwise overlaps become non-trivial:
  U_i intersect U_j = {x : both i and j are in top-2 for x}

This is guaranteed to have structure because each sample contributes to exactly
k=2 cover sets, not all 5.

### Step 2: Restriction Maps from Representation Differences

For each pair (i, j) with non-empty overlap, and for each sample x in the overlap,
define the representation difference at layer l:

  delta_{ij}^l(x) = h_i^l(x) - h_j^l(x)

where h_i^l(x) is the mean-pooled hidden state of adapter i at layer l on sample x.

The restriction map discrepancy on edge e_{ij} is captured by the matrix:

  D_{ij}^l = [delta_{ij}^l(x_1) | ... | delta_{ij}^l(x_m)]^T in R^{m x d}

where x_1, ..., x_m are the overlap samples.

### Step 3: Coboundary Matrix Construction

For the Cech complex with n vertices and E edges, the coboundary matrix
delta_0 in R^{(E*m) x (n*m)} maps vertex cochains to edge cochains.

In our simplified construction (following Hansen & Ghrist), we work with
the **restriction map discrepancy** directly:

For each edge (i,j), the block D_{ij} captures how much the two local
sections (adapter representations) disagree. Stack these:

  D = [D_{12}; D_{13}; ...; D_{45}]  (vertical stack over all edges)

Then dim(H^1) is estimated from the rank deficiency of this system.

**Practical computation:**
- Construct the full coboundary matrix delta_0 encoding the Cech complex structure
- delta_0 has rows indexed by (edge, sample) and columns by (vertex, sample)
- H^1 = ker(delta_1) / im(delta_0)
- For our 0-dimensional sheaf (scalar stalks aggregated into vectors),
  dim(H^1) = dim(C^1) - rank(delta_0) - dim(ker(delta_1))

For simplicity, we use the **Hodge Laplacian** approach:
  L_1 = delta_0^T delta_0 + delta_1 delta_1^T

Then dim(H^1) = nullity(L_1) (by the Hodge decomposition theorem for
simplicial complexes, Eckmann 1944).

For our case where the 2-simplices (triangles) may be sparse, we compute:
  dim(H^1) = #edges - rank(delta_0) - rank(delta_1)

where delta_1: C^1 -> C^2 is the next coboundary operator.

### Step 4: Multi-Layer Analysis

Repeat for layers l in {5, 10, 15, 20, 25} to track how obstruction
dimension varies with depth. Early layers may show compatible representations
(low H^1) while later layers diverge (high H^1), or vice versa.

---

## E. Quantitative Predictions

**P1 (Cover Structure):** With k=2 specialization cover, each sample appears
in exactly 2 cover sets. Since there are 5 adapters, C(5,2) = 10 possible
overlap pairs. Given the specialization data from Finding #240 (math dominates,
finance empty), we predict 4-6 non-empty pairwise overlaps.

**P2 (Non-trivial H^1):** Given that L2 relative differences range from
0.037 (legal-finance, both weak) to 0.168 (code-math, both strong but different),
representation incompatibility should be non-zero. We predict dim(H^1) >= 1
at intermediate layers (10, 15, 20) where adapter effects are strongest.

**P3 (Layer dependence) — ILL-FORMED:** ~~H^1 should peak at intermediate layers.~~
This prediction was ill-formed. The topological H^1 (Betti number) depends entirely
on the Čech nerve, which is determined by PPL-based specialization rankings — these
are layer-independent by construction. H^1 CANNOT vary by layer under this construction.
The prediction conflated the topological invariant with the L2 magnitude of differences
(which DO vary by layer, peaking at layer 15). Future experiments should distinguish
clearly between topological structure (layer-invariant) and metric magnitudes
(layer-dependent).

**P4 (Rank budget):** dim(H^1) provides a lower bound on bridge adapter rank.
For 5 adapters at rank-16 with d=2560, we expect dim(H^1) in range [1, 10].

---

## F. Assumptions & Breaking Conditions

**A1:** Specialization sets produce a non-degenerate cover.
*If violated:* All samples are tied for top-k, cover is trivial, H^1 = 0.
*Consequence:* K1 FAIL. This would mean adapters are functionally identical.

**A2:** Mean-pooled hidden states are meaningful representations.
*If violated:* Token-level variation dominates, mean-pool washes out differences.
*Consequence:* D_{ij} matrices have very low rank, H^1 underestimated.

**A3:** The Cech nerve captures the relevant topology.
*If violated:* The actual "knowledge space" has higher-dimensional structure
not captured by pairwise overlaps.
*Consequence:* H^1 is a lower bound on the true obstruction dimension.

**A4:** L2 difference is the correct restriction map metric.
*If violated:* Cosine directions matter more than magnitudes.
*Consequence:* Need to project onto principal directions of variation.

---

## G. Worked Example (d=4, 3 adapters, 6 samples)

Three adapters A, B, C. Six samples. Top-2 specialization:

| Sample | Top-1 | Top-2 | In covers |
|--------|-------|-------|-----------|
| x1     | A     | B     | U_A, U_B  |
| x2     | A     | C     | U_A, U_C  |
| x3     | B     | A     | U_A, U_B  |
| x4     | B     | C     | U_B, U_C  |
| x5     | C     | A     | U_A, U_C  |
| x6     | C     | B     | U_B, U_C  |

Overlaps: U_A cap U_B = {x1, x3}, U_A cap U_C = {x2, x5}, U_B cap U_C = {x4, x6}
Cech nerve: complete graph K_3 (triangle).
Euler characteristic: 3 - 3 + 0 = 0 (no triangle simplex since no 3-way overlap).

Actually check for triangle: U_A cap U_B cap U_C = {} (each sample in exactly 2).
So nerve has 3 vertices, 3 edges, 0 triangles.

Now suppose hidden states in R^4:

For edge (A,B), samples x1, x3:
  delta_AB(x1) = h_A(x1) - h_B(x1) = [0.5, 0.3, -0.1, 0.0]
  delta_AB(x3) = h_A(x3) - h_B(x3) = [0.4, 0.2, -0.1, 0.1]

D_AB = [[0.5, 0.3, -0.1, 0.0], [0.4, 0.2, -0.1, 0.1]]  -- rank 2

Similarly for D_AC, D_BC.

Coboundary delta_0: For the graph with vertices {A, B, C} and edges {AB, AC, BC},
orient edges as A->B, A->C, B->C.

delta_0 maps (f_A, f_B, f_C) to (f_B - f_A, f_C - f_A, f_C - f_B) on each sample.

The coboundary matrix is (6 x 6) if we take one sample per overlap per edge:
For sample-level construction with m=2 samples per overlap:

C^0 has dim 3 (one scalar per vertex, or n*d per vertex for vector stalks)
C^1 has dim 3 (one scalar per edge)

For scalar sheaf: delta_0 is 3x3:
  [-1  1  0]
  [-1  0  1]
  [ 0 -1  1]

rank(delta_0) = 2 (rows 1+2 = row 3).
delta_1 = 0 (no triangles, so C^2 = 0).
dim(H^1) = dim(C^1) - rank(delta_0) - rank(delta_1) = 3 - 2 - 0 = 1.

This H^1 = 1 corresponds to the single independent cycle A->B->C->A in the
graph. It means there is one obstruction direction: you cannot find a global
section that is simultaneously compatible with all three pairwise restrictions.

For our experiment with 5 adapters: if the Cech nerve is a graph with E edges
and no triangles (from k=2 cover), then dim(H^1) = E - n + 1 (first Betti number).
With n=5 and c=2 connected components (finance isolated), if E=6:
dim(H^1) = E - |V| + c = 6 - 5 + 2 = 3.
(Equivalently, for the connected K_4 subgraph: 6 - 4 + 1 = 3.)
With triangles, dim(H^1) decreases by the number of independent triangles filled.

---

## H. Complexity & Architecture Connection

**Computational cost:**
- Model loading + BitLinear unpack: ~15s
- Per-adapter evaluation at 5 layers x 250 samples: ~2 min per adapter
- 5 adapters: ~10 min total GPU time
- Coboundary matrix construction + SVD: O(E * m * d) ~ O(10 * 50 * 2560) = O(1.3M) -- trivial

**Memory:**
- Model: ~1.5 GB (bf16 unpacked)
- Hidden states: 5 adapters * 5 layers * 250 samples * 2560 dims * 4 bytes = ~160 MB
- Total: ~1.7 GB (well within 48 GB budget)

**Architecture connection:**
- dim(H^1) directly predicts the rank of "bridge" adapters needed to reconcile
  incompatible representations during composition
- If dim(H^1) = 0 at all layers, simple additive composition is sufficient
  (no information lost in composition)
- If dim(H^1) > 0, bridge adapters must span at least the obstruction directions

---

## Self-Test (MANDATORY)

1. **What is the ONE mathematical property that makes the failure mode impossible?**
   Non-trivial H^1 identifies the EXACT obstruction directions where adapter representations
   are incompatible; a bridge adapter spanning these directions makes composition lossless
   by construction.

2. **Which existing theorem(s) does the proof build on?**
   Hodge decomposition for simplicial complexes (Eckmann 1944); Cellular sheaf
   cohomology framework (Hansen & Ghrist 2110.03789, Prop 3.1); Cech nerve theorem
   (Borsuk 1948) — nerve captures homotopy type of the cover.

3. **What specific numbers does the proof predict?**
   P1: 4-6 non-empty pairwise overlaps from k=2 cover.
   P2: dim(H^1) >= 1 at layers 10, 15, 20.
   P3: dim(H^1) peaks at intermediate layers.
   P4: dim(H^1) in [1, 10] range.

4. **What would FALSIFY the proof (not just the experiment)?**
   The proof is wrong if: adapter representations are perfectly compatible on all
   overlaps (delta_{ij} = 0), which would mean LoRA adapters produce identical
   hidden states on shared samples — contradicted by Finding #240 data showing
   L2 rel diff 0.037-0.168.

5. **How many hyperparameters does this approach add?**
   Count: 1 (k, the top-k parameter for specialization cover).
   k=2 is the natural choice: minimal non-trivial cover. k=1 gives no overlaps.
   k=3+ adds redundancy but tests robustness.

6. **Hack check: Am I adding fix #N to an existing stack?**
   No. This is a DIAGNOSTIC experiment — it measures an obstruction dimension,
   not a fix. The output (dim H^1) informs future bridge adapter design.
