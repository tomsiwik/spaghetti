# MATH.md: Pathway Graph Construction on BitNet-2B

## Type: Guided Exploration (Measurement)

**Proven framework:** Persistent homology on co-activation graphs (1812.09764, 2506.01042).
**Claim to verify:** BitNet-2B-4T has non-trivial topological structure in its
co-activation pathways, and this structure is NOT simply the top-k singular vectors.

---

## A. Failure Mode Identification

**Potential failure:** If the co-activation topology of BitNet-2B is trivial (all
pathways are redundant), then the entire pathway preservation framework is
unnecessary. There would be nothing to preserve.

**Alternative failure:** If high-persistence features ARE simply the top-k singular
vectors, then SVD-based methods already capture everything important. Persistent
homology adds no value over standard linear algebra.

---

## B. The Right Question

Not: "Can we compute persistent homology on neural activations?" (known to work)

**Right question:** "Does BitNet-2B-4T have topological structure in its co-activation
graph that is (a) non-trivial (many persistent features) and (b) not reducible to
spectral rank (not just top-k singular vectors)?"

---

## C. Prior Mathematical Foundations

**Neural Persistence (1812.09764):** Defines weight-space filtrations on neural
networks and computes 0-dimensional persistent homology. Shows that persistence
diagrams capture meaningful structural properties of trained networks.

**Neural Topology Probing (2506.01042):** Builds graph representations of LLM
neuron connectivity. Topology-based probes outperform activation-based probes by
130%. Identifies "hub neurons" that serve as structural bridges.

**Persistent homology guarantee (Stability Theorem):** For two filtered simplicial
complexes K and L, the bottleneck distance between their persistence diagrams
satisfies: d_B(Dgm(K), Dgm(L)) <= d_I(K, L), where d_I is the interleaving
distance. This means small perturbations to the filtration produce small changes
in the persistence diagram.

---

## D. Method and Predictions

**Method:**
1. Sample 10K inputs: 2K from each domain (medical, code, math, legal, finance)
2. For a target FFN layer, record activation vectors h(x) in R^d
3. Compute SVD of activation matrix H = [h(x1), ..., h(xn)]^T to get singular
   directions V = [v1, ..., vd]
4. Build co-activation graph: vertices = top-k singular directions (k=100),
   edge weight w(vi, vj) = fraction of inputs where both |vi^T h(x)| > epsilon
   AND |vj^T h(x)| > epsilon
5. Compute 0-dimensional persistent homology via sublevel filtration
6. Plot persistence diagram, compute statistics

**Quantitative predictions:**

| Prediction | Source | Threshold |
|------------|--------|-----------|
| P1: >= 10 features with persistence > 0.1 | 1812.09764 finds non-trivial topology in trained nets | K623 |
| P2: Rank correlation(persistence rank, SV rank) < 0.5 | 2506.01042 shows topology != spectral rank | K624 |
| P3: Power-law distribution of persistence values | Standard in natural graphs | Diagnostic |
| P4: Cross-domain inputs create longest-persisting bridges | Pathway preservation theory | Diagnostic |

---

## E. Assumptions & Breaking Conditions

1. **Activation threshold epsilon matters.** Too high: sparse graph, trivial topology.
   Too low: dense graph, all features short-lived. We use epsilon = 0.5 * max|vi^T h(x)|
   (50% of max activation per direction) to create a sufficiently sparse graph for
   meaningful PH. A random baseline control is required to distinguish real structure
   from sparsification artifacts.

2. **Top-100 singular directions are sufficient.** If important pathways live in
   directions 101+, we miss them. Mitigation: check energy captured by top-100.

3. **Single layer is representative.** Different layers may have different topology.
   This experiment tests one layer first; extend to all layers if successful.

---

## Self-Test (MANDATORY)

1. What is the ONE mathematical property that makes the failure mode impossible?
   **Honest answer: there is none.** The stability theorem guarantees PH is a
   stable detector, but does NOT guarantee that non-trivial topology must exist
   in BitNet-2B. This is a measurement experiment testing whether meaningful
   structure exists. A random baseline control is needed to distinguish real
   topology from sparsification artifacts.

2. Which existing theorem(s) does the proof build on?
   Stability theorem for persistence diagrams (Cohen-Steiner et al., 2007).
   Neural persistence framework (1812.09764).

3. What specific numbers does the proof predict?
   P1: >= 10 features with persistence > 0.1. P2: rank correlation < 0.5.

4. What would FALSIFY the proof?
   If the co-activation graph has trivial topology (< 10 persistent features)
   or if high persistence = high singular value (correlation > 0.5), the pathway
   preservation framework adds no value.

5. How many hyperparameters does this approach add?
   2: activation threshold epsilon, number of singular directions k. Both have
   principled defaults.

6. Hack check: Am I adding fix #N?
   No. This is the first measurement experiment in the pathway preservation track.
