# Knowledge Region Overlap Mapping: Sheaf Cover Construction

## Type: Guided Exploration

**Proven framework:** Sheaf-theoretic composition (2110.03789, 2502.15476) requires
a well-defined open cover of the input space with non-trivial overlaps.

**Unknown being discovered:** Do domain adapters induce a cover with structured
overlaps? What is the compatibility structure on those overlaps?

---

## A. Failure Mode Identification

**The disease:** When N domain adapters are composed (e.g., by averaging), inputs
that lie in the "overlap region" between domains receive a blended response that
may be worse than either individual adapter. This is the composition interference
problem.

**Why it matters:** Finding #68 established that weight-space orthogonality does
NOT imply data-space orthogonality. Two adapters with orthogonal LoRA matrices
can still compete on the same inputs. The question is: do such data-space overlaps
actually exist, and are they structured enough to warrant correction terms?

**This is not a symptom-level problem.** If overlaps don't exist (all adapters
improve only on their own domain's data), composition is trivially safe and no
bridges are needed. If overlaps exist but are uniformly compatible (both adapters
produce similar hidden states), bridges are also unnecessary. Only if overlaps
exist AND have variable compatibility does the sheaf-theoretic framework apply.

---

## B. The Right Question (Reframe)

**Wrong question:** "How do we prevent interference between adapters?"

**Right question:** "What is the topological structure of the input-space cover
induced by domain adapters, and does it satisfy the prerequisites for sheaf
cohomology to be informative?"

The answer determines whether sheaf theory is the right mathematical framework
for composition, or whether simpler methods (routing, scaling) suffice.

---

## C. Prior Mathematical Foundations

### Sheaf-theoretic setup (2110.03789, Hansen & Ghrist 2019)

A **cellular sheaf** F on a topological space X assigns:
- To each open set U_i, a vector space F(U_i) (the "stalk")
- To each inclusion U_i ∩ U_j -> U_i, a restriction map r_{ij}: F(U_i) -> F(U_i ∩ U_j)

**Global sections** are consistent assignments: elements s_i in F(U_i) such that
r_{ij}(s_i) = r_{ji}(s_j) on every overlap.

**First cohomology H^1(X, F)** measures the obstruction to gluing local sections
into global sections. dim(H^1) = 0 means local data always glues consistently.
dim(H^1) > 0 means there are incompatible local assignments that cannot be
reconciled — these are exactly the "bridge gaps" requiring correction terms.

### Cech nerve (Borsuk 1948)

Given an open cover U = {U_1, ..., U_N}, the **Cech nerve** N(U) is the
simplicial complex where:
- Vertices = {U_i}
- Edge (U_i, U_j) exists iff U_i ∩ U_j is non-empty
- k-simplex (U_{i0}, ..., U_{ik}) exists iff the (k+1)-fold intersection is non-empty

By the **Nerve theorem** (Leray 1945, Borsuk 1948): if all intersections are
contractible, then N(U) is homotopy equivalent to the union. The Betti numbers
of N(U) then give the topology of the covered space.

### Application to adapter composition (FRAMEWORK FOR FUTURE WORK)

**Note:** The sheaf-theoretic application below was not successfully tested in
this experiment because the improvement-based cover was degenerate. This section
describes the intended framework; a future experiment with corrected cover
definitions (specialization sets instead of improvement sets) is needed.

We define:
- X = input space (all evaluation samples)
- U_i = {x in X : PPL_i(x) < PPL_base(x)} = "improvement set" of adapter i
- F(U_i) = hidden state representation h_i(x) at a middle layer

The restriction map r_{ij} on overlap U_i ∩ U_j is:
  r_{ij}(s_i)(x) = h_i(x) for x in U_i ∩ U_j

Compatibility is measured by cosine similarity:
  c_{ij}(x) = cos(h_i(x), h_j(x)) for x in U_i ∩ U_j

If c_{ij}(x) is uniformly high, the restriction maps are approximately equal
and H^1 = 0 (no bridge needed for this pair). If c_{ij}(x) varies significantly,
the representations disagree on how to handle overlap inputs, and H^1 > 0
(bridge correction needed).

---

## D. Predictions (Behavioral AND Quantitative)

This is guided exploration, so predictions are informed hypotheses, not proof
consequences.

### Structural predictions:
1. **Non-trivial cover:** At least 3 of the C(5,2) = 10 pairwise overlaps
   should be non-empty, because many inputs have cross-domain relevance
   (e.g., medical billing uses legal language, algorithmic problems use math notation)

2. **Semantic proximity predicts overlap size:** Medical-legal and code-math
   overlaps should be among the largest (shared vocabulary, shared reasoning patterns).
   Finance-code overlap should be smallest (minimal semantic connection).

3. **Compatibility is variable, not uniform:** Within each overlap, some inputs
   should be well-served by either adapter (high cosine, compatible), while others
   should be "contested" (low cosine, incompatible). This variability is measured
   by std(c_{ij}) > 0.1 within each overlap.

### Quantitative predictions (order-of-magnitude):
- Improvement sets: |U_i| should be ~60-80% of domain i's own samples and
  ~10-30% of other domains' samples (adapters generalize somewhat beyond their
  training domain)
- Overlap sizes: |U_i ∩ U_j| ~ 20-80 samples (out of 250 total validation samples)
- Mean cosine similarity on overlaps: 0.5-0.9 (representations similar but not identical)
- Std of cosine: 0.1-0.3 (meaningful variability)

### Cech nerve prediction:
- The nerve should be a connected simplicial complex (most adapters share some
  overlap) but NOT the full 4-simplex (not all 5 overlap simultaneously)
- Expected: 6-8 edges (non-empty pairwise overlaps), 1-3 triangles (triple overlaps)

---

## E. Assumptions and Breaking Conditions

1. **PPL is a meaningful proxy for "adapter helps."** If PPL_adapter < PPL_base
   does not correlate with actual quality improvement, the improvement sets U_i
   are meaningless. Finding #236 showed PPL doesn't predict MMLU accuracy
   (r=0.08), but Finding #238 showed behavioral quality does correlate with
   adapter application. We use PPL as a necessary proxy here (extracting
   behavioral quality for 250 samples is prohibitive) but acknowledge this caveat.

2. **Hidden states at a single layer capture representation structure.** We
   extract from layer 15/30 (middle). If the relevant representation differences
   are at earlier or later layers, our compatibility measurements may miss them.

3. **250 validation samples are sufficient.** The kill criterion requires
   |U_i ∩ U_j| > 50 for at least 3 pairs. With 250 total samples and 5 domains
   of 50 each, this requires substantial cross-domain improvement. If adapters
   are too domain-specific (only improve their own domain), overlaps will be
   small. We additionally use 400 train samples per domain (2000 total) evaluated
   as held-out-from-perspective-of-other-adapters to increase sample size.

4. **Cosine similarity of hidden states is a meaningful compatibility measure.**
   If two adapters produce hidden states in orthogonal subspaces but both lead
   to correct outputs, cosine could be low despite functional compatibility.
   We acknowledge this limitation.

---

## F. Worked Example (d=4, N=3 adapters, 6 samples)

Consider 3 adapters on 6 samples with PPL values:

| Sample | Base | A1    | A2    | A3    |
|--------|------|-------|-------|-------|
| x1     | 10.0 | 8.0   | 9.0   | 11.0  |
| x2     | 10.0 | 11.0  | 7.0   | 9.0   |
| x3     | 10.0 | 8.5   | 8.0   | 12.0  |
| x4     | 10.0 | 12.0  | 11.0  | 7.0   |
| x5     | 10.0 | 9.0   | 9.5   | 8.0   |
| x6     | 10.0 | 7.0   | 6.0   | 9.5   |

Improvement sets:
- U1 = {x1, x3, x5, x6} (PPL < 10)
- U2 = {x2, x3, x6} (PPL < 10)
- U3 = {x2, x4, x5, x6} (PPL < 10)

Pairwise overlaps:
- U1 ∩ U2 = {x3, x6} — size 2
- U1 ∩ U3 = {x5, x6} — size 2
- U2 ∩ U3 = {x2, x6} — size 2

Triple overlap:
- U1 ∩ U2 ∩ U3 = {x6} — size 1

Cech nerve: Complete graph K3 with one 2-simplex (triangle).
This is contractible, so H^1 = 0 in this tiny example.

Now suppose hidden states at layer 2 (d=4):
- h1(x3) = [1, 0, 0, 0], h2(x3) = [0.9, 0.3, 0, 0] → cos = 0.95 (compatible)
- h1(x6) = [0, 1, 0, 0], h2(x6) = [0, 0, 1, 0] → cos = 0.0 (incompatible)

std(cos) on U1 ∩ U2 = std([0.95, 0.0]) = 0.475 >> 0.1 → K2 would PASS.

The pair (A1, A2) agrees on how to handle x3 but disagrees on x6. A bridge
adapter for this overlap would need to reconcile these two views of x6.

---

## G. Complexity and Architecture Connection

**Computational cost:**
- PPL computation: O(N * |X| * L * d^2) where N=5 adapters, |X|~250 samples,
  L=30 layers, d=2560. Each forward pass ~15ms on M5 Pro. Total: 5*250*0.015 = ~19s
  per configuration, plus base = 6 configurations = ~2 min.
- Hidden state extraction: same forward passes, cache layer 15 output.
  Storage: 250 * 2560 * 4 bytes * 5 adapters = ~12.8 MB (negligible).
- Cosine computation: O(|overlap| * d) per pair, negligible.
- Cech nerve: O(2^N) simplices to check, N=5 so 32 subsets — negligible.

**Total estimated runtime:** ~30-45 min (dominated by model loading/unloading
between adapter configurations).

**Connection to architecture:** If overlaps are non-trivial (K1 PASS) and
compatibility is variable (K2 PASS), the next step is Exp 4 (sheaf cohomology
dimension estimation) which determines the rank budget for bridge adapters.
Bridge adapters would be lightweight LoRA modules trained on overlap regions
to reconcile incompatible representations.

---

## Self-Test (MANDATORY)

1. What is the ONE mathematical property that makes the failure mode impossible?
   This experiment does not propose a fix — it MEASURES whether the precondition
   for sheaf-theoretic composition (non-trivial structured overlaps) exists.

2. Which existing theorem(s) does the proof build on?
   Nerve theorem (Leray 1945, Borsuk 1948); sheaf cohomology framework
   (2110.03789, Hansen & Ghrist 2019).

3. What specific numbers does the proof predict?
   |U_i| ~ 60-80% own domain, ~10-30% cross-domain; |U_i ∩ U_j| ~ 20-80;
   mean cosine 0.5-0.9; std cosine > 0.1.

4. What would FALSIFY the proof (not just the experiment)?
   If the nerve theorem's contractibility condition is violated (intersections
   are not contractible in input space), the simplicial approximation is wrong.
   In practice, this means measuring discrete sets rather than continuous regions.

5. How many hyperparameters does this approach add?
   Count: 1 — the layer from which to extract hidden states (layer 15/30).
   Why can't it be derived? The "optimal" layer depends on where domain-specific
   information is encoded, which is an empirical question. Layer 15 is a
   reasonable default (middle of the network).

6. Hack check: Am I adding fix #N to an existing stack?
   No. This is a measurement experiment, not a fix. It determines whether the
   sheaf framework is applicable.
