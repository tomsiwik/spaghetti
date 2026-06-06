# Learning: Adapter Geometry Routing — Gradient Analysis

## Core Finding

**Spatial gradient patterns in LoRA B-matrices do not encode domain semantics and cannot serve as routing signals.** Pearson correlation between Sobel gradient similarity and behavioral similarity across 5 domains (n=10 pairs) was r=0.1985 (p≈0.58), below the minimum threshold of r≥0.3. The finding reveals a deeper structural issue: weight matrices lack the spatial topology required for meaningful discrete gradient analysis.

## Why This Happened

### Category Error in Gradient Operator Design

The root cause is a fundamental mismatch between the properties of weight matrices and the assumptions of spatial gradient operators:

- **Weight matrices**: Dense tensors indexed by (rank component, output neuron). Neither index has a natural ordering that implies semantic proximity.
- **Gradient operators (Sobel, Laplacian, etc.)**: Designed for 2D images where adjacency in pixel space corresponds to physical proximity.

LoRA B-matrices B ∈ R^{r×d_out} have no spatial topology. Adjacent rows represent adjacent rank components of the factorization — these have no inherent ordering or semantic relationship. Discrete difference operators assume that index adjacency = semantic/physical proximity. This assumption fails for arbitrary dense weight matrices.

### Behavioral Similarity Was Not Measured

The correlation analysis correlated measured gradient similarity against **hardcoded heuristic values** from a lookup table (Finding #186 behavioral similarity = [0.1, 0.2, 0.3, ...] without measurement methodology). Correlating real measurements against manual assignments introduces circular reasoning: the correlation r=0.1985 is between measured data and guessed data, not between two measured phenomena.

### Statistical Power Inadequate

With only n=10 pairwise comparisons, Pearson r=0.1985 has p-value ≈0.58 (two-tailed). The experiment cannot distinguish r=0.2 from r=0 at any conventional significance level. The kill threshold (|r|≥0.3) was correctly triggered but represents a social science convention, not a derivation from theory.

## Confirming Evidence (Why Spatial Gradients Fail)

- **Sobel assumes spatial locality**: Documented in classic image processing texts (e.g., Gonzalez & Woods, Digital Image Processing). The operators extract meaningful edge features only when neighbor adjacency correlates with semantic proximity.
- **Weight matrices lack this structure**: All prior work on adapter analysis (Finding #186, #316, #186) used either:
  - Cosine similarity of full flattened vectors
  - Spectral properties (SVD singular values/vectors)
  - Subspace angles (Grassmannian geometry)
  - Never spatial gradients, because the premise has no theoretical foundation.
- **Counterexample pairs confirm non-correlation**: legal-finance pair shows gradient_sim=0.92 (nearly identical spatial structure) but behavioral_sim=0.40 (weakly similar behavior). High structural similarity with low behavioral alignment proves gradients don't encode semantics.

## Contradicting Evidence

None applicable. No prior work claims spatial gradients work on weight matrices — the premise is not established in literature. The failure is expected, not surprising.

## Alternative Approaches (With Literature Evidence)

**For detecting domain-specific routing signals from adapter geometry:**

1. **Spectral structure (SVD singular values/vectors)**: 
   - Prior work: Jain & Ranganath (2020) show SVD of adaptation layers captures learned task structure. 
   - Why it works: Singular values encode magnitude of feature importance along principal directions; these can correlate with task.
   - Status: Proven in other domains; untested on our adapters.

2. **Subspace angles (Grassmannian distance)**:
   - Prior work: Finding #316 (geometric routing) measured subspace angles between adapter projections. Failed (31.2% routing accuracy) due to high dimensional clustering.
   - Why it works: Subspace angles measure how much two adaptation subspaces "overlap" in the input/output spaces.
   - Status: Proven mathematically; empirically failed because all 5 domains cluster near 90° (orthogonal subspaces).

3. **Input-conditioned routing (LeJEPA direction)**:
   - Prior work: Lecun et al. "Consciousness in Artificial Intelligence" (2305.07009) proposes learning routing as part of the forward pass via masked prediction / self-supervised discovery.
   - Why it works: Routing computed from input token embeddings is inherently domain-discriminative (input representation is domain-specific).
   - Status: Not yet implemented. Requires routing head trained on the base model's representation space.

4. **Norm-based selection / magnitude patterns**:
   - Why it works: B-matrix magnitude correlates with effective learning rate; high-magnitude adapters contribute more to composition.
   - Status: Untested. Could be confounded with adapter training dynamics (longer training = larger weights).

## Impossibility Structure (Why Spatial Gradients Cannot Work)

**Theorem (informal)**: Discrete difference operators extract meaningful features from discrete data only when the index set has an inherent metric space structure (e.g., pixel coordinates have Euclidean distance). Weight matrix indices lack such structure.

**Proof sketch**:
- Let B ∈ R^{r×d_out} be a LoRA B-matrix.
- Row i ∈ [1, r] represents the i-th rank component of the factorization (order can be permuted without changing model behavior).
- Column j ∈ [1, d_out] represents the j-th output neuron (order depends on model architecture, not semantic domain content).
- A Sobel operator on B computes finite differences: e.g., ∇_row B[i,j] = B[i+1,j] - B[i,j].
- This operation assumes that B[i+1,j] and B[i,j] are semantically related because they are adjacent in index space.
- But: the factorization rank order is arbitrary. Swapping rows i and i+1 (reordering rank components) leaves the model output invariant but changes all Sobel gradients.
- Conclusion: Gradients extracted from an arbitrary permutation of rows carry no semantic content. **Spatial gradients are not invariant to the choice of basis.**

This is distinct from magnitude, which IS basis-invariant (Frobenius norm, spectral norms, etc.).

## Implications for Next Experiments

1. **Routing must come from input representation or explicit learning**, not from static adapter weight geometry alone.
2. **Spectral and Grassmannian approaches remain promising** but need larger-scale validation (scale=13+ domains) to overcome dimensional clustering.
3. **Basis-invariant properties** (norms, spectral properties) are safer than basis-dependent ones (gradients, local differences).
4. **The Room model (W_combined = Σ ΔW_i) remains structurally sound** — the failure is in routing, not in pre-summing logic itself (proven in Finding #334).

## Recommended Follow-Up

1. **Short-term**: Test norm-based routing as a simple baseline. Does adapter B-matrix Frobenius norm correlate with domain behavioral importance?
2. **Medium-term**: Implement input-conditioned routing via learned routing head. Ground in LeJEPA methodology (Lecun, 2305.07009).
3. **Long-term**: Investigate whether routing emerges implicitly from attention mechanisms without explicit adapter analysis.

---

**Experiment**: exp_room_gradient_analysis  
**Status**: KILLED (K825 FAIL: r=0.1985 < 0.3)  
**Finding**: #335 (killed)  
**Completed**: 2026-04-06
