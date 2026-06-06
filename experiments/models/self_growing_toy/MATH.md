# Self-Growing Model: Mathematical Framework

**Experiment type:** Frontier extension
**Proven result being extended:** ReLoRA (Lialin et al., 2307.05695) proves periodic
LoRA merging is equivalent to gradient accumulation. We extend this to sequential
domain promotion on a random-init base.

## A. Failure Mode Identification

**The disease:** Sequential adapter promotion could fail in three ways:
1. **Catastrophic forgetting:** Promoting domain B's delta overwrites domain A's knowledge
2. **Subspace saturation:** Later promotions land in already-occupied subspaces, producing
   interference instead of new knowledge
3. **Training degradation:** The promoted base becomes harder to train adapters on
   (loss landscape roughens with each promotion)

These are symptoms of a single underlying problem: **uncontrolled perturbation accumulation.**
After K promotions, the base weight is W_K = W_0 + sum_{i=1}^K delta_i. If the deltas
are not magnitude-controlled, the accumulated perturbation can overwhelm the original
weight structure, even if each individual delta is small.

## B. The Right Question

Not: "How do we prevent forgetting during promotion?"
But: "Under what conditions does W_K = W_0 + sum delta_i preserve training dynamics
for future adapters while retaining all previously promoted knowledge?"

The answer comes from ReLoRA (Lialin et al., 2307.05695, Theorem 1 informal):

**ReLoRA equivalence:** Periodically merging LoRA updates into the base and restarting
LoRA training is mathematically equivalent to full-rank gradient accumulation, because:
- LoRA captures a rank-r projection of the gradient at each step
- Merging W <- W + scale * B^T A^T adds that projection to the base
- The next LoRA adapter operates on the updated base, capturing the next rank-r projection
- Over K merge cycles, the accumulated update spans up to rank K*r

## C. Prior Mathematical Foundations

**Theorem (ReLoRA, Lialin et al. 2023):** Let W_0 be initial weights. Training with
periodic LoRA merging over K cycles produces W_K = W_0 + sum_{k=1}^K B_k^T A_k^T.
This is equivalent to full-rank training where at each cycle, the gradient is projected
to rank r. The accumulated weight matrix W_K has effective rank up to K*r.

**Davis-Kahan sin-theta theorem (Finding #326):** For a perturbation delta to weight
matrix W, the subspace rotation is bounded by sin(theta) <= ||delta||_op / gap, where
gap is the spectral gap of W. Smaller ||delta||_F means less subspace rotation, and
thus less forgetting of existing knowledge.

**Key insight from Finding #329:** SVD solidification destroys Grassmannian structure
under COMPOSITION (multi-adapter averaging). But promotion is different: we add ONE
delta to the base at a time. There is no multi-adapter averaging. The delta is
directly absorbed. The Grassmannian structure of future adapters is preserved because
each new adapter gets fresh Grassmannian A-matrices.

## D. Proof of Growth Guarantee

**Claim 1 (Monotonic improvement).** If domain i's adapter achieves loss L_i < L_base
on domain i's data, then promotion (W <- W + delta_i) reduces the base's loss on
domain i, provided ||delta_i|| is controlled to avoid catastrophic interference with
other domains.

*Argument.* The adapter was trained to minimize loss on domain i starting from base W.
The optimal delta minimizes L(W + delta; D_i). After promotion, W' = W + delta_i.
By construction, L(W'; D_i) <= L(W; D_i) (the adapter improved domain i loss).

For other domains j != i, the perturbation delta_i may increase L(W'; D_j). By
Davis-Kahan, this increase is bounded by O(||delta_i||_op / gap_j). At rank r=4 with
controlled scale, the perturbation magnitude is small relative to the base weight
norms.

The net effect across all K promotions: domain k's loss is directly reduced by its
own promotion and may be slightly increased by subsequent promotions. The magnitude
control via SVD rank truncation (or scale reduction) limits cross-domain interference.

**Claim 2 (Later adapters benefit from richer base).** After K promotions, the base
W_K = W_0 + sum delta_i has more structured features than W_0 (random init). Training
a new adapter on W_K should converge faster because:
- The base already captures useful representations
- The adapter only needs to learn domain-specific refinements, not general features
- This is the standard transfer learning argument (Yosinski et al. 2014)

**Claim 3 (Growth vs joint training).** The grown base W_K and a jointly-trained model
W_joint both see the same total data across K domains. The jointly-trained model
optimizes all domains simultaneously (full gradient), while the grown model uses
sequential rank-r projections. By ReLoRA's convergence result, the gap narrows as
(a) rank r increases, (b) number of training steps per adapter increases, and
(c) number of promotion cycles increases. The gap should be bounded but nonzero
due to the sequential nature of promotion.

## E. Quantitative Predictions

| Prediction | Source | Expected Value |
|-----------|--------|----------------|
| P1: Each promotion reduces loss on its domain | Claim 1 | domain_improvement > 0 for all 5 |
| P2: Mean loss improves monotonically | Claim 1 + Davis-Kahan | mean_improvement >= 0 for all 5 |
| P3: 5th adapter trains no slower than 1st | Claim 2 | time_5 / time_1 <= 1.5 |
| P4: Grown/baseline ratio < 3.0 | Claim 3 + ReLoRA | ratio < 3.0 (kill), target < 2.0 |
| P5: Final grown loss << random init loss | Direct | improvement > 50% vs random init |

## F. Assumptions & Breaking Conditions

1. **Scale control:** Each delta_i must have controlled magnitude. If adapters are
   over-scaled (as seen with scale=20 in Finding #326), interference dominates.
   Breaking: if any ||delta_i|| > ||W||, Claim 1 breaks.

2. **Domain separability:** The 5 toy domains must be learnable by a d=64 model.
   Breaking: if d=64 is too small, adapters won't learn anything.

3. **Rank capacity:** K=5 promotions at rank=4 produce an accumulated perturbation
   of effective rank <= 20. At d=64, this uses 20/64 = 31% of the weight space capacity.
   Breaking: if K*r approaches d, saturation occurs.

4. **Sequential vs simultaneous:** The grown base sees domains sequentially. If
   domain order matters significantly, results may not generalize.
   We can check: does the last-promoted domain always have lowest loss?

## G. Worked Example (d=4, r=2, K=2)

Base: W_0 = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] (identity)

Domain A adapter: A_a = [[1,0],[0,1],[0,0],[0,0]], B_a = [[0.1,0],[0,0.2]]
Delta_A = scale * B_a^T @ A_a^T = 2 * [[0.1,0,0,0],[0,0.2,0,0]] (rank 2)
W_1 = W_0 + Delta_A = [[1.2,0,0,0],[0,1.4,0,0],[0,0,1,0],[0,0,0,1]]

Domain B adapter: A_b = [[0,0],[0,0],[1,0],[0,1]], B_b = [[0.15,0],[0,0.1]]
Delta_B = 2 * [[0,0,0.15,0],[0,0,0,0.1]] (rank 2, orthogonal subspace)
W_2 = W_1 + Delta_B = [[1.2,0,0.3,0],[0,1.4,0,0.2],[0,0,1,0],[0,0,0,1]]

Observations:
- W_2 has effective rank 4 (full), accumulated from two rank-2 updates
- Domain A's contribution (first two dims enhanced) is preserved after domain B
- No interference because A_a and A_b span orthogonal subspaces
- This is the ideal case; real adapters will have some overlap

## H. Complexity

- Per promotion: train LoRA (O(steps * d * r)), extract delta (O(d^2 * r)), SVD solidify
  (O(d^2 * min(d,d))), promote (O(d^2))
- Total: O(K * steps * d * r) for training + O(K * d^2 * d) for SVD
- At d=64, r=4, K=5: training dominates, SVD is negligible
- Memory: one model + one adapter at a time = O(d^2) constant

## Self-Test

1. **One property:** Sequential promotion with magnitude-controlled deltas is equivalent
   to gradient accumulation (ReLoRA), so the base improves monotonically.

2. **Existing theorems:** ReLoRA equivalence (Lialin et al. 2023), Davis-Kahan sin-theta
   bound for cross-domain interference control.

3. **Specific numbers:** P1-P5 above. Each promotion improves its domain. Grown base
   within 3x of jointly-trained. No training slowdown.

4. **Falsification:** If mean loss degrades after ANY promotion (K841), the
   gradient-accumulation equivalence breaks at this scale.

5. **Hyperparameters:** scale=2.0 (from LoRA convention), SVD rank=4 (matches LoRA rank).
   Scale could be derived from Davis-Kahan but at toy scale, 2.0 is conservative.

6. **Hack check:** No. Single mechanism: promote delta into base. No stacking of fixes.
