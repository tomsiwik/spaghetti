# P5.B0: Spectral Surgery on PoLAR Adapters — Mathematical Proof

## Reference: arXiv:2603.03995 (Spectral Surgery: Training-Free LoRA Refinement)

## Theorem 1: SVD Decomposition of PoLAR's Theta Matrix

**Setup.** A PoLAR adapter has the form:

  delta_W = X @ Theta @ Y^T

where X in St(d_in, r) (Stiefel), Y in St(d_out, r) (Stiefel), Theta in R^{r x r}.

Since X and Y are orthonormal, the singular values of delta_W are exactly the
singular values of Theta:

  sv(delta_W) = sv(X @ Theta @ Y^T) = sv(Theta)

**Proof.** Let Theta = U_theta @ diag(sigma) @ V_theta^T be the SVD of Theta. Then:

  delta_W = (X @ U_theta) @ diag(sigma) @ (Y @ V_theta)^T

Since X and U_theta are both orthonormal, X @ U_theta is orthonormal.
Similarly Y @ V_theta is orthonormal. This is a valid SVD of delta_W.
Therefore sv(delta_W) = sigma.  QED.

**Consequence:** Spectral surgery on PoLAR reduces to editing the r x r matrix
Theta. We decompose Theta = U @ diag(sigma) @ V^T and reweight sigma.

## Theorem 2: Gradient-Sensitivity Identifies Useful Spectral Components

**Setup.** Given trained adapter delta_W with SVD components sigma_1 >= ... >= sigma_r,
define the sensitivity of component k as:

  s_k = |d(Loss) / d(sigma_k)|

estimated on a small calibration set D_cal:

  s_k = (1/|D_cal|) sum_{x in D_cal} |d L(x; W_base + delta_W) / d sigma_k|

**Claim.** Components with s_k >> 0 are task-relevant; components with s_k ~ 0
are neutral or detrimental.

**Proof.** By chain rule:

  d L / d sigma_k = Tr((d L / d delta_W)^T @ (d delta_W / d sigma_k))
                  = Tr((d L / d delta_W)^T @ u_k @ v_k^T)
                  = (d L / d delta_W)_{u_k, v_k}

where u_k and v_k are the k-th left and right singular vectors. This is the
projection of the loss gradient onto the rank-1 direction u_k v_k^T. If this
projection is near zero, the k-th component does not affect the loss — it is
noise from the training process.

**For PoLAR:** Since sr(delta_W) = r exactly (joint Stiefel), all r components
have non-degenerate singular values. Standard LoRA with sr~1.8 has most components
near-zero already. PoLAR's full-rank Theta means MORE components to potentially
prune — spectral surgery has MORE room to improve quality.

## Theorem 3: Reweighting Under Magnitude Constraint

**Optimization.** Given sensitivities s_1, ..., s_r and current singular values
sigma_1, ..., sigma_r, find new values sigma'_1, ..., sigma'_r that:

  minimize  sum_k (sigma'_k - sigma_k)^2     (stay close to trained values)
  subject to sum_k (sigma'_k)^2 = sum_k sigma_k^2   (preserve Frobenius norm)
             sigma'_k >= 0  for all k           (non-negative singular values)

**Solution (Lagrangian):**

  sigma'_k = max(0, sigma_k + lambda * s_k)

where lambda is chosen to satisfy the norm constraint. This can be solved
by bisection in O(r log(1/epsilon)) time.

**Intuition:** Components with high sensitivity get boosted, components with
low sensitivity get shrunk. The norm constraint prevents the adapter from
growing/shrinking overall.

## Theorem 4: Spectral Surgery Preserves Grassmannian Orthogonality

**Statement.** For two PoLAR adapters with Grassmannian Y_1, Y_2 (Y_1^T Y_2 = 0),
spectral surgery on Theta_1 and Theta_2 independently preserves:

  <delta_W_1', delta_W_2'>_F = 0

**Proof.** Spectral surgery modifies Theta but NOT X or Y. Therefore:

  <X_1 Theta'_1 Y_1^T, X_2 Theta'_2 Y_2^T>_F
  = Tr(Y_1 Theta'_1^T X_1^T X_2 Theta'_2 Y_2^T)
  = Tr(Theta'_1^T X_1^T X_2 Theta'_2 Y_2^T Y_1)
  = Tr(Theta'_1^T X_1^T X_2 Theta'_2 @ 0)       [Y_2^T Y_1 = 0]
  = 0

Spectral surgery is a per-adapter operation on Theta only. The Grassmannian
orthogonality (which lives in Y) is untouched.  QED.

**Impossibility of failure:** No reweighting of sigma values can violate the
Grassmannian guarantee. The interference is zero regardless of what sigma' values
are chosen. This is a structural invariant of the PoLAR-Grassmannian construction.

## Theorem 5: Flat Spectrum of Retracted PoLAR (Counter-prediction)

**Statement.** After Stiefel retraction, delta_W = A @ B where A^T A = I_r
and B B^T = I_r. All r singular values of delta_W are exactly 1.

**Proof.** The Gram matrix of delta_W is:

  delta_W^T delta_W = (AB)^T (AB) = B^T A^T A B = B^T I_r B = B^T B

The nonzero eigenvalues of B^T B equal the nonzero eigenvalues of B B^T = I_r.
Therefore all r nonzero eigenvalues are 1, and sv(delta_W) = [1, 1, ..., 1].  QED.

**Verification:** Finding #442 measured sr(PoLAR r=6) = 6.0000, confirming
all singular values are equal (stable rank = r iff flat spectrum).

**Critical Gap in Theorem 2:** The claim "PoLAR's full-rank Theta means MORE
components to potentially prune" is incorrect. The code parameterization has
delta_W = A @ B (no separate Theta). After retraction, the effective Theta = I_r.
All components have equal singular values — there is LESS to prune, not more.


## Theorem 6: Ill-Definedness of Surgery on Flat Spectra

**Statement.** When all singular values sigma_k = c for all k, spectral surgery
is not well-defined: the result depends on the arbitrary choice of SVD basis.

**Proof.** For sigma = c * I_r, the SVD delta_W = U diag(sigma) V^T = c * U V^T.
For any R in O(r), the matrices U' = UR and V' = VR give an equally valid SVD:

  U' diag(sigma) V'^T = (UR)(cI)(VR)^T = c U R R^T V^T = c U V^T = delta_W  ✓

The gradient sensitivity s_k = |u_k^T nabla_W L v_k| depends on the basis vectors
u_k, v_k. In the rotated basis, s'_k = |(UR)_k^T nabla_W L (VR)_k| which generally
differs from s_k. Since the SVD basis is arbitrary, the surgery result is arbitrary.

**Prediction:** Sensitivity vectors computed in two different valid SVD bases will
have cosine similarity << 1 (basis-dependent), confirming surgery is ill-defined.


## Theorem 7: Surgery Breaks Stiefel Constraint

**Statement.** Any non-trivial spectral surgery (sigma' != sigma) on a retracted
PoLAR adapter produces delta_W' that cannot be factored as A' @ B' with both
A', B' on their respective Stiefel manifolds.

**Proof.** If delta_W' = U diag(sigma') V^T with sigma' != [1,...,1], then
sv(delta_W') = sigma' which is non-uniform. But for any A' in St(d_in, r) and
B' with B'B'^T = I_r:

  sv(A' B') = [1, ..., 1]  (by Theorem 5)

Therefore delta_W' != A' B' for any Stiefel A', B'.  QED.

**Consequence:** Spectral surgery breaks the Grassmannian orthogonality guarantee
(Theorem 4 is vacuous because the surgically modified adapter is no longer PoLAR).
Even if surgery improved single-adapter quality, it would destroy composition.


## Kill Criteria Derivation (Updated with Theorems 5-7)

| K | Criterion | Theorem | Prediction |
|---|-----------|---------|-----------|
| K1270 | GSM8K improvement >= 2pp | Thm 5+6: flat spectrum + basis non-uniqueness | **FAIL: ~0pp** (no well-defined improvement direction) |
| K1271 | PPL preserved within 2pp | Thm 6: surgery is noise on flat spectrum | **PASS: ~0pp** (basis-dependent reweighting is weak noise) |
| K1272 | Surgery < 60s | Thm 1: r x r SVD + finite differences | **PASS: ~10s** |

**Control prediction (LoRA):** Standard LoRA has non-flat spectrum (sr~1.8, Finding #442).
Surgery should show measurable (possibly positive) effect, confirming the surgery
procedure works — the issue is specific to PoLAR's flat spectrum, not the implementation.

## Impossibility Structure

Three independent reasons spectral surgery cannot work on retracted PoLAR:
1. **Flat spectrum** (Thm 5): Nothing to reweight — all components already equal
2. **Basis non-uniqueness** (Thm 6): Surgery result depends on arbitrary SVD basis
3. **Stiefel violation** (Thm 7): Any non-trivial surgery breaks composition guarantee
