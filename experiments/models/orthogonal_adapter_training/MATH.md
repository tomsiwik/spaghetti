# Orthogonal Adapter Training: Mathematical Foundation

## Type: Guided Exploration (Type 2)

Proven framework: OPLoRA (arXiv:2510.13003) — orthogonal projection preserves
top-k singular triples. Unknown: optimal k for ternary adapters on BitNet-2B.

---

## A. Failure Mode: Direction Interference in Knowledge Subspace

**The disease:** When a LoRA adapter delta Delta_W = s * B^T A^T is added to base
weight W, the composed weight W' = W + Delta_W has perturbed singular structure.
If Delta_W has nonzero projection onto the top singular directions of W, it
directly corrupts the knowledge-retrieval pathways that those directions encode.

**Why DARE fails:** DARE applies a random Bernoulli mask to Delta_W, reducing
density but preserving direction (in expectation). The expected DARE delta is
E[Delta_W_DARE] = Delta_W. So the direction interference is unchanged in expectation.
Finding #268 confirmed this: MMLU math degrades -20pp to -35pp across ALL drop rates.

**Formal statement of failure:** Let W = U Sigma V^T be the SVD of the base weight.
Let U_k, V_k be the top-k left/right singular vectors. The direction interference
metric is:

  rho_k(Delta_W) = ||U_k^T Delta_W V_k||_F / ||Delta_W||_F

When rho_k > 0, the adapter delta has nonzero projection onto the knowledge subspace,
directly perturbing the top-k singular triples that encode factual recall.

**This is a stable failure mode:** Standard LoRA training with random or Grassmannian A
matrices has no constraint preventing gradients from aligning with the top singular
directions of W. Since these directions have the largest singular values, they
dominate the gradient landscape and attract learning — making interference worse,
not better, as training progresses.

---

## B. The Right Question

NOT: "How do we reduce interference after training?" (DARE, post-hoc)
RIGHT: "What constraint during training makes direction interference ZERO by construction?"

**Answer:** Force all adapter updates into the orthogonal complement of the base
model's principal subspace. This is a training-time constraint, not a post-hoc fix.

---

## C. Prior Mathematical Foundations

**Theorem (OPLoRA, arXiv:2510.13003, Theorem 1).**
Let W = U Sigma V^T with top-k singular triples (u_i, sigma_i, v_i) for i=1..k.
Define projection matrices:
  P_L = I - U_k U_k^T   (projects away from top-k left singular space)
  P_R = I - V_k V_k^T   (projects away from top-k right singular space)

If the LoRA update satisfies Delta_W = P_L * Delta_W * P_R, then:
  W' = W + Delta_W preserves the top-k singular triples exactly.

*Proof sketch:* W' u_i = (W + Delta_W) u_i = sigma_i v_i + Delta_W u_i.
Since u_i in span(U_k), we have P_L u_i = 0, so
  Delta_W u_i = P_L (Delta_W) P_R u_i   [but P_L projects out u_i]
Wait — more carefully: Delta_W = P_L * M * P_R for some M. Then:
  Delta_W u_i = P_L M P_R u_i.
Since u_i in col(U_k), P_L u_i = 0, but the projection acts on Delta_W's rows, not on u_i directly.

**Correct proof:** The double-sided projection ensures:
  U_k^T Delta_W = U_k^T P_L M P_R = (P_L^T U_k)^T M P_R = 0^T M P_R = 0
  Delta_W V_k = P_L M P_R V_k = P_L M (P_R V_k) = P_L M 0 = 0

Therefore U_k^T (W + Delta_W) = U_k^T W and (W + Delta_W) V_k = W V_k.
The top-k singular triples are exactly preserved. QED.

**Implication:** rho_k(Delta_W) = 0 exactly. No direction interference.

**Johnson-Lindenstrauss context:** Our Grassmannian A matrices already ensure
inter-adapter orthogonality (Finding: |cos|=0.00125). The orthogonal projection
is COMPLEMENTARY — it ensures each adapter is also orthogonal to the BASE MODEL's
knowledge subspace, not just to other adapters.

---

## D. Proof of Guarantee

**Theorem 1 (Knowledge Preservation Under Orthogonal Projection).**
Let W in R^{m x n} have SVD W = U Sigma V^T. Let U_k in R^{m x k} and V_k in R^{n x k}
be the top-k singular vectors. Define:
  P_L = I_m - U_k U_k^T
  P_R = I_n - V_k V_k^T

For any LoRA update Delta_W = s * B^T A^T, define the projected update:
  Delta_W_orth = P_L * Delta_W * P_R

Then:
1. The top-k singular triples of W + Delta_W_orth equal those of W.
2. rho_k(Delta_W_orth) = 0 exactly.
3. The remaining capacity for learning is rank(W) - k singular directions.

*Proof of (1):* From the OPLoRA construction above, U_k^T Delta_W_orth = 0 and
Delta_W_orth V_k = 0. Therefore the top-k left and right singular vectors of W
are still eigenvectors of (W + Delta_W_orth)^T(W + Delta_W_orth) and
(W + Delta_W_orth)(W + Delta_W_orth)^T with the same eigenvalues. QED.

*Proof of (2):* rho_k = ||U_k^T Delta_W_orth V_k||_F / ||Delta_W_orth||_F.
Numerator = 0 by construction. QED.

*Proof of (3):* P_L has rank m-k, P_R has rank n-k. The projected Delta_W_orth
lives in a subspace of dimension (m-k)(n-k), which spans all directions orthogonal
to the knowledge subspace. QED.

**Theorem 2 (Gradient Projection Equivalence).**
Instead of projecting the delta post-hoc, we can equivalently project the gradient
of B during training. At each step, replace:
  grad_B <- P_A * grad_B * P_R
where P_A = A^T P_L A is the projection in the B-parameter space (since
Delta_W = s * B^T A^T, the left projection on Delta_W induces a projection on B
through A).

More precisely, for Delta_W = s * B^T A^T:
  P_L Delta_W P_R = s * P_L (B^T A^T) P_R = s * (P_L B^T)(A^T P_R)

So we need: B^T -> P_L B^T, i.e., project columns of B^T (rows of B) to lie in
range(P_L). And A^T -> A^T P_R, i.e., project rows of A^T (columns of A) to lie
in range(P_R).

Since A is frozen (Grassmannian), we can PRE-COMPUTE A_orth = P_R A at init time.
For B, we project the gradient: grad_B -> grad_B * P_L^T = grad_B * P_L (symmetric).

Actually, let's be more careful. We have:
  Delta_W = s * (A @ B)^T = s * B^T @ A^T  [where A is (d_in, r), B is (r, d_out)]

So Delta_W is (d_out, d_in). The base weight W is also (d_out, d_in).

Left projection P_L in R^{d_out x d_out} acts on rows of Delta_W.
Right projection P_R in R^{d_in x d_in} acts on columns of Delta_W.

P_L Delta_W P_R = P_L (s B^T A^T) P_R = s (P_L B^T)(A^T P_R)

Since A^T is (r, d_in) and P_R is (d_in, d_in):
  A^T P_R = A_orth^T where A_orth = P_R A  [pre-computed, (d_in, r)]

Since P_L B^T: B^T is (d_out, r), P_L is (d_out, d_out):
  P_L B^T = (B P_L^T)^T = (B P_L)^T  [since P_L symmetric]

So the constraint is: use A_orth = P_R @ A (pre-computed, frozen) and project
B's gradient so that B @ P_L = B, i.e., rows of B lie in range(P_L).

**Simplified implementation:** Replace A with A_orth = P_R @ A at initialization.
Project grad_B at each step: grad_B <- grad_B @ P_L. This ensures the composed
delta s * B^T @ A_orth^T = s * B^T @ (A^T P_R) is already right-projected,
and the left projection is enforced through B.

**Even simpler:** Since P_L B^T means projecting each column of B^T (= each row
of B), we project: B <- P_L @ B^T then transpose back... No.

Let me be precise. B is (r, d_out). grad_B is (r, d_out).
Delta_W = s * B^T @ A^T is (d_out, d_in).
P_L is (d_out, d_out). P_L @ Delta_W = P_L @ (s B^T A^T) = s (P_L B^T) A^T.
So P_L B^T means: B^T is (d_out, r), P_L @ B^T is (d_out, r).
Equivalently: (P_L @ B^T)^T = B @ P_L^T = B @ P_L.
So we need B -> B @ P_L, i.e., right-multiply B by P_L.

For gradient projection: grad_B -> grad_B @ P_L.

**Summary of implementation:**
1. At init: compute SVD of each base weight W, extract U_k, V_k
2. Compute P_L = I - U_k @ U_k^T, P_R = I - V_k @ V_k^T
3. Replace A with A_orth = P_R @ A (frozen, pre-computed)
4. After each gradient step on B: project grad_B -> grad_B @ P_L

This adds ZERO hyperparameters beyond k (the number of singular directions
to preserve). k is the ONE unknown we explore (Type 2).

---

## D. Predictions

### Behavioral Predictions
1. MMLU math degradation will be <=15pp (vs -25pp with DARE) because
   the top-k singular directions encoding factual knowledge are preserved exactly.
2. GSM8K gains (+6pp) will be preserved because mathematical reasoning uses
   the orthogonal complement (procedural, not factual recall).
3. In-distribution accuracy will be >=90% of baseline because the effective
   rank available for learning is (min(d_out,d_in) - k), which at k=16 and
   d=2560 leaves 2544 directions — ample capacity.

### Quantitative Predictions (from Theorem 1)
| Prediction | Source | Expected Value |
|-----------|--------|----------------|
| rho_k after training | Theorem 1 | 0.0 exactly |
| MMLU math degradation | K1 | <=15pp (vs -25pp baseline) |
| GSM8K gain over base | K2 | >=+3pp |
| In-dist accuracy ratio | K3 | >=0.90 |
| Training loss convergence | Capacity (d-k >> r) | Within 1.1x of baseline |

---

## E. Assumptions & Breaking Conditions

1. **Top-k singular directions encode knowledge.** If knowledge is distributed
   across ALL singular directions (not concentrated in top-k), preserving top-k
   is insufficient. Breaking: MMLU math still degrades despite rho_k=0.
   → Would need k -> rank(W), which leaves zero learning capacity.

2. **Orthogonal complement has sufficient capacity.** At d=2560 and k=16,
   the complement has dimension 2544. If k is too large (e.g., k=256),
   capacity shrinks. Breaking: training loss fails to converge.

3. **Grassmannian A + projected A_orth still provides good subspace.**
   The projection P_R @ A may reduce the effective rank of A if A already
   has components in span(V_k). Breaking: adapter quality degrades >10%.

4. **SVD is meaningful for ternary weights.** BitNet weights are ternary
   {-1, 0, 1} * scale. Their SVD may have a flatter spectrum than FP16
   weights. Breaking: no clear gap between top-k and remaining singular values.
   → Mitigated: we unpack to bf16 before SVD.

---

## F. Worked Example (d=8, r=2, k=2)

Base weight W (8x8), take a simple example:
  W = diag(10, 8, 3, 2, 1, 1, 1, 1)  [diagonal for clarity]

SVD: U = V = I, Sigma = diag(10, 8, 3, 2, 1, 1, 1, 1)
Top-2: U_2 = [e1, e2], V_2 = [e1, e2]

P_L = I - U_2 U_2^T = diag(0, 0, 1, 1, 1, 1, 1, 1)
P_R = I - V_2 V_2^T = diag(0, 0, 1, 1, 1, 1, 1, 1)

Grassmannian A (8x2), say A = [[0,0], [0,0], [1,0], [0,1], [0,0], [0,0], [0,0], [0,0]]
A_orth = P_R @ A = A  (A already in complement — no change)

If A = [[1,0], [0,1], [0,0], ...] (overlaps with V_2):
A_orth = P_R @ A = [[0,0], [0,0], [0,0], ...] = 0!
→ A has been zeroed because it lived entirely in the knowledge subspace.

More realistic A = [[0.5, 0.3], [0.2, 0.7], [0.6, 0.1], [0.1, 0.5], ...]:
A_orth rows 1,2 are zeroed, rows 3+ preserved.
→ Effective rank of A_orth may be reduced. This is the capacity tradeoff.

At d=2560, k=16: only 16/2560 = 0.6% of input directions are blocked.
Grassmannian A with r=16 has 16 columns in R^2560 — probability that all 16
columns lie in span(V_16) is negligible. Expected rank reduction: ~16/2560 * 16 = 0.1 columns.

---

## G. Complexity & Architecture

**One-time SVD cost:** For each of 30 layers x 7 projections = 210 weight matrices.
Typical size: 2560x2560 (q,k,v,o) and 2560x6912 (gate,up) and 6912x2560 (down).
SVD of 2560x2560: O(d^3) ~ 1.7e10 FLOPs. Total: 210 * ~1.7e10 = 3.5e12 FLOPs.
On M5 Pro: ~1-2 minutes. Done once, cached.

**Per-step gradient projection:** grad_B @ P_L where P_L = I - U_k U_k^T.
Instead of materializing P_L, compute: grad_B - (grad_B @ U_k) @ U_k^T.
Cost: 2 * r * d_out * k per projection. At r=16, d_out=2560, k=16: ~1.3M FLOPs.
vs normal gradient step: negligible overhead.

**Storage:** U_k (d_out x k) + V_k (d_in x k) per weight matrix.
At k=16: 2 * 2560 * 16 * 2 bytes = 163 KB per matrix. 210 matrices = 34 MB total.
Fits easily in memory.

---

## Self-Test (MANDATORY)

1. What is the ONE mathematical property that makes the failure mode impossible?
   **Orthogonal projection ensures Delta_W has zero component in the top-k singular
   subspace of W, making knowledge corruption impossible by construction.**

2. Which existing theorem(s) does the proof build on?
   **OPLoRA Theorem 1 (arXiv:2510.13003): double-sided orthogonal projection
   preserves top-k singular triples exactly.**

3. What specific numbers does the proof predict?
   **rho_k = 0.0 exactly; MMLU math <=15pp degradation; GSM8K >=+3pp; in-dist >=90%.**

4. What would FALSIFY the proof (not just the experiment)?
   **The proof is wrong if: (a) knowledge is NOT stored in top-k singular directions
   (distributed across full spectrum), or (b) the SVD of ternary weights is degenerate
   (no spectral gap, so "top-k" is arbitrary).**

5. How many hyperparameters does this approach add?
   **Count: 1 (k, number of preserved singular directions). Cannot be derived from
   math alone because it depends on how much knowledge is in top-k vs remaining
   directions — this is the Type 2 exploration target.**

6. Hack check: Am I adding fix #N to an existing stack?
   **No. This replaces DARE (post-hoc sparsification) with a training-time constraint.
   The constraint is a single mechanism (orthogonal projection) that makes direction
   interference impossible. DARE is no longer needed for this component.**
