# Spectral Surgery Post-Composition: Mathematical Analysis

## Type: Guided Exploration (Type 2)

The Grassmannian orthogonality framework is proven. The unknown is whether post-composition
SVD reweighting can improve quality beyond raw summation.

## A. Failure Mode Identification (Disease, Not Symptom)

**The proposed disease:** After summing N adapter deltas, some singular components of
the composed delta represent cross-domain interference rather than useful domain knowledge.
Reweighting singular values could suppress interference while preserving signal.

**But is this the real disease?** Let us examine what the composition actually looks like.

Each adapter delta has the form:
$$\Delta_i = s_i \cdot B_i^T A_i^T$$

where $A_i \in \mathbb{R}^{r \times d_{in}}$ are Grassmannian-orthogonal frozen projections,
$B_i \in \mathbb{R}^{r \times d_{out}}$ are learned, $s_i$ are domain-specific scales.

The composed delta is:
$$\Delta_{comp} = \sum_{i=1}^N s_i B_i^T A_i^T$$

## B. The Right Question (Reframe)

**Wrong question:** "Can we clean up the composed delta's spectrum to improve quality?"

**Right question:** "Does the composed delta HAVE harmful spectral components that
don't exist in the individual adapters?"

If the answer is NO (the composition doesn't create interference artifacts), then
spectral surgery has nothing to fix and will only degrade quality by removing signal.

## C. Prior Mathematical Foundations

### Theorem (Grassmannian Orthogonality Spectral Decomposition)

**Claim.** If $A_i^T A_j = 0$ for all $i \neq j$ (Grassmannian orthogonality), then
the singular values of $\Delta_{comp} = \sum_i s_i B_i^T A_i^T$ are exactly the
union of the scaled singular values of each individual adapter.

**Proof.**

Let $A_i \in \mathbb{R}^{r \times d}$ be row-orthogonal to $A_j$ for $i \neq j$.
Define $P_i = A_i^T A_i$ as the orthogonal projector onto the row space of $A_i$.

Since $A_i A_j^T = 0$ (from $A_i^T A_j = 0$ noting dimensions), the projectors satisfy
$P_i P_j = 0$ for $i \neq j$.

The Gram matrix of the composed delta is:
$$\Delta_{comp}^T \Delta_{comp} = \sum_{i,j} s_i s_j A_i B_i B_j^T A_j^T$$

For $i \neq j$: The term $A_i B_i B_j^T A_j^T$ maps from $\text{col}(A_j^T)$
to $\text{col}(A_i^T)$. Since $P_i P_j = 0$, these subspaces are orthogonal.

Actually, let us be more careful. We have $\Delta_i = s_i B_i^T A_i^T$ where
$B_i \in \mathbb{R}^{r \times d_{out}}$, $A_i \in \mathbb{R}^{r \times d_{in}}$.

$$\Delta_{comp}^T \Delta_{comp} = \sum_{i,j} s_i s_j (A_i B_i)(B_j^T A_j^T)$$

Wait — let me be precise about dimensions.

$\Delta_i = s_i B_i^T A_i^T \in \mathbb{R}^{d_{out} \times d_{in}}$

where $B_i \in \mathbb{R}^{r \times d_{out}}$, so $B_i^T \in \mathbb{R}^{d_{out} \times r}$,
and $A_i^T \in \mathbb{R}^{r \times d_{in}}$.

Wait, that's $d_{out} \times r$ times $r \times d_{in}$ = $d_{out} \times d_{in}$. But
$B_i^T A_i^T$ doesn't contract correctly. Let me recheck the convention.

From the code: `delta = scale * (b_mx.T @ a_mx.T)` where $B \in \mathbb{R}^{d_{out} \times r}$
and $A \in \mathbb{R}^{r \times d_{in}}$. So:

$$\Delta_i = s_i B_i A_i \in \mathbb{R}^{d_{out} \times d_{in}}$$

where $B_i \in \mathbb{R}^{d_{out} \times r}$, $A_i \in \mathbb{R}^{r \times d_{in}}$.

The Gram matrix (right):
$$\Delta_{comp}^T \Delta_{comp} = \left(\sum_i s_i A_i^T B_i^T\right)\left(\sum_j s_j B_j A_j\right)$$
$$= \sum_{i,j} s_i s_j A_i^T B_i^T B_j A_j$$

For $i \neq j$: $A_i^T B_i^T B_j A_j$ maps $d_{in} \to d_{in}$ via:
$x \mapsto A_i^T (B_i^T B_j) A_j x$

The range of $A_j$ is the row space of $A_j$ (embedded in $\mathbb{R}^r$).
After multiplication by $B_i^T B_j$, we get a vector in $\mathbb{R}^r$.
Then $A_i^T$ maps back to $d_{in}$.

The image of $A_j$ lies in $\text{col}(A_j^T) \perp \text{col}(A_i^T)$ (by Grassmannian orthogonality).
Wait — $A_i$ has rows that are orthogonal to $A_j$'s rows. This means $A_i A_j^T = 0_{r \times r}$.

So $A_i^T (B_i^T B_j) A_j$ — the $A_j$ maps from $d_{in}$ into $\mathbb{R}^r$, then $B_i^T B_j$
maps $\mathbb{R}^r \to \mathbb{R}^r$, then $A_i^T$ maps $\mathbb{R}^r \to d_{in}$.

This is NOT necessarily zero because $A_i^T$ and $A_j$ operate on DIFFERENT copies of $\mathbb{R}^r$.
The Grassmannian orthogonality $A_i A_j^T = 0$ means the ROW SPACES of $A_i$ and $A_j$ are orthogonal
in $\mathbb{R}^{d_{in}}$. It does NOT imply $A_i^T M A_j = 0$ for arbitrary $M$.

But for the LEFT Gram matrix:
$$\Delta_{comp} \Delta_{comp}^T = \sum_{i,j} s_i s_j B_i A_i A_j^T B_j^T$$

For $i \neq j$: $A_i A_j^T = 0$ (Grassmannian orthogonality). So:
$$\Delta_{comp} \Delta_{comp}^T = \sum_i s_i^2 B_i A_i A_i^T B_i^T = \sum_i s_i^2 B_i B_i^T$$

Wait, that's only true if $A_i A_i^T = I_r$ (each $A_i$ has orthonormal rows). The Grassmannian
construction gives us orthonormal rows, so $A_i A_i^T = I_r$. Therefore:

$$\Delta_{comp} \Delta_{comp}^T = \sum_i s_i^2 B_i B_i^T$$

This is a sum of rank-$r$ PSD matrices. Its eigenvalues determine the singular values of
$\Delta_{comp}$. The cross terms vanished because of Grassmannian orthogonality.

**But the $B_i$ matrices are NOT orthogonal to each other.** $B_i B_j^T \neq 0$ in general.
However, this doesn't appear in the Gram matrix — the cross terms vanished due to $A_i A_j^T = 0$.

**Key insight:** The LEFT singular vectors of $\Delta_{comp}$ come from the $B_i$'s,
which DO share subspace. The RIGHT singular vectors come from the $A_i$'s, which are
perfectly orthogonal. The singular values of $\Delta_{comp}$ are determined by the
eigenvalues of $\sum_i s_i^2 B_i B_i^T$.

## D. Main Result

**Theorem 1 (Spectral Structure of Grassmannian-Orthogonal Composition).**

Let $\Delta_i = s_i B_i A_i$ with $A_i A_j^T = 0$ for $i \neq j$ and $A_i A_i^T = I_r$.
Then:

(a) $\Delta_{comp} \Delta_{comp}^T = \sum_i s_i^2 B_i B_i^T$ (cross-terms vanish)

(b) If additionally $B_i^T B_j = 0$ for $i \neq j$ (output spaces orthogonal), then
the singular values of $\Delta_{comp}$ are exactly $\{s_i \sigma_k(B_i) : k=1,...,r, i=1,...,N\}$.

(c) In general (B_i non-orthogonal), the singular values of $\Delta_{comp}$ satisfy
the Weyl interlacing inequalities but are NOT simply the union of individual singular values.

*Proof of (a).* $\Delta_{comp} \Delta_{comp}^T = \sum_{i,j} s_i s_j B_i (A_i A_j^T) B_j^T$
$= \sum_i s_i^2 B_i (A_i A_i^T) B_i^T = \sum_i s_i^2 B_i B_i^T$. QED.

*Proof of (b).* If $B_i^T B_j = 0$, then $\text{col}(B_i) \perp \text{col}(B_j)$.
So $\sum_i s_i^2 B_i B_i^T$ is block-diagonal in a basis aligned with $\text{col}(B_i)$,
and its eigenvalues are the union of eigenvalues of each $s_i^2 B_i B_i^T$. QED.

**Corollary 1 (Surgery Cannot Help Under Perfect Orthogonality).**

If both $A_i \perp A_j$ AND $B_i \perp B_j$, then the singular values of $\Delta_{comp}$
are exactly the scaled individual singular values. There is nothing to "clean up" — every
singular component belongs to exactly one adapter. Removing any singular component can only
hurt quality.

**Corollary 2 (Surgery Target: B-matrix Overlap).**

The ONLY potential target for spectral surgery is the cross-interaction of $B_i$ matrices
in the output space. If $B_i^T B_j \neq 0$, the eigenvalues of $\sum_i s_i^2 B_i B_i^T$
differ from the union of individual eigenvalues. The difference represents constructive
or destructive interference between output-space components.

## D (continued). Quantitative Predictions

**Prediction 1 (Cross-term magnitude):** The project has measured B-matrix cosine similarity
at ~0.03 (Finding: "17x filter"). With such low B-overlap, the eigenvalues of
$\sum_i s_i^2 B_i B_i^T$ should be very close to the union of individual eigenvalues.
Specifically, by Weyl's inequality:

$$|\sigma_k(\Delta_{comp}) - \sigma_k^{indep}| \leq \|E\|_2$$

where $E = \sum_{i \neq j} s_i s_j B_i (A_i A_j^T) B_j^T = 0$ (from Grassmannian orthogonality).

Wait — $E = 0$ exactly. The cross-terms vanish in the LEFT Gram matrix. So the eigenvalues
of $\Delta_{comp} \Delta_{comp}^T$ equal the eigenvalues of $\sum_i s_i^2 B_i B_i^T$, which
is a sum of N rank-r PSD matrices.

The deviation from "union of individual spectra" is:
$$\sum_i s_i^2 B_i B_i^T - \text{diag}(s_i^2 B_i B_i^T)$$

where "diag" means block-diagonal in the B_i column spaces. The off-diagonal contribution
has Frobenius norm $\leq \sum_{i<j} 2 s_i s_j |B_i^T B_j|_F$.

With empirical B-cosine ~0.03 and $||B_i||_F \approx ||B_j||_F$:
$$\text{off-diag contribution} \leq N(N-1) \cdot s_{max}^2 \cdot 0.03 \cdot ||B||_F^2$$

At N=5 with s_max=20: the off-diagonal is at most ~$20 \cdot 400 \cdot 0.03 = 240$ times
$||B||_F^2$... this needs more careful calculation with actual norms.

**Let me instead predict what the experiment will show:**

1. **Singular value structure:** The composed delta's SVD will show ~N*r = 80 non-trivial
   singular values (5 adapters x rank 16). The spectrum will be approximately the sorted
   union of individual scaled spectra. Deviation from exact union: < 5% relative.

2. **Surgery effect on PPL:** Since the spectrum is already "clean" (each component belongs
   primarily to one adapter), removing/reweighting components will either:
   - Do nothing (reweighting converges to identity)
   - Hurt quality (removing signal components)
   Predicted improvement: < 0.5% (within noise), likely negative.

3. **Calibration speed:** SVD of the composed delta (d_out x d_in matrices per layer)
   is expensive. For BitNet-2B with 30 layers x 7 target keys = 210 matrices, each up
   to 2560x2560. SVD of a 2560x2560 matrix: ~1-2s per matrix on M5 Pro.
   Total: ~210-420s for full SVD. Gradient-based sensitivity adds another forward pass.
   Prediction: > 30s (K697 FAIL).

4. **Correlation of "harmful" components with interference:** Since cross-terms vanish
   (Theorem 1a), there ARE no cross-domain interference components in the SVD. Any
   component identified as "harmful" by gradient sensitivity will be a low-signal
   component of a single adapter, not an interference artifact.
   Prediction: correlation near 0 (K698 FAIL).

## E. Assumptions & Breaking Conditions

1. **Grassmannian orthogonality holds.** Confirmed empirically (|cos| = 0.00125).
   If violated, cross-terms in the Gram matrix become non-zero and interference
   artifacts WOULD appear in the SVD — making surgery potentially useful.

2. **Adapters are low-rank (r << d).** At r=16, d=2560, each adapter uses 0.625%
   of the available spectrum. With N=5 adapters, 3.125% used. Overlap is sparse.

3. **B-matrix overlap is small.** Measured at ~0.03 cosine. If B overlap were large
   (say >0.3), the eigenvalue deviation from "union of spectra" could be significant,
   and surgery might find genuine interference to remove.

## F. Worked Example (d=8, r=2, N=2)

Let $A_1 = \begin{pmatrix} 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \end{pmatrix}$,
$A_2 = \begin{pmatrix} 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \end{pmatrix}$.

Clearly $A_1 A_2^T = 0$. Take $s_1 = s_2 = 1$.

Let $B_1 = \begin{pmatrix} 3 & 0 \\ 0 & 1 \end{pmatrix}$,
$B_2 = \begin{pmatrix} 2 & 0.5 \\ 0.5 & 2 \end{pmatrix}$.

$\Delta_1 = B_1 A_1$ has singular values $\{3, 1\}$.
$\Delta_2 = B_2 A_2$ has singular values $\{\sigma(B_2)\} = \{2.5, 1.5\}$.

$\Delta_{comp} \Delta_{comp}^T = B_1 B_1^T + B_2 B_2^T = \begin{pmatrix} 9 & 0 \\ 0 & 1 \end{pmatrix} + \begin{pmatrix} 4.25 & 2 \\ 2 & 4.25 \end{pmatrix} = \begin{pmatrix} 13.25 & 2 \\ 2 & 5.25 \end{pmatrix}$

Eigenvalues: $\frac{18.5 \pm \sqrt{(8)^2 + 16}}{2} = \frac{18.5 \pm \sqrt{80}}{2} = \frac{18.5 \pm 8.944}{2}$
$= \{13.72, 4.78\}$.

So singular values of $\Delta_{comp}$: $\{\sqrt{13.72}, \sqrt{4.78}\} = \{3.70, 2.19\}$.

Compare with union-of-individuals: $\{3, 2.5, 1.5, 1\}$ → sorted $\{3, 2.5, 1.5, 1\}$.
But $\Delta_{comp}$ is only rank 2 (since B's share the same 2D output space), so it has
only 2 singular values, not 4!

**Wait — this is the key.** When B matrices share output space dimensions, the composed
delta has LOWER rank than the sum of individual ranks. The singular values are NOT the
union — they are a MIXED version.

In this example: individual σ's = {3, 2.5, 1.5, 1}, composed σ's = {3.70, 2.19}.
The composition MIXES the spectral energy. The largest component (3.70 > 3.0) got BOOSTED
by constructive interference, while other components got redistributed.

But in practice, with d_out = 2560, the B matrices are d_out x r = 2560 x 16. The column
spaces of B_i span at most r=16 dimensions each, and with low cosine (~0.03), they are
nearly non-overlapping. So the composed delta has rank very close to N*r, and its singular
values are very close to the union.

**Revised prediction:** With near-orthogonal B matrices at the scale we operate
(d_out=2560, r=16, N=5, B-cosine 0.03), the composed singular values deviate < 3%
from the sorted union. Surgery has almost nothing to work with.

## G. Complexity & Architecture Connection

- **SVD cost:** O(d^2 * min(d_out, d_in)) per layer-key. With d=2560: ~O(d^3) = O(1.68e10).
  For 210 layer-keys: ~3.5e12 FLOPs. On M5 Pro at ~10 TFLOPS: ~350s.
- **Gradient sensitivity:** One forward pass through 2.4B model on calibration data (~128 samples).
  At 97 tok/s: ~128 * 256 / 97 = ~338s. Plus backprop: ~2x = ~676s.
- **Total expected time:** ~1000s = ~17 minutes for full pipeline.
- **Memory:** SVD of 2560x2560 matrix: ~50MB per matrix. Can process sequentially.

## Self-Test

1. **What is the ONE mathematical property that makes the failure mode impossible?**
   Grassmannian orthogonality ($A_i A_j^T = 0$) eliminates cross-terms in the LEFT
   Gram matrix, so the composed delta's spectral structure is determined entirely by
   $\sum_i s_i^2 B_i B_i^T$ with no interference artifacts. There is nothing for
   spectral surgery to fix.

2. **Which existing theorem(s) does the proof build on?**
   Weyl's inequality for eigenvalue perturbation (1912). Properties of orthogonal
   projectors. SVD structure of sums of rank-r matrices.

3. **What specific numbers does the proof predict?**
   - Composed singular values within 3% of sorted union of individual singular values
   - Surgery improves PPL by < 0.5% (noise floor)
   - Calibration time > 30s (K697 FAIL predicted)
   - No correlation between "harmful" components and cross-domain interference (K698 FAIL)

4. **What would FALSIFY the proof?**
   If composed singular values deviate > 10% from sorted union of individual singular values,
   then B-matrix overlap is larger than measured, and the cross-term cancellation is incomplete.
   This would indicate Grassmannian orthogonality alone is insufficient for spectral purity.

5. **How many hyperparameters?**
   Count: 3 (eta_sup, eta_amp for surgery, n_calibration for gradient sensitivity).
   These come from the paper, not from the proof. The proof predicts surgery is unnecessary,
   so these hyperparameters become moot.

6. **Hack check:** This is NOT adding a fix. This is TESTING whether a proposed fix
   (spectral surgery) is necessary. The proof predicts it is NOT, and the experiment
   verifies that prediction. If the prediction is correct, this closes a research direction.
