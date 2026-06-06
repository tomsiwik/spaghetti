# OSRM-Constrained Adapter Init: Mathematical Foundation

## Section 0: Failure Mode & Impossibility Structure

### Failure Mode
Standard LoRA composition fails when adapter cross-terms are large:
```
h_composed = Wx + B_1 A_1 x + B_2 A_2 x
```
When adapter 2 processes input from domain 1, the cross-activation `A_2 @ h_1` projects domain-1 features into adapter-2's subspace. If `||A_2 @ H_1^T||_F` is large, adapter 2 corrupts domain-1 inputs.

**Our prior finding (#68):** At d=2560, weight-space orthogonality does NOT imply data-space orthogonality. 100% of adapter pairs fail OSRM data-orthogonality (mean ratio 0.86). Yet composition still works (PPL 8.35 composed vs 8.58 best individual).

### Impossibility Structure
OSRM (arXiv:2505.22934, ACL 2025) constrains A matrices to make cross-activation mathematically minimal:

**Objective:** For adapter i serving domain i, initialize A_i in the subspace where OTHER domains' features have minimal variance.

Given feature matrix H_j in R^{k x d} (k samples, d=2560) from domain j:
```
S_j = H_j^T H_j / (k-1)     # Covariance matrix, d x d
```

For adapter i, compute the combined covariance of all OTHER domains:
```
S_{-i} = sum_{j != i} S_j / (N-1)
```

Eigendecompose: `S_{-i} = V diag(lambda) V^T`

**OSRM init:** Set A_i = V[:, n-r:n]^T (bottom-r eigenvectors = directions of MINIMAL variance in other domains' features).

**Guarantee:** By construction, ||A_i @ H_j^T||_F^2 = sum_{l in bottom-r} lambda_l * ||H_j @ v_l||^2, which is minimized over all rank-r subspaces.

### What Could Break This
1. **Minimal-variance directions may carry no useful signal for domain i either** -- constraining A_i to avoid others may starve it of its own signal.
2. **Covariance eigenspectrum may be flat** -- at d=2560, if hidden states span a low-dimensional manifold (~22 intrinsic dims per finding #68), most directions have near-zero variance. OSRM init becomes nearly equivalent to random.
3. **B matrices learn to compensate** -- even with constrained A, B can rotate back into interfering subspaces. The constraint is on A only.

### Quantitative Predictions
- **Individual quality:** OSRM adapters within 5% of random init (K1 threshold). The bottom-r subspace should still contain useful domain-specific signal.
- **Merge quality:** OSRM merge +5pp better than unconstrained merge (success S1). Lower cross-activation -> cleaner composition.
- **Comparison to Grassmannian:** Grassmannian is geometry-aware (packing on Gr(r,d)), OSRM is data-aware (covariance-constrained). OSRM should win for merge quality; Grassmannian may win for individual quality.

## Section 1: OSRM Algorithm

### Step 1: Feature Collection
For each domain i, collect hidden states from frozen base model:
```
H_i = [h_i^1, h_i^2, ..., h_i^k]^T    # k x d matrix
```
where h_i^j = mean_pool(base_model(x_i^j)) over sequence positions.

### Step 2: Covariance Computation
Per-domain covariance:
```
S_i = (H_i - mean(H_i))^T (H_i - mean(H_i)) / (k-1)    # d x d
```

Leave-one-out combined covariance for adapter i:
```
S_{-i} = (1/(N-1)) sum_{j != i} S_j    # d x d
```

### Step 3: Constrained A Initialization
Eigendecompose S_{-i} = V Lambda V^T.
Set A_i = V[:, d-r:d]^T (last r columns = smallest eigenvalues).
Shape: A_i in R^{r x d}, satisfying A_i A_i^T = I_r.

### Step 4: Training
Freeze A_i. Train B_i with STE ternary quantization (standard pipeline).
Each adapter sees only its own domain data.

### Step 5: Composition
Merge: h = Wx + (1/N) sum_i B_i A_i x (uniform 1/N scaling).
Cross-activation metric: ||A_i @ H_j^T||_F / (||A_i|| * ||H_j||) for all i != j.

## Section 2: Experimental Design

### Three Conditions
1. **Random init (baseline):** A_i ~ N(0, 1/d), orthogonalized via QR
2. **Grassmannian init:** A_i from AP-packed frames on Gr(r, d)
3. **OSRM init:** A_i from minimal-variance eigenvectors of S_{-i}

### Metrics
- Per-adapter PPL on own domain (individual quality)
- Per-adapter PPL on other domains (cross-domain interference)
- Composed PPL on all domains (merge quality)
- Cross-activation norms ||A_i @ H_j^T||_F for i != j
- Eigenspectrum of S_{-i} (flatness check)

### Kill Criteria
- K1: OSRM adapters >5% worse individually than random init
- K2: Merged quality not better than unconstrained (random) merge

### Success Criteria
- S1: Merged quality within 5% of best individual, +5pp over unconstrained merge
