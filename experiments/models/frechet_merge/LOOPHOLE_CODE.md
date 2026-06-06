# Code Analyzer Audit: frechet_merge

## 1. Metric Hacking via Mathematical Tautology (Circular Evaluation)
In `frechet_merge.py`, the success of the algorithm (K1 criteria) is evaluated using the `subspace_preservation` metric:
```python
overlap = np.linalg.norm(merged_frame.T @ Q[:, :r], 'fro') ** 2 / r
```
The sum of this metric across all N experts is proportional to `Tr(merged_frame^T P_avg merged_frame)` where `P_avg = (1/N) sum_i U_i U_i^T`. 
However, the `chordal_frechet_mean` function computes exactly the top-r eigenvectors of `P_avg`. By definition (the variational characterization of eigenvalues / PCA), this is the mathematically guaranteed argmax of the `subspace_preservation` metric. 
The K1 evaluation is a tautology: the proposed algorithm "beats" the naive baseline only because it is analytically solving for the exact proxy metric being measured. 

## 2. Unfair Capacity Mismatch in Downstream Testing
In `downstream_ntp_test.py`, the baseline `W_naive` is evaluated unfairly against `W_chordal`:
```python
# Naive addition has rank N*r
delta_naive = naive_addition(A_list, B_list, alpha=1.0, rank=rank)
W_naive = W_base + delta_naive

# Chordal is compressed to rank r
merged_chordal = chordal_frechet_mean(A_list, rank)
delta_chordal = make_delta_from_subspace(merged_chordal, B_list, A_list, alpha=1.0, rank=rank)
W_chordal = W_base + delta_chordal
```
The naive sum yields a weight perturbation of rank $N \times r$ (no compression), whereas the chordal merge outputs a rank $r$ perturbation. It is fundamentally invalid to compare the downstream reconstruction MSE of an uncompressed baseline (using N times more capacity) against a compressed model. This unfair comparison is why the `downstream_results.json` shows massive negative relative performance (-69%) for chordal merge.

## 3. Suboptimal Rank-r Reconstruction vs Truncated SVD
In `frechet_merge.py`, the weight delta reconstruction `make_delta_from_subspace` does the following:
```python
alignment = merged_frame.T @ Q[:, :r]
B_merged += alignment @ B
delta = scale * (merged_frame @ B_merged)
```
Mathematically, this reduces exactly to projecting the naive sum onto the merged subspace:
$\Delta_{chordal} = P_{chordal} \Delta_{naive}$
By the Eckart-Young-Mirsky theorem, the optimal rank-r approximation to $\Delta_{naive}$ in the Frobenius norm is its truncated SVD ($U_r \Sigma_r V_r^T$). Projecting onto $P_{chordal}$ (which ignores the magnitudes in the $B$ matrices) is strictly suboptimal compared to simply running SVD on the naive sum. The method is functionally inferior to a baseline that performs SVD truncation on the naive sum.

## 4. Misleading Latency Assessment (K2)
The script assesses the K2 latency criteria as surviving by claiming:
```python
# At serving time, naive = N matmuls; chordal = 1 SVD + N matmuls
# NOTE: This is one-time merge cost. Per-token cost after merge is identical.
k2_killed = False  # Per-token cost is identical
```
This is contradictory. If the merged weights are applied identically (i.e., folded into a dense matrix $W_{base} + \Delta$), then the serving latency is always identical, rendering K2 moot. If they are applied as LoRA adapters, naive addition would require $N \times r$ operations (or require SVD truncation to reach rank $r$), while chordal merge requires $r$ operations. By evaluating offline merge time for `timing` metrics but citing identical per-token costs for survival, the benchmark obfuscates the real overheads of low-rank vs full-rank serving.