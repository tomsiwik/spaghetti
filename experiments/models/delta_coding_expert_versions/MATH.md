# Delta Coding for Expert Version Management: Mathematical Foundations

## 1. Setup

A LoRA expert at version t has parameters theta_t = {A_t^{(l,s)}, B_t^{(l,s)}}
for each layer l in {1, ..., L} and sublayer s in {fc1, fc2}, where:
- A_t^{(l,s)} in R^{d_in x r} (down-projection)
- B_t^{(l,s)} in R^{r x d_out} (up-projection)
- r is the LoRA rank

The effective weight delta at version t is:
  dW_t^{(l,s)} = (alpha / r) * A_t^{(l,s)} @ B_t^{(l,s)}  in R^{d_in x d_out}

## 2. Video Codec Analogy

We borrow terminology from video compression (MPEG, H.264):

| Video Term | Model Term | Definition |
|------------|------------|------------|
| I-frame (keyframe) | Full snapshot | Complete theta_t stored as-is |
| P-frame (delta frame) | Version delta | delta_{t-1,t} = theta_t - theta_{t-1} |
| GOP (group of pictures) | Keyframe interval K | Max chain length before next keyframe |
| Decode chain | Reconstruction | theta_t = theta_kf + sum_{i=kf+1}^{t} delta_{i-1,i} |

## 3. Raw Delta Coding

**Inter-version delta** between consecutive versions t-1 and t:
  delta_{t-1,t}^{(l,s,A)} = A_t^{(l,s)} - A_{t-1}^{(l,s)}
  delta_{t-1,t}^{(l,s,B)} = B_t^{(l,s)} - B_{t-1}^{(l,s)}

**Reconstruction** of version t from keyframe at version kf:
  A_t^{(l,s)} = A_kf^{(l,s)} + sum_{i=kf+1}^{t} delta_{i-1,i}^{(l,s,A)}
  B_t^{(l,s)} = B_kf^{(l,s)} + sum_{i=kf+1}^{t} delta_{i-1,i}^{(l,s,B)}

**Exactness guarantee**: This reconstruction is EXACT (up to floating-point
precision) because it is pure addition/subtraction with no nonlinearity.
Measured error: ~1.6e-08 relative Frobenius norm after 4 chained deltas.

**Storage cost (raw)**:
- Full storage of N versions: N * P parameters (where P = total LoRA params)
- Delta storage with keyframe interval K:
  N_keyframes = ceil(N / K)
  N_deltas = N - N_keyframes
  Total = (N_keyframes + N_deltas) * P = N * P  (no savings with raw deltas)

Raw deltas are the SAME SIZE as full snapshots because A and B matrices
have the same dimensions in both.

## 4. Compressed Delta Coding (SVD Truncation)

The key observation: inter-version deltas are more compressible than full
params because they represent INCREMENTAL CHANGES, which often lie in a
lower-dimensional subspace than the full parameter space.

**Truncated SVD compression** of each delta matrix D in R^{m x n}:
  D = U @ diag(S) @ V^T  (full SVD)
  D_r = U[:, :r'] @ diag(S[:r']) @ V^T[:r', :]  (truncated to rank r')

**Compressed storage** per delta matrix:
  Full: m * n elements
  Compressed: m*r' + r' + r'*n = r'*(m + n + 1) elements

**Compression ratio** (per matrix):
  rho = r'*(m + n + 1) / (m * n)

For LoRA matrices with d_in=64, r=8:
  A: shape (64, 8) -> rho(r'=1) = 1*(64+8+1)/512 = 0.143
  A: shape (64, 8) -> rho(r'=2) = 2*(64+8+1)/512 = 0.285
  A: shape (64, 8) -> rho(r'=4) = 4*(64+8+1)/512 = 0.570

For B: shape (8, 256):
  rho(r'=1) = 1*(8+256+1)/2048 = 0.129
  rho(r'=2) = 2*(8+256+1)/2048 = 0.259

Weighted average across all matrices: ~0.132 (rank 1), ~0.264 (rank 2), ~0.528 (rank 4)

**Reconstruction with compression error**:
  A_t^{approx} = A_kf + sum_{i=kf+1}^{t} delta_{i-1,i}^{approx}

The approximation error accumulates additively along the chain:
  ||A_t^{approx} - A_t^{true}|| <= sum_{i=kf+1}^{t} ||delta_{i-1,i}^{approx} - delta_{i-1,i}^{true}||

This is the "chain drift" phenomenon, analogous to P-frame error accumulation
in lossy video codecs.

## 5. Storage Cost with Compression

**Delta-coded storage with SVD compression**:
  Total = N_keyframes * P + N_deltas * rho * P
        = P * (N_keyframes + rho * N_deltas)

**Overall storage ratio** vs full:
  R = (N_keyframes + rho * N_deltas) / N

For N=5, K=5 (1 keyframe, 4 deltas):
  R = (1 + rho * 4) / 5

  rho=0.132 (SVD rank 1): R = (1 + 0.528) / 5 = 0.306  (69.4% savings)
  rho=0.264 (SVD rank 2): R = (1 + 1.056) / 5 = 0.411  (58.9% savings)
  rho=0.528 (SVD rank 4): R = (1 + 2.112) / 5 = 0.623  (37.7% savings)

## 6. Quality-Storage Tradeoff

The fundamental tradeoff is between compression ratio rho and quality drift d:

  d(rho, chain_len) ~ f(rho) * chain_len

where f(rho) is the per-delta compression error. Empirically:
  f(0.132) ~ 0.4% per delta (relative loss change)
  f(0.264) ~ 0.2% per delta
  f(0.528) ~ 0.05% per delta

The optimal operating point depends on the acceptable drift threshold:
- For KC1 threshold of 1%: SVD rank 2 (rho=0.264) is optimal
  -- achieves 41.1% storage ratio with <0.8% max drift across 4 chained deltas
- For tighter thresholds: SVD rank 4 at 62.3% storage

## 7. Worked Example

d=64, r=8, L=4 layers, 2 sublayers (fc1, fc2), N=5 versions, K=5

**LoRA params per version:**
  Per layer: (64*8 + 8*256) + (256*8 + 8*64) = 2560 + 2560 = 5120
  Total: 4 * 5120 = 20,480

**Full storage (5 versions):** 5 * 20,480 = 102,400 params

**Delta-coded, SVD rank 2:**
  1 keyframe: 20,480 params
  4 deltas at rho=0.264: 4 * 0.264 * 20,480 = 21,627 params
  Total: 42,107 params
  Ratio: 42,107 / 102,400 = 0.411 (58.9% savings)
  Max quality drift: <0.8% (3-seed max)

## 8. Assumptions

1. Expert versions evolve gradually (consecutive fine-tuning, not random init).
   If ||delta_{t-1,t}|| << ||theta_t||, deltas are more compressible.
   Measured: ||delta|| / ||params|| ~ 0.37 (not trivially small but structured).

2. The delta lies in a low-rank subspace. This holds because gradient-based
   updates during short fine-tuning (80 steps) produce low-rank weight changes.
   Measured: SVD rank 2 captures ~60% of delta energy (relative error ~0.58).

3. Floating-point accumulation error is negligible. Measured: ~1.6e-08 relative
   error after 4 chained additions. This is below any meaningful threshold.

4. The base model is frozen across all versions. If the base changes, all
   deltas must be recomputed (analogous to changing the I-frame codec).

## 9. Connection to LoRA

LoRA itself IS delta coding: dW = (alpha/r) * A @ B is a compressed
representation of the weight change relative to the base model.

This experiment extends delta coding to the TEMPORAL dimension:
- LoRA: spatial delta (base -> adapted, stored as low-rank factors)
- Our work: temporal delta (version_t -> version_{t+1}, further compressed via SVD)

The two compose: a version-t expert is (base + LoRA_t), and the delta between
versions is (LoRA_t+1 - LoRA_t), which can be SVD-compressed.

## 10. Falsification Criteria

| Criterion | Threshold | Measured | Result |
|-----------|-----------|----------|--------|
| Quality drift per delta (raw) | <1% | 0.0000% | PASS |
| Quality drift per delta (SVD r=2) | <1% | 0.796% | PASS |
| Storage ratio (SVD r=2) | <50% | 41.1% | PASS |
| Storage ratio (SVD r=1) | <50% | 30.6% | PASS (but drift fails) |

## 11. Computational Cost

**Snapshot**: O(P) copy
**Delta computation**: O(P) subtraction
**SVD compression per delta**: O(min(m,n)^2 * max(m,n)) per matrix
  For A (64x8): O(8^2 * 64) = O(4096)
  For B (8x256): O(8^2 * 256) = O(16384)
  Total per version per layer: ~20K FLOPs (negligible)
**Reconstruction**: O(chain_len * P) additions

Total experiment time: ~144s for 3 seeds (48s per seed).
