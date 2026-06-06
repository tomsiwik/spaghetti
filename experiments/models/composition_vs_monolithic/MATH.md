# Composition vs Monolithic: Mathematical Foundations

## 1. Setup

Given a frozen base model with parameters W_base in R^{d x d}, we compare two
approaches for learning N domains {D_1, ..., D_N}:

**Composed (SOLE):** Train N independent experts, each producing a delta:
  W_k = W_base + Delta_k    (expert k trained on domain D_k only)

At inference, route query to the appropriate expert:
  W(query) = W_base + Delta_{route(query)}

For budget matching, truncate each Delta_k to rank r via SVD:
  Delta_k^{(r)} = U_k[:, :r] diag(S_k[:r]) V_k[:r, :]^T

Total composition rank budget: N * r.

**Monolithic:** Train one model on combined data D_1 union ... union D_N:
  W_mono = W_base + Delta_mono

Truncate to rank N*r for budget matching:
  Delta_mono^{(Nr)} = U_m[:, :Nr] diag(S_m[:Nr]) V_m[:Nr, :]^T

## 2. Parameter Budget Equivalence

For a weight matrix W in R^{d_in x d_out}:

  Composed: N experts x rank-r each = N * r * (d_in + d_out) parameters
  Monolithic: 1 model x rank-(Nr) = Nr * (d_in + d_out) parameters

These are identical: N * r * (d_in + d_out) = Nr * (d_in + d_out).

## 3. Information-Theoretic Argument

The monolithic model can allocate its rank-Nr budget optimally across all
domains via the SVD. The top singular values capture cross-domain shared
structure, while lower singular values capture domain-specific features.

The composed model is constrained: each rank-r slice can only represent
one domain. It cannot represent shared structure across domains (each expert
is independent). This is an information-theoretic disadvantage for the
composed approach when domains share structure.

The gap should decrease when:
  (a) Domains are genuinely independent (no shared structure to exploit)
  (b) Rank r is large enough to capture each domain's complexity
  (c) d is large (rank-r captures proportionally more of the delta)

## 4. SVD Signal Retention

When truncating Delta_k to rank r, the fraction of signal retained is:

  rho_k = ||Delta_k^{(r)}||_F / ||Delta_k||_F = sqrt(sum_{i=1}^r sigma_i^2 / sum_{i=1}^d sigma_i^2)

At d=32, r=4: measured rho ~ 0.80 (80% retained).
At d=896, r=16: expected rho ~ 0.95+ (from delta_rank_scaling results).

The signal loss compounds: if each expert retains 80% of its signal, the
effective quality ceiling for each expert is 80% of the full-rank quality.

## 5. Composition Strategies

### 5.1 Sum: W = W_base + sum_k Delta_k
Catastrophic at any scale when N > 1. The total perturbation magnitude
scales as O(N), overwhelming the base model.

### 5.2 Average: W = W_base + (1/N) sum_k Delta_k
Perturbation magnitude O(1), but each expert's contribution is diluted by
1/N. Only works when all experts are simultaneously needed (generalist model).

### 5.3 Routed (SOLE): W(query) = W_base + Delta_{route(query)}
Each query sees exactly one expert. No dilution. Quality equals the
specialist quality on each domain. This is the SOLE architecture.

## 6. Theoretical Quality Gap

Let L_k(W) be the loss of model W on domain D_k.

Routed composition quality on domain k:
  L_k(W_base + Delta_k^{(r)}) = L_k(W_base + Delta_k) + eps_trunc_k

where eps_trunc_k ~ (1 - rho_k) * ||Delta_k||_F is the truncation error.

Monolithic quality on domain k:
  L_k(W_base + Delta_mono^{(Nr)}) = L_k(W_base + Delta_mono) + eps_trunc_mono

The quality gap is:
  gap_k = L_k(composed) - L_k(mono)
        = [L_k(full_expert) + eps_trunc_k] - [L_k(full_mono) + eps_trunc_mono]

This gap has two components:
  1. Full-rank gap: L_k(full_expert) - L_k(full_mono)
     Expert sees only domain k data; monolithic sees all domains.
     Can be positive (mono better via transfer) or negative (expert focuses).
  2. Truncation gap: eps_trunc_k - eps_trunc_mono
     Expert loses more signal per matrix (rank r vs Nr).

## 7. Worked Example (Micro Scale)

d=32, L=2, H=2, V=42, N=5, r=4:

| Metric | Base | Routed | Avg | Mono(r=20) | Seq |
|--------|------|--------|-----|------------|-----|
| Avg loss | 3.756 | 1.635 | 3.001 | 0.956 | 5.221 |
| vs base | 0% | -56% | -20% | -75% | +39% |

Per expert: sig_ret ~ 0.80, meaning rank-4 captures 80% of the full delta.
Full-rank experts: avg loss ~ 1.0 (comparable to monolithic).
Truncated experts: avg loss ~ 1.6 (sig_ret penalty).

The 71% gap (routed vs mono) decomposes as:
  - Full-rank gap: ~5% (mono slightly better due to cross-domain transfer)
  - Truncation gap: ~66% (rank-4 at d=32 is severely capacity-constrained)

At macro scale (d=896, r=16), rho ~ 0.95+, so truncation gap shrinks to ~5-10%.

## 8. Forgetting Analysis

Sequential monolithic (train on D_1, then D_2, ... then D_N):
  After training on D_k, loss on D_j (j < k) increases by:
  forgetting_j = max_{k>j} L_j(after_k) - L_j(after_j)

Measured: mean forgetting = 4.72 (126% of base loss).
This makes sequential monolithic WORSE than not training at all.

Composed approach: forgetting = 0 by construction (experts are independent).
This is a structural guarantee, not an empirical observation.

## 9. Complexity

| Operation | Composed | Monolithic |
|-----------|----------|------------|
| Training | O(N * n_k * T) parallel | O(N * n * T) sequential |
| Storage | N * r * (d_in + d_out) | Nr * (d_in + d_out) |
| Inference | O(d^2) (one expert) | O(d^2) (one model) |
| Add domain | O(n_new * T) | O(N * n * T) retrain |
| Remove domain | O(1) delete | O(N * n * T) retrain |

where n_k = samples per domain, n = total samples, T = training epochs.

## 10. Assumptions

1. **SVD truncation as LoRA proxy:** At micro scale, we train full-rank and
   truncate post-hoc. In production, LoRA trains directly in the low-rank
   subspace. The truncation approach may overestimate the quality gap
   (LoRA can adapt its subspace during training).

2. **Oracle routing:** The routed condition uses perfect domain labels.
   In production, hash ring routing achieves ~96% cluster accuracy
   (proven in content_aware_routing).

3. **Independent domains:** The synthetic domains share no structure
   (arithmetic vs reversal vs parity). Real domains may have more
   shared structure, which favors the monolithic approach.

4. **Fixed rank budget:** We fix r per expert. Adaptive rank (e.g.,
   more rank for complex domains) could improve composition quality.
