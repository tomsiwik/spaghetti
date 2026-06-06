# Self-Organizing Adapter Slots — Killed by Prior Findings

## Proposed Mechanism

Train A-matrices with a loss including a slot-position term, constrained to null(W_v).
Adapters learn WHERE in null space to sit, maximizing quality and routability.

## Why This Is Killed Before Running

### Theorem (Null-Space Information Barrier, Findings #495 + #500)

Let W_v be a weight matrix, A_i the LoRA A-matrix for adapter i, constrained to null(W_v).
Let D_i = domain information for adapter i, which lives in range(W_v) (by construction of LoRA: ΔW = BA adds to range).

**Claim:** No function f of null-space positions {A_i} can extract domain-relevant routing or quality information.

**Proof:**
1. Domain features D live in range(W_v) (the column space where ΔW = BA operates)
2. A-matrices constrained to null(W_v) satisfy: for all vectors s in null(W_v) and d in range(W_v), ⟨s, d⟩ = 0
3. Any function of A-matrix position (direction, magnitude, relative angle) operates purely in null(W_v)
4. By orthogonality: I(f(A_i, x); domain(x)) = 0 for any f and any input x
5. Therefore: self-organized positions cannot learn routing-relevant structure

**Corollary (Finding #495):** Routing via |A_i Q^T x|^2 achieves 20% accuracy (chance at N=5). Even normalized projections carry zero domain signal. Spearman r = -0.19.

**Corollary (Finding #500):** Quality prediction via projection magnitude achieves AUC = 0.4293 (below chance). I(||A_i Q^T x||^2; quality) ≈ 0.

**Corollary (Finding #498):** A-matrices cluster by initialization method (cos ≈ 0.82 standard, cos ≈ 0 Grassmannian), NOT by domain. Training cannot move them to domain-relevant positions because no domain signal exists in null space.

QED

## Kill Criteria Assessment

- **K1309 (routing >= 85%):** KILLED. Null-space position carries zero routing signal (Finding #495). Self-organized positions cannot improve on 20% (chance).
- **K1310 (orthogonality max|cos| < 0.1):** MOOT. Grassmannian initialization already achieves this optimally. Training A-matrices adds no benefit — the mathematical optimum is already a closed-form solution (QR factorization on Grassmannian manifold).
- **K1311 (new adapter finds slot automatically):** KILLED. Depends on K1309 — automatic slot-finding requires position to carry domain signal, which it cannot.

## Impossibility Structure

The null space of W_v is the orthogonal complement of range(W_v). Domain-specific adaptation modifies range(W_v) via ΔW = BA. Position within null(W_v) is structurally decoupled from domain semantics. No training objective, regularizer, or loss term operating on null-space coordinates can inject domain information.

Grassmannian QR initialization is already the optimal solution for orthogonal packing in null space — it maximizes packing density without any training. Self-organization cannot beat a closed-form optimum.

## What This Closes

The P7 null-space geometry line is now fully resolved:
- **Finding #495:** Null-space cannot route (20% accuracy)
- **Finding #496:** Weighted composition works via ensemble (no geometry needed)
- **Finding #497:** Direct prompting dominates adapter reasoning strategies
- **Finding #498:** A-matrices are init-determined, not domain-determined
- **Finding #500:** Null-space cannot predict quality (AUC=0.43)
- **This experiment:** Self-organizing null-space positions cannot learn routing/quality

Null space is an **isolation tool** (interference prevention via orthogonality), not an **information source** (routing, quality, domain detection).
