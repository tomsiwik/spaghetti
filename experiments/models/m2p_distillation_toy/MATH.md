# MATH.md: M2P Distillation with Grassmannian Composition Guarantee

## Experiment Type
Guided exploration — proven framework (Grassmannian orthogonality), unknown parameter (M2P quality achievable on synthetic domains).

## Prior Results
- Finding #339 (exp_shine_session_adapter, SUPPORTED): M2P generates adapters at 66.6% of SFT quality in 1.15ms on real Qwen3-4B-4bit hidden states.
- Finding #225 (near-lossless composition, N=5): composition interference cos ≈ 0.0002 with domain-matched SFT adapters.
- Finding #337 (adapter_promotion, KILLED): scale=20 LoRA deltas violate linearization; scale ≤ 5–13 is safe.

---

## A. Problem Statement

Standard LoRA composition suffers interference: when two adapters Δ_i = B_i A_i^T and Δ_j = B_j A_j^T are summed, their inner product ⟨Δ_i, Δ_j⟩_F can be non-zero, causing activation-space destructive interference.

**Goal:** Architect adapter generation so that **parameter-space** composition interference is **mathematically impossible**, regardless of what the M2P generates for B-matrices.

**Scope caveat:** Theorem 1 proves zero Frobenius inner product between weight perturbations — this is *parameter-space* orthogonality. It does **not** guarantee zero activation-space interference. When both adapters are active on the same input x, the cross-term `trace(B_j^T A_j^T x^T x A_i B_i)` is zero by Theorem 1 only if `A_j^T A_i = 0` (which holds), but the norms of individual activation corrections are not suppressed. In practice, at rank 4 in dimension 64, the adapters occupy low-dimensional subspaces of the activation space, so activation-space interference is empirically small — but it is not formally zero for arbitrary data.

---

## B. Setup and Notation

**Toy GPT:** d = 64, L = 4 layers, H = 4 heads, vocab = 95 (printable ASCII).

**LoRA adapter:** For input x ∈ ℝ^{d_in}, the adapter output is:
```
output = x @ W_base^T + scale × (x @ A) @ B
```
where A ∈ ℝ^{d_in × r} (A-matrix), B ∈ ℝ^{r × d_out} (B-matrix), r = 4 (rank).

In weight-matrix notation, the effective perturbation is:
```
ΔW = scale × B^T A^T   (d_out × d_in matrix)
```

**Frobenius inner product between two adapters:**
```
⟨ΔW_i, ΔW_j⟩_F = trace(ΔW_i^T ΔW_j)
                 = s² trace(A_i B_i B_j^T A_j^T)
                 = s² trace(B_j^T A_j^T A_i B_i)    [cyclic trace identity]
```

If A_i^T A_j = 0, then the entire expression is 0 **for any B_i, B_j**.

---

## C. Theorem 1: Grassmannian A-Slot Orthogonality Guarantee

**Theorem 1 (Parameter-Space Zero Interference).**
Let Q ∈ ℝ^{d × (N r)} be a matrix with orthonormal columns (d ≥ Nr). Define domain A-matrices as non-overlapping column slices:
```
A_i = Q[:, i·r : (i+1)·r]   for i = 0, …, N-1
```

Then for any i ≠ j:
```
A_i^T A_j = 0_{r×r}
```
and therefore, for any B-matrices {B_i} ∈ ℝ^{r × d_out}:
```
⟨ΔW_i, ΔW_j⟩_F = 0
```

**Proof.**
Since Q has orthonormal columns, Q^T Q = I_{Nr}.
Partition Q^T Q into r×r blocks: (Q^T Q)_{ij} = A_i^T A_j.
For i ≠ j, the (i,j)-th r×r off-diagonal block of I_{Nr} is 0_{r×r}.
Therefore A_i^T A_j = 0_{r×r}.

Substituting into the interference formula:
```
⟨ΔW_i, ΔW_j⟩_F = s² trace(B_j^T (A_j^T A_i) B_i)
                 = s² trace(B_j^T · 0_{r×r} · B_i)
                 = 0
```
This holds for **any** B_i, B_j ∈ ℝ^{r × d_out}. □

**Existence condition:** d ≥ N·r. For this experiment: d=64, N=5, r=4 → 64 ≥ 20. ✓

**Construction:** QR decomposition of a random (64, 20) matrix gives orthonormal Q. Column slices are the A-matrices.

---

## D. Theorem 2: M2P Quality Lower Bound (Guided Exploration)

**Setup:** M2P is a lightweight transformer (2 layers, 4 heads) that reads mean-pooled hidden states from each layer of the base model and outputs B-matrices. It is trained end-to-end on task loss.

**Claim (guided exploration — unknown constant to be discovered):**

There exists a constant c > 0 such that after T training steps, M2P achieves:
```
quality_ratio = (PPL_base - PPL_M2P) / (PPL_base - PPL_SFT) ≥ c
```
where quality_ratio measures how much of the SFT improvement M2P recovers.

**Prior evidence for c > 0:**
- Finding #339: c = 0.666 on Qwen3-4B with real domain data (medical)
- SHINE (arXiv:2602.06358): M2P achieves 50–80% of SFT quality with similar architecture

**Prediction for this experiment:**
- Synthetic domains (arithmetic, reverse, sort, etc.) have sharp pattern structure → M2P should learn faster than real-text domains
- Toy GPT is small (d=64) → M2P capacity (M=16 memory tokens × 64 dims) is proportionally adequate
- **Predicted range:** c ∈ [0.30, 0.70] after 500 M2P training steps

**Kill criterion:** K847 requires c ≥ 0.25 (25% of SFT quality).

---

## E. Dimension Check (Capacity Analysis)

**Memory capacity per domain:**
- M = 16 memory tokens, H = 64 dims → 16 × 64 = 1,024 values per domain
- Target B-matrices: 5 modules × 4 layers × (4 × 64) = 5,120 values per domain

The M2P memory tokens (1,024) are insufficient to directly encode all B values (5,120). This forces the M2P to learn **compressed representations** — it cannot memorize adapters directly but must learn a mapping from context to adapter.

**Key implication:** M2P must generalize, not memorize. If K847 passes, it demonstrates genuine context-to-adapter learning.

Note: The code uses a flat reshape to extract B values from memory (`flat = memory[li].reshape(-1)`), which only uses the first `rank × d_model` values per module. The actual usable capacity per layer is 16 × 64 / 5 ≈ 205 values vs. 4 × 64 = 256 target B values per module — a compression ratio of ~1.25×. M2P must share representations across modules.

---

## F. Kill Criteria and Predictions

| Criterion | Description | Predicted outcome | Threshold |
|-----------|-------------|-------------------|-----------|
| K847 | M2P quality ≥ 25% of SFT | c ∈ [0.30, 0.70] | ≥ 0.25 |
| K848 | Grassmannian A orthogonality | cos ≤ 1e-5 (float32 precision) | Structural guarantee |

**K847 reasoning:** Finding #339 achieved 66.6% with harder (real) data. Synthetic patterns are more compressible. Expect PASS.

**K848 reasoning:** Mathematical guarantee from Theorem 1. QR decomposition in float32 gives orthogonality up to ~1e-6. Will observe |cos| ≈ 1e-6 in the Grassmannian verification step.

---

## G. Prediction-vs-Measurement Table (to be filled after run)

| Metric | Predicted | Measured | Pass/Fail |
|--------|-----------|----------|-----------|
| Grassmannian \|cos\| | ≤ 1e-5 | **0.000000** | **PASS** |
| Mean M2P quality ratio | 0.30–0.70 | **11.5%** (outlier: repeat = -146.8%) | **FAIL** |
| K847 (quality ≥ 25%) | PASS | **FAIL** (dominated by repeat outlier) | — |
| K848 (Grassmannian orthogonality) | PASS | **PASS** | — |

---

## H. Connection to Vision

This experiment validates the **decoupled architecture** for M2P distillation:
- A-matrices: frozen, generated once, guarantee composition
- B-matrices: generated per-context by M2P, encode domain knowledge
- Scale: fixed at 2.0 (toy); will be learned via preservation loss in next experiment

If K847 passes, we have a proof-of-concept for the full pipeline:
context → M2P → B-matrices → compose with Grassmannian A → zero-interference multi-domain model.

The vision: adding domain N+1 costs one M2P forward pass (~1ms), not 300 SFT training steps.

---

## I. Self-Test

**Q1 (Impossibility property):** What structural feature would cause failure?
A: Round-robin training across domains with heterogeneous loss scales causes the M2P to converge to a B-matrix centroid (averaging conflicting gradient signals) rather than per-domain optima. A domain with high loss magnitude will dominate updates, collapsing B-matrices toward its optimum. Formally: without domain conditioning or loss normalization, the B-matrix gradient is `Σ_d g_d`, not per-domain, so the optima are never individually reachable.

**Q2 (Cited theorems):**
- QR decomposition orthonormality (standard linear algebra) — conditions met (d=64 ≥ N·r=20).
- SHINE (arXiv:2602.06358) — empirical paper establishing M2P viability; cited for motivation, not as a theorem.
- Finding #339 (exp_shine_session_adapter, SUPPORTED) — M2P achieves 66.6% of SFT quality in practice.

**Q3 (Predicted numbers):** K847 predicts quality ratio c ∈ [0.30, 0.70] per domain after 500 training steps. K848 predicts |cos(A_i, A_j)| ≤ 1e-5 (float32 precision from QR). Both are specific and falsifiable.

**Q4 (Falsification condition):** K847 < 0.25 (mean/median quality) kills. K848 > 1e-5 kills (structural guarantee failure). Present.

**Q5 (Hyperparameter count):** M2P_LR=1e-3, M2P_PRETRAIN_STEPS=500, N_MEMORY=16, scale=2.0, M2P_LAYERS=2, N_HEADS=4 — 6 hyperparameters. None are swept; all are fixed by analogy to exp_shine_session_adapter. This is a guided-exploration: the unknown is the achievable quality c, not hyperparameter sensitivity.

**Q6 (Hack check):** Single mechanism (Grassmannian A + M2P B). No stacked fixes. CLEAN.
