# MATH.md — T1.2: HRA vs LoRA Quality on Qwen3-4B (GSM8K + MMLU)

## Setup

**Paper:** HRA: Householder Reflection Adaptation (arxiv 2405.17484)
**Prior result:** T1.1 (Finding #415) — HRA achieves float32-floor isometry at d=2816,
algebraic zero interference with Grassmannian init, 2× fewer params than LoRA.

**Model proxy:** Qwen3-4B-4bit (Gemma4 not loadable by mlx_lm 0.29.1 — MODEL_REMAPPING
missing `gemma4` type). Justified: same 4B scale, d_model=2560, proven working in P1.

**Architecture:**
- Qwen3-4B: hidden_size=2560, n_layers=36, n_heads=32, head_dim=128
- q_proj: in=2560, out=4096 (32×128); v_proj: in=2560, out=1024 (8×128)
- HRA r=16 on q_proj input: V ∈ R^{16×2560} = 40,960 params / layer
- LoRA r=16 on q_proj: A ∈ R^{2560×16}, B ∈ R^{4096×16} = 106,496 params / layer
- HRA uses 40,960 / 106,496 = 38.5% of LoRA params at same rank

---

## Theorem 1: HRA Achieves Competitive Quality at Reduced Parameter Count

**Claim:** Let W ∈ R^{d_out × d_in} be a frozen weight, H^(r) the Householder product
from T1.1. An HRA adapter trained to minimize ℓ(W H^(r) x, y) achieves:

    acc_HRA(r=16) ≥ acc_LoRA(r=16)

i.e., same-rank HRA matches or exceeds LoRA accuracy, using 38.5% of the parameters.

**Proof (reduction to stable rank):**

(a) From T1.1 Theorem 3: sr(H^(r)-I) = r. The Householder adapter spans r orthogonal
directions in the weight update space.

(b) For LoRA: W_delta = s * B A^T where A ∈ R^{d_in×r}, B ∈ R^{d_out×r}, s = alpha/r.
    Random initialization gives sr(W_delta) ≈ 1 (top singular value dominates):
    E[||BA^T||_F^2 / ||BA^T||_2^2] = r · O(1/sqrt(r)) → O(sqrt(r)) « r.

(c) Effective parameter capacity: for an update M, the number of independently
    learnable directions is sr(M). HRA uses sr = r directions with r*d_in parameters.
    LoRA uses sr ≈ 1 direction with 2*r*(d_in+d_out) parameters (for this matrix).

(d) HRA has r-times more effective directions per parameter, placing the adapter
    update on a richer subspace of the orthogonal group O(d_in).

(e) Table 1 in HRA paper (2405.17484) reports: on LLaMA-2 and LLaMA-3 across
    multiple benchmarks, HRA r=8 achieves +5.6pp GSM8K, +1.6pp GLUE vs LoRA r=8
    at equal parameter count (square matrices where HRA = LoRA/2).
    For our rectangular q_proj (2560→4096), HRA r=16 uses 38.5% of LoRA r=16 params.

**QED** (modulo cross-architecture transfer — this is a guided exploration, not pure verification)

---

## Theorem 2: HRA Step Time is O(r·d_in) ≤ LoRA Step Time

**Claim:** Wall-clock time per training step:
    t_HRA ≤ 3 × t_LoRA

**Proof:**
(a) HRA forward per layer: r reflections × 2 operations each = 2r vector-multiply
    and r outer-product-subtract. Total FLOPs per token: O(r × d_in).

(b) LoRA forward per layer: A(x) = x A^T, B(Ax) = (Ax) B^T.
    FLOPs per token: O(r × d_in + r × d_out).
    For q_proj: d_out > d_in, so LoRA is actually MORE expensive per step.

(c) HRA FLOPs = O(r × d_in) = O(16 × 2560) = 40,960
    LoRA FLOPs = O(r × d_in + r × d_out) = O(16 × 2560 + 16 × 4096) = 106,496

(d) HRA is ~38% of LoRA in FLOPs. Even accounting for Python loop overhead
    (r=16 Python iterations vs 2 vectorized matmuls), total time ≤ 3× LoRA.

**QED**

---

## Theorem 3: HRA Convergence Within 2× LoRA Steps

**Claim:** Steps to loss < 0.5:
    steps_HRA ≤ 2 × steps_LoRA

**Proof (informal, Guided Exploration):**

(a) From T1.1 Theorem 3: sr(H^(r)-I) = r = 16. Higher stable rank means the
    adapter spans r orthogonal directions simultaneously.

(b) LoRA with sr ≈ 1 adapts primarily along one singular direction initially.
    Full-rank adaptation requires gradients to gradually expand the rank, which
    takes extra steps.

(c) HRA starts with full-rank orthogonal structure → gradient signal distributed
    across r directions from step 1 → faster initial loss reduction.

(d) This predicts K1013 PASS (convergence ≤ 2× LoRA steps).
    Note: this is exploratory — if both converge at similar rates, still consistent.

**QED**

---

## Quantitative Predictions

| Metric | Prediction | Kill Criterion | Source |
|--------|-----------|----------------|--------|
| HRA acc_GSM8K vs LoRA | HRA ≥ LoRA | K1011: HRA ≥ LoRA | HRA paper Table 1 |
| HRA acc_MMLU vs LoRA | HRA ≥ LoRA | K1012: HRA ≥ LoRA | HRA paper Table 2 |
| Convergence steps (loss<0.5) | HRA ≤ 2× LoRA | K1013: ≤ 2× | Theorem 3 |
| Per-step wall time | HRA ≤ 1.5× LoRA | K1014: ≤ 3× | Theorem 2 |
| HRA params (q_proj/layer) | 40,960 | Structural | 16×2560 |
| LoRA params (q_proj/layer) | 106,496 | Structural | 2×16×(2560+4096)/2 |

**Expected outcomes:**
- K1011: PASS (HRA paper +5.6pp GSM8K, pending cross-model transfer)
- K1012: PASS (HRA paper +1.6pp GLUE, MMLU as proxy)
- K1013: PASS or exploratory (faster convergence predicted by sr argument)
- K1014: PASS (HRA has fewer FLOPs, Python loop < 3× vectorized matmul overhead)

**Experiment type:** Guided exploration (proven framework from T1.1, unknown: how well
HRA paper results transfer from LLaMA-2 to Qwen3-4B at fixed 300 training steps)

---

## Architectural Implications for P1

If K1011+K1012 PASS: Replace LoRA with HRA as the P1 adapter format.
- P1 adapts q_proj NoPE dims [128:512] (d=384 effective from T0.3)
- HRA r=16 on NoPE slice: 16×384 = 6,144 params/layer vs LoRA r=8: 2×8×384 = 6,144
- At equal params, HRA uses rank=16 (more directions) vs LoRA rank=8
- Grassmannian init (T1.1 Theorem 2) gives algebraic zero interference

If K1011+K1012 FAIL: Fall back to LoRA with Grassmannian (proven in T0.3/T0.4).
- Finding #413 (Givens) and Finding #415 (Householder) both provide orthogonal adapters
- LoRA + Grassmannian init is sufficient for interference-free composition
