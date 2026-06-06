# MATH.md — Room Model on Gemma 4 E4B (speed + equivalence + exact reversibility)

## 0. Scope and stance

This experiment re-tests the **Room Model** (W_combined = Σ_i ΔW_i, pre-summed delta
injected as a single matrix) on:

- **New hardware**: Apple M5 Pro 48GB (273 GB/s bandwidth, unified memory).
- **New base**: `mlx-community/gemma-4-e4b-it-4bit` (MoE, ~4B active, 4-bit quantized).

Prior art has **killed** this hypothesis on Qwen-class bases (Findings #302, #303,
#315; `micro/models/room_model_poc/`, `micro/models/room_model_wcombined/`):

- Full-model logit MSE = 53.9 with per-module MSE = 5.6e-7 → **LayerNorm/attention/SiLU
  nonlinearity compounds ΔW_i through L=30 layers**; equivalence breaks globally.
- Dense W_combined cost: 4.17 GB vs 18 MB factored → 2.3× speed penalty (41.9 tok/s).
- Theoretical ceiling (Zhong et al., 2504.10957, ICLR 2025): task arithmetic is
  effective only when tasks are *irrelevant or aligned*; N=5 full-α deltas violate
  this by construction.

**Why run it again anyway?** The prior kill predictions are derived at a regime (dense
float LoRA, Qwen3 base, older hardware). Porting to Gemma 4 E4B + M5 Pro provides
*quantified* replication on the target platform before we permanently close the door
for the v8 design (PLAN.md Part 2). A replication that quantifies the gap — not a
hyperparameter tweak — is the only honest next step.

We pre-register **kill-expected** predictions consistent with prior theorems.
This is an honest verification run, not a hopeful resurrection attempt.

## 1. Setup and notation

Let the base model have linear layers `v_proj_ℓ, o_proj_ℓ` for ℓ ∈ {0…L−1} with
weight `W_ℓ ∈ ℝ^{d_out × d_in}` (d = 2560 for E4B).

For each adapter i ∈ {1…N}, N = 5, we build a PoLAR-r=6 rank-r decomposition:

  A_i ∈ ℝ^{r × d_in},   B_i ∈ ℝ^{d_out × r},   α_i = 1   (no LORA_SCALE inflation).

The A_i are drawn on the Grassmannian so A_i A_j^T ≈ 0 for i ≠ j (Finding #126):
construct as Q_i from stacked-and-sliced QR of a random [N·r × d_in] matrix.
B_i ∼ 𝒩(0, σ² I) with σ = 0.02 (no training — identity of results does not require
training; see §4).

Per-layer delta:

  ΔW_{ℓ,i} = B_{ℓ,i} · A_{ℓ,i}         (d_out × d_in, rank ≤ r)

Pre-summed room matrix:

  W^room_ℓ = Σ_{i=1}^{N} ΔW_{ℓ,i}       (one dense matrix per layer)

Room injection applies `y = W_ℓ x + W^room_ℓ x` at every layer ℓ.

## 2. Theorems (pre-registered)

### Theorem 1 — Per-module linearity (expected PASS)

For any x ∈ ℝ^{d_in},

  W^room x = (Σ_i B_i A_i) x = Σ_i B_i (A_i x)

by distributivity of matmul over addition. This is pure linear algebra and holds up
to bfloat16 roundoff (≤ 2^{-7} · ‖W^room‖_F ≈ 1e-3 absolute per element). Measuring
this per-module gives MSE ≪ 1e-5 in bf16. Measured prior: 5.6e-7 on Qwen3.

### Theorem 2 — Global equivalence breaks through nonlinearity (expected FAIL)

LayerNorm applied at layer ℓ+1 re-normalizes the hidden state h_{ℓ+1}. Write
h_{ℓ+1}^{room} = f(W_ℓ x + Σ_i ΔW_{ℓ,i} x) where f is LayerNorm + attention softmax.
Taylor-expand around h^{base}_{ℓ+1}:

  h_{ℓ+1}^{room} − h_{ℓ+1}^{base} = Σ_i (∂f / ∂W_ℓ) · ΔW_{ℓ,i} x + 𝒪(‖Σ_i ΔW_i x‖²)

Even if the first-order term decomposes across i, the **cross-term** Σ_{i≠j} ΔW_i ·
ΔW_j · ∂²f ≠ 0 in general. Zhong et al. 2504.10957 prove this cross-interference is
necessary — it vanishes only for *irrelevant or aligned* tasks. N=5 random Grassmannian
walls do *not* satisfy that; hence **logit-cosine(W^room, explicit routing) ≪ 0.999**.

Prior measurement (room_model_poc): full-model logit MSE = 53.9 with N=5 trained
adapters on Qwen3-0.6B. We predict comparable logit divergence on Gemma 4.

### Theorem 3 — Exact reversibility by bit subtraction (expected PASS)

W^room_{with k} − ΔW_k = Σ_{i≠k} ΔW_i = W^room_{without k} (linearity; deterministic
up to floating-point order of summation). If we always add/remove in the same
associative order (e.g. left-to-right), the operation is **bitwise exact** in bf16
provided no non-associative reduction is used.

Measured quantity: max |W^room_add_remove − W^room_fresh| over all layers. KC1690
accepts ≤ 1 ULP bf16 = 2^{-7} · max|W^room| ≈ 1e-3. Prior expectation: PASS.

## 3. Speed model (KC1688)

Bandwidth accounting per token at batch=1, ℓ=30 layers, d_out=d_in=2560 (v_proj),
d_in=2560, d_out=2048 (o_proj intermediate) — per-module bytes for W^room in bf16:

  mem_v = 2 · 2560² = 13.1 MB
  mem_o = 2 · 2560·2048 = 10.5 MB
  per-layer = mem_v + mem_o = 23.6 MB
  all layers: 30 · 23.6 = 708 MB dense W^room

Plus base model activations/KV cache (~1.2 GB at short context) → token-step
bandwidth ≈ 2 GB. At 273 GB/s peak bandwidth: ceiling ≈ 136 tok/s.

**Prediction (pre-registered): KC1688 (≥ 150 tok/s) FAILS** — bandwidth math forbids
≥150 even in the ideal case; expected measurement in [50, 120] tok/s.

Factored path for comparison (no W^room, run each adapter in parallel with h @ A @ B):
per-layer adapter bandwidth 2·(r·d_in + d_out·r)·N = 2·(6·2560 + 2560·6)·5 = 307 KB
→ negligible. Factored ceiling matches base model bandwidth.

## 4. Why no training is required

All three KCs test *mathematical identities* of the injection mechanism:

- KC1688: speed is a function of the matrix size and tensor operations, not of the
  values in B. Random B at the correct dtype/shape matches trained performance.
- KC1689: logit divergence is driven by cross-terms ΔW_i · ΔW_j · (nonlinear second
  derivative), which scale with ‖ΔW_i‖ · ‖ΔW_j‖ · ‖h‖². Random B at σ=0.02 produces
  comparable ‖ΔW‖ to trained adapters (typical trained σ: 0.01–0.05).
- KC1690: exact reversibility is a statement about the arithmetic of addition/
  subtraction in bf16, independent of the values.

Using random adapters sidesteps the compounding C1 blocker (missing trained adapters,
mem-antipattern-017, 9 instances on 2026-04-18) **without altering the test of the
claim**.

## 5. Kill criteria (pre-registered, LOCKED — do not edit after data)

- **KC1688**: W^room pre-sum of N=5 adapters on Gemma 4 E4B achieves ≥ 150 tok/s on
  M5 Pro. *Expected: FAIL by bandwidth theorem (§3).*
- **KC1689**: Logit cosine(W^room output, explicit-routing output) > 0.999.
  *Expected: FAIL by Theorem 2 (LayerNorm cross-terms, Zhong et al. 2504.10957).*
- **KC1690**: Add/remove via W^room += / −= ΔW_k yields bitwise-exact result vs
  freshly-computed W^room_{without k}. *Expected: PASS by Theorem 3.*

"Explicit-routing output" in KC1689 is defined as: for each token, apply **only** the
single adapter assigned to that token (ground-truth domain label). This is the
strongest version of routing (Finding #312 shows top-1 routing is better than soft).
Measuring cosine against this gives the smallest possible divergence — if W^room
still diverges, no softer routing benchmark can save it.

## 6. Antipattern self-audit (pre-run)

- **composition math (mem-antipattern-001)**: code computes ΔW_i = B_i @ A_i
  (d_out × d_in), then sums. Never sums B or A independently. ✓
- **tautological routing (mem-antipattern-002)**: routing assignment comes from a
  held-out label set, not from val[d][0]. ✓
- **LORA_SCALE inflation (mem-antipattern-003)**: α_i = 1.0. Measured σ=0.02 on B. ✓
- **smoke-as-supported (mem-antipattern-015)**: is_smoke=false; N=5 is the target; not
  a miniature run. ✓
- **preflight-adapter-persistence (mem-antipattern-017)**: no safetensors loaded;
  adapters are random init in-process. ✓
- **thinking-mode truncation (mem-antipattern-008)**: no generation sampling is scored
  for quality; only logit-cosine + tok/s. Thinking mode irrelevant. ✓
- **proxy-model substitution (mem-antipattern-014)**: the target claim is about Gemma
  4 E4B; we measure on Gemma 4 E4B. ✓

## 7. References

- Zhong et al. 2504.10957, ICLR 2025 — theoretical impossibility of task arithmetic
  outside irrelevant/aligned regime.
- Ilharco et al. 2212.04089 — empirical degradation of task arithmetic at scale.
- Finding #126 — Grassmannian initialization gives A_i A_j^T ≈ 0.
- Finding #302, #303, #315 — prior kills of Room Model on Qwen3.
- Finding #300 — bandwidth bottleneck on Apple Silicon.
- S-LoRA 2311.03285, Punica 2310.18547 — factored serving > dense.
- PLAN.md Part 2 — target platform + base model.
