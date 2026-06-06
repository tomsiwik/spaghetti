# MATH.md: Cross-Attention Context Conditioning for M2P

**Experiment ID:** `exp_followup_m2p_cross_attention_conditioning`
**Type:** Verification (architectural ablation of the context-injection mechanism in M2P).
**Motivating kill:** `exp_m2p_scale_calibrated` — KILLED on K850 (adapter magnitude CV = 0.0093, predicted > 0.05). Closure C1 tag: `additive-context-injection-blocks-calibration`.
**Scope:** Single domain (arithmetic), toy GPT-2. Isolates ONLY the M2P context-injection path. All downstream loss (L_task + λ·L_preserve), the base GPT, the frozen Grassmannian A-matrices, and the unpacking head are reused unchanged.

## A. Failure mode identified

In the killed sibling `exp_m2p_scale_calibrated`, M2P conditioned on a task context `c ∈ ℝ^{T×d}` by

```
task_ctx = mean(task_embed(c), axis=1, keepdims=True)   # (1, 1, d)
mem      = mem_init + broadcast(task_ctx)               # (1, N, d)
```

i.e. the full context was pooled to **one vector** and added identically to every memory token before self-attention.

**Claim (Lemma 1, proved below).** Under pure additive mean-pool conditioning, the map from context to post-transformer memory has rank ≤ 1 in the context direction: the Jacobian `∂mem/∂c` factors as `(1_N · w^T) ⊗ (∂task_ctx/∂c)`, where `1_N` is the all-ones vector over memory positions. Per-memory-token variation across contexts is therefore collinear with a single scalar gain. The unpacking head `B_proj ∈ ℝ^{N·d → ΣB_params}` applied to `mem.reshape(1, -1)` inherits this rank-1 restriction: `∂‖B‖_F / ∂c` is a single scalar direction, so `Var_c(‖B‖_F) ≤ ε²·‖∂task_ctx/∂c‖²` with ε the operator norm of the post-transform. Empirically the measured CV of 0.0093 is consistent with this bound: noise dominates, and context sensitivity is lost.

*Proof of Lemma 1 (sketch).*
Let `e(c) = mean(task_embed(c), axis=1)`. Then `mem⁰ = mem_init + 1_N ⊗ e(c)`. Self-attention is permutation-equivariant and acts identically on every slot; an input that is constant across slots plus a fixed offset contributes to every slot's query/key/value through the same affine map. Differentiating each output slot `mem_k^L` w.r.t. `c` and stacking over `k`, the column pattern of `∂mem^L/∂e` is `1_N · ξ(e)^T` for some vector `ξ(e) ∈ ℝ^d` (equivariance forces identical per-row response to a slot-uniform perturbation). Composing with `∂e/∂c` gives the rank-1 factorization. QED.

**Why mean-pool is the right target.** This is a clean architectural failure, not a training-regime failure: the Jacobian bound applies at every point in parameter space, so no amount of L_preserve tuning, lr sweep, or seed variation recovers per-token contextual richness. Closure rule `additive-context-injection-blocks-calibration`.

## B. Reframe — the right question

**Wrong question:** "can we regularise M2P into varying its output?"
**Right question:** "what is the minimum architectural change that breaks Lemma 1's rank-1 Jacobian, so that the K850 measurement becomes a real test of Theorem 1 Step 5?"

**Answer:** give each memory token its own **query** into the context tokens, so every memory slot receives a different linear combination of context positions. This is the standard cross-attention primitive from Vaswani et al. 2017, already the mechanism used by Perceiver (Jaegle et al. 2021, `arxiv:2103.03206`) for conditioning a fixed set of latents on a variable-length input.

## C. Prior mathematical foundations

### C.1 Cross-attention as a rank-increasing map (Bahdanau et al. 2014, Vaswani et al. 2017)

Given queries `Q ∈ ℝ^{N×d_k}`, keys `K(c) ∈ ℝ^{T×d_k}`, values `V(c) ∈ ℝ^{T×d_v}`:

```
CrossAttn(Q; c) = softmax(Q K(c)^T / √d_k) · V(c)
```

The output `mem_k = Σ_t α_{k,t}(c) · V(c)_t` with `α_{k,t}(c) = softmax_t(Q_k · K_t / √d_k)`.

### C.2 Generic rank of the cross-attention Jacobian

**Lemma 2.** Let `f: ℝ^{T×d} → ℝ^{N×d_v}`, `f(c) = CrossAttn(Q; c)`. Assume (i) `Q` has rank `min(N, d_k)` (generic weights), (ii) the context embedding `V` is not a constant function of `c`. Then `rank(∂f/∂c) ≥ min(N, T, d_k)` generically.

*Proof.* `∂mem_k/∂V_t = α_{k,t} · I_{d_v}`. The `α`-pattern matrix `[α_{k,t}]_{N×T}` has rank `min(N, T)` almost everywhere in Q-parameter space (softmax of a generic rank-`min(N, T, d_k)` matrix). Therefore `∂f/∂V` has rank at least `min(N, T) · d_v` restricted to the subspace of non-degenerate `V` perturbations, and composing with the linear embedding `V(c)` gives rank `min(N, T, d_k)` for `∂f/∂c` generically. QED.

With the cross-attention replacement `N=8`, `T=48`, `d_k=16` (4 heads × 16) on our toy scale, the Jacobian rank bound is `≥ 8`. That is strictly greater than the `= 1` bound of mean-pool conditioning, and is sufficient for per-context variation to propagate through the unpacking head `B_proj`.

### C.3 Self-calibration restated (inheritance from parent MATH)

Theorem 1 of `exp_m2p_scale_calibrated` Step 5 predicts `||B(c)||_F` varies monotonically with measured task-gradient difficulty, giving CV > 0.05 across a mixed easy/hard suite of 20 contexts. The proof applies unchanged **once the rank-1 bottleneck is removed**: the mathematics of KKT equilibrium assumes M2P can represent context-dependent outputs (Assumption 4 in the sibling PAPER); Lemma 1 above formalises how the previous architecture violated that assumption, and Lemma 2 formalises how cross-attention restores it.

## D. Architecture — single atomic change

| Component | Killed sibling | This experiment |
|---|---|---|
| Context embedding | `task_embed(c)` | `task_embed(c)` *(unchanged)* |
| Pooling | `mean(…, axis=1)` then broadcast-add to `mem` | — *(removed)* |
| Context-to-memory | additive | **cross-attention: `mem = mem_init + CrossAttn(Q=mem_init, K=ctx, V=ctx)`** |
| Self-attention blocks | M2PBlock × 2 | M2PBlock × 2 *(unchanged)* |
| Final norm + `B_proj` | unchanged | unchanged |

Cross-attention is a standard MultiHead block with query = `mem_init`, key/value = `task_embed(c)`. 4 heads, `d = D_M2P = 64`, head-dim = 16. Total added params: `4 * D_M2P * D_M2P = 16384` (Q, K, V, O projections). Negligible vs existing 8.4M `B_proj`.

**Every other hyperparameter is inherited verbatim** from the killed sibling: `D_MODEL=256`, `N_LAYERS=2`, `LORA_RANK=4`, `N_MEMORY=8`, `M2P_LAYERS=2`, `LAMBDA_PRESERVE=0.1`, `N_CONTEXT_VARIANTS=20`, `M2P_STEPS=600`, `BATCH_SIZE=8`, `LR=3e-4`, `SEED=42`. This is critical: it isolates the single architectural change as the treatment.

## E. Predictions

Let CV = std(‖B(c)‖_F) / mean(‖B(c)‖_F) across 20 contexts (10 easy, 10 hard).

| # | Prediction | Derivation |
|---|---|---|
| P1 | `CV_cross_attn > 0.05` on the same data + hparams that gave `CV_mean_pool = 0.0093` | Lemma 2 rank-8 vs Lemma 1 rank-1 bound |
| P2 | `CV_cross_attn > 3 × CV_mean_pool` | ratio of Jacobian ranks, 8 / 1; generous factor-3 floor accounts for softmax saturation |
| P3 | `CV_mean_pool_control ≤ 0.02` in this run (reproducing Finding #343 baseline within noise) | direct replication of killed sibling measurement under identical seed & data |
| P4 | `||B(c)||_F` under cross-attention retains monotonic hard > easy ordering (Theorem 1 Step 5) at ratio ≥ 1.10 | Lemma 2 permits this; not guaranteed but a finer behavioral prediction |
| P5 | General quality degradation under cross-attn is within ±10pp of mean-pool baseline (K849-style sanity check) | architecture change is rank-increasing at the conditioning path only; does not move the KKT operating point of `L_total` |

## F. Pre-registered kill criteria

KC #1556 in DB. Concretely:

- **K1556a (headline):** `CV_cross_attn > 0.05` across the 20-context eval suite. Failure ⇒ cross-attention did NOT resolve the mean-pool bottleneck.
- **K1556b (control):** `CV_mean_pool_baseline_this_run ≤ 0.02`. If this fails we cannot attribute any CV gain to the architecture change (the measurement is noisy under this seed). The run is then `provisional` and must be reclassified.
- **K1556c (relative):** `CV_cross_attn ≥ 3 × CV_mean_pool_baseline_this_run`. Guards against the pathological case where both conditions drift simultaneously.
- **Headline verdict = SUPPORTED iff all three pass.** Any single failure ⇒ KILLED.

**No KC edits after data comes in.** These three are frozen at the current git commit of `MATH.md`.

## G. What a negative result would mean

If K1556a fails despite K1556b passing: cross-attention is not the rank-increasing primitive we needed, or the unpacking head `B_proj` itself has a rank-1 bottleneck (its weight matrix only depends on the single `flat = mem.reshape(1, -1)` vector; if cross-attention makes `mem` vary meaningfully but the post-block self-attention pools it back, the fix is necessary-but-not-sufficient). Either finding is a concrete architectural result for the next follow-up, not a null.

If K1556a passes but K1556b also passes (CV_mean_pool > 0.02), we've drifted off the killed-sibling's measurement regime; the run is `provisional`, no KC is retroactively edited, and a v2 with tighter data reproduction is required.

## H. Assumptions and scope

- Toy-scale proxy: `D_MODEL=256`, not Gemma 4 scale. This is deliberate — matches the sibling kill so the comparison is apples-to-apples. The sibling's closure C2 and C3 (KKT direction of gradient, L_preserve rigidity) are independent of the mean-pool bottleneck and are **not** tested here. This experiment cleanly isolates the architectural sub-claim.
- Single domain. Multi-domain is explicitly out of scope (Finding #341 gradient conflicts).
- `mlx` version: pinned by the repo `uv` environment. `mlx-lm` is not used (no HF model load).
