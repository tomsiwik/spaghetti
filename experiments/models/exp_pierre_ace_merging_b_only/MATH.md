# MATH.md — ACE-Merging adapted to Pierre's shared-A B-only architecture

## Hypothesis

ACE-Merging (arxiv 2603.02945) provides a **closed-form covariance-weighted
merge** that is a principled generalization of mean averaging:

$$\bar W = \left(\sum_t \tilde W_t \cdot \hat\Sigma_{t,\text{reg}}\right) \cdot \left(\sum_t \hat\Sigma_{t,\text{reg}} + C_{\text{agg}}\right)^{-1}$$

where $\hat\Sigma_t \propto \tilde W_t^\top \tilde W_t$ is the per-task input
covariance inferred directly from the centered task vector — **no calibration
data needed** (Theorem 1 in the paper).

Why ACE is the strongest candidate of the surveyed methods:

1. **Closed-form math** — no iterative optimization, no learned weights.
2. **Working public code** ([unravel-xu/ACE-Merging](https://github.com/unravel-xu/ACE-Merging)) — implementation patterns verified, not just paper pseudocode.
3. **Strict generalization of Fisher-Rao** — at $\hat\Sigma_t = c \cdot I$ (isotropic) and $C_{\text{agg}} = c'\cdot I$, the formula reduces to a scalar-rescaled mean, recovering Fisher-Rao behavior.
4. **Adaptive heterogeneity branch** — when task vectors have widely varying magnitudes, ACE applies an additional spectral refinement (top-k singular value isotropization). Pierre's 7-adapter mix (4 strategy + 3 domain) is heterogeneous, so this branch may engage.

> **Does ACE-Merging close the +6.7pp gap measured between Fisher-Rao
> (64.7%) and full-delta DARE (71.3%) in `exp_pierre_dare_b_vs_fisher_rao`?**

## Adaptation to Pierre's shared-A architecture

ACE operates on full $\Delta W$. Pierre stores adapters as B-only with shared frozen A. The adaptation:

1. **At storage time**: still B-only (compatible with Pierre's existing loading).
2. **At composition time**: materialize per-adapter $\Delta W_t = \text{scale} \cdot A \cdot B_t$ transiently (~117MB per layer at d=2048).
3. **Run ACE merge** on the materialized $\Delta W_t$'s to produce a single fused $\Delta W_{\text{fused}}$.
4. **Apply** via `_FusedDeltaLinear` wrapper (Finding #831 canonical pattern) instead of `PoLARLinear`.

**Net effect**: B-only on disk; full-delta during a one-time merge step at adapter load; fused-delta at inference. The Grassmannian sibling theorem still holds at training time — composition just operates in a different space.

## Algorithm (verbatim from released code, see `compose_methods.py`)

For each (layer, module) key:

1. Materialize $\Delta W_t = s_t \cdot A \cdot B_t$ for each of K adapters.
2. Compute heterogeneity flag $\gamma$:
   - $\gamma = \text{Var}_t[\log \|\Delta W_t\|^2] / \mathbb{E}_t[\log \|\Delta W_t\|^2]^2$
   - flag = $\gamma > \tau$ where $\tau = 0.3$
3. Per-task centered task vector and covariance:
   - $\tilde W_t = \Delta W_t - \text{col\_mean}(\Delta W_t)$
   - $\hat\Sigma_t = \tilde W_t^\top \tilde W_t$ (shape $d_{\text{out}} \times d_{\text{out}}$)
   - If flag: $\hat\Sigma_t \leftarrow \hat\Sigma_t / \text{tr}(\hat\Sigma_t)$, $\varepsilon_t = \varepsilon / \text{tr}$
   - Else: keep raw, $\varepsilon_t = \varepsilon$
   - $\hat\Sigma_{t,\text{reg}} = \hat\Sigma_t + \varepsilon_t I$
4. Closed-form merge:
   - $N = \sum_t \tilde W_t \cdot \hat\Sigma_{t,\text{reg}}$
   - $D = \sum_t \hat\Sigma_{t,\text{reg}} + C_{\text{agg}}$ (with $C_{\text{agg}} = \varepsilon I$ per released code)
   - $\bar W = N \cdot D^{-1}$
5. Restore mean column contribution: $\bar W \mathrel{+}= \text{col\_mean}(\Delta W)_t$ averaged over t.
6. If flag (heterogeneity): apply spectral refinement
   - $U \Sigma V^\top = \text{SVD}(\bar W)$
   - $\sigma_{\text{iso}} = \text{mean}(\Sigma_{1..k})$ where $k = \lfloor 0.3 \cdot \min(d_{\text{in}}, d_{\text{out}})\rfloor$
   - $\bar W \mathrel{+}= \sigma_{\text{iso}} \cdot U_{:,:k} V_{:k,:}^\top$

## Pre-registered Kill Criteria

- **K1 (DECISION)** ACE avg ≥ Fisher-Rao avg + 3pp.
- **K2 (ARCH GAP CLOSURE)** Full-delta DARE avg − ACE avg ≤ 4pp.
- **K3 (PREPROCESS BUDGET)** ACE preprocessing time ≤ 30s total (heavier than Pico because of d_out × d_out covariance and matrix inversions per layer).
- **K4 (SANITY)** Verdict gating only — relies on K1 + K2 + K3 directly.

## Verdict logic

| K1 | K2 | K3 | Outcome |
|----|----|----|---------|
| ✓ | ✓ | ✓ | **SUPPORTED** — Pierre default becomes ACE merge. Strongest candidate confirmed. |
| ✓ | ✓ | ✗ | **SUPPORTED** with caveat — adopt for offline/RAG use; flag inference-time slowdown. |
| ✓ | ✗ | * | **SUPPORTED** — adopt; shared-A still leaves headroom (consider per-adapter A future work). |
| ✗ | * | * | **KILLED** — ACE does not transfer; revisit Pico/OrthoMerge. |

## Eval protocol

Same as `exp_pierre_dare_b_vs_fisher_rao`:
- N=50 per benchmark, fixed seed=42
- Benchmarks: GSM8K, HumanEval, MedQA via `scripts/polar_train.py::eval_*`
- Adapters: same 7 PoLAR (4 strategy + 3 domain)
- Base: `mlx-community/gemma-4-e4b-it-4bit`
- Shared-A donor: `strategy_full`

## Honest gaps & implementation choices

- **No native LoRA path** in ACE's released code — the adaptation here treats Pierre's shared-A architecture as full-delta during merge, which is the natural extension. Released code targets ViT/GPT-2/BERT.
- **`C_agg` simplification**: released code uses `eps · I`; paper's formula is `1 · (mean column energy of Σ_{t,scaled})^T` — a rank-1 broadcast. We follow the released code's simplification for tractability; flagging as a port choice.
- **Eps default**: code uses `eps=1e-2`; paper uses `eps ∈ {1e-5, 2e-4, 4e-2}` per architecture. We use `1e-2` and may tune if K1 marginal.
- **Inverse via `mx.linalg.inv` on CPU**: numerically safer than GPU on Apple Silicon for small (d_out=2048) matrices. Cost ~50ms per layer → ~2.1s total for 42 layers.
- **Spectral refinement applied per-layer**: SVD on (d_in, d_out) = (2560, 2048) is ~200ms per layer on CPU. Total ~8s if heterogeneity branch engages. Within K3 budget.

## References

- ACE-Merging paper (arxiv 2603.02945): https://arxiv.org/abs/2603.02945
- Reference implementation: https://github.com/unravel-xu/ACE-Merging (`src/merge/strategy.py`)
- Implementation spec: research agent `acbe00274a1a6eb9c` (verified against released code)
- Prior measurement: `exp_pierre_dare_b_vs_fisher_rao` (Fisher-Rao 64.7%, full-delta DARE 71.3%)
- Finding #831 canonical `_FusedDeltaLinear` pattern (used here for fused-delta install)
