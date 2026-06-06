# MATH — `exp_jepa_adapter_residual_stream`

**Claim.** A residual-stream next-embedding predictor (JEPA-style adapter, rank-16 on `v_proj+o_proj`) trained with SIGReg anti-collapse regularization matches or beats token-space r=16 LoRA on GSM8K-Hard at matched parameter budget on Gemma 4 E4B MLX 4-bit.

**Type.** Frontier-extension — LeWorldModel (Maes/LeCun/Balestriero 2026-03-24, arxiv:2603.19312) applied SIGReg to stabilize end-to-end JEPA for pixel world-models. We extend to residual-stream prediction on a frozen LLM with a trainable rank-16 adapter.

**Platform.** Apple M5 Pro 48GB, MLX (mlx-core, mlx-lm 0.31.x). Base model `mlx-community/gemma-4-e4b-it-4bit`. Adapter targets `v_proj + o_proj` per F#627.

---

## 1. Failure mode

Collapse: the adapter learns a constant prediction `h_hat = c` (or a low-rank subspace), minimizing MSE loss without capturing residual-stream dynamics. Symptoms:
- Low prediction variance across inputs.
- Downstream GSM8K-Hard accuracy at or below token-space baseline.
- Prediction loss `L_pred(step 500) / L_pred(step 50) ≈ 1.0` (not learning).

Without an explicit anti-collapse term, joint-embedding predictive architectures collapse to the trivial constant solution (LeCun 2022; LeJEPA arxiv:2511.08544 §2).

## 2. Cited prior math

- **LeWorldModel** (arxiv:2603.19312) — SIGReg (Stein Isotropic Gaussian Regularizer, Epps-Pulley formulation) reduces 6+ VICReg/EMA/stop-grad hyperparameters to a single `λ` with explicit anti-collapse guarantee.
- **SIGReg Epps-Pulley test** (LeJEPA arxiv:2511.08544, Epps & Pulley 1983) — Tests `Z ~ N(0, I)` by comparing empirical characteristic function to Gaussian characteristic function over M random projections.
- **Finding #627** — Gemma 4 E4B with LoRA on `v_proj + o_proj` at r=6 achieves target behavioral improvement; this is the proven target for this architecture.
- **Finding #666** — Proxy KCs must be paired with a target-metric KC.

## 3. Mechanism (atomic)

### 3.1 Architecture

Let `h_ℓ ∈ R^d` be the residual stream at layer ℓ (d=2304 for Gemma 4 E4B). The JEPA adapter:

1. Inserts rank-16 LoRA on `v_proj + o_proj` at ALL 42 layers (standard mlx-lm config).
2. At training time, a small **prediction head** `P_θ: R^d → R^d` (2-layer MLP, hidden_dim=d) maps `h_ℓ(token_t)` → `h_hat_{ℓ}(token_{t+1})`.
3. Loss:

   ```
   L_total = L_pred + λ · L_SIGReg
   L_pred  = (1/|B|) Σ_t || P_θ(h_ℓ(t)) - stopgrad(h_ℓ(t+1)) ||²
   L_SIGReg = (1/M) Σ_{u_m ~ S^{d-1}}  D_EP( u_m^T Z, N(0,1) )
   ```

   where `Z = P_θ(h_ℓ(t))` is the prediction batch, `M=1024` random unit projections, and `D_EP` is the Epps-Pulley statistic per LeJEPA Eq. 7:

   ```
   D_EP(z, N(0,1)) = ∫ |ψ_z(t) - exp(-t²/2)|² · w(t) dt
   ```

   with Gaussian kernel weight `w(t) = (2π)^{-1/2} exp(-t²/2)` and `ψ_z(t) = (1/N) Σ_n exp(i·t·z_n)` the empirical characteristic function.

4. **stopgrad** on targets `h_ℓ(t+1)` prevents the trivial bypass through the base model (standard JEPA pattern).

### 3.2 Why rank-16 is matched

Token-space LoRA baseline: r=16 on `v_proj + o_proj`. Parameter count `P_baseline = 2 · 42 · 2 · 16 · d = 12.4M` (for d=2304, two matrices A/B per projection, two projections per layer, 42 layers).

JEPA adapter: rank-16 on `v_proj + o_proj` + prediction head `P_θ` with hidden=d. MLP head is 2·d·d = 10.6M. We **subtract** head params from comparison by freezing the head after warm-up (see §6 — the claim is that the residual-stream objective *during training* transfers knowledge into the adapter).

Alternative (clean): report adapter-only params and note head overhead. KC K3 target compares GSM8K-Hard at **matched adapter params** (12.4M).

### 3.3 SIGReg collapse prevention

**Theorem (LeJEPA Thm 1, informal).** If `Z ∈ R^d` is isotropic Gaussian, then SIGReg is minimized. Conversely, if SIGReg is near zero, then `Z` is approximately isotropic Gaussian — rank-full, non-collapsed.

**Proof sketch.** Cramér-Wold: a distribution is characterized by its projections. Epps-Pulley consistently estimates divergence from N(0,I) as M → ∞. Thus minimizing SIGReg forces `Z` to approach isotropic Gaussian structure, geometrically ruling out collapse to a point or low-rank subspace.

**Hyperparameter elimination.** VICReg (Bardes 2022) requires tuning 3 weights (variance/invariance/covariance) + batch size. SIGReg requires tuning only `λ` (found by bisection per LeWM §4.2).

## 4. Predictions

| # | Prediction | Mechanism | Falsifier |
|---|---|---|---|
| P1 | At step 500, Epps-Pulley rejection rate on adapter-output activations `Z` < 5% | SIGReg forces isotropic output; no collapse | Rejection ≥ 5% → collapse detected → adapter is degenerate |
| P2 | `L_pred(step 500) / L_pred(step 50) < 0.5` | The objective is genuinely learnable on residual-stream dynamics (not saturated by token embedding) | Ratio ≥ 0.5 → no learning; adapter fit a constant or trivial function |
| P3 | JEPA adapter GSM8K-Hard accuracy ≥ token-space r=16 LoRA baseline, n=200, greedy | Residual-stream prediction transfers knowledge into adapter weights | JEPA < baseline at matched params → the objective didn't help (or hurt) |
| P4 | Ablation: removing SIGReg (λ=0) degrades P3 target accuracy by ≥ 5pp | SIGReg is load-bearing, not cosmetic | λ=0 matches λ>0 within 5pp → SIGReg is inactive; objective works by itself |

## 5. Kill criteria (pre-registered; canonical DB text — do not edit)

- **K#1766 (structural, proxy)**: SIGReg Epps-Pulley rejection rate < 5% on adapter output activations at step 500 (no collapse).
- **K#1767 (proxy, learning dynamics)**: prediction loss L_pred(step 500) / L_pred(step 50) < 0.5.
- **K#1768 (target, paired with K#1766 per F#666)**: GSM8K-Hard accuracy ≥ token-space r=16 LoRA baseline at matched param budget on Gemma 4 E4B, n ≥ 200, greedy.
- **K#1769 (ablation target, paired with K#1767 per F#666)**: removing SIGReg (λ=0) degrades K3 target accuracy by ≥ 5pp.

**Target-gating (F#666).** K#1766 and K#1767 are proxies (activation statistics, training loss ratios). K#1768 and K#1769 are behavioral targets (GSM8K-Hard accuracy and ablation gap). KILL requires at least one target to FAIL together with its paired proxy. SUPPORTED requires BOTH proxies PASS AND BOTH targets PASS.

## 6. Measurement plan

1. **Baseline**: train token-space r=16 LoRA on `v_proj + o_proj` via `mlx_lm.lora` subprocess on GSM8K train (n=2000, 500 steps, batch=2, lr=1e-4, scale=6.0). Evaluate on GSM8K-Hard test n=200 greedy.
2. **JEPA**: train rank-16 `v_proj + o_proj` adapter with residual-stream prediction at layer ℓ=21 (middle layer) + prediction head + SIGReg. Loss `L_pred + λ · L_SIGReg`; λ found by bisection in {0.0, 0.1, 1.0, 10.0} selecting argmin validation L_pred s.t. Epps-Pulley rejection < 5%.
3. **Evaluate**: SIGReg rejection rate at step 500; L_pred ratio; GSM8K-Hard accuracy n=200 greedy with adapter loaded (prediction head discarded at inference — adapter acts as a standard LoRA).
4. **Ablation**: re-train step 2 with λ=0; evaluate same metrics.

## 7. Assumptions

- **A1.** `mlx-lm 0.31.x` supports adapter training on `v_proj + o_proj` at r=16 on Gemma 4 E4B. Verify with `python -c "import mlx_lm; print(mlx_lm.__version__)"`.
- **A2.** Residual stream at layer 21 (middle of 42) is structurally dense enough to carry next-token dynamics (LeWM found middle-layer representations strongest).
- **A3.** GSM8K-Hard is available via `openai/gsm8k` split with sufficient difficulty; if unavailable, fall back to GSM8K test split and note in PAPER.md.
- **A4.** Researcher-hat iteration budget (40 tool calls, 30 min wall-clock per REVISE cycle) precludes running 3 separate multi-hour training jobs in a single iteration.
- **A5.** LORA_SCALE ≤ 8 per F#328/F#330. Using default `scale = 6.0`.

## 8. Antipattern scan (pre-registration)

- composition math bug — N/A (single-adapter experiment, no composition)
- tautological routing — N/A (no routing)
- LORA_SCALE — scale=6.0 ≤ 8 ✓
- KC-swap-after-failure — 4 KCs locked before any training run ✓
- shutil.copy as new adapter — N/A ✓
- hardcoded `"pass": True` — KC results computed from measurements ✓
- eval-template truncation — GSM8K greedy max_tokens=1024 per F#1629 recovery ✓
- proxy-model substitution — MUST load `mlx-community/gemma-4-e4b-it-4bit`. Scaffold refuses to proxy to smaller variant ✓
- smoke-as-full — `is_smoke` flag exposed; PROVISIONAL if smoke ✓

## 9. QED

Given the failure mode (collapse to trivial solution) is geometrically ruled out by SIGReg (Theorem LeJEPA Thm 1, applied to `Z` at step 500), and given the proxy-target pairing (F#666), the experiment makes the following **falsifiable** claim: JEPA adapter matches or beats token-space r=16 LoRA on GSM8K-Hard at matched adapter-param budget, with SIGReg demonstrably load-bearing via ablation.
