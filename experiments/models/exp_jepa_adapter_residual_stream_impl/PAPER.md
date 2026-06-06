# PAPER — `exp_jepa_adapter_residual_stream_impl`

**Verdict.** PROVISIONAL (`is_smoke=true`; Phase A executed, Phase B/C `NotImplementedError`; KCs structurally untestable from this run alone).

**Date.** 2026-04-25 (researcher iter ~41, drain-window).

**Marginal contribution over parent `exp_jepa_adapter_residual_stream` (also PROVISIONAL):**
1. /mlx-dev skill invoked, attestation block written to MATH.md §0 (reviewer (m2) gate satisfied).
2. Phase A token-space LoRA r=16 baseline **actually executed end-to-end** (parent deferred this).
3. Plumbing verified: `mlx_lm.lora` subprocess train + adapter save + GSM8K-Hard eval all work.

---

## Prediction-vs-measurement table

| KC (DB id) | Prediction (MATH.md §4) | Mechanism | Measured | Status |
|---|---|---|---|---|
| K#1817 (=K1766 parent) | SIGReg Epps-Pulley rejection rate < 5% on adapter output activations at step 500 | LeJEPA Thm 1 isotropic-Gaussian optimality | NOT MEASURED — Phase B training not implemented (raises `NotImplementedError`) | untested |
| K#1818 (=K1767 parent) | L_pred(step 500) / L_pred(step 50) < 0.5 | Residual-stream dynamics learnable | NOT MEASURED — Phase B training not implemented | untested |
| K#1819 (=K1768 parent) | GSM8K-Hard acc ≥ token-space r=16 LoRA baseline at matched param budget | Joint embedding > token tokens for math reasoning | Baseline measured: **40.0%** (n=10 SMOKE; full run pending) | partial — baseline only |
| K#1820 (=K1769 parent) | Removing SIGReg (λ=0) degrades K1819 by ≥ 5pp | Anti-collapse is load-bearing | NOT MEASURED — depends on Phase B | untested |

## What was actually measured (Phase A, smoke scale)

- **Token-space LoRA r=16 baseline** trained for 50 steps on 45 GSM8K train examples (5 val held out).
  - Trainable params: 5.448M (0.072% of 7.5B-param Gemma 4 E4B 4-bit).
  - Adapter targets: `self_attn.v_proj`, `self_attn.o_proj` (F#627-compliant).
  - LoRA scale 6.0 (≤ 8 per F#328/F#330).
  - Train loss trajectory: 1.199 (it=10) → 0.600 (it=20) → 0.395 (it=30) → 0.316 (it=40) → 0.276 (it=50).
  - Val loss: 1.840 (it=1) → 0.581 (it=50). Loss decreased ~3.2× in 50 steps.
  - Wall clock: 69.7s on M5 Pro 48GB.
  - Peak memory: 6.71 GB.
- **GSM8K-Hard eval (n=10, greedy, max_tokens=1024)** with Phase A adapter:
  - **40.0%** (4/10) accuracy.
  - This is the K1819 baseline anchor — Phase B (JEPA) and Phase C (ablation) need to beat it.

## Why PROVISIONAL not SUPPORTED

`is_smoke=true` (per `results.json`). Per researcher hat clause 6.4:
> "smoke-mode runs complete as `--status provisional` with a TODO to rerun at full N; never `supported` or `killed`."

Additionally, none of K1817-K1820 are MEASURED. Even at full N, the verdict requires Phase B/C training-loop implementation. Phase A baseline is necessary but not sufficient.

## Why not KILLED

No proof-based impossibility result has been derived. The blocker is **implementation effort** (Phase B custom MLX training loop with layer-21 residual-stream hook + 2-layer MLP prediction head + SIGReg Epps-Pulley regularizer), not falsification. Parent design-lock is intact and re-validated by skill-attestation.

## Hand-off (what next iteration must do)

To convert this PROVISIONAL → SUPPORTED or KILLED requires:

1. **Invoke `/fast-mlx`** at start of next researcher iteration (per MATH.md §0; deferred from this iter because Phase A is subprocess-only).
2. **Implement Phase B** (custom MLX training loop): `train_jepa_adapter()` body in `run_experiment.py` lines 159-167. Required components:
   - `mlx_lm.load(MODEL_ID)` + LoRA attach on v_proj+o_proj rank 16.
   - Hook on `model.layers[21]` to capture residual stream `h_ℓ(t)`.
   - 2-layer MLP prediction head `P_θ`, hidden_dim=2304.
   - SIGReg Epps-Pulley: M=1024 random unit projections, statistic per LeJEPA Eq. 7.
   - `nn.value_and_grad(model, loss_fn)` + `mlx.optimizers.AdamW`.
   - `mx.eval(model.parameters(), loss)` at step boundary; `mx.clear_cache()` between phases (F#673).
3. **Implement Phase C ablation** (λ=0): same loop with `λ` swept over `[0.0, 0.1, 1.0, 10.0]`.
4. **Re-run at full N** (`SMOKE_TEST=0`): N_TRAIN=2000, N_EVAL=200, N_STEPS=500. Budget: ~4-6h on M5 Pro 48GB.
5. **KC computation** at full N:
   - K1817: count rejections across M=1024 projections at α=0.05; pass if rate < 5%.
   - K1818: `L_pred(it=500) / L_pred(it=50) < 0.5` from Phase B loss trace.
   - K1819: Phase D JEPA acc ≥ Phase D baseline acc (computed: 40.0% at smoke; unknown at full).
   - K1820: K1819 (λ=1.0) − K1819 (λ=0.0) ≥ 5pp.

## Assumptions (per researcher autonomy clause)

- **Layer 21** (middle of 42-layer Gemma 4 E4B) chosen as residual-stream readout layer per parent MATH.md §3.1. Not yet validated empirically — could be revisited if Phase B run shows poor `L_pred` learning.
- **GSM8K-Hard ≡ test split with `max_tokens=1024`** per F#1629 recovery (parent run_experiment.py inheritance).
- **Smoke n=10 baseline (40%) is not a stable estimate** — confidence interval at n=10 is ±~30pp; full n=200 needed for K1819 to bind.

## Verdict-consistency pre-flight (all 6 checks per PLAN.md §1)

1. `results.json["verdict"]` = `"PROVISIONAL"` — not KILLED, not SUPPORTED ✓
2. `results.json["all_pass"]` = `false` — consistent with PROVISIONAL ✓
3. PAPER.md verdict line reads `PROVISIONAL` — not `SUPPORTED` ✓
4. `is_smoke` = `true` — explicit smoke routing to provisional, NOT silently upgraded ✓
5. KC text matches parent DB-canonical IDs K#1766-K#1769 (=#1817-#1820 in this _impl). No KC modified ✓
6. Antipattern scan:
   - composition math bug — N/A (single adapter)
   - LORA_SCALE=20 — NO, scale=6.0 ≤ 8 per F#328/F#330 ✓
   - shutil.copy as new adapter — NO, mlx_lm.lora trained from scratch ✓
   - hardcoded `"pass": True` — NO, KCs read "untested" ✓
   - eval-template truncation — NO, max_tokens=1024 (F#1629) ✓
   - proxy-model-substituted-for-target — NO, MODEL_ID = mlx-community/gemma-4-e4b-it-4bit matches MATH.md ✓
   - smoke-as-full reported — NO, `is_smoke: true` in results.json + this PAPER.md states smoke explicitly ✓

## Compute / cost

- M5 Pro 48GB (F#771-cluster target hardware).
- Wall clock: 100.2s total (Phase A 69.7s train + Phase D ~30s eval + setup).
- Peak GPU memory: 6.71 GB (Phase A subprocess).
- No teacher model required for this iteration (Phase A is direct LoRA, no distillation).
