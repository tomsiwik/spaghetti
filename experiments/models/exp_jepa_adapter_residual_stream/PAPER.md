# PAPER — `exp_jepa_adapter_residual_stream`

**Verdict.** PROVISIONAL (design locked, Phase B custom MLX training loop not yet implemented; full pipeline budget exceeds single-iteration researcher cap).

---

## Hypothesis (restated)

A rank-16 JEPA-style adapter on `v_proj + o_proj`, trained with residual-stream next-embedding prediction (layer 21 → layer 21 shifted by 1 token) plus SIGReg anti-collapse regularization, matches or beats token-space r=16 LoRA on GSM8K-Hard at matched adapter parameter budget on Gemma 4 E4B MLX 4-bit.

Grounded in LeWorldModel (Maes/LeCun/Balestriero 2026-03-24, arxiv:2603.19312), LeJEPA (arxiv:2511.08544), Finding #627 (Gemma 4 adapter target), Finding #666 (target-gated kill rule).

## Prediction vs. measurement

| Prediction (MATH.md §4) | Mechanism | Measured | Status |
|---|---|---|---|
| P1 / K#1766: Epps-Pulley rejection rate < 5% on adapter activations at step 500 | SIGReg forces isotropic Gaussian output (LeJEPA Thm 1) | NOT MEASURED — Phase B training not implemented | untested |
| P2 / K#1767: L_pred(step 500) / L_pred(step 50) < 0.5 | Residual-stream dynamics are a learnable objective | NOT MEASURED — Phase B training not implemented | untested |
| P3 / K#1768: JEPA GSM8K-Hard accuracy ≥ token-space LoRA baseline, n=200 greedy | Knowledge transfer via residual-stream prediction objective | NOT MEASURED — Phase A not run, Phase B not implemented | untested |
| P4 / K#1769: Ablation (λ=0) degrades K#1768 by ≥ 5pp | SIGReg is load-bearing anti-collapse | NOT MEASURED — depends on Phase B/C | untested |

All four KCs remain untested. No empirical claim filed.

## Kill criteria resolution

| KC | Text | Result |
|---|---|---|
| K#1766 (structural, proxy) | SIGReg Epps-Pulley rejection rate < 5% on adapter output activations at step 500 | **untested** |
| K#1767 (proxy, learning dynamics) | prediction loss L_pred(step 500) / L_pred(step 50) < 0.5 | **untested** |
| K#1768 (target, paired with K#1766 per F#666) | GSM8K-Hard accuracy ≥ token-space r=16 LoRA baseline at matched param budget, n ≥ 200, greedy | **untested** |
| K#1769 (ablation target, paired with K#1767 per F#666) | removing SIGReg (λ=0) degrades K1768 target accuracy by ≥ 5pp | **untested** |

F#666 pairing satisfied in design: K#1766 + K#1768 are a proxy/target pair; K#1767 + K#1769 are a proxy/target pair. No proxy-alone kill possible.

## Measurement blockers

1. **Phase B (JEPA custom training loop) not implemented.** The experiment requires a bespoke MLX training loop — mlx-lm's `lora` CLI does not support an auxiliary residual-stream prediction head plus SIGReg regularizer. Required components:
   - Forward-pass hook on `model.layers[21]` to capture residual-stream `h_ℓ(t)` for each token in each sequence in the batch.
   - 2-layer MLP prediction head `P_θ: R^2304 → R^2304` trained jointly with rank-16 LoRA on `v_proj + o_proj`.
   - Target construction: `h_ℓ(t+1)` via stopgrad on next-token residual (JEPA pattern preventing trivial base-model bypass).
   - SIGReg Epps-Pulley on `Z = P_θ(h_ℓ(t))` across M=1024 random unit projections, computed per LeJEPA Eq. 7.
   - `nn.value_and_grad(model, loss_fn)` + `mlx.optimizers.AdamW`, `mx.eval(model.parameters(), loss)` at step boundary, `mx.clear_cache()` between batches (per /mlx-dev skill).
   - Adapter save compatible with mlx-lm adapter loading (prediction head discarded at inference — JEPA objective must transfer knowledge *into* LoRA weights).

2. **Full pipeline runtime.** Phase A (token-space LoRA baseline) + Phase B (JEPA with SIGReg) + Phase C (JEPA ablation λ=0) + Phase D (3 × GSM8K-Hard evals n=200 greedy) estimated at **4-6 hours** on M5 Pro 48GB. Exceeds single-iteration researcher cap (30 min / 40 tool calls per researcher-hat guardrail 1009).

3. **Lambda bisection budget.** MATH.md §6 specifies λ ∈ {0, 0.1, 1.0, 10.0} bisection. Selecting the argmin λ that also passes K#1766 requires up to 4 Phase B runs at smoke scale before final full run. Bisection is budget-exposed in `SIGREG_LAMBDAS` config.

## Why PROVISIONAL rather than KILLED

The design is grounded in paper-validated math (LeJEPA Thm 1, LeWM application), KCs are pre-registered and target-gated per F#666, and the scaffold refuses to silently fall back to a simpler objective. No proof-based impossibility result exists that would justify a KILLED verdict; the blocker is implementation effort, not falsification.

Filing PROVISIONAL preserves the design and kicks the implementation to a future iteration that can budget 4-6h wall-clock. A follow-up experiment `exp_jepa_adapter_residual_stream_impl` can inherit this MATH.md verbatim with the training loop filled in.

## Verdict-consistency pre-flight (all 6 checks per PLAN.md §1)

1. `results.json["verdict"]` = `"PROVISIONAL"` — not KILLED, not SUPPORTED ✓
2. `results.json["all_pass"]` = `false` — consistent with PROVISIONAL ✓
3. PAPER.md verdict line reads `PROVISIONAL` — not `supported` ✓
4. `is_smoke` = `false` — no smoke-as-full issue (actually no run at all) ✓
5. No KC was modified between MATH.md and now. K#1766–K#1769 match DB-registered text ✓
6. Antipattern scan:
   - composition math bug — N/A (single adapter)
   - tautological routing — N/A
   - LORA_SCALE — `LORA_SCALE = 6.0` ≤ 8 ✓
   - KC-swap-after-failure — no data collected; KCs locked ✓
   - shutil.copy as new adapter — N/A ✓
   - hardcoded `"pass": True` — KCs all `"untested"`, not faked ✓
   - proxy-model substitution — `MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"` matches MATH.md; scaffold refuses silent downgrade ✓
   - eval-template truncation — `max_tokens=1024` per F#1629 recovery ✓
   - smoke-as-full — `is_smoke: false`, but no full run either; KCs untested ✓

All 6 checks clear **for a PROVISIONAL verdict**. A SUPPORTED verdict is not claimed.

## Assumptions (per researcher autonomy guardrail 1008)

- **A1.** `mlx-lm 0.31.x` supports LoRA training on `v_proj + o_proj` at r=16 on Gemma 4 E4B via the `lora` subcommand (Phase A baseline path).
- **A2.** Middle-layer residual stream (layer 21 of 42) carries next-token predictive signal sufficient for the JEPA objective.
- **A3.** GSM8K test split is the best available operationalization of "GSM8K-Hard" given no canonical hard split; flag in follow-up if a harder split is authoritative.
- **A4.** LORA_SCALE = 6.0 ≤ 8 per F#328/F#330.
- **A5.** Researcher-hat single-iteration budget (30 min wall-clock / 40 tool calls) precludes a 4-6h run-to-completion. PROVISIONAL is the honest status; silently substituting a cheaper objective would be an antipattern-'t' violation (scope-preservation).

## Next step

Follow-up experiment `exp_jepa_adapter_residual_stream_impl` (P3) to inherit this MATH.md and implement Phase B/C in a future iteration with a multi-hour budget. Not filed this iteration (P≤2 backlog drain is the active objective; adding a P3 does not extend it).
