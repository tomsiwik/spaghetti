# PAPER — `exp_rdt_jepa_loop_adapter`

**Verdict.** PROVISIONAL (design locked; novel-mechanism custom MLX training loop + infra dep on sibling K1765 speedup pushes empirical pipeline to ~6–10h, ~12–20× above single-iteration researcher-hat cap).

---

## Hypothesis (restated)

A rank-16 RDT (recurrent-depth transformer) loop adapter on `v_proj + o_proj` over layers [12, 20] of Gemma 4 E4B MLX 4-bit, trained with a **JEPA next-embedding prediction objective** (predict `h_{d+1}` from `h_d` across recurrent depth iterates, stopgrad targets) plus SIGReg Epps-Pulley anti-collapse, closes the parent `exp_rdt_loop_lora_gemma4_bench` (F#674 PROVISIONAL) behavioral gap: +5pp GSM8K-Hard at T=3 + saturating-exp depth elasticity R² > 0.90 across T ∈ {1..6}.

Grounded in LeWorldModel (arxiv:2603.19312, Maes/LeCun/Balestriero 2026-03-24), LeJEPA (arxiv:2511.08544), Bae 2024 Relaxed Recursive Transformers (arxiv:2410.20672), Finding #627 (Gemma 4 adapter target), Finding #666 (target-gated kill rule), Finding #674 (parent PROVISIONAL).

## Prediction vs. measurement

| Prediction (MATH.md §5) | KC | Mechanism | Measured | Status |
|---|---|---|---|---|
| P1: `max_d rho(A_d) < 1` across 500+ real training steps | K#1770 | Auxiliary loss preserves parent's contractive guarantee (F#674 K1739) | **NOT MEASURED** — training loop not implemented | untested |
| P2: Epps-Pulley rejection rate < 5% on `P_θ(h_d)` at each d ∈ {1..6} at step 500 | K#1771 | SIGReg forces per-d isotropic Gaussian; rules out cross-depth collapse (LeJEPA Thm 1) | **NOT MEASURED** — training loop not implemented | untested |
| P3: `L_pred(step 500) / L_pred(step 50) < 0.5` with monotone decrease across d | K#1772 | Cross-depth residual-stream dynamics are learnable; JEPA objective not saturated | **NOT MEASURED** — training loop not implemented | untested |
| P4: GSM8K-Hard accuracy +5pp at T=3, n=200 greedy vs base | K#1773 | JEPA auxiliary signal routes knowledge into Δ during training; eval with Δ alone (P_θ discarded) | **NOT MEASURED** — training loop not implemented AND infra dep on K1765 speedup (sibling `exp_rdt_loop_kv_cache_impl` P3) | untested |
| P5: Depth-elasticity saturating-exp R² > 0.90, T ∈ {1..6} at n ≥ 30 per T | K#1774 | JEPA training pressurizes every iterate; depth signal becomes structured (saturating-exp), not noise | **NOT MEASURED** — training loop not implemented AND eval-budget dep on K1765 | untested |

All five KCs remain untested. No empirical claim filed.

## Kill criteria resolution

| KC | Text | Result |
|---|---|---|
| K#1770 (structural, inherited F#674) | `max_d rho(A_d) < 1` across 500+ real GSM8K-loss steps | **untested** |
| K#1771 (structural, proxy) | SIGReg Epps-Pulley pass at each d ∈ {1..6} (no cross-depth collapse) | **untested** |
| K#1772 (proxy, learning dynamics) | `L_pred(step 500) / L_pred(step 50) < 0.5` + monotone across d | **untested** |
| K#1773 (target, paired K#1772 per F#666) | GSM8K-Hard +5pp at T=3, n ≥ 200, greedy | **untested** |
| K#1774 (target, paired K#1771 per F#666) | Depth-elasticity saturating-exp R² > 0.90, T ∈ {1..6}, n ≥ 30 | **untested** |

F#666 pairing satisfied in design: K#1771 ↔ K#1774 (isotropy ↔ depth elasticity), K#1772 ↔ K#1773 (learning dynamics ↔ GSM8K). K#1770 is structural precondition (stability) without required pairing per F#666 structural-KC carve-out. No proxy-alone kill possible.

## Measurement blockers

### Blocker 1 — Novel-mechanism custom MLX training loop

The experiment requires a bespoke MLX training loop that `mlx_lm.lora` CLI does not support. Required components (per MATH.md §6):

1. Monkey-patch or subclass `Gemma4TextModel.__call__` to expose intermediate residuals `h_d` at each depth iterate d ∈ [LOOP_START, LOOP_END) × T ∈ {1..6}. Sibling `exp_rdt_loop_kv_cache` MATH.md §1 has a matching patch pattern; reuse as template.
2. 2-layer MLP prediction head `P_θ: R^2560 → R^2560` with hidden=2560, trained jointly with rank-16 LoRA Δ on `v_proj + o_proj`.
3. Cross-depth residual collection per batch token: stack `h_0..h_T`, apply `P_θ` to h_0..h_{T-1}, construct stopgrad targets from h_1..h_T.
4. SIGReg on `Z = concat_d P_θ(h_d)`: sample M=1024 unit vectors u_m, project `Z · u_m`, compute Epps-Pulley statistic vs N(0,1) per LeJEPA Eq. 7 (numerical ECF integration via Gauss-Hermite quadrature K=32).
5. `nn.value_and_grad(model, loss_fn)` + `mlx.optimizers.AdamW`; `mx.eval(model.parameters(), loss)` at step boundary; `mx.clear_cache()` between batches per `/mlx-dev` skill (F#673 lineage — OOM otherwise).
6. Adapter save compatible with mlx-lm adapter loading (P_θ discarded at inference — JEPA must transfer knowledge into Δ).

Estimated ~4–6h of careful MLX engineering to land correctly (bigger than sibling F#682 due to cross-depth complexity + T=6 dimension).

### Blocker 2 — Full pipeline wall-clock budget

End-to-end pipeline estimated at 6–10h on M5 Pro 48GB:

- Phase A (base baseline eval, T=1): ~20 min × 1 arm = 20 min.
- Phase B (train JEPA+SIGReg adapter, 500 steps × 6-depth-iterate forward + SIGReg on M=1024 projections): 3–5h at T=6, plus λ bisection over {0.0, 0.1, 1.0, 10.0} requires up to 4 smoke runs before final full run.
- Phase C (ablation λ=0, same training): 3–5h (can be pruned to one replication).
- Phase D (GSM8K-Hard eval n=200, 3 arms, max_tokens=1024): 1–2h.
- Phase E (depth elasticity, 6 Ts × 30 prompts × 2 arms): 1–2h.

Total 6–10h; single-iteration researcher-hat cap is 30 min. ~12–20× over.

### Blocker 3 — Infra dep on K1765 speedup (analyst C2 routing)

Parent `exp_rdt_loop_kv_cache` PROVISIONAL (F#690). Its K1765 (5× cached vs uncached speedup) is required for n=200 GSM8K eval at T=3 to fit inside 2h:

- Without K1765: n=200 × T=3 uncached ≈ 90 min per arm (parent's K1740-BENCH estimate). × 3 arms = 4.5h just for Phase D.
- With K1765 (5× speedup): ~18 min per arm × 3 = 54 min for Phase D, fits.

**Analyst routing (C2 over C1).** F#669 preempt-KILL applies to child KCs transitively requiring parent's *behavioral target* SUPPORTED. Here, parent's K1764/K1765 are **infra-feasibility** claims (bit-exact + speedup), not behavioral mechanism claims. Child's behavioral targets (K#1773, K#1774) do not semantically require parent's targets SUPPORTED — they require parent's `_impl` to be runnable, which is the same axis as this experiment's own `_impl` budget. Therefore C2 (PROVISIONAL-as-design with `_impl` dep-linked at P3) is the correct routing, not C1 (F#669 preempt-KILL).

## Why PROVISIONAL rather than KILLED

The design is grounded in paper-validated math (LeJEPA Thm 1, LeWM application, Bae 2024 RDT construction), 5 KCs are pre-registered with F#666 proxy-target pairing, and the scaffold refuses silent scope-swap. No proof-based impossibility result exists that would justify a KILLED verdict; the blocker is implementation effort + compute budget + transitively an infra-speedup dep, not falsification.

Canonical precedent: F#682 (`exp_jepa_adapter_residual_stream` — layer-wise JEPA), F#683/F#684 (hedgehog cos-sim distillation), F#685 (MEMENTO Gemma 4 replication), F#686 (`exp_g4_adapter_class_composition_full` — macro-scope 15-training pipeline). All filed PROVISIONAL-as-design with `_impl` at P3. This experiment is structurally identical in filing pattern.

## Verdict-consistency pre-flight (all 6 checks per PLAN.md §1)

1. `results.json["verdict"]` = `"PROVISIONAL"` — not KILLED, not SUPPORTED ✓
2. `results.json["all_pass"]` = `false` — consistent with PROVISIONAL ✓
3. PAPER.md verdict line reads `PROVISIONAL` — not `supported` ✓
4. `is_smoke` = `false` — no smoke-as-full issue (no run at all) ✓
5. No KC was modified between MATH.md and now. K#1770–K#1774 match DB-registered text ✓
6. Antipattern scan (per MATH.md §7):
   - composition math bug — N/A (single Δ recurrently applied; parent F#674 already verified composition math) ✓
   - tautological routing — N/A (no routing) ✓
   - LORA_SCALE — `LORA_SCALE = 2.0` ≤ 8 per F#328/F#330 ✓
   - KC-swap-after-failure — KCs pre-registered; no data collected ✓
   - shutil.copy as new adapter — N/A (would train Δ from scratch) ✓
   - hardcoded `"pass": True` — all 5 KCs `"not_measured"`, not faked ✓
   - proxy-model substitution — `MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"` matches MATH.md §0 F1 scope lock; scaffold refuses silent downgrade ✓
   - eval-template truncation — `max_tokens=1024` per F#1629 recovery (pre-registered for `_impl`) ✓
   - smoke-as-full — `is_smoke: false`, scaffold-only, KCs untested ✓
   - novel-mechanism single-iteration-scope — EXPLICITLY TRIGGERED: filed PROVISIONAL-as-design per memory ✓
   - preempt-F#669 — checked and ruled out: infra-feasibility axis distinct from behavioral-KC transitivity axis (analyst C2) ✓

All 6 checks clear **for a PROVISIONAL verdict**. A SUPPORTED verdict is not claimed.

## Assumptions (per researcher autonomy guardrail 1008)

- **A1.** `mlx-lm 0.31.2` exposes `Gemma4TextModel.layers` amenable to monkey-patched forward returning intermediate residuals. Sibling `exp_rdt_loop_kv_cache` MATH.md §1 confirms this is possible.
- **A2.** Loop-region residual stream (output of each loop iterate) carries signal for JEPA prediction; parent F#674 K1739 PASS implies non-degenerate iterate geometry.
- **A3.** SIGReg M=1024 projections sufficient for d=2560. LeJEPA uses M ∈ {512, 1024, 4096}; 1024 is mid-range.
- **A4.** λ bisection over {0.0, 0.1, 1.0, 10.0} wide enough. LeWM §4.2 protocol.
- **A5.** LORA_SCALE ≤ 8 per F#328/F#330. Parent F#674 uses 2.0; inherited.
- **A6.** GSM8K test split with max_tokens=1024 = faithful "GSM8K-Hard" operationalization (no canonical hard split).
- **A7.** Researcher-hat single-iteration budget (30 min / 40 tool calls) insufficient for 6–10h pipeline; PROVISIONAL-as-design is the honest status.
- **A8.** Analyst routing C2 (PROVISIONAL-as-design with infra dep) selected over C1 (F#669 preempt-KILL); axis distinction (behavioral-KC vs. infra-feasibility) is the routing criterion.

## Next step

Follow-up `exp_rdt_jepa_loop_adapter_impl` at P3 inherits this MATH.md verbatim with all 5 KC IDs. Dep-linked to `exp_rdt_loop_kv_cache_impl` (P3, infra unblock) per analyst C2 handoff. Not filed in this iteration's backlog — P3 is explicitly out of the P≤2 drain objective per researcher.md step 2.

Empirical verification is the `_impl`'s job, not this filing's.
