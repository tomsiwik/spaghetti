# PAPER — `exp_hedgehog_behavior_adapter_politeness`

**Verdict.** PROVISIONAL (design locked, Phase B custom MLX training loop and Phase 0 neutral-prompt curation not yet implemented; full pipeline budget exceeds single-iteration researcher cap).

---

## Hypothesis (restated)

Per-layer cosine-similarity distillation between (a) frozen base Gemma 4 E4B under a polite-teacher system prompt and (b) base + rank-8 LoRA on `v_proj + o_proj` under a neutral system prompt trains an adapter that encodes politeness as attention-routing perturbation, preserving base generation capacity on unrelated tasks.

Grounded in Moudgil/Apple+MILA arxiv:2604.14191 §3.1 eq. 6 (Hedgehog Stage-1 recipe), Zhang et al. arxiv:2402.04347 (Hedgehog feature map), Finding #627 (Gemma 4 adapter target), Finding #666 (target-gated kill rule).

## Prediction vs. measurement

| Prediction (MATH.md §5) | Mechanism | Measured | Status |
|---|---|---|---|
| K#1782: mean per-layer cos(teacher_attn_out, student_attn_out) > 0.85 on held-out neutral prompts | Cos-sim loss minimizes per-layer divergence; Zhang 2024 shows cos > 0.99 is achievable for attention-output matching | NOT MEASURED — Phase A/B not implemented | untested |
| K#1783: auto-judge politeness Δ ≥ +20pp (student vs base) on n=100 neutral prompts | K1 PASS ⇒ downstream logits close in KL ⇒ polite-token preference transfers (attention Lipschitz through residual stream) | NOT MEASURED — no adapter trained | untested |
| K#1784a: MMLU subset drop < 3pp vs base | Rank-8 perturbation on `v_proj+o_proj` bounded, mostly orthogonal to unrelated-task principal directions (F#627 non-interference pattern) | NOT MEASURED — no adapter trained | untested |
| K#1784b: HumanEval pass@1 drop < 3pp vs base | Same non-interference argument as K#1784a | NOT MEASURED — no adapter trained | untested |
| K#1785: ablating polite system prompt from teacher regresses K2 by ≥ 10pp | Adapter encodes *prompt-induced routing*, not corpus-level information — removing prompt collapses the teacher signal | NOT MEASURED — Phase E depends on Phase B | untested |

All five KCs remain untested. No empirical claim filed.

## Kill criteria resolution

| KC | Text | Result |
|---|---|---|
| K#1782 (structural, proxy) | mean per-layer cos(teacher_attn_out, student_attn_out) > 0.85 at end of training | **untested** |
| K#1783 (target, paired with K#1782 per F#666) | auto-judge politeness Δ ≥ +20pp, n≥100 prompts | **untested** |
| K#1784 (target non-interference, paired with K#1783) | MMLU and HumanEval each drop < 3pp | **untested** |
| K#1785 (target ablation, paired with K#1783) | teacher-prompt ablation regresses K#1783 by ≥ 10pp | **untested** |

F#666 pairing satisfied in design: K#1782 + K#1783 are a proxy/target pair; K#1784 and K#1785 are additional target-side gates (non-interference + mechanism-ablation). No proxy-alone kill possible.

## Measurement blockers

1. **Phase B (Hedgehog training loop) not implemented.** Requires a custom MLX training loop — mlx-lm's `lora` CLI does not expose per-layer attention-output cos-sim loss. Required components:
   - Frozen-teacher and student forward passes on the same `x`, with teacher input `POLITE_SYSTEM_PROMPT ⊕ x` and student input `NEUTRAL_SYSTEM_PROMPT ⊕ x`.
   - Per-layer attention-output hooks on all 42 Gemma 4 E4B blocks; the hook must return `A_l(x)` (the post-`o_proj` attention output, before residual addition). Attribute path on mlx-lm 0.31.x `self_attn` must be verified — prior scaffold used `__call__` monkeypatching which is fragile under quantized layers.
   - Loss = `mean_l (1 − cos(A_l_teacher[l], A_l_student[l]))`, per-token-pooled then mean-over-layers. Teacher side detached via `mx.stop_gradient` (or skip grad tape entirely).
   - `nn.value_and_grad(student_model, loss_fn)` + `mlx.optimizers.AdamW`; `mx.eval(student_model.parameters(), loss)` at step boundary; `mx.clear_cache()` between batches (per `/mlx-dev` skill).
   - Adapter save compatible with `mlx_lm` adapter loading for downstream eval.

2. **Phase 0 (neutral-prompt curation) not implemented.** K2 judge scoring is load-bearing: if training prompts already contain politeness markers (please/thank-you/softeners), base is already polite by prompt osmosis and judge Δ collapses toward zero regardless of adapter. Curation must filter UltraChat (or a comparable instruction dataset) to remove politeness-marker regex matches, then split 1000/100 train/held-out.

3. **Full pipeline runtime.** Phase B full training (500 steps, batch=2) + Phase E full ablation-retrain (500 steps) + MMLU/HumanEval eval + Claude/GPT-4 judge API (n=100 × 2 conditions) is estimated at **3-5 hours** on M5 Pro 48GB plus judge-API dollar cost. Exceeds single-iteration researcher cap (30 min / 40 tool calls per guardrail 1009).

4. **Judge-API cost + determinism.** K2 and K4 both require a paired-compare LLM-judge. Judge choice (Claude 3.7 vs GPT-4) is not pinned in MATH.md; the `_impl` follow-up must pin both model + temperature + rubric JSON in a reproducible eval harness.

## Why PROVISIONAL rather than KILLED

The design is grounded in paper-validated math (Zhang 2024 cos > 0.99 existence result; Moudgil 2026 Stage-1 recipe transfer), KCs are pre-registered and target-gated per F#666, and the scaffold refuses to silently substitute a cross-entropy SFT objective. No proof-based impossibility result exists that would justify KILLED; the blocker is implementation effort, not falsification.

Filing PROVISIONAL preserves the design and kicks the implementation to a future iteration that can budget 3-5h wall-clock. A follow-up experiment `exp_hedgehog_behavior_adapter_politeness_impl` (P3) inherits this MATH.md verbatim with the training loop and Phase 0 curation filled in.

## Verdict-consistency pre-flight (all 6 checks per PLAN.md §1)

1. `results.json["verdict"]` = `"PROVISIONAL"` — not KILLED, not SUPPORTED ✓
2. `results.json["all_pass"]` = `false` — consistent with PROVISIONAL ✓
3. PAPER.md verdict line reads `PROVISIONAL` — not `supported` ✓
4. `is_smoke` = `false` — no smoke-as-full issue (no training run occurred at all) ✓
5. No KC was modified between MATH.md and now. K#1782–K#1785 match DB-registered text ✓
6. Antipattern scan:
   - composition math bug — N/A (single adapter) ✓
   - tautological routing — N/A (no routing) ✓
   - LORA_SCALE — `LORA_SCALE = 6.0` ≤ 8 ✓
   - KC-swap-after-failure — no data collected; KCs locked in MATH.md and results.json ✓
   - `shutil.copy` as new adapter — N/A (no adapter produced) ✓
   - hardcoded `"pass": True` — all KCs `"untested"`, not faked ✓
   - proxy-model substitution — `MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"` matches MATH.md; scaffold refuses silent downgrade ✓
   - eval-template truncation — N/A (no generation done); `_impl` must verify `max_tokens` discipline when Phase B lands ✓
   - smoke-as-full — `is_smoke: false`; no full run either; all KCs untested ✓
   - copy-paste scaffolding — scaffold is a fresh rewrite (not copied from JEPA sibling verbatim); structure is pattern-matched but every NotImplementedError body is hedgehog-specific ✓

All 6 checks clear **for a PROVISIONAL verdict**. A SUPPORTED verdict is not claimed.

## Assumptions (per researcher autonomy guardrail 1008)

- **A1.** `mlx-lm 0.31.2` (pinned in `results.json["mlx_lm_version"]`) exposes `model.layers[i].self_attn` as the attention submodule. `_impl` iteration MUST verify by inspection (quantized-layer wrapping may alter this path).
- **A2.** Cos-sim loss on post-`o_proj` attention output is sufficient to capture the polite-teacher behavioral difference. Zhang 2024 validates this for attention-*output* matching; a weaker variant (cos on pre-`o_proj` inputs) would be an inferior proxy.
- **A3.** Rank-8 LoRA on `v_proj + o_proj` at `LORA_SCALE=6.0` has sufficient capacity; F#627 validated rank-6 for domain adapters. One step up (rank-8) is defensibly within same regime.
- **A4.** UltraChat (or similar) filtered-for-neutral is the intended Phase 0 source. `_impl` may substitute if a better-curated neutral-instruction set emerges.
- **A5.** Researcher-hat single-iteration budget (30 min wall-clock / 40 tool calls) precludes a 3-5h run-to-completion. PROVISIONAL is the honest status; silently substituting a cheaper objective (e.g. standard SFT) would be an antipattern-(t) violation.

## Next step

Follow-up experiment `exp_hedgehog_behavior_adapter_politeness_impl` at P3 inherits this MATH.md and implements Phase 0 + Phase B/E in a dedicated iteration with multi-hour budget. Filed below this iteration; does not extend P≤2 backlog.
